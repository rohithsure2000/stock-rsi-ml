# Databricks notebook source
# MAGIC %md
# MAGIC # Stock Market Prediction Using RSI + Spark MLlib
# MAGIC
# MAGIC This notebook mirrors `src/` in the repository root, reimplemented with
# MAGIC **PySpark / Spark MLlib** for direct use on a Databricks cluster (this is
# MAGIC the environment the original version of this project was built on).
# MAGIC
# MAGIC Pipeline stages:
# MAGIC 1. Load OHLCV data into a Spark DataFrame
# MAGIC 2. Compute RSI(14) and supporting technical indicators via Spark window functions
# MAGIC 3. Generate 70-30 and 50-30 RSI crossover signals
# MAGIC 4. Assemble a feature vector and train Linear Regression, Random Forest, and
# MAGIC    Gradient-Boosted Tree regressors from `pyspark.ml`, tuned via `CrossValidator`
# MAGIC 5. Compare models on R^2 / RMSE and translate predictions into buy/sell thresholds
# MAGIC
# MAGIC NOTE: This file is written in Databricks' "notebook as .py source" format
# MAGIC (`# COMMAND ----------` cell markers) so it can be imported directly as a
# MAGIC Databricks notebook via Repos, or run locally with `spark-submit` given a
# MAGIC local Spark install. It is not executed as part of this repo's CI/tests,
# MAGIC since that runs on the lightweight scikit-learn implementation in `src/`.

# COMMAND ----------

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline

spark = SparkSession.builder.appName("StockRSIPrediction").getOrCreate()

# COMMAND ----------

# MAGIC %md ## 1. Load data
# MAGIC On Databricks, point this at a Delta table or a mounted CSV of historical
# MAGIC OHLCV data (e.g. via a market data ingestion job). For local/demo runs,
# MAGIC this loads the same synthetic CSV produced by `src/data_ingestion.py` in
# MAGIC the repo root, so results are directly comparable to the sklearn version.

# COMMAND ----------

ohlcv_path = "data/raw/ohlcv.csv"  # or a Delta table path / dbfs mount on Databricks

df = (
    spark.read.option("header", True)
    .option("inferSchema", True)
    .csv(ohlcv_path)
    .withColumnRenamed("date", "trade_date")
    .orderBy("trade_date")
)

# COMMAND ----------

# MAGIC %md ## 2. Feature engineering with Spark window functions

# COMMAND ----------

w = Window.orderBy("trade_date")

df = df.withColumn("prev_close", F.lag("close", 1).over(w))
df = df.withColumn("change", F.col("close") - F.col("prev_close"))
df = df.withColumn("gain", F.when(F.col("change") > 0, F.col("change")).otherwise(0.0))
df = df.withColumn("loss", F.when(F.col("change") < 0, -F.col("change")).otherwise(0.0))

# Wilder's smoothing approximated with a trailing simple moving average over
# a 14-day window (a common, close approximation used in Spark pipelines
# where a true recursive EMA is awkward to express as a window function).
rsi_window = Window.orderBy("trade_date").rowsBetween(-13, 0)
df = df.withColumn("avg_gain", F.avg("gain").over(rsi_window))
df = df.withColumn("avg_loss", F.avg("loss").over(rsi_window))
df = df.withColumn(
    "rsi_14",
    F.when(F.col("avg_loss") == 0, F.lit(100.0)).otherwise(
        100 - (100 / (1 + (F.col("avg_gain") / F.col("avg_loss"))))
    ),
)

sma_window_5 = Window.orderBy("trade_date").rowsBetween(-4, 0)
sma_window_10 = Window.orderBy("trade_date").rowsBetween(-9, 0)
sma_window_20 = Window.orderBy("trade_date").rowsBetween(-19, 0)

df = df.withColumn("sma_5", F.avg("close").over(sma_window_5))
df = df.withColumn("sma_10", F.avg("close").over(sma_window_10))
df = df.withColumn("sma_20", F.avg("close").over(sma_window_20))
df = df.withColumn("rolling_vol_10", F.stddev("change").over(sma_window_10))
df = df.withColumn("price_to_sma20", F.col("close") / F.col("sma_20") - 1)

df = df.withColumn(
    "target_next_return",
    (F.lead("close", 1).over(w) - F.col("close")) / F.col("close"),
)

feature_cols = ["rsi_14", "sma_5", "sma_10", "sma_20", "rolling_vol_10", "price_to_sma20"]
modeling_df = df.dropna(subset=feature_cols + ["target_next_return"])

# COMMAND ----------

# MAGIC %md ## 3. RSI 70-30 / 50-30 crossover signals

# COMMAND ----------

df = df.withColumn("prev_rsi", F.lag("rsi_14", 1).over(w))
df = df.withColumn(
    "signal_70_30",
    F.when((F.col("prev_rsi") < 30) & (F.col("rsi_14") >= 30), 1)
    .when((F.col("prev_rsi") > 70) & (F.col("rsi_14") <= 70), -1)
    .otherwise(0),
)
df = df.withColumn(
    "signal_50_30",
    F.when((F.col("prev_rsi") < 50) & (F.col("rsi_14") >= 50), 1)
    .when((F.col("prev_rsi") > 50) & (F.col("rsi_14") <= 50), -1)
    .otherwise(0),
)

# COMMAND ----------

# MAGIC %md ## 4. Train / tune Spark MLlib regressors

# COMMAND ----------

train_df, test_df = modeling_df.randomSplit([0.8, 0.2], seed=42)

assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
scaler = StandardScaler(inputCol="raw_features", outputCol="features", withMean=True, withStd=True)

evaluator_rmse = RegressionEvaluator(labelCol="target_next_return", predictionCol="prediction", metricName="rmse")
evaluator_r2 = RegressionEvaluator(labelCol="target_next_return", predictionCol="prediction", metricName="r2")

results = {}

# --- Linear Regression ---
lr = LinearRegression(labelCol="target_next_return", featuresCol="features")
pipeline_lr = Pipeline(stages=[assembler, scaler, lr])
model_lr = pipeline_lr.fit(train_df)
pred_lr = model_lr.transform(test_df)
results["linear_regression"] = {
    "rmse": evaluator_rmse.evaluate(pred_lr),
    "r2": evaluator_r2.evaluate(pred_lr),
}

# --- Random Forest ---
rf = RandomForestRegressor(labelCol="target_next_return", featuresCol="features", seed=42)
pipeline_rf = Pipeline(stages=[assembler, scaler, rf])
grid_rf = (
    ParamGridBuilder()
    .addGrid(rf.numTrees, [100, 200])
    .addGrid(rf.maxDepth, [3, 5, 8])
    .build()
)
cv_rf = CrossValidator(estimator=pipeline_rf, estimatorParamMaps=grid_rf, evaluator=evaluator_rmse, numFolds=5)
model_rf = cv_rf.fit(train_df)
pred_rf = model_rf.transform(test_df)
results["random_forest"] = {
    "rmse": evaluator_rmse.evaluate(pred_rf),
    "r2": evaluator_r2.evaluate(pred_rf),
}

# --- Gradient-Boosted Trees ---
gbt = GBTRegressor(labelCol="target_next_return", featuresCol="features", seed=42)
pipeline_gbt = Pipeline(stages=[assembler, scaler, gbt])
grid_gbt = (
    ParamGridBuilder()
    .addGrid(gbt.maxIter, [100, 200])
    .addGrid(gbt.maxDepth, [2, 3])
    .addGrid(gbt.stepSize, [0.01, 0.05, 0.1])
    .build()
)
cv_gbt = CrossValidator(estimator=pipeline_gbt, estimatorParamMaps=grid_gbt, evaluator=evaluator_rmse, numFolds=5)
model_gbt = cv_gbt.fit(train_df)
pred_gbt = model_gbt.transform(test_df)
results["gradient_boosted_trees"] = {
    "rmse": evaluator_rmse.evaluate(pred_gbt),
    "r2": evaluator_r2.evaluate(pred_gbt),
}

for name, m in results.items():
    print(f"{name}: RMSE={m['rmse']:.6f}  R2={m['r2']:.4f}")

# COMMAND ----------

# MAGIC %md ## 5. Translate best model's predictions into buy/sell thresholds

# COMMAND ----------

best_name = min(results, key=lambda k: results[k]["rmse"])
best_preds = {"linear_regression": pred_lr, "random_forest": pred_rf, "gradient_boosted_trees": pred_gbt}[best_name]

signal_df = best_preds.withColumn(
    "model_signal",
    F.when(F.col("prediction") >= 0.0025, 1).when(F.col("prediction") <= -0.0025, -1).otherwise(0),
)
display(signal_df.select("trade_date", "close", "rsi_14", "signal_70_30", "prediction", "model_signal"))
