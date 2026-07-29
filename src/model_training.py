"""Trains/tunes LR, RF, GBT to predict next-day return and picks the best
on held-out RMSE. CV is TimeSeriesSplit (no shuffling future into past).

Uses scikit-learn instead of Spark MLlib so it runs without a cluster -
see notebooks/databricks_pyspark_pipeline.py for the Spark version.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


MODEL_GRID = {
    "linear_regression": {
        "estimator": LinearRegression(),
        "param_grid": {},  # no meaningful hyperparameters to tune for OLS
    },
    "random_forest": {
        "estimator": RandomForestRegressor(random_state=42),
        "param_grid": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [3, 5, 8],
            "model__min_samples_leaf": [5, 10],
        },
    },
    "gradient_boosted_trees": {
        "estimator": GradientBoostingRegressor(random_state=42),
        "param_grid": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [2, 3],
            "model__learning_rate": [0.01, 0.05, 0.1],
        },
    },
}


def time_ordered_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    n_test = int(len(X) * test_size)
    X_train, X_test = X.iloc[:-n_test], X.iloc[-n_test:]
    y_train, y_test = y.iloc[:-n_test], y.iloc[-n_test:]
    return X_train, X_test, y_train, y_test


def train_and_tune_all(X_train: pd.DataFrame, y_train: pd.Series, n_splits: int = 5) -> dict:
    """GridSearchCV per model in MODEL_GRID, returns fitted best estimators + CV scores."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = {}

    for name, spec in MODEL_GRID.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("model", spec["estimator"])])

        if spec["param_grid"]:
            search = GridSearchCV(
                pipe,
                param_grid=spec["param_grid"],
                scoring="neg_root_mean_squared_error",
                cv=tscv,
                n_jobs=-1,
            )
            search.fit(X_train, y_train)
            best_estimator = search.best_estimator_
            best_params = search.best_params_
            cv_rmse = -search.best_score_
        else:
            pipe.fit(X_train, y_train)
            best_estimator = pipe
            best_params = {}
            # cross-validate manually since there's no grid to search
            scores = []
            for tr_idx, val_idx in tscv.split(X_train):
                p = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
                p.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
                pred = p.predict(X_train.iloc[val_idx])
                scores.append(np.sqrt(mean_squared_error(y_train.iloc[val_idx], pred)))
            cv_rmse = float(np.mean(scores))

        results[name] = {
            "estimator": best_estimator,
            "best_params": best_params,
            "cv_rmse": cv_rmse,
        }
        logger.info("%s: CV RMSE=%.6f params=%s", name, cv_rmse, best_params)

    return results


def evaluate_on_test(results: dict, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    metrics = {}
    for name, r in results.items():
        pred = r["estimator"].predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        r2 = float(r2_score(y_test, pred))
        metrics[name] = {
            "test_rmse": rmse,
            "test_r2": r2,
            "cv_rmse": r["cv_rmse"],
            "best_params": r["best_params"],
        }
    return metrics


def run_training_pipeline(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    output_path: str | Path = "outputs/model_metrics.json",
) -> tuple[dict, dict]:
    modeling_df = df.dropna(subset=feature_columns + [target_column])
    X = modeling_df[feature_columns]
    y = modeling_df[target_column]

    X_train, X_test, y_train, y_test = time_ordered_split(X, y)
    logger.info("Train size=%d, Test size=%d", len(X_train), len(X_test))

    results = train_and_tune_all(X_train, y_train)
    metrics = evaluate_on_test(results, X_test, y_test)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info("Wrote metrics -> %s", output_path)

    # attach test predictions for downstream backtest/signal work
    for name, r in results.items():
        modeling_df.loc[X_test.index, f"pred_{name}"] = r["estimator"].predict(X_test)

    return results, metrics, modeling_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from data_ingestion import get_dataset
    from feature_engineering import build_feature_frame, FEATURE_COLUMNS, TARGET_COLUMN

    raw = get_dataset()
    feats = build_feature_frame(raw)
    results, metrics, modeling_df = run_training_pipeline(feats, FEATURE_COLUMNS, TARGET_COLUMN)

    print(json.dumps(metrics, indent=2))
