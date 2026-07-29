"""End-to-end: data -> features -> models -> signals -> backtest -> figures.
Run via `python scripts/run_pipeline.py` or `docker compose run pipeline`."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data_ingestion import get_dataset
from .feature_engineering import build_feature_frame, FEATURE_COLUMNS, TARGET_COLUMN
from .model_training import run_training_pipeline
from .signal_generation import rsi_70_30_signal, rsi_50_30_signal, combine_with_model_prediction
from .backtest import backtest_strategy, summarize_backtest

logger = logging.getLogger(__name__)

FIGURES_DIR = Path("outputs/figures")
PROCESSED_DIR = Path("data/processed")
DEFAULT_CONFIG_PATH = Path("config/config.yaml")


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        logger.warning("Config file %s not found, using built-in defaults", config_path)
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def plot_price_with_rsi(modeling_df, out_path: Path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(modeling_df.index, modeling_df["close"], color="#1f77b4", linewidth=1)
    ax1.set_title("Close Price")
    ax1.set_ylabel("Price ($)")

    ax2.plot(modeling_df.index, modeling_df["rsi_14"], color="#ff7f0e", linewidth=1)
    ax2.axhline(70, color="red", linestyle="--", linewidth=0.8)
    ax2.axhline(30, color="green", linestyle="--", linewidth=0.8)
    ax2.set_ylabel("RSI(14)")
    ax2.set_ylim(0, 100)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_model_comparison(metrics: dict, out_path: Path):
    names = list(metrics.keys())
    rmse = [metrics[n]["test_rmse"] for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, rmse, color=["#4c72b0", "#dd8452", "#55a868"])
    ax.set_ylabel("Test RMSE (lower is better)")
    ax.set_title("Model Comparison - Next-Day Return Prediction")
    for b, v in zip(bars, rmse):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_equity_curve(bt_df, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(bt_df.index, bt_df["strategy_equity"], label="RSI + ML Strategy")
    ax.plot(bt_df.index, bt_df["buy_hold_equity"], label="Buy & Hold", linestyle="--")
    ax.set_ylabel("Growth of $1")
    ax.set_title("Strategy vs. Buy & Hold (held-out test period)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def run(ticker: str | None = None, prefer_real: bool | None = None, config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    cfg = load_config(config_path)
    data_cfg = cfg.get("data", {})
    signal_cfg = cfg.get("signals", {}).get("model_blend", {})
    modeling_cfg = cfg.get("modeling", {})
    output_cfg = cfg.get("output", {})

    # CLI/explicit args override config file values, which override built-in defaults.
    ticker = ticker or data_cfg.get("ticker", "AAPL")
    prefer_real = prefer_real if prefer_real is not None else data_cfg.get("prefer_real_data", False)
    cache_path = data_cfg.get("cache_path", "data/raw/ohlcv.csv")
    rsi_period = cfg.get("features", {}).get("rsi_period", 14)
    buy_thr = signal_cfg.get("buy_threshold", 0.0025)
    sell_thr = signal_cfg.get("sell_threshold", -0.0025)
    metrics_path = output_cfg.get("metrics_path", "outputs/model_metrics.json")
    report_path = output_cfg.get("report_path", "outputs/final_report.json")
    figures_dir = Path(output_cfg.get("figures_dir", "outputs/figures"))
    figures_dir.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Step 1/5: Loading data")
    raw = get_dataset(
        ticker=ticker,
        prefer_real=prefer_real,
        start=data_cfg.get("start_date", "2022-01-01"),
        end=data_cfg.get("end_date", "2024-12-31"),
        cache_path=cache_path,
    )

    logger.info("Step 2/5: Engineering features")
    feats = build_feature_frame(raw, rsi_period=rsi_period)
    feats.to_csv(PROCESSED_DIR / "features.csv")

    logger.info("Step 3/5: Training and tuning models")
    results, metrics, modeling_df = run_training_pipeline(
        feats, FEATURE_COLUMNS, TARGET_COLUMN,
        output_path=metrics_path,
    )

    best_model_name = min(metrics, key=lambda k: metrics[k]["test_rmse"])
    logger.info("Best model by test RMSE: %s", best_model_name)

    logger.info("Step 4/5: Generating signals and backtesting")
    rsi_sig_70_30 = rsi_70_30_signal(modeling_df["rsi_14"])
    rsi_sig_50_30 = rsi_50_30_signal(modeling_df["rsi_14"])
    combined_sig = combine_with_model_prediction(
        rsi_sig_70_30, modeling_df[f"pred_{best_model_name}"].fillna(0),
        buy_threshold=buy_thr, sell_threshold=sell_thr,
    )
    modeling_df["signal_rsi_70_30"] = rsi_sig_70_30
    modeling_df["signal_rsi_50_30"] = rsi_sig_50_30
    modeling_df["signal_combined"] = combined_sig

    test_slice = modeling_df.dropna(subset=[f"pred_{best_model_name}"])
    bt = backtest_strategy(test_slice, "signal_combined")
    bt_summary = summarize_backtest(bt)

    logger.info("Step 5/5: Saving figures and final report")
    plot_price_with_rsi(modeling_df, figures_dir / "price_rsi.png")
    plot_model_comparison(metrics, figures_dir / "model_comparison.png")
    plot_equity_curve(bt, figures_dir / "equity_curve.png")

    report = {
        "ticker": ticker,
        "best_model": best_model_name,
        "model_metrics": metrics,
        "backtest_summary": bt_summary,
    }
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    report = run()
    print(json.dumps(report, indent=2))
