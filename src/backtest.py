"""Turns signals + realized returns into a strategy equity curve vs.
buy-and-hold. No fees/slippage/sizing - illustrative, not a live system."""

from __future__ import annotations

import numpy as np
import pandas as pd


def backtest_strategy(
    modeling_df: pd.DataFrame,
    signal_column: str,
    return_column: str = "target_next_return",
) -> pd.DataFrame:
    """signal_column is -1/0/1; multiplies by realized return per day."""
    df = modeling_df.dropna(subset=[signal_column, return_column]).copy()

    df["strategy_return"] = df[signal_column] * df[return_column]
    df["buy_hold_return"] = df[return_column]

    df["strategy_equity"] = (1 + df["strategy_return"]).cumprod()
    df["buy_hold_equity"] = (1 + df["buy_hold_return"]).cumprod()

    return df


def summarize_backtest(bt_df: pd.DataFrame) -> dict:
    n_days = len(bt_df)
    ann_factor = 252 / n_days if n_days > 0 else 0

    strat_total_return = bt_df["strategy_equity"].iloc[-1] - 1 if n_days else 0.0
    bh_total_return = bt_df["buy_hold_equity"].iloc[-1] - 1 if n_days else 0.0

    strat_ann_return = (1 + strat_total_return) ** ann_factor - 1 if n_days else 0.0
    bh_ann_return = (1 + bh_total_return) ** ann_factor - 1 if n_days else 0.0

    strat_vol = bt_df["strategy_return"].std() * np.sqrt(252)
    strat_sharpe = (bt_df["strategy_return"].mean() * 252) / strat_vol if strat_vol > 0 else 0.0

    n_trades = (bt_df["strategy_return"] != 0).sum()
    hit_rate = (bt_df.loc[bt_df["strategy_return"] != 0, "strategy_return"] > 0).mean() if n_trades else 0.0

    max_drawdown = (
        (bt_df["strategy_equity"] / bt_df["strategy_equity"].cummax()) - 1
    ).min()

    return {
        "n_days": int(n_days),
        "n_signals_taken": int(n_trades),
        "strategy_total_return": float(strat_total_return),
        "buy_hold_total_return": float(bh_total_return),
        "strategy_annualized_return": float(strat_ann_return),
        "buy_hold_annualized_return": float(bh_ann_return),
        "strategy_annualized_vol": float(strat_vol),
        "strategy_sharpe": float(strat_sharpe),
        "hit_rate": float(hit_rate),
        "max_drawdown": float(max_drawdown),
    }


if __name__ == "__main__":
    import json
    import logging

    logging.basicConfig(level=logging.INFO)

    from data_ingestion import get_dataset
    from feature_engineering import build_feature_frame, FEATURE_COLUMNS, TARGET_COLUMN
    from model_training import run_training_pipeline
    from signal_generation import rsi_70_30_signal, combine_with_model_prediction

    raw = get_dataset()
    feats = build_feature_frame(raw)
    results, metrics, modeling_df = run_training_pipeline(feats, FEATURE_COLUMNS, TARGET_COLUMN)

    best_model_name = min(metrics, key=lambda k: metrics[k]["test_rmse"])
    print(f"Best model by test RMSE: {best_model_name}")

    rsi_sig = rsi_70_30_signal(modeling_df["rsi_14"])
    combined_sig = combine_with_model_prediction(
        rsi_sig, modeling_df[f"pred_{best_model_name}"].fillna(0)
    )
    modeling_df["combined_signal"] = combined_sig

    bt = backtest_strategy(modeling_df.dropna(subset=[f"pred_{best_model_name}"]), "combined_signal")
    summary = summarize_backtest(bt)
    print(json.dumps(summary, indent=2))
