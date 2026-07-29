"""RSI 70-30 and 50-30 crossover signals. -1/0/1 = sell/hold/buy."""

from __future__ import annotations

import pandas as pd


def rsi_70_30_signal(rsi: pd.Series) -> pd.Series:
    signal = pd.Series(0, index=rsi.index, dtype=int)
    prev = rsi.shift(1)

    buy = (prev < 30) & (rsi >= 30)
    sell = (prev > 70) & (rsi <= 70)

    signal[buy] = 1
    signal[sell] = -1
    return signal


def rsi_50_30_signal(rsi: pd.Series) -> pd.Series:
    signal = pd.Series(0, index=rsi.index, dtype=int)
    prev = rsi.shift(1)

    # confirm bullish only if RSI was actually oversold recently
    was_oversold = rsi.rolling(10, min_periods=1).min().shift(1) < 30
    buy = (prev < 50) & (rsi >= 50) & was_oversold

    was_overbought = rsi.rolling(10, min_periods=1).max().shift(1) > 70
    sell = (prev > 50) & (rsi <= 50) & was_overbought

    signal[buy] = 1
    signal[sell] = -1
    return signal


def combine_with_model_prediction(
    rsi_signal: pd.Series,
    predicted_return: pd.Series,
    buy_threshold: float = 0.0025,
    sell_threshold: float = -0.0025,
) -> pd.Series:
    """Only fire the RSI signal if the model doesn't disagree with it in
    direction (or fall back to the model alone when RSI is silent)."""
    model_dir = pd.Series(0, index=predicted_return.index, dtype=int)
    model_dir[predicted_return >= buy_threshold] = 1
    model_dir[predicted_return <= sell_threshold] = -1

    combined = pd.Series(0, index=rsi_signal.index, dtype=int)
    combined[(rsi_signal == 1) & (model_dir != -1)] = 1
    combined[(rsi_signal == -1) & (model_dir != 1)] = -1
    combined[(rsi_signal == 0) & (model_dir == 1)] = 1
    combined[(rsi_signal == 0) & (model_dir == -1)] = -1

    return combined


if __name__ == "__main__":
    from data_ingestion import get_dataset
    from feature_engineering import build_feature_frame

    raw = get_dataset()
    feats = build_feature_frame(raw)

    s70 = rsi_70_30_signal(feats["rsi_14"])
    s50 = rsi_50_30_signal(feats["rsi_14"])

    print("70-30 signal counts:\n", s70.value_counts())
    print("\n50-30 signal counts:\n", s50.value_counts())
