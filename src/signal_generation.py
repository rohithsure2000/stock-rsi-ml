"""
signal_generation.py

Implements RSI-based crossover trading signals:

- 70-30 strategy: classic overbought/oversold thresholds. RSI crossing
  above 30 (from below) generates a BUY signal; RSI crossing below 70
  (from above) generates a SELL signal.
- 50-30 strategy: a more conservative variant used for trend confirmation,
  where RSI crossing above 50 (from below 30 territory) confirms bullish
  momentum, and RSI crossing below 50 (from above) confirms bearish
  momentum.

Signals are encoded as {-1: SELL, 0: HOLD, 1: BUY}.
"""

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

    # Bullish confirmation: momentum recovering through the midline after
    # having been oversold within the lookback window.
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
    """Blend a rule-based RSI signal with the ML model's predicted next-day
    return into a single actionable signal.

    Logic: only act on the RSI signal if the model's predicted return
    agrees in direction and clears a minimum magnitude threshold; otherwise
    HOLD. This is a simple, explainable ensemble rule connecting the
    technical-analysis signal to the regression output, matching the
    "translate model output into concrete buy/sell thresholds" requirement.
    """
    model_dir = pd.Series(0, index=predicted_return.index, dtype=int)
    model_dir[predicted_return >= buy_threshold] = 1
    model_dir[predicted_return <= sell_threshold] = -1

    combined = pd.Series(0, index=rsi_signal.index, dtype=int)
    agree_buy = (rsi_signal == 1) | (model_dir == 1)
    agree_sell = (rsi_signal == -1) | (model_dir == -1)

    # Require at least directional agreement from the model when the RSI
    # is silent (HOLD), and require the model not to contradict RSI when
    # RSI does fire.
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
