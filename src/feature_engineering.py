"""
feature_engineering.py

Computes RSI and a supporting set of technical indicators, then assembles
the feature matrix and regression target used by the modeling stage.
"""

from __future__ import annotations

import pandas as pd
import numpy as np


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Classic Wilder's RSI using an exponential (Wilder) moving average
    of gains and losses.
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # Edge cases where avg_loss == 0: RSI is 100 if there were gains, and
    # neutral (50) only in the degenerate case of literally no price change.
    all_flat = (avg_gain == 0) & (avg_loss == 0)
    only_gains = (avg_loss == 0) & (avg_gain > 0)
    rsi[only_gains] = 100.0
    rsi[all_flat] = 50.0
    rsi = rsi.fillna(50)  # remaining NaNs occur before `period` warmup rows
    return rsi


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def compute_bollinger_bands(close: pd.Series, window: int = 20, n_std: float = 2.0):
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    pct_b = (close - lower) / (upper - lower)
    return upper, mid, lower, pct_b


def build_feature_frame(df: pd.DataFrame, rsi_period: int = 14) -> pd.DataFrame:
    """Takes raw OHLCV data and returns a DataFrame of engineered features
    plus the regression target (`target_next_return`: next-day close-to-close
    percentage return).
    """
    out = df.copy()

    out["rsi_14"] = compute_rsi(out["close"], period=rsi_period)

    out["sma_5"] = out["close"].rolling(5).mean()
    out["sma_10"] = out["close"].rolling(10).mean()
    out["sma_20"] = out["close"].rolling(20).mean()
    out["ema_12"] = out["close"].ewm(span=12, adjust=False).mean()
    out["ema_26"] = out["close"].ewm(span=26, adjust=False).mean()

    macd_line, signal_line, hist = compute_macd(out["close"])
    out["macd"] = macd_line
    out["macd_signal"] = signal_line
    out["macd_hist"] = hist

    _, _, _, pct_b = compute_bollinger_bands(out["close"])
    out["bb_pct_b"] = pct_b

    out["daily_return"] = out["close"].pct_change()
    out["return_lag_1"] = out["daily_return"].shift(1)
    out["return_lag_2"] = out["daily_return"].shift(2)
    out["return_lag_3"] = out["daily_return"].shift(3)

    out["rolling_vol_10"] = out["daily_return"].rolling(10).std()
    out["volume_change"] = out["volume"].pct_change()
    out["volume_zscore_20"] = (
        out["volume"] - out["volume"].rolling(20).mean()
    ) / out["volume"].rolling(20).std()

    out["price_to_sma20"] = out["close"] / out["sma_20"] - 1
    out["sma5_sma20_spread"] = out["sma_5"] / out["sma_20"] - 1

    # Regression target: next trading day's close-to-close return.
    out["target_next_return"] = out["close"].pct_change().shift(-1)

    return out


FEATURE_COLUMNS = [
    "rsi_14",
    "sma_5",
    "sma_10",
    "sma_20",
    "ema_12",
    "ema_26",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_pct_b",
    "return_lag_1",
    "return_lag_2",
    "return_lag_3",
    "rolling_vol_10",
    "volume_change",
    "volume_zscore_20",
    "price_to_sma20",
    "sma5_sma20_spread",
]

TARGET_COLUMN = "target_next_return"


if __name__ == "__main__":
    from data_ingestion import get_dataset

    raw = get_dataset()
    feats = build_feature_frame(raw)
    print(feats[["close", "rsi_14", "macd", "bb_pct_b", TARGET_COLUMN]].tail(10))
    print("\nNaN rows to be dropped before modeling:", feats[FEATURE_COLUMNS + [TARGET_COLUMN]].isna().any(axis=1).sum())
