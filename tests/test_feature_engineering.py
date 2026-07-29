import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.feature_engineering import compute_rsi, build_feature_frame, FEATURE_COLUMNS, TARGET_COLUMN


def _make_price_series(n=100, seed=1):
    rng = np.random.default_rng(seed)
    prices = 100 + np.cumsum(rng.normal(0, 1, n))
    prices = np.clip(prices, 1, None)
    idx = pd.bdate_range("2023-01-02", periods=n)
    return pd.Series(prices, index=idx)


def test_rsi_bounded_between_0_and_100():
    prices = _make_price_series()
    rsi = compute_rsi(prices)
    assert rsi.min() >= 0
    assert rsi.max() <= 100


def test_rsi_is_100_when_only_gains():
    prices = pd.Series(np.arange(1, 30, dtype=float))
    rsi = compute_rsi(prices, period=14)
    # After enough consecutive gains, RSI should approach 100
    assert rsi.iloc[-1] > 95


def test_rsi_is_near_0_when_only_losses():
    prices = pd.Series(np.arange(30, 1, -1, dtype=float))
    rsi = compute_rsi(prices, period=14)
    assert rsi.iloc[-1] < 5


def test_build_feature_frame_has_expected_columns():
    idx = pd.bdate_range("2023-01-02", periods=60)
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 1, 60))
    df = pd.DataFrame(
        {
            "open": close + rng.normal(0, 0.5, 60),
            "high": close + np.abs(rng.normal(0, 1, 60)),
            "low": close - np.abs(rng.normal(0, 1, 60)),
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, 60),
        },
        index=idx,
    )
    feats = build_feature_frame(df)
    for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
        assert col in feats.columns


def test_target_is_shifted_return():
    idx = pd.bdate_range("2023-01-02", periods=5)
    df = pd.DataFrame(
        {
            "open": [10, 11, 12, 13, 14],
            "high": [10, 11, 12, 13, 14],
            "low": [10, 11, 12, 13, 14],
            "close": [10, 11, 12, 13, 14],
            "volume": [100, 100, 100, 100, 100],
        },
        index=idx,
    )
    feats = build_feature_frame(df)
    expected_first_target = 12 / 11 - 1  # next day's return at row index 1
    assert feats[TARGET_COLUMN].iloc[1] == pytest.approx(expected_first_target)
    assert pd.isna(feats[TARGET_COLUMN].iloc[-1])  # last row has no "next day"
