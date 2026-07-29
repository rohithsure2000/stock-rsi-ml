import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.signal_generation import rsi_70_30_signal, combine_with_model_prediction


def test_rsi_70_30_buy_on_upward_cross_through_30():
    idx = pd.RangeIndex(5)
    rsi = pd.Series([25, 28, 29, 32, 40], index=idx, dtype=float)
    sig = rsi_70_30_signal(rsi)
    assert sig.iloc[3] == 1  # crosses from below 30 to >= 30


def test_rsi_70_30_sell_on_downward_cross_through_70():
    idx = pd.RangeIndex(5)
    rsi = pd.Series([75, 74, 72, 68, 60], index=idx, dtype=float)
    sig = rsi_70_30_signal(rsi)
    assert sig.iloc[3] == -1  # crosses from above 70 to <= 70


def test_rsi_70_30_hold_when_no_cross():
    idx = pd.RangeIndex(5)
    rsi = pd.Series([50, 51, 52, 53, 54], index=idx, dtype=float)
    sig = rsi_70_30_signal(rsi)
    assert (sig == 0).all()


def test_combine_with_model_prediction_requires_no_contradiction():
    idx = pd.RangeIndex(3)
    rsi_signal = pd.Series([1, -1, 0], index=idx)
    predicted_return = pd.Series([0.01, 0.01, 0.01], index=idx)  # bullish model
    combined = combine_with_model_prediction(rsi_signal, predicted_return)
    assert combined.iloc[0] == 1   # RSI buy + model agrees -> buy
    assert combined.iloc[1] == 0   # RSI sell but model bullish -> contradiction -> hold
    assert combined.iloc[2] == 1   # RSI silent, model bullish -> buy
