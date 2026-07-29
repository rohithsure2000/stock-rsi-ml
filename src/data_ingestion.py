"""Loads OHLCV data - real via yfinance, or a synthetic fallback so the
pipeline runs without network access."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_from_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Pull real OHLCV data via yfinance. Raises if unavailable - caller
    should catch and fall back to synthetic data."""
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for ticker={ticker!r} in range {start}..{end}")

    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    df.index.name = "date"
    return df[["open", "high", "low", "close", "volume"]]


def generate_synthetic_ohlcv(
    n_days: int = 756,
    start_price: float = 150.0,
    annual_drift: float = 0.14,
    annual_vol: float = 0.28,
    vol_persistence: float = 0.90,
    vol_reversion: float = 0.06,
    long_run_var: float | None = None,
    base_volume: int = 55_000_000,
    seed: int = 42,
    start_date: str = "2022-01-03",
) -> pd.DataFrame:
    """GARCH(1,1)-style volatility on top of GBM, so it clusters like a
    real stock instead of constant-vol random walk. Defaults are roughly
    AAPL-shaped but tunable."""
    rng = np.random.default_rng(seed)

    dt = 1 / 252
    if long_run_var is None:
        long_run_var = annual_vol ** 2

    # variance recursion (GARCH-ish)
    variances = np.empty(n_days)
    variances[0] = long_run_var
    shocks = rng.standard_normal(n_days)
    for t in range(1, n_days):
        prev_resid_sq = (shocks[t - 1] ** 2) * variances[t - 1]
        variances[t] = (
            long_run_var * vol_reversion
            + vol_persistence * variances[t - 1]
            + (1 - vol_persistence - vol_reversion) * prev_resid_sq
        )
        variances[t] = max(variances[t], 1e-6)

    daily_vol = np.sqrt(variances * dt)
    daily_drift = (annual_drift - 0.5 * variances) * dt

    log_returns = daily_drift + daily_vol * shocks
    close = start_price * np.exp(np.cumsum(log_returns))

    # open/high/low from the close path
    prev_close = np.concatenate([[start_price], close[:-1]])
    intraday_range_pct = np.abs(rng.normal(loc=daily_vol * 0.6, scale=daily_vol * 0.25))
    intraday_range_pct = np.clip(intraday_range_pct, 0.001, None)

    open_ = prev_close * (1 + rng.normal(0, daily_vol * 0.15))
    high = np.maximum(open_, close) * (1 + intraday_range_pct * rng.uniform(0.3, 1.0, n_days))
    low = np.minimum(open_, close) * (1 - intraday_range_pct * rng.uniform(0.3, 1.0, n_days))
    low = np.clip(low, 0.01, None)

    # bigger volume on bigger move days
    abs_ret_z = np.abs(log_returns) / (daily_vol + 1e-9)
    volume = base_volume * (1 + 0.35 * abs_ret_z) * rng.lognormal(mean=0, sigma=0.18, size=n_days)
    volume = volume.astype(np.int64)

    dates = pd.bdate_range(start=start_date, periods=n_days)

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )
    df.index.name = "date"
    df = df.round({"open": 2, "high": 2, "low": 2, "close": 2})
    return df


def get_dataset(
    ticker: str = "AAPL",
    prefer_real: bool = False,
    start: str = "2022-01-01",
    end: str = "2024-12-31",
    cache_path: str | Path = "data/raw/ohlcv.csv",
) -> pd.DataFrame:
    """Real data if prefer_real=True and reachable, else synthetic.
    Caches to cache_path either way."""
    cache_path = Path(cache_path)

    if prefer_real:
        try:
            df = load_from_yfinance(ticker, start, end)
            logger.info("Loaded real market data for %s via yfinance (%d rows)", ticker, len(df))
            df.to_csv(cache_path)
            return df
        except Exception as exc:  # noqa: BLE001
            logger.warning("Falling back to synthetic data: %s", exc)

    df = generate_synthetic_ohlcv(start_date=start)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path)
    logger.info("Generated synthetic OHLCV dataset (%d rows) -> %s", len(df), cache_path)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data = get_dataset()
    print(data.head())
    print(data.describe())
