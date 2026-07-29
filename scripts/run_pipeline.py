#!/usr/bin/env python3
"""
CLI entry point for the RSI + ML stock prediction pipeline.

Usage:
    python scripts/run_pipeline.py --ticker AAPL
    python scripts/run_pipeline.py --ticker MSFT --real-data
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import run  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Run the RSI + ML stock prediction pipeline.")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Ticker symbol (used for labeling / real-data mode).")
    parser.add_argument(
        "--real-data",
        action="store_true",
        help="Attempt to pull real historical data via yfinance (requires network + `pip install yfinance`). "
        "Falls back to synthetic data automatically if unavailable.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    report = run(ticker=args.ticker, prefer_real=args.real_data)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
