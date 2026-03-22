"""Multi-timeframe OHLCV aggregator for MNQ data."""

from __future__ import annotations

import pandas as pd

_AGG_RULES = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}

TIMEFRAMES = ("5min", "15min", "1h", "4h", "1D", "1W")


def aggregate(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample OHLCV to a coarser timeframe."""
    ohlcv = ["open", "high", "low", "close", "volume"]
    cols = [c for c in ohlcv if c in df.columns]
    rules = {c: _AGG_RULES[c] for c in cols}
    out = df[cols].resample(timeframe).agg(rules).dropna(subset=["open"])
    # Carry symbol as the last seen contract
    if "symbol" in df.columns:
        out["symbol"] = df["symbol"].resample(timeframe).last().reindex(out.index)
    return out


def aggregate_all(df: pd.DataFrame, timeframes: tuple[str, ...] = TIMEFRAMES) -> dict[str, pd.DataFrame]:
    """Resample to all standard timeframes. Returns dict keyed by timeframe."""
    return {tf: aggregate(df, tf) for tf in timeframes}
