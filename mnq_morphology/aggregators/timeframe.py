"""Multi-timeframe OHLCV aggregator.

Resamples 1-minute bars into higher timeframes (5m, 15m, 1H, 4H, 1D, 1W)
using standard OHLCV aggregation rules.
"""

from __future__ import annotations

import pandas as pd

# Standard OHLCV resampling rules
_AGG_RULES = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}

TIMEFRAMES = ("5min", "15min", "1h", "4h", "1D", "1W")


def aggregate_timeframe(
    df: pd.DataFrame,
    timeframe: str,
) -> pd.DataFrame:
    """Resample an OHLCV DataFrame to a coarser timeframe.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a DatetimeIndex and columns: open, high, low, close, volume.
    timeframe : str
        Pandas offset alias, e.g. '5min', '15min', '1h', '4h', '1D', '1W'.

    Returns
    -------
    pd.DataFrame
        Resampled OHLCV with the same column layout.
    """
    ohlcv_cols = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    rules = {c: _AGG_RULES[c] for c in ohlcv_cols}
    resampled = df[ohlcv_cols].resample(timeframe).agg(rules).dropna(subset=["open"])
    return resampled


def aggregate_all_timeframes(
    df: pd.DataFrame,
    timeframes: tuple[str, ...] = TIMEFRAMES,
) -> dict[str, pd.DataFrame]:
    """Resample to all standard timeframes.

    Returns a dict keyed by timeframe string.
    """
    return {tf: aggregate_timeframe(df, tf) for tf in timeframes}
