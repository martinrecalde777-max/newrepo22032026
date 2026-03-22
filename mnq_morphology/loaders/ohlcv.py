"""Loader for Databento glbx-mdp3 OHLCV 1-minute CSV files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Databento OHLCV-1m CSV columns (standard schema)
_DATABENTO_COLUMNS = {
    "ts_event": "datetime",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "int64",
}

# Fallback: generic OHLCV column names
_GENERIC_COLUMNS = ["datetime", "open", "high", "low", "close", "volume"]


def load_ohlcv(
    path: str | Path,
    *,
    tz: str = "US/Eastern",
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Load an OHLCV CSV and return a clean DataFrame indexed by datetime.

    Supports both Databento glbx-mdp3 schema and generic OHLCV CSVs.

    Parameters
    ----------
    path : str or Path
        Path to the CSV file.
    tz : str
        Timezone to localize/convert timestamps to (default US/Eastern for CME).
    start, end : str or None
        Optional ISO-8601 date strings to slice the data.

    Returns
    -------
    pd.DataFrame
        Columns: open, high, low, close, volume.  DatetimeIndex named 'datetime'.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"OHLCV file not found: {path}")

    df = pd.read_csv(path, low_memory=False)

    # --- Normalise column names -----------------------------------------------
    df.columns = df.columns.str.strip().str.lower()

    # Databento uses ts_event as the timestamp column
    if "ts_event" in df.columns:
        df = df.rename(columns={"ts_event": "datetime"})

    # Try to find a usable datetime column
    dt_col = None
    for candidate in ("datetime", "date", "timestamp", "time", "ts"):
        if candidate in df.columns:
            dt_col = candidate
            break

    if dt_col is None:
        raise ValueError(
            f"Cannot identify datetime column. Available: {list(df.columns)}"
        )

    df[dt_col] = pd.to_datetime(df[dt_col], utc=True)
    df = df.set_index(dt_col)
    df.index.name = "datetime"

    if tz:
        df.index = df.index.tz_convert(tz)

    # --- Keep only OHLCV columns ---------------------------------------------
    ohlcv = ["open", "high", "low", "close", "volume"]
    missing = [c for c in ohlcv if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}")

    df = df[ohlcv].copy()

    # Databento prices are in fixed-point (1e-9) — detect and convert
    if (df["close"] > 1_000_000).any():
        for col in ("open", "high", "low", "close"):
            df[col] = df[col] / 1e9

    df["volume"] = df["volume"].astype("int64")

    # --- Optional time slice --------------------------------------------------
    if start:
        df = df.loc[start:]
    if end:
        df = df.loc[:end]

    df = df.sort_index()
    return df
