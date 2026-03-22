"""Loader and continuous contract builder for Databento MNQ OHLCV-1m parquet.

Real data characteristics (glbx-mdp3-20210312-20260311.ohlcv-1m.parquet):
- 2,823,464 total bars (2,772,393 pure contracts + ~51K spreads)
- 25 pure quarterly contracts: MNQH1 through MNQH7
- ~70 symbols total including calendar spreads (e.g. MNQH2-MNQM2)
- Prices: 13,000–27,000 range, 0.25 tick size
- Volume: 1–26,825 per 1-min bar
- Date range: 2021-03-12 to 2026-03-11 UTC
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_parquet(
    path: str | Path = "data/glbx-mdp3-20210312-20260311.ohlcv-1m.parquet",
    *,
    tz: str = "US/Eastern",
    drop_spreads: bool = True,
) -> pd.DataFrame:
    """Load the Databento MNQ parquet file.

    Parameters
    ----------
    path : path to parquet file
    tz : timezone for index conversion (CME default: US/Eastern)
    drop_spreads : if True, remove calendar spread symbols (contain '-')

    Returns
    -------
    DataFrame with DatetimeIndex, columns: open, high, low, close, volume, symbol
    """
    path = Path(path)
    df = pd.read_parquet(path)

    # Parse Databento ts_event → DatetimeIndex
    df["datetime"] = pd.to_datetime(df["ts_event"], utc=True)
    df = df.set_index("datetime").sort_index()

    if tz:
        df.index = df.index.tz_convert(tz)

    # Drop Databento metadata columns, keep OHLCV + symbol
    df = df[["open", "high", "low", "close", "volume", "symbol"]].copy()

    # Remove calendar spreads (negative prices, not useful for morphology)
    if drop_spreads:
        df = df[~df["symbol"].str.contains("-")]

    return df


def build_continuous(df: pd.DataFrame) -> pd.DataFrame:
    """Build continuous front-month series by selecting the highest-volume
    contract per day.

    Parameters
    ----------
    df : multi-symbol OHLCV from load_parquet()

    Returns
    -------
    Single continuous OHLCV series with 'symbol' tracking which contract was active.
    """
    # Daily volume per symbol
    daily_vol = df.groupby([df.index.date, "symbol"])["volume"].sum()
    daily_vol.index.names = ["date", "symbol"]

    # Front-month = highest daily volume per date
    front = daily_vol.groupby("date").idxmax().apply(lambda x: x[1])
    front_map = front.to_dict()  # {date: symbol}

    # Filter: keep only bars where symbol == front-month for that date
    df = df.copy()
    dates = pd.Series(df.index.date, index=df.index)
    expected_symbol = dates.map(front_map)
    result = df[df["symbol"].values == expected_symbol.values]

    return result
