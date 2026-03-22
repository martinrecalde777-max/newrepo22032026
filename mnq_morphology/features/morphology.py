"""Candle morphology feature engineering.

Computes structural features of each candlestick that characterise its
shape ("morphology"): body size, wick ratios, direction, momentum proxies,
and volatility metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_morphology(df: pd.DataFrame) -> pd.DataFrame:
    """Append morphology features to an OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: open, high, low, close, volume.

    Returns
    -------
    pd.DataFrame
        Original columns plus all morphology features.
    """
    out = df.copy()

    o, h, l, c = out["open"], out["high"], out["low"], out["close"]

    # --- Basic candle anatomy -------------------------------------------------
    out["body"] = c - o  # signed body (positive = bullish)
    out["body_abs"] = out["body"].abs()
    out["range"] = h - l  # full candle range
    out["upper_wick"] = h - np.maximum(o, c)
    out["lower_wick"] = np.minimum(o, c) - l
    out["midpoint"] = (h + l) / 2

    # --- Ratios (safe division) -----------------------------------------------
    range_safe = out["range"].replace(0, np.nan)

    out["body_ratio"] = out["body_abs"] / range_safe  # body as % of range
    out["upper_wick_ratio"] = out["upper_wick"] / range_safe
    out["lower_wick_ratio"] = out["lower_wick"] / range_safe
    out["wick_imbalance"] = (out["upper_wick"] - out["lower_wick"]) / range_safe

    # --- Direction & strength -------------------------------------------------
    out["direction"] = np.sign(out["body"]).astype("int8")  # +1, 0, -1
    out["body_pct"] = out["body"] / o.replace(0, np.nan)  # % move within candle

    # --- Volatility proxies ---------------------------------------------------
    out["true_range"] = _true_range(h, l, c)
    out["range_ma5"] = out["range"].rolling(5, min_periods=1).mean()
    out["range_ma20"] = out["range"].rolling(20, min_periods=1).mean()
    out["range_z"] = (out["range"] - out["range_ma20"]) / out["range"].rolling(
        20, min_periods=1
    ).std().replace(0, np.nan)

    # --- Momentum proxies -----------------------------------------------------
    out["close_change"] = c.diff()
    out["close_pct_change"] = c.pct_change()
    out["gap"] = o - c.shift(1)  # gap from prior close

    # --- Volume features ------------------------------------------------------
    out["vol_ma5"] = out["volume"].rolling(5, min_periods=1).mean()
    out["vol_ma20"] = out["volume"].rolling(20, min_periods=1).mean()
    out["vol_ratio"] = out["volume"] / out["vol_ma20"].replace(0, np.nan)
    out["dollar_volume"] = out["volume"] * out["midpoint"]

    # --- Streak tracking ------------------------------------------------------
    out["streak"] = _streak(out["direction"])

    return out


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Compute True Range (handles gaps via prior close)."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def _streak(direction: pd.Series) -> pd.Series:
    """Count consecutive same-direction candles (signed)."""
    streak = pd.Series(0, index=direction.index, dtype="int64")
    prev = 0
    for i in range(len(direction)):
        d = direction.iat[i]
        if d == 0:
            prev = 0
        elif i == 0:
            prev = d
        elif d == direction.iat[i - 1]:
            prev = prev + d  # accumulates sign
        else:
            prev = d
        streak.iat[i] = prev
    return streak
