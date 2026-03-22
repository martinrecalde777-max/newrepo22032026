"""Candle morphology feature engine calibrated to real MNQ 1-minute data.

Calibration source: glbx-mdp3-20210312-20260311, 1.77M continuous front-month bars.

Key MNQ 1-min statistics:
    range:      mean=7.0 pts, median=5.0 pts, p95=19.5 pts, p99=34.0 pts
    body:       mean=3.5 pts, median=2.0 pts, p95=11.5 pts
    body_ratio: mean=0.46, median=0.46, p10=0.10, p90=0.82, p95=0.89
    upper_wick: mean=0.27, p50=0.22, p95=0.68
    lower_wick: mean=0.27, p50=0.23, p95=0.69
    direction:  48.4% bullish, 47.6% bearish, 4.0% flat
    volume:     median=334, mean=925, p95=3829
    tick_size:  0.25 points
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# MNQ tick size
TICK = 0.25


def compute_morphology(df: pd.DataFrame) -> pd.DataFrame:
    """Compute candle morphology features on OHLCV data.

    Parameters
    ----------
    df : DataFrame with columns open, high, low, close, volume

    Returns
    -------
    DataFrame with original columns + all morphology features.
    """
    out = df.copy()
    o, h, l, c, v = out["open"], out["high"], out["low"], out["close"], out["volume"]

    # === Candle anatomy ===
    out["body"] = c - o                          # signed body
    out["body_abs"] = out["body"].abs()           # unsigned body
    out["range"] = h - l                          # full candle range
    out["upper_wick"] = h - np.maximum(o, c)      # wick above body
    out["lower_wick"] = np.minimum(o, c) - l      # wick below body
    out["midpoint"] = (h + l) / 2

    # === Ratios (NaN for zero-range bars, <0.02% of data) ===
    rng = out["range"].replace(0, np.nan)
    out["body_ratio"] = out["body_abs"] / rng
    out["upper_wick_ratio"] = out["upper_wick"] / rng
    out["lower_wick_ratio"] = out["lower_wick"] / rng
    out["wick_imbalance"] = (out["upper_wick"] - out["lower_wick"]) / rng

    # === Direction & strength ===
    out["direction"] = np.sign(out["body"]).astype("int8")
    out["body_ticks"] = out["body_abs"] / TICK     # body in ticks
    out["range_ticks"] = out["range"] / TICK       # range in ticks

    # === Volatility ===
    prev_c = c.shift(1)
    tr1 = h - l
    tr2 = (h - prev_c).abs()
    tr3 = (l - prev_c).abs()
    out["true_range"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    out["atr_5"] = out["true_range"].rolling(5, min_periods=1).mean()
    out["atr_20"] = out["true_range"].rolling(20, min_periods=1).mean()
    out["range_vs_atr20"] = out["range"] / out["atr_20"].replace(0, np.nan)

    # === Momentum ===
    out["close_chg"] = c.diff()
    out["close_pct"] = c.pct_change()
    out["gap"] = o - prev_c                       # gap from prior close

    # === Volume context ===
    out["vol_ma5"] = v.rolling(5, min_periods=1).mean()
    out["vol_ma20"] = v.rolling(20, min_periods=1).mean()
    out["vol_ratio"] = v / out["vol_ma20"].replace(0, np.nan)

    # === Streak (vectorised) ===
    out["streak"] = _streak_vectorised(out["direction"].values)

    return out


def _streak_vectorised(direction: np.ndarray) -> np.ndarray:
    """Count consecutive same-direction candles (signed)."""
    n = len(direction)
    streak = np.zeros(n, dtype="int64")
    if n == 0:
        return streak
    streak[0] = direction[0]
    for i in range(1, n):
        d = direction[i]
        if d == 0:
            streak[i] = 0
        elif d == np.sign(streak[i - 1]) or streak[i - 1] == 0:
            streak[i] = streak[i - 1] + d
        else:
            streak[i] = d
    return streak
