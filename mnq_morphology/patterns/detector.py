"""Candlestick pattern detector.

Identifies classic single-candle and multi-candle patterns from morphology
features.  Each detector returns a boolean Series; `detect_patterns` runs
them all and returns a DataFrame of pattern flags.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Single-candle patterns
# ---------------------------------------------------------------------------

def _is_doji(df: pd.DataFrame, threshold: float = 0.05) -> pd.Series:
    """Body is very small relative to range."""
    return df["body_ratio"].fillna(0) < threshold


def _is_hammer(df: pd.DataFrame, body_max: float = 0.33, wick_min: float = 0.60) -> pd.Series:
    """Small body near the top, long lower wick."""
    return (
        (df["body_ratio"].fillna(0) < body_max)
        & (df["lower_wick_ratio"].fillna(0) > wick_min)
        & (df["upper_wick_ratio"].fillna(0) < 0.15)
    )


def _is_inverted_hammer(df: pd.DataFrame, body_max: float = 0.33, wick_min: float = 0.60) -> pd.Series:
    """Small body near the bottom, long upper wick."""
    return (
        (df["body_ratio"].fillna(0) < body_max)
        & (df["upper_wick_ratio"].fillna(0) > wick_min)
        & (df["lower_wick_ratio"].fillna(0) < 0.15)
    )


def _is_marubozu(df: pd.DataFrame, body_min: float = 0.90) -> pd.Series:
    """Nearly all body, minimal wicks — strong conviction candle."""
    return df["body_ratio"].fillna(0) > body_min


def _is_spinning_top(df: pd.DataFrame, body_max: float = 0.30) -> pd.Series:
    """Small body with roughly equal wicks on both sides."""
    return (
        (df["body_ratio"].fillna(0) < body_max)
        & ((df["upper_wick_ratio"].fillna(0) - df["lower_wick_ratio"].fillna(0)).abs() < 0.15)
        & ~_is_doji(df)
    )


def _is_high_wave(df: pd.DataFrame) -> pd.Series:
    """Extremely long wicks on both sides with tiny body — indecision."""
    return (
        (df["body_ratio"].fillna(0) < 0.15)
        & (df["upper_wick_ratio"].fillna(0) > 0.35)
        & (df["lower_wick_ratio"].fillna(0) > 0.35)
    )


# ---------------------------------------------------------------------------
# Two-candle patterns
# ---------------------------------------------------------------------------

def _is_engulfing_bullish(df: pd.DataFrame) -> pd.Series:
    """Current bullish candle fully engulfs prior bearish candle."""
    prev_dir = df["direction"].shift(1)
    return (
        (df["direction"] == 1)
        & (prev_dir == -1)
        & (df["open"] <= df["close"].shift(1))
        & (df["close"] >= df["open"].shift(1))
    )


def _is_engulfing_bearish(df: pd.DataFrame) -> pd.Series:
    """Current bearish candle fully engulfs prior bullish candle."""
    prev_dir = df["direction"].shift(1)
    return (
        (df["direction"] == -1)
        & (prev_dir == 1)
        & (df["open"] >= df["close"].shift(1))
        & (df["close"] <= df["open"].shift(1))
    )


def _is_tweezer_top(df: pd.DataFrame, tol: float = 0.0002) -> pd.Series:
    """Two candles with nearly identical highs at a local high."""
    return (
        ((df["high"] - df["high"].shift(1)).abs() / df["high"].replace(0, np.nan) < tol)
        & (df["direction"].shift(1) == 1)
        & (df["direction"] == -1)
    )


def _is_tweezer_bottom(df: pd.DataFrame, tol: float = 0.0002) -> pd.Series:
    """Two candles with nearly identical lows at a local low."""
    return (
        ((df["low"] - df["low"].shift(1)).abs() / df["low"].replace(0, np.nan) < tol)
        & (df["direction"].shift(1) == -1)
        & (df["direction"] == 1)
    )


# ---------------------------------------------------------------------------
# Three-candle patterns
# ---------------------------------------------------------------------------

def _is_morning_star(df: pd.DataFrame) -> pd.Series:
    """Bearish → small body (star) → bullish reversal."""
    d1 = df["direction"].shift(2)
    d3 = df["direction"]
    star_body = df["body_ratio"].shift(1).fillna(0)
    big1 = df["body_ratio"].shift(2).fillna(0)
    big3 = df["body_ratio"].fillna(0)
    return (d1 == -1) & (star_body < 0.20) & (d3 == 1) & (big1 > 0.50) & (big3 > 0.50)


def _is_evening_star(df: pd.DataFrame) -> pd.Series:
    """Bullish → small body (star) → bearish reversal."""
    d1 = df["direction"].shift(2)
    d3 = df["direction"]
    star_body = df["body_ratio"].shift(1).fillna(0)
    big1 = df["body_ratio"].shift(2).fillna(0)
    big3 = df["body_ratio"].fillna(0)
    return (d1 == 1) & (star_body < 0.20) & (d3 == -1) & (big1 > 0.50) & (big3 > 0.50)


def _is_three_white_soldiers(df: pd.DataFrame) -> pd.Series:
    """Three consecutive bullish candles with strong bodies."""
    return (
        (df["direction"] == 1)
        & (df["direction"].shift(1) == 1)
        & (df["direction"].shift(2) == 1)
        & (df["body_ratio"].fillna(0) > 0.55)
        & (df["body_ratio"].shift(1).fillna(0) > 0.55)
        & (df["body_ratio"].shift(2).fillna(0) > 0.55)
        & (df["close"] > df["close"].shift(1))
        & (df["close"].shift(1) > df["close"].shift(2))
    )


def _is_three_black_crows(df: pd.DataFrame) -> pd.Series:
    """Three consecutive bearish candles with strong bodies."""
    return (
        (df["direction"] == -1)
        & (df["direction"].shift(1) == -1)
        & (df["direction"].shift(2) == -1)
        & (df["body_ratio"].fillna(0) > 0.55)
        & (df["body_ratio"].shift(1).fillna(0) > 0.55)
        & (df["body_ratio"].shift(2).fillna(0) > 0.55)
        & (df["close"] < df["close"].shift(1))
        & (df["close"].shift(1) < df["close"].shift(2))
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

PATTERN_REGISTRY: dict[str, callable] = {
    # single
    "doji": _is_doji,
    "hammer": _is_hammer,
    "inverted_hammer": _is_inverted_hammer,
    "marubozu": _is_marubozu,
    "spinning_top": _is_spinning_top,
    "high_wave": _is_high_wave,
    # two-candle
    "engulfing_bullish": _is_engulfing_bullish,
    "engulfing_bearish": _is_engulfing_bearish,
    "tweezer_top": _is_tweezer_top,
    "tweezer_bottom": _is_tweezer_bottom,
    # three-candle
    "morning_star": _is_morning_star,
    "evening_star": _is_evening_star,
    "three_white_soldiers": _is_three_white_soldiers,
    "three_black_crows": _is_three_black_crows,
}


def detect_patterns(
    df: pd.DataFrame,
    patterns: list[str] | None = None,
) -> pd.DataFrame:
    """Run pattern detectors and return a DataFrame of boolean flags.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame **with morphology features already computed**.
    patterns : list[str] or None
        Subset of pattern names to detect.  None = all.

    Returns
    -------
    pd.DataFrame
        One boolean column per pattern, same index as *df*.
    """
    if patterns is None:
        patterns = list(PATTERN_REGISTRY)

    unknown = set(patterns) - set(PATTERN_REGISTRY)
    if unknown:
        raise ValueError(f"Unknown patterns: {unknown}")

    result = pd.DataFrame(index=df.index)
    for name in patterns:
        result[f"pat_{name}"] = PATTERN_REGISTRY[name](df).astype(bool)

    return result
