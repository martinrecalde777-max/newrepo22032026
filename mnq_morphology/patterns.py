"""Candlestick pattern detector calibrated to real MNQ 1-minute data.

Thresholds derived from actual distributions:
    doji:       body_ratio < 0.10 (p10 of real data)
    marubozu:   body_ratio > 0.89 (p95 of real data)
    small_body: body_ratio < 0.25 (p25)
    big_body:   body_ratio > 0.67 (p75)
    long_wick:  wick_ratio > 0.55 (roughly p85)
    high_vol:   vol_ratio > 2.0 (well above p75=1.21)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --- Thresholds calibrated to MNQ 1-min data ---
DOJI_BODY_MAX = 0.10          # p10 of body_ratio distribution
SMALL_BODY_MAX = 0.25         # p25
BIG_BODY_MIN = 0.67           # p75
MARUBOZU_BODY_MIN = 0.89      # p95
HAMMER_LONG_WICK = 0.55       # long wick threshold
HAMMER_SHORT_WICK = 0.10      # short wick threshold
TWEEZER_TOL_TICKS = 1.0       # 1 tick = 0.25 pts tolerance for tweezer matching
HIGH_VOL_RATIO = 2.0          # vol_ratio threshold for volume confirmation


# ============================================================
# Single-candle patterns
# ============================================================

def _doji(df: pd.DataFrame) -> pd.Series:
    """Tiny body relative to range — indecision."""
    return df["body_ratio"].fillna(0) < DOJI_BODY_MAX


def _hammer(df: pd.DataFrame) -> pd.Series:
    """Small body near top, long lower wick — potential bullish reversal."""
    return (
        (df["body_ratio"].fillna(0) < SMALL_BODY_MAX)
        & (df["lower_wick_ratio"].fillna(0) > HAMMER_LONG_WICK)
        & (df["upper_wick_ratio"].fillna(0) < HAMMER_SHORT_WICK)
    )


def _inverted_hammer(df: pd.DataFrame) -> pd.Series:
    """Small body near bottom, long upper wick."""
    return (
        (df["body_ratio"].fillna(0) < SMALL_BODY_MAX)
        & (df["upper_wick_ratio"].fillna(0) > HAMMER_LONG_WICK)
        & (df["lower_wick_ratio"].fillna(0) < HAMMER_SHORT_WICK)
    )


def _marubozu(df: pd.DataFrame) -> pd.Series:
    """Nearly all body, tiny wicks — strong conviction."""
    return df["body_ratio"].fillna(0) > MARUBOZU_BODY_MIN


def _spinning_top(df: pd.DataFrame) -> pd.Series:
    """Small body with roughly equal wicks."""
    wick_diff = (df["upper_wick_ratio"].fillna(0) - df["lower_wick_ratio"].fillna(0)).abs()
    return (
        (df["body_ratio"].fillna(0) < SMALL_BODY_MAX)
        & (wick_diff < 0.15)
        & (df["body_ratio"].fillna(0) >= DOJI_BODY_MAX)  # not a doji
    )


def _high_wave(df: pd.DataFrame) -> pd.Series:
    """Very long wicks both sides, tiny body — extreme indecision."""
    return (
        (df["body_ratio"].fillna(0) < DOJI_BODY_MAX)
        & (df["upper_wick_ratio"].fillna(0) > 0.35)
        & (df["lower_wick_ratio"].fillna(0) > 0.35)
    )


def _big_body_bullish(df: pd.DataFrame) -> pd.Series:
    """Large bullish candle (body > p75) — MNQ strong move up."""
    return (df["direction"] == 1) & (df["body_ratio"].fillna(0) > BIG_BODY_MIN)


def _big_body_bearish(df: pd.DataFrame) -> pd.Series:
    """Large bearish candle (body > p75) — MNQ strong move down."""
    return (df["direction"] == -1) & (df["body_ratio"].fillna(0) > BIG_BODY_MIN)


# ============================================================
# Two-candle patterns
# ============================================================

def _engulfing_bullish(df: pd.DataFrame) -> pd.Series:
    """Current bullish candle fully engulfs prior bearish candle body."""
    return (
        (df["direction"] == 1)
        & (df["direction"].shift(1) == -1)
        & (df["open"] <= df["close"].shift(1))
        & (df["close"] >= df["open"].shift(1))
    )


def _engulfing_bearish(df: pd.DataFrame) -> pd.Series:
    """Current bearish candle fully engulfs prior bullish candle body."""
    return (
        (df["direction"] == -1)
        & (df["direction"].shift(1) == 1)
        & (df["open"] >= df["close"].shift(1))
        & (df["close"] <= df["open"].shift(1))
    )


def _tweezer_top(df: pd.DataFrame) -> pd.Series:
    """Two candles with nearly identical highs, bull→bear reversal.
    Tolerance: 1 tick (0.25 pts) for MNQ."""
    return (
        ((df["high"] - df["high"].shift(1)).abs() <= TWEEZER_TOL_TICKS * 0.25)
        & (df["direction"].shift(1) == 1)
        & (df["direction"] == -1)
    )


def _tweezer_bottom(df: pd.DataFrame) -> pd.Series:
    """Two candles with nearly identical lows, bear→bull reversal."""
    return (
        ((df["low"] - df["low"].shift(1)).abs() <= TWEEZER_TOL_TICKS * 0.25)
        & (df["direction"].shift(1) == -1)
        & (df["direction"] == 1)
    )


# ============================================================
# Three-candle patterns
# ============================================================

def _morning_star(df: pd.DataFrame) -> pd.Series:
    """Bearish big body → small body star → bullish big body."""
    return (
        (df["direction"].shift(2) == -1)
        & (df["body_ratio"].shift(2).fillna(0) > BIG_BODY_MIN)
        & (df["body_ratio"].shift(1).fillna(0) < SMALL_BODY_MAX)
        & (df["direction"] == 1)
        & (df["body_ratio"].fillna(0) > BIG_BODY_MIN)
    )


def _evening_star(df: pd.DataFrame) -> pd.Series:
    """Bullish big body → small body star → bearish big body."""
    return (
        (df["direction"].shift(2) == 1)
        & (df["body_ratio"].shift(2).fillna(0) > BIG_BODY_MIN)
        & (df["body_ratio"].shift(1).fillna(0) < SMALL_BODY_MAX)
        & (df["direction"] == -1)
        & (df["body_ratio"].fillna(0) > BIG_BODY_MIN)
    )


def _three_white_soldiers(df: pd.DataFrame) -> pd.Series:
    """Three consecutive bullish big-body candles, each closing higher."""
    return (
        (df["direction"] == 1)
        & (df["direction"].shift(1) == 1)
        & (df["direction"].shift(2) == 1)
        & (df["body_ratio"].fillna(0) > BIG_BODY_MIN)
        & (df["body_ratio"].shift(1).fillna(0) > BIG_BODY_MIN)
        & (df["body_ratio"].shift(2).fillna(0) > BIG_BODY_MIN)
        & (df["close"] > df["close"].shift(1))
        & (df["close"].shift(1) > df["close"].shift(2))
    )


def _three_black_crows(df: pd.DataFrame) -> pd.Series:
    """Three consecutive bearish big-body candles, each closing lower."""
    return (
        (df["direction"] == -1)
        & (df["direction"].shift(1) == -1)
        & (df["direction"].shift(2) == -1)
        & (df["body_ratio"].fillna(0) > BIG_BODY_MIN)
        & (df["body_ratio"].shift(1).fillna(0) > BIG_BODY_MIN)
        & (df["body_ratio"].shift(2).fillna(0) > BIG_BODY_MIN)
        & (df["close"] < df["close"].shift(1))
        & (df["close"].shift(1) < df["close"].shift(2))
    )


# ============================================================
# Volume-confirmed variants
# ============================================================

def _vol_spike_reversal(df: pd.DataFrame) -> pd.Series:
    """Direction reversal on a high-volume bar (vol_ratio > 2x)."""
    return (
        (df["direction"] != df["direction"].shift(1))
        & (df["direction"] != 0)
        & (df["direction"].shift(1) != 0)
        & (df["vol_ratio"] > HIGH_VOL_RATIO)
    )


# ============================================================
# Registry & public API
# ============================================================

PATTERN_REGISTRY: dict[str, callable] = {
    "doji": _doji,
    "hammer": _hammer,
    "inverted_hammer": _inverted_hammer,
    "marubozu": _marubozu,
    "spinning_top": _spinning_top,
    "high_wave": _high_wave,
    "big_body_bullish": _big_body_bullish,
    "big_body_bearish": _big_body_bearish,
    "engulfing_bullish": _engulfing_bullish,
    "engulfing_bearish": _engulfing_bearish,
    "tweezer_top": _tweezer_top,
    "tweezer_bottom": _tweezer_bottom,
    "morning_star": _morning_star,
    "evening_star": _evening_star,
    "three_white_soldiers": _three_white_soldiers,
    "three_black_crows": _three_black_crows,
    "vol_spike_reversal": _vol_spike_reversal,
}


def detect_patterns(
    df: pd.DataFrame,
    patterns: list[str] | None = None,
) -> pd.DataFrame:
    """Run pattern detectors on morphology-enriched data.

    Parameters
    ----------
    df : DataFrame with morphology features (from compute_morphology)
    patterns : subset of pattern names, or None for all

    Returns
    -------
    DataFrame with one boolean column per pattern (pat_<name>).
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
