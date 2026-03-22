"""Tests for morphology features and pattern detection."""

import numpy as np
import pandas as pd
import pytest

from mnq_morphology.features.morphology import compute_morphology
from mnq_morphology.patterns.detector import detect_patterns, PATTERN_REGISTRY
from mnq_morphology.aggregators.timeframe import aggregate_timeframe


def _make_ohlcv(rows: list[tuple], freq: str = "1min") -> pd.DataFrame:
    """Helper: build a small OHLCV DataFrame from (o, h, l, c, v) tuples."""
    idx = pd.date_range("2025-01-06 09:30", periods=len(rows), freq=freq, tz="US/Eastern")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=idx)
    df.index.name = "datetime"
    return df


# ---------------------------------------------------------------------------
# Morphology features
# ---------------------------------------------------------------------------

class TestMorphology:
    def test_bullish_candle(self):
        df = _make_ohlcv([(100, 105, 99, 104, 1000)])
        m = compute_morphology(df)
        assert m["body"].iloc[0] == 4.0  # close - open
        assert m["direction"].iloc[0] == 1
        assert m["range"].iloc[0] == 6.0
        assert m["upper_wick"].iloc[0] == 1.0  # 105 - max(100,104)
        assert m["lower_wick"].iloc[0] == 1.0  # min(100,104) - 99

    def test_bearish_candle(self):
        df = _make_ohlcv([(104, 105, 99, 100, 1000)])
        m = compute_morphology(df)
        assert m["body"].iloc[0] == -4.0
        assert m["direction"].iloc[0] == -1

    def test_doji_candle(self):
        df = _make_ohlcv([(100, 105, 95, 100, 500)])
        m = compute_morphology(df)
        assert m["body"].iloc[0] == 0.0
        assert m["direction"].iloc[0] == 0
        assert m["body_ratio"].iloc[0] == 0.0

    def test_body_ratio_range(self):
        """body_ratio should be between 0 and 1."""
        df = _make_ohlcv([
            (100, 110, 90, 108, 100),
            (108, 109, 107, 107.5, 200),
            (107.5, 115, 105, 115, 300),
        ])
        m = compute_morphology(df)
        assert (m["body_ratio"].dropna() >= 0).all()
        assert (m["body_ratio"].dropna() <= 1).all()

    def test_streak(self):
        df = _make_ohlcv([
            (100, 102, 99, 101, 100),   # +1
            (101, 104, 100, 103, 100),   # +1
            (103, 106, 102, 105, 100),   # +1
            (105, 106, 100, 101, 100),   # -1
        ])
        m = compute_morphology(df)
        assert list(m["streak"]) == [1, 2, 3, -1]


# ---------------------------------------------------------------------------
# Pattern detection
# ---------------------------------------------------------------------------

class TestPatterns:
    def test_doji_detected(self):
        df = _make_ohlcv([(100, 110, 90, 100, 500)])
        m = compute_morphology(df)
        p = detect_patterns(m, ["doji"])
        assert p["pat_doji"].iloc[0] is np.True_

    def test_marubozu_detected(self):
        # Almost all body, minimal wicks
        df = _make_ohlcv([(100, 110.1, 99.9, 110, 1000)])
        m = compute_morphology(df)
        p = detect_patterns(m, ["marubozu"])
        assert p["pat_marubozu"].iloc[0] is np.True_

    def test_hammer_detected(self):
        # Small body at top, long lower wick
        df = _make_ohlcv([(109, 110, 100, 110, 500)])
        m = compute_morphology(df)
        p = detect_patterns(m, ["hammer"])
        assert p["pat_hammer"].iloc[0] is np.True_

    def test_engulfing_bullish(self):
        df = _make_ohlcv([
            (105, 106, 103, 104, 100),  # bearish
            (103, 107, 102, 106, 200),  # bullish engulfs
        ])
        m = compute_morphology(df)
        p = detect_patterns(m, ["engulfing_bullish"])
        assert p["pat_engulfing_bullish"].iloc[1] is np.True_

    def test_morning_star(self):
        df = _make_ohlcv([
            (110, 111, 100, 101, 300),  # big bearish (body_ratio ~0.82)
            (101, 101.3, 100.7, 101.1, 100),  # tiny star (body_ratio ~0.17)
            (101.1, 111, 101, 110, 300),  # big bullish (body_ratio ~0.89)
        ])
        m = compute_morphology(df)
        p = detect_patterns(m, ["morning_star"])
        assert p["pat_morning_star"].iloc[2] is np.True_

    def test_all_patterns_run(self):
        """All registered patterns should execute without error."""
        df = _make_ohlcv([(100 + i, 105 + i, 95 + i, 102 + i, 100) for i in range(20)])
        m = compute_morphology(df)
        p = detect_patterns(m)
        assert len(p.columns) == len(PATTERN_REGISTRY)

    def test_unknown_pattern_raises(self):
        df = _make_ohlcv([(100, 105, 95, 102, 100)])
        m = compute_morphology(df)
        with pytest.raises(ValueError, match="Unknown patterns"):
            detect_patterns(m, ["nonexistent_pattern"])


# ---------------------------------------------------------------------------
# Timeframe aggregation
# ---------------------------------------------------------------------------

class TestAggregation:
    def test_5min_aggregation(self):
        rows = [(100 + i, 105 + i, 95 + i, 102 + i, 100) for i in range(10)]
        df = _make_ohlcv(rows)
        agg = aggregate_timeframe(df, "5min")
        assert len(agg) == 2
        # First 5-min bar: open of first bar, high of highest, etc.
        assert agg["open"].iloc[0] == 100
        assert agg["high"].iloc[0] == 109  # max of 105..109
        assert agg["low"].iloc[0] == 95    # min of 95..99
        assert agg["close"].iloc[0] == 106  # close of 5th bar
        assert agg["volume"].iloc[0] == 500

    def test_aggregation_preserves_ohlcv(self):
        rows = [(100, 110, 90, 105, 200)] * 30
        df = _make_ohlcv(rows)
        agg = aggregate_timeframe(df, "1h")
        assert len(agg) == 1
        assert agg["open"].iloc[0] == 100
        assert agg["high"].iloc[0] == 110
        assert agg["low"].iloc[0] == 90
        assert agg["close"].iloc[0] == 105
        assert agg["volume"].iloc[0] == 6000
