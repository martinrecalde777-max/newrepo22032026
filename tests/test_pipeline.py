"""Tests for MNQ morphology pipeline using real data.

All tests use the actual Databento parquet file to ensure the pipeline
works correctly on real MNQ price action, not synthetic data.
"""

import numpy as np
import pandas as pd
import pytest

from mnq_morphology.loader import load_parquet, build_continuous
from mnq_morphology.morphology import compute_morphology, TICK
from mnq_morphology.patterns import detect_patterns, PATTERN_REGISTRY
from mnq_morphology.aggregator import aggregate
from mnq_morphology.statistics import compute_forward_returns, pattern_stats, by_session
from mnq_morphology.confluence import detect_confluence, confluence_stats, find_confluence_events
from mnq_morphology.group_morphology import (
    compute_window_features, classify_shape, group_forward_returns, Shape,
)

DATA_PATH = "data/glbx-mdp3-20210312-20260311.ohlcv-1m.parquet"


@pytest.fixture(scope="module")
def raw():
    """Load full raw dataset once per test module."""
    return load_parquet(DATA_PATH)


@pytest.fixture(scope="module")
def continuous(raw):
    """Build continuous front-month series once."""
    return build_continuous(raw)


@pytest.fixture(scope="module")
def morphology(continuous):
    """Compute morphology features once."""
    return compute_morphology(continuous)


# ============================================================
# Loader tests
# ============================================================

class TestLoader:
    def test_loads_correct_shape(self, raw):
        assert len(raw) == 2_772_393
        assert set(raw.columns) == {"open", "high", "low", "close", "volume", "symbol"}

    def test_no_spreads(self, raw):
        """Calendar spreads (symbols with '-') should be excluded."""
        assert not raw["symbol"].str.contains("-").any()

    def test_25_pure_contracts(self, raw):
        assert raw["symbol"].nunique() == 25

    def test_datetime_index(self, raw):
        assert raw.index.name == "datetime"
        assert raw.index.tz is not None  # timezone-aware

    def test_no_nans_in_ohlcv(self, raw):
        for col in ["open", "high", "low", "close", "volume"]:
            assert raw[col].isna().sum() == 0

    def test_price_range_sane(self, raw):
        """MNQ prices should be in 10,000-30,000 range (no fixed-point artifacts)."""
        assert raw["close"].min() > 10_000
        assert raw["close"].max() < 30_000

    def test_date_range(self, raw):
        assert raw.index.min().year == 2021
        assert raw.index.max().year == 2026


# ============================================================
# Continuous contract tests
# ============================================================

class TestContinuous:
    def test_single_symbol_per_timestamp(self, continuous):
        """Each timestamp should have exactly one bar (front-month only)."""
        dupes = continuous.index.duplicated().sum()
        assert dupes == 0

    def test_fewer_bars_than_raw(self, raw, continuous):
        assert len(continuous) < len(raw)

    def test_no_gaps_in_active_sessions(self, continuous):
        """Spot-check: no multi-hour gaps (CME has 1h daily halt 5-6 PM ET)."""
        sample = continuous.loc["2024-06-03":"2024-06-03"]
        if len(sample) > 1:
            gaps = sample.index.to_series().diff().dropna()
            # Max gap should be ~1h (daily maintenance halt) not hours
            max_gap = gaps.max()
            assert max_gap <= pd.Timedelta("65min")

    def test_rollovers_happen(self, continuous):
        """Should have multiple contract rollovers over 5 years."""
        changes = (continuous["symbol"] != continuous["symbol"].shift()).sum()
        assert changes >= 15  # ~20 expected quarterly rolls


# ============================================================
# Morphology tests
# ============================================================

class TestMorphology:
    def test_all_features_present(self, morphology):
        expected = [
            "body", "body_abs", "range", "upper_wick", "lower_wick",
            "midpoint", "body_ratio", "upper_wick_ratio", "lower_wick_ratio",
            "wick_imbalance", "direction", "body_ticks", "range_ticks",
            "true_range", "atr_5", "atr_20", "range_vs_atr20",
            "close_chg", "close_pct", "gap", "vol_ma5", "vol_ma20",
            "vol_ratio", "streak",
        ]
        for f in expected:
            assert f in morphology.columns, f"Missing feature: {f}"

    def test_body_ratio_bounds(self, morphology):
        br = morphology["body_ratio"].dropna()
        assert (br >= 0).all()
        assert (br <= 1).all()

    def test_wick_ratios_non_negative(self, morphology):
        for col in ["upper_wick_ratio", "lower_wick_ratio"]:
            vals = morphology[col].dropna()
            assert (vals >= -1e-10).all()  # tiny float tolerance

    def test_ratios_sum_to_one(self, morphology):
        """body_ratio + upper_wick_ratio + lower_wick_ratio ≈ 1.0."""
        s = (
            morphology["body_ratio"]
            + morphology["upper_wick_ratio"]
            + morphology["lower_wick_ratio"]
        ).dropna()
        assert np.allclose(s, 1.0, atol=1e-10)

    def test_direction_values(self, morphology):
        assert set(morphology["direction"].unique()) <= {-1, 0, 1}

    def test_body_ticks_are_quarter_point_multiples(self, morphology):
        """MNQ tick = 0.25, so body_ticks should be integers."""
        bt = morphology["body_ticks"].dropna()
        assert np.allclose(bt, bt.round(), atol=1e-8)

    def test_mean_range_matches_known_value(self, morphology):
        """Mean range should be ~7 points (known from data analysis)."""
        assert 5.0 < morphology["range"].mean() < 9.0

    def test_mean_body_ratio_matches_known_value(self, morphology):
        """Mean body_ratio should be ~0.46 (known from data analysis)."""
        assert 0.40 < morphology["body_ratio"].mean() < 0.52

    def test_streak_sign_matches_direction(self, morphology):
        """Non-zero streaks should have same sign as their direction."""
        nonzero = morphology[morphology["streak"] != 0]
        assert (np.sign(nonzero["streak"]) == nonzero["direction"]).all()


# ============================================================
# Pattern tests
# ============================================================

class TestPatterns:
    def test_all_17_patterns_detected(self, morphology):
        pats = detect_patterns(morphology)
        assert len(pats.columns) == 17

    def test_doji_frequency(self, morphology):
        """~10% of bars should be doji (body_ratio < 0.10)."""
        pats = detect_patterns(morphology, ["doji"])
        pct = pats["pat_doji"].mean()
        assert 0.05 < pct < 0.15

    def test_marubozu_frequency(self, morphology):
        """~5% of bars should be marubozu (body_ratio > 0.89 = p95)."""
        pats = detect_patterns(morphology, ["marubozu"])
        pct = pats["pat_marubozu"].mean()
        assert 0.02 < pct < 0.10

    def test_morning_star_rarer_than_doji(self, morphology):
        pats = detect_patterns(morphology, ["doji", "morning_star"])
        assert pats["pat_morning_star"].sum() < pats["pat_doji"].sum()

    def test_engulfing_symmetry(self, morphology):
        """Bullish and bearish engulfing should have similar counts (±20%)."""
        pats = detect_patterns(morphology, ["engulfing_bullish", "engulfing_bearish"])
        bull = pats["pat_engulfing_bullish"].sum()
        bear = pats["pat_engulfing_bearish"].sum()
        ratio = bull / bear if bear > 0 else 0
        assert 0.8 < ratio < 1.2

    def test_unknown_pattern_raises(self, morphology):
        with pytest.raises(ValueError, match="Unknown"):
            detect_patterns(morphology, ["fake_pattern"])


# ============================================================
# Aggregation tests
# ============================================================

class TestAggregation:
    def test_5min_reduces_bars(self, continuous):
        agg = aggregate(continuous, "5min")
        assert len(agg) < len(continuous)
        # ~5x fewer bars
        ratio = len(continuous) / len(agg)
        assert 4.5 < ratio < 5.5

    def test_1h_bar_count(self, continuous):
        agg = aggregate(continuous, "1h")
        assert 25_000 < len(agg) < 35_000

    def test_daily_bar_count(self, continuous):
        agg = aggregate(continuous, "1D")
        # ~5 years, ~260 trading days/year → ~1300
        assert 1_200 < len(agg) < 2_000

    def test_aggregation_preserves_high_low(self, continuous):
        """Aggregated high must >= all component highs."""
        sample = continuous.loc["2024-06-03"]
        agg = aggregate(sample, "1h")
        if len(agg) > 0:
            assert agg["high"].max() == sample["high"].max()
            assert agg["low"].min() == sample["low"].min()

    def test_volume_sums_correctly(self, continuous):
        """Daily aggregated volume should equal sum of 1-min volumes."""
        day = "2024-06-03"
        day_1min = continuous.loc[day]
        agg = aggregate(day_1min, "1D")
        if len(agg) > 0:
            assert agg["volume"].sum() == day_1min["volume"].sum()

    def test_morphology_works_on_aggregated(self, continuous):
        """Full morphology + patterns should work on aggregated data."""
        agg = aggregate(continuous, "1h")
        morph = compute_morphology(agg)
        pats = detect_patterns(morph)
        assert len(morph) == len(agg)
        assert len(pats) == len(agg)


# ============================================================
# Statistics tests
# ============================================================

@pytest.fixture(scope="module")
def full_with_fwd(morphology):
    """Morphology + patterns + forward returns."""
    pats = detect_patterns(morphology)
    full = pd.concat([morphology, pats], axis=1)
    return compute_forward_returns(full)


class TestStatistics:
    def test_forward_returns_computed(self, full_with_fwd):
        for h in [1, 5, 20]:
            assert f"fwd_{h}" in full_with_fwd.columns

    def test_fwd1_mean_near_zero(self, full_with_fwd):
        """Unconditional 1-bar forward return should be near zero."""
        mean = full_with_fwd["fwd_1"].mean()
        assert abs(mean) < 0.5  # less than 0.5 points bias

    def test_pattern_stats_shape(self, full_with_fwd):
        stats = pattern_stats(full_with_fwd)
        assert "BASELINE" in stats.index
        # 17 patterns + BASELINE = 18 rows
        assert len(stats) == 18

    def test_baseline_edge_is_zero(self, full_with_fwd):
        stats = pattern_stats(full_with_fwd)
        assert stats.loc["BASELINE", "fwd1_edge"] == 0.0

    def test_win_rate_bounds(self, full_with_fwd):
        stats = pattern_stats(full_with_fwd)
        for h in [1, 5, 20]:
            wr = stats[f"fwd{h}_win_rate"].dropna()
            assert (wr >= 0).all()
            assert (wr <= 1).all()

    def test_big_body_bearish_has_bullish_reversion(self, full_with_fwd):
        """Real data shows big_body_bearish → positive fwd1 (mean reversion)."""
        stats = pattern_stats(full_with_fwd)
        edge = stats.loc["big_body_bearish", "fwd1_edge"]
        # Confirmed: +0.044 pts edge, p=0.0017
        assert edge > 0

    def test_three_black_crows_reversal(self, full_with_fwd):
        """Real data shows three_black_crows → positive fwd1 (reversal)."""
        stats = pattern_stats(full_with_fwd)
        edge = stats.loc["three_black_crows", "fwd1_edge"]
        assert edge > 0

    def test_session_breakdown_has_three_sessions(self, full_with_fwd):
        sess = by_session(full_with_fwd, horizon=1)
        sessions = set(sess["session"])
        assert sessions == {"asia", "london", "ny"}


# ============================================================
# Confluence tests
# ============================================================

@pytest.fixture(scope="module")
def confluence_data(continuous):
    """Confluence on recent 3 months for speed."""
    recent = continuous.loc["2025-06-01":"2025-09-01"]
    if len(recent) < 1000:
        recent = continuous.iloc[-50000:]
    return recent


class TestConfluence:
    def test_confluence_runs(self, confluence_data):
        conf = detect_confluence(confluence_data, higher_timeframes=("5min", "15min"))
        assert len(conf) == len(confluence_data)

    def test_confluence_count_columns(self, confluence_data):
        conf = detect_confluence(confluence_data, higher_timeframes=("5min",))
        assert "conf_bullish_count" in conf.columns
        assert "conf_bearish_count" in conf.columns
        assert "conf_net" in conf.columns
        assert "conf_strong_bullish" in conf.columns
        assert "conf_strong_bearish" in conf.columns

    def test_bullish_count_non_negative(self, confluence_data):
        conf = detect_confluence(confluence_data, higher_timeframes=("5min",))
        assert (conf["conf_bullish_count"] >= 0).all()
        assert (conf["conf_bearish_count"] >= 0).all()

    def test_confluence_net_equals_diff(self, confluence_data):
        conf = detect_confluence(confluence_data, higher_timeframes=("5min",))
        expected = conf["conf_bullish_count"] - conf["conf_bearish_count"]
        assert (conf["conf_net"] == expected).all()

    def test_higher_confluence_is_rarer(self, confluence_data):
        """More TFs aligned should be less frequent."""
        conf = detect_confluence(confluence_data, higher_timeframes=("5min", "15min"))
        counts = conf["conf_bullish_count"].value_counts().sort_index()
        # Count at level 0 should be > count at level 2
        if 0 in counts.index and 2 in counts.index:
            assert counts[0] > counts[2]

    def test_confluence_stats_returns_data(self, confluence_data):
        conf = detect_confluence(confluence_data, higher_timeframes=("5min", "15min"))
        morph = compute_morphology(confluence_data)
        fwd = compute_forward_returns(morph)
        cstats = confluence_stats(conf, fwd)
        assert len(cstats) > 0
        assert "direction" in cstats.columns
        assert "confluence_level" in cstats.columns

    def test_find_events_returns_signals(self, confluence_data):
        conf = detect_confluence(confluence_data, higher_timeframes=("5min", "15min"))
        events = find_confluence_events(conf, min_bullish=2, min_bearish=2)
        if len(events) > 0:
            assert "signal" in events.columns
            assert set(events["signal"].unique()) <= {"BULLISH", "BEARISH"}


# ============================================================
# Group morphology tests
# ============================================================

@pytest.fixture(scope="module")
def group_sample(continuous):
    """Use 50K bars for group morphology tests (speed)."""
    return continuous.iloc[-50000:]


class TestGroupMorphology:
    def test_window_features_shape(self, group_sample):
        feats = compute_window_features(group_sample, window=15, step=15)
        # Non-overlapping: ~50000/15 ≈ 3333 windows
        assert 3000 < len(feats) < 4000
        assert "gm_slope" in feats.columns
        assert "gm_r2" in feats.columns

    def test_all_12_features_present(self, group_sample):
        feats = compute_window_features(group_sample, window=15, step=15)
        expected = [
            "gm_slope", "gm_r2", "gm_compression", "gm_chop_ratio",
            "gm_close_pos", "gm_vol_profile", "gm_high_pos", "gm_low_pos",
            "gm_range", "gm_net_move", "gm_max_dd", "gm_max_runup",
        ]
        for f in expected:
            assert f in feats.columns, f"Missing: {f}"

    def test_r2_bounds(self, group_sample):
        feats = compute_window_features(group_sample, window=15, step=15)
        assert (feats["gm_r2"] >= -0.01).all()  # float tolerance
        assert (feats["gm_r2"] <= 1.01).all()

    def test_close_pos_bounds(self, group_sample):
        feats = compute_window_features(group_sample, window=15, step=15)
        assert (feats["gm_close_pos"] >= 0).all()
        assert (feats["gm_close_pos"] <= 1).all()

    def test_classify_returns_valid_shapes(self, group_sample):
        feats = compute_window_features(group_sample, window=30, step=30)
        shapes = classify_shape(feats)
        valid = {s for s in Shape}
        assert all(s in valid for s in shapes.unique())

    def test_shape_distribution_not_all_neutral(self, group_sample):
        """With 50K bars, should detect multiple shape types."""
        feats = compute_window_features(group_sample, window=15, step=5)
        shapes = classify_shape(feats)
        unique_shapes = shapes.unique()
        assert len(unique_shapes) >= 5

    def test_forward_returns_computed(self, group_sample):
        feats = compute_window_features(group_sample, window=15, step=15)
        shapes = classify_shape(feats)
        stats = group_forward_returns(group_sample, feats, shapes, forward_windows=(15,))
        assert len(stats) > 0
        assert "n" in stats.columns
        assert "fwd15_mean" in stats.columns

    def test_different_windows_different_results(self, group_sample):
        f15 = compute_window_features(group_sample, window=15, step=15)
        f60 = compute_window_features(group_sample, window=60, step=60)
        # 60-bar windows should have fewer rows
        assert len(f60) < len(f15)

    def test_exhaustion_down_has_positive_fwd(self, continuous):
        """Real data: exhaustion_down (60-bar) → strong positive reversal."""
        feats = compute_window_features(continuous, window=60, step=5)
        shapes = classify_shape(feats)
        stats = group_forward_returns(continuous, feats, shapes, forward_windows=(60,))
        if Shape.EXHAUSTION_DOWN.value in stats.index:
            mean = stats.loc[Shape.EXHAUSTION_DOWN.value, "fwd60_mean"]
            assert mean > 0  # confirmed: +10.56 pts

    def test_momentum_burst_down_continues(self, continuous):
        """Real data: momentum_burst_down (60-bar) → negative continuation."""
        feats = compute_window_features(continuous, window=60, step=5)
        shapes = classify_shape(feats)
        stats = group_forward_returns(continuous, feats, shapes, forward_windows=(60,))
        if Shape.MOMENTUM_BURST_DOWN.value in stats.index:
            mean = stats.loc[Shape.MOMENTUM_BURST_DOWN.value, "fwd60_mean"]
            assert mean < 0  # confirmed: -5.84 pts
