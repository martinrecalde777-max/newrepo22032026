"""Tests for signal generator and backtester on real MNQ data."""

import numpy as np
import pandas as pd
import pytest

from mnq_morphology.loader import load_parquet, build_continuous
from mnq_morphology.contextual_morphology import (
    compute_contextual_features, classify_context, scan_setups,
    compute_forward_returns as ctx_forward_returns,
)
from mnq_morphology.signals import (
    compute_atr, build_setup_table, generate_signals, filter_no_overlap,
)
from mnq_morphology.backtest import (
    run_backtest, trades_to_dataframe, analyze_by_setup, analyze_by_exit,
    MNQ_POINT_VALUE,
)


DATA_PATH = "data/glbx-mdp3-20210312-20260311.ohlcv-1m.parquet"


@pytest.fixture(scope="module")
def continuous():
    raw = load_parquet(DATA_PATH)
    return build_continuous(raw)


@pytest.fixture(scope="module")
def sample(continuous):
    """Last 100K bars for signal/backtest tests."""
    return continuous.iloc[-100000:]


@pytest.fixture(scope="module")
def classified(sample):
    feats = compute_contextual_features(sample, shape_window=30, context_window=90, step=10)
    return classify_context(feats)


@pytest.fixture(scope="module")
def scan_results(sample, classified):
    fwd = ctx_forward_returns(sample, classified, forward_bars=(30, 60))
    return scan_setups(classified, fwd, min_n=50)


# ============================================================
# ATR tests
# ============================================================

class TestATR:
    def test_atr_computed(self, sample):
        atr = compute_atr(sample, period=20)
        assert len(atr) == len(sample)
        assert atr.iloc[25] > 0  # should be positive after warmup

    def test_atr_reasonable_range(self, sample):
        """MNQ ATR-20 on 1-min bars should be 3-15 points."""
        atr = compute_atr(sample, period=20).dropna()
        median_atr = atr.median()
        assert 2.0 < median_atr < 20.0

    def test_atr_first_values_nan(self, sample):
        atr = compute_atr(sample, period=20)
        assert atr.iloc[:19].isna().all()


# ============================================================
# Setup table tests
# ============================================================

class TestSetupTable:
    def test_builds_from_scan(self, scan_results):
        table = build_setup_table(scan_results, min_n=50, min_abs_mean=2.0, min_win_rate=0.50)
        assert len(table) > 0

    def test_has_direction(self, scan_results):
        table = build_setup_table(scan_results, min_n=50, min_abs_mean=2.0, min_win_rate=0.50)
        assert "direction" in table.columns
        assert set(table["direction"].unique()) <= {"long", "short"}

    def test_has_confidence(self, scan_results):
        table = build_setup_table(scan_results, min_n=50, min_abs_mean=2.0, min_win_rate=0.50)
        assert "confidence" in table.columns
        assert (table["confidence"] > 0).all()
        assert (table["confidence"] <= 1.0).all()

    def test_empty_scan_returns_empty(self):
        empty = pd.DataFrame()
        result = build_setup_table(empty)
        assert len(result) == 0

    def test_strict_filter_reduces_setups(self, scan_results):
        loose = build_setup_table(scan_results, min_n=50, min_abs_mean=1.0, min_win_rate=0.50)
        strict = build_setup_table(scan_results, min_n=100, min_abs_mean=5.0, min_win_rate=0.55)
        assert len(strict) <= len(loose)


# ============================================================
# Signal generation tests
# ============================================================

class TestSignalGeneration:
    def test_generates_signals(self, sample, classified, scan_results):
        table = build_setup_table(scan_results, min_n=50, min_abs_mean=2.0, min_win_rate=0.50)
        signals = generate_signals(sample, classified, table)
        assert len(signals) > 0

    def test_signal_columns(self, sample, classified, scan_results):
        table = build_setup_table(scan_results, min_n=50, min_abs_mean=2.0, min_win_rate=0.50)
        signals = generate_signals(sample, classified, table)
        expected_cols = [
            "direction", "entry_price", "stop_price", "target_price",
            "stop_distance", "target_distance", "holding_bars",
            "setup_key", "confidence", "atr",
        ]
        for col in expected_cols:
            assert col in signals.columns, f"Missing column: {col}"

    def test_stop_below_entry_for_long(self, sample, classified, scan_results):
        table = build_setup_table(scan_results, min_n=50, min_abs_mean=2.0, min_win_rate=0.50)
        signals = generate_signals(sample, classified, table)
        longs = signals[signals["direction"] == "long"]
        if len(longs) > 0:
            assert (longs["stop_price"] < longs["entry_price"]).all()
            assert (longs["target_price"] > longs["entry_price"]).all()

    def test_stop_above_entry_for_short(self, sample, classified, scan_results):
        table = build_setup_table(scan_results, min_n=50, min_abs_mean=2.0, min_win_rate=0.50)
        signals = generate_signals(sample, classified, table)
        shorts = signals[signals["direction"] == "short"]
        if len(shorts) > 0:
            assert (shorts["stop_price"] > shorts["entry_price"]).all()
            assert (shorts["target_price"] < shorts["entry_price"]).all()

    def test_empty_table_no_signals(self, sample, classified):
        signals = generate_signals(sample, classified, pd.DataFrame())
        assert len(signals) == 0


# ============================================================
# No-overlap filter tests
# ============================================================

class TestNoOverlap:
    def test_reduces_signal_count(self, sample, classified, scan_results):
        table = build_setup_table(scan_results, min_n=50, min_abs_mean=2.0, min_win_rate=0.50)
        signals = generate_signals(sample, classified, table)
        filtered = filter_no_overlap(signals, min_gap_bars=60)
        assert len(filtered) <= len(signals)

    def test_min_gap_respected(self, sample, classified, scan_results):
        table = build_setup_table(scan_results, min_n=50, min_abs_mean=2.0, min_win_rate=0.50)
        signals = generate_signals(sample, classified, table)
        filtered = filter_no_overlap(signals, min_gap_bars=60)
        if len(filtered) > 1:
            gaps = filtered.index.to_series().diff().dropna().dt.total_seconds() / 60
            assert (gaps >= 60).all()

    def test_single_signal_unchanged(self):
        sig = pd.DataFrame(
            {"direction": ["long"], "confidence": [0.5]},
            index=pd.DatetimeIndex(["2024-01-01"], name="timestamp"),
        )
        result = filter_no_overlap(sig, min_gap_bars=60)
        assert len(result) == 1


# ============================================================
# Backtest tests
# ============================================================

class TestBacktest:
    @pytest.fixture(scope="class")
    def bt_result(self, sample, classified, scan_results):
        table = build_setup_table(scan_results, min_n=50, min_abs_mean=2.0, min_win_rate=0.50)
        signals = generate_signals(sample, classified, table)
        filtered = filter_no_overlap(signals, min_gap_bars=60)
        return run_backtest(sample, filtered, slippage_pts=0.5, commission_pts=0.5)

    def test_trades_executed(self, bt_result):
        assert bt_result.total_trades > 0

    def test_winners_plus_losers_equals_total(self, bt_result):
        assert bt_result.winners + bt_result.losers + bt_result.breakeven == bt_result.total_trades

    def test_win_rate_bounds(self, bt_result):
        assert 0 <= bt_result.win_rate <= 1

    def test_equity_curve_length(self, bt_result):
        assert len(bt_result.equity_curve) == bt_result.total_trades

    def test_max_drawdown_non_negative(self, bt_result):
        assert bt_result.max_drawdown_dollars >= 0

    def test_summary_string(self, bt_result):
        s = bt_result.summary()
        assert "BACKTEST RESULTS" in s
        assert "Profit Factor" in s

    def test_no_signals_returns_empty(self, sample):
        empty = pd.DataFrame()
        result = run_backtest(sample, empty)
        assert result.total_trades == 0

    def test_trades_to_dataframe(self, bt_result):
        df = trades_to_dataframe(bt_result)
        assert len(df) == bt_result.total_trades
        assert "pnl_pts" in df.columns
        assert "exit_reason" in df.columns

    def test_analyze_by_setup(self, bt_result):
        by_setup = analyze_by_setup(bt_result)
        if len(by_setup) > 0:
            assert "setup_key" in by_setup.columns
            assert "win_rate" in by_setup.columns
            assert "net_pnl" in by_setup.columns

    def test_analyze_by_exit(self, bt_result):
        by_exit = analyze_by_exit(bt_result)
        assert len(by_exit) > 0
        assert "exit_reason" in by_exit.columns
        # Should have at least 2 exit types
        assert len(by_exit) >= 2

    def test_exit_reasons_valid(self, bt_result):
        for trade in bt_result.trades:
            assert trade.exit_reason in {"stop", "target", "timeout"}

    def test_slippage_applied(self, sample, classified, scan_results):
        """Verify that higher slippage reduces PnL."""
        table = build_setup_table(scan_results, min_n=50, min_abs_mean=2.0, min_win_rate=0.50)
        signals = generate_signals(sample, classified, table)
        filtered = filter_no_overlap(signals, min_gap_bars=60)
        low_slip = run_backtest(sample, filtered, slippage_pts=0.25, commission_pts=0.0)
        high_slip = run_backtest(sample, filtered, slippage_pts=2.0, commission_pts=0.0)
        if low_slip.total_trades > 0 and high_slip.total_trades > 0:
            assert high_slip.net_pnl_dollars < low_slip.net_pnl_dollars
