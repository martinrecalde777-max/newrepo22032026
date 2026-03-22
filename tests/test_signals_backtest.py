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
    classify_session, add_session_filter, add_mtf_confirmation,
    filter_mtf_confirmed,
)
from mnq_morphology.backtest import (
    run_backtest, trades_to_dataframe, analyze_by_setup, analyze_by_exit,
    MNQ_POINT_VALUE,
)
from mnq_morphology.optimizer import grid_search


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


# ============================================================
# Session filter tests
# ============================================================

class TestSessionFilter:
    def test_classify_session_values(self, sample):
        sessions = classify_session(sample.index)
        valid = {"asia", "london", "ny", "off_hours"}
        assert set(sessions.unique()) <= valid

    def test_all_sessions_present(self, sample):
        sessions = classify_session(sample.index)
        assert "asia" in sessions.values
        assert "london" in sessions.values
        assert "ny" in sessions.values

    def test_ny_is_930_to_1600(self, sample):
        sessions = classify_session(sample.index)
        ny_bars = sample.index[sessions == "ny"]
        if len(ny_bars) > 0:
            hours = ny_bars.hour
            # NY starts at 9:30, so hour 9 with min >= 30, or hours 10-15
            assert all((h >= 9) and (h < 16) for h in hours)

    def test_session_filter_reduces_signals(self, sample, classified, scan_results):
        table = build_setup_table(scan_results, min_n=50, min_abs_mean=2.0, min_win_rate=0.50)
        signals = generate_signals(sample, classified, table)
        all_sess = add_session_filter(signals, allowed_sessions=("asia", "london", "ny"))
        ny_only = add_session_filter(signals, allowed_sessions=("ny",))
        assert len(ny_only) <= len(all_sess)

    def test_session_column_added(self, sample, classified, scan_results):
        table = build_setup_table(scan_results, min_n=50, min_abs_mean=2.0, min_win_rate=0.50)
        signals = generate_signals(sample, classified, table)
        filtered = add_session_filter(signals, allowed_sessions=("ny",))
        assert "session" in filtered.columns
        assert (filtered["session"] == "ny").all()


# ============================================================
# Multi-timeframe confirmation tests
# ============================================================

class TestMTFConfirmation:
    @pytest.fixture(scope="class")
    def mtf_signals(self, sample, classified, scan_results):
        table = build_setup_table(scan_results, min_n=50, min_abs_mean=2.0, min_win_rate=0.50)
        signals = generate_signals(sample, classified, table)
        return add_mtf_confirmation(signals, sample, higher_timeframes=("5min",))

    def test_mtf_columns_added(self, mtf_signals):
        assert "mtf_score" in mtf_signals.columns
        assert "mtf_confirm" in mtf_signals.columns

    def test_mtf_score_range(self, mtf_signals):
        # With 1 higher TF, score range is -1 to +1
        assert mtf_signals["mtf_score"].min() >= -1
        assert mtf_signals["mtf_score"].max() <= 1

    def test_filter_mtf_reduces(self, mtf_signals):
        all_sigs = len(mtf_signals)
        confirmed = filter_mtf_confirmed(mtf_signals, min_score=1)
        assert len(confirmed) <= all_sigs

    def test_confirmed_signals_have_positive_score(self, mtf_signals):
        confirmed = filter_mtf_confirmed(mtf_signals, min_score=1)
        if len(confirmed) > 0:
            assert (confirmed["mtf_score"] >= 1).all()

    def test_empty_signals_returns_empty(self, sample):
        empty = pd.DataFrame()
        result = add_mtf_confirmation(empty, sample)
        assert len(result) == 0


# ============================================================
# Optimizer tests
# ============================================================

class TestOptimizer:
    def test_grid_search_runs(self, sample, classified, scan_results):
        table = build_setup_table(scan_results, min_n=50, min_abs_mean=2.0, min_win_rate=0.50)
        results = grid_search(
            sample, classified, table,
            stop_atr_range=(0.75, 1.0),
            target_atr_range=(None, 2.0),
            holding_range=(30, 60),
            session_combos=(("asia", "london", "ny"),),
            mtf_scores=(0,),
            min_trades=10,
            higher_timeframes=("5min",),
            verbose=False,
        )
        assert len(results) > 0

    def test_grid_search_columns(self, sample, classified, scan_results):
        table = build_setup_table(scan_results, min_n=50, min_abs_mean=2.0, min_win_rate=0.50)
        results = grid_search(
            sample, classified, table,
            stop_atr_range=(1.0,),
            target_atr_range=(None,),
            holding_range=(60,),
            session_combos=(("asia", "london", "ny"),),
            mtf_scores=(0,),
            min_trades=10,
            higher_timeframes=("5min",),
            verbose=False,
        )
        expected_cols = [
            "stop_atr", "target_atr", "holding_bars", "sessions", "mtf_min",
            "trades", "net_pnl_pts", "win_rate", "profit_factor", "sharpe",
        ]
        for col in expected_cols:
            assert col in results.columns, f"Missing: {col}"

    def test_grid_search_sorted_by_pnl(self, sample, classified, scan_results):
        table = build_setup_table(scan_results, min_n=50, min_abs_mean=2.0, min_win_rate=0.50)
        results = grid_search(
            sample, classified, table,
            stop_atr_range=(0.75, 1.0),
            target_atr_range=(None,),
            holding_range=(30, 60),
            session_combos=(("asia", "london", "ny"),),
            mtf_scores=(0,),
            min_trades=10,
            higher_timeframes=("5min",),
            verbose=False,
        )
        if len(results) > 1:
            # Should be sorted descending by net_pnl_pts
            pnls = results["net_pnl_pts"].values
            assert pnls[0] >= pnls[-1]
