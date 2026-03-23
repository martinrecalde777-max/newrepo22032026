"""Parameter optimizer — grid search for best stop/target/holding combos.

Tests combinations of:
    - stop_atr_mult: how tight/wide the stop loss
    - target_atr_mult: how far the target (or None = use expected_move)
    - holding_bars: max bars before timeout exit
    - session filter: which sessions to trade
    - mtf_confirm: whether to require multi-TF confirmation

Uses walk-forward validation: optimize on train window, test on next window,
roll forward. This prevents overfitting to historical data.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

from mnq_morphology.backtest import run_backtest, BacktestResult, MNQ_POINT_VALUE
from mnq_morphology.signals import (
    build_setup_table,
    generate_signals,
    filter_no_overlap,
    add_session_filter,
    add_mtf_confirmation,
    filter_mtf_confirmed,
    precompute_mtf_slopes,
)


@dataclass
class OptResult:
    """Result of a single parameter combination."""
    stop_atr: float
    target_atr: float | None
    holding_bars: int
    sessions: tuple[str, ...]
    mtf_min_score: int
    n_trades: int
    net_pnl_pts: float
    win_rate: float
    profit_factor: float
    sharpe: float
    max_dd_pts: float
    expectancy_pts: float
    stop_ev_ratio: float | None = None
    target_ev_ratio: float | None = None


def grid_search(
    df: pd.DataFrame,
    features: pd.DataFrame,
    setup_table: pd.DataFrame,
    stop_atr_range: tuple[float, ...] = (0.5, 0.75, 1.0, 1.5),
    target_atr_range: tuple[float | None, ...] = (None, 1.0, 2.0, 3.0),
    holding_range: tuple[int, ...] = (30, 60, 120),
    session_combos: tuple[tuple[str, ...], ...] = (
        ("london", "ny"),
    ),
    mtf_scores: tuple[int, ...] = (0, 1),
    slippage_pts: float = 0.5,
    commission_pts: float = 0.5,
    min_trades: int = 50,
    higher_timeframes: tuple[str, ...] = ("5min", "1h"),
    stop_ev_ratios: tuple[float | None, ...] = (None,),
    target_ev_ratios: tuple[float | None, ...] = (None,),
    verbose: bool = True,
) -> pd.DataFrame:
    """Exhaustive grid search over parameter space.

    Parameters
    ----------
    df : OHLCV 1-min DataFrame
    features : classified contextual features
    setup_table : from build_setup_table()
    stop_atr_range : ATR multipliers to test for stops
    target_atr_range : ATR multipliers for targets (None = use expected_move)
    holding_range : max holding periods to test
    session_combos : session filter combinations
    mtf_scores : minimum MTF scores to test (0 = no filter)
    slippage_pts : per-side slippage
    commission_pts : round-trip commission in points
    min_trades : skip combos with fewer trades
    higher_timeframes : TFs for MTF confirmation
    verbose : print progress

    Returns
    -------
    DataFrame of results sorted by net PnL.
    """
    t0 = time.time()

    # Pre-generate all raw signals for each stop/target/holding combo
    # to avoid recomputing contextual features each time
    combos = list(product(stop_atr_range, target_atr_range, holding_range, session_combos, mtf_scores, stop_ev_ratios, target_ev_ratios))
    total = len(combos)

    if verbose:
        print(f"Grid search: {total} combinations")
        print(f"  stop_atr: {stop_atr_range}")
        print(f"  target_atr: {target_atr_range}")
        print(f"  holding: {holding_range}")
        print(f"  sessions: {session_combos}")
        print(f"  mtf_min: {mtf_scores}")
        if any(r is not None for r in stop_ev_ratios):
            print(f"  stop_ev_ratio: {stop_ev_ratios}")
        if any(r is not None for r in target_ev_ratios):
            print(f"  target_ev_ratio: {target_ev_ratios}")

    # Pre-compute MTF slopes ONCE (the expensive part)
    needs_mtf = max(mtf_scores) > 0
    mtf_slopes = None
    if needs_mtf:
        if verbose:
            print("  Pre-computing MTF slopes...")
        mtf_slopes = precompute_mtf_slopes(df, higher_timeframes)

    # Cache base signals by (stop, target, holding) to avoid regenerating
    signal_cache: dict[tuple, pd.DataFrame] = {}
    mtf_cache: dict[tuple, pd.DataFrame] = {}

    results: list[OptResult] = []

    for i, (stop_atr, target_atr, holding, sessions, mtf_min, stop_ev, target_ev) in enumerate(combos):
        sig_key = (stop_atr, target_atr, holding, stop_ev, target_ev)

        if sig_key not in signal_cache:
            sigs = generate_signals(
                df, features, setup_table,
                stop_atr_mult=stop_atr,
                target_atr_mult=target_atr,
                holding_bars=holding,
                stop_ev_ratio=stop_ev,
                target_ev_ratio=target_ev,
            )
            signal_cache[sig_key] = sigs

        sigs = signal_cache[sig_key]

        if len(sigs) == 0:
            continue

        # Session filter
        filtered = add_session_filter(sigs, allowed_sessions=sessions)

        # MTF confirmation (cache per sig_key, uses pre-computed slopes)
        if mtf_min > 0:
            mtf_key = sig_key
            if mtf_key not in mtf_cache:
                mtf_sigs = add_mtf_confirmation(
                    sigs, df, higher_timeframes=higher_timeframes,
                    precomputed_slopes=mtf_slopes,
                )
                mtf_cache[mtf_key] = mtf_sigs

            mtf_sigs = mtf_cache[mtf_key]
            # Re-apply session filter on mtf version
            filtered = add_session_filter(mtf_sigs, allowed_sessions=sessions)
            filtered = filter_mtf_confirmed(filtered, min_score=mtf_min)

        # Overlap filter
        filtered = filter_no_overlap(filtered, min_gap_bars=holding)

        if len(filtered) < min_trades:
            continue

        # Backtest
        bt = run_backtest(df, filtered, slippage_pts=slippage_pts, commission_pts=commission_pts)

        results.append(OptResult(
            stop_atr=stop_atr,
            target_atr=target_atr,
            holding_bars=holding,
            sessions=sessions,
            mtf_min_score=mtf_min,
            n_trades=bt.total_trades,
            net_pnl_pts=bt.net_pnl_pts,
            win_rate=bt.win_rate,
            profit_factor=bt.profit_factor,
            sharpe=bt.sharpe_ratio,
            max_dd_pts=bt.max_drawdown_pts,
            expectancy_pts=bt.expectancy_pts,
            stop_ev_ratio=stop_ev,
            target_ev_ratio=target_ev,
        ))

        if verbose and (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{total}] {elapsed:.0f}s elapsed...")

    if verbose:
        print(f"  Grid search done: {len(results)} valid combos in {time.time()-t0:.1f}s")

    # Build result DataFrame
    rows = []
    for r in results:
        row = {
            "stop_atr": r.stop_atr,
            "target_atr": r.target_atr if r.target_atr is not None else "expected",
            "holding_bars": r.holding_bars,
            "sessions": "|".join(r.sessions),
            "mtf_min": r.mtf_min_score,
            "trades": r.n_trades,
            "net_pnl_pts": r.net_pnl_pts,
            "win_rate": r.win_rate,
            "profit_factor": r.profit_factor,
            "sharpe": r.sharpe,
            "max_dd_pts": r.max_dd_pts,
            "expectancy_pts": r.expectancy_pts,
            "pnl_per_dd": r.net_pnl_pts / r.max_dd_pts if r.max_dd_pts > 0 else 0,
        }
        if hasattr(r, "stop_ev_ratio"):
            row["stop_ev"] = r.stop_ev_ratio
        if hasattr(r, "target_ev_ratio"):
            row["target_ev"] = r.target_ev_ratio
        rows.append(row)

    result_df = pd.DataFrame(rows)
    if len(result_df) > 0:
        result_df = result_df.sort_values("net_pnl_pts", ascending=False)

    return result_df


def walk_forward(
    df: pd.DataFrame,
    features: pd.DataFrame,
    scan_fn,
    train_months: int = 12,
    test_months: int = 3,
    stop_atr_range: tuple[float, ...] = (0.5, 0.75, 1.0),
    target_atr_range: tuple[float | None, ...] = (None, 1.0, 2.0),
    holding_range: tuple[int, ...] = (30, 60),
    slippage_pts: float = 0.5,
    commission_pts: float = 0.5,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[BacktestResult]]:
    """Walk-forward optimization to validate robustness.

    1. Split data into rolling windows: [train_months | test_months]
    2. Optimize on train, test on next window
    3. Roll forward by test_months

    Parameters
    ----------
    df : full OHLCV DataFrame
    features : classified contextual features (must cover full df range)
    scan_fn : callable(features_subset, df_subset) -> setup_table
    train_months, test_months : window sizes
    stop_atr_range, target_atr_range, holding_range : param grids
    slippage_pts, commission_pts : costs

    Returns
    -------
    (summary_df, list of BacktestResults per fold)
    """
    from mnq_morphology.contextual_morphology import (
        compute_forward_returns as ctx_fwd, scan_setups,
    )

    dates = df.index
    start = dates.min()
    end = dates.max()

    fold_results = []
    fold_summaries = []
    fold_num = 0

    current = start
    while True:
        train_end = current + pd.DateOffset(months=train_months)
        test_end = train_end + pd.DateOffset(months=test_months)

        if test_end > end:
            break

        fold_num += 1
        if verbose:
            print(f"\n--- Fold {fold_num}: train {current.date()}→{train_end.date()}, "
                  f"test {train_end.date()}→{test_end.date()} ---")

        # Split
        train_df = df.loc[current:train_end]
        test_df = df.loc[train_end:test_end]
        train_feats = features.loc[features.index.isin(train_df.index)]
        test_feats = features.loc[features.index.isin(test_df.index)]

        if len(train_feats) < 1000 or len(test_feats) < 100:
            current = current + pd.DateOffset(months=test_months)
            continue

        # Optimize on train
        fwd_train = ctx_fwd(train_df, train_feats, forward_bars=(30, 60))
        setups_train = scan_setups(train_feats, fwd_train, min_n=50)
        table_train = build_setup_table(setups_train, min_n=50, min_abs_mean=2.0, min_win_rate=0.50)

        if len(table_train) == 0:
            current = current + pd.DateOffset(months=test_months)
            continue

        best_pnl = -np.inf
        best_params = None

        for stop_atr, target_atr, holding in product(stop_atr_range, target_atr_range, holding_range):
            sigs = generate_signals(train_df, train_feats, table_train,
                                    stop_atr_mult=stop_atr, target_atr_mult=target_atr,
                                    holding_bars=holding)
            sigs = filter_no_overlap(sigs, min_gap_bars=holding)
            if len(sigs) < 20:
                continue
            bt = run_backtest(train_df, sigs, slippage_pts=slippage_pts, commission_pts=commission_pts)
            if bt.net_pnl_pts > best_pnl:
                best_pnl = bt.net_pnl_pts
                best_params = (stop_atr, target_atr, holding)

        if best_params is None:
            current = current + pd.DateOffset(months=test_months)
            continue

        # Test with best params from train
        stop_atr, target_atr, holding = best_params
        if verbose:
            print(f"  Best train params: stop={stop_atr}, target={target_atr}, holding={holding}")
            print(f"  Train PnL: {best_pnl:+.1f} pts")

        test_sigs = generate_signals(test_df, test_feats, table_train,
                                     stop_atr_mult=stop_atr, target_atr_mult=target_atr,
                                     holding_bars=holding)
        test_sigs = filter_no_overlap(test_sigs, min_gap_bars=holding)
        test_bt = run_backtest(test_df, test_sigs, slippage_pts=slippage_pts, commission_pts=commission_pts)

        if verbose:
            print(f"  Test: {test_bt.total_trades} trades, PnL={test_bt.net_pnl_pts:+.1f} pts, "
                  f"WR={test_bt.win_rate:.1%}, PF={test_bt.profit_factor:.2f}")

        fold_results.append(test_bt)
        fold_summaries.append({
            "fold": fold_num,
            "train_start": current,
            "train_end": train_end,
            "test_start": train_end,
            "test_end": test_end,
            "best_stop": stop_atr,
            "best_target": target_atr,
            "best_holding": holding,
            "train_pnl": best_pnl,
            "test_trades": test_bt.total_trades,
            "test_pnl_pts": test_bt.net_pnl_pts,
            "test_win_rate": test_bt.win_rate,
            "test_pf": test_bt.profit_factor,
            "test_sharpe": test_bt.sharpe_ratio,
        })

        current = current + pd.DateOffset(months=test_months)

    summary = pd.DataFrame(fold_summaries)
    if verbose and len(summary) > 0:
        print(f"\n{'='*60}")
        print("WALK-FORWARD SUMMARY")
        print(f"{'='*60}")
        print(f"  Folds: {len(summary)}")
        print(f"  Total test PnL: {summary['test_pnl_pts'].sum():+.1f} pts")
        print(f"  Avg test PnL/fold: {summary['test_pnl_pts'].mean():+.1f} pts")
        print(f"  Profitable folds: {(summary['test_pnl_pts'] > 0).sum()}/{len(summary)}")
        print(f"  Avg win rate: {summary['test_win_rate'].mean():.1%}")

    return summary, fold_results
