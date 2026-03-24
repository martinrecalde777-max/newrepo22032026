"""Full analysis: backtest + session filter + MTF confirmation + optimization.

Runs everything end-to-end on real MNQ data.
"""

import time
import pandas as pd
import numpy as np

from mnq_morphology.loader import load_parquet, build_continuous
from mnq_morphology.contextual_morphology import (
    compute_contextual_features, classify_context, scan_setups,
    compute_forward_returns as ctx_forward_returns,
)
from mnq_morphology.signals import (
    build_setup_table, generate_signals, filter_no_overlap,
    add_session_filter, add_mtf_confirmation, filter_mtf_confirmed,
    classify_session, precompute_mtf_slopes,
)
from mnq_morphology.backtest import (
    run_backtest, trades_to_dataframe, analyze_by_setup, analyze_by_exit,
)
from mnq_morphology.optimizer import grid_search

DATA_PATH = "data/glbx-mdp3-20210312-20260311.ohlcv-1m.parquet"

t0 = time.time()
print("=" * 70)
print("FULL ANALYSIS — MNQ Morphology Pipeline")
print("=" * 70)

# ============================================================
# 1. Load + contextual features
# ============================================================
print("\n[1/6] Loading data + computing contextual features...")
raw = load_parquet(DATA_PATH)
cont = build_continuous(raw)
print(f"  {len(cont):,} bars ({cont.index.min()} → {cont.index.max()})")

feats = compute_contextual_features(cont, shape_window=60, context_window=120, step=12)
classified = classify_context(feats)
# Include 240-bar horizon (4hrs) — best setups need longer holding
fwd = ctx_forward_returns(cont, classified, forward_bars=(60, 120, 240))
setups = scan_setups(classified, fwd, min_n=80, exclude_shapes=("other",))
# HIGH quality gates: min 10pts edge, 55% win rate, 1.3 PF
table = build_setup_table(setups, min_n=80, min_abs_mean=10.0, min_win_rate=0.55, min_pf=1.3)
print(f"  {len(classified):,} windows, {len(setups)} raw setups, {len(table)} qualifying")

# Show qualifying setups with quality metrics
if len(table) > 0:
    print(f"\n  QUALIFYING HIGH-EDGE SETUPS:")
    print(f"  {'Setup':<55s} {'Dir':>5s} {'EV':>8s} {'WR':>6s} {'PF':>6s} {'N':>5s} {'Hold':>5s}")
    print(f"  {'-'*90}")
    for _, row in table.iterrows():
        key = f"{row['ctx_trend']}|{row['shape_type']}|{row['vol_regime']}"
        hold = int(row.get('optimal_holding', 60))
        ev = row['expected_move']
        win_col = row.get('best_win', 0)
        pf_col = row.get('best_pf', 0)
        direction = row['direction']
        ev_display = ev if direction == 'long' else -ev
        print(f"  {key:<55s} {direction:>5s} {ev_display:>+7.1f} {win_col:>5.1%} {pf_col:>5.2f} {row['n']:>5.0f} {hold:>4d}m")

# ============================================================
# 2. TIMEOUT-ONLY vs ATR-STOP comparison (diagnose the gap)
# ============================================================
print("\n" + "=" * 70)
print("[2/6] TIMEOUT-ONLY vs ATR-STOP COMPARISON")
print("  Goal: verify scan EV translates to backtest, diagnose stop impact")
print("=" * 70)

configs = [
    # (stop_atr, target_atr, holding, label)
    (50.0, None, 240, "Timeout-only 240bar (no stop/target)"),
    (50.0, None, 120, "Timeout-only 120bar"),
    (50.0, None,  60, "Timeout-only 60bar"),
    ( 5.0, 5.0,  240, "ATR stop=5, target=5, 240bar"),
    ( 5.0, None, 240, "ATR stop=5, target=EV, 240bar"),
    ( 3.0, 3.0,  240, "ATR stop=3, target=3, 240bar"),
    (10.0, 10.0, 240, "ATR stop=10, target=10, 240bar"),
    (10.0, None, 240, "ATR stop=10, target=EV, 240bar"),
]

print(f"\n  London+NY, no MTF filter:")
for stop, target, hold, label in configs:
    sigs = generate_signals(cont, classified, table, stop_atr_mult=stop,
                            target_atr_mult=target, holding_bars=hold)
    sigs = add_session_filter(sigs, allowed_sessions=("london", "ny"))
    sigs = filter_no_overlap(sigs, min_gap_bars=hold)
    if len(sigs) < 10:
        print(f"    {label:45s}: too few signals ({len(sigs)})")
        continue
    bt = run_backtest(cont, sigs, slippage_pts=0.5, commission_pts=0.5)
    print(f"    {label:45s}: {bt.total_trades:4d} trades | "
          f"WR={bt.win_rate:.1%} | PnL={bt.net_pnl_pts:+8.1f} pts | "
          f"PF={bt.profit_factor:.2f} | Sharpe={bt.sharpe_ratio:+.2f} | "
          f"Exp={bt.expectancy_pts:+.2f} pts/trade")

# Per-setup detail for timeout-only mode
print(f"\n  PER-SETUP detail (timeout-only 240bar, London+NY):")
sigs = generate_signals(cont, classified, table, stop_atr_mult=50.0,
                        target_atr_mult=None, holding_bars=240)
sigs = add_session_filter(sigs, allowed_sessions=("london", "ny"))
sigs = filter_no_overlap(sigs, min_gap_bars=240)
bt_timeout = run_backtest(cont, sigs, slippage_pts=0.5, commission_pts=0.5)
by_setup_to = analyze_by_setup(bt_timeout)
if len(by_setup_to) > 0:
    print(f"  {'Setup':<55s} {'N':>4s} {'WR':>6s} {'Avg PnL':>8s} {'PF':>6s} {'Bars':>5s}")
    print(f"  {'-'*85}")
    for _, row in by_setup_to.iterrows():
        print(f"  {row['setup_key']:<55s} {row['trades']:>4.0f} {row['win_rate']:>5.1%} "
              f"{row['avg_pnl']:>+7.1f} {row['profit_factor']:>5.2f} {row['avg_bars']:>5.0f}")

# ============================================================
# 3. Session filter analysis (best stop/target config)
# ============================================================
print("\n" + "=" * 70)
print("[3/6] SESSION FILTER ANALYSIS")
print("=" * 70)

# Use stop=10 ATR, target=10 ATR to test wider stops
base_sigs = generate_signals(cont, classified, table, stop_atr_mult=10.0,
                              target_atr_mult=10.0, holding_bars=240)
base_sigs_with_session = add_session_filter(base_sigs, allowed_sessions=("asia", "london", "ny", "off_hours"))

for session_combo, label in [
    (("asia", "london", "ny"), "All sessions"),
    (("london", "ny"), "London + NY only"),
    (("ny",), "NY only"),
    (("london",), "London only"),
]:
    sigs = base_sigs_with_session[base_sigs_with_session["session"].isin(session_combo)]
    sigs = filter_no_overlap(sigs, min_gap_bars=240)
    if len(sigs) < 10:
        print(f"  {label:20s}: too few signals ({len(sigs)})")
        continue
    bt = run_backtest(cont, sigs, slippage_pts=0.5, commission_pts=0.5)
    print(f"  {label:20s}: {bt.total_trades:5d} trades | "
          f"WR={bt.win_rate:.1%} | PnL={bt.net_pnl_pts:+8.1f} pts | "
          f"PF={bt.profit_factor:.2f} | Sharpe={bt.sharpe_ratio:+.2f} | "
          f"DD={bt.max_drawdown_pts:.0f} pts | "
          f"Exp={bt.expectancy_pts:+.2f} pts/trade")

# ============================================================
# 4. Multi-timeframe confirmation
# ============================================================
print("\n" + "=" * 70)
print("[4/6] MULTI-TIMEFRAME CONFIRMATION")
print("=" * 70)

print("  Pre-computing MTF slopes (5min, 1h)...")
t_mtf = time.time()
mtf_slopes = precompute_mtf_slopes(cont, higher_timeframes=("5min", "1h"))
mtf_sigs = add_mtf_confirmation(base_sigs, cont, higher_timeframes=("5min", "1h"),
                                 precomputed_slopes=mtf_slopes)
print(f"  MTF computed in {time.time()-t_mtf:.1f}s")

print(f"\n  MTF Score distribution:")
if len(mtf_sigs) > 0:
    for score in sorted(mtf_sigs["mtf_score"].unique()):
        n = (mtf_sigs["mtf_score"] == score).sum()
        print(f"    score={score:+d}: {n:5d} signals ({n/len(mtf_sigs):.1%})")

# Combined: session + MTF
print("\n  Combined filters (session + MTF):")
for sessions, mtf_min in [
    (("london", "ny"), 0),
    (("london", "ny"), 1),
    (("london", "ny"), 2),
]:
    sigs = add_session_filter(mtf_sigs, allowed_sessions=sessions)
    if mtf_min > 0:
        sigs = filter_mtf_confirmed(sigs, min_score=mtf_min)
    sigs = filter_no_overlap(sigs, min_gap_bars=240)
    label = f"London+NY MTF>={mtf_min}"
    if len(sigs) < 10:
        print(f"    {label:30s}: too few signals ({len(sigs)})")
        continue
    bt = run_backtest(cont, sigs, slippage_pts=0.5, commission_pts=0.5)
    print(f"    {label:30s}: {bt.total_trades:5d} trades | "
          f"WR={bt.win_rate:.1%} | PnL={bt.net_pnl_pts:+8.1f} pts | "
          f"PF={bt.profit_factor:.2f} | Sharpe={bt.sharpe_ratio:+.2f} | "
          f"Exp={bt.expectancy_pts:+.2f} pts/trade")

# ============================================================
# 5. Parameter optimization (grid search)
# ============================================================
print("\n" + "=" * 70)
print("[5/6] PARAMETER OPTIMIZATION (Grid Search)")
print("=" * 70)

opt_results = grid_search(
    cont, classified, table,
    stop_atr_range=(5.0, 10.0, 20.0, 50.0),  # very wide stops to let trades breathe
    target_atr_range=(None, 5.0, 10.0),       # None = expected_move
    holding_range=(120, 240),
    session_combos=(
        ("london", "ny"),
    ),
    mtf_scores=(0, 1),
    slippage_pts=0.5,
    commission_pts=0.5,
    min_trades=30,
    higher_timeframes=("5min", "1h"),
    verbose=True,
)

print(f"\n  TOP 15 PARAMETER COMBINATIONS (by net PnL):")
print("-" * 130)
if len(opt_results) > 0:
    display_cols = [
        "stop_atr", "target_atr", "holding_bars", "sessions", "mtf_min",
        "trades", "net_pnl_pts", "win_rate", "profit_factor", "sharpe",
        "max_dd_pts", "expectancy_pts", "pnl_per_dd",
    ]
    top = opt_results.head(15)[display_cols]
    print(top.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # Best by different criteria
    print(f"\n  BEST BY CRITERIA:")
    print(f"  {'Best PnL:':<20s} stop={opt_results.iloc[0]['stop_atr']}, "
          f"target={opt_results.iloc[0]['target_atr']}, "
          f"holding={opt_results.iloc[0]['holding_bars']}, "
          f"mtf={opt_results.iloc[0]['mtf_min']}")

    best_sharpe = opt_results.sort_values("sharpe", ascending=False).iloc[0]
    print(f"  {'Best Sharpe:':<20s} stop={best_sharpe['stop_atr']}, "
          f"target={best_sharpe['target_atr']}, "
          f"holding={best_sharpe['holding_bars']}, "
          f"Sharpe={best_sharpe['sharpe']:+.2f}")

    # Best expectancy with enough trades
    best_exp = opt_results[opt_results["trades"] >= 50].sort_values("expectancy_pts", ascending=False)
    if len(best_exp) > 0:
        be = best_exp.iloc[0]
        print(f"  {'Best Exp (n>=50):':<20s} stop={be['stop_atr']}, "
              f"target={be['target_atr']}, "
              f"holding={be['holding_bars']}, "
              f"Exp={be['expectancy_pts']:+.2f} pts/trade")

# ============================================================
# 6. Best config deep dive
# ============================================================
print("\n" + "=" * 70)
print("[6/6] BEST CONFIGURATION — DEEP DIVE")
print("=" * 70)

if len(opt_results) > 0:
    best = opt_results.iloc[0]
    stop_atr = best["stop_atr"]
    target_atr = None if best["target_atr"] == "expected" else best["target_atr"]
    holding = int(best["holding_bars"])
    sessions = tuple(best["sessions"].split("|"))
    mtf_min = int(best["mtf_min"])
    stop_ev = best.get("stop_ev")
    target_ev = best.get("target_ev")
    if pd.notna(stop_ev) and stop_ev is not None:
        stop_ev = float(stop_ev)
    else:
        stop_ev = None
    if pd.notna(target_ev) and target_ev is not None:
        target_ev = float(target_ev)
    else:
        target_ev = None

    print(f"\n  Config: stop_atr={stop_atr}, target_atr={target_atr}, holding={holding}")
    print(f"          sessions={sessions}, mtf_min={mtf_min}")
    if stop_ev is not None:
        print(f"          stop_ev_ratio={stop_ev}, target_ev_ratio={target_ev}")

    # Re-run with best params
    sigs = generate_signals(cont, classified, table, stop_atr_mult=stop_atr,
                            target_atr_mult=target_atr, holding_bars=holding,
                            stop_ev_ratio=stop_ev, target_ev_ratio=target_ev)
    sigs = add_session_filter(sigs, allowed_sessions=sessions)
    if mtf_min > 0:
        sigs = add_mtf_confirmation(sigs, cont, higher_timeframes=("5min", "1h"),
                                     precomputed_slopes=mtf_slopes)
        sigs = filter_mtf_confirmed(sigs, min_score=mtf_min)
    sigs = filter_no_overlap(sigs, min_gap_bars=holding)

    bt = run_backtest(cont, sigs, slippage_pts=0.5, commission_pts=0.5)
    print(f"\n{bt.summary()}")

    # By setup
    by_setup = analyze_by_setup(bt)
    if len(by_setup) > 0:
        print("\nPERFORMANCE BY SETUP:")
        print(by_setup.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # By exit
    by_exit = analyze_by_exit(bt)
    if len(by_exit) > 0:
        print("\nPERFORMANCE BY EXIT REASON:")
        print(by_exit.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # By year
    trades_df = trades_to_dataframe(bt)
    if len(trades_df) > 0:
        trades_df["year"] = trades_df["entry_time"].dt.year
        yearly = trades_df.groupby("year").agg(
            trades=("net_pnl", "count"),
            net_pnl=("net_pnl", "sum"),
            avg_pnl=("net_pnl", "mean"),
            win_rate=("net_pnl", lambda x: (x > 0).mean()),
        ).round(2)
        print("\nPERFORMANCE BY YEAR:")
        print(yearly.to_string())

    # By session
    if "session" in sigs.columns:
        print("\nPERFORMANCE BY SESSION:")
        for sess in ["london", "ny"]:
            sub_sigs = sigs[sigs["session"] == sess]
            sub_sigs = filter_no_overlap(sub_sigs, min_gap_bars=holding)
            if len(sub_sigs) < 5:
                continue
            sub_bt = run_backtest(cont, sub_sigs, slippage_pts=0.5, commission_pts=0.5)
            print(f"  {sess:8s}: {sub_bt.total_trades:4d} trades | "
                  f"WR={sub_bt.win_rate:.1%} | PnL={sub_bt.net_pnl_pts:+8.1f} pts | "
                  f"PF={sub_bt.profit_factor:.2f}")

print(f"\n\nTotal analysis time: {time.time()-t0:.1f}s")
