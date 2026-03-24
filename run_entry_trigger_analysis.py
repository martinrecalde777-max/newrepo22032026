"""Entry trigger analysis — compare baseline vs precision triggers.

Tests all trigger types and parameter variants against the baseline
(enter at signal bar close). Measures impact on:
    - Stop-out rate (target: reduce from 56% baseline)
    - Win rate, profit factor, avg winner
    - Trade count (trigger rate — how many signals actually fire)
    - Net PnL and Sharpe ratio

Architecture:
    1. Load data, compute contextual features, build setups (same as targeted search)
    2. Generate base signals with session filter
    3. Run baseline backtest (no trigger)
    4. Run each trigger variant
    5. Compare all results in a single table
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
    build_setup_table, compute_atr, classify_session, filter_no_overlap,
)
from mnq_morphology.backtest import (
    run_backtest, run_backtest_triggered, trades_to_dataframe,
    analyze_by_exit, MNQ_POINT_VALUE,
)
from mnq_morphology.entry_triggers import TRIGGER_CONFIGS

DATA_PATH = "data/glbx-mdp3-20210312-20260311.ohlcv-1m.parquet"

t0 = time.time()
print("=" * 80)
print("ENTRY TRIGGER ANALYSIS — Precision Entry Layer")
print("=" * 80)

# ── 1. Load & Features ──
print("\n[1/6] Loading data...")
raw = load_parquet(DATA_PATH)
cont = build_continuous(raw)
print(f"  {len(cont):,} bars")

print("\n[2/6] Computing contextual features...")
t1 = time.time()
feats = compute_contextual_features(cont, shape_window=60, context_window=120, step=12)
classified = classify_context(feats)
print(f"  {len(classified):,} windows in {time.time()-t1:.1f}s")

print("\n[3/6] Forward returns & setup scan...")
fwd = ctx_forward_returns(cont, classified, forward_bars=(30, 60, 120, 240, 480))
setups = scan_setups(classified, fwd, min_n=50)
print(f"  {len(setups)} raw setups")

# Progressive relaxation to find setups
table = None
for min_aw, min_wr, min_pf_val, min_mean in [
    (40, 0.55, 1.5, 20), (30, 0.55, 1.3, 15),
    (20, 0.52, 1.2, 10), (0, 0.52, 1.0, 8),
]:
    table = build_setup_table(
        setups, min_n=50, min_abs_mean=min_mean,
        min_win_rate=min_wr, min_pf=min_pf_val, min_avg_winner=min_aw,
    )
    if len(table) > 0:
        print(f"  {len(table)} setups (aw>={min_aw}, wr>={min_wr}, pf>={min_pf_val})")
        break

if table is None or len(table) == 0:
    print("  NO setups found. Exiting.")
    exit()

# ── 4. Build base signals ──
print("\n[4/6] Building base signals...")
t2 = time.time()

setup_lookup = {}
for _, row in table.iterrows():
    key = (row["ctx_trend"], row["shape_type"], row["vol_regime"])
    if key not in setup_lookup:
        setup_lookup[key] = row

atr = compute_atr(cont, 20)

base_signals = []
for ts in classified.index:
    ctx = classified.loc[ts, "ctx_trend"]
    shape = classified.loc[ts, "shape_type"]
    vol = classified.loc[ts, "vol_regime"]
    key = (ctx, shape, vol)

    if key not in setup_lookup:
        continue
    if ts not in atr.index or pd.isna(atr.loc[ts]):
        continue

    setup = setup_lookup[key]
    entry = cont.loc[ts, "close"]
    current_atr = atr.loc[ts]
    if current_atr <= 0:
        continue

    base_signals.append({
        "timestamp": ts,
        "direction": setup["direction"],
        "entry_price": entry,
        "atr": current_atr,
        "setup_key": f"{ctx}|{shape}|{vol}",
        "confidence": setup["confidence"],
        "expected_move": setup["expected_move"],
        "optimal_holding": int(setup.get("optimal_holding", 60)),
    })

base_df = pd.DataFrame(base_signals).set_index("timestamp").sort_index()
print(f"  {len(base_df):,} base signals in {time.time()-t2:.1f}s")

# Session filter
session = classify_session(base_df.index)
base_df["session"] = session
base_df = base_df[base_df["session"].isin(("london", "ny"))]
print(f"  {len(base_df):,} after session filter")

# ── 5. Run baseline + all trigger variants ──
print("\n[5/6] Running backtests...")
print("=" * 80)

# Test grid: stop/target combos to test with each trigger
# Use a representative set, not full grid
STOP_TARGET_COMBOS = [
    (20, 60, 120),   # wide stop, wide target
    (25, 70, 120),
    (30, 80, 120),
    (15, 50, 60),    # tighter
    (20, 50, 60),
    (25, 60, 240),   # longer hold
    (30, 100, 240),
    (35, 80, 120),
]

TRIGGER_WINDOWS = [3, 5, 8, 10]


def make_signals(base, s_pts, t_pts, holding):
    """Apply stop/target/holding to base signals."""
    sigs = base.copy()
    ml = sigs["direction"] == "long"
    sigs.loc[ml, "stop_price"] = sigs.loc[ml, "entry_price"] - s_pts
    sigs.loc[ml, "target_price"] = sigs.loc[ml, "entry_price"] + t_pts
    sigs.loc[~ml, "stop_price"] = sigs.loc[~ml, "entry_price"] + s_pts
    sigs.loc[~ml, "target_price"] = sigs.loc[~ml, "entry_price"] - t_pts
    sigs["stop_distance"] = float(s_pts)
    sigs["target_distance"] = float(t_pts)
    sigs["holding_bars"] = holding
    sigs = filter_no_overlap(sigs, min_gap_bars=holding)
    return sigs


all_results = []
t3 = time.time()

# Run for each stop/target combo
for combo_idx, (s_pts, t_pts, holding) in enumerate(STOP_TARGET_COMBOS):
    sigs = make_signals(base_df, s_pts, t_pts, holding)
    if len(sigs) < 10:
        continue

    combo_label = f"s{s_pts}/t{t_pts}/h{holding}"

    # A) Baseline (no trigger)
    bt = run_backtest(cont, sigs, slippage_pts=0.5, commission_pts=0.5)
    if bt.total_trades < 10:
        continue

    exit_df = analyze_by_exit(bt)
    stop_pct = 0
    if len(exit_df) > 0 and "stop" in exit_df["exit_reason"].values:
        stop_pct = float(exit_df[exit_df["exit_reason"] == "stop"]["pct"].iloc[0])

    all_results.append({
        "combo": combo_label,
        "trigger": "BASELINE",
        "window": 0,
        "trades": bt.total_trades,
        "trigger_rate": 1.0,
        "win_rate": bt.win_rate,
        "profit_factor": bt.profit_factor,
        "avg_win_pts": bt.avg_win_pts,
        "avg_loss_pts": bt.avg_loss_pts,
        "stop_pct": stop_pct,
        "net_pnl_pts": bt.net_pnl_pts,
        "net_pnl_$": bt.net_pnl_dollars,
        "sharpe": bt.sharpe_ratio,
        "max_dd_pts": bt.max_drawdown_pts,
        "expectancy_pts": bt.expectancy_pts,
    })

    # B) Each trigger type × window size
    for trig_name, trig_cfg in TRIGGER_CONFIGS.items():
        trig_type = trig_cfg["type"]
        trig_params = {k: v for k, v in trig_cfg.items() if k != "type"}

        for tw in TRIGGER_WINDOWS:
            bt_t = run_backtest_triggered(
                cont, sigs,
                trigger_type=trig_type,
                trigger_window=tw,
                trigger_params=trig_params,
                slippage_pts=0.5,
                commission_pts=0.5,
            )

            if bt_t.total_trades < 5:
                continue

            exit_df_t = analyze_by_exit(bt_t)
            stop_pct_t = 0
            if len(exit_df_t) > 0 and "stop" in exit_df_t["exit_reason"].values:
                stop_pct_t = float(exit_df_t[exit_df_t["exit_reason"] == "stop"]["pct"].iloc[0])

            trig_rate = getattr(bt_t, "trigger_rate", 0)

            all_results.append({
                "combo": combo_label,
                "trigger": trig_name,
                "window": tw,
                "trades": bt_t.total_trades,
                "trigger_rate": trig_rate,
                "win_rate": bt_t.win_rate,
                "profit_factor": bt_t.profit_factor,
                "avg_win_pts": bt_t.avg_win_pts,
                "avg_loss_pts": bt_t.avg_loss_pts,
                "stop_pct": stop_pct_t,
                "net_pnl_pts": bt_t.net_pnl_pts,
                "net_pnl_$": bt_t.net_pnl_dollars,
                "sharpe": bt_t.sharpe_ratio,
                "max_dd_pts": bt_t.max_drawdown_pts,
                "expectancy_pts": bt_t.expectancy_pts,
            })

    elapsed = time.time() - t3
    print(f"  [{combo_idx+1}/{len(STOP_TARGET_COMBOS)}] {combo_label} done ({elapsed:.0f}s)")

print(f"\n  All backtests done in {time.time()-t3:.1f}s")

# ── 6. Results ──
print("\n" + "=" * 80)
print("[6/6] RESULTS — Entry Trigger Comparison")
print("=" * 80)

results_df = pd.DataFrame(all_results)

if len(results_df) == 0:
    print("  No results.")
    exit()

# A) Overall: aggregate across all combos per trigger
print("\n  A) AGGREGATE BY TRIGGER (averaged across stop/target combos)")
print("  " + "-" * 75)

agg = results_df.groupby("trigger").agg(
    configs=("combo", "count"),
    avg_trades=("trades", "mean"),
    avg_trig_rate=("trigger_rate", "mean"),
    avg_win_rate=("win_rate", "mean"),
    avg_pf=("profit_factor", "mean"),
    avg_avg_win=("avg_win_pts", "mean"),
    avg_stop_pct=("stop_pct", "mean"),
    total_net_pnl=("net_pnl_pts", "sum"),
    avg_sharpe=("sharpe", "mean"),
    avg_expectancy=("expectancy_pts", "mean"),
).round(3)

# Sort: baseline first, then by avg_win_rate descending
baseline_row = agg.loc[["BASELINE"]] if "BASELINE" in agg.index else pd.DataFrame()
non_baseline = agg.drop("BASELINE", errors="ignore").sort_values("avg_win_rate", ascending=False)
agg_sorted = pd.concat([baseline_row, non_baseline])

print(f"\n{'Trigger':<22s} {'Cfgs':>4s} {'Trades':>7s} {'TrigRate':>8s} "
      f"{'WinRate':>7s} {'PF':>6s} {'AvgWin':>7s} {'StopPct':>7s} "
      f"{'NetPnL':>8s} {'Sharpe':>7s} {'Expect':>7s}")
print("  " + "-" * 100)

for idx, row in agg_sorted.iterrows():
    marker = " <<<" if idx == "BASELINE" else ""
    print(f"  {idx:<20s} {row['configs']:>4.0f} {row['avg_trades']:>7.0f} "
          f"{row['avg_trig_rate']:>7.1%} "
          f"{row['avg_win_rate']:>6.1%} {row['avg_pf']:>6.2f} "
          f"{row['avg_avg_win']:>+7.1f} {row['avg_stop_pct']:>6.1%} "
          f"{row['total_net_pnl']:>+8.0f} {row['avg_sharpe']:>+7.2f} "
          f"{row['avg_expectancy']:>+7.2f}{marker}")

# B) Best trigger combos meeting strict criteria
print("\n\n  B) BEST TRIGGER COMBOS (WR>=50%, PF>=2, AvgWin>=40pts)")
print("  " + "-" * 75)

strict = results_df[
    (results_df["win_rate"] >= 0.50) &
    (results_df["profit_factor"] >= 2.0) &
    (results_df["avg_win_pts"] >= 40.0) &
    (results_df["trigger"] != "BASELINE")
].sort_values("net_pnl_pts", ascending=False)

if len(strict) > 0:
    for _, r in strict.head(25).iterrows():
        print(f"  {r['trigger']:<20s} w={r['window']:<3.0f} {r['combo']:<15s} "
              f"n={r['trades']:<5.0f} trig={r['trigger_rate']:.0%} "
              f"WR={r['win_rate']:.1%} PF={r['profit_factor']:.2f} "
              f"AvgW={r['avg_win_pts']:+.1f} Stop={r['stop_pct']:.0%} "
              f"Net={r['net_pnl_pts']:+.0f}pts Sh={r['sharpe']:.2f}")
else:
    print("  No combos meet strict criteria. Showing best by composite score:")
    non_bl = results_df[results_df["trigger"] != "BASELINE"].copy()
    if len(non_bl) > 0:
        non_bl["score"] = (
            (non_bl["win_rate"] / 0.55).clip(upper=1.0) * 0.25 +
            (non_bl["profit_factor"] / 3.0).clip(upper=1.0) * 0.25 +
            (non_bl["avg_win_pts"] / 60.0).clip(upper=1.0) * 0.25 +
            (1 - non_bl["stop_pct"]).clip(lower=0) * 0.25  # reward LOW stop rate
        )
        top = non_bl.nlargest(25, "score")
        for _, r in top.iterrows():
            print(f"  {r['trigger']:<20s} w={r['window']:<3.0f} {r['combo']:<15s} "
                  f"n={r['trades']:<5.0f} trig={r['trigger_rate']:.0%} "
                  f"WR={r['win_rate']:.1%} PF={r['profit_factor']:.2f} "
                  f"AvgW={r['avg_win_pts']:+.1f} Stop={r['stop_pct']:.0%} "
                  f"Net={r['net_pnl_pts']:+.0f}pts Sh={r['sharpe']:.2f}")

# C) Stop-out improvement: baseline vs best trigger per combo
print("\n\n  C) STOP-OUT RATE IMPROVEMENT BY COMBO")
print("  " + "-" * 75)

for combo in results_df["combo"].unique():
    combo_df = results_df[results_df["combo"] == combo]
    bl = combo_df[combo_df["trigger"] == "BASELINE"]
    non_bl = combo_df[combo_df["trigger"] != "BASELINE"]

    if len(bl) == 0 or len(non_bl) == 0:
        continue

    bl_stop = bl.iloc[0]["stop_pct"]
    bl_wr = bl.iloc[0]["win_rate"]
    bl_pf = bl.iloc[0]["profit_factor"]

    # Best trigger = lowest stop_pct with decent trade count
    best_trig = non_bl[non_bl["trades"] >= 10].nsmallest(1, "stop_pct")
    if len(best_trig) == 0:
        continue

    bt = best_trig.iloc[0]
    delta_stop = bt["stop_pct"] - bl_stop
    delta_wr = bt["win_rate"] - bl_wr

    print(f"  {combo:<15s}  BL: stop={bl_stop:.0%} WR={bl_wr:.1%} PF={bl_pf:.2f}  |  "
          f"Best: {bt['trigger']:<18s} w={bt['window']:.0f} "
          f"stop={bt['stop_pct']:.0%} ({delta_stop:+.0%}) "
          f"WR={bt['win_rate']:.1%} ({delta_wr:+.0%}) "
          f"PF={bt['profit_factor']:.2f} "
          f"n={bt['trades']:.0f}")

# D) Trigger rate vs quality tradeoff
print("\n\n  D) TRIGGER SELECTIVITY vs QUALITY")
print("  " + "-" * 75)
print(f"  {'Trigger':<22s} {'TrigRate':>8s} {'WR':>6s} {'PF':>6s} "
      f"{'StopPct':>7s} {'Quality':>7s}")
print("  " + "-" * 60)

for trig in sorted(results_df["trigger"].unique()):
    tdf = results_df[results_df["trigger"] == trig]
    tr = tdf["trigger_rate"].mean()
    wr = tdf["win_rate"].mean()
    pf = tdf["profit_factor"].mean()
    sp = tdf["stop_pct"].mean()
    # Quality = WR * PF * (1 - stop_pct)
    quality = wr * min(pf, 5) * (1 - sp)
    print(f"  {trig:<22s} {tr:>7.1%} {wr:>5.1%} {pf:>6.2f} "
          f"{sp:>6.1%} {quality:>7.3f}")

# E) Ultra-strict: WR>=55%, PF>=3, AvgWin>=60
print("\n\n  E) ULTRA-STRICT: WR>=55%, PF>=3.0, AvgWin>=60pts")
print("  " + "-" * 75)

ultra = results_df[
    (results_df["win_rate"] >= 0.55) &
    (results_df["profit_factor"] >= 3.0) &
    (results_df["avg_win_pts"] >= 60.0)
].sort_values("net_pnl_pts", ascending=False)

if len(ultra) > 0:
    print(f"  {len(ultra)} combos meet ALL ultra-strict criteria!")
    for _, r in ultra.head(15).iterrows():
        marker = " <<<" if r["trigger"] == "BASELINE" else ""
        print(f"  {r['trigger']:<20s} w={r['window']:<3.0f} {r['combo']:<15s} "
              f"n={r['trades']:<5.0f} trig={r['trigger_rate']:.0%} "
              f"WR={r['win_rate']:.1%} PF={r['profit_factor']:.2f} "
              f"AvgW={r['avg_win_pts']:+.1f} Stop={r['stop_pct']:.0%} "
              f"Net={r['net_pnl_pts']:+.0f}pts${marker}")
else:
    print("  No combos meet ultra-strict. Closest:")
    results_df["ultra_score"] = (
        (results_df["win_rate"] / 0.55).clip(upper=1.0) * 0.33 +
        (results_df["profit_factor"] / 3.0).clip(upper=1.0) * 0.33 +
        (results_df["avg_win_pts"] / 60.0).clip(upper=1.0) * 0.34
    )
    top = results_df.nlargest(15, "ultra_score")
    for _, r in top.iterrows():
        wr_ok = "Y" if r["win_rate"] >= 0.55 else "N"
        pf_ok = "Y" if r["profit_factor"] >= 3.0 else "N"
        aw_ok = "Y" if r["avg_win_pts"] >= 60.0 else "N"
        print(f"  {r['trigger']:<20s} w={r['window']:<3.0f} {r['combo']:<15s} "
              f"WR={r['win_rate']:.1%}[{wr_ok}] PF={r['profit_factor']:.2f}[{pf_ok}] "
              f"AvgW={r['avg_win_pts']:+.1f}[{aw_ok}] Stop={r['stop_pct']:.0%} "
              f"n={r['trades']:.0f}")

print(f"\n\nTotal time: {time.time()-t0:.1f}s")
print("=" * 80)
