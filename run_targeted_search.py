"""Targeted search — find setups that meet strict criteria.

Criteria (>=):
    - Win Rate >= 55%
    - Profit Factor >= 3.0
    - Avg Winner >= 60 pts

Optimized: generates base signal matches ONCE, then sweeps stop/target
by recomputing prices from stored entry data. Much faster than
regenerating signals per combo.
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
    build_setup_table, compute_atr, add_session_filter,
    classify_session, filter_no_overlap,
)
from mnq_morphology.backtest import (
    run_backtest, trades_to_dataframe, analyze_by_setup, analyze_by_exit,
    MNQ_POINT_VALUE,
)

# ── Criteria ──
MIN_WIN_RATE = 0.55
MIN_PROFIT_FACTOR = 3.0
MIN_AVG_WINNER = 60.0  # points

DATA_PATH = "data/glbx-mdp3-20210312-20260311.ohlcv-1m.parquet"

t0 = time.time()
print("=" * 70)
print("TARGETED SEARCH — WR>=55%, PF>=3, AvgWin>=60pts")
print("=" * 70)

# ── 1. Load ──
print("\n[1/7] Loading data...")
raw = load_parquet(DATA_PATH)
cont = build_continuous(raw)
print(f"  {len(cont):,} bars ({cont.index.min()} → {cont.index.max()})")

# ── 2. Features ──
print("\n[2/7] Computing contextual features...")
t1 = time.time()
feats = compute_contextual_features(cont, shape_window=60, context_window=120, step=12)
classified = classify_context(feats)
print(f"  {len(classified):,} windows in {time.time()-t1:.1f}s")

# ── 3. Forward returns ──
print("\n[3/7] Forward returns (30, 60, 120, 240, 480 bars)...")
fwd = ctx_forward_returns(cont, classified, forward_bars=(30, 60, 120, 240, 480))

# ── 4. Scan ──
print("\n[4/7] Scanning setups...")
setups = scan_setups(classified, fwd, min_n=50)
print(f"  {len(setups)} raw setups")

if "best_avg_winner" in setups.columns:
    top = setups.nlargest(10, "best_avg_winner")
    print(f"\n  Top 10 by statistical avg_winner:")
    for _, r in top.iterrows():
        print(f"    {r['ctx_trend']:15s} | {r['shape_type']:20s} | {r['vol_regime']:15s} | "
              f"n={r['n']:.0f} | mean={r['best_mean']:+.1f} | "
              f"WR={r.get('best_win',0):.1%} | PF={r.get('best_pf',0):.2f} | "
              f"avg_win={r.get('best_avg_winner',0):.1f} | "
              f"horizon={r.get('best_horizon_bars','?')}b")

# ── 5. Filter ──
print("\n[5/7] Filtering setups...")
table = build_setup_table(
    setups, min_n=50, min_abs_mean=20.0,
    min_win_rate=0.55, min_pf=1.5, min_avg_winner=40.0,
)
if len(table) == 0:
    for min_aw, min_wr, min_pf_val, min_mean in [
        (30, 0.55, 1.3, 15), (20, 0.52, 1.2, 10), (0, 0.52, 1.0, 8),
    ]:
        table = build_setup_table(
            setups, min_n=50, min_abs_mean=min_mean,
            min_win_rate=min_wr, min_pf=min_pf_val, min_avg_winner=min_aw,
        )
        if len(table) > 0:
            print(f"  Relaxed: {len(table)} setups (aw>={min_aw}, wr>={min_wr})")
            break

if len(table) == 0:
    print("  NO setups found. Exiting.")
    exit()

print(f"  {len(table)} qualifying setups:")
for _, row in table.iterrows():
    print(f"    {row['direction']:5s} | {row['ctx_trend']:15s} | {row['shape_type']:20s} | "
          f"{row['vol_regime']:15s} | EV={row['expected_move']:.1f}")

# ── 6. FAST grid search ──
# KEY OPTIMIZATION: build base signal matches ONCE, then apply different
# stop/target prices. The setup matching (which is slow) only needs to
# happen once since it's the same setups.
print("\n[6/7] Building base signals & grid search...")
t2 = time.time()

# Build lookup
setup_lookup = {}
for _, row in table.iterrows():
    key = (row["ctx_trend"], row["shape_type"], row["vol_regime"])
    if key not in setup_lookup:
        setup_lookup[key] = row

# Compute ATR once
atr = compute_atr(cont, 20)

# Match features to setups ONCE — this is the expensive part
print("  Matching features to setups (one-time)...")
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
print(f"  {len(base_df):,} base signal matches in {time.time()-t2:.1f}s")

# Session filter (one-time)
session = classify_session(base_df.index)
base_df["session"] = session
base_df = base_df[base_df["session"].isin(("london", "ny"))]
print(f"  {len(base_df):,} after session filter (London + NY only)")

# Now sweep stop/target/holding — this is FAST because we just
# compute prices and run backtest
stop_pts_grid = [15, 20, 25, 30, 35, 40]
target_pts_grid = [50, 60, 70, 80, 100]
holding_grid = [30, 60, 120, 240, 480]

combos = [(s, t, h) for s in stop_pts_grid for t in target_pts_grid
          for h in holding_grid if t / s >= 1.5]
print(f"\n  Sweeping {len(combos)} stop/target/holding combos...")

results = []
t3 = time.time()

for i, (s_pts, t_pts, holding) in enumerate(combos):
    # Apply stop/target prices to base signals
    sigs = base_df.copy()

    mask_long = sigs["direction"] == "long"
    mask_short = ~mask_long

    sigs.loc[mask_long, "stop_price"] = sigs.loc[mask_long, "entry_price"] - s_pts
    sigs.loc[mask_long, "target_price"] = sigs.loc[mask_long, "entry_price"] + t_pts
    sigs.loc[mask_short, "stop_price"] = sigs.loc[mask_short, "entry_price"] + s_pts
    sigs.loc[mask_short, "target_price"] = sigs.loc[mask_short, "entry_price"] - t_pts

    sigs["stop_distance"] = float(s_pts)
    sigs["target_distance"] = float(t_pts)
    sigs["holding_bars"] = holding

    # Overlap filter
    sigs = filter_no_overlap(sigs, min_gap_bars=holding)

    if len(sigs) < 20:
        continue

    bt = run_backtest(cont, sigs, slippage_pts=0.5, commission_pts=0.5)

    results.append({
        "stop_pts": s_pts,
        "target_pts": t_pts,
        "holding": holding,
        "trades": bt.total_trades,
        "win_rate": bt.win_rate,
        "profit_factor": bt.profit_factor,
        "avg_win_pts": bt.avg_win_pts,
        "avg_loss_pts": bt.avg_loss_pts,
        "net_pnl_pts": bt.net_pnl_pts,
        "net_pnl_$": bt.net_pnl_dollars,
        "sharpe": bt.sharpe_ratio,
        "max_dd_pts": bt.max_drawdown_pts,
        "expectancy_pts": bt.expectancy_pts,
        "rr_ratio": abs(bt.avg_win_pts / bt.avg_loss_pts) if bt.avg_loss_pts != 0 else 0,
    })

    if (i + 1) % 40 == 0:
        print(f"  [{i+1}/{len(combos)}] {time.time()-t3:.0f}s elapsed...")

print(f"  Grid done: {len(results)} valid in {time.time()-t3:.1f}s")

if not results:
    print("  No valid results.")
    exit()

results_df = pd.DataFrame(results)

# ── 7. Results ──
print("\n" + "=" * 70)
print("[7/7] RESULTS")
print("=" * 70)

# A) ALL criteria met
criteria_mask = (
    (results_df["win_rate"] >= MIN_WIN_RATE) &
    (results_df["profit_factor"] >= MIN_PROFIT_FACTOR) &
    (results_df["avg_win_pts"] >= MIN_AVG_WINNER)
)
matching = results_df[criteria_mask].sort_values("net_pnl_pts", ascending=False)

print(f"\n  Combos meeting ALL criteria (WR>={MIN_WIN_RATE:.0%}, PF>={MIN_PROFIT_FACTOR}, AvgWin>={MIN_AVG_WINNER}pts):")
print(f"  {len(matching)} / {len(results_df)} combinations\n")

if len(matching) > 0:
    for _, r in matching.head(20).iterrows():
        print(f"  stop={r['stop_pts']:.0f} | target={r['target_pts']:.0f} | "
              f"hold={r['holding']:.0f} | "
              f"n={r['trades']:.0f} | WR={r['win_rate']:.1%} | "
              f"PF={r['profit_factor']:.2f} | "
              f"avg_win={r['avg_win_pts']:+.1f} | avg_loss={r['avg_loss_pts']:+.1f} | "
              f"R:R={r['rr_ratio']:.2f} | "
              f"net={r['net_pnl_pts']:+.0f}pts (${r['net_pnl_$']:+,.0f}) | "
              f"DD={r['max_dd_pts']:.0f}pts | Sharpe={r['sharpe']:.2f}")
else:
    print("  >> NO combinations meet ALL criteria simultaneously <<\n")

    # Show partial matches
    for label, col, thresh in [
        ("WR >= 55%", "win_rate", MIN_WIN_RATE),
        ("PF >= 3", "profit_factor", MIN_PROFIT_FACTOR),
        ("AvgWin >= 60", "avg_win_pts", MIN_AVG_WINNER),
    ]:
        n_pass = (results_df[col] >= thresh).sum()
        print(f"    {label}: {n_pass}/{len(results_df)} pass")

    # Composite score
    results_df["score"] = (
        (results_df["win_rate"] / MIN_WIN_RATE).clip(upper=1.0) * 0.33 +
        (results_df["profit_factor"] / MIN_PROFIT_FACTOR).clip(upper=1.0) * 0.33 +
        (results_df["avg_win_pts"] / MIN_AVG_WINNER).clip(upper=1.0) * 0.34
    )
    best = results_df.nlargest(15, "score")
    print("\n  Closest matches:")
    for _, r in best.iterrows():
        wr_ok = "OK" if r["win_rate"] >= MIN_WIN_RATE else "  "
        pf_ok = "OK" if r["profit_factor"] >= MIN_PROFIT_FACTOR else "  "
        aw_ok = "OK" if r["avg_win_pts"] >= MIN_AVG_WINNER else "  "
        print(f"  stop={r['stop_pts']:.0f} | target={r['target_pts']:.0f} | "
              f"hold={r['holding']:.0f} | n={r['trades']:.0f} | "
              f"WR={r['win_rate']:.1%} [{wr_ok}] | "
              f"PF={r['profit_factor']:.2f} [{pf_ok}] | "
              f"avg_win={r['avg_win_pts']:+.1f} [{aw_ok}] | "
              f"net={r['net_pnl_pts']:+.0f}pts")

# B) Best per criterion
print("\n\n  BEST BY CRITERION:")
print("  " + "-" * 60)
for label, col in [("Win Rate", "win_rate"), ("Profit Factor", "profit_factor"),
                    ("Avg Winner", "avg_win_pts"), ("Net PnL", "net_pnl_pts")]:
    br = results_df.loc[results_df[col].idxmax()]
    print(f"  Best {label:15s}: {br[col]:>8.2f} | "
          f"stop={br['stop_pts']:.0f} target={br['target_pts']:.0f} "
          f"hold={br['holding']:.0f} | "
          f"WR={br['win_rate']:.1%} PF={br['profit_factor']:.2f} "
          f"AvgWin={br['avg_win_pts']:.1f} | n={br['trades']:.0f}")

# C) Detailed breakdown
print("\n\n  DETAILED BREAKDOWN — Best combo:")
print("  " + "-" * 60)

if len(matching) > 0:
    best_p = matching.iloc[0]
else:
    best_p = results_df.nlargest(1, "score").iloc[0]

s_pts = float(best_p["stop_pts"])
t_pts = float(best_p["target_pts"])
holding = int(best_p["holding"])
print(f"  Params: stop={s_pts:.0f}pts target={t_pts:.0f}pts holding={holding}bars")

# Rebuild signals for this combo
sigs = base_df.copy()
mask_long = sigs["direction"] == "long"
sigs.loc[mask_long, "stop_price"] = sigs.loc[mask_long, "entry_price"] - s_pts
sigs.loc[mask_long, "target_price"] = sigs.loc[mask_long, "entry_price"] + t_pts
sigs.loc[~mask_long, "stop_price"] = sigs.loc[~mask_long, "entry_price"] + s_pts
sigs.loc[~mask_long, "target_price"] = sigs.loc[~mask_long, "entry_price"] - t_pts
sigs["stop_distance"] = s_pts
sigs["target_distance"] = t_pts
sigs["holding_bars"] = holding
sigs = filter_no_overlap(sigs, min_gap_bars=holding)

bt = run_backtest(cont, sigs, slippage_pts=0.5, commission_pts=0.5)
print(bt.summary())

print("\n  BY SETUP:")
by_setup = analyze_by_setup(bt)
if len(by_setup) > 0:
    print(by_setup.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

print("\n  BY EXIT REASON:")
by_exit = analyze_by_exit(bt)
if len(by_exit) > 0:
    print(by_exit.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

trades_df = trades_to_dataframe(bt)
if len(trades_df) > 0:
    print("\n  BY HOUR:")
    trades_df["hour"] = trades_df["entry_time"].dt.hour
    hourly = trades_df.groupby("hour").agg(
        trades=("net_pnl", "count"),
        avg_pnl=("net_pnl", "mean"),
        win_rate=("net_pnl", lambda x: (x > 0).mean()),
    ).round(2)
    print(hourly.to_string())

print(f"\n\nTotal time: {time.time()-t0:.1f}s")
