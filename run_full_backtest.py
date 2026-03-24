"""Run full backtest on all MNQ data — baseline numbers."""

import time
import pandas as pd

from mnq_morphology.loader import load_parquet, build_continuous
from mnq_morphology.contextual_morphology import (
    compute_contextual_features, classify_context, scan_setups,
    compute_forward_returns as ctx_forward_returns,
)
from mnq_morphology.signals import (
    build_setup_table, generate_signals, filter_no_overlap,
)
from mnq_morphology.backtest import (
    run_backtest, trades_to_dataframe, analyze_by_setup, analyze_by_exit,
)

DATA_PATH = "data/glbx-mdp3-20210312-20260311.ohlcv-1m.parquet"

t0 = time.time()
print("=" * 70)
print("FULL BACKTEST — MNQ Morphology Pipeline")
print("=" * 70)

# 1. Load
print("\n[1/6] Loading data...")
raw = load_parquet(DATA_PATH)
cont = build_continuous(raw)
print(f"  {len(cont):,} bars ({cont.index.min()} → {cont.index.max()})")

# 2. Contextual features
print("\n[2/6] Computing contextual features (60-bar shape, 120-bar context)...")
t1 = time.time()
feats = compute_contextual_features(cont, shape_window=60, context_window=120, step=12)
classified = classify_context(feats)
print(f"  {len(classified):,} windows in {time.time()-t1:.1f}s")
print(f"  Shape types: {classified['shape_type'].value_counts().to_dict()}")

# 3. Forward returns + scan
print("\n[3/6] Scanning for best setups...")
fwd = ctx_forward_returns(cont, classified, forward_bars=(30, 60, 120, 240, 480))
setups = scan_setups(classified, fwd, min_n=80)
print(f"  {len(setups)} raw setups found")

# 4. Build setup table + signals
print("\n[4/6] Generating signals...")
table = build_setup_table(setups, min_n=80, min_abs_mean=8.0, min_win_rate=0.55, min_pf=1.3)
print(f"  {len(table)} qualifying setups:")
for _, row in table.head(10).iterrows():
    stat_aw = f"avg_win={row['stat_avg_winner']:.1f}" if "stat_avg_winner" in row.index else ""
    print(f"    {row['direction']:5s} | {row['ctx_trend']:15s} | {row['shape_type']:20s} | {row['vol_regime']:15s} | "
          f"mean={row['expected_move']:.1f}pts | n={row['n']:.0f} | conf={row['confidence']:.2f} | {stat_aw}")

signals = generate_signals(cont, classified, table, stop_atr_mult=1.5, holding_bars=60)
print(f"\n  {len(signals):,} raw signals generated")

filtered = filter_no_overlap(signals, min_gap_bars=60)
print(f"  {len(filtered):,} signals after overlap filter")

# 5. Backtest
print("\n[5/6] Running backtest (slippage=0.5, commission=0.5)...")
t2 = time.time()
result = run_backtest(cont, filtered, slippage_pts=0.5, commission_pts=0.5)
print(f"  Completed in {time.time()-t2:.1f}s")

# 6. Results
print("\n[6/6] RESULTS\n")
print(result.summary())

# Breakdown by setup
print("\n\nPERFORMANCE BY SETUP:")
print("-" * 90)
by_setup = analyze_by_setup(result)
if len(by_setup) > 0:
    print(by_setup.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

# Breakdown by exit
print("\n\nPERFORMANCE BY EXIT REASON:")
print("-" * 60)
by_exit = analyze_by_exit(result)
if len(by_exit) > 0:
    print(by_exit.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

# Session breakdown
print("\n\nSIGNAL DISTRIBUTION BY HOUR:")
print("-" * 40)
trades_df = trades_to_dataframe(result)
if len(trades_df) > 0:
    trades_df["hour"] = trades_df["entry_time"].dt.hour
    hourly = trades_df.groupby("hour").agg(
        trades=("net_pnl", "count"),
        avg_pnl=("net_pnl", "mean"),
        total_pnl=("net_pnl", "sum"),
        win_rate=("net_pnl", lambda x: (x > 0).mean()),
    ).round(2)
    print(hourly.to_string())

# Direction breakdown
print("\n\nBY DIRECTION:")
print("-" * 40)
if len(trades_df) > 0:
    for d in ["long", "short"]:
        sub = trades_df[trades_df["direction"] == d]
        if len(sub) > 0:
            print(f"  {d.upper():6s}: {len(sub):4d} trades | "
                  f"win={((sub['net_pnl']>0).mean()):.1%} | "
                  f"net=${sub['net_pnl'].sum():+,.0f} | "
                  f"avg=${sub['net_pnl'].mean():+,.2f}")

print(f"\n\nTotal time: {time.time()-t0:.1f}s")
