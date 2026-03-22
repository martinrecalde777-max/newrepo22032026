"""
FASE 5B — Sensitivity Analysis
Varía parámetros ±40% para verificar que el edge no depende de ajuste exacto.
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/user/newrepo22032026/outputs'

print("=" * 80)
print("FASE 5B — SENSITIVITY ANALYSIS")
print("=" * 80)

# Load data
df = pd.read_parquet(os.path.join(OUTPUT_DIR, 'mnq_clean.parquet'))
df['ts_et'] = pd.to_datetime(df['ts_et'])
n_bars = len(df)
opens = df['open'].values.astype(np.float64)
highs = df['high'].values.astype(np.float64)
lows = df['low'].values.astype(np.float64)
closes = df['close'].values.astype(np.float64)
timestamps = df['ts_et'].values

gap_mask = np.zeros(n_bars, dtype=bool)
ts_diff = np.diff(timestamps.astype('int64')) / 1e9
gap_mask[1:] = ts_diff > 300
gap_cumsum = np.cumsum(gap_mask)

TICK_SIZE = 0.25
POINT_VALUE = 2.0
COMMISSION_RT = 1.24
SLIPPAGE_TICKS = 1

cluster_data = {}
for W in [3, 5, 8]:
    data = np.load(os.path.join(OUTPUT_DIR, f'fase2_W{W}_data.npz'))
    cluster_data[W] = {'labels': data['labels'], 'end_idx': data['end_idx']}

print(f"  Barras: {n_bars:,}")


def quick_backtest(W, signal_clusters, hold_bars, stop_loss_pts, take_profit_pts=0):
    """Fast backtest returning trade PnLs."""
    data = cluster_data[W]
    labels = data['labels']
    end_idx = data['end_idx']
    trades = []

    for i in range(len(end_idx)):
        cluster = labels[i]
        if cluster not in signal_clusters:
            continue
        direction = signal_clusters[cluster]
        entry_bar = end_idx[i] + 1
        if entry_bar >= n_bars or gap_mask[entry_bar]:
            continue

        entry_price = opens[entry_bar]
        if direction == 'long':
            entry_price += SLIPPAGE_TICKS * TICK_SIZE
        else:
            entry_price -= SLIPPAGE_TICKS * TICK_SIZE

        exit_price = None
        for j in range(1, hold_bars + 1):
            bar_idx = entry_bar + j
            if bar_idx >= n_bars:
                exit_price = closes[n_bars - 1]
                break
            if gap_cumsum[bar_idx] - gap_cumsum[entry_bar] > 0:
                exit_price = closes[bar_idx - 1]
                break
            if stop_loss_pts > 0:
                if direction == 'long' and lows[bar_idx] <= entry_price - stop_loss_pts:
                    exit_price = entry_price - stop_loss_pts
                    break
                elif direction == 'short' and highs[bar_idx] >= entry_price + stop_loss_pts:
                    exit_price = entry_price + stop_loss_pts
                    break
            if take_profit_pts > 0:
                if direction == 'long' and highs[bar_idx] >= entry_price + take_profit_pts:
                    exit_price = entry_price + take_profit_pts
                    break
                elif direction == 'short' and lows[bar_idx] <= entry_price - take_profit_pts:
                    exit_price = entry_price - take_profit_pts
                    break
            if j == hold_bars:
                exit_price = closes[bar_idx]
                break

        if exit_price is None:
            continue

        if direction == 'long':
            exit_price -= SLIPPAGE_TICKS * TICK_SIZE
            pnl = (exit_price - entry_price) * POINT_VALUE - COMMISSION_RT
        else:
            exit_price += SLIPPAGE_TICKS * TICK_SIZE
            pnl = (entry_price - exit_price) * POINT_VALUE - COMMISSION_RT
        trades.append(pnl)

    return trades


# ============================================================
# SENSITIVITY GRIDS
# ============================================================

# Strategy configs: vary hold_bars and stop_loss
STRATS = {
    'W3_momentum': {'W': 3, 'signal_clusters': {3: 'long', 4: 'short'}, 'base_hold': 5, 'base_sl': 10},
    'W5_C2long': {'W': 5, 'signal_clusters': {2: 'long'}, 'base_hold': 5, 'base_sl': 30},
    'W8_fadeC2': {'W': 8, 'signal_clusters': {2: 'short'}, 'base_hold': 1, 'base_sl': 40},
    'W3_fade_C1': {'W': 3, 'signal_clusters': {1: 'short'}, 'base_hold': 10, 'base_sl': 30},
}

hold_multipliers = [0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
sl_multipliers = [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 2.0]

sensitivity_results = {}

for strat_name, cfg in STRATS.items():
    print(f"\n  Analizando {strat_name}...")
    W = cfg['W']
    sc = cfg['signal_clusters']
    base_hold = cfg['base_hold']
    base_sl = cfg['base_sl']

    # Grid: hold_bars vs stop_loss
    hold_values = [max(1, int(base_hold * m)) for m in hold_multipliers]
    sl_values = [max(2, int(base_sl * m)) for m in sl_multipliers]

    # Remove duplicates while preserving order
    hold_values = list(dict.fromkeys(hold_values))
    sl_values = list(dict.fromkeys(sl_values))

    grid_pnl = np.zeros((len(hold_values), len(sl_values)))
    grid_sharpe = np.zeros((len(hold_values), len(sl_values)))
    grid_wr = np.zeros((len(hold_values), len(sl_values)))
    grid_ntrades = np.zeros((len(hold_values), len(sl_values)))

    for hi, h in enumerate(hold_values):
        for si, sl in enumerate(sl_values):
            trades = quick_backtest(W, sc, h, sl)
            n = len(trades)
            if n > 10:
                pnls = np.array(trades)
                grid_pnl[hi, si] = pnls.sum()
                grid_sharpe[hi, si] = pnls.mean() / max(pnls.std(), 1e-10) * np.sqrt(252)
                grid_wr[hi, si] = (pnls > 0).mean()
                grid_ntrades[hi, si] = n
            else:
                grid_pnl[hi, si] = 0
                grid_sharpe[hi, si] = 0
                grid_wr[hi, si] = 0
                grid_ntrades[hi, si] = n

    # Count how many cells are profitable
    total_cells = grid_pnl.size
    profitable_cells = (grid_pnl > 0).sum()
    pct_profitable = profitable_cells / total_cells * 100

    sensitivity_results[strat_name] = {
        'hold_values': hold_values,
        'sl_values': sl_values,
        'grid_pnl': grid_pnl.tolist(),
        'grid_sharpe': grid_sharpe.tolist(),
        'grid_wr': grid_wr.tolist(),
        'grid_ntrades': grid_ntrades.tolist(),
        'pct_profitable': float(pct_profitable),
        'base_hold': base_hold,
        'base_sl': base_sl,
    }

    print(f"    Grid: {len(hold_values)}x{len(sl_values)} = {total_cells} celdas")
    print(f"    Celdas rentables: {profitable_cells}/{total_cells} ({pct_profitable:.0f}%)")
    print(f"    PnL rango: ${grid_pnl.min():+,.0f} a ${grid_pnl.max():+,.0f}")
    print(f"    Sharpe rango: {grid_sharpe[grid_ntrades > 10].min():.2f} a {grid_sharpe.max():.2f}")


# ============================================================
# VISUALIZACIÓN
# ============================================================
print("\n[3] Generando heatmaps de sensibilidad...")

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.flatten()

for idx, (strat_name, res) in enumerate(sensitivity_results.items()):
    ax = axes[idx]
    grid = np.array(res['grid_pnl'])
    hold_vals = res['hold_values']
    sl_vals = res['sl_values']

    im = ax.imshow(grid, cmap='RdYlGn', aspect='auto', origin='lower')
    ax.set_xticks(range(len(sl_vals)))
    ax.set_xticklabels([str(s) for s in sl_vals], fontsize=8)
    ax.set_yticks(range(len(hold_vals)))
    ax.set_yticklabels([str(h) for h in hold_vals], fontsize=8)
    ax.set_xlabel('Stop Loss (pts)')
    ax.set_ylabel('Hold Bars')
    ax.set_title(f'{strat_name}\n{res["pct_profitable"]:.0f}% profitable cells', fontsize=10)
    plt.colorbar(im, ax=ax, label='PnL ($)', shrink=0.8)

    # Annotate
    for hi in range(len(hold_vals)):
        for si in range(len(sl_vals)):
            val = grid[hi, si]
            color = 'white' if abs(val) > grid.max() * 0.5 else 'black'
            ax.text(si, hi, f'${val/1000:.0f}K', ha='center', va='center',
                    fontsize=6, color=color, fontweight='bold')

    # Mark base config
    if res['base_hold'] in hold_vals and res['base_sl'] in sl_vals:
        bh_idx = hold_vals.index(res['base_hold'])
        bs_idx = sl_vals.index(res['base_sl'])
        ax.plot(bs_idx, bh_idx, 'r*', markersize=15, markeredgecolor='black')

plt.suptitle('Sensitivity Analysis — PnL Heatmaps (★ = base config)', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fase5b_sensitivity.png'), dpi=150, bbox_inches='tight')
print(f"  Guardado: fase5b_sensitivity.png")

with open(os.path.join(OUTPUT_DIR, 'fase5b_sensitivity.json'), 'w') as f:
    json.dump(sensitivity_results, f, indent=2, default=str)
print(f"  Guardado: fase5b_sensitivity.json")

print("\n[FASE 5B COMPLETADA]")
