"""
FASE 5A — Walk-Forward Analysis
Pipeline de descubrimiento de morfologías predictivas MNQ futures

Valida las top estrategias con ventanas rodantes train/test para
detectar degradación temporal y overfitting.
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
import time
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/user/newrepo22032026/outputs'

print("=" * 80)
print("FASE 5A — WALK-FORWARD ANALYSIS")
print("=" * 80)

# ============================================================
# 1. CARGAR DATOS
# ============================================================
print("\n[1] Cargando datos...")

df = pd.read_parquet(os.path.join(OUTPUT_DIR, 'mnq_clean.parquet'))
df['ts_et'] = pd.to_datetime(df['ts_et'])
n_bars = len(df)

opens = df['open'].values.astype(np.float64)
highs = df['high'].values.astype(np.float64)
lows = df['low'].values.astype(np.float64)
closes = df['close'].values.astype(np.float64)
timestamps = df['ts_et'].values
dates = pd.to_datetime(timestamps)

gap_mask = np.zeros(n_bars, dtype=bool)
ts_diff = np.diff(timestamps.astype('int64')) / 1e9
gap_mask[1:] = ts_diff > 300
gap_cumsum = np.cumsum(gap_mask)

# Contract specs
TICK_SIZE = 0.25
POINT_VALUE = 2.0
COMMISSION_RT = 1.24
SLIPPAGE_TICKS = 1

# Load cluster data
cluster_data = {}
for W in [3, 5, 8]:
    data = np.load(os.path.join(OUTPUT_DIR, f'fase2_W{W}_data.npz'))
    cluster_data[W] = {
        'labels': data['labels'],
        'end_idx': data['end_idx'],
    }

print(f"  Barras: {n_bars:,}")

# ============================================================
# 2. TOP STRATEGIES TO VALIDATE
# ============================================================

TOP_STRATS = {
    'W5_C2long_H5': {'W': 5, 'signal_clusters': {2: 'long'}, 'hold_bars': 5, 'stop_loss_pts': 30, 'take_profit_pts': 0},
    'W3_momentum_SL10': {'W': 3, 'signal_clusters': {3: 'long', 4: 'short'}, 'hold_bars': 5, 'stop_loss_pts': 10, 'take_profit_pts': 0},
    'W3_momentum_SL20': {'W': 3, 'signal_clusters': {3: 'long', 4: 'short'}, 'hold_bars': 5, 'stop_loss_pts': 20, 'take_profit_pts': 0},
    'W3_fade_C1_H10': {'W': 3, 'signal_clusters': {1: 'short'}, 'hold_bars': 10, 'stop_loss_pts': 30, 'take_profit_pts': 0},
    'W8_fadeC2_H1': {'W': 8, 'signal_clusters': {2: 'short'}, 'hold_bars': 1, 'stop_loss_pts': 40, 'take_profit_pts': 0},
}


def run_backtest_range(W, config, start_idx, end_idx_range):
    """Run backtest on a subset of windows within index range."""
    data = cluster_data[W]
    labels = data['labels']
    end_idx = data['end_idx']

    mask = (end_idx >= start_idx) & (end_idx < end_idx_range)
    sub_labels = labels[mask]
    sub_end_idx = end_idx[mask]

    signal_clusters = config['signal_clusters']
    hold_bars = config['hold_bars']
    stop_loss_pts = config.get('stop_loss_pts', 0)
    take_profit_pts = config.get('take_profit_pts', 0)

    trades = []

    for i in range(len(sub_end_idx)):
        cluster = sub_labels[i]
        if cluster not in signal_clusters:
            continue

        direction = signal_clusters[cluster]
        entry_bar = sub_end_idx[i] + 1

        if entry_bar >= n_bars or gap_mask[entry_bar]:
            continue

        entry_price = opens[entry_bar]
        if direction == 'long':
            entry_price += SLIPPAGE_TICKS * TICK_SIZE
        else:
            entry_price -= SLIPPAGE_TICKS * TICK_SIZE

        exit_price = None
        exit_bar = None
        exit_reason = None

        for j in range(1, hold_bars + 1):
            bar_idx = entry_bar + j
            if bar_idx >= n_bars:
                bar_idx = n_bars - 1
                exit_price = closes[bar_idx]
                exit_reason = 'end'
                exit_bar = bar_idx
                break

            if gap_cumsum[bar_idx] - gap_cumsum[entry_bar] > 0:
                exit_price = closes[bar_idx - 1]
                exit_reason = 'gap'
                exit_bar = bar_idx - 1
                break

            if stop_loss_pts > 0:
                if direction == 'long' and lows[bar_idx] <= entry_price - stop_loss_pts:
                    exit_price = entry_price - stop_loss_pts
                    exit_reason = 'sl'
                    exit_bar = bar_idx
                    break
                elif direction == 'short' and highs[bar_idx] >= entry_price + stop_loss_pts:
                    exit_price = entry_price + stop_loss_pts
                    exit_reason = 'sl'
                    exit_bar = bar_idx
                    break

            if take_profit_pts > 0:
                if direction == 'long' and highs[bar_idx] >= entry_price + take_profit_pts:
                    exit_price = entry_price + take_profit_pts
                    exit_reason = 'tp'
                    exit_bar = bar_idx
                    break
                elif direction == 'short' and lows[bar_idx] <= entry_price - take_profit_pts:
                    exit_price = entry_price - take_profit_pts
                    exit_reason = 'tp'
                    exit_bar = bar_idx
                    break

            if j == hold_bars:
                exit_price = closes[bar_idx]
                exit_reason = 'time'
                exit_bar = bar_idx
                break

        if exit_price is None:
            continue

        if direction == 'long':
            exit_price -= SLIPPAGE_TICKS * TICK_SIZE
            pnl_pts = exit_price - entry_price
        else:
            exit_price += SLIPPAGE_TICKS * TICK_SIZE
            pnl_pts = entry_price - exit_price

        pnl_dollars = pnl_pts * POINT_VALUE - COMMISSION_RT
        trades.append(pnl_dollars)

    return trades


# ============================================================
# 3. WALK-FORWARD: 6-fold (5 months train, 1 month test ~approx)
# ============================================================
print("\n[2] Walk-Forward Analysis...")

# Split data into ~6 equal chunks by bar index
N_FOLDS = 6
chunk_size = n_bars // N_FOLDS
fold_boundaries = [i * chunk_size for i in range(N_FOLDS + 1)]
fold_boundaries[-1] = n_bars

# Walk-forward: train on folds 0..k-1, test on fold k
wf_results = {}

for strat_name, config in TOP_STRATS.items():
    W = config['W']
    wf_folds = []

    for k in range(1, N_FOLDS):  # test folds 1..5, train on all prior
        train_start = 0
        train_end = fold_boundaries[k]
        test_start = fold_boundaries[k]
        test_end = fold_boundaries[k + 1]

        # Test period trades only
        test_trades = run_backtest_range(W, config, test_start, test_end)

        n_trades = len(test_trades)
        if n_trades > 0:
            pnls = np.array(test_trades)
            total_pnl = float(pnls.sum())
            avg_pnl = float(pnls.mean())
            win_rate = float((pnls > 0).mean())
            sharpe = float(pnls.mean() / max(pnls.std(), 1e-10) * np.sqrt(252))
        else:
            total_pnl = avg_pnl = win_rate = sharpe = 0.0

        fold_start_date = str(timestamps[test_start])[:10]
        fold_end_date = str(timestamps[min(test_end - 1, n_bars - 1)])[:10]

        wf_folds.append({
            'fold': k,
            'period': f'{fold_start_date} → {fold_end_date}',
            'n_trades': n_trades,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'win_rate': win_rate,
            'sharpe': sharpe,
        })

    wf_results[strat_name] = wf_folds

    # Print results
    print(f"\n  {strat_name}:")
    all_profitable = True
    for fold in wf_folds:
        status = "+" if fold['total_pnl'] > 0 else "-"
        if fold['total_pnl'] <= 0:
            all_profitable = False
        print(f"    Fold {fold['fold']}: {fold['period']} | {fold['n_trades']:>4} trades | "
              f"${fold['total_pnl']:>+10.2f} | WR={fold['win_rate']:.1%} | Sharpe={fold['sharpe']:.2f} [{status}]")

    profitable_folds = sum(1 for f in wf_folds if f['total_pnl'] > 0)
    total_wf_pnl = sum(f['total_pnl'] for f in wf_folds)
    print(f"    → Folds rentables: {profitable_folds}/{len(wf_folds)} | PnL total WF: ${total_wf_pnl:+,.2f}")
    if all_profitable:
        print(f"    ★ TODOS los folds rentables — señal robusta")


# ============================================================
# 4. VISUALIZACIÓN WF
# ============================================================
print("\n[3] Generando gráficos Walk-Forward...")

fig, axes = plt.subplots(len(TOP_STRATS), 1, figsize=(14, 3.5 * len(TOP_STRATS)), sharex=False)
if len(TOP_STRATS) == 1:
    axes = [axes]

for idx, (strat_name, folds) in enumerate(wf_results.items()):
    ax = axes[idx]
    fold_nums = [f['fold'] for f in folds]
    pnls = [f['total_pnl'] for f in folds]
    colors = ['green' if p > 0 else 'red' for p in pnls]

    bars = ax.bar(fold_nums, pnls, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_title(f'{strat_name} — Walk-Forward PnL por Fold', fontsize=11)
    ax.set_xlabel('Fold')
    ax.set_ylabel('PnL ($)')

    for bar, pnl, fold in zip(bars, pnls, folds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'${pnl:+,.0f}\n{fold["n_trades"]}t',
                ha='center', va='bottom' if pnl > 0 else 'top', fontsize=8)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fase5a_walkforward.png'), dpi=150, bbox_inches='tight')
print(f"  Guardado: fase5a_walkforward.png")

# Save results
with open(os.path.join(OUTPUT_DIR, 'fase5a_walkforward.json'), 'w') as f:
    json.dump(wf_results, f, indent=2, default=str)

print(f"\n  Guardado: fase5a_walkforward.json")
print("\n[FASE 5A COMPLETADA]")
