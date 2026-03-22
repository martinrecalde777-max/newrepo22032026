"""
FASE 6A+6B — Rolling Re-Clustering + Clean Backtest (Sin Look-Ahead Bias)

PROBLEMA: En Fases 1-5, KMeans se entrenó con TODO el dataset (2021-2026).
Los clusters del período de test "vieron" datos futuros → resultados inflados.

SOLUCIÓN: Ventana rolling de entrenamiento.
- Cada 60 días hábiles (~3 meses), re-entrenar KMeans solo con datos pasados
- Asignar clusters a las ventanas del período forward (siguiente 60 días)
- Backtest SOLO sobre períodos forward (nunca sobre datos de entrenamiento)

Esto elimina completamente el look-ahead bias.
"""

import pandas as pd
import numpy as np
import os
import json
import time
import warnings
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/user/newrepo22032026/outputs'

print("=" * 80)
print("FASE 6A+6B — ROLLING RE-CLUSTERING + CLEAN BACKTEST")
print("  Eliminando look-ahead bias completamente")
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
volumes = df['volume'].values.astype(np.float64)
timestamps = df['ts_et'].values
dates = pd.to_datetime(timestamps)

gap_mask = np.zeros(n_bars, dtype=bool)
ts_diff = np.diff(timestamps.astype('int64')) / 1e9
gap_mask[1:] = ts_diff > 300
gap_cumsum = np.cumsum(gap_mask)

TICK_SIZE = 0.25
POINT_VALUE = 2.0
COMMISSION_RT = 1.24
SLIPPAGE_TICKS = 1

print(f"  Barras: {n_bars:,}")
print(f"  Rango: {str(timestamps[0])[:10]} → {str(timestamps[-1])[:10]}")

# ============================================================
# 2. FUNCIONES DE EXTRACCIÓN DE FEATURES (igual que Fase 2)
# ============================================================

def extract_windows(W, opens, highs, lows, closes, volumes, gap_cumsum, n_bars):
    """Extract sliding windows of W bars, skip gaps."""
    features_list = []
    end_indices = []

    for i in range(W - 1, n_bars):
        start = i - W + 1
        if gap_cumsum[i] - gap_cumsum[start] > 0:
            continue

        o = opens[start:i+1]
        h = highs[start:i+1]
        l = lows[start:i+1]
        c = closes[start:i+1]
        v = volumes[start:i+1]

        base = o[0]
        if base == 0:
            continue

        # Normalize to relative returns from first open
        o_norm = (o - base) / base * 1000
        h_norm = (h - base) / base * 1000
        l_norm = (l - base) / base * 1000
        c_norm = (c - base) / base * 1000

        # Features: OHLC normalized + volume ratio
        feat = []
        for j in range(W):
            feat.extend([o_norm[j], h_norm[j], l_norm[j], c_norm[j]])

        # Add volume features (normalized to mean)
        v_mean = v.mean() if v.mean() > 0 else 1
        for j in range(W):
            feat.append(v[j] / v_mean)

        features_list.append(feat)
        end_indices.append(i)

    return np.array(features_list, dtype=np.float32), np.array(end_indices, dtype=np.int64)


# ============================================================
# 3. ROLLING RE-CLUSTERING
# ============================================================

TRAIN_BARS = 294700 * 2   # ~6 meses de barras de 1min para entrenar
FORWARD_BARS = 294700     # ~3 meses forward para testear
N_CLUSTERS = 8
MIN_TRAIN_WINDOWS = 1000  # Mínimo de ventanas para entrenar

# Strategies to test (the 3 distinct ones)
STRATEGIES = {
    'W3_momentum': {'W': 3, 'hold_bars': 5, 'stop_loss_pts': 10},
    'W3_fade_C1': {'W': 3, 'hold_bars': 10, 'stop_loss_pts': 30},
    'W8_fadeC2': {'W': 8, 'hold_bars': 1, 'stop_loss_pts': 40},
}

# We need to discover which cluster maps to which direction dynamically
# since cluster labels can swap between re-trainings

def find_best_cluster_mapping(W, features, end_indices, labels, n_clusters,
                               opens, highs, lows, closes, gap_mask, gap_cumsum, n_bars):
    """
    After clustering, find which clusters have directional edge.
    Returns dict: {cluster_id: ('long'|'short', avg_return)}
    """
    cluster_returns = {c: [] for c in range(n_clusters)}

    for i in range(len(end_indices)):
        cluster = labels[i]
        entry_bar = end_indices[i] + 1
        if entry_bar >= n_bars or gap_mask[entry_bar]:
            continue

        # Simple 1-bar forward return
        if entry_bar + 1 >= n_bars:
            continue
        if gap_cumsum[entry_bar + 1] - gap_cumsum[entry_bar] > 0:
            continue

        ret = (closes[entry_bar + 1] - opens[entry_bar]) / opens[entry_bar]
        cluster_returns[cluster].append(ret)

    mapping = {}
    for c in range(n_clusters):
        rets = cluster_returns[c]
        if len(rets) < 20:
            continue
        avg = np.mean(rets)
        std = np.std(rets)
        if std == 0:
            continue
        t_stat = avg / (std / np.sqrt(len(rets)))
        # Need t_stat > 1.5 for some directional edge
        if abs(t_stat) > 1.5:
            direction = 'long' if avg > 0 else 'short'
            mapping[c] = (direction, float(avg), float(t_stat), len(rets))

    return mapping


def run_backtest_with_mapping(W, cluster_mapping, strategy_type, end_indices, labels,
                                hold_bars, stop_loss_pts,
                                opens, highs, lows, closes, gap_mask, gap_cumsum, n_bars):
    """
    Run backtest using dynamically discovered cluster mappings.
    strategy_type: 'momentum' (follow direction) or 'fade' (reverse direction)
    """
    trades = []

    # For momentum: use clusters with strong directional edge, trade in same direction
    # For fade: use clusters, trade in opposite direction
    signal_clusters = {}
    for c, (direction, avg_ret, t_stat, n) in cluster_mapping.items():
        if strategy_type == 'momentum':
            # Trade in same direction as the edge
            if abs(t_stat) > 2.0:  # stricter for momentum
                signal_clusters[c] = direction
        elif strategy_type == 'fade_short':
            # Only short signals (fade upward moves)
            if direction == 'long' and t_stat > 2.0:
                signal_clusters[c] = 'short'  # fade it
        elif strategy_type == 'fade_down':
            # Fade downward moves
            if direction == 'short' and t_stat < -2.0:
                signal_clusters[c] = 'long'  # fade it

    if not signal_clusters:
        return trades

    for i in range(len(end_indices)):
        cluster = labels[i]
        if cluster not in signal_clusters:
            continue

        direction = signal_clusters[cluster]
        entry_bar = end_indices[i] + 1

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
                exit_price = closes[n_bars - 1]
                exit_reason = 'end'
                exit_bar = n_bars - 1
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

        trades.append({
            'entry_bar': int(entry_bar),
            'exit_bar': int(exit_bar),
            'direction': direction,
            'cluster': int(cluster),
            'entry_price': float(entry_price),
            'exit_price': float(exit_price),
            'pnl_pts': float(pnl_pts),
            'pnl_dollars': float(pnl_dollars),
            'exit_reason': exit_reason,
        })

    return trades


# ============================================================
# MAIN ROLLING LOOP
# ============================================================
print("\n[2] Rolling re-clustering...")

rolling_results = {}

for strat_name, cfg in STRATEGIES.items():
    W = cfg['W']
    hold_bars = cfg['hold_bars']
    stop_loss_pts = cfg['stop_loss_pts']

    # Determine strategy type
    if 'momentum' in strat_name:
        strat_type = 'momentum'
    elif 'fade' in strat_name:
        strat_type = 'fade_short'
    else:
        strat_type = 'momentum'

    print(f"\n  === {strat_name} (W={W}, type={strat_type}) ===")

    # Extract ALL windows for this W
    print(f"    Extrayendo ventanas W={W}...")
    t0 = time.time()
    all_features, all_end_idx = extract_windows(W, opens, highs, lows, closes, volumes, gap_cumsum, n_bars)
    print(f"    {len(all_features):,} ventanas en {time.time()-t0:.1f}s")

    # Define rolling periods
    # Start training after we have enough data
    all_trades = []
    period_stats = []

    # Step through the data
    step = 0
    cursor = TRAIN_BARS  # Start first forward period after TRAIN_BARS of training data

    while cursor < n_bars:
        step += 1
        train_end_bar = cursor
        forward_end_bar = min(cursor + FORWARD_BARS, n_bars)

        # Get training windows (end_idx < train_end_bar)
        train_mask = all_end_idx < train_end_bar
        train_features = all_features[train_mask]
        train_end_indices = all_end_idx[train_mask]

        # Get forward windows (train_end_bar <= end_idx < forward_end_bar)
        forward_mask = (all_end_idx >= train_end_bar) & (all_end_idx < forward_end_bar)
        forward_features = all_features[forward_mask]
        forward_end_indices = all_end_idx[forward_mask]

        if len(train_features) < MIN_TRAIN_WINDOWS or len(forward_features) < 100:
            cursor = forward_end_bar
            continue

        # Scale features
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_features)
        forward_scaled = scaler.transform(forward_features)

        # Train KMeans on training data ONLY
        kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=10, max_iter=300, random_state=42)
        train_labels = kmeans.fit_predict(train_scaled)

        # Predict clusters for forward data
        forward_labels = kmeans.predict(forward_scaled)

        # Find cluster mapping from TRAINING data only
        cluster_mapping = find_best_cluster_mapping(
            W, train_features, train_end_indices, train_labels, N_CLUSTERS,
            opens, highs, lows, closes, gap_mask, gap_cumsum, n_bars
        )

        # Backtest on FORWARD data only using training-derived mapping
        period_trades = run_backtest_with_mapping(
            W, cluster_mapping, strat_type,
            forward_end_indices, forward_labels,
            hold_bars, stop_loss_pts,
            opens, highs, lows, closes, gap_mask, gap_cumsum, n_bars
        )

        period_start_date = str(timestamps[train_end_bar])[:10]
        period_end_date = str(timestamps[min(forward_end_bar - 1, n_bars - 1)])[:10]

        n_trades = len(period_trades)
        if n_trades > 0:
            pnls = np.array([t['pnl_dollars'] for t in period_trades])
            total_pnl = float(pnls.sum())
            avg_pnl = float(pnls.mean())
            win_rate = float((pnls > 0).mean())
            n_signal_clusters = len(cluster_mapping)
        else:
            total_pnl = avg_pnl = win_rate = 0.0
            n_signal_clusters = len(cluster_mapping)

        status = "+" if total_pnl > 0 else ("0" if n_trades == 0 else "-")
        print(f"    Period {step}: {period_start_date} → {period_end_date} | "
              f"train={len(train_features):,} | fwd={len(forward_features):,} | "
              f"clusters_signal={n_signal_clusters} | "
              f"{n_trades:>4} trades | ${total_pnl:>+10.2f} | WR={win_rate:.1%} [{status}]")

        period_stats.append({
            'period': step,
            'start': period_start_date,
            'end': period_end_date,
            'n_train_windows': len(train_features),
            'n_forward_windows': len(forward_features),
            'n_signal_clusters': n_signal_clusters,
            'n_trades': n_trades,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'win_rate': win_rate,
        })

        all_trades.extend(period_trades)
        cursor = forward_end_bar

    # Summary
    n_total = len(all_trades)
    if n_total > 0:
        all_pnls = np.array([t['pnl_dollars'] for t in all_trades])
        total_pnl = float(all_pnls.sum())
        avg_pnl = float(all_pnls.mean())
        win_rate = float((all_pnls > 0).mean())
        sharpe = float(all_pnls.mean() / max(all_pnls.std(), 1e-10) * np.sqrt(252))
        cum = np.cumsum(all_pnls)
        max_dd = float((cum - np.maximum.accumulate(cum)).min())
    else:
        total_pnl = avg_pnl = win_rate = sharpe = max_dd = 0.0

    profitable_periods = sum(1 for p in period_stats if p['total_pnl'] > 0)
    total_periods = len(period_stats)

    print(f"\n    RESUMEN {strat_name} (CLEAN, sin look-ahead):")
    print(f"    Trades: {n_total}")
    print(f"    PnL total: ${total_pnl:+,.2f}")
    print(f"    Avg PnL/trade: ${avg_pnl:+.2f}")
    if n_total > 0:
        print(f"    Avg pts/trade: {(avg_pnl + COMMISSION_RT) / POINT_VALUE:+.1f}")
    print(f"    Win Rate: {win_rate:.1%}")
    print(f"    Sharpe: {sharpe:.2f}")
    print(f"    Max DD: ${max_dd:,.2f}")
    print(f"    Períodos rentables: {profitable_periods}/{total_periods}")

    rolling_results[strat_name] = {
        'trades': all_trades,
        'period_stats': period_stats,
        'summary': {
            'n_trades': n_total,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'avg_pts': (avg_pnl + COMMISSION_RT) / POINT_VALUE if n_total > 0 else 0,
            'win_rate': win_rate,
            'sharpe': sharpe,
            'max_dd': max_dd,
            'profitable_periods': profitable_periods,
            'total_periods': total_periods,
        }
    }


# ============================================================
# ALSO TEST W8 with fade_down type
# ============================================================
print("\n  === W8_fadeC2 (fade_down variant) ===")

W = 8
hold_bars = 1
stop_loss_pts = 40
strat_type = 'fade_down'

all_features_w8, all_end_idx_w8 = all_features, all_end_idx  # reuse if W=8 was last
if W != 8:
    all_features_w8, all_end_idx_w8 = extract_windows(W, opens, highs, lows, closes, volumes, gap_cumsum, n_bars)

all_trades_w8fd = []
period_stats_w8fd = []
cursor = TRAIN_BARS
step = 0

while cursor < n_bars:
    step += 1
    train_end_bar = cursor
    forward_end_bar = min(cursor + FORWARD_BARS, n_bars)

    train_mask = all_end_idx_w8 < train_end_bar
    train_features = all_features_w8[train_mask]
    train_end_indices = all_end_idx_w8[train_mask]

    forward_mask = (all_end_idx_w8 >= train_end_bar) & (all_end_idx_w8 < forward_end_bar)
    forward_features = all_features_w8[forward_mask]
    forward_end_indices = all_end_idx_w8[forward_mask]

    if len(train_features) < MIN_TRAIN_WINDOWS or len(forward_features) < 100:
        cursor = forward_end_bar
        continue

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    forward_scaled = scaler.transform(forward_features)

    kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=10, max_iter=300, random_state=42)
    train_labels = kmeans.fit_predict(train_scaled)
    forward_labels = kmeans.predict(forward_scaled)

    cluster_mapping = find_best_cluster_mapping(
        W, train_features, train_end_indices, train_labels, N_CLUSTERS,
        opens, highs, lows, closes, gap_mask, gap_cumsum, n_bars
    )

    period_trades = run_backtest_with_mapping(
        W, cluster_mapping, strat_type,
        forward_end_indices, forward_labels,
        hold_bars, stop_loss_pts,
        opens, highs, lows, closes, gap_mask, gap_cumsum, n_bars
    )

    period_start_date = str(timestamps[train_end_bar])[:10]
    period_end_date = str(timestamps[min(forward_end_bar - 1, n_bars - 1)])[:10]

    n_trades = len(period_trades)
    total_pnl = sum(t['pnl_dollars'] for t in period_trades) if n_trades else 0
    win_rate = np.mean([t['pnl_dollars'] > 0 for t in period_trades]) if n_trades else 0

    status = "+" if total_pnl > 0 else ("0" if n_trades == 0 else "-")
    print(f"    Period {step}: {period_start_date} → {period_end_date} | "
          f"{n_trades:>4} trades | ${total_pnl:>+10.2f} | WR={win_rate:.1%} [{status}]")

    period_stats_w8fd.append({
        'period': step, 'start': period_start_date, 'end': period_end_date,
        'n_trades': n_trades, 'total_pnl': total_pnl, 'win_rate': float(win_rate),
    })
    all_trades_w8fd.extend(period_trades)
    cursor = forward_end_bar

if all_trades_w8fd:
    pnls = np.array([t['pnl_dollars'] for t in all_trades_w8fd])
    summary = {
        'n_trades': len(pnls), 'total_pnl': float(pnls.sum()),
        'avg_pnl': float(pnls.mean()),
        'avg_pts': float((pnls.mean() + COMMISSION_RT) / POINT_VALUE),
        'win_rate': float((pnls > 0).mean()),
        'sharpe': float(pnls.mean() / max(pnls.std(), 1e-10) * np.sqrt(252)),
        'max_dd': float((np.cumsum(pnls) - np.maximum.accumulate(np.cumsum(pnls))).min()),
        'profitable_periods': sum(1 for p in period_stats_w8fd if p['total_pnl'] > 0),
        'total_periods': len(period_stats_w8fd),
    }
    print(f"\n    RESUMEN W8_fadeC2_fade_down: {summary['n_trades']} trades, ${summary['total_pnl']:+,.2f}, "
          f"Sharpe={summary['sharpe']:.2f}, WR={summary['win_rate']:.1%}")
    rolling_results['W8_fadeC2_fade_down'] = {
        'trades': all_trades_w8fd, 'period_stats': period_stats_w8fd, 'summary': summary
    }


# ============================================================
# 4. COMPARACIÓN: ORIGINAL vs CLEAN
# ============================================================
print("\n\n" + "=" * 80)
print("COMPARACIÓN: ORIGINAL (con look-ahead) vs CLEAN (sin look-ahead)")
print("=" * 80)

original_stats = {
    'W3_momentum': {'total_pnl': 55577.48, 'avg_pts': 37.8, 'win_rate': 0.362, 'sharpe': 6.49, 'n_trades': 748},
    'W3_fade_C1': {'total_pnl': 30890.26, 'avg_pts': 41.7, 'win_rate': 0.428, 'sharpe': 6.34, 'n_trades': 376},
    'W8_fadeC2': {'total_pnl': 32959.90, 'avg_pts': 19.1, 'win_rate': 0.367, 'sharpe': 3.95, 'n_trades': 890},
}

print(f"\n{'Strategy':<20} {'Metric':<12} {'ORIGINAL':>12} {'CLEAN':>12} {'Change':>12}")
print("-" * 70)

for strat_name in ['W3_momentum', 'W3_fade_C1', 'W8_fadeC2']:
    if strat_name not in rolling_results:
        continue
    orig = original_stats.get(strat_name, {})
    clean = rolling_results[strat_name]['summary']

    for metric, orig_key, clean_key, fmt in [
        ('PnL', 'total_pnl', 'total_pnl', '${:+,.0f}'),
        ('Pts/trade', 'avg_pts', 'avg_pts', '{:+.1f}'),
        ('Win Rate', 'win_rate', 'win_rate', '{:.1%}'),
        ('Sharpe', 'sharpe', 'sharpe', '{:.2f}'),
        ('Trades', 'n_trades', 'n_trades', '{:,.0f}'),
    ]:
        o = orig.get(orig_key, 0)
        c = clean.get(clean_key, 0)
        if metric == 'PnL':
            change = f'{c/max(o,1)*100:.0f}%' if o != 0 else 'N/A'
        elif metric in ('Pts/trade', 'Sharpe'):
            change = f'{c - o:+.1f}'
        elif metric == 'Win Rate':
            change = f'{(c - o)*100:+.1f}pp'
        else:
            change = f'{c - o:+,.0f}'
        print(f"{strat_name:<20} {metric:<12} {fmt.format(o):>12} {fmt.format(c):>12} {change:>12}")
    print()


# ============================================================
# 5. VISUALIZACIÓN
# ============================================================
print("[5] Generando gráficos...")

fig, axes = plt.subplots(len(rolling_results), 2, figsize=(18, 5 * len(rolling_results)))
if len(rolling_results) == 1:
    axes = axes.reshape(1, -1)

for idx, (strat_name, res) in enumerate(rolling_results.items()):
    trades = res['trades']
    if not trades:
        continue

    pnls = np.array([t['pnl_dollars'] for t in trades])
    cum_pnl = np.cumsum(pnls)

    # Equity curve
    ax = axes[idx, 0]
    ax.plot(cum_pnl, linewidth=1, color='navy')
    ax.fill_between(range(len(cum_pnl)), cum_pnl, alpha=0.15, color='steelblue')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_title(f'{strat_name} — CLEAN Equity Curve (no look-ahead)\n'
                 f'${res["summary"]["total_pnl"]:+,.0f} | {res["summary"]["n_trades"]} trades | '
                 f'Sharpe={res["summary"]["sharpe"]:.2f}', fontsize=10)
    ax.set_xlabel('Trade #')
    ax.set_ylabel('Cumulative PnL ($)')
    ax.grid(True, alpha=0.3)

    # Period PnL bars
    ax = axes[idx, 1]
    periods = res['period_stats']
    period_nums = [p['period'] for p in periods]
    period_pnls = [p['total_pnl'] for p in periods]
    colors = ['green' if p > 0 else 'red' for p in period_pnls]
    bars = ax.bar(period_nums, period_pnls, color=colors, alpha=0.7, edgecolor='black', linewidth=0.3)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_title(f'{strat_name} — PnL por Período Rolling\n'
                 f'{res["summary"]["profitable_periods"]}/{res["summary"]["total_periods"]} períodos rentables',
                 fontsize=10)
    ax.set_xlabel('Período')
    ax.set_ylabel('PnL ($)')
    ax.grid(axis='y', alpha=0.3)

    for bar, pnl, period in zip(bars, period_pnls, periods):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'${pnl/1000:.0f}K\n{period["n_trades"]}t',
                ha='center', va='bottom' if pnl > 0 else 'top', fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fase6ab_rolling_clean.png'), dpi=150, bbox_inches='tight')
print(f"  Guardado: fase6ab_rolling_clean.png")

# Save results (without full trade lists for JSON)
save_results = {}
for name, res in rolling_results.items():
    save_results[name] = {
        'summary': res['summary'],
        'period_stats': res['period_stats'],
        'n_trades_detail': len(res['trades']),
    }

with open(os.path.join(OUTPUT_DIR, 'fase6ab_rolling_clean.json'), 'w') as f:
    json.dump(save_results, f, indent=2, default=str)
print(f"  Guardado: fase6ab_rolling_clean.json")

# Save trades for next phase
import pickle
with open(os.path.join(OUTPUT_DIR, 'fase6ab_trades.pkl'), 'wb') as f:
    pickle.dump(rolling_results, f)
print(f"  Guardado: fase6ab_trades.pkl")

print("\n[FASE 6A+6B COMPLETADAS]")
