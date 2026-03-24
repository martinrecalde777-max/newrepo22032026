"""
FASE 7 — Features Enriquecidos + Señal Mejorada + Transiciones + Cooldown

Todo rolling, sin look-ahead, 1-2 trades/sesión como objetivo.
"""

import pandas as pd
import numpy as np
import os
import json
import time
import pickle
import warnings
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/user/newrepo22032026/outputs'

print("=" * 80)
print("FASE 7 — ENRICHED FEATURES + SIGNAL IMPROVEMENTS + COOLDOWN")
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

# Gap detection
gap_mask = np.zeros(n_bars, dtype=bool)
ts_diff = np.diff(timestamps.astype('int64')) / 1e9
gap_mask[1:] = ts_diff > 300
gap_cumsum = np.cumsum(gap_mask)

# Session info
hours = dates.hour
minutes = dates.minute
time_minutes = hours * 60 + minutes  # minutes since midnight
day_of_week = dates.dayofweek  # 0=Mon, 4=Fri

TICK_SIZE = 0.25
POINT_VALUE = 2.0
COMMISSION_RT = 1.24
SLIPPAGE_TICKS = 1

print(f"  Barras: {n_bars:,}")

# ============================================================
# 2. PRE-COMPUTE ENRICHED INDICATORS
# ============================================================
print("\n[2] Pre-computing indicators...")
t0 = time.time()

# ATR(14)
atr_period = 14
tr = np.maximum(highs - lows,
       np.maximum(np.abs(highs - np.roll(closes, 1)),
                  np.abs(lows - np.roll(closes, 1))))
tr[0] = highs[0] - lows[0]
atr = np.zeros(n_bars)
atr[:atr_period] = np.nan
for i in range(atr_period, n_bars):
    if gap_cumsum[i] - gap_cumsum[max(0, i - atr_period)] > 0:
        atr[i] = tr[i]
    else:
        atr[i] = np.mean(tr[i-atr_period+1:i+1])

# Bollinger Width (20-bar)
bb_period = 20
bb_width = np.zeros(n_bars)
sma20 = np.zeros(n_bars)
for i in range(bb_period - 1, n_bars):
    if gap_cumsum[i] - gap_cumsum[i - bb_period + 1] > 0:
        bb_width[i] = np.nan
        continue
    window = closes[i-bb_period+1:i+1]
    m = window.mean()
    s = window.std()
    sma20[i] = m
    bb_width[i] = (2 * s / m * 100) if m > 0 else 0  # as % of price

# Returns for multi-timeframe
ret_1 = np.zeros(n_bars)
ret_5 = np.zeros(n_bars)
ret_15 = np.zeros(n_bars)
ret_60 = np.zeros(n_bars)

for shift, arr in [(1, ret_1), (5, ret_5), (15, ret_15), (60, ret_60)]:
    for i in range(shift, n_bars):
        if gap_cumsum[i] - gap_cumsum[i - shift] == 0 and closes[i - shift] > 0:
            arr[i] = (closes[i] - closes[i - shift]) / closes[i - shift] * 1000

# Volume ratio (current bar vs 20-bar avg)
vol_ratio = np.ones(n_bars)
for i in range(20, n_bars):
    if gap_cumsum[i] - gap_cumsum[i - 20] == 0:
        avg_v = volumes[i-20:i].mean()
        vol_ratio[i] = volumes[i] / max(avg_v, 1)

print(f"  Indicators computed in {time.time()-t0:.1f}s")

# ============================================================
# 3. ENRICHED WINDOW EXTRACTION
# ============================================================
W = 3  # Keep W=3 (best from Phase 6)

def extract_enriched_windows(W, n_bars):
    """Extract windows with enriched features."""
    features_list = []
    end_indices = []

    for i in range(max(W - 1, 60), n_bars):  # need 60 bars for indicators
        start = i - W + 1
        if gap_cumsum[i] - gap_cumsum[start] > 0:
            continue
        if np.isnan(atr[i]) or np.isnan(bb_width[i]):
            continue

        o = opens[start:i+1]
        h = highs[start:i+1]
        l = lows[start:i+1]
        c = closes[start:i+1]
        v = volumes[start:i+1]

        base = o[0]
        if base == 0:
            continue

        # === ORIGINAL FEATURES (normalized OHLCV) ===
        feat = []
        for j in range(W):
            feat.extend([
                (o[j] - base) / base * 1000,
                (h[j] - base) / base * 1000,
                (l[j] - base) / base * 1000,
                (c[j] - base) / base * 1000,
            ])
        v_mean = v.mean() if v.mean() > 0 else 1
        for j in range(W):
            feat.append(v[j] / v_mean)

        # === NEW: Volatility context ===
        feat.append(atr[i] / base * 1000)        # ATR normalized
        feat.append(bb_width[i])                   # Bollinger width %

        # === NEW: Multi-timeframe returns ===
        feat.append(ret_5[i])    # 5-bar momentum
        feat.append(ret_15[i])   # 15-bar momentum
        feat.append(ret_60[i])   # 60-bar momentum (1hr trend)

        # === NEW: Volume context ===
        feat.append(vol_ratio[i])  # volume vs 20-bar avg

        # === NEW: Temporal context ===
        # Hour as sin/cos (cyclical encoding)
        hour_rad = hours[i] / 24.0 * 2 * np.pi
        feat.append(np.sin(hour_rad))
        feat.append(np.cos(hour_rad))

        # Session phase: 0=premarket, 1=open, 2=midday, 3=close, 4=afterhours
        h = hours[i]
        m = minutes[i]
        tm = h * 60 + m
        if tm < 570:      # before 9:30
            session = 0.0
        elif tm < 630:     # 9:30-10:30 (first hour)
            session = 1.0
        elif tm < 900:     # 10:30-15:00 (midday)
            session = 2.0
        elif tm < 960:     # 15:00-16:00 (power hour)
            session = 3.0
        else:
            session = 4.0
        feat.append(session / 4.0)  # normalize 0-1

        features_list.append(feat)
        end_indices.append(i)

    return np.array(features_list, dtype=np.float32), np.array(end_indices, dtype=np.int64)

print(f"\n[3] Extracting enriched windows (W={W})...")
t0 = time.time()
all_features, all_end_idx = extract_enriched_windows(W, n_bars)
N_FEATURES = all_features.shape[1]
print(f"  {len(all_features):,} windows, {N_FEATURES} features each ({time.time()-t0:.1f}s)")


# ============================================================
# 4. ROLLING CLUSTERING + TRANSITION DETECTION
# ============================================================
TRAIN_BARS = 294700 * 2
FORWARD_BARS = 294700
N_CLUSTERS = 8
MIN_TRAIN = 1000

print(f"\n[4] Rolling clustering + transitions...")

def compute_transitions_and_signals(end_indices, labels, pnls_1bar, gap_cumsum, n_bars):
    """
    Compute:
    1. Per-cluster edge
    2. Per-transition (cluster_from → cluster_to) edge
    3. Confidence (distance to centroid)
    """
    n = len(end_indices)

    # Cluster-level returns
    cluster_rets = {c: [] for c in range(N_CLUSTERS)}
    # Transition-level returns
    transition_rets = {}

    prev_label = None
    for i in range(n):
        c = labels[i]
        entry_bar = end_indices[i] + 1
        if entry_bar >= n_bars or gap_mask[entry_bar]:
            prev_label = c
            continue
        if entry_bar + 1 >= n_bars or gap_cumsum[entry_bar+1] - gap_cumsum[entry_bar] > 0:
            prev_label = c
            continue

        ret = (closes[entry_bar + 1] - opens[entry_bar]) / opens[entry_bar]
        cluster_rets[c].append(ret)

        # Transition
        if prev_label is not None:
            key = (prev_label, c)
            if key not in transition_rets:
                transition_rets[key] = []
            transition_rets[key].append(ret)

        prev_label = c

    # Find edges
    cluster_edges = {}
    for c in range(N_CLUSTERS):
        r = cluster_rets[c]
        if len(r) < 30:
            continue
        avg = np.mean(r)
        std = np.std(r)
        if std == 0:
            continue
        t = avg / (std / np.sqrt(len(r)))
        if abs(t) > 2.0:
            cluster_edges[c] = {'direction': 'long' if avg > 0 else 'short',
                                'avg_ret': float(avg), 't_stat': float(t), 'n': len(r)}

    transition_edges = {}
    for key, r in transition_rets.items():
        if len(r) < 20:
            continue
        avg = np.mean(r)
        std = np.std(r)
        if std == 0:
            continue
        t = avg / (std / np.sqrt(len(r)))
        if abs(t) > 2.5:  # stricter for transitions
            transition_edges[key] = {'direction': 'long' if avg > 0 else 'short',
                                     'avg_ret': float(avg), 't_stat': float(t), 'n': len(r)}

    return cluster_edges, transition_edges


def backtest_with_improvements(end_indices, labels, features_scaled, kmeans,
                                cluster_edges, transition_edges,
                                hold_bars, stop_loss_pts,
                                cooldown_bars):
    """
    Backtest with:
    - Cluster signals + transition signals
    - Confidence (distance to centroid)
    - Cooldown between trades
    - Session filter
    """
    trades = []
    last_exit_bar = -cooldown_bars - 1  # allow first trade
    prev_label = None

    # Pre-compute distances to centroids
    centers = kmeans.cluster_centers_
    distances = np.linalg.norm(features_scaled - centers[labels], axis=1)
    # Normalize distances per cluster
    dist_normalized = np.zeros(len(distances))
    for c in range(N_CLUSTERS):
        mask = labels == c
        if mask.sum() > 0:
            d = distances[mask]
            med = np.median(d)
            if med > 0:
                dist_normalized[mask] = d / med

    for i in range(len(end_indices)):
        c = labels[i]
        entry_bar = end_indices[i] + 1

        if entry_bar >= n_bars or gap_mask[entry_bar]:
            prev_label = c
            continue

        # === COOLDOWN CHECK ===
        if entry_bar - last_exit_bar < cooldown_bars:
            prev_label = c
            continue

        # === SESSION FILTER: only trade regular hours (9:30-15:30 ET) ===
        bar_hour = dates[entry_bar].hour
        bar_min = dates[entry_bar].minute
        bar_tm = bar_hour * 60 + bar_min
        if bar_tm < 570 or bar_tm > 930:  # 9:30 to 15:30
            prev_label = c
            continue

        # === CONFIDENCE: skip if too far from centroid ===
        if dist_normalized[i] > 1.5:  # more than 1.5x median distance
            prev_label = c
            continue

        # === SIGNAL: cluster OR transition ===
        direction = None
        signal_type = None

        # Check transition signal first (stronger)
        if prev_label is not None:
            trans_key = (prev_label, c)
            if trans_key in transition_edges:
                direction = transition_edges[trans_key]['direction']
                signal_type = 'transition'

        # Fall back to cluster signal
        if direction is None and c in cluster_edges:
            direction = cluster_edges[c]['direction']
            signal_type = 'cluster'

        prev_label = c

        if direction is None:
            continue

        # === EXECUTE TRADE ===
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
        last_exit_bar = exit_bar

        trades.append({
            'entry_bar': int(entry_bar),
            'exit_bar': int(exit_bar),
            'direction': direction,
            'signal_type': signal_type,
            'cluster': int(c),
            'confidence': float(1.0 - min(dist_normalized[i], 2.0) / 2.0),
            'pnl_pts': float(pnl_pts),
            'pnl_dollars': float(pnl_dollars),
            'exit_reason': exit_reason,
            'session_time': int(bar_tm),
        })

    return trades


# ============================================================
# 5. SWEEP COOLDOWN + HOLD BARS
# ============================================================
print("\n[5] Sweeping cooldown and parameters...")

# Target: 1-2 trades per session
# Regular session = 390 minutes (9:30-16:00), we trade 9:30-15:30 = 360 min
# 1 trade/session → cooldown ~ 360 bars
# 2 trades/session → cooldown ~ 180 bars

CONFIGS = [
    {'name': 'momentum_cd120', 'hold': 5, 'sl': 10, 'cooldown': 120},
    {'name': 'momentum_cd180', 'hold': 5, 'sl': 10, 'cooldown': 180},
    {'name': 'momentum_cd240', 'hold': 5, 'sl': 10, 'cooldown': 240},
    {'name': 'momentum_cd360', 'hold': 5, 'sl': 10, 'cooldown': 360},
    {'name': 'hold10_cd180', 'hold': 10, 'sl': 15, 'cooldown': 180},
    {'name': 'hold10_cd360', 'hold': 10, 'sl': 15, 'cooldown': 360},
    {'name': 'hold20_cd180', 'hold': 20, 'sl': 20, 'cooldown': 180},
    {'name': 'hold20_cd360', 'hold': 20, 'sl': 20, 'cooldown': 360},
]

all_results = {}

for cfg in CONFIGS:
    name = cfg['name']
    hold_bars = cfg['hold']
    stop_loss_pts = cfg['sl']
    cooldown = cfg['cooldown']

    all_trades = []
    period_stats = []
    cursor = TRAIN_BARS
    step = 0

    while cursor < n_bars:
        step += 1
        train_end = cursor
        forward_end = min(cursor + FORWARD_BARS, n_bars)

        train_mask = all_end_idx < train_end
        fwd_mask = (all_end_idx >= train_end) & (all_end_idx < forward_end)

        train_feat = all_features[train_mask]
        train_idx = all_end_idx[train_mask]
        fwd_feat = all_features[fwd_mask]
        fwd_idx = all_end_idx[fwd_mask]

        if len(train_feat) < MIN_TRAIN or len(fwd_feat) < 100:
            cursor = forward_end
            continue

        scaler = StandardScaler()
        train_sc = scaler.fit_transform(train_feat)
        fwd_sc = scaler.transform(fwd_feat)

        kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=10, max_iter=300, random_state=42)
        train_labels = kmeans.fit_predict(train_sc)
        fwd_labels = kmeans.predict(fwd_sc)

        # Compute edges from training data
        cluster_edges, transition_edges = compute_transitions_and_signals(
            train_idx, train_labels, None, gap_cumsum, n_bars
        )

        # Backtest on forward data
        period_trades = backtest_with_improvements(
            fwd_idx, fwd_labels, fwd_sc, kmeans,
            cluster_edges, transition_edges,
            hold_bars, stop_loss_pts, cooldown
        )

        n_t = len(period_trades)
        pnl = sum(t['pnl_dollars'] for t in period_trades) if n_t else 0

        # Estimate sessions in forward period
        fwd_dates = pd.to_datetime(timestamps[train_end:min(forward_end, n_bars)])
        n_sessions = fwd_dates.normalize().nunique()
        trades_per_session = n_t / max(n_sessions, 1)

        period_stats.append({
            'period': step, 'n_trades': n_t, 'total_pnl': pnl,
            'n_sessions': int(n_sessions), 'trades_per_session': float(trades_per_session),
        })

        all_trades.extend(period_trades)
        cursor = forward_end

    # Summary
    n_total = len(all_trades)
    if n_total > 0:
        pnls = np.array([t['pnl_dollars'] for t in all_trades])
        total_pnl = float(pnls.sum())
        avg_pnl = float(pnls.mean())
        avg_pts = float((avg_pnl + COMMISSION_RT) / POINT_VALUE)
        win_rate = float((pnls > 0).mean())
        sharpe = float(pnls.mean() / max(pnls.std(), 1e-10) * np.sqrt(252))
        cum = np.cumsum(pnls)
        max_dd = float((cum - np.maximum.accumulate(cum)).min())

        total_sessions = sum(p['n_sessions'] for p in period_stats)
        avg_tps = n_total / max(total_sessions, 1)

        # Signal type breakdown
        n_cluster = sum(1 for t in all_trades if t['signal_type'] == 'cluster')
        n_transition = sum(1 for t in all_trades if t['signal_type'] == 'transition')
        pnl_cluster = sum(t['pnl_dollars'] for t in all_trades if t['signal_type'] == 'cluster')
        pnl_transition = sum(t['pnl_dollars'] for t in all_trades if t['signal_type'] == 'transition')

        # Session breakdown
        session_pnls = {}
        for t in all_trades:
            tm = t['session_time']
            if tm < 630:
                s = 'open_930_1030'
            elif tm < 780:
                s = 'mid_1030_1300'
            elif tm < 900:
                s = 'afternoon_1300_1500'
            else:
                s = 'close_1500_1530'
            if s not in session_pnls:
                session_pnls[s] = []
            session_pnls[s].append(t['pnl_dollars'])
    else:
        total_pnl = avg_pnl = avg_pts = win_rate = sharpe = max_dd = 0
        avg_tps = 0
        n_cluster = n_transition = 0
        pnl_cluster = pnl_transition = 0
        session_pnls = {}

    profitable_periods = sum(1 for p in period_stats if p['total_pnl'] > 0)

    print(f"\n  {name}: {n_total} trades | ${total_pnl:+,.0f} | {avg_pts:+.1f} pts | "
          f"WR={win_rate:.1%} | Sharpe={sharpe:.2f} | DD=${max_dd:,.0f} | "
          f"{avg_tps:.1f} trades/session | {profitable_periods}/{len(period_stats)} periods+")

    if n_cluster + n_transition > 0:
        print(f"    Signals: cluster={n_cluster} (${pnl_cluster:+,.0f}) | "
              f"transition={n_transition} (${pnl_transition:+,.0f})")

    if session_pnls:
        for s in sorted(session_pnls.keys()):
            sp = np.array(session_pnls[s])
            print(f"    {s}: {len(sp)} trades, ${sp.sum():+,.0f}, WR={float((sp>0).mean()):.1%}")

    all_results[name] = {
        'trades': all_trades,
        'period_stats': period_stats,
        'summary': {
            'n_trades': n_total, 'total_pnl': total_pnl,
            'avg_pnl': avg_pnl, 'avg_pts': avg_pts,
            'win_rate': win_rate, 'sharpe': sharpe, 'max_dd': max_dd,
            'avg_trades_per_session': avg_tps,
            'profitable_periods': profitable_periods,
            'total_periods': len(period_stats),
            'n_cluster_signals': n_cluster, 'n_transition_signals': n_transition,
            'pnl_cluster': pnl_cluster, 'pnl_transition': pnl_transition,
        }
    }


# ============================================================
# 6. BEST CONFIG ANALYSIS
# ============================================================
print("\n\n" + "=" * 80)
print("RANKING DE CONFIGURACIONES")
print("=" * 80)

ranked = sorted(all_results.items(),
                key=lambda x: x[1]['summary']['sharpe'] if x[1]['summary']['n_trades'] > 50 else -999,
                reverse=True)

print(f"\n{'Config':<22} {'Trades':>6} {'T/Sess':>6} {'PnL':>12} {'Pts':>6} "
      f"{'WR':>6} {'Sharpe':>7} {'MaxDD':>10} {'Per+':>5}")
print("-" * 90)

for name, res in ranked:
    s = res['summary']
    print(f"{name:<22} {s['n_trades']:>6} {s['avg_trades_per_session']:>6.1f} "
          f"${s['total_pnl']:>+10,.0f} {s['avg_pts']:>+5.1f} "
          f"{s['win_rate']:>5.1%} {s['sharpe']:>7.2f} ${s['max_dd']:>9,.0f} "
          f"{s['profitable_periods']}/{s['total_periods']}")


# ============================================================
# 7. COMPARISON WITH PHASE 6 BASELINE
# ============================================================
print("\n\n" + "=" * 80)
print("COMPARACIÓN: Fase 6 (baseline) vs Fase 7 (enriched)")
print("=" * 80)

# Phase 6 best was W3_momentum: 728 events, $11,639, 8.6 pts, Sharpe 2.57
best_name = ranked[0][0] if ranked else None
if best_name:
    best = all_results[best_name]['summary']
    print(f"\n  {'Metric':<25} {'Fase 6 (1t/event)':>18} {'Fase 7 best':>18}")
    print(f"  {'-'*61}")
    print(f"  {'Config':<25} {'W3_momentum':>18} {best_name:>18}")
    print(f"  {'Trades':<25} {'728':>18} {best['n_trades']:>18,}")
    print(f"  {'Trades/session':<25} {'~4.7/week':>18} {best['avg_trades_per_session']:>17.1f}")
    print(f"  {'Total PnL':<25} {'$11,639':>18} {'${:+,.0f}'.format(best['total_pnl']):>18}")
    print(f"  {'Avg pts/trade':<25} {'+8.6':>18} {'{:+.1f}'.format(best['avg_pts']):>18}")
    print(f"  {'Win Rate':<25} {'34.3%':>18} {'{:.1%}'.format(best['win_rate']):>18}")
    print(f"  {'Sharpe':<25} {'2.57':>18} {'{:.2f}'.format(best['sharpe']):>18}")
    print(f"  {'Max DD':<25} {'$-1,650':>18} {'${:,.0f}'.format(best['max_dd']):>18}")


# ============================================================
# 8. VISUALIZACIÓN
# ============================================================
print(f"\n[8] Generating plots...")

n_configs = min(len(ranked), 4)
fig, axes = plt.subplots(n_configs, 2, figsize=(18, 5 * n_configs))
if n_configs == 1:
    axes = axes.reshape(1, -1)

for idx in range(n_configs):
    name, res = ranked[idx]
    trades = res['trades']
    if not trades:
        continue

    pnls = np.array([t['pnl_dollars'] for t in trades])
    cum = np.cumsum(pnls)

    # Equity curve
    ax = axes[idx, 0]
    ax.plot(cum, linewidth=1, color='navy')
    ax.fill_between(range(len(cum)), cum, alpha=0.15, color='steelblue')
    ax.axhline(0, color='black', linewidth=0.5)
    s = res['summary']
    ax.set_title(f'{name}\n${s["total_pnl"]:+,.0f} | {s["n_trades"]} trades | '
                 f'{s["avg_pts"]:+.1f} pts | Sharpe={s["sharpe"]:.2f} | '
                 f'{s["avg_trades_per_session"]:.1f} t/sess', fontsize=9)
    ax.set_xlabel('Trade #')
    ax.set_ylabel('PnL ($)')
    ax.grid(True, alpha=0.3)

    # PnL by signal type and session
    ax = axes[idx, 1]
    cluster_pnls = [t['pnl_dollars'] for t in trades if t['signal_type'] == 'cluster']
    trans_pnls = [t['pnl_dollars'] for t in trades if t['signal_type'] == 'transition']

    if cluster_pnls:
        ax.plot(np.cumsum(cluster_pnls), linewidth=1, color='blue', alpha=0.7,
                label=f'Cluster ({len(cluster_pnls)}t, ${sum(cluster_pnls):+,.0f})')
    if trans_pnls:
        ax.plot(np.cumsum(trans_pnls), linewidth=1, color='red', alpha=0.7,
                label=f'Transition ({len(trans_pnls)}t, ${sum(trans_pnls):+,.0f})')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_title(f'{name} — Signal Type Breakdown', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fase7_enriched.png'), dpi=150, bbox_inches='tight')
print(f"  Saved: fase7_enriched.png")

# Save results
save_data = {}
for name, res in all_results.items():
    save_data[name] = {
        'summary': res['summary'],
        'period_stats': res['period_stats'],
    }
with open(os.path.join(OUTPUT_DIR, 'fase7_enriched.json'), 'w') as f:
    json.dump(save_data, f, indent=2, default=str)

with open(os.path.join(OUTPUT_DIR, 'fase7_trades.pkl'), 'wb') as f:
    pickle.dump(all_results, f)

print(f"  Saved: fase7_enriched.json, fase7_trades.pkl")
print("\n[FASE 7 COMPLETADA]")
