"""
FASE 8 — MULTI-TIMEFRAME MORPHOLOGY

El insight: no buscar patrones en 1 vela de 1 min (ruido),
sino en AGRUPACIONES de velas agregadas a distintos timeframes.

Approach:
1. Agregar 1-min bars → 5min, 15min, 30min OHLCV
2. Extraer morfología de GRUPOS de N velas agregadas
   - 3x5min  = 15 min de estructura de mercado
   - 3x15min = 45 min de estructura
   - 5x5min  = 25 min de estructura
   - 3x30min = 90 min de estructura
3. Clustering + rolling backtest (sin look-ahead)
4. Solo operar en regular hours, con costos realistas
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
print("FASE 8 — MULTI-TIMEFRAME MORPHOLOGY")
print("  Patrones en AGRUPACIONES de velas, no en velas individuales")
print("=" * 80)

# ============================================================
# 1. CARGAR DATOS 1-MIN
# ============================================================
print("\n[1] Cargando datos 1-min...")
df = pd.read_parquet(os.path.join(OUTPUT_DIR, 'mnq_clean.parquet'))
df['ts_et'] = pd.to_datetime(df['ts_et'])
df = df.sort_values('ts_et').reset_index(drop=True)
n_bars_1m = len(df)
print(f"  {n_bars_1m:,} barras de 1-min")

TICK_SIZE = 0.25
POINT_VALUE = 2.0
COMMISSION_RT = 1.24
SLIPPAGE_TICKS = 1

# ============================================================
# 2. AGREGAR A MULTI-TIMEFRAME
# ============================================================
print("\n[2] Agregando a múltiples timeframes...")


def aggregate_bars(df_1m, tf_minutes):
    """
    Aggregate 1-min bars to tf_minutes bars.
    Respects session boundaries (gaps > 5 min break the aggregation).
    Returns DataFrame with OHLCV at the new timeframe.
    """
    ts = df_1m['ts_et'].values
    opens = df_1m['open'].values.astype(np.float64)
    highs = df_1m['high'].values.astype(np.float64)
    lows = df_1m['low'].values.astype(np.float64)
    closes = df_1m['close'].values.astype(np.float64)
    volumes = df_1m['volume'].values.astype(np.float64)

    # Detect gaps (>5 min between bars)
    ts_int = ts.astype('int64') // 10**9
    gaps = np.zeros(len(ts), dtype=bool)
    gaps[1:] = np.diff(ts_int) > 300

    agg_rows = []
    i = 0
    n = len(ts)

    while i < n:
        # Start of new aggregated bar
        agg_open = opens[i]
        agg_high = highs[i]
        agg_low = lows[i]
        agg_close = closes[i]
        agg_vol = volumes[i]
        agg_ts = ts[i]
        count = 1
        # Index of last 1-min bar included in this agg bar
        last_1m_idx = i

        j = i + 1
        while j < n and count < tf_minutes:
            if gaps[j]:
                break  # session break, close current bar
            agg_high = max(agg_high, highs[j])
            agg_low = min(agg_low, lows[j])
            agg_close = closes[j]
            agg_vol += volumes[j]
            last_1m_idx = j
            count += 1
            j += 1

        # Only keep complete bars (got enough 1-min bars)
        if count == tf_minutes:
            agg_rows.append({
                'ts_et': agg_ts,
                'open': agg_open,
                'high': agg_high,
                'low': agg_low,
                'close': agg_close,
                'volume': agg_vol,
                'last_1m_idx': last_1m_idx,  # for backtest entry reference
            })

        i = j

    result = pd.DataFrame(agg_rows)
    return result


timeframes = {
    '5m': 5,
    '15m': 15,
    '30m': 30,
}

agg_data = {}
for tf_name, tf_min in timeframes.items():
    t0 = time.time()
    agg_df = aggregate_bars(df, tf_min)
    agg_data[tf_name] = agg_df
    print(f"  {tf_name}: {len(agg_df):,} barras ({time.time()-t0:.1f}s)")


# ============================================================
# 3. EXTRAER FEATURES MORFOLÓGICOS DE GRUPOS
# ============================================================
print("\n[3] Extrayendo features morfológicos de grupos...")

# Original 1-min arrays for backtesting
opens_1m = df['open'].values.astype(np.float64)
highs_1m = df['high'].values.astype(np.float64)
lows_1m = df['low'].values.astype(np.float64)
closes_1m = df['close'].values.astype(np.float64)
volumes_1m = df['volume'].values.astype(np.float64)
timestamps_1m = df['ts_et'].values
dates_1m = pd.to_datetime(timestamps_1m)

# Gap info for 1-min (for backtest)
gap_mask_1m = np.zeros(n_bars_1m, dtype=bool)
ts_diff_1m = np.diff(timestamps_1m.astype('int64')) / 1e9
gap_mask_1m[1:] = ts_diff_1m > 300
gap_cumsum_1m = np.cumsum(gap_mask_1m)


def extract_mtf_features(agg_df, group_size):
    """
    Extract morphological features from groups of `group_size` aggregated bars.

    Features per group:
    - Per-bar: normalized OHLC (relative to group open), body ratio, wick ratios
    - Group-level: total range, trend direction, volume profile
    """
    opens = agg_df['open'].values.astype(np.float64)
    highs = agg_df['high'].values.astype(np.float64)
    lows = agg_df['low'].values.astype(np.float64)
    closes = agg_df['close'].values.astype(np.float64)
    volumes = agg_df['volume'].values.astype(np.float64)
    last_1m = agg_df['last_1m_idx'].values.astype(np.int64)
    ts = agg_df['ts_et'].values

    n = len(agg_df)
    features_list = []
    entry_1m_indices = []  # 1-min bar index for trade entry (bar AFTER the group)

    # Detect gaps in aggregated bars (>gap between consecutive agg bars)
    ts_int = ts.astype('int64') // 10**9
    agg_gaps = np.zeros(n, dtype=bool)
    agg_gaps[1:] = np.diff(ts_int) > 3600  # >1 hour gap in agg bars = session break

    for i in range(group_size - 1, n):
        start = i - group_size + 1

        # Check no gap within group
        if np.any(agg_gaps[start+1:i+1]):
            continue

        o = opens[start:i+1]
        h = highs[start:i+1]
        l = lows[start:i+1]
        c = closes[start:i+1]
        v = volumes[start:i+1]

        base = o[0]
        if base == 0:
            continue

        # The entry bar in 1-min is the bar AFTER the last bar of the group
        entry_1m = last_1m[i] + 1
        if entry_1m >= n_bars_1m:
            continue

        feat = []

        # === Per-bar OHLC normalized to group open ===
        for j in range(group_size):
            feat.extend([
                (o[j] - base) / base * 1000,
                (h[j] - base) / base * 1000,
                (l[j] - base) / base * 1000,
                (c[j] - base) / base * 1000,
            ])

        # === Per-bar candle morphology ===
        for j in range(group_size):
            bar_range = h[j] - l[j]
            if bar_range > 0:
                body = abs(c[j] - o[j])
                body_ratio = body / bar_range
                upper_wick = (h[j] - max(o[j], c[j])) / bar_range
                lower_wick = (min(o[j], c[j]) - l[j]) / bar_range
                direction = 1.0 if c[j] >= o[j] else -1.0
            else:
                body_ratio = 0
                upper_wick = 0
                lower_wick = 0
                direction = 0
            feat.extend([body_ratio, upper_wick, lower_wick, direction])

        # === Volume profile across group ===
        v_total = v.sum()
        if v_total > 0:
            for j in range(group_size):
                feat.append(v[j] / v_total)
        else:
            feat.extend([1.0 / group_size] * group_size)

        # === Group-level summary features ===
        group_range = (h.max() - l.min()) / base * 1000
        group_trend = (c[-1] - o[0]) / base * 1000
        # Efficiency: how much of the range was captured by trend
        group_efficiency = group_trend / group_range if group_range > 0 else 0
        # Volatility (sum of bar ranges) vs group range
        bar_ranges = h - l
        sum_bar_ranges = bar_ranges.sum() / base * 1000
        choppiness = sum_bar_ranges / group_range if group_range > 0 else 1
        # Close position within group range
        close_position = (c[-1] - l.min()) / (h.max() - l.min()) if (h.max() - l.min()) > 0 else 0.5

        feat.extend([
            group_range,
            group_trend,
            group_efficiency,
            choppiness,
            close_position,
        ])

        features_list.append(feat)
        entry_1m_indices.append(entry_1m)

    if not features_list:
        return np.array([]), np.array([])

    return np.array(features_list, dtype=np.float32), np.array(entry_1m_indices, dtype=np.int64)


# Define configurations: (timeframe, group_size, description)
CONFIGS = [
    ('5m',  3, '3x5m=15min'),
    ('5m',  5, '5x5m=25min'),
    ('5m',  6, '6x5m=30min'),
    ('15m', 3, '3x15m=45min'),
    ('15m', 4, '4x15m=60min'),
    ('15m', 5, '5x15m=75min'),
    ('30m', 3, '3x30m=90min'),
    ('30m', 4, '4x30m=120min'),
]

mtf_features = {}
for tf_name, grp_size, desc in CONFIGS:
    key = f'{tf_name}_g{grp_size}'
    t0 = time.time()
    feats, entry_idx = extract_mtf_features(agg_data[tf_name], grp_size)
    mtf_features[key] = {'features': feats, 'entry_1m_idx': entry_idx, 'desc': desc}
    n_feats = feats.shape[1] if len(feats) > 0 else 0
    print(f"  {key} ({desc}): {len(feats):,} windows, {n_feats} features ({time.time()-t0:.1f}s)")


# ============================================================
# 4. ROLLING CLUSTERING + BACKTEST PER CONFIG
# ============================================================
print("\n[4] Rolling clustering + backtest per multi-TF config...")

N_CLUSTERS = 10
TRAIN_FRAC = 0.6      # Use first 60% for training
MIN_TRAIN = 500

# Trade params to sweep
TRADE_CONFIGS = [
    {'name': 'h5_sl10',  'hold': 5,  'sl': 10},
    {'name': 'h10_sl15', 'hold': 10, 'sl': 15},
    {'name': 'h15_sl20', 'hold': 15, 'sl': 20},
    {'name': 'h30_sl25', 'hold': 30, 'sl': 25},
]


def find_cluster_edges(entry_indices, labels, n_clusters):
    """Find directional edge per cluster from training data."""
    cluster_rets = {c: [] for c in range(n_clusters)}

    for i in range(len(entry_indices)):
        c = labels[i]
        entry_bar = entry_indices[i]
        if entry_bar >= n_bars_1m or gap_mask_1m[entry_bar]:
            continue
        # 2-bar forward return (enter at open of entry_bar, exit at close of entry_bar+1)
        exit_bar = entry_bar + 1
        if exit_bar >= n_bars_1m or gap_cumsum_1m[exit_bar] - gap_cumsum_1m[entry_bar] > 0:
            continue
        ret = (closes_1m[exit_bar] - opens_1m[entry_bar]) / opens_1m[entry_bar]
        cluster_rets[c].append(ret)

    edges = {}
    for c in range(n_clusters):
        r = cluster_rets[c]
        if len(r) < 30:
            continue
        avg = np.mean(r)
        std = np.std(r)
        if std == 0:
            continue
        t = avg / (std / np.sqrt(len(r)))
        if abs(t) > 2.0:
            edges[c] = {
                'direction': 'long' if avg > 0 else 'short',
                'avg_ret': float(avg),
                't_stat': float(t),
                'n': len(r),
            }
    return edges


def backtest_mtf(entry_indices, labels, cluster_edges, hold_bars, stop_loss_pts):
    """Backtest on forward data using training-derived edges."""
    trades = []

    for i in range(len(entry_indices)):
        c = labels[i]
        if c not in cluster_edges:
            continue

        entry_bar = int(entry_indices[i])
        if entry_bar >= n_bars_1m or gap_mask_1m[entry_bar]:
            continue

        # Session filter: only regular hours 9:30-15:30 ET
        h = dates_1m[entry_bar].hour
        m = dates_1m[entry_bar].minute
        tm = h * 60 + m
        if tm < 570 or tm > 930:  # 9:30 to 15:30
            continue

        direction = cluster_edges[c]['direction']
        entry_price = opens_1m[entry_bar]
        if direction == 'long':
            entry_price += SLIPPAGE_TICKS * TICK_SIZE
        else:
            entry_price -= SLIPPAGE_TICKS * TICK_SIZE

        exit_price = None
        exit_bar = None
        exit_reason = None

        for j in range(1, hold_bars + 1):
            bar_idx = entry_bar + j
            if bar_idx >= n_bars_1m:
                exit_price = closes_1m[n_bars_1m - 1]
                exit_reason = 'end'
                exit_bar = n_bars_1m - 1
                break
            if gap_cumsum_1m[bar_idx] - gap_cumsum_1m[entry_bar] > 0:
                exit_price = closes_1m[bar_idx - 1]
                exit_reason = 'gap'
                exit_bar = bar_idx - 1
                break
            if stop_loss_pts > 0:
                if direction == 'long' and lows_1m[bar_idx] <= entry_price - stop_loss_pts:
                    exit_price = entry_price - stop_loss_pts
                    exit_reason = 'sl'
                    exit_bar = bar_idx
                    break
                elif direction == 'short' and highs_1m[bar_idx] >= entry_price + stop_loss_pts:
                    exit_price = entry_price + stop_loss_pts
                    exit_reason = 'sl'
                    exit_bar = bar_idx
                    break
            if j == hold_bars:
                exit_price = closes_1m[bar_idx]
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
            'entry_bar': entry_bar,
            'exit_bar': exit_bar,
            'direction': direction,
            'cluster': int(c),
            'pnl_pts': float(pnl_pts),
            'pnl_dollars': float(pnl_dollars),
            'exit_reason': exit_reason,
            'session_time': int(tm),
        })

    return trades


# ============================================================
# MAIN ROLLING LOOP
# ============================================================

# Rolling parameters (in terms of 1-min bar indices)
TRAIN_BARS_1M = 294700 * 2   # ~6 months training
FORWARD_BARS_1M = 294700     # ~3 months forward

all_results = {}

for tf_name, grp_size, desc in CONFIGS:
    key = f'{tf_name}_g{grp_size}'
    feats = mtf_features[key]['features']
    entry_idx = mtf_features[key]['entry_1m_idx']

    if len(feats) < MIN_TRAIN + 100:
        print(f"\n  {key}: Insufficient data ({len(feats)}), skipping")
        continue

    print(f"\n  === {key} ({desc}) — {len(feats):,} windows ===")

    for tcfg in TRADE_CONFIGS:
        hold_bars = tcfg['hold']
        sl_pts = tcfg['sl']
        cfg_name = f'{key}_{tcfg["name"]}'

        all_trades = []
        period_stats = []
        cursor = TRAIN_BARS_1M
        step = 0

        while cursor < n_bars_1m:
            step += 1
            train_end = cursor
            forward_end = min(cursor + FORWARD_BARS_1M, n_bars_1m)

            train_mask = entry_idx < train_end
            fwd_mask = (entry_idx >= train_end) & (entry_idx < forward_end)

            train_feat = feats[train_mask]
            train_entry = entry_idx[train_mask]
            fwd_feat = feats[fwd_mask]
            fwd_entry = entry_idx[fwd_mask]

            if len(train_feat) < MIN_TRAIN or len(fwd_feat) < 50:
                cursor = forward_end
                continue

            scaler = StandardScaler()
            train_sc = scaler.fit_transform(train_feat)
            fwd_sc = scaler.transform(fwd_feat)

            kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=10, max_iter=300, random_state=42)
            train_labels = kmeans.fit_predict(train_sc)
            fwd_labels = kmeans.predict(fwd_sc)

            # Find edges from training
            edges = find_cluster_edges(train_entry, train_labels, N_CLUSTERS)

            # Backtest forward
            period_trades = backtest_mtf(fwd_entry, fwd_labels, edges, hold_bars, sl_pts)

            n_t = len(period_trades)
            pnl = sum(t['pnl_dollars'] for t in period_trades) if n_t else 0

            period_stats.append({
                'period': step, 'n_trades': n_t, 'total_pnl': pnl,
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
            profitable_periods = sum(1 for p in period_stats if p['total_pnl'] > 0)
        else:
            total_pnl = avg_pnl = avg_pts = win_rate = sharpe = max_dd = 0
            profitable_periods = 0

        total_periods = len(period_stats)

        all_results[cfg_name] = {
            'trades': all_trades,
            'period_stats': period_stats,
            'summary': {
                'n_trades': n_total, 'total_pnl': total_pnl,
                'avg_pnl': avg_pnl, 'avg_pts': avg_pts,
                'win_rate': win_rate, 'sharpe': sharpe, 'max_dd': max_dd,
                'profitable_periods': profitable_periods,
                'total_periods': total_periods,
                'tf': tf_name, 'group_size': grp_size, 'desc': desc,
                'hold_bars': hold_bars, 'sl_pts': sl_pts,
            }
        }

        if n_total > 10:
            marker = "***" if sharpe > 1.0 else ""
            print(f"    {tcfg['name']}: {n_total:>5} trades | ${total_pnl:>+10,.0f} | "
                  f"{avg_pts:>+5.1f} pts | WR={win_rate:.1%} | Sharpe={sharpe:>6.2f} | "
                  f"DD=${max_dd:>8,.0f} | {profitable_periods}/{total_periods} per+ {marker}")


# ============================================================
# 5. RANKING
# ============================================================
print("\n\n" + "=" * 80)
print("RANKING — TOP 20 MULTI-TIMEFRAME CONFIGS")
print("=" * 80)

ranked = sorted(
    [(k, v) for k, v in all_results.items() if v['summary']['n_trades'] > 30],
    key=lambda x: x[1]['summary']['sharpe'],
    reverse=True
)[:20]

print(f"\n{'Config':<32} {'TF':>5} {'Trades':>6} {'PnL':>12} {'Pts':>6} "
      f"{'WR':>6} {'Sharpe':>7} {'MaxDD':>10} {'Per+':>5}")
print("-" * 100)

for name, res in ranked:
    s = res['summary']
    print(f"{name:<32} {s['desc']:>5} {s['n_trades']:>6} "
          f"${s['total_pnl']:>+10,.0f} {s['avg_pts']:>+5.1f} "
          f"{s['win_rate']:>5.1%} {s['sharpe']:>7.2f} ${s['max_dd']:>9,.0f} "
          f"{s['profitable_periods']}/{s['total_periods']}")


# ============================================================
# 6. SESSION ANALYSIS OF TOP CONFIGS
# ============================================================
print("\n\n" + "=" * 80)
print("SESSION ANALYSIS — TOP 5 CONFIGS")
print("=" * 80)

for name, res in ranked[:5]:
    trades = res['trades']
    if not trades:
        continue
    print(f"\n  {name} ({res['summary']['desc']}):")

    session_pnls = {}
    for t in trades:
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

    for s in sorted(session_pnls.keys()):
        sp = np.array(session_pnls[s])
        avg_p = (sp.mean() + COMMISSION_RT) / POINT_VALUE
        print(f"    {s}: {len(sp):>4} trades, ${sp.sum():>+10,.0f}, "
              f"WR={float((sp>0).mean()):.1%}, avg={avg_p:+.1f} pts")

    # Direction breakdown
    long_pnl = sum(t['pnl_dollars'] for t in trades if t['direction'] == 'long')
    short_pnl = sum(t['pnl_dollars'] for t in trades if t['direction'] == 'short')
    n_long = sum(1 for t in trades if t['direction'] == 'long')
    n_short = sum(1 for t in trades if t['direction'] == 'short')
    print(f"    Long:  {n_long:>4} trades, ${long_pnl:>+10,.0f}")
    print(f"    Short: {n_short:>4} trades, ${short_pnl:>+10,.0f}")


# ============================================================
# 7. WALK-FORWARD STABILITY TEST ON TOP 3
# ============================================================
print("\n\n" + "=" * 80)
print("WALK-FORWARD PERIOD STABILITY — TOP 3")
print("=" * 80)

for name, res in ranked[:3]:
    ps = res['period_stats']
    trades = res['trades']
    s = res['summary']
    print(f"\n  {name}:")
    print(f"    {'Period':>8} {'Trades':>7} {'PnL':>12}")
    print(f"    {'-'*30}")
    for p in ps:
        marker = "+" if p['total_pnl'] > 0 else "-"
        print(f"    {p['period']:>8} {p['n_trades']:>7} ${p['total_pnl']:>+10,.0f} [{marker}]")
    print(f"    {'TOTAL':>8} {s['n_trades']:>7} ${s['total_pnl']:>+10,.0f} "
          f"[{s['profitable_periods']}/{s['total_periods']} positive]")


# ============================================================
# 8. VISUALIZACIÓN
# ============================================================
print(f"\n[8] Generating plots...")

n_plot = min(len(ranked), 6)
fig, axes = plt.subplots(n_plot, 2, figsize=(18, 5 * n_plot))
if n_plot == 1:
    axes = axes.reshape(1, -1)

for idx in range(n_plot):
    name, res = ranked[idx]
    trades = res['trades']
    if not trades:
        continue

    pnls = np.array([t['pnl_dollars'] for t in trades])
    cum = np.cumsum(pnls)
    s = res['summary']

    # Equity curve
    ax = axes[idx, 0]
    ax.plot(cum, linewidth=1, color='navy')
    ax.fill_between(range(len(cum)), cum, alpha=0.15, color='steelblue')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_title(f'{name} ({s["desc"]})\n${s["total_pnl"]:+,.0f} | {s["n_trades"]} trades | '
                 f'{s["avg_pts"]:+.1f} pts | Sharpe={s["sharpe"]:.2f}', fontsize=9)
    ax.set_xlabel('Trade #')
    ax.set_ylabel('PnL ($)')
    ax.grid(True, alpha=0.3)

    # Period PnL
    ax = axes[idx, 1]
    ps = res['period_stats']
    period_nums = [p['period'] for p in ps]
    period_pnls = [p['total_pnl'] for p in ps]
    colors = ['green' if p > 0 else 'red' for p in period_pnls]
    ax.bar(period_nums, period_pnls, color=colors, alpha=0.7, edgecolor='black', linewidth=0.3)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_title(f'{name} — PnL per Rolling Period\n'
                 f'{s["profitable_periods"]}/{s["total_periods"]} profitable', fontsize=9)
    ax.set_xlabel('Period')
    ax.set_ylabel('PnL ($)')
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fase8_multitimeframe.png'), dpi=150, bbox_inches='tight')
print(f"  Saved: fase8_multitimeframe.png")

# Save results
save_data = {}
for name, res in all_results.items():
    save_data[name] = {
        'summary': res['summary'],
        'period_stats': res['period_stats'],
    }
with open(os.path.join(OUTPUT_DIR, 'fase8_multitimeframe.json'), 'w') as f:
    json.dump(save_data, f, indent=2, default=str)

with open(os.path.join(OUTPUT_DIR, 'fase8_trades.pkl'), 'wb') as f:
    pickle.dump({k: {'trades': v['trades'], 'summary': v['summary']} for k, v in all_results.items()}, f)

print(f"  Saved: fase8_multitimeframe.json, fase8_trades.pkl")

# ============================================================
# 9. FINAL VERDICT
# ============================================================
print("\n\n" + "=" * 80)
print("VEREDICTO FASE 8")
print("=" * 80)

if ranked:
    best_name, best_res = ranked[0]
    bs = best_res['summary']
    print(f"\n  Mejor config: {best_name} ({bs['desc']})")
    print(f"  Trades: {bs['n_trades']}")
    print(f"  PnL: ${bs['total_pnl']:+,.0f}")
    print(f"  Pts/trade: {bs['avg_pts']:+.1f}")
    print(f"  Win Rate: {bs['win_rate']:.1%}")
    print(f"  Sharpe: {bs['sharpe']:.2f}")
    print(f"  Max DD: ${bs['max_dd']:,.0f}")
    print(f"  Períodos rentables: {bs['profitable_periods']}/{bs['total_periods']}")

    if bs['sharpe'] > 1.5 and bs['profitable_periods'] >= bs['total_periods'] * 0.6:
        print(f"\n  *** PROMISING: Sharpe > 1.5 y >60% períodos rentables ***")
    elif bs['sharpe'] > 0.5:
        print(f"\n  Resultado moderado — hay señal pero necesita refinamiento")
    else:
        print(f"\n  Sin edge significativo en multi-timeframe morphology tampoco")
else:
    print("\n  No configs with enough trades to evaluate.")

print("\n[FASE 8 COMPLETADA]")
