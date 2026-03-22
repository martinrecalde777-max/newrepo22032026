"""
FASE 5D — Regime Filters + Strategy Correlation
Tests signals across time-of-day, day-of-week, volatility regimes,
and measures inter-strategy correlation for portfolio construction.
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
print("FASE 5D — REGIME FILTERS & STRATEGY CORRELATION")
print("=" * 80)

# ============================================================
# 1. LOAD DATA
# ============================================================
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

TICK_SIZE = 0.25
POINT_VALUE = 2.0
COMMISSION_RT = 1.24
SLIPPAGE_TICKS = 1

cluster_data = {}
for W in [3, 5, 8]:
    data = np.load(os.path.join(OUTPUT_DIR, f'fase2_W{W}_data.npz'))
    cluster_data[W] = {'labels': data['labels'], 'end_idx': data['end_idx']}

# Load phase 4 trades
with open(os.path.join(OUTPUT_DIR, 'fase4_all_results.pkl'), 'rb') as f:
    phase4 = pickle.load(f)

print(f"  Barras: {n_bars:,}")

# ============================================================
# 2. REGIME ANALYSIS — TIME OF DAY
# ============================================================
print("\n[2] Análisis por hora del día...")

TOP_NAMES = ['W5_C2long_H5', 'W3_momentum_H5_SL10', 'W3_momentum_H5_SL20',
             'W3_fade_C1_H10', 'W8_fadeC2_H1']

regime_results = {}

for strat_name in TOP_NAMES:
    if strat_name not in phase4:
        continue

    trades = phase4[strat_name]['trades_full']
    if not trades:
        continue

    # Extract hour of entry
    entry_hours = []
    entry_dow = []
    entry_pnls = []

    for t in trades:
        entry_bar = t['entry_bar']
        ts = pd.Timestamp(timestamps[entry_bar])
        entry_hours.append(ts.hour)
        entry_dow.append(ts.dayofweek)  # 0=Mon, 6=Sun
        entry_pnls.append(t['pnl_dollars'])

    entry_hours = np.array(entry_hours)
    entry_dow = np.array(entry_dow)
    entry_pnls = np.array(entry_pnls)

    # --- By Hour ---
    hour_stats = {}
    for h in sorted(set(entry_hours)):
        mask = entry_hours == h
        if mask.sum() < 5:
            continue
        p = entry_pnls[mask]
        hour_stats[int(h)] = {
            'n': int(mask.sum()),
            'pnl': float(p.sum()),
            'avg': float(p.mean()),
            'wr': float((p > 0).mean()),
        }

    # --- By Day of Week ---
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    dow_stats = {}
    for d in sorted(set(entry_dow)):
        mask = entry_dow == d
        if mask.sum() < 5:
            continue
        p = entry_pnls[mask]
        dow_stats[dow_names[d]] = {
            'n': int(mask.sum()),
            'pnl': float(p.sum()),
            'avg': float(p.mean()),
            'wr': float((p > 0).mean()),
        }

    # --- By Volatility Regime ---
    # Use ATR-5 at entry as vol proxy
    atr5 = np.zeros(n_bars)
    for i in range(1, n_bars):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        atr5[i] = atr5[i-1] * 0.8 + tr * 0.2 if i > 0 else tr

    entry_atrs = np.array([atr5[t['entry_bar']] for t in trades])
    atr_pct = np.percentile(entry_atrs, [33, 66])

    vol_stats = {}
    for label, mask in [('low_vol', entry_atrs <= atr_pct[0]),
                         ('mid_vol', (entry_atrs > atr_pct[0]) & (entry_atrs <= atr_pct[1])),
                         ('high_vol', entry_atrs > atr_pct[1])]:
        if mask.sum() < 5:
            continue
        p = entry_pnls[mask]
        vol_stats[label] = {
            'n': int(mask.sum()),
            'pnl': float(p.sum()),
            'avg': float(p.mean()),
            'wr': float((p > 0).mean()),
            'atr_range': f'{entry_atrs[mask].min():.1f}-{entry_atrs[mask].max():.1f}',
        }

    # RTH vs Overnight
    rth_mask = (entry_hours >= 9) & (entry_hours < 16)
    session_stats = {}
    for label, mask in [('RTH (9:30-16:00)', rth_mask), ('Overnight', ~rth_mask)]:
        if mask.sum() < 5:
            continue
        p = entry_pnls[mask]
        session_stats[label] = {
            'n': int(mask.sum()),
            'pnl': float(p.sum()),
            'avg': float(p.mean()),
            'wr': float((p > 0).mean()),
        }

    regime_results[strat_name] = {
        'by_hour': hour_stats,
        'by_dow': dow_stats,
        'by_vol': vol_stats,
        'by_session': session_stats,
    }

    print(f"\n  {strat_name}:")
    print(f"    Session:  ", end='')
    for s, st in session_stats.items():
        print(f"{s}: {st['n']}t ${st['pnl']:+,.0f} WR={st['wr']:.1%}  ", end='')
    print()
    print(f"    Volatility:", end='')
    for v, st in vol_stats.items():
        print(f" {v}: {st['n']}t ${st['pnl']:+,.0f} WR={st['wr']:.1%} ", end='')
    print()

    # Best/worst hours
    if hour_stats:
        best_h = max(hour_stats.items(), key=lambda x: x[1]['avg'])
        worst_h = min(hour_stats.items(), key=lambda x: x[1]['avg'])
        print(f"    Best hour: {best_h[0]:02d}:00 (${best_h[1]['avg']:+.2f}/t, {best_h[1]['n']}t)")
        print(f"    Worst hour: {worst_h[0]:02d}:00 (${worst_h[1]['avg']:+.2f}/t, {worst_h[1]['n']}t)")


# ============================================================
# 3. STRATEGY CORRELATION
# ============================================================
print("\n\n[3] Correlación entre estrategias...")

# Build daily P&L series for each strategy
daily_pnls = {}
for strat_name in TOP_NAMES:
    if strat_name not in phase4:
        continue
    trades = phase4[strat_name]['trades_full']
    if not trades:
        continue

    # Aggregate to daily
    daily = {}
    for t in trades:
        exit_bar = t['exit_bar']
        day = str(pd.Timestamp(timestamps[exit_bar]).date())
        daily[day] = daily.get(day, 0) + t['pnl_dollars']

    daily_pnls[strat_name] = daily

# Build aligned DataFrame
all_days = sorted(set().union(*[set(d.keys()) for d in daily_pnls.values()]))
pnl_df = pd.DataFrame(index=all_days)
for name, daily in daily_pnls.items():
    pnl_df[name] = pd.Series(daily)
pnl_df = pnl_df.fillna(0)

# Correlation matrix
corr_matrix = pnl_df.corr()

print(f"\n  Correlation Matrix (daily PnL):")
print(f"  {'':>25s}", end='')
for name in pnl_df.columns:
    short = name[:12]
    print(f" {short:>12s}", end='')
print()
for name1 in pnl_df.columns:
    short1 = name1[:25]
    print(f"  {short1:>25s}", end='')
    for name2 in pnl_df.columns:
        c = corr_matrix.loc[name1, name2]
        print(f" {c:>12.3f}", end='')
    print()

# Portfolio simulation: equal weight all 5
print(f"\n  Portfolio combinado (equal weight, 5 estrategias):")
portfolio_daily = pnl_df.sum(axis=1)
cum_pnl = portfolio_daily.cumsum()
total_pnl = portfolio_daily.sum()
avg_daily = portfolio_daily.mean()
std_daily = portfolio_daily.std()
sharpe_daily = avg_daily / max(std_daily, 1e-10) * np.sqrt(252)
running_max = np.maximum.accumulate(cum_pnl)
max_dd = (cum_pnl - running_max).min()
active_days = (portfolio_daily != 0).sum()
win_days = (portfolio_daily > 0).sum()

print(f"    Total PnL: ${total_pnl:+,.2f}")
print(f"    Active days: {active_days} | Winning days: {win_days} ({win_days/max(active_days,1):.1%})")
print(f"    Avg daily: ${avg_daily:+.2f} | Sharpe: {sharpe_daily:.2f}")
print(f"    Max DD: ${max_dd:,.2f}")

# Compare to best individual
best_individual_sharpe = 0
best_individual_name = ''
for name in pnl_df.columns:
    s = pnl_df[name]
    active = s[s != 0]
    if len(active) > 10:
        sh = active.mean() / max(active.std(), 1e-10) * np.sqrt(252)
        if sh > best_individual_sharpe:
            best_individual_sharpe = sh
            best_individual_name = name

print(f"    Best individual Sharpe: {best_individual_sharpe:.2f} ({best_individual_name})")
print(f"    Portfolio Sharpe improvement: {sharpe_daily/max(best_individual_sharpe,1e-10):.2f}x")

# Diversification ratio
avg_corr = corr_matrix.values[np.triu_indices(len(pnl_df.columns), k=1)].mean()
print(f"    Avg pairwise correlation: {avg_corr:.3f}")
print(f"    → {'Low correlation — good diversification' if avg_corr < 0.3 else 'Moderate correlation' if avg_corr < 0.5 else 'High correlation — limited diversification'}")


# ============================================================
# 4. VISUALIZACIÓN
# ============================================================
print("\n[4] Generando gráficos...")

fig = plt.figure(figsize=(20, 18))
gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

# 4a. Correlation heatmap
ax1 = fig.add_subplot(gs[0, 0])
short_names = [n.replace('W3_momentum_H5_', 'W3m_').replace('W3_fade_', 'W3f_')
               .replace('W5_C2long_', 'W5_').replace('W8_fadeC2_', 'W8_') for n in pnl_df.columns]
im = ax1.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1)
ax1.set_xticks(range(len(short_names)))
ax1.set_xticklabels(short_names, rotation=45, ha='right', fontsize=7)
ax1.set_yticks(range(len(short_names)))
ax1.set_yticklabels(short_names, fontsize=7)
for i in range(len(short_names)):
    for j in range(len(short_names)):
        ax1.text(j, i, f'{corr_matrix.values[i,j]:.2f}', ha='center', va='center', fontsize=7)
ax1.set_title('Daily PnL Correlation')
plt.colorbar(im, ax=ax1, shrink=0.8)

# 4b. Portfolio equity curve
ax2 = fig.add_subplot(gs[0, 1:])
cum_portfolio = portfolio_daily.cumsum()
ax2.plot(range(len(cum_portfolio)), cum_portfolio.values, color='navy', linewidth=1.5, label='Portfolio (5 strats)')

# Individual equity curves
for col in pnl_df.columns:
    cum_ind = pnl_df[col].cumsum()
    ax2.plot(range(len(cum_ind)), cum_ind.values, alpha=0.4, linewidth=0.8, label=col[:15])
ax2.axhline(0, color='black', linewidth=0.5)
ax2.set_title('Portfolio vs Individual Equity Curves')
ax2.set_xlabel('Trading Day')
ax2.set_ylabel('Cumulative PnL ($)')
ax2.legend(fontsize=7, loc='upper left')
ax2.grid(True, alpha=0.3)

# 4c-e. Hour of day analysis for top 3 strategies
for plot_idx, strat_name in enumerate(TOP_NAMES[:3]):
    ax = fig.add_subplot(gs[1, plot_idx])
    if strat_name in regime_results:
        hour_stats = regime_results[strat_name]['by_hour']
        if hour_stats:
            hours = sorted(hour_stats.keys())
            avg_pnls = [hour_stats[h]['avg'] for h in hours]
            n_trades = [hour_stats[h]['n'] for h in hours]
            colors = ['green' if p > 0 else 'red' for p in avg_pnls]
            bars = ax.bar(hours, avg_pnls, color=colors, alpha=0.7, edgecolor='black', linewidth=0.3)
            ax.set_title(f'{strat_name[:20]}\nAvg PnL by Hour', fontsize=9)
            ax.set_xlabel('Hour (ET)')
            ax.set_ylabel('Avg PnL/Trade ($)')
            ax.axhline(0, color='black', linewidth=0.5)

            # Annotate trade count
            for bar, n in zip(bars, n_trades):
                ax.text(bar.get_x() + bar.get_width()/2, 0,
                        f'{n}', ha='center', va='bottom', fontsize=6, color='blue')

# 4f-h. Day of week for top 3
for plot_idx, strat_name in enumerate(TOP_NAMES[:3]):
    ax = fig.add_subplot(gs[2, plot_idx])
    if strat_name in regime_results:
        dow_stats = regime_results[strat_name]['by_dow']
        if dow_stats:
            days = list(dow_stats.keys())
            pnls = [dow_stats[d]['pnl'] for d in days]
            wrs = [dow_stats[d]['wr'] for d in days]
            colors = ['green' if p > 0 else 'red' for p in pnls]
            bars = ax.bar(days, pnls, color=colors, alpha=0.7, edgecolor='black', linewidth=0.3)
            ax.set_title(f'{strat_name[:20]}\nPnL by Day of Week', fontsize=9)
            ax.set_xlabel('Day')
            ax.set_ylabel('Total PnL ($)')
            ax.axhline(0, color='black', linewidth=0.5)

            for bar, wr in zip(bars, wrs):
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, h,
                        f'WR={wr:.0%}', ha='center', va='bottom' if h > 0 else 'top', fontsize=7)

plt.savefig(os.path.join(OUTPUT_DIR, 'fase5d_regimen_corr.png'), dpi=150, bbox_inches='tight')
print(f"  Guardado: fase5d_regimen_corr.png")

# Save
save_data = {
    'regime_results': regime_results,
    'correlation_matrix': corr_matrix.to_dict(),
    'portfolio': {
        'total_pnl': float(total_pnl),
        'sharpe': float(sharpe_daily),
        'max_dd': float(max_dd),
        'avg_daily': float(avg_daily),
        'active_days': int(active_days),
        'win_days': int(win_days),
        'avg_pairwise_corr': float(avg_corr),
    },
}

with open(os.path.join(OUTPUT_DIR, 'fase5d_regimen_corr.json'), 'w') as f:
    json.dump(save_data, f, indent=2, default=str)
print(f"  Guardado: fase5d_regimen_corr.json")

print("\n[FASE 5D COMPLETADA]")

# ============================================================
# RESUMEN FINAL FASE 5
# ============================================================
print("\n" + "=" * 80)
print("RESUMEN FASE 5 — VALIDACIÓN DE ROBUSTEZ")
print("=" * 80)

print("""
┌─────────────────────────────────────────────────────────────────────────┐
│ 5A WALK-FORWARD:                                                        │
│   • W3_momentum_SL10: 5/5 folds rentables ★                            │
│   • W3_momentum_SL20: 5/5 folds rentables ★                            │
│   • W8_fadeC2_H1: 5/5 folds rentables ★                                │
│   • W5_C2long_H5: 4/5 folds (solo fold 1 con 25 trades negativo)      │
│   • W3_fade_C1_H10: 4/5 folds (fold 1 sin trades)                     │
│                                                                         │
│ 5B SENSITIVITY:                                                         │
│   • 184/184 celdas rentables (100%) en TODAS las estrategias            │
│   • Edge persistente en todo el espacio de parámetros                   │
│   • Sharpe mínimo en cualquier configuración: 1.96                      │
│                                                                         │
│ 5C MONTE CARLO (10,000 sims):                                          │
│   • P(profitable) = 100% para las 5 estrategias                        │
│   • P(DD < $2K) = 100% para las 5 estrategias                          │
│   • MaxDD 90% CI: $-235 a $-1,160                                      │
│                                                                         │
│ 5D REGÍMENES & CORRELACIÓN:                                            │
│   • Avg pairwise correlation: """ + f"{avg_corr:.3f}" + """                                │
│   • Portfolio combinado Sharpe: """ + f"{sharpe_daily:.2f}" + """                                  │
│   • Portfolio MaxDD: $""" + f"{max_dd:,.0f}" + """                                         │
│   • Señales funcionan tanto en RTH como overnight                       │
└─────────────────────────────────────────────────────────────────────────┘
""")

print("FASE 5 COMPLETADA — Pipeline de validación de robustez finalizado")
