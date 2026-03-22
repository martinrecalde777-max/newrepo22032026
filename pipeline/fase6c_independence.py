"""
FASE 6C — Análisis de Independencia de Trades en Ráfaga

PREGUNTA: Cuando hay 60 trades en un día, ¿son señales independientes
o es la misma señal repetida? Si son redundantes, el PnL real es mucho menor.

TESTS:
1. Autocorrelación temporal de PnL entre trades consecutivos
2. Trades overlapping (misma barra de entry/exit)
3. Clustering temporal: ¿cuántos "eventos" únicos hay?
4. PnL por evento único vs por trade individual
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

warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/user/newrepo22032026/outputs'

print("=" * 80)
print("FASE 6C — ANÁLISIS DE INDEPENDENCIA DE TRADES")
print("=" * 80)

# Load clean trades from 6AB
with open(os.path.join(OUTPUT_DIR, 'fase6ab_trades.pkl'), 'rb') as f:
    rolling_results = pickle.load(f)

# Load timestamps
df = pd.read_parquet(os.path.join(OUTPUT_DIR, 'mnq_clean.parquet'))
df['ts_et'] = pd.to_datetime(df['ts_et'])
timestamps = df['ts_et'].values

independence_results = {}

for strat_name in ['W3_momentum', 'W3_fade_C1']:
    res = rolling_results.get(strat_name)
    if not res or not res['trades']:
        continue

    trades = res['trades']
    n_trades = len(trades)

    print(f"\n{'='*60}")
    print(f"  {strat_name} — {n_trades} trades")
    print(f"{'='*60}")

    entry_bars = np.array([t['entry_bar'] for t in trades])
    exit_bars = np.array([t['exit_bar'] for t in trades])
    pnls = np.array([t['pnl_dollars'] for t in trades])
    directions = [t['direction'] for t in trades]

    # ============================================================
    # 1. TEMPORAL CLUSTERING — Define "events"
    # ============================================================
    # An "event" = group of trades whose entry_bars are within GAP_THRESHOLD bars of each other
    GAP_THRESHOLD = 30  # 30 bars = 30 minutes

    sorted_idx = np.argsort(entry_bars)
    sorted_entries = entry_bars[sorted_idx]
    sorted_pnls = pnls[sorted_idx]
    sorted_exits = exit_bars[sorted_idx]

    # Group into events
    events = []
    current_event = [0]

    for i in range(1, len(sorted_entries)):
        if sorted_entries[i] - sorted_entries[current_event[-1]] <= GAP_THRESHOLD:
            current_event.append(i)
        else:
            events.append(current_event)
            current_event = [i]
    events.append(current_event)

    n_events = len(events)
    event_sizes = [len(e) for e in events]

    # Event-level PnL
    event_pnls = [sorted_pnls[e].sum() for e in events]
    event_avg_pnl = [sorted_pnls[e].mean() for e in events]

    print(f"\n  [1] CLUSTERING TEMPORAL (gap > {GAP_THRESHOLD} bars = nuevo evento)")
    print(f"      Trades totales: {n_trades}")
    print(f"      Eventos únicos: {n_events}")
    print(f"      Trades/evento: min={min(event_sizes)} max={max(event_sizes)} "
          f"avg={np.mean(event_sizes):.1f} median={np.median(event_sizes):.0f}")
    print(f"      Eventos de 1 trade: {sum(1 for s in event_sizes if s == 1)} ({sum(1 for s in event_sizes if s == 1)/n_events*100:.0f}%)")
    print(f"      Eventos de >10 trades: {sum(1 for s in event_sizes if s > 10)} ({sum(1 for s in event_sizes if s > 10)/n_events*100:.0f}%)")

    # ============================================================
    # 2. OVERLAP ANALYSIS
    # ============================================================
    # How many trades are "alive" at the same time?
    n_overlapping = 0
    for i in range(1, len(sorted_entries)):
        if sorted_entries[i] <= sorted_exits[i-1]:
            n_overlapping += 1

    overlap_pct = n_overlapping / max(n_trades - 1, 1) * 100
    print(f"\n  [2] OVERLAP (trade entra antes de que salga el anterior)")
    print(f"      Trades con overlap: {n_overlapping}/{n_trades} ({overlap_pct:.0f}%)")

    # ============================================================
    # 3. AUTOCORRELACIÓN DE PnL
    # ============================================================
    # Si trades consecutivos tienen alta autocorrelación, no son independientes
    if len(sorted_pnls) > 10:
        autocorr_1 = np.corrcoef(sorted_pnls[:-1], sorted_pnls[1:])[0, 1]
        autocorr_5 = np.corrcoef(sorted_pnls[:-5], sorted_pnls[5:])[0, 1] if len(sorted_pnls) > 10 else 0

        # Within-event autocorrelation
        within_corrs = []
        for event in events:
            if len(event) > 3:
                ep = sorted_pnls[event]
                c = np.corrcoef(ep[:-1], ep[1:])[0, 1]
                if not np.isnan(c):
                    within_corrs.append(c)

        avg_within_corr = np.mean(within_corrs) if within_corrs else 0

        print(f"\n  [3] AUTOCORRELACIÓN de PnL")
        print(f"      Lag-1 (trade consecutivo): {autocorr_1:.3f}")
        print(f"      Lag-5: {autocorr_5:.3f}")
        print(f"      Within-event avg autocorr: {avg_within_corr:.3f}")
        if abs(autocorr_1) > 0.3:
            print(f"      ⚠ Alta autocorrelación — trades NO son independientes")
        elif abs(autocorr_1) > 0.1:
            print(f"      ⚡ Autocorrelación moderada")
        else:
            print(f"      ✓ Baja autocorrelación — trades razonablemente independientes")

    # ============================================================
    # 4. SAME DIRECTION WITHIN EVENT
    # ============================================================
    same_direction_pct = []
    for event in events:
        if len(event) > 1:
            dirs = [directions[sorted_idx[i]] for i in event]
            most_common = max(set(dirs), key=dirs.count)
            pct = dirs.count(most_common) / len(dirs)
            same_direction_pct.append(pct)

    avg_same_dir = np.mean(same_direction_pct) if same_direction_pct else 0
    print(f"\n  [4] DIRECCIÓN DENTRO DE EVENTOS")
    print(f"      % misma dirección (avg): {avg_same_dir:.1%}")
    if avg_same_dir > 0.9:
        print(f"      → Casi todos los trades del evento van en la misma dirección")
        print(f"      → Son señales REDUNDANTES del mismo movimiento")
    elif avg_same_dir > 0.7:
        print(f"      → Mayoría misma dirección — parcialmente redundantes")
    else:
        print(f"      → Dirección mixta — señales más independientes")

    # ============================================================
    # 5. EVENT-LEVEL STATISTICS (la métrica real)
    # ============================================================
    event_pnls_arr = np.array(event_pnls)
    event_total = event_pnls_arr.sum()
    event_avg = event_pnls_arr.mean()
    event_wr = (event_pnls_arr > 0).mean()
    event_sharpe = event_pnls_arr.mean() / max(event_pnls_arr.std(), 1e-10) * np.sqrt(252)
    event_cum = np.cumsum(event_pnls_arr)
    event_dd = (event_cum - np.maximum.accumulate(event_cum)).min()

    # Pts per event
    event_avg_pts = (event_avg + 1.24) / 2.0  # approximate

    print(f"\n  [5] ESTADÍSTICAS POR EVENTO (vs por trade)")
    print(f"      {'Métrica':<20} {'Por Trade':>15} {'Por Evento':>15}")
    print(f"      {'-'*50}")
    print(f"      {'Count':<20} {n_trades:>15,} {n_events:>15,}")
    print(f"      {'Total PnL':<20} {'${:+,.0f}'.format(pnls.sum()):>15} {'${:+,.0f}'.format(event_total):>15}")
    print(f"      {'Avg PnL':<20} {'${:+,.2f}'.format(pnls.mean()):>15} {'${:+,.2f}'.format(event_avg):>15}")
    print(f"      {'Win Rate':<20} {'{:.1%}'.format((pnls>0).mean()):>15} {'{:.1%}'.format(event_wr):>15}")
    print(f"      {'Sharpe':<20} {'{:.2f}'.format(pnls.mean()/max(pnls.std(),1e-10)*np.sqrt(252)):>15} {'{:.2f}'.format(event_sharpe):>15}")
    print(f"      {'Max DD':<20} {'${:,.0f}'.format((np.cumsum(pnls)-np.maximum.accumulate(np.cumsum(pnls))).min()):>15} {'${:,.0f}'.format(event_dd):>15}")

    # ============================================================
    # 6. WHAT IF WE TAKE ONLY 1 TRADE PER EVENT?
    # ============================================================
    # Take only the first trade of each event
    first_trade_pnls = np.array([sorted_pnls[e[0]] for e in events])
    ft_total = first_trade_pnls.sum()
    ft_avg = first_trade_pnls.mean()
    ft_wr = (first_trade_pnls > 0).mean()
    ft_sharpe = first_trade_pnls.mean() / max(first_trade_pnls.std(), 1e-10) * np.sqrt(252)
    ft_avg_pts = (ft_avg + 1.24) / 2.0

    print(f"\n  [6] SOLO 1 TRADE POR EVENTO (first trade):")
    print(f"      Trades: {n_events}")
    print(f"      Total PnL: ${ft_total:+,.2f}")
    print(f"      Avg PnL: ${ft_avg:+.2f} ({ft_avg_pts:+.1f} pts)")
    print(f"      Win Rate: {ft_wr:.1%}")
    print(f"      Sharpe: {ft_sharpe:.2f}")

    independence_results[strat_name] = {
        'n_trades': n_trades,
        'n_events': n_events,
        'trades_per_event_avg': float(np.mean(event_sizes)),
        'trades_per_event_median': float(np.median(event_sizes)),
        'overlap_pct': overlap_pct,
        'autocorr_lag1': float(autocorr_1) if 'autocorr_1' in dir() else 0,
        'avg_same_direction': float(avg_same_dir),
        'per_trade': {
            'total_pnl': float(pnls.sum()),
            'avg_pnl': float(pnls.mean()),
            'avg_pts': float((pnls.mean() + 1.24) / 2.0),
            'win_rate': float((pnls > 0).mean()),
            'sharpe': float(pnls.mean() / max(pnls.std(), 1e-10) * np.sqrt(252)),
        },
        'per_event': {
            'total_pnl': float(event_total),
            'avg_pnl': float(event_avg),
            'avg_pts': float(event_avg_pts),
            'win_rate': float(event_wr),
            'sharpe': float(event_sharpe),
            'max_dd': float(event_dd),
        },
        'first_trade_only': {
            'total_pnl': float(ft_total),
            'avg_pnl': float(ft_avg),
            'avg_pts': float(ft_avg_pts),
            'win_rate': float(ft_wr),
            'sharpe': float(ft_sharpe),
        },
    }


# ============================================================
# VISUALIZACIÓN
# ============================================================
print(f"\n\n[7] Generando gráficos...")

fig, axes = plt.subplots(2, 3, figsize=(20, 12))

for idx, strat_name in enumerate(['W3_momentum', 'W3_fade_C1']):
    res = rolling_results.get(strat_name)
    if not res or not res['trades']:
        continue

    trades = res['trades']
    entry_bars = np.array([t['entry_bar'] for t in trades])
    pnls = np.array([t['pnl_dollars'] for t in trades])
    ir = independence_results[strat_name]

    sorted_idx = np.argsort(entry_bars)
    sorted_pnls = pnls[sorted_idx]

    # 1. Distribution of event sizes
    ax = axes[idx, 0]
    events_data = []
    sorted_entries = entry_bars[sorted_idx]
    current_event = [0]
    for i in range(1, len(sorted_entries)):
        if sorted_entries[i] - sorted_entries[current_event[-1]] <= 30:
            current_event.append(i)
        else:
            events_data.append(len(current_event))
            current_event = [i]
    events_data.append(len(current_event))

    ax.hist(events_data, bins=50, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.3)
    ax.set_title(f'{strat_name}\nDistribución de tamaño de eventos', fontsize=10)
    ax.set_xlabel('Trades por evento')
    ax.set_ylabel('Frecuencia')
    ax.axvline(np.median(events_data), color='red', linewidth=2, label=f'Mediana: {np.median(events_data):.0f}')
    ax.legend()

    # 2. PnL autocorrelation
    ax = axes[idx, 1]
    max_lag = min(50, len(sorted_pnls) // 5)
    autocorrs = []
    for lag in range(1, max_lag + 1):
        c = np.corrcoef(sorted_pnls[:-lag], sorted_pnls[lag:])[0, 1]
        autocorrs.append(c if not np.isnan(c) else 0)
    ax.bar(range(1, max_lag + 1), autocorrs, color='steelblue', alpha=0.7)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axhline(1.96 / np.sqrt(len(sorted_pnls)), color='red', linestyle='--', alpha=0.5, label='95% CI')
    ax.axhline(-1.96 / np.sqrt(len(sorted_pnls)), color='red', linestyle='--', alpha=0.5)
    ax.set_title(f'{strat_name}\nAutocorrelación de PnL', fontsize=10)
    ax.set_xlabel('Lag (trades)')
    ax.set_ylabel('Autocorrelación')
    ax.legend()

    # 3. Equity curves comparison: all trades vs event-level vs first-trade
    ax = axes[idx, 2]

    # All trades
    cum_all = np.cumsum(sorted_pnls)
    ax.plot(cum_all, linewidth=0.5, alpha=0.5, color='blue', label=f'Todos ({len(sorted_pnls)} trades)')

    # Event-level
    sorted_entries2 = entry_bars[sorted_idx]
    events2 = []
    current = [0]
    for i in range(1, len(sorted_entries2)):
        if sorted_entries2[i] - sorted_entries2[current[-1]] <= 30:
            current.append(i)
        else:
            events2.append(current)
            current = [i]
    events2.append(current)

    event_pnls2 = [sorted_pnls[e].sum() for e in events2]
    cum_event = np.cumsum(event_pnls2)
    ax.plot(cum_event, linewidth=1.5, color='navy', label=f'Por evento ({len(events2)} eventos)')

    first_pnls2 = [sorted_pnls[e[0]] for e in events2]
    cum_first = np.cumsum(first_pnls2)
    ax.plot(cum_first, linewidth=1.5, color='darkred', linestyle='--', label=f'1st trade/evento ({len(events2)})')

    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_title(f'{strat_name}\nEquity: todos vs eventos vs 1st trade', fontsize=10)
    ax.set_xlabel('Índice')
    ax.set_ylabel('PnL acumulado ($)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fase6c_independence.png'), dpi=150, bbox_inches='tight')
print(f"  Guardado: fase6c_independence.png")

with open(os.path.join(OUTPUT_DIR, 'fase6c_independence.json'), 'w') as f:
    json.dump(independence_results, f, indent=2, default=str)
print(f"  Guardado: fase6c_independence.json")


# ============================================================
# RESUMEN FINAL
# ============================================================
print("\n\n" + "=" * 80)
print("RESUMEN FASE 6C — INDEPENDENCIA DE TRADES")
print("=" * 80)

for strat_name, ir in independence_results.items():
    print(f"\n  {strat_name}:")
    print(f"    {ir['n_trades']} trades → {ir['n_events']} eventos únicos ({ir['trades_per_event_avg']:.1f} trades/evento)")
    print(f"    Overlap: {ir['overlap_pct']:.0f}% | Same direction: {ir['avg_same_direction']:.0%}")
    print(f"    Autocorrelación lag-1: {ir['autocorr_lag1']:.3f}")
    print(f"")
    print(f"    Vista conservadora (1 trade/evento):")
    print(f"      {ir['n_events']} trades | ${ir['first_trade_only']['total_pnl']:+,.0f} | "
          f"{ir['first_trade_only']['avg_pts']:+.1f} pts/trade | "
          f"WR={ir['first_trade_only']['win_rate']:.1%} | Sharpe={ir['first_trade_only']['sharpe']:.2f}")

print("\n[FASE 6C COMPLETADA]")
