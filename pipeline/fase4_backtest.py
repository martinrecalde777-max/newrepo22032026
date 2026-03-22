"""
FASE 4 — Backtesting de Estrategias Basadas en Morfología
Pipeline de descubrimiento de morfologías predictivas MNQ futures

Simula estrategias de trading usando las señales de clusters descubiertas
en Fase 3, con ejecución realista y métricas de rendimiento.
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
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/user/newrepo22032026/outputs'

print("=" * 80)
print("FASE 4 — BACKTESTING DE ESTRATEGIAS MORFOLÓGICAS")
print("=" * 80)

# ============================================================
# 1. CARGAR DATOS
# ============================================================
print("\n[1] Cargando datos...")

df = pd.read_parquet(os.path.join(OUTPUT_DIR, 'mnq_clean.parquet'))
df['ts_et'] = pd.to_datetime(df['ts_et'])
n_bars = len(df)
print(f"  Barras totales: {n_bars:,}")

opens = df['open'].values.astype(np.float64)
highs = df['high'].values.astype(np.float64)
lows = df['low'].values.astype(np.float64)
closes = df['close'].values.astype(np.float64)
volumes = df['volume'].values.astype(np.float64)
timestamps = df['ts_et'].values
dates = pd.to_datetime(timestamps)

# Gap mask for continuity checks
ts_diff = np.diff(timestamps.astype('int64')) / 1e9
gap_mask = np.zeros(n_bars, dtype=bool)
gap_mask[1:] = ts_diff > 300
gap_cumsum = np.cumsum(gap_mask)

# Load Phase 2 cluster data for W=3 (best predictive signals)
BEST_Ws = [3, 5, 8]
print(f"  Ventanas a backtestear: {BEST_Ws}")

# MNQ contract specs
TICK_SIZE = 0.25
TICK_VALUE = 0.50  # $0.50 per tick for MNQ
POINT_VALUE = 2.0  # $2.00 per point for MNQ
COMMISSION_RT = 1.24  # round-trip commission per contract
SLIPPAGE_TICKS = 1  # 1 tick slippage per side


# ============================================================
# 2. FUNCIONES DE BACKTESTING
# ============================================================

def classify_bar_window(W, end_idx, model, scaler, feature_extractor):
    """Classify a window ending at end_idx using the Phase 2 model."""
    pass  # We'll use pre-computed labels instead


def run_backtest(W, labels, end_idx, strategy_config):
    """
    Run backtest for a given window size and strategy.

    strategy_config: dict with keys:
      - signal_clusters: dict mapping cluster_id -> 'long' or 'short'
      - hold_bars: number of bars to hold
      - stop_loss_pts: stop loss in points (0 = no stop)
      - take_profit_pts: take profit in points (0 = no TP)
      - max_positions: max concurrent positions (1 = no stacking)
    """
    signal_clusters = strategy_config['signal_clusters']
    hold_bars = strategy_config['hold_bars']
    stop_loss_pts = strategy_config.get('stop_loss_pts', 0)
    take_profit_pts = strategy_config.get('take_profit_pts', 0)

    # Sort by time
    sort_order = np.argsort(end_idx)
    end_idx_sorted = end_idx[sort_order]
    labels_sorted = labels[sort_order]

    trades = []
    equity = [0.0]
    equity_timestamps = [timestamps[0]]

    for i in range(len(end_idx_sorted)):
        cluster = labels_sorted[i]
        if cluster not in signal_clusters:
            continue

        direction = signal_clusters[cluster]  # 'long' or 'short'
        entry_bar_idx = end_idx_sorted[i] + 1  # enter on next bar after pattern

        if entry_bar_idx >= n_bars:
            continue

        # Check continuity at entry
        if gap_mask[entry_bar_idx]:
            continue

        # Entry price: open of next bar + slippage
        entry_price = opens[entry_bar_idx]
        if direction == 'long':
            entry_price += SLIPPAGE_TICKS * TICK_SIZE
        else:
            entry_price -= SLIPPAGE_TICKS * TICK_SIZE

        # Simulate bar-by-bar
        exit_price = None
        exit_bar_idx = None
        exit_reason = None

        for j in range(1, hold_bars + 1):
            bar_idx = entry_bar_idx + j
            if bar_idx >= n_bars:
                # Force exit at last available bar
                bar_idx = n_bars - 1
                exit_price = closes[bar_idx]
                exit_reason = 'end_of_data'
                exit_bar_idx = bar_idx
                break

            # Check gap — force exit at previous close if gap
            if gap_cumsum[bar_idx] - gap_cumsum[entry_bar_idx] > 0:
                exit_price = closes[bar_idx - 1]
                exit_reason = 'gap'
                exit_bar_idx = bar_idx - 1
                break

            bar_high = highs[bar_idx]
            bar_low = lows[bar_idx]

            # Check stop loss
            if stop_loss_pts > 0:
                if direction == 'long' and bar_low <= entry_price - stop_loss_pts:
                    exit_price = entry_price - stop_loss_pts
                    exit_reason = 'stop_loss'
                    exit_bar_idx = bar_idx
                    break
                elif direction == 'short' and bar_high >= entry_price + stop_loss_pts:
                    exit_price = entry_price + stop_loss_pts
                    exit_reason = 'stop_loss'
                    exit_bar_idx = bar_idx
                    break

            # Check take profit
            if take_profit_pts > 0:
                if direction == 'long' and bar_high >= entry_price + take_profit_pts:
                    exit_price = entry_price + take_profit_pts
                    exit_reason = 'take_profit'
                    exit_bar_idx = bar_idx
                    break
                elif direction == 'short' and bar_low <= entry_price - take_profit_pts:
                    exit_price = entry_price - take_profit_pts
                    exit_reason = 'take_profit'
                    exit_bar_idx = bar_idx
                    break

            # Time exit
            if j == hold_bars:
                exit_price = closes[bar_idx]
                exit_reason = 'time_exit'
                exit_bar_idx = bar_idx
                break

        if exit_price is None:
            continue

        # Apply exit slippage
        if direction == 'long':
            exit_price -= SLIPPAGE_TICKS * TICK_SIZE
        else:
            exit_price += SLIPPAGE_TICKS * TICK_SIZE

        # P&L calculation
        if direction == 'long':
            pnl_points = exit_price - entry_price
        else:
            pnl_points = entry_price - exit_price

        pnl_dollars = pnl_points * POINT_VALUE - COMMISSION_RT

        # MFE/MAE during trade
        trade_highs = highs[entry_bar_idx:exit_bar_idx + 1]
        trade_lows = lows[entry_bar_idx:exit_bar_idx + 1]
        if direction == 'long':
            mfe = (trade_highs.max() - entry_price) * POINT_VALUE
            mae = (entry_price - trade_lows.min()) * POINT_VALUE
        else:
            mfe = (entry_price - trade_lows.min()) * POINT_VALUE
            mae = (trade_highs.max() - entry_price) * POINT_VALUE

        trades.append({
            'entry_bar': int(entry_bar_idx),
            'exit_bar': int(exit_bar_idx),
            'entry_time': str(timestamps[entry_bar_idx]),
            'exit_time': str(timestamps[exit_bar_idx]),
            'direction': direction,
            'cluster': int(cluster),
            'entry_price': float(entry_price),
            'exit_price': float(exit_price),
            'pnl_points': float(pnl_points),
            'pnl_dollars': float(pnl_dollars),
            'exit_reason': exit_reason,
            'hold_bars_actual': int(exit_bar_idx - entry_bar_idx),
            'mfe': float(mfe),
            'mae': float(mae),
        })

        # Track equity
        cum_pnl = sum(t['pnl_dollars'] for t in trades)
        equity.append(cum_pnl)
        equity_timestamps.append(timestamps[exit_bar_idx])

    return trades, equity, equity_timestamps


def compute_metrics(trades):
    """Compute strategy performance metrics from trade list."""
    if not trades:
        return {'n_trades': 0}

    pnls = np.array([t['pnl_dollars'] for t in trades])
    n_trades = len(trades)
    winners = pnls > 0
    losers = pnls < 0

    # Equity curve for drawdown
    cum_pnl = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cum_pnl)
    drawdowns = cum_pnl - running_max
    max_dd = drawdowns.min() if len(drawdowns) > 0 else 0

    # Time span
    first_entry = trades[0]['entry_time']
    last_exit = trades[-1]['exit_time']

    # Annualized (approximate: 252 trading days)
    pnl_per_trade = pnls.mean()

    gross_profit = pnls[winners].sum() if winners.any() else 0
    gross_loss = abs(pnls[losers].sum()) if losers.any() else 1e-10

    avg_win = pnls[winners].mean() if winners.any() else 0
    avg_loss = abs(pnls[losers].mean()) if losers.any() else 1e-10

    # Trade duration
    durations = [t['hold_bars_actual'] for t in trades]
    mfes = [t['mfe'] for t in trades]
    maes = [t['mae'] for t in trades]

    # Exit reason distribution
    exit_reasons = {}
    for t in trades:
        r = t['exit_reason']
        exit_reasons[r] = exit_reasons.get(r, 0) + 1

    return {
        'n_trades': n_trades,
        'total_pnl': float(cum_pnl[-1]),
        'avg_pnl_per_trade': float(pnl_per_trade),
        'win_rate': float(winners.mean()),
        'profit_factor': float(gross_profit / max(gross_loss, 1e-10)),
        'avg_win': float(avg_win),
        'avg_loss': float(avg_loss),
        'payoff_ratio': float(avg_win / max(avg_loss, 1e-10)),
        'max_drawdown': float(max_dd),
        'sharpe': float(pnls.mean() / max(pnls.std(), 1e-10) * np.sqrt(252)),
        'max_consecutive_wins': int(max_consecutive(winners)),
        'max_consecutive_losses': int(max_consecutive(losers)),
        'avg_hold_bars': float(np.mean(durations)),
        'avg_mfe': float(np.mean(mfes)),
        'avg_mae': float(np.mean(maes)),
        'exit_reasons': exit_reasons,
        'first_trade': first_entry,
        'last_trade': last_exit,
    }


def max_consecutive(bool_array):
    """Max consecutive True values."""
    if not bool_array.any():
        return 0
    max_run = 0
    current = 0
    for v in bool_array:
        if v:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run


# ============================================================
# 3. DEFINIR ESTRATEGIAS
# ============================================================
print("\n[2] Definiendo estrategias...")

strategies = {}

# --- Strategy 1: W=3 Momentum (C3=long, C4=short) ---
# C3: strong uptrend (slope_norm=+0.018, dir=0.98), 82-89% continuation
# C4: strong downtrend (slope_norm=-0.018, dir=-0.98), 76-80% continuation
for H in [3, 5, 10]:
    strategies[f'W3_momentum_H{H}'] = {
        'W': 3,
        'signal_clusters': {3: 'long', 4: 'short'},
        'hold_bars': H,
        'stop_loss_pts': 0,
        'take_profit_pts': 0,
        'description': f'W=3 momentum: C3→Long, C4→Short, hold {H} bars',
    }

# --- Strategy 2: W=3 Momentum with stops ---
for sl in [10, 20, 30]:
    strategies[f'W3_momentum_H5_SL{sl}'] = {
        'W': 3,
        'signal_clusters': {3: 'long', 4: 'short'},
        'hold_bars': 5,
        'stop_loss_pts': sl,
        'take_profit_pts': 0,
        'description': f'W=3 momentum H=5 with SL={sl}pts',
    }

# --- Strategy 3: W=3 Momentum with TP ---
for tp in [10, 20, 30]:
    strategies[f'W3_momentum_H10_TP{tp}'] = {
        'W': 3,
        'signal_clusters': {3: 'long', 4: 'short'},
        'hold_bars': 10,
        'stop_loss_pts': 20,
        'take_profit_pts': tp,
        'description': f'W=3 momentum H=10 SL=20 TP={tp}',
    }

# --- Strategy 4: W=3 Mean-reversion on C1 (strong up bars → fade) ---
# C1 in W=3: strong uptrend but post-pattern shows 34-37% up → fade
for H in [3, 5, 10]:
    strategies[f'W3_fade_C1_H{H}'] = {
        'W': 3,
        'signal_clusters': {1: 'short'},
        'hold_bars': H,
        'stop_loss_pts': 30,
        'take_profit_pts': 0,
        'description': f'W=3 fade C1 (strong up→short) H={H} SL=30',
    }

# --- Strategy 5: W=5 cluster signals ---
# C2 at H=5,20: >53% up direction
for H in [5, 10, 20]:
    strategies[f'W5_C2long_H{H}'] = {
        'W': 5,
        'signal_clusters': {2: 'long'},
        'hold_bars': H,
        'stop_loss_pts': 30,
        'take_profit_pts': 0,
        'description': f'W=5 C2→Long H={H} SL=30',
    }

# --- Strategy 6: W=8 C2 fade (large range no direction → mean revert) ---
# C2 at H=1: 41.5% up, ret=-0.08% → short bias
for H in [1, 3, 5]:
    strategies[f'W8_fadeC2_H{H}'] = {
        'W': 8,
        'signal_clusters': {2: 'short'},
        'hold_bars': H,
        'stop_loss_pts': 40,
        'take_profit_pts': 0,
        'description': f'W=8 fade C2 (large range→short) H={H} SL=40',
    }

print(f"  Estrategias definidas: {len(strategies)}")


# ============================================================
# 4. EJECUTAR BACKTESTS
# ============================================================
print("\n[3] Ejecutando backtests...")

backtest_results = {}

# Load cluster data per W
loaded_data = {}
for W in BEST_Ws:
    data = np.load(os.path.join(OUTPUT_DIR, f'fase2_W{W}_data.npz'))
    loaded_data[W] = {
        'labels': data['labels'],
        'end_idx': data['end_idx'],
    }
    print(f"  W={W}: {len(data['labels']):,} ventanas cargadas")

for strat_name, config in strategies.items():
    t0 = time.time()
    W = config['W']
    data = loaded_data[W]

    # Temporal split: use last 20% for out-of-sample
    n_total = len(data['end_idx'])
    oos_start = int(n_total * 0.8)

    # Full period backtest
    trades_full, equity_full, eq_ts_full = run_backtest(
        W, data['labels'], data['end_idx'], config
    )

    # Out-of-sample only
    oos_mask = np.arange(n_total) >= oos_start
    trades_oos, equity_oos, eq_ts_oos = run_backtest(
        W, data['labels'][oos_mask], data['end_idx'][oos_mask], config
    )

    metrics_full = compute_metrics(trades_full)
    metrics_oos = compute_metrics(trades_oos)

    elapsed = time.time() - t0

    backtest_results[strat_name] = {
        'config': config,
        'full': metrics_full,
        'oos': metrics_oos,
        'equity_full': equity_full,
        'equity_ts_full': eq_ts_full,
        'trades_full': trades_full,
        'elapsed': elapsed,
    }

    n_full = metrics_full['n_trades']
    n_oos = metrics_oos['n_trades']
    pnl_f = metrics_full.get('total_pnl', 0)
    pnl_o = metrics_oos.get('total_pnl', 0)
    wr_f = metrics_full.get('win_rate', 0)
    wr_o = metrics_oos.get('win_rate', 0)
    pf_f = metrics_full.get('profit_factor', 0)

    print(f"  {strat_name:35s} | Full: {n_full:>5} trades ${pnl_f:>+10.2f} WR={wr_f:.1%} PF={pf_f:.2f} | "
          f"OOS: {n_oos:>4} trades ${pnl_o:>+9.2f} WR={wr_o:.1%}")


# ============================================================
# 5. RANKING DE ESTRATEGIAS
# ============================================================
print("\n" + "=" * 80)
print("RANKING DE ESTRATEGIAS")
print("=" * 80)

# Rank by OOS Sharpe (robustness)
ranked = []
for name, res in backtest_results.items():
    mf = res['full']
    mo = res['oos']
    if mf['n_trades'] < 20:
        continue
    ranked.append({
        'name': name,
        'n_trades_full': mf['n_trades'],
        'n_trades_oos': mo['n_trades'],
        'pnl_full': mf.get('total_pnl', 0),
        'pnl_oos': mo.get('total_pnl', 0),
        'wr_full': mf.get('win_rate', 0),
        'wr_oos': mo.get('win_rate', 0),
        'pf_full': mf.get('profit_factor', 0),
        'pf_oos': mo.get('profit_factor', 0),
        'sharpe_full': mf.get('sharpe', 0),
        'sharpe_oos': mo.get('sharpe', 0),
        'max_dd_full': mf.get('max_drawdown', 0),
        'max_dd_oos': mo.get('max_drawdown', 0),
        'avg_pnl': mf.get('avg_pnl_per_trade', 0),
        'payoff': mf.get('payoff_ratio', 0),
    })

ranked.sort(key=lambda x: x['pnl_oos'], reverse=True)

print(f"\n{'Estrategia':>38s} | {'#Full':>5s} | {'PnL Full':>10s} | {'WR%':>5s} | "
      f"{'PF':>5s} | {'#OOS':>5s} | {'PnL OOS':>10s} | {'WR%':>5s} | {'MaxDD':>9s}")
print("-" * 115)
for r in ranked:
    print(f"{r['name']:>38s} | {r['n_trades_full']:>5d} | ${r['pnl_full']:>+9.2f} | "
          f"{r['wr_full']:.1%} | {r['pf_full']:>5.2f} | {r['n_trades_oos']:>5d} | "
          f"${r['pnl_oos']:>+9.2f} | {r['wr_oos']:.1%} | ${r['max_dd_full']:>8.2f}")


# ============================================================
# 6. ANÁLISIS DETALLADO DE MEJORES ESTRATEGIAS
# ============================================================
print("\n" + "=" * 80)
print("ANÁLISIS DETALLADO — TOP ESTRATEGIAS")
print("=" * 80)

top_strategies = ranked[:5] if len(ranked) >= 5 else ranked

for r in top_strategies:
    name = r['name']
    res = backtest_results[name]
    mf = res['full']
    mo = res['oos']

    print(f"\n{'─'*60}")
    print(f"  {name}")
    print(f"  {res['config']['description']}")
    print(f"{'─'*60}")
    print(f"  FULL PERIOD:")
    print(f"    Trades: {mf['n_trades']:,} | PnL: ${mf['total_pnl']:+,.2f} | Avg: ${mf['avg_pnl_per_trade']:+.2f}/trade")
    print(f"    Win Rate: {mf['win_rate']:.1%} | Profit Factor: {mf['profit_factor']:.2f} | Payoff: {mf['payoff_ratio']:.2f}")
    print(f"    Sharpe: {mf['sharpe']:.2f} | Max DD: ${mf['max_drawdown']:,.2f}")
    print(f"    Avg Hold: {mf['avg_hold_bars']:.1f} bars | MFE: ${mf['avg_mfe']:.2f} | MAE: ${mf['avg_mae']:.2f}")
    print(f"    Consec Wins: {mf['max_consecutive_wins']} | Consec Losses: {mf['max_consecutive_losses']}")
    print(f"    Exit reasons: {mf['exit_reasons']}")
    print(f"  OUT-OF-SAMPLE:")
    print(f"    Trades: {mo['n_trades']:,} | PnL: ${mo.get('total_pnl', 0):+,.2f} | Avg: ${mo.get('avg_pnl_per_trade', 0):+.2f}/trade")
    print(f"    Win Rate: {mo.get('win_rate', 0):.1%} | Profit Factor: {mo.get('profit_factor', 0):.2f}")
    print(f"    Max DD: ${mo.get('max_drawdown', 0):,.2f}")

    # Trade distribution by direction
    trades = res['trades_full']
    longs = [t for t in trades if t['direction'] == 'long']
    shorts = [t for t in trades if t['direction'] == 'short']
    if longs:
        l_pnl = sum(t['pnl_dollars'] for t in longs)
        l_wr = sum(1 for t in longs if t['pnl_dollars'] > 0) / len(longs)
        print(f"    Longs:  {len(longs):>5} trades ${l_pnl:>+10.2f} WR={l_wr:.1%}")
    if shorts:
        s_pnl = sum(t['pnl_dollars'] for t in shorts)
        s_wr = sum(1 for t in shorts if t['pnl_dollars'] > 0) / len(shorts)
        print(f"    Shorts: {len(shorts):>5} trades ${s_pnl:>+10.2f} WR={s_wr:.1%}")


# ============================================================
# 7. VISUALIZACIONES
# ============================================================
print("\n[7] Generando visualizaciones...")

fig = plt.figure(figsize=(20, 16))
gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

# 7a. Equity curves — top strategies
ax1 = fig.add_subplot(gs[0, :])
colors = plt.cm.Set1(np.linspace(0, 1, min(len(top_strategies), 9)))
for i, r in enumerate(top_strategies):
    name = r['name']
    eq = backtest_results[name]['equity_full']
    ax1.plot(range(len(eq)), eq, label=name, color=colors[i], linewidth=1.2)
ax1.axhline(0, color='black', linewidth=0.5, linestyle='--')
ax1.set_title('Equity Curves — Top Estrategias (Full Period)')
ax1.set_xlabel('Trade #')
ax1.set_ylabel('PnL Acumulado ($)')
ax1.legend(fontsize=7, ncol=2, loc='upper left')
ax1.grid(True, alpha=0.3)

# 7b. Win rate comparison (Full vs OOS)
ax2 = fig.add_subplot(gs[1, 0])
if ranked:
    names_short = [r['name'].replace('W3_momentum_', 'M_').replace('W3_fade_', 'F_')
                   .replace('W5_C2long_', '5C2_').replace('W8_fadeC2_', '8F_')
                   for r in ranked]
    x = np.arange(len(ranked))
    width = 0.35
    wr_full = [r['wr_full'] for r in ranked]
    wr_oos = [r['wr_oos'] for r in ranked]
    ax2.barh(x - width/2, wr_full, width, label='Full', color='steelblue')
    ax2.barh(x + width/2, wr_oos, width, label='OOS', color='coral')
    ax2.axvline(0.5, color='red', linestyle='--', alpha=0.5)
    ax2.set_yticks(x)
    ax2.set_yticklabels(names_short, fontsize=7)
    ax2.set_xlabel('Win Rate')
    ax2.set_title('Win Rate: Full vs Out-of-Sample')
    ax2.legend(fontsize=8)
    ax2.invert_yaxis()

# 7c. PnL comparison
ax3 = fig.add_subplot(gs[1, 1])
if ranked:
    pnl_full = [r['pnl_full'] for r in ranked]
    pnl_oos = [r['pnl_oos'] for r in ranked]
    ax3.barh(x - width/2, pnl_full, width, label='Full', color='steelblue')
    ax3.barh(x + width/2, pnl_oos, width, label='OOS', color='coral')
    ax3.axvline(0, color='black', linestyle='-', alpha=0.3)
    ax3.set_yticks(x)
    ax3.set_yticklabels(names_short, fontsize=7)
    ax3.set_xlabel('PnL ($)')
    ax3.set_title('PnL Total: Full vs Out-of-Sample')
    ax3.legend(fontsize=8)
    ax3.invert_yaxis()

# 7d. Trade P&L distribution for best strategy
ax4 = fig.add_subplot(gs[2, 0])
if top_strategies:
    best_name = top_strategies[0]['name']
    best_trades = backtest_results[best_name]['trades_full']
    if best_trades:
        pnls = [t['pnl_dollars'] for t in best_trades]
        ax4.hist(pnls, bins=80, color='teal', alpha=0.7, edgecolor='black', linewidth=0.3)
        ax4.axvline(0, color='red', linewidth=1.5)
        ax4.axvline(np.mean(pnls), color='gold', linewidth=1.5, linestyle='--',
                    label=f'Mean=${np.mean(pnls):.2f}')
        ax4.set_title(f'Distribución P&L: {best_name}')
        ax4.set_xlabel('P&L por Trade ($)')
        ax4.set_ylabel('Frecuencia')
        ax4.legend()

# 7e. MFE/MAE scatter for best strategy
ax5 = fig.add_subplot(gs[2, 1])
if top_strategies:
    best_trades = backtest_results[top_strategies[0]['name']]['trades_full']
    if best_trades:
        mfes = [t['mfe'] for t in best_trades]
        maes = [t['mae'] for t in best_trades]
        colors_scatter = ['green' if t['pnl_dollars'] > 0 else 'red' for t in best_trades]
        ax5.scatter(maes, mfes, c=colors_scatter, alpha=0.3, s=8)
        max_val = max(max(mfes), max(maes)) if mfes and maes else 100
        ax5.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='MFE=MAE')
        ax5.set_xlabel('MAE ($) — Max Adverse Excursion')
        ax5.set_ylabel('MFE ($) — Max Favorable Excursion')
        ax5.set_title(f'MFE vs MAE: {top_strategies[0]["name"]}')
        ax5.legend(fontsize=8)

plt.savefig(os.path.join(OUTPUT_DIR, 'fase4_backtest.png'), dpi=150, bbox_inches='tight')
print(f"  Guardado: {OUTPUT_DIR}/fase4_backtest.png")


# ============================================================
# 8. GUARDAR RESULTADOS
# ============================================================

# Save summary (without full trade lists and equity arrays)
summary = {
    'strategies_tested': len(strategies),
    'strategies_with_trades': sum(1 for r in backtest_results.values() if r['full']['n_trades'] > 0),
    'contract_specs': {
        'tick_size': TICK_SIZE,
        'tick_value': TICK_VALUE,
        'point_value': POINT_VALUE,
        'commission_rt': COMMISSION_RT,
        'slippage_ticks': SLIPPAGE_TICKS,
    },
    'ranking': ranked,
    'detailed_results': {},
}

for name, res in backtest_results.items():
    summary['detailed_results'][name] = {
        'config': {k: v for k, v in res['config'].items() if k != 'signal_clusters'},
        'signal_clusters': {str(k): v for k, v in res['config']['signal_clusters'].items()},
        'full': res['full'],
        'oos': res['oos'],
    }

with open(os.path.join(OUTPUT_DIR, 'fase4_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2, default=str)

with open(os.path.join(OUTPUT_DIR, 'fase4_all_results.pkl'), 'wb') as f:
    pickle.dump(backtest_results, f)

print(f"  Guardados: fase4_summary.json, fase4_all_results.pkl")

print("\n" + "=" * 80)
print("FASE 4 COMPLETADA — Esperando confirmación para Fase 5")
print("=" * 80)
