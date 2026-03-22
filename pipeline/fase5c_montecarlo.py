"""
FASE 5C — Monte Carlo Simulation + Bootstrap Confidence Intervals
Permuta trades para estimar distribuciones de métricas bajo aleatoriedad.
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
print("FASE 5C — MONTE CARLO & BOOTSTRAP")
print("=" * 80)

# Load Phase 4 results
with open(os.path.join(OUTPUT_DIR, 'fase4_all_results.pkl'), 'rb') as f:
    phase4 = pickle.load(f)

N_SIM = 10000
np.random.seed(42)

TOP_NAMES = ['W5_C2long_H5', 'W3_momentum_H5_SL10', 'W3_momentum_H5_SL20',
             'W3_fade_C1_H10', 'W8_fadeC2_H1']

mc_results = {}

for strat_name in TOP_NAMES:
    if strat_name not in phase4:
        continue

    trades = phase4[strat_name]['trades_full']
    if len(trades) < 20:
        continue

    pnls = np.array([t['pnl_dollars'] for t in trades])
    n_trades = len(pnls)

    print(f"\n  {strat_name} ({n_trades} trades)...")

    # --- Monte Carlo: shuffle trade order, compute equity paths ---
    mc_final_pnl = np.zeros(N_SIM)
    mc_max_dd = np.zeros(N_SIM)
    mc_sharpe = np.zeros(N_SIM)

    for sim in range(N_SIM):
        shuffled = np.random.permutation(pnls)
        cum = np.cumsum(shuffled)
        mc_final_pnl[sim] = cum[-1]
        running_max = np.maximum.accumulate(cum)
        dd = cum - running_max
        mc_max_dd[sim] = dd.min()
        mc_sharpe[sim] = shuffled.mean() / max(shuffled.std(), 1e-10) * np.sqrt(252)

    # --- Bootstrap: resample with replacement ---
    boot_avg_pnl = np.zeros(N_SIM)
    boot_wr = np.zeros(N_SIM)
    boot_pf = np.zeros(N_SIM)

    for sim in range(N_SIM):
        sample = np.random.choice(pnls, size=n_trades, replace=True)
        boot_avg_pnl[sim] = sample.mean()
        boot_wr[sim] = (sample > 0).mean()
        gross_win = sample[sample > 0].sum() if (sample > 0).any() else 0
        gross_loss = abs(sample[sample < 0].sum()) if (sample < 0).any() else 1e-10
        boot_pf[sim] = gross_win / max(gross_loss, 1e-10)

    # Confidence intervals
    ci = lambda arr, lo=5, hi=95: (float(np.percentile(arr, lo)), float(np.percentile(arr, hi)))

    actual_pnl = float(pnls.sum())
    actual_sharpe = float(pnls.mean() / max(pnls.std(), 1e-10) * np.sqrt(252))
    actual_wr = float((pnls > 0).mean())
    actual_dd = float((np.cumsum(pnls) - np.maximum.accumulate(np.cumsum(pnls))).min())
    actual_pf = float(pnls[pnls > 0].sum() / max(abs(pnls[pnls < 0].sum()), 1e-10))

    res = {
        'n_trades': n_trades,
        'actual': {
            'total_pnl': actual_pnl,
            'sharpe': actual_sharpe,
            'win_rate': actual_wr,
            'max_dd': actual_dd,
            'profit_factor': actual_pf,
        },
        'mc_pnl_ci90': ci(mc_final_pnl),
        'mc_dd_ci90': ci(mc_max_dd),
        'mc_sharpe_ci90': ci(mc_sharpe),
        'boot_avg_pnl_ci90': ci(boot_avg_pnl),
        'boot_wr_ci90': ci(boot_wr),
        'boot_pf_ci90': ci(boot_pf),
        'mc_pnl_median': float(np.median(mc_final_pnl)),
        'mc_dd_median': float(np.median(mc_max_dd)),
        'prob_profitable': float((mc_final_pnl > 0).mean()),
        'prob_dd_under_2k': float((mc_max_dd > -2000).mean()),
    }

    mc_results[strat_name] = res

    # Store distributions for plotting
    mc_results[strat_name]['_mc_final_pnl'] = mc_final_pnl
    mc_results[strat_name]['_mc_max_dd'] = mc_max_dd
    mc_results[strat_name]['_boot_wr'] = boot_wr

    print(f"    PnL: ${actual_pnl:+,.2f} | MC 90% CI: ${res['mc_pnl_ci90'][0]:+,.0f} to ${res['mc_pnl_ci90'][1]:+,.0f}")
    print(f"    MaxDD: ${actual_dd:,.2f} | MC 90% CI: ${res['mc_dd_ci90'][0]:,.0f} to ${res['mc_dd_ci90'][1]:,.0f}")
    print(f"    Sharpe: {actual_sharpe:.2f} | MC 90% CI: {res['mc_sharpe_ci90'][0]:.2f} to {res['mc_sharpe_ci90'][1]:.2f}")
    print(f"    Win Rate: {actual_wr:.1%} | Boot 90% CI: {res['boot_wr_ci90'][0]:.1%} to {res['boot_wr_ci90'][1]:.1%}")
    print(f"    P(profitable): {res['prob_profitable']:.1%} | P(DD < $2K): {res['prob_dd_under_2k']:.1%}")


# ============================================================
# VISUALIZACIÓN
# ============================================================
print("\n[3] Generando gráficos Monte Carlo...")

n_strats = len(mc_results)
fig, axes = plt.subplots(n_strats, 3, figsize=(18, 4 * n_strats))
if n_strats == 1:
    axes = axes.reshape(1, -1)

for idx, (name, res) in enumerate(mc_results.items()):
    # PnL distribution
    ax = axes[idx, 0]
    mc_pnl = res['_mc_final_pnl']
    ax.hist(mc_pnl, bins=80, color='steelblue', alpha=0.7, edgecolor='none')
    ax.axvline(res['actual']['total_pnl'], color='red', linewidth=2, label=f'Actual ${res["actual"]["total_pnl"]:+,.0f}')
    ax.axvline(res['mc_pnl_ci90'][0], color='orange', linewidth=1.5, linestyle='--', label=f'5th: ${res["mc_pnl_ci90"][0]:+,.0f}')
    ax.axvline(res['mc_pnl_ci90'][1], color='orange', linewidth=1.5, linestyle='--', label=f'95th: ${res["mc_pnl_ci90"][1]:+,.0f}')
    ax.set_title(f'{name}\nMC Final PnL (P(>0)={res["prob_profitable"]:.1%})', fontsize=9)
    ax.set_xlabel('PnL ($)')
    ax.legend(fontsize=7)

    # Max DD distribution
    ax = axes[idx, 1]
    mc_dd = res['_mc_max_dd']
    ax.hist(mc_dd, bins=80, color='salmon', alpha=0.7, edgecolor='none')
    ax.axvline(res['actual']['max_dd'], color='red', linewidth=2, label=f'Actual ${res["actual"]["max_dd"]:,.0f}')
    ax.axvline(res['mc_dd_ci90'][0], color='darkred', linewidth=1.5, linestyle='--')
    ax.axvline(res['mc_dd_ci90'][1], color='darkred', linewidth=1.5, linestyle='--')
    ax.set_title(f'MC Max Drawdown\nP(DD>-$2K)={res["prob_dd_under_2k"]:.1%}', fontsize=9)
    ax.set_xlabel('Max DD ($)')
    ax.legend(fontsize=7)

    # Win Rate bootstrap
    ax = axes[idx, 2]
    boot_wr = res['_boot_wr']
    ax.hist(boot_wr, bins=60, color='mediumseagreen', alpha=0.7, edgecolor='none')
    ax.axvline(res['actual']['win_rate'], color='red', linewidth=2, label=f'Actual {res["actual"]["win_rate"]:.1%}')
    ax.axvline(res['boot_wr_ci90'][0], color='darkgreen', linewidth=1.5, linestyle='--')
    ax.axvline(res['boot_wr_ci90'][1], color='darkgreen', linewidth=1.5, linestyle='--')
    ax.set_title(f'Bootstrap Win Rate\n90% CI: {res["boot_wr_ci90"][0]:.1%}-{res["boot_wr_ci90"][1]:.1%}', fontsize=9)
    ax.set_xlabel('Win Rate')
    ax.legend(fontsize=7)

plt.suptitle('Monte Carlo & Bootstrap Analysis (10,000 simulations)', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fase5c_montecarlo.png'), dpi=150, bbox_inches='tight')
print(f"  Guardado: fase5c_montecarlo.png")

# Save JSON (without numpy arrays)
save_res = {}
for name, res in mc_results.items():
    save_res[name] = {k: v for k, v in res.items() if not k.startswith('_')}

with open(os.path.join(OUTPUT_DIR, 'fase5c_montecarlo.json'), 'w') as f:
    json.dump(save_res, f, indent=2, default=str)
print(f"  Guardado: fase5c_montecarlo.json")

print("\n[FASE 5C COMPLETADA]")
