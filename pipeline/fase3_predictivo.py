"""
FASE 3 — Análisis Predictivo
Pipeline de descubrimiento de morfologías predictivas MNQ futures

Evalúa si los clusters morfológicos descubiertos en Fase 2 tienen poder
predictivo sobre el comportamiento futuro del precio.
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
import time
import warnings

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    mean_absolute_error, r2_score, roc_auc_score
)
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/user/newrepo22032026/outputs'

print("=" * 80)
print("FASE 3 — ANÁLISIS PREDICTIVO")
print("=" * 80)

# ============================================================
# 1. CARGAR DATOS
# ============================================================
print("\n[1] Cargando datos de Fases 1 y 2...")

df = pd.read_parquet(os.path.join(OUTPUT_DIR, 'mnq_clean.parquet'))
df['ts_et'] = pd.to_datetime(df['ts_et'])
print(f"  Barras totales: {len(df):,}")

opens = df['open'].values.astype(np.float64)
highs = df['high'].values.astype(np.float64)
lows = df['low'].values.astype(np.float64)
closes = df['close'].values.astype(np.float64)
volumes = df['volume'].values.astype(np.float64)
timestamps = df['ts_et'].values

with open(os.path.join(OUTPUT_DIR, 'fase2_summary.json')) as f:
    fase2_summary = json.load(f)

optimal_Ws = fase2_summary['optimal_Ws']
print(f"  Ventanas óptimas de Fase 2: {optimal_Ws}")

# ============================================================
# 2. DEFINIR TARGETS FORWARD-LOOKING
# ============================================================
FORWARD_HORIZONS = [1, 3, 5, 10, 20]

print(f"\n[2] Horizontes forward: {FORWARD_HORIZONS} barras")


def compute_forward_targets(end_indices, horizon):
    """
    Para cada ventana que termina en end_indices[i], computa targets
    basados en las siguientes 'horizon' barras.
    """
    n_bars = len(closes)
    n_windows = len(end_indices)

    # Targets
    fwd_return = np.full(n_windows, np.nan)
    fwd_direction = np.full(n_windows, np.nan)
    fwd_volatility = np.full(n_windows, np.nan)
    fwd_max_runup = np.full(n_windows, np.nan)
    fwd_max_drawdown = np.full(n_windows, np.nan)
    fwd_range = np.full(n_windows, np.nan)

    # Check timestamp continuity
    ts_diff = np.diff(timestamps.astype('int64')) / 1e9
    gap_mask = np.zeros(n_bars, dtype=bool)
    gap_mask[1:] = ts_diff > 300
    gap_cumsum = np.cumsum(gap_mask)

    for i, end_idx in enumerate(end_indices):
        start_fwd = end_idx + 1
        end_fwd = start_fwd + horizon

        if end_fwd > n_bars:
            continue

        # Check no gaps in forward window
        if gap_cumsum[end_fwd - 1] - gap_cumsum[start_fwd] > 0:
            continue

        c_start = closes[end_idx]
        if c_start < 1e-10:
            continue

        fwd_closes = closes[start_fwd:end_fwd]
        fwd_highs = highs[start_fwd:end_fwd]
        fwd_lows = lows[start_fwd:end_fwd]

        # Return
        fwd_return[i] = (fwd_closes[-1] - c_start) / c_start

        # Direction (binary: 1=up, 0=down)
        fwd_direction[i] = 1.0 if fwd_closes[-1] > c_start else 0.0

        # Volatility (std of log returns)
        log_c = np.log(np.maximum(fwd_closes, 1e-10))
        log_c0 = np.log(max(c_start, 1e-10))
        all_log = np.concatenate([[log_c0], log_c])
        rets = np.diff(all_log)
        fwd_volatility[i] = np.std(rets) if len(rets) > 1 else 0.0

        # Max runup / drawdown from entry
        price_path = np.concatenate([[c_start], fwd_closes])
        cumulative_return = (price_path - c_start) / c_start
        fwd_max_runup[i] = cumulative_return.max()
        fwd_max_drawdown[i] = cumulative_return.min()

        # Forward range
        fwd_range[i] = (fwd_highs.max() - fwd_lows.min()) / c_start

    return {
        'fwd_return': fwd_return,
        'fwd_direction': fwd_direction,
        'fwd_volatility': fwd_volatility,
        'fwd_max_runup': fwd_max_runup,
        'fwd_max_drawdown': fwd_max_drawdown,
        'fwd_range': fwd_range,
    }


# ============================================================
# 3. ANÁLISIS POR VENTANA ÓPTIMA
# ============================================================

all_predictive_results = {}

for W in optimal_Ws:
    print(f"\n{'='*60}")
    print(f"  VENTANA W = {W}")
    print(f"{'='*60}")
    t0 = time.time()

    # Load Phase 2 data
    data = np.load(os.path.join(OUTPUT_DIR, f'fase2_W{W}_data.npz'))
    X = data['X']
    X_scaled = data['X_scaled']
    end_idx = data['end_idx']
    labels = data['labels']

    print(f"  Ventanas: {len(X):,} | Clusters: {len(set(labels[labels >= 0]))} "
          f"(+ noise: {(labels == -1).sum():,})")

    # Remove noise points
    valid_mask = labels >= 0
    X_valid = X[valid_mask]
    X_scaled_valid = X_scaled[valid_mask]
    end_idx_valid = end_idx[valid_mask]
    labels_valid = labels[valid_mask]

    print(f"  Ventanas sin ruido: {len(X_valid):,}")

    w_results = {'W': W, 'horizons': {}}

    for H in FORWARD_HORIZONS:
        print(f"\n  --- Horizonte H={H} ---")

        targets = compute_forward_targets(end_idx_valid, H)

        # Filter to valid targets
        has_target = np.isfinite(targets['fwd_return'])
        n_valid = has_target.sum()
        print(f"    Ventanas con target válido: {n_valid:,} / {len(X_valid):,}")

        if n_valid < 500:
            print(f"    Insuficientes — skip")
            continue

        X_h = X_valid[has_target]
        X_h_scaled = X_scaled_valid[has_target]
        labels_h = labels_valid[has_target]
        end_idx_h = end_idx_valid[has_target]
        targets_h = {k: v[has_target] for k, v in targets.items()}

        # --- 3a. Cluster-level predictive analysis ---
        cluster_ids = sorted(set(labels_h))
        cluster_predictive = {}

        for c in cluster_ids:
            mask_c = labels_h == c
            n_c = mask_c.sum()
            if n_c < 30:
                continue

            ret_c = targets_h['fwd_return'][mask_c]
            dir_c = targets_h['fwd_direction'][mask_c]
            vol_c = targets_h['fwd_volatility'][mask_c]
            runup_c = targets_h['fwd_max_runup'][mask_c]
            dd_c = targets_h['fwd_max_drawdown'][mask_c]

            cluster_predictive[int(c)] = {
                'n': int(n_c),
                'return_mean': float(np.mean(ret_c)),
                'return_std': float(np.std(ret_c)),
                'return_median': float(np.median(ret_c)),
                'direction_pct_up': float(np.mean(dir_c)),
                'volatility_mean': float(np.mean(vol_c)),
                'max_runup_mean': float(np.mean(runup_c)),
                'max_drawdown_mean': float(np.mean(dd_c)),
                'sharpe': float(np.mean(ret_c) / max(np.std(ret_c), 1e-10)),
                'profit_factor': float(
                    np.abs(ret_c[ret_c > 0].sum()) / max(np.abs(ret_c[ret_c < 0].sum()), 1e-10)
                ) if (ret_c > 0).any() and (ret_c < 0).any() else 0.0,
            }

        print(f"    Clusters analizados: {len(cluster_predictive)}")
        for c, cp in sorted(cluster_predictive.items()):
            arrow = "↑" if cp['direction_pct_up'] > 0.52 else ("↓" if cp['direction_pct_up'] < 0.48 else "→")
            print(f"      C{c} (n={cp['n']:,}): ret={cp['return_mean']*100:+.4f}% "
                  f"dir={cp['direction_pct_up']:.1%}{arrow} "
                  f"sharpe={cp['sharpe']:.3f} pf={cp['profit_factor']:.2f}")

        # --- 3b. Feature + Cluster → Direction classification ---
        # Build feature matrix: morphological features + one-hot cluster
        n_clusters_total = max(cluster_ids) + 1
        cluster_onehot = np.zeros((len(X_h), n_clusters_total))
        for i, c in enumerate(labels_h):
            if c >= 0:
                cluster_onehot[i, c] = 1.0

        X_model = np.hstack([X_h_scaled, cluster_onehot])
        y_dir = targets_h['fwd_direction'].astype(int)
        y_ret = targets_h['fwd_return']

        # Temporal split: sort by end_idx
        sort_order = np.argsort(end_idx_h)
        X_model = X_model[sort_order]
        y_dir = y_dir[sort_order]
        y_ret = y_ret[sort_order]
        end_idx_sorted = end_idx_h[sort_order]

        # 80/20 temporal split
        split_point = int(len(X_model) * 0.8)
        X_train, X_test = X_model[:split_point], X_model[split_point:]
        y_dir_train, y_dir_test = y_dir[:split_point], y_dir[split_point:]
        y_ret_train, y_ret_test = y_ret[:split_point], y_ret[split_point:]

        print(f"    Train: {len(X_train):,} | Test: {len(X_test):,}")

        # Direction classifier
        clf = lgb.LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            num_leaves=31, min_child_samples=50,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1, n_jobs=-1,
        )
        clf.fit(X_train, y_dir_train,
                eval_set=[(X_test, y_dir_test)],
                callbacks=[lgb.early_stopping(30, verbose=False)])

        y_dir_pred = clf.predict(X_test)
        y_dir_proba = clf.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_dir_test, y_dir_pred)
        f1 = f1_score(y_dir_test, y_dir_pred, average='weighted')
        try:
            auc = roc_auc_score(y_dir_test, y_dir_proba)
        except ValueError:
            auc = 0.5

        # Baseline: always predict majority class
        baseline_acc = max(y_dir_test.mean(), 1 - y_dir_test.mean())

        print(f"    [Dirección] Acc={acc:.4f} (base={baseline_acc:.4f}) "
              f"F1={f1:.4f} AUC={auc:.4f}")

        # Return regression
        reg = lgb.LGBMRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            num_leaves=31, min_child_samples=50,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1, n_jobs=-1,
        )
        reg.fit(X_train, y_ret_train,
                eval_set=[(X_test, y_ret_test)],
                callbacks=[lgb.early_stopping(30, verbose=False)])

        y_ret_pred = reg.predict(X_test)
        mae = mean_absolute_error(y_ret_test, y_ret_pred)
        r2 = r2_score(y_ret_test, y_ret_pred)

        # Directional accuracy from regression
        dir_from_reg = (np.sign(y_ret_pred) == np.sign(y_ret_test)).mean()

        print(f"    [Retorno]   MAE={mae:.6f} R²={r2:.4f} DirAcc={dir_from_reg:.4f}")

        # Feature importance
        feat_names = [
            'total_range', 'mean_bar_range', 'body_range_ratio', 'slope_norm',
            'intra_vol', 'wick_asymmetry', 'vol_ratio', 'vol_mean',
            'close_pos_total', 'direction', 'dominant_session', 'crosses_session',
            'gap', 'vol_relative'
        ] + [f'cluster_{i}' for i in range(n_clusters_total)]

        importance = clf.feature_importances_
        top_feat_idx = np.argsort(importance)[::-1][:10]
        top_feats = [(feat_names[i] if i < len(feat_names) else f'f{i}',
                      float(importance[i])) for i in top_feat_idx]

        print(f"    Top features: {', '.join(f'{n}({v:.0f})' for n, v in top_feats[:5])}")

        h_result = {
            'horizon': H,
            'n_train': int(len(X_train)),
            'n_test': int(len(X_test)),
            'cluster_predictive': cluster_predictive,
            'direction': {
                'accuracy': float(acc),
                'baseline_accuracy': float(baseline_acc),
                'f1_weighted': float(f1),
                'auc': float(auc),
                'lift_vs_baseline': float(acc - baseline_acc),
            },
            'return_regression': {
                'mae': float(mae),
                'r2': float(r2),
                'directional_accuracy': float(dir_from_reg),
            },
            'top_features': top_feats,
        }

        w_results['horizons'][H] = h_result

    elapsed = time.time() - t0
    w_results['elapsed_seconds'] = elapsed
    all_predictive_results[W] = w_results
    print(f"\n  W={W} completado en {elapsed:.1f}s")


# ============================================================
# 4. RESUMEN COMPARATIVO
# ============================================================
print("\n" + "=" * 80)
print("RESUMEN COMPARATIVO DE PODER PREDICTIVO")
print("=" * 80)

print(f"\n{'W':>4s} | {'H':>4s} | {'Acc':>7s} | {'Base':>7s} | {'Lift':>7s} | "
      f"{'AUC':>7s} | {'MAE':>9s} | {'R²':>7s} | {'DirAcc':>7s}")
print("-" * 85)

best_configs = []
for W, wr in sorted(all_predictive_results.items()):
    for H, hr in sorted(wr['horizons'].items()):
        d = hr['direction']
        r = hr['return_regression']
        lift = d['lift_vs_baseline']
        print(f"{W:>4d} | {H:>4d} | {d['accuracy']:>7.4f} | {d['baseline_accuracy']:>7.4f} | "
              f"{lift:>+7.4f} | {d['auc']:>7.4f} | {r['mae']:>9.6f} | "
              f"{r['r2']:>7.4f} | {r['directional_accuracy']:>7.4f}")
        best_configs.append({
            'W': W, 'H': H, 'lift': lift, 'auc': d['auc'],
            'dir_acc': d['accuracy'], 'r2': r['r2'],
        })

# Best configurations
best_configs.sort(key=lambda x: x['auc'], reverse=True)
print(f"\nMejores configuraciones por AUC:")
for bc in best_configs[:5]:
    print(f"  W={bc['W']} H={bc['H']}: AUC={bc['auc']:.4f} Lift={bc['lift']:+.4f}")


# ============================================================
# 5. ANÁLISIS DE CLUSTERS COMO SEÑALES
# ============================================================
print("\n" + "=" * 80)
print("CLUSTERS COMO SEÑALES DE TRADING")
print("=" * 80)

# For best W, analyze each cluster as a potential signal
best_W = optimal_Ws[0]
wr = all_predictive_results[best_W]

signal_clusters = []
for H in FORWARD_HORIZONS:
    if H not in wr['horizons']:
        continue
    hr = wr['horizons'][H]
    for c, cp in hr['cluster_predictive'].items():
        if cp['n'] < 50:
            continue
        # Signal strength: how far from 50/50 is the direction
        bias = abs(cp['direction_pct_up'] - 0.5)
        if bias > 0.03:  # At least 3% edge
            signal_clusters.append({
                'W': best_W, 'H': H, 'cluster': c,
                'n': cp['n'],
                'direction_pct': cp['direction_pct_up'],
                'return_mean': cp['return_mean'],
                'sharpe': cp['sharpe'],
                'profit_factor': cp['profit_factor'],
                'bias': bias,
            })

signal_clusters.sort(key=lambda x: x['bias'], reverse=True)
print(f"\nW={best_W} — Clusters con sesgo direccional >3%:")
print(f"{'C':>4s} | {'H':>4s} | {'n':>7s} | {'Dir%':>7s} | {'AvgRet':>10s} | {'Sharpe':>7s} | {'PF':>7s}")
print("-" * 60)
for sc in signal_clusters:
    dir_str = f"{sc['direction_pct']*100:.1f}%"
    print(f"  C{sc['cluster']} | {sc['H']:>4d} | {sc['n']:>7,} | {dir_str:>7s} | "
          f"{sc['return_mean']*100:>+9.4f}% | {sc['sharpe']:>7.3f} | {sc['profit_factor']:>7.2f}")


# ============================================================
# 6. VISUALIZACIONES
# ============================================================
print("\n[6] Generando visualizaciones...")

n_Ws = len(all_predictive_results)
n_Hs = len(FORWARD_HORIZONS)

fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# 6a. AUC heatmap (W x H)
ax = axes[0, 0]
Ws_sorted = sorted(all_predictive_results.keys())
auc_matrix = np.full((len(Ws_sorted), n_Hs), np.nan)
for i, W in enumerate(Ws_sorted):
    for j, H in enumerate(FORWARD_HORIZONS):
        if H in all_predictive_results[W]['horizons']:
            auc_matrix[i, j] = all_predictive_results[W]['horizons'][H]['direction']['auc']

im = ax.imshow(auc_matrix, aspect='auto', cmap='RdYlGn', vmin=0.45, vmax=0.60)
ax.set_xticks(range(n_Hs))
ax.set_xticklabels([f'H={h}' for h in FORWARD_HORIZONS])
ax.set_yticks(range(len(Ws_sorted)))
ax.set_yticklabels([f'W={w}' for w in Ws_sorted])
ax.set_title('AUC por Ventana × Horizonte')
for i in range(len(Ws_sorted)):
    for j in range(n_Hs):
        if not np.isnan(auc_matrix[i, j]):
            ax.text(j, i, f'{auc_matrix[i,j]:.3f}', ha='center', va='center', fontsize=8)
fig.colorbar(im, ax=ax, shrink=0.8)

# 6b. Lift vs baseline
ax = axes[0, 1]
lift_matrix = np.full((len(Ws_sorted), n_Hs), np.nan)
for i, W in enumerate(Ws_sorted):
    for j, H in enumerate(FORWARD_HORIZONS):
        if H in all_predictive_results[W]['horizons']:
            lift_matrix[i, j] = all_predictive_results[W]['horizons'][H]['direction']['lift_vs_baseline']

im2 = ax.imshow(lift_matrix, aspect='auto', cmap='RdBu', vmin=-0.05, vmax=0.05)
ax.set_xticks(range(n_Hs))
ax.set_xticklabels([f'H={h}' for h in FORWARD_HORIZONS])
ax.set_yticks(range(len(Ws_sorted)))
ax.set_yticklabels([f'W={w}' for w in Ws_sorted])
ax.set_title('Lift vs Baseline (Accuracy)')
for i in range(len(Ws_sorted)):
    for j in range(n_Hs):
        if not np.isnan(lift_matrix[i, j]):
            ax.text(j, i, f'{lift_matrix[i,j]:+.3f}', ha='center', va='center', fontsize=8)
fig.colorbar(im2, ax=ax, shrink=0.8)

# 6c. Cluster direction bias for best W
ax = axes[1, 0]
best_wr = all_predictive_results[best_W]
bar_data = {}
for H in FORWARD_HORIZONS:
    if H not in best_wr['horizons']:
        continue
    cp = best_wr['horizons'][H]['cluster_predictive']
    for c, vals in cp.items():
        if c not in bar_data:
            bar_data[c] = {}
        bar_data[c][H] = vals['direction_pct_up']

x_positions = np.arange(len(FORWARD_HORIZONS))
width = 0.15
clusters_to_plot = sorted(bar_data.keys())
colors = plt.cm.Set2(np.linspace(0, 1, max(len(clusters_to_plot), 1)))
for k, c in enumerate(clusters_to_plot):
    vals = [bar_data[c].get(H, np.nan) for H in FORWARD_HORIZONS]
    ax.bar(x_positions + k * width, vals, width, label=f'C{c}', color=colors[k])

ax.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='50/50')
ax.set_xticks(x_positions + width * len(clusters_to_plot) / 2)
ax.set_xticklabels([f'H={h}' for h in FORWARD_HORIZONS])
ax.set_ylabel('% Dirección Up')
ax.set_title(f'W={best_W}: Sesgo Direccional por Cluster × Horizonte')
ax.legend(fontsize=7, ncol=3)
ax.set_ylim(0.35, 0.65)

# 6d. Feature importance (best W, best H by AUC)
ax = axes[1, 1]
if best_configs:
    best_cfg = best_configs[0]
    bW, bH = best_cfg['W'], best_cfg['H']
    hr = all_predictive_results[bW]['horizons'][bH]
    top_f = hr['top_features'][:10]
    names = [f[0] for f in top_f][::-1]
    vals = [f[1] for f in top_f][::-1]
    ax.barh(names, vals, color='teal')
    ax.set_title(f'Feature Importance (W={bW}, H={bH})')
    ax.set_xlabel('Importancia')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fase3_predictivo.png'), dpi=150, bbox_inches='tight')
print(f"  Guardado: {OUTPUT_DIR}/fase3_predictivo.png")


# ============================================================
# 7. GUARDAR RESULTADOS
# ============================================================
summary = {
    'optimal_Ws_analyzed': optimal_Ws,
    'forward_horizons': FORWARD_HORIZONS,
    'best_configs_by_auc': best_configs[:10],
    'signal_clusters': signal_clusters,
    'results': {},
}

for W, wr in all_predictive_results.items():
    w_summary = {'W': W, 'horizons': {}}
    for H, hr in wr['horizons'].items():
        w_summary['horizons'][str(H)] = {
            'direction': hr['direction'],
            'return_regression': hr['return_regression'],
            'cluster_predictive': hr['cluster_predictive'],
            'top_features': hr['top_features'],
            'n_train': hr['n_train'],
            'n_test': hr['n_test'],
        }
    summary['results'][str(W)] = w_summary

with open(os.path.join(OUTPUT_DIR, 'fase3_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2, default=str)

with open(os.path.join(OUTPUT_DIR, 'fase3_all_results.pkl'), 'wb') as f:
    pickle.dump(all_predictive_results, f)

print(f"  Guardados: fase3_summary.json, fase3_all_results.pkl")

print("\n" + "=" * 80)
print("FASE 3 COMPLETADA — Esperando confirmación para Fase 4")
print("=" * 80)
