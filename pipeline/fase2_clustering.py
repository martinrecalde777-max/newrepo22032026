"""
FASE 2 — Descubrimiento de Agrupaciones Óptimas
Pipeline de descubrimiento de morfologías predictivas MNQ futures
Optimized with vectorized feature extraction.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import hdbscan
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
import os
import json
import pickle
import time

warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/user/newrepo22032026/outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("FASE 2 — DESCUBRIMIENTO DE AGRUPACIONES ÓPTIMAS")
print("=" * 80)

# ============================================================
# CARGAR DATOS LIMPIOS
# ============================================================
print("\n[0] Cargando dataset limpio de Fase 1...")
df = pd.read_parquet(os.path.join(OUTPUT_DIR, 'mnq_clean.parquet'))
df['ts_et'] = pd.to_datetime(df['ts_et'])
print(f"  Barras cargadas: {len(df):,}")

SESSION_MAP = {'Asia': 0, 'London': 1, 'RTH': 2, 'Post-Market': 3}
df['session_code'] = df['session'].map(SESSION_MAP).astype(float)

# Pre-compute arrays for speed
opens = df['open'].values.astype(np.float64)
highs = df['high'].values.astype(np.float64)
lows = df['low'].values.astype(np.float64)
closes = df['close'].values.astype(np.float64)
volumes = df['volume'].values.astype(np.float64)
session_codes = df['session_code'].values
gaps = df['gap'].values.astype(np.float64)
vol_rel = df['vol_relative_hour'].values.astype(np.float64)
timestamps = df['ts_et'].values

FEATURE_NAMES = [
    'total_range', 'mean_bar_range', 'body_range_ratio', 'slope_norm',
    'intra_vol', 'wick_asymmetry', 'vol_ratio', 'vol_mean',
    'close_pos_total', 'direction', 'dominant_session', 'crosses_session',
    'gap', 'vol_relative'
]

WINDOW_SIZES = [3, 5, 8, 10, 13, 15, 20, 30, 45, 60, 90, 120]


def extract_features_vectorized(W, sample_step=1):
    """Vectorized feature extraction for sliding windows of size W."""
    n = len(opens)

    # Subsample to keep total windows manageable (~300K max)
    target_windows = 300000
    n_possible = n - W + 1
    if n_possible > target_windows:
        sample_step = max(sample_step, n_possible // target_windows)

    # Check continuity: mark where gaps > 5 min exist
    ts_diff = np.diff(timestamps.astype('int64')) / 1e9  # seconds
    gap_mask = np.zeros(n, dtype=bool)
    gap_mask[1:] = ts_diff > 300  # gap > 5 min
    # Cumulative gap count - windows crossing a gap are invalid
    gap_cumsum = np.cumsum(gap_mask)

    all_indices = np.arange(0, n - W + 1, sample_step)

    # Filter: only windows where no gap occurs inside
    valid_mask = (gap_cumsum[all_indices + W - 1] - gap_cumsum[all_indices]) == 0
    indices = all_indices[valid_mask]

    if len(indices) == 0:
        return None, None

    n_win = len(indices)
    features = np.zeros((n_win, 14), dtype=np.float64)

    # Pre-compute rolling arrays using stride tricks for small W,
    # or loop in chunks for larger W
    print(f"    Computing features for {n_win:,} windows (step={sample_step})...")

    # Batch process
    batch_size = 50000
    for batch_start in range(0, n_win, batch_size):
        batch_end = min(batch_start + batch_size, n_win)
        batch_idx = indices[batch_start:batch_end]

        for local_i, start in enumerate(batch_idx):
            i = batch_start + local_i
            end = start + W

            o = opens[start:end]
            h = highs[start:end]
            l = lows[start:end]
            c = closes[start:end]
            v = volumes[start:end]
            sc = session_codes[start:end]

            h_max = h.max()
            l_min = l.min()
            total_range = h_max - l_min
            if total_range < 1e-10:
                total_range = 1e-10

            bar_ranges = h - l
            mean_bar_range = bar_ranges.mean()

            bodies = np.abs(c - o)
            safe_ranges = np.where(bar_ranges > 0, bar_ranges, 1e-10)
            body_range_ratio = (bodies / safe_ranges).mean()

            # Slope (linear regression on closes)
            x = np.arange(W, dtype=np.float64)
            c_mean = c.mean()
            x_mean = (W - 1) / 2.0
            slope = np.dot(x - x_mean, c - c_mean) / max(np.dot(x - x_mean, x - x_mean), 1e-10)
            slope_norm = slope / max(c_mean, 1e-10) * W

            # Intra-window volatility
            log_c = np.log(np.maximum(c, 1e-10))
            rets = np.diff(log_c)
            intra_vol = np.std(rets) if len(rets) > 1 else 0.0

            # Wick asymmetry
            upper_wicks = h - np.maximum(o, c)
            lower_wicks = np.minimum(o, c) - l
            total_wick = upper_wicks.sum() + lower_wicks.sum()
            wick_asym = (upper_wicks.sum() - lower_wicks.sum()) / total_wick if total_wick > 0 else 0.0

            # Volume ratio (first half vs second half)
            half = W // 2
            vol_first = v[:half].mean() if half > 0 else v[0]
            vol_last = v[half:].mean()
            vol_ratio_val = vol_first / max(vol_last, 1) - 1.0

            vol_mean_val = v.mean()
            close_pos_total = (c[-1] - l_min) / total_range
            direction = (c[-1] - o[0]) / total_range

            # Session features
            # Dominant session (mode)
            counts = np.bincount(sc.astype(int), minlength=4)
            dominant_session = np.argmax(counts)
            crosses_session = 1.0 if len(set(sc.astype(int))) > 1 else 0.0

            gap_val = gaps[start] if np.isfinite(gaps[start]) else 0.0
            vol_rel_mean = np.nanmean(vol_rel[start:end])

            features[i] = [
                total_range, mean_bar_range, body_range_ratio, slope_norm,
                intra_vol, wick_asym, vol_ratio_val, vol_mean_val,
                close_pos_total, direction, dominant_session, crosses_session,
                gap_val, vol_rel_mean
            ]

    end_indices = indices + W - 1
    return features, end_indices


def evaluate_kmeans(X_scaled, k_range):
    scores = {}
    sample_size = min(10000, len(X_scaled))
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42, max_iter=300)
        labels = km.fit_predict(X_scaled)
        if len(set(labels)) < 2:
            continue
        sil = silhouette_score(X_scaled, labels, sample_size=sample_size)
        ch = calinski_harabasz_score(X_scaled, labels)
        scores[k] = {'silhouette': sil, 'calinski_harabasz': ch, 'labels': labels, 'model': km}
    return scores


def evaluate_hdbscan(X_scaled, min_cluster_sizes=[50, 100, 200, 500]):
    best = None
    best_sil = -1
    results = {}
    for mcs in min_cluster_sizes:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=mcs, min_samples=10,
            metric='euclidean', cluster_selection_method='eom',
            prediction_data=True
        )
        labels = clusterer.fit_predict(X_scaled)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        if n_clusters >= 2:
            mask = labels != -1
            if mask.sum() > n_clusters:
                sil = silhouette_score(X_scaled[mask], labels[mask],
                                       sample_size=min(10000, int(mask.sum())))
                ch = calinski_harabasz_score(X_scaled[mask], labels[mask])
            else:
                sil, ch = -1, 0
        else:
            sil, ch = -1, 0
        results[mcs] = {
            'n_clusters': n_clusters, 'n_noise': n_noise,
            'noise_pct': n_noise / len(labels) * 100,
            'silhouette': sil, 'calinski_harabasz': ch,
            'labels': labels, 'model': clusterer
        }
        if sil > best_sil:
            best_sil = sil
            best = mcs
    return results, best


def temporal_stability(labels, end_indices, n_periods=5):
    ts = timestamps[end_indices]
    ts_min, ts_max = ts.min(), ts.max()
    period_edges = np.linspace(ts_min.astype('int64'), ts_max.astype('int64'), n_periods + 1)
    ts_int = ts.astype('int64')

    cluster_ids = sorted(set(labels[labels >= 0]))
    if not cluster_ids:
        return 0.0, {}

    presence = {c: [] for c in cluster_ids}
    for i in range(n_periods):
        mask = (ts_int >= period_edges[i]) & (ts_int < period_edges[i+1])
        period_labels = labels[mask]
        for c in cluster_ids:
            presence[c].append(int((period_labels == c).sum()))

    stability_detail = {}
    stable = 0
    for c in cluster_ids:
        pp = sum(1 for x in presence[c] if x >= 10)
        stability_detail[int(c)] = {
            'periods_present': pp,
            'counts_per_period': presence[c],
            'stable': pp >= n_periods - 1
        }
        if pp >= n_periods - 1:
            stable += 1

    return stable / len(cluster_ids), stability_detail


# ============================================================
# MAIN LOOP
# ============================================================
all_results = {}

for W in WINDOW_SIZES:
    print(f"\n{'='*60}")
    print(f"  VENTANA W = {W}")
    print(f"{'='*60}")
    t0 = time.time()

    X, end_idx = extract_features_vectorized(W)
    if X is None:
        print(f"  No hay ventanas válidas para W={W}")
        continue

    print(f"  Ventanas extraídas: {len(X):,}")

    # Clean NaN/Inf
    mask_valid = np.isfinite(X).all(axis=1)
    X = X[mask_valid]
    end_idx = end_idx[mask_valid]
    print(f"  Ventanas válidas: {len(X):,}")

    if len(X) < 500:
        print(f"  Insuficientes ventanas")
        continue

    # For very large datasets, subsample for clustering then propagate
    MAX_CLUSTER_SAMPLES = 50000
    if len(X) > MAX_CLUSTER_SAMPLES:
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(len(X), MAX_CLUSTER_SAMPLES, replace=False)
        X_sample = X[sample_idx]
    else:
        X_sample = X
        sample_idx = None

    scaler = StandardScaler()
    X_sample_scaled = scaler.fit_transform(X_sample)

    # --- HDBSCAN ---
    print(f"  [HDBSCAN] Evaluando...")
    hdb_results, best_mcs = evaluate_hdbscan(X_sample_scaled)
    if best_mcs is not None:
        hdb_best = hdb_results[best_mcs]
        print(f"    Mejor mcs={best_mcs}: {hdb_best['n_clusters']} clusters, "
              f"sil={hdb_best['silhouette']:.4f}, noise={hdb_best['noise_pct']:.1f}%")
    else:
        hdb_best = None

    # --- K-Means ---
    print(f"  [K-Means] Evaluando k=2..15...")
    km_results = evaluate_kmeans(X_sample_scaled, range(2, 16))
    if km_results:
        best_k = max(km_results, key=lambda k: km_results[k]['silhouette'])
        km_best = km_results[best_k]
        print(f"    Mejor k={best_k}: sil={km_best['silhouette']:.4f}, CH={km_best['calinski_harabasz']:.1f}")
    else:
        km_best = None
        best_k = None

    # Select best
    hdb_sil = hdb_best['silhouette'] if hdb_best else -1
    km_sil = km_best['silhouette'] if km_best else -1

    if hdb_sil >= km_sil and hdb_best is not None:
        best_method = 'HDBSCAN'
        best_labels_sample = hdb_best['labels']
        best_sil = hdb_sil
        best_ch = hdb_best['calinski_harabasz']
        best_model = hdb_best['model']
    elif km_best is not None:
        best_method = f'KMeans(k={best_k})'
        best_labels_sample = km_best['labels']
        best_sil = km_sil
        best_ch = km_best['calinski_harabasz']
        best_model = km_best['model']
    else:
        print(f"  Ningún método produjo clusters válidos para W={W}")
        continue

    # Propagate labels to full dataset if subsampled
    if sample_idx is not None:
        X_all_scaled = scaler.transform(X)
        if 'KMeans' in best_method:
            best_labels = best_model.predict(X_all_scaled)
        else:
            # For HDBSCAN, use approximate_predict
            best_labels, _ = hdbscan.approximate_predict(best_model, X_all_scaled)
        end_idx_final = end_idx
    else:
        best_labels = best_labels_sample
        end_idx_final = end_idx
        X_all_scaled = X_sample_scaled

    # Stability
    stability_score, stability_detail = temporal_stability(best_labels, end_idx_final)
    print(f"  Mejor: {best_method} | sil={best_sil:.4f} | stability={stability_score:.2f}")

    # Cluster sizes
    unique_labels = sorted(set(best_labels))
    cluster_sizes = {int(c): int((best_labels == c).sum()) for c in unique_labels}
    valid_clusters = [c for c, s in cluster_sizes.items() if c >= 0 and s >= 50]
    print(f"  Clusters (>=50): {len(valid_clusters)} de {len([c for c in unique_labels if c >= 0])}")
    for c in sorted(valid_clusters):
        print(f"    C{c}: {cluster_sizes[c]:,}")

    # Cluster stats
    cluster_stats = {}
    for c in valid_clusters:
        mask_c = best_labels == c
        X_c = X[mask_c]
        stats_c = {}
        for j, fname in enumerate(FEATURE_NAMES):
            vals = X_c[:, j]
            stats_c[fname] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
                'median': float(np.median(vals)),
            }
        cluster_stats[int(c)] = stats_c

    elapsed = time.time() - t0

    all_results[W] = {
        'n_windows': len(X),
        'best_method': best_method,
        'silhouette': float(best_sil),
        'calinski_harabasz': float(best_ch),
        'stability_score': float(stability_score),
        'stability_detail': stability_detail,
        'n_clusters_valid': len(valid_clusters),
        'cluster_sizes': cluster_sizes,
        'valid_clusters': valid_clusters,
        'cluster_stats': cluster_stats,
        'elapsed_seconds': elapsed,
    }

    # Save per-W data
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, f'fase2_W{W}_data.npz'),
        X=X, X_scaled=scaler.transform(X), end_idx=end_idx_final, labels=best_labels
    )
    with open(os.path.join(OUTPUT_DIR, f'fase2_W{W}_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(OUTPUT_DIR, f'fase2_W{W}_model.pkl'), 'wb') as f:
        pickle.dump(best_model, f)

    print(f"  Completado en {elapsed:.1f}s")


# ============================================================
# RANKING
# ============================================================
print("\n" + "=" * 80)
print("RANKING DE VENTANAS ÓPTIMAS")
print("=" * 80)

ranking = []
for W, res in sorted(all_results.items()):
    composite = res['silhouette'] * res['stability_score'] * np.log1p(res['n_clusters_valid'])
    ranking.append({
        'W': W,
        'method': res['best_method'],
        'silhouette': res['silhouette'],
        'CH': res['calinski_harabasz'],
        'stability': res['stability_score'],
        'n_clusters': res['n_clusters_valid'],
        'n_windows': res['n_windows'],
        'composite': composite,
    })

ranking.sort(key=lambda x: x['composite'], reverse=True)

print(f"\n{'W':>4s} | {'Método':>15s} | {'Sil':>8s} | {'CH':>10s} | {'Stab':>5s} | {'Cls':>4s} | {'Windows':>9s} | {'Score':>7s}")
print("-" * 80)
for r in ranking:
    print(f"{r['W']:>4d} | {r['method']:>15s} | {r['silhouette']:>8.4f} | {r['CH']:>10.1f} | "
          f"{r['stability']:>5.2f} | {r['n_clusters']:>4d} | {r['n_windows']:>9,} | {r['composite']:>7.4f}")

optimal_Ws = [r['W'] for r in ranking if r['composite'] > 0 and r['n_clusters'] >= 2][:5]
if not optimal_Ws:
    # Fallback: top 3 by silhouette
    optimal_Ws = [r['W'] for r in sorted(ranking, key=lambda x: x['silhouette'], reverse=True)[:3]]
print(f"\nVentanas óptimas seleccionadas: {optimal_Ws}")

# ============================================================
# VISUALIZATIONS
# ============================================================
print("\n[Generando visualizaciones...]")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

ax = axes[0, 0]
ws = [r['W'] for r in ranking]
sils = [r['silhouette'] for r in ranking]
ax.barh([str(w) for w in ws], sils, color='steelblue')
ax.set_xlabel('Silhouette Score')
ax.set_title('Silhouette por tamaño de ventana')
ax.invert_yaxis()

ax = axes[0, 1]
stabs = [r['stability'] for r in ranking]
ax.barh([str(w) for w in ws], stabs, color='forestgreen')
ax.set_xlabel('Estabilidad Temporal')
ax.set_title('Estabilidad temporal por ventana')
ax.invert_yaxis()

ax = axes[1, 0]
comps = [r['composite'] for r in ranking]
colors_bar = ['gold' if r['W'] in optimal_Ws else 'gray' for r in ranking]
ax.barh([str(w) for w in ws], comps, color=colors_bar)
ax.set_xlabel('Score Compuesto')
ax.set_title('Score compuesto (óptimos en dorado)')
ax.invert_yaxis()

ax = axes[1, 1]
if optimal_Ws:
    best_W = optimal_Ws[0]
    data = np.load(os.path.join(OUTPUT_DIR, f'fase2_W{best_W}_data.npz'))
    X_best = data['X_scaled']
    labels_best = data['labels']

    n_sample = min(15000, len(X_best))
    rng = np.random.RandomState(42)
    idx_sample = rng.choice(len(X_best), n_sample, replace=False)
    X_sub = X_best[idx_sample]
    labels_sub = labels_best[idx_sample]

    try:
        import umap
        print(f"  Calculando UMAP para W={best_W}...")
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
        embedding = reducer.fit_transform(X_sub)
        unique_labs = sorted(set(labels_sub))
        cmap = plt.cm.get_cmap('tab20', max(len(unique_labs), 1))
        for i, c in enumerate(unique_labs):
            mask = labels_sub == c
            lbl = 'Noise' if c == -1 else f'C{c}'
            alpha = 0.1 if c == -1 else 0.5
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                      c=[cmap(i)], s=3, alpha=alpha, label=lbl)
        ax.set_title(f'UMAP — W={best_W} ({all_results[best_W]["best_method"]})')
        ax.legend(fontsize=7, markerscale=3, ncol=2, loc='best')
    except Exception as e:
        ax.text(0.5, 0.5, f'UMAP error: {e}', transform=ax.transAxes, ha='center', va='center')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fase2_clustering.png'), dpi=150, bbox_inches='tight')
print(f"  Guardado: {OUTPUT_DIR}/fase2_clustering.png")

# ============================================================
# CLUSTER DESCRIPTIONS
# ============================================================
print("\n" + "=" * 80)
print("DESCRIPCIÓN ESTADÍSTICA DE CLUSTERS PRINCIPALES")
print("=" * 80)

for W in optimal_Ws[:3]:
    res = all_results[W]
    print(f"\n--- W = {W} ({res['best_method']}) ---")
    print(f"  Sil: {res['silhouette']:.4f} | CH: {res['calinski_harabasz']:.1f} | Stab: {res['stability_score']:.2f}")

    for c in sorted(res['valid_clusters']):
        size = res['cluster_sizes'].get(c, 0)
        stats = res['cluster_stats'].get(c, {})
        print(f"\n  Cluster {c} (n={size:,}):")
        for fname in ['total_range', 'slope_norm', 'intra_vol', 'body_range_ratio',
                       'wick_asymmetry', 'direction', 'vol_mean', 'vol_relative']:
            if fname in stats:
                s = stats[fname]
                print(f"    {fname:20s}: mean={s['mean']:>10.4f}  std={s['std']:>10.4f}  med={s['median']:>10.4f}")

# Save summaries
with open(os.path.join(OUTPUT_DIR, 'fase2_summary.json'), 'w') as f:
    save_results = {}
    for W, res in all_results.items():
        save_res = {k: v for k, v in res.items() if k != 'cluster_stats'}
        save_results[str(W)] = save_res
    json.dump({
        'ranking': ranking,
        'optimal_Ws': optimal_Ws,
        'results': save_results,
    }, f, indent=2, default=str)

with open(os.path.join(OUTPUT_DIR, 'fase2_all_results.pkl'), 'wb') as f:
    pickle.dump(all_results, f)

print(f"\n  Guardados: fase2_summary.json, fase2_all_results.pkl")

print("\n" + "=" * 80)
print("FASE 2 COMPLETADA — Esperando confirmación para Fase 3")
print("=" * 80)
