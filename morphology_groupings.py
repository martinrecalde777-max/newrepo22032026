#!/usr/bin/env python3
"""
=============================================================================
MORPHOLOGICAL GROUPING ANALYSIS - MNQ Futures
=============================================================================
Estudia la MORFOLOGÍA de agrupaciones de velas de 1 minuto:
  - Clasifica la FORMA de cada agrupación (no cada vela individual)
  - Incluye CONTEXTO: de dónde viene (agrupación previa, dirección, volumen)
  - Predice a dónde va (EV en puntos NQ)
  - Filtra solo patrones con EV >= 50 puntos

Agrupaciones estudiadas: 15, 30, 60, 120, 240 velas de 1 min
=============================================================================
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import json
import os
from itertools import product
from collections import defaultdict

# ============================================================================
# 1. LOAD DATA
# ============================================================================

print("=" * 80)
print("MORPHOLOGICAL GROUPING ANALYSIS - MNQ FUTURES")
print("=" * 80)

df = pd.read_parquet('data/mnq_preprocessed.parquet')
print(f"Data: {len(df):,} bars | {df.timestamp.min()} to {df.timestamp.max()}")
print(f"Price range: {df.close.min():.2f} - {df.close.max():.2f}")

# ============================================================================
# 2. DEFINE MORPHOLOGY CLASSIFICATION
# ============================================================================
# Para cada agrupación de N velas, calculamos:
#   - net_return: close[-1] - close[0] (cuánto se movió)
#   - range: max(high) - min(low) (cuánto osciló)
#   - efficiency: |net_return| / range (qué tan directo fue el movimiento)
#   - close_position: dónde cierra respecto al rango (0=piso, 1=techo)
#   - open_position: dónde abre respecto al rango
#   - internal_path: suma de |close[i]-close[i-1]| / range (cuánto zigzagueó)
#   - volume_profile: volumen primera mitad vs segunda mitad
#   - direction_consistency: % de velas en la dirección dominante

GROUPING_SIZES = [15, 30, 60, 120, 240]
FUTURE_HORIZONS = [15, 30, 60, 120]  # minutos para medir EV
MIN_EV_POINTS = 50.0
MIN_SAMPLES = 30  # mínimo de ocurrencias para considerar el patrón

def classify_morphology(net_ret, grp_range, efficiency, close_pos, open_pos,
                        vol_first_half_ratio, dir_consistency, internal_path_ratio):
    """
    Clasifica la morfología de una agrupación en categorías ricas.

    Returns: string con el nombre de la morfología
    """
    # Normalizar
    is_bullish = net_ret > 0
    is_bearish = net_ret < 0
    is_flat = abs(net_ret) < grp_range * 0.05 if grp_range > 0 else True

    high_efficiency = efficiency > 0.65
    med_efficiency = 0.35 < efficiency <= 0.65
    low_efficiency = efficiency <= 0.35

    closes_high = close_pos > 0.75
    closes_low = close_pos < 0.25
    closes_mid = 0.25 <= close_pos <= 0.75

    opens_high = open_pos > 0.75
    opens_low = open_pos < 0.25

    vol_front_loaded = vol_first_half_ratio > 1.3
    vol_back_loaded = vol_first_half_ratio < 0.7

    high_consistency = dir_consistency > 0.65
    low_consistency = dir_consistency < 0.45

    high_zigzag = internal_path_ratio > 3.0

    # === MORFOLOGÍAS BULLISH ===
    if is_bullish and high_efficiency and closes_high and high_consistency:
        return "IMPULSO_BULL"  # Tendencia alcista fuerte y limpia

    if is_bullish and high_efficiency and closes_high and not high_consistency:
        return "RALLY_ERRATICO_BULL"  # Sube fuerte pero con pullbacks

    if is_bullish and closes_high and opens_high and low_efficiency:
        return "CONSOLIDACION_ALTA"  # Se queda arriba, poco movimiento neto

    if is_bullish and closes_high and opens_low:
        return "V_BOTTOM"  # Abre abajo, cierra arriba (reversal alcista)

    if is_bullish and med_efficiency and vol_back_loaded:
        return "ACUMULACION_BULL"  # Volumen creciente hacia arriba

    if is_bullish and med_efficiency and vol_front_loaded:
        return "AGOTAMIENTO_BULL"  # Volumen decreciente, pierde fuerza

    # === MORFOLOGÍAS BEARISH ===
    if is_bearish and high_efficiency and closes_low and high_consistency:
        return "IMPULSO_BEAR"  # Tendencia bajista fuerte y limpia

    if is_bearish and high_efficiency and closes_low and not high_consistency:
        return "SELL_ERRATICO_BEAR"  # Baja fuerte pero con bounces

    if is_bearish and closes_low and opens_low and low_efficiency:
        return "CONSOLIDACION_BAJA"  # Se queda abajo

    if is_bearish and closes_low and opens_high:
        return "INV_V_TOP"  # Abre arriba, cierra abajo (reversal bajista)

    if is_bearish and med_efficiency and vol_back_loaded:
        return "DISTRIBUCION_BEAR"  # Volumen creciente hacia abajo

    if is_bearish and med_efficiency and vol_front_loaded:
        return "AGOTAMIENTO_BEAR"  # Volumen decreciente bajando

    # === MORFOLOGÍAS DE RANGO / INDECISIÓN ===
    if is_flat and low_efficiency and high_zigzag:
        return "RANGO_VOLATIL"  # Mucho movimiento, nada neto (chop)

    if is_flat and low_efficiency and not high_zigzag:
        return "COMPRESION"  # Poco movimiento, poca volatilidad (calma antes de tormenta?)

    if closes_high and opens_high and is_flat:
        return "TECHO_PLANO"  # Lateraliza en zona alta

    if closes_low and opens_low and is_flat:
        return "PISO_PLANO"  # Lateraliza en zona baja

    # === MORFOLOGÍAS ESPECIALES ===
    if closes_mid and high_zigzag and not is_flat:
        return "ZIGZAG_MEDIO"  # Mucho ruido, cierra al medio

    if is_bullish and low_efficiency:
        return "DRIFT_BULL"  # Deriva alcista lenta sin convicción

    if is_bearish and low_efficiency:
        return "DRIFT_BEAR"  # Deriva bajista lenta sin convicción

    return "INDEFINIDO"


def classify_volume_regime(vol_ratio):
    """Clasifica el régimen de volumen."""
    if vol_ratio > 2.0:
        return "VOL_EXPLOSIVO"
    elif vol_ratio > 1.3:
        return "VOL_ALTO"
    elif vol_ratio > 0.7:
        return "VOL_NORMAL"
    elif vol_ratio > 0.3:
        return "VOL_BAJO"
    else:
        return "VOL_MUERTO"


def classify_volatility_regime(vol_now, vol_ma):
    """Clasifica el régimen de volatilidad de precio."""
    if vol_ma == 0 or np.isnan(vol_ma):
        return "VOLA_NORMAL"
    ratio = vol_now / vol_ma if vol_ma > 0 else 1.0
    if ratio > 2.0:
        return "VOLA_EXPLOSION"
    elif ratio > 1.3:
        return "VOLA_ALTA"
    elif ratio > 0.7:
        return "VOLA_NORMAL"
    else:
        return "VOLA_COMPRIMIDA"


# ============================================================================
# 3. COMPUTE MORPHOLOGIES FOR EACH GROUPING SIZE
# ============================================================================

results_all = {}

for GRP_SIZE in GROUPING_SIZES:
    print(f"\n{'='*80}")
    print(f"  AGRUPACIÓN DE {GRP_SIZE} VELAS ({GRP_SIZE} MINUTOS)")
    print(f"{'='*80}")

    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    opn = df['open'].values
    volume = df['volume'].values
    direction = df['direction'].values
    vol_ratio = df['volume_ratio'].values
    volatility_20 = df['volatility_20'].values
    volatility_60 = df['volatility_60'].values

    n = len(df)

    # Pre-compute rolling metrics using vectorized operations
    print(f"  Computing morphological features...")

    # Rolling high/low for grouping range
    from numpy.lib.stride_tricks import sliding_window_view

    # We need enough data for the grouping + context + future
    max_future = max(FUTURE_HORIZONS)
    required_lookback = GRP_SIZE * 2  # current + previous grouping
    start_idx = required_lookback
    end_idx = n - max_future

    if end_idx <= start_idx:
        print(f"  SKIP: Not enough data for grouping {GRP_SIZE}")
        continue

    # Arrays to store results
    morphologies = []
    contexts = []
    futures = {}
    for h in FUTURE_HORIZONS:
        futures[h] = []

    # Vectorized computation of grouping features
    print(f"  Processing {end_idx - start_idx:,} potential patterns...")

    # Current grouping metrics (vectorized)
    grp_high = pd.Series(high).rolling(GRP_SIZE).max().values
    grp_low = pd.Series(low).rolling(GRP_SIZE).min().values
    grp_range = grp_high - grp_low
    grp_net_ret = close - np.roll(close, GRP_SIZE)
    grp_net_ret[:GRP_SIZE] = np.nan

    # Efficiency
    grp_efficiency = np.where(grp_range > 0, np.abs(grp_net_ret) / grp_range, 0)

    # Close position within range
    grp_close_pos = np.where(grp_range > 0, (close - grp_low) / grp_range, 0.5)

    # Open position (open of GRP_SIZE bars ago)
    open_of_group = np.roll(opn, GRP_SIZE - 1)
    open_of_group[:GRP_SIZE] = np.nan
    grp_open_pos = np.where(grp_range > 0, (open_of_group - grp_low) / grp_range, 0.5)

    # Volume first half vs second half ratio
    half = GRP_SIZE // 2
    vol_first = pd.Series(volume).rolling(half).sum().shift(half).values  # first half
    vol_second = pd.Series(volume).rolling(half).sum().values  # second half
    vol_half_ratio = np.where(vol_second > 0, vol_first / vol_second, 1.0)

    # Direction consistency: % of bars moving in dominant direction
    dir_sum = pd.Series(direction).rolling(GRP_SIZE).sum().values
    grp_dir_consistency = np.abs(dir_sum) / GRP_SIZE

    # Internal path ratio: sum of |bar moves| / range
    bar_moves = np.abs(np.diff(close, prepend=close[0]))
    internal_path = pd.Series(bar_moves).rolling(GRP_SIZE).sum().values
    grp_internal_path_ratio = np.where(grp_range > 0, internal_path / grp_range, 1.0)

    # Volume regime for the grouping
    grp_vol_total = pd.Series(volume).rolling(GRP_SIZE).sum().values
    grp_vol_ma_long = pd.Series(volume).rolling(GRP_SIZE * 5).mean().values * GRP_SIZE
    grp_vol_ratio = np.where(grp_vol_ma_long > 0, grp_vol_total / grp_vol_ma_long, 1.0)

    # Previous grouping metrics (shift by GRP_SIZE)
    prev_net_ret = np.roll(grp_net_ret, GRP_SIZE)
    prev_net_ret[:GRP_SIZE * 2] = np.nan
    prev_efficiency = np.roll(grp_efficiency, GRP_SIZE)
    prev_close_pos = np.roll(grp_close_pos, GRP_SIZE)
    prev_open_pos = np.roll(grp_open_pos, GRP_SIZE)
    prev_vol_half_ratio = np.roll(vol_half_ratio, GRP_SIZE)
    prev_dir_consistency = np.roll(grp_dir_consistency, GRP_SIZE)
    prev_internal_path_ratio = np.roll(grp_internal_path_ratio, GRP_SIZE)
    prev_range = np.roll(grp_range, GRP_SIZE)
    prev_range[:GRP_SIZE * 2] = np.nan

    # Classify all morphologies vectorized (via loop over valid indices but with precomputed arrays)
    print(f"  Classifying morphologies...")

    morph_current = np.empty(n, dtype=object)
    morph_prev = np.empty(n, dtype=object)
    vol_regime = np.empty(n, dtype=object)
    vola_regime = np.empty(n, dtype=object)

    for i in range(start_idx, end_idx):
        # Current morphology
        morph_current[i] = classify_morphology(
            grp_net_ret[i], grp_range[i], grp_efficiency[i],
            grp_close_pos[i], grp_open_pos[i],
            vol_half_ratio[i], grp_dir_consistency[i], grp_internal_path_ratio[i]
        )

        # Previous morphology (context)
        if not np.isnan(prev_net_ret[i]) and not np.isnan(prev_range[i]):
            morph_prev[i] = classify_morphology(
                prev_net_ret[i], prev_range[i], prev_efficiency[i],
                prev_close_pos[i], prev_open_pos[i],
                prev_vol_half_ratio[i], prev_dir_consistency[i], prev_internal_path_ratio[i]
            )
        else:
            morph_prev[i] = "UNKNOWN"

        # Volume regime
        vol_regime[i] = classify_volume_regime(grp_vol_ratio[i])

        # Volatility regime
        vola_regime[i] = classify_volatility_regime(
            volatility_20[i] if not np.isnan(volatility_20[i]) else 0,
            volatility_60[i] if not np.isnan(volatility_60[i]) else 1
        )

    # Future returns
    future_rets = {}
    for h in FUTURE_HORIZONS:
        future_rets[h] = np.roll(close, -h) - close
        future_rets[h][-h:] = np.nan

    # Build DataFrame for analysis
    print(f"  Building analysis DataFrame...")

    valid_mask = np.zeros(n, dtype=bool)
    valid_mask[start_idx:end_idx] = True
    # Also filter out any NaN futures
    for h in FUTURE_HORIZONS:
        valid_mask &= ~np.isnan(future_rets[h])
    # Filter out NaN in pre-computed arrays
    valid_mask &= ~np.isnan(grp_net_ret)
    valid_mask &= ~np.isnan(grp_range)
    valid_mask &= (morph_current != None)
    valid_mask &= (morph_prev != None)

    valid_indices = np.where(valid_mask)[0]
    print(f"  Valid patterns: {len(valid_indices):,}")

    analysis_df = pd.DataFrame({
        'idx': valid_indices,
        'morph_current': morph_current[valid_indices],
        'morph_prev': morph_prev[valid_indices],
        'vol_regime': vol_regime[valid_indices],
        'vola_regime': vola_regime[valid_indices],
        'grp_net_ret': grp_net_ret[valid_indices],
        'grp_range': grp_range[valid_indices],
        'grp_efficiency': grp_efficiency[valid_indices],
        'hour': df['hour'].values[valid_indices],
    })

    for h in FUTURE_HORIZONS:
        analysis_df[f'future_{h}'] = future_rets[h][valid_indices]

    # ========================================================================
    # 4. ANALYZE: MORPHOLOGY + CONTEXT -> FUTURE
    # ========================================================================

    print(f"\n  --- DISTRIBUTION OF MORPHOLOGIES ---")
    morph_counts = analysis_df['morph_current'].value_counts()
    for m, cnt in morph_counts.items():
        pct = cnt / len(analysis_df) * 100
        print(f"    {m:30s}: {cnt:8,} ({pct:5.1f}%)")

    # Group by: morph_current + morph_prev + vol_regime
    # This gives us: "AFTER morphology X, coming FROM morphology Y, with volume Z"

    print(f"\n  --- PATTERNS WITH EV >= {MIN_EV_POINTS} POINTS ---")
    print(f"  (Format: PREV_MORPH -> CURRENT_MORPH [VOL_REGIME] => future EV)")

    group_cols = ['morph_prev', 'morph_current', 'vol_regime']
    grouped = analysis_df.groupby(group_cols)

    patterns_found = []

    for (prev, curr, vol), group in grouped:
        if len(group) < MIN_SAMPLES:
            continue

        for h in FUTURE_HORIZONS:
            future_col = f'future_{h}'
            fut = group[future_col]

            ev = fut.mean()
            ev_abs = abs(ev)

            if ev_abs < MIN_EV_POINTS:
                continue

            std = fut.std()
            win_rate = (fut > 0).mean() if ev > 0 else (fut < 0).mean()
            median_ret = fut.median()
            p25 = fut.quantile(0.25)
            p75 = fut.quantile(0.75)
            max_ret = fut.max()
            min_ret = fut.min()
            sharpe = ev / std if std > 0 else 0

            # Directional EV (signed)
            direction_str = "LONG" if ev > 0 else "SHORT"

            pattern = {
                'grp_size': GRP_SIZE,
                'morph_prev': prev,
                'morph_current': curr,
                'vol_regime': vol,
                'horizon': h,
                'direction': direction_str,
                'n_samples': len(group),
                'ev_points': round(ev, 2),
                'ev_abs': round(ev_abs, 2),
                'median': round(median_ret, 2),
                'std': round(std, 2),
                'sharpe': round(sharpe, 4),
                'win_rate': round(win_rate, 4),
                'p25': round(p25, 2),
                'p75': round(p75, 2),
                'max': round(max_ret, 2),
                'min': round(min_ret, 2),
                # Average hour distribution
                'avg_hour': round(group['hour'].mean(), 1),
            }
            patterns_found.append(pattern)

    # Sort by |EV| descending
    patterns_found.sort(key=lambda x: x['ev_abs'], reverse=True)

    print(f"\n  FOUND {len(patterns_found)} patterns with EV >= {MIN_EV_POINTS} pts")
    print(f"  {'─'*100}")

    for i, p in enumerate(patterns_found[:40]):  # Top 40
        print(f"  #{i+1:3d} | {p['morph_prev']:25s} -> {p['morph_current']:25s} "
              f"[{p['vol_regime']:15s}] "
              f"| {p['direction']:5s} {p['horizon']:3d}min "
              f"| EV={p['ev_points']:+8.1f} pts "
              f"| med={p['median']:+7.1f} "
              f"| WR={p['win_rate']:.1%} "
              f"| N={p['n_samples']:6,} "
              f"| Sharpe={p['sharpe']:+.3f} "
              f"| ~{p['avg_hour']:.0f}h")

    results_all[GRP_SIZE] = patterns_found

# ============================================================================
# 5. CROSS-GROUPING ANALYSIS: MULTI-TIMEFRAME CONFLUENCE
# ============================================================================

print(f"\n\n{'='*80}")
print("  CROSS-GROUPING ANALYSIS: CONFLUENCIA MULTI-TEMPORAL")
print(f"{'='*80}")

# Flatten all patterns
all_patterns = []
for grp_size, patterns in results_all.items():
    all_patterns.extend(patterns)

# Group by morph_current to find which morphologies are consistently profitable
morph_ev = defaultdict(list)
for p in all_patterns:
    key = (p['morph_current'], p['direction'])
    morph_ev[key].append(p['ev_abs'])

print(f"\n  --- MORPHOLOGIES WITH CONSISTENT HIGH EV ACROSS TIMEFRAMES ---")
print(f"  {'Morphology':30s} {'Dir':6s} {'AvgEV':>8s} {'MaxEV':>8s} {'Count':>6s}")
print(f"  {'─'*70}")

morph_summary = []
for (morph, direction), evs in sorted(morph_ev.items(), key=lambda x: -np.mean(x[1])):
    if len(evs) >= 2:
        morph_summary.append({
            'morphology': morph,
            'direction': direction,
            'avg_ev': round(np.mean(evs), 1),
            'max_ev': round(np.max(evs), 1),
            'count': len(evs),
        })
        print(f"  {morph:30s} {direction:6s} {np.mean(evs):8.1f} {np.max(evs):8.1f} {len(evs):6d}")

# ============================================================================
# 6. TRANSITION MATRIX: WHAT FOLLOWS WHAT
# ============================================================================

print(f"\n\n{'='*80}")
print("  TRANSITION ANALYSIS: SECUENCIAS DE MORFOLOGÍAS MÁS RENTABLES")
print(f"{'='*80}")

# Best transitions (prev -> current) aggregated across all groupings & horizons
transition_ev = defaultdict(list)
for p in all_patterns:
    key = f"{p['morph_prev']} -> {p['morph_current']}"
    transition_ev[key].append({
        'ev': p['ev_points'],
        'ev_abs': p['ev_abs'],
        'direction': p['direction'],
        'grp_size': p['grp_size'],
        'horizon': p['horizon'],
        'n_samples': p['n_samples'],
        'win_rate': p['win_rate'],
        'sharpe': p['sharpe'],
    })

print(f"\n  --- TOP TRANSITIONS BY AVERAGE |EV| ---")
print(f"  {'Transition':55s} {'Dir':6s} {'AvgEV':>8s} {'MaxEV':>8s} {'AvgWR':>8s} {'AvgN':>8s} {'#Occurs':>8s}")
print(f"  {'─'*110}")

transition_sorted = sorted(
    transition_ev.items(),
    key=lambda x: np.mean([e['ev_abs'] for e in x[1]]),
    reverse=True
)

for key, entries in transition_sorted[:50]:
    avg_ev = np.mean([e['ev'] for e in entries])
    avg_ev_abs = np.mean([e['ev_abs'] for e in entries])
    max_ev = max([e['ev_abs'] for e in entries])
    avg_wr = np.mean([e['win_rate'] for e in entries])
    avg_n = np.mean([e['n_samples'] for e in entries])
    direction = entries[0]['direction']  # They could be mixed

    # Check if all same direction
    dirs = set(e['direction'] for e in entries)
    dir_str = direction if len(dirs) == 1 else "MIXED"

    print(f"  {key:55s} {dir_str:6s} {avg_ev:+8.1f} {max_ev:8.1f} {avg_wr:8.1%} {avg_n:8.0f} {len(entries):8d}")


# ============================================================================
# 7. HORA DEL DIA: MORPHOLOGÍAS CON EV ALTO POR HORA
# ============================================================================

print(f"\n\n{'='*80}")
print("  HOURLY ANALYSIS: MEJORES PATRONES POR HORA DEL DÍA")
print(f"{'='*80}")

hourly_patterns = defaultdict(list)
for p in all_patterns:
    hour = int(p['avg_hour'])
    hourly_patterns[hour].append(p)

for hour in sorted(hourly_patterns.keys()):
    pts = hourly_patterns[hour]
    if not pts:
        continue
    best = sorted(pts, key=lambda x: x['ev_abs'], reverse=True)[:3]
    print(f"\n  Hour {hour:02d}:00 UTC ({len(pts)} patterns)")
    for b in best:
        print(f"    {b['morph_prev']:20s} -> {b['morph_current']:20s} "
              f"[{b['vol_regime']:12s}] "
              f"{b['direction']:5s} {b['horizon']:3d}min "
              f"EV={b['ev_points']:+7.1f} WR={b['win_rate']:.1%} N={b['n_samples']:,}")


# ============================================================================
# 8. SAVE RESULTS
# ============================================================================

output = {
    'metadata': {
        'grouping_sizes': GROUPING_SIZES,
        'future_horizons': FUTURE_HORIZONS,
        'min_ev_points': MIN_EV_POINTS,
        'min_samples': MIN_SAMPLES,
        'total_bars': len(df),
        'date_range': f"{df.timestamp.min()} to {df.timestamp.max()}",
    },
    'patterns_by_grouping': {str(k): v for k, v in results_all.items()},
    'total_patterns_found': len(all_patterns),
    'morphology_summary': morph_summary,
}

with open('data/morphology_grouping_results.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\n\n{'='*80}")
print(f"  RESUMEN FINAL")
print(f"{'='*80}")
print(f"  Total patterns with EV >= {MIN_EV_POINTS} pts: {len(all_patterns)}")
for grp, pats in results_all.items():
    n_long = sum(1 for p in pats if p['direction'] == 'LONG')
    n_short = sum(1 for p in pats if p['direction'] == 'SHORT')
    if pats:
        best = max(pats, key=lambda x: x['ev_abs'])
        print(f"  GRP-{grp:3d}min: {len(pats):4d} patterns ({n_long} LONG, {n_short} SHORT) "
              f"| Best: {best['morph_prev']}->{best['morph_current']} "
              f"EV={best['ev_points']:+.1f} WR={best['win_rate']:.1%}")

print(f"\n  Results saved to data/morphology_grouping_results.json")
print(f"{'='*80}")
