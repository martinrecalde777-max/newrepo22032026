#!/usr/bin/env python3
"""
=============================================================================
DEEP DIVE: MORPHOLOGICAL GROUPINGS WITH ENRICHED CONTEXT
=============================================================================
Profundiza en los patrones encontrados:
1. Agrega contexto de TREND MACRO (SMA 200, SMA 50 position)
2. Agrega contexto de SESSION (Asia/Europe/US)
3. Explora cadenas de 3 morfologías (grandparent -> parent -> current)
4. Statistical validation (bootstrap confidence intervals)
5. Relaxed EV threshold to 30pts to show more patterns with context
=============================================================================
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import json
from collections import defaultdict

# ============================================================================
# LOAD DATA
# ============================================================================
print("=" * 100)
print("DEEP DIVE: MORPHOLOGICAL GROUPINGS - ENRICHED CONTEXT")
print("=" * 100)

df = pd.read_parquet('data/mnq_preprocessed.parquet')
print(f"Data: {len(df):,} bars")

close = df['close'].values
high = df['high'].values
low = df['low'].values
opn = df['open'].values
volume = df['volume'].values
direction = df['direction'].values

# ============================================================================
# ADD MACRO CONTEXT
# ============================================================================
print("\nComputing macro context...")

# SMA 200 and SMA 50 (in 1-min bars: 200*390 ≈ 78000 bars for daily, but we'll use
# shorter period proxies that still capture the "big picture")
# We use 3000-bar and 12000-bar MAs as proxies for "medium" and "long" trend
sma_medium = pd.Series(close).rolling(3000).mean().values  # ~2 trading days
sma_long = pd.Series(close).rolling(12000).mean().values   # ~8 trading days

# Macro regime
macro_regime = np.full(len(df), "NEUTRAL", dtype=object)
for i in range(len(df)):
    if np.isnan(sma_medium[i]) or np.isnan(sma_long[i]):
        macro_regime[i] = "UNKNOWN"
    elif close[i] > sma_medium[i] > sma_long[i]:
        macro_regime[i] = "BULL_TREND"
    elif close[i] < sma_medium[i] < sma_long[i]:
        macro_regime[i] = "BEAR_TREND"
    elif close[i] > sma_long[i] and close[i] < sma_medium[i]:
        macro_regime[i] = "PULLBACK_IN_BULL"
    elif close[i] < sma_long[i] and close[i] > sma_medium[i]:
        macro_regime[i] = "BOUNCE_IN_BEAR"
    else:
        macro_regime[i] = "CHOPPY"

# Session classification (UTC times for CME)
hours = df['hour'].values
session = np.full(len(df), "OTHER", dtype=object)
for i in range(len(df)):
    h = hours[i]
    if 0 <= h < 8:       # Asia/Early Europe
        session[i] = "ASIA"
    elif 8 <= h < 13:    # Europe / US premarket
        session[i] = "EUROPE"
    elif 13 <= h < 20:   # US cash session (9:00-16:00 ET ≈ 13:00-20:00 UTC)
        session[i] = "US_CASH"
    else:
        session[i] = "OVERNIGHT"

# Range expansion context: is current ATR expanding or contracting?
atr_20 = pd.Series(df['range'].values).rolling(20).mean().values
atr_100 = pd.Series(df['range'].values).rolling(100).mean().values
atr_regime = np.full(len(df), "NORMAL", dtype=object)
for i in range(len(df)):
    if np.isnan(atr_20[i]) or np.isnan(atr_100[i]) or atr_100[i] == 0:
        continue
    ratio = atr_20[i] / atr_100[i]
    if ratio > 1.5:
        atr_regime[i] = "ATR_EXPANDING"
    elif ratio > 1.15:
        atr_regime[i] = "ATR_SLIGHTLY_HIGH"
    elif ratio < 0.65:
        atr_regime[i] = "ATR_COMPRESSED"
    elif ratio < 0.85:
        atr_regime[i] = "ATR_SLIGHTLY_LOW"


# ============================================================================
# MORPHOLOGY CLASSIFICATION (same as before but faster with numpy)
# ============================================================================

def classify_morphology(net_ret, grp_range, efficiency, close_pos, open_pos,
                        vol_half_ratio, dir_consistency, internal_path_ratio):
    if grp_range == 0 or np.isnan(grp_range):
        return "INDEFINIDO"

    is_bullish = net_ret > 0
    is_bearish = net_ret < 0
    is_flat = abs(net_ret) < grp_range * 0.05

    high_eff = efficiency > 0.65
    med_eff = 0.35 < efficiency <= 0.65
    closes_high = close_pos > 0.75
    closes_low = close_pos < 0.25
    opens_high = open_pos > 0.75
    opens_low = open_pos < 0.25
    vol_front = vol_half_ratio > 1.3
    vol_back = vol_half_ratio < 0.7
    high_consist = dir_consistency > 0.65
    high_zigzag = internal_path_ratio > 3.0

    if is_bullish and high_eff and closes_high and high_consist:
        return "IMPULSO_BULL"
    if is_bullish and high_eff and closes_high:
        return "RALLY_ERRATICO"
    if is_bullish and closes_high and opens_low:
        return "V_BOTTOM"
    if is_bullish and med_eff and vol_back:
        return "ACUMULACION"
    if is_bullish and med_eff and vol_front:
        return "AGOTAMIENTO_BULL"
    if is_bearish and high_eff and closes_low and high_consist:
        return "IMPULSO_BEAR"
    if is_bearish and high_eff and closes_low:
        return "SELL_ERRATICO"
    if is_bearish and closes_low and opens_high:
        return "INV_V_TOP"
    if is_bearish and med_eff and vol_back:
        return "DISTRIBUCION"
    if is_bearish and med_eff and vol_front:
        return "AGOTAMIENTO_BEAR"
    if is_flat and high_zigzag:
        return "CHOP"
    if is_flat:
        return "COMPRESION"
    if is_bullish:
        return "DRIFT_UP"
    if is_bearish:
        return "DRIFT_DOWN"
    return "OTRO"


def classify_volume_regime(vol_ratio):
    if vol_ratio > 2.0:
        return "EXPLOSIVO"
    if vol_ratio > 1.3:
        return "ALTO"
    if vol_ratio > 0.7:
        return "NORMAL"
    return "BAJO"


# ============================================================================
# MAIN ANALYSIS: ENRICHED CONTEXT
# ============================================================================

GROUPING_SIZES = [30, 60, 120, 240]
FUTURE_HORIZONS = [15, 30, 60, 120]
MIN_EV = 50.0
MIN_SAMPLES = 30

all_results = {}

for GRP in GROUPING_SIZES:
    print(f"\n{'='*100}")
    print(f"  GROUPING: {GRP} BARS ({GRP} MINUTES)")
    print(f"{'='*100}")

    # Pre-compute rolling features
    grp_high = pd.Series(high).rolling(GRP).max().values
    grp_low = pd.Series(low).rolling(GRP).min().values
    grp_range = grp_high - grp_low
    grp_net = close - np.roll(close, GRP)
    grp_net[:GRP] = np.nan

    grp_eff = np.where(grp_range > 0, np.abs(grp_net) / grp_range, 0)
    grp_cpos = np.where(grp_range > 0, (close - grp_low) / grp_range, 0.5)

    open_grp = np.roll(opn, GRP - 1)
    open_grp[:GRP] = np.nan
    grp_opos = np.where(grp_range > 0, (open_grp - grp_low) / grp_range, 0.5)

    half = max(1, GRP // 2)
    vol_first = pd.Series(volume).rolling(half).sum().shift(half).values
    vol_second = pd.Series(volume).rolling(half).sum().values
    vol_hr = np.where(vol_second > 0, vol_first / vol_second, 1.0)

    dir_sum = pd.Series(direction).rolling(GRP).sum().values
    dir_consist = np.abs(dir_sum) / GRP

    bar_moves = np.abs(np.diff(close, prepend=close[0]))
    int_path = pd.Series(bar_moves).rolling(GRP).sum().values
    ipr = np.where(grp_range > 0, int_path / grp_range, 1.0)

    grp_vol_total = pd.Series(volume).rolling(GRP).sum().values
    grp_vol_ma = pd.Series(volume).rolling(min(GRP * 5, len(df) // 2)).mean().values * GRP
    grp_vol_ratio = np.where(grp_vol_ma > 0, grp_vol_total / grp_vol_ma, 1.0)

    # Future returns
    fut = {}
    for h in FUTURE_HORIZONS:
        fut[h] = np.roll(close, -h) - close
        fut[h][-h:] = np.nan

    # Classify morphologies
    print("  Classifying...")
    start_idx = GRP * 3  # need 3 groupings back
    end_idx = len(df) - max(FUTURE_HORIZONS)

    morph = np.full(len(df), "", dtype=object)
    for i in range(GRP, len(df)):
        if np.isnan(grp_net[i]):
            morph[i] = "SKIP"
            continue
        morph[i] = classify_morphology(
            grp_net[i], grp_range[i], grp_eff[i], grp_cpos[i], grp_opos[i],
            vol_hr[i], dir_consist[i], ipr[i]
        )

    vol_reg = np.array([classify_volume_regime(grp_vol_ratio[i]) for i in range(len(df))])

    # Previous and grandparent morphologies
    prev_morph = np.roll(morph, GRP)
    prev_morph[:GRP * 2] = "SKIP"
    gp_morph = np.roll(morph, GRP * 2)
    gp_morph[:GRP * 3] = "SKIP"

    # Build analysis dataframe with ALL context
    valid = np.ones(len(df), dtype=bool)
    valid[:start_idx] = False
    valid[end_idx:] = False
    valid &= (morph != "SKIP") & (morph != "")
    valid &= (prev_morph != "SKIP") & (prev_morph != "")
    valid &= (gp_morph != "SKIP") & (gp_morph != "")
    valid &= (macro_regime != "UNKNOWN")
    for h in FUTURE_HORIZONS:
        valid &= ~np.isnan(fut[h])

    idx = np.where(valid)[0]
    print(f"  Valid patterns: {len(idx):,}")

    adf = pd.DataFrame({
        'morph': morph[idx],
        'prev': prev_morph[idx],
        'grandparent': gp_morph[idx],
        'vol_regime': vol_reg[idx],
        'macro': macro_regime[idx],
        'session': session[idx],
        'atr_regime': atr_regime[idx],
        'hour': hours[idx],
        'grp_range': grp_range[idx],
        'grp_net': grp_net[idx],
    })
    for h in FUTURE_HORIZONS:
        adf[f'f{h}'] = fut[h][idx]

    # ====================================================================
    # ANALYSIS 1: Morph + Prev + Vol Regime + Macro (4-dimensional context)
    # ====================================================================
    print(f"\n  --- 4D CONTEXT: prev_morph + morph + vol_regime + macro ---")
    print(f"  Filtering EV >= {MIN_EV} pts, N >= {MIN_SAMPLES}")

    patterns = []
    for (prev, curr, vol, macro), grp in adf.groupby(['prev', 'morph', 'vol_regime', 'macro']):
        if len(grp) < MIN_SAMPLES:
            continue
        for h in FUTURE_HORIZONS:
            vals = grp[f'f{h}']
            ev = vals.mean()
            if abs(ev) < MIN_EV:
                continue

            std_val = vals.std()
            direction_str = "LONG" if ev > 0 else "SHORT"
            wr = (vals > 0).mean() if ev > 0 else (vals < 0).mean()

            # Bootstrap 95% CI
            rng = np.random.RandomState(42)
            boot_means = [vals.sample(n=len(vals), replace=True, random_state=rng).mean() for _ in range(500)]
            ci_low = np.percentile(boot_means, 2.5)
            ci_high = np.percentile(boot_means, 97.5)
            # Is the CI entirely on one side of zero?
            ci_significant = (ci_low > 0 and ev > 0) or (ci_high < 0 and ev < 0)

            patterns.append({
                'context': f"{prev} -> {curr}",
                'vol': vol,
                'macro': macro,
                'horizon': h,
                'dir': direction_str,
                'ev': round(ev, 1),
                'median': round(vals.median(), 1),
                'std': round(std_val, 1),
                'sharpe': round(ev / std_val, 3) if std_val > 0 else 0,
                'wr': round(wr, 3),
                'n': len(grp),
                'ci_low': round(ci_low, 1),
                'ci_high': round(ci_high, 1),
                'ci_sig': ci_significant,
                'avg_hour': round(grp['hour'].mean(), 1),
                'session_mode': grp['session'].mode().iloc[0] if len(grp) > 0 else "",
            })

    patterns.sort(key=lambda x: abs(x['ev']), reverse=True)

    # Print the STATISTICALLY SIGNIFICANT ones first
    sig_patterns = [p for p in patterns if p['ci_sig']]
    nonsig_patterns = [p for p in patterns if not p['ci_sig']]

    print(f"\n  STATISTICALLY SIGNIFICANT (95% CI doesn't cross zero): {len(sig_patterns)}")
    print(f"  {'─'*120}")

    for i, p in enumerate(sig_patterns[:30]):
        print(f"  #{i+1:2d} {p['context']:50s} [{p['vol']:10s}] [{p['macro']:18s}] "
              f"{p['dir']:5s} {p['horizon']:3d}min "
              f"EV={p['ev']:+7.1f} med={p['median']:+7.1f} "
              f"WR={p['wr']:.1%} N={p['n']:5,} "
              f"Sharpe={p['sharpe']:+.3f} "
              f"CI=[{p['ci_low']:+.1f}, {p['ci_high']:+.1f}] "
              f"@{p['avg_hour']:.0f}h {p['session_mode']}")

    # ====================================================================
    # ANALYSIS 2: 3-LEVEL CHAINS (grandparent -> parent -> current)
    # ====================================================================
    print(f"\n  --- 3-LEVEL CHAINS: grandparent -> parent -> current ---")

    chain_patterns = []
    for (gp, prev, curr), grp in adf.groupby(['grandparent', 'prev', 'morph']):
        if len(grp) < MIN_SAMPLES:
            continue
        for h in FUTURE_HORIZONS:
            vals = grp[f'f{h}']
            ev = vals.mean()
            if abs(ev) < MIN_EV:
                continue
            std_val = vals.std()
            wr = (vals > 0).mean() if ev > 0 else (vals < 0).mean()
            direction_str = "LONG" if ev > 0 else "SHORT"

            chain_patterns.append({
                'chain': f"{gp} -> {prev} -> {curr}",
                'horizon': h,
                'dir': direction_str,
                'ev': round(ev, 1),
                'wr': round(wr, 3),
                'n': len(grp),
                'sharpe': round(ev / std_val, 3) if std_val > 0 else 0,
            })

    chain_patterns.sort(key=lambda x: abs(x['ev']), reverse=True)

    print(f"  Found {len(chain_patterns)} chain patterns")
    for i, p in enumerate(chain_patterns[:25]):
        print(f"  #{i+1:2d} {p['chain']:75s} "
              f"{p['dir']:5s} {p['horizon']:3d}min "
              f"EV={p['ev']:+7.1f} WR={p['wr']:.1%} N={p['n']:5,} Sh={p['sharpe']:+.3f}")

    # ====================================================================
    # ANALYSIS 3: SESSION-SPECIFIC PATTERNS
    # ====================================================================
    print(f"\n  --- SESSION-SPECIFIC PATTERNS ---")

    for sess in ["US_CASH", "EUROPE", "ASIA"]:
        sess_df = adf[adf['session'] == sess]
        if len(sess_df) < 100:
            continue

        sess_patterns = []
        for (prev, curr), grp in sess_df.groupby(['prev', 'morph']):
            if len(grp) < MIN_SAMPLES:
                continue
            for h in FUTURE_HORIZONS:
                vals = grp[f'f{h}']
                ev = vals.mean()
                if abs(ev) < MIN_EV:
                    continue
                std_val = vals.std()
                wr = (vals > 0).mean() if ev > 0 else (vals < 0).mean()
                sess_patterns.append({
                    'transition': f"{prev} -> {curr}",
                    'horizon': h,
                    'dir': "LONG" if ev > 0 else "SHORT",
                    'ev': round(ev, 1),
                    'wr': round(wr, 3),
                    'n': len(grp),
                    'sharpe': round(ev / std_val, 3) if std_val > 0 else 0,
                })

        sess_patterns.sort(key=lambda x: abs(x['ev']), reverse=True)
        print(f"\n  {sess} session ({len(sess_df):,} bars): {len(sess_patterns)} patterns with EV>={MIN_EV}")
        for i, p in enumerate(sess_patterns[:10]):
            print(f"    #{i+1:2d} {p['transition']:50s} "
                  f"{p['dir']:5s} {p['horizon']:3d}min "
                  f"EV={p['ev']:+7.1f} WR={p['wr']:.1%} N={p['n']:5,}")

    all_results[GRP] = {
        '4d_context': patterns,
        'chains': chain_patterns,
    }

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print(f"\n\n{'='*100}")
print("  FINAL SUMMARY: ALL SIGNIFICANT PATTERNS (EV >= 50, CI significant)")
print(f"{'='*100}")

all_sig = []
for grp, res in all_results.items():
    for p in res['4d_context']:
        if p['ci_sig']:
            p['grp_size'] = grp
            all_sig.append(p)

all_sig.sort(key=lambda x: abs(x['ev']), reverse=True)

print(f"\n  Total statistically significant patterns: {len(all_sig)}")
print(f"  {'─'*130}")

for i, p in enumerate(all_sig):
    print(f"  #{i+1:2d} GRP-{p['grp_size']:3d} | {p['context']:50s} [{p['vol']:10s}] [{p['macro']:18s}] "
          f"{p['dir']:5s} {p['horizon']:3d}min "
          f"EV={p['ev']:+7.1f} WR={p['wr']:.1%} N={p['n']:5,} "
          f"Sharpe={p['sharpe']:+.3f} CI=[{p['ci_low']:+.1f},{p['ci_high']:+.1f}]")

# Save everything
output = {
    'significant_patterns': all_sig,
    'total_by_grouping': {str(k): {
        'n_4d_patterns': len(v['4d_context']),
        'n_sig_4d': len([p for p in v['4d_context'] if p.get('ci_sig', False)]),
        'n_chains': len(v['chains']),
    } for k, v in all_results.items()},
}

with open('data/morphology_deep_dive_results.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\n  Results saved to data/morphology_deep_dive_results.json")
