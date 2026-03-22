"""
FASE 1 — Carga, Limpieza y Exploración
Pipeline de descubrimiento de morfologías predictivas MNQ futures
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
import os
import json

warnings.filterwarnings('ignore')

DATA_PATH = '/home/user/newrepo22032026/data_mnq.csv'
OUTPUT_DIR = '/home/user/newrepo22032026/outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("FASE 1 — CARGA, LIMPIEZA Y EXPLORACIÓN")
print("=" * 80)

# ============================================================
# 1. CARGA DEL DATASET
# ============================================================
print("\n[1] Cargando dataset completo...")
df = pd.read_csv(DATA_PATH, parse_dates=['ts_event'])

print(f"  Filas totales en CSV: {len(df):,}")
print(f"  Columnas: {list(df.columns)}")
print(f"  Símbolos únicos: {df['symbol'].nunique()} → {sorted(df['symbol'].unique())}")

# Filtrar calendar spreads (contienen '-' en el símbolo, ej. MNQZ1-MNQH2)
# Los spreads tienen precios ~0-20 pts vs ~13000-26000 de outrights
outright_syms = [s for s in df['symbol'].unique() if '-' not in s]
spread_syms = [s for s in df['symbol'].unique() if '-' in s]
print(f"\n  Contratos outright: {len(outright_syms)} → {sorted(outright_syms)}")
print(f"  Calendar spreads (excluidos): {len(spread_syms)}")
df = df[df['symbol'].isin(outright_syms)].copy()
print(f"  Filas después de excluir spreads: {len(df):,}")

# MNQ tiene múltiples contratos activos simultáneamente (front month, back months)
# Necesitamos el contrato más líquido por cada timestamp (continuous contract)
print("\n[1.1] Construyendo contrato continuo (barra más líquida por timestamp)...")

# Para cada timestamp, tomar la barra con mayor volumen (front month)
df_sorted = df.sort_values(['ts_event', 'volume'], ascending=[True, False])
df_cont = df_sorted.drop_duplicates(subset='ts_event', keep='first').copy()
df_cont = df_cont.sort_values('ts_event').reset_index(drop=True)

print(f"  Barras después de dedup por timestamp: {len(df_cont):,}")
print(f"  Contratos usados: {df_cont['symbol'].nunique()}")

# Convertir timezone a US/Eastern para clasificación de sesiones
df_cont['ts_event'] = pd.to_datetime(df_cont['ts_event'], utc=True)
df_cont['ts_et'] = df_cont['ts_event'].dt.tz_convert('US/Eastern')

print(f"\n  Rango de fechas (UTC): {df_cont['ts_event'].min()} → {df_cont['ts_event'].max()}")
print(f"  Rango de fechas (ET):  {df_cont['ts_et'].min()} → {df_cont['ts_et'].max()}")

# Estadísticas básicas OHLCV
print("\n[1.2] Estadísticas descriptivas OHLCV:")
ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
desc = df_cont[ohlcv_cols].describe()
print(desc.to_string())

# ============================================================
# 2. CLASIFICACIÓN DE SESIONES
# ============================================================
print("\n" + "=" * 80)
print("[2] Clasificando sesiones de trading...")

def classify_session(ts_et):
    """
    Asia:       18:00-02:00 ET (domingo-jueves noche)
    Londres:    02:00-09:30 ET
    RTH:        09:30-16:00 ET
    Post-Market: 16:00-18:00 ET
    """
    h = ts_et.hour
    m = ts_et.minute
    t = h * 60 + m  # minutos desde medianoche

    if t >= 18 * 60:        # 18:00-23:59
        return 'Asia'
    elif t < 2 * 60:        # 00:00-01:59
        return 'Asia'
    elif t < 9 * 60 + 30:   # 02:00-09:29
        return 'London'
    elif t < 16 * 60:       # 09:30-15:59
        return 'RTH'
    else:                    # 16:00-17:59
        return 'Post-Market'

df_cont['session'] = df_cont['ts_et'].apply(classify_session)

session_counts = df_cont['session'].value_counts()
print("  Distribución de barras por sesión:")
for s, c in session_counts.items():
    print(f"    {s:12s}: {c:>10,} barras ({100*c/len(df_cont):5.1f}%)")

# ============================================================
# 3. DETECCIÓN DE ANOMALÍAS
# ============================================================
print("\n" + "=" * 80)
print("[3] Detectando anomalías...")

# 3a. Barras con volumen cero
zero_vol = (df_cont['volume'] == 0).sum()
print(f"\n  [3a] Barras con volumen = 0: {zero_vol:,} ({100*zero_vol/len(df_cont):.2f}%)")

# 3b. Gaps de datos (barras faltantes)
# MNQ opera Sun 18:00 ET - Fri 17:00 ET, con pausa 17:00-18:00 ET diaria
time_diffs = df_cont['ts_event'].diff().dt.total_seconds() / 60
expected_1min = (time_diffs == 1.0).sum()
gaps = time_diffs[time_diffs > 1.0]
print(f"\n  [3b] Gaps de datos:")
print(f"    Barras consecutivas (1 min): {expected_1min:,}")
print(f"    Gaps > 1 min: {len(gaps):,}")

# Distribución de gaps
gap_bins = [(1, 5), (5, 15), (15, 60), (60, 120), (120, 1440), (1440, float('inf'))]
print("    Distribución de gaps:")
for lo, hi in gap_bins:
    n = ((gaps > lo) & (gaps <= hi)).sum()
    label = f"{lo}-{hi}" if hi != float('inf') else f">{lo}"
    print(f"      {label:>12s} min: {n:>6,}")

# Los gaps grandes son esperados (fines de semana, feriados, mantenimiento)
big_gaps = gaps[gaps > 120]
if len(big_gaps) > 0:
    print(f"\n    Top 10 gaps más grandes:")
    for idx in big_gaps.nlargest(10).index:
        gap_min = time_diffs.iloc[idx]
        ts_before = df_cont['ts_et'].iloc[idx-1]
        ts_after = df_cont['ts_et'].iloc[idx]
        print(f"      {ts_before} → {ts_after} ({gap_min:.0f} min = {gap_min/60:.1f}h)")

# 3c. Outliers extremos (>5σ) en returns
df_cont['ret_arith'] = df_cont['close'].pct_change()
df_cont['ret_log'] = np.log(df_cont['close'] / df_cont['close'].shift(1))

ret_mean = df_cont['ret_log'].mean()
ret_std = df_cont['ret_log'].std()
outliers_5sigma = df_cont[np.abs(df_cont['ret_log'] - ret_mean) > 5 * ret_std]
print(f"\n  [3c] Outliers >5σ en log-returns: {len(outliers_5sigma):,}")
if len(outliers_5sigma) > 0:
    print("    Top 10 outliers:")
    for _, row in outliers_5sigma.nlargest(10, 'ret_log', keep='first').iterrows():
        print(f"      {row['ts_et']} ret={row['ret_log']:.6f} ({row['ret_log']/ret_std:.1f}σ) "
              f"close={row['close']:.2f} vol={row['volume']}")

# 3d. Barras donde high < low o open fuera de rango
invalid_hl = (df_cont['high'] < df_cont['low']).sum()
invalid_range = ((df_cont['open'] > df_cont['high']) | (df_cont['open'] < df_cont['low'])).sum()
print(f"\n  [3d] Barras inválidas:")
print(f"    high < low: {invalid_hl}")
print(f"    open fuera de [low, high]: {invalid_range}")

# ============================================================
# 4. FEATURES BASE POR BARRA
# ============================================================
print("\n" + "=" * 80)
print("[4] Calculando features base por barra...")

# Returns (ya calculados arriba)
# Rango
df_cont['range_hl'] = df_cont['high'] - df_cont['low']

# Body (|close - open|)
df_cont['body'] = np.abs(df_cont['close'] - df_cont['open'])

# Wicks
df_cont['upper_wick'] = df_cont['high'] - df_cont[['open', 'close']].max(axis=1)
df_cont['lower_wick'] = df_cont[['open', 'close']].min(axis=1) - df_cont['low']

# Gap desde cierre anterior
df_cont['gap'] = df_cont['open'] - df_cont['close'].shift(1)

# Posición del close dentro del rango de la barra
df_cont['close_position'] = np.where(
    df_cont['range_hl'] > 0,
    (df_cont['close'] - df_cont['low']) / df_cont['range_hl'],
    0.5
)

# Volatilidad realizada rolling
for w in [5, 10, 20, 60]:
    df_cont[f'realized_vol_{w}'] = df_cont['ret_log'].rolling(w).std() * np.sqrt(w)

# Volumen relativo vs media de la misma hora
df_cont['hour_et'] = df_cont['ts_et'].dt.hour
hourly_vol_mean = df_cont.groupby('hour_et')['volume'].transform('mean')
df_cont['vol_relative_hour'] = df_cont['volume'] / hourly_vol_mean.replace(0, np.nan)

# Volumen relativo vs media de la misma sesión
session_vol_mean = df_cont.groupby('session')['volume'].transform('mean')
df_cont['vol_relative_session'] = df_cont['volume'] / session_vol_mean.replace(0, np.nan)

# Posición del close dentro del rango de la sesión actual (rolling por sesión)
# Creamos un identificador de sesión única (fecha + sesión)
df_cont['date_et'] = df_cont['ts_et'].dt.date
# Para sesiones que cruzan medianoche (Asia), ajustar fecha
df_cont['session_date'] = df_cont['date_et']
# Asia 18:00+ pertenece al "día" siguiente en términos de sesión
mask_asia_pm = (df_cont['session'] == 'Asia') & (df_cont['ts_et'].dt.hour >= 18)
df_cont.loc[mask_asia_pm, 'session_date'] = df_cont.loc[mask_asia_pm, 'ts_et'].dt.date + pd.Timedelta(days=1)

session_group = df_cont.groupby(['session_date', 'session'])
df_cont['session_high'] = session_group['high'].transform('cummax')
df_cont['session_low'] = session_group['low'].transform('cummin')
df_cont['close_pos_session'] = np.where(
    (df_cont['session_high'] - df_cont['session_low']) > 0,
    (df_cont['close'] - df_cont['session_low']) / (df_cont['session_high'] - df_cont['session_low']),
    0.5
)

# ============================================================
# REPORTE ESTADÍSTICO COMPLETO
# ============================================================
print("\n" + "=" * 80)
print("REPORTE ESTADÍSTICO COMPLETO")
print("=" * 80)

# Returns
print("\n[A] Estadísticas de Returns (log):")
ret_stats = df_cont['ret_log'].dropna()
print(f"  N observaciones: {len(ret_stats):,}")
print(f"  Media:    {ret_stats.mean():.8f}")
print(f"  Std:      {ret_stats.std():.8f}")
print(f"  Skewness: {ret_stats.skew():.4f}")
print(f"  Kurtosis: {ret_stats.kurtosis():.4f} (exceso)")
print(f"  Min:      {ret_stats.min():.6f}")
print(f"  Max:      {ret_stats.max():.6f}")
print(f"  Percentil 1%:  {ret_stats.quantile(0.01):.6f}")
print(f"  Percentil 99%: {ret_stats.quantile(0.99):.6f}")

# Test de normalidad (Jarque-Bera)
jb_stat, jb_pval = stats.jarque_bera(ret_stats.dropna())
print(f"  Jarque-Bera: stat={jb_stat:.2f}, p={jb_pval:.2e}")

print("\n[B] Estadísticas de Returns por sesión:")
for session in ['Asia', 'London', 'RTH', 'Post-Market']:
    s_ret = df_cont[df_cont['session'] == session]['ret_log'].dropna()
    print(f"  {session:12s}: mean={s_ret.mean():.8f}, std={s_ret.std():.8f}, "
          f"skew={s_ret.skew():.4f}, kurt={s_ret.kurtosis():.4f}, n={len(s_ret):,}")

print("\n[C] Distribución de Volumen:")
vol = df_cont['volume']
print(f"  Media:   {vol.mean():.1f}")
print(f"  Mediana: {vol.median():.1f}")
print(f"  Std:     {vol.std():.1f}")
print(f"  Min:     {vol.min()}")
print(f"  Max:     {vol.max()}")
pcts = [10, 25, 50, 75, 90, 95, 99]
for p in pcts:
    print(f"  P{p:>2d}:     {vol.quantile(p/100):.0f}")

print("\n[D] Volumen por sesión:")
for session in ['Asia', 'London', 'RTH', 'Post-Market']:
    s_vol = df_cont[df_cont['session'] == session]['volume']
    print(f"  {session:12s}: mean={s_vol.mean():.1f}, median={s_vol.median():.0f}, "
          f"total={s_vol.sum():,.0f}")

print("\n[E] Features base — estadísticas:")
feature_cols = ['range_hl', 'body', 'upper_wick', 'lower_wick', 'gap',
                'close_position', 'vol_relative_hour', 'vol_relative_session',
                'close_pos_session',
                'realized_vol_5', 'realized_vol_10', 'realized_vol_20', 'realized_vol_60']
feat_desc = df_cont[feature_cols].describe()
print(feat_desc.to_string())

# ============================================================
# RESUMEN DE LIMPIEZA
# ============================================================
print("\n" + "=" * 80)
print("RESUMEN DE LIMPIEZA Y CALIDAD DEL DATASET")
print("=" * 80)

n_total = len(df_cont)
n_nan_ret = df_cont['ret_log'].isna().sum()
n_zero_vol = (df_cont['volume'] == 0).sum()
n_invalid = invalid_hl + invalid_range

print(f"  Total barras (contrato continuo): {n_total:,}")
print(f"  Barras con NaN en returns:        {n_nan_ret:,} (primera barra + gaps)")
print(f"  Barras volumen = 0:               {n_zero_vol:,}")
print(f"  Barras inválidas OHLC:            {n_invalid}")
print(f"  Outliers >5σ:                     {len(outliers_5sigma):,}")
print(f"  Dataset LIMPIO: {n_total - n_nan_ret:,} barras con returns válidos")
print(f"  (Barras con vol=0 y outliers se mantienen — son datos reales del mercado)")

# ============================================================
# VISUALIZACIONES
# ============================================================
print("\n[Generando visualizaciones...]")

fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# 1. Histograma de returns
ax = axes[0, 0]
ax.hist(ret_stats.values, bins=500, density=True, alpha=0.7, color='steelblue', label='Empírica')
x_range = np.linspace(ret_stats.quantile(0.001), ret_stats.quantile(0.999), 200)
ax.plot(x_range, stats.norm.pdf(x_range, ret_stats.mean(), ret_stats.std()),
        'r-', lw=2, label='Normal teórica')
ax.set_title('Distribución Log-Returns (1 min)')
ax.set_xlabel('Log Return')
ax.legend()
ax.set_xlim(ret_stats.quantile(0.001), ret_stats.quantile(0.999))

# 2. QQ plot
ax = axes[0, 1]
stats.probplot(ret_stats.dropna().values[::10], dist="norm", plot=ax)  # subsample for speed
ax.set_title('QQ Plot Log-Returns vs Normal')

# 3. Volumen por hora
ax = axes[0, 2]
hourly_vol = df_cont.groupby('hour_et')['volume'].mean()
colors_hour = []
for h in hourly_vol.index:
    if h >= 18 or h < 2:
        colors_hour.append('purple')
    elif h < 9 or (h == 9 and True):
        colors_hour.append('blue')
    elif h < 16:
        colors_hour.append('green')
    else:
        colors_hour.append('orange')
ax.bar(hourly_vol.index, hourly_vol.values, color=colors_hour, alpha=0.8)
ax.set_title('Volumen promedio por hora (ET)')
ax.set_xlabel('Hora ET')
ax.set_ylabel('Volumen medio')

# 4. Volatilidad por hora
ax = axes[1, 0]
hourly_vol_ret = df_cont.groupby('hour_et')['ret_log'].std()
ax.bar(hourly_vol_ret.index, hourly_vol_ret.values * 100, color=colors_hour, alpha=0.8)
ax.set_title('Volatilidad (std returns) por hora (ET)')
ax.set_xlabel('Hora ET')
ax.set_ylabel('Std Returns (%)')

# 5. Precio close a lo largo del tiempo
ax = axes[1, 1]
# Subsample for plotting
plot_sub = df_cont.iloc[::60]  # cada hora
ax.plot(plot_sub['ts_et'].values, plot_sub['close'].values, lw=0.5, color='steelblue')
ax.set_title('MNQ Close (5 años)')
ax.set_xlabel('Fecha')
ax.set_ylabel('Precio')
ax.tick_params(axis='x', rotation=45)

# 6. Box plot de returns por sesión
ax = axes[1, 2]
sessions_ordered = ['Asia', 'London', 'RTH', 'Post-Market']
data_box = [df_cont[df_cont['session'] == s]['ret_log'].dropna().values for s in sessions_ordered]
# Clip for visualization
data_box_clipped = [np.clip(d, np.percentile(d, 1), np.percentile(d, 99)) for d in data_box]
bp = ax.boxplot(data_box_clipped, labels=sessions_ordered, patch_artist=True)
colors_bp = ['purple', 'blue', 'green', 'orange']
for patch, color in zip(bp['boxes'], colors_bp):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)
ax.set_title('Returns por sesión (P1-P99)')
ax.set_ylabel('Log Return')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fase1_exploracion.png'), dpi=150, bbox_inches='tight')
print(f"  → Guardado: {OUTPUT_DIR}/fase1_exploracion.png")

# ============================================================
# GUARDAR DATASET PROCESADO
# ============================================================
print("\n[Guardando dataset procesado...]")

# Guardar en parquet para eficiencia en fases siguientes
save_cols = ['ts_event', 'ts_et', 'open', 'high', 'low', 'close', 'volume', 'symbol',
             'session', 'ret_arith', 'ret_log', 'range_hl', 'body', 'upper_wick', 'lower_wick',
             'gap', 'close_position', 'hour_et', 'vol_relative_hour', 'vol_relative_session',
             'close_pos_session', 'session_date',
             'realized_vol_5', 'realized_vol_10', 'realized_vol_20', 'realized_vol_60']

# Convert timezone-aware to UTC strings for parquet compatibility
df_save = df_cont[save_cols].copy()
df_save['ts_et'] = df_save['ts_et'].dt.tz_localize(None)  # Remove tz for parquet
df_save.to_parquet(os.path.join(OUTPUT_DIR, 'mnq_clean.parquet'), index=False)
print(f"  → Guardado: {OUTPUT_DIR}/mnq_clean.parquet ({len(df_save):,} barras)")

# Guardar resumen como JSON
summary = {
    'total_barras_csv': len(df),
    'total_barras_continuo': len(df_cont),
    'barras_con_returns': int(n_total - n_nan_ret),
    'rango_fechas': [str(df_cont['ts_event'].min()), str(df_cont['ts_event'].max())],
    'contratos_usados': int(df_cont['symbol'].nunique()),
    'barras_volumen_cero': int(n_zero_vol),
    'outliers_5sigma': len(outliers_5sigma),
    'barras_invalidas': int(n_invalid),
    'returns_log_stats': {
        'mean': float(ret_stats.mean()),
        'std': float(ret_stats.std()),
        'skew': float(ret_stats.skew()),
        'kurtosis': float(ret_stats.kurtosis()),
    },
    'sesiones': {s: int(c) for s, c in session_counts.items()},
}
with open(os.path.join(OUTPUT_DIR, 'fase1_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print(f"  → Guardado: {OUTPUT_DIR}/fase1_summary.json")

print("\n" + "=" * 80)
print("FASE 1 COMPLETADA — Esperando confirmación para continuar a Fase 2")
print("=" * 80)
