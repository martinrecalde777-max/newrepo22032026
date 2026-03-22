# REPORTE DE RESULTADOS: Sistema de Trading Cuantitativo MNQ
## Análisis Completo con 1,769,455 Barras Reales de 1 Minuto (2021-2026)

---

## 1. RESUMEN EJECUTIVO

Se analizaron **1,769,455 barras de 1 minuto** del Micro Nasdaq Futures (MNQ) cubriendo 5 años completos (marzo 2021 - marzo 2026). El análisis empleó:

- **Machine Learning**: XGBoost, LightGBM (ensemble)
- **Lógica Bayesiana**: Inferencia condicional sobre morfologías
- **Cadenas de Markov**: Transiciones de estado discreto y HMM para regímenes
- **Teoría del Caos**: Exponente de Hurst, entropía aproximada
- **Teoría del Juego**: Equilibrio de Nash para decisión de trading
- **145 features** generados desde 8 agrupaciones de barras óptimas
- **Walk-forward validation** riguroso (múltiples ventanas out-of-sample)
- **Monte Carlo** simulation para intervalos de confianza

### Conclusión Principal

> **El mercado MNQ a escala de 1 minuto es extraordinariamente eficiente.** Los modelos ML detectan un sesgo direccional real pero pequeño (~53% accuracy en horizontes de 30-120 barras), que resulta insuficiente para generar alpha neto consistente después de costos de transacción en walk-forward validation. Sin embargo, se identificaron condiciones específicas donde existe ventaja marginal, y el sistema completo de análisis tiene valor significativo como herramienta de decisión.

---

## 2. DATOS UTILIZADOS

| Parámetro | Valor |
|-----------|-------|
| Fuente | Databento CME MDP3 |
| Instrumento | MNQ (Micro Nasdaq E-mini) |
| Periodo | 2021-03-12 a 2026-03-11 |
| Timeframe | 1 minuto |
| Barras totales (raw) | 2,823,464 |
| Barras (serie continua front-month) | 1,769,455 |
| Rango de precio | 10,495.75 - 26,396.25 |
| Volumen medio por barra | 925 contratos |

**Construcción de serie continua**: Se seleccionó el contrato front-month (mayor volumen diario) para cada día de trading, con 21 rollovers trimestrales (HMUZ).

---

## 3. DESCUBRIMIENTO DE AGRUPACIONES ÓPTIMAS DE BARRAS

Se evaluaron 16 tamaños candidatos usando:
- **Mutual Information** (MI): Información compartida entre features del grupo y retorno futuro
- **Autocorrelación**: Persistencia del retorno agrupado
- **Poder Predictivo**: Correlación lineal features-target

### Resultados (Target: retorno a 60 barras)

| Tamaño | MI | AutoCorr | Pred.Power | Composite | Rank |
|--------|-----|----------|------------|-----------|------|
| 90 | 0.06391 | 0.98898 | 0.00724 | **0.32443** | #1 |
| 60 | 0.06831 | 0.98335 | 0.00537 | **0.32394** | #2 |
| 120 | 0.05931 | 0.99176 | 0.00870 | **0.32386** | #3 |
| 180 | 0.05468 | 0.99454 | 0.00942 | **0.32306** | #4 |
| 45 | 0.06896 | 0.97767 | 0.00524 | **0.32246** | #5 |
| 240 | 0.04881 | 0.99591 | 0.00987 | **0.32126** | #6 |
| 30 | 0.07068 | 0.96635 | 0.00530 | **0.31977** | #7 |
| 25 | 0.07261 | 0.95967 | 0.00530 | **0.31853** | #8 |

### Interpretación

Las agrupaciones de **90 y 60 barras** (1.5 horas y 1 hora) son las más informativas. La autocorrelación crece monotónicamente con el tamaño del grupo (esperado), pero la MI muestra un pico en tamaños medianos (25-30 barras), lo que sugiere que la **información de corto plazo se pierde al agregar demasiado**. Se seleccionaron las 8 mejores agrupaciones: **[25, 30, 45, 60, 90, 120, 180, 240]**.

---

## 4. FEATURES GENERADOS (145 TOTAL)

### 4.1 Features por grupo (15 features × 8 grupos = 120)
Para cada agrupación se computaron:
- `range`: Rango total del grupo (High máximo - Low mínimo)
- `range_pct`: Rango normalizado por precio medio
- `return` / `return_pct`: Retorno del grupo
- `body`: Cierre actual - Apertura primera barra del grupo
- `body_range_ratio`: Eficiencia del cuerpo respecto al rango
- `volatility`: Desviación estándar de log-returns dentro del grupo
- `vol_change`: Cambio de volatilidad respecto al grupo anterior
- `vol_total` / `vol_mean` / `vol_trend`: Perfil de volumen
- `momentum`: Precio vs media móvil del grupo
- `efficiency`: |movimiento neto| / camino total recorrido
- `up_pct`: Porcentaje de barras alcistas
- `close_pos`: Posición del cierre dentro del rango del grupo [0-1]

### 4.2 Features cross-group (10)
- Ratio de volatilidad corto/largo plazo
- Alineación de momentum entre escalas consecutivas
- Eficiencia media y dispersión cross-group

### 4.3 Features base (7)
- Volatilidad (20 y 60 barras), ratio de volumen, hora, día de semana, momentum 5 y 20

### 4.4 Morfologías codificadas (8)
- Encoding numérico de clasificación morfológica por cada grupo

### Top 10 Features por Mutual Information

| # | Feature | MI Score |
|---|---------|----------|
| 1 | grp_240_range_pct | 0.04102 |
| 2 | grp_120_range_pct | 0.03596 |
| 3 | grp_180_range_pct | 0.03298 |
| 4 | grp_90_range_pct | 0.02898 |
| 5 | grp_60_range_pct | 0.02263 |
| 6 | grp_45_range_pct | 0.02180 |
| 7 | grp_30_range_pct | 0.01484 |
| 8 | grp_25_range_pct | 0.01102 |
| 9 | mom_align_60_90 | 0.00803 |
| 10 | mom_align_180_240 | 0.00648 |

**Hallazgo clave**: Los features de `range_pct` (volatilidad realizada normalizada) dominan completamente. Esto indica que **la volatilidad es el factor predictivo más importante**, no la dirección. Las alineaciones de momentum cross-escala son el segundo grupo más informativo.

---

## 5. CLASIFICACIÓN MORFOLÓGICA DE GRUPOS

Se clasificaron las barras en 8 patrones morfológicos:

### Distribución de Morfologías (grupo de 60 barras)

| Patrón | Frecuencia | % |
|--------|-----------|---|
| Neutral | 851,412 | 48.2% |
| Trend-Up | 233,652 | 13.2% |
| Trend-Down | 180,120 | 10.2% |
| V-Bottom | 121,221 | 6.9% |
| Consolidation | 118,852 | 6.7% |
| Inverted-V | 91,564 | 5.2% |
| Breakout-Down | 87,120 | 4.9% |
| Breakout-Up | 85,514 | 4.8% |

### Análisis de Edge por Morfología

Se evaluaron las 175 combinaciones morfología × horizonte. **Ninguna morfología individual produce expected value positivo neto después de costos de transacción ($3.04/trade)**. El mejor setup individual genera EV = -2.0 pts/trade.

### Combinaciones de Morfologías (Confluencia Bayesiana)

Se encontraron 57 combinaciones de 2 morfologías con EV > 5 pts, pero con muestras insuficientes (200-400) para significancia estadística robusta. Ejemplo:

| Combo | Horizonte | Dirección | n | EV/trade |
|-------|-----------|-----------|---|----------|
| morph_30=Breakout-Down + morph_60=Trend-Up | 120 | SHORT | 242 | 8.5 pts |
| morph_60=Trend-Down + morph_90=Trend-Up | 90 | LONG | 256 | 7.9 pts |
| morph_120=Breakout-Down + morph_240=Trend-Up | 120 | LONG | 406 | 6.9 pts |

> **Advertencia**: Estas combinaciones tienen muestras pequeñas y alto riesgo de overfitting. No son recomendables como señales autónomas.

---

## 6. MODELOS PREDICTIVOS

### 6.1 Ensemble XGBoost + LightGBM

| Modelo | Accuracy (test) | RMSE | Dir. Accuracy |
|--------|----------------|------|---------------|
| XGBoost Direction | 53.07% | - | 53.07% |
| LightGBM Direction | 52.86% | - | 52.86% |
| XGBoost Return | - | 54.50 pts | 52.58% |
| LightGBM Return | - | 54.50 pts | 52.89% |
| **Ensemble Direction** | **53.04%** | - | - |
| Magnitude (>=50pts) | **81.82%** | - | - |

### 6.2 Análisis Multi-Horizonte

| Horizonte (min) | Accuracy | Mejor PnL (in-sample) | EV/trade | Big Move Rate |
|-----------------|----------|----------------------|----------|---------------|
| 15 | 50.96% | 67,908 pts | 0.3 pts | 6.0% |
| 30 | 51.78% | 109,623 pts | 0.4 pts | 11.6% |
| 60 | 53.02% | 253,757 pts | 1.1 pts | 20.1% |
| 90 | 53.74% | 264,522 pts | 1.0 pts | 26.5% |
| 120 | 54.20% | 376,389 pts | 1.4 pts | 32.2% |
| 180 | 54.69% | 540,425 pts | 2.0 pts | 40.9% |
| 240 | 54.90% | 725,833 pts | 2.7 pts | 47.7% |

**Observación**: La accuracy aumenta con el horizonte (más tiempo = más tendencia), pero el EV/trade sigue siendo bajo (1-3 pts) versus los costos ($3.04/trade).

### 6.3 Importancia de Features (Ensemble)

| Feature | Importancia Normalizada |
|---------|----------------------|
| grp_120_volatility | 0.0392 |
| grp_240_vol_change | 0.0334 |
| grp_240_body | 0.0329 |
| grp_60_range_pct | 0.0284 |
| grp_90_momentum | 0.0268 |
| grp_240_up_pct | 0.0266 |
| grp_240_range | 0.0220 |
| hour | 0.0214 |
| grp_120_vol_total | 0.0213 |
| grp_180_vol_change | 0.0212 |

---

## 7. BACKTESTING CON GESTIÓN DE RIESGO

### 7.1 Búsqueda Exhaustiva de Estrategias (27 configuraciones)

Se probaron 3 horizontes × 3 selectividades × 3 configuraciones de stops:

#### Top 5 por Total PnL (in-sample, 30% test set)

| Config | PnL | Trades | Win Rate | EV/trade | PF | Sharpe |
|--------|-----|--------|----------|----------|-----|--------|
| H=30 S=2.0% SL=3×ATR TP=5×ATR | 3,808 pts | 1,136 | 47.7% | 3.4 | 1.17 | 0.84 |
| H=30 S=0.5% SL=2×ATR TP=4×ATR | 2,608 pts | 333 | 46.2% | 7.8 | 1.37 | 1.65 |
| H=30 S=0.5% SL=3×ATR TP=5×ATR | 2,480 pts | 289 | 51.9% | 8.6 | **1.38** | 1.65 |
| H=30 S=1.0% SL=3×ATR TP=5×ATR | 2,421 pts | 613 | 49.9% | 3.9 | 1.17 | 0.82 |
| H=30 S=1.0% SL=1.5×ATR TP=3×ATR | 2,291 pts | 860 | 41.2% | 2.7 | 1.15 | 0.74 |

### 7.2 Walk-Forward Validation (Gold Standard)

La mejor configuración in-sample (H=30, S=2%, SL=3×ATR, TP=5×ATR):

| Ventana | PnL | Trades | Win Rate | EV/trade |
|---------|-----|--------|----------|----------|
| WF 0 | -2,320 | 629 | 33.9% | -3.7 |
| WF 1 | -1,273 | 595 | 38.3% | -2.1 |
| WF 2 | -1,098 | 597 | 40.0% | -1.8 |
| **WF 3** | **+1,577** | 476 | 45.4% | **+3.3** |
| **WF 4** | **+444** | 527 | 46.5% | **+0.8** |

**Total Walk-Forward: -2,670 pts | Win Rate: 40.4% | Profit Factor: 0.93**

### 7.3 Monte Carlo (2,000 simulaciones sobre trades walk-forward)

| Métrica | Valor |
|---------|-------|
| PnL Medio | -2,598 pts |
| PnL 5to percentil | -6,618 pts |
| PnL 95to percentil | +1,336 pts |
| P(profitable) | 13.4% |

---

## 8. ANÁLISIS POR RÉGIMEN

### 8.1 Hora del Día (target a 30 barras)

| Sesión | Barras | Up Rate | Volatilidad | Big Move Rate |
|--------|--------|---------|-------------|---------------|
| PreMarket (0-9) | 697,292 | 51.2% | 19.5 pts | 2.7% |
| Open (9-10) | 77,460 | 50.3% | 21.6 pts | 3.2% |
| Morning (10-12) | 154,920 | 51.7% | 22.1 pts | 3.3% |
| Lunch (12-14) | 154,859 | 50.6% | 47.6 pts | 17.9% |
| **Afternoon (14-16)** | **154,710** | **51.9%** | **54.0 pts** | **26.2%** |
| AfterHours (16-24) | 530,184 | 51.6% | 38.8 pts | 11.3% |

**Hallazgo**: La sesión de la tarde (14-16 ET) tiene el mayor sesgo alcista (51.9%) y la mayor volatilidad con 26.2% de probabilidad de movimiento >=50 pts.

### 8.2 Régimen de Volatilidad

| Quintil | Barras | Up Rate | Mean Return | Volatilidad | Big Move |
|---------|--------|---------|-------------|-------------|----------|
| Q1 (más baja) | 353,887 | 50.5% | +0.21 | 13.4 pts | 0.7% |
| Q2 | 353,887 | 51.3% | +0.24 | 19.5 pts | 2.0% |
| Q3 | 353,887 | 51.3% | -0.05 | 26.1 pts | 4.7% |
| **Q4** | **353,861** | **52.1%** | **+0.25** | **33.8 pts** | **9.8%** |
| Q5 (más alta) | 353,883 | 51.5% | +0.34 | 56.6 pts | 26.4% |

**Hallazgo**: La volatilidad alta (Q4-Q5) combina mayor sesgo alcista con movimientos más grandes, pero el ruido también es mayor.

### 8.3 Filtrado Combinado RTH + Régimen Vol (Walk-Forward)

| Filtro | Select. | PnL | Trades | WR | EV | PF |
|--------|---------|-----|--------|-----|-----|-----|
| All regimes | 0.5% | -1,468 | 263 | 40.3% | -5.6 | 0.72 |
| **High vol (Q4-5)** | **1.0%** | **+275** | **291** | **47.4%** | **+0.9** | **1.05** |
| Mid vol (Q2-4) | 0.5% | -292 | 172 | 42.4% | -1.7 | 0.82 |
| Low vol (Q1-2) | 0.5% | -355 | 104 | 36.5% | -3.4 | 0.55 |

**Único escenario marginalmente positivo**: Alta volatilidad + 1% selectividad, pero PF=1.05 es insuficiente para trading real.

---

## 9. DIAGNÓSTICO PROFUNDO: POR QUÉ EL EDGE ES PEQUEÑO

### 9.1 Desde la Teoría del Caos
El MNQ opera en un régimen cercano al **random walk** en escala de 1 minuto:
- Las autocorrelaciones de retorno son extremadamente pequeñas
- La entropía del mercado es alta (baja predictibilidad)
- El exponente de Hurst oscila alrededor de 0.5

### 9.2 Desde la Teoría del Juego
El mercado NQ es uno de los más líquidos del mundo:
- Los participantes sofisticados (HFT, market makers) arbitran ineficiencias a velocidad de microsegundos
- Cualquier patrón explotable a escala de minutos es rápidamente arbitrado
- El equilibrio de Nash del juego trader-mercado converge a no-trade (costos > edge)

### 9.3 Desde la Lógica Bayesiana
- Las distribuciones posteriores con evidencia morfológica se mantienen muy cerca de la prior
- La likelihood ratio (señal/ruido) es insuficiente para mover la posterior significativamente
- La acumulación de múltiples fuentes de evidencia (naive Bayes) no logra superar el umbral de trading

### 9.4 Desde Cadenas de Markov
- Las transiciones de estado son casi uniformes (baja estructura)
- Los regímenes detectados por HMM tienen distribuciones de retorno similares
- La memoria del proceso es muy corta

---

## 10. HALLAZGOS POSITIVOS Y VALOR DEL SISTEMA

A pesar de la ausencia de alpha neto en modo automático, el sistema aporta:

### 10.1 Como Herramienta de Análisis
1. **Detector de régimen de volatilidad**: 81.8% accuracy para predecir movimientos >=50 pts
2. **Perfil hora del día**: La sesión 14-16 ET concentra las mejores oportunidades
3. **Multi-escala**: Los features de rango porcentual en múltiples escalas dan lectura del "estado del mercado"
4. **Morfologías**: Clasificación en tiempo real del tipo de acción del precio

### 10.2 Como Filtro de Trading Discrecional
1. El modelo puede señalar CUÁNDO el mercado es más probable que se mueva (magnitude classifier)
2. Las alineaciones de momentum cross-escala (`mom_align_60_90`, `mom_align_180_240`) son señales de confluencia
3. Los regímenes de volatilidad ayudan a calibrar stops y targets

### 10.3 Edge Marginal Identificado
- **Horizonte óptimo**: 30 barras (30 minutos)
- **Configuración óptima**: SL=3×ATR, TP=5×ATR
- **Selectividad**: 0.5% (solo las señales más extremas)
- **In-sample**: PF=1.38, Sharpe=1.65 (289 trades)
- **Mejor contexto**: Alta volatilidad (Q4-Q5) + RTH

---

## 11. RECOMENDACIONES

### Para Trading Automático
1. **No se recomienda** el trading automático puro con estos modelos en MNQ 1-minuto
2. El edge es demasiado delgado para superar costos + slippage + desviaciones de ejecución
3. Se necesitaría investigar: datos de book de órdenes (Level 2), tick data, o instrumentos menos eficientes

### Para Trading Asistido
1. Usar el modelo de magnitud como filtro: "¿Es probable un movimiento de 50+ puntos?"
2. Usar morfologías y alineaciones cross-escala como confirmación
3. Usar régimen de volatilidad para calibrar gestión de riesgo
4. Concentrar trading en la sesión 14-16 ET (mayor oportunidad)

### Para Investigación Futura
1. **Datos de mayor resolución** (tick data, book depth)
2. **Variables exógenas** (VIX, yields, correlaciones cross-asset)
3. **Deep learning temporal** (LSTM, Transformers sobre secuencias de features)
4. **Ejecución adaptativa** (trailing stops dinámicos basados en régimen)
5. **Instrumentos menos eficientes** (small caps, commodities exóticas)

---

## 12. ESPECIFICACIONES TÉCNICAS DEL SISTEMA

### Arquitectura
```
config.py               - Configuración global
core/
  data_loader.py        - Ingesta y preprocesamiento OHLCV
  feature_engineering.py - Agrupaciones, morfologías, features caos
models/
  bayesian_model.py     - Inferencia bayesiana
  markov_model.py       - Cadenas de Markov + HMM
  ensemble_model.py     - XGBoost/LightGBM + Teoría del Juego
backtest/
  engine.py             - Motor de backtesting con gestión de riesgo
bot/
  trading_core.py       - Núcleo para trading en tiempo real
  signal_engine.py      - Motor de señales
main.py                 - Orquestador principal
```

### Dependencias
```
pandas, numpy, scipy, scikit-learn, xgboost, lightgbm,
hmmlearn, statsmodels, matplotlib, seaborn, joblib, pyarrow
```

### Ejecución
```bash
python main.py --data /path/to/mnq_data.parquet
```

---

*Reporte generado el 2026-03-22*
*Sistema: MNQ Quantitative Trading System v1.0*
*Datos: 1,769,455 barras reales de 1 minuto (2021-2026)*
*Todos los resultados son con datos reales, sin simulación alguna.*
