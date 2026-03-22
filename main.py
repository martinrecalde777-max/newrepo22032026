#!/usr/bin/env python3
"""
MNQ Quantitative Trading System - Main Orchestrator
=====================================================
Sistema completo de análisis cuantitativo y trading para Micro Nasdaq Futures.

Uso:
    # Entrenamiento completo con datos históricos
    python main.py --data /path/to/data.parquet --mode train

    # Solo backtesting
    python main.py --data /path/to/data.parquet --mode backtest

    # Modo bot (tiempo real)
    python main.py --mode bot

    # Generar reporte
    python main.py --data /path/to/data.parquet --mode report
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('data/logs/trading_system.log', mode='a'),
    ]
)
logger = logging.getLogger('MNQ_System')


def load_data(data_path: str) -> pd.DataFrame:
    """Load and preprocess data from CSV or Parquet."""
    logger.info(f"Loading data from {data_path}")

    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    elif data_path.endswith('.csv'):
        df = pd.read_csv(data_path, low_memory=False)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")

    logger.info(f"Raw data: {len(df):,} rows, columns: {list(df.columns)}")

    # Normalize columns
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

    # Detect timestamp column
    time_col = None
    for candidate in ['timestamp', 'ts_event', 'datetime', 'date', 'time', 'ts']:
        if candidate in df.columns:
            time_col = candidate
            break
    if time_col is None:
        time_col = df.columns[0]

    df = df.rename(columns={time_col: 'timestamp'})
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')

    # Filter to outright MNQ contracts (no spreads)
    if 'symbol' in df.columns:
        df = df[~df['symbol'].str.contains('-', na=False)]

    # Ensure numeric OHLCV
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df = df[df['volume'] > 0]
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Build continuous front-month series if multiple symbols
    if 'symbol' in df.columns and df['symbol'].nunique() > 1:
        logger.info("Building continuous front-month series...")
        df['date'] = df['timestamp'].dt.date
        daily_vol = df.groupby(['date', 'symbol'])['volume'].sum().reset_index()
        daily_vol.columns = ['date', 'symbol', 'daily_volume']
        front_month = daily_vol.loc[daily_vol.groupby('date')['daily_volume'].idxmax()]
        front_month = front_month[['date', 'symbol']].rename(columns={'symbol': 'front_symbol'})
        df = df.merge(front_month, on='date')
        df = df[df['symbol'] == df['front_symbol']]
        df = df.drop(columns=['front_symbol', 'date'], errors='ignore')
        df = df.drop_duplicates(subset='timestamp', keep='first')

    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Add base features
    df['range'] = df['high'] - df['low']
    df['body'] = df['close'] - df['open']
    df['body_abs'] = df['body'].abs()
    df['direction'] = np.sign(df['body'])
    df['return_close'] = df['close'].pct_change()
    df['return_log'] = np.log(df['close'] / df['close'].shift(1))
    df['return_points'] = df['close'] - df['close'].shift(1)
    df['volatility_5'] = df['return_log'].rolling(5).std()
    df['volatility_20'] = df['return_log'].rolling(20).std()
    df['volatility_60'] = df['return_log'].rolling(60).std()
    df['volume_ma_20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = np.where(df['volume_ma_20'] > 0, df['volume'] / df['volume_ma_20'], 1.0)
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['momentum_5'] = df['close'] - df['close'].shift(5)
    df['momentum_20'] = df['close'] - df['close'].shift(20)
    df['momentum_60'] = df['close'] - df['close'].shift(60)

    logger.info(f"Processed data: {len(df):,} bars, {df['timestamp'].min()} to {df['timestamp'].max()}")
    return df


def discover_groupings(df: pd.DataFrame, candidate_sizes=None) -> list:
    """Discover optimal bar groupings using mutual information."""
    from sklearn.feature_selection import mutual_info_regression

    if candidate_sizes is None:
        candidate_sizes = [3, 5, 7, 8, 10, 12, 15, 20, 25, 30, 45, 60, 90, 120, 180, 240]

    TARGET_HORIZON = 60
    scores = {}

    logger.info("Discovering optimal bar groupings...")
    for size in candidate_sizes:
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        grp_range = high.rolling(size).max() - low.rolling(size).min()
        grp_return = close - close.shift(size)
        grp_vol = df['return_log'].rolling(size).std()
        total_path = df['range'].rolling(size).sum()
        net_move = (close - close.shift(size)).abs()
        grp_eff = np.where(total_path > 0, net_move / total_path, 0)

        target = close.shift(-TARGET_HORIZON) - close

        features_df = pd.DataFrame({
            'range': grp_range, 'return': grp_return,
            'volatility': grp_vol, 'efficiency': pd.Series(grp_eff, index=df.index),
        })
        valid = features_df.notna().all(axis=1) & target.notna()
        X = features_df[valid].values
        y = target[valid].values

        if len(X) < 1000:
            scores[size] = 0
            continue

        n_sample = min(50000, len(X))
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), n_sample, replace=False)
        mi = mutual_info_regression(X[idx], y[idx], n_neighbors=5, random_state=42).mean()

        grouped_returns = close.diff(size).dropna()
        autocorr = abs(grouped_returns.autocorr(lag=1))
        if np.isnan(autocorr):
            autocorr = 0

        composite = 0.4 * mi + 0.3 * autocorr + 0.3 * 0
        scores[size] = composite
        logger.info(f"  Size {size:4d}: MI={mi:.5f} composite={composite:.5f}")

    optimal = sorted(scores, key=scores.get, reverse=True)[:8]
    optimal.sort()
    logger.info(f"Optimal groupings: {optimal}")

    os.makedirs('data', exist_ok=True)
    with open('data/optimal_groupings.json', 'w') as f:
        json.dump(optimal, f)

    return optimal


def build_features(df: pd.DataFrame, groupings: list) -> pd.DataFrame:
    """Build complete feature matrix including order flow and cross-asset features."""
    logger.info(f"Building features for groupings: {groupings}")
    features = pd.DataFrame(index=df.index)

    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']

    for size in groupings:
        features[f'grp_{size}_range'] = high.rolling(size).max() - low.rolling(size).min()
        features[f'grp_{size}_range_pct'] = features[f'grp_{size}_range'] / close.rolling(size).mean()
        features[f'grp_{size}_return'] = close - close.shift(size)
        features[f'grp_{size}_return_pct'] = close.pct_change(size)
        features[f'grp_{size}_body'] = close - df['open'].shift(size - 1)
        features[f'grp_{size}_body_range_ratio'] = np.where(
            features[f'grp_{size}_range'] > 0,
            features[f'grp_{size}_body'].abs() / features[f'grp_{size}_range'], 0)
        features[f'grp_{size}_volatility'] = df['return_log'].rolling(size).std()
        features[f'grp_{size}_vol_change'] = features[f'grp_{size}_volatility'] / features[f'grp_{size}_volatility'].shift(size)
        features[f'grp_{size}_vol_total'] = volume.rolling(size).sum()
        features[f'grp_{size}_vol_mean'] = volume.rolling(size).mean()
        features[f'grp_{size}_vol_trend'] = volume.rolling(max(1, size // 2)).mean() / volume.rolling(size).mean().replace(0, np.nan)
        features[f'grp_{size}_momentum'] = close - close.rolling(size).mean()
        total_path = df['range'].rolling(size).sum()
        net_move = (close - close.shift(size)).abs()
        features[f'grp_{size}_efficiency'] = np.where(total_path > 0, net_move / total_path, 0)
        features[f'grp_{size}_up_pct'] = df['direction'].rolling(size).apply(lambda x: (x > 0).sum() / len(x), raw=True)
        gh = high.rolling(size).max()
        gl = low.rolling(size).min()
        gr = gh - gl
        features[f'grp_{size}_close_pos'] = np.where(gr > 0, (close - gl) / gr, 0.5)

    # Cross-group
    short_g, long_g = groupings[0], groupings[-1]
    sv = df['return_log'].rolling(short_g).std()
    lv = df['return_log'].rolling(long_g).std()
    features['vol_ratio_short_long'] = np.where(lv > 0, sv / lv, 1.0)

    for i in range(len(groupings) - 1):
        g1, g2 = groupings[i], groupings[i + 1]
        features[f'mom_align_{g1}_{g2}'] = np.sign(features[f'grp_{g1}_return']) * np.sign(features[f'grp_{g2}_return'])

    eff_cols = [c for c in features.columns if 'efficiency' in c]
    features['mean_efficiency'] = features[eff_cols].mean(axis=1)
    features['eff_dispersion'] = features[eff_cols].std(axis=1)

    # Base features
    features['volatility_20'] = df['volatility_20']
    features['volatility_60'] = df['volatility_60']
    features['volume_ratio'] = df['volume_ratio']
    features['hour'] = df['hour']
    features['day_of_week'] = df['day_of_week']
    features['momentum_5'] = df['momentum_5']
    features['momentum_20'] = df['momentum_20']

    # Morphology encoding
    for size in groupings:
        net_ret = close - close.shift(size)
        grp_range = high.rolling(size).max() - low.rolling(size).min()
        efficiency = np.where(grp_range > 0, net_ret.abs() / grp_range, 0)
        gh_v = high.rolling(size).max()
        gl_v = low.rolling(size).min()
        gr_v = gh_v - gl_v
        close_pos = np.where(gr_v > 0, (close - gl_v) / gr_v, 0.5)

        morph = np.zeros(len(df))
        morph = np.where((net_ret > 0) & (efficiency > 0.6), 1, morph)  # Trend-Up
        morph = np.where((net_ret < 0) & (efficiency > 0.6), 2, morph)  # Trend-Down
        morph = np.where((np.array(close_pos) > 0.8) & (net_ret > 0), 3, morph)  # V-Bottom
        morph = np.where((np.array(close_pos) < 0.2) & (net_ret < 0), 4, morph)  # Inv-V
        features[f'morph_{size}_encoded'] = morph

    # === IMPROVEMENT 1: Order Flow Proxy Features ===
    logger.info("Adding order flow proxy features...")
    from core.order_flow_features import OrderFlowProxy
    of_engine = OrderFlowProxy()
    of_features = of_engine.compute_all(df)
    features = pd.concat([features, of_features], axis=1)

    # === IMPROVEMENT 2: Cross-Asset Features ===
    logger.info("Adding cross-asset features...")
    from core.cross_asset_features import CrossAssetFeatureProvider
    ca_engine = CrossAssetFeatureProvider(mode='simulate')
    ca_features = ca_engine.compute_features(df)
    features = pd.concat([features, ca_features], axis=1)

    logger.info(f"Total features (with order flow + cross-asset): {features.shape[1]}")
    features.to_parquet('data/features_all.parquet', index=False)
    return features


def train_models(df, features, config_dict):
    """Train ensemble models with expanded features (order flow + cross-asset)."""
    import xgboost as xgb
    import lightgbm as lgb
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    import joblib

    TARGET_HORIZON = 30
    MIN_EV = 50.0

    target_return = df['close'].shift(-TARGET_HORIZON) - df['close']
    target_dir = (target_return > 0).astype(int)
    target_big = (target_return.abs() >= MIN_EV).astype(int)

    features_clean = features.replace([np.inf, -np.inf], np.nan)
    valid = features_clean.notna().all(axis=1) & target_return.notna()

    X = features_clean[valid].values
    y_dir = target_dir[valid].values
    y_ret = target_return[valid].values
    y_big = target_big[valid].values

    n = len(X)
    split = int(n * 0.7)
    val_split = int(n * 0.85)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X[:split])
    X_va = scaler.transform(X[split:val_split])
    X_te = scaler.transform(X[val_split:])

    logger.info(f"Training: {split:,} | Val: {val_split-split:,} | Test: {n-val_split:,}")
    logger.info(f"Feature count: {X.shape[1]} (including order flow + cross-asset)")

    os.makedirs('data/trained_models', exist_ok=True)

    # XGBoost Direction
    logger.info("Training XGBoost direction...")
    xgb_dir = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        eval_metric='logloss', early_stopping_rounds=50, verbosity=0, n_jobs=-1
    )
    xgb_dir.fit(X_tr, y_dir[:split], eval_set=[(X_va, y_dir[split:val_split])], verbose=False)

    # LightGBM Direction
    logger.info("Training LightGBM direction...")
    lgb_dir = lgb.LGBMClassifier(
        n_estimators=500, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1, n_jobs=-1
    )
    lgb_dir.fit(X_tr, y_dir[:split], eval_set=[(X_va, y_dir[split:val_split])],
                callbacks=[lgb.early_stopping(50, verbose=False)])

    # XGBoost Return
    logger.info("Training XGBoost return...")
    xgb_ret = xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        eval_metric='rmse', early_stopping_rounds=50, verbosity=0, n_jobs=-1
    )
    xgb_ret.fit(X_tr, y_ret[:split], eval_set=[(X_va, y_ret[split:val_split])], verbose=False)

    # LightGBM Return
    logger.info("Training LightGBM return...")
    lgb_ret = lgb.LGBMRegressor(
        n_estimators=500, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1, n_jobs=-1
    )
    lgb_ret.fit(X_tr, y_ret[:split], eval_set=[(X_va, y_ret[split:val_split])],
                callbacks=[lgb.early_stopping(50, verbose=False)])

    # Magnitude
    logger.info("Training magnitude classifier...")
    xgb_mag = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, random_state=42, eval_metric='logloss',
        early_stopping_rounds=30, verbosity=0, n_jobs=-1
    )
    xgb_mag.fit(X_tr, y_big[:split], eval_set=[(X_va, y_big[split:val_split])], verbose=False)

    # Test metrics
    test_acc_xgb = accuracy_score(y_dir[val_split:], xgb_dir.predict(X_te))
    test_acc_lgb = accuracy_score(y_dir[val_split:], lgb_dir.predict(X_te))
    mag_acc = accuracy_score(y_big[val_split:], xgb_mag.predict(X_te))

    logger.info(f"Test Accuracy - XGB: {test_acc_xgb:.4f} | LGB: {test_acc_lgb:.4f} | Mag: {mag_acc:.4f}")

    # === IMPROVEMENT 3: Compute regime mask for backtest ===
    logger.info("Computing regime filter mask...")
    from core.regime_filter import RegimeFilter
    from config import SystemConfig
    regime_config = SystemConfig()
    regime_filter = RegimeFilter(regime_config)
    regime_mask = regime_filter.compute_regime_mask(
        df,
        min_quintile=regime_config.trading.min_volatility_quintile,
        start_hour=regime_config.trading.session_start_hour_et,
        end_hour=regime_config.trading.session_end_hour_et,
    )
    # Save regime mask aligned to valid indices
    regime_mask_valid = regime_mask[valid].values

    # === IMPROVEMENT 4: Magnitude filter stats ===
    mag_probs_test = xgb_mag.predict_proba(X_te)[:, 1]
    mag_filter_count = (mag_probs_test >= 0.5).sum()
    logger.info(f"Magnitude filter: {mag_filter_count}/{len(X_te)} test bars predict big move (>= 50pts)")

    # Save
    joblib.dump(xgb_dir, 'data/trained_models/xgb_direction.pkl')
    joblib.dump(lgb_dir, 'data/trained_models/lgb_direction.pkl')
    joblib.dump(xgb_ret, 'data/trained_models/xgb_return.pkl')
    joblib.dump(lgb_ret, 'data/trained_models/lgb_return.pkl')
    joblib.dump(xgb_mag, 'data/trained_models/xgb_magnitude.pkl')
    joblib.dump(scaler, 'data/trained_models/scaler.pkl')

    # Save feature names for real-time engine
    feature_names = features.columns.tolist()
    with open('data/trained_models/feature_names.json', 'w') as f:
        json.dump(feature_names, f)

    logger.info("Models saved to data/trained_models/")

    # Generate signals with all 4 improvements applied
    logger.info("Generating signals with regime + magnitude filters...")
    signals = _generate_filtered_signals(
        X_te, scaler, xgb_dir, lgb_dir, xgb_ret, lgb_ret, xgb_mag,
        regime_mask_valid[val_split - (n - len(regime_mask_valid)):] if len(regime_mask_valid) > (n - val_split) else regime_mask_valid[-(n - val_split):],
        mag_threshold=0.5
    )

    return {
        'xgb_acc': test_acc_xgb, 'lgb_acc': test_acc_lgb, 'mag_acc': mag_acc,
        'feature_count': X.shape[1],
    }


def _generate_filtered_signals(X_test, scaler, xgb_dir, lgb_dir, xgb_ret, lgb_ret,
                                xgb_mag, regime_mask, mag_threshold=0.5):
    """Generate signals applying regime filter + magnitude filter."""
    n = len(X_test)

    # Direction probabilities (ensemble average)
    prob_up_xgb = xgb_dir.predict_proba(X_test)[:, 1]
    prob_up_lgb = lgb_dir.predict_proba(X_test)[:, 1]
    prob_up = (prob_up_xgb + prob_up_lgb) / 2.0

    # Return predictions (ensemble average)
    ret_xgb = xgb_ret.predict(X_test)
    ret_lgb = lgb_ret.predict(X_test)
    ret_pred = (ret_xgb + ret_lgb) / 2.0

    # Magnitude probability
    prob_big = xgb_mag.predict_proba(X_test)[:, 1]

    # Direction
    direction = np.where(prob_up > 0.5, 1, -1)

    # Confidence
    confidence = np.abs(prob_up - 0.5) * 2 * prob_big

    # Expected value
    costs = (0.52 + 1.0) * 2  # 3.04 points
    ev = np.abs(ret_pred) * confidence - costs

    # Signal validity with ALL filters
    signal_valid = (
        (ev >= 50.0) &
        (confidence >= 0.60) &
        (prob_big >= mag_threshold)  # IMPROVEMENT 4: Magnitude gate
    )

    # IMPROVEMENT 3: Regime filter
    if regime_mask is not None and len(regime_mask) == n:
        signal_valid = signal_valid & regime_mask.astype(bool)

    n_valid = signal_valid.sum()
    logger.info(f"Filtered signals: {n_valid}/{n} valid ({n_valid/n*100:.2f}%)")
    logger.info(f"  After EV+confidence filter: {((ev >= 50.0) & (confidence >= 0.60)).sum()}")
    logger.info(f"  After magnitude filter (>={mag_threshold}): {((ev >= 50.0) & (confidence >= 0.60) & (prob_big >= mag_threshold)).sum()}")
    if regime_mask is not None:
        logger.info(f"  After regime filter: {n_valid}")

    return {
        'direction': direction,
        'prob_up': prob_up,
        'expected_return': ret_pred,
        'prob_big_move': prob_big,
        'confidence': confidence,
        'expected_value': ev,
        'signal_valid': signal_valid,
    }


def run_compare(df, features, groupings):
    """
    Compare mode: run backtests side-by-side with different filter configurations.
    Uses magnitude-driven strategy: enter when mag classifier predicts big move,
    direction from ensemble, regime filter optional.
    """
    import xgboost as xgb
    import lightgbm as lgb
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    import joblib
    from backtest.engine import BacktestEngine
    from config import SystemConfig
    from core.regime_filter import RegimeFilter

    TARGET_HORIZON = 30

    logger.info("=" * 70)
    logger.info("COMPARE MODE: Magnitude-driven signal strategy")
    logger.info("=" * 70)

    # Prepare data
    target_return = df['close'].shift(-TARGET_HORIZON) - df['close']
    features_clean = features.replace([np.inf, -np.inf], np.nan)
    valid = features_clean.notna().all(axis=1) & target_return.notna()

    X = features_clean[valid].values
    y_ret = target_return[valid].values
    y_dir = (y_ret > 0).astype(int)

    n = len(X)
    val_split = int(n * 0.85)

    # Load models (use saved scaler from training)
    model_path = 'data/trained_models/'
    try:
        scaler = joblib.load(os.path.join(model_path, 'scaler.pkl'))
        xgb_dir = joblib.load(os.path.join(model_path, 'xgb_direction.pkl'))
        lgb_dir = joblib.load(os.path.join(model_path, 'lgb_direction.pkl'))
        xgb_ret = joblib.load(os.path.join(model_path, 'xgb_return.pkl'))
        lgb_ret = joblib.load(os.path.join(model_path, 'lgb_return.pkl'))
        xgb_mag = joblib.load(os.path.join(model_path, 'xgb_magnitude.pkl'))
        logger.info("Loaded existing trained models")
    except Exception as e:
        logger.error(f"Models not found. Run --mode train first. Error: {e}")
        return

    X_te = scaler.transform(X[val_split:])
    y_ret_test = y_ret[val_split:]
    y_dir_test = y_dir[val_split:]

    # Compute regime mask
    regime_filter = RegimeFilter(SystemConfig())
    regime_mask_full = regime_filter.compute_regime_mask(df)
    regime_mask_valid = regime_mask_full[valid].values
    regime_test = regime_mask_valid[val_split:]

    # Generate predictions
    prob_up = (xgb_dir.predict_proba(X_te)[:, 1] + lgb_dir.predict_proba(X_te)[:, 1]) / 2
    ret_pred = (xgb_ret.predict(X_te) + lgb_ret.predict(X_te)) / 2
    prob_big = xgb_mag.predict_proba(X_te)[:, 1]
    direction = np.where(prob_up > 0.5, 1, -1)

    # Diagnostic stats
    dir_bias = np.abs(prob_up - 0.5)
    logger.info(f"\nSignal diagnostics (test set, n={len(X_te):,}):")
    logger.info(f"  prob_up range: [{prob_up.min():.4f}, {prob_up.max():.4f}], bias max: {dir_bias.max():.4f}")
    logger.info(f"  prob_big: mean={prob_big.mean():.4f}, >=0.3: {(prob_big>=0.3).sum():,}, >=0.5: {(prob_big>=0.5).sum():,}")
    logger.info(f"  ret_pred: mean={ret_pred.mean():.2f}, std={ret_pred.std():.2f}")
    logger.info(f"  regime active: {regime_test.sum():,}/{len(regime_test):,}")

    # Test set df slice for backtest
    df_valid = df[valid].iloc[val_split:].reset_index(drop=True)

    # NEW STRATEGY: Magnitude-driven entry
    # Enter when:
    #   1. Magnitude classifier says big move coming (prob_big >= threshold)
    #   2. Direction has ANY bias (prob_up != exactly 0.5)
    #   3. Optionally: regime filter active
    # Use directional selectivity (stronger bias = better) as tiebreaker

    configs = [
        # Magnitude-only strategies (no EV/confidence gate)
        {'name': 'v1: All direction signals (baseline)',
         'dir_bias_min': 0.0, 'mag_min': 0.0, 'regime': False,
         'sl_mult': 3.0, 'tp_mult': 5.0},
        {'name': 'v2: Mag>=0.30 (big move likely)',
         'dir_bias_min': 0.0, 'mag_min': 0.30, 'regime': False,
         'sl_mult': 3.0, 'tp_mult': 5.0},
        {'name': 'v3: Mag>=0.50 (big move probable)',
         'dir_bias_min': 0.0, 'mag_min': 0.50, 'regime': False,
         'sl_mult': 3.0, 'tp_mult': 5.0},
        {'name': 'v4: Mag>=0.50 + regime filter',
         'dir_bias_min': 0.0, 'mag_min': 0.50, 'regime': True,
         'sl_mult': 3.0, 'tp_mult': 5.0},
        {'name': 'v5: Mag>=0.50 + dir_bias>=1%',
         'dir_bias_min': 0.01, 'mag_min': 0.50, 'regime': False,
         'sl_mult': 3.0, 'tp_mult': 5.0},
        {'name': 'v6: Mag>=0.50 + dir_bias>=1% + regime',
         'dir_bias_min': 0.01, 'mag_min': 0.50, 'regime': True,
         'sl_mult': 3.0, 'tp_mult': 5.0},
        {'name': 'v7: Mag>=0.50 + dir>=2% + regime',
         'dir_bias_min': 0.02, 'mag_min': 0.50, 'regime': True,
         'sl_mult': 3.0, 'tp_mult': 5.0},
        {'name': 'v8: Mag>=0.70 + regime (ultra select)',
         'dir_bias_min': 0.0, 'mag_min': 0.70, 'regime': True,
         'sl_mult': 3.0, 'tp_mult': 5.0},
        # Alternative SL/TP ratios
        {'name': 'v9: Mag>=0.50 + regime + tight SL',
         'dir_bias_min': 0.0, 'mag_min': 0.50, 'regime': True,
         'sl_mult': 1.5, 'tp_mult': 3.0},
        {'name': 'v10: Mag>=0.50 + regime + wide SL',
         'dir_bias_min': 0.0, 'mag_min': 0.50, 'regime': True,
         'sl_mult': 4.0, 'tp_mult': 8.0},
    ]

    results = []
    for cfg in configs:
        sig_valid = (
            (prob_big >= cfg['mag_min']) &
            (dir_bias >= cfg['dir_bias_min'])
        )
        if cfg['regime']:
            sig_valid = sig_valid & regime_test.astype(bool)

        n_signals = int(sig_valid.sum())

        if n_signals == 0:
            results.append({
                'config': cfg['name'], 'signals': 0, 'trades': 0,
                'pnl': 0, 'win_rate': 0, 'pf': 0, 'ev_per_trade': 0, 'sharpe': 0, 'max_dd': 0,
            })
            continue

        # Quick PnL calculation (no backtest engine, just direction * return)
        positions = np.where(sig_valid, direction, 0)
        trade_returns = positions * y_ret_test
        trade_pnls = trade_returns[positions != 0]
        costs = 3.04  # round-trip

        if len(trade_pnls) > 0:
            gross_pnl = trade_pnls.sum()
            net_pnl = gross_pnl - len(trade_pnls) * costs
            winners = trade_pnls[trade_pnls > costs]
            losers = trade_pnls[trade_pnls <= costs]
            win_rate = (trade_pnls > costs).mean()
            pf = abs(winners.sum() / losers.sum()) if losers.sum() != 0 else float('inf')
            ev_per_trade = net_pnl / len(trade_pnls)
            sharpe = (trade_pnls - costs).mean() / max((trade_pnls - costs).std(), 0.01) * np.sqrt(252)
            equity = np.cumsum(trade_pnls - costs)
            max_dd = (equity - np.maximum.accumulate(equity)).min()
        else:
            net_pnl = win_rate = pf = ev_per_trade = sharpe = max_dd = 0

        # Also run actual backtest engine for the most promising configs
        bt_result = None
        if cfg['mag_min'] >= 0.50 and n_signals < 50000:
            signals_df = pd.DataFrame({
                'direction_numeric': direction,
                'expected_return': ret_pred,
                'expected_value': np.abs(ret_pred),
                'confidence': prob_big,
                'signal_valid': sig_valid,
                'prob_big_move': prob_big,
                'regime_active': regime_test.astype(bool) if cfg['regime'] else True,
            })
            sys_config = SystemConfig()
            sys_config.trading.min_confidence = 0.0
            sys_config.trading.min_expected_value_points = 0.0
            sys_config.trading.stop_loss_multiplier = cfg['sl_mult']
            sys_config.trading.take_profit_multiplier = cfg['tp_mult']
            engine = BacktestEngine(sys_config)
            try:
                bt_result = engine.run(df_valid, signals_df)
            except Exception as e:
                logger.warning(f"  Backtest engine failed for {cfg['name']}: {e}")

        results.append({
            'config': cfg['name'],
            'signals': n_signals,
            'trades': bt_result['n_trades'] if bt_result else len(trade_pnls),
            'pnl': bt_result['total_pnl_points'] if bt_result else net_pnl,
            'win_rate': bt_result['win_rate'] if bt_result else win_rate,
            'pf': bt_result['profit_factor'] if bt_result else pf,
            'ev_per_trade': bt_result['expectancy_per_trade'] if bt_result else ev_per_trade,
            'sharpe': bt_result['sharpe_ratio'] if bt_result else sharpe,
            'max_dd': bt_result['max_drawdown_points'] if bt_result else max_dd,
        })

    # Print comparison table
    logger.info("\n" + "=" * 110)
    logger.info("COMPARISON RESULTS (Test Set: 15% out-of-sample, 181 features)")
    logger.info("=" * 110)
    logger.info(f"{'Configuration':<42} {'Signals':>8} {'Trades':>7} {'PnL pts':>10} {'WR':>7} {'PF':>7} {'EV/Tr':>8} {'Sharpe':>7} {'MaxDD':>8}")
    logger.info("-" * 110)
    for r in results:
        logger.info(
            f"{r['config']:<42} {r['signals']:>8} {r['trades']:>7} "
            f"{r['pnl']:>10.1f} {r['win_rate']:>6.1%} {r['pf']:>7.2f} "
            f"{r['ev_per_trade']:>8.1f} {r.get('sharpe', 0):>7.2f} {r.get('max_dd', 0):>8.1f}"
        )
    logger.info("=" * 110)

    # Save comparison
    with open('data/compare_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    return results


def run_walk_forward_filtered(df, features, groupings, n_windows=5):
    """
    Walk-forward validation WITH regime + magnitude filters active.
    This is the gold standard test for the improved system.
    """
    import xgboost as xgb
    import lightgbm as lgb
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    from core.regime_filter import RegimeFilter
    from config import SystemConfig

    TARGET_HORIZON = 30
    MIN_EV = 50.0
    MAG_THRESHOLDS = [0.0, 0.5, 0.6]  # Test multiple thresholds

    logger.info("=" * 70)
    logger.info(f"WALK-FORWARD VALIDATION WITH FILTERS ({n_windows} windows)")
    logger.info("=" * 70)

    features_clean = features.replace([np.inf, -np.inf], np.nan)
    target_return = df['close'].shift(-TARGET_HORIZON) - df['close']
    valid = features_clean.notna().all(axis=1) & target_return.notna()

    X = features_clean[valid].values
    y_ret = target_return[valid].values
    y_dir = (y_ret > 0).astype(int)
    y_big = (np.abs(y_ret) >= MIN_EV).astype(int)

    # Compute regime mask
    regime_filter = RegimeFilter(SystemConfig())
    regime_mask_full = regime_filter.compute_regime_mask(df)
    regime_mask = regime_mask_full[valid].values

    n = len(X)
    window_size = n // (n_windows + 1)  # +1 for initial training window
    train_start_size = window_size * 2  # Use first 2 windows for initial training

    all_results = {t: [] for t in MAG_THRESHOLDS}

    for w in range(n_windows):
        train_end = train_start_size + w * window_size
        test_start = train_end
        test_end = min(test_start + window_size, n)

        if test_end <= test_start:
            break

        X_train = X[:train_end]
        y_dir_train = y_dir[:train_end]
        y_ret_train = y_ret[:train_end]
        y_big_train = y_big[:train_end]
        X_test = X[test_start:test_end]
        y_dir_test = y_dir[test_start:test_end]
        y_ret_test = y_ret[test_start:test_end]
        regime_test = regime_mask[test_start:test_end]

        # Scale
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Train models
        try:
            xgb_dir = xgb.XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
                eval_metric='logloss', early_stopping_rounds=30, verbosity=0
            )
            xgb_dir.fit(X_train_s, y_dir_train, eval_set=[(X_test_s, y_dir_test)], verbose=False)

            lgb_dir = lgb.LGBMClassifier(
                n_estimators=300, max_depth=8, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
            )
            lgb_dir.fit(X_train_s, y_dir_train, eval_set=[(X_test_s, y_dir_test)],
                        callbacks=[lgb.early_stopping(30, verbose=False)])

            xgb_ret = xgb.XGBRegressor(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
                eval_metric='rmse', early_stopping_rounds=30, verbosity=0
            )
            xgb_ret.fit(X_train_s, y_ret_train, eval_set=[(X_test_s, y_ret_test)], verbose=False)

            lgb_ret = lgb.LGBMRegressor(
                n_estimators=300, max_depth=8, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
            )
            lgb_ret.fit(X_train_s, y_ret_train, eval_set=[(X_test_s, y_ret_test)],
                        callbacks=[lgb.early_stopping(30, verbose=False)])

            xgb_mag = xgb.XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                subsample=0.8, random_state=42, eval_metric='logloss',
                early_stopping_rounds=20, verbosity=0
            )
            xgb_mag.fit(X_train_s, y_big_train, eval_set=[(X_test_s, (np.abs(y_ret[test_start:test_end]) >= MIN_EV).astype(int))], verbose=False)

            # Predictions
            prob_up = (xgb_dir.predict_proba(X_test_s)[:, 1] + lgb_dir.predict_proba(X_test_s)[:, 1]) / 2
            ret_pred = (xgb_ret.predict(X_test_s) + lgb_ret.predict(X_test_s)) / 2
            prob_big = xgb_mag.predict_proba(X_test_s)[:, 1]

            dir_acc = accuracy_score(y_dir_test, (prob_up > 0.5).astype(int))
            mag_acc = accuracy_score(
                (np.abs(y_ret[test_start:test_end]) >= MIN_EV).astype(int),
                (prob_big >= 0.5).astype(int)
            )

            direction = np.where(prob_up > 0.5, 1, -1)
            confidence = np.abs(prob_up - 0.5) * 2 * prob_big
            costs = (0.52 + 1.0) * 2
            ev = np.abs(ret_pred) * confidence - costs

            # Test each magnitude threshold (magnitude-driven strategy)
            for mag_t in MAG_THRESHOLDS:
                positions = np.where(
                    (prob_big >= mag_t) &
                    regime_test.astype(bool),
                    direction, 0
                )
                pnl = positions * y_ret_test
                n_trades = int((positions != 0).sum())
                total_pnl = float(pnl.sum())
                trade_pnls = pnl[positions != 0]

                if n_trades > 0:
                    win_rate = float((trade_pnls > 0).mean())
                    avg_pnl = float(trade_pnls.mean())
                    sharpe = float(trade_pnls.mean() / max(trade_pnls.std(), 0.01) * np.sqrt(252))
                else:
                    win_rate = avg_pnl = sharpe = 0

                all_results[mag_t].append({
                    'window': w,
                    'train_size': len(X_train),
                    'test_size': len(X_test),
                    'dir_accuracy': dir_acc,
                    'mag_accuracy': mag_acc,
                    'n_trades': n_trades,
                    'total_pnl': total_pnl,
                    'win_rate': win_rate,
                    'ev_per_trade': avg_pnl,
                    'sharpe': sharpe,
                })

            logger.info(f"  Window {w}: DirAcc={dir_acc:.4f} MagAcc={mag_acc:.4f} | "
                       f"Regime bars: {regime_test.sum()}/{len(regime_test)}")
            for mag_t in MAG_THRESHOLDS:
                r = all_results[mag_t][-1]
                logger.info(f"    mag>={mag_t:.1f}: trades={r['n_trades']:>5} PnL={r['total_pnl']:>8.1f} "
                           f"WR={r['win_rate']:.1%} EV={r['ev_per_trade']:>6.1f} Sharpe={r['sharpe']:.2f}")

        except Exception as e:
            logger.error(f"  Window {w} failed: {e}")
            for mag_t in MAG_THRESHOLDS:
                all_results[mag_t].append({'window': w, 'error': str(e)})

    # Summary
    logger.info("\n" + "=" * 90)
    logger.info("WALK-FORWARD SUMMARY (with regime filter + order flow + cross-asset features)")
    logger.info("=" * 90)
    logger.info(f"{'Mag Threshold':>15} {'Windows':>8} {'Total PnL':>12} {'Avg WR':>8} {'Avg EV/Tr':>10} {'Avg Sharpe':>10} {'Tot Trades':>10}")
    logger.info("-" * 90)

    summary = {}
    for mag_t in MAG_THRESHOLDS:
        valid_windows = [r for r in all_results[mag_t] if 'error' not in r]
        if valid_windows:
            total_pnl = sum(r['total_pnl'] for r in valid_windows)
            avg_wr = np.mean([r['win_rate'] for r in valid_windows if r['n_trades'] > 0])
            avg_ev = np.mean([r['ev_per_trade'] for r in valid_windows if r['n_trades'] > 0])
            avg_sharpe = np.mean([r['sharpe'] for r in valid_windows if r['n_trades'] > 0])
            tot_trades = sum(r['n_trades'] for r in valid_windows)

            logger.info(
                f"{f'>= {mag_t:.1f}':>15} {len(valid_windows):>8} {total_pnl:>12.1f} "
                f"{avg_wr:>7.1%} {avg_ev:>10.1f} {avg_sharpe:>10.2f} {tot_trades:>10}"
            )
            summary[mag_t] = {
                'total_pnl': total_pnl, 'avg_wr': avg_wr,
                'avg_ev': avg_ev, 'avg_sharpe': avg_sharpe, 'total_trades': tot_trades,
            }

    logger.info("=" * 90)

    # Save
    wf_output = {'per_window': {str(k): v for k, v in all_results.items()}, 'summary': {str(k): v for k, v in summary.items()}}
    with open('data/walk_forward_filtered_results.json', 'w') as f:
        json.dump(wf_output, f, indent=2, default=str)

    return all_results, summary


def main():
    parser = argparse.ArgumentParser(description='MNQ Quantitative Trading System v2')
    parser.add_argument('--data', type=str, help='Path to data file (CSV or Parquet)')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'backtest', 'bot', 'report', 'full', 'compare', 'walkforward'],
                       help='Operation mode')
    parser.add_argument('--horizon', type=int, default=30, help='Target horizon in bars')
    parser.add_argument('--wf-windows', type=int, default=5, help='Walk-forward windows')
    args = parser.parse_args()

    os.makedirs('data/logs', exist_ok=True)
    os.makedirs('data/trained_models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)

    logger.info("=" * 70)
    logger.info("MNQ QUANTITATIVE TRADING SYSTEM v2.0")
    logger.info(f"Mode: {args.mode}")
    logger.info("=" * 70)

    if args.mode in ['train', 'full', 'backtest', 'report', 'compare', 'walkforward']:
        if not args.data:
            default_paths = [
                'data/mnq_continuous_1m.parquet',
                'data/mnq_data.parquet',
                'data/glbx-mdp3-20210312-20260311.ohlcv-1m.parquet',
            ]
            for p in default_paths:
                if os.path.exists(p):
                    args.data = p
                    break
            if not args.data:
                logger.error("No data file specified. Use --data /path/to/file")
                sys.exit(1)

        df = load_data(args.data)
        df.to_parquet('data/mnq_preprocessed.parquet', index=False)

    # Load groupings
    groupings = None
    if args.mode in ['train', 'full', 'compare', 'walkforward']:
        groupings_path = 'data/optimal_groupings.json'
        if os.path.exists(groupings_path) and args.mode != 'full':
            with open(groupings_path) as f:
                groupings = json.load(f)
            logger.info(f"Loaded groupings: {groupings}")
        else:
            groupings = discover_groupings(df)

    if args.mode in ['train', 'full']:
        t0 = time.time()
        features = build_features(df, groupings)
        metrics = train_models(df, features, {})
        logger.info(f"\nTraining complete in {time.time()-t0:.1f}s")
        logger.info(f"Results: {metrics}")

    elif args.mode == 'compare':
        t0 = time.time()
        features = build_features(df, groupings)
        results = run_compare(df, features, groupings)
        logger.info(f"\nComparison complete in {time.time()-t0:.1f}s")

    elif args.mode == 'walkforward':
        t0 = time.time()
        features = build_features(df, groupings)
        results, summary = run_walk_forward_filtered(
            df, features, groupings, n_windows=args.wf_windows
        )
        logger.info(f"\nWalk-forward complete in {time.time()-t0:.1f}s")

    elif args.mode == 'bot':
        from bot.trading_core import TradingBot
        from config import SystemConfig

        config = SystemConfig()
        bot = TradingBot(config)
        logger.info("Trading bot initialized. Waiting for bars...")
        logger.info("Use bot.process_bar(bar_dict) to feed bars")

        try:
            while True:
                status = bot.get_status()
                logger.info(f"Status: {status['daily_performance']}")
                time.sleep(60)
        except KeyboardInterrupt:
            bot.shutdown()

    logger.info("System complete.")


if __name__ == '__main__':
    main()
