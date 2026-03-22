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
    """Build complete feature matrix."""
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

    logger.info(f"Total features: {features.shape[1]}")
    features.to_parquet('data/features_all.parquet', index=False)
    return features


def train_models(df, features, config_dict):
    """Train ensemble models."""
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

    # Save
    joblib.dump(xgb_dir, 'data/trained_models/xgb_direction.pkl')
    joblib.dump(lgb_dir, 'data/trained_models/lgb_direction.pkl')
    joblib.dump(xgb_ret, 'data/trained_models/xgb_return.pkl')
    joblib.dump(lgb_ret, 'data/trained_models/lgb_return.pkl')
    joblib.dump(xgb_mag, 'data/trained_models/xgb_magnitude.pkl')
    joblib.dump(scaler, 'data/trained_models/scaler.pkl')

    logger.info("Models saved to data/trained_models/")
    return {'xgb_acc': test_acc_xgb, 'lgb_acc': test_acc_lgb, 'mag_acc': mag_acc}


def main():
    parser = argparse.ArgumentParser(description='MNQ Quantitative Trading System')
    parser.add_argument('--data', type=str, help='Path to data file (CSV or Parquet)')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'backtest', 'bot', 'report', 'full'],
                       help='Operation mode')
    parser.add_argument('--horizon', type=int, default=30, help='Target horizon in bars')
    args = parser.parse_args()

    os.makedirs('data/logs', exist_ok=True)
    os.makedirs('data/trained_models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)

    logger.info("=" * 70)
    logger.info("MNQ QUANTITATIVE TRADING SYSTEM")
    logger.info(f"Mode: {args.mode}")
    logger.info("=" * 70)

    if args.mode in ['train', 'full', 'backtest', 'report']:
        if not args.data:
            # Try default paths
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

    if args.mode in ['train', 'full']:
        t0 = time.time()

        # Discover groupings
        groupings = discover_groupings(df)

        # Build features
        features = build_features(df, groupings)

        # Train models
        metrics = train_models(df, features, {})

        logger.info(f"\nTraining complete in {time.time()-t0:.1f}s")
        logger.info(f"Results: {metrics}")

    elif args.mode == 'bot':
        from bot.trading_core import TradingBot
        from config import SystemConfig

        config = SystemConfig()
        bot = TradingBot(config)
        logger.info("Trading bot initialized. Waiting for bars...")
        logger.info("Use bot.process_bar(bar_dict) to feed bars")

        # Interactive mode
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
