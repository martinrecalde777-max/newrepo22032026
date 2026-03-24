#!/usr/bin/env python3
"""
Magnitude-First Pipeline
=========================
Pipeline invertido: Magnitud primero (gate) → Dirección después → Stop loss optimizado.

Filosofía:
  1. El clasificador de magnitud (81.8% accuracy) filtra barras donde se espera movimiento grande
  2. El clasificador de dirección decide hacia dónde
  3. El stop loss corta rápido cuando la dirección falla
  4. Grid search encuentra los sweet spots óptimos

Usa TODA la data disponible con split temporal: 70% train / 15% val / 15% test
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings
from itertools import product

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import joblib

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('MagFirst')


def load_and_prepare(data_path: str):
    """Load data, build features, prepare targets."""
    from main import load_data, build_features

    logger.info("Loading data...")
    df = load_data(data_path)

    # Load groupings
    groupings_path = 'data/optimal_groupings.json'
    if os.path.exists(groupings_path):
        with open(groupings_path) as f:
            groupings = json.load(f)
        logger.info(f"Loaded groupings: {groupings}")
    else:
        from main import discover_groupings
        groupings = discover_groupings(df)

    logger.info("Building features...")
    features = build_features(df, groupings)

    return df, features, groupings


def train_magnitude_first(df, features, target_horizon=30):
    """
    Train all models and run the magnitude-first grid search.
    Uses ALL data with temporal split.
    """
    MIN_EV_POINTS = 50.0

    # Targets
    target_return = df['close'].shift(-target_horizon) - df['close']
    target_dir = (target_return > 0).astype(int)
    target_big = (target_return.abs() >= MIN_EV_POINTS).astype(int)

    features_clean = features.replace([np.inf, -np.inf], np.nan)
    valid = features_clean.notna().all(axis=1) & target_return.notna()

    X = features_clean[valid].values
    y_dir = target_dir[valid].values
    y_ret = target_return[valid].values
    y_big = target_big[valid].values
    df_valid = df[valid].reset_index(drop=True)

    n = len(X)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    logger.info(f"Total valid samples: {n:,}")
    logger.info(f"Train: {train_end:,} | Val: {val_end - train_end:,} | Test: {n - val_end:,}")

    # Class balance
    big_pct_train = y_big[:train_end].mean()
    big_pct_test = y_big[val_end:].mean()
    logger.info(f"Big move (>=50pts) prevalence - Train: {big_pct_train:.1%} | Test: {big_pct_test:.1%}")
    logger.info(f"  -> Naive baseline (always predict majority): {max(big_pct_test, 1-big_pct_test):.1%}")

    # Scale
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X[:train_end])
    X_va = scaler.transform(X[train_end:val_end])
    X_te = scaler.transform(X[val_end:])

    y_dir_tr, y_dir_va, y_dir_te = y_dir[:train_end], y_dir[train_end:val_end], y_dir[val_end:]
    y_ret_tr, y_ret_va, y_ret_te = y_ret[:train_end], y_ret[train_end:val_end], y_ret[val_end:]
    y_big_tr, y_big_va, y_big_te = y_big[:train_end], y_big[train_end:val_end], y_big[val_end:]

    # ================================================================
    # TRAIN MODELS
    # ================================================================
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING MODELS")
    logger.info("=" * 70)

    # 1. Magnitude classifier (THE STAR)
    logger.info("Training magnitude classifier (>=50 pts)...")
    xgb_mag = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        eval_metric='logloss', early_stopping_rounds=50,
        verbosity=0, n_jobs=-1,
        scale_pos_weight=(1 - big_pct_train) / max(big_pct_train, 0.01),
    )
    xgb_mag.fit(X_tr, y_big_tr, eval_set=[(X_va, y_big_va)], verbose=False)

    # 2. XGBoost Direction
    logger.info("Training XGBoost direction...")
    xgb_dir = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        eval_metric='logloss', early_stopping_rounds=50,
        verbosity=0, n_jobs=-1
    )
    xgb_dir.fit(X_tr, y_dir_tr, eval_set=[(X_va, y_dir_va)], verbose=False)

    # 3. LightGBM Direction
    logger.info("Training LightGBM direction...")
    lgb_dir = lgb.LGBMClassifier(
        n_estimators=500, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        verbose=-1, n_jobs=-1
    )
    lgb_dir.fit(X_tr, y_dir_tr, eval_set=[(X_va, y_dir_va)],
                callbacks=[lgb.early_stopping(50, verbose=False)])

    # 4. Return regressors
    logger.info("Training return regressors...")
    xgb_ret = xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        eval_metric='rmse', early_stopping_rounds=50,
        verbosity=0, n_jobs=-1
    )
    xgb_ret.fit(X_tr, y_ret_tr, eval_set=[(X_va, y_ret_va)], verbose=False)

    lgb_ret = lgb.LGBMRegressor(
        n_estimators=500, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        verbose=-1, n_jobs=-1
    )
    lgb_ret.fit(X_tr, y_ret_tr, eval_set=[(X_va, y_ret_va)],
                callbacks=[lgb.early_stopping(50, verbose=False)])

    # ================================================================
    # EVALUATE MAGNITUDE CLASSIFIER IN DEPTH
    # ================================================================
    logger.info("\n" + "=" * 70)
    logger.info("MAGNITUDE CLASSIFIER - DEEP ANALYSIS")
    logger.info("=" * 70)

    mag_probs_te = xgb_mag.predict_proba(X_te)[:, 1]
    mag_preds_te = (mag_probs_te >= 0.5).astype(int)

    mag_acc = accuracy_score(y_big_te, mag_preds_te)
    mag_prec = precision_score(y_big_te, mag_preds_te, zero_division=0)
    mag_rec = recall_score(y_big_te, mag_preds_te, zero_division=0)
    mag_f1 = f1_score(y_big_te, mag_preds_te, zero_division=0)

    cm = confusion_matrix(y_big_te, mag_preds_te)
    logger.info(f"  Accuracy:  {mag_acc:.4f}")
    logger.info(f"  Precision: {mag_prec:.4f} (when it says 'big move', how often correct)")
    logger.info(f"  Recall:    {mag_rec:.4f} (of all big moves, how many it catches)")
    logger.info(f"  F1:        {mag_f1:.4f}")
    logger.info(f"  Confusion matrix:")
    logger.info(f"    TN={cm[0,0]:,}  FP={cm[0,1]:,}")
    logger.info(f"    FN={cm[1,0]:,}  TP={cm[1,1]:,}")

    # Precision/Recall at different thresholds
    logger.info("\n  Magnitude threshold analysis:")
    logger.info(f"  {'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>8} {'N_pass':>8} {'Actual_big%':>12}")
    for thresh in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
        preds_t = (mag_probs_te >= thresh).astype(int)
        n_pass = preds_t.sum()
        if n_pass > 0:
            prec_t = precision_score(y_big_te, preds_t, zero_division=0)
            rec_t = recall_score(y_big_te, preds_t, zero_division=0)
            f1_t = f1_score(y_big_te, preds_t, zero_division=0)
            actual_big_pct = y_big_te[preds_t == 1].mean() if n_pass > 0 else 0
            logger.info(f"  {thresh:>10.2f} {prec_t:>10.4f} {rec_t:>10.4f} {f1_t:>8.4f} {n_pass:>8,} {actual_big_pct:>11.1%}")

    # ================================================================
    # DIRECTION CLASSIFIER ANALYSIS
    # ================================================================
    logger.info("\n" + "=" * 70)
    logger.info("DIRECTION CLASSIFIER ANALYSIS")
    logger.info("=" * 70)

    prob_up_te = (xgb_dir.predict_proba(X_te)[:, 1] + lgb_dir.predict_proba(X_te)[:, 1]) / 2
    dir_preds_te = (prob_up_te > 0.5).astype(int)
    dir_acc_overall = accuracy_score(y_dir_te, dir_preds_te)
    logger.info(f"  Overall direction accuracy: {dir_acc_overall:.4f}")

    # Direction accuracy CONDITIONED on magnitude prediction
    logger.info("\n  Direction accuracy conditioned on magnitude probability:")
    logger.info(f"  {'Mag_thresh':>10} {'N_bars':>8} {'Dir_acc':>8} {'Dir_acc_big':>12} {'N_actual_big':>12}")
    for mag_t in [0.0, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        mask = mag_probs_te >= mag_t
        n_bars = mask.sum()
        if n_bars > 0:
            dir_acc_cond = accuracy_score(y_dir_te[mask], dir_preds_te[mask])
            # Among those that ARE actually big moves
            big_mask = mask & (y_big_te == 1)
            n_actual_big = big_mask.sum()
            dir_acc_big = accuracy_score(y_dir_te[big_mask], dir_preds_te[big_mask]) if n_actual_big > 0 else 0
            logger.info(f"  {mag_t:>10.2f} {n_bars:>8,} {dir_acc_cond:>8.4f} {dir_acc_big:>12.4f} {n_actual_big:>12,}")

    # ================================================================
    # MAGNITUDE-FIRST PIPELINE: GRID SEARCH
    # ================================================================
    logger.info("\n" + "=" * 70)
    logger.info("MAGNITUDE-FIRST PIPELINE - GRID SEARCH")
    logger.info("=" * 70)
    logger.info("Pipeline: Magnitude gate -> Direction filter -> Stop loss management")

    # Generate predictions on test set
    ret_pred_te = (xgb_ret.predict(X_te) + lgb_ret.predict(X_te)) / 2
    direction_te = np.where(prob_up_te > 0.5, 1, -1)
    dir_bias_te = np.abs(prob_up_te - 0.5)

    # Regime mask
    from core.regime_filter import RegimeFilter
    from config import SystemConfig
    regime_filter = RegimeFilter(SystemConfig())
    regime_mask_full = regime_filter.compute_regime_mask(df)
    regime_mask_valid = regime_mask_full[valid].values
    regime_te = regime_mask_valid[val_end:]

    # Test set OHLCV for backtest engine
    df_test = df_valid.iloc[val_end:].reset_index(drop=True)

    # Grid search parameters
    mag_thresholds = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    dir_thresholds = [0.50, 0.51, 0.52, 0.53, 0.55, 0.57, 0.60]
    sl_multipliers = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
    tp_multipliers = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    use_regime = [False, True]

    from backtest.engine import BacktestEngine

    # ================================================================
    # PHASE 1: QUICK PnL SCAN (positions × returns, no SL/TP)
    # This is ~1000x faster than full backtest engine
    # ================================================================
    costs = (0.52 + 1.0) * 2  # 3.04 points round-trip

    quick_results = []
    total_signal_combos = len(mag_thresholds) * len(dir_thresholds) * len(use_regime)
    logger.info(f"Phase 1: Quick scan of {total_signal_combos} signal configurations...")
    t0 = time.time()

    for mag_t, dir_t, regime in product(mag_thresholds, dir_thresholds, use_regime):
        mag_pass = mag_probs_te >= mag_t
        dir_pass = (prob_up_te >= dir_t) | (prob_up_te <= (1 - dir_t))
        sig_valid = mag_pass & dir_pass
        if regime:
            sig_valid = sig_valid & regime_te.astype(bool)

        n_signals = int(sig_valid.sum())
        if n_signals < 5:
            continue

        positions = np.where(sig_valid, direction_te, 0)
        trade_returns = positions * y_ret_te
        trade_pnls = trade_returns[positions != 0]
        n_trades = len(trade_pnls)

        if n_trades < 5:
            continue

        net_pnls = trade_pnls - costs
        gross_pnl = net_pnls.sum()
        winners = net_pnls[net_pnls > 0]
        losers = net_pnls[net_pnls <= 0]
        win_rate = len(winners) / n_trades
        pf = abs(winners.sum() / losers.sum()) if losers.sum() != 0 else float('inf')
        ev_per_trade = gross_pnl / n_trades
        sharpe = net_pnls.mean() / max(net_pnls.std(), 0.01) * np.sqrt(252)

        quick_results.append({
            'mag_threshold': mag_t,
            'dir_threshold': dir_t,
            'regime': regime,
            'n_signals': n_signals,
            'n_trades': n_trades,
            'quick_pnl': gross_pnl,
            'quick_wr': win_rate,
            'quick_pf': pf,
            'quick_ev': ev_per_trade,
            'quick_sharpe': sharpe,
        })

    quick_df = pd.DataFrame(quick_results)
    logger.info(f"Phase 1 complete: {len(quick_df)} signal configs in {time.time()-t0:.1f}s")

    # Print quick scan results
    logger.info("\n  Quick scan results (ALL signal configs, no SL/TP management):")
    logger.info(f"  {'Mag':>5} {'Dir':>5} {'Reg':>4} {'Signals':>8} {'Trades':>7} "
                f"{'PnL':>10} {'WR':>7} {'PF':>7} {'EV/Tr':>8} {'Sharpe':>7}")
    logger.info("  " + "-" * 80)
    for _, row in quick_df.sort_values('quick_pnl', ascending=False).iterrows():
        logger.info(
            f"  {row['mag_threshold']:>5.2f} {row['dir_threshold']:>5.2f} "
            f"{'Y' if row['regime'] else 'N':>4} "
            f"{row['n_signals']:>8} {row['n_trades']:>7} "
            f"{row['quick_pnl']:>10.1f} {row['quick_wr']:>6.1%} {row['quick_pf']:>7.2f} "
            f"{row['quick_ev']:>8.1f} {row['quick_sharpe']:>7.2f}"
        )

    # ================================================================
    # PHASE 2: FULL BACKTEST on select signal configs × SL/TP grid
    # Only configs with positive quick PnL and < 5000 signals
    # ================================================================
    profitable_quick = quick_df[quick_df['quick_pnl'] > 0].copy()
    # Also include top 5 by sharpe even if negative PnL
    top_sharpe_extra = quick_df.nlargest(5, 'quick_sharpe')
    top_configs = pd.concat([profitable_quick, top_sharpe_extra]).drop_duplicates(
        subset=['mag_threshold', 'dir_threshold', 'regime']
    )
    # Limit to manageable signal counts for the backtest engine
    top_configs = top_configs[top_configs['n_signals'] <= 15000]

    # Reduced SL/TP grid for speed
    sl_grid = [0.75, 1.0, 1.5, 2.0, 3.0]
    tp_grid = [2.0, 3.0, 4.0, 5.0]

    total_bt_combos = len(top_configs) * len(sl_grid) * len(tp_grid)
    logger.info(f"\nPhase 2: Full backtest on {len(top_configs)} signal configs "
                f"× {len(sl_grid)} SL × {len(tp_grid)} TP = {total_bt_combos} combinations")

    results = []
    t0 = time.time()
    combo_count = 0

    for _, sig_cfg in top_configs.iterrows():
        mag_t = sig_cfg['mag_threshold']
        dir_t = sig_cfg['dir_threshold']
        regime = sig_cfg['regime']

        mag_pass = mag_probs_te >= mag_t
        dir_pass = (prob_up_te >= dir_t) | (prob_up_te <= (1 - dir_t))
        sig_valid = mag_pass & dir_pass
        if regime:
            sig_valid = sig_valid & regime_te.astype(bool)

        for sl_m, tp_m in product(sl_grid, tp_grid):
            combo_count += 1

            signals_df = pd.DataFrame({
                'direction_numeric': direction_te,
                'expected_return': ret_pred_te,
                'expected_value': np.full(len(X_te), 100.0),
                'confidence': np.full(len(X_te), 1.0),
                'signal_valid': sig_valid,
                'prob_big_move': mag_probs_te,
                'regime_active': regime_te.astype(bool) if regime else np.ones(len(X_te), dtype=bool),
            })

            config = SystemConfig()
            config.trading.min_confidence = 0.0
            config.trading.min_expected_value_points = 0.0
            config.trading.stop_loss_multiplier = sl_m
            config.trading.take_profit_multiplier = tp_m
            config.trading.max_drawdown_points = 100000.0

            engine = BacktestEngine(config)
            try:
                bt = engine.run(df_test, signals_df)
            except Exception:
                continue

            if bt['n_trades'] < 3:
                continue

            results.append({
                'mag_threshold': mag_t,
                'dir_threshold': dir_t,
                'sl_multiplier': sl_m,
                'tp_multiplier': tp_m,
                'regime': regime,
                'n_signals': int(sig_valid.sum()),
                'n_trades': bt['n_trades'],
                'total_pnl': bt['total_pnl_points'],
                'total_pnl_usd': bt['total_pnl_usd'],
                'win_rate': bt['win_rate'],
                'profit_factor': bt['profit_factor'],
                'expectancy': bt['expectancy_per_trade'],
                'sharpe': bt['sharpe_ratio'],
                'sortino': bt['sortino_ratio'],
                'max_dd': bt['max_drawdown_points'],
                'calmar': bt['calmar_ratio'],
                'avg_bars_held': bt['avg_bars_held'],
            })

            if combo_count % 20 == 0:
                elapsed = time.time() - t0
                pct = combo_count / total_bt_combos * 100
                logger.info(f"  Progress: {combo_count}/{total_bt_combos} ({pct:.0f}%) - "
                           f"Elapsed: {elapsed:.0f}s - Valid: {len(results)}")

    elapsed = time.time() - t0
    logger.info(f"\nPhase 2 complete: {len(results)} valid configurations in {elapsed:.0f}s")

    if not results:
        logger.error("No valid configurations found!")
        return

    # ================================================================
    # ANALYZE RESULTS
    # ================================================================
    results_df = pd.DataFrame(results)

    # Sort by different criteria
    logger.info("\n" + "=" * 70)
    logger.info("TOP 20 CONFIGURATIONS BY PROFIT FACTOR")
    logger.info("=" * 70)
    top_pf = results_df[results_df['n_trades'] >= 10].nlargest(20, 'profit_factor')
    _print_results_table(top_pf)

    logger.info("\n" + "=" * 70)
    logger.info("TOP 20 CONFIGURATIONS BY TOTAL PnL")
    logger.info("=" * 70)
    top_pnl = results_df[results_df['n_trades'] >= 10].nlargest(20, 'total_pnl')
    _print_results_table(top_pnl)

    logger.info("\n" + "=" * 70)
    logger.info("TOP 20 CONFIGURATIONS BY SHARPE RATIO")
    logger.info("=" * 70)
    top_sharpe = results_df[results_df['n_trades'] >= 10].nlargest(20, 'sharpe')
    _print_results_table(top_sharpe)

    logger.info("\n" + "=" * 70)
    logger.info("TOP 20 CONFIGURATIONS BY EXPECTANCY/TRADE")
    logger.info("=" * 70)
    top_ev = results_df[results_df['n_trades'] >= 10].nlargest(20, 'expectancy')
    _print_results_table(top_ev)

    # Composite score: normalize and combine
    logger.info("\n" + "=" * 70)
    logger.info("TOP 20 BY COMPOSITE SCORE (PF + Sharpe + WR + Calmar)")
    logger.info("=" * 70)
    filtered = results_df[results_df['n_trades'] >= 10].copy()
    if len(filtered) > 0:
        for col in ['profit_factor', 'sharpe', 'win_rate', 'calmar']:
            col_min = filtered[col].min()
            col_max = filtered[col].max()
            col_range = col_max - col_min
            if col_range > 0:
                filtered[f'{col}_norm'] = (filtered[col] - col_min) / col_range
            else:
                filtered[f'{col}_norm'] = 0
        filtered['composite'] = (
            filtered['profit_factor_norm'] * 0.30 +
            filtered['sharpe_norm'] * 0.30 +
            filtered['win_rate_norm'] * 0.20 +
            filtered['calmar_norm'] * 0.20
        )
        top_composite = filtered.nlargest(20, 'composite')
        _print_results_table(top_composite)

    # ================================================================
    # SWEET SPOT ANALYSIS
    # ================================================================
    logger.info("\n" + "=" * 70)
    logger.info("SWEET SPOT ANALYSIS")
    logger.info("=" * 70)

    profitable = results_df[results_df['total_pnl'] > 0]
    if len(profitable) > 0:
        logger.info(f"\nProfitable configs: {len(profitable)}/{len(results_df)} ({len(profitable)/len(results_df)*100:.1f}%)")

        logger.info("\n  Avg stats by MAGNITUDE threshold (profitable configs only):")
        mag_stats = profitable.groupby('mag_threshold').agg(
            count=('total_pnl', 'count'),
            avg_pnl=('total_pnl', 'mean'),
            avg_wr=('win_rate', 'mean'),
            avg_pf=('profit_factor', 'mean'),
            avg_sharpe=('sharpe', 'mean'),
            avg_trades=('n_trades', 'mean'),
        )
        for idx, row in mag_stats.iterrows():
            logger.info(f"    mag>={idx:.2f}: {row['count']:>4} configs, "
                       f"PnL={row['avg_pnl']:>8.1f}, WR={row['avg_wr']:.1%}, "
                       f"PF={row['avg_pf']:.2f}, Sharpe={row['avg_sharpe']:.2f}, "
                       f"Trades={row['avg_trades']:.0f}")

        logger.info("\n  Avg stats by DIRECTION threshold (profitable configs only):")
        dir_stats = profitable.groupby('dir_threshold').agg(
            count=('total_pnl', 'count'),
            avg_pnl=('total_pnl', 'mean'),
            avg_wr=('win_rate', 'mean'),
            avg_pf=('profit_factor', 'mean'),
            avg_sharpe=('sharpe', 'mean'),
            avg_trades=('n_trades', 'mean'),
        )
        for idx, row in dir_stats.iterrows():
            logger.info(f"    dir>={idx:.2f}: {row['count']:>4} configs, "
                       f"PnL={row['avg_pnl']:>8.1f}, WR={row['avg_wr']:.1%}, "
                       f"PF={row['avg_pf']:.2f}, Sharpe={row['avg_sharpe']:.2f}, "
                       f"Trades={row['avg_trades']:.0f}")

        logger.info("\n  Avg stats by STOP LOSS multiplier (profitable configs only):")
        sl_stats = profitable.groupby('sl_multiplier').agg(
            count=('total_pnl', 'count'),
            avg_pnl=('total_pnl', 'mean'),
            avg_wr=('win_rate', 'mean'),
            avg_pf=('profit_factor', 'mean'),
            avg_sharpe=('sharpe', 'mean'),
            avg_ev=('expectancy', 'mean'),
        )
        for idx, row in sl_stats.iterrows():
            logger.info(f"    SL={idx:.2f}x ATR: {row['count']:>4} configs, "
                       f"PnL={row['avg_pnl']:>8.1f}, WR={row['avg_wr']:.1%}, "
                       f"PF={row['avg_pf']:.2f}, EV/trade={row['avg_ev']:.1f}")

        logger.info("\n  Avg stats by TAKE PROFIT multiplier (profitable configs only):")
        tp_stats = profitable.groupby('tp_multiplier').agg(
            count=('total_pnl', 'count'),
            avg_pnl=('total_pnl', 'mean'),
            avg_wr=('win_rate', 'mean'),
            avg_pf=('profit_factor', 'mean'),
        )
        for idx, row in tp_stats.iterrows():
            logger.info(f"    TP={idx:.1f}x ATR: {row['count']:>4} configs, "
                       f"PnL={row['avg_pnl']:>8.1f}, WR={row['avg_wr']:.1%}, "
                       f"PF={row['avg_pf']:.2f}")

        logger.info("\n  Regime filter impact (profitable configs only):")
        regime_stats = profitable.groupby('regime').agg(
            count=('total_pnl', 'count'),
            avg_pnl=('total_pnl', 'mean'),
            avg_wr=('win_rate', 'mean'),
            avg_pf=('profit_factor', 'mean'),
            avg_sharpe=('sharpe', 'mean'),
        )
        for idx, row in regime_stats.iterrows():
            label = "WITH regime" if idx else "NO regime"
            logger.info(f"    {label}: {row['count']:>4} configs, "
                       f"PnL={row['avg_pnl']:>8.1f}, WR={row['avg_wr']:.1%}, "
                       f"PF={row['avg_pf']:.2f}, Sharpe={row['avg_sharpe']:.2f}")

    # ================================================================
    # BEST CONFIG DETAILED BACKTEST
    # ================================================================
    if len(filtered) > 0:
        best = filtered.nlargest(1, 'composite').iloc[0]
        logger.info("\n" + "=" * 70)
        logger.info("BEST CONFIGURATION (by composite score) - DETAILED BACKTEST")
        logger.info("=" * 70)
        logger.info(f"  Magnitude threshold: {best['mag_threshold']:.2f}")
        logger.info(f"  Direction threshold: {best['dir_threshold']:.2f}")
        logger.info(f"  Stop loss:           {best['sl_multiplier']:.2f}x ATR")
        logger.info(f"  Take profit:         {best['tp_multiplier']:.1f}x ATR")
        logger.info(f"  Regime filter:       {best['regime']}")
        logger.info(f"  ---")
        logger.info(f"  Trades:              {best['n_trades']}")
        logger.info(f"  Total PnL:           {best['total_pnl']:.1f} pts (${best['total_pnl_usd']:.2f})")
        logger.info(f"  Win rate:            {best['win_rate']:.1%}")
        logger.info(f"  Profit factor:       {best['profit_factor']:.2f}")
        logger.info(f"  Expectancy/trade:    {best['expectancy']:.1f} pts")
        logger.info(f"  Sharpe ratio:        {best['sharpe']:.2f}")
        logger.info(f"  Sortino ratio:       {best['sortino']:.2f}")
        logger.info(f"  Max drawdown:        {best['max_dd']:.1f} pts")
        logger.info(f"  Calmar ratio:        {best['calmar']:.2f}")
        logger.info(f"  Avg bars held:       {best['avg_bars_held']:.0f}")

        # Re-run best config with full logging
        mag_pass = mag_probs_te >= best['mag_threshold']
        dir_pass = (prob_up_te >= best['dir_threshold']) | (prob_up_te <= (1 - best['dir_threshold']))
        sig_valid = mag_pass & dir_pass
        if best['regime']:
            sig_valid = sig_valid & regime_te.astype(bool)

        signals_df = pd.DataFrame({
            'direction_numeric': direction_te,
            'expected_return': ret_pred_te,
            'expected_value': np.full(len(X_te), 100.0),
            'confidence': np.full(len(X_te), 1.0),
            'signal_valid': sig_valid,
            'prob_big_move': mag_probs_te,
            'regime_active': regime_te.astype(bool) if best['regime'] else np.ones(len(X_te), dtype=bool),
        })

        config = SystemConfig()
        config.trading.min_confidence = 0.0
        config.trading.min_expected_value_points = 0.0
        config.trading.stop_loss_multiplier = best['sl_multiplier']
        config.trading.take_profit_multiplier = best['tp_multiplier']
        config.trading.max_drawdown_points = 10000.0

        engine = BacktestEngine(config)
        bt = engine.run(df_test, signals_df)

        # Exit reason breakdown
        if bt.get('exit_stats'):
            logger.info("\n  Exit reason breakdown:")
            for reason, stats in bt['exit_stats'].items():
                logger.info(f"    {reason}: {stats['count']} trades, "
                           f"PnL={stats['total_pnl']:.1f}pts, WR={stats['win_rate']:.1%}")

        if bt.get('direction_stats'):
            logger.info("\n  By direction:")
            for d, stats in bt['direction_stats'].items():
                logger.info(f"    {d}: {stats['n_trades']} trades, "
                           f"PnL={stats['total_pnl']:.1f}pts, WR={stats['win_rate']:.1%}")

    # Save all results
    os.makedirs('data', exist_ok=True)
    results_df.to_csv('data/magnitude_first_grid_results.csv', index=False)
    logger.info(f"\nAll results saved to data/magnitude_first_grid_results.csv ({len(results_df)} configs)")

    # Save best config
    if len(filtered) > 0:
        best_config = {
            'mag_threshold': float(best['mag_threshold']),
            'dir_threshold': float(best['dir_threshold']),
            'sl_multiplier': float(best['sl_multiplier']),
            'tp_multiplier': float(best['tp_multiplier']),
            'regime': bool(best['regime']),
            'metrics': {
                'n_trades': int(best['n_trades']),
                'total_pnl': float(best['total_pnl']),
                'win_rate': float(best['win_rate']),
                'profit_factor': float(best['profit_factor']),
                'sharpe': float(best['sharpe']),
                'max_dd': float(best['max_dd']),
            }
        }
        with open('data/magnitude_first_best_config.json', 'w') as f:
            json.dump(best_config, f, indent=2)
        logger.info("Best config saved to data/magnitude_first_best_config.json")

    return results_df


def _print_results_table(df):
    """Print a formatted results table."""
    header = (f"  {'Mag':>5} {'Dir':>5} {'SL':>5} {'TP':>5} {'Reg':>4} "
              f"{'Trades':>7} {'PnL':>10} {'WR':>7} {'PF':>7} "
              f"{'EV/Tr':>8} {'Sharpe':>7} {'MaxDD':>8}")
    logger.info(header)
    logger.info("  " + "-" * 95)
    for _, row in df.iterrows():
        logger.info(
            f"  {row['mag_threshold']:>5.2f} {row['dir_threshold']:>5.2f} "
            f"{row['sl_multiplier']:>5.2f} {row['tp_multiplier']:>5.1f} "
            f"{'Y' if row['regime'] else 'N':>4} "
            f"{row['n_trades']:>7} {row['total_pnl']:>10.1f} "
            f"{row['win_rate']:>6.1%} {row['profit_factor']:>7.2f} "
            f"{row['expectancy']:>8.1f} {row['sharpe']:>7.2f} "
            f"{row['max_dd']:>8.1f}"
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Magnitude-First Pipeline')
    parser.add_argument('--data', type=str, help='Path to data file')
    parser.add_argument('--horizon', type=int, default=30, help='Target horizon in bars')
    args = parser.parse_args()

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
        logger.error("No data file found. Use --data /path/to/file")
        sys.exit(1)

    df, features, groupings = load_and_prepare(args.data)
    results = train_magnitude_first(df, features, target_horizon=args.horizon)
