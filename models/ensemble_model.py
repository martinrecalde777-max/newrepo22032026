"""
Ensemble Predictive Model
==========================
Combina XGBoost, LightGBM, y modelos estadísticos en un ensemble
ponderado por rendimiento out-of-sample.

Incluye:
- Game Theory: Nash equilibrium para decisiones de trading
  (modelar el mercado como juego adversarial)
- Expected Value filtering (>= 50 pts)
- Walk-forward validation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from scipy import stats
import joblib
import logging
import warnings
import os

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class GameTheoryModule:
    """
    Game Theory applied to trading decisions.
    Models the market as an adversarial game where:
    - Player 1 (Trader): chooses LONG, SHORT, or FLAT
    - Player 2 (Market): moves UP, DOWN, or SIDEWAYS

    Uses minimax regret and Nash equilibrium concepts.
    """

    def __init__(self, config):
        self.config = config
        self.payoff_matrix = None
        self.nash_equilibrium = None

    def compute_payoff_matrix(self, predictions: pd.DataFrame,
                               actual_returns: pd.Series) -> np.ndarray:
        """
        Compute empirical payoff matrix from historical predictions vs outcomes.

        Rows: Trader actions (LONG, SHORT, FLAT)
        Cols: Market outcomes (UP_BIG, UP_SMALL, FLAT, DOWN_SMALL, DOWN_BIG)
        """
        min_ev = self.config.trading.min_expected_value_points
        commission = self.config.trading.commission_per_side * 2
        slippage = self.config.trading.slippage_per_side * 2
        costs = commission + slippage

        # Discretize market outcomes
        thresholds = [-min_ev, -min_ev / 2, min_ev / 2, min_ev]
        market_states = np.digitize(actual_returns.values, thresholds)

        # Payoff matrix: 3 actions x 5 market states
        payoff = np.zeros((3, 5))  # LONG, SHORT, FLAT x 5 states
        counts = np.zeros((3, 5))

        for i in range(len(actual_returns)):
            ret = actual_returns.iloc[i]
            state = min(market_states[i], 4)

            # LONG payoff
            payoff[0, state] += ret - costs
            counts[0, state] += 1

            # SHORT payoff
            payoff[1, state] += -ret - costs
            counts[1, state] += 1

            # FLAT payoff (0 minus opportunity cost estimate)
            payoff[2, state] += 0
            counts[2, state] += 1

        # Average payoffs
        with np.errstate(divide='ignore', invalid='ignore'):
            payoff = np.where(counts > 0, payoff / counts, 0)

        self.payoff_matrix = payoff
        return payoff

    def compute_nash_equilibrium(self) -> Dict:
        """
        Compute mixed-strategy Nash equilibrium for the trading game.
        Uses linear programming approach for 2-player zero-sum approximation.
        """
        if self.payoff_matrix is None:
            return {'strategy': 'FLAT', 'confidence': 0.0}

        payoff = self.payoff_matrix

        # For each market state, find best action (maximin strategy)
        # Market plays to minimize trader's payoff (adversarial)
        trader_min_payoffs = payoff.min(axis=1)  # Worst case for each action
        best_action = np.argmax(trader_min_payoffs)  # Maximin

        # Mixed strategy: softmax over expected payoffs
        expected_payoffs = payoff.mean(axis=1)
        temperature = 1.0
        exp_payoffs = np.exp(expected_payoffs / max(temperature, 0.01))
        mixed_strategy = exp_payoffs / exp_payoffs.sum()

        actions = ['LONG', 'SHORT', 'FLAT']

        self.nash_equilibrium = {
            'pure_strategy': actions[best_action],
            'mixed_strategy': {actions[i]: float(mixed_strategy[i]) for i in range(3)},
            'expected_payoffs': {actions[i]: float(expected_payoffs[i]) for i in range(3)},
            'maximin_payoff': float(trader_min_payoffs[best_action]),
        }

        return self.nash_equilibrium

    def filter_by_game_theory(self, signal: Dict) -> Dict:
        """Apply game theory filter to a trading signal."""
        if self.nash_equilibrium is None:
            return signal

        direction = signal.get('direction', 'FLAT')
        expected_return = signal.get('expected_return', 0)

        # Check if direction aligns with Nash equilibrium
        nash_payoff = self.nash_equilibrium['expected_payoffs'].get(direction, 0)
        nash_prob = self.nash_equilibrium['mixed_strategy'].get(direction, 0)

        signal['nash_payoff'] = nash_payoff
        signal['nash_probability'] = nash_prob
        signal['nash_aligned'] = nash_prob > 0.3  # Action has >30% in mixed strategy

        return signal


class EnsemblePredictor:
    """
    Ensemble model combining XGBoost, LightGBM, and statistical models.
    Uses walk-forward validation for robust out-of-sample testing.
    """

    def __init__(self, config):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.model_weights = {}
        self.feature_importance = None
        self.game_theory = GameTheoryModule(config)
        self.is_fitted = False
        self.metrics = {}

    def fit(self, df: pd.DataFrame, features: pd.DataFrame,
            target_horizon: int = 60) -> 'EnsemblePredictor':
        """
        Fit ensemble models with walk-forward validation.
        """
        logger.info("=" * 60)
        logger.info("FITTING ENSEMBLE PREDICTIVE MODELS")
        logger.info("=" * 60)

        # Prepare target
        target_return = df['close'].shift(-target_horizon) - df['close']
        target_direction = (target_return > 0).astype(int)
        target_magnitude = target_return.abs()

        # Align and clean
        valid_mask = features.notna().all(axis=1) & target_return.notna()
        X = features[valid_mask].values
        y_return = target_return[valid_mask].values
        y_direction = target_direction[valid_mask].values
        y_magnitude = target_magnitude[valid_mask].values

        feature_names = features.columns.tolist()

        logger.info(f"Training samples: {len(X):,}")
        logger.info(f"Features: {len(feature_names)}")

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['main'] = scaler

        # Walk-forward split
        n = len(X)
        train_size = int(n * self.config.model.walk_forward_train_pct)

        # Use last portion for validation
        X_train, X_val = X_scaled[:train_size], X_scaled[train_size:]
        y_dir_train, y_dir_val = y_direction[:train_size], y_direction[train_size:]
        y_ret_train, y_ret_val = y_return[:train_size], y_return[train_size:]
        y_mag_train, y_mag_val = y_magnitude[:train_size], y_magnitude[train_size:]

        logger.info(f"Walk-forward: Train={len(X_train):,} | Val={len(X_val):,}")

        # 1. XGBoost Direction Classifier
        logger.info("Training XGBoost direction classifier...")
        xgb_dir = xgb.XGBClassifier(
            n_estimators=self.config.model.xgb_n_estimators,
            max_depth=self.config.model.xgb_max_depth,
            learning_rate=self.config.model.xgb_learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss',
            early_stopping_rounds=50,
            verbosity=0,
        )
        xgb_dir.fit(X_train, y_dir_train,
                     eval_set=[(X_val, y_dir_val)],
                     verbose=False)
        self.models['xgb_direction'] = xgb_dir

        # 2. XGBoost Return Regressor
        logger.info("Training XGBoost return regressor...")
        xgb_ret = xgb.XGBRegressor(
            n_estimators=self.config.model.xgb_n_estimators,
            max_depth=self.config.model.xgb_max_depth,
            learning_rate=self.config.model.xgb_learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='rmse',
            early_stopping_rounds=50,
            verbosity=0,
        )
        xgb_ret.fit(X_train, y_ret_train,
                     eval_set=[(X_val, y_ret_val)],
                     verbose=False)
        self.models['xgb_return'] = xgb_ret

        # 3. LightGBM Direction Classifier
        logger.info("Training LightGBM direction classifier...")
        lgb_dir = lgb.LGBMClassifier(
            n_estimators=self.config.model.lgb_n_estimators,
            max_depth=self.config.model.lgb_max_depth,
            learning_rate=self.config.model.lgb_learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
        lgb_dir.fit(X_train, y_dir_train,
                     eval_set=[(X_val, y_dir_val)],
                     callbacks=[lgb.early_stopping(50, verbose=False)])
        self.models['lgb_direction'] = lgb_dir

        # 4. LightGBM Return Regressor
        logger.info("Training LightGBM return regressor...")
        lgb_ret = lgb.LGBMRegressor(
            n_estimators=self.config.model.lgb_n_estimators,
            max_depth=self.config.model.lgb_max_depth,
            learning_rate=self.config.model.lgb_learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
        lgb_ret.fit(X_train, y_ret_train,
                     eval_set=[(X_val, y_ret_val)],
                     callbacks=[lgb.early_stopping(50, verbose=False)])
        self.models['lgb_return'] = lgb_ret

        # 5. Magnitude classifier (is move >= 50 points?)
        logger.info("Training magnitude classifier (>=50 pts)...")
        y_big_move_train = (y_mag_train >= self.config.trading.min_expected_value_points).astype(int)
        y_big_move_val = (y_mag_val >= self.config.trading.min_expected_value_points).astype(int)

        xgb_mag = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
            eval_metric='logloss',
            early_stopping_rounds=30,
            verbosity=0,
        )
        xgb_mag.fit(X_train, y_big_move_train,
                     eval_set=[(X_val, y_big_move_val)],
                     verbose=False)
        self.models['xgb_magnitude'] = xgb_mag

        # Evaluate models
        self._evaluate_models(X_val, y_dir_val, y_ret_val, y_mag_val)

        # Feature importance (average across models)
        self._compute_feature_importance(feature_names)

        # Compute model weights based on validation performance
        self._compute_model_weights(X_val, y_dir_val, y_ret_val)

        # Game theory module
        val_predictions = self._raw_predict(X_val)
        val_returns = pd.Series(y_ret_val)
        self.game_theory.compute_payoff_matrix(val_predictions, val_returns)
        self.game_theory.compute_nash_equilibrium()

        if self.game_theory.nash_equilibrium:
            logger.info(f"Nash Equilibrium: {self.game_theory.nash_equilibrium['pure_strategy']}")
            logger.info(f"Mixed strategy: {self.game_theory.nash_equilibrium['mixed_strategy']}")

        self.is_fitted = True
        return self

    def _evaluate_models(self, X_val, y_dir_val, y_ret_val, y_mag_val):
        """Evaluate all models on validation set."""
        logger.info("\n--- Model Evaluation (Validation Set) ---")

        # Direction accuracy
        for name in ['xgb_direction', 'lgb_direction']:
            model = self.models[name]
            preds = model.predict(X_val)
            acc = accuracy_score(y_dir_val, preds)
            self.metrics[f'{name}_accuracy'] = acc
            logger.info(f"  {name}: accuracy={acc:.4f}")

        # Return prediction
        for name in ['xgb_return', 'lgb_return']:
            model = self.models[name]
            preds = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_ret_val, preds))
            mae = mean_absolute_error(y_ret_val, preds)
            # Directional accuracy of return predictor
            dir_acc = ((preds > 0) == (y_ret_val > 0)).mean()
            self.metrics[f'{name}_rmse'] = rmse
            self.metrics[f'{name}_mae'] = mae
            self.metrics[f'{name}_dir_accuracy'] = dir_acc
            logger.info(f"  {name}: RMSE={rmse:.2f}, MAE={mae:.2f}, DirAcc={dir_acc:.4f}")

        # Magnitude classifier
        mag_preds = self.models['xgb_magnitude'].predict(X_val)
        y_big_move = (y_mag_val >= self.config.trading.min_expected_value_points).astype(int)
        mag_acc = accuracy_score(y_big_move, mag_preds)
        self.metrics['magnitude_accuracy'] = mag_acc
        logger.info(f"  Magnitude (>=50pts) classifier: accuracy={mag_acc:.4f}")

    def _compute_feature_importance(self, feature_names: List[str]):
        """Compute averaged feature importance across tree models."""
        importances = {}

        for name in ['xgb_direction', 'lgb_direction', 'xgb_return', 'lgb_return']:
            model = self.models[name]
            if hasattr(model, 'feature_importances_'):
                imp = model.feature_importances_
                for i, fname in enumerate(feature_names[:len(imp)]):
                    if fname not in importances:
                        importances[fname] = []
                    importances[fname].append(imp[i])

        # Average importance
        avg_importance = {k: np.mean(v) for k, v in importances.items()}
        self.feature_importance = pd.Series(avg_importance).sort_values(ascending=False)

        logger.info("\nTop 15 features by importance:")
        for feat, imp in self.feature_importance.head(15).items():
            logger.info(f"  {feat}: {imp:.4f}")

    def _compute_model_weights(self, X_val, y_dir_val, y_ret_val):
        """Compute ensemble weights based on validation performance."""
        # Direction model weights (by accuracy)
        dir_accs = {}
        for name in ['xgb_direction', 'lgb_direction']:
            preds = self.models[name].predict(X_val)
            dir_accs[name] = accuracy_score(y_dir_val, preds)

        total = sum(dir_accs.values())
        self.model_weights['direction'] = {k: v / total for k, v in dir_accs.items()}

        # Return model weights (by inverse RMSE)
        ret_scores = {}
        for name in ['xgb_return', 'lgb_return']:
            preds = self.models[name].predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_ret_val, preds))
            ret_scores[name] = 1.0 / max(rmse, 0.01)

        total = sum(ret_scores.values())
        self.model_weights['return'] = {k: v / total for k, v in ret_scores.items()}

        logger.info(f"\nEnsemble weights:")
        logger.info(f"  Direction: {self.model_weights['direction']}")
        logger.info(f"  Return: {self.model_weights['return']}")

    def _raw_predict(self, X: np.ndarray) -> pd.DataFrame:
        """Raw predictions without game theory filter."""
        results = pd.DataFrame()

        # Direction probabilities (ensemble)
        dir_probs = np.zeros(len(X))
        for name, weight in self.model_weights.get('direction', {}).items():
            model = self.models[name]
            probs = model.predict_proba(X)[:, 1]  # P(up)
            dir_probs += probs * weight

        results['prob_up'] = dir_probs
        results['direction'] = np.where(dir_probs > 0.5, 'LONG', 'SHORT')
        results['direction_numeric'] = np.where(dir_probs > 0.5, 1, -1)

        # Return prediction (ensemble)
        ret_pred = np.zeros(len(X))
        for name, weight in self.model_weights.get('return', {}).items():
            model = self.models[name]
            preds = model.predict(X)
            ret_pred += preds * weight

        results['expected_return'] = ret_pred

        # Magnitude probability
        results['prob_big_move'] = self.models['xgb_magnitude'].predict_proba(X)[:, 1]

        # Confidence
        results['confidence'] = (
            np.abs(dir_probs - 0.5) * 2 *  # Direction certainty
            results['prob_big_move']          # Magnitude certainty
        )

        return results

    def predict(self, features: pd.DataFrame,
                regime_mask: pd.Series = None) -> pd.DataFrame:
        """
        Full prediction pipeline with game theory, EV filtering,
        magnitude gate (Improvement 4), and regime filter (Improvement 3).
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        X = self.scalers['main'].transform(features.values)
        results = self._raw_predict(X)
        results.index = features.index

        # Expected Value calculation
        costs = (self.config.trading.commission_per_side + self.config.trading.slippage_per_side) * 2
        results['expected_value'] = (
            results['expected_return'].abs() * results['confidence'] - costs
        )

        # Base filter: EV and confidence
        min_ev = self.config.trading.min_expected_value_points
        results['signal_valid'] = (
            (results['expected_value'] >= min_ev) &
            (results['confidence'] >= self.config.trading.min_confidence)
        )

        # === IMPROVEMENT 4: Magnitude classifier as hard gate ===
        if getattr(self.config.trading, 'use_magnitude_filter', True):
            mag_threshold = getattr(self.config.trading, 'magnitude_prob_threshold', 0.5)
            results['magnitude_gate'] = results['prob_big_move'] >= mag_threshold
            n_before = results['signal_valid'].sum()
            results['signal_valid'] = results['signal_valid'] & results['magnitude_gate']
            n_after = results['signal_valid'].sum()
            logger.info(f"Magnitude filter (>={mag_threshold}): {n_before} -> {n_after} valid signals")

        # === IMPROVEMENT 3: Regime filter ===
        if regime_mask is not None:
            aligned_mask = regime_mask.reindex(results.index, fill_value=False)
            results['regime_active'] = aligned_mask
            n_before = results['signal_valid'].sum()
            results['signal_valid'] = results['signal_valid'] & aligned_mask.astype(bool)
            n_after = results['signal_valid'].sum()
            logger.info(f"Regime filter: {n_before} -> {n_after} valid signals")
        else:
            results['regime_active'] = True

        # Game theory alignment
        if self.game_theory.nash_equilibrium:
            for idx in results.index:
                signal = results.loc[idx].to_dict()
                signal = self.game_theory.filter_by_game_theory(signal)
                results.loc[idx, 'nash_aligned'] = signal.get('nash_aligned', False)
                results.loc[idx, 'nash_probability'] = signal.get('nash_probability', 0)
        else:
            results['nash_aligned'] = True
            results['nash_probability'] = 0.5

        return results

    def walk_forward_validation(self, df: pd.DataFrame, features: pd.DataFrame,
                                 target_horizon: int = 60) -> pd.DataFrame:
        """
        Walk-forward out-of-sample validation.
        This is the gold standard for evaluating trading models.
        """
        logger.info("=" * 60)
        logger.info("WALK-FORWARD VALIDATION")
        logger.info("=" * 60)

        n_windows = self.config.model.walk_forward_windows
        train_pct = self.config.model.walk_forward_train_pct

        target_return = df['close'].shift(-target_horizon) - df['close']
        valid_mask = features.notna().all(axis=1) & target_return.notna()

        X = features[valid_mask].values
        y = target_return[valid_mask].values
        y_dir = (y > 0).astype(int)

        n = len(X)
        window_size = n // n_windows
        all_predictions = []
        all_actuals = []
        window_metrics = []

        for w in range(n_windows - 1):
            train_end = (w + 1) * window_size
            test_start = train_end
            test_end = min(test_start + window_size, n)

            if test_end <= test_start:
                continue

            X_train = X[:train_end]
            y_train = y[:train_end]
            y_dir_train = y_dir[:train_end]
            X_test = X[test_start:test_end]
            y_test = y[test_start:test_end]
            y_dir_test = y_dir[test_start:test_end]

            # Scale
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            # Train quick models for this window
            try:
                xgb_model = xgb.XGBClassifier(
                    n_estimators=200, max_depth=5, learning_rate=0.05,
                    random_state=42, verbosity=0
                )
                xgb_model.fit(X_train_s, y_dir_train)
                preds = xgb_model.predict(X_test_s)
                probs = xgb_model.predict_proba(X_test_s)[:, 1]

                acc = accuracy_score(y_dir_test, preds)

                # Simulated PnL
                positions = np.where(probs > 0.55, 1, np.where(probs < 0.45, -1, 0))
                pnl = positions * y_test
                total_pnl = pnl.sum()
                sharpe = pnl.mean() / max(pnl.std(), 0.01) * np.sqrt(252 * 390)  # annualized

                window_metrics.append({
                    'window': w,
                    'train_size': len(X_train),
                    'test_size': len(X_test),
                    'accuracy': acc,
                    'total_pnl': total_pnl,
                    'sharpe': sharpe,
                    'n_trades': int((positions != 0).sum()),
                })

                all_predictions.extend(probs.tolist())
                all_actuals.extend(y_dir_test.tolist())

                logger.info(f"  Window {w}: acc={acc:.4f}, PnL={total_pnl:.1f}pts, "
                           f"Sharpe={sharpe:.2f}, trades={int((positions != 0).sum())}")
            except Exception as e:
                logger.error(f"  Window {w} failed: {e}")

        wf_results = pd.DataFrame(window_metrics)
        if len(wf_results) > 0:
            logger.info(f"\nWalk-Forward Summary:")
            logger.info(f"  Mean accuracy: {wf_results['accuracy'].mean():.4f}")
            logger.info(f"  Mean PnL/window: {wf_results['total_pnl'].mean():.1f} pts")
            logger.info(f"  Mean Sharpe: {wf_results['sharpe'].mean():.2f}")
            logger.info(f"  Win rate (profitable windows): "
                       f"{(wf_results['total_pnl'] > 0).mean():.1%}")

        self.metrics['walk_forward'] = wf_results
        return wf_results

    def save_models(self, path: str):
        """Save all trained models to disk."""
        os.makedirs(path, exist_ok=True)
        for name, model in self.models.items():
            joblib.dump(model, os.path.join(path, f'{name}.pkl'))
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, os.path.join(path, f'scaler_{name}.pkl'))
        joblib.dump(self.model_weights, os.path.join(path, 'model_weights.pkl'))
        joblib.dump(self.feature_importance, os.path.join(path, 'feature_importance.pkl'))
        logger.info(f"Models saved to {path}")

    def load_models(self, path: str):
        """Load trained models from disk."""
        for name in ['xgb_direction', 'xgb_return', 'lgb_direction', 'lgb_return', 'xgb_magnitude']:
            fpath = os.path.join(path, f'{name}.pkl')
            if os.path.exists(fpath):
                self.models[name] = joblib.load(fpath)

        scaler_path = os.path.join(path, 'scaler_main.pkl')
        if os.path.exists(scaler_path):
            self.scalers['main'] = joblib.load(scaler_path)

        weights_path = os.path.join(path, 'model_weights.pkl')
        if os.path.exists(weights_path):
            self.model_weights = joblib.load(weights_path)

        self.is_fitted = True
        logger.info(f"Models loaded from {path}")
