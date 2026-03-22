"""
Markov Chain & Hidden Markov Model
====================================
Implementación de cadenas de Markov y HMM para detección de regímenes
de mercado y predicción de transiciones de estado.

- Cadenas de Markov de orden N para secuencias de estados discretos
- Hidden Markov Model para detección de regímenes latentes
- Transiciones condicionales con parámetros de mercado
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from sklearn.preprocessing import KBinsDiscretizer
from scipy import stats
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    logger.warning("hmmlearn not available. HMM features will be disabled.")


class MarkovChainPredictor:
    """
    N-th order Markov Chain for market state prediction.
    States are discretized market conditions (direction + magnitude + volatility regime).
    """

    def __init__(self, config):
        self.config = config
        self.n_states = config.model.n_markov_states
        self.lookback = config.model.markov_lookback

        # Transition matrices for different orders
        self.transition_counts = {}  # {order: {state_tuple: {next_state: count}}}
        self.transition_probs = {}   # {order: {state_tuple: {next_state: prob}}}
        self.state_returns = {}      # {state: [returns]}

        # State encoding
        self.state_encoder = None
        self.state_labels = None

        self.is_fitted = False

    def _discretize_states(self, df: pd.DataFrame) -> pd.Series:
        """Discretize market conditions into N states."""
        # Features for state definition
        returns = df['return_points'].fillna(0)
        volatility = df['volatility_20'].fillna(df['volatility_20'].median())
        volume_ratio = df['volume_ratio'].fillna(1.0)

        # Create composite state variable
        features = pd.DataFrame({
            'return': returns,
            'volatility': volatility,
            'volume_ratio': volume_ratio
        }).dropna()

        # Discretize using quantile-based binning
        try:
            discretizer = KBinsDiscretizer(
                n_bins=self.n_states, encode='ordinal', strategy='quantile'
            )
            # Use return as primary state variable
            states = discretizer.fit_transform(features[['return']].values).ravel().astype(int)
            self.state_encoder = discretizer

            # Create state labels
            edges = discretizer.bin_edges_[0]
            self.state_labels = {}
            for i in range(self.n_states):
                lo = edges[i]
                hi = edges[i + 1]
                self.state_labels[i] = f"S{i}[{lo:.1f},{hi:.1f}]"

        except Exception:
            # Fallback: simple sign-based states
            states = np.digitize(
                features['return'].values,
                bins=np.percentile(features['return'].values,
                                  np.linspace(0, 100, self.n_states + 1)[1:-1])
            )
            self.state_labels = {i: f"S{i}" for i in range(self.n_states)}

        result = pd.Series(np.nan, index=df.index)
        result.iloc[features.index] = states
        return result.astype('Int64')

    def fit(self, df: pd.DataFrame, target_horizon: int = 60) -> 'MarkovChainPredictor':
        """Fit Markov chain transition matrices from data."""
        logger.info(f"Fitting Markov Chain (N_states={self.n_states}, lookback={self.lookback})")

        states = self._discretize_states(df)
        target = df['close'].shift(-target_horizon) - df['close']

        valid = states.notna() & target.notna()
        states_clean = states[valid].values
        target_clean = target[valid].values

        # Build transition matrices for orders 1 through lookback
        for order in range(1, self.lookback + 1):
            counts = defaultdict(lambda: defaultdict(int))
            returns_by_transition = defaultdict(list)

            for i in range(order, len(states_clean)):
                history = tuple(states_clean[i - order:i])
                next_state = states_clean[i]

                if any(pd.isna(h) for h in history) or pd.isna(next_state):
                    continue

                counts[history][next_state] += 1
                returns_by_transition[(history, next_state)].append(target_clean[i])

            # Convert counts to probabilities
            probs = {}
            for history, next_counts in counts.items():
                total = sum(next_counts.values())
                probs[history] = {s: c / total for s, c in next_counts.items()}

            self.transition_counts[order] = dict(counts)
            self.transition_probs[order] = probs

            logger.info(f"  Order {order}: {len(probs)} unique state sequences")

        # State-level return statistics
        for state in range(self.n_states):
            mask = states_clean == state
            if mask.any():
                self.state_returns[state] = {
                    'mean_return': float(np.mean(target_clean[mask])),
                    'std_return': float(np.std(target_clean[mask])),
                    'up_prob': float((target_clean[mask] > 0).mean()),
                    'n': int(mask.sum())
                }

        self.is_fitted = True
        return self

    def predict(self, recent_states: List[int]) -> Dict:
        """
        Predict next state and expected return from recent state history.

        Args:
            recent_states: List of recent state values (most recent last)

        Returns:
            Prediction dict with probabilities and expected returns
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        predictions = {}

        # Try each order from highest to lowest
        for order in range(min(self.lookback, len(recent_states)), 0, -1):
            history = tuple(recent_states[-order:])

            if order in self.transition_probs and history in self.transition_probs[order]:
                probs = self.transition_probs[order][history]

                # Expected next state distribution
                expected_return = 0.0
                for state, prob in probs.items():
                    if state in self.state_returns:
                        expected_return += prob * self.state_returns[state]['mean_return']

                # Most likely next state
                most_likely = max(probs, key=probs.get)
                confidence = probs[most_likely]

                predictions[f'order_{order}'] = {
                    'next_state_probs': probs,
                    'most_likely_state': most_likely,
                    'state_confidence': confidence,
                    'expected_return': expected_return,
                }

        if not predictions:
            return {
                'expected_return': 0.0,
                'confidence': 0.0,
                'direction': 'FLAT',
                'direction_numeric': 0,
            }

        # Ensemble across orders (weight higher orders more)
        total_weight = 0
        weighted_return = 0
        weighted_confidence = 0

        for order_key, pred in predictions.items():
            order = int(order_key.split('_')[1])
            weight = order  # Higher order = more weight
            weighted_return += pred['expected_return'] * weight
            weighted_confidence += pred['state_confidence'] * weight
            total_weight += weight

        if total_weight > 0:
            expected_return = weighted_return / total_weight
            confidence = weighted_confidence / total_weight
        else:
            expected_return = 0.0
            confidence = 0.0

        direction = 'LONG' if expected_return > 0 else 'SHORT' if expected_return < 0 else 'FLAT'

        return {
            'expected_return': expected_return,
            'confidence': confidence,
            'direction': direction,
            'direction_numeric': 1 if expected_return > 0 else -1 if expected_return < 0 else 0,
            'order_predictions': predictions,
        }

    def batch_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions for all rows."""
        states = self._discretize_states(df)
        results = []

        for i in range(len(df)):
            start = max(0, i - self.lookback)
            recent = states.iloc[start:i + 1].dropna().values.tolist()

            if len(recent) < 1:
                results.append({
                    'markov_expected_return': 0.0,
                    'markov_confidence': 0.0,
                    'markov_direction': 0,
                })
            else:
                pred = self.predict([int(s) for s in recent])
                results.append({
                    'markov_expected_return': pred['expected_return'],
                    'markov_confidence': pred['confidence'],
                    'markov_direction': pred['direction_numeric'],
                })

        return pd.DataFrame(results, index=df.index)


class HMMRegimeDetector:
    """
    Hidden Markov Model for latent market regime detection.
    Identifies underlying market states (trending, ranging, volatile, quiet)
    from observable features.
    """

    def __init__(self, config):
        self.config = config
        self.n_states = config.model.n_hmm_states
        self.model = None
        self.regime_stats = {}
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> 'HMMRegimeDetector':
        """Fit HMM to detect latent regimes."""
        if not HMM_AVAILABLE:
            logger.warning("HMM not available, skipping")
            return self

        logger.info(f"Fitting HMM regime detector (n_states={self.n_states})")

        # Observable features
        features = pd.DataFrame({
            'return': df['return_log'].fillna(0),
            'volatility': df['volatility_20'].fillna(method='ffill').fillna(0),
            'volume_ratio': df['volume_ratio'].fillna(1),
            'range_norm': (df['range'] / df['close']).fillna(0),
        })

        # Remove inf
        features = features.replace([np.inf, -np.inf], 0)
        valid_mask = features.notna().all(axis=1)
        X = features[valid_mask].values

        if len(X) < 1000:
            logger.warning("Insufficient data for HMM fitting")
            return self

        # Subsample for fitting if too large (HMM can be slow)
        if len(X) > 500000:
            # Use every Nth sample to get ~500k
            step = len(X) // 500000
            X_fit = X[::step]
        else:
            X_fit = X

        # Fit Gaussian HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=self.config.model.hmm_iterations,
            random_state=42,
            verbose=False
        )

        try:
            self.model.fit(X_fit)
        except Exception as e:
            logger.error(f"HMM fitting failed: {e}")
            return self

        # Decode states for full dataset
        states = pd.Series(np.nan, index=df.index)
        try:
            hidden_states = self.model.predict(X)
            states.iloc[valid_mask.values.nonzero()[0]] = hidden_states
        except Exception as e:
            logger.error(f"HMM prediction failed: {e}")
            return self

        # Compute regime statistics
        target = df['close'].diff(60)  # 60-bar forward return
        for state in range(self.n_states):
            mask = states == state
            if mask.sum() < 100:
                continue

            state_returns = target[mask].dropna()
            state_vol = df['volatility_20'][mask].dropna()

            self.regime_stats[state] = {
                'n_bars': int(mask.sum()),
                'pct_time': float(mask.mean()),
                'mean_return': float(state_returns.mean()) if len(state_returns) > 0 else 0,
                'std_return': float(state_returns.std()) if len(state_returns) > 0 else 0,
                'mean_volatility': float(state_vol.mean()) if len(state_vol) > 0 else 0,
                'up_probability': float((state_returns > 0).mean()) if len(state_returns) > 0 else 0.5,
            }

        self.is_fitted = True

        # Log regime summary
        for state, s in self.regime_stats.items():
            label = self._get_regime_label(s)
            logger.info(f"  Regime {state} ({label}): "
                       f"{s['pct_time']:.1%} of time, "
                       f"mean_ret={s['mean_return']:.2f}, "
                       f"vol={s['mean_volatility']:.4f}")

        return self

    def _get_regime_label(self, stats: Dict) -> str:
        """Label a regime based on its statistics."""
        vol = stats['mean_volatility']
        ret = stats['mean_return']
        up_prob = stats['up_probability']

        if vol > 0 and stats.get('std_return', 1) > 0:
            vol_regime = 'high_vol' if vol > 0.001 else 'low_vol'
        else:
            vol_regime = 'normal'

        if up_prob > 0.55:
            trend = 'bullish'
        elif up_prob < 0.45:
            trend = 'bearish'
        else:
            trend = 'neutral'

        return f"{trend}_{vol_regime}"

    def predict_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict current regime for each bar."""
        result = pd.DataFrame(index=df.index)

        if not self.is_fitted or self.model is None:
            result['hmm_regime'] = -1
            result['hmm_regime_prob'] = 0.0
            return result

        features = pd.DataFrame({
            'return': df['return_log'].fillna(0),
            'volatility': df['volatility_20'].fillna(method='ffill').fillna(0),
            'volume_ratio': df['volume_ratio'].fillna(1),
            'range_norm': (df['range'] / df['close']).fillna(0),
        }).replace([np.inf, -np.inf], 0)

        valid_mask = features.notna().all(axis=1)
        X = features[valid_mask].values

        try:
            states = self.model.predict(X)
            probs = self.model.predict_proba(X)

            result['hmm_regime'] = np.nan
            result['hmm_regime_prob'] = np.nan
            valid_idx = valid_mask.values.nonzero()[0]

            result.iloc[valid_idx, result.columns.get_loc('hmm_regime')] = states
            result.iloc[valid_idx, result.columns.get_loc('hmm_regime_prob')] = probs.max(axis=1)

            # Add regime-based expected return
            result['hmm_expected_return'] = result['hmm_regime'].map(
                {s: stats['mean_return'] for s, stats in self.regime_stats.items()}
            )
            result['hmm_up_prob'] = result['hmm_regime'].map(
                {s: stats['up_probability'] for s, stats in self.regime_stats.items()}
            )

        except Exception as e:
            logger.error(f"HMM prediction error: {e}")
            result['hmm_regime'] = -1
            result['hmm_regime_prob'] = 0.0
            result['hmm_expected_return'] = 0.0
            result['hmm_up_prob'] = 0.5

        return result
