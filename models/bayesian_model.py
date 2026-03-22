"""
Bayesian Inference Model
========================
Modelo de inferencia bayesiana para predicción de dirección y magnitud
del movimiento de precios del MNQ.

Implementa:
- Prior computation basado en distribución histórica de retornos
- Likelihood estimation condicional a morfologías de barras
- Posterior update con evidencia observada en tiempo real
- Bayesian credible intervals para señales de trading
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from scipy import stats
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class BayesianPredictor:
    """
    Bayesian predictor that computes posterior probabilities of
    price direction and magnitude conditioned on observed bar group patterns.
    """

    def __init__(self, config):
        self.config = config
        self.prior_strength = config.model.bayesian_prior_strength
        self.min_samples = config.model.bayesian_min_samples

        # Prior distributions
        self.prior_up_prob = 0.5
        self.prior_return_mean = 0.0
        self.prior_return_std = 1.0

        # Conditional distributions: {condition_key: {stats}}
        self.conditional_stats = defaultdict(lambda: {
            'n': 0, 'up_count': 0, 'down_count': 0,
            'returns': [], 'magnitudes': []
        })

        # Transition matrices for morphology sequences
        self.morphology_transitions = {}

        self.is_fitted = False

    def fit(self, df: pd.DataFrame, features: pd.DataFrame,
            morphologies: pd.DataFrame, target_horizon: int = 60) -> 'BayesianPredictor':
        """
        Fit the Bayesian model by computing conditional distributions.

        Args:
            df: OHLCV data
            features: Group features
            morphologies: Morphology classifications
            target_horizon: Bars ahead for target
        """
        logger.info(f"Fitting Bayesian model with target horizon = {target_horizon} bars")

        # Target: future return in points
        target_return = df['close'].shift(-target_horizon) - df['close']

        # Compute priors from full dataset
        valid_returns = target_return.dropna()
        self.prior_up_prob = (valid_returns > 0).mean()
        self.prior_return_mean = valid_returns.mean()
        self.prior_return_std = valid_returns.std()

        logger.info(f"  Prior: P(up)={self.prior_up_prob:.3f}, "
                    f"mean_return={self.prior_return_mean:.2f}, "
                    f"std={self.prior_return_std:.2f}")

        # Build conditional distributions from morphologies
        for morph_col in morphologies.columns:
            self._build_conditional_from_morphology(
                morphologies[morph_col], target_return, morph_col
            )

        # Build conditional from discretized features
        self._build_conditional_from_features(features, target_return)

        # Build morphology transition probabilities
        for morph_col in morphologies.columns:
            self._build_transition_matrix(morphologies[morph_col], target_return, morph_col)

        self.is_fitted = True
        logger.info(f"  Bayesian model fitted with {len(self.conditional_stats)} condition keys")
        return self

    def _build_conditional_from_morphology(self, morphology: pd.Series,
                                            target: pd.Series, prefix: str):
        """Build P(return | morphology_pattern) distributions."""
        valid_mask = morphology.notna() & target.notna()
        morph_clean = morphology[valid_mask]
        target_clean = target[valid_mask]

        for pattern in morph_clean.unique():
            if pattern == 'Unknown':
                continue
            mask = morph_clean == pattern
            returns = target_clean[mask].values

            if len(returns) < self.min_samples:
                continue

            key = f"{prefix}={pattern}"
            self.conditional_stats[key] = {
                'n': len(returns),
                'up_count': int((returns > 0).sum()),
                'down_count': int((returns <= 0).sum()),
                'mean_return': float(np.mean(returns)),
                'std_return': float(np.std(returns)),
                'median_return': float(np.median(returns)),
                'skew': float(stats.skew(returns)),
                'p75': float(np.percentile(returns, 75)),
                'p25': float(np.percentile(returns, 25)),
                'mean_magnitude': float(np.mean(np.abs(returns))),
            }

    def _build_conditional_from_features(self, features: pd.DataFrame,
                                          target: pd.Series):
        """Build conditionals from discretized quantitative features."""
        # Select key features for discretization
        key_features = [c for c in features.columns
                       if any(kw in c for kw in ['volatility', 'efficiency', 'volume_ratio',
                                                   'close_position', 'body_range'])]

        target_clean = target.dropna()

        for col in key_features[:20]:  # Limit to top 20 features
            series = features[col].reindex(target_clean.index)
            valid = series.notna() & target_clean.notna()
            if valid.sum() < 100:
                continue

            s = series[valid]
            t = target_clean[valid]

            # Discretize into quartiles
            try:
                quartiles = pd.qcut(s, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
            except Exception:
                continue

            for q in quartiles.unique():
                if pd.isna(q):
                    continue
                mask = quartiles == q
                returns = t[mask].values

                if len(returns) < self.min_samples:
                    continue

                key = f"{col}={q}"
                self.conditional_stats[key] = {
                    'n': len(returns),
                    'up_count': int((returns > 0).sum()),
                    'down_count': int((returns <= 0).sum()),
                    'mean_return': float(np.mean(returns)),
                    'std_return': float(np.std(returns)),
                    'median_return': float(np.median(returns)),
                    'skew': float(stats.skew(returns)),
                    'p75': float(np.percentile(returns, 75)),
                    'p25': float(np.percentile(returns, 25)),
                    'mean_magnitude': float(np.mean(np.abs(returns))),
                }

    def _build_transition_matrix(self, morphology: pd.Series, target: pd.Series, prefix: str):
        """Build Bayesian transition matrix: P(return | morph_t, morph_t-1)."""
        valid_mask = morphology.notna() & morphology.shift(1).notna() & target.notna()
        morph_now = morphology[valid_mask]
        morph_prev = morphology.shift(1)[valid_mask]
        target_clean = target[valid_mask]

        transitions = {}
        for prev_state in morph_prev.unique():
            for curr_state in morph_now.unique():
                if prev_state == 'Unknown' or curr_state == 'Unknown':
                    continue
                mask = (morph_prev == prev_state) & (morph_now == curr_state)
                returns = target_clean[mask].values

                if len(returns) < self.min_samples:
                    continue

                key = f"{prefix}_transition:{prev_state}->{curr_state}"
                transitions[key] = {
                    'n': len(returns),
                    'up_prob': float((returns > 0).mean()),
                    'mean_return': float(np.mean(returns)),
                    'std_return': float(np.std(returns)),
                    'mean_magnitude': float(np.mean(np.abs(returns))),
                }

        self.morphology_transitions[prefix] = transitions

    def predict(self, current_conditions: Dict[str, str]) -> Dict:
        """
        Compute posterior prediction given current observed conditions.

        Args:
            current_conditions: Dict of {feature_name: observed_value}

        Returns:
            Dict with posterior probabilities, expected return, confidence
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        # Start with prior
        log_odds_up = np.log(self.prior_up_prob / (1 - self.prior_up_prob))
        weighted_returns = [self.prior_return_mean]
        weights = [self.prior_strength]
        evidence_count = 0

        # Update with each piece of evidence (naive Bayes assumption)
        for feature, value in current_conditions.items():
            key = f"{feature}={value}"
            if key in self.conditional_stats:
                stats_dict = self.conditional_stats[key]
                n = stats_dict['n']

                if n < self.min_samples:
                    continue

                # Likelihood ratio for direction
                p_up_given_evidence = stats_dict['up_count'] / n
                p_down_given_evidence = stats_dict['down_count'] / n

                if p_up_given_evidence > 0 and p_down_given_evidence > 0:
                    # Log-likelihood ratio update
                    llr = np.log(p_up_given_evidence / p_down_given_evidence)
                    # Weight by sample size (diminishing returns)
                    weight = np.log1p(n / self.min_samples)
                    log_odds_up += llr * weight

                # Return estimate
                weighted_returns.append(stats_dict['mean_return'])
                weights.append(np.sqrt(n))
                evidence_count += 1

        # Check transition evidence
        for prefix, transitions in self.morphology_transitions.items():
            for key, t_stats in transitions.items():
                # Check if this transition matches current conditions
                parts = key.split(':')
                if len(parts) == 2:
                    trans = parts[1]  # "prev->curr"
                    for feat, val in current_conditions.items():
                        if prefix in feat and val in trans:
                            log_odds_up += np.log(
                                t_stats['up_prob'] / max(1 - t_stats['up_prob'], 0.01)
                            ) * 0.5
                            weighted_returns.append(t_stats['mean_return'])
                            weights.append(np.sqrt(t_stats['n']) * 0.5)
                            evidence_count += 1

        # Convert log-odds to probability
        posterior_up = 1.0 / (1.0 + np.exp(-np.clip(log_odds_up, -10, 10)))

        # Weighted average expected return
        weights = np.array(weights)
        weighted_returns = np.array(weighted_returns)
        expected_return = np.average(weighted_returns, weights=weights)

        # Confidence based on evidence count and posterior extremity
        posterior_extremity = abs(posterior_up - 0.5) * 2  # 0 to 1
        evidence_factor = min(1.0, evidence_count / 5)  # Saturates at 5 pieces
        confidence = posterior_extremity * evidence_factor

        # Expected value calculation
        direction = 1 if posterior_up > 0.5 else -1
        expected_value = expected_return * direction if direction * expected_return > 0 else expected_return

        return {
            'posterior_up': posterior_up,
            'posterior_down': 1 - posterior_up,
            'expected_return': expected_return,
            'expected_value': expected_value,
            'confidence': confidence,
            'evidence_count': evidence_count,
            'direction': 'LONG' if posterior_up > 0.5 else 'SHORT',
            'direction_numeric': direction,
        }

    def batch_predict(self, df: pd.DataFrame, features: pd.DataFrame,
                      morphologies: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions for all rows in the dataset."""
        results = []

        morph_cols = [c for c in morphologies.columns if morphologies[c].notna().any()]

        # Key features to discretize for conditions
        key_feat_cols = [c for c in features.columns
                        if any(kw in c for kw in ['volatility', 'efficiency', 'volume_ratio',
                                                    'close_position', 'body_range'])][:10]

        for i in range(len(df)):
            conditions = {}

            # Add morphology conditions
            for mc in morph_cols:
                val = morphologies[mc].iloc[i]
                if pd.notna(val) and val != 'Unknown':
                    conditions[mc] = str(val)

            # Add discretized feature conditions
            for fc in key_feat_cols:
                val = features[fc].iloc[i]
                if pd.notna(val):
                    # Simple quartile binning
                    col_data = features[fc].dropna()
                    if len(col_data) > 0:
                        q = pd.Series([val]).rank(pct=True).iloc[0]
                        if q <= 0.25:
                            conditions[fc] = 'Q1'
                        elif q <= 0.5:
                            conditions[fc] = 'Q2'
                        elif q <= 0.75:
                            conditions[fc] = 'Q3'
                        else:
                            conditions[fc] = 'Q4'

            pred = self.predict(conditions)
            results.append(pred)

        return pd.DataFrame(results, index=df.index)
