"""
Feature Engineering & Bar Grouping Discovery
==============================================
Motor de descubrimiento automático de agrupaciones de barras relevantes.
Utiliza mutual information, análisis de autocorrelación, y métodos de
información teórica para determinar los tamaños de agrupación óptimos.

Analiza morfologías de grupos, parámetros intra-agrupación:
- Rango (range)
- Acción del precio (price action patterns)
- Volumen (volume profile)
- Volatilidad (realized volatility)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import KBinsDiscretizer
from scipy import stats
from scipy.signal import find_peaks
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class BarGroupAnalyzer:
    """Discovers optimal bar groupings and extracts group-level features."""

    def __init__(self, config):
        self.config = config
        self.optimal_groupings = []
        self.grouping_scores = {}
        self.morphology_catalog = {}

    def discover_optimal_groupings(self, df: pd.DataFrame) -> List[int]:
        """
        Discover which bar grouping sizes carry the most predictive information.
        Uses mutual information, autocorrelation analysis, and spectral methods.
        """
        logger.info("=" * 60)
        logger.info("DISCOVERING OPTIMAL BAR GROUPINGS")
        logger.info("=" * 60)

        candidate_sizes = self.config.grouping.candidate_sizes
        scores = {}

        for size in candidate_sizes:
            score = self._evaluate_grouping_size(df, size)
            scores[size] = score
            logger.info(f"  Group size {size:4d}: MI={score['mutual_info']:.4f} | "
                       f"AutoCorr={score['autocorr']:.4f} | "
                       f"PredPower={score['predictive_power']:.4f} | "
                       f"Composite={score['composite']:.4f}")

        self.grouping_scores = scores

        # Select top groupings by composite score
        sorted_sizes = sorted(scores.keys(), key=lambda s: scores[s]['composite'], reverse=True)

        # Filter by minimum thresholds
        optimal = []
        for size in sorted_sizes:
            s = scores[size]
            if (s['mutual_info'] >= self.config.grouping.min_mutual_info and
                s['predictive_power'] >= self.config.grouping.min_predictive_power):
                optimal.append(size)
            if len(optimal) >= self.config.grouping.max_groupings:
                break

        # Always include at least 3 groupings
        if len(optimal) < 3:
            for size in sorted_sizes:
                if size not in optimal:
                    optimal.append(size)
                if len(optimal) >= 3:
                    break

        self.optimal_groupings = sorted(optimal)
        logger.info(f"\nOptimal groupings selected: {self.optimal_groupings}")
        return self.optimal_groupings

    def _evaluate_grouping_size(self, df: pd.DataFrame, size: int) -> Dict:
        """Evaluate the predictive quality of a specific grouping size."""
        # Create grouped features
        group_features = self._compute_group_features(df, size)

        if len(group_features) < 100:
            return {'mutual_info': 0, 'autocorr': 0, 'predictive_power': 0, 'composite': 0}

        # Target: future return over next `size` bars (in points)
        target = df['close'].shift(-size) - df['close']
        target = target.iloc[size:-size]  # Remove edges

        # Align features with target
        features_aligned = group_features.iloc[size:size + len(target)]

        # Drop NaN
        valid_mask = features_aligned.notna().all(axis=1) & target.notna()
        features_clean = features_aligned[valid_mask]
        target_clean = target[valid_mask]

        if len(features_clean) < 100:
            return {'mutual_info': 0, 'autocorr': 0, 'predictive_power': 0, 'composite': 0}

        # 1. Mutual Information
        try:
            mi = mutual_info_regression(
                features_clean.values,
                target_clean.values,
                n_neighbors=5,
                random_state=42
            ).mean()
        except Exception:
            mi = 0.0

        # 2. Autocorrelation of grouped returns
        grouped_returns = df['close'].diff(size).dropna()
        try:
            autocorr = abs(grouped_returns.autocorr(lag=1))
            if np.isnan(autocorr):
                autocorr = 0.0
        except Exception:
            autocorr = 0.0

        # 3. Predictive power (simple linear correlation)
        try:
            corrs = features_clean.corrwith(target_clean).abs()
            pred_power = corrs.mean()
            if np.isnan(pred_power):
                pred_power = 0.0
        except Exception:
            pred_power = 0.0

        # Composite score
        composite = 0.4 * mi + 0.3 * autocorr + 0.3 * pred_power

        return {
            'mutual_info': mi,
            'autocorr': autocorr,
            'predictive_power': pred_power,
            'composite': composite
        }

    def _compute_group_features(self, df: pd.DataFrame, size: int) -> pd.DataFrame:
        """Compute rolling group-level features for a given group size."""
        features = pd.DataFrame(index=df.index)

        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        open_price = df['open']

        # Range features
        features[f'grp_{size}_range'] = high.rolling(size).max() - low.rolling(size).min()
        features[f'grp_{size}_range_pct'] = features[f'grp_{size}_range'] / close.rolling(size).mean()

        # Price action
        features[f'grp_{size}_return'] = close - close.shift(size)
        features[f'grp_{size}_return_pct'] = close.pct_change(size)

        # Body (open of first bar to close of last bar in group)
        features[f'grp_{size}_body'] = close - open_price.shift(size - 1)
        features[f'grp_{size}_body_range_ratio'] = np.where(
            features[f'grp_{size}_range'] > 0,
            features[f'grp_{size}_body'].abs() / features[f'grp_{size}_range'],
            0
        )

        # Volatility
        features[f'grp_{size}_volatility'] = df['return_log'].rolling(size).std()
        features[f'grp_{size}_volatility_change'] = (
            features[f'grp_{size}_volatility'] / features[f'grp_{size}_volatility'].shift(size)
        )

        # Volume profile
        features[f'grp_{size}_volume_total'] = volume.rolling(size).sum()
        features[f'grp_{size}_volume_mean'] = volume.rolling(size).mean()
        features[f'grp_{size}_volume_std'] = volume.rolling(size).std()
        features[f'grp_{size}_volume_trend'] = (
            volume.rolling(size // 2 + 1).mean() /
            volume.rolling(size).mean().replace(0, np.nan)
        )

        # Momentum within group
        features[f'grp_{size}_momentum'] = close - close.rolling(size).mean()

        # Efficiency ratio (net move vs total path)
        total_path = df['range'].rolling(size).sum()
        net_move = (close - close.shift(size)).abs()
        features[f'grp_{size}_efficiency'] = np.where(
            total_path > 0, net_move / total_path, 0
        )

        # Internal bar statistics
        features[f'grp_{size}_up_bars_pct'] = (
            df['direction'].rolling(size).apply(lambda x: (x > 0).sum() / len(x), raw=True)
        )

        # High/low position within range
        group_high = high.rolling(size).max()
        group_low = low.rolling(size).min()
        group_range = group_high - group_low
        features[f'grp_{size}_close_position'] = np.where(
            group_range > 0,
            (close - group_low) / group_range,
            0.5
        )

        return features

    def build_all_group_features(self, df: pd.DataFrame, groupings: Optional[List[int]] = None) -> pd.DataFrame:
        """Build feature matrix for all optimal groupings."""
        if groupings is None:
            groupings = self.optimal_groupings

        logger.info(f"Building features for groupings: {groupings}")
        all_features = pd.DataFrame(index=df.index)

        for size in groupings:
            gf = self._compute_group_features(df, size)
            all_features = pd.concat([all_features, gf], axis=1)

        # Add cross-group features
        all_features = self._add_cross_group_features(all_features, df, groupings)

        logger.info(f"Total features generated: {all_features.shape[1]}")
        return all_features

    def _add_cross_group_features(self, features: pd.DataFrame, df: pd.DataFrame,
                                   groupings: List[int]) -> pd.DataFrame:
        """Add features that compare across different grouping sizes."""
        if len(groupings) < 2:
            return features

        # Volatility regime comparison (short vs long)
        short_g = min(groupings)
        long_g = max(groupings)

        short_vol = df['return_log'].rolling(short_g).std()
        long_vol = df['return_log'].rolling(long_g).std()
        features['vol_ratio_short_long'] = np.where(
            long_vol > 0, short_vol / long_vol, 1.0
        )

        # Momentum alignment across scales
        for i in range(len(groupings) - 1):
            g1, g2 = groupings[i], groupings[i + 1]
            r1_col = f'grp_{g1}_return'
            r2_col = f'grp_{g2}_return'
            if r1_col in features.columns and r2_col in features.columns:
                features[f'momentum_align_{g1}_{g2}'] = np.sign(features[r1_col]) * np.sign(features[r2_col])

        # Multi-scale efficiency
        eff_cols = [c for c in features.columns if 'efficiency' in c]
        if len(eff_cols) >= 2:
            features['mean_efficiency'] = features[eff_cols].mean(axis=1)
            features['efficiency_dispersion'] = features[eff_cols].std(axis=1)

        return features

    def classify_morphologies(self, df: pd.DataFrame, groupings: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Classify bar group morphologies into discrete patterns.
        Patterns: Trend-Up, Trend-Down, Consolidation, Breakout-Up, Breakout-Down,
                  V-Bottom, Inverted-V, Inside, Expansion.
        """
        if groupings is None:
            groupings = self.optimal_groupings

        morphologies = pd.DataFrame(index=df.index)

        for size in groupings:
            morph = self._classify_single_group_morphology(df, size)
            morphologies[f'morph_{size}'] = morph

        return morphologies

    def _classify_single_group_morphology(self, df: pd.DataFrame, size: int) -> pd.Series:
        """Classify morphology for a single group size using vectorized operations."""
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        # Metrics
        net_return = close - close.shift(size)
        group_range = high.rolling(size).max() - low.rolling(size).min()
        efficiency = np.where(group_range > 0, net_return.abs() / group_range, 0)
        vol_ratio = df['return_log'].rolling(size).std() / df['return_log'].rolling(size * 2).std().replace(0, np.nan)

        # Close position in range
        group_high = high.rolling(size).max()
        group_low = low.rolling(size).min()
        close_pos = np.where(group_range > 0, (close - group_low) / group_range, 0.5)

        # Half-group analysis (first half vs second half)
        half = max(1, size // 2)
        first_half_range = high.rolling(half).max().shift(half) - low.rolling(half).min().shift(half)
        second_half_range = high.rolling(half).max() - low.rolling(half).min()
        range_expansion = np.where(
            first_half_range > 0,
            second_half_range / first_half_range,
            1.0
        )

        # Classify
        morphology = pd.Series('Unknown', index=df.index)

        # Trend-Up: strong positive return, high efficiency
        morphology = np.where(
            (net_return > 0) & (efficiency > 0.6),
            'Trend-Up', morphology
        )
        # Trend-Down: strong negative return, high efficiency
        morphology = np.where(
            (net_return < 0) & (efficiency > 0.6),
            'Trend-Down', morphology
        )
        # Consolidation: low range, low efficiency
        median_range = pd.Series(group_range).rolling(size * 10, min_periods=size).median()
        morphology = np.where(
            (group_range < median_range * 0.5) & (np.array(efficiency) < 0.3),
            'Consolidation', morphology
        )
        # Breakout-Up: range expansion + positive close near high
        morphology = np.where(
            (np.array(range_expansion) > 1.5) & (net_return > 0) & (np.array(close_pos) > 0.75),
            'Breakout-Up', morphology
        )
        # Breakout-Down: range expansion + negative close near low
        morphology = np.where(
            (np.array(range_expansion) > 1.5) & (net_return < 0) & (np.array(close_pos) < 0.25),
            'Breakout-Down', morphology
        )
        # V-Bottom: close near high, low in first half
        morphology = np.where(
            (np.array(close_pos) > 0.8) & (net_return > 0) & (np.array(efficiency) > 0.3) &
            (np.array(efficiency) <= 0.6),
            'V-Bottom', morphology
        )
        # Inverted-V: close near low, high in first half
        morphology = np.where(
            (np.array(close_pos) < 0.2) & (net_return < 0) & (np.array(efficiency) > 0.3) &
            (np.array(efficiency) <= 0.6),
            'Inverted-V', morphology
        )

        return pd.Series(morphology, index=df.index)


class ChaosFeatureExtractor:
    """
    Chaos Theory features: Lyapunov exponents, fractal dimension,
    Hurst exponent, entropy measures.
    """

    @staticmethod
    def hurst_exponent(series: pd.Series, max_lag: int = 100) -> float:
        """Compute Hurst exponent via R/S analysis."""
        series = series.dropna().values
        if len(series) < max_lag * 2:
            return 0.5

        lags = range(2, min(max_lag, len(series) // 4))
        rs_values = []

        for lag in lags:
            n_chunks = len(series) // lag
            rs_chunk = []
            for i in range(n_chunks):
                chunk = series[i * lag:(i + 1) * lag]
                mean_chunk = chunk.mean()
                deviate = np.cumsum(chunk - mean_chunk)
                r = deviate.max() - deviate.min()
                s = chunk.std()
                if s > 0:
                    rs_chunk.append(r / s)
            if rs_chunk:
                rs_values.append((np.log(lag), np.log(np.mean(rs_chunk))))

        if len(rs_values) < 3:
            return 0.5

        log_lags, log_rs = zip(*rs_values)
        slope, _, _, _, _ = stats.linregress(log_lags, log_rs)
        return slope

    @staticmethod
    def approximate_entropy(series: pd.Series, m: int = 2, r_factor: float = 0.2) -> float:
        """Compute approximate entropy (ApEn) - measures unpredictability."""
        data = series.dropna().values
        N = len(data)
        if N < 50:
            return 0.0

        # Use subset for performance
        if N > 5000:
            data = data[-5000:]
            N = len(data)

        r = r_factor * data.std()
        if r == 0:
            return 0.0

        def phi(m_val):
            patterns = np.array([data[i:i + m_val] for i in range(N - m_val)])
            count = np.zeros(len(patterns))
            for i, p in enumerate(patterns):
                dists = np.max(np.abs(patterns - p), axis=1)
                count[i] = np.sum(dists <= r) / len(patterns)
            return np.log(count[count > 0]).mean()

        return abs(phi(m) - phi(m + 1))

    @staticmethod
    def compute_rolling_hurst(df: pd.DataFrame, window: int = 500) -> pd.Series:
        """Rolling Hurst exponent."""
        hurst_values = pd.Series(np.nan, index=df.index)
        close = df['close'].values
        returns = np.diff(np.log(close))

        for i in range(window, len(returns)):
            chunk = returns[i - window:i]
            h = ChaosFeatureExtractor.hurst_exponent(pd.Series(chunk), max_lag=50)
            hurst_values.iloc[i + 1] = h

        return hurst_values

    @staticmethod
    def compute_chaos_features(df: pd.DataFrame, window: int = 500) -> pd.DataFrame:
        """Compute all chaos-theory features."""
        features = pd.DataFrame(index=df.index)

        # Rolling Hurst exponent
        features['hurst'] = ChaosFeatureExtractor.compute_rolling_hurst(df, window)

        # Hurst regime (>0.5 trending, <0.5 mean-reverting, ~0.5 random)
        features['hurst_regime'] = pd.cut(
            features['hurst'],
            bins=[0, 0.4, 0.6, 1.0],
            labels=['mean_revert', 'random', 'trending']
        ).astype(str)

        # Rolling approximate entropy (smaller window for tractability)
        apen_window = min(200, window)
        apen_values = pd.Series(np.nan, index=df.index)
        returns = df['return_log'].values

        step = 50  # Compute every 50 bars for performance
        for i in range(apen_window, len(returns), step):
            chunk = returns[i - apen_window:i]
            chunk_clean = chunk[~np.isnan(chunk)]
            if len(chunk_clean) > 50:
                apen_values.iloc[i] = ChaosFeatureExtractor.approximate_entropy(
                    pd.Series(chunk_clean), m=2
                )

        features['approx_entropy'] = apen_values.ffill()

        # Volatility fractal dimension approximation (Higuchi method simplified)
        features['vol_of_vol'] = df['return_log'].rolling(window).std().rolling(window).std()

        return features
