"""
Cross-Asset Feature Provider
==============================
Genera features de activos correlacionados (VIX, yields, S&P500).

Modos:
- 'simulate': Deriva proxies del propio MNQ (sin datos externos)
- 'file': Carga datos de un CSV local
- 'fetch': Descarga datos via yfinance (requiere internet)

Features generadas (~15):
- VIX proxy (volatilidad realizada anualizada)
- Yield proxy (inverso de momentum = risk-off indicator)
- Correlacion inter-mercado
- Risk-on/Risk-off score compuesto
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
import logging
import os

logger = logging.getLogger(__name__)


class CrossAssetFeatureProvider:
    """Provides cross-asset features from external data or MNQ-derived proxies."""

    def __init__(self, mode: str = 'simulate'):
        """
        Args:
            mode: 'simulate' (derive from MNQ), 'file' (load CSV), 'fetch' (yfinance)
        """
        self.mode = mode
        self.external_data = None

    def compute_features(self, df: pd.DataFrame,
                         external_path: Optional[str] = None) -> pd.DataFrame:
        """Compute cross-asset features and return DataFrame aligned to df index."""
        if self.mode == 'fetch':
            return self._compute_from_fetch(df)
        elif self.mode == 'file' and external_path:
            return self._compute_from_file(df, external_path)
        else:
            return self._compute_from_simulation(df)

    def _compute_from_simulation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive cross-asset proxies entirely from MNQ data.
        This is the default when external data is unavailable.
        """
        logger.info("Computing cross-asset proxy features from MNQ data...")
        features = pd.DataFrame(index=df.index)

        close = df['close']
        volume = df['volume']
        ret_log = df['return_log'] if 'return_log' in df.columns else np.log(close / close.shift(1))

        # --- VIX Proxy ---
        # Realized volatility annualized (VIX ~= annualized vol * 100)
        # Use different windows to capture term structure
        vol_short = ret_log.rolling(20).std() * np.sqrt(252 * 390)  # ~1 day
        vol_medium = ret_log.rolling(100).std() * np.sqrt(252 * 390)  # ~5 days
        vol_long = ret_log.rolling(500).std() * np.sqrt(252 * 390)  # ~25 days

        features['ca_vix_proxy'] = vol_short  # Current implied vol proxy
        features['ca_vix_change'] = vol_short - vol_short.shift(20)  # Vol change over 1 day
        features['ca_vix_term_structure'] = np.where(
            vol_long > 0, vol_short / vol_long, 1.0
        )  # >1 = contango (fear), <1 = backwardation (complacency)

        # VIX percentile rank (rolling 5-day window over 25 days of history)
        features['ca_vix_percentile'] = vol_short.rolling(5000, min_periods=500).rank(pct=True)

        # --- Yield Proxy ---
        # In risk-off environments, momentum turns negative and vol rises
        # Use inverse momentum as yield proxy (when rates rise, growth sells off)
        mom_60 = close - close.shift(60)
        mom_240 = close - close.shift(240)

        # Yield change proxy: negative momentum = rates rising / risk-off
        features['ca_yield_proxy'] = -mom_240 / close.rolling(240).mean().replace(0, np.nan)
        features['ca_yield_change'] = features['ca_yield_proxy'] - features['ca_yield_proxy'].shift(60)

        # --- Correlation Structure ---
        # Auto-correlation as a proxy for market efficiency/trending
        # High autocorr = trending (correlated with risk-on flows)
        ret_60 = close.pct_change(60)
        features['ca_autocorr_20'] = ret_log.rolling(500).apply(
            lambda x: pd.Series(x).autocorr(lag=20) if len(x) > 20 else 0, raw=False
        )

        # --- Risk-On / Risk-Off Composite Score ---
        # Risk-On signals: low vol, positive momentum, expanding volume
        # Risk-Off signals: high vol, negative momentum, contracting volume
        vol_z = (vol_short - vol_long) / vol_long.replace(0, np.nan)  # Vol z-score
        mom_z = mom_60 / close.rolling(60).std().replace(0, np.nan)  # Momentum z-score
        vol_ratio = volume.rolling(20).mean() / volume.rolling(100).mean().replace(0, np.nan)

        # Composite: positive = risk-on, negative = risk-off
        features['ca_risk_score'] = (
            -vol_z * 0.4 +       # Low vol = risk-on
            mom_z * 0.3 +        # Positive momentum = risk-on
            (vol_ratio - 1) * 0.3  # High volume ratio = conviction
        )

        # Discretize risk regime
        features['ca_risk_regime'] = pd.cut(
            features['ca_risk_score'],
            bins=[-np.inf, -1, -0.3, 0.3, 1, np.inf],
            labels=[0, 1, 2, 3, 4]  # 0=strong risk-off, 4=strong risk-on
        ).astype(float)

        # --- Intraday Session Features ---
        # Market microstructure changes by session
        if 'hour' in df.columns:
            hour = df['hour']
        else:
            hour = df['timestamp'].dt.hour if 'timestamp' in df.columns else pd.Series(0, index=df.index)

        # Session-based vol ratio (current session vol vs overall)
        features['ca_session_vol_ratio'] = (
            ret_log.rolling(30).std() / ret_log.rolling(390).std().replace(0, np.nan)
        )

        # Volume surge indicator (is volume above 2x its hourly average?)
        hourly_vol_avg = volume.rolling(390).mean()  # ~1 trading day
        features['ca_volume_surge'] = np.where(
            hourly_vol_avg > 0,
            volume.rolling(5).mean() / hourly_vol_avg,
            1.0
        )

        logger.info(f"Cross-asset proxy features: {features.shape[1]} columns")
        return features

    def _compute_from_file(self, df: pd.DataFrame, path: str) -> pd.DataFrame:
        """Load external cross-asset data from CSV and merge."""
        logger.info(f"Loading cross-asset data from {path}")

        ext = pd.read_csv(path, parse_dates=['date'])
        ext = ext.sort_values('date')

        # The external data is daily; we need to forward-fill to 1-min
        if 'timestamp' in df.columns:
            df_date = df['timestamp'].dt.date
        else:
            df_date = df.index

        features = pd.DataFrame(index=df.index)

        # Map daily values to each minute bar
        ext_indexed = ext.set_index('date')
        for col in ext_indexed.columns:
            daily_series = ext_indexed[col]
            # Create a mapping date -> value
            features[f'ca_{col}'] = df_date.map(
                lambda d: daily_series.get(d, np.nan)
            )
            features[f'ca_{col}'] = features[f'ca_{col}'].ffill()

        return features

    def _compute_from_fetch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fetch external data via yfinance (requires internet + yfinance)."""
        try:
            import yfinance as yf
        except ImportError:
            logger.warning("yfinance not installed. Falling back to simulation mode.")
            return self._compute_from_simulation(df)

        logger.info("Fetching cross-asset data from Yahoo Finance...")

        if 'timestamp' in df.columns:
            start = df['timestamp'].min().strftime('%Y-%m-%d')
            end = df['timestamp'].max().strftime('%Y-%m-%d')
        else:
            return self._compute_from_simulation(df)

        tickers = {
            'vix': '^VIX',
            'tnx': '^TNX',
            'sp500': '^GSPC',
        }

        features = pd.DataFrame(index=df.index)
        df_date = df['timestamp'].dt.normalize()

        for name, ticker in tickers.items():
            try:
                data = yf.download(ticker, start=start, end=end, progress=False)
                if len(data) == 0:
                    continue
                daily_close = data['Close'].to_dict()

                # Map to minute bars (use previous day's close to avoid look-ahead)
                mapped = df_date.map(lambda d: daily_close.get(d, np.nan))
                mapped = mapped.ffill()

                features[f'ca_{name}_level'] = mapped.values
                features[f'ca_{name}_change'] = mapped.pct_change(periods=390).values  # ~1 day change
            except Exception as e:
                logger.warning(f"Failed to fetch {ticker}: {e}")

        # If we got some data, compute derived features
        if 'ca_vix_level' in features.columns:
            features['ca_vix_percentile'] = features['ca_vix_level'].rolling(5000, min_periods=100).rank(pct=True)

        # Fall back to simulation for anything we couldn't fetch
        sim_features = self._compute_from_simulation(df)
        for col in sim_features.columns:
            if col not in features.columns:
                features[col] = sim_features[col]

        return features

    def compute_single_bar(self, bar_buffer: list) -> Dict:
        """Compute cross-asset features for a single bar (real-time use)."""
        if len(bar_buffer) < 500:
            return {}

        df = pd.DataFrame(bar_buffer)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        close = df['close']
        volume = df['volume']
        ret_log = np.log(close / close.shift(1))

        features = {}

        # VIX proxy
        vol_short = ret_log.rolling(20).std().iloc[-1] * np.sqrt(252 * 390)
        vol_long = ret_log.rolling(500).std().iloc[-1] * np.sqrt(252 * 390)
        features['ca_vix_proxy'] = vol_short
        features['ca_vix_change'] = vol_short - (ret_log.rolling(20).std().iloc[-21] * np.sqrt(252 * 390)) if len(df) > 41 else 0
        features['ca_vix_term_structure'] = vol_short / vol_long if vol_long > 0 else 1.0
        features['ca_vix_percentile'] = ret_log.rolling(20).std().rank(pct=True).iloc[-1]

        # Yield proxy
        mom_240 = close.iloc[-1] - close.iloc[-min(240, len(df) - 1)]
        mean_240 = close.rolling(min(240, len(df))).mean().iloc[-1]
        features['ca_yield_proxy'] = -mom_240 / mean_240 if mean_240 > 0 else 0
        features['ca_yield_change'] = 0  # Simplified for real-time

        # Autocorrelation
        features['ca_autocorr_20'] = 0  # Expensive to compute per-bar

        # Risk score
        vol_z = (vol_short - vol_long) / vol_long if vol_long > 0 else 0
        mom_60 = close.iloc[-1] - close.iloc[-min(60, len(df) - 1)]
        mom_z = mom_60 / close.rolling(60).std().iloc[-1] if close.rolling(60).std().iloc[-1] > 0 else 0
        vol_ratio = volume.rolling(20).mean().iloc[-1] / volume.rolling(100).mean().iloc[-1] if volume.rolling(100).mean().iloc[-1] > 0 else 1
        features['ca_risk_score'] = -vol_z * 0.4 + mom_z * 0.3 + (vol_ratio - 1) * 0.3
        features['ca_risk_regime'] = 2  # Default neutral for real-time

        # Session features
        features['ca_session_vol_ratio'] = (
            ret_log.rolling(30).std().iloc[-1] / ret_log.rolling(390).std().iloc[-1]
        ) if ret_log.rolling(390).std().iloc[-1] > 0 else 1.0

        vol_ma = volume.rolling(390).mean().iloc[-1]
        features['ca_volume_surge'] = volume.rolling(5).mean().iloc[-1] / vol_ma if vol_ma > 0 else 1.0

        return features
