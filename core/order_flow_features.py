"""
Order Flow Proxy Features
==========================
Deriva proxies de order flow a partir de datos OHLCV.
Sin datos Level 2 reales, estimamos presion compradora/vendedora
usando relaciones precio-volumen (Chaikin, OBV, MFI, VWAP).

Features generadas (~30):
- VWAP y desviacion del precio
- On-Balance Volume (OBV) y pendiente
- Volume Delta proxy (Chaikin-style)
- Buying/Selling pressure ratio
- Accumulation/Distribution line
- Money Flow Index (MFI)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class OrderFlowProxy:
    """Derives order flow proxy features from OHLCV data."""

    WINDOWS = [5, 15, 30, 60]

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all order flow proxy features."""
        logger.info("Computing order flow proxy features...")
        features = pd.DataFrame(index=df.index)

        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        open_price = df['open']

        # 1. VWAP features
        vwap_feats = self._vwap_features(close, high, low, volume)
        features = pd.concat([features, vwap_feats], axis=1)

        # 2. OBV features
        obv_feats = self._obv_features(close, volume)
        features = pd.concat([features, obv_feats], axis=1)

        # 3. Volume delta proxy (Chaikin-style)
        delta_feats = self._volume_delta_proxy(close, high, low, volume)
        features = pd.concat([features, delta_feats], axis=1)

        # 4. Buying/Selling pressure
        pressure_feats = self._buying_selling_pressure(close, high, low)
        features = pd.concat([features, pressure_feats], axis=1)

        # 5. Accumulation/Distribution line
        ad_feats = self._accumulation_distribution(close, high, low, volume)
        features = pd.concat([features, ad_feats], axis=1)

        # 6. Money Flow Index
        mfi_feats = self._money_flow_index(close, high, low, volume)
        features = pd.concat([features, mfi_feats], axis=1)

        # 7. Volume-price divergence
        div_feats = self._volume_price_divergence(close, volume)
        features = pd.concat([features, div_feats], axis=1)

        logger.info(f"Order flow features generated: {features.shape[1]} columns")
        return features

    def _vwap_features(self, close, high, low, volume) -> pd.DataFrame:
        """Rolling VWAP and price deviation from VWAP."""
        feats = pd.DataFrame(index=close.index)
        typical_price = (high + low + close) / 3.0

        for w in self.WINDOWS:
            tp_vol = (typical_price * volume).rolling(w).sum()
            cum_vol = volume.rolling(w).sum()
            vwap = np.where(cum_vol > 0, tp_vol / cum_vol, close)
            vwap = pd.Series(vwap, index=close.index)

            # Normalized distance from VWAP
            feats[f'of_vwap_dev_{w}'] = (close - vwap) / vwap.replace(0, np.nan)

        return feats

    def _obv_features(self, close, volume) -> pd.DataFrame:
        """On-Balance Volume and its slope."""
        feats = pd.DataFrame(index=close.index)

        # OBV: cumulative volume * sign of price change
        price_change = close.diff()
        obv_direction = np.sign(price_change).fillna(0)
        obv = (volume * obv_direction).cumsum()

        for w in self.WINDOWS:
            # Normalized OBV slope (linear regression over window)
            obv_ma = obv.rolling(w).mean()
            feats[f'of_obv_slope_{w}'] = (obv - obv_ma) / volume.rolling(w).mean().replace(0, np.nan)

        return feats

    def _volume_delta_proxy(self, close, high, low, volume) -> pd.DataFrame:
        """
        Chaikin-style volume delta: estimates buy vs sell volume
        delta = volume * (2 * (close - low) / (high - low) - 1)
        Positive = more buying, Negative = more selling
        """
        feats = pd.DataFrame(index=close.index)
        hl_range = (high - low).replace(0, np.nan)
        close_location = (2.0 * (close - low) / hl_range - 1.0).fillna(0)
        bar_delta = volume * close_location

        for w in self.WINDOWS:
            feats[f'of_delta_sum_{w}'] = bar_delta.rolling(w).sum()
            # Normalized by total volume
            total_vol = volume.rolling(w).sum().replace(0, np.nan)
            feats[f'of_delta_ratio_{w}'] = bar_delta.rolling(w).sum() / total_vol

        return feats

    def _buying_selling_pressure(self, close, high, low) -> pd.DataFrame:
        """
        Buying pressure = close - low (bulls pushed price up from low)
        Selling pressure = high - close (bears pushed price down from high)
        """
        feats = pd.DataFrame(index=close.index)
        bp = close - low  # buying pressure
        sp = high - close  # selling pressure
        total_pressure = (bp + sp).replace(0, np.nan)

        for w in [10, 20, 60]:
            bp_avg = bp.rolling(w).mean()
            sp_avg = sp.rolling(w).mean()
            total_avg = (bp_avg + sp_avg).replace(0, np.nan)
            feats[f'of_bp_ratio_{w}'] = bp_avg / total_avg

        return feats

    def _accumulation_distribution(self, close, high, low, volume) -> pd.DataFrame:
        """
        Accumulation/Distribution line:
        AD = cumsum(((close - low) - (high - close)) / (high - low) * volume)
        """
        feats = pd.DataFrame(index=close.index)
        hl_range = (high - low).replace(0, np.nan)
        clv = ((close - low) - (high - close)) / hl_range  # Close Location Value [-1, 1]
        clv = clv.fillna(0)
        ad_flow = clv * volume
        ad_line = ad_flow.cumsum()

        for w in [20, 60]:
            ad_ma = ad_line.rolling(w).mean()
            feats[f'of_ad_slope_{w}'] = (ad_line - ad_ma) / volume.rolling(w).mean().replace(0, np.nan)

        return feats

    def _money_flow_index(self, close, high, low, volume, period: int = 14) -> pd.DataFrame:
        """
        Money Flow Index: volume-weighted RSI.
        MFI = 100 - 100 / (1 + positive_flow / negative_flow)
        """
        feats = pd.DataFrame(index=close.index)
        typical_price = (high + low + close) / 3.0
        money_flow = typical_price * volume
        tp_diff = typical_price.diff()

        positive_flow = pd.Series(
            np.where(tp_diff > 0, money_flow, 0), index=close.index
        )
        negative_flow = pd.Series(
            np.where(tp_diff < 0, money_flow, 0), index=close.index
        )

        pos_sum = positive_flow.rolling(period).sum()
        neg_sum = negative_flow.rolling(period).sum().replace(0, np.nan)
        money_ratio = pos_sum / neg_sum
        mfi = 100.0 - 100.0 / (1.0 + money_ratio)

        feats['of_mfi_14'] = mfi / 100.0  # Normalize to 0-1

        # Also longer MFI
        pos_sum_30 = positive_flow.rolling(30).sum()
        neg_sum_30 = negative_flow.rolling(30).sum().replace(0, np.nan)
        money_ratio_30 = pos_sum_30 / neg_sum_30
        feats['of_mfi_30'] = (100.0 - 100.0 / (1.0 + money_ratio_30)) / 100.0

        return feats

    def _volume_price_divergence(self, close, volume) -> pd.DataFrame:
        """
        Detect divergences between price momentum and volume momentum.
        Positive divergence: price falling but volume declining (exhaustion)
        Negative divergence: price rising but volume declining (weak rally)
        """
        feats = pd.DataFrame(index=close.index)

        for w in [20, 60]:
            price_mom = close - close.shift(w)
            vol_mom = volume.rolling(w).mean() - volume.rolling(w).mean().shift(w)

            # Normalized divergence: sign(price_mom) * sign(vol_mom)
            # -1 = divergence, +1 = confirmation
            price_sign = np.sign(price_mom).fillna(0)
            vol_sign = np.sign(vol_mom).fillna(0)
            feats[f'of_pv_confirm_{w}'] = price_sign * vol_sign

        return feats

    def compute_single_bar(self, bar_buffer: list, groupings: list = None) -> Dict:
        """Compute order flow features for a single bar (real-time use)."""
        if len(bar_buffer) < 60:
            return {}

        df = pd.DataFrame(bar_buffer)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        features = {}
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        # VWAP deviation
        for w in self.WINDOWS:
            if len(df) >= w:
                tp = (high + low + close) / 3.0
                tp_vol = (tp * volume).rolling(w).sum()
                cum_vol = volume.rolling(w).sum()
                vwap = tp_vol.iloc[-1] / cum_vol.iloc[-1] if cum_vol.iloc[-1] > 0 else close.iloc[-1]
                features[f'of_vwap_dev_{w}'] = (close.iloc[-1] - vwap) / vwap if vwap > 0 else 0

        # OBV slope
        price_change = close.diff()
        obv_dir = np.sign(price_change).fillna(0)
        obv = (volume * obv_dir).cumsum()
        for w in self.WINDOWS:
            if len(df) >= w:
                obv_ma = obv.rolling(w).mean().iloc[-1]
                vol_ma = volume.rolling(w).mean().iloc[-1]
                features[f'of_obv_slope_{w}'] = (obv.iloc[-1] - obv_ma) / vol_ma if vol_ma > 0 else 0

        # Delta ratio
        hl_range = (high - low).replace(0, np.nan)
        close_loc = (2.0 * (close - low) / hl_range - 1.0).fillna(0)
        bar_delta = volume * close_loc
        for w in self.WINDOWS:
            if len(df) >= w:
                total_vol = volume.rolling(w).sum().iloc[-1]
                features[f'of_delta_sum_{w}'] = bar_delta.rolling(w).sum().iloc[-1]
                features[f'of_delta_ratio_{w}'] = bar_delta.rolling(w).sum().iloc[-1] / total_vol if total_vol > 0 else 0

        # Buying pressure ratio
        bp = close - low
        sp = high - close
        for w in [10, 20, 60]:
            if len(df) >= w:
                bp_avg = bp.rolling(w).mean().iloc[-1]
                sp_avg = sp.rolling(w).mean().iloc[-1]
                total = bp_avg + sp_avg
                features[f'of_bp_ratio_{w}'] = bp_avg / total if total > 0 else 0.5

        # AD slope
        clv = ((close - low) - (high - close)) / hl_range
        clv = clv.fillna(0)
        ad_flow = clv * volume
        ad_line = ad_flow.cumsum()
        for w in [20, 60]:
            if len(df) >= w:
                ad_ma = ad_line.rolling(w).mean().iloc[-1]
                vol_ma = volume.rolling(w).mean().iloc[-1]
                features[f'of_ad_slope_{w}'] = (ad_line.iloc[-1] - ad_ma) / vol_ma if vol_ma > 0 else 0

        # MFI
        tp = (high + low + close) / 3.0
        mf = tp * volume
        tp_diff = tp.diff()
        pos_flow = pd.Series(np.where(tp_diff > 0, mf, 0), index=df.index)
        neg_flow = pd.Series(np.where(tp_diff < 0, mf, 0), index=df.index)
        for period, name in [(14, 'of_mfi_14'), (30, 'of_mfi_30')]:
            if len(df) >= period:
                ps = pos_flow.rolling(period).sum().iloc[-1]
                ns = neg_flow.rolling(period).sum().iloc[-1]
                mr = ps / ns if ns > 0 else 100
                features[name] = (100.0 - 100.0 / (1.0 + mr)) / 100.0

        # Price-volume confirmation
        for w in [20, 60]:
            if len(df) >= w * 2:
                price_mom = close.iloc[-1] - close.iloc[-w]
                vol_now = volume.rolling(w).mean().iloc[-1]
                vol_prev = volume.rolling(w).mean().iloc[-w]
                vol_mom = vol_now - vol_prev
                features[f'of_pv_confirm_{w}'] = np.sign(price_mom) * np.sign(vol_mom)

        return features
