"""
Regime Filter
==============
Filtro estricto de regimen: solo permite trading en condiciones optimas.

Condiciones para operar:
1. Alta volatilidad (quintil Q4 o Q5)
2. Sesion de la tarde (14:00-16:00 ET)

Basado en el analisis empirico que muestra:
- Sesion tarde: 54 pts vol promedio, 26.2% big moves, 51.9% bullish bias
- High vol Q4-Q5: unico regimen con PnL positivo en walk-forward (PF=1.05)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class RegimeFilter:
    """Filters signals based on volatility regime and trading session."""

    def __init__(self, config):
        self.config = config
        self.vol_quintiles = None

    def compute_volatility_quintile(self, df: pd.DataFrame,
                                     lookback: int = 5000) -> pd.Series:
        """
        Compute rolling volatility quintile (1-5) for each bar.
        Uses volatility_20 if available, otherwise computes from return_log.
        """
        if 'volatility_20' in df.columns:
            vol = df['volatility_20']
        else:
            ret = np.log(df['close'] / df['close'].shift(1))
            vol = ret.rolling(20).std()

        # Rolling quintile rank
        quintile = vol.rolling(lookback, min_periods=500).rank(pct=True)
        # Map to 1-5
        quintile_num = pd.cut(
            quintile,
            bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=[1, 2, 3, 4, 5],
            include_lowest=True
        ).astype(float)

        self.vol_quintiles = quintile_num
        return quintile_num

    def compute_session_mask(self, df: pd.DataFrame,
                              start_hour: int = 14,
                              end_hour: int = 16) -> pd.Series:
        """
        Compute boolean mask for afternoon session (14:00-16:00 ET).
        Timestamps are assumed UTC; ET = UTC - 4 (EDT) or UTC - 5 (EST).
        We use UTC-4 as a reasonable approximation for most of the year.
        """
        if 'timestamp' in df.columns:
            ts = df['timestamp']
            if ts.dt.tz is not None:
                # Convert UTC to ET (approximate with -4 hours for EDT)
                hour_et = (ts.dt.hour - 4) % 24
            else:
                # Assume already in ET
                hour_et = ts.dt.hour
        elif 'hour' in df.columns:
            # If hour column exists, assume it's UTC
            hour_et = (df['hour'] - 4) % 24
        else:
            logger.warning("No timestamp/hour info. Session filter disabled.")
            return pd.Series(True, index=df.index)

        mask = (hour_et >= start_hour) & (hour_et < end_hour)
        return mask

    def compute_regime_mask(self, df: pd.DataFrame,
                            min_quintile: int = 4,
                            start_hour: int = 14,
                            end_hour: int = 16) -> pd.Series:
        """
        Combined regime mask: high volatility AND afternoon session.
        Returns boolean Series where True = allowed to trade.
        """
        vol_q = self.compute_volatility_quintile(df)
        session = self.compute_session_mask(df, start_hour, end_hour)

        vol_ok = vol_q >= min_quintile
        regime_mask = vol_ok & session

        n_total = len(df)
        n_eligible = regime_mask.sum()
        pct = n_eligible / n_total * 100 if n_total > 0 else 0

        logger.info(f"Regime filter: {n_eligible:,}/{n_total:,} bars eligible ({pct:.1f}%)")
        logger.info(f"  Vol Q>={min_quintile}: {vol_ok.sum():,} bars")
        logger.info(f"  Session {start_hour}-{end_hour} ET: {session.sum():,} bars")

        return regime_mask

    def apply_filter(self, signals: pd.DataFrame,
                     regime_mask: pd.Series) -> pd.DataFrame:
        """Set signal_valid=False for bars outside the allowed regime."""
        signals = signals.copy()
        # Align mask to signals index
        aligned_mask = regime_mask.reindex(signals.index, fill_value=False)
        signals.loc[~aligned_mask, 'signal_valid'] = False
        signals['regime_active'] = aligned_mask

        n_before = signals['signal_valid'].sum() if 'signal_valid' in signals.columns else 0
        n_filtered = (signals['regime_active'] & signals.get('signal_valid', True)).sum()
        logger.info(f"Regime filter: {n_before} valid signals -> {n_filtered} after regime filter")

        return signals

    def check_current_regime(self, current_vol_20: float,
                              current_hour_utc: int,
                              vol_history: pd.Series = None) -> Dict:
        """
        Check regime for a single bar (real-time use).

        Args:
            current_vol_20: Current 20-bar rolling volatility
            current_hour_utc: Current hour in UTC
            vol_history: Series of historical vol_20 values for percentile calc

        Returns:
            Dict with regime info
        """
        # Session check
        hour_et = (current_hour_utc - 4) % 24
        in_session = 14 <= hour_et < 16

        # Volatility quintile
        if vol_history is not None and len(vol_history) > 100:
            pct_rank = (vol_history < current_vol_20).mean()
            quintile = int(min(5, max(1, np.ceil(pct_rank * 5))))
        else:
            quintile = 3  # Default to middle if no history

        high_vol = quintile >= 4
        is_active = in_session and high_vol

        return {
            'in_session': in_session,
            'hour_et': hour_et,
            'vol_quintile': quintile,
            'high_vol': high_vol,
            'regime_active': is_active,
        }
