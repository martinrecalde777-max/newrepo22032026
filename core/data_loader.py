"""
Data Loader Module
==================
Carga y preprocesa datos OHLCV de 1 minuto del MNQ.
Maneja la detección automática de formato CSV, limpieza de datos,
y generación de features base para el pipeline.
"""

import os
import pandas as pd
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MNQDataLoader:
    """Loads and preprocesses MNQ 1-minute OHLCV data."""

    EXPECTED_COLUMNS_VARIANTS = [
        # Standard OHLCV
        ['timestamp', 'open', 'high', 'low', 'close', 'volume'],
        ['date', 'open', 'high', 'low', 'close', 'volume'],
        ['datetime', 'open', 'high', 'low', 'close', 'volume'],
        ['time', 'open', 'high', 'low', 'close', 'volume'],
        # With symbol
        ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume'],
        ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume'],
        # Databento format
        ['ts_event', 'open', 'high', 'low', 'close', 'volume'],
    ]

    def __init__(self, config):
        self.config = config
        self.raw_data = None
        self.processed_data = None

    def find_data_file(self) -> str:
        """Find the data file from configured paths."""
        paths = [self.config.data.csv_path] + self.config.data.fallback_paths
        for path in paths:
            expanded = os.path.expanduser(path)
            if os.path.exists(expanded):
                logger.info(f"Found data file at: {expanded}")
                return expanded
        raise FileNotFoundError(
            f"Data file not found. Searched paths:\n" +
            "\n".join(f"  - {p}" for p in paths) +
            "\n\nPlease ensure the CSV file is at one of these locations."
        )

    def load(self) -> pd.DataFrame:
        """Load raw CSV data with automatic format detection."""
        filepath = self.find_data_file()
        logger.info(f"Loading data from {filepath}...")

        # Try to detect separator and header
        with open(filepath, 'r') as f:
            first_lines = [f.readline() for _ in range(5)]

        # Detect separator
        sep = ','
        if '\t' in first_lines[0]:
            sep = '\t'
        elif ';' in first_lines[0]:
            sep = ';'

        # Load with pandas
        df = pd.read_csv(filepath, sep=sep, low_memory=False)
        logger.info(f"Loaded {len(df):,} rows with columns: {list(df.columns)}")

        self.raw_data = df
        return df

    def preprocess(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Full preprocessing pipeline."""
        if df is None:
            df = self.raw_data.copy()
        else:
            df = df.copy()

        # Normalize column names
        df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

        # Identify and rename time column
        time_col = None
        for candidate in ['timestamp', 'ts_event', 'datetime', 'date', 'time', 'ts']:
            if candidate in df.columns:
                time_col = candidate
                break

        if time_col is None:
            # Try first column as datetime
            try:
                pd.to_datetime(df.iloc[:, 0].head(10))
                time_col = df.columns[0]
            except Exception:
                raise ValueError(f"Cannot identify datetime column. Columns: {list(df.columns)}")

        df = df.rename(columns={time_col: 'timestamp'})

        # Parse timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
        df = df.dropna(subset=['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Ensure OHLCV columns exist and are numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows with NaN OHLCV
        df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])

        # Filter MNQ symbol if symbol column exists
        if 'symbol' in df.columns:
            mnq_mask = df['symbol'].str.contains('MNQ', case=False, na=False)
            if mnq_mask.any():
                df = df[mnq_mask]
                logger.info(f"Filtered to MNQ: {len(df):,} rows")

        # Remove zero-volume bars (likely non-trading)
        df = df[df['volume'] > 0]

        # Validate OHLC consistency
        valid = (df['high'] >= df['low']) & \
                (df['high'] >= df['open']) & \
                (df['high'] >= df['close']) & \
                (df['low'] <= df['open']) & \
                (df['low'] <= df['close'])
        invalid_count = (~valid).sum()
        if invalid_count > 0:
            logger.warning(f"Removed {invalid_count} rows with invalid OHLC relationships")
            df = df[valid]

        # Add base derived features
        df = self._add_base_features(df)

        df = df.reset_index(drop=True)
        self.processed_data = df

        logger.info(f"Preprocessing complete: {len(df):,} clean bars")
        logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

        return df

    def _add_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add fundamental per-bar features."""
        # Price action
        df['range'] = df['high'] - df['low']
        df['body'] = df['close'] - df['open']
        df['body_abs'] = df['body'].abs()
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']

        # Relative measures
        df['body_range_ratio'] = np.where(df['range'] > 0, df['body_abs'] / df['range'], 0)
        df['upper_wick_ratio'] = np.where(df['range'] > 0, df['upper_wick'] / df['range'], 0)
        df['lower_wick_ratio'] = np.where(df['range'] > 0, df['lower_wick'] / df['range'], 0)

        # Bar direction
        df['direction'] = np.sign(df['body'])

        # Returns
        df['return_close'] = df['close'].pct_change()
        df['return_log'] = np.log(df['close'] / df['close'].shift(1))
        df['return_points'] = df['close'] - df['close'].shift(1)

        # Volatility (rolling)
        df['volatility_5'] = df['return_log'].rolling(5).std()
        df['volatility_20'] = df['return_log'].rolling(20).std()
        df['volatility_60'] = df['return_log'].rolling(60).std()

        # Volume features
        df['volume_ma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = np.where(
            df['volume_ma_20'] > 0,
            df['volume'] / df['volume_ma_20'],
            1.0
        )

        # Time features
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['time_in_session'] = df['hour'] * 60 + df['minute']

        # Momentum
        df['momentum_5'] = df['close'] - df['close'].shift(5)
        df['momentum_20'] = df['close'] - df['close'].shift(20)
        df['momentum_60'] = df['close'] - df['close'].shift(60)

        # VWAP approximation (daily reset)
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['tp_volume'] = df['typical_price'] * df['volume']

        return df

    def get_train_val_test_split(self, df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Chronological train/validation/test split."""
        if df is None:
            df = self.processed_data

        n = len(df)
        train_end = int(n * self.config.model.train_ratio)
        val_end = int(n * (self.config.model.train_ratio + self.config.model.validation_ratio))

        train = df.iloc[:train_end].copy()
        val = df.iloc[train_end:val_end].copy()
        test = df.iloc[val_end:].copy()

        logger.info(f"Split: Train={len(train):,} | Val={len(val):,} | Test={len(test):,}")
        return train, val, test
