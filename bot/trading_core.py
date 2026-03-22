"""
Trading Core - Base de Datos y Motor de Señales en Tiempo Real
===============================================================
Núcleo para el bot de trading en tiempo real.
Almacena modelos entrenados, genera señales, y gestiona estado.

Componentes:
- SQLite database para estado, trades, señales
- Motor de features en tiempo real
- Generador de señales con filtros multi-capa
- Gestor de posiciones
"""

import sqlite3
import json
import os
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
import joblib

logger = logging.getLogger(__name__)


class TradingDatabase:
    """SQLite database for trading state management."""

    def __init__(self, db_path: str = './data/trading_core.db'):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self):
        """Create all required tables."""
        cursor = self.conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                direction INTEGER NOT NULL,
                confidence REAL NOT NULL,
                expected_return REAL,
                expected_value REAL,
                prob_up REAL,
                prob_big_move REAL,
                morphology TEXT,
                vol_regime INTEGER,
                is_valid INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id INTEGER,
                entry_time TEXT NOT NULL,
                exit_time TEXT,
                direction INTEGER NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                stop_loss REAL,
                take_profit REAL,
                pnl_points REAL,
                pnl_net REAL,
                exit_reason TEXT,
                status TEXT DEFAULT 'open',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (signal_id) REFERENCES signals(id)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_state (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                close_price REAL NOT NULL,
                volatility_20 REAL,
                vol_regime INTEGER,
                hurst REAL,
                current_morphology TEXT,
                atr_20 REAL,
                volume_ratio REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                total_pnl REAL,
                n_trades INTEGER,
                win_rate REAL,
                max_drawdown REAL,
                sharpe REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        self.conn.commit()

    def insert_signal(self, signal: Dict) -> int:
        """Insert a new trading signal."""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO signals (timestamp, direction, confidence, expected_return,
                               expected_value, prob_up, prob_big_move, morphology,
                               vol_regime, is_valid)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal.get('timestamp', datetime.now(timezone.utc).isoformat()),
            signal.get('direction', 0),
            signal.get('confidence', 0),
            signal.get('expected_return', 0),
            signal.get('expected_value', 0),
            signal.get('prob_up', 0.5),
            signal.get('prob_big_move', 0),
            signal.get('morphology', ''),
            signal.get('vol_regime', 0),
            1 if signal.get('is_valid', False) else 0,
        ))
        self.conn.commit()
        return cursor.lastrowid

    def insert_trade(self, trade: Dict) -> int:
        """Insert a new trade."""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO trades (signal_id, entry_time, direction, entry_price,
                              stop_loss, take_profit, status)
            VALUES (?, ?, ?, ?, ?, ?, 'open')
        ''', (
            trade.get('signal_id'),
            trade.get('entry_time', datetime.now(timezone.utc).isoformat()),
            trade['direction'],
            trade['entry_price'],
            trade.get('stop_loss'),
            trade.get('take_profit'),
        ))
        self.conn.commit()
        return cursor.lastrowid

    def close_trade(self, trade_id: int, exit_price: float,
                    exit_reason: str, pnl_points: float, pnl_net: float):
        """Close an open trade."""
        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE trades SET
                exit_time = ?,
                exit_price = ?,
                exit_reason = ?,
                pnl_points = ?,
                pnl_net = ?,
                status = 'closed'
            WHERE id = ?
        ''', (
            datetime.now(timezone.utc).isoformat(),
            exit_price,
            exit_reason,
            pnl_points,
            pnl_net,
            trade_id,
        ))
        self.conn.commit()

    def get_open_trades(self) -> List[Dict]:
        """Get all open trades."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM trades WHERE status = 'open'")
        columns = [d[0] for d in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_recent_signals(self, limit: int = 100) -> List[Dict]:
        """Get recent signals."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM signals ORDER BY id DESC LIMIT ?", (limit,)
        )
        columns = [d[0] for d in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_daily_performance(self, date: str = None) -> Dict:
        """Get performance for a specific date."""
        if date is None:
            date = datetime.now(timezone.utc).strftime('%Y-%m-%d')

        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT
                COUNT(*) as n_trades,
                COALESCE(SUM(pnl_net), 0) as total_pnl,
                COALESCE(AVG(CASE WHEN pnl_net > 0 THEN 1.0 ELSE 0.0 END), 0) as win_rate
            FROM trades
            WHERE DATE(entry_time) = ? AND status = 'closed'
        ''', (date,))
        row = cursor.fetchone()
        return {
            'date': date,
            'n_trades': row[0],
            'total_pnl': row[1],
            'win_rate': row[2],
        }

    def save_config(self, key: str, value):
        """Save a configuration value."""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO model_config (key, value, updated_at)
            VALUES (?, ?, ?)
        ''', (key, json.dumps(value), datetime.now(timezone.utc).isoformat()))
        self.conn.commit()

    def load_config(self, key: str, default=None):
        """Load a configuration value."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT value FROM model_config WHERE key = ?", (key,))
        row = cursor.fetchone()
        if row:
            return json.loads(row[0])
        return default

    def close(self):
        """Close database connection."""
        self.conn.close()


class RealTimeFeatureEngine:
    """Compute features from a rolling window of bars for real-time use.
    Includes order flow proxies and cross-asset features."""

    def __init__(self, groupings: List[int], feature_names: List[str]):
        self.groupings = groupings
        self.feature_names = feature_names
        self.max_lookback = max(max(groupings) + 10, 600)  # At least 600 for cross-asset
        self.bar_buffer = []
        # Lazy-load feature engines
        self._of_engine = None
        self._ca_engine = None

    def _get_of_engine(self):
        if self._of_engine is None:
            from core.order_flow_features import OrderFlowProxy
            self._of_engine = OrderFlowProxy()
        return self._of_engine

    def _get_ca_engine(self):
        if self._ca_engine is None:
            from core.cross_asset_features import CrossAssetFeatureProvider
            self._ca_engine = CrossAssetFeatureProvider(mode='simulate')
        return self._ca_engine

    def update(self, bar: Dict):
        """Add a new bar to the buffer."""
        self.bar_buffer.append(bar)
        if len(self.bar_buffer) > self.max_lookback * 2:
            self.bar_buffer = self.bar_buffer[-self.max_lookback:]

    def compute_features(self) -> Optional[np.ndarray]:
        """Compute current feature vector from bar buffer."""
        if len(self.bar_buffer) < max(self.groupings):
            return None

        df = pd.DataFrame(self.bar_buffer[-self.max_lookback:])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Base features
        df['range'] = df['high'] - df['low']
        df['body'] = df['close'] - df['open']
        df['direction'] = np.sign(df['body'])
        df['return_log'] = np.log(df['close'] / df['close'].shift(1))

        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        features = {}

        for size in self.groupings:
            features[f'grp_{size}_range'] = (high.rolling(size).max() - low.rolling(size).min()).iloc[-1]
            features[f'grp_{size}_range_pct'] = features[f'grp_{size}_range'] / close.rolling(size).mean().iloc[-1]
            features[f'grp_{size}_return'] = (close.iloc[-1] - close.iloc[-size])
            features[f'grp_{size}_return_pct'] = close.pct_change(size).iloc[-1]
            features[f'grp_{size}_body'] = close.iloc[-1] - df['open'].iloc[-size]
            grp_range = features[f'grp_{size}_range']
            features[f'grp_{size}_body_range_ratio'] = abs(features[f'grp_{size}_body']) / grp_range if grp_range > 0 else 0
            features[f'grp_{size}_volatility'] = df['return_log'].rolling(size).std().iloc[-1]
            vol_curr = df['return_log'].rolling(size).std().iloc[-1]
            vol_prev = df['return_log'].rolling(size).std().iloc[-size] if len(df) > size * 2 else vol_curr
            features[f'grp_{size}_vol_change'] = vol_curr / vol_prev if vol_prev > 0 else 1
            features[f'grp_{size}_vol_total'] = volume.rolling(size).sum().iloc[-1]
            features[f'grp_{size}_vol_mean'] = volume.rolling(size).mean().iloc[-1]
            half = max(1, size // 2)
            vol_recent = volume.rolling(half).mean().iloc[-1]
            vol_full = volume.rolling(size).mean().iloc[-1]
            features[f'grp_{size}_vol_trend'] = vol_recent / vol_full if vol_full > 0 else 1
            features[f'grp_{size}_momentum'] = close.iloc[-1] - close.rolling(size).mean().iloc[-1]
            total_path = df['range'].rolling(size).sum().iloc[-1]
            net_move = abs(close.iloc[-1] - close.iloc[-size])
            features[f'grp_{size}_efficiency'] = net_move / total_path if total_path > 0 else 0
            features[f'grp_{size}_up_pct'] = df['direction'].rolling(size).apply(
                lambda x: (x > 0).sum() / len(x), raw=True
            ).iloc[-1]
            gh = high.rolling(size).max().iloc[-1]
            gl = low.rolling(size).min().iloc[-1]
            gr = gh - gl
            features[f'grp_{size}_close_pos'] = (close.iloc[-1] - gl) / gr if gr > 0 else 0.5

        # Cross-group features
        short_g, long_g = self.groupings[0], self.groupings[-1]
        sv = df['return_log'].rolling(short_g).std().iloc[-1]
        lv = df['return_log'].rolling(long_g).std().iloc[-1]
        features['vol_ratio_short_long'] = sv / lv if lv > 0 else 1

        for i in range(len(self.groupings) - 1):
            g1, g2 = self.groupings[i], self.groupings[i + 1]
            r1 = features.get(f'grp_{g1}_return', 0)
            r2 = features.get(f'grp_{g2}_return', 0)
            features[f'mom_align_{g1}_{g2}'] = np.sign(r1) * np.sign(r2)

        eff_vals = [features.get(f'grp_{s}_efficiency', 0) for s in self.groupings]
        features['mean_efficiency'] = np.mean(eff_vals)
        features['eff_dispersion'] = np.std(eff_vals)

        # Base features
        features['volatility_20'] = df['return_log'].rolling(20).std().iloc[-1]
        features['volatility_60'] = df['return_log'].rolling(60).std().iloc[-1]
        vol_ma = volume.rolling(20).mean().iloc[-1]
        features['volume_ratio'] = volume.iloc[-1] / vol_ma if vol_ma > 0 else 1
        # Time features from actual timestamp
        if 'timestamp' in df.columns:
            features['hour'] = pd.to_datetime(df['timestamp'].iloc[-1]).hour
            features['day_of_week'] = pd.to_datetime(df['timestamp'].iloc[-1]).weekday()
        else:
            features['hour'] = 0
            features['day_of_week'] = 0
        features['momentum_5'] = close.iloc[-1] - close.iloc[-5]
        features['momentum_20'] = close.iloc[-1] - close.iloc[-20]

        # Morphology encodings
        for g in self.groupings:
            if len(df) >= g:
                net_ret = close.iloc[-1] - close.iloc[-g]
                grp_range_v = high.rolling(g).max().iloc[-1] - low.rolling(g).min().iloc[-1]
                eff = abs(net_ret) / grp_range_v if grp_range_v > 0 else 0
                cp = (close.iloc[-1] - low.rolling(g).min().iloc[-1]) / grp_range_v if grp_range_v > 0 else 0.5
                morph = 0
                if net_ret > 0 and eff > 0.6:
                    morph = 1
                elif net_ret < 0 and eff > 0.6:
                    morph = 2
                elif cp > 0.8 and net_ret > 0:
                    morph = 3
                elif cp < 0.2 and net_ret < 0:
                    morph = 4
                features[f'morph_{g}_encoded'] = morph
            else:
                features[f'morph_{g}_encoded'] = 0

        # === IMPROVEMENT 1: Order Flow Proxy Features ===
        of_feats = self._get_of_engine().compute_single_bar(self.bar_buffer)
        features.update(of_feats)

        # === IMPROVEMENT 2: Cross-Asset Features ===
        ca_feats = self._get_ca_engine().compute_single_bar(self.bar_buffer)
        features.update(ca_feats)

        # Build feature vector in correct order
        feature_vector = []
        for name in self.feature_names:
            val = features.get(name, 0)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                val = 0
            feature_vector.append(val)

        return np.array(feature_vector).reshape(1, -1)


class SignalEngine:
    """
    Generates trading signals by combining ML models with filters.
    Includes magnitude gate (Improvement 4) and regime filter (Improvement 3).
    Designed for real-time operation.
    """

    def __init__(self, model_path: str, config):
        self.config = config
        self.models = {}
        self.scaler = None
        self.model_weights = {}
        self.feature_engine = None
        self.vol_history = pd.Series(dtype=float)  # For regime filter
        self._regime_filter = None
        self._load_models(model_path)

    def _get_regime_filter(self):
        if self._regime_filter is None:
            from core.regime_filter import RegimeFilter
            self._regime_filter = RegimeFilter(self.config)
        return self._regime_filter

    def _load_models(self, path: str):
        """Load trained models from disk."""
        model_files = {
            'xgb_direction': 'xgb_direction.pkl',
            'xgb_return': 'xgb_return.pkl',
            'lgb_direction': 'lgb_direction.pkl',
            'lgb_return': 'lgb_return.pkl',
            'xgb_magnitude': 'xgb_magnitude.pkl',
        }

        for name, filename in model_files.items():
            filepath = os.path.join(path, filename)
            if os.path.exists(filepath):
                self.models[name] = joblib.load(filepath)
                logger.info(f"Loaded model: {name}")

        scaler_path = os.path.join(path, 'scaler.pkl')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)

    def generate_signal(self, features: np.ndarray,
                        current_bar: Dict = None) -> Dict:
        """
        Generate a trading signal from features.
        Applies magnitude gate and regime filter.

        Returns dict with:
        - direction: 1 (LONG), -1 (SHORT), 0 (FLAT)
        - confidence: 0-1
        - expected_return: in points
        - expected_value: EV in points
        - is_valid: whether signal passes all filters
        - regime_active: whether regime conditions are met
        - magnitude_gate: whether magnitude classifier predicts big move
        """
        if self.scaler is None or not self.models:
            return {'direction': 0, 'confidence': 0, 'is_valid': False,
                    'regime_active': False, 'magnitude_gate': False}

        # Handle NaN in features
        features_clean = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        X = self.scaler.transform(features_clean)

        # Direction probability
        prob_up = 0.5
        n_dir_models = 0
        for name in ['xgb_direction', 'lgb_direction']:
            if name in self.models:
                p = self.models[name].predict_proba(X)[0, 1]
                prob_up += p
                n_dir_models += 1
        if n_dir_models > 0:
            prob_up = (prob_up - 0.5) / n_dir_models  # Average

        # Return prediction
        ret_pred = 0.0
        n_ret_models = 0
        for name in ['xgb_return', 'lgb_return']:
            if name in self.models:
                r = self.models[name].predict(X)[0]
                ret_pred += r
                n_ret_models += 1
        if n_ret_models > 0:
            ret_pred /= n_ret_models

        # Magnitude probability
        prob_big_move = 0.0
        if 'xgb_magnitude' in self.models:
            prob_big_move = self.models['xgb_magnitude'].predict_proba(X)[0, 1]

        # Direction
        if prob_up > 0.5:
            direction = 1
        elif prob_up < 0.5:
            direction = -1
        else:
            direction = 0

        # Confidence
        confidence = abs(prob_up - 0.5) * 2 * max(prob_big_move, 0.3)

        # Expected value
        costs = (self.config.trading.commission_per_side + self.config.trading.slippage_per_side) * 2
        expected_value = abs(ret_pred) * confidence - costs

        # === IMPROVEMENT 4: Magnitude gate ===
        mag_threshold = getattr(self.config.trading, 'magnitude_prob_threshold', 0.5)
        magnitude_gate = prob_big_move >= mag_threshold

        # === IMPROVEMENT 3: Regime filter ===
        regime_active = True
        if getattr(self.config.trading, 'regime_filter_enabled', True) and current_bar:
            # Get current hour (UTC)
            ts = current_bar.get('timestamp', '')
            try:
                hour_utc = pd.to_datetime(ts).hour
            except Exception:
                hour_utc = datetime.now(timezone.utc).hour

            # Update vol history
            vol_20 = features_clean[0][self._find_feature_idx('volatility_20')] if features_clean.shape[1] > 0 else 0
            self.vol_history = pd.concat([
                self.vol_history, pd.Series([vol_20])
            ]).tail(10000)

            regime_info = self._get_regime_filter().check_current_regime(
                current_vol_20=vol_20,
                current_hour_utc=hour_utc,
                vol_history=self.vol_history,
            )
            regime_active = regime_info['regime_active']

        # Combined validity with ALL filters
        is_valid = (
            confidence >= self.config.trading.min_confidence and
            expected_value >= self.config.trading.min_expected_value_points and
            direction != 0 and
            magnitude_gate and     # IMPROVEMENT 4
            regime_active          # IMPROVEMENT 3
        )

        return {
            'direction': direction,
            'confidence': float(confidence),
            'prob_up': float(prob_up),
            'prob_big_move': float(prob_big_move),
            'expected_return': float(ret_pred),
            'expected_value': float(expected_value),
            'is_valid': is_valid,
            'magnitude_gate': magnitude_gate,
            'regime_active': regime_active,
            'timestamp': datetime.now(timezone.utc).isoformat(),
        }

    def _find_feature_idx(self, name: str) -> int:
        """Find index of a feature by name (for real-time feature extraction)."""
        if self.feature_engine and hasattr(self.feature_engine, 'feature_names'):
            try:
                return self.feature_engine.feature_names.index(name)
            except ValueError:
                pass
        return 0


class TradingBot:
    """
    Main trading bot class that orchestrates signal generation,
    position management, and trade execution.
    """

    def __init__(self, config, model_path: str = './data/trained_models/',
                 db_path: str = './data/trading_core.db'):
        self.config = config
        self.db = TradingDatabase(db_path)
        self.signal_engine = SignalEngine(model_path, config)
        self.feature_engine = None
        self.is_running = False

        # Load feature names and groupings from saved config
        self._load_runtime_config()

    def _load_runtime_config(self):
        """Load runtime configuration from database or files."""
        groupings_path = './data/optimal_groupings.json'
        if os.path.exists(groupings_path):
            with open(groupings_path) as f:
                groupings = json.load(f)
        else:
            groupings = [25, 30, 45, 60, 90, 120, 180, 240]

        # Try loading feature names from JSON first (faster), fall back to parquet
        feature_names_path = './data/trained_models/feature_names.json'
        features_path = './data/features_all.parquet'
        if os.path.exists(feature_names_path):
            with open(feature_names_path) as f:
                feature_names = json.load(f)
        elif os.path.exists(features_path):
            feature_names = pd.read_parquet(features_path, columns=[]).columns.tolist()
        else:
            feature_names = []

        if feature_names:
            self.feature_engine = RealTimeFeatureEngine(groupings, feature_names)
            self.signal_engine.feature_engine = self.feature_engine

    def process_bar(self, bar: Dict) -> Optional[Dict]:
        """
        Process a new bar and potentially generate a signal/trade.

        Args:
            bar: Dict with keys: timestamp, open, high, low, close, volume

        Returns:
            Signal dict if a signal was generated, None otherwise
        """
        if self.feature_engine is None:
            logger.warning("Feature engine not initialized")
            return None

        # Update feature engine
        self.feature_engine.update(bar)

        # Compute features
        features = self.feature_engine.compute_features()
        if features is None:
            return None

        # Generate signal (with regime + magnitude filters)
        signal = self.signal_engine.generate_signal(features, current_bar=bar)
        signal['timestamp'] = bar.get('timestamp', datetime.now(timezone.utc).isoformat())

        # Store signal in database
        signal_id = self.db.insert_signal(signal)

        # Check for open positions
        open_trades = self.db.get_open_trades()

        if signal['is_valid'] and not open_trades:
            # Calculate stops
            recent_bars = self.feature_engine.bar_buffer[-20:]
            atr = np.mean([b['high'] - b['low'] for b in recent_bars])
            entry_price = bar['close']

            if signal['direction'] == 1:
                sl = entry_price - atr * self.config.trading.stop_loss_multiplier
                tp = entry_price + atr * self.config.trading.take_profit_multiplier
            else:
                sl = entry_price + atr * self.config.trading.stop_loss_multiplier
                tp = entry_price - atr * self.config.trading.take_profit_multiplier

            trade = {
                'signal_id': signal_id,
                'direction': signal['direction'],
                'entry_price': entry_price,
                'stop_loss': sl,
                'take_profit': tp,
                'entry_time': signal['timestamp'],
            }
            trade_id = self.db.insert_trade(trade)
            signal['trade_opened'] = True
            signal['trade_id'] = trade_id

            logger.info(f"TRADE OPENED: {'LONG' if signal['direction']==1 else 'SHORT'} "
                       f"@ {entry_price:.2f} | SL={sl:.2f} TP={tp:.2f} | "
                       f"Conf={signal['confidence']:.2f} EV={signal['expected_value']:.1f}")

        elif open_trades:
            # Check stop/TP for open positions
            for trade in open_trades:
                self._manage_position(trade, bar)

        return signal

    def _manage_position(self, trade: Dict, bar: Dict):
        """Check and manage an open position."""
        direction = trade['direction']
        entry_price = trade['entry_price']
        sl = trade['stop_loss']
        tp = trade['take_profit']
        current_high = bar['high']
        current_low = bar['low']
        current_close = bar['close']

        exit_price = None
        exit_reason = None

        if direction == 1:  # LONG
            if current_low <= sl:
                exit_price = sl
                exit_reason = 'stop_loss'
            elif current_high >= tp:
                exit_price = tp
                exit_reason = 'take_profit'
        else:  # SHORT
            if current_high >= sl:
                exit_price = sl
                exit_reason = 'stop_loss'
            elif current_low <= tp:
                exit_price = tp
                exit_reason = 'take_profit'

        if exit_price is not None:
            pnl_points = direction * (exit_price - entry_price)
            costs = (self.config.trading.commission_per_side +
                    self.config.trading.slippage_per_side) * 2
            pnl_net = pnl_points - costs

            self.db.close_trade(
                trade['id'], exit_price, exit_reason, pnl_points, pnl_net
            )

            logger.info(f"TRADE CLOSED: {exit_reason} @ {exit_price:.2f} | "
                       f"PnL={pnl_net:.1f} pts")

    def get_status(self) -> Dict:
        """Get current bot status."""
        open_trades = self.db.get_open_trades()
        daily_perf = self.db.get_daily_performance()
        recent_signals = self.db.get_recent_signals(10)

        return {
            'open_positions': len(open_trades),
            'open_trades': open_trades,
            'daily_performance': daily_perf,
            'recent_signals': recent_signals,
        }

    def shutdown(self):
        """Gracefully shutdown the bot."""
        self.is_running = False
        self.db.close()
        logger.info("Trading bot shutdown complete")
