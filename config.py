"""
MNQ Intraday Trading System - Configuration
=============================================
Sistema de trading cuantitativo para Micro Nasdaq Futures (MNQ)
Desarrollado con metodologías de física aplicada y ciencias computacionales.
"""

import os
from dataclasses import dataclass, field
from typing import List

@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    csv_path: str = '/Users/martincode/Library/Mobile Documents/com~apple~CloudDocs/glbx-mdp3-20210312-20260311.ohlcv-1m.csv'
    # Fallback paths
    fallback_paths: List[str] = field(default_factory=lambda: [
        './data/glbx-mdp3-20210312-20260311.ohlcv-1m.csv',
        '../glbx-mdp3-20210312-20260311.ohlcv-1m.csv',
    ])
    symbol: str = 'MNQ'
    timeframe: str = '1min'
    tick_value: float = 0.25  # MNQ tick size
    point_value: float = 2.0  # USD per point for MNQ
    contract_value: float = 2.0  # Multiplier

@dataclass
class GroupingConfig:
    """Configuration for bar grouping analysis."""
    # Candidate group sizes to evaluate
    candidate_sizes: List[int] = field(default_factory=lambda: [
        3, 5, 7, 8, 10, 12, 15, 20, 25, 30, 45, 60, 90, 120, 180, 240
    ])
    # Minimum mutual information threshold for a grouping to be considered relevant
    min_mutual_info: float = 0.01
    # Minimum predictive power (R² or accuracy) to keep a grouping
    min_predictive_power: float = 0.02
    # Maximum number of groupings to keep
    max_groupings: int = 8

@dataclass
class ModelConfig:
    """Configuration for predictive models."""
    # Markov chain
    n_markov_states: int = 8
    markov_lookback: int = 3

    # Hidden Markov Model
    n_hmm_states: int = 5
    hmm_iterations: int = 200

    # Bayesian
    bayesian_prior_strength: float = 1.0
    bayesian_min_samples: int = 30

    # Ensemble
    xgb_n_estimators: int = 500
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.05
    lgb_n_estimators: int = 500
    lgb_max_depth: int = 8
    lgb_learning_rate: float = 0.05

    # Train/test split
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15

    # Walk-forward
    walk_forward_windows: int = 10
    walk_forward_train_pct: float = 0.80

@dataclass
class TradingConfig:
    """Configuration for trading rules and backtesting."""
    # Minimum expected value per trade in points
    min_expected_value_points: float = 50.0
    # Minimum confidence threshold
    min_confidence: float = 0.60
    # Maximum positions
    max_positions: int = 1
    # Commission per side in points
    commission_per_side: float = 0.52  # ~$1.04 round trip for MNQ
    # Slippage per side in points
    slippage_per_side: float = 1.0
    # Maximum drawdown before halt (in points)
    max_drawdown_points: float = 500.0
    # Trading hours (CME MNQ)
    trading_start_hour: int = 9   # 9:30 ET approximated
    trading_start_minute: int = 30
    trading_end_hour: int = 16
    trading_end_minute: int = 0
    # Risk management
    stop_loss_multiplier: float = 1.5
    take_profit_multiplier: float = 2.5
    trailing_stop_activation: float = 30.0  # points
    trailing_stop_distance: float = 15.0    # points

@dataclass
class BotConfig:
    """Configuration for real-time trading bot."""
    db_path: str = './data/trading_core.db'
    model_path: str = './data/trained_models/'
    signal_refresh_seconds: int = 5
    heartbeat_seconds: int = 30
    log_path: str = './data/logs/'

@dataclass
class SystemConfig:
    """Master configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    grouping: GroupingConfig = field(default_factory=GroupingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    bot: BotConfig = field(default_factory=BotConfig)
    report_path: str = './reports/'
    random_seed: int = 42
    n_jobs: int = -1  # Use all CPU cores
