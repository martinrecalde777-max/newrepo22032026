"""
Backtesting Engine
==================
Motor de backtesting completo para el sistema de trading MNQ.

Características:
- Simulación bar-a-bar con slippage y comisiones reales
- Gestión de posiciones con stop-loss, take-profit, trailing stop
- Métricas exhaustivas: Sharpe, Sortino, Calmar, max drawdown, win rate
- Monte Carlo para estimación de distribución de retornos
- Análisis por hora del día, día de la semana, régimen de mercado
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from scipy import stats
import logging

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a single completed trade."""
    entry_idx: int
    exit_idx: int
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: int  # 1 for LONG, -1 for SHORT
    entry_price: float
    exit_price: float
    pnl_points: float
    pnl_net: float  # After costs
    bars_held: int
    exit_reason: str
    signal_confidence: float
    signal_ev: float
    prob_big_move: float = 0.0       # Magnitude classifier probability
    regime_active: bool = True        # Whether regime filter was active at entry


@dataclass
class Position:
    """Represents an open position."""
    entry_idx: int
    entry_time: pd.Timestamp
    direction: int
    entry_price: float
    stop_loss: float
    take_profit: float
    trailing_stop: Optional[float] = None
    trailing_activated: bool = False
    best_price: float = 0.0
    signal_confidence: float = 0.0
    signal_ev: float = 0.0
    prob_big_move: float = 0.0
    regime_active: bool = True


class BacktestEngine:
    """
    High-fidelity backtesting engine for MNQ intraday trading.
    """

    def __init__(self, config):
        self.config = config
        self.trades: List[Trade] = []
        self.equity_curve = []
        self.position: Optional[Position] = None
        self.daily_pnl = {}

    def run(self, df: pd.DataFrame, signals: pd.DataFrame) -> Dict:
        """
        Run backtest on historical data with trading signals.

        Args:
            df: OHLCV data with timestamps
            signals: DataFrame with columns:
                - direction_numeric: 1 (LONG), -1 (SHORT), 0 (FLAT)
                - expected_return: predicted return in points
                - expected_value: expected value in points
                - confidence: signal confidence 0-1
                - signal_valid: bool, whether signal passes filters

        Returns:
            Dict with comprehensive backtest results
        """
        logger.info("=" * 60)
        logger.info("RUNNING BACKTEST")
        logger.info("=" * 60)

        self.trades = []
        self.equity_curve = []
        self.position = None
        cumulative_pnl = 0.0

        costs_per_trade = (self.config.trading.commission_per_side +
                          self.config.trading.slippage_per_side) * 2

        for i in range(len(df)):
            row = df.iloc[i]
            current_price = row['close']
            current_high = row['high']
            current_low = row['low']
            timestamp = row['timestamp'] if 'timestamp' in df.columns else df.index[i]

            # Check if we have a position to manage
            if self.position is not None:
                exit_reason = self._check_exit_conditions(
                    self.position, current_high, current_low, current_price, i
                )

                if exit_reason:
                    # Close position
                    if self.position.direction == 1:
                        exit_price = current_low if exit_reason == 'stop_loss' else current_price
                        pnl = exit_price - self.position.entry_price
                    else:
                        exit_price = current_high if exit_reason == 'stop_loss' else current_price
                        pnl = self.position.entry_price - exit_price

                    pnl_net = pnl - costs_per_trade

                    trade = Trade(
                        entry_idx=self.position.entry_idx,
                        exit_idx=i,
                        entry_time=self.position.entry_time,
                        exit_time=timestamp,
                        direction=self.position.direction,
                        entry_price=self.position.entry_price,
                        exit_price=exit_price,
                        pnl_points=pnl,
                        pnl_net=pnl_net,
                        bars_held=i - self.position.entry_idx,
                        exit_reason=exit_reason,
                        signal_confidence=self.position.signal_confidence,
                        signal_ev=self.position.signal_ev,
                    )
                    self.trades.append(trade)
                    cumulative_pnl += pnl_net
                    self.position = None
                else:
                    # Update trailing stop
                    self._update_trailing_stop(self.position, current_high, current_low)

            # Check for new entry signal
            if self.position is None and i < len(signals):
                signal = signals.iloc[i]

                if (signal.get('signal_valid', False) and
                    signal.get('confidence', 0) >= self.config.trading.min_confidence and
                    abs(signal.get('expected_value', 0)) >= self.config.trading.min_expected_value_points):

                    direction = int(signal.get('direction_numeric', 0))
                    if direction != 0:
                        # Calculate stop and target
                        atr = df['range'].iloc[max(0, i - 20):i + 1].mean()
                        if np.isnan(atr) or atr <= 0:
                            atr = 10.0  # Default ATR

                        if direction == 1:
                            stop_loss = current_price - atr * self.config.trading.stop_loss_multiplier
                            take_profit = current_price + atr * self.config.trading.take_profit_multiplier
                        else:
                            stop_loss = current_price + atr * self.config.trading.stop_loss_multiplier
                            take_profit = current_price - atr * self.config.trading.take_profit_multiplier

                        self.position = Position(
                            entry_idx=i,
                            entry_time=timestamp,
                            direction=direction,
                            entry_price=current_price,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            best_price=current_price,
                            signal_confidence=signal.get('confidence', 0),
                            signal_ev=signal.get('expected_value', 0),
                            prob_big_move=signal.get('prob_big_move', 0),
                            regime_active=signal.get('regime_active', True),
                        )

            # Check max drawdown halt
            if cumulative_pnl < -self.config.trading.max_drawdown_points:
                if self.position is not None:
                    pnl = (current_price - self.position.entry_price) * self.position.direction
                    pnl_net = pnl - costs_per_trade
                    trade = Trade(
                        entry_idx=self.position.entry_idx,
                        exit_idx=i,
                        entry_time=self.position.entry_time,
                        exit_time=timestamp,
                        direction=self.position.direction,
                        entry_price=self.position.entry_price,
                        exit_price=current_price,
                        pnl_points=pnl,
                        pnl_net=pnl_net,
                        bars_held=i - self.position.entry_idx,
                        exit_reason='max_drawdown_halt',
                        signal_confidence=self.position.signal_confidence,
                        signal_ev=self.position.signal_ev,
                    )
                    self.trades.append(trade)
                    cumulative_pnl += pnl_net
                    self.position = None
                logger.warning(f"Max drawdown reached at bar {i}. Halting.")
                break

            self.equity_curve.append(cumulative_pnl)

        # Close any remaining position
        if self.position is not None:
            last_price = df.iloc[-1]['close']
            pnl = (last_price - self.position.entry_price) * self.position.direction
            pnl_net = pnl - costs_per_trade
            trade = Trade(
                entry_idx=self.position.entry_idx,
                exit_idx=len(df) - 1,
                entry_time=self.position.entry_time,
                exit_time=df.iloc[-1].get('timestamp', df.index[-1]),
                direction=self.position.direction,
                entry_price=self.position.entry_price,
                exit_price=last_price,
                pnl_points=pnl,
                pnl_net=pnl_net,
                bars_held=len(df) - 1 - self.position.entry_idx,
                exit_reason='end_of_data',
                signal_confidence=self.position.signal_confidence,
                signal_ev=self.position.signal_ev,
            )
            self.trades.append(trade)
            self.position = None

        # Compute metrics
        results = self._compute_metrics(df)
        return results

    def _check_exit_conditions(self, pos: Position, high: float, low: float,
                                close: float, bar_idx: int) -> Optional[str]:
        """Check all exit conditions for an open position."""
        if pos.direction == 1:  # LONG
            if low <= pos.stop_loss:
                return 'stop_loss'
            if high >= pos.take_profit:
                return 'take_profit'
            if pos.trailing_stop is not None and low <= pos.trailing_stop:
                return 'trailing_stop'
        else:  # SHORT
            if high >= pos.stop_loss:
                return 'stop_loss'
            if low <= pos.take_profit:
                return 'take_profit'
            if pos.trailing_stop is not None and high >= pos.trailing_stop:
                return 'trailing_stop'

        # Time-based exit (end of day)
        # Not implemented here as we'd need actual session times

        return None

    def _update_trailing_stop(self, pos: Position, high: float, low: float):
        """Update trailing stop if conditions are met."""
        activation = self.config.trading.trailing_stop_activation
        distance = self.config.trading.trailing_stop_distance

        if pos.direction == 1:  # LONG
            pos.best_price = max(pos.best_price, high)
            profit = pos.best_price - pos.entry_price

            if profit >= activation:
                pos.trailing_activated = True
                new_stop = pos.best_price - distance
                if pos.trailing_stop is None or new_stop > pos.trailing_stop:
                    pos.trailing_stop = new_stop
        else:  # SHORT
            pos.best_price = min(pos.best_price, low) if pos.best_price > 0 else low
            profit = pos.entry_price - pos.best_price

            if profit >= activation:
                pos.trailing_activated = True
                new_stop = pos.best_price + distance
                if pos.trailing_stop is None or new_stop < pos.trailing_stop:
                    pos.trailing_stop = new_stop

    def _compute_metrics(self, df: pd.DataFrame) -> Dict:
        """Compute comprehensive backtest metrics."""
        if not self.trades:
            logger.warning("No trades executed in backtest")
            return {'n_trades': 0, 'total_pnl': 0}

        trades_df = pd.DataFrame([{
            'entry_time': t.entry_time,
            'exit_time': t.exit_time,
            'direction': t.direction,
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'pnl_points': t.pnl_points,
            'pnl_net': t.pnl_net,
            'bars_held': t.bars_held,
            'exit_reason': t.exit_reason,
            'confidence': t.signal_confidence,
            'signal_ev': t.signal_ev,
            'prob_big_move': t.prob_big_move,
            'regime_active': t.regime_active,
        } for t in self.trades])

        pnl = trades_df['pnl_net'].values
        n_trades = len(pnl)
        winners = pnl[pnl > 0]
        losers = pnl[pnl <= 0]

        # Basic metrics
        total_pnl = pnl.sum()
        win_rate = len(winners) / n_trades if n_trades > 0 else 0
        avg_win = winners.mean() if len(winners) > 0 else 0
        avg_loss = losers.mean() if len(losers) > 0 else 0
        profit_factor = abs(winners.sum() / losers.sum()) if len(losers) > 0 and losers.sum() != 0 else float('inf')
        expectancy = pnl.mean() if n_trades > 0 else 0

        # Equity curve metrics
        equity = np.cumsum(pnl)
        running_max = np.maximum.accumulate(equity)
        drawdowns = equity - running_max
        max_drawdown = drawdowns.min()
        max_drawdown_pct = max_drawdown / max(running_max.max(), 1) * 100

        # Risk metrics
        if pnl.std() > 0:
            sharpe = pnl.mean() / pnl.std() * np.sqrt(252)  # Annualized approx
        else:
            sharpe = 0

        downside_returns = pnl[pnl < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino = pnl.mean() / downside_returns.std() * np.sqrt(252)
        else:
            sortino = 0

        if abs(max_drawdown) > 0:
            calmar = total_pnl / abs(max_drawdown)
        else:
            calmar = 0

        # By exit reason
        exit_stats = trades_df.groupby('exit_reason').agg(
            count=('pnl_net', 'count'),
            total_pnl=('pnl_net', 'sum'),
            avg_pnl=('pnl_net', 'mean'),
            win_rate=('pnl_net', lambda x: (x > 0).mean())
        ).to_dict('index')

        # By direction
        direction_stats = {}
        for d in [1, -1]:
            d_trades = trades_df[trades_df['direction'] == d]
            if len(d_trades) > 0:
                d_pnl = d_trades['pnl_net'].values
                direction_stats['LONG' if d == 1 else 'SHORT'] = {
                    'n_trades': len(d_trades),
                    'total_pnl': float(d_pnl.sum()),
                    'win_rate': float((d_pnl > 0).mean()),
                    'avg_pnl': float(d_pnl.mean()),
                }

        # Consecutive wins/losses
        is_win = pnl > 0
        max_consec_wins = self._max_consecutive(is_win, True)
        max_consec_losses = self._max_consecutive(is_win, False)

        # Monte Carlo simulation
        mc_results = self._monte_carlo_simulation(pnl, n_simulations=1000)

        results = {
            'n_trades': n_trades,
            'total_pnl_points': float(total_pnl),
            'total_pnl_usd': float(total_pnl * self.config.data.point_value),
            'win_rate': float(win_rate),
            'avg_win_points': float(avg_win),
            'avg_loss_points': float(avg_loss),
            'profit_factor': float(profit_factor),
            'expectancy_per_trade': float(expectancy),
            'expectancy_usd': float(expectancy * self.config.data.point_value),
            'max_drawdown_points': float(max_drawdown),
            'max_drawdown_pct': float(max_drawdown_pct),
            'sharpe_ratio': float(sharpe),
            'sortino_ratio': float(sortino),
            'calmar_ratio': float(calmar),
            'avg_bars_held': float(trades_df['bars_held'].mean()),
            'max_consec_wins': int(max_consec_wins),
            'max_consec_losses': int(max_consec_losses),
            'exit_stats': exit_stats,
            'direction_stats': direction_stats,
            'mc_95_percentile_drawdown': float(mc_results['dd_95']),
            'mc_median_total_pnl': float(mc_results['median_pnl']),
            'mc_5_percentile_pnl': float(mc_results['pnl_5']),
            'trades_df': trades_df,
            'equity_curve': equity.tolist(),
        }

        self._log_results(results)
        return results

    def _max_consecutive(self, series: np.ndarray, value: bool) -> int:
        """Count maximum consecutive occurrences of a value."""
        max_count = 0
        current = 0
        for v in series:
            if v == value:
                current += 1
                max_count = max(max_count, current)
            else:
                current = 0
        return max_count

    def _monte_carlo_simulation(self, pnl: np.ndarray, n_simulations: int = 1000) -> Dict:
        """Monte Carlo simulation of trade sequences."""
        total_pnls = []
        max_drawdowns = []

        for _ in range(n_simulations):
            # Reshuffle trade PnLs
            shuffled = np.random.choice(pnl, size=len(pnl), replace=True)
            equity = np.cumsum(shuffled)
            total_pnls.append(equity[-1])

            running_max = np.maximum.accumulate(equity)
            dd = (equity - running_max).min()
            max_drawdowns.append(dd)

        return {
            'median_pnl': float(np.median(total_pnls)),
            'pnl_5': float(np.percentile(total_pnls, 5)),
            'pnl_95': float(np.percentile(total_pnls, 95)),
            'dd_95': float(np.percentile(max_drawdowns, 5)),  # 95th worst
            'dd_median': float(np.median(max_drawdowns)),
        }

    def _log_results(self, results: Dict):
        """Log backtest results."""
        logger.info("\n" + "=" * 60)
        logger.info("BACKTEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"  Total trades:        {results['n_trades']}")
        logger.info(f"  Total PnL:           {results['total_pnl_points']:.1f} pts "
                    f"(${results['total_pnl_usd']:.2f})")
        logger.info(f"  Win rate:            {results['win_rate']:.1%}")
        logger.info(f"  Avg win:             {results['avg_win_points']:.1f} pts")
        logger.info(f"  Avg loss:            {results['avg_loss_points']:.1f} pts")
        logger.info(f"  Profit factor:       {results['profit_factor']:.2f}")
        logger.info(f"  Expectancy/trade:    {results['expectancy_per_trade']:.1f} pts "
                    f"(${results['expectancy_usd']:.2f})")
        logger.info(f"  Max drawdown:        {results['max_drawdown_points']:.1f} pts")
        logger.info(f"  Sharpe ratio:        {results['sharpe_ratio']:.2f}")
        logger.info(f"  Sortino ratio:       {results['sortino_ratio']:.2f}")
        logger.info(f"  Calmar ratio:        {results['calmar_ratio']:.2f}")
        logger.info(f"  Avg bars held:       {results['avg_bars_held']:.0f}")
        logger.info(f"  Max consec wins:     {results['max_consec_wins']}")
        logger.info(f"  Max consec losses:   {results['max_consec_losses']}")
        logger.info(f"\n  Monte Carlo (1000 sims):")
        logger.info(f"    Median PnL:        {results['mc_median_total_pnl']:.1f} pts")
        logger.info(f"    5th pctl PnL:      {results['mc_5_percentile_pnl']:.1f} pts")
        logger.info(f"    95th pctl DD:      {results['mc_95_percentile_drawdown']:.1f} pts")

        if results.get('direction_stats'):
            logger.info(f"\n  By Direction:")
            for d, s in results['direction_stats'].items():
                logger.info(f"    {d}: {s['n_trades']} trades, "
                           f"PnL={s['total_pnl']:.1f}pts, WR={s['win_rate']:.1%}")

        if results.get('exit_stats'):
            logger.info(f"\n  By Exit Reason:")
            for reason, s in results['exit_stats'].items():
                logger.info(f"    {reason}: {s['count']} trades, "
                           f"PnL={s['total_pnl']:.1f}pts, WR={s['win_rate']:.1%}")
