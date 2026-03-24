"""Backtester — simulates trading signals on bar-level MNQ data.

Realistic execution model:
    - Entry at signal bar close + slippage
    - Exit on stop, target, or max holding period (whichever comes first)
    - Per-trade commission (round-trip)
    - Bar-by-bar simulation using high/low for stop/target checks

Metrics:
    - Total PnL, net of commissions and slippage
    - Win rate, profit factor, max drawdown
    - Sharpe ratio (annualized, assuming 252 trading days)
    - Average trade duration
    - Expectancy per trade
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# MNQ contract specs
MNQ_TICK_SIZE = 0.25       # minimum price increment
MNQ_TICK_VALUE = 0.50      # $0.50 per tick
MNQ_POINT_VALUE = 2.00     # $2.00 per point (1 point = 4 ticks)


@dataclass
class TradeResult:
    """Single completed trade."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str
    entry_price: float
    exit_price: float
    pnl_points: float
    pnl_dollars: float
    commission: float
    net_pnl: float
    exit_reason: str          # 'stop', 'target', 'timeout'
    bars_held: int
    setup_key: str
    confidence: float


@dataclass
class BacktestResult:
    """Full backtest summary."""
    trades: list[TradeResult] = field(default_factory=list)
    total_signals: int = 0
    total_trades: int = 0
    winners: int = 0
    losers: int = 0
    breakeven: int = 0
    gross_pnl_pts: float = 0.0
    total_commission: float = 0.0
    net_pnl_pts: float = 0.0
    net_pnl_dollars: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_pts: float = 0.0
    max_drawdown_dollars: float = 0.0
    sharpe_ratio: float = 0.0
    avg_win_pts: float = 0.0
    avg_loss_pts: float = 0.0
    avg_bars_held: float = 0.0
    expectancy_pts: float = 0.0
    expectancy_dollars: float = 0.0
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "BACKTEST RESULTS — MNQ Morphology Signals",
            "=" * 60,
            f"  Total signals:     {self.total_signals:>8,}",
            f"  Trades executed:   {self.total_trades:>8,}",
            f"  Winners:           {self.winners:>8,}  ({self.win_rate:.1%})",
            f"  Losers:            {self.losers:>8,}",
            f"  Breakeven:         {self.breakeven:>8,}",
            "",
            f"  Gross PnL:         {self.gross_pnl_pts:>+10.1f} pts  (${self.gross_pnl_pts * MNQ_POINT_VALUE:>+10,.0f})",
            f"  Commission:        {self.total_commission:>10.1f} pts  (${self.total_commission * MNQ_POINT_VALUE:>10,.0f})",
            f"  Net PnL:           {self.net_pnl_pts:>+10.1f} pts  (${self.net_pnl_dollars:>+10,.0f})",
            "",
            f"  Profit Factor:     {self.profit_factor:>10.2f}",
            f"  Sharpe Ratio:      {self.sharpe_ratio:>+10.2f}",
            f"  Max Drawdown:      {self.max_drawdown_pts:>10.1f} pts  (${self.max_drawdown_dollars:>10,.0f})",
            "",
            f"  Avg Win:           {self.avg_win_pts:>+10.1f} pts",
            f"  Avg Loss:          {self.avg_loss_pts:>+10.1f} pts",
            f"  Avg Bars Held:     {self.avg_bars_held:>10.1f}",
            f"  Expectancy/Trade:  {self.expectancy_pts:>+10.2f} pts  (${self.expectancy_dollars:>+10.2f})",
            "=" * 60,
        ]
        return "\n".join(lines)


def run_backtest(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    slippage_pts: float = 0.5,
    commission_pts: float = 0.5,
    point_value: float = MNQ_POINT_VALUE,
) -> BacktestResult:
    """Run bar-by-bar backtest on signals.

    Parameters
    ----------
    df : OHLCV DataFrame (1-min bars) with DatetimeIndex
    signals : output of signals.generate_signals() or filter_no_overlap()
    slippage_pts : slippage per side in points (total = 2x)
    commission_pts : round-trip commission in points equivalent
    point_value : dollar value per point ($2 for MNQ)

    Returns
    -------
    BacktestResult with all trades and summary metrics.
    """
    if len(signals) == 0:
        return BacktestResult()

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    ts_index = df.index
    ts_to_loc = {t: i for i, t in enumerate(ts_index)}

    total_slippage = slippage_pts * 2  # entry + exit

    trades: list[TradeResult] = []

    for sig_ts, sig in signals.iterrows():
        if sig_ts not in ts_to_loc:
            continue

        entry_loc = ts_to_loc[sig_ts]
        direction = sig["direction"]
        raw_entry = sig["entry_price"]
        stop_price = sig["stop_price"]
        target_price = sig["target_price"]
        max_bars = int(sig["holding_bars"])

        # Apply entry slippage
        if direction == "long":
            entry_price = raw_entry + slippage_pts
        else:
            entry_price = raw_entry - slippage_pts

        # Walk forward bar by bar
        exit_price = None
        exit_reason = "timeout"
        exit_loc = min(entry_loc + max_bars, len(close) - 1)

        for bar in range(entry_loc + 1, min(entry_loc + max_bars + 1, len(close))):
            bar_high = high[bar]
            bar_low = low[bar]

            if direction == "long":
                # Check stop first (worst case)
                if bar_low <= stop_price:
                    exit_price = stop_price - slippage_pts
                    exit_reason = "stop"
                    exit_loc = bar
                    break
                # Check target
                if bar_high >= target_price:
                    exit_price = target_price - slippage_pts
                    exit_reason = "target"
                    exit_loc = bar
                    break
            else:  # short
                # Check stop first
                if bar_high >= stop_price:
                    exit_price = stop_price + slippage_pts
                    exit_reason = "stop"
                    exit_loc = bar
                    break
                # Check target
                if bar_low <= target_price:
                    exit_price = target_price + slippage_pts
                    exit_reason = "target"
                    exit_loc = bar
                    break

        # Timeout exit at close
        if exit_price is None:
            if direction == "long":
                exit_price = close[exit_loc] - slippage_pts
            else:
                exit_price = close[exit_loc] + slippage_pts

        # Calculate PnL
        if direction == "long":
            pnl_points = exit_price - entry_price
        else:
            pnl_points = entry_price - exit_price

        pnl_dollars = pnl_points * point_value
        commission_dollars = commission_pts * point_value
        net_pnl = pnl_dollars - commission_dollars

        trades.append(TradeResult(
            entry_time=sig_ts,
            exit_time=ts_index[exit_loc],
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl_points=pnl_points,
            pnl_dollars=pnl_dollars,
            commission=commission_dollars,
            net_pnl=net_pnl,
            exit_reason=exit_reason,
            bars_held=exit_loc - entry_loc,
            setup_key=sig.get("setup_key", ""),
            confidence=sig.get("confidence", 0),
        ))

    return _compute_metrics(trades, signals, commission_pts, point_value)


def _compute_metrics(
    trades: list[TradeResult],
    signals: pd.DataFrame,
    commission_pts: float,
    point_value: float,
) -> BacktestResult:
    """Compute summary metrics from trade list."""
    result = BacktestResult(trades=trades, total_signals=len(signals))

    if not trades:
        return result

    net_pnls = np.array([t.net_pnl for t in trades])
    pnl_pts = np.array([t.pnl_points for t in trades])
    bars = np.array([t.bars_held for t in trades])

    result.total_trades = len(trades)
    result.winners = int((net_pnls > 0).sum())
    result.losers = int((net_pnls < 0).sum())
    result.breakeven = int((net_pnls == 0).sum())

    result.gross_pnl_pts = float(pnl_pts.sum())
    result.total_commission = commission_pts * len(trades)
    result.net_pnl_pts = result.gross_pnl_pts - result.total_commission
    result.net_pnl_dollars = result.net_pnl_pts * point_value

    result.win_rate = result.winners / result.total_trades if result.total_trades > 0 else 0

    gross_wins = net_pnls[net_pnls > 0].sum()
    gross_losses = abs(net_pnls[net_pnls < 0].sum())
    result.profit_factor = gross_wins / gross_losses if gross_losses > 0 else np.inf

    win_pts = pnl_pts[pnl_pts > 0]
    loss_pts = pnl_pts[pnl_pts < 0]
    result.avg_win_pts = float(win_pts.mean()) if len(win_pts) > 0 else 0
    result.avg_loss_pts = float(loss_pts.mean()) if len(loss_pts) > 0 else 0

    result.avg_bars_held = float(bars.mean())
    result.expectancy_pts = float(net_pnls.mean()) / point_value
    result.expectancy_dollars = float(net_pnls.mean())

    # Equity curve and drawdown
    equity = np.cumsum(net_pnls)
    result.equity_curve = pd.Series(
        equity,
        index=[t.exit_time for t in trades],
        name="equity",
    )
    running_max = np.maximum.accumulate(equity)
    drawdown = running_max - equity
    result.max_drawdown_dollars = float(drawdown.max()) if len(drawdown) > 0 else 0
    result.max_drawdown_pts = result.max_drawdown_dollars / point_value

    # Sharpe ratio (annualized)
    # Assume ~1500 trades/year as a rough estimate for intraday
    if len(net_pnls) > 1:
        mean_ret = net_pnls.mean()
        std_ret = net_pnls.std()
        if std_ret > 0:
            trades_per_year = min(len(trades), 1500)
            result.sharpe_ratio = (mean_ret / std_ret) * np.sqrt(trades_per_year)
        else:
            result.sharpe_ratio = 0
    else:
        result.sharpe_ratio = 0

    return result


def trades_to_dataframe(result: BacktestResult) -> pd.DataFrame:
    """Convert trade list to DataFrame for analysis."""
    if not result.trades:
        return pd.DataFrame()

    rows = []
    for t in result.trades:
        rows.append({
            "entry_time": t.entry_time,
            "exit_time": t.exit_time,
            "direction": t.direction,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "pnl_pts": t.pnl_points,
            "pnl_dollars": t.pnl_dollars,
            "commission": t.commission,
            "net_pnl": t.net_pnl,
            "exit_reason": t.exit_reason,
            "bars_held": t.bars_held,
            "setup_key": t.setup_key,
            "confidence": t.confidence,
        })
    return pd.DataFrame(rows)


def run_backtest_triggered(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    trigger_type: str = "pullback",
    trigger_window: int = 5,
    trigger_params: dict | None = None,
    slippage_pts: float = 0.5,
    commission_pts: float = 0.5,
    point_value: float = MNQ_POINT_VALUE,
) -> BacktestResult:
    """Trigger-aware backtest: wait for micro-level confirmation before entering.

    Instead of entering at signal bar close, opens a trigger window of N bars
    after the signal. Scans each bar for a trigger condition. If triggered,
    enters at the trigger price. If the window expires, skips the trade.

    Parameters
    ----------
    df : OHLCV DataFrame (1-min bars)
    signals : output of generate_signals() or filter_no_overlap()
    trigger_type : one of 'pullback', 'momentum', 'wick_rejection',
                   'break_retest', 'composite'
    trigger_window : max bars to wait for trigger after signal
    trigger_params : extra kwargs for the trigger function
    slippage_pts : slippage per side
    commission_pts : round-trip commission in points
    point_value : dollar value per point

    Returns
    -------
    BacktestResult with trigger stats added.
    """
    from mnq_morphology.entry_triggers import (
        check_pullback, check_momentum, check_wick_rejection,
        check_break_retest, check_composite,
    )

    if len(signals) == 0:
        return BacktestResult()

    params = trigger_params or {}

    open_arr = df["open"].values.astype(float)
    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    volume = df["volume"].values.astype(float)
    ts_index = df.index
    ts_to_loc = {t: i for i, t in enumerate(ts_index)}

    # Precompute ATR5 and vol_ma20 for trigger checks
    from mnq_morphology.signals import compute_atr
    atr_series = compute_atr(df, 5).values
    vol_ma20 = pd.Series(volume).rolling(20, min_periods=1).mean().values

    total_slippage = slippage_pts * 2
    trades: list[TradeResult] = []
    triggered_count = 0
    skipped_count = 0
    break_state: dict = {}  # state tracker for break_retest

    for sig_idx, (sig_ts, sig) in enumerate(signals.iterrows()):
        if sig_ts not in ts_to_loc:
            continue

        signal_loc = ts_to_loc[sig_ts]
        direction = sig["direction"]
        signal_close = sig["entry_price"]
        stop_price = sig["stop_price"]
        target_price = sig["target_price"]
        max_bars = int(sig["holding_bars"])

        # Signal bar high/low for break_retest
        signal_high = high[signal_loc]
        signal_low = low[signal_loc]

        # Scan trigger window
        entry_price = None
        entry_loc = None

        tw_end = min(signal_loc + trigger_window + 1, len(close))
        for bar in range(signal_loc + 1, tw_end):
            bar_atr = atr_series[bar] if not np.isnan(atr_series[bar]) else 7.0
            bar_vol_ma = vol_ma20[bar] if not np.isnan(vol_ma20[bar]) else 300.0

            fired = False
            trigger_price = 0.0

            if trigger_type == "pullback":
                fired, trigger_price = check_pullback(
                    bar, direction, signal_close,
                    high, low, close, bar_atr,
                    pullback_pct=params.get("pullback_pct", 0.5),
                )
            elif trigger_type == "momentum":
                fired, trigger_price = check_momentum(
                    bar, direction, open_arr, high, low, close, bar_atr,
                    min_body_atr=params.get("min_body_atr", 0.8),
                    min_body_ratio=params.get("min_body_ratio", 0.55),
                )
            elif trigger_type == "wick_rejection":
                fired, trigger_price = check_wick_rejection(
                    bar, direction, open_arr, high, low, close, bar_atr,
                    min_wick_atr=params.get("min_wick_atr", 0.5),
                    max_body_ratio=params.get("max_body_ratio", 0.35),
                )
            elif trigger_type == "break_retest":
                fired, trigger_price = check_break_retest(
                    bar, direction, signal_high, signal_low,
                    high, low, close, break_state, sig_idx,
                )
            elif trigger_type == "composite":
                fired, trigger_price = check_composite(
                    bar, direction, signal_close,
                    open_arr, high, low, close, volume,
                    bar_atr, bar_vol_ma,
                )

            if fired:
                entry_price = trigger_price
                entry_loc = bar
                break

        if entry_price is None:
            skipped_count += 1
            continue

        triggered_count += 1

        # Apply slippage to entry
        if direction == "long":
            entry_price += slippage_pts
        else:
            entry_price -= slippage_pts

        # Recompute stop/target from TRIGGER entry price (not signal close)
        stop_dist = sig["stop_distance"]
        target_dist = sig["target_distance"]
        if direction == "long":
            stop_price = entry_price - stop_dist
            target_price = entry_price + target_dist
        else:
            stop_price = entry_price + stop_dist
            target_price = entry_price - target_dist

        # Walk forward from trigger bar
        exit_price = None
        exit_reason = "timeout"
        exit_loc = min(entry_loc + max_bars, len(close) - 1)

        for bar in range(entry_loc + 1, min(entry_loc + max_bars + 1, len(close))):
            bar_high = high[bar]
            bar_low = low[bar]

            if direction == "long":
                if bar_low <= stop_price:
                    exit_price = stop_price - slippage_pts
                    exit_reason = "stop"
                    exit_loc = bar
                    break
                if bar_high >= target_price:
                    exit_price = target_price - slippage_pts
                    exit_reason = "target"
                    exit_loc = bar
                    break
            else:
                if bar_high >= stop_price:
                    exit_price = stop_price + slippage_pts
                    exit_reason = "stop"
                    exit_loc = bar
                    break
                if bar_low <= target_price:
                    exit_price = target_price + slippage_pts
                    exit_reason = "target"
                    exit_loc = bar
                    break

        if exit_price is None:
            if direction == "long":
                exit_price = close[exit_loc] - slippage_pts
            else:
                exit_price = close[exit_loc] + slippage_pts

        if direction == "long":
            pnl_points = exit_price - entry_price
        else:
            pnl_points = entry_price - exit_price

        pnl_dollars = pnl_points * point_value
        commission_dollars = commission_pts * point_value
        net_pnl = pnl_dollars - commission_dollars

        trades.append(TradeResult(
            entry_time=ts_index[entry_loc],
            exit_time=ts_index[exit_loc],
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl_points=pnl_points,
            pnl_dollars=pnl_dollars,
            commission=commission_dollars,
            net_pnl=net_pnl,
            exit_reason=exit_reason,
            bars_held=exit_loc - entry_loc,
            setup_key=sig.get("setup_key", ""),
            confidence=sig.get("confidence", 0),
        ))

    result = _compute_metrics(trades, signals, commission_pts, point_value)
    # Attach trigger stats
    result.triggered_count = triggered_count
    result.skipped_count = skipped_count
    result.trigger_rate = triggered_count / (triggered_count + skipped_count) if (triggered_count + skipped_count) > 0 else 0
    return result


def analyze_by_setup(result: BacktestResult) -> pd.DataFrame:
    """Break down performance by setup key."""
    df = trades_to_dataframe(result)
    if len(df) == 0:
        return pd.DataFrame()

    groups = df.groupby("setup_key")
    rows = []
    for key, grp in groups:
        net = grp["net_pnl"]
        wins = net[net > 0]
        losses = net[net < 0]
        rows.append({
            "setup_key": key,
            "trades": len(grp),
            "win_rate": (net > 0).mean(),
            "net_pnl": net.sum(),
            "avg_pnl": net.mean(),
            "profit_factor": wins.sum() / losses.abs().sum() if losses.abs().sum() > 0 else np.inf,
            "avg_bars": grp["bars_held"].mean(),
            "stops": (grp["exit_reason"] == "stop").sum(),
            "targets": (grp["exit_reason"] == "target").sum(),
            "timeouts": (grp["exit_reason"] == "timeout").sum(),
        })

    return pd.DataFrame(rows).sort_values("net_pnl", ascending=False).reset_index(drop=True)


def analyze_by_exit(result: BacktestResult) -> pd.DataFrame:
    """Break down performance by exit reason."""
    df = trades_to_dataframe(result)
    if len(df) == 0:
        return pd.DataFrame()

    groups = df.groupby("exit_reason")
    rows = []
    for reason, grp in groups:
        rows.append({
            "exit_reason": reason,
            "count": len(grp),
            "pct": len(grp) / len(df),
            "avg_pnl": grp["net_pnl"].mean(),
            "total_pnl": grp["net_pnl"].sum(),
            "avg_bars": grp["bars_held"].mean(),
        })
    return pd.DataFrame(rows)
