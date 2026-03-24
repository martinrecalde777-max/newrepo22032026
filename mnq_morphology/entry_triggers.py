"""Precision entry triggers — micro-level confirmation within a trigger window.

Problem: 56% of trades exit by stop. The morphological context (shape + trend +
vol regime) correctly identifies high-EV environments, but entering at signal
bar close is too crude. Price often moves against us before going our way.

Solution: After the context match fires, open a "trigger window" of N bars.
Within that window, check each bar for a micro-level entry condition. If the
trigger fires, enter at the trigger bar's price. If the window expires without
a trigger, skip the trade entirely.

This layer sits BETWEEN signal generation and backtesting. It doesn't change
what setups are identified — it only changes WHEN we enter.

Trigger types:
    1. PULLBACK: Price retraces to a favorable level before continuing
    2. MOMENTUM: Strong directional bar confirms intent
    3. WICK_REJECTION: Failed probe against us → snap back entry
    4. BREAK_RETEST: Break signal bar extreme, then enter on retest
    5. COMPOSITE: Multiple micro-conditions must align
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def check_pullback(
    bar_idx: int,
    direction: str,
    signal_close: float,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr: float,
    pullback_pct: float = 0.5,
) -> tuple[bool, float]:
    """Pullback trigger: price retraces then resumes.

    For longs: bar low dips below signal_close - pullback_depth, then
    bar closes above signal_close (bought the dip).
    For shorts: bar high rises above signal_close + pullback_depth, then
    bar closes below signal_close.

    Returns (triggered, entry_price).
    """
    pullback_depth = atr * pullback_pct
    bar_high = high[bar_idx]
    bar_low = low[bar_idx]
    bar_close = close[bar_idx]

    if direction == "long":
        # Price dipped below our ideal entry zone AND closed back up
        dip_level = signal_close - pullback_depth
        if bar_low <= dip_level and bar_close > signal_close - (pullback_depth * 0.3):
            # Enter at the dip level (limit order fill)
            return True, dip_level
    else:
        # Price spiked above our ideal entry zone AND closed back down
        spike_level = signal_close + pullback_depth
        if bar_high >= spike_level and bar_close < signal_close + (pullback_depth * 0.3):
            return True, spike_level

    return False, 0.0


def check_momentum(
    bar_idx: int,
    direction: str,
    open_arr: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr: float,
    min_body_atr: float = 0.8,
    min_body_ratio: float = 0.6,
) -> tuple[bool, float]:
    """Momentum trigger: strong directional bar confirms intent.

    For longs: bullish bar with body >= min_body_atr * ATR and
    body_ratio >= min_body_ratio (mostly body, not wicks).
    Entry at bar close.
    """
    bar_open = open_arr[bar_idx]
    bar_close = close[bar_idx]
    bar_high = high[bar_idx]
    bar_low = low[bar_idx]

    body = bar_close - bar_open
    body_abs = abs(body)
    bar_range = bar_high - bar_low

    if bar_range <= 0 or atr <= 0:
        return False, 0.0

    body_ratio = body_abs / bar_range

    if direction == "long":
        if body > 0 and body_abs >= min_body_atr * atr and body_ratio >= min_body_ratio:
            return True, bar_close
    else:
        if body < 0 and body_abs >= min_body_atr * atr and body_ratio >= min_body_ratio:
            return True, bar_close

    return False, 0.0


def check_wick_rejection(
    bar_idx: int,
    direction: str,
    open_arr: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr: float,
    min_wick_atr: float = 0.5,
    max_body_ratio: float = 0.35,
) -> tuple[bool, float]:
    """Wick rejection trigger: failed probe, snap back.

    For longs: bar has long lower wick (>= min_wick_atr * ATR), small body,
    closes in upper half — sellers tried but failed.
    For shorts: bar has long upper wick, small body, closes in lower half.
    Entry at bar close.
    """
    bar_open = open_arr[bar_idx]
    bar_close = close[bar_idx]
    bar_high = high[bar_idx]
    bar_low = low[bar_idx]

    bar_range = bar_high - bar_low
    if bar_range <= 0 or atr <= 0:
        return False, 0.0

    body_abs = abs(bar_close - bar_open)
    body_ratio = body_abs / bar_range
    upper_wick = bar_high - max(bar_open, bar_close)
    lower_wick = min(bar_open, bar_close) - bar_low

    if direction == "long":
        # Long lower wick = hammer = buyers stepped in
        if (lower_wick >= min_wick_atr * atr and
                body_ratio <= max_body_ratio and
                bar_close >= bar_low + bar_range * 0.5):
            return True, bar_close
    else:
        # Long upper wick = inverted hammer = sellers stepped in
        if (upper_wick >= min_wick_atr * atr and
                body_ratio <= max_body_ratio and
                bar_close <= bar_low + bar_range * 0.5):
            return True, bar_close

    return False, 0.0


def check_break_retest(
    bar_idx: int,
    direction: str,
    signal_high: float,
    signal_low: float,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    _broken: dict,
    sig_id: int,
) -> tuple[bool, float]:
    """Break-and-retest trigger: break signal bar extreme, retest, enter.

    For longs: first break above signal bar high (phase 1), then wait for
    bar that touches/dips back near that level but closes above (phase 2).
    Entry at signal bar high (the retest level).

    _broken tracks state across bars for this signal.
    """
    bar_high = high[bar_idx]
    bar_low = low[bar_idx]
    bar_close = close[bar_idx]

    key = f"break_{sig_id}"

    if direction == "long":
        if key not in _broken:
            # Phase 1: wait for break above signal high
            if bar_high > signal_high:
                _broken[key] = True
            return False, 0.0
        else:
            # Phase 2: wait for retest of signal high from above
            if bar_low <= signal_high + 1.0 and bar_close > signal_high:
                return True, signal_high
    else:
        if key not in _broken:
            # Phase 1: wait for break below signal low
            if bar_low < signal_low:
                _broken[key] = True
            return False, 0.0
        else:
            # Phase 2: wait for retest of signal low from below
            if bar_high >= signal_low - 1.0 and bar_close < signal_low:
                return True, signal_low

    return False, 0.0


def check_composite(
    bar_idx: int,
    direction: str,
    signal_close: float,
    open_arr: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    atr: float,
    vol_ma: float,
) -> tuple[bool, float]:
    """Composite trigger: multiple micro-conditions must align.

    Requires at least 2 of:
    1. Price in favorable zone (pullback from signal close)
    2. Volume above average (commitment)
    3. Bar closes in trade direction with decent body

    This is the most selective trigger — fewest entries, highest quality.
    """
    bar_open = open_arr[bar_idx]
    bar_close = close[bar_idx]
    bar_high = high[bar_idx]
    bar_low = low[bar_idx]
    bar_vol = volume[bar_idx]

    if atr <= 0:
        return False, 0.0

    score = 0

    # 1. Price in favorable zone
    if direction == "long":
        if bar_close <= signal_close + atr * 0.3:  # not too far above signal
            score += 1
    else:
        if bar_close >= signal_close - atr * 0.3:
            score += 1

    # 2. Volume confirmation
    if vol_ma > 0 and bar_vol >= vol_ma * 1.2:
        score += 1

    # 3. Directional close with body
    body = bar_close - bar_open
    bar_range = bar_high - bar_low
    body_ratio = abs(body) / bar_range if bar_range > 0 else 0

    if direction == "long" and body > 0 and body_ratio >= 0.4:
        score += 1
    elif direction == "short" and body < 0 and body_ratio >= 0.4:
        score += 1

    if score >= 2:
        return True, bar_close

    return False, 0.0


# Registry of trigger functions and their parameter sets for grid search
TRIGGER_CONFIGS = {
    "pullback_tight": {"type": "pullback", "pullback_pct": 0.3},
    "pullback_med": {"type": "pullback", "pullback_pct": 0.5},
    "pullback_wide": {"type": "pullback", "pullback_pct": 0.8},
    "momentum_strong": {"type": "momentum", "min_body_atr": 1.0, "min_body_ratio": 0.65},
    "momentum_med": {"type": "momentum", "min_body_atr": 0.8, "min_body_ratio": 0.55},
    "momentum_light": {"type": "momentum", "min_body_atr": 0.5, "min_body_ratio": 0.45},
    "wick_rejection": {"type": "wick_rejection", "min_wick_atr": 0.5, "max_body_ratio": 0.35},
    "wick_loose": {"type": "wick_rejection", "min_wick_atr": 0.3, "max_body_ratio": 0.45},
    "break_retest": {"type": "break_retest"},
    "composite": {"type": "composite"},
}
