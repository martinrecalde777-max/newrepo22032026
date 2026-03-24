"""Signal generator — converts contextual morphology setups into trade signals.

Takes the best (ctx_trend, shape_type, vol_regime) combos identified by
scan_setups and generates concrete long/short signals with expected
holding period and confidence level.

Each signal includes:
    - direction: 'long' or 'short'
    - entry_price: close at signal bar
    - stop_distance: ATR-based stop in points
    - target_distance: based on historical mean forward return
    - holding_bars: expected holding period
    - confidence: sample size / min_n ratio (capped at 1.0)
    - session: asia / london / ny / off_hours
    - mtf_confirm: multi-timeframe confirmation score
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_atr(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Average True Range in points."""
    h = df["high"]
    l = df["low"]
    c = df["close"].shift(1)
    tr = pd.concat([h - l, (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def build_setup_table(
    scan_results: pd.DataFrame,
    min_n: int = 80,
    min_abs_mean: float = 8.0,
    min_win_rate: float = 0.55,
    min_pf: float = 1.3,
    min_avg_winner: float = 0.0,
    max_p_value: float | None = None,
) -> pd.DataFrame:
    """Filter scan_setups output to keep only HIGH-QUALITY setups.

    The key lesson: trading many weak setups dilutes the edge.
    Only trade setups with strong statistical evidence.

    Parameters
    ----------
    scan_results : output of contextual_morphology.scan_setups()
    min_n : minimum sample size
    min_abs_mean : minimum absolute mean forward return (points)
        Previous value of 3.0 was too low — gets eaten by costs.
        8.0+ ensures real edge after slippage+commission.
    min_win_rate : minimum win rate (directional)
    min_pf : minimum profit factor at the statistical level
    min_avg_winner : minimum average winning trade (points, absolute).
        Filters at the statistical level before backtesting.
    max_p_value : if provided, filter by p-value column (optional)

    Returns
    -------
    DataFrame of qualifying setups with direction assigned.
    """
    if len(scan_results) == 0:
        return pd.DataFrame()

    # Use best_mean/best_win/best_pf if available (multi-horizon),
    # otherwise fall back to first fwd column
    if "best_mean" in scan_results.columns:
        fwd_mean = "best_mean"
        fwd_win = "best_win"
        fwd_pf = "best_pf"
        fwd_avg_winner = "best_avg_winner"
    else:
        mean_cols = [c for c in scan_results.columns if c.endswith("_mean")]
        if not mean_cols:
            return pd.DataFrame()
        fwd_mean = mean_cols[0]
        fwd_win = fwd_mean.replace("_mean", "_win")
        fwd_pf = fwd_mean.replace("_mean", "_pf")
        fwd_avg_winner = fwd_mean.replace("_mean", "_avg_winner")

    # Quality gate: sample size + absolute edge
    mask = (
        (scan_results["n"] >= min_n)
        & (scan_results[fwd_mean].abs() >= min_abs_mean)
    )

    # Win rate filter (direction-aware)
    if fwd_win in scan_results.columns:
        long_mask = (scan_results[fwd_mean] > 0) & (scan_results[fwd_win] >= min_win_rate)
        short_mask = (scan_results[fwd_mean] < 0) & ((1 - scan_results[fwd_win]) >= min_win_rate)
        mask = mask & (long_mask | short_mask)

    # Profit factor filter — require PF >= min_pf
    if fwd_pf in scan_results.columns:
        mask = mask & (scan_results[fwd_pf] >= min_pf)

    # Average winner filter — reject setups where winners are too small
    if min_avg_winner > 0 and fwd_avg_winner in scan_results.columns:
        mask = mask & (scan_results[fwd_avg_winner].abs() >= min_avg_winner)

    out = scan_results[mask].copy()
    if len(out) == 0:
        return pd.DataFrame()

    out["direction"] = np.where(out[fwd_mean] > 0, "long", "short")
    out["expected_move"] = out[fwd_mean].abs()

    # Avg winner from scan data (statistical, pre-backtest)
    if fwd_avg_winner in out.columns:
        out["stat_avg_winner"] = out[fwd_avg_winner].abs()

    # Per-setup optimal holding period (from best horizon analysis)
    if "best_horizon_bars" in out.columns:
        out["optimal_holding"] = out["best_horizon_bars"]
    else:
        out["optimal_holding"] = 60  # default

    # Confidence: how much data backs this setup (capped at 1.0)
    out["confidence"] = (out["n"] / (min_n * 5)).clip(upper=1.0)

    return out.sort_values("expected_move", ascending=False).reset_index(drop=True)


def generate_signals(
    df: pd.DataFrame,
    features: pd.DataFrame,
    setup_table: pd.DataFrame,
    atr_period: int = 20,
    stop_atr_mult: float = 1.5,
    target_atr_mult: float | None = None,
    holding_bars: int = 60,
    stop_ev_ratio: float | None = None,
    target_ev_ratio: float | None = None,
    stop_pts: float | None = None,
    target_pts: float | None = None,
) -> pd.DataFrame:
    """Generate trade signals on the bar-level DataFrame.

    Parameters
    ----------
    df : OHLCV DataFrame (1-min bars)
    features : classified contextual features (output of classify_context)
    setup_table : filtered setups from build_setup_table()
    atr_period : period for ATR calculation
    stop_atr_mult : stop distance = ATR * this multiplier (used when stop_ev_ratio is None)
    target_atr_mult : target distance = ATR * this. If None, uses setup's expected_move.
    holding_bars : max bars to hold if neither stop nor target hit
    stop_ev_ratio : if set, stop = expected_move * this ratio (e.g., 0.4 = risk 40% of EV)
        Overrides stop_atr_mult. This scales the stop proportionally to the
        setup's edge, preventing tight ATR stops from killing high-EV setups.
    target_ev_ratio : if set, target = expected_move * this ratio (e.g., 0.8)
        Overrides target_atr_mult.
    stop_pts : fixed stop distance in points. Overrides all other stop methods.
    target_pts : fixed target distance in points. Overrides all other target methods.

    Returns
    -------
    DataFrame of signals with columns:
        timestamp, direction, entry_price, stop_price, target_price,
        holding_bars, setup_key, confidence
    """
    if len(setup_table) == 0:
        return pd.DataFrame()

    atr = compute_atr(df, atr_period)

    # Build lookup: (ctx_trend, shape_type, vol_regime) -> setup info
    setup_lookup = {}
    for _, row in setup_table.iterrows():
        key = (row["ctx_trend"], row["shape_type"], row["vol_regime"])
        if key not in setup_lookup:
            setup_lookup[key] = row

    signals = []
    required_cols = {"ctx_trend", "shape_type", "vol_regime"}
    if not required_cols.issubset(features.columns):
        return pd.DataFrame()

    for ts in features.index:
        ctx = features.loc[ts, "ctx_trend"]
        shape = features.loc[ts, "shape_type"]
        vol = features.loc[ts, "vol_regime"]
        key = (ctx, shape, vol)

        if key not in setup_lookup:
            continue

        setup = setup_lookup[key]
        if ts not in atr.index or pd.isna(atr.loc[ts]):
            continue

        current_atr = atr.loc[ts]
        if current_atr <= 0:
            continue

        entry = df.loc[ts, "close"]
        ev = setup["expected_move"]

        # Stop distance: fixed pts > EV-proportional > ATR-based
        if stop_pts is not None:
            stop_dist = stop_pts
        elif stop_ev_ratio is not None:
            stop_dist = ev * stop_ev_ratio
        else:
            stop_dist = current_atr * stop_atr_mult

        # Target distance: fixed pts > EV-proportional > ATR-based > expected_move
        if target_pts is not None:
            target_dist = target_pts
        elif target_ev_ratio is not None:
            target_dist = ev * target_ev_ratio
        elif target_atr_mult is not None:
            target_dist = current_atr * target_atr_mult
        else:
            target_dist = ev

        direction = setup["direction"]
        if direction == "long":
            stop_price = entry - stop_dist
            target_price = entry + target_dist
        else:
            stop_price = entry + stop_dist
            target_price = entry - target_dist

        # Use per-setup optimal holding period if available, else global
        setup_holding = int(setup.get("optimal_holding", holding_bars))
        if holding_bars > 0:
            setup_holding = min(setup_holding, holding_bars) if holding_bars != 60 else setup_holding

        signals.append({
            "timestamp": ts,
            "direction": direction,
            "entry_price": entry,
            "stop_price": stop_price,
            "target_price": target_price,
            "stop_distance": stop_dist,
            "target_distance": target_dist,
            "holding_bars": setup_holding,
            "setup_key": f"{ctx}|{shape}|{vol}",
            "confidence": setup["confidence"],
            "atr": current_atr,
        })

    result = pd.DataFrame(signals)
    if len(result) > 0:
        result = result.set_index("timestamp").sort_index()
    return result


def classify_session(index: pd.DatetimeIndex) -> pd.Series:
    """Classify each timestamp into a trading session.

    CME MNQ sessions (US/Eastern):
        Asia:   18:00–02:00 (prior day evening → early morning)
        London: 02:00–09:30
        NY:     09:30–16:00
        Off:    16:00–18:00 (daily maintenance halt)
    """
    hour = index.hour
    minute = index.minute
    session = pd.Series("off_hours", index=index)
    session[(hour >= 18) | (hour < 2)] = "asia"
    session[(hour >= 2) & (hour < 9) | ((hour == 9) & (minute < 30))] = "london"
    session[((hour == 9) & (minute >= 30)) | ((hour >= 10) & (hour < 16))] = "ny"
    return session


def add_session_filter(
    signals: pd.DataFrame,
    allowed_sessions: tuple[str, ...] = ("asia", "london", "ny"),
) -> pd.DataFrame:
    """Filter signals to only allowed sessions."""
    if len(signals) == 0:
        return signals
    session = classify_session(signals.index)
    signals = signals.copy()
    signals["session"] = session
    return signals[signals["session"].isin(allowed_sessions)]


def precompute_mtf_slopes(
    df_1min: pd.DataFrame,
    higher_timeframes: tuple[str, ...] = ("5min", "1h"),
) -> dict[str, pd.Series]:
    """Pre-compute higher-TF slopes once, forward-filled to 1-min index.

    Call this once and pass the result to add_mtf_confirmation() to avoid
    recomputing aggregations on every call.
    """
    from mnq_morphology.aggregator import aggregate

    tf_slopes = {}
    for tf in higher_timeframes:
        agg = aggregate(df_1min, tf)
        slope = agg["close"].diff(20) / 20
        slope_1min = slope.reindex(df_1min.index, method="ffill")
        tf_slopes[tf] = slope_1min
    return tf_slopes


def add_mtf_confirmation(
    signals: pd.DataFrame,
    df_1min: pd.DataFrame,
    higher_timeframes: tuple[str, ...] = ("5min", "1h"),
    precomputed_slopes: dict[str, pd.Series] | None = None,
) -> pd.DataFrame:
    """Add multi-timeframe trend confirmation score to signals.

    For each signal, checks if higher-TF trend agrees with direction:
        - Computes slope of close over last N bars at each higher TF
        - Score +1 if slope agrees with direction, -1 if disagrees, 0 if flat
        - mtf_score = sum across TFs (range: -len(TFs) to +len(TFs))
        - mtf_confirm = True if mtf_score > 0

    Pass precomputed_slopes (from precompute_mtf_slopes()) to avoid
    recomputing aggregations on every call.
    """
    if len(signals) == 0:
        return signals

    signals = signals.copy()

    tf_slopes = precomputed_slopes or precompute_mtf_slopes(df_1min, higher_timeframes)

    scores = []
    for ts in signals.index:
        score = 0
        direction = signals.loc[ts, "direction"]
        for tf, slope_s in tf_slopes.items():
            if ts in slope_s.index and not pd.isna(slope_s.loc[ts]):
                sl = slope_s.loc[ts]
                if direction == "long" and sl > 0.5:
                    score += 1
                elif direction == "long" and sl < -0.5:
                    score -= 1
                elif direction == "short" and sl < -0.5:
                    score += 1
                elif direction == "short" and sl > 0.5:
                    score -= 1
        scores.append(score)

    signals["mtf_score"] = scores
    signals["mtf_confirm"] = [s > 0 for s in scores]
    return signals


def filter_mtf_confirmed(signals: pd.DataFrame, min_score: int = 1) -> pd.DataFrame:
    """Keep only signals with multi-timeframe confirmation >= min_score."""
    if "mtf_score" not in signals.columns:
        return signals
    return signals[signals["mtf_score"] >= min_score]


def filter_no_overlap(signals: pd.DataFrame, min_gap_bars: int = 60) -> pd.DataFrame:
    """Remove overlapping signals — keep only one per min_gap_bars window.

    When multiple signals fire within the holding period, we keep the one
    with highest confidence.
    """
    if len(signals) <= 1:
        return signals

    signals = signals.sort_index()
    kept = [0]  # always keep first signal

    for i in range(1, len(signals)):
        prev_ts = signals.index[kept[-1]]
        curr_ts = signals.index[i]

        # Compute bar gap (works for DatetimeIndex with freq)
        gap = (curr_ts - prev_ts).total_seconds() / 60  # minutes = bars for 1-min data

        if gap >= min_gap_bars:
            kept.append(i)
        else:
            # If new signal has higher confidence, replace
            if signals.iloc[i]["confidence"] > signals.iloc[kept[-1]]["confidence"]:
                kept[-1] = i

    return signals.iloc[kept]
