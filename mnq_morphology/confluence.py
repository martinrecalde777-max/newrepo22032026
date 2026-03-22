"""Multi-timeframe pattern confluence detector for MNQ.

Detects when candlestick patterns align across multiple timeframes
simultaneously, which may indicate stronger signals than single-TF patterns.

Example: morning_star on 1H + engulfing_bullish on 5m at the same timestamp.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from mnq_morphology.morphology import compute_morphology
from mnq_morphology.patterns import detect_patterns
from mnq_morphology.aggregator import aggregate


def _build_tf_patterns(
    base_1min: pd.DataFrame,
    timeframes: tuple[str, ...] = ("5min", "15min", "1h"),
) -> dict[str, pd.DataFrame]:
    """Compute morphology + patterns for each timeframe, forward-fill to 1-min index."""
    result = {}

    for tf in timeframes:
        agg = aggregate(base_1min, tf)
        morph = compute_morphology(agg)
        pats = detect_patterns(morph)
        # Forward-fill higher-TF patterns to 1-min index
        # A 1H pattern is "active" for all 1-min bars within that hour
        pats_reindexed = pats.reindex(base_1min.index, method="ffill")
        # Rename columns to include timeframe
        pats_reindexed.columns = [f"{c}_{tf}" for c in pats_reindexed.columns]
        result[tf] = pats_reindexed

    return result


def detect_confluence(
    base_1min: pd.DataFrame,
    higher_timeframes: tuple[str, ...] = ("5min", "15min", "1h"),
) -> pd.DataFrame:
    """Detect multi-timeframe pattern confluence events.

    Parameters
    ----------
    base_1min : OHLCV DataFrame at 1-min resolution (with or without morphology)
    higher_timeframes : timeframes to check for alignment

    Returns
    -------
    DataFrame indexed like base_1min with confluence columns:
        - conf_bullish_count: # of bullish patterns active across all TFs
        - conf_bearish_count: # of bearish patterns active across all TFs
        - conf_bullish_tfs: which TFs have active bullish patterns
        - conf_bearish_tfs: which TFs have active bearish patterns
        - One column per (higher_tf, pattern) combination
    """
    # Patterns on the base 1-min data
    morph_1min = compute_morphology(base_1min)
    pats_1min = detect_patterns(morph_1min)
    pats_1min.columns = [f"{c}_1min" for c in pats_1min.columns]

    # Higher timeframe patterns forward-filled to 1-min
    tf_pats = _build_tf_patterns(base_1min, higher_timeframes)

    # Combine all pattern columns
    all_pats = pd.concat([pats_1min] + list(tf_pats.values()), axis=1)

    # Classify patterns as bullish or bearish
    BULLISH = {"hammer", "inverted_hammer", "big_body_bullish", "engulfing_bullish",
               "tweezer_bottom", "morning_star", "three_white_soldiers"}
    BEARISH = {"big_body_bearish", "engulfing_bearish", "tweezer_top",
               "evening_star", "three_black_crows"}

    all_tfs = ("1min",) + higher_timeframes
    out = pd.DataFrame(index=base_1min.index)

    # Count bullish/bearish confluences per bar
    bull_counts = pd.Series(0, index=base_1min.index, dtype="int8")
    bear_counts = pd.Series(0, index=base_1min.index, dtype="int8")
    bull_tfs_list = [[] for _ in range(len(base_1min))]
    bear_tfs_list = [[] for _ in range(len(base_1min))]

    for tf in all_tfs:
        tf_bull = pd.Series(False, index=base_1min.index)
        tf_bear = pd.Series(False, index=base_1min.index)

        for pat_name in BULLISH:
            col = f"pat_{pat_name}_{tf}"
            if col in all_pats.columns:
                tf_bull |= all_pats[col].fillna(False).astype(bool)

        for pat_name in BEARISH:
            col = f"pat_{pat_name}_{tf}"
            if col in all_pats.columns:
                tf_bear |= all_pats[col].fillna(False).astype(bool)

        bull_counts += tf_bull.astype("int8")
        bear_counts += tf_bear.astype("int8")

    out["conf_bullish_count"] = bull_counts
    out["conf_bearish_count"] = bear_counts
    out["conf_net"] = bull_counts - bear_counts  # positive = bullish bias

    # Strong confluence = 3+ TFs agree
    n_tfs = len(all_tfs)
    out["conf_strong_bullish"] = bull_counts >= min(3, n_tfs)
    out["conf_strong_bearish"] = bear_counts >= min(3, n_tfs)

    # Include the full pattern matrix for detailed analysis
    out = pd.concat([out, all_pats], axis=1)

    return out


def confluence_stats(
    conf: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 5, 20),
) -> pd.DataFrame:
    """Measure forward returns conditioned on confluence level.

    Parameters
    ----------
    conf : output of detect_confluence()
    fwd_returns : DataFrame with fwd_1, fwd_5, fwd_20 columns

    Returns
    -------
    Stats per confluence level showing edge amplification.
    """
    rows = []

    for direction, count_col in [("bullish", "conf_bullish_count"), ("bearish", "conf_bearish_count")]:
        for level in range(5):  # 0, 1, 2, 3, 4
            mask = conf[count_col] == level
            n = mask.sum()
            if n < 30:
                continue

            row = {"direction": direction, "confluence_level": level, "n": n}

            for h in horizons:
                fc = f"fwd_{h}"
                if fc not in fwd_returns.columns:
                    continue
                vals = fwd_returns.loc[mask, fc].dropna()
                if len(vals) < 30:
                    continue

                # For bearish confluence, we expect negative returns (short edge)
                # so we measure "edge in expected direction"
                if direction == "bearish":
                    edge_vals = -vals  # flip sign for bearish
                else:
                    edge_vals = vals

                row[f"fwd{h}_mean"] = vals.mean()
                row[f"fwd{h}_win_rate"] = (edge_vals > 0).mean()
                row[f"fwd{h}_std"] = vals.std()

            rows.append(row)

    return pd.DataFrame(rows)


def find_confluence_events(
    conf: pd.DataFrame,
    min_bullish: int = 3,
    min_bearish: int = 3,
) -> pd.DataFrame:
    """Extract specific high-confluence events for review.

    Returns a compact DataFrame of timestamps where strong confluence occurred.
    """
    bull_mask = conf["conf_bullish_count"] >= min_bullish
    bear_mask = conf["conf_bearish_count"] >= min_bearish

    events = []

    if bull_mask.any():
        bull_events = conf.loc[bull_mask, ["conf_bullish_count", "conf_bearish_count", "conf_net"]].copy()
        bull_events["signal"] = "BULLISH"
        events.append(bull_events)

    if bear_mask.any():
        bear_events = conf.loc[bear_mask, ["conf_bullish_count", "conf_bearish_count", "conf_net"]].copy()
        bear_events["signal"] = "BEARISH"
        events.append(bear_events)

    if not events:
        return pd.DataFrame()

    result = pd.concat(events).sort_index()
    # Remove duplicates where both bullish and bearish fire (conflict)
    result = result[~result.index.duplicated(keep="first")]
    return result
