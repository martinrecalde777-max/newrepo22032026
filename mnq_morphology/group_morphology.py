"""Group morphology — shape analysis of N-bar windows of MNQ 1-min data.

Instead of studying individual candles, this module classifies the
SHAPE of groups of consecutive 1-min bars (e.g., 15, 30, 60 bars)
and measures what typically follows each shape.

Calibrated to real MNQ data (1.77M bars, 2021–2026):
    15-bar windows:
        slope:       mean≈0, std=1.69 pts/bar, p5=-2.3, p95=+2.2
        R²:          mean=0.42 (moderate trending), p5=0.006, p95=0.877
        compression: mean=1.18, squeeze < 0.5, expansion > 2.0
        choppiness:  mean=7.1 direction changes (of 14 possible)
        close_pos:   mean=0.52 (centered), 0=at-low, 1=at-high

Shape types:
    CLEAN_RAMP_UP:    strong uptrend, high R², positive slope
    CLEAN_RAMP_DOWN:  strong downtrend, high R², negative slope
    V_BOTTOM:         low in first half, reversal up, close near high
    V_TOP:            high in first half, reversal down, close near low
    SQUEEZE:          range compressing (2nd half much tighter than 1st)
    EXPANSION:        range expanding (breakout / volatility burst)
    CHOP:             high direction changes, low R², no clear trend
    MOMENTUM_BURST:   big slope + big range + back-loaded volume
    EXHAUSTION:       big slope but R² fading + volume dying
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd


class Shape(str, Enum):
    CLEAN_RAMP_UP = "clean_ramp_up"
    CLEAN_RAMP_DOWN = "clean_ramp_down"
    V_BOTTOM = "v_bottom"
    V_TOP = "v_top"
    SQUEEZE = "squeeze"
    EXPANSION = "expansion"
    CHOP = "chop"
    MOMENTUM_BURST_UP = "momentum_burst_up"
    MOMENTUM_BURST_DOWN = "momentum_burst_down"
    EXHAUSTION_UP = "exhaustion_up"
    EXHAUSTION_DOWN = "exhaustion_down"
    NEUTRAL = "neutral"


# Thresholds calibrated to MNQ 15-bar window distributions
# Adjusted for window_size != 15 via scaling
_BASE_WINDOW = 15

# R² thresholds
R2_TREND = 0.65       # p75 — good trend quality
R2_STRONG = 0.80      # p90 — very clean trend
R2_CHOPPY = 0.15      # p25 — no trend

# Slope thresholds (pts/bar for 15-bar window)
SLOPE_STRONG = 1.5    # roughly p80
SLOPE_WEAK = 0.3      # roughly p35

# Compression thresholds
COMPRESS_SQUEEZE = 0.50   # strong squeeze
COMPRESS_EXPAND = 2.00    # strong expansion

# Choppiness (direction changes / max possible)
CHOP_HIGH = 0.60      # >60% of possible direction changes

# Volume profile
VOL_BACKLOADED = 1.5   # back half has 50% more volume
VOL_DYING = 0.6        # back half has much less volume


def compute_window_features(
    df: pd.DataFrame,
    window: int = 15,
    step: int = 1,
) -> pd.DataFrame:
    """Compute shape features for rolling windows of N bars.

    Parameters
    ----------
    df : OHLCV DataFrame (1-min continuous)
    window : number of bars per group
    step : step size (1 = fully overlapping, window = non-overlapping)

    Returns
    -------
    DataFrame indexed at the LAST bar of each window, with shape features.
    """
    c = df["close"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    v = df["volume"].values.astype(float)
    n = len(c)
    W = window

    if n < W:
        return pd.DataFrame()

    # Pre-compute linear regression constants
    x = np.arange(W, dtype=float)
    x_c = x - x.mean()
    ss_x = (x_c ** 2).sum()
    half = W // 2

    indices = np.arange(W - 1, n, step)
    results = np.empty((len(indices), 12), dtype=float)

    for idx_out, end in enumerate(indices):
        start = end - W + 1
        wc = c[start:end + 1]
        wh = h[start:end + 1]
        wl = l[start:end + 1]
        wv = v[start:end + 1]

        # 1. Slope & R² (linear regression on close)
        y = wc - wc[0]
        y_mean = y.mean()
        slope = (x_c * (y - y_mean)).sum() / ss_x
        ss_tot = ((y - y_mean) ** 2).sum()
        if ss_tot > 0:
            fitted = slope * x_c + y_mean
            ss_res = ((y - fitted) ** 2).sum()
            r2 = 1.0 - ss_res / ss_tot
        else:
            r2 = 0.0

        # 2. Compression (2nd-half range / 1st-half range)
        r1 = wh[:half].max() - wl[:half].min()
        r2_range = wh[half:].max() - wl[half:].min()
        compression = r2_range / r1 if r1 > 0 else 1.0

        # 3. Choppiness (direction changes / max possible)
        dirs = np.sign(np.diff(wc))
        nonzero_dirs = dirs[dirs != 0]
        if len(nonzero_dirs) > 1:
            changes = (nonzero_dirs[1:] != nonzero_dirs[:-1]).sum()
            chop_ratio = changes / (len(nonzero_dirs) - 1)
        else:
            chop_ratio = 0.0

        # 4. Close position (0=at window low, 1=at window high)
        w_high = wh.max()
        w_low = wl.min()
        w_range = w_high - w_low
        close_pos = (wc[-1] - w_low) / w_range if w_range > 0 else 0.5

        # 5. Volume profile (back / front)
        front_vol = wv[:half].sum()
        back_vol = wv[half:].sum()
        vol_profile = back_vol / front_vol if front_vol > 0 else 1.0

        # 6. High/Low positions within window (0=start, 1=end)
        high_pos = np.argmax(wh) / (W - 1)
        low_pos = np.argmin(wl) / (W - 1)

        # 7. Total range in points
        total_range = w_range

        # 8. Net move (close[-1] - close[0])
        net_move = wc[-1] - wc[0]

        # 9. Max drawdown from running high
        running_high = np.maximum.accumulate(wc)
        drawdowns = running_high - wc
        max_dd = drawdowns.max()

        # 10. Max runup from running low
        running_low = np.minimum.accumulate(wc)
        runups = wc - running_low
        max_runup = runups.max()

        results[idx_out] = [
            slope, r2, compression, chop_ratio, close_pos,
            vol_profile, high_pos, low_pos, total_range, net_move,
            max_dd, max_runup,
        ]

    out_index = df.index[indices]
    out = pd.DataFrame(
        results,
        index=out_index,
        columns=[
            "gm_slope", "gm_r2", "gm_compression", "gm_chop_ratio",
            "gm_close_pos", "gm_vol_profile", "gm_high_pos", "gm_low_pos",
            "gm_range", "gm_net_move", "gm_max_dd", "gm_max_runup",
        ],
    )
    out.index.name = df.index.name

    return out


def classify_shape(features: pd.DataFrame) -> pd.Series:
    """Classify each window into a Shape category based on its features."""
    n = len(features)
    shapes = pd.Series(Shape.NEUTRAL, index=features.index, dtype=object)

    slope = features["gm_slope"]
    r2 = features["gm_r2"]
    comp = features["gm_compression"]
    chop = features["gm_chop_ratio"]
    close_pos = features["gm_close_pos"]
    vol_prof = features["gm_vol_profile"]
    high_pos = features["gm_high_pos"]
    low_pos = features["gm_low_pos"]

    # --- CLEAN RAMPS (strong trend, high R²) ---
    clean_up = (slope > SLOPE_STRONG) & (r2 > R2_TREND)
    clean_down = (slope < -SLOPE_STRONG) & (r2 > R2_TREND)

    # --- V-BOTTOM: low in first half, close near high ---
    v_bottom = (
        (low_pos < 0.5)
        & (close_pos > 0.70)
        & (r2 < R2_TREND)  # not a clean ramp
        & (slope > SLOPE_WEAK)
    )

    # --- V-TOP: high in first half, close near low ---
    v_top = (
        (high_pos < 0.5)
        & (close_pos < 0.30)
        & (r2 < R2_TREND)
        & (slope < -SLOPE_WEAK)
    )

    # --- SQUEEZE: 2nd half much tighter than 1st ---
    squeeze = comp < COMPRESS_SQUEEZE

    # --- EXPANSION: 2nd half much wider than 1st ---
    expansion = comp > COMPRESS_EXPAND

    # --- CHOP: lots of direction changes, low R² ---
    chop_mask = (chop > CHOP_HIGH) & (r2 < R2_CHOPPY)

    # --- MOMENTUM BURST: steep slope + expansion + back-loaded volume ---
    momentum_up = (
        (slope > SLOPE_STRONG)
        & (r2 > R2_TREND)
        & (vol_prof > VOL_BACKLOADED)
    )
    momentum_down = (
        (slope < -SLOPE_STRONG)
        & (r2 > R2_TREND)
        & (vol_prof > VOL_BACKLOADED)
    )

    # --- EXHAUSTION: strong slope but volume dying ---
    exhaust_up = (
        (slope > SLOPE_STRONG)
        & (vol_prof < VOL_DYING)
    )
    exhaust_down = (
        (slope < -SLOPE_STRONG)
        & (vol_prof < VOL_DYING)
    )

    # Apply in priority order (more specific first)
    shapes[chop_mask] = Shape.CHOP
    shapes[squeeze] = Shape.SQUEEZE
    shapes[expansion] = Shape.EXPANSION
    shapes[v_bottom] = Shape.V_BOTTOM
    shapes[v_top] = Shape.V_TOP
    shapes[clean_up] = Shape.CLEAN_RAMP_UP
    shapes[clean_down] = Shape.CLEAN_RAMP_DOWN
    # Override ramps with momentum/exhaustion variants
    shapes[momentum_up] = Shape.MOMENTUM_BURST_UP
    shapes[momentum_down] = Shape.MOMENTUM_BURST_DOWN
    shapes[exhaust_up] = Shape.EXHAUSTION_UP
    shapes[exhaust_down] = Shape.EXHAUSTION_DOWN

    return shapes


def group_forward_returns(
    df: pd.DataFrame,
    features: pd.DataFrame,
    shapes: pd.Series,
    forward_windows: tuple[int, ...] = (15, 30, 60),
) -> pd.DataFrame:
    """Measure what happens AFTER each shape type.

    Forward returns are measured in MNQ points over the next N bars.
    """
    close = df["close"].reindex(features.index)

    # Compute forward returns
    fwd = pd.DataFrame(index=features.index)
    for fw in forward_windows:
        fwd[f"fwd_{fw}"] = df["close"].reindex(features.index).values
        shifted = df["close"].shift(-fw).reindex(features.index).values
        fwd[f"fwd_{fw}"] = shifted - close.values

    fwd["shape"] = shapes

    rows = []
    for shape in Shape:
        mask = fwd["shape"] == shape
        n = mask.sum()
        if n < 50:
            continue

        row = {"shape": shape.value, "n": n, "pct": n / len(fwd) * 100}

        for fw in forward_windows:
            col = f"fwd_{fw}"
            vals = fwd.loc[mask, col].dropna()
            if len(vals) < 50:
                continue

            row[f"fwd{fw}_mean"] = vals.mean()
            row[f"fwd{fw}_median"] = vals.median()
            row[f"fwd{fw}_std"] = vals.std()
            row[f"fwd{fw}_win_rate"] = (vals > 0).mean()
            row[f"fwd{fw}_sharpe"] = vals.mean() / vals.std() if vals.std() > 0 else 0

            # Profit factor
            wins = vals[vals > 0].sum()
            losses = vals[vals < 0].abs().sum()
            row[f"fwd{fw}_pf"] = wins / losses if losses > 0 else np.inf

        rows.append(row)

    return pd.DataFrame(rows).set_index("shape")


def run_group_analysis(
    df: pd.DataFrame,
    windows: tuple[int, ...] = (15, 30, 60),
    step: int = 1,
    forward_windows: tuple[int, ...] = (15, 30, 60),
) -> dict[int, pd.DataFrame]:
    """Run full group morphology analysis for multiple window sizes.

    Returns dict of {window_size: stats_df}.
    """
    results = {}
    for w in windows:
        features = compute_window_features(df, window=w, step=step)
        shapes = classify_shape(features)
        stats = group_forward_returns(df, features, shapes, forward_windows)
        results[w] = stats
    return results
