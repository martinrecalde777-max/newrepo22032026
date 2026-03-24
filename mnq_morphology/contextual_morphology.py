"""Contextual group morphology — shape + prior context + volume regime.

This is the core analytical module. Instead of classifying candle groups
in isolation, it conditions on:
    1. SHAPE of the current N-bar window (slope, R², compression, close position)
    2. CONTEXT of the prior M-bar window (trend direction, range)
    3. VOLUME REGIME (shape volume vs context volume)

Real MNQ findings (1.77M bars):
    Without context: best edges 3–10 pts
    With context: best edges 20–49 pts
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _linreg(y: np.ndarray, x_c: np.ndarray, ss_x: float) -> tuple[float, float]:
    y_m = y.mean()
    slope = (x_c * (y - y_m)).sum() / ss_x
    ss_tot = ((y - y_m) ** 2).sum()
    if ss_tot > 0:
        fitted = slope * x_c + y_m
        r2 = 1.0 - ((y - fitted) ** 2).sum() / ss_tot
    else:
        r2 = 0.0
    return slope, r2


def compute_contextual_features(
    df: pd.DataFrame,
    shape_window: int = 60,
    context_window: int = 120,
    step: int = 1,
) -> pd.DataFrame:
    """Compute shape + context features for rolling windows.

    Parameters
    ----------
    df : OHLCV DataFrame
    shape_window : bars for current shape
    context_window : bars for prior context
    step : step between windows

    Returns
    -------
    DataFrame indexed at the last bar of each shape window.
    """
    c = df["close"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    v = df["volume"].values.astype(float)
    n = len(c)
    W = shape_window
    C = context_window
    total = W + C

    if n < total:
        return pd.DataFrame()

    x_s = np.arange(W, dtype=float)
    x_s_c = x_s - x_s.mean()
    ss_xs = (x_s_c ** 2).sum()

    x_ctx = np.arange(C, dtype=float)
    x_ctx_c = x_ctx - x_ctx.mean()
    ss_xctx = (x_ctx_c ** 2).sum()

    half = W // 2
    indices = np.arange(total - 1, n, step)
    results = np.empty((len(indices), 14), dtype=float)

    for idx_out, end in enumerate(indices):
        s_start = end - W + 1
        wc_s = c[s_start:end + 1]
        wh_s = h[s_start:end + 1]
        wl_s = l[s_start:end + 1]
        wv_s = v[s_start:end + 1]

        ctx_end = s_start - 1
        ctx_start = s_start - C
        wc_ctx = c[ctx_start:ctx_end + 1]
        wh_ctx = h[ctx_start:ctx_end + 1]
        wl_ctx = l[ctx_start:ctx_end + 1]
        wv_ctx = v[ctx_start:ctx_end + 1]

        # Shape features
        slope_s, r2_s = _linreg(wc_s - wc_s[0], x_s_c, ss_xs)
        r1 = wh_s[:half].max() - wl_s[:half].min()
        r2_range = wh_s[half:].max() - wl_s[half:].min()
        compression = r2_range / r1 if r1 > 0 else 1.0
        w_high = wh_s.max()
        w_low = wl_s.min()
        w_range = w_high - w_low
        close_pos = (wc_s[-1] - w_low) / w_range if w_range > 0 else 0.5
        front_vol = wv_s[:half].sum()
        back_vol = wv_s[half:].sum()
        vol_profile = back_vol / front_vol if front_vol > 0 else 1.0
        net_move_s = wc_s[-1] - wc_s[0]

        # Context features
        slope_ctx, r2_ctx = _linreg(wc_ctx - wc_ctx[0], x_ctx_c, ss_xctx)
        net_move_ctx = wc_ctx[-1] - wc_ctx[0]
        ctx_range = wh_ctx.max() - wl_ctx.min()
        ctx_vol_mean = wv_ctx.mean()
        shape_vol_mean = wv_s.mean()
        vol_ratio = shape_vol_mean / ctx_vol_mean if ctx_vol_mean > 0 else 1.0

        results[idx_out] = [
            slope_s, r2_s, compression, close_pos, vol_profile, net_move_s, w_range,
            slope_ctx, r2_ctx, net_move_ctx, ctx_range, vol_ratio,
            shape_vol_mean, ctx_vol_mean,
        ]

    out_index = df.index[indices]
    return pd.DataFrame(
        results,
        index=out_index,
        columns=[
            "slope_s", "r2_s", "compression", "close_pos", "vol_profile",
            "net_move_s", "range_s",
            "slope_ctx", "r2_ctx", "net_move_ctx", "range_ctx", "vol_ratio_ctx",
            "vol_mean_s", "vol_mean_ctx",
        ],
    )


def classify_context(features: pd.DataFrame) -> pd.DataFrame:
    """Add discrete labels for context trend, shape type, and volume regime."""
    out = features.copy()

    # Context trend
    out["ctx_trend"] = pd.cut(
        out["slope_ctx"],
        bins=[-np.inf, -1.5, -0.5, 0.5, 1.5, np.inf],
        labels=["ctx_strong_dn", "ctx_mild_dn", "ctx_flat", "ctx_mild_up", "ctx_strong_up"],
    )

    # Volume regime
    out["vol_regime"] = pd.cut(
        out["vol_ratio_ctx"],
        bins=[0, 0.5, 0.8, 1.2, 2.0, np.inf],
        labels=["vol_climax_die", "vol_low", "vol_normal", "vol_elevated", "vol_spike"],
    )

    # Shape type — granular classification
    # Key insight: "la forma sin contexto no vale nada"
    # We need to distinguish SILENT ramps (low vol, high edge) from noisy ones,
    # and detect squeeze traps (fake breakouts from compression).
    shape = pd.Series("other", index=out.index)
    slope = out["slope_s"]
    r2 = out["r2_s"]
    close_pos = out["close_pos"]
    net = out["net_move_s"]
    comp = out["compression"]
    vp = out["vol_profile"]

    # Base shapes
    shape[(slope > 1.5) & (r2 > 0.7)] = "strong_ramp_up"
    shape[(slope < -1.5) & (r2 > 0.7)] = "strong_ramp_dn"
    shape[(close_pos > 0.8) & (r2 < 0.35) & (net > 0)] = "v_bottom"
    shape[(close_pos < 0.2) & (r2 < 0.35) & (net < 0)] = "v_top"
    shape[comp < 0.35] = "tight_squeeze"
    shape[comp > 2.5] = "big_expansion"

    # Split ramps by volume profile: silent (declining vol) vs noisy
    # "Ramp up silencioso" = price trending up but volume DECLINING → vol_profile < 0.7
    # These have the highest edge (49pts EV in prior analysis)
    shape[(slope > 1.5) & (r2 > 0.7) & (vp < 0.7)] = "silent_ramp_up"
    shape[(slope < -1.5) & (r2 > 0.7) & (vp < 0.7)] = "silent_ramp_dn"

    # Exhaustion: strong slope but volume dying at the end
    shape[(slope > 1.5) & (vp < 0.5)] = "exhaust_up"
    shape[(slope < -1.5) & (vp < 0.5)] = "exhaust_dn"

    # Volume burst: strong move WITH increasing volume (momentum confirmation)
    shape[(slope > 1.5) & (vp > 2.0)] = "vol_burst_up"
    shape[(slope < -1.5) & (vp > 2.0)] = "vol_burst_dn"

    # Squeeze trap: tight compression that breaks out with volume
    # (fake breakout from range → reversal)
    shape[(comp < 0.35) & (vp > 1.5)] = "squeeze_breakout"

    out["shape_type"] = shape
    return out


def compute_forward_returns(
    df: pd.DataFrame,
    features: pd.DataFrame,
    forward_bars: tuple[int, ...],
) -> pd.DataFrame:
    """Compute forward returns at each feature index."""
    close = df["close"]
    fwd = pd.DataFrame(index=features.index)
    for fb in forward_bars:
        shifted = close.shift(-fb).reindex(features.index)
        fwd[f"fwd_{fb}"] = shifted.values - close.reindex(features.index).values
    return fwd


def scan_setups(
    features: pd.DataFrame,
    fwd: pd.DataFrame,
    min_n: int = 80,
    exclude_shapes: tuple[str, ...] = ("other",),
) -> pd.DataFrame:
    """Scan all (ctx_trend, shape_type, vol_regime) combos for edge.

    Parameters
    ----------
    features : classified contextual features
    fwd : forward returns
    min_n : minimum sample size per combo
    exclude_shapes : shape types to exclude (default: "other" has no edge)

    Returns DataFrame sorted by absolute mean forward return.
    Picks the BEST forward horizon per setup (highest |mean|).
    """
    fwd_cols = [c for c in fwd.columns if c.startswith("fwd_")]
    combined = pd.concat([features[["ctx_trend", "shape_type", "vol_regime"]], fwd], axis=1)

    rows = []
    for (ctx, shape, vol), group in combined.groupby(["ctx_trend", "shape_type", "vol_regime"]):
        # Skip garbage shapes — "other" has no defined morphology
        if shape in exclude_shapes:
            continue

        n = len(group)
        if n < min_n:
            continue

        row = {"ctx_trend": ctx, "shape_type": shape, "vol_regime": vol, "n": n}

        # Track best horizon for this setup
        best_abs_mean = 0
        best_horizon = None

        for fc in fwd_cols:
            vals = group[fc].dropna()
            if len(vals) < min_n:
                continue
            mean = vals.mean()
            std = vals.std()
            wins = vals[vals > 0]
            losses = vals[vals < 0]
            row[f"{fc}_mean"] = mean
            row[f"{fc}_win"] = (vals > 0).mean()
            row[f"{fc}_sharpe"] = mean / std if std > 0 else 0
            row[f"{fc}_pf"] = wins.sum() / losses.abs().sum() if losses.abs().sum() > 0 else np.inf
            row[f"{fc}_avg_winner"] = float(wins.mean()) if len(wins) > 0 else 0.0
            row[f"{fc}_avg_loser"] = float(losses.mean()) if len(losses) > 0 else 0.0

            if abs(mean) > best_abs_mean:
                best_abs_mean = abs(mean)
                best_horizon = fc

        # Store best horizon info
        if best_horizon is not None:
            horizon_bars = int(best_horizon.split("_")[1])
            row["best_horizon"] = best_horizon
            row["best_horizon_bars"] = horizon_bars
            row["best_mean"] = row[f"{best_horizon}_mean"]
            row["best_win"] = row.get(f"{best_horizon}_win", 0)
            row["best_pf"] = row.get(f"{best_horizon}_pf", 0)
            row["best_avg_winner"] = row.get(f"{best_horizon}_avg_winner", 0)
            row["best_avg_loser"] = row.get(f"{best_horizon}_avg_loser", 0)

        rows.append(row)

    result = pd.DataFrame(rows)
    if len(result) == 0:
        return result

    # Sort by best forward mean (absolute) across ALL horizons
    if "best_mean" in result.columns:
        result = result.sort_values("best_mean", key=abs, ascending=False)
    else:
        first_fwd = fwd_cols[0]
        mean_col = f"{first_fwd}_mean"
        if mean_col in result.columns:
            result = result.sort_values(mean_col, key=abs, ascending=False)

    return result
