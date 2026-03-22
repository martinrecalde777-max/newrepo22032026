"""Statistical analysis of pattern predictive power on real MNQ data.

Measures forward returns after each pattern occurrence to determine
whether patterns have genuine predictive edge.

Metrics computed per pattern:
    - Forward returns at 1, 5, 20 bars (mean, median, std)
    - Win rate (% of positive forward returns)
    - Expectancy (average P&L per occurrence in points)
    - Edge vs baseline (pattern return − unconditional return)
    - t-statistic and p-value for statistical significance
    - Profit factor (gross wins / gross losses)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


FORWARD_HORIZONS = (1, 5, 20)


def compute_forward_returns(
    df: pd.DataFrame,
    horizons: tuple[int, ...] = FORWARD_HORIZONS,
) -> pd.DataFrame:
    """Add forward return columns to morphology DataFrame.

    Returns are in points (not percent) since MNQ is a futures contract.
    """
    out = df.copy()
    for h in horizons:
        out[f"fwd_{h}"] = out["close"].shift(-h) - out["close"]
    return out


def pattern_stats(
    df: pd.DataFrame,
    horizons: tuple[int, ...] = FORWARD_HORIZONS,
) -> pd.DataFrame:
    """Compute full statistical profile for every pattern column.

    Parameters
    ----------
    df : DataFrame with pat_* columns and fwd_* columns

    Returns
    -------
    DataFrame indexed by pattern name, with stats for each horizon.
    """
    pat_cols = [c for c in df.columns if c.startswith("pat_")]
    fwd_cols = [f"fwd_{h}" for h in horizons]

    # Baseline (unconditional) stats
    baseline = {}
    for fc in fwd_cols:
        vals = df[fc].dropna()
        baseline[fc] = {
            "mean": vals.mean(),
            "median": vals.median(),
            "std": vals.std(),
            "win_rate": (vals > 0).mean(),
        }

    rows = []
    for pc in pat_cols:
        name = pc.replace("pat_", "")
        mask = df[pc].astype(bool)
        n_occurrences = mask.sum()

        row = {"pattern": name, "occurrences": n_occurrences}

        for h, fc in zip(horizons, fwd_cols):
            vals = df.loc[mask, fc].dropna()
            n = len(vals)

            if n < 30:
                # Not enough data for reliable stats
                row.update({
                    f"fwd{h}_mean": np.nan, f"fwd{h}_median": np.nan,
                    f"fwd{h}_std": np.nan, f"fwd{h}_win_rate": np.nan,
                    f"fwd{h}_expectancy": np.nan, f"fwd{h}_edge": np.nan,
                    f"fwd{h}_t_stat": np.nan, f"fwd{h}_p_value": np.nan,
                    f"fwd{h}_profit_factor": np.nan, f"fwd{h}_n": n,
                })
                continue

            mean = vals.mean()
            wins = vals[vals > 0]
            losses = vals[vals < 0]

            # t-test: is pattern mean significantly different from baseline?
            t_stat, p_val = scipy_stats.ttest_1samp(vals, baseline[fc]["mean"])

            # Profit factor
            gross_win = wins.sum() if len(wins) > 0 else 0
            gross_loss = losses.abs().sum() if len(losses) > 0 else 0
            pf = gross_win / gross_loss if gross_loss > 0 else np.inf

            row.update({
                f"fwd{h}_mean": mean,
                f"fwd{h}_median": vals.median(),
                f"fwd{h}_std": vals.std(),
                f"fwd{h}_win_rate": (vals > 0).mean(),
                f"fwd{h}_expectancy": mean,  # in points per occurrence
                f"fwd{h}_edge": mean - baseline[fc]["mean"],
                f"fwd{h}_t_stat": t_stat,
                f"fwd{h}_p_value": p_val,
                f"fwd{h}_profit_factor": pf,
                f"fwd{h}_n": n,
            })

        rows.append(row)

    result = pd.DataFrame(rows).set_index("pattern")

    # Add baseline row
    bl_row = {"occurrences": len(df)}
    for h, fc in zip(horizons, fwd_cols):
        bl_row.update({
            f"fwd{h}_mean": baseline[fc]["mean"],
            f"fwd{h}_median": baseline[fc]["median"],
            f"fwd{h}_std": baseline[fc]["std"],
            f"fwd{h}_win_rate": baseline[fc]["win_rate"],
            f"fwd{h}_expectancy": baseline[fc]["mean"],
            f"fwd{h}_edge": 0.0,
            f"fwd{h}_t_stat": 0.0,
            f"fwd{h}_p_value": 1.0,
            f"fwd{h}_profit_factor": np.nan,
            f"fwd{h}_n": len(df[fc].dropna()),
        })
    bl_df = pd.DataFrame([bl_row], index=pd.Index(["BASELINE"], name="pattern"))
    result = pd.concat([bl_df, result])

    return result


def summarize_stats(stats_df: pd.DataFrame, horizon: int = 1) -> str:
    """Produce a human-readable summary for a given horizon."""
    h = horizon
    cols = [
        "occurrences",
        f"fwd{h}_mean", f"fwd{h}_win_rate", f"fwd{h}_edge",
        f"fwd{h}_t_stat", f"fwd{h}_p_value", f"fwd{h}_profit_factor",
    ]
    sub = stats_df[cols].copy()
    sub.columns = ["N", "mean_pts", "win_rate", "edge_pts", "t_stat", "p_value", "profit_factor"]

    lines = [
        f"=== PATTERN EDGE ANALYSIS — {h}-bar forward returns (MNQ points) ===",
        "",
        sub.to_string(float_format=lambda x: f"{x:.4f}" if abs(x) < 100 else f"{x:.1f}"),
        "",
        "Significance: p < 0.05 = *, p < 0.01 = **, p < 0.001 = ***",
    ]

    # Flag significant edges
    sig = stats_df[stats_df[f"fwd{h}_p_value"] < 0.05].drop("BASELINE", errors="ignore")
    if len(sig) > 0:
        sig_sorted = sig.sort_values(f"fwd{h}_edge", key=abs, ascending=False)
        lines.append("")
        lines.append(f"Statistically significant patterns (p<0.05) at {h}-bar horizon:")
        for name, row in sig_sorted.iterrows():
            direction = "BULLISH" if row[f"fwd{h}_edge"] > 0 else "BEARISH"
            stars = "***" if row[f"fwd{h}_p_value"] < 0.001 else ("**" if row[f"fwd{h}_p_value"] < 0.01 else "*")
            lines.append(
                f"  {name:25s} edge={row[f'fwd{h}_edge']:+.4f} pts  "
                f"win={row[f'fwd{h}_win_rate']:.1%}  PF={row[f'fwd{h}_profit_factor']:.2f}  "
                f"{direction} {stars}"
            )
    else:
        lines.append(f"\nNo statistically significant patterns at {h}-bar horizon.")

    return "\n".join(lines)


def by_session(
    df: pd.DataFrame,
    horizon: int = 1,
) -> pd.DataFrame:
    """Break down pattern edge by trading session (Asia / London / NY).

    CME MNQ sessions (US/Eastern):
        Asia:   18:00–02:00 (prior day evening → early morning)
        London: 02:00–09:30
        NY:     09:30–16:00
    """
    fwd_col = f"fwd_{horizon}"
    pat_cols = [c for c in df.columns if c.startswith("pat_")]

    hour = df.index.hour
    # Sessions in ET
    session = pd.Series("off_hours", index=df.index)
    session[(hour >= 18) | (hour < 2)] = "asia"
    session[(hour >= 2) & (hour < 9) | ((hour == 9) & (df.index.minute < 30))] = "london"
    session[((hour == 9) & (df.index.minute >= 30)) | ((hour >= 10) & (hour < 16))] = "ny"

    rows = []
    for pc in pat_cols:
        name = pc.replace("pat_", "")
        for sess in ["asia", "london", "ny"]:
            mask = df[pc].astype(bool) & (session == sess)
            vals = df.loc[mask, fwd_col].dropna()
            n = len(vals)
            if n < 30:
                continue
            rows.append({
                "pattern": name,
                "session": sess,
                "n": n,
                "mean_pts": vals.mean(),
                "win_rate": (vals > 0).mean(),
                "std": vals.std(),
            })

    return pd.DataFrame(rows)
