"""Main pipeline orchestrator.

Loads data → computes morphology → detects patterns → aggregates timeframes
→ exports results.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from mnq_morphology.loaders import load_ohlcv
from mnq_morphology.features import compute_morphology
from mnq_morphology.patterns import detect_patterns
from mnq_morphology.aggregators import aggregate_timeframe
from mnq_morphology.aggregators.timeframe import TIMEFRAMES


def run_pipeline(
    csv_path: str | Path,
    *,
    output_dir: str | Path = "output",
    timeframes: tuple[str, ...] = TIMEFRAMES,
    tz: str = "US/Eastern",
    start: str | None = None,
    end: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Execute the full morphology pipeline.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys: '1min', and each aggregated timeframe.
        Each DataFrame contains OHLCV + morphology features + pattern flags.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load
    print(f"[1/4] Loading data from {csv_path} …")
    df = load_ohlcv(csv_path, tz=tz, start=start, end=end)
    print(f"       Loaded {len(df):,} bars  ({df.index.min()} → {df.index.max()})")

    results: dict[str, pd.DataFrame] = {}

    # 2-3. Morphology + Patterns on base 1-min timeframe
    print("[2/4] Computing morphology features …")
    df_morph = compute_morphology(df)
    print("[3/4] Detecting candlestick patterns …")
    df_pats = detect_patterns(df_morph)
    df_full = pd.concat([df_morph, df_pats], axis=1)
    results["1min"] = df_full

    # Save base timeframe
    out_path = output_dir / "mnq_1min_morphology.parquet"
    df_full.to_parquet(out_path)
    print(f"       Saved → {out_path}")

    # 4. Aggregate to higher timeframes and repeat
    print("[4/4] Aggregating to higher timeframes …")
    for tf in timeframes:
        df_tf = aggregate_timeframe(df, tf)
        df_tf_morph = compute_morphology(df_tf)
        df_tf_pats = detect_patterns(df_tf_morph)
        df_tf_full = pd.concat([df_tf_morph, df_tf_pats], axis=1)
        results[tf] = df_tf_full

        safe_name = tf.replace("min", "m").lower()
        out_path = output_dir / f"mnq_{safe_name}_morphology.parquet"
        df_tf_full.to_parquet(out_path)
        print(f"       {tf:>5}: {len(df_tf_full):>10,} bars → {out_path}")

    # Summary
    _print_summary(results)

    return results


def _print_summary(results: dict[str, pd.DataFrame]) -> None:
    """Print a quick summary of pattern detections across timeframes."""
    print("\n" + "=" * 60)
    print("PATTERN SUMMARY")
    print("=" * 60)
    pat_cols = [c for c in results["1min"].columns if c.startswith("pat_")]
    header = f"{'Timeframe':>10}" + "".join(f"{p.replace('pat_',''):>18}" for p in pat_cols)
    print(header)
    print("-" * len(header))
    for tf, df in results.items():
        counts = "".join(f"{df[p].sum():>18,}" for p in pat_cols)
        print(f"{tf:>10}{counts}")
    print("=" * 60)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="MNQ Morphology Pipeline — candlestick analysis for MNQ futures",
    )
    parser.add_argument(
        "csv",
        help="Path to the OHLCV CSV file (Databento glbx-mdp3 or generic).",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="output",
        help="Directory for output parquet files (default: output/).",
    )
    parser.add_argument(
        "--tz",
        default="US/Eastern",
        help="Timezone for timestamps (default: US/Eastern).",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Start date filter (ISO-8601).",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="End date filter (ISO-8601).",
    )
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=list(TIMEFRAMES),
        help=f"Timeframes to aggregate (default: {' '.join(TIMEFRAMES)}).",
    )
    args = parser.parse_args(argv)

    run_pipeline(
        args.csv,
        output_dir=args.output_dir,
        timeframes=tuple(args.timeframes),
        tz=args.tz,
        start=args.start,
        end=args.end,
    )


if __name__ == "__main__":
    main()
