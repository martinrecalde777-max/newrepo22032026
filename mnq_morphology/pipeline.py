"""MNQ Morphology Pipeline — full end-to-end execution.

Usage:
    python -m mnq_morphology.pipeline [--output-dir output] [--start 2024-01-01] [--end 2025-01-01]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from mnq_morphology.loader import load_parquet, build_continuous
from mnq_morphology.morphology import compute_morphology
from mnq_morphology.patterns import detect_patterns
from mnq_morphology.aggregator import aggregate, TIMEFRAMES


DATA_PATH = "data/glbx-mdp3-20210312-20260311.ohlcv-1m.parquet"


def run(
    data_path: str = DATA_PATH,
    *,
    output_dir: str = "output",
    timeframes: tuple[str, ...] = TIMEFRAMES,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Execute the full pipeline on real MNQ data."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = {}

    # 1. Load
    t0 = time.time()
    print(f"[1/5] Loading {data_path} …")
    raw = load_parquet(data_path)
    print(f"       {len(raw):,} bars, {raw['symbol'].nunique()} contracts in {time.time()-t0:.1f}s")

    # 2. Continuous series
    t1 = time.time()
    print("[2/5] Building continuous front-month series …")
    cont = build_continuous(raw)
    if start:
        cont = cont.loc[start:]
    if end:
        cont = cont.loc[:end]
    print(f"       {len(cont):,} bars ({cont.index.min()} → {cont.index.max()}) in {time.time()-t1:.1f}s")

    # 3. Morphology
    t2 = time.time()
    print("[3/5] Computing morphology features …")
    morph = compute_morphology(cont)
    print(f"       {morph.shape[1]} features in {time.time()-t2:.1f}s")

    # 4. Patterns
    t3 = time.time()
    print("[4/5] Detecting candlestick patterns …")
    pats = detect_patterns(morph)
    full = pd.concat([morph, pats], axis=1)
    results["1min"] = full

    fp = out / "mnq_1min_morphology.parquet"
    full.to_parquet(fp)
    print(f"       17 patterns detected in {time.time()-t3:.1f}s → {fp}")

    # 5. Multi-timeframe
    t4 = time.time()
    print("[5/5] Aggregating to higher timeframes …")
    for tf in timeframes:
        agg = aggregate(cont, tf)
        agg_morph = compute_morphology(agg)
        agg_pats = detect_patterns(agg_morph)
        agg_full = pd.concat([agg_morph, agg_pats], axis=1)
        results[tf] = agg_full

        safe = tf.replace("min", "m").lower()
        fp = out / f"mnq_{safe}_morphology.parquet"
        agg_full.to_parquet(fp)
        print(f"       {tf:>5}: {len(agg_full):>10,} bars → {fp}")

    print(f"\nTotal time: {time.time()-t0:.1f}s")

    # Summary
    _summary(results)
    return results


def _summary(results: dict[str, pd.DataFrame]) -> None:
    pat_cols = [c for c in results["1min"].columns if c.startswith("pat_")]
    print("\n" + "=" * 70)
    print("PATTERN SUMMARY ACROSS TIMEFRAMES")
    print("=" * 70)
    for tf, df in results.items():
        total = len(df)
        counts = {p.replace("pat_", ""): int(df[p].sum()) for p in pat_cols}
        top3 = sorted(counts.items(), key=lambda x: -x[1])[:3]
        top_str = ", ".join(f"{n}={c:,}" for n, c in top3)
        print(f"  {tf:>5} ({total:>10,} bars): {top_str}")
    print("=" * 70)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="MNQ Morphology Pipeline")
    parser.add_argument("--data", default=DATA_PATH, help="Path to parquet file")
    parser.add_argument("-o", "--output-dir", default="output")
    parser.add_argument("--start", default=None, help="Start date (ISO-8601)")
    parser.add_argument("--end", default=None, help="End date (ISO-8601)")
    parser.add_argument("--timeframes", nargs="+", default=list(TIMEFRAMES))
    args = parser.parse_args(argv)

    run(args.data, output_dir=args.output_dir, timeframes=tuple(args.timeframes),
        start=args.start, end=args.end)


if __name__ == "__main__":
    main()
