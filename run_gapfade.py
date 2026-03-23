"""CLI entry point for Overnight Gap Fade strategy.

Usage:
    python run_gapfade.py                    # baseline (paper params)
    python run_gapfade.py --optimize         # grid search (180 combos)
    python run_gapfade.py --walk-forward     # walk-forward analysis
    python run_gapfade.py --report           # generate report
"""

import argparse
import json
from pathlib import Path

from src.data_loader import load_processed
from src.metrics import calculate_metrics
from src.strategies.gapfade import (
    GapFadeStrategy,
    build_param_grid,
    generate_report,
    run_backtest,
    run_grid_search,
    run_walk_forward,
)

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")


def get_data(instrument: str):
    """Load data from Parquet cache."""
    cache = PROCESSED_DIR / f"{instrument}_1m.parquet"
    if not cache.exists():
        raise FileNotFoundError(f"{cache} not found. Run run_backtest.py first.")
    print(f"  {instrument}: loading from {cache}")
    return load_processed(cache)


def run_baseline(args):
    """Run baseline with paper-exact parameters."""
    print("Loading data...")
    data = {}
    for instrument in ["MES", "MNQ"]:
        data[instrument] = get_data(instrument)
        print(f"  {instrument}: {len(data[instrument]):,} bars")

    print("\nRunning baseline (paper parameters)...")
    strategy = GapFadeStrategy()  # all defaults = paper params
    trades = run_backtest(data, strategy)
    print(f"  {len(trades)} trades")

    metrics = calculate_metrics(trades)

    # Print summary
    print(f"\n{'='*50}")
    print("  GAP FADE — BASELINE")
    print(f"{'='*50}")
    for key in ["total_trades", "win_rate", "profit_factor", "total_pnl",
                "sharpe_ratio", "max_drawdown", "avg_pnl_per_trade"]:
        val = metrics.get(key, 0)
        if isinstance(val, float):
            print(f"  {key:<25} {val:>12.4f}")
        else:
            print(f"  {key:<25} {val:>12}")
    print(f"{'='*50}")

    # Print per-instrument breakdown
    if "by_instrument" in metrics:
        print("\n  Per instrument:")
        print(metrics["by_instrument"].to_string())

    # Print exit reason breakdown
    if "exit_reasons" in metrics:
        print("\n  Exit reasons:")
        for reason, count in metrics["exit_reasons"].items():
            pct = count / metrics["total_trades"] * 100
            print(f"    {reason:<16} : {count:>5}  ({pct:.1f}%)")

    # Print long/short breakdown
    print(f"\n  Long  trades: {metrics.get('trades_long', 0):>5}  "
          f"PnL: ${metrics.get('pnl_long', 0):>10,.2f}  "
          f"WR: {metrics.get('win_rate_long', 0):.1%}")
    print(f"  Short trades: {metrics.get('trades_short', 0):>5}  "
          f"PnL: ${metrics.get('pnl_short', 0):>10,.2f}  "
          f"WR: {metrics.get('win_rate_short', 0):.1%}")

    # Save
    RESULTS_DIR.mkdir(exist_ok=True)
    baseline = {k: metrics.get(k, 0) for k in [
        "total_trades", "win_rate", "profit_factor", "total_pnl",
        "sharpe_ratio", "max_drawdown", "avg_pnl_per_trade",
    ]}
    path = RESULTS_DIR / "gapfade_baseline.json"
    with open(path, "w") as f:
        json.dump(baseline, f, indent=2)
    print(f"\n  Saved to {path}")


def run_optimize(args):
    """Run grid search over 180 combinations."""
    print("Loading data...")
    data = {}
    for instrument in ["MES", "MNQ"]:
        data[instrument] = get_data(instrument)
        print(f"  {instrument}: {len(data[instrument]):,} bars")

    print("\nBuilding parameter grid...")
    param_grid = build_param_grid()
    print(f"  {len(param_grid)} combinations")

    print("\nRunning grid search...")
    results = run_grid_search(data, param_grid, progress=True)

    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / "gapfade_grid.csv"
    results.to_csv(path, index=False)
    print(f"\nResults saved to {path}")

    # Show top 10
    top = results.nlargest(10, "total_pnl")[
        ["gap_measure_time", "min_gap_pct", "fill_pct", "stop_gap_multiple",
         "total_trades", "win_rate", "profit_factor", "total_pnl", "sharpe_ratio"]
    ]
    print("\nTop 10 by total PnL:")
    print(top.to_string(index=False))


def run_walk_forward_cmd(args):
    """Run walk-forward analysis."""
    print("Loading data...")
    data = {}
    for instrument in ["MES", "MNQ"]:
        data[instrument] = get_data(instrument)
        print(f"  {instrument}: {len(data[instrument]):,} bars")

    print("\nBuilding parameter grid...")
    param_grid = build_param_grid()
    print(f"  {len(param_grid)} combinations")

    print("\nRunning walk-forward analysis (24-month train, 6-month test)...")
    wf_results = run_walk_forward(data, param_grid)

    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / "gapfade_walk_forward.csv"
    wf_results.to_csv(path, index=False)
    print(f"\nResults saved to {path}")
    print(f"  Shape: {wf_results.shape}")


def run_report(args):
    """Generate report from saved results."""
    import pandas as pd

    grid_path = RESULTS_DIR / "gapfade_grid.csv"
    wf_path = RESULTS_DIR / "gapfade_walk_forward.csv"
    baseline_path = RESULTS_DIR / "gapfade_baseline.json"

    grid_results = None
    if grid_path.exists():
        grid_results = pd.read_csv(grid_path)
        print(f"Loaded grid: {grid_results.shape[0]} combinations")

    wf_results = None
    if wf_path.exists():
        wf_results = pd.read_csv(wf_path)
        print(f"Loaded walk-forward: {wf_results.shape[0]} rows")

    baseline = None
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)
        print(f"Loaded baseline: {baseline.get('total_trades', 0)} trades")

    print("\n")
    print(generate_report(grid_results, wf_results, baseline))


def main():
    parser = argparse.ArgumentParser(description="Overnight Gap Fade Strategy")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--optimize", action="store_true",
                       help="Run grid search (180 combos)")
    group.add_argument("--walk-forward", action="store_true",
                       help="Run walk-forward analysis")
    group.add_argument("--report", action="store_true",
                       help="Generate report from saved results")
    args = parser.parse_args()

    if args.optimize:
        run_optimize(args)
    elif args.walk_forward:
        run_walk_forward_cmd(args)
    elif args.report:
        run_report(args)
    else:
        run_baseline(args)


if __name__ == "__main__":
    main()
