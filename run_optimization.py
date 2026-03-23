"""CLI entry point for SRS strategy optimization.

Usage:
    python run_optimization.py --grid           # Run grid search (default)
    python run_optimization.py --walk-forward   # Walk-forward analysis (Step 3)
    python run_optimization.py --report         # Generate report (Step 4)
"""

import argparse
from pathlib import Path

import json

from src.data_loader import load_and_prepare, load_processed, save_processed
from src.strategies.srs import SRSStrategy

DATA_PATH = Path("data/raw/glbx-mdp3-20100606-20260315.ohlcv-1m.dbn.zst")
PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")


def get_data(instrument: str):
    """Load data from Parquet cache if available, else process raw."""
    cache = PROCESSED_DIR / f"{instrument}_1m.parquet"
    if cache.exists():
        print(f"  Loading from cache: {cache}")
        return load_processed(cache)

    print(f"  Processing raw data for {instrument}...")
    df = load_and_prepare(DATA_PATH, instrument)
    save_processed(df, PROCESSED_DIR, instrument)
    print(f"  Cache saved: {cache}")
    return df


def run_grid(args):
    """Run grid search over 432 parameter combinations."""
    print("Loading data...")
    data = {}
    for instrument in ["MES", "MNQ"]:
        data[instrument] = get_data(instrument)
        print(f"  {instrument}: {len(data[instrument]):,} bars")

    print(f"\nBuilding parameter grid...")
    param_grid = SRSStrategy.build_param_grid()
    print(f"  {len(param_grid)} combinations")

    print("\nRunning grid search...")
    results_df = SRSStrategy.run_grid_search(data, param_grid, progress=True)

    RESULTS_DIR.mkdir(exist_ok=True)
    output_path = RESULTS_DIR / "optimization_grid.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    print(f"  Shape: {results_df.shape}")

    # Show top 10 by total_pnl
    top = results_df.nlargest(10, "total_pnl")[
        ["rr_ratio", "range_minutes", "reentry_check", "sessions",
         "total_trades", "win_rate", "profit_factor", "total_pnl", "sharpe_ratio"]
    ]
    print("\nTop 10 combos by total PnL:")
    print(top.to_string(index=False))


def run_walk_forward_cmd(args):
    """Run walk-forward analysis and save results/walk_forward_results.csv."""
    print("Loading data...")
    data = {}
    for instrument in ["MES", "MNQ"]:
        data[instrument] = get_data(instrument)
        print(f"  {instrument}: {len(data[instrument]):,} bars")

    print("\nBuilding parameter grid...")
    param_grid = SRSStrategy.build_param_grid()
    print(f"  {len(param_grid)} combinations")

    print("\nRunning walk-forward analysis (24-month train, 6-month test)...")
    wf_results = SRSStrategy.run_walk_forward(data, param_grid)

    RESULTS_DIR.mkdir(exist_ok=True)
    output_path = RESULTS_DIR / "walk_forward_results.csv"
    wf_results.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    print(f"  Shape: {wf_results.shape}")

    if not wf_results.empty:
        n_windows = wf_results["window_id"].nunique()
        oos_win_rate = wf_results.groupby("params")["oos_pnl"].apply(
            lambda x: (x > 0).mean()
        )
        robust_count = (oos_win_rate >= 0.7).sum()
        print(f"\n{robust_count} robust parameters over {n_windows} windows (>= 70% OOS profitable)")


def run_report(args):
    """Load saved results and generate a comprehensive text report."""
    grid_path = RESULTS_DIR / "optimization_grid.csv"
    wf_path = RESULTS_DIR / "walk_forward_results.csv"
    baseline_path = RESULTS_DIR / "baseline.json"

    if not grid_path.exists():
        print(f"Error: {grid_path} not found. Run --grid first.")
        return

    import pandas as pd
    grid_results = pd.read_csv(grid_path)
    print(f"Loaded grid results: {grid_results.shape[0]} combinations")

    wf_results = None
    if wf_path.exists():
        wf_results = pd.read_csv(wf_path)
        print(f"Loaded walk-forward results: {wf_results.shape[0]} rows")
    else:
        print("No walk-forward results found (run --walk-forward to compute OOS metrics)")

    baseline = None
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)
        print(f"Loaded baseline: {baseline.get('total_trades', 0)} trades")
    else:
        print("No baseline.json found")

    print("\nGenerating report...\n")
    report = SRSStrategy.generate_report(grid_results, wf_results, baseline)
    print(report)


def main():
    parser = argparse.ArgumentParser(description="SRS Strategy Optimizer")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--grid", action="store_true", default=True,
                       help="Run grid search (default)")
    group.add_argument("--walk-forward", action="store_true",
                       help="Run walk-forward analysis")
    group.add_argument("--report", action="store_true",
                       help="Generate results report")
    args = parser.parse_args()

    if args.walk_forward:
        run_walk_forward_cmd(args)
        return

    if args.report:
        run_report(args)
        return

    run_grid(args)


if __name__ == "__main__":
    main()
