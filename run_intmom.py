"""CLI entry point for Intraday Momentum strategy.

Usage:
    python run_intmom.py                     # baseline (paper params)
    python run_intmom.py --optimize          # grid search
    python run_intmom.py --walk-forward      # walk-forward analysis
    python run_intmom.py --report            # generate report
"""

import argparse
import json
from pathlib import Path

from src.data_loader import load_processed
from src.metrics import calculate_metrics
from src.strategy_intmom import (
    IntradayMomentumStrategy,
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
    strategy = IntradayMomentumStrategy()  # all defaults = paper params
    trades = run_backtest(data, strategy)
    print(f"  {len(trades)} trades")

    metrics = calculate_metrics(trades)

    # Print summary
    print(f"\n{'='*50}")
    print("  INTRADAY MOMENTUM — BASELINE")
    print(f"{'='*50}")
    for key in ["total_trades", "win_rate", "profit_factor", "total_pnl",
                 "sharpe_ratio", "max_drawdown", "avg_pnl_per_trade"]:
        val = metrics.get(key, 0)
        if isinstance(val, float):
            print(f"  {key:<25} {val:>12.4f}")
        else:
            print(f"  {key:<25} {val:>12}")
    print(f"{'='*50}")

    # Save
    RESULTS_DIR.mkdir(exist_ok=True)
    baseline = {k: metrics.get(k, 0) for k in [
        "total_trades", "win_rate", "profit_factor", "total_pnl",
        "sharpe_ratio", "max_drawdown", "avg_pnl_per_trade",
    ]}
    path = RESULTS_DIR / "intmom_baseline.json"
    with open(path, "w") as f:
        json.dump(baseline, f, indent=2)
    print(f"\n  Saved to {path}")


def run_optimize(args):
    """Run grid search over 1,024 combinations."""
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
    path = RESULTS_DIR / "intmom_grid.csv"
    results.to_csv(path, index=False)
    print(f"\nResults saved to {path}")

    # Show top 10
    top = results.nlargest(10, "total_pnl")[
        ["signal_end", "entry_time", "min_signal_pct", "stop_loss_ticks",
         "take_profit_ticks", "total_trades", "win_rate", "profit_factor",
         "total_pnl", "sharpe_ratio"]
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
    path = RESULTS_DIR / "intmom_walk_forward.csv"
    wf_results.to_csv(path, index=False)
    print(f"\nResults saved to {path}")
    print(f"  Shape: {wf_results.shape}")


def run_report(args):
    """Generate report from saved results."""
    import pandas as pd

    grid_path = RESULTS_DIR / "intmom_grid.csv"
    wf_path = RESULTS_DIR / "intmom_walk_forward.csv"
    baseline_path = RESULTS_DIR / "intmom_baseline.json"

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
    parser = argparse.ArgumentParser(description="Intraday Momentum Strategy")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--optimize", action="store_true",
                       help="Run grid search")
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
