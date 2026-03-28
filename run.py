"""Unified CLI entry point for all strategies.

Usage:
    python run.py <strategy>                     # baseline backtest
    python run.py <strategy> --optimize          # grid search
    python run.py <strategy> --walk-forward      # walk-forward analysis
    python run.py <strategy> --report            # generate report
    python run.py --list                         # list available strategies

Examples:
    python run.py srs
    python run.py srs --optimize
    python run.py intmom --walk-forward
    python run.py gapfade --report
"""

import argparse
import json
from pathlib import Path

from src.backtester import run_backtest
from src.data_loader import load_processed
from src.metrics import calculate_metrics
from src.strategies import get_strategy, list_strategies

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")


def get_data(instrument: str):
    """Load data from Parquet cache."""
    cache = PROCESSED_DIR / f"{instrument}_1m.parquet"
    if not cache.exists():
        raise FileNotFoundError(
            f"{cache} not found. Run run_backtest.py first to generate cache."
        )
    print(f"  {instrument}: loading from {cache}")
    return load_processed(cache)


def load_data():
    """Load both instruments."""
    print("Loading data...")
    data = {}
    for instrument in ["MES", "MNQ"]:
        data[instrument] = get_data(instrument)
        print(f"  {instrument}: {len(data[instrument]):,} bars")
    return data


def cmd_baseline(strategy_cls, args):
    """Run baseline backtest with default parameters."""
    data = load_data()

    print(f"\nRunning baseline ({strategy_cls.name}, default params)...")
    strategy = strategy_cls.from_params(strategy_cls.default_params())
    trades = run_backtest(strategy, data)
    print(f"  {len(trades)} trades")

    metrics = calculate_metrics(trades)

    print(f"\n{'='*50}")
    print(f"  {strategy_cls.name.upper()} — BASELINE")
    print(f"{'='*50}")
    for key in ["total_trades", "win_rate", "profit_factor", "total_pnl",
                "sharpe_ratio", "max_drawdown", "avg_pnl_per_trade"]:
        val = metrics.get(key, 0)
        if isinstance(val, float):
            print(f"  {key:<25} {val:>12.4f}")
        else:
            print(f"  {key:<25} {val:>12}")
    print(f"{'='*50}")

    # Save full metrics (serialize DataFrames/Series to dicts)
    RESULTS_DIR.mkdir(exist_ok=True)
    import pandas as pd
    baseline = {}
    for k, v in metrics.items():
        if isinstance(v, (pd.DataFrame, pd.Series)):
            baseline[k] = v.to_dict()
        elif isinstance(v, (float, int, str, bool, type(None))):
            baseline[k] = v
        else:
            baseline[k] = str(v)
    path = RESULTS_DIR / f"{strategy_cls.name}_baseline.json"
    with open(path, "w") as f:
        json.dump(baseline, f, indent=2, default=str)
    print(f"\n  Saved to {path}")


def cmd_optimize(strategy_cls, args):
    """Run grid search."""
    data = load_data()

    print(f"\nBuilding parameter grid...")
    param_grid = strategy_cls.build_param_grid()
    print(f"  {len(param_grid)} combinations")

    if args.coarse_to_fine:
        from src.optimizer import coarse_to_fine_grid_search

        print(f"\nRunning coarse-to-fine grid search ({strategy_cls.name})...")
        results = coarse_to_fine_grid_search(
            strategy_cls, data, param_grid,
            progress_file=args.progress_file,
        )
    elif args.progress_file:
        from src.optimizer import generic_grid_search

        print(f"\nRunning grid search ({strategy_cls.name})...")
        results = generic_grid_search(
            strategy_cls, data, param_grid,
            progress_file=args.progress_file,
        )
    else:
        print(f"\nRunning grid search ({strategy_cls.name})...")
        results = strategy_cls.run_grid_search(data, param_grid, progress=True)

    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{strategy_cls.name}_grid.csv"
    results.to_csv(path, index=False)
    print(f"\nResults saved to {path}")

    # Show top 10 by profit factor
    top = results.nlargest(10, "profit_factor")
    print("\nTop 10 by profit factor:")
    cols = strategy_cls.param_columns() + ["total_trades", "win_rate", "profit_factor", "total_pnl"]
    display_cols = [c for c in cols if c in top.columns]
    print(top[display_cols].to_string(index=False))


def cmd_walk_forward(strategy_cls, args):
    """Run walk-forward analysis."""
    data = load_data()

    print(f"\nBuilding parameter grid...")
    param_grid = strategy_cls.build_param_grid()
    print(f"  {len(param_grid)} combinations")

    print(f"\nRunning walk-forward ({strategy_cls.name}, 24-month train, 6-month test)...")
    if args.progress_file:
        from src.optimizer import generic_walk_forward

        wf_results = generic_walk_forward(
            strategy_cls, data, param_grid,
            progress_file=args.progress_file,
        )
    else:
        wf_results = strategy_cls.run_walk_forward(data, param_grid)

    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{strategy_cls.name}_wf.csv"
    wf_results.to_csv(path, index=False)
    print(f"\nResults saved to {path}")
    print(f"  Shape: {wf_results.shape}")


def cmd_report(strategy_cls, args):
    """Generate report from saved results."""
    import pandas as pd

    grid_path = RESULTS_DIR / f"{strategy_cls.name}_grid.csv"
    wf_path = RESULTS_DIR / f"{strategy_cls.name}_wf.csv"
    baseline_path = RESULTS_DIR / f"{strategy_cls.name}_baseline.json"

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
    print(strategy_cls.generate_report(grid_results, wf_results, baseline))


def main():
    parser = argparse.ArgumentParser(
        description="Unified backtester CLI for all strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "strategy", nargs="?",
        help="Strategy name (use --list to see available)",
    )
    parser.add_argument("--list", action="store_true",
                        help="List available strategies")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--optimize", action="store_true",
                       help="Run grid search")
    group.add_argument("--walk-forward", action="store_true",
                       help="Run walk-forward analysis")
    group.add_argument("--report", action="store_true",
                       help="Generate report from saved results")
    parser.add_argument("--coarse-to-fine", action="store_true",
                        help="Use coarse-to-fine two-phase grid search (requires --optimize)")
    parser.add_argument("--progress-file", type=str, default=None,
                        help="Write optimization progress to JSON file")
    args = parser.parse_args()

    if args.coarse_to_fine and not args.optimize:
        parser.error("--coarse-to-fine requires --optimize")

    if args.list:
        print("Available strategies:")
        for name in list_strategies():
            cls = get_strategy(name)
            n_params = len(cls.build_param_grid())
            print(f"  {name:<20} ({n_params} grid combos)")
        return

    if not args.strategy:
        parser.error("strategy name is required (use --list to see available)")

    strategy_cls = get_strategy(args.strategy)

    if args.optimize:
        cmd_optimize(strategy_cls, args)
    elif args.walk_forward:
        cmd_walk_forward(strategy_cls, args)
    elif args.report:
        cmd_report(strategy_cls, args)
    else:
        cmd_baseline(strategy_cls, args)


if __name__ == "__main__":
    main()
