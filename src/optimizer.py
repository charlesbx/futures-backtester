"""Generic optimization: grid search, walk-forward analysis, and reporting.

Strategy-agnostic — works with any BaseStrategy subclass.
Strategies can override run_grid_search/run_walk_forward on their class
for performance optimizations (e.g. precomputing signals).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from tqdm import tqdm

from .backtester import run_backtest
from .metrics import calculate_metrics

if TYPE_CHECKING:
    from .trade import BaseStrategy

METRIC_COLUMNS = [
    "total_trades", "win_rate", "profit_factor", "total_pnl",
    "avg_pnl_per_trade", "max_drawdown", "sharpe_ratio",
    "pnl_long", "pnl_short", "trades_long", "trades_short",
    "win_rate_long", "win_rate_short",
    "profit_concentration",
    "pf_ci_low", "pf_ci_high",
    "sharpe_ci_low", "sharpe_ci_high",
    "max_dd_duration_trades",
]


def _extract_metrics(metrics: dict) -> dict:
    """Extract standard metric columns from a metrics dict."""
    return {k: metrics.get(k, 0) for k in METRIC_COLUMNS}


def generic_grid_search(
    strategy_cls: type[BaseStrategy],
    data: dict[str, pd.DataFrame],
    param_grid: list[dict] | None = None,
    slippage_ticks: float = 1,
    commission: float = 1.24,
    progress: bool = True,
) -> pd.DataFrame:
    """Naive grid search — runs full backtest for each param combination.

    For better performance, strategies should override
    ``BaseStrategy.run_grid_search`` with precomputation logic.
    """
    if param_grid is None:
        param_grid = strategy_cls.build_param_grid()

    rows: list[dict] = []
    iterator = (
        tqdm(param_grid, desc=f"Grid search ({strategy_cls.name})")
        if progress else param_grid
    )

    for params in iterator:
        strategy = strategy_cls.from_params(params)
        trades = run_backtest(strategy, data, slippage_ticks, commission)
        metrics = calculate_metrics(trades)
        row = {**strategy.to_params(), **_extract_metrics(metrics)}
        rows.append(row)

    return pd.DataFrame(rows)


def apply_fdr_correction(grid_results: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Apply Benjamini-Hochberg FDR correction to grid search results.

    Adds 'fdr_significant' column: True if the result survives correction.
    Uses a simple heuristic: profit factor significantly > 1.0 based on
    trade count and win rate variance.
    """
    from scipy import stats

    df = grid_results.copy()
    p_values = []

    for _, row in df.iterrows():
        n = int(row.get("total_trades", 0))
        pf = row.get("profit_factor", 0)
        wr = row.get("win_rate", 0)

        if n < 10 or pf <= 0:
            p_values.append(1.0)
            continue

        # Test if win rate is significantly different from 50% using binomial test
        wins = int(round(wr * n))
        p = stats.binom_test(wins, n, 0.5, alternative="greater") if n > 0 else 1.0
        p_values.append(p)

    df["p_value"] = p_values

    # Benjamini-Hochberg procedure
    n_tests = len(df)
    if n_tests > 0:
        sorted_idx = np.argsort(p_values)
        ranks = np.empty_like(sorted_idx)
        ranks[sorted_idx] = np.arange(1, n_tests + 1)
        thresholds = alpha * ranks / n_tests
        df["fdr_significant"] = df["p_value"] <= thresholds
    else:
        df["fdr_significant"] = False

    return df


def generic_walk_forward(
    strategy_cls: type[BaseStrategy],
    data: dict[str, pd.DataFrame],
    param_grid: list[dict] | None = None,
    train_months: int = 24,
    test_months: int = 6,
    top_n: int = 10,
    slippage_ticks: float = 1,
    commission: float = 1.24,
) -> pd.DataFrame:
    """Rolling walk-forward analysis.

    Windows: train_months train, test_months test, test_months step.
    Train and test windows never overlap.

    For each window:
    1. Run grid search on train period (using strategy's optimized version)
    2. Select top_n by profit_factor
    3. Evaluate each out-of-sample on test period
    """
    if param_grid is None:
        param_grid = strategy_cls.build_param_grid()

    tz = next(iter(data.values())).index.tz
    global_start = max(df.index.min().date() for df in data.values())
    global_end = min(df.index.max().date() for df in data.values())

    # Build rolling windows (step = test_months, no test overlap)
    windows: list[tuple] = []
    current = global_start
    while True:
        train_start = pd.Timestamp(current, tz=tz)
        train_end = train_start + pd.DateOffset(months=train_months)
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=test_months)

        if test_end.date() > global_end:
            break

        windows.append((train_start, train_end, test_start, test_end))
        current = (train_start + pd.DateOffset(months=test_months)).date()

    param_cols = strategy_cls.param_columns()
    rows: list[dict] = []

    for wid, (train_start, train_end, test_start, test_end) in enumerate(
        tqdm(windows, desc="Walk-forward")
    ):
        train_data = {
            inst: df[(df.index >= train_start) & (df.index < train_end)]
            for inst, df in data.items()
        }
        test_data = {
            inst: df[(df.index >= test_start) & (df.index < test_end)]
            for inst, df in data.items()
        }

        if any(d.empty for d in train_data.values()):
            continue

        # In-sample grid search (uses strategy's optimized version)
        is_results = strategy_cls.run_grid_search(
            train_data, param_grid, slippage_ticks, commission, progress=False,
        )
        if is_results.empty:
            continue

        top_rows = is_results.nlargest(top_n, "profit_factor")

        for _, is_row in top_rows.iterrows():
            params = {col: is_row[col] for col in param_cols if col in is_row}
            strategy = strategy_cls.from_params(params)
            oos_trades = run_backtest(
                strategy, test_data, slippage_ticks, commission,
            )
            oos_metrics = calculate_metrics(oos_trades)

            rows.append({
                "window_id": wid,
                "train_start": str(train_start.date()),
                "train_end": str(train_end.date()),
                "test_start": str(test_start.date()),
                "test_end": str(test_end.date()),
                "params": json.dumps(
                    strategy.to_params(), sort_keys=True, default=str,
                ),
                "is_pnl": is_row.get("total_pnl", 0),
                "oos_pnl": oos_metrics.get("total_pnl", 0),
                "is_sharpe": is_row.get("sharpe_ratio", 0),
                "oos_sharpe": oos_metrics.get("sharpe_ratio", 0),
                "is_pf": is_row.get("profit_factor", 0),
                "oos_pf": oos_metrics.get("profit_factor", 0),
            })

    result_df = pd.DataFrame(rows)

    if not result_df.empty:
        # OOS consistency metrics
        by_window = result_df.groupby("window_id").agg(
            oos_pf_mean=("oos_pf", "mean"),
            oos_pnl_sum=("oos_pnl", "sum"),
        )
        result_df.attrs["oos_windows_total"] = int(by_window.shape[0])
        result_df.attrs["oos_windows_profitable"] = int((by_window["oos_pnl_sum"] > 0).sum())
        result_df.attrs["oos_win_rate"] = float((by_window["oos_pnl_sum"] > 0).mean())
        result_df.attrs["oos_pf_mean"] = float(result_df["oos_pf"].mean())
        result_df.attrs["oos_pf_std"] = float(result_df["oos_pf"].std())
        result_df.attrs["oos_pf_min"] = float(result_df["oos_pf"].min())
        result_df.attrs["oos_pf_max"] = float(result_df["oos_pf"].max())

    return result_df


def default_report(
    strategy_name: str,
    grid_results: pd.DataFrame | None,
    wf_results: pd.DataFrame | None,
    baseline: dict | None,
    param_columns: list[str],
) -> str:
    """Generate a basic text report from optimization results."""
    lines: list[str] = []
    SEP = "=" * 72

    def section(title: str) -> None:
        lines.extend(["", SEP, f"  {title}", SEP])

    # ── 1. BASELINE ──
    section(f"1. BASELINE — {strategy_name.upper()}")
    if baseline and baseline.get("total_trades", 0) > 0:
        b = baseline
        lines.append(f"  Trades         : {b.get('total_trades', 'N/A')}")
        lines.append(f"  Win rate       : {b.get('win_rate', 0):.1%}")
        lines.append(f"  Profit factor  : {b.get('profit_factor', 0):.3f}")
        lines.append(f"  Total PnL      : ${b.get('total_pnl', 0):>10,.2f}")
        lines.append(f"  Sharpe ratio   : {b.get('sharpe_ratio', 0):.3f}")
        lines.append(f"  Max drawdown   : ${b.get('max_drawdown', 0):>10,.2f}")
    else:
        lines.append("  No baseline data available.")

    # ── 2. TOP 10 ──
    has_wf = wf_results is not None and not wf_results.empty
    sort_label = "OOS PF" if has_wf else "IS PF"
    section(f"2. TOP 10 COMBINATIONS — sorted by {sort_label}")

    if grid_results is not None and not grid_results.empty:
        top10 = grid_results.nlargest(10, "profit_factor")
        for _, row in top10.iterrows():
            params_str = "  ".join(
                f"{c}={row.get(c, '?')}" for c in param_columns
            )
            lines.append(
                f"  {params_str}  PF={row.get('profit_factor', 0):.3f}  "
                f"WR={row.get('win_rate', 0):.1%}  "
                f"PnL=${row.get('total_pnl', 0):>9,.0f}  "
                f"trades={int(row.get('total_trades', 0))}"
            )
    else:
        lines.append("  No grid search results available.")

    # ── 3. SENSITIVITY ──
    if grid_results is not None and not grid_results.empty:
        section("3. SENSITIVITY ANALYSIS — Mean Profit Factor by Parameter")
        for param in param_columns:
            if param not in grid_results.columns:
                continue
            grouped = (
                grid_results.groupby(param)["profit_factor"]
                .mean()
                .reset_index()
            )
            lines.append(f"\n  {param}:")
            for _, row in grouped.iterrows():
                lines.append(
                    f"    {str(row[param]):<16}  PF={row['profit_factor']:.3f}"
                )

    # ── 4. WALK-FORWARD ──
    if has_wf:
        section("4. WALK-FORWARD — IS vs OOS")
        lines.append(
            f"  Windows         : {wf_results['window_id'].nunique()}"
        )
        lines.append(f"  Mean IS PF      : {wf_results['is_pf'].mean():.3f}")
        lines.append(
            f"  Mean OOS PF     : {wf_results['oos_pf'].mean():.3f}"
        )
        lines.append(
            f"  Total OOS PnL   : ${wf_results['oos_pnl'].sum():,.0f}"
        )
        oos_pos = (wf_results["oos_pnl"] > 0).sum()
        lines.append(
            f"  OOS positive    : {oos_pos}/{len(wf_results)} "
            f"({oos_pos / len(wf_results):.1%})"
        )

    # ── 5. OOS CONSISTENCY ──
    if has_wf and hasattr(wf_results, 'attrs') and wf_results.attrs.get("oos_windows_total"):
        section("5. OOS CONSISTENCY")
        a = wf_results.attrs
        lines.append(f"  Windows total   : {a['oos_windows_total']}")
        lines.append(f"  Windows profit. : {a['oos_windows_profitable']} ({a['oos_win_rate']:.1%})")
        lines.append(f"  OOS PF mean     : {a['oos_pf_mean']:.3f}")
        lines.append(f"  OOS PF std      : {a['oos_pf_std']:.3f}")
        lines.append(f"  OOS PF range    : {a['oos_pf_min']:.3f} — {a['oos_pf_max']:.3f}")

    # ── SUMMARY ──
    lines.extend(["", SEP])
    if grid_results is not None and not grid_results.empty:
        n_pf1 = int((grid_results["profit_factor"] > 1.0).sum())
        lines.append(
            f"  Combos with PF > 1.0: {n_pf1} / {len(grid_results)}"
        )
    lines.extend([SEP, ""])

    return "\n".join(lines)
