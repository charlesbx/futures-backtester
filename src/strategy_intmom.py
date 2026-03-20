"""Intraday Momentum Strategy.

Based on Gao, Han, Li, Zhou (JFE 2018):
The first 30 minutes' return predicts the last 30 minutes' return.

Provides signal detection, trade simulation, grid search,
walk-forward analysis, and report generation.
"""

from dataclasses import dataclass
from datetime import time
import itertools
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from .data_loader import INSTRUMENTS, load_processed


@dataclass
class Signal:
    """A daily trading signal from morning return."""

    date: pd.Timestamp
    direction: str          # 'long' or 'short'
    signal_return: float    # morning return that generated the signal
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    instrument: str


@dataclass
class Trade:
    """An executed trade, compatible with metrics.calculate_metrics()."""

    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    pnl_ticks: float
    pnl_dollars: float
    exit_reason: str        # 'tp', 'sl', 'session_end'
    session: int            # always 0 for intmom
    instrument: str
    date: pd.Timestamp
    range_high: float       # unused, set to 0.0
    range_low: float        # unused, set to 0.0


class IntradayMomentumStrategy:
    """Intraday Momentum strategy.

    Args:
        signal_start: Start of signal window (default 9:30 AM ET)
        signal_end: End of signal window (default 10:00 AM ET)
        entry_time: Trade entry time (default 3:30 PM ET)
        exit_time: Trade exit / session end (default 4:00 PM ET)
        min_signal_pct: Minimum abs(return) to generate signal (default 0.0)
        stop_loss_ticks: Stop-loss in ticks from entry (None = disabled)
        take_profit_ticks: Take-profit in ticks from entry (None = disabled)
    """

    def __init__(
        self,
        signal_start: time = time(9, 30),
        signal_end: time = time(10, 0),
        entry_time: time = time(15, 30),
        exit_time: time = time(16, 0),
        min_signal_pct: float = 0.0,
        stop_loss_ticks: int | None = None,
        take_profit_ticks: int | None = None,
    ):
        self.signal_start = signal_start
        self.signal_end = signal_end
        self.entry_time = entry_time
        self.exit_time = exit_time
        self.min_signal_pct = min_signal_pct
        self.stop_loss_ticks = stop_loss_ticks
        self.take_profit_ticks = take_profit_ticks

    def find_signals(self, df: pd.DataFrame, instrument: str) -> list[Signal]:
        """Find daily trading signals from morning returns.

        Args:
            df: OHLCV 1-min DataFrame with Eastern Time index
            instrument: 'MES' or 'MNQ'

        Returns:
            List of Signal objects, one per trading day with valid signal
        """
        signals = []
        tz = df.index.tz

        # Pre-compute time-of-day for each bar once (fast, avoids O(n²) scans)
        bar_times = df.index.time  # array of datetime.time objects

        sig_start = self.signal_start
        sig_end = self.signal_end
        entry_t = self.entry_time
        exit_t = self.exit_time

        expected_bars = (sig_end.hour * 60 + sig_end.minute
                         - sig_start.hour * 60 - sig_start.minute)

        # Group by date using normalize (single pass, O(n))
        date_col = df.index.normalize()

        for date, day_data in df.groupby(date_col):
            if day_data.empty:
                continue

            day_times = day_data.index.time

            # Signal window mask
            sig_mask = (day_times >= sig_start) & (day_times < sig_end)
            signal_window = day_data.loc[sig_mask]

            if len(signal_window) < expected_bars * 0.8:
                continue

            # Entry window mask
            entry_mask = (day_times >= entry_t) & (day_times < exit_t)
            entry_bars = day_data.loc[entry_mask]
            if len(entry_bars) < 2:
                continue

            # Morning return
            open_price = signal_window.iloc[0]["open"]
            close_price = signal_window.iloc[-1]["close"]
            if open_price == 0:
                continue
            morning_return = (close_price - open_price) / open_price

            if abs(morning_return) <= self.min_signal_pct:
                continue

            direction = "long" if morning_return > 0 else "short"

            # Build timezone-aware entry/exit timestamps
            date_naive = pd.Timestamp(date.date())
            entry_ts = date_naive + pd.Timedelta(hours=entry_t.hour, minutes=entry_t.minute)
            exit_ts = date_naive + pd.Timedelta(hours=exit_t.hour, minutes=exit_t.minute)
            if tz is not None:
                entry_ts = entry_ts.tz_localize(tz)
                exit_ts = exit_ts.tz_localize(tz)

            signals.append(Signal(
                date=date,
                direction=direction,
                signal_return=morning_return,
                entry_time=entry_ts,
                exit_time=exit_ts,
                instrument=instrument,
            ))

        return signals


def simulate_trade(
    df: pd.DataFrame,
    signal: Signal,
    strategy: IntradayMomentumStrategy,
    slippage_ticks: float = 1,
    commission: float = 1.24,
) -> Trade | None:
    """Simulate a single trade bar-by-bar.

    Args:
        df: OHLCV 1-min DataFrame (Eastern Time index)
        signal: Signal with entry/exit times and direction
        strategy: Strategy instance (for SL/TP parameters)
        slippage_ticks: Slippage in ticks per trade
        commission: Round-trip commission in dollars

    Returns:
        Trade object or None if entry bar not found
    """
    tick_size = INSTRUMENTS[signal.instrument]["tick_size"]
    tick_value = INSTRUMENTS[signal.instrument]["tick_value"]

    # Get bars from entry to exit
    trade_bars = df[(df.index >= signal.entry_time) & (df.index < signal.exit_time)]
    if trade_bars.empty:
        return None

    # Entry price with slippage
    entry_bar = trade_bars.iloc[0]
    if signal.direction == "long":
        entry_price = entry_bar["open"] + slippage_ticks * tick_size
    else:
        entry_price = entry_bar["open"] - slippage_ticks * tick_size

    # Compute SL/TP levels
    sl_price = None
    tp_price = None
    if strategy.stop_loss_ticks is not None:
        if signal.direction == "long":
            sl_price = entry_price - strategy.stop_loss_ticks * tick_size
        else:
            sl_price = entry_price + strategy.stop_loss_ticks * tick_size

    if strategy.take_profit_ticks is not None:
        if signal.direction == "long":
            tp_price = entry_price + strategy.take_profit_ticks * tick_size
        else:
            tp_price = entry_price - strategy.take_profit_ticks * tick_size

    # Walk bar-by-bar (skip entry bar to avoid same-bar SL/TP exit)
    exit_price = None
    exit_time = None
    exit_reason = "session_end"

    for _, bar in trade_bars.iloc[1:].iterrows():
        if signal.direction == "long":
            # Check SL (hit on low)
            if sl_price is not None and bar["low"] <= sl_price:
                exit_price = sl_price
                exit_time = bar.name
                exit_reason = "sl"
                break
            # Check TP (hit on high)
            if tp_price is not None and bar["high"] >= tp_price:
                exit_price = tp_price
                exit_time = bar.name
                exit_reason = "tp"
                break
        else:
            # Check SL (hit on high)
            if sl_price is not None and bar["high"] >= sl_price:
                exit_price = sl_price
                exit_time = bar.name
                exit_reason = "sl"
                break
            # Check TP (hit on low)
            if tp_price is not None and bar["low"] <= tp_price:
                exit_price = tp_price
                exit_time = bar.name
                exit_reason = "tp"
                break

    # Session end exit
    if exit_price is None:
        last_bar = trade_bars.iloc[-1]
        exit_price = last_bar["close"]
        exit_time = last_bar.name

    # PnL calculation
    if signal.direction == "long":
        pnl_ticks = (exit_price - entry_price) / tick_size
    else:
        pnl_ticks = (entry_price - exit_price) / tick_size

    pnl_dollars = pnl_ticks * tick_value - commission

    return Trade(
        entry_time=trade_bars.iloc[0].name,
        exit_time=exit_time,
        direction=signal.direction,
        entry_price=entry_price,
        exit_price=exit_price,
        stop_loss=sl_price if sl_price is not None else 0.0,
        take_profit=tp_price if tp_price is not None else 0.0,
        pnl_ticks=pnl_ticks,
        pnl_dollars=pnl_dollars,
        exit_reason=exit_reason,
        session=0,
        instrument=signal.instrument,
        date=signal.date,
        range_high=0.0,
        range_low=0.0,
    )


def run_backtest(
    data: dict[str, pd.DataFrame],
    strategy: IntradayMomentumStrategy,
    slippage_ticks: float = 1,
    commission: float = 1.24,
) -> list[Trade]:
    """Run backtest across all instruments.

    Args:
        data: Dict of instrument -> OHLCV DataFrame
        strategy: IntradayMomentumStrategy instance
        slippage_ticks: Slippage in ticks
        commission: Round-trip commission in dollars

    Returns:
        List of all executed trades
    """
    all_trades = []
    for instrument, df in data.items():
        signals = strategy.find_signals(df, instrument)
        for signal in signals:
            trade = simulate_trade(df, signal, strategy, slippage_ticks, commission)
            if trade is not None:
                all_trades.append(trade)
    return all_trades


def build_param_grid() -> list[dict]:
    """Generate 1,024 parameter combinations.

    signal_end: [10:00, 10:15, 10:30, 11:00]  (4)
    entry_time: [15:00, 15:15, 15:30, 15:45]  (4)
    min_signal_pct: [0.0, 0.05, 0.10, 0.20]   (4)
    stop_loss_ticks: [None, 20, 40, 60]        (4)
    take_profit_ticks: [None, 20, 40, 60]      (4)
    Total: 4^5 = 1024
    """
    signal_ends = [time(10, 0), time(10, 15), time(10, 30), time(11, 0)]
    entry_times = [time(15, 0), time(15, 15), time(15, 30), time(15, 45)]
    min_signals = [0.0, 0.05, 0.10, 0.20]
    sl_ticks = [None, 20, 40, 60]
    tp_ticks = [None, 20, 40, 60]

    grid = []
    for se, et, ms, sl, tp in itertools.product(
        signal_ends, entry_times, min_signals, sl_ticks, tp_ticks
    ):
        grid.append({
            "signal_end": se,
            "entry_time": et,
            "min_signal_pct": ms,
            "stop_loss_ticks": sl,
            "take_profit_ticks": tp,
        })
    return grid


def _run_single_config(
    data: dict[str, pd.DataFrame],
    params: dict,
) -> tuple[dict, list[Trade]]:
    """Run a single parameter configuration and return metrics + trades."""
    from .metrics import calculate_metrics

    strategy = IntradayMomentumStrategy(
        signal_end=params["signal_end"],
        entry_time=params["entry_time"],
        min_signal_pct=params["min_signal_pct"],
        stop_loss_ticks=params["stop_loss_ticks"],
        take_profit_ticks=params["take_profit_ticks"],
    )
    trades = run_backtest(data, strategy)
    metrics = calculate_metrics(trades)
    return metrics, trades


def run_grid_search(
    data: dict[str, pd.DataFrame],
    param_grid: list[dict],
    progress: bool = False,
) -> pd.DataFrame:
    """Run grid search over parameter combinations.

    Returns DataFrame with one row per combination.
    """
    results = []
    iterator = tqdm(param_grid, desc="Grid search") if progress else param_grid

    for params in iterator:
        metrics, trades = _run_single_config(data, params)

        results.append({
            "signal_end": params["signal_end"].strftime("%H:%M"),
            "entry_time": params["entry_time"].strftime("%H:%M"),
            "min_signal_pct": params["min_signal_pct"],
            "stop_loss_ticks": params["stop_loss_ticks"],
            "take_profit_ticks": params["take_profit_ticks"],
            "total_trades": metrics.get("total_trades", 0),
            "win_rate": metrics.get("win_rate", 0),
            "profit_factor": metrics.get("profit_factor", 0),
            "total_pnl": metrics.get("total_pnl", 0),
            "sharpe_ratio": metrics.get("sharpe_ratio", 0),
            "max_drawdown": metrics.get("max_drawdown", 0),
            "avg_pnl_per_trade": metrics.get("avg_pnl_per_trade", 0),
            "pnl_long": metrics.get("pnl_long", 0),
            "pnl_short": metrics.get("pnl_short", 0),
        })

    return pd.DataFrame(results)


def _filter_data_by_date(
    data: dict[str, pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    """Filter all instrument DataFrames to a date range."""
    return {
        inst: df[(df.index >= start) & (df.index < end)]
        for inst, df in data.items()
    }


def run_walk_forward(
    data: dict[str, pd.DataFrame],
    param_grid: list[dict],
    train_months: int = 24,
    test_months: int = 6,
    top_n: int = 10,
) -> pd.DataFrame:
    """Rolling walk-forward analysis.

    For each window:
    1. Train: run grid search, select top_n by profit_factor
    2. Test: evaluate top_n out-of-sample
    3. Record IS and OOS metrics
    """
    tz = next(iter(data.values())).index.tz
    global_start = max(df.index.min().date() for df in data.values())
    global_end = min(df.index.max().date() for df in data.values())

    # Build rolling windows
    windows = []
    current = global_start
    while True:
        train_start = pd.Timestamp(current, tz=tz)
        train_end = train_start + pd.DateOffset(months=train_months)
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=test_months)

        if test_end.date() > global_end:
            break

        windows.append((train_start, train_end, test_start, test_end))
        # Step by test_months (not train_months) to produce overlapping train
        # windows and non-overlapping test windows, matching SRS optimizer pattern
        current = (train_start + pd.DateOffset(months=test_months)).date()

    print(f"  {len(windows)} walk-forward windows")
    wf_results = []

    for wid, (train_start, train_end, test_start, test_end) in enumerate(windows):
        print(f"\n  Window {wid}: train {train_start.date()}→{train_end.date()}, "
              f"test {test_start.date()}→{test_end.date()}")

        train_data = _filter_data_by_date(data, train_start, train_end)
        test_data = _filter_data_by_date(data, test_start, test_end)

        # In-sample grid search
        is_results = run_grid_search(train_data, param_grid)
        if is_results.empty:
            continue

        # Select top N by profit factor
        top_params = is_results.nlargest(top_n, "profit_factor")

        # Evaluate OOS
        for _, row in top_params.iterrows():
            params = {
                "signal_end": time(*map(int, row["signal_end"].split(":"))),
                "entry_time": time(*map(int, row["entry_time"].split(":"))),
                "min_signal_pct": row["min_signal_pct"],
                "stop_loss_ticks": None if pd.isna(row["stop_loss_ticks"]) else int(row["stop_loss_ticks"]),
                "take_profit_ticks": None if pd.isna(row["take_profit_ticks"]) else int(row["take_profit_ticks"]),
            }

            oos_metrics, _ = _run_single_config(test_data, params)

            wf_results.append({
                "window_id": wid,
                "train_start": train_start.date(),
                "train_end": train_end.date(),
                "test_start": test_start.date(),
                "test_end": test_end.date(),
                "params": json.dumps({k: str(v) for k, v in params.items()}),
                "is_pnl": row["total_pnl"],
                "oos_pnl": oos_metrics.get("total_pnl", 0),
                "is_sharpe": row["sharpe_ratio"],
                "oos_sharpe": oos_metrics.get("sharpe_ratio", 0),
                "is_pf": row["profit_factor"],
                "oos_pf": oos_metrics.get("profit_factor", 0),
            })

    return pd.DataFrame(wf_results)


def generate_report(
    grid_results: pd.DataFrame | None,
    wf_results: pd.DataFrame | None,
    baseline: dict | None,
) -> str:
    """Generate comprehensive text report from saved results."""
    lines = []
    lines.append("=" * 70)
    lines.append("  INTRADAY MOMENTUM STRATEGY — REPORT")
    lines.append("=" * 70)

    # Baseline
    if baseline:
        lines.append("\n--- BASELINE (Paper Parameters) ---")
        for key, val in baseline.items():
            if isinstance(val, float):
                lines.append(f"  {key:<25} {val:>15.4f}")
            else:
                lines.append(f"  {key:<25} {val:>15}")

    # Grid search
    if grid_results is not None and not grid_results.empty:
        lines.append(f"\n--- GRID SEARCH ({len(grid_results)} combinations) ---")

        # Summary stats
        profitable = grid_results[grid_results["profit_factor"] > 1.0]
        lines.append(f"  Profitable (PF > 1.0): {len(profitable)} ({len(profitable)/len(grid_results):.1%})")
        lines.append(f"  Mean PF: {grid_results['profit_factor'].mean():.3f}")
        lines.append(f"  Max PF: {grid_results['profit_factor'].max():.3f}")
        lines.append(f"  Mean PnL: ${grid_results['total_pnl'].mean():,.0f}")

        # Top 10
        lines.append("\n  Top 10 by Total PnL:")
        top = grid_results.nlargest(10, "total_pnl")
        header = f"  {'sig_end':>7} {'entry':>7} {'min_sig':>7} {'SL':>5} {'TP':>5} {'trades':>6} {'WR':>6} {'PF':>6} {'PnL':>10} {'Sharpe':>7}"
        lines.append(header)
        lines.append("  " + "-" * 72)
        for _, r in top.iterrows():
            sl_str = str(int(r['stop_loss_ticks'])) if pd.notna(r['stop_loss_ticks']) else "None"
            tp_str = str(int(r['take_profit_ticks'])) if pd.notna(r['take_profit_ticks']) else "None"
            lines.append(
                f"  {r['signal_end']:>7} {r['entry_time']:>7} {r['min_signal_pct']:>7.2f} "
                f"{sl_str:>5} {tp_str:>5} {r['total_trades']:>6.0f} {r['win_rate']:>5.1%} "
                f"{r['profit_factor']:>6.3f} ${r['total_pnl']:>9,.0f} {r['sharpe_ratio']:>7.3f}"
            )

        # Parameter sensitivity
        lines.append("\n  Parameter Sensitivity (mean PF):")
        for param in ["signal_end", "entry_time", "min_signal_pct", "stop_loss_ticks", "take_profit_ticks"]:
            lines.append(f"\n  By {param}:")
            grouped = grid_results.groupby(param).agg(
                mean_pf=("profit_factor", "mean"),
                mean_pnl=("total_pnl", "mean"),
            )
            for val, row in grouped.iterrows():
                val_str = str(val) if pd.notna(val) else "None"
                lines.append(f"    {val_str:<10} PF={row['mean_pf']:.3f}  PnL=${row['mean_pnl']:>8,.0f}")

    # Walk-forward
    if wf_results is not None and not wf_results.empty:
        lines.append(f"\n--- WALK-FORWARD ({wf_results['window_id'].nunique()} windows) ---")
        lines.append(f"  Total OOS PnL: ${wf_results['oos_pnl'].sum():,.0f}")
        lines.append(f"  Mean OOS PF: {wf_results['oos_pf'].mean():.3f}")
        lines.append(f"  Mean IS PF: {wf_results['is_pf'].mean():.3f}")
        oos_positive = (wf_results['oos_pnl'] > 0).sum()
        lines.append(f"  OOS positive: {oos_positive}/{len(wf_results)} ({oos_positive/len(wf_results):.1%})")
        lines.append(f"  IS→OOS degradation: {1 - wf_results['oos_pf'].mean() / max(wf_results['is_pf'].mean(), 1e-9):.1%}")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)
