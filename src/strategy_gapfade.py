"""Overnight Gap Fade Strategy.

Gaps between the previous session close and the current session open
tend to fill. Fade the gap (buy gap-downs, sell gap-ups), targeting
partial fill with an optional stop based on gap widening.

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
class GapSignal:
    """A daily gap fade signal."""

    date: pd.Timestamp
    direction: str          # 'long' or 'short'
    gap_pct: float          # (measure_price - prev_close) / prev_close
    prev_close: float       # previous trading day's 4:00 PM close
    measure_price: float    # price at gap_measure_time
    entry_time: pd.Timestamp
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
    session: int            # always 0 for gap fade
    instrument: str
    date: pd.Timestamp
    range_high: float       # unused, set to 0.0
    range_low: float        # unused, set to 0.0


class GapFadeStrategy:
    """Overnight Gap Fade strategy.

    Args:
        gap_measure_time: Time at which to measure the gap (default 9:30 AM ET)
        min_gap_pct: Minimum abs(gap_pct) to generate a signal (default 0.001 = 0.1%)
        fill_pct: Target fill fraction: 0.75 means target 75% of the way back to prev_close
        stop_gap_multiple: If set, stop = entry +/- stop_gap_multiple * abs(gap_size)
    """

    def __init__(
        self,
        gap_measure_time: time = time(9, 30),
        min_gap_pct: float = 0.001,
        fill_pct: float = 0.75,
        stop_gap_multiple: float | None = None,
    ):
        self.gap_measure_time = gap_measure_time
        self.min_gap_pct = min_gap_pct
        self.fill_pct = fill_pct
        self.stop_gap_multiple = stop_gap_multiple

    def find_signals(self, df: pd.DataFrame, instrument: str) -> list[GapSignal]:
        """Find daily gap fade signals.

        Args:
            df: OHLCV 1-min DataFrame with Eastern Time index
            instrument: 'MES' or 'MNQ'

        Returns:
            List of GapSignal objects, one per trading day with valid gap
        """
        signals = []
        tz = df.index.tz
        measure_t = self.gap_measure_time
        close_t = time(16, 0)  # 4:00 PM ET

        # Group by date once
        date_col = df.index.normalize()
        day_groups = {date: day_data for date, day_data in df.groupby(date_col)}
        sorted_dates = sorted(day_groups.keys())

        prev_close = None
        prev_date = None

        for date in sorted_dates:
            day_data = day_groups[date]
            day_times = day_data.index.time

            # Get 4:00 PM close for this day (for use as prev_close next day)
            close_mask = day_times == close_t
            close_bars = day_data.loc[close_mask]
            this_close = close_bars.iloc[-1]["close"] if not close_bars.empty else None

            if prev_close is not None:
                # Get price at gap_measure_time
                measure_mask = day_times == measure_t
                measure_bars = day_data.loc[measure_mask]

                if not measure_bars.empty:
                    measure_price = measure_bars.iloc[0]["open"]
                    if prev_close != 0:
                        gap_pct = (measure_price - prev_close) / prev_close

                        if abs(gap_pct) > self.min_gap_pct:
                            direction = "long" if gap_pct < 0 else "short"

                            date_naive = pd.Timestamp(date.date()) if hasattr(date, 'date') else pd.Timestamp(date)
                            entry_ts = date_naive + pd.Timedelta(
                                hours=measure_t.hour, minutes=measure_t.minute
                            )
                            if tz is not None:
                                entry_ts = entry_ts.tz_localize(tz)

                            signals.append(GapSignal(
                                date=date,
                                direction=direction,
                                gap_pct=gap_pct,
                                prev_close=prev_close,
                                measure_price=measure_price,
                                entry_time=entry_ts,
                                instrument=instrument,
                            ))

            # Update prev_close for next iteration
            if this_close is not None:
                prev_close = this_close
                prev_date = date

        return signals


def simulate_trade(
    signal: GapSignal,
    day_data: pd.DataFrame,
    strategy: GapFadeStrategy,
    tick_size: float,
    tick_value: float,
    slippage_ticks: float = 1,
    commission: float = 1.24,
) -> Trade | None:
    """Simulate a single gap fade trade bar-by-bar.

    Args:
        signal: GapSignal with direction, prev_close, measure_price, entry_time
        day_data: Pre-sliced day OHLCV DataFrame (Eastern Time)
        strategy: GapFadeStrategy instance (for fill_pct and stop_gap_multiple)
        tick_size: Instrument tick size
        tick_value: Instrument tick value per tick
        slippage_ticks: Slippage in ticks per side
        commission: Round-trip commission in dollars

    Returns:
        Trade object or None if entry bar not found
    """
    measure_t = signal.entry_time.time() if hasattr(signal.entry_time, 'time') else signal.entry_time
    session_end_t = time(16, 0)

    day_times = day_data.index.time
    # All bars from measure_time to 4:00 PM
    trade_bars = day_data.loc[(day_times >= measure_t) & (day_times < session_end_t)]
    if trade_bars.empty:
        return None

    # Entry at gap_measure_time bar open + slippage
    entry_bar = trade_bars.iloc[0]
    if signal.direction == "long":
        entry_price = entry_bar["open"] + slippage_ticks * tick_size
    else:
        entry_price = entry_bar["open"] - slippage_ticks * tick_size

    # Fill target: linear interpolation between measure_price and prev_close
    # fill_target = measure_price + fill_pct * (prev_close - measure_price)
    fill_target = signal.measure_price + strategy.fill_pct * (signal.prev_close - signal.measure_price)

    # Stop: entry +/- stop_gap_multiple * abs(gap_size)
    gap_size = signal.measure_price - signal.prev_close  # negative for gap-down, positive for gap-up
    sl_price = None
    if strategy.stop_gap_multiple is not None:
        if signal.direction == "long":
            sl_price = entry_price - strategy.stop_gap_multiple * abs(gap_size)
        else:
            sl_price = entry_price + strategy.stop_gap_multiple * abs(gap_size)

    # Walk bar-by-bar (skip entry bar)
    exit_price = None
    exit_time = None
    exit_reason = "session_end"

    for _, bar in trade_bars.iloc[1:].iterrows():
        if signal.direction == "long":
            # Check stop first (hit on low)
            if sl_price is not None and bar["low"] <= sl_price:
                exit_price = sl_price
                exit_time = bar.name
                exit_reason = "sl"
                break
            # Check fill target (hit on high)
            if bar["high"] >= fill_target:
                exit_price = fill_target
                exit_time = bar.name
                exit_reason = "tp"
                break
        else:
            # Check stop first (hit on high)
            if sl_price is not None and bar["high"] >= sl_price:
                exit_price = sl_price
                exit_time = bar.name
                exit_reason = "sl"
                break
            # Check fill target (hit on low)
            if bar["low"] <= fill_target:
                exit_price = fill_target
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
        take_profit=fill_target,
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
    strategy: GapFadeStrategy,
    slippage_ticks: float = 1,
    commission: float = 1.24,
) -> list[Trade]:
    """Run backtest across all instruments.

    Args:
        data: Dict of instrument -> OHLCV DataFrame
        strategy: GapFadeStrategy instance
        slippage_ticks: Slippage in ticks
        commission: Round-trip commission in dollars

    Returns:
        List of all executed trades
    """
    all_trades = []
    for instrument, df in data.items():
        tick_size = INSTRUMENTS[instrument]["tick_size"]
        tick_value = INSTRUMENTS[instrument]["tick_value"]
        date_col = df.index.normalize()
        day_groups = {date: gdf for date, gdf in df.groupby(date_col)}

        signals = strategy.find_signals(df, instrument)
        for signal in signals:
            day_data = day_groups.get(signal.date)
            if day_data is None:
                continue
            trade = simulate_trade(
                signal, day_data, strategy, tick_size, tick_value,
                slippage_ticks, commission
            )
            if trade is not None:
                all_trades.append(trade)
    return all_trades


def build_param_grid() -> list[dict]:
    """Generate 180 parameter combinations.

    gap_measure_time: [9:15, 9:30, 9:31]                     (3)
    min_gap_pct:      [0.0005, 0.001, 0.0015, 0.0025, 0.005] (5)
    fill_pct:         [0.50, 0.75, 1.00]                      (3)
    stop_gap_multiple:[None, 0.25, 0.50, 1.00]                (4)
    Total: 3 × 5 × 3 × 4 = 180
    """
    measure_times = [time(9, 15), time(9, 30), time(9, 31)]
    min_gaps = [0.0005, 0.001, 0.0015, 0.0025, 0.005]
    fill_pcts = [0.50, 0.75, 1.00]
    stop_multiples = [None, 0.25, 0.50, 1.00]

    grid = []
    for mt, mg, fp, sm in itertools.product(measure_times, min_gaps, fill_pcts, stop_multiples):
        grid.append({
            "gap_measure_time": mt,
            "min_gap_pct": mg,
            "fill_pct": fp,
            "stop_gap_multiple": sm,
        })
    return grid


def _precompute_gap_data(
    data: dict[str, pd.DataFrame],
    measure_time_values: list[time],
) -> dict[tuple[str, str], pd.DataFrame]:
    """Precompute gap data for each (instrument, gap_measure_time).

    Returns dict keyed by (instrument, measure_time_str) -> DataFrame with
    columns: date, prev_close, measure_price, gap_pct, direction
    Only 6 precomputations needed (2 instruments × 3 measure times).
    """
    cache = {}
    close_t = time(16, 0)

    for instrument, df in data.items():
        date_col = df.index.normalize()
        day_groups = {date: gdf for date, gdf in df.groupby(date_col)}
        sorted_dates = sorted(day_groups.keys())

        for measure_t in measure_time_values:
            key = (instrument, measure_t.strftime("%H:%M"))
            rows = []
            prev_close = None

            for date in sorted_dates:
                day_data = day_groups[date]
                day_times = day_data.index.time

                # Get 4:00 PM close for this day
                close_mask = day_times == close_t
                close_bars = day_data.loc[close_mask]
                this_close = close_bars.iloc[-1]["close"] if not close_bars.empty else None

                if prev_close is not None:
                    measure_mask = day_times == measure_t
                    measure_bars = day_data.loc[measure_mask]

                    if not measure_bars.empty and prev_close != 0:
                        measure_price = measure_bars.iloc[0]["open"]
                        gap_pct = (measure_price - prev_close) / prev_close
                        rows.append({
                            "date": date,
                            "prev_close": prev_close,
                            "measure_price": measure_price,
                            "gap_pct": gap_pct,
                        })

                if this_close is not None:
                    prev_close = this_close

            cache[key] = (
                pd.DataFrame(rows)
                if rows
                else pd.DataFrame(columns=["date", "prev_close", "measure_price", "gap_pct"])
            )

    return cache


def run_grid_search(
    data: dict[str, pd.DataFrame],
    param_grid: list[dict],
    progress: bool = False,
) -> pd.DataFrame:
    """Run grid search over parameter combinations.

    Precomputes gap data for each (instrument, gap_measure_time) to avoid
    redundant signal scanning. Only 6 precomputations for 180 combos.
    """
    from .metrics import calculate_metrics

    # Extract unique gap_measure_time values for precomputation
    measure_time_values = sorted(
        set(p["gap_measure_time"] for p in param_grid),
        key=lambda t: (t.hour, t.minute),
    )
    print(f"  Precomputing gap data for {len(measure_time_values)} measure_time values × {len(data)} instruments...")
    gap_cache = _precompute_gap_data(data, measure_time_values)

    # Precompute day groups for fast trade simulation
    print(f"  Precomputing day groups...")
    day_groups = {}
    for instrument, df in data.items():
        day_groups[instrument] = {date: gdf for date, gdf in df.groupby(df.index.normalize())}

    # Precompute timezone per instrument
    tz_map = {instrument: df.index.tz for instrument, df in data.items()}

    print(f"  Done. Running {len(param_grid)} parameter combinations...")

    results = []
    iterator = tqdm(param_grid, desc="Grid search") if progress else param_grid

    for params in iterator:
        measure_t = params["gap_measure_time"]
        measure_t_str = measure_t.strftime("%H:%M")
        min_gap = params["min_gap_pct"]
        fill_pct = params["fill_pct"]
        stop_mult = params["stop_gap_multiple"]

        strategy = GapFadeStrategy(
            gap_measure_time=measure_t,
            min_gap_pct=min_gap,
            fill_pct=fill_pct,
            stop_gap_multiple=stop_mult,
        )

        all_trades = []
        for instrument, df in data.items():
            tz = tz_map[instrument]
            tick_size = INSTRUMENTS[instrument]["tick_size"]
            tick_value = INSTRUMENTS[instrument]["tick_value"]
            gaps_df = gap_cache.get((instrument, measure_t_str))
            if gaps_df is None or gaps_df.empty:
                continue

            for _, row in gaps_df.iterrows():
                gp = row["gap_pct"]
                if abs(gp) <= min_gap:
                    continue

                date = row["date"]
                direction = "long" if gp < 0 else "short"

                date_naive = pd.Timestamp(date.date()) if hasattr(date, 'date') else pd.Timestamp(date)
                entry_ts = date_naive + pd.Timedelta(hours=measure_t.hour, minutes=measure_t.minute)
                if tz is not None:
                    entry_ts = entry_ts.tz_localize(tz)

                signal = GapSignal(
                    date=date,
                    direction=direction,
                    gap_pct=gp,
                    prev_close=row["prev_close"],
                    measure_price=row["measure_price"],
                    entry_time=entry_ts,
                    instrument=instrument,
                )

                dd = day_groups[instrument].get(date)
                if dd is None:
                    continue
                trade = simulate_trade(signal, dd, strategy, tick_size, tick_value)
                if trade is not None:
                    all_trades.append(trade)

        metrics = calculate_metrics(all_trades)

        results.append({
            "gap_measure_time": measure_t_str,
            "min_gap_pct": min_gap,
            "fill_pct": fill_pct,
            "stop_gap_multiple": stop_mult,
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


def _run_single_config(
    data: dict[str, pd.DataFrame],
    params: dict,
) -> tuple[dict, list[Trade]]:
    """Run a single parameter configuration and return metrics + trades."""
    from .metrics import calculate_metrics

    strategy = GapFadeStrategy(
        gap_measure_time=params["gap_measure_time"],
        min_gap_pct=params["min_gap_pct"],
        fill_pct=params["fill_pct"],
        stop_gap_multiple=params["stop_gap_multiple"],
    )
    trades = run_backtest(data, strategy)
    metrics = calculate_metrics(trades)
    return metrics, trades


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
        # Step by test_months — non-overlapping test windows, overlapping train windows
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
            # Reconstruct params — convert string times back to time objects
            measure_t_str = row["gap_measure_time"]
            h, m = map(int, measure_t_str.split(":"))
            stop_mult = None if pd.isna(row["stop_gap_multiple"]) else float(row["stop_gap_multiple"])
            params = {
                "gap_measure_time": time(h, m),
                "min_gap_pct": row["min_gap_pct"],
                "fill_pct": row["fill_pct"],
                "stop_gap_multiple": stop_mult,
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
    lines.append("  OVERNIGHT GAP FADE STRATEGY — REPORT")
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

        profitable = grid_results[grid_results["profit_factor"] > 1.0]
        lines.append(f"  Profitable (PF > 1.0): {len(profitable)} ({len(profitable)/len(grid_results):.1%})")
        lines.append(f"  Mean PF: {grid_results['profit_factor'].mean():.3f}")
        lines.append(f"  Max PF: {grid_results['profit_factor'].max():.3f}")
        lines.append(f"  Mean PnL: ${grid_results['total_pnl'].mean():,.0f}")

        # Top 10
        lines.append("\n  Top 10 by Total PnL:")
        top = grid_results.nlargest(10, "total_pnl")
        header = (f"  {'meas_t':>7} {'min_gap':>8} {'fill':>6} {'stop':>6} "
                  f"{'trades':>6} {'WR':>6} {'PF':>6} {'PnL':>10} {'Sharpe':>7}")
        lines.append(header)
        lines.append("  " + "-" * 68)
        for _, r in top.iterrows():
            stop_str = f"{r['stop_gap_multiple']:.2f}" if pd.notna(r['stop_gap_multiple']) else "None"
            lines.append(
                f"  {r['gap_measure_time']:>7} {r['min_gap_pct']:>8.4f} {r['fill_pct']:>6.2f} "
                f"{stop_str:>6} {r['total_trades']:>6.0f} {r['win_rate']:>5.1%} "
                f"{r['profit_factor']:>6.3f} ${r['total_pnl']:>9,.0f} {r['sharpe_ratio']:>7.3f}"
            )

        # Parameter sensitivity
        lines.append("\n  Parameter Sensitivity (mean PF):")
        for param in ["gap_measure_time", "min_gap_pct", "fill_pct", "stop_gap_multiple"]:
            lines.append(f"\n  By {param}:")
            grouped = grid_results.groupby(param, dropna=False).agg(
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
        lines.append(
            f"  IS→OOS degradation: "
            f"{1 - wf_results['oos_pf'].mean() / max(wf_results['is_pf'].mean(), 1e-9):.1%}"
        )

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)
