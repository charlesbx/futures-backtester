"""Asian Range Breakout / Fade Strategy.

The Asian session (6:00 PM to ~2:00 AM ET) establishes a range that
US-session traders react to. Two modes:

- **breakout**: trade in the direction of the breakout above/below the
  Asian range during US hours.
- **fade**: fade the breakout, expecting price to return inside the range.

Provides Asian range detection, signal generation, trade simulation,
grid search, walk-forward analysis, and report generation.
"""

import itertools
import json
from dataclasses import dataclass
from datetime import time

import pandas as pd
from tqdm import tqdm

from .data_loader import INSTRUMENTS, load_processed

# Asian session always starts at 6:00 PM ET (previous calendar day)
ASIAN_START = time(18, 0)

# Minimum number of 1-min bars required to form a valid range
MIN_RANGE_BARS = 5


@dataclass
class AsianRange:
    """Overnight Asian session range for a single trading date."""

    date: pd.Timestamp
    asian_high: float
    asian_low: float
    asian_start: pd.Timestamp
    asian_end: pd.Timestamp
    range_ticks: float
    instrument: str


@dataclass
class AsianSignal:
    """A breakout or fade signal off the Asian range."""

    date: pd.Timestamp
    direction: str          # 'long' or 'short'
    mode: str               # 'breakout' or 'fade'
    entry_price: float
    entry_time: pd.Timestamp
    asian_range: AsianRange
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
    session: int            # always 0 for Asian range strategy
    instrument: str
    date: pd.Timestamp
    range_high: float
    range_low: float


class AsianRangeStrategy:
    """Asian Range Breakout / Fade strategy.

    Args:
        mode: 'breakout' or 'fade'
        asian_end: Time marking the end of the Asian range window (default 2:00 AM ET)
        trade_start: Earliest time to enter a trade (default 9:30 AM ET)
        trade_end: Latest time; force-exit at this time (default 4:00 PM ET)
        min_range_ticks: Minimum range size in ticks to consider (default 10)
        max_range_ticks: Maximum range size in ticks (default None = no limit)
        stop_type: 'opposite' (opposite side of range, breakout only) or 'multiple' (stop_multiple * range)
        tp_type: 'rr' (risk/reward), 'opposite' (measured move / boundary), or 'midpoint'
        rr_ratio: Risk/reward ratio for take-profit (default 2.0)
        stop_multiple: Multiplier for stop distance when stop_type='multiple'
    """

    # Asian session always starts at 6:00 PM ET
    ASIAN_START = time(18, 0)

    def __init__(
        self,
        mode: str = "breakout",
        asian_end: time = time(2, 0),
        trade_start: time = time(9, 30),
        trade_end: time = time(16, 0),
        min_range_ticks: int = 10,
        max_range_ticks: int | None = None,
        stop_type: str = "opposite",
        tp_type: str = "rr",
        rr_ratio: float = 2.0,
        stop_multiple: float = 1.0,
    ):
        self.mode = mode
        self.asian_end = asian_end
        self.trade_start = trade_start
        self.trade_end = trade_end
        self.min_range_ticks = min_range_ticks
        self.max_range_ticks = max_range_ticks
        self.stop_type = stop_type
        self.tp_type = tp_type
        self.rr_ratio = rr_ratio
        self.stop_multiple = stop_multiple

    # ------------------------------------------------------------------
    # SL / TP computation
    # ------------------------------------------------------------------

    def compute_sl_tp(
        self,
        signal: "AsianSignal",
        entry_price: float,
    ) -> tuple[float, float]:
        """Compute stop-loss and take-profit for a signal.

        Args:
            signal: The entry signal (contains direction, mode, asian_range)
            entry_price: Actual entry price (after slippage)

        Returns:
            (stop_loss, take_profit) tuple
        """
        ar = signal.asian_range
        range_size = ar.asian_high - ar.asian_low

        # --- Stop-loss ---
        if self.stop_type == "opposite":
            # Opposite side of the Asian range
            if signal.direction == "long":
                sl = ar.asian_low
            else:
                sl = ar.asian_high
        elif self.stop_type == "multiple":
            if signal.direction == "long":
                sl = entry_price - self.stop_multiple * range_size
            else:
                sl = entry_price + self.stop_multiple * range_size
        else:
            raise ValueError(f"Unknown stop_type: {self.stop_type!r}")

        # --- Take-profit ---
        risk_distance = abs(entry_price - sl)

        if self.tp_type == "rr":
            if signal.direction == "long":
                tp = entry_price + self.rr_ratio * risk_distance
            else:
                tp = entry_price - self.rr_ratio * risk_distance
        elif self.tp_type == "opposite":
            if signal.mode == "breakout":
                # Measured move: extend by one range_size beyond the boundary
                if signal.direction == "long":
                    tp = ar.asian_high + range_size
                else:
                    tp = ar.asian_low - range_size
            else:
                # Fade: target the opposite boundary
                if signal.direction == "long":
                    tp = ar.asian_high
                else:
                    tp = ar.asian_low
        elif self.tp_type == "midpoint":
            tp = (ar.asian_high + ar.asian_low) / 2
        else:
            raise ValueError(f"Unknown tp_type: {self.tp_type!r}")

        return sl, tp

    # ------------------------------------------------------------------
    # Range detection
    # ------------------------------------------------------------------

    def find_asian_ranges(
        self,
        df: pd.DataFrame,
        instrument: str,
        tick_size: float,
    ) -> list[AsianRange]:
        """Find Asian session ranges in the data.

        CRITICAL: The Asian range spans midnight. For trading date D,
        the window runs from D-1 18:00 ET to D asian_end ET.

        Uses absolute timestamps (not day groups / normalize()) because
        the range crosses the calendar-day boundary.

        Args:
            df: OHLCV 1-min DataFrame with Eastern Time DatetimeIndex
            instrument: 'MES' or 'MNQ'
            tick_size: Instrument tick size (e.g. 0.25)

        Returns:
            List of AsianRange objects, one per valid trading date
        """
        if df.empty:
            return []

        tz = df.index.tz
        ranges: list[AsianRange] = []

        # Get unique calendar dates present in the data
        dates = sorted(pd.Series(df.index.date).unique())

        for date in dates:
            # For trading date D, Asian window is D-1 18:00 to D asian_end
            prev_day = pd.Timestamp(date) - pd.Timedelta(days=1)
            window_start = pd.Timestamp(
                year=prev_day.year, month=prev_day.month, day=prev_day.day,
                hour=self.ASIAN_START.hour, minute=self.ASIAN_START.minute,
            )
            window_end = pd.Timestamp(
                year=date.year, month=date.month, day=date.day,
                hour=self.asian_end.hour, minute=self.asian_end.minute,
            )

            # Localize to match the DataFrame timezone
            # Use nonexistent='shift_forward' for DST spring-forward gaps
            # (e.g. 2:00 AM doesn't exist on spring-forward dates)
            if tz is not None:
                window_start = window_start.tz_localize(
                    tz, nonexistent="shift_forward",
                )
                window_end = window_end.tz_localize(
                    tz, nonexistent="shift_forward",
                )

            # Filter bars within the Asian window
            mask = (df.index >= window_start) & (df.index < window_end)
            asian_bars = df.loc[mask]

            # Validate: enough bars
            if len(asian_bars) < MIN_RANGE_BARS:
                continue

            asian_high = asian_bars["high"].max()
            asian_low = asian_bars["low"].min()

            # Validate: range is not zero
            if asian_high <= asian_low:
                continue

            range_ticks = round((asian_high - asian_low) / tick_size)

            # Validate: range size filters
            if range_ticks < self.min_range_ticks:
                continue
            if self.max_range_ticks is not None and range_ticks > self.max_range_ticks:
                continue

            ranges.append(AsianRange(
                date=pd.Timestamp(date),
                asian_high=asian_high,
                asian_low=asian_low,
                asian_start=window_start,
                asian_end=window_end,
                range_ticks=range_ticks,
                instrument=instrument,
            ))

        return ranges

    # ------------------------------------------------------------------
    # Signal detection
    # ------------------------------------------------------------------

    def find_signals(
        self,
        df: pd.DataFrame,
        ranges: list[AsianRange],
    ) -> list[AsianSignal]:
        """Find breakout or fade signals for each Asian range.

        For each range, builds a trade window from ``self.trade_start`` to
        ``self.trade_end`` on the range's date, then dispatches to the
        appropriate detector based on ``self.mode``.

        Args:
            df: OHLCV 1-min DataFrame with Eastern Time DatetimeIndex
            ranges: List of AsianRange objects (from find_asian_ranges)

        Returns:
            List of AsianSignal objects (at most one per day)
        """
        signals: list[AsianSignal] = []
        tz = df.index.tz

        for ar in ranges:
            # Build trade window timestamps on the range's date
            d = ar.date
            window_start = pd.Timestamp(
                year=d.year, month=d.month, day=d.day,
                hour=self.trade_start.hour, minute=self.trade_start.minute,
            )
            window_end = pd.Timestamp(
                year=d.year, month=d.month, day=d.day,
                hour=self.trade_end.hour, minute=self.trade_end.minute,
            )
            if tz is not None:
                window_start = window_start.tz_localize(
                    tz, nonexistent="shift_forward",
                )
                window_end = window_end.tz_localize(
                    tz, nonexistent="shift_forward",
                )

            mask = (df.index >= window_start) & (df.index < window_end)
            window_bars = df.loc[mask]

            if window_bars.empty:
                continue

            if self.mode == "breakout":
                sig = self._find_breakout_signal(window_bars, ar)
            else:
                sig = self._find_fade_signal(window_bars, ar)

            if sig is not None:
                signals.append(sig)

        return signals

    def _find_breakout_signal(
        self,
        window_bars: pd.DataFrame,
        ar: AsianRange,
    ) -> AsianSignal | None:
        """Detect the first breakout above/below the Asian range.

        - Long: first bar where high > asian_high — entry at asian_high
        - Short: first bar where low < asian_low — entry at asian_low
        - Same-bar tiebreaker: open >= midpoint → long, else short
          (matches SRS pattern in backtester._detect_breakout)

        Returns:
            AsianSignal or None if no breakout in the window
        """
        highs = window_bars["high"]
        lows = window_bars["low"]

        long_mask = highs > ar.asian_high
        short_mask = lows < ar.asian_low

        long_breaks = window_bars.index[long_mask]
        short_breaks = window_bars.index[short_mask]

        first_long = long_breaks[0] if len(long_breaks) > 0 else None
        first_short = short_breaks[0] if len(short_breaks) > 0 else None

        # Determine direction and entry
        if first_long is not None and first_short is not None:
            if first_long < first_short:
                direction, entry_time, entry_price = "long", first_long, ar.asian_high
            elif first_short < first_long:
                direction, entry_time, entry_price = "short", first_short, ar.asian_low
            else:
                # Same bar — use open relative to range midpoint
                bar = window_bars.loc[first_long]
                mid = (ar.asian_high + ar.asian_low) / 2
                if bar["open"] >= mid:
                    direction, entry_time, entry_price = "long", first_long, ar.asian_high
                else:
                    direction, entry_time, entry_price = "short", first_short, ar.asian_low
        elif first_long is not None:
            direction, entry_time, entry_price = "long", first_long, ar.asian_high
        elif first_short is not None:
            direction, entry_time, entry_price = "short", first_short, ar.asian_low
        else:
            return None

        return AsianSignal(
            date=ar.date,
            direction=direction,
            mode="breakout",
            entry_price=entry_price,
            entry_time=entry_time,
            asian_range=ar,
            instrument=ar.instrument,
        )

    def _find_fade_signal(
        self,
        window_bars: pd.DataFrame,
        ar: AsianRange,
    ) -> AsianSignal | None:
        """Detect the first fade (rejection) off the Asian range boundary.

        - Short fade: bar high touches/exceeds asian_high but close < asian_high
        - Long fade: bar low touches/goes below asian_low but close > asian_low
        - Returns on the first rejection found (bar-by-bar walk)

        Returns:
            AsianSignal or None if no rejection in the window
        """
        for ts, bar in window_bars.iterrows():
            # Short fade: wick above but rejected
            if bar["high"] >= ar.asian_high and bar["close"] < ar.asian_high:
                return AsianSignal(
                    date=ar.date,
                    direction="short",
                    mode="fade",
                    entry_price=bar["close"],
                    entry_time=ts,
                    asian_range=ar,
                    instrument=ar.instrument,
                )
            # Long fade: wick below but rejected
            if bar["low"] <= ar.asian_low and bar["close"] > ar.asian_low:
                return AsianSignal(
                    date=ar.date,
                    direction="long",
                    mode="fade",
                    entry_price=bar["close"],
                    entry_time=ts,
                    asian_range=ar,
                    instrument=ar.instrument,
                )

        return None


# ======================================================================
# Trade simulation
# ======================================================================


def simulate_trade(
    signal: AsianSignal,
    day_data: pd.DataFrame,
    strategy: AsianRangeStrategy,
    tick_size: float,
    tick_value: float,
    slippage_ticks: float = 1,
    commission: float = 1.24,
) -> Trade | None:
    """Simulate a single Asian range trade bar-by-bar.

    Args:
        signal: AsianSignal with direction, mode, entry_price, entry_time
        day_data: Pre-sliced day OHLCV DataFrame (Eastern Time)
        strategy: AsianRangeStrategy instance (for SL/TP params and trade_end)
        tick_size: Instrument tick size (e.g. 0.25)
        tick_value: Instrument tick value per tick (e.g. 1.25 for MES)
        slippage_ticks: Slippage in ticks per side (default 1)
        commission: Round-trip commission in dollars (default 1.24)

    Returns:
        Trade object or None if no bars in the trade window
    """
    ar = signal.asian_range

    # Build trade window: entry_time to trade_end
    entry_t = signal.entry_time
    trade_end_t = strategy.trade_end

    # Construct the trade_end timestamp on the signal's date
    d = signal.date
    tz = day_data.index.tz
    trade_end_ts = pd.Timestamp(
        year=d.year, month=d.month, day=d.day,
        hour=trade_end_t.hour, minute=trade_end_t.minute,
    )
    if tz is not None:
        trade_end_ts = trade_end_ts.tz_localize(
            tz, nonexistent="shift_forward",
        )

    # Filter bars from entry_time to trade_end
    mask = (day_data.index >= entry_t) & (day_data.index < trade_end_ts)
    trade_bars = day_data.loc[mask]
    if trade_bars.empty:
        return None

    # Entry price with slippage
    slippage = slippage_ticks * tick_size
    if signal.direction == "long":
        entry_price = signal.entry_price + slippage
    else:
        entry_price = signal.entry_price - slippage

    # Compute SL/TP
    sl, tp = strategy.compute_sl_tp(signal, entry_price)

    # Walk bar-by-bar (skip entry bar)
    exit_price = None
    exit_time = None
    exit_reason = "session_end"

    for _, bar in trade_bars.iloc[1:].iterrows():
        if signal.direction == "long":
            # Check SL first (conservative: SL before TP on same bar)
            if bar["low"] <= sl:
                exit_price = sl
                exit_time = bar.name
                exit_reason = "sl"
                break
            # Check TP
            if bar["high"] >= tp:
                exit_price = tp
                exit_time = bar.name
                exit_reason = "tp"
                break
        else:
            # Short: check SL first
            if bar["high"] >= sl:
                exit_price = sl
                exit_time = bar.name
                exit_reason = "sl"
                break
            # Check TP
            if bar["low"] <= tp:
                exit_price = tp
                exit_time = bar.name
                exit_reason = "tp"
                break

    # Forced exit at last bar close if no SL/TP hit
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
        stop_loss=sl,
        take_profit=tp,
        pnl_ticks=pnl_ticks,
        pnl_dollars=pnl_dollars,
        exit_reason=exit_reason,
        session=0,
        instrument=signal.instrument,
        date=signal.date,
        range_high=ar.asian_high,
        range_low=ar.asian_low,
    )


# ======================================================================
# Backtest runner
# ======================================================================


def run_backtest(
    data: dict[str, pd.DataFrame],
    strategy: AsianRangeStrategy,
    slippage_ticks: float = 1,
    commission: float = 1.24,
) -> list[Trade]:
    """Run the Asian Range strategy across multiple instruments.

    Args:
        data: Dict mapping instrument name (e.g. 'MES') to OHLCV DataFrame
        strategy: Configured AsianRangeStrategy instance
        slippage_ticks: Slippage in ticks per side (default 1)
        commission: Round-trip commission in dollars (default 1.24)

    Returns:
        List of all Trade objects across all instruments
    """
    all_trades: list[Trade] = []

    for instrument, df in data.items():
        tick_size = INSTRUMENTS[instrument]["tick_size"]
        tick_value = INSTRUMENTS[instrument]["tick_value"]

        # Compute Asian ranges
        ranges = strategy.find_asian_ranges(df, instrument, tick_size)

        # Find signals
        signals = strategy.find_signals(df, ranges)

        # Precompute day groups for fast lookup
        day_groups: dict[pd.Timestamp, pd.DataFrame] = {
            date: gdf for date, gdf in df.groupby(df.index.normalize())
        }

        # Simulate each signal
        tz = df.index.tz
        for signal in signals:
            # signal.date is tz-naive; day_groups keys are tz-aware
            lookup_date = signal.date
            if tz is not None and lookup_date.tz is None:
                lookup_date = lookup_date.tz_localize(tz)
            day_data = day_groups.get(lookup_date)
            if day_data is None:
                continue
            trade = simulate_trade(
                signal, day_data, strategy,
                tick_size, tick_value,
                slippage_ticks, commission,
            )
            if trade is not None:
                all_trades.append(trade)

    return all_trades


# ======================================================================
# Parameter grid
# ======================================================================


def build_param_grid() -> list[dict]:
    """Build a grid of all valid parameter combinations.

    Generates exactly 4,050 combinations:
    - Breakout: 162 base x 15 exit combos = 2,430
    - Fade: 162 base x 10 exit combos = 1,620
    - Total: 4,050

    CRITICAL: stop_type='opposite' is excluded when mode='fade'.
    Fade entries are near the boundary, making opposite-side stops useless.

    Returns:
        List of dicts, each with all 10 parameter keys
    """
    modes = ["breakout", "fade"]
    asian_ends = [time(0, 0), time(1, 0), time(2, 0)]
    trade_starts = [time(8, 0), time(9, 0), time(9, 30)]
    trade_ends = [time(12, 0), time(14, 0), time(16, 0)]
    min_range_ticks_vals = [5, 10, 15]
    max_range_ticks_vals = [75, None]

    stop_types = ["opposite", "multiple"]
    tp_types = ["rr", "opposite", "midpoint"]
    rr_ratios = [1.0, 2.0, 3.0]
    stop_multiples = [0.5, 1.0]

    grid: list[dict] = []

    base_combos = itertools.product(
        modes, asian_ends, trade_starts, trade_ends,
        min_range_ticks_vals, max_range_ticks_vals,
    )

    for mode, asian_end, trade_start, trade_end, min_rt, max_rt in base_combos:
        # Build exit parameter combos
        for stop_type in stop_types:
            # CRITICAL: skip opposite stop for fade mode
            if mode == "fade" and stop_type == "opposite":
                continue

            # Determine stop_multiple values
            if stop_type == "multiple":
                sm_values = stop_multiples
            else:
                # opposite stop: stop_multiple unused, use default
                sm_values = [0.5]

            for tp_type in tp_types:
                # Determine rr_ratio values
                if tp_type == "rr":
                    rr_values = rr_ratios
                else:
                    # non-rr tp: rr_ratio unused, use default
                    rr_values = [2.0]

                for rr_ratio in rr_values:
                    for stop_multiple in sm_values:
                        grid.append({
                            "mode": mode,
                            "asian_end": asian_end,
                            "trade_start": trade_start,
                            "trade_end": trade_end,
                            "min_range_ticks": min_rt,
                            "max_range_ticks": max_rt,
                            "stop_type": stop_type,
                            "tp_type": tp_type,
                            "rr_ratio": rr_ratio,
                            "stop_multiple": stop_multiple,
                        })

    return grid


# ======================================================================
# Grid search with precomputation
# ======================================================================


def _precompute_asian_ranges(
    data: dict[str, pd.DataFrame],
    asian_end_values: list[time],
    min_ticks_values: list[float],
    max_ticks_values: list[float | None],
) -> dict[tuple[str, str, float, float | None], list[AsianRange]]:
    """Precompute Asian ranges for all parameter combinations.

    The key optimization: only ``len(asian_end_values) × len(data)``
    actual range computations are performed (e.g. 3 × 2 = 6), then each
    result is filtered by (min_ticks, max_ticks) to build the full cache.

    Args:
        data: Dict mapping instrument name to OHLCV DataFrame
        asian_end_values: Unique asian_end time values from the grid
        min_ticks_values: Unique min_range_ticks values from the grid
        max_ticks_values: Unique max_range_ticks values from the grid

    Returns:
        Dict keyed by (instrument, asian_end_str, min_ticks, max_ticks)
        mapping to a list of AsianRange objects that pass the filters
    """
    cache: dict[tuple[str, str, float, float | None], list[AsianRange]] = {}

    for instrument, df in data.items():
        tick_size = INSTRUMENTS[instrument]["tick_size"]

        for asian_end in asian_end_values:
            ae_str = asian_end.strftime("%H:%M")

            # Compute ALL ranges with no filtering (min=0, max=None)
            unfiltered_strategy = AsianRangeStrategy(
                asian_end=asian_end,
                min_range_ticks=0,
                max_range_ticks=None,
            )
            all_ranges = unfiltered_strategy.find_asian_ranges(
                df, instrument, tick_size,
            )

            # Filter by each (min_ticks, max_ticks) combination
            for mn in min_ticks_values:
                for mx in max_ticks_values:
                    filtered = [
                        r for r in all_ranges
                        if r.range_ticks >= mn
                        and (mx is None or r.range_ticks <= mx)
                    ]
                    cache[(instrument, ae_str, mn, mx)] = filtered

    return cache


def run_grid_search(
    data: dict[str, pd.DataFrame],
    param_grid: list[dict],
    progress: bool = False,
) -> pd.DataFrame:
    """Run grid search over parameter combinations with precomputation.

    Precomputes Asian ranges for each unique (instrument, asian_end) pair,
    then filters by min/max ticks. This avoids redundant range scans —
    only ``len(asian_end_values) × len(instruments)`` actual computations
    instead of one per grid combination.

    Args:
        data: Dict mapping instrument name to OHLCV DataFrame
        param_grid: List of parameter dicts (from build_param_grid())
        progress: Whether to show a tqdm progress bar

    Returns:
        DataFrame with one row per parameter combination and metric columns
    """
    from .metrics import calculate_metrics

    # Extract unique parameter values for precomputation
    asian_end_values = sorted(
        {p["asian_end"] for p in param_grid},
        key=lambda t: (t.hour, t.minute),
    )
    min_ticks_values = sorted({p["min_range_ticks"] for p in param_grid})
    max_ticks_values = sorted(
        {p["max_range_ticks"] for p in param_grid},
        key=lambda x: (0, x) if x is not None else (1, 0),
    )

    print(
        f"  Precomputing Asian ranges for "
        f"{len(asian_end_values)} asian_end values "
        f"× {len(data)} instruments..."
    )
    range_cache = _precompute_asian_ranges(
        data, asian_end_values, min_ticks_values, max_ticks_values,
    )

    # Precompute day groups for fast trade simulation
    print("  Precomputing day groups...")
    day_groups: dict[str, dict[pd.Timestamp, pd.DataFrame]] = {}
    for instrument, df in data.items():
        day_groups[instrument] = {
            date: gdf for date, gdf in df.groupby(df.index.normalize())
        }

    print(f"  Done. Running {len(param_grid)} parameter combinations...")

    results: list[dict] = []
    iterator = tqdm(param_grid, desc="Grid search") if progress else param_grid

    for params in iterator:
        strategy = AsianRangeStrategy(**params)

        ae_str = params["asian_end"].strftime("%H:%M")
        mn = params["min_range_ticks"]
        mx = params["max_range_ticks"]

        all_trades: list[Trade] = []

        for instrument, df in data.items():
            tick_size = INSTRUMENTS[instrument]["tick_size"]
            tick_value = INSTRUMENTS[instrument]["tick_value"]

            # Look up cached ranges
            ranges = range_cache.get((instrument, ae_str, mn, mx), [])
            if not ranges:
                continue

            # Find signals from cached ranges
            signals = strategy.find_signals(df, ranges)

            # Simulate each signal
            inst_day_groups = day_groups[instrument]
            inst_tz = df.index.tz
            for signal in signals:
                # signal.date is tz-naive; day_groups keys are tz-aware
                lookup_date = signal.date
                if inst_tz is not None and lookup_date.tz is None:
                    lookup_date = lookup_date.tz_localize(inst_tz)
                day_data = inst_day_groups.get(lookup_date)
                if day_data is None:
                    continue
                trade = simulate_trade(
                    signal, day_data, strategy,
                    tick_size, tick_value,
                )
                if trade is not None:
                    all_trades.append(trade)

        metrics = calculate_metrics(all_trades)

        results.append({
            "mode": params["mode"],
            "asian_end": ae_str,
            "trade_start": params["trade_start"].strftime("%H:%M"),
            "trade_end": params["trade_end"].strftime("%H:%M"),
            "min_range_ticks": mn,
            "max_range_ticks": mx,
            "stop_type": params["stop_type"],
            "tp_type": params["tp_type"],
            "rr_ratio": params["rr_ratio"],
            "stop_multiple": params["stop_multiple"],
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


# ======================================================================
# Walk-forward helpers
# ======================================================================


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


# ======================================================================
# Walk-forward analysis
# ======================================================================


def run_walk_forward(
    data: dict[str, pd.DataFrame],
    param_grid: list[dict],
    train_months: int = 24,
    test_months: int = 6,
    top_n: int = 10,
) -> pd.DataFrame:
    """Rolling walk-forward analysis.

    For each window:
    1. Train: run grid search, select top_n by profit_factor (min 20 trades)
    2. Test: evaluate top_n out-of-sample
    3. Record IS and OOS metrics
    """
    from .metrics import calculate_metrics

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

        # In-sample grid search (no progress bar)
        is_results = run_grid_search(train_data, param_grid)
        if is_results.empty:
            continue

        # Filter: minimum 20 trades
        qualified = is_results[is_results["total_trades"] >= 20]
        if qualified.empty:
            continue

        # Select top N by profit factor
        top_params = qualified.nlargest(top_n, "profit_factor")

        # Evaluate OOS
        for _, row in top_params.iterrows():
            # Reconstruct params — convert string times back to time objects
            h, m = map(int, row["asian_end"].split(":"))
            ae = time(h, m)
            h, m = map(int, row["trade_start"].split(":"))
            ts = time(h, m)
            h, m = map(int, row["trade_end"].split(":"))
            te = time(h, m)

            max_rt = None if pd.isna(row["max_range_ticks"]) else int(row["max_range_ticks"])

            params = {
                "mode": row["mode"],
                "asian_end": ae,
                "trade_start": ts,
                "trade_end": te,
                "min_range_ticks": int(row["min_range_ticks"]),
                "max_range_ticks": max_rt,
                "stop_type": row["stop_type"],
                "tp_type": row["tp_type"],
                "rr_ratio": float(row["rr_ratio"]),
                "stop_multiple": float(row["stop_multiple"]),
            }

            strategy = AsianRangeStrategy(**params)
            oos_trades = run_backtest(test_data, strategy)
            oos_metrics = calculate_metrics(oos_trades)

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


# ======================================================================
# Report generation
# ======================================================================


def generate_report(
    grid_results: pd.DataFrame | None,
    wf_results: pd.DataFrame | None,
    baseline: dict | None,
) -> str:
    """Generate comprehensive text report from saved results.

    Args:
        grid_results: DataFrame from run_grid_search() with columns:
            mode, asian_end, trade_start, trade_end, min_range_ticks,
            max_range_ticks, stop_type, tp_type, rr_ratio, stop_multiple,
            total_trades, win_rate, profit_factor, total_pnl, sharpe_ratio,
            max_drawdown, avg_pnl_per_trade, pnl_long, pnl_short
        wf_results: DataFrame from run_walk_forward() with columns:
            window_id, train_start, train_end, test_start, test_end,
            params, is_pnl, oos_pnl, is_sharpe, oos_sharpe, is_pf, oos_pf
        baseline: Dict with keys: total_trades, win_rate, profit_factor,
            total_pnl, sharpe_ratio, max_drawdown, avg_pnl_per_trade

    Returns:
        Formatted text report string
    """
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("  ASIAN RANGE BREAKOUT / FADE STRATEGY — REPORT")
    lines.append("=" * 70)

    # ------------------------------------------------------------------
    # Section 1: Baseline
    # ------------------------------------------------------------------
    if baseline:
        lines.append("\n--- BASELINE (Default Parameters) ---")
        for key in [
            "total_trades", "win_rate", "profit_factor", "total_pnl",
            "sharpe_ratio", "max_drawdown", "avg_pnl_per_trade",
        ]:
            val = baseline.get(key)
            if val is None:
                continue
            if isinstance(val, float):
                lines.append(f"  {key:<25} {val:>15.4f}")
            else:
                lines.append(f"  {key:<25} {val:>15}")

    # ------------------------------------------------------------------
    # Sections 2-5: Grid search
    # ------------------------------------------------------------------
    if grid_results is not None and not grid_results.empty:
        # Section 2: Grid Search Summary
        total_combos = len(grid_results)
        profitable = grid_results[grid_results["profit_factor"] > 1.0]
        pct_profitable = len(profitable) / total_combos if total_combos else 0

        lines.append(f"\n--- GRID SEARCH ({total_combos} combinations) ---")
        lines.append(
            f"  Profitable (PF > 1.0): {len(profitable)} "
            f"({pct_profitable:.1%})"
        )
        lines.append(f"  Mean PF: {grid_results['profit_factor'].mean():.3f}")
        lines.append(f"  Max PF: {grid_results['profit_factor'].max():.3f}")
        lines.append(f"  Mean PnL: ${grid_results['total_pnl'].mean():,.0f}")

        # Section 3: Top 10 by Total PnL
        lines.append("\n  Top 10 by Total PnL:")
        top = grid_results.nlargest(10, "total_pnl")
        header = (
            f"  {'mode':>9} {'asian_end':>9} {'trade_start':>11} "
            f"{'trade_end':>9} {'stop':>6} {'tp':>8} "
            f"{'trades':>6} {'WR':>6} {'PF':>6} {'PnL':>10}"
        )
        lines.append(header)
        lines.append("  " + "-" * 86)
        for _, r in top.iterrows():
            lines.append(
                f"  {r['mode']:>9} {r['asian_end']:>9} "
                f"{r['trade_start']:>11} {r['trade_end']:>9} "
                f"{r['stop_type']:>6} {r['tp_type']:>8} "
                f"{r['total_trades']:>6.0f} {r['win_rate']:>5.1%} "
                f"{r['profit_factor']:>6.3f} ${r['total_pnl']:>9,.0f}"
            )

        # Section 4: Parameter Sensitivity
        lines.append("\n  Parameter Sensitivity (mean PF & mean PnL):")
        sensitivity_params = [
            "mode", "asian_end", "trade_start", "trade_end",
            "min_range_ticks", "max_range_ticks", "stop_type", "tp_type",
        ]
        for param in sensitivity_params:
            lines.append(f"\n  By {param}:")
            grouped = grid_results.groupby(param, dropna=False).agg(
                mean_pf=("profit_factor", "mean"),
                mean_pnl=("total_pnl", "mean"),
            )
            for val, row in grouped.iterrows():
                val_str = str(val) if pd.notna(val) else "None"
                lines.append(
                    f"    {val_str:<15} PF={row['mean_pf']:.3f}  "
                    f"PnL=${row['mean_pnl']:>8,.0f}"
                )

        # Section 5: Mode Comparison — Breakout vs Fade
        lines.append("\n--- MODE COMPARISON — Breakout vs Fade ---")
        for mode_val in ["breakout", "fade"]:
            subset = grid_results[grid_results["mode"] == mode_val]
            if subset.empty:
                continue
            mode_profitable = subset[subset["profit_factor"] > 1.0]
            pct = len(mode_profitable) / len(subset) if len(subset) else 0
            lines.append(f"\n  {mode_val.upper()}:")
            lines.append(f"    Total combos:  {len(subset)}")
            lines.append(f"    Profitable:    {len(mode_profitable)} ({pct:.1%})")
            lines.append(f"    Mean PF:       {subset['profit_factor'].mean():.3f}")
            lines.append(f"    Best PnL:      ${subset['total_pnl'].max():,.0f}")

        # Section 7: Instrument Analysis (Top 5 by PnL)
        if "pnl_long" in grid_results.columns and "pnl_short" in grid_results.columns:
            lines.append("\n--- INSTRUMENT ANALYSIS (Top 5 by PnL) ---")
            top5 = grid_results.nlargest(5, "total_pnl")
            header_inst = (
                f"  {'mode':>9} {'PnL_Long':>12} {'PnL_Short':>12}"
            )
            lines.append(header_inst)
            lines.append("  " + "-" * 35)
            for _, r in top5.iterrows():
                lines.append(
                    f"  {r['mode']:>9} ${r['pnl_long']:>11,.0f} "
                    f"${r['pnl_short']:>11,.0f}"
                )

    # ------------------------------------------------------------------
    # Section 6: Walk-Forward OOS
    # ------------------------------------------------------------------
    if wf_results is not None and not wf_results.empty:
        lines.append(
            f"\n--- WALK-FORWARD ({wf_results['window_id'].nunique()} windows) ---"
        )
        lines.append(f"  Total OOS PnL: ${wf_results['oos_pnl'].sum():,.0f}")
        lines.append(f"  Mean OOS PF: {wf_results['oos_pf'].mean():.3f}")
        lines.append(f"  Mean IS PF: {wf_results['is_pf'].mean():.3f}")
        oos_positive = (wf_results["oos_pnl"] > 0).sum()
        total_wf = len(wf_results)
        lines.append(
            f"  OOS positive: {oos_positive}/{total_wf} "
            f"({oos_positive / total_wf:.1%})"
        )
        is_mean = wf_results["is_pf"].mean()
        oos_mean = wf_results["oos_pf"].mean()
        degradation = 1 - oos_mean / max(is_mean, 1e-9)
        lines.append(f"  IS→OOS degradation: {degradation:.1%}")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)
