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
from dataclasses import dataclass
from datetime import time

import pandas as pd

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
            if tz is not None:
                window_start = window_start.tz_localize(tz)
                window_end = window_end.tz_localize(tz)

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
                window_start = window_start.tz_localize(tz)
                window_end = window_end.tz_localize(tz)

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
        trade_end_ts = trade_end_ts.tz_localize(tz)

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
        for signal in signals:
            day_data = day_groups.get(signal.date)
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
