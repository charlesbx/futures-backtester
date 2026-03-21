"""Asian Range Breakout / Fade Strategy.

The Asian session (6:00 PM to ~2:00 AM ET) establishes a range that
US-session traders react to. Two modes:

- **breakout**: trade in the direction of the breakout above/below the
  Asian range during US hours.
- **fade**: fade the breakout, expecting price to return inside the range.

Provides Asian range detection, signal generation, trade simulation,
grid search, walk-forward analysis, and report generation.
"""

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
        stop_type: 'range' (opposite side) or 'multiple' (stop_multiple * range)
        tp_type: 'rr' (risk/reward ratio) or 'range_multiple'
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
        stop_type: str = "range",
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
