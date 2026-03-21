"""Tests for Asian Range strategy — range detection and data structures."""

from datetime import time

import pandas as pd
import pytest

from src.strategy_asian import (
    ASIAN_START,
    MIN_RANGE_BARS,
    AsianRange,
    AsianRangeStrategy,
    AsianSignal,
    Trade,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TZ = "US/Eastern"
TICK_SIZE_MES = 0.25


def make_bars(
    start: str,
    periods: int,
    freq: str = "1min",
    open_price: float = 4000.0,
    high_offset: float = 1.0,
    low_offset: float = 1.0,
    close_price: float | None = None,
    volume: int = 100,
) -> pd.DataFrame:
    """Build a simple OHLCV DataFrame with constant-ish prices.

    Useful for constructing precise windows where you control high/low.

    Args:
        start: Timestamp string (will be localized to US/Eastern)
        periods: Number of 1-min bars
        freq: Bar frequency (default '1min')
        open_price: Open price for every bar
        high_offset: Added to open_price to get high
        low_offset: Subtracted from open_price to get low
        close_price: Close price (defaults to open_price)
        volume: Volume per bar
    """
    idx = pd.date_range(start, periods=periods, freq=freq, tz=TZ)
    close = close_price if close_price is not None else open_price
    df = pd.DataFrame(
        {
            "open": open_price,
            "high": open_price + high_offset,
            "low": open_price - low_offset,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )
    return df


def make_overnight_bars(
    date: str,
    asian_high: float = 4010.0,
    asian_low: float = 3990.0,
    us_high: float = 4020.0,
    us_low: float = 3980.0,
    asian_end_hour: int = 2,
    asian_end_minute: int = 0,
) -> pd.DataFrame:
    """Build 1-min bars spanning an overnight session for trading date *date*.

    The Asian window runs from (date-1) 18:00 ET to date asian_end ET.
    Then US-session bars run from 09:30 ET to 16:00 ET on *date*.

    High/low values are placed exactly at the midpoint of the respective
    windows so they are easy to verify.

    Args:
        date: Trading date as 'YYYY-MM-DD'
        asian_high: Highest price during the Asian window
        asian_low: Lowest price during the Asian window
        us_high: Highest price during the US session
        us_low: Lowest price during the US session
        asian_end_hour: Hour for Asian window end
        asian_end_minute: Minute for Asian window end
    """
    d = pd.Timestamp(date)
    prev = d - pd.Timedelta(days=1)

    # --- Asian bars: prev day 18:00 to date asian_end ---
    asian_start_ts = pd.Timestamp(
        year=prev.year, month=prev.month, day=prev.day,
        hour=18, minute=0,
    )
    asian_end_ts = pd.Timestamp(
        year=d.year, month=d.month, day=d.day,
        hour=asian_end_hour, minute=asian_end_minute,
    )
    # Number of minutes in the Asian window
    asian_minutes = int((asian_end_ts - asian_start_ts).total_seconds() / 60)
    if asian_minutes <= 0:
        asian_minutes = 1

    asian_mid = (asian_high + asian_low) / 2.0
    asian_idx = pd.date_range(asian_start_ts, periods=asian_minutes, freq="1min", tz=TZ)
    asian_df = pd.DataFrame(
        {
            "open": asian_mid,
            "high": asian_mid + 1.0,   # default; overridden below
            "low": asian_mid - 1.0,    # default; overridden below
            "close": asian_mid,
            "volume": 100,
        },
        index=asian_idx,
    )
    # Place the extreme high and low at the midpoint bar of the Asian window
    mid_bar = len(asian_df) // 2
    asian_df.iloc[mid_bar, asian_df.columns.get_loc("high")] = asian_high
    asian_df.iloc[mid_bar, asian_df.columns.get_loc("low")] = asian_low

    # --- US session bars: 09:30 to 16:00 ---
    us_start_ts = pd.Timestamp(
        year=d.year, month=d.month, day=d.day,
        hour=9, minute=30,
    )
    us_minutes = 390  # 09:30 to 16:00 = 6.5 hours
    us_mid = (us_high + us_low) / 2.0
    us_idx = pd.date_range(us_start_ts, periods=us_minutes, freq="1min", tz=TZ)
    us_df = pd.DataFrame(
        {
            "open": us_mid,
            "high": us_mid + 1.0,
            "low": us_mid - 1.0,
            "close": us_mid,
            "volume": 200,
        },
        index=us_idx,
    )
    mid_us = len(us_df) // 2
    us_df.iloc[mid_us, us_df.columns.get_loc("high")] = us_high
    us_df.iloc[mid_us, us_df.columns.get_loc("low")] = us_low

    # Concatenate and sort
    combined = pd.concat([asian_df, us_df])
    combined.sort_index(inplace=True)
    return combined


# ---------------------------------------------------------------------------
# Dataclass sanity tests
# ---------------------------------------------------------------------------


class TestDataclasses:
    """Verify that all dataclasses can be instantiated with expected fields."""

    def test_asian_range_fields(self):
        ar = AsianRange(
            date=pd.Timestamp("2025-01-02"),
            asian_high=4010.0,
            asian_low=3990.0,
            asian_start=pd.Timestamp("2025-01-01 18:00", tz=TZ),
            asian_end=pd.Timestamp("2025-01-02 02:00", tz=TZ),
            range_ticks=80.0,
            instrument="MES",
        )
        assert ar.asian_high == 4010.0
        assert ar.range_ticks == 80.0
        assert ar.instrument == "MES"

    def test_asian_signal_fields(self):
        ar = AsianRange(
            date=pd.Timestamp("2025-01-02"),
            asian_high=4010.0,
            asian_low=3990.0,
            asian_start=pd.Timestamp("2025-01-01 18:00", tz=TZ),
            asian_end=pd.Timestamp("2025-01-02 02:00", tz=TZ),
            range_ticks=80.0,
            instrument="MES",
        )
        sig = AsianSignal(
            date=pd.Timestamp("2025-01-02"),
            direction="long",
            mode="breakout",
            entry_price=4010.25,
            entry_time=pd.Timestamp("2025-01-02 09:45", tz=TZ),
            asian_range=ar,
            instrument="MES",
        )
        assert sig.direction == "long"
        assert sig.mode == "breakout"
        assert sig.asian_range is ar

    def test_trade_compatible_with_metrics(self):
        """Trade dataclass has all fields expected by metrics.calculate_metrics()."""
        t = Trade(
            entry_time=pd.Timestamp("2025-01-02 09:45", tz=TZ),
            exit_time=pd.Timestamp("2025-01-02 11:00", tz=TZ),
            direction="long",
            entry_price=4010.25,
            exit_price=4030.0,
            stop_loss=3990.0,
            take_profit=4050.0,
            pnl_ticks=79.0,
            pnl_dollars=97.51,
            exit_reason="tp",
            session=0,
            instrument="MES",
            date=pd.Timestamp("2025-01-02"),
            range_high=4010.0,
            range_low=3990.0,
        )
        assert t.exit_reason == "tp"
        assert t.session == 0
        assert t.range_high == 4010.0


# ---------------------------------------------------------------------------
# find_asian_ranges tests
# ---------------------------------------------------------------------------


class TestFindAsianRanges:
    """Tests for AsianRangeStrategy.find_asian_ranges()."""

    def test_basic_range_detection(self):
        """Detect a single Asian range with correct high/low/ticks."""
        df = make_overnight_bars(
            "2025-01-02",
            asian_high=4010.0,
            asian_low=3990.0,
        )
        strategy = AsianRangeStrategy(min_range_ticks=1)
        ranges = strategy.find_asian_ranges(df, "MES", TICK_SIZE_MES)

        assert len(ranges) == 1
        ar = ranges[0]
        assert ar.asian_high == 4010.0
        assert ar.asian_low == 3990.0
        expected_ticks = round((4010.0 - 3990.0) / TICK_SIZE_MES)  # 80
        assert ar.range_ticks == expected_ticks
        assert ar.instrument == "MES"
        assert ar.date == pd.Timestamp("2025-01-02")

    def test_min_range_filter(self):
        """Ranges smaller than min_range_ticks are excluded."""
        # Range = 2.0 points = 8 ticks at 0.25 tick_size
        df = make_overnight_bars(
            "2025-01-02",
            asian_high=4001.0,
            asian_low=3999.0,
        )
        # min_range_ticks=10 should exclude 8-tick range
        strategy = AsianRangeStrategy(min_range_ticks=10)
        ranges = strategy.find_asian_ranges(df, "MES", TICK_SIZE_MES)
        assert len(ranges) == 0

        # min_range_ticks=5 should include it
        strategy2 = AsianRangeStrategy(min_range_ticks=5)
        ranges2 = strategy2.find_asian_ranges(df, "MES", TICK_SIZE_MES)
        assert len(ranges2) == 1

    def test_max_range_filter(self):
        """Ranges larger than max_range_ticks are excluded."""
        # Range = 20 points = 80 ticks
        df = make_overnight_bars(
            "2025-01-02",
            asian_high=4010.0,
            asian_low=3990.0,
        )
        strategy = AsianRangeStrategy(min_range_ticks=1, max_range_ticks=50)
        ranges = strategy.find_asian_ranges(df, "MES", TICK_SIZE_MES)
        assert len(ranges) == 0

        # max_range_ticks=100 should include it
        strategy2 = AsianRangeStrategy(min_range_ticks=1, max_range_ticks=100)
        ranges2 = strategy2.find_asian_ranges(df, "MES", TICK_SIZE_MES)
        assert len(ranges2) == 1

    def test_different_asian_end_times(self):
        """Asian end times of 00:00, 02:00, and 03:00 all work."""
        for end_hour, end_min in [(0, 0), (2, 0), (3, 0)]:
            df = make_overnight_bars(
                "2025-01-02",
                asian_high=4010.0,
                asian_low=3990.0,
                asian_end_hour=end_hour,
                asian_end_minute=end_min,
            )
            strategy = AsianRangeStrategy(
                min_range_ticks=1,
                asian_end=time(end_hour, end_min),
            )
            ranges = strategy.find_asian_ranges(df, "MES", TICK_SIZE_MES)
            assert len(ranges) == 1, (
                f"Expected 1 range for asian_end={end_hour}:{end_min:02d}, "
                f"got {len(ranges)}"
            )
            ar = ranges[0]
            assert ar.asian_high == 4010.0
            assert ar.asian_low == 3990.0

    def test_multiple_dates(self):
        """Multiple trading dates produce multiple ranges."""
        df1 = make_overnight_bars("2025-01-02", asian_high=4010.0, asian_low=3990.0)
        df2 = make_overnight_bars("2025-01-03", asian_high=4015.0, asian_low=3985.0)
        df = pd.concat([df1, df2]).sort_index()

        strategy = AsianRangeStrategy(min_range_ticks=1)
        ranges = strategy.find_asian_ranges(df, "MES", TICK_SIZE_MES)

        # Should find ranges for both dates
        assert len(ranges) >= 2
        dates_found = {r.date for r in ranges}
        assert pd.Timestamp("2025-01-02") in dates_found
        assert pd.Timestamp("2025-01-03") in dates_found

    def test_insufficient_bars_excluded(self):
        """Fewer than MIN_RANGE_BARS bars in the window -> no range."""
        # Only 3 bars in the Asian window
        df = make_bars(
            "2025-01-01 18:00",
            periods=3,
            open_price=4000.0,
            high_offset=5.0,
            low_offset=5.0,
        )
        strategy = AsianRangeStrategy(min_range_ticks=1)
        ranges = strategy.find_asian_ranges(df, "MES", TICK_SIZE_MES)
        assert len(ranges) == 0

    def test_zero_range_excluded(self):
        """A flat range (high == low) is excluded."""
        df = make_bars(
            "2025-01-01 18:00",
            periods=60,
            open_price=4000.0,
            high_offset=0.0,
            low_offset=0.0,
        )
        strategy = AsianRangeStrategy(min_range_ticks=0)
        ranges = strategy.find_asian_ranges(df, "MES", TICK_SIZE_MES)
        assert len(ranges) == 0


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Verify module-level constants."""

    def test_asian_start(self):
        assert ASIAN_START == time(18, 0)

    def test_min_range_bars(self):
        assert MIN_RANGE_BARS == 5
