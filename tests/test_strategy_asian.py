"""Tests for Asian Range strategy — range detection, signals, and trade simulation."""

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
    build_param_grid,
    simulate_trade,
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
# Signal detection — Breakout
# ---------------------------------------------------------------------------


class TestBreakoutSignals:
    """Tests for AsianRangeStrategy._find_breakout_signal() via find_signals()."""

    @staticmethod
    def _make_range() -> AsianRange:
        """Build a standard AsianRange for 2024-01-08."""
        return AsianRange(
            date=pd.Timestamp("2024-01-08"),
            asian_high=5010.0,
            asian_low=4990.0,
            asian_start=pd.Timestamp("2024-01-07 18:00", tz=TZ),
            asian_end=pd.Timestamp("2024-01-08 02:00", tz=TZ),
            range_ticks=round((5010.0 - 4990.0) / TICK_SIZE_MES),
            instrument="MES",
        )

    def test_long_breakout(self):
        """High exceeds asian_high -> long signal with entry_price = asian_high."""
        df = make_overnight_bars(
            "2024-01-08",
            asian_high=5010.0,
            asian_low=4990.0,
            us_high=5020.0,  # breaks above asian_high
            us_low=4995.0,   # stays above asian_low
        )
        strategy = AsianRangeStrategy(mode="breakout", min_range_ticks=1)
        ar = self._make_range()
        signals = strategy.find_signals(df, [ar])

        assert len(signals) == 1
        sig = signals[0]
        assert sig.direction == "long"
        assert sig.mode == "breakout"
        assert sig.entry_price == 5010.0

    def test_short_breakout(self):
        """Low breaks below asian_low -> short signal with entry_price = asian_low."""
        df = make_overnight_bars(
            "2024-01-08",
            asian_high=5010.0,
            asian_low=4990.0,
            us_high=5005.0,  # stays below asian_high
            us_low=4980.0,   # breaks below asian_low
        )
        strategy = AsianRangeStrategy(mode="breakout", min_range_ticks=1)
        ar = self._make_range()
        signals = strategy.find_signals(df, [ar])

        assert len(signals) == 1
        sig = signals[0]
        assert sig.direction == "short"
        assert sig.mode == "breakout"
        assert sig.entry_price == 4990.0

    def test_no_breakout(self):
        """Price stays inside range -> no signal."""
        df = make_overnight_bars(
            "2024-01-08",
            asian_high=5010.0,
            asian_low=4990.0,
            us_high=5005.0,  # inside range
            us_low=4995.0,   # inside range
        )
        strategy = AsianRangeStrategy(mode="breakout", min_range_ticks=1)
        ar = self._make_range()
        signals = strategy.find_signals(df, [ar])

        assert len(signals) == 0


# ---------------------------------------------------------------------------
# Signal detection — Fade
# ---------------------------------------------------------------------------


class TestFadeSignals:
    """Tests for AsianRangeStrategy._find_fade_signal() via find_signals()."""

    @staticmethod
    def _make_range() -> AsianRange:
        """Build a standard AsianRange for 2024-01-08."""
        return AsianRange(
            date=pd.Timestamp("2024-01-08"),
            asian_high=5010.0,
            asian_low=4990.0,
            asian_start=pd.Timestamp("2024-01-07 18:00", tz=TZ),
            asian_end=pd.Timestamp("2024-01-08 02:00", tz=TZ),
            range_ticks=round((5010.0 - 4990.0) / TICK_SIZE_MES),
            instrument="MES",
        )

    def test_fade_high_short(self):
        """High touches boundary but close below -> short fade signal at bar close."""
        # Build bars where a bar's high >= asian_high but close < asian_high
        df = make_overnight_bars(
            "2024-01-08",
            asian_high=5010.0,
            asian_low=4990.0,
            us_high=5010.0,  # exactly touches asian_high
            us_low=4995.0,   # stays above asian_low
        )
        # The spike bar in make_overnight_bars has high=us_high and close=us_mid.
        # us_mid = (5010 + 4995)/2 = 5002.5, which is < 5010 (asian_high).
        # So high >= asian_high AND close < asian_high => short fade.
        strategy = AsianRangeStrategy(mode="fade", min_range_ticks=1)
        ar = self._make_range()
        signals = strategy.find_signals(df, [ar])

        assert len(signals) == 1
        sig = signals[0]
        assert sig.direction == "short"
        assert sig.mode == "fade"
        # Entry price should be the bar's close
        assert sig.entry_price == 5002.5

    def test_no_fade_when_close_above(self):
        """High touches boundary and close stays above -> no fade (it's a breakout)."""
        # Build bars where the high exceeds asian_high AND close >= asian_high
        df = make_overnight_bars(
            "2024-01-08",
            asian_high=5010.0,
            asian_low=4990.0,
            us_high=5030.0,  # well above
            us_low=5011.0,   # all bars stay above asian_high
        )
        # us_mid = (5030 + 5011)/2 = 5020.5, close=5020.5 which is > 5010
        # The default bars have high=us_mid+1=5021.5 and low=us_mid-1=5019.5
        # The spike bar has high=5030 but close=5020.5 which is > asian_high (5010)
        # So high >= asian_high but close >= asian_high => NOT a short fade
        # Low never touches asian_low (all lows >= 5019.5 > 4990) => no long fade either
        strategy = AsianRangeStrategy(mode="fade", min_range_ticks=1)
        ar = self._make_range()
        signals = strategy.find_signals(df, [ar])

        assert len(signals) == 0


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Verify module-level constants."""

    def test_asian_start(self):
        assert ASIAN_START == time(18, 0)

    def test_min_range_bars(self):
        assert MIN_RANGE_BARS == 5


# ---------------------------------------------------------------------------
# Helpers for SL/TP and trade simulation tests
# ---------------------------------------------------------------------------


def _make_range(
    asian_high: float = 5010.0,
    asian_low: float = 4990.0,
) -> AsianRange:
    """Build a standard AsianRange for testing."""
    return AsianRange(
        date=pd.Timestamp("2024-01-08"),
        asian_high=asian_high,
        asian_low=asian_low,
        asian_start=pd.Timestamp("2024-01-07 18:00", tz=TZ),
        asian_end=pd.Timestamp("2024-01-08 02:00", tz=TZ),
        range_ticks=round((asian_high - asian_low) / TICK_SIZE_MES),
        instrument="MES",
    )


def _make_signal(
    direction: str = "long",
    mode: str = "breakout",
    entry_price: float | None = None,
    asian_high: float = 5010.0,
    asian_low: float = 4990.0,
) -> AsianSignal:
    """Build an AsianSignal for testing.

    Default entry_price:
        breakout long  -> asian_high
        breakout short -> asian_low
        fade long      -> asian_low + 1.0  (close above low)
        fade short     -> asian_high - 1.0 (close below high)
    """
    ar = _make_range(asian_high=asian_high, asian_low=asian_low)

    if entry_price is None:
        if mode == "breakout":
            entry_price = asian_high if direction == "long" else asian_low
        else:
            # Fade: entry at bar close near the boundary
            entry_price = (asian_low + 1.0) if direction == "long" else (asian_high - 1.0)

    return AsianSignal(
        date=pd.Timestamp("2024-01-08"),
        direction=direction,
        mode=mode,
        entry_price=entry_price,
        entry_time=pd.Timestamp("2024-01-08 09:45", tz=TZ),
        asian_range=ar,
        instrument="MES",
    )


# ---------------------------------------------------------------------------
# compute_sl_tp tests
# ---------------------------------------------------------------------------


class TestComputeSlTp:
    """Tests for AsianRangeStrategy.compute_sl_tp()."""

    def test_range_stop_breakout_long(self):
        """stop_type='range', breakout long: SL = asian_low, TP via rr_ratio."""
        sig = _make_signal(direction="long", mode="breakout")
        # entry_price = asian_high = 5010.0
        strategy = AsianRangeStrategy(
            mode="breakout", stop_type="opposite", tp_type="rr",
            rr_ratio=2.0, min_range_ticks=1,
        )
        entry = sig.entry_price + 0.25  # slippage applied externally
        sl, tp = strategy.compute_sl_tp(sig, entry)

        assert sl == 4990.0  # asian_low
        risk = entry - sl  # 5010.25 - 4990.0 = 20.25
        expected_tp = entry + 2.0 * risk
        assert tp == pytest.approx(expected_tp)

    def test_multiple_stop_fade_short(self):
        """stop_type='multiple', fade short: SL = entry + 0.5 * range, TP = asian_low (opposite)."""
        sig = _make_signal(direction="short", mode="fade")
        # entry_price = asian_high - 1.0 = 5009.0
        strategy = AsianRangeStrategy(
            mode="fade", stop_type="multiple", tp_type="opposite",
            stop_multiple=0.5, min_range_ticks=1,
        )
        entry = sig.entry_price - 0.25  # slippage for short
        sl, tp = strategy.compute_sl_tp(sig, entry)

        range_size = 5010.0 - 4990.0  # 20.0
        expected_sl = entry + 0.5 * range_size  # 5008.75 + 10.0 = 5018.75
        assert sl == pytest.approx(expected_sl)
        # Fade short -> opposite boundary = asian_low
        assert tp == 4990.0

    def test_midpoint_tp(self):
        """tp_type='midpoint': TP = (asian_high + asian_low) / 2."""
        sig = _make_signal(direction="long", mode="breakout")
        strategy = AsianRangeStrategy(
            mode="breakout", stop_type="opposite", tp_type="midpoint",
            min_range_ticks=1,
        )
        entry = sig.entry_price + 0.25
        sl, tp = strategy.compute_sl_tp(sig, entry)

        assert sl == 4990.0  # range stop
        assert tp == pytest.approx((5010.0 + 4990.0) / 2)  # 5000.0

    def test_opposite_tp_breakout_long(self):
        """tp_type='opposite', breakout long: TP = asian_high + range_size (measured move)."""
        sig = _make_signal(direction="long", mode="breakout")
        strategy = AsianRangeStrategy(
            mode="breakout", stop_type="opposite", tp_type="opposite",
            min_range_ticks=1,
        )
        entry = sig.entry_price + 0.25
        sl, tp = strategy.compute_sl_tp(sig, entry)

        range_size = 5010.0 - 4990.0  # 20.0
        assert tp == pytest.approx(5010.0 + range_size)  # 5030.0

    def test_opposite_tp_breakout_short(self):
        """tp_type='opposite', breakout short: TP = asian_low - range_size."""
        sig = _make_signal(direction="short", mode="breakout")
        strategy = AsianRangeStrategy(
            mode="breakout", stop_type="opposite", tp_type="opposite",
            min_range_ticks=1,
        )
        entry = sig.entry_price - 0.25
        sl, tp = strategy.compute_sl_tp(sig, entry)

        range_size = 5010.0 - 4990.0
        assert tp == pytest.approx(4990.0 - range_size)  # 4970.0

    def test_opposite_tp_fade_long(self):
        """tp_type='opposite', fade long: TP = asian_high (opposite boundary)."""
        sig = _make_signal(direction="long", mode="fade")
        strategy = AsianRangeStrategy(
            mode="fade", stop_type="opposite", tp_type="opposite",
            min_range_ticks=1,
        )
        entry = sig.entry_price + 0.25
        sl, tp = strategy.compute_sl_tp(sig, entry)

        assert tp == 5010.0  # asian_high

    def test_rr_ratio_short(self):
        """RR ratio applied correctly for short direction."""
        sig = _make_signal(direction="short", mode="breakout")
        strategy = AsianRangeStrategy(
            mode="breakout", stop_type="opposite", tp_type="rr",
            rr_ratio=1.5, min_range_ticks=1,
        )
        entry = sig.entry_price - 0.25  # 4989.75
        sl, tp = strategy.compute_sl_tp(sig, entry)

        assert sl == 5010.0  # asian_high
        risk = sl - entry  # 5010.0 - 4989.75 = 20.25
        expected_tp = entry - 1.5 * risk
        assert tp == pytest.approx(expected_tp)


# ---------------------------------------------------------------------------
# simulate_trade tests
# ---------------------------------------------------------------------------


class TestSimulateTrade:
    """Tests for simulate_trade()."""

    @staticmethod
    def _make_trade_bars(
        entry_time: str = "2024-01-08 09:45",
        n_bars: int = 20,
        base_price: float = 5010.0,
        highs: list[float] | None = None,
        lows: list[float] | None = None,
        closes: list[float] | None = None,
    ) -> pd.DataFrame:
        """Build trade-window bars with controllable highs/lows/closes."""
        idx = pd.date_range(entry_time, periods=n_bars, freq="1min", tz=TZ)
        df = pd.DataFrame(
            {
                "open": base_price,
                "high": base_price + 1.0,
                "low": base_price - 1.0,
                "close": base_price,
                "volume": 100,
            },
            index=idx,
        )
        if highs is not None:
            for i, h in enumerate(highs):
                if h is not None and i < len(df):
                    df.iloc[i, df.columns.get_loc("high")] = h
        if lows is not None:
            for i, lo in enumerate(lows):
                if lo is not None and i < len(df):
                    df.iloc[i, df.columns.get_loc("low")] = lo
        if closes is not None:
            for i, c in enumerate(closes):
                if c is not None and i < len(df):
                    df.iloc[i, df.columns.get_loc("close")] = c
        return df

    def test_long_tp_hit(self):
        """Long trade hits TP on bar where high >= tp."""
        sig = _make_signal(direction="long", mode="breakout")
        strategy = AsianRangeStrategy(
            mode="breakout", stop_type="opposite", tp_type="rr",
            rr_ratio=2.0, min_range_ticks=1,
        )
        tick_size = 0.25
        tick_value = 1.25
        slippage = 1 * tick_size  # 0.25
        entry = sig.entry_price + slippage  # 5010.25
        sl_expected = 4990.0
        risk = entry - sl_expected  # 20.25
        tp_expected = entry + 2.0 * risk  # 5010.25 + 40.50 = 5050.75

        # Build bars: bar 0 is entry, bars 1-4 normal, bar 5 high hits TP
        highs = [None, None, None, None, None, tp_expected + 1.0]
        day_data = self._make_trade_bars(highs=highs)

        trade = simulate_trade(sig, day_data, strategy, tick_size, tick_value)

        assert trade is not None
        assert trade.exit_reason == "tp"
        assert trade.exit_price == pytest.approx(tp_expected)
        assert trade.direction == "long"
        assert trade.pnl_ticks == pytest.approx((tp_expected - entry) / tick_size)
        assert trade.range_high == 5010.0
        assert trade.range_low == 4990.0
        assert trade.session == 0

    def test_short_sl_hit(self):
        """Short trade hits SL when bar high >= sl."""
        sig = _make_signal(direction="short", mode="breakout")
        strategy = AsianRangeStrategy(
            mode="breakout", stop_type="opposite", tp_type="rr",
            rr_ratio=2.0, min_range_ticks=1,
        )
        tick_size = 0.25
        tick_value = 1.25
        entry = sig.entry_price - 0.25  # 4989.75
        sl_expected = 5010.0  # asian_high

        # Bar 3 high exceeds SL
        highs = [None, None, None, sl_expected + 1.0]
        day_data = self._make_trade_bars(base_price=4990.0, highs=highs)

        trade = simulate_trade(sig, day_data, strategy, tick_size, tick_value)

        assert trade is not None
        assert trade.exit_reason == "sl"
        assert trade.exit_price == pytest.approx(sl_expected)
        pnl_ticks = (entry - sl_expected) / tick_size  # negative
        assert trade.pnl_ticks == pytest.approx(pnl_ticks)

    def test_session_end_exit(self):
        """No SL or TP hit -> forced exit at last bar close."""
        sig = _make_signal(direction="long", mode="breakout")
        strategy = AsianRangeStrategy(
            mode="breakout", stop_type="opposite", tp_type="rr",
            rr_ratio=2.0, min_range_ticks=1,
        )
        tick_size = 0.25
        tick_value = 1.25

        # All bars stay well within SL/TP range
        day_data = self._make_trade_bars(n_bars=10, base_price=5010.0)

        trade = simulate_trade(sig, day_data, strategy, tick_size, tick_value)

        assert trade is not None
        assert trade.exit_reason == "session_end"
        assert trade.exit_price == day_data.iloc[-1]["close"]

    def test_sl_before_tp_same_bar(self):
        """When both SL and TP could trigger on the same bar, SL wins (conservative)."""
        sig = _make_signal(direction="long", mode="breakout")
        strategy = AsianRangeStrategy(
            mode="breakout", stop_type="opposite", tp_type="rr",
            rr_ratio=2.0, min_range_ticks=1,
        )
        tick_size = 0.25
        tick_value = 1.25
        entry = sig.entry_price + 0.25  # 5010.25
        sl_expected = 4990.0
        risk = entry - sl_expected
        tp_expected = entry + 2.0 * risk

        # Bar 2: low hits SL AND high hits TP
        highs = [None, None, tp_expected + 5.0]
        lows = [None, None, sl_expected - 5.0]
        day_data = self._make_trade_bars(highs=highs, lows=lows)

        trade = simulate_trade(sig, day_data, strategy, tick_size, tick_value)

        assert trade is not None
        assert trade.exit_reason == "sl"  # SL checked first

    def test_pnl_dollars_includes_commission(self):
        """PnL dollars = pnl_ticks * tick_value - commission."""
        sig = _make_signal(direction="long", mode="breakout")
        strategy = AsianRangeStrategy(
            mode="breakout", stop_type="opposite", tp_type="rr",
            rr_ratio=2.0, min_range_ticks=1,
        )
        tick_size = 0.25
        tick_value = 1.25
        commission = 2.50

        day_data = self._make_trade_bars(n_bars=5, base_price=5010.0)
        trade = simulate_trade(
            sig, day_data, strategy, tick_size, tick_value,
            commission=commission,
        )

        assert trade is not None
        expected_dollars = trade.pnl_ticks * tick_value - commission
        assert trade.pnl_dollars == pytest.approx(expected_dollars)

    def test_empty_bars_returns_none(self):
        """No bars in trade window returns None."""
        sig = _make_signal(direction="long", mode="breakout")
        strategy = AsianRangeStrategy(
            mode="breakout", min_range_ticks=1,
            trade_end=time(9, 30),  # end before entry -> no bars
        )
        tick_size = 0.25
        tick_value = 1.25

        day_data = self._make_trade_bars(
            entry_time="2024-01-08 09:45",
            n_bars=10,
        )
        trade = simulate_trade(sig, day_data, strategy, tick_size, tick_value)
        assert trade is None

    def test_slippage_applied(self):
        """Slippage is added for long, subtracted for short."""
        sig_long = _make_signal(direction="long", mode="breakout")
        sig_short = _make_signal(direction="short", mode="breakout")
        strategy = AsianRangeStrategy(mode="breakout", min_range_ticks=1)
        tick_size = 0.25
        tick_value = 1.25

        day_data = self._make_trade_bars(n_bars=10, base_price=5000.0)

        trade_long = simulate_trade(
            sig_long, day_data, strategy, tick_size, tick_value, slippage_ticks=2,
        )
        trade_short = simulate_trade(
            sig_short, day_data, strategy, tick_size, tick_value, slippage_ticks=2,
        )

        assert trade_long is not None
        assert trade_long.entry_price == sig_long.entry_price + 2 * tick_size

        assert trade_short is not None
        assert trade_short.entry_price == sig_short.entry_price - 2 * tick_size


# ---------------------------------------------------------------------------
# Parameter grid tests
# ---------------------------------------------------------------------------


class TestParamGrid:
    """Tests for build_param_grid()."""

    def test_grid_size(self):
        """Grid must contain exactly 4,050 valid parameter combinations."""
        grid = build_param_grid()
        assert len(grid) == 4050

    def test_no_fade_opposite_stop(self):
        """No combination should have mode='fade' AND stop_type='opposite'."""
        grid = build_param_grid()
        fade_opposite = [
            p for p in grid
            if p["mode"] == "fade" and p["stop_type"] == "opposite"
        ]
        assert len(fade_opposite) == 0
