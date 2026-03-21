# Asian Range Strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Strategy #4 (Asian Range) — breakout and fade modes on the overnight session range for MES/MNQ futures, with grid search (~4,050 combos) and walk-forward validation.

**Architecture:** Single self-contained module `src/strategy_asian.py` following the `strategy_gapfade.py` pattern. All strategy logic, grid search, walk-forward, and report generation in one file. CLI entry point in `run_asian.py`. No changes to shared infrastructure (`data_loader.py`, `metrics.py`).

**Tech Stack:** Python 3.10+, pandas, numpy, tqdm. Existing `src/data_loader.py` for data loading, `src/metrics.py` for performance metrics.

**Spec:** `docs/superpowers/specs/2026-03-21-asian-range-strategy-design.md`

**Pattern reference:** `src/strategy_gapfade.py` (closest existing strategy), `run_gapfade.py` (CLI pattern)

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/strategy_asian.py` | Create | All strategy logic: dataclasses, range computation, signal detection, trade simulation, grid search, walk-forward, report |
| `tests/test_strategy_asian.py` | Create | Unit tests for range computation, signal detection, trade simulation, parameter grid |
| `run_asian.py` | Create | CLI entry point: baseline, grid, walk-forward, report |

---

### Task 1: Data Structures and Asian Range Computation

**Files:**
- Create: `src/strategy_asian.py`
- Create: `tests/test_strategy_asian.py`

- [ ] **Step 1: Create test file with synthetic data fixture**

Create `tests/test_strategy_asian.py` with a helper that builds a small 1-min OHLCV DataFrame spanning an overnight session (6 PM to next day 4 PM) in Eastern Time. This fixture is reused by all tests.

```python
"""Tests for Asian Range Strategy."""

import pandas as pd
import numpy as np
import pytest

from src.strategy_asian import (
    AsianRange,
    AsianRangeStrategy,
)


def make_bars(
    start: str,
    end: str,
    base_price: float = 5000.0,
    tick_size: float = 0.25,
    tz: str = "US/Eastern",
) -> pd.DataFrame:
    """Create synthetic 1-min OHLCV bars between start and end.

    Bars have small random noise around base_price.
    """
    rng = np.random.default_rng(42)
    idx = pd.date_range(start, end, freq="1min", tz=tz)
    n = len(idx)
    noise = rng.normal(0, 2, n) * tick_size
    opens = base_price + noise
    highs = opens + rng.uniform(0, 4, n) * tick_size
    lows = opens - rng.uniform(0, 4, n) * tick_size
    closes = opens + rng.normal(0, 1, n) * tick_size
    volumes = rng.integers(100, 1000, n)

    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=idx,
    )


def make_overnight_bars(
    trade_date: str = "2024-01-08",
    asian_high: float = 5010.0,
    asian_low: float = 4990.0,
    us_high: float = 5020.0,
    us_low: float = 4980.0,
) -> pd.DataFrame:
    """Create bars for a full overnight+day session with controlled highs/lows.

    Asian session: previous day 18:00 to trade_date 02:00
    US session: trade_date 08:00 to 16:00

    Bars are at base_price=5000 with the specified extremes injected
    at specific times.
    """
    date = pd.Timestamp(trade_date)
    prev_date = date - pd.Timedelta(days=1)
    tz = "US/Eastern"

    # Asian session: prev day 18:00 to trade_date 02:00
    asian_bars = make_bars(
        f"{prev_date.date()} 18:00", f"{date.date()} 01:59", base_price=5000.0
    )
    # Inject controlled high at 20:00 and low at 22:00
    high_time = pd.Timestamp(f"{prev_date.date()} 20:00", tz=tz)
    low_time = pd.Timestamp(f"{prev_date.date()} 22:00", tz=tz)
    if high_time in asian_bars.index:
        asian_bars.loc[high_time, "high"] = asian_high
        asian_bars.loc[high_time, "open"] = asian_high - 0.5
        asian_bars.loc[high_time, "close"] = asian_high - 0.75
    if low_time in asian_bars.index:
        asian_bars.loc[low_time, "low"] = asian_low
        asian_bars.loc[low_time, "open"] = asian_low + 0.5
        asian_bars.loc[low_time, "close"] = asian_low + 0.75

    # US session: trade_date 08:00 to 16:00
    us_bars = make_bars(
        f"{date.date()} 08:00", f"{date.date()} 15:59", base_price=5000.0
    )
    # Inject breakout high at 09:45 and low at 10:30
    bo_high_time = pd.Timestamp(f"{date.date()} 09:45", tz=tz)
    bo_low_time = pd.Timestamp(f"{date.date()} 10:30", tz=tz)
    if bo_high_time in us_bars.index:
        us_bars.loc[bo_high_time, "high"] = us_high
        us_bars.loc[bo_high_time, "open"] = us_high - 1.0
        us_bars.loc[bo_high_time, "close"] = us_high - 0.5
    if bo_low_time in us_bars.index:
        us_bars.loc[bo_low_time, "low"] = us_low
        us_bars.loc[bo_low_time, "open"] = us_low + 1.0
        us_bars.loc[bo_low_time, "close"] = us_low + 0.5

    return pd.concat([asian_bars, us_bars]).sort_index()
```

- [ ] **Step 2: Create strategy_asian.py with dataclasses**

Create `src/strategy_asian.py` with the three dataclasses and an empty `AsianRangeStrategy` class:

```python
"""Asian Range Strategy.

Tests breakout and fade modes on the overnight Asian session range
(high/low between 6 PM ET and midnight-2 AM ET) for MES/MNQ futures.

Provides signal detection, trade simulation, grid search,
walk-forward analysis, and report generation.
"""

from dataclasses import dataclass
from datetime import time, timedelta
import itertools
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from .data_loader import INSTRUMENTS, load_processed


@dataclass
class AsianRange:
    """Overnight Asian session range for one trading day."""

    date: pd.Timestamp          # US trading date this range precedes
    asian_high: float
    asian_low: float
    asian_start: pd.Timestamp
    asian_end: pd.Timestamp
    range_ticks: float          # (high - low) / tick_size
    instrument: str


@dataclass
class AsianSignal:
    """An entry signal from the Asian range strategy."""

    date: pd.Timestamp
    direction: str              # 'long' or 'short'
    mode: str                   # 'breakout' or 'fade'
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
    exit_reason: str            # 'tp', 'sl', 'session_end'
    session: int                # always 0 for asian range
    instrument: str
    date: pd.Timestamp
    range_high: float           # asian_high
    range_low: float            # asian_low


MIN_RANGE_BARS = 5


class AsianRangeStrategy:
    """Asian Range strategy with breakout and fade modes.

    Args:
        mode: 'breakout' or 'fade'
        asian_end: End of Asian session (default midnight)
        trade_start: Start of US trading window (default 9:30 AM)
        trade_end: End of US trading window / forced exit (default 4:00 PM)
        min_range_ticks: Minimum range size in ticks (default 10)
        max_range_ticks: Maximum range size in ticks (default None = no cap)
        stop_type: 'opposite' (breakout only) or 'multiple'
        tp_type: 'rr', 'opposite', or 'midpoint'
        rr_ratio: Risk/reward ratio (used when tp_type='rr', default 2.0)
        stop_multiple: Stop as fraction of range (used when stop_type='multiple', default 0.5)
    """

    ASIAN_START = time(18, 0)  # 6:00 PM ET, fixed

    def __init__(
        self,
        mode: str = "breakout",
        asian_end: time = time(0, 0),
        trade_start: time = time(9, 30),
        trade_end: time = time(16, 0),
        min_range_ticks: float = 10,
        max_range_ticks: float | None = None,
        stop_type: str = "opposite",
        tp_type: str = "rr",
        rr_ratio: float = 2.0,
        stop_multiple: float = 0.5,
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
```

- [ ] **Step 3: Implement find_asian_ranges() with cross-midnight logic**

Add to `AsianRangeStrategy`:

```python
    def find_asian_ranges(
        self, df: pd.DataFrame, instrument: str, tick_size: float,
    ) -> list[AsianRange]:
        """Compute Asian session high/low for each trading day.

        The Asian range spans midnight: for trading date D, the window
        is D-1 18:00 ET to D asian_end ET.  Uses absolute timestamp
        filtering (not day groups) to handle the cross-midnight query.
        """
        tz = df.index.tz
        asian_end_t = self.asian_end

        # Get unique trading dates (dates that have bars after midnight)
        dates = sorted(pd.Series(df.index.date).unique())

        ranges = []
        for date in dates:
            prev_date = date - timedelta(days=1)

            # Build absolute window: D-1 18:00 to D asian_end
            window_start = pd.Timestamp(
                year=prev_date.year, month=prev_date.month, day=prev_date.day,
                hour=self.ASIAN_START.hour, minute=self.ASIAN_START.minute,
                tz=tz,
            )
            # asian_end is after midnight (00:00, 01:00, or 02:00)
            window_end = pd.Timestamp(
                year=date.year, month=date.month, day=date.day,
                hour=asian_end_t.hour, minute=asian_end_t.minute,
                tz=tz,
            )

            # Filter bars in window
            range_bars = df[(df.index >= window_start) & (df.index < window_end)]

            if len(range_bars) < MIN_RANGE_BARS:
                continue

            asian_high = range_bars["high"].max()
            asian_low = range_bars["low"].min()

            if asian_high <= asian_low:
                continue

            range_ticks = (asian_high - asian_low) / tick_size

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
```

- [ ] **Step 4: Write tests for find_asian_ranges()**

Add to `tests/test_strategy_asian.py`:

```python
class TestFindAsianRanges:
    """Test Asian range computation."""

    def test_basic_range_detection(self):
        """Detects range from overnight bars with correct high/low."""
        df = make_overnight_bars(
            trade_date="2024-01-08", asian_high=5010.0, asian_low=4990.0,
        )
        strategy = AsianRangeStrategy(asian_end=time(2, 0))
        ranges = strategy.find_asian_ranges(df, "MES", tick_size=0.25)

        assert len(ranges) >= 1
        r = ranges[0]
        assert r.asian_high == 5010.0
        assert r.asian_low == 4990.0
        assert r.range_ticks == (5010.0 - 4990.0) / 0.25  # 80 ticks

    def test_min_range_filter(self):
        """Ranges below min_range_ticks are excluded."""
        df = make_overnight_bars(
            trade_date="2024-01-08", asian_high=5001.0, asian_low=5000.0,
        )
        # Range = 1.0 / 0.25 = 4 ticks, min is 10
        strategy = AsianRangeStrategy(asian_end=time(2, 0), min_range_ticks=10)
        ranges = strategy.find_asian_ranges(df, "MES", tick_size=0.25)
        assert len(ranges) == 0

    def test_max_range_filter(self):
        """Ranges above max_range_ticks are excluded."""
        df = make_overnight_bars(
            trade_date="2024-01-08", asian_high=5050.0, asian_low=4950.0,
        )
        # Range = 100 / 0.25 = 400 ticks, max is 75
        strategy = AsianRangeStrategy(
            asian_end=time(2, 0), min_range_ticks=5, max_range_ticks=75,
        )
        ranges = strategy.find_asian_ranges(df, "MES", tick_size=0.25)
        assert len(ranges) == 0

    def test_different_asian_end_times(self):
        """Different asian_end values produce valid ranges."""
        df = make_overnight_bars(
            trade_date="2024-01-08", asian_high=5010.0, asian_low=4990.0,
        )
        for end_hour in [0, 1, 2]:
            strategy = AsianRangeStrategy(
                asian_end=time(end_hour, 0), min_range_ticks=5,
            )
            ranges = strategy.find_asian_ranges(df, "MES", tick_size=0.25)
            # Should find at least the range (injected high/low are at 20:00/22:00,
            # all within the midnight window)
            assert len(ranges) >= 1 if end_hour > 0 else True
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_strategy_asian.py::TestFindAsianRanges -v`
Expected: All 4 tests pass (or fail if fixture needs adjustment — fix iteratively).

- [ ] **Step 6: Commit**

```bash
git add src/strategy_asian.py tests/test_strategy_asian.py
git commit -m "feat(asian): add data structures and range computation with tests"
```

---

### Task 2: Signal Detection (Breakout + Fade)

**Files:**
- Modify: `src/strategy_asian.py`
- Modify: `tests/test_strategy_asian.py`

- [ ] **Step 1: Implement find_signals() for breakout mode**

Add to `AsianRangeStrategy`:

```python
    def find_signals(
        self, df: pd.DataFrame, ranges: list[AsianRange],
    ) -> list[AsianSignal]:
        """Detect entry signals for all computed ranges.

        Dispatches to _find_breakout_signal or _find_fade_signal based on mode.
        """
        signals = []
        tz = df.index.tz

        for ar in ranges:
            # Build trade window timestamps
            trade_start_ts = pd.Timestamp(
                year=ar.date.year, month=ar.date.month, day=ar.date.day,
                hour=self.trade_start.hour, minute=self.trade_start.minute,
                tz=tz,
            )
            trade_end_ts = pd.Timestamp(
                year=ar.date.year, month=ar.date.month, day=ar.date.day,
                hour=self.trade_end.hour, minute=self.trade_end.minute,
                tz=tz,
            )

            window_bars = df[(df.index >= trade_start_ts) & (df.index < trade_end_ts)]
            if window_bars.empty:
                continue

            if self.mode == "breakout":
                signal = self._find_breakout_signal(window_bars, ar)
            else:
                signal = self._find_fade_signal(window_bars, ar)

            if signal is not None:
                signals.append(signal)

        return signals

    def _find_breakout_signal(
        self, window_bars: pd.DataFrame, ar: AsianRange,
    ) -> AsianSignal | None:
        """Detect first breakout above asian_high or below asian_low."""
        highs = window_bars["high"]
        lows = window_bars["low"]

        long_mask = highs > ar.asian_high
        short_mask = lows < ar.asian_low

        long_breaks = window_bars.index[long_mask]
        short_breaks = window_bars.index[short_mask]

        first_long = long_breaks[0] if len(long_breaks) > 0 else None
        first_short = short_breaks[0] if len(short_breaks) > 0 else None

        if first_long is not None and first_short is not None:
            if first_long < first_short:
                direction, entry_time = "long", first_long
            elif first_short < first_long:
                direction, entry_time = "short", first_short
            else:
                # Same bar — use open relative to midpoint (SRS pattern)
                bar = window_bars.loc[first_long]
                mid = (ar.asian_high + ar.asian_low) / 2
                if bar["open"] >= mid:
                    direction, entry_time = "long", first_long
                else:
                    direction, entry_time = "short", first_short
        elif first_long is not None:
            direction, entry_time = "long", first_long
        elif first_short is not None:
            direction, entry_time = "short", first_short
        else:
            return None

        # Entry at boundary level (slippage applied later in simulate_trade)
        entry_price = ar.asian_high if direction == "long" else ar.asian_low

        return AsianSignal(
            date=ar.date,
            direction=direction,
            mode="breakout",
            entry_price=entry_price,
            entry_time=entry_time,
            asian_range=ar,
            instrument=ar.instrument,
        )
```

- [ ] **Step 2: Implement _find_fade_signal()**

Add to `AsianRangeStrategy`:

```python
    def _find_fade_signal(
        self, window_bars: pd.DataFrame, ar: AsianRange,
    ) -> AsianSignal | None:
        """Detect first rejection at an Asian range boundary.

        Short fade: bar high >= asian_high AND bar close < asian_high
        Long fade:  bar low <= asian_low AND bar close > asian_low
        """
        for ts, bar in window_bars.iterrows():
            # Check fade-the-high (short)
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
            # Check fade-the-low (long)
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
```

- [ ] **Step 3: Write tests for signal detection**

Add to `tests/test_strategy_asian.py`:

```python
from datetime import time

from src.strategy_asian import AsianRange, AsianSignal, AsianRangeStrategy


class TestBreakoutSignals:
    """Test breakout signal detection."""

    def _make_range(self, date="2024-01-08"):
        tz = "US/Eastern"
        return AsianRange(
            date=pd.Timestamp(date),
            asian_high=5010.0,
            asian_low=4990.0,
            asian_start=pd.Timestamp(f"2024-01-07 18:00", tz=tz),
            asian_end=pd.Timestamp(f"{date} 02:00", tz=tz),
            range_ticks=80.0,
            instrument="MES",
        )

    def test_long_breakout(self):
        """Detects long when high exceeds asian_high."""
        df = make_overnight_bars(
            trade_date="2024-01-08",
            asian_high=5010.0, asian_low=4990.0,
            us_high=5020.0, us_low=4995.0,  # breaks high but not low
        )
        ar = self._make_range()
        strategy = AsianRangeStrategy(mode="breakout", trade_start=time(9, 0))
        signals = strategy.find_signals(df, [ar])

        assert len(signals) == 1
        assert signals[0].direction == "long"
        assert signals[0].mode == "breakout"
        assert signals[0].entry_price == 5010.0

    def test_short_breakout(self):
        """Detects short when low breaks below asian_low."""
        df = make_overnight_bars(
            trade_date="2024-01-08",
            asian_high=5010.0, asian_low=4990.0,
            us_high=5005.0, us_low=4980.0,  # breaks low but not high
        )
        ar = self._make_range()
        strategy = AsianRangeStrategy(mode="breakout", trade_start=time(9, 0))
        signals = strategy.find_signals(df, [ar])

        assert len(signals) == 1
        assert signals[0].direction == "short"
        assert signals[0].entry_price == 4990.0

    def test_no_breakout(self):
        """No signal when price stays within range."""
        df = make_overnight_bars(
            trade_date="2024-01-08",
            asian_high=5010.0, asian_low=4990.0,
            us_high=5005.0, us_low=4995.0,  # stays inside range
        )
        ar = self._make_range()
        strategy = AsianRangeStrategy(mode="breakout", trade_start=time(9, 0))
        signals = strategy.find_signals(df, [ar])
        assert len(signals) == 0


class TestFadeSignals:
    """Test fade signal detection."""

    def _make_range(self, date="2024-01-08"):
        tz = "US/Eastern"
        return AsianRange(
            date=pd.Timestamp(date),
            asian_high=5010.0,
            asian_low=4990.0,
            asian_start=pd.Timestamp(f"2024-01-07 18:00", tz=tz),
            asian_end=pd.Timestamp(f"{date} 02:00", tz=tz),
            range_ticks=80.0,
            instrument="MES",
        )

    def test_fade_high_short(self):
        """Detects short fade when high touches boundary but close is below."""
        df = make_overnight_bars(
            trade_date="2024-01-08",
            asian_high=5010.0, asian_low=4990.0,
            us_high=5020.0, us_low=4995.0,
        )
        # The injected bar at 09:45 has high=5020 and close=5019.5
        # which is > asian_high, so it won't trigger fade (close must be < asian_high)
        # We need to adjust: set close below asian_high
        tz = "US/Eastern"
        bo_time = pd.Timestamp("2024-01-08 09:45", tz=tz)
        if bo_time in df.index:
            df.loc[bo_time, "high"] = 5010.5   # touches asian_high
            df.loc[bo_time, "close"] = 5008.0  # closes below

        ar = self._make_range()
        strategy = AsianRangeStrategy(mode="fade", trade_start=time(9, 0))
        signals = strategy.find_signals(df, [ar])

        assert len(signals) == 1
        assert signals[0].direction == "short"
        assert signals[0].mode == "fade"
        assert signals[0].entry_price == 5008.0
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_strategy_asian.py -v`
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/strategy_asian.py tests/test_strategy_asian.py
git commit -m "feat(asian): add breakout and fade signal detection with tests"
```

---

### Task 3: Trade Simulation

**Files:**
- Modify: `src/strategy_asian.py`
- Modify: `tests/test_strategy_asian.py`

- [ ] **Step 1: Implement compute_sl_tp() helper**

Add to `AsianRangeStrategy`:

```python
    def compute_sl_tp(
        self, signal: AsianSignal, entry_price: float,
    ) -> tuple[float, float]:
        """Compute stop-loss and take-profit prices.

        Returns (stop_loss, take_profit).
        """
        ar = signal.asian_range
        range_size = ar.asian_high - ar.asian_low

        # Stop-loss
        if self.stop_type == "opposite":
            # Breakout only (fade excluded in grid)
            if signal.direction == "long":
                sl = ar.asian_low
            else:
                sl = ar.asian_high
        else:  # multiple
            if signal.direction == "long":
                sl = entry_price - self.stop_multiple * range_size
            else:
                sl = entry_price + self.stop_multiple * range_size

        # Risk distance for R:R calculation
        risk_distance = abs(entry_price - sl)

        # Take-profit
        if self.tp_type == "rr":
            if signal.direction == "long":
                tp = entry_price + self.rr_ratio * risk_distance
            else:
                tp = entry_price - self.rr_ratio * risk_distance
        elif self.tp_type == "opposite":
            if signal.mode == "breakout":
                # Measured move: one range beyond the boundary
                if signal.direction == "long":
                    tp = ar.asian_high + range_size
                else:
                    tp = ar.asian_low - range_size
            else:
                # Fade: literal opposite boundary
                if signal.direction == "long":
                    tp = ar.asian_high
                else:
                    tp = ar.asian_low
        else:  # midpoint
            tp = (ar.asian_high + ar.asian_low) / 2

        return sl, tp
```

- [ ] **Step 2: Implement simulate_trade()**

Add as a module-level function:

```python
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

    Entry: bar after signal detection.
    Exit: SL, TP, or forced at trade_end (session_end).
    SL checked before TP on same bar (conservative).
    """
    ar = signal.asian_range
    trade_end_t = strategy.trade_end

    # Bars from signal entry time to trade_end
    day_times = day_data.index.time
    trade_bars = day_data.loc[
        (day_data.index >= signal.entry_time) & (day_times < trade_end_t)
    ]
    if trade_bars.empty:
        return None

    # Apply slippage to entry price
    if signal.mode == "breakout":
        if signal.direction == "long":
            entry_price = signal.entry_price + slippage_ticks * tick_size
        else:
            entry_price = signal.entry_price - slippage_ticks * tick_size
    else:
        # Fade: entry at bar close, slippage applied directionally
        if signal.direction == "long":
            entry_price = signal.entry_price + slippage_ticks * tick_size
        else:
            entry_price = signal.entry_price - slippage_ticks * tick_size

    sl, tp = strategy.compute_sl_tp(signal, entry_price)

    # Walk bar-by-bar (skip entry bar)
    exit_price = None
    exit_time = None
    exit_reason = "session_end"

    for _, bar in trade_bars.iloc[1:].iterrows():
        if signal.direction == "long":
            # SL first (conservative)
            if bar["low"] <= sl:
                exit_price, exit_time, exit_reason = sl, bar.name, "sl"
                break
            if bar["high"] >= tp:
                exit_price, exit_time, exit_reason = tp, bar.name, "tp"
                break
        else:
            if bar["high"] >= sl:
                exit_price, exit_time, exit_reason = sl, bar.name, "sl"
                break
            if bar["low"] <= tp:
                exit_price, exit_time, exit_reason = tp, bar.name, "tp"
                break

    # Forced exit at session end
    if exit_price is None:
        last_bar = trade_bars.iloc[-1]
        exit_price = last_bar["close"]
        exit_time = last_bar.name

    # PnL
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
```

- [ ] **Step 3: Write tests for SL/TP computation and trade simulation**

Add to `tests/test_strategy_asian.py`:

```python
from src.strategy_asian import simulate_trade, Trade


class TestComputeSlTp:
    """Test stop-loss and take-profit calculations."""

    def _make_signal(self, direction="long", mode="breakout", entry_price=5010.0):
        tz = "US/Eastern"
        ar = AsianRange(
            date=pd.Timestamp("2024-01-08"),
            asian_high=5010.0, asian_low=4990.0,
            asian_start=pd.Timestamp("2024-01-07 18:00", tz=tz),
            asian_end=pd.Timestamp("2024-01-08 02:00", tz=tz),
            range_ticks=80.0, instrument="MES",
        )
        return AsianSignal(
            date=ar.date, direction=direction, mode=mode,
            entry_price=entry_price,
            entry_time=pd.Timestamp("2024-01-08 09:45", tz=tz),
            asian_range=ar, instrument="MES",
        )

    def test_opposite_stop_breakout_long(self):
        signal = self._make_signal("long", "breakout", 5010.25)
        strategy = AsianRangeStrategy(stop_type="opposite", tp_type="rr", rr_ratio=2.0)
        sl, tp = strategy.compute_sl_tp(signal, 5010.25)
        assert sl == 4990.0  # asian_low
        assert tp == 5010.25 + 2.0 * (5010.25 - 4990.0)  # entry + 2*risk

    def test_multiple_stop_fade_short(self):
        signal = self._make_signal("short", "fade", 5008.0)
        strategy = AsianRangeStrategy(
            mode="fade", stop_type="multiple", stop_multiple=0.5,
            tp_type="opposite",
        )
        sl, tp = strategy.compute_sl_tp(signal, 5008.0)
        range_size = 5010.0 - 4990.0  # 20
        assert sl == 5008.0 + 0.5 * range_size  # 5018.0
        assert tp == 4990.0  # fade short -> opposite boundary = asian_low

    def test_midpoint_tp(self):
        signal = self._make_signal("long", "breakout", 5010.25)
        strategy = AsianRangeStrategy(stop_type="opposite", tp_type="midpoint")
        sl, tp = strategy.compute_sl_tp(signal, 5010.25)
        assert tp == (5010.0 + 4990.0) / 2  # 5000.0
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_strategy_asian.py -v`
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/strategy_asian.py tests/test_strategy_asian.py
git commit -m "feat(asian): add trade simulation with SL/TP computation and tests"
```

---

### Task 4: Backtest Runner and Parameter Grid

**Files:**
- Modify: `src/strategy_asian.py`
- Modify: `tests/test_strategy_asian.py`

- [ ] **Step 1: Implement run_backtest()**

Add as a module-level function:

```python
def run_backtest(
    data: dict[str, pd.DataFrame],
    strategy: AsianRangeStrategy,
    slippage_ticks: float = 1,
    commission: float = 1.24,
) -> list[Trade]:
    """Run backtest across all instruments.

    For each instrument:
    1. Compute Asian ranges
    2. Find signals (breakout or fade)
    3. Simulate each trade bar-by-bar
    """
    all_trades = []
    for instrument, df in data.items():
        tick_size = INSTRUMENTS[instrument]["tick_size"]
        tick_value = INSTRUMENTS[instrument]["tick_value"]

        ranges = strategy.find_asian_ranges(df, instrument, tick_size)
        signals = strategy.find_signals(df, ranges)

        # Precompute day groups for fast bar lookup
        date_col = df.index.normalize()
        day_groups = {date: gdf for date, gdf in df.groupby(date_col)}

        for signal in signals:
            day_data = day_groups.get(signal.date)
            if day_data is None:
                continue
            trade = simulate_trade(
                signal, day_data, strategy, tick_size, tick_value,
                slippage_ticks, commission,
            )
            if trade is not None:
                all_trades.append(trade)

    return all_trades
```

- [ ] **Step 2: Implement build_param_grid()**

Add as a module-level function:

```python
def build_param_grid() -> list[dict]:
    """Generate ~4,050 parameter combinations.

    Respects dependencies:
    - rr_ratio only with tp_type='rr'
    - stop_multiple only with stop_type='multiple'
    - stop_type='opposite' excluded when mode='fade'
    """
    modes = ["breakout", "fade"]
    asian_ends = [time(0, 0), time(1, 0), time(2, 0)]
    trade_starts = [time(8, 0), time(9, 0), time(9, 30)]
    trade_ends = [time(12, 0), time(14, 0), time(16, 0)]
    min_range_ticks_values = [5, 10, 15]
    max_range_ticks_values = [75, None]

    # Exit param combos: (stop_type, stop_multiple, tp_type, rr_ratio)
    exit_combos = []
    for stop_type in ["opposite", "multiple"]:
        stop_mults = [None] if stop_type == "opposite" else [0.5, 1.0]
        for stop_mult in stop_mults:
            for tp_type in ["rr", "opposite", "midpoint"]:
                rr_values = [1.0, 2.0, 3.0] if tp_type == "rr" else [None]
                for rr in rr_values:
                    exit_combos.append((stop_type, stop_mult, tp_type, rr))

    grid = []
    for mode, ae, ts, te, mn, mx in itertools.product(
        modes, asian_ends, trade_starts, trade_ends,
        min_range_ticks_values, max_range_ticks_values,
    ):
        for stop_type, stop_mult, tp_type, rr in exit_combos:
            # Exclude stop_type=opposite when mode=fade
            if mode == "fade" and stop_type == "opposite":
                continue

            grid.append({
                "mode": mode,
                "asian_end": ae,
                "trade_start": ts,
                "trade_end": te,
                "min_range_ticks": mn,
                "max_range_ticks": mx,
                "stop_type": stop_type,
                "tp_type": tp_type,
                "rr_ratio": rr if rr is not None else 2.0,
                "stop_multiple": stop_mult if stop_mult is not None else 0.5,
            })

    return grid
```

- [ ] **Step 3: Write test for grid size**

Add to `tests/test_strategy_asian.py`:

```python
from src.strategy_asian import build_param_grid


class TestParamGrid:
    def test_grid_size(self):
        """Grid should produce exactly 4,050 combinations."""
        grid = build_param_grid()
        assert len(grid) == 4050

    def test_no_fade_opposite_stop(self):
        """No fade combo should have stop_type=opposite."""
        grid = build_param_grid()
        for params in grid:
            if params["mode"] == "fade":
                assert params["stop_type"] != "opposite", (
                    f"Found fade + opposite stop: {params}"
                )
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_strategy_asian.py::TestParamGrid -v`
Expected: Both tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/strategy_asian.py tests/test_strategy_asian.py
git commit -m "feat(asian): add backtest runner and parameter grid builder"
```

---

### Task 5: Grid Search with Precomputation

**Files:**
- Modify: `src/strategy_asian.py`

- [ ] **Step 1: Implement _precompute_asian_ranges()**

Add as a module-level function:

```python
def _precompute_asian_ranges(
    data: dict[str, pd.DataFrame],
    asian_end_values: list[time],
    min_ticks_values: list[float],
    max_ticks_values: list[float | None],
) -> dict[tuple[str, str, float, float | None], list[AsianRange]]:
    """Precompute Asian ranges for each (instrument, asian_end, min_ticks, max_ticks).

    Only 3 asian_end × 2 instruments = 6 raw range computations.
    Range filtering by min/max ticks is cheap so we precompute all combos.
    """
    cache = {}

    for instrument, df in data.items():
        tick_size = INSTRUMENTS[instrument]["tick_size"]

        for ae in asian_end_values:
            # Compute ranges with no range filter (min=0, max=None)
            strategy = AsianRangeStrategy(
                asian_end=ae, min_range_ticks=0, max_range_ticks=None,
            )
            all_ranges = strategy.find_asian_ranges(df, instrument, tick_size)

            # Apply each min/max filter combination
            for mn in min_ticks_values:
                for mx in max_ticks_values:
                    key = (instrument, ae.strftime("%H:%M"), mn, mx)
                    filtered = [
                        r for r in all_ranges
                        if r.range_ticks >= mn
                        and (mx is None or r.range_ticks <= mx)
                    ]
                    cache[key] = filtered

    return cache
```

- [ ] **Step 2: Implement run_grid_search()**

Add as a module-level function:

```python
def run_grid_search(
    data: dict[str, pd.DataFrame],
    param_grid: list[dict],
    progress: bool = False,
) -> pd.DataFrame:
    """Run grid search over parameter combinations.

    Precomputes Asian ranges and day groups to avoid redundant work.
    """
    from .metrics import calculate_metrics

    # Extract unique values for precomputation
    asian_end_values = sorted(
        set(p["asian_end"] for p in param_grid),
        key=lambda t: (t.hour, t.minute),
    )
    min_ticks_values = sorted(set(p["min_range_ticks"] for p in param_grid))
    max_ticks_values = sorted(
        set(p["max_range_ticks"] for p in param_grid),
        key=lambda x: (0, x) if x is not None else (1, 0),
    )

    print(f"  Precomputing Asian ranges for {len(asian_end_values)} end times "
          f"× {len(data)} instruments...")
    range_cache = _precompute_asian_ranges(
        data, asian_end_values, min_ticks_values, max_ticks_values,
    )

    # Precompute day groups
    print("  Precomputing day groups...")
    day_groups = {}
    for instrument, df in data.items():
        day_groups[instrument] = {
            date: gdf for date, gdf in df.groupby(df.index.normalize())
        }

    tz_map = {instrument: df.index.tz for instrument, df in data.items()}
    print(f"  Done. Running {len(param_grid)} parameter combinations...")

    results = []
    iterator = tqdm(param_grid, desc="Grid search") if progress else param_grid

    for params in iterator:
        strategy = AsianRangeStrategy(**params)

        all_trades = []
        for instrument, df in data.items():
            tick_size = INSTRUMENTS[instrument]["tick_size"]
            tick_value = INSTRUMENTS[instrument]["tick_value"]
            tz = tz_map[instrument]

            ae_str = params["asian_end"].strftime("%H:%M")
            key = (instrument, ae_str, params["min_range_ticks"],
                   params["max_range_ticks"])
            ranges = range_cache.get(key, [])

            signals = strategy.find_signals(df, ranges)

            for signal in signals:
                dd = day_groups[instrument].get(signal.date)
                if dd is None:
                    continue
                trade = simulate_trade(
                    signal, dd, strategy, tick_size, tick_value,
                )
                if trade is not None:
                    all_trades.append(trade)

        metrics = calculate_metrics(all_trades)

        results.append({
            "mode": params["mode"],
            "asian_end": params["asian_end"].strftime("%H:%M"),
            "trade_start": params["trade_start"].strftime("%H:%M"),
            "trade_end": params["trade_end"].strftime("%H:%M"),
            "min_range_ticks": params["min_range_ticks"],
            "max_range_ticks": params["max_range_ticks"],
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
```

- [ ] **Step 3: Commit**

```bash
git add src/strategy_asian.py
git commit -m "feat(asian): add grid search with precomputed ranges"
```

---

### Task 6: Walk-Forward Analysis

**Files:**
- Modify: `src/strategy_asian.py`

- [ ] **Step 1: Implement _filter_data_by_date() and run_walk_forward()**

Add as module-level functions:

```python
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
        current = (train_start + pd.DateOffset(months=test_months)).date()

    print(f"  {len(windows)} walk-forward windows")
    wf_results = []

    for wid, (train_start, train_end, test_start, test_end) in enumerate(windows):
        print(f"\n  Window {wid}: train {train_start.date()}->{train_end.date()}, "
              f"test {test_start.date()}->{test_end.date()}")

        train_data = _filter_data_by_date(data, train_start, train_end)
        test_data = _filter_data_by_date(data, test_start, test_end)

        # In-sample grid search
        is_results = run_grid_search(train_data, param_grid)
        if is_results.empty:
            continue

        # Select top N by profit factor (with enough trades)
        qualified = is_results[is_results["total_trades"] >= 20]
        if qualified.empty:
            qualified = is_results
        top_params = qualified.nlargest(top_n, "profit_factor")

        # Evaluate OOS
        for _, row in top_params.iterrows():
            # Reconstruct params
            h, m = map(int, row["asian_end"].split(":"))
            ae = time(h, m)
            h2, m2 = map(int, row["trade_start"].split(":"))
            ts = time(h2, m2)
            h3, m3 = map(int, row["trade_end"].split(":"))
            te = time(h3, m3)
            mx = None if pd.isna(row["max_range_ticks"]) else float(row["max_range_ticks"])

            params = {
                "mode": row["mode"],
                "asian_end": ae,
                "trade_start": ts,
                "trade_end": te,
                "min_range_ticks": row["min_range_ticks"],
                "max_range_ticks": mx,
                "stop_type": row["stop_type"],
                "tp_type": row["tp_type"],
                "rr_ratio": row["rr_ratio"],
                "stop_multiple": row["stop_multiple"],
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
```

- [ ] **Step 2: Commit**

```bash
git add src/strategy_asian.py
git commit -m "feat(asian): add walk-forward analysis"
```

---

### Task 7: Report Generation

**Files:**
- Modify: `src/strategy_asian.py`

- [ ] **Step 1: Implement generate_report()**

Add as a module-level function:

```python
def generate_report(
    grid_results: pd.DataFrame | None,
    wf_results: pd.DataFrame | None,
    baseline: dict | None,
) -> str:
    """Generate comprehensive text report."""
    lines = []
    SEP = "=" * 70
    lines.append(SEP)
    lines.append("  ASIAN RANGE STRATEGY — REPORT")
    lines.append(SEP)

    # 1. Baseline
    if baseline:
        lines.append("\n--- 1. BASELINE (Default Parameters) ---")
        for key, val in baseline.items():
            if isinstance(val, float):
                lines.append(f"  {key:<25} {val:>15.4f}")
            else:
                lines.append(f"  {key:<25} {val:>15}")

    # 2. Grid search summary
    if grid_results is not None and not grid_results.empty:
        lines.append(f"\n--- 2. GRID SEARCH ({len(grid_results)} combinations) ---")

        profitable = grid_results[grid_results["profit_factor"] > 1.0]
        lines.append(f"  Profitable (PF > 1.0): {len(profitable)} "
                      f"({len(profitable)/len(grid_results):.1%})")
        lines.append(f"  Mean PF: {grid_results['profit_factor'].mean():.3f}")
        lines.append(f"  Max PF: {grid_results['profit_factor'].max():.3f}")
        lines.append(f"  Mean PnL: ${grid_results['total_pnl'].mean():,.0f}")

        # 3. Top 10
        lines.append("\n--- 3. TOP 10 BY TOTAL PnL ---")
        top = grid_results.nlargest(10, "total_pnl")
        header = (f"  {'mode':<9} {'ae':>5} {'ts':>5} {'te':>5} {'stop':<8} "
                  f"{'tp':<8} {'trades':>6} {'WR':>6} {'PF':>6} {'PnL':>10}")
        lines.append(header)
        lines.append("  " + "-" * 76)
        for _, r in top.iterrows():
            lines.append(
                f"  {r['mode']:<9} {r['asian_end']:>5} {r['trade_start']:>5} "
                f"{r['trade_end']:>5} {r['stop_type']:<8} {r['tp_type']:<8} "
                f"{r['total_trades']:>6.0f} {r['win_rate']:>5.1%} "
                f"{r['profit_factor']:>6.3f} ${r['total_pnl']:>9,.0f}"
            )

        # 4. Parameter sensitivity
        lines.append("\n--- 4. PARAMETER SENSITIVITY (mean PF) ---")
        for param in ["mode", "asian_end", "trade_start", "trade_end",
                       "min_range_ticks", "max_range_ticks",
                       "stop_type", "tp_type"]:
            lines.append(f"\n  By {param}:")
            grouped = grid_results.groupby(param, dropna=False).agg(
                mean_pf=("profit_factor", "mean"),
                mean_pnl=("total_pnl", "mean"),
            )
            for val, row in grouped.iterrows():
                val_str = str(val) if pd.notna(val) else "None"
                lines.append(f"    {val_str:<12} PF={row['mean_pf']:.3f}  "
                              f"PnL=${row['mean_pnl']:>8,.0f}")

        # 5. Mode comparison
        lines.append("\n--- 5. MODE COMPARISON — Breakout vs Fade ---")
        for mode_val in ["breakout", "fade"]:
            mode_data = grid_results[grid_results["mode"] == mode_val]
            if mode_data.empty:
                continue
            mp = mode_data[mode_data["profit_factor"] > 1.0]
            lines.append(
                f"  {mode_val:<10}: {len(mode_data)} combos, "
                f"{len(mp)} profitable ({len(mp)/len(mode_data):.1%}), "
                f"mean PF={mode_data['profit_factor'].mean():.3f}, "
                f"best PnL=${mode_data['total_pnl'].max():,.0f}"
            )

    # 6. Walk-forward
    if wf_results is not None and not wf_results.empty:
        lines.append(f"\n--- 6. WALK-FORWARD ({wf_results['window_id'].nunique()} windows) ---")
        lines.append(f"  Total OOS PnL: ${wf_results['oos_pnl'].sum():,.0f}")
        lines.append(f"  Mean OOS PF: {wf_results['oos_pf'].mean():.3f}")
        lines.append(f"  Mean IS PF: {wf_results['is_pf'].mean():.3f}")
        oos_positive = (wf_results['oos_pnl'] > 0).sum()
        lines.append(f"  OOS positive: {oos_positive}/{len(wf_results)} "
                      f"({oos_positive/len(wf_results):.1%})")
        degradation = 1 - wf_results['oos_pf'].mean() / max(wf_results['is_pf'].mean(), 1e-9)
        lines.append(f"  IS->OOS degradation: {degradation:.1%}")

    # 7. Instrument analysis (from grid top 5)
    if grid_results is not None and not grid_results.empty:
        lines.append("\n--- 7. INSTRUMENT ANALYSIS (Top 5 by PnL) ---")
        if "pnl_long" in grid_results.columns:
            top5 = grid_results.nlargest(5, "total_pnl")
            lines.append(f"  {'mode':<9} {'PnL_L':>10} {'PnL_S':>10}")
            lines.append("  " + "-" * 30)
            for _, r in top5.iterrows():
                lines.append(
                    f"  {r['mode']:<9} "
                    f"${r.get('pnl_long', 0):>9,.0f} "
                    f"${r.get('pnl_short', 0):>9,.0f}"
                )

    lines.append("\n" + SEP)
    return "\n".join(lines)
```

- [ ] **Step 2: Commit**

```bash
git add src/strategy_asian.py
git commit -m "feat(asian): add report generation"
```

---

### Task 8: CLI Script and Baseline Run

**Files:**
- Create: `run_asian.py`

- [ ] **Step 1: Create run_asian.py**

Following the `run_gapfade.py` pattern:

```python
"""CLI entry point for Asian Range strategy.

Usage:
    python run_asian.py                    # baseline (default params)
    python run_asian.py --optimize         # grid search (~4,050 combos)
    python run_asian.py --walk-forward     # walk-forward analysis
    python run_asian.py --report           # generate report
"""

import argparse
import json
from pathlib import Path

from src.data_loader import load_processed
from src.metrics import calculate_metrics
from src.strategy_asian import (
    AsianRangeStrategy,
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
    """Run baseline with default parameters."""
    print("Loading data...")
    data = {}
    for instrument in ["MES", "MNQ"]:
        data[instrument] = get_data(instrument)
        print(f"  {instrument}: {len(data[instrument]):,} bars")

    print("\nRunning baseline (default parameters: breakout, midnight, opposite stop, RR=2.0)...")
    strategy = AsianRangeStrategy()  # all defaults
    trades = run_backtest(data, strategy)
    print(f"  {len(trades)} trades")

    metrics = calculate_metrics(trades)

    # Print summary
    print(f"\n{'='*50}")
    print("  ASIAN RANGE — BASELINE")
    print(f"{'='*50}")
    for key in ["total_trades", "win_rate", "profit_factor", "total_pnl",
                "sharpe_ratio", "max_drawdown", "avg_pnl_per_trade"]:
        val = metrics.get(key, 0)
        if isinstance(val, float):
            print(f"  {key:<25} {val:>12.4f}")
        else:
            print(f"  {key:<25} {val:>12}")
    print(f"{'='*50}")

    if "by_instrument" in metrics:
        print("\n  Per instrument:")
        print(metrics["by_instrument"].to_string())

    if "exit_reasons" in metrics:
        print("\n  Exit reasons:")
        for reason, count in metrics["exit_reasons"].items():
            pct = count / metrics["total_trades"] * 100
            print(f"    {reason:<16} : {count:>5}  ({pct:.1f}%)")

    print(f"\n  Long  trades: {metrics.get('trades_long', 0):>5}  "
          f"PnL: ${metrics.get('pnl_long', 0):>10,.2f}  "
          f"WR: {metrics.get('win_rate_long', 0):.1%}")
    print(f"  Short trades: {metrics.get('trades_short', 0):>5}  "
          f"PnL: ${metrics.get('pnl_short', 0):>10,.2f}  "
          f"WR: {metrics.get('win_rate_short', 0):.1%}")

    # Save
    RESULTS_DIR.mkdir(exist_ok=True)
    baseline = {k: metrics.get(k, 0) for k in [
        "total_trades", "win_rate", "profit_factor", "total_pnl",
        "sharpe_ratio", "max_drawdown", "avg_pnl_per_trade",
    ]}
    path = RESULTS_DIR / "asian_baseline.json"
    with open(path, "w") as f:
        json.dump(baseline, f, indent=2)
    print(f"\n  Saved to {path}")


def run_optimize(args):
    """Run grid search over ~4,050 combinations."""
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
    path = RESULTS_DIR / "asian_grid.csv"
    results.to_csv(path, index=False)
    print(f"\nResults saved to {path}")

    # Show top 10
    top = results.nlargest(10, "total_pnl")[
        ["mode", "asian_end", "trade_start", "trade_end",
         "stop_type", "tp_type",
         "total_trades", "win_rate", "profit_factor", "total_pnl", "sharpe_ratio"]
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
    path = RESULTS_DIR / "asian_wf.csv"
    wf_results.to_csv(path, index=False)
    print(f"\nResults saved to {path}")
    print(f"  Shape: {wf_results.shape}")


def run_report(args):
    """Generate report from saved results."""
    import pandas as pd

    grid_path = RESULTS_DIR / "asian_grid.csv"
    wf_path = RESULTS_DIR / "asian_wf.csv"
    baseline_path = RESULTS_DIR / "asian_baseline.json"

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

    report = generate_report(grid_results, wf_results, baseline)
    print("\n")
    print(report)

    # Save report
    RESULTS_DIR.mkdir(exist_ok=True)
    report_path = RESULTS_DIR / "asian_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n  Report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Asian Range Strategy")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--optimize", action="store_true",
                       help="Run grid search (~4,050 combos)")
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
```

- [ ] **Step 2: Commit**

```bash
git add run_asian.py
git commit -m "feat(asian): add CLI entry point"
```

- [ ] **Step 3: Run baseline on real data**

Run: `python run_asian.py`
Expected: Loads MES/MNQ data, computes Asian ranges, finds breakout signals, simulates trades, prints baseline metrics. Verify trade count is reasonable (expect ~1,000-2,000 trades over 7 years, 2 instruments).

- [ ] **Step 4: Run all tests**

Run: `python -m pytest tests/test_strategy_asian.py -v`
Expected: All tests pass.

- [ ] **Step 5: Final commit with baseline results**

```bash
git add -A
git commit -m "feat(asian): complete Asian Range Strategy implementation

Strategy #4 with breakout and fade modes on overnight Asian range.
~4,050 parameter combos for grid search + walk-forward validation.
Baseline run completed."
```

---

## Execution Order

Tasks 1-4 must be sequential (each builds on the previous).
Tasks 5-7 must be sequential (grid search → walk-forward → report).
Task 8 depends on all previous tasks.

```
Task 1 → Task 2 → Task 3 → Task 4 → Task 5 → Task 6 → Task 7 → Task 8
```

All tasks are in one file (`src/strategy_asian.py`) so there's no opportunity for parallel work.
