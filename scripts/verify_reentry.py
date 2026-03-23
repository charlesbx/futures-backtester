#!/usr/bin/env python3
"""Verify that the range_reentry proxy (minute % 15 == 14) is equivalent
to true 15-min resampling relative to range_end for default strategy parameters.

Decision criteria:
  - Divergence <= 1%: proxy is acceptable
  - Divergence > 1%: proxy needs fixing (do NOT fix here — handled in Step 1)

Usage:
    python scripts/verify_reentry.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.data_loader import load_processed, INSTRUMENTS
from src.strategies.srs import SRSStrategy, SessionRange

PROCESSED_DIR = Path("data/processed")


# ---------------------------------------------------------------------------
# Two independent implementations of the reentry detection logic
# ---------------------------------------------------------------------------

def find_reentry_proxy(remaining: pd.DataFrame, range_info: SessionRange):
    """Current proxy: last minute of a calendar-aligned 15-min block (minute % 15 == 14)."""
    is_15min_close = remaining.index.minute % 15 == 14
    inside_range = (
        (remaining["close"] >= range_info.range_low)
        & (remaining["close"] <= range_info.range_high)
    )
    hits = remaining.index[is_15min_close & inside_range]
    return hits[0] if len(hits) > 0 else None


def find_reentry_reference(remaining: pd.DataFrame, range_info: SessionRange):
    """Reference: true 15-min resample relative to range_end, check close inside range.

    Uses pandas resample with origin=range_end to build 15-min candles aligned
    to the start of trading (not to the clock).  The "close" of each candle is
    the last available 1-min bar inside that bin.
    """
    if remaining.empty:
        return None

    closes = (
        remaining["close"]
        .resample("15min", origin=range_info.range_end)
        .last()
        .dropna()
    )

    inside = closes[
        (closes >= range_info.range_low) & (closes <= range_info.range_high)
    ]
    if inside.empty:
        return None

    # For each 15-min bin whose close is inside the range, find the actual
    # 1-min bar timestamp that produced that close value.
    for bin_start in inside.index:
        bin_end = bin_start + pd.Timedelta(minutes=15)
        bin_bars = remaining.loc[
            (remaining.index >= bin_start) & (remaining.index < bin_end)
        ]
        if not bin_bars.empty:
            return bin_bars.index[-1]

    return None


# ---------------------------------------------------------------------------
# Per-range simulation (breakout detection + dual exit comparison)
# ---------------------------------------------------------------------------

def compare_exits_for_range(
    df: pd.DataFrame,
    range_info: SessionRange,
    strategy: SRSStrategy,
):
    """Return (proxy_exit_reason, reference_exit_reason) for one session range.

    Returns None if no breakout occurs (no trade → nothing to compare).
    """
    mask = (df.index >= range_info.range_end) & (df.index < range_info.session_end)
    session_bars = df.loc[mask]

    if session_bars.empty:
        return None

    # Breakout detection (mirrors Backtester._detect_breakout)
    long_breaks = session_bars.index[session_bars["high"] > range_info.range_high]
    short_breaks = session_bars.index[session_bars["low"] < range_info.range_low]

    first_long = long_breaks[0] if len(long_breaks) > 0 else None
    first_short = short_breaks[0] if len(short_breaks) > 0 else None

    if first_long is None and first_short is None:
        return None  # No breakout — no trade

    if first_long is not None and first_short is not None:
        if first_long <= first_short:
            direction, entry_time = "long", first_long
        else:
            direction, entry_time = "short", first_short
    elif first_long is not None:
        direction, entry_time = "long", first_long
    else:
        direction, entry_time = "short", first_short

    params = strategy.compute_trade_params(range_info, direction)
    stop_loss = params["stop_loss"]
    take_profit = params["take_profit"]

    remaining = session_bars.loc[session_bars.index > entry_time]
    if remaining.empty:
        return ("session_end", "session_end")

    # SL / TP hits (same for both methods)
    if direction == "long":
        sl_hits = remaining.index[remaining["low"] <= stop_loss]
        tp_hits = remaining.index[remaining["high"] >= take_profit]
    else:
        sl_hits = remaining.index[remaining["high"] >= stop_loss]
        tp_hits = remaining.index[remaining["low"] <= take_profit]

    first_sl = sl_hits[0] if len(sl_hits) > 0 else None
    first_tp = tp_hits[0] if len(tp_hits) > 0 else None

    proxy_reentry = find_reentry_proxy(remaining, range_info)
    ref_reentry = find_reentry_reference(remaining, range_info)

    def resolve_exit(reentry_hit):
        # Priority: SL(0) > TP(1) > reentry(2) > session_end
        candidates = []
        if first_sl is not None:
            candidates.append((first_sl, "sl", 0))
        if first_tp is not None:
            candidates.append((first_tp, "tp", 1))
        if reentry_hit is not None:
            candidates.append((reentry_hit, "range_reentry", 2))
        if not candidates:
            return "session_end"
        candidates.sort(key=lambda x: (x[0], x[2]))
        return candidates[0][1]

    return resolve_exit(proxy_reentry), resolve_exit(ref_reentry)


# ---------------------------------------------------------------------------
# Per-instrument verification
# ---------------------------------------------------------------------------

def verify_instrument(instrument: str) -> tuple[int, int]:
    """Return (total_trades, divergent_trades) for one instrument."""
    cache = PROCESSED_DIR / f"{instrument}_1m.parquet"
    if not cache.exists():
        print(f"  {instrument}: No processed data found at {cache}")
        return 0, 0

    print(f"\nLoading {instrument}...")
    df = load_processed(cache)
    print(f"  {len(df):,} bars | {df.index.min()} → {df.index.max()}")

    strategy = SRSStrategy(rr_ratio=2.25, range_minutes=30)
    ranges = strategy.find_session_ranges(df, instrument)
    print(f"  {len(ranges)} session ranges")

    total = 0
    divergent = 0

    for range_info in ranges:
        result = compare_exits_for_range(df, range_info, strategy)
        if result is None:
            continue  # No breakout → no trade
        proxy_exit, ref_exit = result
        total += 1
        if proxy_exit != ref_exit:
            divergent += 1

    return total, divergent


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Verification: range_reentry proxy vs. true 15-min resample")
    print("=" * 60)

    grand_total = 0
    grand_divergent = 0

    for instrument in ["MES", "MNQ"]:
        total, divergent = verify_instrument(instrument)
        grand_total += total
        grand_divergent += divergent
        if total > 0:
            rate = divergent / total * 100
            print(f"  {instrument}: {divergent}/{total} divergent ({rate:.2f}%)")
        else:
            print(f"  {instrument}: No trades found")

    print()
    print("=" * 60)

    if grand_total == 0:
        print("  No trades found. Ensure processed data exists in data/processed/")
        print("=" * 60)
        return

    overall_rate = grand_divergent / grand_total * 100
    print(f"  OVERALL: {grand_divergent}/{grand_total} divergent ({overall_rate:.2f}%)")
    print()

    if overall_rate <= 1.0:
        print(f"  CONCLUSION: Proxy is ACCEPTABLE (divergence {overall_rate:.2f}% <= 1%)")
        print()
        print("  Explanation: With range_minutes=30 and sessions starting at :00,")
        print("  range_end is always at :30. True 15-min candle closes from :30 land")
        print("  at :44, :59, :14, :29 — all of which satisfy (minute % 15 == 14).")
        print("  The proxy is mathematically exact for these default parameters.")
        print()
        print("  NOTE: The proxy would diverge if range_minutes were changed to a value")
        print("  not divisible by 15, or if sessions started at non-:00 offsets.")
    else:
        print(f"  CONCLUSION: Proxy NEEDS FIXING (divergence {overall_rate:.2f}% > 1%)")
        print("  Step 1 should implement the corrected 15-min resample approach.")

    print("=" * 60)


if __name__ == "__main__":
    main()
