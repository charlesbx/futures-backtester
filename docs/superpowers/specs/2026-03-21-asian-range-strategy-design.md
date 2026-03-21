# Asian Range Strategy — Design Spec

**Date**: 2026-03-21
**Status**: Approved
**Strategy #4** in the SRS Backtester project

## Overview

Test a strategy based on the Asian session range (overnight high/low) on MES and MNQ micro E-mini futures. The Asian range is the high and low formed during the CME Globex overnight session (6 PM ET onward). Two modes are tested: **breakout** (trade in the direction of the range break) and **fade** (trade the rejection back into the range).

## Motivation

Three previous strategies tested (SRS breakout, Intraday Momentum, Overnight Gap Fade) all failed to produce a tradeable edge. Key learnings:
- Breakout continuation is weak on liquid index futures
- Exit management matters more than entry
- High win-rate strategies (Gap Fade 67%) still fail if per-trade edge is too thin

The Asian range is a widely-used concept in futures trading. The overnight session establishes a range that acts as support/resistance during the US session. Testing both breakout and fade on the same range gives a direct comparison of trend-following vs. mean-reversion on these instruments.

## 1. Asian Range Computation

### Time Window

- **asian_start**: fixed at 6:00 PM ET (CME Globex open)
- **asian_end**: parameter — `[12:00 AM, 1:00 AM, 2:00 AM]` ET

The range spans midnight: Monday's Asian range starts Sunday 6 PM and ends Monday 12-2 AM. The range is anchored to the US trading date it precedes.

### Implementation Note: Cross-Midnight Query

All existing strategies group data by `df.index.normalize()` which splits at midnight. The Asian range spans two calendar dates (6 PM on D-1, ending at 12-2 AM on D). Implementation should use absolute timestamps rather than day groups: for each US trading date D, build the window as `pd.Timestamp(D-1, 18:00, tz=tz)` to `pd.Timestamp(D, asian_end, tz=tz)` and filter the full DataFrame directly. Alternatively, iterate over consecutive date pairs from the day-group dict and concatenate bars from the evening of D-1 with early-morning bars of D.

### DST Handling

ET = UTC-5 (EST, winter) / UTC-4 (EDT, summer). Data is already in Eastern Time with automatic DST handling, so time comparisons work directly.

### Range Calculation

For each US trading day:
1. Extract all 1-min bars between `asian_start` (previous evening) and `asian_end`
2. `asian_high` = max(high) across those bars
3. `asian_low` = min(low) across those bars
4. `range_ticks` = (asian_high - asian_low) / tick_size

### Validation Rules

- Minimum 5 bars in the range period (data integrity)
- `asian_high > asian_low` (skip flat sessions)
- `range_ticks >= min_range_ticks` parameter
- `range_ticks <= max_range_ticks` parameter (if set)

### Data Structure

```python
@dataclass
class AsianRange:
    date: pd.Timestamp          # US trading date this range precedes
    asian_high: float
    asian_low: float
    asian_start: pd.Timestamp
    asian_end: pd.Timestamp
    range_ticks: float          # (high - low) / tick_size
    instrument: str
```

## 2. Entry Logic

### Breakout Mode

During the US trading window (`trade_start` to `trade_end`), walk bar-by-bar:

1. **Long**: first bar where `high > asian_high` — enter at `asian_high + slippage`
2. **Short**: first bar where `low < asian_low` — enter at `asian_low - slippage`
3. Same-bar tiebreaker: direction of open relative to range midpoint (same as SRS)
4. One trade per day — first breakout only, no re-entry

### Fade Mode

During the US trading window, walk bar-by-bar:

1. **Short (fade the high)**: bar `high >= asian_high` AND bar `close < asian_high` — enter short at bar close
2. **Long (fade the low)**: bar `low <= asian_low` AND bar `close > asian_low` — enter long at bar close
3. One trade per day — first rejection only

Fade uses bar close as entry price because the rejection is only confirmed at bar close.

### Trade Window Parameters

- `trade_start`: `[8:00 AM, 9:00 AM, 9:30 AM]` ET
- `trade_end`: `[12:00 PM, 2:00 PM, 4:00 PM]` ET

## 3. Exit Logic

### Stop-Loss

Parameter `stop_type`:

- **`opposite`** (**breakout mode only**): stop at the opposite side of the Asian range
  - Long: SL = `asian_low`
  - Short: SL = `asian_high`
  - **Not used with fade mode** — fade entries are near the boundary, so the opposite-side stop would be at essentially the same price as entry, producing near-zero risk distance. Excluded from the grid.
- **`multiple`**: stop at `entry +/- stop_multiple * range_size`
  - Parameter `stop_multiple`: `[0.5, 1.0]`

### Take-Profit

Parameter `tp_type`:

- **`rr`**: fixed risk/reward — `tp = entry +/- rr_ratio * risk_distance`
  - Parameter `rr_ratio`: `[1.0, 2.0, 3.0]`
- **`opposite`**: target depends on mode and direction:

  | Mode + Direction | TP (`opposite`) |
  |---|---|
  | Breakout long | `asian_high + range_size` (measured move) |
  | Breakout short | `asian_low - range_size` (measured move) |
  | Fade long (from low) | `asian_high` (literal opposite boundary) |
  | Fade short (from high) | `asian_low` (literal opposite boundary) |

  Note: breakout uses a measured-move target (one range beyond the boundary), while fade targets the literal opposite boundary (mean reversion to other side).

- **`midpoint`**: target range midpoint = `(asian_high + asian_low) / 2`

### Forced Exit

If neither TP nor SL hit by `trade_end`, exit at bar close.

### Priority on Same Bar

SL checked before TP (conservative, consistent with existing strategies).

### Parameter Dependencies

- `tp_type=rr` uses `rr_ratio`; `tp_type=opposite|midpoint` ignores it
- `stop_type=multiple` uses `stop_multiple`; `stop_type=opposite` ignores it
- `stop_type=opposite` is excluded when `mode=fade`

## 4. Parameter Grid

| Parameter | Values | Count |
|-----------|--------|-------|
| `mode` | `breakout`, `fade` | 2 |
| `asian_end` | `00:00`, `01:00`, `02:00` | 3 |
| `trade_start` | `08:00`, `09:00`, `09:30` | 3 |
| `trade_end` | `12:00`, `14:00`, `16:00` | 3 |
| `min_range_ticks` | `5`, `10`, `15` | 3 |
| `max_range_ticks` | `75`, `None` | 2 |
| `stop_type` | `opposite` (breakout only), `multiple` | 2 |
| `tp_type` | `rr`, `opposite`, `midpoint` | 3 |
| `rr_ratio` | `1.0`, `2.0`, `3.0` | 3 (only with tp_type=rr) |
| `stop_multiple` | `0.5`, `1.0` | 2 (only with stop_type=multiple) |

**Effective total: ~4,050 combinations**

Base per mode: 3 (asian_end) × 3 (trade_start) × 3 (trade_end) × 3 (min_range) × 2 (max_range) = **162**

Breakout (all stop types):
- stop=opposite + tp=rr: 162 × 3 = 486
- stop=opposite + tp=opposite|midpoint: 162 × 2 = 324
- stop=multiple + tp=rr: 162 × 2 × 3 = 972
- stop=multiple + tp=opposite|midpoint: 162 × 2 × 2 = 648
- **Breakout subtotal: 2,430**

Fade (stop=multiple only, stop=opposite excluded):
- stop=multiple + tp=rr: 162 × 2 × 3 = 972
- stop=multiple + tp=opposite|midpoint: 162 × 2 × 2 = 648
- **Fade subtotal: 1,620**

**Grand total: 2,430 + 1,620 = 4,050**

### Precomputation Strategy

- Asian ranges: 3 `asian_end` values × 2 instruments = **6 precomputations**
- Day groups: precomputed once per instrument for trade simulation
- This avoids recomputing ranges for each of the 4,050 parameter combos

## 5. Module Structure

### File: `src/strategy_asian.py`

Self-contained module following the `strategy_gapfade.py` pattern:

```
AsianRange (dataclass)          — range data for one trading day
AsianSignal (dataclass)         — entry signal with direction, mode, prices
Trade (dataclass)               — executed trade, compatible with metrics.calculate_metrics()

AsianRangeStrategy (class)      — holds all parameters
  find_asian_ranges()           — compute Asian high/low per day, apply filters
  find_signals()                — detect breakout or fade entries

simulate_trade()                — bar-by-bar exit simulation (SL/TP/session end)
run_backtest()                  — loop over instruments and signals
build_param_grid()              — generate ~4,050 valid combos
_precompute_asian_ranges()      — cache ranges per (instrument, asian_end)
run_grid_search()               — grid search with precomputation
run_walk_forward()              — rolling train/test OOS validation
generate_report()               — text report
```

### Run Script: `run_asian.py`

CLI interface:
- `--baseline`: run with default params, print report
- `--grid`: run grid search, save to `results/`
- `--walk-forward`: run walk-forward analysis
- `--report`: generate report from saved results

### Shared Infrastructure

- `data_loader.py`: no changes (already provides 24h data in ET)
- `metrics.py`: no changes (already generic)

### Default Baseline Parameters

- `mode=breakout`
- `asian_end=00:00` (midnight)
- `trade_start=09:30`, `trade_end=16:00`
- `min_range_ticks=10`, `max_range_ticks=None`
- `stop_type=opposite`, `tp_type=rr`, `rr_ratio=2.0`

## 6. Walk-Forward Configuration

- **Train**: 24 months
- **Test**: 6 months
- **Step**: 6 months (non-overlapping test windows)
- **Selection**: top 10 by profit factor in-sample
- **Data period**: May 2019 — March 2026 (~7 years) → ~7-8 windows

## 7. Report Sections

1. **Baseline** — default params metrics
2. **Grid Search Summary** — total combos, % profitable, best PF
3. **Top 10 by PnL** — with all key metrics
4. **Parameter Sensitivity** — mean PF by each parameter
5. **Mode Comparison** — breakout vs fade head-to-head (trades, WR, PF, PnL)
6. **Walk-Forward OOS** — IS vs OOS degradation, OOS positive rate
7. **Instrument Analysis** — MES vs MNQ breakdown for top combos

## 8. Results Directory

Files stored in `results/` with `asian_` prefix, consistent with existing strategies:

```
results/
├── asian_baseline.json
├── asian_grid.csv
├── asian_wf.csv
└── asian_report.txt
```

## 9. Transaction Costs

Consistent with existing strategies:
- **Slippage**: 1 tick per side (default)
- **Commission**: $1.24 round-trip (NinjaTrader micro futures)
- **Tick sizes**: MES = 0.25 ($1.25/tick), MNQ = 0.25 ($0.50/tick)
