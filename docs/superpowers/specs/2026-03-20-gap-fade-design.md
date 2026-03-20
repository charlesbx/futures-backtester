# Overnight Gap Fade Strategy — Design Spec

## Goal

Backtest the Overnight Gap Fade strategy on MES/MNQ micro E-mini futures. Gaps between the previous session close and the current session open tend to fill. Fade the gap (buy gap-downs, sell gap-ups), targeting partial fill with a stop based on gap widening.

## Academic References

- "Statistical Arbitrage with Mean-Reverting Overnight Price Gaps on S&P 500" — MDPI (2019): 89% win rate on SPY gaps of -0.15% to -0.6%
- "Overnight-Intraday Reversal Everywhere" — Della Corte & Kosowski: overnight return reverses intraday across global markets
- Gap fill rates: gaps < 0.5% fill 59-61%, gaps > 1% fill only 28-33%

## Strategy Rules

### Gap Computation
1. Record **previous trading day's close** at 4:00 PM ET (use most recent prior trading day that has a 4:00 PM bar; skip weekends/holidays)
2. Record **gap measurement price** at `gap_measure_time` (default 9:30 AM ET)
3. Compute gap: `gap_pct = (measure_price - prev_close) / prev_close`

### Signal
- If `gap_pct < -min_gap_pct`: gap down → LONG (fade, expect fill upward)
- If `gap_pct > +min_gap_pct`: gap up → SHORT (fade, expect fill downward)
- If `abs(gap_pct) <= min_gap_pct`: no trade

### Entry
- Market entry at `gap_measure_time` in the fade direction
- 1-tick slippage applied

### Exit (priority order)
1. **Stop-loss hit** (checked first per bar): if `stop_gap_multiple` is set, stop = entry_price - `stop_gap_multiple * abs(gap_size)` for longs, or entry_price + `stop_gap_multiple * abs(gap_size)` for shorts. The gap widens beyond the stop threshold.
2. **Fill target hit**: `fill_target = measure_price + fill_pct * (prev_close - measure_price)`. This is a linear interpolation: at `fill_pct=0.75`, the target is 75% of the way from entry back to previous close. Example: prev_close=5000, measure_price=4950 → target = 4950 + 0.75 * 50 = 4987.50.
3. **Session end**: forced exit at 4:00 PM ET.

### Position Management
- One trade per day per instrument
- No re-entry after exit

### Instruments
- MES (tick_size=0.25, tick_value=$1.25)
- MNQ (tick_size=0.25, tick_value=$0.50)
- Commission: $1.24 per round-trip
- Slippage: 1 tick per trade

### Edge Cases
- **No previous close available** (first day of data): skip
- **Gap measurement bar missing** (holiday, early close): skip (return None)
- **Gap size is zero**: no trade (abs(0) <= min_gap_pct)

## Parameters

### Baseline
| Parameter | Value |
|-----------|-------|
| gap_measure_time | 9:30 AM ET |
| min_gap_pct | 0.10% (0.001) |
| fill_pct | 0.75 |
| stop_gap_multiple | None (no stop) |

### Optimization Grid (180 combinations)
| Parameter | Values | Count |
|-----------|--------|-------|
| gap_measure_time | 9:15, 9:30, 9:31 | 3 |
| min_gap_pct | 0.0005, 0.001, 0.0015, 0.0025, 0.005 | 5 |
| fill_pct | 0.50, 0.75, 1.00 | 3 |
| stop_gap_multiple | None, 0.25, 0.50, 1.00 | 4 |

Total: 3 × 5 × 3 × 4 = 180

### Walk-Forward
- Rolling windows: 24-month train, 6-month test, 6-month step
- Select top 10 by profit factor in-sample
- Evaluate out-of-sample

## Module Structure

### New Files
| File | Responsibility |
|------|----------------|
| `src/strategy_gapfade.py` | Strategy class, signal detection, trade simulation, grid search, walk-forward, report |
| `run_gapfade.py` | CLI: baseline, --optimize, --walk-forward, --report |

### Reused (No Modifications)
| File | What's Reused |
|------|---------------|
| `src/data_loader.py` | `load_processed()`, `INSTRUMENTS` dict |
| `src/metrics.py` | `calculate_metrics()` |

## `src/strategy_gapfade.py` — Components

### `GapSignal` dataclass
- date, direction ("long"/"short"), gap_pct (float), prev_close (float), measure_price (float), entry_time (Timestamp), instrument (str)

### `Trade` dataclass
- Same fields as intmom Trade for `calculate_metrics()` compatibility
- entry_time, exit_time, direction, entry_price, exit_price, stop_loss, take_profit, pnl_ticks, pnl_dollars, exit_reason ("tp"/"sl"/"session_end"), session (0), instrument, date, range_high (0.0), range_low (0.0)

### `GapFadeStrategy` class
- `__init__(gap_measure_time, min_gap_pct, fill_pct, stop_gap_multiple)`
- `find_signals(df, instrument)` → list[GapSignal]
  - Groups by date, finds previous close (4:00 PM bar), current open at gap_measure_time
  - Computes gap_pct, filters by min_gap_pct, generates signal

### `simulate_trade(signal, day_data, strategy, tick_size, tick_value, slippage_ticks, commission)` → Trade | None
- Entry at gap_measure_time bar open + slippage
- Compute fill target and stop levels from prev_close and gap size
- Walk bar-by-bar (skip entry bar) checking stop then target
- Exit at session end (4:00 PM) if neither hit

### `run_backtest(data, strategy)` → list[Trade]
- For each instrument, find signals, simulate each trade

### Performance Optimization
- Precompute gap data once: for each (instrument, gap_measure_time), build a DataFrame of (date, prev_close, measure_price, gap_pct)
- Precompute day groups: `{date: day_data}` per instrument
- Grid search reuses precomputed data — only fill_pct, stop_gap_multiple, and min_gap_pct vary per combo
- Expected runtime: ~5-10 minutes for 180 combos (vs intmom's 37 min for 1,024)

### `build_param_grid()` → list[dict] (180 combinations)
### `run_grid_search(data, param_grid)` → DataFrame
### `run_walk_forward(data, param_grid)` → DataFrame
### `generate_report(grid_results, wf_results, baseline)` → str

## `run_gapfade.py` — CLI

```
python run_gapfade.py                    # baseline
python run_gapfade.py --optimize         # grid search (180 combos)
python run_gapfade.py --walk-forward     # walk-forward validation
python run_gapfade.py --report           # generate report
```

## Output Files

| File | Format | Content |
|------|--------|---------|
| `results/gapfade_baseline.json` | JSON | Baseline metrics |
| `results/gapfade_grid.csv` | CSV | 180 rows: params + metrics |
| `results/gapfade_walk_forward.csv` | CSV | OOS results per window |

## Success Criteria

| Metric | Threshold | Meaning |
|--------|-----------|---------|
| Baseline PF | > 1.0 | Gap fade signal exists on MES/MNQ |
| Best grid PF | > 1.2 | Meaningful edge with tuning |
| Best grid Sharpe | > 0.8 | Acceptable risk-adjusted return |
| Walk-forward OOS PF | > 1.1 | Edge survives out-of-sample |

## Biases to Avoid

- **Look-ahead bias**: previous close uses only yesterday's 4 PM bar. Gap measurement uses only the measurement time bar. No future data.
- **Overfitting**: 180 combos is a small grid. Walk-forward provides OOS validation.
- **Transaction costs**: 1-tick slippage + $1.24 commission per trade.
- **Survivorship of gap fill rates**: academic studies on SPY may not transfer to futures. The baseline test will verify this.

## Dependencies

No new pip dependencies. Uses pandas, numpy (already installed).
