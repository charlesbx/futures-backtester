# Intraday Momentum Strategy — Design Spec

## Goal

Backtest the Intraday Momentum strategy (Gao, Han, Li, Zhou 2018) on MES/MNQ micro E-mini futures using existing 1-minute data. Validate the published signal, optimize parameters, and assess out-of-sample robustness via walk-forward analysis.

## Academic Reference

**Paper:** "Market Intraday Momentum" — Gao, Han, Li, Zhou — *Journal of Financial Economics* (2018)

**Core finding:** The first 30 minutes' return predicts the last 30 minutes' return. The mechanism is hedging demand from options market makers creating predictable closing flows.

**Confirmation:** "Intraday time series momentum: Global evidence and links to market characteristics" (Reading University) confirmed the effect across international futures markets.

## Strategy Rules

### Signal
1. At `signal_start` (default 9:30 AM ET, market open), record the price
2. At `signal_end` (default 10:00 AM ET), record the price
3. Compute the morning return: `(price_signal_end - price_signal_start) / price_signal_start`
4. If morning return > `min_signal_pct`: signal = LONG
5. If morning return < `-min_signal_pct`: signal = SHORT
6. If abs(morning return) <= `min_signal_pct`: no trade

### Entry
- Enter at `entry_time` (default 3:30 PM ET) in the signal direction
- Market entry with 1-tick slippage (conservative for micros)

### Exit (priority order)
1. Stop-loss hit (if configured): `stop_loss_ticks` from entry price
2. Take-profit hit (if configured): `take_profit_ticks` from entry price
3. `exit_time` reached (default 4:00 PM ET): close at last bar's close

### Position Management Defaults (Paper Baseline)
- stop_loss_ticks: None (no stop)
- take_profit_ticks: None (no TP)
- Just hold from entry_time to exit_time

### Instruments
- MES (tick_size=0.25, tick_value=$1.25)
- MNQ (tick_size=0.25, tick_value=$0.50)
- Commission: $1.24 per round-trip
- Slippage: 1 tick per trade

### One Trade Per Day
- Maximum one trade per instrument per day
- No re-entry after exit

### Edge Cases
- **Early close days** (e.g., day before Thanksgiving): if the entry_time bar does not exist, skip the day (return None)
- **Missing data**: if signal window has insufficient bars (< 80% of expected), skip the day
- **Holidays / no data**: naturally skipped since no bars exist

## Parameters

### Baseline (Paper-Exact)
| Parameter | Value |
|-----------|-------|
| signal_start | 9:30 AM ET |
| signal_end | 10:00 AM ET |
| entry_time | 3:30 PM ET |
| exit_time | 4:00 PM ET |
| min_signal_pct | 0.0 |
| stop_loss_ticks | None |
| take_profit_ticks | None |

### Optimization Grid (1,024 combinations)
| Parameter | Values | Count |
|-----------|--------|-------|
| signal_end | 10:00, 10:15, 10:30, 11:00 | 4 |
| entry_time | 15:00, 15:15, 15:30, 15:45 | 4 |
| min_signal_pct | 0.0, 0.05, 0.10, 0.20 | 4 |
| stop_loss_ticks | None, 20, 40, 60 | 4 |
| take_profit_ticks | None, 20, 40, 60 | 4 |

Fixed: signal_start=9:30, exit_time=16:00 (anchored to session boundaries).

## Module Structure

### New Files
| File | Responsibility |
|------|----------------|
| `src/strategy_intmom.py` | Strategy class + trade simulation function |
| `run_intmom.py` | CLI entry point: baseline, optimize, walk-forward, report |

### Reused (No Modifications)
| File | What's Reused |
|------|---------------|
| `src/data_loader.py` | `load_processed()`, `INSTRUMENTS` dict |
| `src/metrics.py` | `calculate_metrics()`, `print_report()` |

### Not Reused
- `src/strategy.py` — SRS-specific (SessionRange, breakout logic)
- `src/backtester.py` — too coupled to SRS range/breakout simulation
- `src/optimizer.py` — SRS parameter grid; intraday momentum has its own grid

### Why a New Simulation Loop
The SRS backtester detects ranges, finds breakouts, then simulates exits across variable-length sessions. Intraday Momentum has a fixed entry time, fixed exit time, and at most 30-60 bars of simulation per trade. A standalone `simulate_trade()` function inside `strategy_intmom.py` is simpler and clearer than adapting the SRS backtester.

## `src/strategy_intmom.py` — Components

### `IntradayMomentumStrategy` class
- `__init__(signal_start, signal_end, entry_time, exit_time, min_signal_pct, stop_loss_ticks, take_profit_ticks)`
- `find_signals(df, instrument)` → list of `Signal` dataclass: `{date, direction, signal_return, entry_time, exit_time}`
- Iterates over trading days, computes morning return, generates signal if above threshold

### `Signal` dataclass
- date, direction ("long"/"short"), signal_return (float), entry_time (Timestamp), exit_time (Timestamp)

### `Trade` dataclass
- Reuse the same fields as `src/backtester.Trade` for compatibility with `calculate_metrics()`
- entry_time, exit_time, direction, entry_price, exit_price, stop_loss, take_profit, pnl_ticks, pnl_dollars, exit_reason ("tp"/"sl"/"session_end"), session (always 0 for intmom), instrument, date, range_high (unused, set to 0), range_low (unused, set to 0)

### `simulate_trade(df, signal, instrument, strategy, slippage_ticks, commission)` → Trade | None
- Get the entry bar at `signal.entry_time`
- Apply slippage to entry price
- Walk bar-by-bar from entry to exit_time:
  - Check stop-loss against bar high/low
  - Check take-profit against bar high/low
- If neither triggered, exit at exit_time bar close
- Compute PnL in ticks and dollars

### `run_backtest(data, strategy, slippage_ticks, commission)` → list[Trade]
- For each instrument in data:
  - Find signals
  - Simulate each trade
  - Collect results

## `run_intmom.py` — CLI Entry Point

### `python run_intmom.py` (baseline)
- Load data from Parquet cache
- Run with paper-exact parameters
- Print metrics report
- Save `results/intmom_baseline.json`

### `python run_intmom.py --optimize` (grid search)
- Build parameter grid (1,024 combinations)
- Run all combinations
- Save `results/intmom_grid.csv`
- Print top 10 by total PnL

### `python run_intmom.py --walk-forward`
- Rolling windows: 24-month train, 6-month test, 6-month step
- Select top 10 parameter sets per window in-sample
- Evaluate out-of-sample
- Save `results/intmom_walk_forward.csv`

### `python run_intmom.py --report`
- Load saved results
- Generate comprehensive text report:
  - Baseline performance
  - Top 10 grid combinations
  - Parameter sensitivity (by each parameter)
  - Walk-forward OOS summary
  - IS vs OOS degradation

## Output Files

| File | Format | Content |
|------|--------|---------|
| `results/intmom_baseline.json` | JSON | Baseline metrics (paper params) |
| `results/intmom_grid.csv` | CSV | 1,024 rows: params + metrics |
| `results/intmom_walk_forward.csv` | CSV | OOS results per window per param set |

## Success Criteria

| Metric | Threshold | Meaning |
|--------|-----------|---------|
| Baseline PF | > 1.0 | Paper signal exists on MES/MNQ |
| Best grid PF | > 1.2 | Meaningful edge with tuning |
| Best grid Sharpe | > 0.8 | Acceptable risk-adjusted return |
| Walk-forward OOS PF | > 1.1 | Edge survives out-of-sample |

If baseline PF < 1.0, the intraday momentum signal does not exist on these instruments and we pivot to the Gap Fade strategy.

## Biases to Avoid

- **Look-ahead bias:** Morning return uses only data up to signal_end. Entry price uses only the entry_time bar. No future data leaks.
- **Overfitting:** Walk-forward analysis with rolling windows. The paper baseline provides an independent hypothesis to test before optimization.
- **Transaction costs:** 1-tick slippage + $1.24 commission per trade, same as SRS.

## Dependencies

No new pip dependencies. Uses pandas, numpy (already installed).
