# Trade-Level Analysis — Design Spec

## Goal

Determine whether the SRS strategy has a real signal worth pursuing, and if so, identify where the edge lives. This is a diagnostic tool, not a trading system enhancement.

## Decision Framework

The analysis answers two questions in order:
1. **Is there a signal?** — Does any configuration show stable, non-random profitability?
2. **Where is the edge?** — If signal exists, what trade characteristics predict winners?

## Script

**File:** `scripts/analyze_trades.py`

**Usage:** `python scripts/analyze_trades.py`

**Output:**
- `results/analysis/*.png` — individual charts (12 total)
- `results/analysis/trade_analysis.pdf` — combined PDF in reading order
- Console: summary stats table

## Data Flow

1. Load MES + MNQ 1-minute data from Parquet cache (`data/processed/`)
2. Run backtester twice with two configurations:
   - **Baseline:** rr_ratio=2.25, range_minutes=30, reentry_check="15min", sessions=[1,2]
   - **Best:** rr_ratio=3.0, range_minutes=60, reentry_check="none", sessions=[1]
3. Collect all `Trade` objects, convert to a single DataFrame with a `config` column
4. Compute derived columns:
   - `range_size_ticks` = (range_high - range_low) / tick_size
   - `year` = entry_time.year
   - `weekday` = entry_time.day_name()
   - `quarter` = entry_time.quarter formatted as "YYYY-QN"
   - `pnl_category` = "win" if pnl_dollars > 0, else "loss"
5. Generate 12 charts, save to `results/analysis/`
6. Print summary stats

## Configurations

| Parameter | Baseline | Best |
|-----------|----------|------|
| rr_ratio | 2.25 | 3.0 |
| range_minutes | 30 | 60 |
| reentry_check | "15min" | "none" |
| sessions | [1, 2] | [1] |

## Charts

### Chapter 1 — "Is there a signal?"

**Chart 1: Equity curves side by side**
- Type: line chart
- X: date, Y: cumulative PnL ($)
- Both configs on same axes, different colors
- Purpose: visual trend — does one config trend up while the other bleeds?

**Chart 2: Rolling profit factor (100-trade window)**
- Type: line chart
- Rolling PF on a 100-trade window, plotted over time
- Horizontal reference line at PF=1.0
- Both configs overlaid
- Purpose: is the edge stable, decaying, or concentrated in one period?

**Chart 3: PnL by year (grouped bar chart)**
- Type: grouped bar chart
- Annual total PnL for baseline vs best, side by side
- Purpose: is profitability spread across years or concentrated?

**Chart 4: PnL distribution (overlapping histograms)**
- Type: histogram with two overlapping distributions
- Per-trade PnL for both configs
- Purpose: shape analysis — skew, fat tails, loss clustering

**Chart 5: Exit reason breakdown (stacked bar chart)**
- Type: stacked percentage bar chart
- Exit reasons (TP, SL, range_reentry, session_end) for each config
- Purpose: how does disabling range reentry change the exit mix?

### Chapter 2 — "Where is the edge?"

**Chart 6: PnL by range size (binned bar chart)**
- Type: grouped bar chart
- Range size bins: 0-10, 10-20, 20-30, 30-50, 50+ ticks
- Average PnL per trade in each bin, both configs
- Purpose: is there a range size sweet spot?

**Chart 7: PnL by day of week (grouped bar chart)**
- Type: grouped bar chart
- Average PnL per trade, Monday-Friday, both configs
- Purpose: weekday effect detection

**Chart 8: PnL by direction (grouped bar chart)**
- Type: grouped bar chart
- Total PnL and win rate for long vs short, both configs
- Purpose: directional bias detection

**Chart 9: PnL by instrument (grouped bar chart)**
- Type: grouped bar chart
- Total PnL and trade count for MES vs MNQ, both configs
- Purpose: should you trade one instrument only?

**Chart 10: Range size vs outcome scatter**
- Type: scatter plot
- X: range size (ticks), Y: PnL per trade, color: win/loss
- Best config only
- Purpose: visual pattern detection in range size vs outcome

**Chart 11: Win rate heatmap (range size x quarter)**
- Type: heatmap
- Y: range size bins, X: quarter (YYYY-QN)
- Cell color: win rate, best config only
- Cells with fewer than 10 trades are grayed out (avoid misleading rates from tiny samples)
- Purpose: identify regimes where the strategy works reliably

**Chart 12: Trade duration by exit reason (box plot)**
- Type: box plot
- X: exit reason (TP, SL, range_reentry, session_end), Y: trade duration in minutes
- Both configs side by side
- Purpose: do winners hit TP fast? Do losers linger? Duration patterns reveal signal quality

## Console Output

Summary stats table comparing both configs:

| Metric | Baseline | Best |
|--------|----------|------|
| Total trades | — | — |
| Win rate | — | — |
| Profit factor | — | — |
| Total PnL | — | — |
| Sharpe ratio | — | — |
| Max drawdown | — | — |
| Avg winner ($) | — | — |
| Avg loser ($) | — | — |
| Best year PnL | — | — |
| Worst year PnL | — | — |

## Dependencies

Existing: `pandas`, `numpy`, `matplotlib`

New: `matplotlib.backends.backend_pdf` (PdfPages) — already included with matplotlib.

No new pip dependencies.

## Scope Boundaries

- This script is read-only analysis. It does not modify any existing code.
- It reuses existing `SRSStrategy`, `Backtester`, and `calculate_metrics` directly.
- No new strategy logic or backtester changes.
- Charts use matplotlib defaults with minimal styling (legible, not pretty).
