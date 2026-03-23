# SRS Backtester

**Language:** always respond in English.

## Purpose

Backtester for the SRS (Session Range Strategy) on micro E-mini futures
MES and MNQ. The goal is to validate the statistical edge of the strategy
and to explore/optimize its parameters.

## Strategy Rules — SRS (Session Range Strategy)

### Sessions
- **Session 1**: 9:00 AM Eastern Time (New York)
- **Session 2**: 12:00 PM Eastern Time
- Be mindful of DST: ET = UTC-5 (EST, winter) / UTC-4 (EDT, summer)

### Entry Logic
1. At the session time, record the high and low of the **first 30 minutes** (range)
2. Wait for a **breakout** above the high (long) or below the low (short)
3. Market entry at the moment of the breakout

### Position Management
- **Stop-loss**: opposite side of the range (short → range high, long → range low)
- **Take-profit**: based on a Risk/Reward ratio (default: 2.25)
- The range in ticks = distance between the range high and low

### Exit Rules
1. Take-profit reached
2. Stop-loss reached
3. Price **closes inside the range** on a 15-minute candle
4. End of session (forced exit)

### Instruments
- **MES** (Micro E-mini S&P 500): tick size = 0.25, tick value = $1.25
- **MNQ** (Micro E-mini Nasdaq 100): tick size = 0.25, tick value = $0.50

## Data

### Source
- **Databento** — CME Globex MDP 3.0 (GLBX.MDP3)
- Schema: OHLCV-1m
- Format: DBN (Databento Binary Encoding), zstd compressed
- Period: 2010-06-06 to 2026-03-15 (MES/MNQ have existed since May 2019)

### Files
- `data/raw/`: raw DBN files (not versioned, in .gitignore)
- `data/processed/`: cleaned data in Parquet (not versioned)

## Backtesting Methodology

### Biases to Avoid
- **Look-ahead bias**: always use shift(1) or verify that you are not looking
  into the future
- **Survivorship bias**: not applicable here (futures, not stocks)
- **Overfitting**: split train/test/validation, walk-forward analysis

### Contract rollover
- The data contains all quarterly contracts
- Use the front-month contract and handle the roll properly
- Verify that the roll does not create false signals at transition dates

### Realistic Costs
- **Commissions**: model broker fees (NinjaTrader fees)
- **Slippage**: 1 tick per trade by default (conservative for micros)

### Metrics to Calculate
- Win rate
- Profit factor
- Max drawdown
- Sharpe ratio
- Total number of trades
- Average PnL per trade
- Performance by session (session 1 vs session 2)
- Performance by instrument (MES vs MNQ)
- Trade distribution by day of the week

## Tech Stack
- Python 3.10+
- `databento` — reading DBN files
- `pandas` — data manipulation
- `numpy` — numerical computations
- Custom backtester (pandas-based) to start

## File Structure
```
srs-backtester/
├── CLAUDE.md              ← this file
├── .gitignore
├── requirements.txt
├── run.py                 ← unified CLI: python run.py <strategy> [--optimize|--walk-forward|--report]
├── run_backtest.py        ← SRS baseline backtest (legacy)
├── run_optimization.py    ← SRS optimization (legacy)
├── run_asian.py           ← Asian Range CLI (legacy)
├── run_intmom.py          ← Intraday Momentum CLI (legacy)
├── run_gapfade.py         ← Gap Fade CLI (legacy)
├── data/
│   ├── raw/               ← Databento DBN files (gitignored)
│   └── processed/         ← cleaned Parquet files (gitignored)
├── src/
│   ├── trade.py           ← Trade dataclass + BaseStrategy ABC (plugin interface)
│   ├── backtester.py      ← generic strategy-agnostic backtest engine
│   ├── optimizer.py       ← generic grid search + walk-forward analysis
│   ├── metrics.py         ← performance metrics calculation
│   ├── data_loader.py     ← data loading, rollover, timezone conversion
│   └── strategies/        ← strategy plugins (auto-discovered)
│       ├── __init__.py    ← registry: get_strategy(), list_strategies()
│       ├── srs.py         ← Session Range Strategy
│       ├── asian_range.py ← Asian Range Breakout/Fade
│       ├── intmom.py      ← Intraday Momentum
│       └── gapfade.py     ← Overnight Gap Fade
├── tests/                 ← pytest tests
├── scripts/               ← diagnostic scripts
├── notebooks/             ← exploration and visualization
└── results/               ← backtest results (gitignored)
```

## Adding a New Strategy

1. Create a file in `src/strategies/` (e.g. `my_strategy.py`)
2. Subclass `BaseStrategy` from `src.trade` and implement:
   - `name` class attribute
   - `default_params()`, `from_params()`, `to_params()`
   - `generate_signals(data)`, `simulate_trade(signal, df, slippage, commission)`
   - `build_param_grid()`
3. Decorate with `@register` from `src.strategies`
4. Run: `python run.py my_strategy`

# currentDate
Today's date is 2026-03-18.
