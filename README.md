# Futures Backtester

A strategy-agnostic backtesting engine for micro E-mini futures (MES/MNQ), with built-in grid search optimization and walk-forward validation.

## Features

- **Plugin architecture** — strategies are self-contained Python modules that implement a standard interface. Drop a file in `src/strategies/` and it's auto-discovered.
- **Realistic simulation** — bar-by-bar execution on 1-minute OHLCV data with configurable slippage (default: 1 tick) and commissions (default: $1.24 RT).
- **Grid search** — exhaustive parameter optimization across all combinations, with strategy-level hooks for precomputation.
- **Walk-forward validation** — rolling 24-month train / 6-month test windows to detect overfitting. In-sample and out-of-sample metrics reported side by side.
- **Continuous front-month series** — automatic contract rollover based on daily volume (handles quarterly futures seamlessly).
- **Unified CLI** — one entry point for baseline runs, optimization, walk-forward, and report generation.

## Included Strategies

| Strategy | Description |
|----------|-------------|
| `srs` | Session Range Strategy — 30-min opening range breakout at 9:00 AM / 12:00 PM ET |
| `intmom` | Intraday momentum — trend continuation signals |
| `gapfade` | Overnight gap fade — mean reversion on open gaps |

## Quickstart

```bash
# Clone and set up
git clone https://github.com/charlesbx/futures-backtester.git
cd futures-backtester
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Data

The backtester uses [Databento](https://databento.com/) 1-minute OHLCV data for MES and MNQ futures. You'll need to:

1. Obtain MES/MNQ OHLCV-1m data from Databento (schema: `ohlcv-1m`, dataset: `GLBX.MDP3`)
2. Process raw DBN files into Parquet using the data loader:

```python
from src.data_loader import load_and_prepare, save_processed

df = load_and_prepare("data/raw/your_file.dbn.zst", "MES")
save_processed(df, "data/processed", "MES")
```

This builds a continuous front-month series with automatic contract rollover and Eastern Time conversion.

### Usage

```bash
# List available strategies
python run.py --list

# Run a baseline backtest (default parameters)
python run.py srs

# Run grid search optimization
python run.py srs --optimize

# Run walk-forward validation
python run.py intmom --walk-forward

# Generate a report from saved results
python run.py gapfade --report
```

## Writing a New Strategy

Create a file in `src/strategies/` and implement the `BaseStrategy` interface:

```python
from src.trade import BaseStrategy, Trade
from src.strategies import register

@register
class MyStrategy(BaseStrategy):
    name = "my_strategy"

    @classmethod
    def default_params(cls):
        return {"threshold": 0.5, "stop_atr": 1.5}

    @classmethod
    def from_params(cls, params):
        return cls(threshold=params["threshold"], stop_atr=params["stop_atr"])

    def to_params(self):
        return {"threshold": self.threshold, "stop_atr": self.stop_atr}

    def generate_signals(self, data):
        # data: dict mapping instrument name -> OHLCV 1-min DataFrame
        # Return a list of signal objects (each must have .instrument attribute)
        ...

    def simulate_trade(self, signal, df, slippage_ticks, commission):
        # Simulate a single trade from a signal
        # Return a Trade object, or None if the trade couldn't be executed
        ...

    @classmethod
    def build_param_grid(cls):
        return [
            {"threshold": t, "stop_atr": s}
            for t in [0.3, 0.5, 0.7, 1.0]
            for s in [1.0, 1.5, 2.0]
        ]
```

Then run it:

```bash
python run.py my_strategy
python run.py my_strategy --optimize
python run.py my_strategy --walk-forward
```

Strategies can override `run_grid_search()` and `run_walk_forward()` for performance optimizations (e.g., precomputing signals shared across parameter combinations).

## Architecture

```
src/
├── trade.py           # Trade dataclass + BaseStrategy ABC
├── backtester.py      # Strategy-agnostic backtest engine
├── optimizer.py       # Grid search + walk-forward analysis
├── metrics.py         # Performance metrics (win rate, PF, Sharpe, drawdown, ...)
├── data_loader.py     # Databento data loading, rollover, timezone conversion
└── strategies/        # Strategy plugins (auto-discovered)
    ├── __init__.py    # Registry: get_strategy(), list_strategies()
    ├── srs.py         # Session Range Strategy
    ├── intmom.py      # Intraday Momentum
    └── gapfade.py     # Overnight Gap Fade
```

### How It Works

1. **Data loading** — raw Databento DBN files are processed into continuous front-month Parquet series (rollover by daily volume, UTC → Eastern Time conversion)
2. **Signal generation** — each strategy scans the data and produces a list of trading signals
3. **Trade simulation** — signals are executed bar-by-bar with slippage and commission modeling
4. **Metrics** — win rate, profit factor, Sharpe ratio, max drawdown, PnL by session/instrument/weekday
5. **Optimization** — grid search tests all parameter combinations; walk-forward validates out-of-sample

## Metrics

The engine computes:

- Win rate, profit factor, total PnL, average PnL per trade
- Sharpe ratio (annualized, 252 trading days)
- Maximum drawdown
- Performance breakdown by session, instrument, day of week
- Long/short split metrics
- Exit reason distribution

## License

MIT
