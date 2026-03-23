"""Generic strategy-agnostic backtest engine.

Runs any BaseStrategy subclass on OHLCV data and returns a list of trades.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from .trade import BaseStrategy, Trade


def run_backtest(
    strategy: BaseStrategy,
    data: dict[str, pd.DataFrame],
    slippage_ticks: float = 1,
    commission: float = 1.24,
) -> list[Trade]:
    """Run a strategy on OHLCV data and return executed trades.

    Args:
        strategy: A configured BaseStrategy instance.
        data: Dict mapping instrument name (e.g. 'MES') to OHLCV 1-min
              DataFrame with Eastern Time DatetimeIndex.
        slippage_ticks: Slippage in ticks per trade (default 1).
        commission: Round-trip commission in dollars (default 1.24).

    Returns:
        List of Trade objects.
    """
    signals = strategy.generate_signals(data)
    trades: list[Trade] = []
    for signal in signals:
        trade = strategy.simulate_trade(
            signal, data[signal.instrument], slippage_ticks, commission,
        )
        if trade is not None:
            trades.append(trade)
    return trades
