"""Unified Trade dataclass and BaseStrategy abstract base class.

These are the core abstractions for the strategy-agnostic backtester engine.
Any strategy can be plugged in by subclassing BaseStrategy and implementing
the required methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class Trade:
    """A single executed trade.

    This is the canonical Trade representation used across all strategies.
    The metrics module and backtester engine operate on lists of Trade objects.
    """

    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str       # 'long' or 'short'
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    pnl_ticks: float
    pnl_dollars: float
    exit_reason: str     # 'tp', 'sl', 'session_end', 'range_reentry', etc.
    session: int
    instrument: str
    date: pd.Timestamp
    range_high: float
    range_low: float


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies.

    To create a new strategy:

    1. Subclass BaseStrategy
    2. Set the ``name`` class attribute
    3. Implement all abstract methods
    4. Place the file in ``src/strategies/`` for auto-discovery

    Example::

        class MyStrategy(BaseStrategy):
            name = "my_strategy"

            @classmethod
            def default_params(cls):
                return {"threshold": 0.5}

            @classmethod
            def from_params(cls, params):
                return cls(threshold=params["threshold"])

            def to_params(self):
                return {"threshold": self.threshold}

            def generate_signals(self, data):
                ...

            def simulate_trade(self, signal, df, slippage_ticks, commission):
                ...

            @classmethod
            def build_param_grid(cls):
                return [{"threshold": t} for t in [0.3, 0.5, 0.7]]
    """

    name: str  # Must be set as a class attribute

    @classmethod
    @abstractmethod
    def default_params(cls) -> dict:
        """Return default parameter values for the strategy."""

    @classmethod
    @abstractmethod
    def from_params(cls, params: dict) -> BaseStrategy:
        """Create a new strategy instance from a parameter dictionary."""

    @abstractmethod
    def to_params(self) -> dict:
        """Serialize current strategy parameters to a dictionary."""

    @abstractmethod
    def generate_signals(self, data: dict[str, pd.DataFrame]) -> list:
        """Generate trading signals from OHLCV data.

        Args:
            data: Dict mapping instrument name to OHLCV 1-min DataFrame
                  with Eastern Time DatetimeIndex.

        Returns:
            List of signal objects. Each signal must have an ``instrument``
            attribute (str) so the engine knows which DataFrame to use
            for trade simulation.
        """

    @abstractmethod
    def simulate_trade(
        self,
        signal: Any,
        df: pd.DataFrame,
        slippage_ticks: float,
        commission: float,
    ) -> Trade | None:
        """Simulate a single trade from a signal.

        Args:
            signal: A signal object produced by generate_signals().
            df: Full OHLCV 1-min DataFrame for the signal's instrument.
            slippage_ticks: Slippage in ticks per trade.
            commission: Round-trip commission in dollars.

        Returns:
            A Trade object, or None if the trade could not be executed.
        """

    @classmethod
    @abstractmethod
    def build_param_grid(cls) -> list[dict]:
        """Return parameter combinations for grid search optimization."""

    @classmethod
    def param_columns(cls) -> list[str]:
        """Column names for parameters in grid search results DataFrame."""
        return list(cls.default_params().keys())

    @classmethod
    def run_grid_search(
        cls,
        data: dict[str, pd.DataFrame],
        param_grid: list[dict] | None = None,
        slippage_ticks: float = 1,
        commission: float = 1.24,
        progress: bool = True,
    ) -> pd.DataFrame:
        """Run grid search over parameter combinations.

        Override this method for strategy-specific optimizations
        (e.g. precomputing signals shared across param combos).
        """
        from .optimizer import generic_grid_search

        return generic_grid_search(
            cls, data, param_grid, slippage_ticks, commission, progress,
        )

    @classmethod
    def run_walk_forward(
        cls,
        data: dict[str, pd.DataFrame],
        param_grid: list[dict] | None = None,
        train_months: int = 24,
        test_months: int = 6,
        top_n: int = 10,
        slippage_ticks: float = 1,
        commission: float = 1.24,
    ) -> pd.DataFrame:
        """Run rolling walk-forward analysis.

        Override for strategy-specific optimizations.
        """
        from .optimizer import generic_walk_forward

        return generic_walk_forward(
            cls, data, param_grid, train_months, test_months,
            top_n, slippage_ticks, commission,
        )

    @classmethod
    def generate_report(
        cls,
        grid_results: pd.DataFrame | None,
        wf_results: pd.DataFrame | None,
        baseline: dict | None,
    ) -> str:
        """Generate a text report from optimization results.

        Override for strategy-specific report formatting.
        """
        from .optimizer import default_report

        return default_report(
            cls.name, grid_results, wf_results, baseline, cls.param_columns(),
        )
