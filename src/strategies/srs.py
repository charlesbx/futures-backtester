"""SRS (Session Range Strategy) — BaseStrategy implementation.

Migrates the original SRSStrategy + Backtester SRS logic into the
strategy-agnostic BaseStrategy interface.
"""

from __future__ import annotations

import ast
import itertools
import json
from dataclasses import dataclass
from datetime import time
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..backtester import run_backtest
from ..data_loader import INSTRUMENTS
from ..metrics import calculate_metrics
from ..trade import BaseStrategy, Trade
from . import register


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SESSION_CONFIG = {
    1: {"start": time(9, 0), "end": time(12, 0)},
    2: {"start": time(12, 0), "end": time(16, 0)},
}

MIN_RANGE_BARS = 5


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SessionRange:
    """30-minute range identified at the start of a session."""

    date: pd.Timestamp
    session: int
    range_high: float
    range_low: float
    range_start: pd.Timestamp
    range_end: pd.Timestamp
    session_end: pd.Timestamp
    instrument: str


@dataclass
class SRSSignal:
    """A detected breakout signal off a session range."""

    range_info: SessionRange
    direction: str          # 'long' or 'short'
    entry_time: pd.Timestamp
    instrument: str


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

@register
class SRSStrategy(BaseStrategy):
    """Session Range Strategy.

    Identifies session ranges, detects breakouts, and simulates trades
    with SL/TP based on the range size and R:R ratio.

    Parameters
    ----------
    rr_ratio:       Risk/reward ratio for take-profit (default 2.25)
    range_minutes:  Duration of the ranging period in minutes (default 30)
    reentry_check:  Reentry detection method: "15min", "30min", "none", "strict"
    sessions:       List of session numbers to trade; None = all
    """

    name = "srs"

    def __init__(
        self,
        rr_ratio: float = 2.25,
        range_minutes: int = 30,
        reentry_check: str = "15min",
        sessions: list[int] | None = None,
    ):
        self.rr_ratio = rr_ratio
        self.range_minutes = range_minutes
        self.reentry_check = reentry_check
        self.sessions = sessions
        # Populated by run_grid_search for performance optimisation
        self._precomputed_ranges: dict[tuple[str, int], list[SessionRange]] | None = None

    # ------------------------------------------------------------------
    # BaseStrategy abstract methods
    # ------------------------------------------------------------------

    @classmethod
    def default_params(cls) -> dict:
        return {
            "rr_ratio": 2.25,
            "range_minutes": 30,
            "reentry_check": "15min",
            "sessions": "[1, 2]",
        }

    @classmethod
    def from_params(cls, params: dict) -> SRSStrategy:
        sessions = params.get("sessions")
        if sessions is None:
            sessions_parsed: list[int] | None = None
        elif isinstance(sessions, str):
            sessions_parsed = ast.literal_eval(sessions)
        else:
            sessions_parsed = list(sessions)
        return cls(
            rr_ratio=float(params.get("rr_ratio", 2.25)),
            range_minutes=int(params.get("range_minutes", 30)),
            reentry_check=str(params.get("reentry_check", "15min")),
            sessions=sessions_parsed,
        )

    def to_params(self) -> dict:
        return {
            "rr_ratio": self.rr_ratio,
            "range_minutes": self.range_minutes,
            "reentry_check": self.reentry_check,
            "sessions": str(self.sessions if self.sessions is not None else [1, 2]),
        }

    @classmethod
    def build_param_grid(cls) -> list[dict]:
        """Build the 432-combination parameter grid.

        Grid:
          rr_ratio       : 8 values (1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0)
          range_minutes  : 6 values (15, 20, 25, 30, 45, 60)
          reentry_check  : 3 values ("15min", "30min", "none")
          sessions       : 3 values ([1,2], [1], [2])

        Total: 8 x 6 x 3 x 3 = 432
        """
        rr_ratios = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0]
        range_minutes_vals = [15, 20, 25, 30, 45, 60]
        reentry_checks = ["15min", "30min", "none"]
        sessions_vals = [[1, 2], [1], [2]]

        grid: list[dict] = []
        for rr, rm, rc, sess in itertools.product(
            rr_ratios, range_minutes_vals, reentry_checks, sessions_vals
        ):
            grid.append({
                "rr_ratio": rr,
                "range_minutes": rm,
                "reentry_check": rc,
                "sessions": str(sess),
            })
        return grid

    def generate_signals(self, data: dict[str, pd.DataFrame]) -> list[SRSSignal]:
        """Generate SRS breakout signals for all instruments.

        Uses precomputed ranges if available (set by run_grid_search).
        """
        signals: list[SRSSignal] = []

        for instrument, df in data.items():
            # Use precomputed ranges if available
            if self._precomputed_ranges is not None:
                key = (instrument, self.range_minutes)
                all_ranges = self._precomputed_ranges.get(key, [])
                # Apply session filter
                if self.sessions is not None:
                    ranges = [r for r in all_ranges if r.session in self.sessions]
                else:
                    ranges = all_ranges
            else:
                ranges = self._find_session_ranges(df, instrument)

            for range_info in ranges:
                mask = (
                    (df.index >= range_info.range_end)
                    & (df.index < range_info.session_end)
                )
                session_bars = df.loc[mask]
                if session_bars.empty:
                    continue
                result = self._detect_breakout(session_bars, range_info)
                if result is not None:
                    direction, entry_time = result
                    signals.append(SRSSignal(
                        range_info=range_info,
                        direction=direction,
                        entry_time=entry_time,
                        instrument=instrument,
                    ))

        return signals

    def simulate_trade(
        self,
        signal: SRSSignal,
        df: pd.DataFrame,
        slippage_ticks: float,
        commission: float,
    ) -> Trade | None:
        """Simulate a single SRS trade from entry to exit.

        Walks bars from the entry bar to session end, checking SL/TP/reentry
        in priority order.
        """
        inst = INSTRUMENTS[signal.instrument]
        tick_size = inst["tick_size"]
        tick_value = inst["tick_value"]

        range_info = signal.range_info
        direction = signal.direction
        slippage = slippage_ticks * tick_size

        # Entry price with slippage
        params = self._compute_trade_params(range_info, direction)
        if direction == "long":
            entry_price = params["entry_price"] + slippage
        else:
            entry_price = params["entry_price"] - slippage

        stop_loss = params["stop_loss"]
        take_profit = params["take_profit"]

        # Remaining bars after entry (from entry_time onwards, up to session end)
        remaining_mask = (
            (df.index > signal.entry_time)
            & (df.index < range_info.session_end)
        )
        remaining = df.loc[remaining_mask]

        exit_time, exit_price, exit_reason = self._find_exit(
            remaining, direction, stop_loss, take_profit, range_info,
        )

        # If no exit found, force-close at session end bar
        if exit_time is None:
            session_bars_mask = (
                (df.index >= signal.entry_time)
                & (df.index < range_info.session_end)
            )
            session_bars = df.loc[session_bars_mask]
            if session_bars.empty:
                return None
            last_bar = session_bars.iloc[-1]
            exit_price = last_bar["close"]
            exit_time = last_bar.name
            exit_reason = "session_end"

        # PnL
        if direction == "long":
            pnl_ticks = (exit_price - entry_price) / tick_size
        else:
            pnl_ticks = (entry_price - exit_price) / tick_size

        pnl_dollars = pnl_ticks * tick_value - commission

        return Trade(
            entry_time=signal.entry_time,
            exit_time=exit_time,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            pnl_ticks=pnl_ticks,
            pnl_dollars=pnl_dollars,
            exit_reason=exit_reason,
            session=range_info.session,
            instrument=signal.instrument,
            date=range_info.date,
            range_high=range_info.range_high,
            range_low=range_info.range_low,
        )

    # ------------------------------------------------------------------
    # Grid search override — precomputes session ranges
    # ------------------------------------------------------------------

    @classmethod
    def run_grid_search(
        cls,
        data: dict[str, pd.DataFrame],
        param_grid: list[dict] | None = None,
        slippage_ticks: float = 1,
        commission: float = 1.24,
        progress: bool = True,
    ) -> pd.DataFrame:
        """Optimised grid search with range precomputation.

        Session ranges depend only on (instrument, range_minutes).
        Precomputing them once per unique (instrument, range_minutes) pair
        avoids redundant work across the 432 combinations.
        """
        from ..optimizer import _extract_metrics

        if param_grid is None:
            param_grid = cls.build_param_grid()

        # 1. Precompute ranges for each unique (instrument, range_minutes)
        unique_range_minutes: set[int] = set()
        for p in param_grid:
            unique_range_minutes.add(int(p["range_minutes"]))

        precomputed: dict[tuple[str, int], list[SessionRange]] = {}
        for instrument, df in data.items():
            for rm in unique_range_minutes:
                # Build a temporary strategy just to compute ranges (no session filter)
                tmp = cls(range_minutes=rm, sessions=None)
                precomputed[(instrument, rm)] = tmp._find_session_ranges(df, instrument)

        # 2. Run grid search, attaching precomputed ranges to each strategy instance
        rows: list[dict] = []
        iterator = (
            tqdm(param_grid, desc="Grid search (srs)")
            if progress else param_grid
        )

        for params in iterator:
            strategy = cls.from_params(params)
            strategy._precomputed_ranges = precomputed
            trades = run_backtest(strategy, data, slippage_ticks, commission)
            metrics = calculate_metrics(trades)
            row = {**strategy.to_params(), **_extract_metrics(metrics)}
            rows.append(row)

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    @classmethod
    def generate_report(
        cls,
        grid_results: pd.DataFrame | None,
        wf_results: pd.DataFrame | None,
        baseline: dict | None,
    ) -> str:
        """Generate a comprehensive SRS optimization report.

        8 sections:
        1. Baseline
        2. Top 10 combinations (IS or OOS)
        3. Sensitivity analysis
        4. reentry_check x rr_ratio heatmap
        5. Directional bias
        6. IS vs OOS comparison
        7. Detailed metrics for best combo
        8. Analysis by instrument
        """
        lines: list[str] = []
        SEP = "=" * 72

        def section(title: str) -> None:
            lines.extend(["", SEP, f"  {title}", SEP])

        has_wf = wf_results is not None and not wf_results.empty
        has_grid = grid_results is not None and not grid_results.empty

        # ── 1. BASELINE ────────────────────────────────────────────────
        section("1. BASELINE — SRS (default parameters: rr=2.25, range=30min, 15min reentry)")
        if baseline and baseline.get("total_trades", 0) > 0:
            b = baseline
            lines.append(f"  Trades         : {b.get('total_trades', 'N/A')}")
            lines.append(f"  Win rate       : {b.get('win_rate', 0):.1%}")
            lines.append(f"  Profit factor  : {b.get('profit_factor', 0):.3f}")
            lines.append(f"  Total PnL      : ${b.get('total_pnl', 0):>10,.2f}")
            lines.append(f"  Sharpe ratio   : {b.get('sharpe_ratio', 0):.3f}")
            lines.append(f"  Max drawdown   : ${b.get('max_drawdown', 0):>10,.2f}")
        else:
            lines.append("  No baseline data available.")

        # ── 2. TOP 10 COMBINATIONS ─────────────────────────────────────
        sort_label = "OOS PF" if has_wf else "IS PF"
        section(f"2. TOP 10 COMBINATIONS — sorted by {sort_label}")
        param_columns = cls.param_columns()

        if has_grid:
            if has_wf:
                # Aggregate OOS PF by params string
                wf_agg = (
                    wf_results.groupby("params")["oos_pf"]
                    .mean()
                    .reset_index()
                    .rename(columns={"oos_pf": "mean_oos_pf"})
                )
                top_params = wf_agg.nlargest(10, "mean_oos_pf")
                for _, row in top_params.iterrows():
                    try:
                        p = json.loads(row["params"])
                    except Exception:
                        p = {}
                    fdr_flag = ""
                    if has_grid and "fdr_significant" in grid_results.columns:
                        # look up FDR flag in grid results
                        mask = True
                        for k, v in p.items():
                            if k in grid_results.columns:
                                mask = mask & (grid_results[k].astype(str) == str(v))
                        matches = grid_results[mask] if not isinstance(mask, bool) else pd.DataFrame()
                        if not matches.empty and not matches["fdr_significant"].iloc[0]:
                            fdr_flag = "  [!FDR]"
                    params_str = "  ".join(f"{k}={v}" for k, v in p.items())
                    lines.append(
                        f"  {params_str}"
                        f"  OOS_PF={row['mean_oos_pf']:.3f}{fdr_flag}"
                    )
            else:
                top10 = grid_results.nlargest(10, "profit_factor")
                for _, row in top10.iterrows():
                    fdr_flag = ""
                    if "fdr_significant" in row and not row["fdr_significant"]:
                        fdr_flag = "  [!FDR]"
                    params_str = "  ".join(
                        f"{c}={row.get(c, '?')}" for c in param_columns
                    )
                    lines.append(
                        f"  {params_str}"
                        f"  PF={row.get('profit_factor', 0):.3f}"
                        f"  WR={row.get('win_rate', 0):.1%}"
                        f"  PnL=${row.get('total_pnl', 0):>9,.0f}"
                        f"  trades={int(row.get('total_trades', 0))}"
                        f"{fdr_flag}"
                    )
        else:
            lines.append("  No grid search results available.")

        # ── 3. SENSITIVITY ANALYSIS ────────────────────────────────────
        if has_grid:
            section("3. SENSITIVITY ANALYSIS — Mean Profit Factor by Parameter")
            for param in param_columns:
                if param not in grid_results.columns:
                    continue
                grouped = (
                    grid_results.groupby(param)["profit_factor"]
                    .mean()
                    .reset_index()
                )
                lines.append(f"\n  {param}:")
                for _, row in grouped.iterrows():
                    lines.append(
                        f"    {str(row[param]):<20}  PF={row['profit_factor']:.3f}"
                    )

        # ── 4. REENTRY x RR HEATMAP ────────────────────────────────────
        if has_grid and "reentry_check" in grid_results.columns and "rr_ratio" in grid_results.columns:
            section("4. REENTRY_CHECK x RR_RATIO — Mean Profit Factor Heatmap")
            pivot = (
                grid_results.groupby(["reentry_check", "rr_ratio"])["profit_factor"]
                .mean()
                .unstack(level="rr_ratio")
            )
            rr_cols = sorted(pivot.columns)
            header = f"  {'reentry':<12}" + "".join(f"  rr={c:<5}" for c in rr_cols)
            lines.append(header)
            for reentry_val in pivot.index:
                row_vals = "".join(
                    f"  {pivot.loc[reentry_val, c]:.3f}  " for c in rr_cols
                )
                lines.append(f"  {str(reentry_val):<12}{row_vals}")

        # ── 5. DIRECTIONAL BIAS ────────────────────────────────────────
        if has_grid and "pnl_long" in grid_results.columns:
            section("5. DIRECTIONAL BIAS — Long vs Short (top 10 IS combos)")
            top10 = grid_results.nlargest(10, "profit_factor")
            lines.append(
                f"  {'Params':<50}  {'Long PnL':>10}  {'Short PnL':>10}"
                f"  {'WR Long':>8}  {'WR Short':>9}"
            )
            for _, row in top10.iterrows():
                params_str = f"rr={row.get('rr_ratio','?')} rm={row.get('range_minutes','?')}"
                params_str += f" rc={row.get('reentry_check','?')} s={row.get('sessions','?')}"
                lines.append(
                    f"  {params_str:<50}"
                    f"  ${row.get('pnl_long', 0):>9,.0f}"
                    f"  ${row.get('pnl_short', 0):>9,.0f}"
                    f"  {row.get('win_rate_long', 0):>7.1%}"
                    f"  {row.get('win_rate_short', 0):>8.1%}"
                )

        # ── 6. IS vs OOS COMPARISON ────────────────────────────────────
        if has_wf:
            section("6. WALK-FORWARD — IS vs OOS Summary")
            lines.append(f"  Windows         : {wf_results['window_id'].nunique()}")
            lines.append(f"  Total rows      : {len(wf_results)}")
            lines.append(f"  Mean IS PF      : {wf_results['is_pf'].mean():.3f}")
            lines.append(f"  Mean OOS PF     : {wf_results['oos_pf'].mean():.3f}")
            lines.append(f"  Total OOS PnL   : ${wf_results['oos_pnl'].sum():,.0f}")
            oos_pos = (wf_results["oos_pf"] > 1.0).sum()
            lines.append(
                f"  OOS PF > 1.0    : {oos_pos}/{len(wf_results)} "
                f"({oos_pos / len(wf_results):.1%})"
            )
            pnl_pos = (wf_results["oos_pnl"] > 0).sum()
            lines.append(
                f"  OOS PnL > 0     : {pnl_pos}/{len(wf_results)} "
                f"({pnl_pos / len(wf_results):.1%})"
            )

            # Per-window table (top 5 rows per window by OOS PF)
            lines.append("")
            lines.append(
                f"  {'Window':<8}  {'Train':<23}  {'Test':<23}"
                f"  {'IS PF':>6}  {'OOS PF':>7}  {'OOS PnL':>10}"
            )
            for wid in sorted(wf_results["window_id"].unique()):
                w = wf_results[wf_results["window_id"] == wid]
                best = w.nlargest(1, "oos_pf").iloc[0]
                lines.append(
                    f"  {int(wid):<8}  "
                    f"{best['train_start']} → {best['train_end']:<12}  "
                    f"{best['test_start']} → {best['test_end']:<12}  "
                    f"{best['is_pf']:>6.3f}  {best['oos_pf']:>7.3f}  "
                    f"${best['oos_pnl']:>9,.0f}"
                )

        # ── 7. DETAILED METRICS — BEST COMBO ──────────────────────────
        section("7. DETAILED METRICS — Best IS Combination")
        if has_grid:
            best_row = grid_results.nlargest(1, "profit_factor").iloc[0]
            lines.append(f"  Parameters:")
            for c in param_columns:
                lines.append(f"    {c:<20}: {best_row.get(c, 'N/A')}")
            lines.append("")
            lines.append(f"  Trades         : {int(best_row.get('total_trades', 0))}")
            lines.append(f"  Win rate       : {best_row.get('win_rate', 0):.1%}")
            lines.append(f"  Profit factor  : {best_row.get('profit_factor', 0):.3f}")
            lines.append(f"  Total PnL      : ${best_row.get('total_pnl', 0):>10,.2f}")
            lines.append(f"  Sharpe ratio   : {best_row.get('sharpe_ratio', 0):.3f}")
            lines.append(f"  Max drawdown   : ${best_row.get('max_drawdown', 0):>10,.2f}")
            if "fdr_significant" in best_row:
                lines.append(f"  FDR significant: {best_row['fdr_significant']}")
        else:
            lines.append("  No grid search results available.")

        # ── 8. ANALYSIS BY INSTRUMENT ─────────────────────────────────
        section("8. ANALYSIS BY INSTRUMENT — Top 10 IS combos")
        if has_grid:
            top10 = grid_results.nlargest(10, "profit_factor")
            inst_cols_pnl = [c for c in ["pnl_mes", "pnl_mnq"] if c in top10.columns]
            inst_cols_trades = [c for c in ["trades_mes", "trades_mnq"] if c in top10.columns]
            if inst_cols_pnl:
                lines.append(
                    f"  {'Params':<50}  "
                    + "  ".join(f"{c:>12}" for c in inst_cols_pnl)
                )
                for _, row in top10.iterrows():
                    params_str = f"rr={row.get('rr_ratio','?')} rm={row.get('range_minutes','?')}"
                    params_str += f" rc={row.get('reentry_check','?')} s={row.get('sessions','?')}"
                    vals = "  ".join(f"  ${row.get(c, 0):>9,.0f}" for c in inst_cols_pnl)
                    lines.append(f"  {params_str:<50}{vals}")
            else:
                lines.append("  Per-instrument breakdown not available in grid results.")
                lines.append("  (Run grid search with instrument-tagged columns to see this.)")
        else:
            lines.append("  No grid search results available.")

        # ── FDR SUMMARY ────────────────────────────────────────────────
        if has_grid and "fdr_significant" in grid_results.columns:
            lines.extend(["", SEP, "  FDR CONTROL SUMMARY (Benjamini-Hochberg, alpha=0.05)", SEP])
            n_sig = int(grid_results["fdr_significant"].sum())
            n_total = len(grid_results)
            lines.append(f"  Significant combinations: {n_sig} / {n_total} ({n_sig/n_total:.1%})")
            lines.append(f"  Non-significant marked [!FDR] in Top 10 above.")

        # ── COMBOS SUMMARY ────────────────────────────────────────────
        lines.extend(["", SEP])
        if has_grid:
            n_pf1 = int((grid_results["profit_factor"] > 1.0).sum())
            lines.append(
                f"  Combos with PF > 1.0: {n_pf1} / {len(grid_results)}"
            )
        lines.extend([SEP, ""])

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # FDR control (Benjamini-Hochberg) — SRS-specific
    # ------------------------------------------------------------------

    @staticmethod
    def apply_fdr_control(
        results_df: pd.DataFrame,
        trades_by_params: dict[str, list[Any]],
        n_permutations: int = 1000,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """Add p_value, p_value_adjusted, and fdr_significant columns.

        For each combination:
        1. Permute trade pnl_dollars (sign flip) N times
        2. Compute permuted total PnL each time
        3. p-value = fraction of permuted PnLs >= observed
        4. Benjamini-Hochberg correction at alpha

        Args:
            results_df:       Grid search results DataFrame (one row per combo)
            trades_by_params: Dict mapping param hash (json.dumps sorted) to Trade list
            n_permutations:   Number of permutation iterations (default 1000)
            alpha:            FDR threshold (default 0.05)

        Returns:
            Copy of results_df with 'p_value', 'p_value_adjusted', 'fdr_significant'.
        """
        rng = np.random.default_rng(seed=42)
        df = results_df.copy()
        p_values: list[float] = []

        for _, row in df.iterrows():
            # Build param key — must match how run_grid_search stores it
            param_cols = [
                "rr_ratio", "range_minutes", "reentry_check", "sessions",
            ]
            param_dict = {c: row[c] for c in param_cols if c in row}
            key = json.dumps(param_dict, sort_keys=True, default=str)

            trades = trades_by_params.get(key, [])
            if not trades:
                p_values.append(1.0)
                continue

            pnl_arr = np.array([t.pnl_dollars for t in trades])
            observed_pnl = pnl_arr.sum()

            # Permutation: randomly flip signs
            count_gte = 0
            for _ in range(n_permutations):
                signs = rng.choice([-1.0, 1.0], size=len(pnl_arr))
                perm_pnl = (np.abs(pnl_arr) * signs).sum()
                if perm_pnl >= observed_pnl:
                    count_gte += 1

            p_values.append(count_gte / n_permutations)

        df["p_value"] = p_values

        # Benjamini-Hochberg correction
        n = len(p_values)
        sorted_idx = np.argsort(p_values)
        p_adj = np.empty(n)
        min_so_far = 1.0
        for rank_inv, idx in enumerate(reversed(sorted_idx)):
            rank = n - rank_inv  # 1-based rank from largest p
            adjusted = p_values[idx] * n / rank
            min_so_far = min(min_so_far, adjusted)
            p_adj[idx] = min_so_far

        df["p_value_adjusted"] = p_adj
        df["fdr_significant"] = df["p_value_adjusted"] < alpha

        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_session_ranges(
        self, df: pd.DataFrame, instrument: str,
    ) -> list[SessionRange]:
        """Identify all valid session ranges in the data.

        For each trading day and session:
        1. Extract the first range_minutes bars
        2. Compute range high and low
        3. Validate: >= MIN_RANGE_BARS and range_high > range_low
        """
        ranges: list[SessionRange] = []
        tz = df.index.tz
        dates = pd.Series(df.index.date).unique()

        for date in dates:
            for session_num, config in SESSION_CONFIG.items():
                if self.sessions is not None and session_num not in self.sessions:
                    continue
                session_start = pd.Timestamp(
                    year=date.year, month=date.month, day=date.day,
                    hour=config["start"].hour, minute=config["start"].minute,
                    tz=tz,
                )
                session_end = pd.Timestamp(
                    year=date.year, month=date.month, day=date.day,
                    hour=config["end"].hour, minute=config["end"].minute,
                    tz=tz,
                )
                range_end = session_start + pd.Timedelta(minutes=self.range_minutes)

                range_bars = df[
                    (df.index >= session_start) & (df.index < range_end)
                ]

                if len(range_bars) < MIN_RANGE_BARS:
                    continue

                range_high = range_bars["high"].max()
                range_low = range_bars["low"].min()

                if range_high <= range_low:
                    continue

                ranges.append(SessionRange(
                    date=pd.Timestamp(date),
                    session=session_num,
                    range_high=range_high,
                    range_low=range_low,
                    range_start=session_start,
                    range_end=range_end,
                    session_end=session_end,
                    instrument=instrument,
                ))

        return ranges

    def _compute_trade_params(
        self, range_info: SessionRange, direction: str,
    ) -> dict:
        """Compute entry, SL, TP for a range and direction.

        Long : entry = range_high, SL = range_low,  TP = entry + range * R:R
        Short: entry = range_low,  SL = range_high, TP = entry - range * R:R
        """
        range_size = range_info.range_high - range_info.range_low

        if direction == "long":
            entry = range_info.range_high
            sl = range_info.range_low
            tp = entry + range_size * self.rr_ratio
        else:
            entry = range_info.range_low
            sl = range_info.range_high
            tp = entry - range_size * self.rr_ratio

        return {"entry_price": entry, "stop_loss": sl, "take_profit": tp}

    @staticmethod
    def _detect_breakout(
        session_bars: pd.DataFrame,
        range_info: SessionRange,
    ) -> tuple[str, pd.Timestamp] | None:
        """Detect the first breakout above or below the session range.

        Returns (direction, entry_time) or None if no breakout.

        Tiebreaker when the same bar breaks both sides: compare with
        first_long <= first_short → long wins (consistent with verify_reentry.py).
        """
        long_breaks = session_bars.index[session_bars["high"] > range_info.range_high]
        short_breaks = session_bars.index[session_bars["low"] < range_info.range_low]

        first_long = long_breaks[0] if len(long_breaks) > 0 else None
        first_short = short_breaks[0] if len(short_breaks) > 0 else None

        if first_long is None and first_short is None:
            return None

        if first_long is not None and first_short is not None:
            if first_long <= first_short:
                return "long", first_long
            else:
                return "short", first_short
        elif first_long is not None:
            return "long", first_long
        else:
            return "short", first_short

    def _find_exit(
        self,
        remaining: pd.DataFrame,
        direction: str,
        stop_loss: float,
        take_profit: float,
        range_info: SessionRange,
    ) -> tuple[pd.Timestamp | None, float | None, str]:
        """Walk remaining bars to find the exit event.

        Priority order (earlier timestamp wins; same timestamp → SL > TP > reentry):
        1. Stop-loss
        2. Take-profit
        3. Range reentry (method controlled by self.reentry_check)
        4. Returns (None, None, 'session_end') if nothing triggered

        Args:
            remaining:   Bars after entry (exclusive) up to session_end (exclusive)
            direction:   'long' or 'short'
            stop_loss:   SL price
            take_profit: TP price
            range_info:  SessionRange (for reentry boundary and range_end origin)

        Returns:
            (exit_time, exit_price, exit_reason) — all None/str if no exit found.
        """
        if remaining.empty:
            return None, None, "session_end"

        # ── SL / TP hits ──────────────────────────────────────────────
        if direction == "long":
            sl_mask = remaining["low"] <= stop_loss
            tp_mask = remaining["high"] >= take_profit
        else:
            sl_mask = remaining["high"] >= stop_loss
            tp_mask = remaining["low"] <= take_profit

        sl_hits = remaining.index[sl_mask]
        tp_hits = remaining.index[tp_mask]

        first_sl = sl_hits[0] if len(sl_hits) > 0 else None
        first_tp = tp_hits[0] if len(tp_hits) > 0 else None

        # ── Reentry detection ─────────────────────────────────────────
        first_reentry: pd.Timestamp | None = None

        if self.reentry_check == "none":
            pass  # disabled

        elif self.reentry_check == "strict":
            # Any 1-min bar whose close is inside the range
            inside = (
                (remaining["close"] >= range_info.range_low)
                & (remaining["close"] <= range_info.range_high)
            )
            hits = remaining.index[inside]
            first_reentry = hits[0] if len(hits) > 0 else None

        elif self.reentry_check in ("15min", "30min"):
            freq = self.reentry_check  # "15min" or "30min"
            if not remaining.empty:
                closes = (
                    remaining["close"]
                    .resample(freq, origin=range_info.range_end)
                    .last()
                    .dropna()
                )
                inside_closes = closes[
                    (closes >= range_info.range_low)
                    & (closes <= range_info.range_high)
                ]
                for bin_start in inside_closes.index:
                    bin_end = bin_start + pd.Timedelta(freq)
                    bin_bars = remaining.loc[
                        (remaining.index >= bin_start)
                        & (remaining.index < bin_end)
                    ]
                    if not bin_bars.empty:
                        first_reentry = bin_bars.index[-1]
                        break

        else:
            raise ValueError(
                f"Unknown reentry_check: {self.reentry_check!r}. "
                "Expected one of: '15min', '30min', 'none', 'strict'."
            )

        # ── Resolve exit priority ─────────────────────────────────────
        # Earlier timestamp wins; ties broken by SL(0) > TP(1) > reentry(2)
        candidates: list[tuple[pd.Timestamp, str, int, float]] = []
        if first_sl is not None:
            candidates.append((first_sl, "sl", 0, stop_loss))
        if first_tp is not None:
            candidates.append((first_tp, "tp", 1, take_profit))
        if first_reentry is not None:
            reentry_price = remaining.loc[first_reentry, "close"]
            candidates.append((first_reentry, "range_reentry", 2, reentry_price))

        if not candidates:
            return None, None, "session_end"

        candidates.sort(key=lambda x: (x[0], x[2]))
        exit_time, exit_reason, _, exit_price = candidates[0]
        return exit_time, exit_price, exit_reason
