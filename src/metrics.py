"""Performance metrics calculation for the backtester engine."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .trade import Trade


def _bootstrap_ci(trades_pnl: pd.Series, n_boot: int = 1000, ci: float = 0.95) -> dict:
    """Bootstrap confidence intervals for profit factor and Sharpe ratio."""
    rng = np.random.default_rng(42)
    n = len(trades_pnl)
    if n < 10:
        return {"pf_ci_low": 0.0, "pf_ci_high": 0.0, "sharpe_ci_low": 0.0, "sharpe_ci_high": 0.0}

    pf_samples = []
    sharpe_samples = []

    for _ in range(n_boot):
        sample = trades_pnl.iloc[rng.integers(0, n, size=n)]
        gross_win = sample[sample > 0].sum()
        gross_loss = abs(sample[sample <= 0].sum())
        pf = gross_win / gross_loss if gross_loss > 0 else 0.0
        pf_samples.append(pf)

        if sample.std() > 0:
            sharpe_samples.append(sample.mean() / sample.std() * (252 ** 0.5))
        else:
            sharpe_samples.append(0.0)

    alpha = (1 - ci) / 2
    pf_arr = np.array(pf_samples)
    sh_arr = np.array(sharpe_samples)

    return {
        "pf_ci_low": float(np.percentile(pf_arr, alpha * 100)),
        "pf_ci_high": float(np.percentile(pf_arr, (1 - alpha) * 100)),
        "sharpe_ci_low": float(np.percentile(sh_arr, alpha * 100)),
        "sharpe_ci_high": float(np.percentile(sh_arr, (1 - alpha) * 100)),
    }


def calculate_metrics(trades: list[Trade]) -> dict:
    """Calculate all performance metrics.

    Metrics computed:
    - Win rate, profit factor, total PnL, average PnL
    - Max drawdown, Sharpe ratio (annualized, 252 days)
    - Performance by session, instrument, and day of week
    - Exit reason distribution
    """
    if not trades:
        return {"total_trades": 0}

    df = pd.DataFrame([
        {
            "entry_time": t.entry_time,
            "exit_time": t.exit_time,
            "direction": t.direction,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "pnl_ticks": t.pnl_ticks,
            "pnl_dollars": t.pnl_dollars,
            "exit_reason": t.exit_reason,
            "session": t.session,
            "instrument": t.instrument,
            "date": t.date,
        }
        for t in trades
    ])

    total_trades = len(df)
    winners = df["pnl_dollars"] > 0
    losers = ~winners

    win_rate = winners.sum() / total_trades

    gross_profit = df.loc[winners, "pnl_dollars"].sum()
    gross_loss = abs(df.loc[losers, "pnl_dollars"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    total_pnl = df["pnl_dollars"].sum()
    avg_pnl = df["pnl_dollars"].mean()

    # Max drawdown (on cumulative PnL curve, chronological order)
    df_sorted = df.sort_values("entry_time")
    cumulative = df_sorted["pnl_dollars"].cumsum()
    max_drawdown = (cumulative - cumulative.cummax()).min()

    # Annualized Sharpe ratio
    df["entry_date"] = df["entry_time"].dt.date
    daily_pnl = df.groupby("entry_date")["pnl_dollars"].sum()
    if len(daily_pnl) > 1 and daily_pnl.std() > 0:
        sharpe = daily_pnl.mean() / daily_pnl.std() * (252 ** 0.5)
    else:
        sharpe = 0.0

    # Bootstrap confidence intervals for profit factor and Sharpe
    bootstrap = _bootstrap_ci(df["pnl_dollars"])

    # Profit concentration: % of total PnL from top 10% of trades (by absolute PnL)
    top_10pct_count = max(1, int(len(df) * 0.1))
    sorted_by_pnl = df["pnl_dollars"].abs().sort_values(ascending=False)
    top_10pct_pnl = df.loc[sorted_by_pnl.head(top_10pct_count).index, "pnl_dollars"].sum()
    profit_concentration = abs(top_10pct_pnl / total_pnl) if total_pnl != 0 else 0.0

    # Drawdown duration: max consecutive trades where cumulative PnL was below its high-water mark
    cumulative = df_sorted["pnl_dollars"].cumsum()
    underwater = cumulative < cumulative.cummax()
    if underwater.any():
        # Group consecutive underwater periods
        groups = (~underwater).cumsum()
        underwater_groups = underwater.groupby(groups).sum()
        max_dd_days = int(underwater_groups.max()) if len(underwater_groups) > 0 else 0
    else:
        max_dd_days = 0

    # Monthly PnL breakdown
    df_sorted["month"] = df_sorted["entry_time"].dt.to_period("M").astype(str)
    by_month = df_sorted.groupby("month")["pnl_dollars"].agg(
        trades="count", total="sum", moyenne="mean"
    )
    by_month["win_rate"] = df_sorted.groupby("month")["pnl_dollars"].apply(
        lambda x: (x > 0).mean()
    )

    # Performance by session
    _agg = {"trades": "count", "total": "sum", "moyenne": "mean"}
    by_session = df.groupby("session")["pnl_dollars"].agg(**_agg)
    by_session["win_rate"] = df.groupby("session")["pnl_dollars"].apply(
        lambda x: (x > 0).mean()
    )

    # Performance by instrument
    by_instrument = df.groupby("instrument")["pnl_dollars"].agg(**_agg)
    by_instrument["win_rate"] = df.groupby("instrument")["pnl_dollars"].apply(
        lambda x: (x > 0).mean()
    )

    # Performance by day of week
    df["weekday"] = pd.to_datetime(df["date"]).dt.day_name()
    by_weekday = df.groupby("weekday")["pnl_dollars"].agg(**_agg)
    by_weekday["win_rate"] = df.groupby("weekday")["pnl_dollars"].apply(
        lambda x: (x > 0).mean()
    )
    # Order Monday → Friday
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    by_weekday = by_weekday.reindex([d for d in day_order if d in by_weekday.index])

    # Exit reason distribution
    exit_reasons = df["exit_reason"].value_counts()

    # Performance by direction
    long_trades = df[df["direction"] == "long"]
    short_trades = df[df["direction"] == "short"]

    return {
        "total_trades": total_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "total_pnl": total_pnl,
        "avg_pnl_per_trade": avg_pnl,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe,
        "by_session": by_session,
        "by_instrument": by_instrument,
        "by_weekday": by_weekday,
        "exit_reasons": exit_reasons,
        "pnl_long": long_trades["pnl_dollars"].sum() if len(long_trades) > 0 else 0.0,
        "pnl_short": short_trades["pnl_dollars"].sum() if len(short_trades) > 0 else 0.0,
        "trades_long": len(long_trades),
        "trades_short": len(short_trades),
        "win_rate_long": (long_trades["pnl_dollars"] > 0).mean() if len(long_trades) > 0 else 0.0,
        "win_rate_short": (short_trades["pnl_dollars"] > 0).mean() if len(short_trades) > 0 else 0.0,
        "profit_concentration": profit_concentration,
        "max_dd_duration_trades": max_dd_days,
        "by_month": by_month,
        "pf_ci_low": bootstrap["pf_ci_low"],
        "pf_ci_high": bootstrap["pf_ci_high"],
        "sharpe_ci_low": bootstrap["sharpe_ci_low"],
        "sharpe_ci_high": bootstrap["sharpe_ci_high"],
    }


def print_report(metrics: dict) -> None:
    """Print a formatted backtest report."""
    if metrics.get("total_trades", 0) == 0:
        print("No trades executed.")
        return

    print()
    print("=" * 60)
    print("  BACKTEST REPORT")
    print("=" * 60)

    print(f"\n  Total trades          : {metrics['total_trades']}")
    print(f"  Win rate              : {metrics['win_rate']:.1%}")
    print(f"  Profit factor         : {metrics['profit_factor']:.2f}")
    print(f"  Total PnL             : ${metrics['total_pnl']:>10,.2f}")
    print(f"  Avg PnL per trade     : ${metrics['avg_pnl_per_trade']:>10,.2f}")
    print(f"  Max drawdown          : ${metrics['max_drawdown']:>10,.2f}")
    print(f"  Sharpe ratio          : {metrics['sharpe_ratio']:.2f}")

    print("\n  --- Performance by session ---")
    print(metrics["by_session"].to_string())

    print("\n  --- Performance by instrument ---")
    print(metrics["by_instrument"].to_string())

    print("\n  --- Performance by day of week ---")
    print(metrics["by_weekday"].to_string())

    print("\n  --- Exit reasons ---")
    for reason, count in metrics["exit_reasons"].items():
        pct = count / metrics["total_trades"] * 100
        print(f"  {reason:<16} : {count:>5}  ({pct:.1f}%)")

    print("\n" + "=" * 60)
