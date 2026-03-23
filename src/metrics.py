"""Performance metrics calculation for the backtester engine."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from .trade import Trade


def calculate_metrics(trades: list[Trade]) -> dict:
    """Calcule toutes les métriques de performance.

    Métriques calculées :
    - Win rate, profit factor, total PnL, PnL moyen
    - Max drawdown, Sharpe ratio (annualisé, 252 jours)
    - Performance par session, par instrument, par jour de la semaine
    - Distribution des raisons de sortie
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

    # Max drawdown (sur la courbe de PnL cumulé, ordre chronologique)
    df_sorted = df.sort_values("entry_time")
    cumulative = df_sorted["pnl_dollars"].cumsum()
    max_drawdown = (cumulative - cumulative.cummax()).min()

    # Sharpe ratio annualisé
    df["entry_date"] = df["entry_time"].dt.date
    daily_pnl = df.groupby("entry_date")["pnl_dollars"].sum()
    if len(daily_pnl) > 1 and daily_pnl.std() > 0:
        sharpe = daily_pnl.mean() / daily_pnl.std() * (252 ** 0.5)
    else:
        sharpe = 0.0

    # Performance par session
    _agg = {"trades": "count", "total": "sum", "moyenne": "mean"}
    by_session = df.groupby("session")["pnl_dollars"].agg(**_agg)
    by_session["win_rate"] = df.groupby("session")["pnl_dollars"].apply(
        lambda x: (x > 0).mean()
    )

    # Performance par instrument
    by_instrument = df.groupby("instrument")["pnl_dollars"].agg(**_agg)
    by_instrument["win_rate"] = df.groupby("instrument")["pnl_dollars"].apply(
        lambda x: (x > 0).mean()
    )

    # Performance par jour de la semaine
    df["weekday"] = pd.to_datetime(df["date"]).dt.day_name()
    by_weekday = df.groupby("weekday")["pnl_dollars"].agg(**_agg)
    by_weekday["win_rate"] = df.groupby("weekday")["pnl_dollars"].apply(
        lambda x: (x > 0).mean()
    )
    # Ordonner lundi → vendredi
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    by_weekday = by_weekday.reindex([d for d in day_order if d in by_weekday.index])

    # Distribution des raisons de sortie
    exit_reasons = df["exit_reason"].value_counts()

    # Performance par direction
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
    }


def print_report(metrics: dict) -> None:
    """Affiche un rapport formaté des métriques de backtest."""
    if metrics.get("total_trades", 0) == 0:
        print("Aucun trade exécuté.")
        return

    print()
    print("=" * 60)
    print("  RAPPORT DE BACKTEST — SRS (Session Range Strategy)")
    print("=" * 60)

    print(f"\n  Nombre total de trades : {metrics['total_trades']}")
    print(f"  Win rate              : {metrics['win_rate']:.1%}")
    print(f"  Profit factor         : {metrics['profit_factor']:.2f}")
    print(f"  PnL total             : ${metrics['total_pnl']:>10,.2f}")
    print(f"  PnL moyen par trade   : ${metrics['avg_pnl_per_trade']:>10,.2f}")
    print(f"  Max drawdown          : ${metrics['max_drawdown']:>10,.2f}")
    print(f"  Sharpe ratio          : {metrics['sharpe_ratio']:.2f}")

    print("\n  --- Performance par session ---")
    print(metrics["by_session"].to_string())

    print("\n  --- Performance par instrument ---")
    print(metrics["by_instrument"].to_string())

    print("\n  --- Performance par jour de la semaine ---")
    print(metrics["by_weekday"].to_string())

    print("\n  --- Raisons de sortie ---")
    for reason, count in metrics["exit_reasons"].items():
        pct = count / metrics["total_trades"] * 100
        print(f"  {reason:<16} : {count:>5}  ({pct:.1f}%)")

    print("\n" + "=" * 60)
