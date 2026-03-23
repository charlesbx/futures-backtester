"""Trade-level diagnostic analysis for SRS strategy.

Compares baseline vs best configuration at the individual trade level.
Generates 12 charts answering:
  Chapter 1: "Is there a signal?"
  Chapter 2: "Where is the edge?"

Usage:
    python scripts/analyze_trades.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from src.backtester import run_backtest
from src.data_loader import INSTRUMENTS, load_processed
from src.strategies.srs import SRSStrategy
from src.trade import Trade

PROJECT_ROOT = Path(__file__).parent.parent

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "results" / "analysis"

CONFIGS = {
    "baseline": {
        "strategy": {"rr_ratio": 2.25, "range_minutes": 30, "sessions": [1, 2]},
        "backtester": {"reentry_check": "15min"},
    },
    "best": {
        "strategy": {"rr_ratio": 3.0, "range_minutes": 60, "sessions": [1]},
        "backtester": {"reentry_check": "none"},
    },
}

RANGE_BINS = [0, 10, 20, 30, 50, float("inf")]
RANGE_LABELS = ["0-10", "10-20", "20-30", "30-50", "50+"]
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def load_data() -> dict[str, pd.DataFrame]:
    """Load MES and MNQ data from Parquet cache."""
    data = {}
    for instrument in ["MES", "MNQ"]:
        cache = PROCESSED_DIR / f"{instrument}_1m.parquet"
        if not cache.exists():
            raise FileNotFoundError(
                f"{cache} not found. Run run_backtest.py first to generate cache."
            )
        data[instrument] = load_processed(cache)
        print(f"  {instrument}: {len(data[instrument]):,} bars")
    return data


def run_config(data: dict[str, pd.DataFrame], config: dict) -> list[Trade]:
    """Run backtester with a given config across both instruments."""
    strategy = SRSStrategy(**config["strategy"], **config.get("backtester", {}))
    return run_backtest(strategy, data, slippage_ticks=1, commission=1.24)


def trades_to_dataframe(trades: list[Trade], config_name: str) -> pd.DataFrame:
    """Convert Trade list to DataFrame with derived columns."""
    rows = []
    for t in trades:
        tick_size = INSTRUMENTS[t.instrument]["tick_size"]
        range_size_ticks = (t.range_high - t.range_low) / tick_size
        duration_minutes = (t.exit_time - t.entry_time).total_seconds() / 60.0
        rows.append({
            "config": config_name,
            "entry_time": t.entry_time,
            "exit_time": t.exit_time,
            "direction": t.direction,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "stop_loss": t.stop_loss,
            "take_profit": t.take_profit,
            "pnl_ticks": t.pnl_ticks,
            "pnl_dollars": t.pnl_dollars,
            "exit_reason": t.exit_reason,
            "session": t.session,
            "instrument": t.instrument,
            "date": t.date,
            "range_high": t.range_high,
            "range_low": t.range_low,
            "range_size_ticks": range_size_ticks,
            "duration_minutes": duration_minutes,
            "year": t.entry_time.year,
            "weekday": t.entry_time.day_name(),
            "quarter": f"{t.entry_time.year}-Q{t.entry_time.quarter}",
            "pnl_category": "win" if t.pnl_dollars > 0 else "loss",
        })
    return pd.DataFrame(rows)


def build_trade_data(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Run both configs and return combined DataFrame."""
    frames = []
    for name, config in CONFIGS.items():
        print(f"\nRunning {name} config...")
        trades = run_config(data, config)
        print(f"  {len(trades)} trades")
        df = trades_to_dataframe(trades, name)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def print_summary(df: pd.DataFrame) -> None:
    """Print comparison table of both configs to console."""
    print("\n" + "=" * 70)
    print("  SUMMARY: BASELINE vs BEST CONFIG")
    print("=" * 70)

    header = f"{'Metric':<25} {'Baseline':>20} {'Best':>20}"
    print(header)
    print("-" * 65)

    baseline_metrics = {}
    best_metrics = {}

    for config_name in ["baseline", "best"]:
        subset = df[df["config"] == config_name]

        # Compute Sharpe and drawdown upfront
        sharpe_str = "—"
        dd_str = "—"
        if len(subset) > 0:
            sorted_sub = subset.sort_values("entry_time")
            cumulative = sorted_sub["pnl_dollars"].cumsum()
            dd_str = f"${(cumulative - cumulative.cummax()).min():,.0f}"

            daily_pnl = subset.groupby(subset["entry_time"].dt.date)["pnl_dollars"].sum()
            if len(daily_pnl) > 1 and daily_pnl.std() > 0:
                sharpe_str = f"{daily_pnl.mean() / daily_pnl.std() * (252 ** 0.5):.3f}"

        yearly = subset.groupby("year")["pnl_dollars"].sum() if len(subset) > 0 else pd.Series(dtype=float)

        metrics = {
            "Total trades": len(subset),
            "Win rate": f"{(subset['pnl_dollars'] > 0).mean():.1%}" if len(subset) > 0 else "N/A",
            "Profit factor": f"{subset.loc[subset['pnl_dollars'] > 0, 'pnl_dollars'].sum() / max(abs(subset.loc[subset['pnl_dollars'] <= 0, 'pnl_dollars'].sum()), 1e-9):.3f}" if len(subset) > 0 else "N/A",
            "Total PnL": f"${subset['pnl_dollars'].sum():,.0f}" if len(subset) > 0 else "$0",
            "Sharpe ratio": sharpe_str,
            "Max drawdown": dd_str,
            "Avg winner ($)": f"${subset.loc[subset['pnl_dollars'] > 0, 'pnl_dollars'].mean():,.2f}" if (subset['pnl_dollars'] > 0).any() else "$0",
            "Avg loser ($)": f"${subset.loc[subset['pnl_dollars'] <= 0, 'pnl_dollars'].mean():,.2f}" if (subset['pnl_dollars'] <= 0).any() else "$0",
            "Best year PnL": f"${yearly.max():,.0f}" if len(yearly) > 0 else "N/A",
            "Worst year PnL": f"${yearly.min():,.0f}" if len(yearly) > 0 else "N/A",
        }

        if config_name == "baseline":
            baseline_metrics = metrics
        else:
            best_metrics = metrics

    for key in baseline_metrics:
        print(f"  {key:<23} {baseline_metrics[key]:>20} {best_metrics[key]:>20}")

    print("=" * 70)


# ── Chapter 1: "Is there a signal?" ─────────────────────────────────────


def chart_01_equity_curves(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Cumulative PnL over time for both configs."""
    for config_name, color in [("baseline", "tab:red"), ("best", "tab:blue")]:
        subset = df[df["config"] == config_name].sort_values("entry_time")
        cumulative = subset["pnl_dollars"].cumsum()
        ax.plot(subset["entry_time"].values, cumulative.values, label=config_name, color=color, linewidth=0.8)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative PnL ($)")
    ax.set_title("Chart 1: Equity Curves — Baseline vs Best")
    ax.legend()
    ax.grid(True, alpha=0.3)


def chart_02_rolling_pf(df: pd.DataFrame, ax: plt.Axes, window: int = 100) -> None:
    """Rolling profit factor over a N-trade window."""
    for config_name, color in [("baseline", "tab:red"), ("best", "tab:blue")]:
        subset = df[df["config"] == config_name].sort_values("entry_time").reset_index(drop=True)
        wins = subset["pnl_dollars"].clip(lower=0).rolling(window).sum()
        losses = subset["pnl_dollars"].clip(upper=0).abs().rolling(window).sum()
        rolling_pf = wins / losses.replace(0, np.nan)
        ax.plot(subset["entry_time"].values, rolling_pf.values, label=config_name, color=color, linewidth=0.8)
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Profit Factor")
    ax.set_title(f"Chart 2: Rolling Profit Factor ({window}-trade window)")
    ax.legend()
    ax.set_ylim(0, 3)
    ax.grid(True, alpha=0.3)


def chart_03_pnl_by_year(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Annual PnL grouped bar chart."""
    pivot = df.pivot_table(values="pnl_dollars", index="year", columns="config", aggfunc="sum")
    pivot = pivot.reindex(columns=["baseline", "best"])
    pivot.plot(kind="bar", ax=ax, color=["tab:red", "tab:blue"])
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Year")
    ax.set_ylabel("Total PnL ($)")
    ax.set_title("Chart 3: PnL by Year")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(axis="x", rotation=0)


def chart_04_pnl_distribution(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Overlapping histograms of per-trade PnL."""
    for config_name, color in [("baseline", "tab:red"), ("best", "tab:blue")]:
        subset = df[df["config"] == config_name]
        ax.hist(subset["pnl_dollars"], bins=80, alpha=0.5, label=config_name, color=color, density=True)
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("PnL per Trade ($)")
    ax.set_ylabel("Density")
    ax.set_title("Chart 4: PnL Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)


def chart_05_exit_reasons(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Stacked percentage bar chart of exit reasons."""
    reasons = ["tp", "sl", "range_reentry", "session_end"]
    data = {}
    for config_name in ["baseline", "best"]:
        subset = df[df["config"] == config_name]
        total = len(subset) or 1
        data[config_name] = [len(subset[subset["exit_reason"] == r]) / total * 100 for r in reasons]

    x = np.arange(len(["baseline", "best"]))
    bottom = np.zeros(2)
    colors = ["tab:green", "tab:red", "tab:orange", "tab:gray"]
    for i, reason in enumerate(reasons):
        values = [data["baseline"][i], data["best"][i]]
        ax.bar(x, values, bottom=bottom, label=reason, color=colors[i])
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(["baseline", "best"])
    ax.set_ylabel("% of Trades")
    ax.set_title("Chart 5: Exit Reason Breakdown")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 105)


# ── Chapter 2: "Where is the edge?" ─────────────────────────────────────


def chart_06_pnl_by_range_size(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Average PnL per trade by range size bin."""
    df = df.copy()
    df["range_bin"] = pd.cut(df["range_size_ticks"], bins=RANGE_BINS, labels=RANGE_LABELS, right=False)
    pivot = df.pivot_table(values="pnl_dollars", index="range_bin", columns="config", aggfunc="mean")
    pivot = pivot.reindex(columns=["baseline", "best"])
    pivot.plot(kind="bar", ax=ax, color=["tab:red", "tab:blue"])
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Range Size (ticks)")
    ax.set_ylabel("Avg PnL per Trade ($)")
    ax.set_title("Chart 6: PnL by Range Size")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(axis="x", rotation=0)


def chart_07_pnl_by_weekday(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Average PnL per trade by day of week."""
    pivot = df.pivot_table(values="pnl_dollars", index="weekday", columns="config", aggfunc="mean")
    pivot = pivot.reindex(index=DAY_ORDER).reindex(columns=["baseline", "best"])
    pivot.plot(kind="bar", ax=ax, color=["tab:red", "tab:blue"])
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Avg PnL per Trade ($)")
    ax.set_title("Chart 7: PnL by Day of Week")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(axis="x", rotation=0)


def chart_08_pnl_by_direction(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Total PnL and win rate by direction."""
    pivot_pnl = df.pivot_table(values="pnl_dollars", index="direction", columns="config", aggfunc="sum")
    pivot_pnl = pivot_pnl.reindex(columns=["baseline", "best"])
    pivot_pnl.plot(kind="bar", ax=ax, color=["tab:red", "tab:blue"])
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Direction")
    ax.set_ylabel("Total PnL ($)")
    ax.set_title("Chart 8: PnL by Direction")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(axis="x", rotation=0)

    # Annotate win rates
    for i, direction in enumerate(pivot_pnl.index):
        for j, config_name in enumerate(["baseline", "best"]):
            subset = df[(df["config"] == config_name) & (df["direction"] == direction)]
            wr = (subset["pnl_dollars"] > 0).mean()
            bar_x = i + (j - 0.5) * 0.25
            ax.annotate(f"{wr:.0%}", (bar_x, 0), textcoords="offset points",
                        xytext=(0, -15), ha="center", fontsize=7, color="white", fontweight="bold")


def chart_09_pnl_by_instrument(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Total PnL and trade count by instrument."""
    pivot = df.pivot_table(values="pnl_dollars", index="instrument", columns="config", aggfunc="sum")
    pivot = pivot.reindex(columns=["baseline", "best"])
    pivot.plot(kind="bar", ax=ax, color=["tab:red", "tab:blue"])
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Instrument")
    ax.set_ylabel("Total PnL ($)")
    ax.set_title("Chart 9: PnL by Instrument")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(axis="x", rotation=0)

    # Annotate trade counts
    for i, instrument in enumerate(pivot.index):
        for j, config_name in enumerate(["baseline", "best"]):
            subset = df[(df["config"] == config_name) & (df["instrument"] == instrument)]
            n = len(subset)
            bar_x = i + (j - 0.5) * 0.25
            ax.annotate(f"n={n}", (bar_x, 0), textcoords="offset points",
                        xytext=(0, -15), ha="center", fontsize=7, color="white", fontweight="bold")


def chart_10_range_vs_outcome(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Scatter: range size vs PnL, best config only."""
    best = df[df["config"] == "best"]
    wins = best[best["pnl_category"] == "win"]
    losses = best[best["pnl_category"] == "loss"]
    ax.scatter(losses["range_size_ticks"], losses["pnl_dollars"],
               alpha=0.3, s=8, color="tab:red", label="loss")
    ax.scatter(wins["range_size_ticks"], wins["pnl_dollars"],
               alpha=0.3, s=8, color="tab:green", label="win")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Range Size (ticks)")
    ax.set_ylabel("PnL per Trade ($)")
    ax.set_title("Chart 10: Range Size vs Outcome (Best Config)")
    ax.legend()
    ax.grid(True, alpha=0.3)


def chart_11_winrate_heatmap(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Win rate heatmap: range size bins x quarter, best config only."""
    best = df[df["config"] == "best"].copy()
    if best.empty:
        ax.text(0.5, 0.5, "No trades", transform=ax.transAxes, ha="center", va="center")
        ax.set_title("Chart 11: Win Rate Heatmap (Best Config, n<10 grayed)")
        return
    best["range_bin"] = pd.cut(best["range_size_ticks"], bins=RANGE_BINS, labels=RANGE_LABELS, right=False)
    best["is_win"] = (best["pnl_dollars"] > 0).astype(int)

    pivot_wr = best.pivot_table(values="is_win", index="range_bin", columns="quarter", aggfunc="mean")
    pivot_count = best.pivot_table(values="is_win", index="range_bin", columns="quarter", aggfunc="count")

    # Mask cells with < 10 trades
    mask = pivot_count < 10
    pivot_display = pivot_wr.copy()
    pivot_display[mask] = np.nan

    im = ax.imshow(pivot_display.values, aspect="auto", cmap="RdYlGn", vmin=0.2, vmax=0.8)
    ax.set_xticks(range(len(pivot_display.columns)))
    ax.set_xticklabels(pivot_display.columns, rotation=45, ha="right", fontsize=6)
    ax.set_yticks(range(len(pivot_display.index)))
    ax.set_yticklabels(pivot_display.index)
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Range Size (ticks)")
    ax.set_title("Chart 11: Win Rate Heatmap (Best Config, n<10 grayed)")
    plt.colorbar(im, ax=ax, label="Win Rate")


def chart_12_duration_by_exit(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Box plot of trade duration by exit reason, both configs."""
    exit_order = ["tp", "sl", "range_reentry", "session_end"]
    plot_data = []
    labels = []
    positions = []
    colors = []
    pos = 0
    for reason in exit_order:
        for config_name, color in [("baseline", "tab:red"), ("best", "tab:blue")]:
            subset = df[(df["config"] == config_name) & (df["exit_reason"] == reason)]
            if len(subset) > 0:
                plot_data.append(subset["duration_minutes"].values)
                labels.append(f"{reason}\n{config_name}")
                positions.append(pos)
                colors.append(color)
            pos += 1
        pos += 0.5  # gap between exit reason groups

    if not plot_data:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        ax.set_title("Chart 12: Trade Duration by Exit Reason")
        return

    bp = ax.boxplot(plot_data, positions=positions, widths=0.8, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Duration (minutes)")
    ax.set_title("Chart 12: Trade Duration by Exit Reason")
    ax.grid(True, alpha=0.3, axis="y")


# ── Orchestration ────────────────────────────────────────────────────────


CHART_FUNCTIONS = [
    ("01_equity_curves", chart_01_equity_curves),
    ("02_rolling_pf", chart_02_rolling_pf),
    ("03_pnl_by_year", chart_03_pnl_by_year),
    ("04_pnl_distribution", chart_04_pnl_distribution),
    ("05_exit_reasons", chart_05_exit_reasons),
    ("06_pnl_by_range_size", chart_06_pnl_by_range_size),
    ("07_pnl_by_weekday", chart_07_pnl_by_weekday),
    ("08_pnl_by_direction", chart_08_pnl_by_direction),
    ("09_pnl_by_instrument", chart_09_pnl_by_instrument),
    ("10_range_vs_outcome", chart_10_range_vs_outcome),
    ("11_winrate_heatmap", chart_11_winrate_heatmap),
    ("12_duration_by_exit", chart_12_duration_by_exit),
]


def generate_charts(df: pd.DataFrame) -> None:
    """Generate all 12 charts as individual PNGs and a combined PDF."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_path = OUTPUT_DIR / "trade_analysis.pdf"
    with PdfPages(pdf_path) as pdf:
        for name, chart_fn in CHART_FUNCTIONS:
            print(f"  Generating {name}...")
            fig, ax = plt.subplots(figsize=(12, 6))
            chart_fn(df, ax)
            fig.tight_layout()

            # Save individual PNG
            fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=150, bbox_inches="tight")

            # Add to PDF
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"\n  PDF saved to {pdf_path}")
    print(f"  PNGs saved to {OUTPUT_DIR}/")


def main():
    print("Loading data...")
    data = load_data()

    df = build_trade_data(data)
    print(f"\nTotal trades in analysis: {len(df):,}")

    print_summary(df)

    print("\nGenerating charts...")
    generate_charts(df)

    print("\nDone.")


if __name__ == "__main__":
    main()
