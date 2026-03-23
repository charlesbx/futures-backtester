"""Script principal pour exécuter le backtest SRS.

Usage:
    python run_backtest.py
"""

import json
from pathlib import Path

from src.backtester import run_backtest
from src.data_loader import load_and_prepare, load_processed, save_processed
from src.metrics import calculate_metrics, print_report
from src.strategies.srs import SRSStrategy

DATA_PATH = Path("data/raw/glbx-mdp3-20100606-20260315.ohlcv-1m.dbn.zst")
PROCESSED_DIR = Path("data/processed")


def get_data(instrument: str):
    """Charge les données (depuis le cache Parquet si disponible)."""
    cache = PROCESSED_DIR / f"{instrument}_1m.parquet"
    if cache.exists():
        print(f"  Chargement depuis le cache : {cache}")
        return load_processed(cache)

    print(f"  Traitement des données brutes...")
    df = load_and_prepare(DATA_PATH, instrument)
    save_processed(df, PROCESSED_DIR, instrument)
    print(f"  Cache sauvegardé : {cache}")
    return df


def main():
    # Paramètres de la stratégie
    strategy = SRSStrategy(rr_ratio=2.25, range_minutes=30)

    data = {}
    for instrument in ["MES", "MNQ"]:
        print(f"\n{'='*40}")
        print(f"  {instrument}")
        print(f"{'='*40}")

        df = get_data(instrument)
        print(f"  {len(df):,} barres | {df.index.min()} → {df.index.max()}")
        data[instrument] = df

    # Commission NinjaTrader micro futures : ~$0.62/side = $1.24 aller-retour
    all_trades = run_backtest(strategy, data, slippage_ticks=1, commission=1.24)
    print(f"\n  {len(all_trades)} trades trouvés")

    # Rapport global
    metrics = calculate_metrics(all_trades)
    print_report(metrics)

    # Save baseline metrics to results/baseline.json
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    baseline = {
        "total_trades": metrics.get("total_trades", 0),
        "win_rate": metrics.get("win_rate", 0.0),
        "profit_factor": metrics.get("profit_factor", 0.0),
        "total_pnl": metrics.get("total_pnl", 0.0),
        "sharpe_ratio": metrics.get("sharpe_ratio", 0.0),
    }
    baseline_path = results_dir / "baseline.json"
    with open(baseline_path, "w") as f:
        json.dump(baseline, f, indent=2)
    print(f"\n  Baseline saved to {baseline_path}")


if __name__ == "__main__":
    main()
