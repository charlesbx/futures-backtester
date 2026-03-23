"""Chargement et nettoyage des données Databento pour le backtester SRS.

Gère le rollover par volume quotidien et la conversion UTC → Eastern Time.
"""

import re
from pathlib import Path

import databento as db
import pandas as pd

# Tick specifications par instrument
INSTRUMENTS = {
    "MES": {"tick_size": 0.25, "tick_value": 1.25},
    "MNQ": {"tick_size": 0.25, "tick_value": 0.50},
}

# Pattern pour les contrats individuels (exclut les spreads comme MESM9-MESU9)
_CONTRACT_RE = re.compile(r"^(MES|MNQ)[HMUZ]\d$")


def load_raw_dbn(path: str | Path) -> pd.DataFrame:
    """Charge un fichier DBN Databento et retourne un DataFrame OHLCV 1-min."""
    store = db.DBNStore.from_file(str(path))
    return store.to_df()


def filter_individual_contracts(df: pd.DataFrame) -> pd.DataFrame:
    """Garde uniquement les contrats individuels MES/MNQ (exclut les spreads)."""
    mask = df["symbol"].str.match(r"^(MES|MNQ)[HMUZ]\d$")
    return df[mask].copy()


def build_continuous_series(df: pd.DataFrame, instrument: str) -> pd.DataFrame:
    """Construit une série continue pour un instrument via rollover par volume.

    Pour chaque jour de trading (date UTC), sélectionne le contrat avec
    le plus grand volume quotidien. Gère automatiquement le roll.
    """
    # Filtrer sur l'instrument (contrats individuels uniquement)
    mask = df["symbol"].apply(lambda s: bool(_CONTRACT_RE.match(s)) and s.startswith(instrument))
    inst_df = df[mask].copy()

    # Date de trading (date UTC — suffisant pour le rollover quotidien)
    inst_df["trade_date"] = inst_df.index.normalize()

    # Volume quotidien par contrat
    daily_vol = (
        inst_df.groupby(["trade_date", "symbol"])["volume"]
        .sum()
        .reset_index()
    )

    # Front-month = contrat avec le max de volume chaque jour
    idx_max = daily_vol.groupby("trade_date")["volume"].idxmax()
    front_map = daily_vol.loc[idx_max].set_index("trade_date")["symbol"]

    # Ne garder que les barres du contrat front-month
    inst_df["front"] = inst_df["trade_date"].map(front_map)
    result = inst_df[inst_df["symbol"] == inst_df["front"]].copy()
    result.drop(columns=["trade_date", "front"], inplace=True)
    result.sort_index(inplace=True)

    return result


def convert_to_eastern(df: pd.DataFrame) -> pd.DataFrame:
    """Convertit l'index UTC en Eastern Time (gère EST/EDT automatiquement)."""
    df = df.copy()
    df.index = df.index.tz_convert("US/Eastern")
    return df


def save_processed(df: pd.DataFrame, output_dir: str | Path, instrument: str) -> Path:
    """Sauvegarde les données préparées en Parquet."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{instrument}_1m.parquet"
    df.to_parquet(path)
    return path


def load_processed(path: str | Path) -> pd.DataFrame:
    """Charge des données préparées depuis un fichier Parquet."""
    return pd.read_parquet(str(path))


def load_and_prepare(path: str | Path, instrument: str) -> pd.DataFrame:
    """Pipeline complet : charge DBN → filtre → rollover → Eastern Time.

    Args:
        path: Chemin vers le fichier DBN (.dbn.zst)
        instrument: 'MES' ou 'MNQ'

    Returns:
        DataFrame OHLCV 1-min en Eastern Time, série continue front-month
    """
    df = load_raw_dbn(path)
    df = filter_individual_contracts(df)
    df = build_continuous_series(df, instrument)
    df = convert_to_eastern(df)
    return df
