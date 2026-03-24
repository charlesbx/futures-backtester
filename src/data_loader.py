"""Databento data loading and processing for the backtester engine.

Handles contract rollover by daily volume and UTC → Eastern Time conversion.
"""

import re
from pathlib import Path

import databento as db
import pandas as pd

# Tick specifications per instrument
INSTRUMENTS = {
    "MES": {"tick_size": 0.25, "tick_value": 1.25},
    "MNQ": {"tick_size": 0.25, "tick_value": 0.50},
}

# Pattern for individual contracts (excludes spreads like MESM9-MESU9)
_CONTRACT_RE = re.compile(r"^(MES|MNQ)[HMUZ]\d$")


def load_raw_dbn(path: str | Path) -> pd.DataFrame:
    """Load a Databento DBN file and return an OHLCV 1-min DataFrame."""
    store = db.DBNStore.from_file(str(path))
    return store.to_df()


def filter_individual_contracts(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only individual MES/MNQ contracts (excludes spreads)."""
    mask = df["symbol"].str.match(r"^(MES|MNQ)[HMUZ]\d$")
    return df[mask].copy()


def build_continuous_series(df: pd.DataFrame, instrument: str) -> pd.DataFrame:
    """Build a continuous series for an instrument via volume-based rollover.

    For each trading day (UTC date), selects the contract with the highest
    daily volume. Handles rolls automatically.
    """
    # Filter on instrument (individual contracts only)
    mask = df["symbol"].apply(lambda s: bool(_CONTRACT_RE.match(s)) and s.startswith(instrument))
    inst_df = df[mask].copy()

    # Trading date (UTC date — sufficient for daily rollover)
    inst_df["trade_date"] = inst_df.index.normalize()

    # Daily volume per contract
    daily_vol = (
        inst_df.groupby(["trade_date", "symbol"])["volume"]
        .sum()
        .reset_index()
    )

    # Front-month = contract with max volume each day
    idx_max = daily_vol.groupby("trade_date")["volume"].idxmax()
    front_map = daily_vol.loc[idx_max].set_index("trade_date")["symbol"]

    # Keep only bars from the front-month contract
    inst_df["front"] = inst_df["trade_date"].map(front_map)
    result = inst_df[inst_df["symbol"] == inst_df["front"]].copy()
    result.drop(columns=["trade_date", "front"], inplace=True)
    result.sort_index(inplace=True)

    return result


def convert_to_eastern(df: pd.DataFrame) -> pd.DataFrame:
    """Convert UTC index to Eastern Time (handles EST/EDT automatically)."""
    df = df.copy()
    df.index = df.index.tz_convert("US/Eastern")
    return df


def save_processed(df: pd.DataFrame, output_dir: str | Path, instrument: str) -> Path:
    """Save processed data to Parquet."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{instrument}_1m.parquet"
    df.to_parquet(path)
    return path


def load_processed(path: str | Path) -> pd.DataFrame:
    """Load processed data from a Parquet file."""
    return pd.read_parquet(str(path))


def load_and_prepare(path: str | Path, instrument: str) -> pd.DataFrame:
    """Full pipeline: load DBN → filter → rollover → Eastern Time.

    Args:
        path: Path to DBN file (.dbn.zst)
        instrument: 'MES' or 'MNQ'

    Returns:
        OHLCV 1-min DataFrame in Eastern Time, continuous front-month series
    """
    df = load_raw_dbn(path)
    df = filter_individual_contracts(df)
    df = build_continuous_series(df, instrument)
    df = convert_to_eastern(df)
    return df
