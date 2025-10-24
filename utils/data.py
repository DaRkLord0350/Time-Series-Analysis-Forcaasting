# utils/data.py
from pathlib import Path

import pandas as pd
import yaml


def load_config(config_path="configs/datasource.yaml"):
    """Load YAML configuration file or dict."""
    if isinstance(config_path, dict):
        return config_path
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"❌ Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_data(config_path="configs/datasource.yaml"):
    """
    Load dataset based on YAML config or dict.
    Returns:
        df: Raw DataFrame
        id_col, date_col, target_col
    """
    cfg = load_config(config_path)
    source = cfg.get("source", {})
    src_type = source.get("type", "csv")

    # --- Read CSV ---
    if src_type == "csv":
        path = Path(source.get("path"))
        if not path.exists():
            raise FileNotFoundError(f"❌ CSV not found: {path}")
        df = pd.read_csv(path)
    else:
        raise ValueError(f"❌ Unsupported source type: {src_type}")

    # --- Columns from config ---
    date_col = source.get("date_column", "date")
    target_col = source.get("target_column", "target")
    id_col = source.get("id_column")

    # --- Validate required columns ---
    missing_cols = [c for c in [date_col, target_col] if c not in df.columns]
    if missing_cols:
        raise KeyError(f"❌ Missing required columns: {missing_cols} in dataset {path}")

    # --- Auto-add series_id if missing ---
    if id_col is None or id_col not in df.columns:
        print("ℹ️ No 'series_id' column found — assigning default ID: 'airline_total'")
        df["series_id"] = "airline_total"
        id_col = "series_id"

    # --- Parse date ---
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    return df, id_col, date_col, target_col


def load_timeseries(config_path="configs/datasource.yaml"):
    """
    Load standardized dataframe → ['series_id', 'date', 'target']
    """
    df, id_col, date_col, target_col = load_data(config_path)
    df = df.rename(
        columns={id_col: "series_id", date_col: "date", target_col: "target"}
    )
    df = df.sort_values(["series_id", "date"]).dropna(
        subset=["series_id", "date", "target"]
    )
    return df


def preview_data(df, n=5):
    print("\n📊 Preview:")
    print(df.head(n))
    print(f"\nShape: {df.shape}")
    print(f"Date range: {df['date'].min()} → {df['date'].max()}")


if __name__ == "__main__":
    df = load_timeseries("configs/datasource.yaml")
    preview_data(df)
