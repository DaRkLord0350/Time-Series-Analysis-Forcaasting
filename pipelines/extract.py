import pandas as pd


def extract(path: str):
    df = pd.read_csv(path, parse_dates=["ds"]).set_index("ds")
    if df.empty:
        raise ValueError(f"❌ No data found in {path}")
    if "y" not in df.columns:
        raise ValueError(f"❌ Missing required column 'y' in {path}")
    return df
