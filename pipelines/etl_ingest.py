import os

import duckdb
import pandas as pd
import pandera.pandas as pa
import yaml
from pandera.pandas import Check, Column, DataFrameSchema

# ------------------ LOAD CONFIG ------------------ #
CONFIG_PATH = "configs/datasource.yaml"

with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

dataset_info = cfg["dataset"]
paths = cfg["paths"]
schema_cfg = cfg["schema"]
metadata = cfg["metadata"]

RAW_PATH = paths["raw_data"]
PROCESSED_PATH = paths["processed_data"]

ID_COL = schema_cfg["id_col"]
DATE_COL = schema_cfg["date_col"]
TARGET_COL = schema_cfg["target_col"]
FREQ = schema_cfg["freq"]

# ------------------ READ RAW DATA ------------------ #
print(f"📥 Reading dataset: {dataset_info['name']}")

if not os.path.exists(RAW_PATH):
    raise FileNotFoundError(f"❌ Raw data not found at {RAW_PATH}")

df = pd.read_csv(RAW_PATH)

# ------------------ BASIC CLEANING ------------------ #
print("🧹 Cleaning and type-casting...")

# Standardize column names
df.columns = df.columns.str.strip()

# Parse date column
df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

# Type cast numeric target column
df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").astype(float)

# Create series_id if not present
if ID_COL not in df.columns:
    df[ID_COL] = dataset_info["name"]

# Drop rows with invalid data
df = df.dropna(subset=[DATE_COL, TARGET_COL])

# Enforce frequency sorting
df = df.sort_values(DATE_COL).reset_index(drop=True)

# ------------------ VALIDATION ------------------ #
print("✅ Running data validation checks...")

schema = DataFrameSchema(
    {
        ID_COL: Column(str, nullable=False),
        DATE_COL: Column(pa.DateTime, nullable=False),
        TARGET_COL: Column(float, Check.greater_than_or_equal_to(0)),
    }
)

validated_df = schema.validate(df)

# ------------------ SAVE CLEAN DATA ------------------ #
print("💾 Saving cleaned data...")

os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
validated_df.to_csv(PROCESSED_PATH, index=False)
print(f"✅ Cleaned file saved → {PROCESSED_PATH}")

# ------------------ OPTIONAL: WRITE TO DUCKDB ------------------ #
DUCKDB_PATH = "data/forecasting.duckdb"
TABLE_NAME = dataset_info["name"]

con = duckdb.connect(DUCKDB_PATH)
con.register("validated_df", validated_df)
con.execute(
    f"CREATE TABLE IF NOT EXISTS {TABLE_NAME} AS SELECT * FROM validated_df LIMIT 0;"
)
con.execute(f"DELETE FROM {TABLE_NAME};")  # Avoid duplicates
con.execute(f"INSERT INTO {TABLE_NAME} SELECT * FROM validated_df;")
con.close()

print("🎉 ETL pipeline completed successfully!")
print(f"🪶 Total rows ingested: {len(validated_df)}")
