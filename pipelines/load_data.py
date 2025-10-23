import pandas as pd
import yaml


def load_data(config_path="configs/datasource.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_path = config["paths"]["raw_data"]
    df = pd.read_csv(data_path)
    df["series_id"] = "airline_total"
    df[config["schema"]["date_col"]] = pd.to_datetime(df[config["schema"]["date_col"]])
    df = df.rename(columns={config["schema"]["target_col"]: "target"})

    return df


def validate_data(df):
    print("\n🔍 Data Validation Report")
    print("-" * 30)
    print(f"Rows: {len(df)}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"Duplicates: {df.duplicated().sum()}")
    print(f"Date range: {df['Month'].min()} → {df['Month'].max()}")
    print(f"Frequency check (unique months): {df['Month'].dt.to_period('M').nunique()}")


if __name__ == "__main__":
    df = load_data()
    print(df.head())
    validate_data(df)
