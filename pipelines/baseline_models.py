import json
from pathlib import Path

import pandas as pd
import yaml

from utils.metrics import mae, mape, smape

CONFIG_PATH = "configs/datasource.yaml"
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

target_col = cfg["schema"]["target_col"]


def naive_forecast(series):
    """Forecast = last observed value"""
    return series.shift(1)


def seasonal_naive_forecast(series, season_length=12):
    """Forecast = value from same period last season"""
    return series.shift(season_length)


def moving_average_forecast(series, window=3):
    """Forecast = average of previous window"""
    return series.rolling(window=window).mean().shift(1)


def evaluate_baselines(df, target_col=target_col):
    y_true = df[target_col]

    results = {}
    forecasts = {
        "naive": naive_forecast(y_true),
        "seasonal_naive": seasonal_naive_forecast(y_true),
        "moving_average": moving_average_forecast(y_true),
    }

    for name, y_pred in forecasts.items():
        mask = ~y_pred.isna()
        y_true_valid, y_pred_valid = y_true[mask], y_pred[mask]

        results[name] = {
            "MAE": mae(y_true_valid, y_pred_valid),
            "MAPE": mape(y_true_valid, y_pred_valid),
            "sMAPE": smape(y_true_valid, y_pred_valid),
        }

    Path("reports").mkdir(exist_ok=True)
    with open("reports/baselines.json", "w") as f:
        json.dump(results, f, indent=4)

    print("✅ Baseline metrics saved to reports/baselines.json")
    return results


if __name__ == "__main__":
    df = pd.read_csv(
        "data/processed/airline_passenger_processed.csv", parse_dates=["Month"]
    )
    df.set_index("Month", inplace=True)
    evaluate_baselines(df, target_col=target_col)
