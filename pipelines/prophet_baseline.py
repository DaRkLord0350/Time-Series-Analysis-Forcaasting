# pipelines/prophet_baseline.py
import json
from pathlib import Path

import pandas as pd
import yaml
from prophet import Prophet
from prophet.serialize import model_to_json

from utils.cv import sliding_windows
from utils.data import load_timeseries
from utils.metrics import mae, mape, smape


def fit_prophet(df, ds_col, y_col, cfg):
    m = Prophet(
        yearly_seasonality=cfg["model"]["yearly_seasonality"],
        weekly_seasonality=cfg["model"]["weekly_seasonality"],
        daily_seasonality=cfg["model"]["daily_seasonality"],
        changepoint_prior_scale=cfg["model"]["changepoint_prior_scale"],
        seasonality_mode=cfg["model"]["seasonality_mode"],
    )
    country = cfg["model"].get("holidays_country")
    if country:
        m.add_country_holidays(country_name=country)
    train_df = df.rename(columns={ds_col: "ds", y_col: "y"})
    m.fit(train_df)
    return m


def main():
    cfg = yaml.safe_load(open("configs/prophet.yaml"))
    data_cfg = yaml.safe_load(open(cfg["io"]["datasource_config"]))

    # id_col = data_cfg["source"]["id_column"]
    # ds_col = data_cfg["source"]["date_column"]
    # y_col = data_cfg["source"]["target_column"]

    df = load_timeseries(data_cfg)
    models_dir = Path(cfg["io"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    horizon = pd.Timedelta(cfg["train"]["horizon"]).days
    step = pd.Timedelta(cfg["train"]["step"]).days
    min_train_days = cfg["train"]["min_train_periods"]

    all_metrics = []

    for series_id, g in df.groupby("series_id"):
        g = g.sort_values("date").reset_index(drop=True)
        series_scores = []

        for tr, val, meta in sliding_windows(
            g,
            date_col="date",
            min_train_days=min_train_days,
            horizon_days=horizon,
            step_days=step,
        ):
            m = fit_prophet(tr, "date", "target", cfg)
            val_dates = pd.date_range(val["date"].min(), val["date"].max(), freq="D")
            fcst = m.predict(pd.DataFrame({"ds": val_dates}))[["ds", "yhat"]]
            merged = val.rename(columns={"date": "ds", "target": "y"}).merge(
                fcst, on="ds", how="left"
            )

            row = {
                "series_id": series_id,
                "split_start": val["date"].min().strftime("%Y-%m-%d"),
                "split_end": val["date"].max().strftime("%Y-%m-%d"),
                "MAE": mae(merged["y"], merged["yhat"]),
                "MAPE": mape(merged["y"], merged["yhat"]),
                "sMAPE": smape(merged["y"], merged["yhat"]),
            }
            series_scores.append(row)

        avg = pd.DataFrame(series_scores).mean(numeric_only=True).to_dict()
        avg["series_id"] = series_id
        all_metrics.append(avg)

        final_model = fit_prophet(g, "date", "target", cfg)
        with open(models_dir / f"{series_id}.json", "w") as f:
            f.write(model_to_json(final_model))

    out_path = Path(cfg["io"]["metrics_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2)


if __name__ == "__main__":
    main()
