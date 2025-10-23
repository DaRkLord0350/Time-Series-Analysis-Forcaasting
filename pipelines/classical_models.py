"""
pipelines/classical_models.py

Usage (example):
    python -m pipelines.classical_models \
        --data-path data/processed/airline_passenger_processed.csv \
        --id-col series_id --date-col Month --target-col Passengers \
        --freq M --horizon 12 --initial-train 60 --version v1.0.0

What it does:
- For each series (grouped by id_col), runs sliding-window CV using pmdarima.
- Keeps best params (lowest mean sMAPE across folds).
- Retrains on full series with best params and saves model (joblib)
  under models/<version>/series_<id>.pkl
- Writes per-series metrics & params to reports/classical_baselines.json
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from pmdarima import auto_arima
from tqdm import tqdm

from pipelines.metrics import smape  # local helper


# ---------------------------------------------------------------------
# SLIDING WINDOW CROSS-VALIDATION
# ---------------------------------------------------------------------
def sliding_window_cv(
    y: np.ndarray,
    exog: Optional[np.ndarray],
    initial_train_size: int,
    horizon: int,
    step: int = 1,
    seasonal: bool = True,
    m: int = 12,
    max_p: int = 5,
    max_q: int = 5,
):
    """
    Rolling-origin (sliding window) CV generator.

    Yields dicts with:
        - train_y, test_y, train_exog, test_exog, train_idx, test_idx
    """
    n = len(y)
    i = initial_train_size
    while i + horizon <= n:
        train_idx = list(range(0, i))
        test_idx = list(range(i, i + horizon))
        train_y = y[train_idx]
        test_y = y[test_idx]

        if exog is not None:
            train_exog = exog[train_idx]
            test_exog = exog[test_idx]
        else:
            train_exog = test_exog = None

        yield {
            "train_y": train_y,
            "test_y": test_y,
            "train_exog": train_exog,
            "test_exog": test_exog,
            "train_idx": train_idx,
            "test_idx": test_idx,
        }

        i += step


# ---------------------------------------------------------------------
# AUTO-ARIMA EVALUATION ON CV FOLDS
# ---------------------------------------------------------------------
def evaluate_auto_arima_on_cv(
    y: np.ndarray,
    exog: Optional[np.ndarray],
    initial_train_size: int,
    horizon: int,
    step: int = 1,
    seasonal: bool = True,
    m: int = 12,
    **auto_arima_kwargs,
):
    """
    Runs auto_arima for each fold, returns mean sMAPE and keeps the best auto
    _arima model.
    """
    fold_scores = []
    fold_params = []

    for fold in sliding_window_cv(
        y, exog, initial_train_size, horizon, step, seasonal, m
    ):
        train_y = fold["train_y"]
        test_y = fold["test_y"]
        train_exog = fold["train_exog"]
        test_exog = fold["test_exog"]

        try:
            model = auto_arima(
                train_y,
                exogenous=train_exog,
                seasonal=seasonal,
                m=m,
                error_action="ignore",
                suppress_warnings=True,
                **auto_arima_kwargs,
            )
        except Exception as exc:
            print("auto_arima failed on a fold:", exc)
            continue

        # Forecast
        try:
            if test_exog is not None:
                preds = model.predict(n_periods=len(test_y), exogenous=test_exog)
            else:
                preds = model.predict(n_periods=len(test_y))
        except Exception as exc:
            print("Prediction failed on a fold:", exc)
            continue

        score = smape(test_y, preds)
        fold_scores.append(score)
        fold_params.append(model.get_params())

    if not fold_scores:
        raise RuntimeError("No successful CV folds for this series.")

    mean_score = float(np.mean(fold_scores))
    median_score = float(np.median(fold_scores))

    final_model = auto_arima(
        y,
        exogenous=exog,
        seasonal=seasonal,
        m=m,
        error_action="ignore",
        suppress_warnings=True,
        **auto_arima_kwargs,
    )

    return {
        "mean_smape": mean_score,
        "median_smape": median_score,
        "n_folds": len(fold_scores),
        "final_model": final_model,
        "final_params": final_model.get_params(),
    }


# ---------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------
def run_pipeline(
    data_path: str,
    id_col: str,
    date_col: str,
    target_col: str,
    freq: str,
    horizon: int,
    initial_train_size: int,
    version: Optional[str] = None,
    output_dir: str = ".",
    m: Optional[int] = None,
    step: int = 1,
):
    df = pd.read_csv(data_path, parse_dates=[date_col])
    reports = {}
    version = version or f"v{int(time.time())}"

    models_dir = Path(output_dir) / "models" / version
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path(output_dir) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    for series_id, group in tqdm(df.groupby(id_col), desc="series"):
        group = group.sort_values(date_col)
        y = group[target_col].astype(float).values

        if m is None:
            if freq.lower().startswith("m"):
                m_local = 12
            elif freq.lower().startswith("w"):
                m_local = 52
            elif freq.lower().startswith("d"):
                m_local = 7
            else:
                m_local = 1
        else:
            m_local = m

        if len(y) < initial_train_size + horizon:
            print(
                "Series %s too short (%d). Needed %d. Skipping."
                % (series_id, len(y), initial_train_size + horizon)
            )
            continue

        try:
            res = evaluate_auto_arima_on_cv(
                y,
                exog=None,
                initial_train_size=initial_train_size,
                horizon=horizon,
                step=step,
                seasonal=(m_local > 1),
                m=m_local,
                max_p=3,
                max_q=3,
                max_P=1,
                max_Q=1,
                information_criterion="aicc",
                n_jobs=1,
                start_p=0,
                start_q=0,
                seasonal_test="ocsb",
                trace=False,
            )
        except Exception as exc:
            print(f"Auto-ARIMA failed for series {series_id}: {exc}")
            continue

        final_model = res["final_model"]
        metrics = {
            "mean_smape": res["mean_smape"],
            "median_smape": res["median_smape"],
            "n_folds": res["n_folds"],
        }

        model_path = models_dir / f"series_{series_id}.pkl"
        joblib.dump(final_model, model_path)

        params_path = models_dir / f"series_{series_id}_params.json"
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(res["final_params"], f, default=str, indent=2)

        reports[str(series_id)] = {
            "model_path": str(model_path),
            "params_path": str(params_path),
            "metrics": metrics,
        }

    report_path = reports_dir / "classical_baselines.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"version": version, "results": reports}, f, indent=2)

    print("✅ Done. Models saved to:", models_dir)
    print("✅ Report saved to:", report_path)


# ---------------------------------------------------------------------
# CLI ENTRY POINT
# ---------------------------------------------------------------------
def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--id-col", default="series_id")
    parser.add_argument("--date-col", default="date")
    parser.add_argument("--target-col", default="target")
    parser.add_argument("--freq", default="M", help="Data frequency: M, D, W ...")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument(
        "--initial-train", dest="initial_train_size", type=int, default=60
    )
    parser.add_argument("--version", default=None)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument(
        "--m",
        type=int,
        default=None,
        help="Seasonal period; if omitted will be guessed",
    )
    args = parser.parse_args()

    run_pipeline(
        data_path=args.data_path,
        id_col=args.id_col,
        date_col=args.date_col,
        target_col=args.target_col,
        freq=args.freq,
        horizon=args.horizon,
        initial_train_size=args.initial_train_size,
        version=args.version,
        output_dir=args.output_dir,
        m=args.m,
    )


if __name__ == "__main__":
    cli()
