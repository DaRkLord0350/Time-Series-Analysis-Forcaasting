import pickle
from pathlib import Path

import mlflow
import pandas as pd
import yaml
from pmdarima import auto_arima

CONFIG_PATH = Path("configs/config.yaml")


class ModelWrapper:
    def __init__(self, model_path="models/latest_model.pkl"):
        self.model_path = Path(model_path)
        self.model = None
        if self.model_path.exists():
            self.load()

    def load(self):
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)

    def train_and_log(self):
        cfg = yaml.safe_load(open(CONFIG_PATH))
        df = pd.read_csv(cfg["data"]["source"], parse_dates=["ds"])
        df = df.set_index("ds")
        series = df["y"]

        # ✅ Handle short dataset
        m = cfg["model"]["seasonal_period"]
        if len(series) < m * 2:
            print(
                f"⚠️ Not enough data ({len(series)} rows) for seasonality (m={m})."
                f"Disabling seasonal mode."
            )
            seasonal = False
            m = 1
        else:
            seasonal = True

        # ✅ Train model safely
        model = auto_arima(
            series,
            seasonal=seasonal,
            m=m,
            error_action="ignore",
            suppress_warnings=True,
            trace=True,
        )

        # ✅ Save model
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump(model, f)

        # ✅ Log to MLflow
        mlflow.set_experiment(cfg["model"]["mlflow_experiment"])
        with mlflow.start_run():
            mlflow.log_param("model", "auto_arima")
            mlflow.log_param("seasonal", seasonal)
            mlflow.log_metric("n_samples", len(series))
            mlflow.sklearn.log_model(model, "model")

        print(f"✅ Model trained and saved to {self.model_path}")
        self.model = model

    def predict(self, steps=14):
        if not self.model:
            if self.model_path.exists():
                self.load()
        if not self.model:
            raise RuntimeError("Model not trained yet")
        fc, conf_int = self.model.predict(n_periods=steps, return_conf_int=True)
        return [
            {"yhat": float(y), "lower": float(l), "upper": float(u)}
            for y, (l, u) in zip(fc, conf_int)
        ]
