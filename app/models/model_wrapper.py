import pickle 
from pathlib import Path 
import pandas as pd 
import yaml 
import mlflow 
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
        model = auto_arima(series, seasonal=True, m=cfg["model"]["seasonal_period"], suppress_warnings=True) 
        self.model_path.parent.mkdir(parents=True, exist_ok=True) 
        with open(self.model_path, "wb") as f: 
            pickle.dump(model, f) 
        mlflow.set_experiment(cfg["model"]["mlflow_experiment"]) 
        with mlflow.start_run(): 
            mlflow.log_param("model", "auto_arima") 
            mlflow.log_metric("n_samples", len(series)) 
            mlflow.sklearn.log_model(model, "model") 
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
