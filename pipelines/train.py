import sys
from pathlib import Path

import yaml

from app.models.model_wrapper import ModelWrapper
from pipelines.extract import extract
from pipelines.transform import transform

sys.path.append(str(Path(__file__).resolve().parents[1]))


def run_train(cfg_path="configs/config.yaml"):
    cfg = yaml.safe_load(open(cfg_path))
    df = extract(cfg["data"]["source"])
    df = transform(df)
    mw = ModelWrapper(model_path=cfg["model"]["save_path"])
    mw.train_and_log()


if __name__ == "__main__":
    run_train()
