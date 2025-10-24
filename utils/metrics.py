# utils/metrics.py
import numpy as np


def mae(y, yhat):
    y, yhat = np.array(y), np.array(yhat)
    return float(np.mean(np.abs(y - yhat)))


def mape(y, yhat, eps=1e-8):
    y, yhat = np.array(y), np.array(yhat)
    return float(np.mean(np.abs((y - yhat) / np.clip(np.abs(y), eps, None))) * 100)


def smape(y, yhat, eps=1e-8):
    y, yhat = np.array(y), np.array(yhat)
    denom = np.abs(y) + np.abs(yhat) + eps
    return float(np.mean(2.0 * np.abs(y - yhat) / denom) * 100)
