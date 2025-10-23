import numpy as np

from utils.metrics import mae, mape, smape


def test_mae():
    y_true = np.array([100, 200, 300])
    y_pred = np.array([110, 190, 310])
    assert np.isclose(mae(y_true, y_pred), 10)


def test_mape():
    y_true = np.array([100, 200, 300])
    y_pred = np.array([110, 190, 310])
    expected = (10 / 100 + 10 / 200 + 10 / 300) / 3 * 100
    assert np.isclose(mape(y_true, y_pred), expected)


def test_smape():
    y_true = np.array([100, 200, 300])
    y_pred = np.array([110, 190, 310])
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    expected = np.mean(numerator / denominator) * 100
    assert np.isclose(smape(y_true, y_pred), expected)
