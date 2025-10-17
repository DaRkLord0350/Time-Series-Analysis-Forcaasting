import pytest

from app.models.model_wrapper import ModelWrapper


def test_predict_raises_when_no_model(tmp_path, monkeypatch):
    mw = ModelWrapper(model_path=str(tmp_path / "no_model.pkl"))
    with pytest.raises(RuntimeError):
        mw.predict(steps=3)
