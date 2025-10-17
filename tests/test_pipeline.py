import pandas as pd

from pipelines.transform import transform


def test_transform_fills():
    df = pd.DataFrame({"y": [1, None, 3]}, index=pd.date_range("2025-01-01", periods=3))
    out = transform(df)
    assert out.isnull().sum().sum() == 0
