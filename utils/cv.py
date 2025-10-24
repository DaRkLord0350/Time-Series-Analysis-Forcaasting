# utils/cv.py
import pandas as pd


def sliding_windows(df, date_col, min_train_days, horizon_days, step_days):
    df = df.sort_values(date_col).reset_index(drop=True)
    dates = df[date_col]
    start_idx = 0
    while True:
        train_end = dates.iloc[start_idx] + pd.Timedelta(days=min_train_days - 1)
        # validation immediately after train
        val_start = train_end + pd.Timedelta(days=1)
        val_end = val_start + pd.Timedelta(days=horizon_days - 1)

        tr = df[(df[date_col] >= dates.iloc[start_idx]) & (df[date_col] <= train_end)]
        val = df[(df[date_col] >= val_start) & (df[date_col] <= val_end)]
        if tr.empty or val.empty:
            break
        yield tr, val, {"train_end": train_end, "val_end": val_end}

        # slide forward
        start_idx = df.index[
            df[date_col] >= (dates.iloc[start_idx] + pd.Timedelta(days=step_days))
        ][0]
        if (
            dates.iloc[start_idx] + pd.Timedelta(days=min_train_days + horizon_days)
            > dates.max()
        ):
            break
