def transform(df):
    df = df.asfreq("D")
    return df.fillna(method="ffill").fillna(0)
