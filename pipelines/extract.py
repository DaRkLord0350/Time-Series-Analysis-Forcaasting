import pandas as pd 
def extract(path: str): 
    return pd.read_csv(path, parse_dates=["ds"]).set_index("ds") 
