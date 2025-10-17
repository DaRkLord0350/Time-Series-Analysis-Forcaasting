import streamlit as st 
import requests 
import pandas as pd 
 
API_URL = st.secrets.get("API_URL", "http://localhost:8000/predict/") 
 
st.title("Forecasting Dashboard") 
 
steps = st.number_input("Forecast horizon (steps)", value=14, min_value=1) 
 
if st.button("Get forecast"): 
    resp = requests.post(API_URL, json={"steps": int(steps)}) 
    if resp.ok: 
        fc = resp.json()["forecast"] 
        df = pd.DataFrame(fc) 
        st.line_chart(df["yhat"]) 
        st.dataframe(df) 
    else: 
        st.error(f"API error: {resp.text}") 
 
st.subheader("KPIs") 
k1, k2, k3 = st.columns(3) 
k1.metric("MAPE", "-") 
k2.metric("sMAPE", "-") 
k3.metric("Bias", "-") 
