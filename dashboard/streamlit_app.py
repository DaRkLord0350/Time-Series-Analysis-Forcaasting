# app/streamlit_app.py
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="📈 Forecasting System Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 Time Series Forecasting System (Prophet vs SARIMA)")


# --------------------------------------------------------
# Load all available metrics
# --------------------------------------------------------
def load_json(path):
    path = Path(path)
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return None


prophet_data = load_json("reports/prophet_metrics.json")
sarima_data = load_json("reports/sarima_metrics.json")
baseline_data = load_json("reports/baselines.json")

# --------------------------------------------------------
# Sidebar
# --------------------------------------------------------
st.sidebar.header("🔍 Select View")
view = st.sidebar.radio(
    "Choose what to explore:",
    ["Prophet Metrics", "SARIMA Metrics", "Baseline Models", "Model Comparison"],
)

# --------------------------------------------------------
# Prophet Metrics
# --------------------------------------------------------
if view == "Prophet Metrics" and prophet_data:
    st.subheader("📈 Prophet Model Metrics")
    df = pd.DataFrame(prophet_data)
    st.dataframe(df)

    # Visualize
    fig, ax = plt.subplots()
    ax.bar(df["series_id"], df["sMAPE"], label="sMAPE (%)")
    ax.set_ylabel("sMAPE (%)")
    ax.set_title("Prophet Error by Series")
    st.pyplot(fig)

# --------------------------------------------------------
# SARIMA Metrics
# --------------------------------------------------------
elif view == "SARIMA Metrics" and sarima_data:
    # Handle nested dict format
    if isinstance(sarima_data, dict) and "results" in sarima_data:
        records = []
        for sid, content in sarima_data["results"].items():
            m = content["metrics"]
            records.append(
                {
                    "series_id": sid,
                    "mean_sMAPE": m["mean_smape"],
                    "median_sMAPE": m["median_smape"],
                    "n_folds": m["n_folds"],
                }
            )
        df = pd.DataFrame(records)
    else:
        df = pd.DataFrame(sarima_data)

    st.subheader("🌀 SARIMA / AutoARIMA Metrics")
    st.dataframe(df)

    fig, ax = plt.subplots()
    ax.bar(df["series_id"], df["mean_sMAPE"], color="orange", label="Mean sMAPE")
    ax.set_ylabel("sMAPE (%)")
    ax.set_title("SARIMA Model Performance")
    st.pyplot(fig)

# --------------------------------------------------------
# Baseline Metrics
# --------------------------------------------------------
elif view == "Baseline Models" and baseline_data:
    st.subheader("⚙️ Baseline Models (Naïve, Seasonal, Moving Avg)")
    df = pd.DataFrame(baseline_data).T.reset_index()
    df.columns = ["Model", "MAE", "MAPE", "sMAPE"]
    st.dataframe(df)

    fig, ax = plt.subplots()
    df.plot(x="Model", y=["MAE", "MAPE", "sMAPE"], kind="bar", ax=ax)
    plt.title("Baseline Error Comparison")
    plt.ylabel("Error Value")
    st.pyplot(fig)

# --------------------------------------------------------
# Model Comparison
# --------------------------------------------------------
elif view == "Model Comparison":
    st.subheader("🏁 Prophet vs SARIMA Comparison")

    compare_path = Path("reports/model_compare_v1.md")
    if compare_path.exists():
        st.markdown(compare_path.read_text(encoding="utf-8"))
    else:
        st.warning("No comparison report found. Run compare_models.py first.")

    chart_path = Path("reports/model_compare_chart.png")
    if chart_path.exists():
        st.image(str(chart_path), caption="Prophet vs SARIMA sMAPE Comparison")

else:
    st.info("No data available for this section yet. Please run pipelines first.")
