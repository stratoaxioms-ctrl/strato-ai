import streamlit as st
import pandas as pd
from atmo_screener import run_strato_ai_screener
import plotly.express as px

st.set_page_config(
    page_title="STRATO-AI",
    page_icon="☁️",
    layout="wide"
)

st.title("☁️ STRATO-AI — Atmospheric Investment Intelligence")
st.write("AI-powered, mission-aligned atmospheric screener for portfolio decisions.")

st.sidebar.header("Client Preferences — Connor Barwin / MTWB")
risk = st.sidebar.slider("Risk Tolerance", 0, 100, 40)
impact_weight = st.sidebar.slider("MTWB Impact Weight", 0, 100, 70)
weather = st.sidebar.selectbox(
    "ATMO-Weather Mode",
    ["Clear Skies (Bullish)", "Jetstream Momentum", "Heatwave Innovation", "Turbulence (Risk-Off)", "Solar Flare (Defensive)"]
)

uploaded = st.file_uploader("Upload atmo_input.csv", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    with st.spinner("Analyzing atmosphere..."):
        results = run_strato_ai_screener(df, risk, impact_weight, weather)

    st.subheader("Atmospheric Layer Classification")
    st.dataframe(results)

    layer_plot = px.histogram(results, x="Layer", title="STRATO-AI Layer Distribution")
    st.plotly_chart(layer_plot, use_container_width=True)

    st.download_button(
        "Download STRATO-AI Output CSV",
        results.to_csv(index=False),
        "strato_ai_output.csv",
        "text/csv"
    )

else:
    st.info("Upload your atmo_input.csv to begin.")
