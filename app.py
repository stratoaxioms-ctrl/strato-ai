import streamlit as st
import pandas as pd
import plotly.express as px
from atmo_screener import run_strato_ai_screener

st.set_page_config(page_title="STRATO-AI", layout="wide")

st.title("🌤️ STRATO-AI — Atmospheric Investment Screener")
st.write("Upload your dataset and generate atmospheric cluster signals.")

uploaded = st.file_uploader("Upload your atmo_input.csv", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.subheader("📄 Input Preview")
    st.dataframe(df)

    st.subheader("⚙️ MTWB Weighting")
    mtwb_weight = st.slider("Mission Score Weight", 0.0, 1.0, 0.3)

    st.subheader("🌦️ Market Weather")
    weather = st.selectbox("Select Market Weather Condition", [
        "Clear Skies", "Jetstreams", "Heatwave", "Turbulence", "Solar Flare"
    ])

    if st.button("Run STRATO-AI"):
        result = run_strato_ai_screener(df, mtwb_weight, weather)
        st.success("STRATO-AI completed!")
        st.dataframe(result)

        csv = result.to_csv(index=False).encode("utf-8")
        st.download_button("Download Results CSV", csv, "strato_output.csv", "text/csv")
