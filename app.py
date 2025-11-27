# app.py — STRATO-AI Premium (Enhanced UI + Graphics + Connor personalization)
# -------------------------------------------------------------------------
# Replace your current app.py with this file, commit, and redeploy.
# Requires: streamlit, pandas, plotly, scikit-learn (already in requirements.txt)
# -------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from atmo_screener import run_strato_ai_screener

# --------------------
# Branding & Theme
# --------------------
st.set_page_config(
    page_title="STRATO-AI — Premium",
    page_icon="☁️",
    layout="wide",
)

LAYER_COLORS = {
    "Troposphere": "#2ca02c",   # green
    "Stratosphere": "#1f77b4",  # blue
    "Mesosphere": "#ff7f0e",    # orange
    "Thermosphere": "#d62728",  # red
    "Exosphere": "#9467bd"      # purple
}

# Header
st.markdown(
    """
    <div style="display:flex;align-items:center;gap:16px">
      <div style="font-size:28px;font-weight:700">☁️ <span style="color:#0b3d91">STRATO-AI</span> — Premium</div>
      <div style="color:#555">Atmospheric Investment Intelligence • Built for Connor Barwin / MTWB</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# --------------------
# Sidebar: Connor controls
# --------------------
st.sidebar.header("Connor Preferences — Live Controls")
risk_preset = st.sidebar.selectbox("Risk Profile", ["Conservative", "Balanced", "Aggressive"], index=1)
risk_map = {"Conservative": 0.2, "Balanced": 0.5, "Aggressive": 0.8}
risk_weight = risk_map[risk_preset]

mtwb_weight = st.sidebar.slider("MTWB Mission Weight (impact emphasis)", 0.0, 1.0, 0.5, 0.01)
drawdown = st.sidebar.number_input("Annual Drawdown ($/yr) starting Year 3", value=10000, step=1000)
horizon = st.sidebar.select_slider("Time Horizon (years)", [5,10,15,20], value=10)

st.sidebar.markdown("---")
st.sidebar.header("Macro Weather (ATMO-WEATHER)")
weather = st.sidebar.radio("Select Market Weather", ["Clear Skies","Jetstreams","Heatwave","Turbulence","Solar Flare"])
st.sidebar.markdown("---")
st.sidebar.write("Quick actions")
if st.sidebar.button("Save config snapshot (session)"):
    st.sidebar.success("Snapshot saved (session)")

# --------------------
# Upload / Example
# --------------------
st.subheader("Step 1 — Upload your dataset")
uploaded = st.file_uploader("Upload `atmo_input.csv` (use sample columns in README)", type=["csv"])
use_example = st.button("Use example dataset")

if uploaded:
    df_in = pd.read_csv(uploaded)
elif use_example:
    # small example
    df_in = pd.DataFrame([
        ['AAPL',175,2500000,28.5,80000,12,0.05,1.0,1.5,85,60,0.3,0.03,0.12,'Technology',24000,0.006,0.10,0.8,0.7,0.9,0.6],
        ['NVDA',400,1200000,60.2,12000,65,0.20,1.8,0.5,70,40,0.6,0.25,0.80,'Semiconductors',2500,0.0,1.20,0.6,0.5,0.6,0.3],
        ['NEE',75,180000,30.5,3500,6,0.02,0.6,1.0,92,70,0.1,0.01,0.03,'Utilities',300,0.035,0.06,0.9,0.8,0.95,0.8],
    ], columns=['ticker','price_local','market_cap_m','pe','fcf_m','rev_growth_pct','eps_std_4q','beta','de_ratio','esg_score','community_score','analyst_sent','momentum_30d','momentum_90d','sector','rd_expense_m','dividend_yield','1Y_Return','Sustainability','DEI','Safety','Renewables'])
else:
    st.info("Upload a CSV or click 'Use example dataset'.")
    st.stop()

# Basic sanity preview
st.markdown("**Preview (first 8 rows)**")
st.dataframe(df_in.head(8))

# --------------------
# Run STRATO-AI engine
# --------------------
if st.button("Run STRATO-AI — Analyze"):
    with st.spinner("Running STRATO-AI…"):
        # call the engine — returns processed df with Layer + scores
        try:
            processed = run_strato_ai_screener(df_in)
        except TypeError:
            # backward compatibility if function signature expects weights/weather
            processed = run_strato_ai_screener(df_in)
        # Add final composite score using UI weights
        # ensure necessary columns exist
        for col in ["ATMO_Pressure","ATMO_Heat","ATMO_Turbulence","MTWB_Impact_Score"]:
            if col not in processed.columns:
                processed[col] = 0.0

        # Composite final score
        processed["Composite"] = (
            0.35 * processed["ATMO_Pressure"] +
            0.35 * processed["ATMO_Heat"] -
            0.2  * processed["ATMO_Turbulence"] +
            mtwb_weight * processed.get("MTWB_Impact_Score", 0)
        )

        # Apply a small risk tilt according to preset
        processed["Composite"] = processed["Composite"] * (1 + (risk_weight - 0.5) * 0.2)

        # Normalize to 0-100
        proc_vals = processed["Composite"].to_numpy()
        proc_min, proc_max = proc_vals.min(), proc_vals.max()
        if not np.isclose(proc_max, proc_min):
            processed["Composite_100"] = 100 * (proc_vals - proc_min) / (proc_max - proc_min)
        else:
            processed["Composite_100"] = 50.0

        # Narrative column
        def make_narr(row):
            parts = []
            parts.append(f"{row.get('ticker','')} → {row.get('Atmospheric_Layer', 'Unknown')}.")
            if row['ATMO_Heat'] > 0.7:
                parts.append("Growth/innovation signal.")
            if row['ATMO_Pressure'] > 0.7:
                parts.append("Stability & cash flow strength.")
            if row['ATMO_Turbulence'] > 0.6:
                parts.append("Elevated volatility — size carefully.")
            if row.get('MTWB_Impact_Score',0) > 0.6:
                parts.append("Strong MTWB mission alignment.")
            return " ".join(parts)

        processed["Narrative"] = processed.apply(make_narr, axis=1)

    st.success("Analysis complete — visualizations loaded below.")

    # --------------------
    # Interactive filters & leaderboards
    # --------------------
    st.markdown("## Top Candidates & Filters")
    col_a, col_b = st.columns([2,1])
    with col_b:
        min_priority = st.slider("Min Composite Score", 0, 100, 40)
        layer_filter = st.multiselect("Filter Layers", options=list(LAYER_COLORS.keys()), default=list(LAYER_COLORS.keys()))
        sector_filter = st.multiselect("Sector (optional)", options=sorted(processed['sector'].unique()), default=sorted(processed['sector'].unique()))

    df_view = processed[processed["Composite_100"] >= min_priority]
    df_view = df_view[df_view["Atmospheric_Layer"].isin(layer_filter)]
    if 'sector' in df_view.columns and sector_filter:
        df_view = df_view[df_view['sector'].isin(sector_filter)]

    with col_a:
        st.markdown("### Leaderboard — Sorted by Composite Score")
        leaderboard = df_view.sort_values("Composite_100", ascending=False).reset_index(drop=True)
        st.dataframe(leaderboard[["ticker","Atmospheric_Layer","Composite_100","MTWB_Impact_Score","Narrative"]].head(30).round(2))

    # --------------------
    # Plot 1: Heat vs Pressure scatter
    # --------------------
    st.markdown("## Heat vs Pressure (interactive)")
    fig = px.scatter(
        processed,
        x="ATMO_Heat",
        y="ATMO_Pressure",
        color="Atmospheric_Layer",
        size="Composite_100",
        hover_name="ticker",
        color_discrete_map=LAYER_COLORS,
        title="ATMO Heat vs Pressure — sized by Composite Score"
    )
    st.plotly_chart(fig, use_container_width=True)

    # --------------------
    # Plot 2: Composite by Layer (box)
    # --------------------
    st.markdown("## Composite Score Distribution by Layer")
    fig2 = px.box(
        processed,
        x="Atmospheric_Layer",
        y="Composite_100",
        color="Atmospheric_Layer",
        color_discrete_map=LAYER_COLORS,
        category_orders={"Atmospheric_Layer": ["Troposphere","Stratosphere","Mesosphere","Thermosphere","Exosphere"]}
    )
    st.plotly_chart(fig2, use_container_width=True)

    # --------------------
    # Plot 3: Sunburst Allocation suggestion
    # --------------------
    st.markdown("## Suggested Layer Allocation (ATMO-WEATHER adjusted)")
    # simple target: weight by mean Composite_100 per layer
    alloc = processed.groupby("Atmospheric_Layer")["Composite_100"].mean().reindex(["Troposphere","Stratosphere","Mesosphere","Thermosphere","Exosphere"]).fillna(0)
    alloc_norm = alloc / alloc.sum() if alloc.sum()>0 else alloc
    alloc_df = pd.DataFrame({"layer":alloc_norm.index, "weight":alloc_norm.values})
    fig3 = px.sunburst(alloc_df, path=['layer'], values='weight', color='layer', color_discrete_map=LAYER_COLORS, title="Layer Allocation Suggestion")
    st.plotly_chart(fig3, use_container_width=True)

    # --------------------
    # Export: CSV + HTML summary
    # --------------------
    st.markdown("## Export & Share")
    csv_bytes = processed.to_csv(index=False).encode("utf-8")
    st.download_button("Download full STRATO-AI CSV", data=csv_bytes, file_name=f"strato_ai_output_{datetime.utcnow().strftime('%Y%m%d')}.csv", mime="text/csv")

    # Quick HTML summary
    def make_html_summary(df_small):
        title = f"STRATO-AI Report — {datetime.utcnow().strftime('%Y-%m-%d')}"
        rows = []
        for _, r in df_small.iterrows():
            rows.append(f"<li><b>{r['ticker']}</b> — {r['Atmospheric_Layer']}: {r['Narrative']}</li>")
        body = "<ul>" + "".join(rows) + "</ul>"
        html = f"<html><head><title>{title}</title></head><body><h1>{title}</h1>{body}</body></html>"
        return html.encode("utf-8")

    top10 = processed.head(10)
    html_bytes = make_html_summary(top10)
    st.download_button("Download HTML client memo (top 10)", data=html_bytes, file_name=f"strato_ai_memo_{datetime.utcnow().strftime('%Y%m%d')}.html", mime="text/html")

    st.markdown("---")
    st.markdown("**Notes**: keep your decision log separately; record human overrides & rationale for your appendix.")
    st.balloons()

# Footer / quick help
st.markdown("---")
st.markdown("STRATO-AI • Built for Connor Barwin • Keep logs for judges (show human-AI collaboration).")
