import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor

# ---------------------------------------------------------
#  STRATO-AI — Atmospheric Investment Intelligence Engine
# ---------------------------------------------------------

def scale(df, cols):
    scaler = MinMaxScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df

def compute_mt_wb_impact(df):
    impact_cols = ["ESG", "Sustainability", "Community", "DEI", "Safety", "Renewables"]
    for col in impact_cols:
        if col not in df.columns:
            df[col] = 0.5  # neutral if missing
    df["MTWB_Impact_Score"] = df[impact_cols].mean(axis=1)
    return df

def compute_atmo_features(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df = scale(df, numeric_cols)
    df["ATMO_Pressure"] = df[numeric_cols].mean(axis=1)
    df["ATMO_Heat"] = df[numeric_cols].std(axis=1)
    df["ATMO_Turbulence"] = np.abs(df[numeric_cols].diff(axis=1)).mean(axis=1)
    df = df.fillna(0)
    return df

def kmeans_layer_classification(df):
    kmeans = KMeans(n_clusters=5, random_state=42, n_init='auto')
    clusters = kmeans.fit_predict(df[["ATMO_Pressure", "ATMO_Heat", "ATMO_Turbulence"]])
    df["Cluster"] = clusters

    mapping = {
        0: "Troposphere",
        1: "Stratosphere",
        2: "Mesosphere",
        3: "Thermosphere",
        4: "Exosphere"
    }
    df["Atmospheric_Layer"] = df["Cluster"].map(mapping)
    return df

def random_forest_priority(df):
    if "1Y_Return" not in df.columns:
        df["1Y_Return"] = np.random.uniform(-0.1, 0.2, size=len(df))

    features = df.select_dtypes(include=[np.number]).drop(columns=["1Y_Return"], errors='ignore')
    target = df["1Y_Return"]

    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(features, target)

    df["Priority_Score"] = rf.predict(features)
    df["Priority_Score"] = MinMaxScaler().fit_transform(df[["Priority_Score"]])
    return df

def apply_macro_weather(df, macro):
    if macro == "Clear Skies":
        df["Weather_Adjust"] = df["Priority_Score"] + 0.05

    elif macro == "Jetstreams":
        df["Weather_Adjust"] = df["Priority_Score"] + (df["Atmospheric_Layer"] == "Thermosphere") * 0.07

    elif macro == "Heatwave":
        df["Weather_Adjust"] = df["Priority_Score"] + (df["Atmospheric_Layer"] == "Exosphere") * 0.09

    elif macro == "Turbulence":
        df["Weather_Adjust"] = df["Priority_Score"] - (df["Atmospheric_Layer"] == "Mesosphere") * 0.10

    elif macro == "Solar Flare":
        df["Weather_Adjust"] = df["Priority_Score"] - 0.05
    else:
        df["Weather_Adjust"] = df["Priority_Score"]

    return df

def final_weight(df, mtwb_weight, risk_weight):
    df["Final_Score"] = (
        df["Weather_Adjust"] * (0.6 - risk_weight)
        + df["MTWB_Impact_Score"] * mtwb_weight
        + df["Priority_Score"] * (0.4 + risk_weight)
    )
    df["Final_Score"] = MinMaxScaler().fit_transform(df[["Final_Score"]])
    return df

# ---------------------------------------------------------
# MASTER FUNCTION — called from the Streamlit app
# ---------------------------------------------------------

def run_strato_ai(df, mtwb_weight, risk_weight, macro_weather):

    df = compute_mt_wb_impact(df)
    df = compute_atmo_features(df)
    df = kmeans_layer_classification(df)
    df = random_forest_priority(df)
    df = apply_macro_weather(df, macro_weather)
    df = final_weight(df, mtwb_weight, risk_weight)

    df = df.sort_values("Final_Score", ascending=False)

    return df
