import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

def run_strato_ai_screener(df, mtwb_weight, weather):
    df = df.copy()

    numeric_cols = ["ATMO_Pressure", "ATMO_Heat", "ATMO_Turbulence", "MTWB_Impact"]
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    scaler = MinMaxScaler()
    df[["P_scaled", "H_scaled", "T_scaled", "I_scaled"]] = scaler.fit_transform(df[numeric_cols])

    km = KMeans(n_clusters=5, n_init=10, random_state=0)
    df["Cluster"] = km.fit_predict(df[["P_scaled", "H_scaled", "T_scaled", "I_scaled"]])

    layer_map = {
        0: "Troposphere",
        1: "Stratosphere",
        2: "Mesosphere",
        3: "Thermosphere",
        4: "Exosphere"
    }
    df["Layer"] = df["Cluster"].map(layer_map)

    weather_boost = {
        "Clear Skies": {"Troposphere": 4, "Stratosphere": 4},
        "Jetstreams": {"Thermosphere": 5, "Exosphere": 3},
        "Heatwave": {"Exosphere": 6},
        "Turbulence": {"Stratosphere": 5, "Troposphere": 3},
        "Solar Flare": {"Troposphere": 8, "Stratosphere": 6}
    }

    df["ATMO_Score"] = (
        df["P_scaled"] * 0.25 +
        df["H_scaled"] * 0.25 +
        df["T_scaled"] * 0.25 +
        df["I_scaled"] * mtwb_weight
    )

    df["ATMO_Score"] += df["Layer"].map(lambda x: weather_boost.get(weather, {}).get(x, 0))

    df["Rank"] = df["ATMO_Score"].rank(ascending=False)

    return df.sort_values("Rank")

