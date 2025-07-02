import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import joblib

# Load driver ID to name mapping
mapping_df = pd.read_csv("data/final_data.csv")

# === DRIVER MAPPING ===
drivers_df = pd.read_csv("data/raw/drivers.csv")
drivers_df['full_name'] = drivers_df['forename'] + " " + drivers_df['surname']
driver_dict = dict(zip(drivers_df['full_name'], drivers_df['driverId']))

# === STATUS MAPPING ===
status_df = pd.read_csv("data/raw/status.csv")
status_dict = dict(zip(status_df['status'], status_df['statusId']))

# === CIRCUIT MAPPING ===
circuits_df = pd.read_csv("data/raw/circuits.csv")
circuit_dict = dict(zip(circuits_df['name'], circuits_df['circuitId']))

# Load trained XGBoost model
model = joblib.load('notebooks/xgb_model.pkl')

# Load sample format with all required input columns
@st.cache_data
def load_metadata():
    return pd.read_csv("notebooks/sample_input_format.csv")

# Layout
# === Streamlit Page Setup ===
st.set_page_config(
    page_title="🏁 F1 Finish Tier Predictor",
    layout="wide",
    page_icon="🏎️",
)

# === Inject Custom CSS for F1 Aesthetic ===
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Orbitron&display=swap" rel="stylesheet">
    <style>
        html, body, [class*="st-"], h1, h2, h3, .stButton button {
            font-family: 'Orbitron', sans-serif !important;
            letter-spacing: 0.5px;
        }
    </style>
    <style>
        /* Page title */
        .stApp {
            background-color: #0d1117;
            color: #e6edf3;
        }
        h1 {
            color: #ff3838;
            font-family: 'Trebuchet MS', sans-serif;
            font-weight: bold;
        }
        .stButton button {
            background-color: #ff3838;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 0.5em 1.5em;
        }
        .stButton button:hover {
            background-color: #e52e2e;
        }
        .stSidebar {
            background-color: #161b22;
        }
        .stSlider > div {
            color: #e6edf3;
        }
        .stSelectbox label, .stNumberInput label {
            color: #e6edf3;
        }
        .stMarkdown {
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)


# Page Title
st.title("🏎️ F1 Finish Tier Predictor")
st.markdown("#### Predict whether a driver will finish in **Top 3**, **4–10**, or **11+** based on detailed race telemetry and strategy parameters.")

# === SIDEBAR INPUTS ===
st.sidebar.header("Race Parameters")
avg_lap_time = st.sidebar.slider("Avg Lap Time (ms)", 60000, 180000, 95000)
lap_time_std = st.sidebar.slider("Lap Time Std Dev (ms)", 0, 50000, 15000)
total_pit_time = st.sidebar.slider("Total Pit Time (ms)", 0, 200000, 60000)
num_pit_stops = st.sidebar.slider("Pit Stops", 0, 5, 2)
grid = st.sidebar.slider("Starting Grid Position", 1, 24, 10)
total_laps = st.sidebar.slider("Total Laps", 40, 80, 60)
race_distance = avg_lap_time * total_laps

# === DROPDOWNS ===
driver_name = st.selectbox("Driver", list(driver_dict.keys()))
driver_id = driver_dict[driver_name]

status_label = st.sidebar.selectbox("Status", list(status_dict.keys()))
status_id = status_dict[status_label]

circuit_label = st.sidebar.selectbox("Circuit", list(circuit_dict.keys()))
circuit_id = circuit_dict[circuit_label]

# === BUILD INPUT ROW ===
input_row = load_metadata().iloc[0:0]
input_row.loc[0, 'avg_lap_time'] = avg_lap_time
input_row.loc[0, 'lap_time_std'] = lap_time_std
input_row.loc[0, 'total_pit_time'] = total_pit_time
input_row.loc[0, 'num_pit_stops'] = num_pit_stops
input_row.loc[0, 'grid'] = grid
input_row.loc[0, 'race_distance'] = race_distance

# === ONE-HOT ENCODING ===
for col in input_row.columns:
    if 'driverId_' in col:
        input_row[col] = 1 if f"driverId_{driver_id}" == col else 0
    elif 'statusId_' in col:
        input_row[col] = 1 if f"statusId_{status_id}" == col else 0
    elif 'circuitId_' in col:
        input_row[col] = 1 if f"circuitId_{circuit_id}" == col else 0

# === PREDICTION ===
if st.button("Predict Finish Tier"):
    pred = model.predict(input_row)[0]
    pred_map = {
    0: "🏆 Podium (1–3)",
    1: "🎯 Midfield (4–10)",
    2: "🚦 Backmarker (11+)"
}
    finish_colors = ["#00e676", "#ffa726", "#ff5252"]
    finish_tier_text = pred_map[pred]
    finish_color = finish_colors[pred]

    st.markdown(f"""
    <div style='
        background-color: {finish_color};
        padding: 1em;
        border-radius: 8px;
        text-align: center;
        font-family: Orbitron, sans-serif;
        font-size: 24px;
        font-weight: bold;
        color: black;
        box-shadow: 0 0 12px {finish_color};
    '>
    🏁 Finish Tier: {finish_tier_text}
    </div>
    """, unsafe_allow_html=True)


    # Confidence Bar
    proba = model.predict_proba(input_row)[0]
    st.markdown("### 🧠 Confidence Levels")
    fig, ax = plt.subplots(figsize=(6, 2.2))

    bars = ax.bar(["Top 3", "4–10", "11+"], proba, color=['#00e676', '#ffa726', '#ff5252'], edgecolor='white')
    ax.set_ylabel("Probability", color='white')
    ax.set_ylim(0, 1)
    ax.set_facecolor('#0d1117')
    fig.patch.set_facecolor('#0d1117')

    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_color('#0d1117')
    ax.spines['right'].set_color('#0d1117')

    for bar, prob in zip(bars, proba):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03, f"{prob:.2f}",
                ha='center', color='white', fontweight='bold')
    plt.rcParams['font.family'] = 'Orbitron'
    st.pyplot(fig, use_container_width=True)

