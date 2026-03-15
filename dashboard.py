import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import joblib

import urllib.request
import base64

# === Streamlit Page Setup ===
# Must be the very first Streamlit command
st.set_page_config(
    page_title="F1 Finish Tier Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

bg_b64 = get_base64_of_bin_file('assets/f1_multi_bg.jpg')

# === Inject Custom CSS for Premium F1 Aesthetic ===
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700;900&display=swap" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Titillium+Web:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        /* Global Font */
        html, body, [class*="st-"] {
            font-family: 'Titillium Web', sans-serif !important;
        }
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Orbitron', 'Titillium Web', sans-serif !important;
            letter-spacing: 1px;
            text-transform: uppercase;
        }
        
        /* Top Banner & Titles */
        .title-text {
            color: #FF1801 !important;
            font-size: 3rem !important;
            font-weight: 900 !important;
            text-shadow: 2px 2px 4px rgba(255, 24, 1, 0.4);
            margin-bottom: 0px !important;
            padding-bottom: 0px !important;
        }
        .subtitle-text {
            color: #e6edf3;
            font-size: 1.2rem;
            margin-top: 5px !important;
            margin-bottom: 2rem !important;
            opacity: 0.8;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #15151e !important;
            border-right: 3px solid #FF1801 !important;
        }
        
        /* Metric Styling */
        div[data-testid="stMetricValue"] {
            font-family: 'Orbitron', sans-serif !important;
            color: #FF1801 !important;
        }
        div[data-testid="stMetricLabel"] {
            color: #e6edf3 !important;
            opacity: 0.8;
        }
        
        /* Buttons */
        .stButton>button {
            width: 100%;
            background-color: #FF1801 !important;
            color: white !important;
            font-family: 'Orbitron', sans-serif !important;
            font-weight: 700 !important;
            font-size: 1.2rem !important;
            border: none !important;
            border-radius: 4px !important;
            padding: 0.75rem !important;
            transition: all 0.3s ease !important;
            text-transform: uppercase;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        .stButton>button:hover {
            background-color: #cc1300 !important;
            box-shadow: 0 0 15px rgba(255, 24, 1, 0.6) !important;
            transform: translateY(-2px) !important;
        }
        
        /* Headers in main container */
        .section-header {
            border-bottom: 2px solid #38383f;
            padding-bottom: 5px;
            margin-bottom: 20px;
            color: #FF1801;
            font-size: 1.5rem;
            margin-top: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown(f"""
    <style>
        /* Background for main area */
        [data-testid="stAppViewContainer"] {{
            background-image: linear-gradient(rgba(17, 17, 21, 0.75), rgba(17, 17, 21, 0.85)), url("data:image/jpeg;base64,{bg_b64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            background-color: transparent !important; 
        }}
        
        [data-testid="stHeader"] {{
            background-color: transparent !important;
        }}
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
        /* Expander/Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: transparent !important;
            border-radius: 4px 4px 0px 0px;
            color: white !important;
            font-family: 'Orbitron', sans-serif !important;
            font-size: 1.1rem;
            padding: 0 1rem;
        }
        .stTabs [aria-selected="true"] {
            border-bottom: 3px solid #FF1801 !important;
            color: #FF1801 !important;
        }
        
        /* Elegant Slider Styling */
        div[data-baseweb="slider"] {
            margin-top: 10px !important;
            padding-bottom: 20px !important;
        }
        
        /* The unselected track (dark gray) */
        div[data-baseweb="slider"] > div > div:first-child {
            background-color: #2D2D36 !important; 
            height: 8px !important;
            border-radius: 4px !important;
        }

        /* The selected fill area (F1 Red gradient) */
        div[data-baseweb="slider"] > div > div:nth-child(2) {
            background: linear-gradient(90deg, #b81200 0%, #FF1801 100%) !important;
            height: 8px !important;
            border-radius: 4px !important;
        }

        /* The draggable thumb handle */
        div[data-baseweb="slider"] > div > div:nth-child(3) {
            height: 20px !important;
            width: 20px !important;
            background-color: #ffffff !important;
            border: 3px solid #FF1801 !important;
            box-shadow: 0 0 10px rgba(255, 24, 1, 0.6) !important;
            border-radius: 50% !important;
            transition: transform 0.1s ease, box-shadow 0.1s ease !important;
        }
        div[data-baseweb="slider"] > div > div:nth-child(3):hover {
            transform: scale(1.2) !important;
            box-shadow: 0 0 15px rgba(255, 24, 1, 0.9) !important;
            cursor: grab !important;
        }
        div[data-baseweb="slider"] > div > div:nth-child(3):active {
            cursor: grabbing !important;
        }

        /* Tooltip style */
        div[data-baseweb="tooltip"] {
            background-color: #15151e !important;
            border: 1px solid #FF1801 !important;
            color: white !important;
            font-family: 'Orbitron', sans-serif !important;
            border-radius: 4px !important;
            padding: 4px 8px !important;
            font-size: 0.9rem !important;
        }

        /* Enforce bright white text for input labels (Selectbox, Sliders, Number Inputs) */
        label[data-testid="stWidgetLabel"] p {
            color: #FFFFFF !important;
            font-size: 1.1rem !important;
            font-weight: 600 !important;
            letter-spacing: 0.5px;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.8);
        }
        
        /* Make sure general paragraph text in the sidebar or tools is readable */
        p {
            color: #e6edf3 !important;
        }
        
        /* Dropdown selection text color */
        div[data-baseweb="select"] div {
            color: #FFFFFF !important;
        }
    </style>
""", unsafe_allow_html=True)


# === CACHED DATA LOADING ===
@st.cache_data
def load_data():
    # Driver Mapping
    drivers_df = pd.read_csv("data/raw/drivers.csv")
    drivers_df['full_name'] = drivers_df['forename'] + " " + drivers_df['surname']
    driver_dict = dict(zip(drivers_df['full_name'], drivers_df['driverId']))

    # Status Mapping
    status_df = pd.read_csv("data/raw/status.csv")
    status_dict = dict(zip(status_df['status'], status_df['statusId']))

    # Circuit Mapping
    circuits_df = pd.read_csv("data/raw/circuits.csv")
    circuit_dict = dict(zip(circuits_df['name'], circuits_df['circuitId']))
    
    return driver_dict, status_dict, circuit_dict

driver_dict, status_dict, circuit_dict = load_data()

@st.cache_resource
def load_model():
    return joblib.load('notebooks/xgb_model.pkl')

model = load_model()

@st.cache_data
def load_metadata():
    return pd.read_csv("notebooks/sample_input_format.csv")

# === Main UI App ===
st.markdown('<h1 class="title-text">Formula 1 Race Strategy AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Advanced Predictive Modeling for Finish Tier Outcomes</p>', unsafe_allow_html=True)

# Main container layout
tab1, tab2 = st.tabs(["PREDICTION ENGINE", "HOW IT WORKS"])

with tab1:
    col1, col2 = st.columns([1, 2.5], gap="large")
    
    with col1:
        st.markdown('<div class="section-header">RACE PARAMETERS</div>', unsafe_allow_html=True)
        
        # Selectors
        driver_name = st.selectbox("Driver", list(driver_dict.keys()), index=list(driver_dict.keys()).index("Lewis Hamilton") if "Lewis Hamilton" in driver_dict else 0)
        driver_id = driver_dict[driver_name]

        circuit_label = st.selectbox("Circuit", list(circuit_dict.keys()), index=list(circuit_dict.keys()).index("Silverstone Circuit") if "Silverstone Circuit" in circuit_dict else 0)
        circuit_id = circuit_dict[circuit_label]

        status_label = st.selectbox("Finish Status", list(status_dict.keys()), index=list(status_dict.keys()).index("Finished") if "Finished" in status_dict else 0)
        status_id = status_dict[status_label]
        
        st.markdown('<div class="section-header" style="margin-top: 2rem;">TELEMETRY</div>', unsafe_allow_html=True)
        
        grid = st.number_input("Starting Grid Position", min_value=1, max_value=24, value=2)
        avg_lap_time = st.slider("Avg Lap Time (ms)", 60000, 180000, 95000, step=1000)
        lap_time_std = st.slider("Lap Time Variance (ms)", 0, 50000, 15000, step=500)
        num_pit_stops = st.slider("Number of Pit Stops", 0, 5, 2)
        total_pit_time = st.slider("Total Pit Stop Time (ms)", 0, 200000, 48000, step=1000)
        total_laps = st.slider("Total Laps Completed", 0, 80, 52)
        race_distance = avg_lap_time * total_laps

    with col2:
        st.markdown('<div class="section-header">SIMULATION RESULTS</div>', unsafe_allow_html=True)
        
        # Show mini metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Driver", driver_name)
        m2.metric("Grid Position", f"P{grid}")
        m3.metric("Avg Pace", f"{avg_lap_time/1000:.2f}s")
        m4.metric("Pit Strategy", f"{num_pit_stops} Stops")
        
        st.write("")
        st.write("")
        
        predict_button = st.button("RUN AI SIMULATION")
        
        if predict_button:
            with st.spinner('Analyzing telemetry and executing race simulation...'):
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

                input_row = input_row.fillna(0) # Ensure no NaNs remain

                # === PREDICTION ===
                pred = model.predict(input_row)[0]
                proba = model.predict_proba(input_row)[0]
                
                pred_map = {
                    0: "PODIUM FINISH (1–3)",
                    1: "MIDFIELD FINISH (4–10)",
                    2: "BACKMARKER FINISH (11+)"
                }
                finish_colors = ["#00e676", "#ffa726", "#ff5252"]
                
                finish_tier_text = pred_map[pred]
                finish_color = finish_colors[pred]

                st.write("")
                st.markdown(f"""
                <div style='
                    background: linear-gradient(135deg, rgba(20,20,26,1) 0%, rgba(30,30,38,1) 100%);
                    padding: 2em;
                    border-radius: 12px;
                    border: 2px solid {finish_color};
                    text-align: center;
                    font-family: Orbitron, sans-serif;
                    box-shadow: 0 0 30px {finish_color}40;
                    margin-bottom: 2rem;
                '>
                    <h4 style="color: white; margin: 0; padding: 0; font-size: 1.2rem; opacity: 0.8; font-family: 'Titillium Web';">PREDICTED OUTCOME</h4>
                    <h2 style="color: {finish_color}; margin: 10px 0 0 0; font-size: 2.8rem; font-weight: 900; text-shadow: 0 0 10px {finish_color}80;">
                        {finish_tier_text}
                    </h2>
                </div>
                """, unsafe_allow_html=True)

                # Confidence Bar Chart Customization
                st.markdown('<div class="section-header">MODEL CONFIDENCE DISTRIBUTIONS</div>', unsafe_allow_html=True)
                
                fig, ax = plt.subplots(figsize=(8, 3))
                
                # Plot with transparent background that integrates with dark mode
                fig.patch.set_facecolor('#111115')
                ax.set_facecolor('#111115')
                
                bars = ax.bar(["PODIUM (1-3)", "MIDFIELD (4-10)", "BACKMARKER (11+)"], 
                              proba, 
                              color=['#00e676', '#ffa726', '#ff5252'], 
                              edgecolor='#111115',
                              linewidth=2,
                              width=0.6)
                              
                ax.set_ylim(0, 1)
                
                # Styling the plot purely for aesthetics
                ax.tick_params(axis='x', colors='white', labelsize=10, labelrotation=0)
                ax.tick_params(axis='y', left=False, labelleft=False) # Hide y axis
                
                ax.spines['bottom'].set_color('#38383f')
                ax.spines['bottom'].set_linewidth(2)
                ax.spines['left'].set_color('none')
                ax.spines['top'].set_color('none')
                ax.spines['right'].set_color('none')

                for bar, prob in zip(bars, proba):
                    ax.text(bar.get_x() + bar.get_width()/2, 
                            bar.get_height() + 0.05, 
                            f"{prob*100:.1f}%",
                            ha='center', 
                            color='white', 
                            fontweight='bold',
                            fontsize=14,
                            family='Orbitron')
                            
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                st.write("")
        else:
            # Idle state visual
            st.info("Enter the race telemetry parameters on the left and click **RUN AI SIMULATION** to calculate the projected finish tier.")

with tab2:
    st.markdown('<div class="section-header">ABOUT THE MODEL</div>', unsafe_allow_html=True)
    st.markdown("""
    This advanced predictive engine uses an **XGBoost Classifier** trained on historical Formula 1 telemetry, lap time variability, pit stop strategies, and grid configurations.
    
    ### Key Features Analzyed:
    * **Grid Position**: Where the driver starts heavily influences their finish trajectory.
    * **Lap Time Variance**: A measure of consistency. Drivers with highly variable lap times generally struggle holding track position.
    * **Pit Strategy**: Total time spent in the pits and the number of stops affect race delta times.
    * **Circuit Dynamics**: Different tracks have different overtaking probabilities and degradation models, accounted for internally.
    
    _Built by Nandini Patel | Powered by XGBoost and Streamlit_
    """)
