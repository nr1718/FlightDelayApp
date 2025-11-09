# pages/3_Predict_Delay.py

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# -------------------------------------------------
# Page Setup & Constants
# -------------------------------------------------
st.set_page_config(page_title="Predict Flight Delay | Flight Delay App", layout="wide")

st.title("✈️ Predict Flight Delay")
st.markdown("### Enter your flight details and see the predicted delay in minutes.")
st.markdown("---")

# Define the dictionary for renaming columns (must match the main app!)
COLUMN_RENAMES = {
    "destination": "dest",
    "sched_dep_hour": "departure_time",
    "sched_arr_hour": "arrival_time",
    "delay_minutes": "delay"
}

# Define paths for model components
MODEL_PATH = "models/best_model.pkl"
PREPROCESSORS = {
    'scaler': 'models/scaler.pkl',
    'le_airline': 'models/le_airline.pkl',
    'le_origin': 'models/le_origin.pkl',
    'le_dest': 'models/le_dest.pkl',
    'le_weather': 'models/le_weather.pkl'
}
DATA_PATH = "flight_delay_data.csv"
FEATURES = ['airline_enc', 'origin_enc', 'dest_enc', 'departure_time', 'arrival_time', 'distance', 'day_of_week', 'month', 'weather_enc']
COLS_TO_SCALE = ['departure_time', 'arrival_time', 'distance', 'day_of_week', 'month']

# -------------------------------------------------
# Load Data and Preprocessors
# -------------------------------------------------

# 1. Load Data (for dropdowns)
if not os.path.exists(DATA_PATH):
    st.error("⚠️ Dataset not found! Please make sure 'flight_delay_data.csv' is in your project folder.")
    st.stop()
df_raw = pd.read_csv(DATA_PATH)
df = df_raw.rename(columns=COLUMN_RENAMES, errors='ignore') # CRITICAL: RENAME HERE

# 2. Check for renamed columns and stop if critical ones are missing
if not {'airline', 'origin', 'dest', 'departure_time', 'arrival_time', 'weather'}.issubset(df.columns):
    st.warning("⚠️ Critical columns (like 'dest', 'departure_time') are missing after renaming. Please check your data or renaming dictionary.")
    st.stop()
    
# 3. Load Model and Preprocessors
try:
    model = joblib.load(MODEL_PATH)
    # CRITICAL: Load saved preprocessors
    scaler = joblib.load(PREPROCESSORS['scaler'])
    le_airline = joblib.load(PREPROCESSORS['le_airline'])
    le_origin = joblib.load(PREPROCESSORS['le_origin'])
    le_dest = joblib.load(PREPROCESSORS['le_dest'])
    le_weather = joblib.load(PREPROCESSORS['le_weather'])
    st.success("✅ Trained model and preprocessors loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model/preprocessors: {e}. Ensure training ran successfully.")
    st.stop()

# Unique values for dropdowns (using renamed DF)
airlines = sorted(df["airline"].dropna().unique().tolist())
origins = sorted(df["origin"].dropna().unique().tolist())
destinations = sorted(df["dest"].dropna().unique().tolist())
weathers = sorted(df["weather"].dropna().unique().tolist())


# -------------------------------------------------
# Sidebar Inputs
# -------------------------------------------------
st.sidebar.header("🧭 Input Flight Details")

with st.sidebar.form("input_form"):
    airline = st.selectbox("Airline", airlines)
    origin = st.selectbox("Origin Airport", origins)
    destination = st.selectbox("Destination Airport", destinations)
    
    # CRITICAL: Add the weather and arrival time inputs, which were missing but used in training
    weather = st.selectbox("Weather Condition", weathers)
    dep_hour = st.slider("Scheduled Departure (24h)", 0, 23, 10)
    arr_hour = st.slider("Scheduled Arrival (24h)", 0, 23, 12)
    distance = st.number_input("Distance (km)", 50, 5000, 500)
    month = st.slider("Month", 1, 12, 6)
    day_of_week = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 3) # Changed range to 0-6 to match training

    submit_btn = st.form_submit_button("🎯 Predict Delay")

# -------------------------------------------------
# Prediction Logic
# -------------------------------------------------

if submit_btn:
    try:
        # 1. Create Input DataFrame (using the features the model expects)
        X_input = pd.DataFrame({
            'airline_enc': [le_airline.transform([airline])[0]],
            'origin_enc': [le_origin.transform([origin])[0]],
            'dest_enc': [le_dest.transform([destination])[0]],
            'departure_time': [dep_hour],
            'arrival_time': [arr_hour],
            'distance': [distance],
            'day_of_week': [day_of_week],
            'month': [month],
            'weather_enc': [le_weather.transform([weather])[0]]
        })

        # 2. Scale Numerical Features
        X_input_scaled = X_input.copy()
        X_input_scaled[COLS_TO_SCALE] = scaler.transform(X_input[COLS_TO_SCALE])
        
        # 3. Predict (using Regression model)
        # Note: We only pass the required features, which are now correctly scaled/encoded
        pred_delay = model.predict(X_input_scaled[FEATURES])[0]
        predicted_delay_minutes = max(0, int(round(pred_delay)))
        
        # 4. Display Results
        if predicted_delay_minutes >= 30:
            status_color = "#D50000"
            status_text = "🔴 SIGNIFICANT DELAY"
            st.markdown(f"### Prediction Result: <span style='color:{status_color}; font-size:28px'>{status_text}</span>", unsafe_allow_html=True)
            st.balloons()
        elif predicted_delay_minutes >= 15:
            status_color = "#FF8800"
            status_text = "🟠 MODERATE DELAY"
            st.markdown(f"### Prediction Result: <span style='color:{status_color}; font-size:28px'>{status_text}</span>", unsafe_allow_html=True)
        else:
            status_color = "#00C853"
            status_text = "🟢 ON TIME"
            st.markdown(f"### Prediction Result: <span style='color:{status_color}; font-size:28px'>{status_text}</span>", unsafe_allow_html=True)
            
        st.metric("Predicted Delay", f"{predicted_delay_minutes} minutes")

    except ValueError as ve:
        st.error(f"❌ Error during encoding/prediction. Check if the input values ({airline}, {origin}, {destination}, {weather}) were present in the training data. Error: {ve}")
    except Exception as e:
        st.error(f"❌ An unexpected error occurred during prediction: {e}")

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color: gray; font-size: 14px;'>"
    "Developed by <b>Nirmal Raj</b> | Flight Delay App © 2025"
    "</div>",
    unsafe_allow_html=True
)