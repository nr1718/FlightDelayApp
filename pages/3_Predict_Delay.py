import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import os

# -------------------------------------------------
# Page Setup
# -------------------------------------------------
st.set_page_config(page_title="Predict Flight Delay | Flight Delay App", layout="wide")

st.title("✈️ Predict Flight Delay")
st.markdown("### Enter your flight details and see if your flight is likely to be delayed or on time!")

# -------------------------------------------------
# Load Dataset (for dropdowns)
# -------------------------------------------------
data_path = "flight_delay_data.csv"
if not os.path.exists(data_path):
    st.error("⚠️ Dataset not found! Please make sure 'flight_delay_data.csv' is in your project folder.")
    st.stop()

df = pd.read_csv(data_path)

# Basic column check
required_cols = {"airline", "origin", "dest", "month", "day_of_week", "departure_time"}
if not required_cols.issubset(df.columns):
    st.warning("⚠️ Some required columns are missing in your dataset. Please check your CSV file.")
    st.stop()

# Unique values for dropdowns
airlines = sorted(df["airline"].dropna().unique().tolist())
origins = sorted(df["origin"].dropna().unique().tolist())
destinations = sorted(df["dest"].dropna().unique().tolist())

# -------------------------------------------------
# Load Model
# -------------------------------------------------
model_path = "models/best_model.pkl"
if not os.path.exists(model_path):
    st.error("⚠️ No trained model found. Please train and save your model first.")
    st.stop()

try:
    model = joblib.load(model_path)
    st.success("✅ Trained model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# -------------------------------------------------
# Sidebar Inputs
# -------------------------------------------------
st.sidebar.header("🧭 Input Flight Details")

airline = st.sidebar.selectbox("Airline", airlines)
origin = st.sidebar.selectbox("Origin Airport", origins)
destination = st.sidebar.selectbox("Destination Airport", destinations)
month = st.sidebar.slider("Month", 1, 12, 6)
day_of_week = st.sidebar.slider("Day of Week (1=Mon, 7=Sun)", 1, 7, 3)
departure_time = st.sidebar.slider("Scheduled Departure (24h)", 0, 23, 15)

# Display entered info
input_data = pd.DataFrame({
    "airline": [airline],
    "origin": [origin],
    "dest": [destination],
    "month": [month],
    "day_of_week": [day_of_week],
    "departure_time": [departure_time]
})
st.markdown("### 🧾 Flight Details Summary")
st.dataframe(input_data, use_container_width=True)

# -------------------------------------------------
# Encode Categorical Columns
# -------------------------------------------------
# Generate encoders dynamically based on dataset
airline_encoder = {val: idx for idx, val in enumerate(df["airline"].unique())}
origin_encoder = {val: idx for idx, val in enumerate(df["origin"].unique())}
dest_encoder = {val: idx for idx, val in enumerate(df["dest"].unique())}

encoded_data = pd.DataFrame({
    "airline_enc": [airline_encoder.get(airline, 0)],
    "origin_enc": [origin_encoder.get(origin, 0)],
    "dest_enc": [dest_encoder.get(destination, 0)],
    "month": [month],
    "day_of_week": [day_of_week],
    "departure_time": [departure_time]
})

# -------------------------------------------------
# Prediction
# -------------------------------------------------
st.markdown("---")
if st.button("🎯 Predict Flight Delay"):
    try:
        prediction = model.predict(encoded_data)[0]
        proba = model.predict_proba(encoded_data)[0] if hasattr(model, "predict_proba") else [None, None]

        status = "🟢 On Time" if prediction == 0 else "🔴 Delayed"
        color = "#00C853" if prediction == 0 else "#D50000"

        st.markdown(f"### Prediction Result: <span style='color:{color}; font-size:26px'>{status}</span>", unsafe_allow_html=True)

        # Probability visualization
        if proba[1] is not None:
            st.markdown(f"**Confidence:** {proba[1]*100:.2f}%")

            fig = px.bar(
                x=["On Time", "Delayed"],
                y=proba,
                color=["On Time", "Delayed"],
                color_discrete_sequence=["#00C853", "#D50000"],
                text=[f"{p*100:.1f}%" for p in proba],
                title="🚦 Prediction Confidence",
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")

# -------------------------------------------------
# How It Works Section
# -------------------------------------------------
st.markdown("---")
st.markdown("### 💡 How It Works")
st.info("""
This tool predicts flight delays using **machine learning** trained on real flight data.  
It considers:
- Airline ✈️  
- Route (Origin ➜ Destination) 🌍  
- Month & Day of the Week 📅  
- Scheduled Departure Time 🕒  

The system dynamically adapts to new data if you update your CSV file.
""")

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
