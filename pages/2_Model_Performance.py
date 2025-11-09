# pages/2_Model_Performance.py

import streamlit as st
import pandas as pd
import plotly.express as px
import os
import joblib
import numpy as np

st.set_page_config(page_title="Model Performance | Flight Delay App", layout="wide")

st.title("📊 Model Performance Summary")
st.markdown("### Evaluate the performance of your trained model (Regression)")
st.markdown("---")

# -----------------------------------------
# Paths (CORRECTED)
# -----------------------------------------
model_path = "models/best_model.pkl"
performance_csv = "output/model_performance.csv" # CORRECTED PATH

# -----------------------------------------
# Check model existence
# -----------------------------------------
if not os.path.exists(model_path):
    st.error("⚠️ Model file not found. Please ensure `models/best_model.pkl` exists.")
    st.stop()

# Load model and preprocessors
try:
    model = joblib.load(model_path)
    scaler = joblib.load('models/scaler.pkl')
    st.success("✅ Trained model and scaler loaded successfully!")
except Exception as e:
    st.error(f"Error loading model components: {e}")
    st.stop()


# -----------------------------------------
# Load performance summary
# -----------------------------------------
if os.path.exists(performance_csv):
    df_perf = pd.read_csv(performance_csv)
    st.subheader("📈 Model Performance Metrics")
    
    # Display the metrics table (R2, MAE)
    st.table(df_perf) # Using st.table is clean for small metric tables
    
    # Display metrics in columns for better visualization
    metric_dict = df_perf.set_index('Metric')['Value'].to_dict()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R-squared ($R^2$)", metric_dict.get('R-squared ($R^2$)', 'N/A'))
    col2.metric("MAE (Min)", metric_dict.get('Mean Absolute Error (MAE)', 'N/A'))
    col3.metric("Train Set Size", metric_dict.get('Trained On Rows', 'N/A'))
    col4.metric("Test Set Size", metric_dict.get('Test Set Rows', 'N/A'))

else:
    st.error("❌ `outputs/model_performance.csv` not found. Please reload the main app page to run model training first.")
    st.stop()


# -----------------------------------------
# Feature Importance (Coefficient Magnitude for Linear Regression)
# -----------------------------------------
st.markdown("### 🧠 Feature Coefficients (Importance)")

try:
    # Feature names are NOT stored in the LinearRegression model, you must supply them.
    # Assuming the encoded features are:
    feature_names = ['airline_enc', 'origin_enc', 'dest_enc', 'departure_time', 'arrival_time', 'distance', 'day_of_week', 'month', 'weather_enc']
    
    # Get coefficients
    coefs = model.coef_
    
    importance = pd.DataFrame({
        "Feature": feature_names,
        # Use the absolute magnitude of the coefficient for importance, as signs don't matter here
        "Importance": np.abs(coefs), 
        "Coefficient": coefs # Keep the raw coefficient for analysis
    }).sort_values(by="Importance", ascending=True)

    # Plotting coefficients
    fig_imp = px.bar(
        importance,
        x="Importance",
        y="Feature",
        orientation="h",
        # Use Coefficient for color to show positive/negative impact
        color="Coefficient", 
        title="🎯 Feature Impact (Coefficient Magnitude)",
        color_continuous_scale=px.colors.diverging.RdBu, # Red/Blue scale is good for +/-
        hover_data={"Coefficient": ':.4f'}
    )
    st.plotly_chart(fig_imp, use_container_width=True)

except Exception as e:
    st.warning(f"⚠️ Could not load feature coefficients. Ensure training features are consistent. Error: {e}")


# -----------------------------------------
# (Removed sections for Confusion Matrix, ROC Curve, and Accuracy Trend as they are Classification metrics)
# -----------------------------------------

# -----------------------------------------
# Footer
# -----------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center; font-size: 14px; color: gray;'>"
    "Developed by <b>Nirmal Raj</b> | Flight Delay App © 2025"
    "</div>",
    unsafe_allow_html=True
)