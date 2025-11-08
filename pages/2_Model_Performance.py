import streamlit as st
import pandas as pd
import plotly.express as px
import os
import joblib
import numpy as np

st.set_page_config(page_title="Model Performance | Flight Delay App", layout="wide")

st.title("📊 Model Performance Summary")
st.markdown("### Evaluate the performance of your trained model")

# -----------------------------------------
# Paths
# -----------------------------------------
model_path = "models/best_model.pkl"
performance_csv = "model_performance.csv"

# -----------------------------------------
# Check model existence
# -----------------------------------------
if not os.path.exists(model_path):
    st.error("⚠️ Model file not found. Please ensure `models/best_model.pkl` exists.")
    st.stop()

# Load model
model = joblib.load(model_path)
st.success("✅ Trained model loaded successfully!")

# -----------------------------------------
# Load performance summary
# -----------------------------------------
if os.path.exists(performance_csv):
    df_perf = pd.read_csv(performance_csv)
    st.subheader("📈 Model Performance Metrics")
    st.dataframe(df_perf, use_container_width=True)

    # Accuracy trend (if multiple models tested)
    if "Accuracy" in df_perf.columns:
        fig_acc = px.line(
            df_perf,
            x=df_perf.index,
            y="Accuracy",
            title="📈 Accuracy Trend over Model Iterations",
            markers=True,
            color_discrete_sequence=["#00BFFF"]
        )
        st.plotly_chart(fig_acc, use_container_width=True)

    # Precision, Recall, F1-score comparison
    metric_cols = [col for col in ["Precision", "Recall", "F1-Score"] if col in df_perf.columns]
    if metric_cols:
        fig_metrics = px.bar(
            df_perf.melt(value_vars=metric_cols, var_name="Metric", value_name="Score"),
            x="Metric",
            y="Score",
            color="Metric",
            barmode="group",
            title="🔍 Model Metric Comparison",
            color_discrete_sequence=px.colors.sequential.RdPu
        )
        st.plotly_chart(fig_metrics, use_container_width=True)

else:
    st.error("❌ `model_performance.csv` not found. Run model training first.")
    st.stop()

# -----------------------------------------
# Feature Importance (if available)
# -----------------------------------------
st.markdown("### 🧠 Feature Importance")
try:
    if hasattr(model, "feature_importances_"):
        importance = pd.DataFrame({
            "Feature": model.feature_names_in_,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        fig_imp = px.bar(
            importance,
            x="Importance",
            y="Feature",
            orientation="h",
            title="🎯 Feature Importance",
            color="Importance",
            color_continuous_scale="Viridis",
            animation_frame=None
        )
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.warning("⚠️ Model has no feature importance attribute.")
except Exception as e:
    st.warning(f"⚠️ Could not load feature importance: {e}")

# -----------------------------------------
# Confusion Matrix (if saved separately)
# -----------------------------------------
conf_matrix_file = "confusion_matrix.csv"
if os.path.exists(conf_matrix_file):
    st.markdown("### 🔢 Confusion Matrix")
    cm = pd.read_csv(conf_matrix_file, header=None)
    cm.columns = ["Predicted Negative", "Predicted Positive"]
    cm.index = ["Actual Negative", "Actual Positive"]

    fig_cm = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale="Plasma",
        title="📉 Confusion Matrix Heatmap"
    )
    st.plotly_chart(fig_cm, use_container_width=True)
else:
    st.info("ℹ️ Confusion matrix file not found (`confusion_matrix.csv`). Skipping...")

# -----------------------------------------
# ROC Curve (if file exists)
# -----------------------------------------
roc_file = "roc_data.csv"
if os.path.exists(roc_file):
    st.markdown("### 📉 ROC Curve")
    roc_data = pd.read_csv(roc_file)
    fig_roc = px.area(
        roc_data,
        x="False Positive Rate",
        y="True Positive Rate",
        title="💡 Receiver Operating Characteristic (ROC) Curve",
        color_discrete_sequence=["#FF6B6B"],
        animation_frame=None
    )
    st.plotly_chart(fig_roc, use_container_width=True)
else:
    st.info("ℹ️ ROC data file not found (`roc_data.csv`). Skipping...")

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
