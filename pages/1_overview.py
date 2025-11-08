import streamlit as st
import pandas as pd
import os
import plotly.express as px
import time

st.set_page_config(page_title="Overview | Flight Delay App", layout="wide")

st.title("✈️ Flight Delay Prediction Dashboard")
st.markdown("### Overview & Data Insights")

# -----------------------------------------
# 📂 Step 1: Load dataset
# -----------------------------------------
default_file = "flight_delay_data.csv"
uploaded_file = st.file_uploader("📂 Upload a different flight delay CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Loaded file: `{uploaded_file.name}`")
    file_used = uploaded_file.name
else:
    if os.path.exists(default_file):
        df = pd.read_csv(default_file)
        file_used = os.path.abspath(default_file)
        st.success(f"✅ Using default file: `{file_used}`")
    else:
        st.error(f"⚠️ File not found: `{os.path.abspath(default_file)}`")
        st.stop()

# -----------------------------------------
# 🧾 Step 2: Show dataset info
# -----------------------------------------
st.markdown("### 📊 Dataset Summary")
st.write(f"**Rows:** {df.shape[0]}, **Columns:** {df.shape[1]}")
st.write("**Columns:**", list(df.columns))

st.dataframe(df.head(10), use_container_width=True)

# -----------------------------------------
# 🕒 Step 3: Charts
# -----------------------------------------
st.markdown("## 📈 Data Visualizations")

# 1️⃣ Airline Frequency
if "airline" in df.columns:
    fig1 = px.pie(
        df,
        names="airline",
        title="Airline Distribution",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    st.plotly_chart(fig1, use_container_width=True)
else:
    st.warning("⚠️ Column 'airline' not found. Skipping airline chart.")

# 2️⃣ Flight Count per Airline
if "airline" in df.columns:
    airline_counts = df["airline"].value_counts().reset_index()
    airline_counts.columns = ["airline", "count"]
    fig2 = px.bar(
        airline_counts,
        x="airline",
        y="count",
        title="Number of Flights per Airline",
        color="count",
        color_continuous_scale="Blues"
    )
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.warning("⚠️ Column 'airline' not found. Skipping flight count chart.")

# 3️⃣ Delay Distribution
if "delay" in df.columns:
    st.subheader("⏰ Flight Delay Distribution (in minutes)")
    fig3 = px.histogram(
        df,
        x="delay",
        nbins=40,
        title="How Often Do Flights Get Delayed?",
        color_discrete_sequence=["#FF6B6B"],
        animation_frame=None
    )
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.warning("⚠️ Column 'delay' not found in dataset. Skipping delay distribution chart.")

# 4️⃣ Destination Trends (if available)
if "destination" in df.columns:
    st.subheader("🌍 Top 10 Destinations")
    top_dest = df["destination"].value_counts().nlargest(10).reset_index()
    top_dest.columns = ["destination", "count"]
    fig4 = px.bar(
        top_dest,
        x="destination",
        y="count",
        title="Top 10 Destination Airports",
        color="count",
        color_continuous_scale="Sunset"
    )
    st.plotly_chart(fig4, use_container_width=True)

# -----------------------------------------
# 🧭 Step 4: Footer
# -----------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center; font-size: 14px; color: gray;'>"
    "Developed by <b>Nirmal Raj</b> | Flight Delay App © 2025"
    "</div>",
    unsafe_allow_html=True
)
