import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# =============================
# Load the saved model
# =============================
model_data = joblib.load("final_bike_rental_model.joblib")  # <-- replace with your joblib file name
pipeline = model_data["model"]
best_params = model_data["best_params"]
best_score = model_data["best_score"]
test_metrics = model_data["test_metrics"]
feature_columns = model_data["feature_columns"]
training_date = model_data["training_date"]
split_ratio = model_data["split_ratio"]

# =============================
# Streamlit GUI
# =============================
st.set_page_config(page_title="Decision Tree Regressor App", layout="wide")
st.title("🌳 Decision Tree Regression Model (Streamlit GUI)")

# Sidebar: Model Information
st.sidebar.header("📌 Model Information")
st.sidebar.write(f"**Training Date:** {training_date}")
st.sidebar.write(f"**Train/Test Split:** {split_ratio}")
st.sidebar.write(f"**Best CV Score (R²):** {best_score:.4f}")

st.sidebar.subheader("Best Hyperparameters")
for k, v in best_params.items():
    st.sidebar.write(f"- {k}: {v}")

st.sidebar.subheader("Test Metrics")
for metric, value in test_metrics.items():
    st.sidebar.write(f"- {metric}: {value:.4f}")

# =============================
# User Input Section
# =============================
st.header("🔢 Input Features for Prediction")

col1, col2 = st.columns(2)

with col1:
    temperature = st.number_input("Temperature (°C)", value=25.0)
    hour = st.slider("Hour of Day (0–23)", 0, 23, 12)
    humidity = st.slider("Humidity (%)", 0, 100, 50)

with col2:
    rainfall = st.number_input("Rainfall (mm)", value=0.0, step=0.1)
    visibility = st.number_input("Visibility (km)", value=10.0)
    season_winter = st.selectbox("Is it Winter?", ["No", "Yes"])
    season_autumn = st.selectbox("Is it Autumn?", ["No", "Yes"])

# Convert categorical to binary
season_winter_val = 1 if season_winter == "Yes" else 0
season_autumn_val = 1 if season_autumn == "Yes" else 0

# Build dataframe for prediction
input_data = pd.DataFrame([[
    temperature, hour, humidity, rainfall, visibility,
    season_winter_val, season_autumn_val
]], columns=feature_columns["numerical"] + feature_columns["categorical"])

# =============================
# Prediction
# =============================
if st.button("🔮 Predict"):
    prediction = pipeline.predict(input_data)[0]
    st.success(f"**Predicted Value: {prediction:.2f}**")

# =============================
# Expandable: Show Model Pipeline
# =============================
with st.expander("🔍 View Pipeline Details"):
    st.write(pipeline)

