
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load("random_forest_gridsearch.pkl")

st.set_page_config(page_title="Smartphone Price Predictor", layout="centered")

st.title("📱 Smartphone Price Prediction App")

st.markdown("Enter the smartphone specifications below:")

# Input form
with st.form("prediction_form"):
    RAM = st.number_input("RAM (in GB)", min_value=1, max_value=32, value=4)
    storage = st.number_input("Storage (in GB)", min_value=8, max_value=1024, value=64)
    Battery_cap = st.number_input("Battery Capacity (mAh)", min_value=1000, max_value=7000, value=4000)
    has_fast_charging = st.selectbox("Has Fast Charging?", ["Yes", "No"])
    has_nfc = st.selectbox("Has NFC?", ["Yes", "No"])
    has_5g = st.selectbox("Has 5G?", ["Yes", "No"])
    num_core = st.number_input("Number of Cores", min_value=1, max_value=16, value=8)
    primery_rear_camera = st.number_input("Primary Rear Camera (MP)", min_value=1, max_value=200, value=48)
    Num_Rear_Cameras = st.number_input("Number of Rear Cameras", min_value=1, max_value=5, value=3)
    primery_front_camera = st.number_input("Primary Front Camera (MP)", min_value=1, max_value=100, value=16)
    display_size = st.number_input("Display Size (inches)", min_value=3.0, max_value=8.0, value=6.5)
    display_types = st.selectbox("Display Type", ["LCD", "AMOLED", "OLED", "IPS", "TFT"])

    submitted = st.form_submit_button("Predict Price")

# Map input to label-encoded values if necessary
if submitted:
    # Manual encoding (assumed mapping, must match training)
    fast_charging = 1 if has_fast_charging == "Yes" else 0
    nfc = 1 if has_nfc == "Yes" else 0
    g5 = 1 if has_5g == "Yes" else 0
    display_map = {"LCD": 1, "AMOLED": 0, "OLED": 3, "IPS": 2, "TFT": 4}
    display_encoded = display_map.get(display_types, 1)

    features = np.array([[RAM, storage, Battery_cap, fast_charging, nfc, g5, num_core,
                          primery_rear_camera, Num_Rear_Cameras, primery_front_camera,
                          display_size, display_encoded]])
    
    prediction = model.predict(features)[0]
    st.success(f"💸 Predicted Price: ₹{round(prediction, 2)}")
