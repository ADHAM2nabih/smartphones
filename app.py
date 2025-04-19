import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('random_forest_gridsearch.pkl')

st.title("📱 Smartphone Price Predictor")

st.write("Enter the phone's specifications to predict its price:")

# Input features
ram = st.slider("RAM (GB)", 1, 16, 4)
storage = st.slider("Storage (GB)", 8, 512, 64)
battery = st.slider("Battery Capacity (mAh)", 1000, 7000, 4000)
fast_charging = st.selectbox("Has Fast Charging?", [0, 1])
nfc = st.selectbox("Has NFC?", [0, 1])
g5 = st.selectbox("Has 5G?", [0, 1])
cores = st.slider("Number of Cores", 1, 12, 4)
rear_cam = st.slider("Primary Rear Camera (MP)", 5, 200, 48)
num_rear = st.slider("Number of Rear Cameras", 1, 5, 2)
front_cam = st.slider("Primary Front Camera (MP)", 2, 64, 8)
display_size = st.slider("Display Size (inches)", 4.0, 7.5, 6.0)
display_type = st.selectbox("Display Type", ['LCD', 'AMOLED', 'OLED', 'TFT'])

# One-hot encode display type
display_types = ['LCD', 'AMOLED', 'OLED', 'TFT']
display_encoded = [1 if display_type == dtype else 0 for dtype in display_types]

# Combine all features
features = np.array([[ram, storage, battery, fast_charging, nfc, g5, cores,
                      rear_cam, num_rear, front_cam, display_size] + display_encoded])

# Predict
if st.button("Predict Price"):
    prediction = model.predict(features)[0]
    st.success(f"📱 Estimated Price: ${prediction:,.2f}")
