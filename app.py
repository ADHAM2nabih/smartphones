import streamlit as st
import joblib
import numpy as np

# ⬇️ تحميل النموذج
model = joblib.load('random_forest_gridsearch.pkl')

# ⬇️ عنوان التطبيق
st.markdown("<h1 style='font-size: 42px;'>📱 Smartphone Price Predictor</h1>", unsafe_allow_html=True)
st.write("**أدخل مواصفات الموبايل وتوقع السعر بالدولار 💰**")

# ⬇️ مدخلات المستخدم
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

# ⬇️ اختيار نوع الشاشة
display_type = st.selectbox("Display Type", ['LCD', 'AMOLED', 'OLED', 'TFT'])

# ⬇️ One-hot encoding
display_types = ['LCD', 'AMOLED', 'OLED', 'TFT']
display_encoded = [1 if display_type == dtype else 0 for dtype in display_types]

# ⬇️ تجميع كل الخصائص معًا بنفس الترتيب اللي اتدرب عليه النموذج
features = np.array([[ram, storage, battery, fast_charging, nfc, g5, cores,
                      rear_cam, num_rear, front_cam, display_size] + display_encoded])

# ⬇️ زر التنبؤ
if st.button("🎯 Predict Price"):
    try:
        prediction = model.predict(features)[0]
        st.success(f"📱 السعر المتوقع: **${prediction:,.2f}**")
    except ValueError as e:
        st.error("❌ حدث خطأ في عدد الخصائص المدخلة. تأكد من مطابقتها لبيانات التدريب.")
