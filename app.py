import streamlit as st
import numpy as np
import joblib
import os
import requests

API_KEY = "1f8e5fb2fb0083baea9f23a7b0c6c4aa"
MODEL_FILE = "aqi_model.pkl"
SCALER_FILE = "scaler.pkl"

st.set_page_config(
    page_title="Zephyr AI",
    page_icon="🌤️",
    layout="centered"
)

@st.cache_resource
def load_assets():
    if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
        st.error("Model files not found. Run train_model.py first.")
        return None, None
    return joblib.load(MODEL_FILE), joblib.load(SCALER_FILE)

model, scaler = load_assets()

def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good", "🟢"
    elif aqi <= 100:
        return "Satisfactory", "🟡"
    elif aqi <= 200:
        return "Moderate / Unhealthy", "🟠"
    else:
        return "Poor", "🔴"

def get_live_weather(city):
    url = "http://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": API_KEY, "units": "metric"}

    try:
        r = requests.get(url, params=params)
        r.raise_for_status()
        d = r.json()

        return {
            "Temperature": d["main"]["temp"],
            "Humidity": d["main"]["humidity"],
            "Wind_Speed": d["wind"]["speed"] * 3.6,  # km/h
            "Rainfall": d.get("rain", {}).get("1h", 0)
        }
    except Exception as e:
        st.error(f"API Error: {e}")
        return None

st.title("Zephyr AI")
st.markdown("Predicting Air Quality using **Machine Learning (Random Forest)**")

if model:

    st.sidebar.header("Input Mode")
    mode = st.sidebar.radio("Select Mode", ["Live Weather", "Manual Input"])

    input_data = {}

    if mode == "Live Weather":
        cities = [
            "Delhi", "Mumbai", "Bangalore", "Chennai", "Kolkata",
            "Hyderabad", "Pune", "Ahmedabad", "Jaipur", "Bhubaneswar"
        ]
        city = st.selectbox("Select City", cities)

        if st.button("Fetch & Predict"):
            weather = get_live_weather(city)
            if weather:
                input_data = weather
                st.success("Weather data fetched")

    else:
        st.subheader("Manual Input")
        c1, c2 = st.columns(2)

        input_data["Temperature"] = c1.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
        input_data["Humidity"] = c2.number_input("Humidity (%)", 0.0, 100.0, 60.0)
        input_data["Wind_Speed"] = c1.number_input("Wind Speed (km/h)", 0.0, 30.0, 8.0)
        input_data["Rainfall"] = c2.number_input("Rainfall (mm)", 0.0, 50.0, 0.0)

    if input_data:

        features = np.array([[ 
            input_data["Temperature"],
            input_data["Humidity"],
            input_data["Wind_Speed"],
            input_data["Rainfall"]
        ]])

        features_scaled = scaler.transform(features)
        predicted_aqi = model.predict(features_scaled)[0]
        predicted_aqi = np.clip(predicted_aqi, 0, 500)

        category, icon = get_aqi_category(predicted_aqi)

        st.divider()
        st.subheader("Prediction Result")

        col1, col2 = st.columns(2)
        col1.metric("Predicted AQI", f"{predicted_aqi:.1f}")
        col2.markdown(f"## {icon} {category}")

        if category in ["Moderate / Unhealthy", "Poor"]:
            st.warning("⚠️ Avoid prolonged outdoor activities")
        else:
            st.success("✅ Air quality is acceptable")

        st.caption("⚠️ Educational ML model – not an official AQI measurement")
