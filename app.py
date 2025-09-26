import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

# --- Page Configuration ---
st.set_page_config(
    page_title="Air Quality Index (AQI) Predictor",
    page_icon="🌬️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Load Model and Scaler ---
@st.cache_resource
def load_assets():
    """Loads the trained model and scaler from disk."""
    try:
        model = load_model('aqi_predictor_dl_model_station.h5')
        with open('scaler_station.pkl', 'rb') as file:
            scaler = pickle.load(file)
        return model, scaler
    except FileNotFoundError:
        st.error("Model or scaler file not found. Please ensure 'aqi_predictor_dl_model_station.h5' and 'scaler_station.pkl' are in the same directory.")
        return None, None
    except Exception as e:
        st.error(f"An error occurred while loading the model/scaler: {e}")
        return None, None

model, scaler = load_assets()

# --- AQI Category Helper Function ---
def get_aqi_category(aqi):
    """Returns the AQI category and associated color."""
    if aqi <= 50:
        return ("Good", "#4CAF50", "Minimal impact")
    elif aqi <= 100:
        return ("Satisfactory", "#8BC34A", "Minor breathing discomfort to sensitive people")
    elif aqi <= 200:
        return ("Moderate", "#FFEB3B", "Breathing discomfort to people with lung disease")
    elif aqi <= 300:
        return ("Poor", "#FF9800", "Breathing discomfort to most people on prolonged exposure")
    elif aqi <= 400:
        return ("Very Poor", "#F44336", "Respiratory illness on prolonged exposure")
    else:
        return ("Severe", "#B71C1C", "Affects healthy people and seriously impacts those with existing diseases")

# --- UI Elements ---
st.title("🌬️ Air Quality Index (AQI) Predictor")
st.markdown("Enter the values of key pollutants to predict the Air Quality Index (AQI). This model is trained on station-level data for higher accuracy.")

st.sidebar.header("Pollutant Levels")

# --- Input Fields ---
with st.sidebar:
    st.markdown("#### Enter the pollutant concentrations:")
    pm25 = st.number_input("PM2.5 (μg/m³)", min_value=0.0, value=55.5, step=0.1, format="%.1f")
    pm10 = st.number_input("PM10 (μg/m³)", min_value=0.0, value=150.2, step=0.1, format="%.1f")
    no2 = st.number_input("NO₂ (μg/m³)", min_value=0.0, value=30.8, step=0.1, format="%.1f")
    co = st.number_input("CO (mg/m³)", min_value=0.0, value=1.1, step=0.1, format="%.1f")
    so2 = st.number_input("SO₂ (μg/m³)", min_value=0.0, value=14.5, step=0.1, format="%.1f")
    o3 = st.number_input("Ozone (μg/m³)", min_value=0.0, value=35.9, step=0.1, format="%.1f")

# --- Prediction Logic ---
if st.button("Predict AQI", type="primary"):
    if model is not None and scaler is not None:
        # Prepare the input data for prediction
        features = np.array([[pm25, pm10, no2, co, so2, o3]])
        
        # Scale the features
        scaled_features = scaler.transform(features)
        
        # Make a prediction
        prediction = model.predict(scaled_features).flatten()[0]
        predicted_aqi = round(prediction)

        category, color, health_impact = get_aqi_category(predicted_aqi)

        # --- Display Results ---
        st.markdown("---")
        st.subheader("Prediction Result")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Predicted AQI Value", value=predicted_aqi)
        
        with col2:
            st.markdown(f"**Category:**")
            st.markdown(f"<p style='color:black; background-color:{color}; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold;'>{category}</p>", unsafe_allow_html=True)
            
        st.markdown(f"**Health Impact:** `{health_impact}`")

    else:
        st.warning("Model is not loaded. Cannot make a prediction.")

# --- AQI Information Expander ---
with st.expander("Learn More About AQI Categories"):
    st.image("https://i.imgur.com/8YftG2E.png", caption="AQI Categories as defined by the Central Pollution Control Board, India.")

