# AQI Prediction & RNN Model Comparison

This repository contains a dual-purpose system focused on evaluating air quality and predicting Air Quality Index (AQI) values using Deep Learning models. The work consists of:

1. **A Streamlit-based AQI Monitoring and Prediction Dashboard**
2. **A Research Study Comparing RNN-based Deep Learning Models (SimpleRNN, LSTM, GRU)**

The core innovation of this project lies in the **application of Recurrent Neural Networks to tabular AQI datasets**, using a novel sequence reshaping method validated through experimentation.

---

## 🌐 1. Streamlit Dashboard

The application (`app.py`) provides an interactive interface to:

### ✅ Features
| Feature | Description |
|--------|-------------|
| **Live AQI Monitoring** | Fetches real-time air pollution data via OpenWeatherMap API |
| **Manual AQI Prediction** | Predict AQI using a trained TensorFlow model and pollutant inputs |
| **Dual Mode UI** | Separate tabs for live monitoring & manual prediction |
| **Data Visualization** | Interprets AQI levels and categorizes air quality (Good, Moderate, etc.) |

### 🧠 Model Used in App
- Trained Keras model (`aqi_predictor_dl_model_station.h5`)
- Preprocessing StandardScaler (`scaler_station.pkl`)

---

## 🔬 2. Research on RNNs for Tabular AQI Prediction

This research compares three Recurrent Neural Network architectures for AQI regression:

- **SimpleRNN**
- **LSTM**
- **GRU**

### 🎯 Problem Addressed
RNNs traditionally require **sequence-formatted (3D) data**:
