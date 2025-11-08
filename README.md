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
(samples, timesteps, features)

But AQI datasets are **tabular (2D)**:
(samples, features)

### 💡 Novel Solution Proposed
Convert tabular data into single-timestep sequences:
(n_samples, 6_features) → (n_samples, 1_timestep, 6_features)

This allows sequence-based learning on static tabular data.

---

## 📊 Model Performance

| Model      | R² Score  | Mean Absolute Error (MAE) |
|------------|-----------|---------------------------|
| **GRU**    | 0.9995    | 2.23                      |
| **LSTM**   | 0.9994    | 2.31                      |
| SimpleRNN  | 0.9850    | 9.81                      |

➡️ **Conclusion:** GRU performed the best, achieving near-perfect accuracy.

---

## 🚀 Running the Streamlit Application

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Kenche-Koushik/your-repo-name.git
cd your-repo-name
```

### 2️⃣ Create & Activate Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the App
```bash
streamlit run app.py
```

---

## 👨‍💻 Author

Kenche Koushik

Machine Learning | Deep Learning | Data Analysis

Feel free to star ⭐ the repo, contribute, or open issues.
