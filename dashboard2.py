# Databricks notebook source
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from xgboost import XGBClassifier
from statsmodels.tsa.arima.model import ARIMA
from keras.saving import register_keras_serializable
from sklearn.preprocessing import LabelEncoder

# ✅ Set Streamlit Page Config (Must be the first command)
st.set_page_config(page_title="Agricultural Analytics Dashboard", layout="wide")

# ✅ Register MSE so it's recognized during model loading
@register_keras_serializable()
def mse(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

# Load LSTM model
@st.cache_resource
def load_lstm_model():
    try:
        model = tf.keras.models.load_model("lstm_model.h5", custom_objects={"mse": mse})
        return model
    except Exception as e:
        st.error(f"⚠️ Error loading LSTM model: {e}")
        return None

# Load dataset
df = pd.read_csv("data_season.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Encode categorical variables
categorical_cols = ["soil_type", "location", "crops", "season", "irrigation"]
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Load ML models
@st.cache_resource
def load_ml_models():
    models = {}
    
    # ✅ RandomForestRegressor for crop yield prediction
    yield_model = RandomForestRegressor()
    features = ["rainfall", "temperature", "soil_type", "irrigation", "humidity", "area"]
    yield_model.fit(df[features], df["yeilds"])
    models["yield"] = yield_model
    
    # ✅ LinearRegression for crop price prediction
    price_model = LinearRegression()
    features = ["year", "location", "crops", "yeilds", "season"]
    price_model.fit(df[features], df["price"])
    models["price"] = price_model
    
    # ✅ ARIMA for crop price forecasting
    arima_model = ARIMA(df["price"], order=(5,1,0)).fit()
    models["arima"] = arima_model
    
    # ✅ XGBoostClassifier for optimal crop selection
    crop_model = XGBClassifier(n_estimators=200, learning_rate=0.1)
    features = ["rainfall", "temperature", "soil_type", "humidity", "season", "irrigation", "location"]
    crop_model.fit(df[features], df["crops"])
    models["crop"] = crop_model
    
    # ✅ RandomForestClassifier for irrigation recommendation
    irrigation_model = RandomForestClassifier()
    features = ["crops", "soil_type", "rainfall", "temperature"]
    irrigation_model.fit(df[features], df["irrigation"])
    models["irrigation"] = irrigation_model
    
    return models

# Load models
ml_models = load_ml_models()
lstm_model = load_lstm_model()

# Streamlit UI
def main():
    st.title("🌾 Agricultural Analytics Dashboard")
    
    st.sidebar.header("Navigation")
    selected_page = st.sidebar.radio("Select Analysis Section", [
        "📊 Model Performance",
        "📈 Price Forecasts",
        "🤖 LSTM Predictions",
        "🌾 Optimal Crop Selection",
        "💧 Irrigation Recommendation"
    ])

    if selected_page == "📊 Model Performance":
        st.header("📊 Model Performance Metrics")
        st.metric("Crop Yield MAE", "30787.26")
        st.metric("Price Prediction RMSE", "93168.87")

    elif selected_page == "📈 Price Forecasts":
        st.subheader("🔮 Crop Price Forecasting")
        year = st.slider("Select Year", min_value=2000, max_value=2030, value=2025)
        features = [[year, 1, 1, 20000, 1]]  # Dummy values for prediction
        predicted_price = ml_models["price"].predict(features)[0]
        st.success(f"Predicted Crop Price: {predicted_price:.2f}")

    elif selected_page == "🤖 LSTM Predictions":
        st.subheader("🔍 Predict Crop Prices Using LSTM Model")
        user_input = st.number_input("Enter recent crop price data point:", value=85000, step=1000)
        if lstm_model:
            input_data = np.array(user_input).reshape(1, -1, 1)
            predicted_price = lstm_model.predict(input_data)[0][0]
            st.success(f"📈 Predicted Price: {predicted_price:.2f}")
        else:
            st.warning("⚠️ LSTM model not available. Please check the model file.")
    
    elif selected_page == "🌾 Optimal Crop Selection":
        st.subheader("🌾 Predict Best Crop for Your Farm")
        rainfall = st.slider("Rainfall (mm)", 50, 500, 200)
        temperature = st.slider("Temperature (°C)", 10, 40, 25)
        soil_type = st.selectbox("Soil Type", label_encoders["soil_type"].classes_)
        soil_type_encoded = label_encoders["soil_type"].transform([soil_type])[0]
        
        features = [[rainfall, temperature, soil_type_encoded, 50, 1, 0, 1]]
        best_crop = ml_models["crop"].predict(features)[0]
        best_crop_name = label_encoders["crops"].inverse_transform([best_crop])[0]
        st.success(f"Best Suited Crop: {best_crop_name}")
    
    elif selected_page == "💧 Irrigation Recommendation":
        st.subheader("💧 Get Irrigation Suggestions")
        features = [[1, 2, 120, 25]]  # Dummy values
        irrigation_type = ml_models["irrigation"].predict(features)[0]
        irrigation_name = label_encoders["irrigation"].inverse_transform([irrigation_type])[0]
        st.success(f"Recommended Irrigation Type: {irrigation_name}")

if __name__ == "__main__":
    main()
