# Databricks notebook source
# Databricks notebook source
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import numpy as np
from keras.saving import register_keras_serializable

# ✅ Register MSE so it's recognized during model loading
@register_keras_serializable()
def mse(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

def load_lstm_model():
    try:
        model = tf.keras.models.load_model(
            "lstm_model.h5", 
            custom_objects={"mse": mse}  # Ensure 'mse' is recognized
        )
        return model
    except Exception as e:
        st.error(f"⚠️ Error loading LSTM model: {e}")
        return None

def predict_lstm(model, input_data):
    input_data = np.array(input_data).reshape(1, -1, 1)  # Reshape for LSTM input
    prediction = model.predict(input_data)
    return prediction[0][0]

def load_data():
    return {
        'yield_mae': 30787.26,
        'price_rmse': 93168.87,
        'crop_accuracy': 0.8544,
        'irrigation_accuracy': 0.9114,
        'seasonal_impact': pd.DataFrame({
            'Season': ['Winter', 'Summer', 'Monsoon'],
            'Yield': [20184.69, 24916.97, 21995.53],
            'Price': [86171.28, 86684.50, 85456.53]
        }),
        'soil_impact': pd.DataFrame({
            'Soil Type': [f'Type {i}' for i in range(1, 29)],
            'Yield': [24181, 19359, 21629, 14881, 40383, 10979, 8631, 41245, 
                      13524, 9655, 12621, 18641, 2274, 20013, 17715, 16819, 
                      37665, 24452, 14609, 76959, 18161, 36890, 32161, 15705, 
                      16295, 60599, 20851, 14632]
        })
    }

def main():
    st.set_page_config(page_title="Agricultural Analytics Dashboard", layout="wide")
    
    st.title("🌾 Agricultural Analytics Dashboard")
    st.write("An interactive dashboard for monitoring crop yield, price predictions, and soil impact analysis.")
    
    results = load_data()
    model = load_lstm_model()
    
    st.sidebar.header("Navigation")
    selected_page = st.sidebar.radio("Select Analysis Section", [
        "📊 Model Performance",
        "📅 Seasonal Analysis",
        "🌱 Soil Impact",
        "📈 Price Forecasts",
        "📂 Upload Data",
        "🤖 LSTM Predictions"
    ])

    # ✅ Model Performance Section
    if selected_page == "📊 Model Performance":
        st.header("📊 Model Performance Metrics")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Crop Yield MAE", f"{results['yield_mae']:,.2f}")
            st.metric("Crop Accuracy", f"{results['crop_accuracy']*100:.1f}%")
        
        with col2:
            st.metric("Price Prediction RMSE", f"{results['price_rmse']:,.2f}")
            st.metric("Irrigation Accuracy", f"{results['irrigation_accuracy']*100:.1f}%")

    # ✅ Seasonal Analysis Section
    elif selected_page == "📅 Seasonal Analysis":
        st.header("📅 Seasonal Impact Analysis")
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))

        sns.barplot(data=results['seasonal_impact'], x='Season', y='Yield', ax=ax[0], palette="Blues")
        ax[0].set_title("Crop Yield by Season")
        ax[0].set_ylabel("Yield Value")

        sns.barplot(data=results['seasonal_impact'], x='Season', y='Price', ax=ax[1], palette="Greens")
        ax[1].set_title("Price by Season")
        ax[1].set_ylabel("Average Price")

        st.pyplot(fig)

    # ✅ Soil Impact Section
    elif selected_page == "🌱 Soil Impact":
        st.header("🌱 Soil Type Impact Analysis")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(data=results['soil_impact'], x='Soil Type', y='Yield', palette="coolwarm")
        ax.set_title("Yield by Soil Type")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set_ylabel("Yield Value")
        st.pyplot(fig)

    # ✅ Price Forecasting Section
    elif selected_page == "📈 Price Forecasts":
        st.subheader("🔮 Crop Price Forecasting")
        forecast_data = pd.DataFrame({
            'Year': [12, 13, 14, 15],
            'Forecast Price': [82269, 82094, 85579, 82164]
        })
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=forecast_data, x='Year', y='Forecast Price', marker='o', linestyle='-', color='blue')
        ax.set_title("Future Price Forecast")
        ax.set_ylabel("Price in USD")
        st.pyplot(fig)
    
    # ✅ LSTM Predictions Section
    elif selected_page == "🤖 LSTM Predictions":
        st.subheader("🔍 Predict Crop Prices Using LSTM Model")
        user_input = st.number_input("Enter recent crop price data point:", value=85000, step=1000)
        if model:
            predicted_price = predict_lstm(model, [user_input])
            st.success(f"📈 Predicted Price: {predicted_price:.2f}")
        else:
            st.warning("⚠️ LSTM model not available. Please check the model file.")
    
    # ✅ Upload Data Section
    elif selected_page == "📂 Upload Data":
        st.subheader("📤 Upload Your Dataset")
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded_file is not None:
            user_data = pd.read_csv(uploaded_file)
            st.write("### Preview of Uploaded Data:")
            st.dataframe(user_data.head())
            st.write("#### Summary Statistics:")
            st.write(user_data.describe())

if __name__ == "__main__":
    main()
