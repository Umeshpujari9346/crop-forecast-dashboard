# Databricks notebook source
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def main():
    st.title("Agricultural Analytics Dashboard")
    
    # Load precomputed results (replace with actual results from your models)
    results = {
        'yield_mae': 30787.26,
        'price_rmse': 93168.87,
        'crop_accuracy': 0.8544,
        'irrigation_accuracy': 0.9114,
        'seasonal_impact': pd.DataFrame({
            'season': [0, 1, 2],
            'yeilds': [20184.69, 24916.97, 21995.53],
            'price': [86171.28, 86684.50, 85456.53]
        }),
        'soil_impact': pd.DataFrame({
            'soil_type': range(28),
            'yeilds': [24181, 19359, 21629, 14881, 40383, 10979, 8631, 41245, 
                      13524, 9655, 12621, 18641, 2274, 20013, 17715, 16819, 
                      37665, 24452, 14609, 76959, 18161, 36890, 32161, 15705, 
                      16295, 60599, 20851, 14632]
        })
    }

    # Sidebar controls
    st.sidebar.header("Navigation")
    selected_page = st.sidebar.radio("Select Page", [
        "Model Performance",
        "Seasonal Analysis",
        "Soil Impact",
        "Price Forecasts"
    ])

    if selected_page == "Model Performance":
        st.header("Model Performance Metrics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Crop Yield MAE", f"{results['yield_mae']:,.2f}")
            st.metric("Crop Accuracy", f"{results['crop_accuracy']*100:.1f}%")
        
        with col2:
            st.metric("Price Prediction RMSE", f"{results['price_rmse']:,.2f}")
            st.metric("Irrigation Accuracy", f"{results['irrigation_accuracy']*100:.1f}%")

    elif selected_page == "Seasonal Analysis":
        st.header("Seasonal Impact Analysis")
        
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        
        results['seasonal_impact'].plot.bar(x='season', y='yeilds', ax=ax[0])
        ax[0].set_title("Yield by Season")
        ax[0].set_ylabel("Average Yield")
        
        results['seasonal_impact'].plot.bar(x='season', y='price', ax=ax[1])
        ax[1].set_title("Price by Season")
        ax[1].set_ylabel("Average Price")
        
        st.pyplot(fig)

    elif selected_page == "Soil Impact":
        st.header("Soil Type Impact Analysis")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        results['soil_impact'].plot.bar(x='soil_type', y='yeilds', ax=ax)
        ax.set_title("Yield by Soil Type")
        ax.set_xlabel("Soil Type")
        ax.set_ylabel("Average Yield")
        st.pyplot(fig)

    elif selected_page == "Price Forecasts":
        st.header("Price Forecasting")
        
        # Sample forecast data
        forecast_data = pd.DataFrame({
            'Year': [12, 13, 14, 15],
            'Forecast': [82269, 82094, 85579, 82164]
        })
        
        fig, ax = plt.subplots(figsize=(10, 6))
        forecast_data.plot.line(x='Year', y='Forecast', marker='o', ax=ax)
        ax.set_title("ARIMA Price Forecast")
        ax.set_ylabel("Price")
        st.pyplot(fig)

if __name__ == "__main__":
    main()