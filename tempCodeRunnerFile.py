import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from keras.saving import register_keras_serializable
from sklearn.preprocessing import LabelEncoder

# ✅ Suppress TensorFlow oneDNN warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ✅ Set Streamlit Page Config
st.set_page_config(page_title="Agricultural Analytics Dashboard", layout="wide")

# ✅ Register MSE so it's recognized during model loading
@register_keras_serializable()
def mse(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

# Load LSTM model
@st.cache_resource
def load_lstm_model():
    try:
        model = tf.keras.models.load_model("lstm_model_fixed.h5", custom_objects={"mse": mse}, compile=False)
        return model
    except Exception as e:
        st.error(f"⚠️ Error loading LSTM model: {e}")
        return None

# Load dataset
df = pd.read_csv("data_season.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# ✅ Fix for unseen labels (Handle missing values before encoding)
categorical_cols = ["soil_type", "location", "crops", "season", "irrigation"]
df[categorical_cols] = df[categorical_cols].fillna("Unknown")  # Replace NaN with "Unknown"

# Encode categorical variables safely
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))  # Convert to string and encode
    label_encoders[col] = le

# Load ML models
@st.cache_resource
def load_ml_models():
    models = {}
    
    # ✅ Improved Crop Yield Prediction (XGBoostRegressor)
    yield_model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=7)
    features = ["rainfall", "temperature", "soil_type", "irrigation", "humidity", "area"]
    yield_model.fit(df[features], df["yeilds"])
    models["yield"] = yield_model
    
    # ✅ Improved Crop Price Prediction (XGBoostRegressor)
    price_model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6)
    features = ["year", "location", "crops", "yeilds", "season"]
    price_model.fit(df[features], df["price"])
    models["price"] = price_model
    
    # ✅ Optimized ARIMA Model for Time-Series Price Forecasting
    arima_model = auto_arima(df["price"], seasonal=True, stepwise=True, suppress_warnings=True)
    models["arima"] = arima_model
    
    # ✅ XGBoostClassifier for Optimal Crop Selection
    crop_model = XGBClassifier(n_estimators=200, learning_rate=0.1)
    features = ["rainfall", "temperature", "soil_type", "humidity", "season", "irrigation", "location"]
    crop_model.fit(df[features], df["crops"])
    models["crop"] = crop_model
    
    # ✅ RandomForestClassifier for Irrigation Recommendation
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



        # Compute model performance metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    # Prepare test data (Use last 20% for evaluation)
        test_size = int(0.2 * len(df))
        df_train, df_test = df[:-test_size], df[-test_size:]

    # ✅ Crop Yield Prediction Performance
        y_true_yield = df_test["yeilds"]
        y_pred_yield = ml_models["yield"].predict(df_test[["rainfall", "temperature", "soil_type", "irrigation", "humidity", "area"]])
        mae_yield = mean_absolute_error(y_true_yield, y_pred_yield)

    # ✅ Crop Price Prediction Performance
        y_true_price = df_test["price"]
        y_pred_price = ml_models["price"].predict(df_test[["year", "location", "crops", "yeilds", "season"]])
        rmse_price = mean_squared_error(y_true_price, y_pred_price, squared=False)
        r2_price = r2_score(y_true_price, y_pred_price)

    # ✅ Display Metrics in an Engaging Format
        col1, col2, col3 = st.columns(3)
    
        with col1:
            st.metric("📉 Crop Yield MAE", f"{mae_yield:.2f}", "Lower is better")
    
        with col2:
            st.metric("💰 Price Prediction RMSE", f"{rmse_price:.2f}", "Lower is better")

        with col3:
            st.metric("📊 R² Score (Price Model)", f"{r2_price:.2f}", "Closer to 1 is better")

    # ✅ Additional Performance Insights
        st.success(f"🚀 Crop Yield Model Error: ~{mae_yield:.0f} units per prediction")
        st.success(f"📈 Price Model Accuracy (R² Score): {r2_price:.2f} (1 = Perfect Fit)")
            

    elif selected_page == "📈 Price Forecasts":
        st.subheader("🔮 Crop Price Forecasting")
        year = st.slider("Select Year", min_value=2000, max_value=2030, value=2025)
        features = [[year, 1, 1, 20000, 1]]  # Dummy values for prediction
        predicted_price = ml_models["price"].predict(features)[0]
        st.success(f"Predicted Crop Price: {predicted_price:.2f}")

    # elif selected_page == "🤖 LSTM Predictions":
    #     st.subheader("🔍 Predict Crop Prices Using LSTM Model")
    #     user_input = st.text_area("Enter 5 recent crop price data points (comma-separated):", "85000,86000,87000,88000,89000")

    #     # Convert input into a list of 5 values
    #     try:
    #         input_list = [float(x.strip()) for x in user_input.split(",")]
    #         if len(input_list) != 5:
    #             st.error("⚠️ Please enter exactly 5 values.")
    #         else:
    #             input_data = np.array(input_list).reshape(1, 5, 1)  # Correct reshaping for LSTM
    #             predicted_price = lstm_model.predict(input_data)[0][0]
    #             st.success(f"📈 Predicted Price: {predicted_price:.2f}")
    #     except ValueError:
    #         st.error("⚠️ Please enter valid numerical values.")
    elif selected_page == "🤖 LSTM Predictions":
        st.subheader("🔍 Predict Crop Prices Using LSTM Model (Improved)")

        # Take user inputs for meaningful features
        selected_crop = st.selectbox("Select Crop", label_encoders["crops"].classes_)
        rainfall = st.slider("Rainfall (mm)", min_value=0, max_value=500, value=200)
        temperature = st.slider("Temperature (°C)", min_value=5, max_value=50, value=25)
        soil_type = st.selectbox("Select Soil Type", label_encoders["soil_type"].classes_)
        irrigation_type = st.selectbox("Select Irrigation Type", label_encoders["irrigation"].classes_)
        market_demand = st.slider("Market Demand (1 = Low, 10 = High)", min_value=1, max_value=10, value=5)

        # Encode categorical inputs
        selected_crop_encoded = label_encoders["crops"].transform([selected_crop])[0]
        soil_type_encoded = label_encoders["soil_type"].transform([soil_type])[0]
        irrigation_encoded = label_encoders["irrigation"].transform([irrigation_type])[0]

        # Prepare input array for LSTM model
        input_features = np.array([[rainfall, temperature, soil_type_encoded, irrigation_encoded, market_demand]])
        input_features = input_features.reshape(1, 5, 1)  # Ensure correct shape for LSTM

        # Make prediction with LSTM
        if lstm_model:
            predicted_price = lstm_model.predict(input_features)[0][0]
            st.success(f"📈 Predicted Crop Price: ₹{predicted_price:.2f}")
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
        st.subheader("💧 Get Irrigation Suggestions Based on Your Farm Conditions")

        # Take user inputs
        selected_crop = st.selectbox("Select Crop", label_encoders["crops"].classes_)
        soil_type = st.selectbox("Select Soil Type", label_encoders["soil_type"].classes_)
        rainfall = st.slider("Rainfall (mm)", min_value=0, max_value=500, value=200)
        temperature = st.slider("Temperature (°C)", min_value=5, max_value=50, value=25)

        # Encode categorical inputs
        selected_crop_encoded = label_encoders["crops"].transform([selected_crop])[0]
        soil_type_encoded = label_encoders["soil_type"].transform([soil_type])[0]

        # Prepare input features for model
        features = [[selected_crop_encoded, soil_type_encoded, rainfall, temperature]]

        # Predict best irrigation method
        irrigation_type_encoded = ml_models["irrigation"].predict(features)[0]
        irrigation_name = label_encoders["irrigation"].inverse_transform([irrigation_type_encoded])[0]

        # Display recommendation
        st.success(f"✅ Best Recommended Irrigation Type: {irrigation_name}")

if __name__ == "__main__":
    main()
