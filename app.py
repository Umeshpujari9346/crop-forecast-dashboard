# Databricks notebook source
from flask import Flask, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load trained LSTM model
model = load_model("lstm_model.h5")

# Dummy input data (Replace this with real data)
scaler = MinMaxScaler(feature_range=(0, 1))
price_series = np.array([100, 120, 150, 170, 200, 230, 250]).reshape(-1, 1)
scaler.fit(price_series)

def predict_price():
    latest_data = price_series[-3:].reshape(1, 3, 1)  # Using last 3 points
    scaled_prediction = model.predict(latest_data)
    return scaler.inverse_transform(scaled_prediction)[0][0]

@app.route("/predict", methods=["GET"])
def predict():
    predicted_price = predict_price()  # Call the function to get the price
    return jsonify({"predicted_price": float(predicted_price)})

if __name__ == "__main__":
    app.run(debug=True)
