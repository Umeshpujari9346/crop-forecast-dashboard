# Databricks notebook source
import tensorflow as tf

# Load the existing model without compiling
model = tf.keras.models.load_model("lstm_model.h5", compile=False)

# Recompile with required settings
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# Save the recompiled model
model.save("lstm_model_fixed.h5")

print("\n✅ LSTM model recompiled and saved as 'lstm_model_fixed.h5'.")
