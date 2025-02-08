# Databricks notebook source
import tensorflow as tf
import numpy as np

# Load LSTM model safely
try:
    model = tf.keras.models.load_model("lstm_model.h5", compile=False)
    print("\n✅ Model loaded successfully!")
except Exception as e:
    print(f"\n❌ Error loading model: {e}")
    exit()

# Print model summary
print("\n📌 Model Summary:")
model.summary()

# Check model input and output shapes
input_shape = model.input_shape
output_shape = model.output_shape
print(f"\n🔹 Expected Input Shape: {input_shape}")
print(f"🔹 Expected Output Shape: {output_shape}")

# Generate a dummy input to check model prediction
try:
    timesteps = input_shape[1] if input_shape[1] else 10  # Use default 10 timesteps if None
    dummy_input = np.random.rand(1, timesteps, 1)  # Batch=1, Timesteps, Features=1
    prediction = model.predict(dummy_input)
    print(f"\n✅ Model Test Prediction Output: {prediction}")
except Exception as e:
    print(f"\n❌ Error during prediction: {e}")

# Suggestions for Fixing Warnings
print("\n🔧 **Fixing Possible Warnings:**")
print("- If you see 'Compiled the loaded model, but the compiled metrics have yet to be built',")
print("  make sure your model is compiled before saving using:")
print("  `model.compile(optimizer='adam', loss='mse', metrics=['mae'])` before saving the model.")
print("- If your input shape is incorrect, reshape it properly before making predictions.")
print("- To suppress oneDNN warnings, set: `os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'`")
