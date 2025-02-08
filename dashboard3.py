# Databricks notebook source
import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import plotly.express as px
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load dataset
data_path = "data_season.csv"
df = pd.read_csv(data_path)

# Encode categorical variables
label_encoders = {}
categorical_cols = ['Location', 'Soil type', 'Irrigation', 'Crops', 'Season']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Load pre-trained models (to be trained separately)
# models = {"LinearRegression": joblib.load("linear_model.pkl"), ...}
models = {}

# Initialize Dash app
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("Crop Yield and Price Prediction Dashboard"),
    
    # Input fields
    html.Label("Area"),
    dcc.Input(id='area', type='number', value=1000, step=100),
    
    html.Label("Rainfall"),
    dcc.Input(id='rainfall', type='number', value=1000.0, step=0.1),
    
    html.Label("Temperature"),
    dcc.Input(id='temperature', type='number', value=25.0, step=0.1),
    
    html.Label("Humidity"),
    dcc.Input(id='humidity', type='number', value=50.0, step=0.1),
    
    html.Label("Soil Type"),
    dcc.Dropdown(
        id='soil-type',
        options=[{'label': val, 'value': idx} for idx, val in enumerate(label_encoders['Soil type'].classes_)],
        value=0
    ),
    
    html.Label("Irrigation"),
    dcc.Dropdown(
        id='irrigation',
        options=[{'label': val, 'value': idx} for idx, val in enumerate(label_encoders['Irrigation'].classes_)],
        value=0
    ),
    
    html.Label("Crops"),
    dcc.Dropdown(
        id='crops',
        options=[{'label': val, 'value': idx} for idx, val in enumerate(label_encoders['Crops'].classes_)],
        value=0
    ),
    
    html.Label("Season"),
    dcc.Dropdown(
        id='season',
        options=[{'label': val, 'value': idx} for idx, val in enumerate(label_encoders['Season'].classes_)],
        value=0
    ),
    
    html.Button('Predict', id='predict-button', n_clicks=0),
    
    html.H3("Predicted Yield: "),
    html.Div(id='yield-output'),
    
    html.H3("Predicted Price: "),
    html.Div(id='price-output')
])

# Define callback for prediction
@app.callback(
    [Output('yield-output', 'children'), Output('price-output', 'children')],
    [Input('predict-button', 'n_clicks')],
    [State('area', 'value'), State('rainfall', 'value'), State('temperature', 'value'),
     State('humidity', 'value'), State('soil-type', 'value'), State('irrigation', 'value'),
     State('crops', 'value'), State('season', 'value')]
)
def predict(n_clicks, area, rainfall, temperature, humidity, soil, irrigation, crops, season):
    if n_clicks == 0:
        return "Waiting for input...", "Waiting for input..."
    
    features = np.array([[area, rainfall, temperature, humidity, soil, irrigation, crops, season]])
    
    # Predict using multiple models (mock predictions for now)
    yield_pred = "TBD"
    price_pred = "TBD"
    
    return f"{yield_pred} kg", f"₹{price_pred}"

if __name__ == '__main__':
    app.run_server(debug=True)
