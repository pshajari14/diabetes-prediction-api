from flask import Flask, request, jsonify
import pickle
import numpy as np
import os  # 👈 to read environment variables

app = Flask(__name__)

# Load model and scaler using env variables
MODEL_PATH = os.getenv('MODEL_PATH', 'app/model.pkl')
SCALER_PATH = os.getenv('SCALER_PATH', 'app/scaler.pkl')

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return "Welcome to the Diabetes Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    return jsonify({'prediction': int(prediction[0])})
# Trigger rebuild to avoid pywin32 error


