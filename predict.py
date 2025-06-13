import pickle
import pandas as pd

# Load model
with open('app/model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load scaler
with open('app/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Define column names
feature_names = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

# Replace with actual input values
input_data = [[50, 1, 25.6, 80, 120, 85, 100, 95, 140, 70]]

# Create DataFrame
input_df = pd.DataFrame(input_data, columns=feature_names)

# Scale input
input_scaled = scaler.transform(input_df)

# Predict
prediction = model.predict(input_scaled)

# Result
print("🔮 Prediction:", prediction[0])




