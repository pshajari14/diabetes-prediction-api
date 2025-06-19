# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle
import os

# 1. Load the dataset
df = pd.read_csv("diabetes.csv")  # <-- Make sure this file exists in the same folder

# 2. Split into features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler

# 5. Train model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 6. Create 'app' folder if it doesn't exist
os.makedirs("app", exist_ok=True)

# 7. Save model
with open("app/model.pkl", "wb") as f:
    pickle.dump(model, f)

# 8. Save scaler
with open("app/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Model and scaler saved successfully in 'app/' folder.")
