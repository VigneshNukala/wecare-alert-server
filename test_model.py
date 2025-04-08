import joblib

# Load the trained model
model = joblib.load("alert_model.pkl")

# Example test input: temperature, spo2, heart_rate
sample_input = [[101.5, 88, 120]]  # Likely abnormal

# Predict
prediction = model.predict(sample_input)[0]

# Output result
if prediction == 1:
    print("⚠️ Irregularity Detected")
else:
    print("✅ Vitals are Normal")
