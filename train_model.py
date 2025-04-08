import pandas as pd
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
import joblib # type: ignore

# Step 1: Load Data
df = pd.read_csv("patient_data.csv")

X = df[['temperature', 'spo2', 'heart_rate']]
y = df['label']

# Step 2: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Step 3: Train Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 4: Save Model
joblib.dump(model, "alert_model.pkl")
