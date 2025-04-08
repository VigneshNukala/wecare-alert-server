from fastapi import FastAPI
from pydantic import BaseModel
import firebase_admin
from firebase_admin import credentials, db
from fastapi import BackgroundTasks
from datetime import datetime
import joblib

# Init Firebase app
cred = credentials.Certificate("serviceAccountKey.json")  # your service account JSON
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://we-care-948eb-default-rtdb.asia-southeast1.firebasedatabase.app' # from Firebase console
})


def push_alert_to_firebase():
    ref = db.reference("/alerts")
    ref.push({
        "message": "🚨 Abnormal vitals detected",
        "timestamp": datetime.utcnow().isoformat(),
        "doctorPhone": "+91XXXXXXXXXX",
        "patientPhone": "+91XXXXXXXXXX"
    })


app = FastAPI()
model = joblib.load("alert_model.pkl")

class PatientReport(BaseModel):
    temperature: float
    spo2: float
    heart_rate: int

@app.post("/predict")
async def predict(report: PatientReport, background_tasks: BackgroundTasks):
    data = [[report.temperature, report.spo2, report.heart_rate]]
    result = model.predict(data)[0]

    if result == 1:
        background_tasks.add_task(push_alert_to_firebase)
    return {"status" : "received"}
