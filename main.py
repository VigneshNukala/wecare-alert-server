from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import joblib
from typing import List
from statistics import mean, stdev
from db import get_patient_reports, save_report

app = FastAPI()

@app.get("/")
async def root():
    return "Wecare Alerts Server"
model = joblib.load("alert_model.pkl")

class PatientReport(BaseModel):
    patient_id: str
    temperature: float
    spo2: float
    heart_rate: int

async def is_abnormal_for_patient(patient_id: str, current_vitals: List[float]) -> bool:
    # Fetch patient's historical data from MongoDB
    historical_data = await get_patient_reports(patient_id)
    
    if not historical_data or len(historical_data) < 5:
        # If insufficient history, fall back to ML model
        return bool(model.predict([current_vitals])[0])
    
    # Process historical data
    temps = [report['temperature'] for report in historical_data]
    spo2s = [report['spo2'] for report in historical_data]
    heart_rates = [report['heart_rate'] for report in historical_data]
    
    # Calculate mean and standard deviation for each vital
    temp_mean, temp_std = mean(temps), stdev(temps)
    spo2_mean, spo2_std = mean(spo2s), stdev(spo2s)
    hr_mean, hr_std = mean(heart_rates), stdev(heart_rates)
    
    # Check if current vitals are more than 2 standard deviations away
    temp_abnormal = abs(current_vitals[0] - temp_mean) > (2 * temp_std)
    spo2_abnormal = abs(current_vitals[1] - spo2_mean) > (2 * spo2_std)
    hr_abnormal = abs(current_vitals[2] - hr_mean) > (2 * hr_std)
    
    return temp_abnormal or spo2_abnormal or hr_abnormal

@app.post("/predict")
async def predict(report: PatientReport):
    data = [report.temperature, report.spo2, report.heart_rate]
    
    # Store the current report
    report_data = report.dict()
    await save_report(report_data)
    
    # Check if abnormal for this specific patient
    is_abnormal = await is_abnormal_for_patient(report.patient_id, data)
    
    return {"status": "received", "is_abnormal": is_abnormal}
