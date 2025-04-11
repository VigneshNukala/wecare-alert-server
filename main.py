import os
import resend
import httpx
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel, EmailStr
from datetime import datetime
import joblib
from typing import List
from statistics import mean, stdev
from dotenv import load_dotenv
load_dotenv()
from db import get_patient_reports

async def get_token(authorization: str = Header(None)) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    return authorization.replace("Bearer ", "")


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

# Function to fetch patient reports from the database
async def get_patient_profile(patient_id: str, token: str) -> dict:
    headers = {
        "Authorization": f"Bearer {token}",
    }
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"https://wecare-backend-us88.onrender.com/patient/profile",
            params={"patientId": patient_id},
            headers=headers
        )

        if response.status_code == 200:  
            return response.json()["data"]
        raise HTTPException(status_code=404, detail="Patient profile not found")

# Function to send alert email using Resend API
async def send_alert_email(recipient_details: dict, is_emergency_contact: bool = False):
    resend.api_key = os.environ["RESEND_API_KEY"]
    
    if is_emergency_contact:
        subject = f"Emergency Alert - {recipient_details['patient_name']}'s Health Status"
        html_content = f"""
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <h2>Emergency Health Alert</h2>
            <p>Dear {recipient_details['emergency_contact_name']},</p>
            <p>This is to inform you that {recipient_details['patient_name']}'s vital signs are showing abnormal readings.</p>
            <div style="padding: 20px; border: 1px solid #ddd; margin: 20px 0;">
                <h3>Current Vitals:</h3>
                <ul>
                    <li>Temperature: <strong>{recipient_details['temperature']}°C</strong></li>
                    <li>SPO2 Level: <strong>{recipient_details['spo2']}%</strong></li>
                    <li>Heart Rate: <strong>{recipient_details['heart_rate']} BPM</strong></li>
                </ul>
            </div>
            <p><strong>Important:</strong> Please contact them or their healthcare provider immediately.</p>
        </div>
        """
    else:
        subject = "Health Alert - Abnormal Vital Signs Detected"
        html_content = f"""
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <h2>Health Alert</h2>
            <p>Dear {recipient_details['patient_name']},</p>
            <p>Your vital signs are showing abnormal readings. Please take necessary precautions.</p>
            <div style="padding: 20px; border: 1px solid #ddd; margin: 20px 0;">
                <h3>Your Current Vitals:</h3>
                <ul>
                    <li>Temperature: <strong>{recipient_details['temperature']}°C</strong></li>
                    <li>SPO2 Level: <strong>{recipient_details['spo2']}%</strong></li>
                    <li>Heart Rate: <strong>{recipient_details['heart_rate']} BPM</strong></li>
                </ul>
            </div>
            <p><strong>Important:</strong> Please consult your healthcare provider.</p>
            <div style="text-align: center; margin: 30px 0;">
                <a href="{recipient_details['dashboard_link']}" 
                   style="background: #007bff; color: #fff; padding: 10px 20px; text-decoration: none; border-radius: 4px;">
                   View Your Dashboard
                </a>
            </div>
        </div>
        """

    params = {
        "from": "WeCare Alerts <alerts@freshroots.shop>",
        "to": [recipient_details['email']],
        "subject": subject,
        "html": html_content,
    }

    try:
        email = resend.Emails.send(params)
        return {"status": "sent", "email_id": email}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Function to check if the current vitals are abnormal for a specific patient
async def is_abnormal_for_patient(patient_id: str, current_vitals: List[float]) -> bool:
    historical_data = await get_patient_reports(patient_id)

    if not historical_data or len(historical_data) < 5:
        # Use ML model fallback
        features = {
            'temperature': [current_vitals[0]],
            'spo2': [current_vitals[1]],
            'heart_rate': [current_vitals[2]]
        }
        import pandas as pd
        X = pd.DataFrame(features)
        prediction = bool(model.predict(X)[0])
        return prediction
    
    # Process historical data
    temps = [report['temperature'] for report in historical_data]
    spo2s = [report['spo2'] for report in historical_data]
    heart_rates = [report['heart_rate'] for report in historical_data]
    
    # current_temp_c = (current_vitals[0] - 32) * 5 / 9
    # Calculate mean and standard deviation
    temp_mean, temp_std = mean(temps), stdev(temps)
    spo2_mean, spo2_std = mean(spo2s), stdev(spo2s)
    hr_mean, hr_std = mean(heart_rates), stdev(heart_rates)

    # Check abnormality safely
    temp_abnormal = abs(current_vitals[0] - temp_mean) > (2 * temp_std) if temp_std > 0 else current_vitals[0] != temp_mean
    spo2_abnormal = abs(current_vitals[1] - spo2_mean) > (2 * spo2_std) if spo2_std > 0 else current_vitals[1] != spo2_mean
    hr_abnormal = abs(current_vitals[2] - hr_mean) > (2 * hr_std) if hr_std > 0 else current_vitals[2] != hr_mean
    
    return temp_abnormal or spo2_abnormal or hr_abnormal

# Endpoint to receive patient reports and check for abnormalities
@app.post("/predict")
async def predict(report: PatientReport, token: str = Depends(get_token)):
    # Sanity checks for vitals
    if not (86 <= report.temperature <= 113 and 70 <= report.spo2 <= 100 and 40 <= report.heart_rate <= 180):
        raise HTTPException(status_code=400, detail="Invalid sensor readings")

    data = [report.temperature, report.spo2, report.heart_rate]
    
    # Check if abnormal for this specific patient
    is_abnormal = await is_abnormal_for_patient(report.patient_id, data)
    print(is_abnormal)
    
    if is_abnormal:
        patient_profile = await get_patient_profile(report.patient_id, token)

        patient_details = {
            "patient_name": patient_profile["name"],
            "email": patient_profile["email"],
            "temperature": report.temperature,
            "spo2": report.spo2,
            "heart_rate": report.heart_rate,
            "dashboard_link": f"https://wecare-health.com/patient/{report.patient_id}",
            "token": token
        }

        patient_email_result = await send_alert_email(patient_details)

        emergency_notifications = []
        for contact in patient_profile.get("emergencyContacts", []):
            emergency_details = {
                "emergency_contact_name": contact["name"],
                "email": contact["email"],
                "patient_name": patient_profile["name"],
                "temperature": report.temperature,
                "spo2": report.spo2,
                "heart_rate": report.heart_rate
            }
            emergency_result = await send_alert_email(emergency_details, is_emergency_contact=True)
            emergency_notifications.append(emergency_result)
        print(emergency_notifications)
        print(patient_profile.get("emergencyContacts", []))
        return {
            "status": "received",
            "is_abnormal": is_abnormal,
            "patient_notification": patient_email_result,
            "emergency_notifications": emergency_notifications
        }

    return {"status": "received", "is_abnormal": is_abnormal}
