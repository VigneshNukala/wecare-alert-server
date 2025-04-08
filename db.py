from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime, UTC

client = AsyncIOMotorClient("mongodb+srv://gsnr1925:5vaZiEW5Rn433knp@wecare1.ibh32ew.mongodb.net/?retryWrites=true&w=majority&appName=wecare1")
db = client.wecare_db

async def get_patient_reports(patient_id: str):
    cursor = db.reports.find({"patient_id": patient_id})
    return await cursor.to_list(length=None)

async def save_report(report_data: dict):
    report_data["timestamp"] = datetime.now(UTC)
    await db.reports.insert_one(report_data)

async def save_alert(alert_data: dict):
    await db.alerts.insert_one(alert_data)