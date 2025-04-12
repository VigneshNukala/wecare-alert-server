import asyncio
import pytest
import pandas as pd
from main import is_abnormal_for_patient, PatientReport
from db import save_report

# Add event loop fixture
@pytest.fixture(scope="session")
def event_loop():
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(autouse=True)
async def setup_db():
    # Setup before each test
    yield
    # Cleanup after each test if needed

async def setup_patient_history(patient_id: str, reports):
    # Save multiple reports for testing
    for report in reports:
        await save_report({
            "patient_id": patient_id,
            **report
        })

@pytest.mark.asyncio
async def test_extreme_temperature():
    patient_id = "test_temp_patient"
    
    # Setup normal history
    normal_reports = [
        {"temperature": 98.6, "spo2": 98, "heart_rate": 75} for _ in range(5)
    ]
    await setup_patient_history(patient_id, normal_reports)
    
    # Test extreme high temperature
    high_temp = pd.DataFrame([[104.5, 98, 75]], columns=["temperature", "spo2", "heart_rate"])
    assert await is_abnormal_for_patient(patient_id, high_temp.values[0]) == True
    
    # Test extreme low temperature
    low_temp = pd.DataFrame([[95.0, 98, 75]], columns=["temperature", "spo2", "heart_rate"])
    assert await is_abnormal_for_patient(patient_id, low_temp.values[0]) == True

@pytest.mark.asyncio
async def test_extreme_spo2():
    patient_id = "test_spo2_patient"
    
    # Setup normal history
    normal_reports = [
        {"temperature": 98.6, "spo2": 98, "heart_rate": 75} for _ in range(5)
    ]
    await setup_patient_history(patient_id, normal_reports)
    
    # Test extremely low SpO2
    low_spo2 = pd.DataFrame([[98.6, 70, 75]], columns=["temperature", "spo2", "heart_rate"])
    assert await is_abnormal_for_patient(patient_id, low_spo2.values[0]) == True
    
    # Test borderline SpO2
    borderline_spo2 = pd.DataFrame([[98.6, 94, 75]], columns=["temperature", "spo2", "heart_rate"])
    assert await is_abnormal_for_patient(patient_id, borderline_spo2.values[0]) == True

@pytest.mark.asyncio
async def test_extreme_heart_rate():
    patient_id = "test_hr_patient"
    
    # Setup normal history
    normal_reports = [
        {"temperature": 98.6, "spo2": 98, "heart_rate": 75} for _ in range(5)
    ]
    await setup_patient_history(patient_id, normal_reports)
    
    # Test extremely high heart rate
    high_hr = pd.DataFrame([[98.6, 98, 180]], columns=["temperature", "spo2", "heart_rate"])
    assert await is_abnormal_for_patient(patient_id, high_hr.values[0]) == True
    
    # Test extremely low heart rate
    low_hr = pd.DataFrame([[98.6, 98, 35]], columns=["temperature", "spo2", "heart_rate"])
    assert await is_abnormal_for_patient(patient_id, low_hr.values[0]) == True

@pytest.mark.asyncio
async def test_new_patient():
    patient_id = "new_patient"
    
    # Test with no history (should use ML model)
    vitals = pd.DataFrame([[98.6, 98, 75]], columns=["temperature", "spo2", "heart_rate"])
    result = await is_abnormal_for_patient(patient_id, vitals.values[0])
    assert isinstance(result, bool)

@pytest.mark.asyncio
async def test_all_vitals_abnormal():
    patient_id = "test_all_abnormal"
    
    # Setup normal history
    normal_reports = [
        {"temperature": 98.6, "spo2": 98, "heart_rate": 75} for _ in range(5)
    ]
    await setup_patient_history(patient_id, normal_reports)
    
    # Test all vitals abnormal
    extreme_vitals = pd.DataFrame([[104.0, 85, 150]], columns=["temperature", "spo2", "heart_rate"])
    assert await is_abnormal_for_patient(patient_id, extreme_vitals.values[0]) == True

@pytest.mark.asyncio
async def test_random_variations():
    patient_id = "random_test_patient"
    
    # Setup baseline normal history
    normal_reports = [
        {"temperature": 98.6, "spo2": 98, "heart_rate": 75} for _ in range(5)
    ]
    await setup_patient_history(patient_id, normal_reports)
    
    # Generate 1000 random variations
    import random
    
    test_cases = []
    for i in range(1000):
        if random.random() < 0.6:
            # Normal ranges
            temp = round(random.uniform(97.5, 99.5), 1)
            spo2 = random.randint(95, 100)
            hr = random.randint(60, 100)
        else:
            # Abnormal ranges
            temp = round(random.uniform(94.0, 105.0), 1)
            spo2 = random.randint(70, 94)
            hr = random.randint(30, 200)
        
        test_cases.append({
            "case_id": i + 1,
            "temperature": temp,
            "spo2": spo2,
            "heart_rate": hr
        })
    
    # Test each case and collect results
    results = []
    for case in test_cases:
        vitals = pd.DataFrame([[case["temperature"], case["spo2"], case["heart_rate"]]], 
                            columns=["temperature", "spo2", "heart_rate"])
        is_abnormal = await is_abnormal_for_patient(patient_id, vitals.values[0])
        
        results.append({
            **case,
            "is_abnormal": is_abnormal
        })
    
    # Convert results to DataFrame and save to Excel
    df = pd.DataFrame(results)
    excel_path = "test_results.xlsx"
    df.to_excel(excel_path, index=False)
    
    # Print summary
    print("\nTest Results Summary:")
    print("=" * 80)
    abnormal_count = sum(1 for r in results if r["is_abnormal"])
    print(f"Total cases: 1000")
    print(f"Abnormal cases: {abnormal_count}")
    print(f"Normal cases: {1000 - abnormal_count}")
    print(f"\nComplete results saved to: {excel_path}")