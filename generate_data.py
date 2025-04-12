import pandas as pd
import random

def generate_sample_data(num_samples=200):
    data = []

    for _ in range(num_samples):
        # Randomly decide if this sample is normal (0) or irregular (1)
        label = random.choices([0, 1], weights=[0.7, 0.3])[0]  # 70% normal, 30% irregular

        if label == 0:
            # Normal range
            temperature = round(random.uniform(97.5, 99.5), 1)
            spo2 = random.randint(95, 100)
            heart_rate = random.randint(60, 100)
        else:
            # Irregular range
            temperature = round(random.uniform(100, 104), 1)
            spo2 = random.randint(80, 94)
            heart_rate = random.randint(40, 130)

        data.append([temperature, spo2, heart_rate, label])

    return pd.DataFrame(data, columns=["temperature", "spo2", "heart_rate", "label"])

# Generate and save to CSV
df = generate_sample_data(300)
df.to_csv("patient_data.csv", index=False)
print("✅ patient_data.csv generated!")
