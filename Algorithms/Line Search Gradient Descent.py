# Create a sample heart disease dataset
import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Create synthetic heart disease data based on real medical patterns
n_samples = 1000

data = {
    'age': np.random.randint(25, 80, n_samples),
    'sex': np.random.choice([0, 1], n_samples),  # 0: Female, 1: Male
    'chest_pain_type': np.random.choice([0, 1, 2, 3], n_samples),  # 0-3: Different types
    'resting_bp': np.random.randint(90, 200, n_samples),  # Blood pressure
    'cholesterol': np.random.randint(100, 400, n_samples),  # Cholesterol level
    'fasting_blood_sugar': np.random.choice([0, 1], n_samples),  # >120 mg/dl
    'resting_ecg': np.random.choice([0, 1, 2], n_samples),  # ECG results
    'max_heart_rate': np.random.randint(60, 220, n_samples),  # Maximum heart rate
    'exercise_angina': np.random.choice([0, 1], n_samples),  # Exercise induced angina
    'st_depression': np.random.uniform(0, 6, n_samples),  # ST depression
    'st_slope': np.random.choice([0, 1, 2], n_samples),  # Slope of peak exercise ST
    'num_vessels': np.random.choice([0, 1, 2, 3], n_samples),  # Number of vessels colored
    'thalassemia': np.random.choice([0, 1, 2, 3], n_samples)  # Thalassemia type
}

# Create target based on realistic medical correlations
# Higher risk factors increase probability of heart disease
risk_score = (
    (data['age'] - 40) * 0.02 +
    data['sex'] * 0.3 +
    data['chest_pain_type'] * 0.2 +
    (data['resting_bp'] - 120) * 0.01 +
    (data['cholesterol'] - 200) * 0.001 +
    data['fasting_blood_sugar'] * 0.2 +
    data['exercise_angina'] * 0.4 +
    data['st_depression'] * 0.3
)

# Convert to probability and create binary target
probability = 1 / (1 + np.exp(-risk_score))
data['heart_disease'] = (probability > np.random.uniform(0, 1, n_samples)).astype(int)

# Create DataFrame
df = pd.DataFrame(data)

# Display first few rows and basic info
print("Heart Disease Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nTarget Distribution:")
print(df['heart_disease'].value_counts())

print("\nDataset Info:")
print(df.info())

# Save to CSV
df.to_csv('heart_disease_data.csv', index=False)
print("\nDataset saved as 'heart_disease_data.csv'")