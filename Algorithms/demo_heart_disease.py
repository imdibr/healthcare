
# Import our healthcare Adam optimizer
from healthcare_adam_optimizer import AdamOptimizer, main
import pandas as pd
import numpy as np

def quick_demo():
    """
    Quick demonstration of the heart disease prediction system
    """
    print("🏥 Heart Disease Prediction Demo")
    print("=" * 40)

    # Check if data file exists
    try:
        df = pd.read_csv('heart_disease_data.csv')
        print(f"✅ Dataset loaded: {df.shape[0]} patients, {df.shape[1]-1} features")

        # Show class distribution
        disease_count = df['heart_disease'].sum()
        healthy_count = len(df) - disease_count
        print(f"📊 Patients with heart disease: {disease_count}")
        print(f"📊 Healthy patients: {healthy_count}")

    except FileNotFoundError:
        print("❌ heart_disease_data.csv not found!")
        print("Please make sure the CSV file is in the same directory.")
        return

    print("\n🚀 Starting model training...")

    # Run the main training function
    try:
        model, scaler = main()
        print("\n🎉 Training completed successfully!")

        # Example prediction
        print("\n💡 Example: Predicting for a new patient...")
        # Create a sample patient data
        new_patient = np.array([[65, 1, 2, 140, 280, 0, 1, 150, 1, 2.5, 1, 1, 2]])
        new_patient_scaled = scaler.transform(new_patient)

        prediction = model.predict(new_patient_scaled)[0]
        risk_level = "HIGH RISK" if prediction > 0.5 else "LOW RISK"

        print(f"   Patient risk probability: {prediction:.3f}")
        print(f"   Risk level: {risk_level}")

    except Exception as e:
        print(f"❌ Error during training: {e}")

if __name__ == "__main__":
    quick_demo()
