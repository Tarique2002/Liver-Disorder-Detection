import joblib
import pandas as pd
import numpy as np
import os

# Define the feature columns exactly as expected by the model
FEATURE_COLS = [
    'Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin',
    'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
    'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin',
    'Albumin_and_Globulin_Ratio'
]

def load_models():
    """Loads the trained Random Forest model and the scaler."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", "liver_model.pkl")
    scaler_path = os.path.join(base_dir, "models", "scaler.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def predict_liver_disease(input_data, model, scaler):
    """
    Takes raw input data (dict), preprocesses it, and returns prediction.
    """
    # 1. Convert input data to DataFrame
    df = pd.DataFrame([input_data])
    
    # 2. Reorder columns to match training
    df = df[FEATURE_COLS]
    
    # 3. Label Encode Gender
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0, 'Male ': 1, 'Female ': 0, 1: 1, 0: 0})
    df['Gender'] = df['Gender'].fillna(1).astype(int)
    
    # 4. Handle any missing values in input (just in case)
    df = df.fillna(0)
    
    # 5. Scale features
    X_scaled = scaler.transform(df)
    
    # 6. Predict
    prediction = model.predict(X_scaled)[0] # 1 for disease, 0 for healthy
    probabilities = model.predict_proba(X_scaled)[0]
    
    # Probability of getting the disease (class 1)
    risk_prob = probabilities[1] if len(probabilities) > 1 else (1.0 if prediction == 1 else 0.0)
    
    result = {
        'prediction': 'High Risk of Liver Disease' if prediction == 1 else 'Low Risk (Healthy)',
        'risk_percentage': round(risk_prob * 100, 2),
        'confidence_score': round(max(probabilities) * 100, 2),
        'is_high_risk': bool(prediction == 1)
    }
    
    return result

def generate_recommendations(is_high_risk):
    if is_high_risk:
        return [
            "Consult a hepatologist or gastroenterologist immediately.",
            "Avoid alcohol consumption entirely.",
            "Maintain a healthy, balanced diet low in saturated fats.",
            "Avoid self-medicating; some over-the-counter drugs can harm the liver.",
            "Regularly monitor your liver function through blood tests."
        ]
    else:
        return [
            "Continue maintaining a healthy lifestyle.",
            "Eat a balanced diet rich in fruits, vegetables, and whole grains.",
            "Exercise regularly.",
            "Limit alcohol intake.",
            "Stay hydrated."
        ]
