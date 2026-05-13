import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import preprocessing

def train_and_evaluate():
    print("Loading data...")
    df = preprocessing.load_data("dataset/indian_liver_patient.csv")
    
    print("Preprocessing data...")
    X, y, scaler = preprocessing.clean_and_preprocess(df, is_training=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training Random Forest model...")
    # Initialize Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Saving model and scaler...")
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    joblib.dump(rf_model, "models/liver_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    # Using simple dictionary encoding for gender in prediction.py so we don't strictly need encoder.pkl,
    # but we can save a dummy dict if needed.
    
    print("Model saved successfully in 'models/' directory.")

if __name__ == "__main__":
    train_and_evaluate()
