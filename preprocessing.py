import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_data(filepath=None):
    """Loads the dataset."""
    if filepath is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(base_dir, "dataset", "indian_liver_patient.csv")
    df = pd.read_csv(filepath)
    return df

def clean_and_preprocess(df, is_training=True, scaler=None):
    """
    Cleans and preprocesses the dataset.
    - Fills missing values with mean.
    - Encodes 'Gender': Male = 1, Female = 0.
    - Scales features if is_training=True or if scaler is provided.
    """
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()

    # 1. Label Encoding
    if 'Gender' in df.columns:
        # Handle string variations just in case
        df['Gender'] = df['Gender'].replace({'Male': 1, 'Female': 0, 'Male ': 1, 'Female ': 0})
        # If there are any NaN in Gender, fill with mode (1 for Male usually in this dataset)
        df['Gender'] = df['Gender'].fillna(1)
        # Ensure it's integer type
        df['Gender'] = df['Gender'].astype(int)

    # 2. Missing Values handling
    # Fill numeric columns with their mean
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # If training, we separate features and target, and fit scaler
    if is_training:
        X = df.drop('Dataset', axis=1)
        # The dataset target is 1 (Liver Disease) and 2 (Healthy). We map 2 to 0.
        y = df['Dataset'].apply(lambda x: 1 if x == 1 else 0)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
        
        return X_scaled_df, y, scaler
    else:
        # For inference (prediction), we use the loaded scaler
        if scaler is None:
            raise ValueError("Scaler must be provided for inference.")
        
        X_scaled = scaler.transform(df)
        X_scaled_df = pd.DataFrame(X_scaled, columns=df.columns)
        return X_scaled_df

if __name__ == "__main__":
    # Test the preprocessing module
    df = load_data()
    X, y, scaler = clean_and_preprocess(df, is_training=True)
    print("Preprocessing completed.")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
