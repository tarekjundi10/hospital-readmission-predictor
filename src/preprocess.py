import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os, joblib

def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)
    print(f"Original shape: {df.shape}")

    # Create binary target: High Risk (1) vs Low Risk (0)
    # Based on Stress Level + Sleep Hours (clinically validated proxies)
    df["target"] = (
        (df["Stress Level"].str.lower() == "high") |
        (df["Sleep Hours"] < 6)
    ).astype(int)
    print(f"Target distribution:\n{df['target'].value_counts()}")

    # Drop unused columns
    df = df.drop(columns=["Country", "Mental Health Condition", "Stress Level"])

    # Encode categoricals
    categorical_cols = ["Gender", "Exercise Level", "Diet Type"]
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # Features and target
    X = df.drop(columns=["target"])
    y = df["target"]

    # Scale
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Save scaler
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    joblib.dump(scaler, os.path.join(base_dir, "models", "scaler.pkl"))

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, ["Low Risk", "High Risk"]

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filepath = os.path.join(base_dir, "data", "raw", "Mental_Health_Lifestyle_Dataset.csv")
    load_and_preprocess(filepath)
    print("Preprocessing completed")