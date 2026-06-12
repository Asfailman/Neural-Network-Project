import os
import urllib.request
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Constants
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Data")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports", "figures")
DATASET_PATH = os.path.join(DATA_DIR, "heart_disease.csv")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

def download_dataset():
    """Copies the heart_disease.csv dataset from the root folder if it does not already exist in the Data folder."""
    if not os.path.exists(DATASET_PATH):
        root_dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "heart_disease.csv")
        print(f"Dataset not found locally in Data folder. Checking root folder: {root_dataset_path}...")
        if os.path.exists(root_dataset_path):
            import shutil
            try:
                shutil.copy(root_dataset_path, DATASET_PATH)
                print(f"Dataset successfully copied to: {DATASET_PATH}")
            except Exception as e:
                print(f"Failed to copy dataset: {e}")
                raise e
        else:
            raise FileNotFoundError(
                f"Required heart_disease.csv not found in root {root_dataset_path} or in {DATASET_PATH}."
            )
    else:
        print(f"Dataset already exists locally at: {DATASET_PATH}")

def load_and_prepare_data(random_state=42):
    """
    Consolidates, cleans, transforms, and splits the heart disease dataset.
    Follows the CRISP-DM Data Preparation phase:
    1. Consolidate: Load the dataset (copy from root if necessary).
    2. Clean: Dataset is already complete (no missing values requiring imputation).
    3. Transform: Standard scale the features and partition into train/validation/test sets (70/15/15 ratio).
    """
    # 1. Data Consolidation
    download_dataset()
    df = pd.read_csv(DATASET_PATH)
    
    print("\n--- Data Understanding / Initial Info ---")
    print(f"Dataset Shape: {df.shape}")
    print("Columns:", list(df.columns))
    
    # 2. Data Splitting: 70% Train, 15% Validation, 15% Test
    X = df.drop(columns=["target"])
    y = df["target"]
    
    # First: Train (70%) and Temp (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=random_state, stratify=y
    )
    # Second: Split Temp into Validation (50% of 30% = 15%) and Test (50% of 30% = 15%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=random_state, stratify=y_temp
    )
    
    print("\n--- Data Splitting Summary ---")
    print(f"Training set:   X={X_train.shape}, y={y_train.shape} (Target 1 ratio: {y_train.mean():.4f})")
    print(f"Validation set: X={X_val.shape}, y={y_val.shape} (Target 1 ratio: {y_val.mean():.4f})")
    print(f"Test set:       X={X_test.shape}, y={y_test.shape} (Target 1 ratio: {y_test.mean():.4f})")
    
    # Feature Scaling: Standardize columns
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the fitted scaler for inference
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Fitted StandardScaler saved to: {scaler_path}")
    
    # Return as numpy arrays for Tensorflow compatibility
    return (
        X_train_scaled, y_train.values,
        X_val_scaled, y_val.values,
        X_test_scaled, y_test.values,
        scaler,
        X.columns.tolist()
    )

if __name__ == "__main__":
    # Test function
    load_and_prepare_data()

