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
DATASET_PATH = os.path.join(DATA_DIR, "diabetes.csv")
DATASET_URL = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

def download_dataset():
    """Downloads the diabetes dataset if it does not already exist locally."""
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset not found locally. Downloading from: {DATASET_URL}...")
        try:
            urllib.request.urlretrieve(DATASET_URL, DATASET_PATH)
            print(f"Dataset downloaded successfully and saved to: {DATASET_PATH}")
        except Exception as e:
            print(f"Failed to download dataset: {e}")
            raise e
    else:
        print(f"Dataset already exists locally at: {DATASET_PATH}")

def load_and_prepare_data(random_state=42):
    """
    Consolidates, cleans, transforms, and splits the diabetes dataset.
    Follows the CRISP-DM Data Preparation phase:
    1. Consolidate: Load the dataset (download if necessary).
    2. Clean: Replace physically impossible zeros with NaN and impute them class-wise.
    3. Transform: Standard scale the features and partition into train/validation/test sets (70/15/15 ratio).
    """
    # 1. Data Consolidation
    download_dataset()
    df = pd.read_csv(DATASET_PATH)
    
    print("\n--- Data Understanding / Initial Info ---")
    print(f"Dataset Shape: {df.shape}")
    print("Columns:", list(df.columns))
    
    # 2. Data Cleaning (Handling zeros in features where zero is physically invalid)
    zero_fields = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    
    print("\nMissing/Zero values count per feature before imputation:")
    for field in zero_fields:
        zero_count = (df[field] == 0).sum()
        print(f" - {field}: {zero_count} zeros ({zero_count / len(df) * 100:.2f}%)")
        # Replace 0 with NaN
        df[field] = df[field].replace(0, np.nan)
        
    # Impute NaNs class-wise (based on Outcome) using the median of that class
    # This prevents data leakage and is biologically/physiologically sound
    for field in zero_fields:
        # Calculate median per outcome class
        medians = df.groupby("Outcome")[field].transform("median")
        df[field] = df[field].fillna(medians)
        
    print("\nCheck after imputation (NaN count):")
    print(df[zero_fields].isnull().sum())
    
    # 3. Data Transformation & Splitting
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]
    
    # Split: 70% Train, 15% Validation, 15% Test
    # First: Train (70%) and Temp (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=random_state, stratify=y
    )
    # Second: Split Temp into Validation (50% of 30% = 15%) and Test (50% of 30% = 15%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=random_state, stratify=y_temp
    )
    
    print("\n--- Data Splitting Summary ---")
    print(f"Training set:   X={X_train.shape}, y={y_train.shape} (Outcome 1 ratio: {y_train.mean():.4f})")
    print(f"Validation set: X={X_val.shape}, y={y_val.shape} (Outcome 1 ratio: {y_val.mean():.4f})")
    print(f"Test set:       X={X_test.shape}, y={y_test.shape} (Outcome 1 ratio: {y_test.mean():.4f})")
    
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
    
    # Convert back to DataFrame/Series for readability if needed, or return numpy arrays
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
