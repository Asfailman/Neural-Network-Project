"""
DATA PREPROCESSING MODULE
This script handles the ingestion, mapping, one-hot encoding, train-test splitting, 
and feature scaling of the Cardiac Risk Stratification dataset based on CRISP-DM methodology.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def get_data_filepaths():
    """
    Helper function to locate dataset files relative to the project structure.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    
    heart_disease_path = os.path.join(project_root, 'heart_disease.csv')
    
    # Check candidates for raw data file to support running from root or subfolder
    candidates = [
        os.path.join(project_root, 'Data', 'cardiac arrest dataset.csv'),
        os.path.join(project_root, '..', 'Data', 'cardiac arrest dataset.csv'),
        os.path.join(project_root, 'data', 'cardiac arrest dataset.csv'),
        os.path.join(project_root, '..', 'data', 'cardiac arrest dataset.csv')
    ]
    
    raw_dataset_path = candidates[0]
    for cand in candidates:
        if os.path.exists(cand):
            raw_dataset_path = cand
            break
    
    return heart_disease_path, raw_dataset_path

def create_heart_disease_csv():
    """
    Creates the required placeholder 'heart_disease.csv' from the raw dataset 
    by mapping it to UTeM's 12-column layout.
    """
    heart_disease_path, raw_dataset_path = get_data_filepaths()
    
    # If the file already exists in working directory or subproject directory, we are good
    cwd_path = os.path.join(os.getcwd(), 'heart_disease.csv')
    if os.path.exists(cwd_path):
        return cwd_path
    if os.path.exists(heart_disease_path):
        # Copy to CWD if we are running from elsewhere
        if cwd_path != heart_disease_path:
            pd.read_csv(heart_disease_path).to_csv(cwd_path, index=False)
        return heart_disease_path
        
    if not os.path.exists(raw_dataset_path):
        raise FileNotFoundError(
            f"Raw dataset not found at '{raw_dataset_path}'. "
            f"Please ensure 'Data/cardiac arrest dataset.csv' exists."
        )
        
    print(f"[CRISP-DM Phase 3: Data Preparation] Generating 'heart_disease.csv' from raw data...")
    df_raw = pd.read_csv(raw_dataset_path)
    
    # Column mapping according to UTeM specifications
    rename_mapping = {
        'cp': 'chest_pain_type',
        'trestbps': 'resting_bp',
        'chol': 'cholesterol',
        'fbs': 'fasting_blood_sugar',
        'restecg': 'resting_ecg',
        'thalach': 'max_heart_rate',
        'exang': 'exercise_angina',
        'slope': 'st_slope'
    }
    df_raw = df_raw.rename(columns=rename_mapping)
    
    # Selected 12 columns under study
    columns_to_keep = [
        'age', 'sex', 'chest_pain_type', 'resting_bp', 'cholesterol', 
        'fasting_blood_sugar', 'resting_ecg', 'max_heart_rate', 
        'exercise_angina', 'oldpeak', 'st_slope', 'target'
    ]
    df_raw = df_raw[columns_to_keep]
    
    # Scale chest_pain_type nominal values from 0-3 to 1-4 for compliance
    df_raw['chest_pain_type'] = df_raw['chest_pain_type'] + 1
    
    # Save file in subproject and cwd folder
    os.makedirs(os.path.dirname(heart_disease_path), exist_ok=True)
    df_raw.to_csv(heart_disease_path, index=False)
    df_raw.to_csv(cwd_path, index=False)
    
    print(f"-> Successfully created '{heart_disease_path}'")
    print(f"-> Successfully created copy in current directory: '{cwd_path}'")
    return heart_disease_path

def load_and_preprocess_data():
    """
    Loads 'heart_disease.csv', applies one-hot encoding for nominal variables,
    splits into 80/20 train/test sets, and scales the continuous columns.
    
    Returns:
        X_train_scaled (DataFrame): Scaled training features.
        X_test_scaled (DataFrame): Scaled test features.
        y_train (Series): Training targets.
        y_test (Series): Testing targets.
        feature_names (list): List of column names in the processed features.
    """
    # Step 1: Ingest the data
    heart_disease_path = create_heart_disease_csv()
    df = pd.read_csv(heart_disease_path)
    
    print("[CRISP-DM Phase 3: Data Preparation] Loading and preprocessing data...")
    print(f"Initial Dataset Shape: {df.shape}")
    
    # Step 2: One-hot encode nominal variables ('chest_pain_type', 'resting_ecg', 'st_slope')
    # Convert nominal variables to string type so pandas dummy encoder treats them as categories
    nominal_cols = ['chest_pain_type', 'resting_ecg', 'st_slope']
    for col in nominal_cols:
        df[col] = df[col].astype(str)
        
    df = pd.get_dummies(df, columns=nominal_cols, drop_first=False)
    
    # Separate features and labels
    X = df.drop(columns=['target'])
    y = df['target']
    
    # Convert dummy columns to numeric type (float32) for compatibility with TensorFlow
    dummy_cols = [col for col in X.columns if any(nom in col for nom in nominal_cols)]
    for col in dummy_cols:
        X[col] = X[col].astype('float32')
        
    # Step 3: Splitting data into 80% training and 20% testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Step 4: Feature scaling of continuous numerical columns
    continuous_cols = ['age', 'resting_bp', 'cholesterol', 'max_heart_rate', 'oldpeak']
    
    scaler = StandardScaler()
    
    # Prevent data leakage: fit scaler ONLY on train set, transform both train and test sets
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
    X_test_scaled[continuous_cols] = scaler.transform(X_test[continuous_cols])
    
    print(f"Preprocessed Training Set Shape: {X_train_scaled.shape}")
    print(f"Preprocessed Testing Set Shape: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, list(X.columns)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, features = load_and_preprocess_data()
    print("Preprocessed feature columns:")
    print(features)
