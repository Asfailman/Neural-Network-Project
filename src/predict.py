import os
import pickle
import numpy as np
import tensorflow as tf

# Constants
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "cardiac_mlp_model.keras")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")

def predict_single_patient(features, model, scaler):
    """
    Predicts the diabetes risk for a single patient's features.
    
    Parameters:
    - features (list/array): 8 clinical metrics matching Pima dataset columns.
    - model (tf.keras.Model): Loaded MLP model.
    - scaler (StandardScaler): Loaded fitted scaler.
    
    Returns:
    - probability (float): Predicted probability of being diabetic.
    - prediction (int): 0 (Low Risk) or 1 (High Risk) based on 0.5 threshold.
    """
    # Reshape features to 2D array: (1, 8)
    features_arr = np.array(features).reshape(1, -1)
    
    # Scale features
    features_scaled = scaler.transform(features_arr)
    
    # Predict probability
    prob = model.predict(features_scaled, verbose=0)[0][0]
    pred = 1 if prob >= 0.5 else 0
    
    return prob, pred

def load_resources():
    """Loads the saved Keras model and StandardScaler."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            "Model resources not found. Please train the model first by running main.py."
        )
        
    # Load model
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Load scaler
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
        
    return model, scaler

def get_float_input(prompt, min_val=0.0, max_val=1000.0, default=None):
    """Safely retrieves a float input from the user with range checks."""
    while True:
        try:
            val_str = input(prompt).strip()
            if not val_str and default is not None:
                return default
            val = float(val_str)
            if min_val <= val <= max_val:
                return val
            else:
                print(f"Error: Value must be between {min_val} and {max_val}.")
        except ValueError:
            print("Error: Invalid number format. Please try again.")

def run_interactive_cli():
    """Runs a terminal-based interactive CLI loop for custom risk prediction."""
    print("\n" + "="*50)
    print("     CARDIAC RISK STRATIFICATION - MLP INFERENCE CLI    ")
    print("="*50)
    
    try:
        model, scaler = load_resources()
        print("Successfully loaded model weights and StandardScaler scaler config.")
    except Exception as e:
        print(f"Error: Could not load trained resources. {e}")
        return
        
    features_info = [
        ("Age", "Age (years)", 1, 120, 54),
        ("Sex", "Sex (1 = Male, 0 = Female)", 0, 1, 1),
        ("Chest Pain Type", "Chest pain type [0: typical, 1: atypical, 2: non-anginal, 3: asymptomatic]", 0, 3, 1),
        ("Resting BP", "Resting blood pressure [mm Hg]", 50, 250, 130),
        ("Cholesterol", "Serum cholesterol [mg/dL]", 80, 600, 240),
        ("Fasting Blood Sugar", "Fasting blood sugar > 120 mg/dL (1 = True, 0 = False)", 0, 1, 0),
        ("Resting ECG", "Resting electrocardiographic results [0, 1, 2]", 0, 2, 1),
        ("Max Heart Rate", "Maximum heart rate achieved", 50, 220, 150),
        ("Exercise Angina", "Exercise induced angina (1 = Yes, 0 = No)", 0, 1, 0),
        ("Oldpeak", "ST depression induced by exercise relative to rest", 0.0, 10.0, 1.0),
        ("ST Slope", "Slope of peak exercise ST segment [0, 1, 2]", 0, 2, 1)
    ]
    
    while True:
        print("\nEnter patient diagnostic metrics (or type 'q' to quit):")
        patient_inputs = []
        quit_flag = False
        
        for name, desc, min_v, max_v, default in features_info:
            prompt = f" - {name} ({desc}) [default: {default}]: "
            user_input = input(prompt).strip()
            if user_input.lower() == 'q':
                quit_flag = True
                break
                
            if not user_input:
                patient_inputs.append(float(default))
            else:
                try:
                    val = float(user_input)
                    if min_v <= val <= max_v:
                        patient_inputs.append(val)
                    else:
                        print(f"  Warning: Value out of realistic range [{min_v}-{max_v}]. Using default: {default}")
                        patient_inputs.append(float(default))
                except ValueError:
                    print(f"  Warning: Invalid input. Using default: {default}")
                    patient_inputs.append(float(default))
                    
        if quit_flag:
            break
            
        print("\nInput parameters summarized:")
        for (name, _, _, _, _), val in zip(features_info, patient_inputs):
            print(f" - {name:25s}: {val}")
            
        prob, pred = predict_single_patient(patient_inputs, model, scaler)
        
        print("\n" + "-"*40)
        print("          PREDICTION RESULT          ")
        print("-"*40)
        risk_label = "HIGH RISK (CARDIAC RISK)" if pred == 1 else "LOW RISK (NORMAL)"
        color_code = "\033[91m" if pred == 1 else "\033[92m"
        reset_code = "\033[0m"
        
        print(f"Probability of Cardiac Risk: {prob * 100:.2f}%")
        print(f"Risk Classification:        {color_code}{risk_label}{reset_code}")
        
        if pred == 1:
            print("\nAdvice: The patient exhibits a high clinical risk profile. Further diagnosis and lifestyle modifications are recommended.")
        else:
            print("\nAdvice: The patient exhibits a low risk profile. Maintenance of a healthy lifestyle is recommended.")
        print("-"*40)
        
        again = input("\nAnalyze another patient? (y/n) [y]: ").strip().lower()
        if again == 'n':
            break
            
    print("\nExiting inference CLI. Thank you.")

if __name__ == "__main__":
    run_interactive_cli()
