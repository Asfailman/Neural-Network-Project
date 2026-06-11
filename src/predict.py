import os
import pickle
import numpy as np
import tensorflow as tf

# Constants
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "diabetes_mlp_model.keras")
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
    print("      DIABETES RISK PREDICTION - MLP INFERENCE CLI      ")
    print("="*50)
    
    try:
        model, scaler = load_resources()
        print("Successfully loaded model weights and StandardScaler scaler config.")
    except Exception as e:
        print(f"Error: Could not load trained resources. {e}")
        return
        
    features_info = [
        ("Pregnancies", "Number of pregnancies", 0, 20, 0),
        ("Glucose", "Plasma glucose concentration (2 hours in an oral glucose tolerance test) [mg/dL]", 30, 300, 110),
        ("BloodPressure", "Diastolic blood pressure [mm Hg]", 30, 150, 70),
        ("SkinThickness", "Triceps skin fold thickness [mm]", 5, 100, 20),
        ("Insulin", "2-Hour serum insulin [mu U/ml]", 10, 800, 80),
        ("BMI", "Body Mass Index (weight in kg / (height in m)^2)", 10.0, 60.0, 25.0),
        ("DiabetesPedigreeFunction", "Diabetes pedigree genetic factor (0.0 to 2.5)", 0.05, 2.5, 0.47),
        ("Age", "Age (years)", 18, 100, 30)
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
        risk_label = "HIGH RISK (DIABETIC)" if pred == 1 else "LOW RISK (NON-DIABETIC)"
        color_code = "\033[91m" if pred == 1 else "\033[92m"
        reset_code = "\033[0m"
        
        print(f"Probability of Diabetes: {prob * 100:.2f}%")
        print(f"Risk Classification:    {color_code}{risk_label}{reset_code}")
        
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
