"""
MAIN MODEL PIPELINE EXECUTION SCRIPT
This is the central entry point that executes the end-to-end CRISP-DM workflow:
1. Data Preparation (Loading, one-hot encoding, splitting, scaling)
2. Modeling (Building and compiling the MLP model)
3. Model Training & History Plotting
4. Evaluation (Generating confusion matrix and classification reports)
"""

import os
import sys

# Use non-interactive backend for matplotlib to prevent blocking on plt.show()
import matplotlib
matplotlib.use('Agg')

# Ensure local source files in 'src/' are importable regardless of working directory
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, 'src'))
sys.path.insert(0, script_dir)

# Import modular pipeline steps
# pyrefly: ignore [missing-import]
from dataset import load_and_preprocess_data
# pyrefly: ignore [missing-import]
from model import build_mlp_model
# pyrefly: ignore [missing-import]
from train import train_model, plot_training_history
# pyrefly: ignore [missing-import]
from evaluate import evaluate_model

def run_pipeline():
    print("=" * 70)
    print("  CARDIAC RISK STRATIFICATION - MULTILAYER PERCEPTRON (MLP) PIPELINE")
    print("                    CRISP-DM METHODOLOGY STUDY")
    print("=" * 70)
    
    # Define directories for storing models and plots
    plots_dir = os.path.join(script_dir, 'plots')
    models_dir = os.path.join(script_dir, 'models')
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # ------------------ Phase 3: Data Preparation ------------------
    X_train, X_test, y_train, y_test, features = load_and_preprocess_data()
    
    # ------------------ Phase 4: Modeling (Build & Train) ------------------
    # Build model using input dimension matching the processed features
    input_dim = X_train.shape[1]
    model = build_mlp_model(input_dim=input_dim, dropout_rate=0.2)
    
    # Fit the model (100 Epochs, batch size 32, 20% validation split)
    history = train_model(
        model, 
        X_train, 
        y_train, 
        epochs=100, 
        batch_size=32, 
        validation_split=0.2
    )
    
    # Save the trained model structure and weights (recommending Keras standard format)
    model_save_path = os.path.join(models_dir, 'cardiac_mlp_model.keras')
    model.save(model_save_path)
    print(f"\n-> Successfully saved trained MLP model to: {model_save_path}")
    
    # Plot and save "Loss" and "Accuracy" curves
    print("\nVisualizing and saving learning curves...")
    plot_training_history(history, save_dir=plots_dir)
    
    # ------------------ Phase 5: Evaluation ------------------
    # Evaluate model on unseen test set, printing classification report and confusion matrix
    cm, report = evaluate_model(model, X_test, y_test, save_dir=plots_dir)
    
    print("\n" + "=" * 70)
    print("               PIPELINE RUN COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_pipeline()
