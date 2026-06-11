import argparse
import sys
from src.train import train_model
from src.evaluate import evaluate_model
from src.predict import run_interactive_cli

def run_pipeline():
    """Orchestrates the entire CRISP-DM Machine Learning pipeline."""
    print("="*60)
    print("     DIABETES RISK PREDICTION - MLP END-TO-END PIPELINE     ")
    print("="*60)
    
    # 1 & 2. Data Understanding, Preparation & Training (CRISP-DM Phases 2, 3, 4)
    print("\n[Phase 1-3] Running Data Preparation and Model Training...")
    model, history, test_data = train_model(epochs=100)
    
    # 3. Model Evaluation (CRISP-DM Phase 5)
    print("\n[Phase 4] Running Model Evaluation on Test Set...")
    metrics = evaluate_model(test_data=test_data)
    
    print("\n" + "="*60)
    print("                PIPELINE EXECUTION COMPLETE                 ")
    print("="*60)
    print(f"Final Test Accuracy:    {metrics['accuracy']*100:.2f}%")
    print(f"Final Test Recall:      {metrics['recall']*100:.2f}% (Sensitivity)")
    print(f"Final Test AUC-ROC:     {metrics['auc']:.4f}")
    print("="*60)
    print("Trained model and metrics saved to './models' and './reports'.")
    print("Run prediction CLI with: python main.py --predict")

def main():
    parser = argparse.ArgumentParser(description="Diabetes Risk Prediction using MLP Pipeline")
    parser.add_argument(
        "--predict", 
        action="store_true", 
        help="Run the interactive prediction CLI to input patient metrics"
    )
    
    args = parser.parse_args()
    
    if args.predict:
        run_interactive_cli()
    else:
        run_pipeline()

if __name__ == "__main__":
    main()
