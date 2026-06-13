import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)
from src.dataset import load_and_prepare_data

# Constants
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
MODEL_PATH = os.path.join(MODELS_DIR, "cardiac_mlp_model.keras")

os.makedirs(FIGURES_DIR, exist_ok=True)

def evaluate_model(model_path=MODEL_PATH, test_data=None):
    """
    Evaluates the trained MLP model on the test dataset for Cardiac Risk Stratification.
    
    1. Loads the saved model.
    2. Runs predictions on the test set.
    3. Calculates classification metrics (Accuracy, Precision, Recall, F1, Specificity, AUC).
    4. Generates and saves Confusion Matrix and ROC Curve plots.
    5. Saves a text report of the final evaluations.
    """
    # 1. Load data
    if test_data is None:
        _, _, _, _, X_test, y_test, _, _ = load_and_prepare_data()
    else:
        X_test, y_test = test_data
        
    # 2. Load model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at: {model_path}. Please run training first.")
        
    model = tf.keras.models.load_model(model_path)
    print(f"\nTrained model loaded from: {model_path}")
    
    # 3. Predict probabilities and classes
    y_pred_prob = model.predict(X_test).ravel()
    y_pred = (y_pred_prob >= 0.5).astype(int)
    
    # 4. Compute Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Confusion Matrix for Specificity
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    
    # ROC Curve & AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    print("\n--- Model Evaluation Results ---")
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f} (Sensitivity)")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1-Score:    {f1:.4f}")
    print(f"AUC-ROC:     {roc_auc:.4f}")
    
    # Detailed classification report
    class_report = classification_report(y_test, y_pred, target_names=["No Disease", "Heart Disease"])
    print("\nClassification Report:\n", class_report)
    
    # 5. Save Text Report
    report_path = os.path.join(REPORTS_DIR, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write("==================================================\n")
        f.write("CARDIAC RISK STRATIFICATION - MLP EVALUATION REPORT\n")
        f.write("==================================================\n\n")
        f.write(f"Accuracy:    {accuracy:.4f}\n")
        f.write(f"Precision:   {precision:.4f}\n")
        f.write(f"Recall:      {recall:.4f} (Sensitivity)\n")
        f.write(f"Specificity: {specificity:.4f}\n")
        f.write(f"F1-Score:    {f1:.4f}\n")
        f.write(f"AUC-ROC:     {roc_auc:.4f}\n\n")
        f.write("Confusion Matrix counts:\n")
        f.write(f" - True Negatives (TN):  {tn}\n")
        f.write(f" - False Positives (FP): {fp}\n")
        f.write(f" - False Negatives (FN): {fn}\n")
        f.write(f" - True Positives (TP):  {tp}\n\n")
        f.write("Classification Report Details:\n")
        f.write(class_report)
    print(f"Saved evaluation report text to: {report_path}")
    
    # 6. Generate and Save Confusion Matrix Plot
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["No Disease", "Heart Disease"],
                yticklabels=["No Disease", "Heart Disease"])
    plt.title('Confusion Matrix', fontsize=12, fontweight='bold', pad=10)
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_path = os.path.join(FIGURES_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Confusion matrix plot saved to: {cm_path}")
    
    # 7. Generate and Save ROC Curve Plot
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=12, fontweight='bold', pad=10)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    roc_path = os.path.join(FIGURES_DIR, "roc_curve.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"ROC curve plot saved to: {roc_path}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'auc': roc_auc
    }

if __name__ == "__main__":
    evaluate_model()
