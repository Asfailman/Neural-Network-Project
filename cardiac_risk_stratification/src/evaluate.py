"""
MODEL EVALUATION MODULE
This script evaluates the trained Keras MLP model on testing data,
printing performance metrics and plotting a Confusion Matrix heatmap.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, X_test, y_test, save_dir=None):
    """
    Generates predictions, prints the confusion matrix and classification report,
    and plots a heatmap of the confusion matrix.
    
    Args:
        model (Sequential): Trained Keras MLP model.
        X_test (DataFrame): Preprocessed testing features.
        y_test (Series): Testing labels.
        save_dir (str, optional): Directory to save the output heatmap image.
        
    Returns:
        cm (ndarray): The confusion matrix array.
        report (str): The classification report text.
    """
    print("[CRISP-DM Phase 5: Evaluation] Generating predictions on the testing set...")
    
    # 1. Generate predictions on the test set
    y_pred_prob = model.predict(X_test)
    
    # Convert probabilities to binary class labels (threshold = 0.5)
    y_pred = (y_pred_prob >= 0.5).astype(int)
    
    # 2. Print metrics
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Normal (0)', 'Heart Disease (1)'])
    
    print("\n" + "=" * 60)
    print("                MODEL PERFORMANCE EVALUATION REPORT")
    print("=" * 60)
    print("\nClassification Report (Accuracy, Precision, Recall, F1-Score):")
    print(report)
    print("\nConfusion Matrix:")
    print(cm)
    print("=" * 60)
    
    # 3. Polish Confusion Matrix visualization using Seaborn
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        cbar=False,
        xticklabels=['Predicted Normal (0)', 'Predicted Disease (1)'],
        yticklabels=['Actual Normal (0)', 'Actual Disease (1)'],
        annot_kws={'size': 14, 'weight': 'bold'}
    )
    
    plt.title('Confusion Matrix Heatmap', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('True Class label', fontsize=12, labelpad=10)
    plt.xlabel('Predicted Class label', fontsize=12, labelpad=10)
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        cm_plot_path = os.path.join(save_dir, 'confusion_matrix_heatmap.png')
        plt.savefig(cm_plot_path, dpi=300)
        print(f"-> Saved Confusion Matrix Heatmap to: {cm_plot_path}")
        
    plt.show()
    plt.close()
    
    return cm, report
