"""
MODEL TRAINING & VISUALIZATION MODULE
This script fits the compiled Keras model on training data (with validation split)
and visualizes the loss and accuracy metrics over epochs using Matplotlib.
"""

import os
import matplotlib.pyplot as plt

def train_model(model, X_train, y_train, epochs=100, batch_size=32, validation_split=0.2):
    """
    Fits the compiled MLP model on training data.
    
    Args:
        model (Sequential): Compiled Keras MLP model.
        X_train (DataFrame): Preprocessed training features.
        y_train (Series): Training labels.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        validation_split (float): Portion of training data reserved for validation.
        
    Returns:
        history (History): History object containing training and validation losses/accuracies.
    """
    print(f"[CRISP-DM Phase 4: Modeling] Training model for {epochs} epochs with batch size {batch_size}...")
    
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1
    )
    
    return history

def plot_training_history(history, save_dir=None):
    """
    Plots 'Training vs Validation Loss' and 'Training vs Validation Accuracy'
    on two separate line graphs and saves them.
    
    Args:
        history (History): Keras training history object.
        save_dir (str, optional): Directory to save output plot images.
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
    epochs_range = range(1, len(history.history['loss']) + 1)
    
    # ------------------ Plot 1: Training vs Validation Loss ------------------
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, history.history['loss'], 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs_range, history.history['val_loss'], 'r--', label='Validation Loss', linewidth=2)
    
    plt.title('Training vs Validation Loss Across Epochs', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Epochs', fontsize=12, labelpad=10)
    plt.ylabel('Loss (Binary Crossentropy)', fontsize=12, labelpad=10)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    if save_dir:
        loss_plot_path = os.path.join(save_dir, 'training_validation_loss.png')
        plt.savefig(loss_plot_path, dpi=300)
        print(f"-> Saved Loss plot to: {loss_plot_path}")
    plt.show()
    plt.close()
    
    # ------------------ Plot 2: Training vs Validation Accuracy ------------------
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, history.history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    plt.plot(epochs_range, history.history['val_accuracy'], 'r--', label='Validation Accuracy', linewidth=2)
    
    plt.title('Training vs Validation Accuracy Across Epochs', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Epochs', fontsize=12, labelpad=10)
    plt.ylabel('Accuracy', fontsize=12, labelpad=10)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    if save_dir:
        accuracy_plot_path = os.path.join(save_dir, 'training_validation_accuracy.png')
        plt.savefig(accuracy_plot_path, dpi=300)
        print(f"-> Saved Accuracy plot to: {accuracy_plot_path}")
    plt.show()
    plt.close()
