import os
import matplotlib.pyplot as plt
import tensorflow as tf
from src.dataset import load_and_prepare_data
from src.model import create_mlp_model

# Constants
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports", "figures")
MODEL_PATH = os.path.join(MODELS_DIR, "cardiac_mlp_model.keras")

def train_model(epochs=100, batch_size=32, layers=[64, 32], dropout_rate=0.2, learning_rate=0.001):
    """
    Trains the MLP model for Cardiac Risk Stratification.
    
    1. Loads preprocessed datasets.
    2. Builds the MLP network.
    3. Trains the model with EarlyStopping and ModelCheckpoint.
    4. Plots and saves training history (Loss, Accuracy).
    """
    # 1. Load data
    (X_train, y_train, 
     X_val, y_val, 
     X_test, y_test, 
     scaler, 
     feature_names) = load_and_prepare_data()
    
    # 2. Build model
    input_dim = X_train.shape[1]
    model = create_mlp_model(
        input_dim=input_dim, 
        layers=layers, 
        dropout_rate=dropout_rate, 
        learning_rate=learning_rate
    )
    
    print("\n--- Model Architecture ---")
    model.summary()
    
    # 3. Setup Keras Callbacks
    # Early stopping stops training when validation loss stops improving to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )
    
    # Model checkpoint saves the model with the lowest validation loss
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    # 4. Train model
    print("\n--- Starting Training ---")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )
    
    # 5. Plot and Save Training History
    plot_training_history(history)
    
    print(f"\nModel training completed and saved to {MODEL_PATH}")
    return model, history, (X_test, y_test)

def plot_training_history(history):
    """Plots validation and training loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(history.history['loss'], label='Train Loss', color='b', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Val Loss', color='r', linestyle='--', linewidth=2)
    ax1.set_title('Model Loss History', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss (Binary Crossentropy)')
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Accuracy plot
    ax2.plot(history.history['accuracy'], label='Train Acc', color='g', linewidth=2)
    ax2.plot(history.history['val_accuracy'], label='Val Acc', color='orange', linestyle='--', linewidth=2)
    ax2.set_title('Model Accuracy History', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plot_path = os.path.join(REPORTS_DIR, "training_history.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Training history plot saved to: {plot_path}")

if __name__ == "__main__":
    train_model(epochs=100)
