"""
MLP ARCHITECTURE & COMPILATION MODULE
This script defines the Multilayer Perceptron (MLP) Artificial Neural Network (ANN) model structure
using TensorFlow/Keras, compiling it with optimizer, loss function, and metrics.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input

def build_mlp_model(input_dim, dropout_rate=0.2):
    """
    Builds and compiles a Keras Sequential MLP model.
    
    Args:
        input_dim (int): The number of features in the input vector.
        dropout_rate (float): The fraction of the input units to drop during training.
        
    Returns:
        model (Sequential): Compiled Keras Sequential model.
    """
    print(f"[CRISP-DM Phase 4: Modeling] Building MLP Model with input dimension = {input_dim}...")
    
    model = Sequential([
        # 1. Input layer matching feature count
        Input(shape=(input_dim,), name="input_layer"),
        
        # 2. First hidden Dense layer with 64 neurons & ReLU activation
        Dense(64, activation='relu', name="hidden_layer_1"),
        # Dropout layer to prevent overfitting
        Dropout(dropout_rate, name="dropout_1"),
        
        # 3. Second hidden Dense layer with 32 neurons & ReLU activation
        Dense(32, activation='relu', name="hidden_layer_2"),
        # Dropout layer to prevent overfitting
        Dropout(dropout_rate, name="dropout_2"),
        
        # 4. Output layer with 1 neuron and Sigmoid activation for binary classification
        Dense(1, activation='sigmoid', name="output_layer")
    ])
    
    # 5. Model compilation
    # - Optimizer: 'adam' (stochastic gradient descent method based on adaptive estimation of first and second-order moments)
    # - Loss: 'binary_crossentropy' (standard loss for binary classification)
    # - Metric: 'accuracy' (computes how often predictions match labels)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("MLP Model Architecture Summary:")
    model.summary()
    
    return model

if __name__ == "__main__":
    # Test building the model with an arbitrary input dimension of 15
    test_model = build_mlp_model(input_dim=15)
