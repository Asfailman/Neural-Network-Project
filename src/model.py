import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input

def create_mlp_model(input_dim, layers=[64, 32], dropout_rate=0.2, learning_rate=0.001):
    """
    Creates and compiles a Multilayer Perceptron (MLP) for binary classification.
    
    Parameters:
    - input_dim (int): Number of features in the input data.
    - layers (list of int): List specifying the number of units in each hidden layer.
    - dropout_rate (float): Dropout rate for regularization.
    - learning_rate (float): Learning rate for the Adam optimizer.
    
    Returns:
    - model (tf.keras.Model): The compiled Keras Sequential model.
    """
    model = Sequential()
    
    # Input layer
    model.add(Input(shape=(input_dim,)))
    
    # Hidden layers
    for i, units in enumerate(layers):
        model.add(Dense(units, activation='relu', name=f"dense_hidden_{i+1}"))
        model.add(BatchNormalization(name=f"batch_norm_{i+1}"))
        model.add(Dropout(dropout_rate, name=f"dropout_{i+1}"))
        
    # Output layer for binary classification (Outcome: 0 or 1)
    model.add(Dense(1, activation='sigmoid', name="output_layer"))
    
    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.BinaryCrossentropy()
    
    metrics = [
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')
    ]
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    return model

if __name__ == "__main__":
    # Test model creation
    model = create_mlp_model(input_dim=8)
    model.summary()
