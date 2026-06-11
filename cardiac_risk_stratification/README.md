# Cardiac Risk Stratification using Multilayer Perceptron (MLP)

This directory is created to isolate the Neural Network project for Cardiac Risk Stratification and prevent any git or workspace conflicts.

## Folder Structure

Below is the directory structure we are establishing:

```
cardiac_risk_stratification/
├── README.md                  # Project overview and documentation
├── requirements.txt           # Python package dependencies
├── data/                      # Dataset files
└── src/                       # Source code for MLP model
    ├── __init__.py
    ├── dataset.py             # Data loading and scaling utilities
    ├── model.py               # Multilayer Perceptron model definition
    ├── train.py               # Training scripts and loss logging
    └── evaluate.py            # Model evaluation and metrics reports
```

## Setup Instructions

1. **Python Environment**:
   It is recommended to use a virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

2. **Installation**:
   Once the implementation plan is approved, we will define dependencies in `requirements.txt`. You can install them using:
   ```bash
   pip install -r requirements.txt
   ```

3. **Running the Model**:
   Scripts will be executed from the root or source directory. Detailed commands will be documented here as we code.
