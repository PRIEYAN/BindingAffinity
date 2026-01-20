"""
Simple prediction script - easy to use version
Predicts binding affinity for a single protein-ligand pair from CSV file
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys

class NoOpScaler:
    """Used when tree models were trained without scaling (pickled as __main__.NoOpScaler)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X, y=None):
        return X

def predict_binding_affinity(input_file, output_file='prediction_result.csv'):
    """
    Predict binding affinity from a CSV file with features
    
    Args:
        input_file: Path to CSV file with features (same format as training data)
        output_file: Path to save predictions
    """
    
    # Check if model exists
    model_dir = 'models'
    if not os.path.exists(model_dir):
        print("Error: No trained models found. Please run train_binding_affinity.py first.")
        return
    
    # Find model files
    model_files = [f for f in os.listdir(model_dir) if f.startswith('best_model_') and f.endswith('.pkl')]
    if not model_files:
        print("Error: No trained models found. Please run train_binding_affinity.py first.")
        return
    
    # Load the first available model
    model_name = model_files[0].replace('best_model_', '').replace('.pkl', '')
    
    print(f"Loading model: {model_name}")
    
    # Load model
    with open(os.path.join(model_dir, f'best_model_{model_name}.pkl'), 'rb') as f:
        model = pickle.load(f)
    
    # Load scaler
    with open(os.path.join(model_dir, f'scaler_{model_name}.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    
    # Load metadata
    with open(os.path.join(model_dir, f'metadata_{model_name}.pkl'), 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"Model performance: RMSE={metadata['metrics']['val_rmse']:.4f}, R²={metadata['metrics']['val_r2']:.4f}")
    
    # Load input data
    print(f"\nLoading features from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Extract features (assume same format as training: ID in first column, features in rest)
    if df.shape[1] == 16801:  # Features only
        X = df
        ids = None
    elif df.shape[1] == 16802:  # ID + features
        ids = df.iloc[:, 0]
        X = df.iloc[:, 1:]
    else:
        # Try to detect: first column might be ID, last might be target
        ids = df.iloc[:, 0]
        X = df.iloc[:, 1:-1]  # Exclude first and last
    
    print(f"  Found {len(X)} samples with {X.shape[1]} features")
    
    # Check feature count
    if X.shape[1] != metadata['input_dim']:
        print(f"Warning: Expected {metadata['input_dim']} features, got {X.shape[1]}")
        if X.shape[1] > metadata['input_dim']:
            X = X.iloc[:, :metadata['input_dim']]
            print(f"  Using first {metadata['input_dim']} features")
        else:
            print("Error: Not enough features!")
            return
    
    # Handle missing values
    X = X.fillna(0)
    
    # Scale features (or pass-through if NoOpScaler)
    X_scaled = scaler.transform(X)
    
    # Predict
    print("Making predictions...")
    predictions = model.predict(X_scaled)
    
    # Create results
    if ids is not None:
        results = pd.DataFrame({
            'ID': ids,
            'Predicted_Binding_Affinity': predictions
        })
    else:
        results = pd.DataFrame({
            'Sample': range(len(predictions)),
            'Predicted_Binding_Affinity': predictions
        })
    
    # Save results
    results.to_csv(output_file, index=False)
    
    print(f"\nPredictions saved to {output_file}")
    print(f"\nResults:")
    print(results.to_string(index=False))
    print(f"\nStatistics:")
    print(f"  Mean affinity: {predictions.mean():.4f}")
    print(f"  Std: {predictions.std():.4f}")
    print(f"  Min: {predictions.min():.4f}")
    print(f"  Max: {predictions.max():.4f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_simple.py <input_csv_file> [output_csv_file]")
        print("\nExample:")
        print("  python predict_simple.py test_features.csv predictions.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'predictions.csv'
    
    predict_binding_affinity(input_file, output_file)
