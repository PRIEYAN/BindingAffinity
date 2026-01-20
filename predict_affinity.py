"""
Inference script for protein-ligand binding affinity prediction
Loads a trained model and predicts binding affinity for new protein-ligand pairs
"""

import pandas as pd
import numpy as np
import pickle
import os
import argparse
import warnings
warnings.filterwarnings('ignore')

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

class NoOpScaler:
    """Used when tree models were trained without scaling (pickled as __main__.NoOpScaler)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X, y=None):
        return X

def load_model(model_dir='models'):
    """Load the saved model, scaler, and metadata"""
    # Find the best model files
    model_files = [f for f in os.listdir(model_dir) if f.startswith('best_model_')]
    
    if not model_files:
        raise FileNotFoundError(f"No trained models found in {model_dir}. Please train a model first using train_binding_affinity.py")
    
    # Get the model name from the first file
    model_name = model_files[0].replace('best_model_', '').replace('.pkl', '').replace('.pt', '')
    
    # Load metadata
    metadata_path = os.path.join(model_dir, f'metadata_{model_name}.pkl')
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    # Load scaler
    scaler_path = os.path.join(model_dir, f'scaler_{model_name}.pkl')
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Load model
    if model_name == 'pytorch_nn':
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is required for this model. Install with: pip install torch")
        
        # Define the same network architecture
        class BindingAffinityNet(nn.Module):
            def __init__(self, input_dim):
                super(BindingAffinityNet, self).__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 512),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1)
                )
            
            def forward(self, x):
                return self.net(x).squeeze()
        
        model = BindingAffinityNet(metadata['input_dim'])
        model_path = os.path.join(model_dir, f'best_model_{model_name}.pt')
        device = torch.device(metadata.get('device', 'cpu').replace('device(type=\'', '').replace('\')', ''))
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        model.to(device)
        metadata['device'] = device
    else:
        model_path = os.path.join(model_dir, f'best_model_{model_name}.pkl')
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    
    print(f"Loaded model: {model_name}")
    print(f"  Validation RMSE: {metadata['metrics']['val_rmse']:.4f}")
    print(f"  Validation R²: {metadata['metrics']['val_r2']:.4f}")
    
    return model, scaler, metadata

def predict_from_csv(csv_file, model, scaler, metadata):
    """Predict binding affinity from a CSV file with features"""
    print(f"\nLoading features from {csv_file}...")
    
    # Load the CSV file
    df = pd.read_csv(csv_file)
    
    # Check if first column is ID
    if df.shape[1] == 16801:  # Features only (no ID, no target)
        X = df
        ids = None
    elif df.shape[1] == 16802:  # ID + features (no target)
        ids = df.iloc[:, 0]
        X = df.iloc[:, 1:-1] if df.shape[1] == 16802 else df.iloc[:, 1:]
    else:
        # Assume first column is ID, last might be target, middle are features
        ids = df.iloc[:, 0]
        X = df.iloc[:, 1:-1]  # Exclude ID and last column
    
    # Check feature count
    if X.shape[1] != metadata['input_dim']:
        raise ValueError(f"Expected {metadata['input_dim']} features, got {X.shape[1]}")
    
    # Handle missing values
    X = X.fillna(0)
    
    # Scale features (or pass-through if NoOpScaler)
    X_scaled = scaler.transform(X)
    
    # Predict
    if metadata['model_name'] == 'pytorch_nn':
        device = metadata['device']
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        model.eval()
        with torch.no_grad():
            predictions = model(X_tensor).cpu().numpy()
    else:
        predictions = model.predict(X_scaled)
    
    # Create results dataframe
    results = pd.DataFrame({
        'ID': ids if ids is not None else range(len(predictions)),
        'Predicted_Binding_Affinity': predictions
    })
    
    return results

def predict_from_features(features, model, scaler, metadata):
    """Predict binding affinity from a numpy array or list of features"""
    # Convert to numpy array
    if isinstance(features, list):
        features = np.array(features)
    
    # Reshape if needed (single sample)
    if features.ndim == 1:
        features = features.reshape(1, -1)
    
    # Check feature count
    if features.shape[1] != metadata['input_dim']:
        raise ValueError(f"Expected {metadata['input_dim']} features, got {features.shape[1]}")
    
    # Handle missing values
    features = np.nan_to_num(features, nan=0.0)
    
    # Scale features (or pass-through if NoOpScaler)
    features_scaled = scaler.transform(features)
    
    # Predict
    if metadata['model_name'] == 'pytorch_nn':
        device = metadata['device']
        features_tensor = torch.FloatTensor(features_scaled).to(device)
        model.eval()
        with torch.no_grad():
            predictions = model(features_tensor).cpu().numpy()
    else:
        predictions = model.predict(features_scaled)
    
    return predictions

def main():
    parser = argparse.ArgumentParser(description='Predict protein-ligand binding affinity')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input CSV file with features (or use --features for single prediction)')
    parser.add_argument('--output', '-o', type=str, default='predictions.csv',
                        help='Output CSV file for predictions (default: predictions.csv)')
    parser.add_argument('--model_dir', '-m', type=str, default='models',
                        help='Directory containing saved models (default: models)')
    parser.add_argument('--features', '-f', type=str, nargs='+',
                        help='Provide features directly as space-separated values (for single prediction)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Protein-Ligand Binding Affinity Prediction")
    print("=" * 60)
    
    # Load model
    try:
        model, scaler, metadata = load_model(args.model_dir)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Make predictions
    try:
        if args.features:
            # Single prediction from command line
            features = [float(f) for f in args.features]
            predictions = predict_from_features(features, model, scaler, metadata)
            print(f"\nPredicted Binding Affinity: {predictions[0]:.4f}")
        else:
            # Batch prediction from CSV
            results = predict_from_csv(args.input, model, scaler, metadata)
            
            # Save results
            results.to_csv(args.output, index=False)
            print(f"\nPredictions saved to {args.output}")
            print(f"\nSample predictions:")
            print(results.head(10).to_string(index=False))
            
            print(f"\nStatistics:")
            print(f"  Mean: {results['Predicted_Binding_Affinity'].mean():.4f}")
            print(f"  Std: {results['Predicted_Binding_Affinity'].std():.4f}")
            print(f"  Min: {results['Predicted_Binding_Affinity'].min():.4f}")
            print(f"  Max: {results['Predicted_Binding_Affinity'].max():.4f}")
    
    except Exception as e:
        print(f"Error making predictions: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
