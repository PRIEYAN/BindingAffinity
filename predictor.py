"""
XGBoost model predictor
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class NoOpScaler:
    """Used when tree models were trained without scaling."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X
    def fit_transform(self, X, y=None):
        return X


# Global model cache
_model = None
_scaler = None
_metadata = None


def load_model():
    """Load model once at startup"""
    global _model, _scaler, _metadata
    
    if _model is not None:
        return _model, _scaler, _metadata
    
    model_dir = 'models'
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Models directory not found: {model_dir}")
    
    model_files = [f for f in os.listdir(model_dir) if f.startswith('best_model_') and f.endswith('.pkl')]
    if not model_files:
        raise FileNotFoundError("No trained models found")
    
    model_name = model_files[0].replace('best_model_', '').replace('.pkl', '')
    
    with open(os.path.join(model_dir, f'metadata_{model_name}.pkl'), 'rb') as f:
        _metadata = pickle.load(f)
    
    # Load scaler (with NoOpScaler fallback for pickle compatibility)
    try:
        with open(os.path.join(model_dir, f'scaler_{model_name}.pkl'), 'rb') as f:
            _scaler = pickle.load(f)
    except (AttributeError, ModuleNotFoundError) as e:
        if 'NoOpScaler' in str(e):
            # If NoOpScaler is missing during unpickling, use our class
            import sys
            # Temporarily add NoOpScaler to pickle's available classes
            sys.modules['__main__'].NoOpScaler = NoOpScaler
            with open(os.path.join(model_dir, f'scaler_{model_name}.pkl'), 'rb') as f:
                _scaler = pickle.load(f)
        else:
            raise
    
    with open(os.path.join(model_dir, f'best_model_{model_name}.pkl'), 'rb') as f:
        _model = pickle.load(f)
    
    return _model, _scaler, _metadata


def find_complex_in_csv(complex_id: str, csv_path: Optional[str] = None) -> Optional[np.ndarray]:
    """Find complex ID in CSV and return features. Checks all available dataset files."""
    # List of dataset files to check (in order of preference)
    dataset_files = [
        "final_train_features_true.csv",
        "final_valid_features_true.csv",
        "test2013_features_true.csv",
        "coreset2016_features_true.csv"
    ]
    
    # If specific path provided, check that first
    if csv_path:
        dataset_files.insert(0, csv_path)
    
    for csv_file in dataset_files:
        if not os.path.exists(csv_file):
            continue
        
        try:
            for chunk in pd.read_csv(csv_file, chunksize=2000):
                id_col = chunk.columns[0]
                match = chunk[id_col] == complex_id
                if match.any():
                    row = chunk.loc[match].iloc[0]
                    features = row.iloc[1:-1].to_numpy(dtype=np.float32)
                    print(f"Found complex {complex_id} in {csv_file}")
                    return features.reshape(1, -1)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            continue
    
    print(f"Complex {complex_id} not found in any dataset files")
    return None


def predict_affinity(complex_id: str) -> Tuple[float, bool]:
    """Predict affinity for a complex ID. Returns (affinity_value, success)"""
    try:
        model, scaler, metadata = load_model()
        features = find_complex_in_csv(complex_id)
        
        if features is None:
            return (0.0, False)
        
        if features.shape[1] != metadata['input_dim']:
            return (0.0, False)
        
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        
        return (float(prediction), True)
    except Exception as e:
        print(f"Prediction error: {e}")
        return (0.0, False)
