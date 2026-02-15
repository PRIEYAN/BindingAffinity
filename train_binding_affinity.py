"""
Training script for protein-ligand binding affinity prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Try to import PyTorch for GPU neural network
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    PYTORCH_AVAILABLE = True
except ImportError:
    
    PYTORCH_AVAILABLE = False

def check_gpu_availability():
    """Check GPU availability for different libraries"""
    gpu_info = {
        'xgboost': False,
        'lightgbm': False,
        'pytorch': False
    }
    
    # Check XGBoost GPU - we'll try during actual training
    # Just report if CUDA is available
    if PYTORCH_AVAILABLE and torch.cuda.is_available():
        gpu_info['xgboost'] = True  # Likely available if PyTorch sees CUDA
        gpu_info['lightgbm'] = True
        gpu_info['pytorch'] = True
    
    return gpu_info

# Configuration
TRAIN_FILE = 'final_train_features_true.csv'
VALID_FILE = 'final_valid_features_true.csv'
TEST_FILE = 'test2013_features_true.csv'
CORESET_FILE = 'coreset2016_features_true.csv'

def train_pytorch_nn(X_train, y_train, X_val, y_val):
    """Train a PyTorch neural network with GPU support"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {device}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    y_train_tensor = torch.FloatTensor(y_train.values).to(device)
    X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
    y_val_tensor = torch.FloatTensor(y_val.values).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    # Define model
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
    
    model = BindingAffinityNet(X_train_scaled.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    n_epochs = 50
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break
    
    # Predictions
    model.eval()
    with torch.no_grad():
        y_train_pred = model(X_train_tensor).cpu().numpy()
        y_val_pred = model(X_val_tensor).cpu().numpy()
    
    # Evaluate
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    
    print(f"  Train RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}")
    print(f"  Val RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}")
    
    return model, scaler, {
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'train_r2': train_r2,
        'val_rmse': val_rmse,
        'val_mae': val_mae,
        'val_r2': val_r2,
        'device': device
    }

class NoOpScaler:
    """A drop-in replacement for sklearn scalers when scaling is not needed."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X, y=None):
        return X

def load_data(file_path, target_col=None):
    """Load data from CSV file"""
    print(f"Loading {file_path}...")
    df = pd.read_csv(file_path)
    
    # First column is ID
    id_col = df.columns[0]
    
    # Determine target column
    if target_col is None:
        # Try to find target column (last column or one with 'pKa', 'affinity', etc.)
        possible_targets = [c for c in df.columns if any(term in c.lower() 
                          for term in ['pka', 'affinity', 'kd', 'ic50', 'binding'])]
        if possible_targets:
            target_col = possible_targets[-1]  # Take the last matching column
        else:
            target_col = df.columns[-1]  # Default to last column
    
    # Separate features and target
    X = df.drop([id_col, target_col], axis=1)
    y = df[target_col]

    # Reduce memory: use float32 for features/target
    # (Tree models work well with float32 and it halves RAM vs float64)
    X = X.astype(np.float32, copy=False)
    y = y.astype(np.float32, copy=False)
    
    print(f"  Shape: {df.shape}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Target column: {target_col}")
    print(f"  Target stats: min={y.min():.2f}, max={y.max():.2f}, mean={y.mean():.2f}, std={y.std():.2f}")
    
    return X, y, df[id_col]

def train_model(X_train, y_train, X_val, y_val, model_name='random_forest'):
    """Train a regression model"""

    # IMPORTANT: Scaling creates huge extra arrays (OOM risk).
    # Tree-based models (RF/XGB/LGBM/GBDT) do NOT need scaling.
    needs_scaling = model_name in {'ridge', 'lasso', 'pytorch_nn'}

    if needs_scaling:
        scaler = StandardScaler()
        X_train_in = scaler.fit_transform(X_train)
        X_val_in = scaler.transform(X_val)
    else:
        scaler = NoOpScaler()
        X_train_in = X_train.values if hasattr(X_train, "values") else X_train
        X_val_in = X_val.values if hasattr(X_val, "values") else X_val
    
    # Initialize model
    if model_name == 'random_forest':
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    elif model_name == 'gradient_boosting':
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
    elif model_name == 'xgboost':
        # Try GPU first, fallback to CPU
        try:
            # Try newer API first (XGBoost 2.0+)
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                device='cuda',  # Use GPU
                tree_method='hist'
            )
            print("  Using GPU for XGBoost")
        except:
            try:
                # Try older API (XGBoost < 2.0)
                model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    tree_method='gpu_hist'  # GPU method
                )
                print("  Using GPU for XGBoost (gpu_hist)")
            except:
                # Fallback to CPU
                model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=-1
                )
                print("  Using CPU for XGBoost (GPU not available)")
    elif model_name == 'lightgbm':
        # Try GPU first, fallback to CPU
        try:
            model = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                device='gpu',  # Use GPU
                gpu_platform_id=0,
                gpu_device_id=0,
                verbose=-1
            )
            print("  Using GPU for LightGBM")
        except:
            # Fallback to CPU
            model = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            print("  Using CPU for LightGBM (GPU not available)")
    elif model_name == 'pytorch_nn':
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed. Install with: pip install torch")
        # PyTorch neural network will be handled separately
        return train_pytorch_nn(X_train, y_train, X_val, y_val)
    elif model_name == 'ridge':
        model = Ridge(alpha=1.0)
    elif model_name == 'lasso':
        model = Lasso(alpha=0.1)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Train
    print(f"\nTraining {model_name}...")
    model.fit(X_train_in, y_train)
    
    # Predict
    y_train_pred = model.predict(X_train_in)
    y_val_pred = model.predict(X_val_in)
    
    # Evaluate
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    
    print(f"  Train RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}")
    print(f"  Val RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}")
    
    return model, scaler, {
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'train_r2': train_r2,
        'val_rmse': val_rmse,
        'val_mae': val_mae,
        'val_r2': val_r2
    }

def main():
    print("=" * 60)
    print("Protein-Ligand Binding Affinity Prediction")
    print("=" * 60)
    
    # Check GPU availability
    print("\nChecking GPU availability...")
    gpu_info = check_gpu_availability()
    if PYTORCH_AVAILABLE:
        print(f"  PyTorch: Available, CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU Device: {torch.cuda.get_device_name(0)}")
    else:
        print("  PyTorch: Not installed (optional for neural networks)")
    
    # Load data
    X_train, y_train, train_ids = load_data(TRAIN_FILE)
    X_val, y_val, val_ids = load_data(VALID_FILE)
    
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    
    # Check for missing values
    print(f"\nMissing values in training: {X_train.isnull().sum().sum()}")
    print(f"Missing values in validation: {X_val.isnull().sum().sum()}")
    
    # Handle missing values
    if X_train.isnull().sum().sum() > 0:
        X_train = X_train.fillna(0)
    if X_val.isnull().sum().sum() > 0:
        X_val = X_val.fillna(0)
    
    # Train multiple models
    # Keep this memory-safe by default: tree boosters first (and often best).
    # RandomForest is CPU-only and can be heavy; enable it later if you have plenty of RAM.
    models_to_try = ['xgboost', 'lightgbm']
    # You can add these back if RAM allows:
    # models_to_try += ['random_forest', 'gradient_boosting']
    if PYTORCH_AVAILABLE and torch.cuda.is_available():
        models_to_try.append('pytorch_nn')
    results = {}
    
    for model_name in models_to_try:
        try:
            model, scaler, metrics = train_model(X_train, y_train, X_val, y_val, model_name)
            results[model_name] = {
                'model': model,
                'scaler': scaler,
                'metrics': metrics
            }
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            continue
    
    # Find best model
    if results:
        best_model_name = min(results.keys(), 
                            key=lambda x: results[x]['metrics']['val_rmse'])
        best_model = results[best_model_name]
        
        print(f"\n{'=' * 60}")
        print(f"Best Model: {best_model_name}")
        print(f"Validation RMSE: {best_model['metrics']['val_rmse']:.4f}")
        print(f"Validation MAE: {best_model['metrics']['val_mae']:.4f}")
        print(f"Validation R²: {best_model['metrics']['val_r2']:.4f}")
        print(f"{'=' * 60}")
        
        # Save the best model
        print(f"\nSaving best model ({best_model_name})...")
        os.makedirs('models', exist_ok=True)
        
        # Save model and scaler
        model_path = f'models/best_model_{best_model_name}.pkl'
        scaler_path = f'models/scaler_{best_model_name}.pkl'
        metadata_path = f'models/metadata_{best_model_name}.pkl'
        
        # Save model (handle PyTorch separately)
        if best_model_name == 'pytorch_nn':
            torch.save(best_model['model'].state_dict(), f'models/best_model_{best_model_name}.pt')
            with open(scaler_path, 'wb') as f:
                pickle.dump(best_model['scaler'], f)
            with open(metadata_path, 'wb') as f:
                pickle.dump({
                    'model_name': best_model_name,
                    'metrics': best_model['metrics'],
                    'device': str(best_model['metrics']['device']),
                    'input_dim': X_train.shape[1]
                }, f)
            print(f"  Model saved to: models/best_model_{best_model_name}.pt")
        else:
            with open(model_path, 'wb') as f:
                pickle.dump(best_model['model'], f)
            with open(scaler_path, 'wb') as f:
                pickle.dump(best_model['scaler'], f)
            with open(metadata_path, 'wb') as f:
                pickle.dump({
                    'model_name': best_model_name,
                    'metrics': best_model['metrics'],
                    'input_dim': X_train.shape[1]
                }, f)
            print(f"  Model saved to: {model_path}")
        
        print(f"  Scaler saved to: {scaler_path}")
        print(f"  Metadata saved to: {metadata_path}")
        
        # Test on test set if available
        try:
            X_test, y_test, test_ids = load_data(TEST_FILE)
            if X_test.isnull().sum().sum() > 0:
                X_test = X_test.fillna(0)
            
            # Handle PyTorch model differently
            if best_model_name == 'pytorch_nn':
                X_test_scaled = best_model['scaler'].transform(X_test)
                X_test_tensor = torch.FloatTensor(X_test_scaled).to(best_model['metrics']['device'])
                best_model['model'].eval()
                with torch.no_grad():
                    y_test_pred = best_model['model'](X_test_tensor).cpu().numpy()
            else:
                X_test_scaled = best_model['scaler'].transform(X_test)
                y_test_pred = best_model['model'].predict(X_test_scaled)
            
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            test_mae = mean_absolute_error(y_test, y_test_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            
            print(f"\nTest Set Performance:")
            print(f"  RMSE: {test_rmse:.4f}")
            print(f"  MAE: {test_mae:.4f}")
            print(f"  R²: {test_r2:.4f}")
        except Exception as e:
            print(f"\nCould not evaluate on test set: {e}")
    
    print("\nTraining complete!")

if __name__ == "__main__":
    main()
