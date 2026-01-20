"""
Simple training script for binding affinity prediction
Quick start version with basic Random Forest model
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")

# Load training data
train_df = pd.read_csv('final_train_features_true.csv')
X_train = train_df.iloc[:, 1:-1]  # All columns except first (ID) and last (target)
y_train = train_df.iloc[:, -1]    # Last column is target (pKa)

# Load validation data
val_df = pd.read_csv('final_valid_features_true.csv')
X_val = val_df.iloc[:, 1:-1]
y_val = val_df.iloc[:, -1]

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Features: {X_train.shape[1]}")

# Handle missing values
X_train = X_train.fillna(0)
X_val = X_val.fillna(0)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Train Random Forest model
print("\nTraining Random Forest model...")
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)

# Predictions
y_train_pred = model.predict(X_train_scaled)
y_val_pred = model.predict(X_val_scaled)

# Evaluate
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
val_mae = mean_absolute_error(y_val, y_val_pred)
val_r2 = r2_score(y_val, y_val_pred)

print("\nResults:")
print(f"Training - RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}")
print(f"Validation - RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}")

# Test on test set if available
try:
    test_df = pd.read_csv('test2013_features_true.csv')
    X_test = test_df.iloc[:, 1:-1].fillna(0)
    y_test = test_df.iloc[:, -1]
    X_test_scaled = scaler.transform(X_test)
    y_test_pred = model.predict(X_test_scaled)
    
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"\nTest Set - RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")
except Exception as e:
    print(f"\nTest set evaluation skipped: {e}")

print("\nDone!")
