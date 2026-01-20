# Example Usage Guide

## Step 1: Train the Model

First, train the model on your dataset:

```bash
python train_binding_affinity.py
```

This will:
- Train multiple models (XGBoost, LightGBM, Random Forest, etc.)
- Compare their performance
- Save the best model to `models/` directory
- Show validation and test set performance

**Expected output:**
```
============================================================
Protein-Ligand Binding Affinity Prediction
============================================================

Checking GPU availability...
  PyTorch: Available, CUDA: True
  GPU Device: NVIDIA GeForce RTX 3090

Loading final_train_features_true.csv...
  Shape: (16226, 16802)
  Features: 16800
  Target column: pKa
  ...

Training xgboost...
  Using GPU for XGBoost
  Train RMSE: 1.2345, MAE: 0.9876, R²: 0.8765
  Val RMSE: 1.3456, MAE: 1.0123, R²: 0.8543

...

Best Model: xgboost
Validation RMSE: 1.3456
Validation MAE: 1.0123
Validation R²: 0.8543

Saving best model (xgboost)...
  Model saved to: models/best_model_xgboost.pkl
  Scaler saved to: models/scaler_xgboost.pkl
  Metadata saved to: models/metadata_xgboost.pkl
```

## Step 2: Prepare Input Data

Create a CSV file with features for your protein-ligand pairs. The file should have the same format as the training data.

**Example `new_protein_ligand_features.csv`:**
```csv
protein_ligand_id,GLY_H_0,GLY_C_1,GLY_O_2,...,pKa
1a2b/1a2b_cplx.pdb,0.0,0.0,0.0,...,(features)
2c3d/2c3d_cplx.pdb,1.0,2.0,3.0,...,(features)
```

**Note**: 
- The ID column is optional
- The last column (if present) will be ignored if it looks like a target
- You need exactly 16,800 feature columns

## Step 3: Make Predictions

### Option A: Simple Script

```bash
python predict_simple.py new_protein_ligand_features.csv predictions.csv
```

**Output:**
```
Loading model: xgboost
Model performance: RMSE=1.3456, R²=0.8543

Loading features from new_protein_ligand_features.csv...
  Found 2 samples with 16800 features
Making predictions...

Predictions saved to predictions.csv

Results:
ID                          Predicted_Binding_Affinity
1a2b/1a2b_cplx.pdb         6.2345
2c3d/2c3d_cplx.pdb         5.9876

Statistics:
  Mean affinity: 6.1106
  Std: 0.1235
  Min: 5.9876
  Max: 6.2345
```

### Option B: Advanced Script

```bash
python predict_affinity.py --input new_protein_ligand_features.csv --output predictions.csv
```

## Step 4: Interpret Results

The predicted binding affinity values are in the same units as your training data (likely pKa values, ranging from ~2 to ~15).

- **Lower values** (2-5): Weaker binding
- **Higher values** (8-15): Stronger binding

## Troubleshooting

### "No trained models found"
- Make sure you've run `train_binding_affinity.py` first
- Check that the `models/` directory exists and contains model files

### "Expected 16800 features, got X"
- Your input CSV doesn't have the correct number of features
- Make sure you're using the same feature extraction method as training
- Check that you haven't accidentally included/excluded columns

### "GPU not being used"
- Random Forest always uses CPU (no GPU support)
- XGBoost and LightGBM will automatically use GPU if available
- Check GPU_SETUP.md for troubleshooting

## Batch Processing

To predict on multiple files:

```bash
for file in data/*.csv; do
    python predict_simple.py "$file" "predictions/$(basename $file)"
done
```
