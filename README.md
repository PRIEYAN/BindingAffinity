# Protein-Ligand Binding Affinity Prediction

This project trains machine learning models to predict the binding affinity between proteins and ligands.

## Dataset

The dataset contains:
- **Training set**: `final_train_features_true.csv` (16,226 samples)
- **Validation set**: `final_valid_features_true.csv` (1,000 samples)
- **Test set**: `test2013_features_true.csv` (195 samples)
- **Coreset**: `coreset2016_features_true.csv` (285 samples)

Each sample contains:
- First column: Protein-ligand complex identifier (PDB path)
- Features: 16,801 numerical features describing protein-ligand interactions
- Target: Binding affinity (last column, likely pKa or similar metric)

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the training script:
```bash
python train_binding_affinity.py
```

The script will:
1. Load training and validation data
2. Train multiple models (Random Forest, XGBoost, LightGBM, Gradient Boosting, optional PyTorch NN)
3. Evaluate models on validation set
4. Select the best model based on validation RMSE
5. Save the best model to `models/` directory
6. Evaluate the best model on test set

## Models

The script trains and compares:
- **Random Forest**: Ensemble of decision trees (CPU only - no GPU support)
- **XGBoost**: Gradient boosting with XGBoost (GPU support available)
- **LightGBM**: Fast gradient boosting framework (GPU support available)
- **Gradient Boosting**: Scikit-learn's gradient boosting (CPU only)
- **PyTorch Neural Network**: Deep learning model (GPU support available, optional)

**Note**: Random Forest and Gradient Boosting from scikit-learn use CPU only. XGBoost and LightGBM will automatically use GPU if available.

## Evaluation Metrics

- **RMSE** (Root Mean Squared Error): Lower is better
- **MAE** (Mean Absolute Error): Lower is better
- **R²** (Coefficient of Determination): Higher is better (closer to 1.0)

## Output

The script prints:
- Dataset statistics
- Training progress for each model
- Validation metrics for each model
- Best model selection
- Test set performance (if test set is available)

## Making Predictions

After training, you can use the saved model to predict binding affinity for new protein-ligand pairs.

### Simple Prediction Script

For quick predictions from a CSV file:

```bash
python predict_simple.py input_features.csv [output_predictions.csv]
```

Example:
```bash
python predict_simple.py test_features.csv predictions.csv
```

### Advanced Prediction Script

For more control and options:

```bash
python predict_affinity.py --input input_features.csv --output predictions.csv
```

Or predict from command line features:
```bash
python predict_affinity.py --features 0.0 0.0 0.0 ... (16800 features)
```

### Input Format

The input CSV file should have the same format as the training data:
- **Option 1**: 16,800 features (no ID column)
- **Option 2**: ID in first column + 16,800 features
- **Option 3**: ID + features + target (target will be ignored)

The script will automatically detect the format and extract features.

### Output Format

The output CSV contains:
- `ID`: Identifier from input (or sample number)
- `Predicted_Binding_Affinity`: Predicted binding affinity value

## Model Files

After training, the following files are saved in the `models/` directory:
- `best_model_<model_name>.pkl` or `.pt`: The trained model
- `scaler_<model_name>.pkl`: Feature scaler
- `metadata_<model_name>.pkl`: Model metadata and performance metrics

## Customization

You can modify the script to:
- Adjust hyperparameters for each model
- Add more models
- Use different feature scaling methods
- Implement cross-validation
- Change model saving location
