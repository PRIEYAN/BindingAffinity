"""
Interactive affinity predictor.

Important: this model does NOT take raw protein/ligand structures directly.
It predicts from the SAME 16,800 engineered features used in your CSV files.

So this script asks you for a complex ID (first column in your CSV, e.g. "3rqw/3rqw_cplx.pdb"),
looks up its feature row inside the provided dataset CSVs, then runs the saved model.

Can also be called with command-line argument:
  python3 run.py <complex_id>
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from predict_affinity import load_model


DATASET_FILES = [
    "final_train_features_true.csv",
    "final_valid_features_true.csv",
    "test2013_features_true.csv",
    "coreset2016_features_true.csv",
]

class NoOpScaler:
    """Used when tree models were trained without scaling (pickled as __main__.NoOpScaler)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X, y=None):
        return X


def find_features_by_id(csv_path: str, complex_id: str, chunksize: int = 2000) -> Optional[np.ndarray]:
    """
    Find one row by ID (first column) and return its features as shape (1, n_features).
    Uses chunked reading to avoid loading big CSVs into RAM.
    """
    if not os.path.exists(csv_path):
        return None

    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        id_col = chunk.columns[0]
        match = chunk[id_col] == complex_id
        if not match.any():
            continue

        row = chunk.loc[match].iloc[0]
        # Drop ID + target (last column)
        features = row.iloc[1:-1].to_numpy(dtype=np.float32, copy=False)
        return features.reshape(1, -1)

    return None


def predict_one(model, scaler, metadata, features: np.ndarray) -> float:
    features_scaled = scaler.transform(features)
    pred = model.predict(features_scaled)
    return float(pred[0])


def main() -> int:
    print("=" * 60)
    print("Interactive Protein–Ligand Binding Affinity Predictor")
    print("=" * 60)

    try:
        model, scaler, metadata = load_model("models")
    except Exception as e:
        print(f"\nERROR: Could not load model from `models/`: {e}")
        print("Run training first so `models/` exists and contains best_model/scaler/metadata files.")
        return 1

    # Check if complex_id provided as command-line argument
    if len(sys.argv) > 1:
        complex_id = sys.argv[1].strip()
        print(f"\nUsing complex ID from command line: {complex_id}")
    else:
        print("\nEnter the COMPLEX ID exactly as it appears in your dataset CSV first column.")
        print('Example: 3rqw/3rqw_cplx.pdb')
        complex_id = input("\nComplex ID: ").strip()
    
    if not complex_id:
        print("No ID provided.")
        return 1

    features = None
    found_in = None
    for f in DATASET_FILES:
        feats = find_features_by_id(f, complex_id)
        if feats is not None:
            features = feats
            found_in = f
            break

    if features is None:
        print("\nNot found in the default dataset files:")
        for f in DATASET_FILES:
            print(f" - {f}")
        print("\nIf you want to predict a NEW protein/ligand not in these CSVs, you must first compute")
        print("the same 16,800 features and save them to a CSV, then use:")
        print("  python3 predict_affinity.py --input your_features.csv --output predictions.csv")
        return 1

    if features.shape[1] != metadata["input_dim"]:
        print(f"\nERROR: Feature dimension mismatch. Model expects {metadata['input_dim']}, got {features.shape[1]}.")
        return 1

    pred = predict_one(model, scaler, metadata, features)
    print(f"\nFound in: {found_in}")
    print(f"Predicted affinity (same scale as training target `{metadata['model_name']}` / pKa): {pred:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

