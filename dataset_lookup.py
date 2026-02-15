"""
Create a lookup of all complex IDs in the dataset for fast validation
"""

import csv
import os
from typing import Set

def load_dataset_complex_ids() -> Set[str]:
    """Load all complex IDs from all dataset CSV files"""
    complex_ids = set()
    
    dataset_files = [
        "final_train_features_true.csv",
        "final_valid_features_true.csv",
        "test2013_features_true.csv",
        "coreset2016_features_true.csv"
    ]
    
    for csv_file in dataset_files:
        if not os.path.exists(csv_file):
            continue
        
        try:
            with open(csv_file, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    if row and len(row) > 0:
                        complex_id = row[0].strip()
                        if complex_id and '/' in complex_id:
                            complex_ids.add(complex_id)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
    
    return complex_ids


# Global cache
_dataset_complex_ids = None

def get_dataset_complex_ids() -> Set[str]:
    """Get cached dataset complex IDs"""
    global _dataset_complex_ids
    if _dataset_complex_ids is None:
        _dataset_complex_ids = load_dataset_complex_ids()
    return _dataset_complex_ids


def filter_pdb_ids_by_dataset(pdb_ids: list) -> list:
    """Filter PDB IDs to only those that exist in the dataset"""
    dataset_ids = get_dataset_complex_ids()
    
    # Convert dataset complex IDs to PDB IDs (lowercase)
    dataset_pdb_ids = {cid.split('/')[0].lower() for cid in dataset_ids}
    
    # Filter input PDB IDs
    filtered = [pid for pid in pdb_ids if pid.lower() in dataset_pdb_ids]
    
    return filtered
