"""
Extract protein and ligand names from dataset CSV files
"""

import pandas as pd
import requests
import time
from typing import List, Tuple

def get_pdb_info(pdb_id: str) -> Tuple[str, str]:
    """
    Get protein and ligand names from RCSB PDB API
    """
    try:
        # Get structure summary
        url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id.upper()}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            title = data.get('struct', {}).get('title', '')
            
            # Try to extract protein name from title
            protein = title.split('with')[0].strip() if 'with' in title else title.split('-')[0].strip()
            
            # Get ligand info
            ligand_url = f"https://data.rcsb.org/rest/v1/core/chemcomp/{pdb_id.upper()}"
            ligand = "Unknown"
            
            # Try polymer entity for protein name
            polymer_url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id.upper()}/1"
            polymer_response = requests.get(polymer_url, timeout=10)
            if polymer_response.status_code == 200:
                polymer_data = polymer_response.json()
                entity_names = polymer_data.get('rcsb_entity_source_organism', [{}])[0]
                if entity_names:
                    protein = entity_names.get('ncbi_scientific_name', protein)
            
            return protein[:50], ligand  # Limit length
    except Exception as e:
        print(f"Error fetching {pdb_id}: {e}")
    
    return "Unknown", "Unknown"


def extract_complex_ids_from_csv(csv_file: str, limit: int = 50) -> List[str]:
    """Extract complex IDs from CSV file"""
    complex_ids = []
    try:
        # Read in chunks to handle large files
        for chunk in pd.read_csv(csv_file, chunksize=1000, nrows=limit*1000):
            id_col = chunk.columns[0]
            complex_ids.extend(chunk[id_col].unique().tolist())
            if len(complex_ids) >= limit:
                break
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
    
    return complex_ids[:limit]


def main():
    """Extract protein-ligand pairs from dataset"""
    print("Extracting complex IDs from dataset files...")
    
    # Get complex IDs from training set
    complex_ids = extract_complex_ids_from_csv("final_train_features_true.csv", limit=30)
    
    print(f"Found {len(complex_ids)} complex IDs")
    print("Fetching protein and ligand names from RCSB PDB...")
    
    results = []
    
    for i, complex_id in enumerate(complex_ids, 1):
        pdb_id = complex_id.split('/')[0] if '/' in complex_id else complex_id
        print(f"[{i}/{len(complex_ids)}] Processing {pdb_id}...")
        
        # Get info from RCSB
        protein, ligand = get_pdb_info(pdb_id)
        
        # If we got Unknown, try a simpler approach - use PDB ID as reference
        if protein == "Unknown":
            protein = f"Protein from {pdb_id}"
        if ligand == "Unknown":
            ligand = f"Ligand from {pdb_id}"
        
        results.append({
            'complex_id': complex_id,
            'pdb_id': pdb_id,
            'protein': protein,
            'ligand': ligand
        })
        
        time.sleep(0.5)  # Rate limiting
    
    # Write to file
    output_file = "protein_ligand_pairs.txt"
    with open(output_file, 'w') as f:
        f.write("Protein-Ligand Pairs from Dataset\n")
        f.write("=" * 60 + "\n\n")
        
        for item in results:
            f.write(f"Complex ID: {item['complex_id']}\n")
            f.write(f"PDB ID: {item['pdb_id']}\n")
            f.write(f"Protein: {item['protein']}\n")
            f.write(f"Ligand: {item['ligand']}\n")
            f.write("-" * 60 + "\n")
        
        # Also write a simple list format
        f.write("\n\nSimple Format (for copy-paste):\n")
        f.write("=" * 60 + "\n")
        for item in results:
            f.write(f"{item['protein']} | {item['ligand']}\n")
    
    print(f"\n✓ Saved {len(results)} protein-ligand pairs to {output_file}")
    print(f"\nSample entries:")
    for item in results[:5]:
        print(f"  - {item['protein']} + {item['ligand']} ({item['pdb_id']})")


if __name__ == "__main__":
    main()
