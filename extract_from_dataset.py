"""
Extract protein-ligand pairs directly from dataset CSV files
This ensures all complex IDs are guaranteed to exist in the dataset
"""

import csv
import requests
import time
from typing import List, Dict, Optional

def get_pdb_info_simple(pdb_id: str) -> tuple:
    """Get basic info from RCSB PDB - simplified version"""
    try:
        # Get entry summary
        url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id.upper()}"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            title = data.get('struct', {}).get('title', '')
            
            # Try to extract protein and ligand from title
            protein = "Unknown"
            ligand = "Unknown"
            
            # Common patterns in PDB titles
            if 'with' in title.lower():
                parts = title.split('with')
                protein = parts[0].strip()[:60]
                ligand = parts[1].strip()[:40] if len(parts) > 1 else "Unknown"
            elif 'complex' in title.lower():
                parts = title.split('complex')
                protein = parts[0].strip()[:60]
            else:
                protein = title[:60] if title else pdb_id
            
            # Clean up protein name
            protein = protein.replace('Crystal structure of', '').replace('Structure of', '').strip()
            protein = protein.split(',')[0].split(';')[0].strip()
            
            return protein, ligand
    except Exception as e:
        pass
    
    return None, None


def extract_complex_ids_from_all_datasets(max_per_file: int = 20) -> List[str]:
    """Extract complex IDs from all dataset CSV files"""
    all_complex_ids = []
    
    dataset_files = [
        "final_train_features_true.csv",
        "final_valid_features_true.csv", 
        "test2013_features_true.csv",
        "coreset2016_features_true.csv"
    ]
    
    for csv_file in dataset_files:
        try:
            complex_ids = []
            with open(csv_file, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for i, row in enumerate(reader):
                    if i >= max_per_file:
                        break
                    if row and len(row) > 0:
                        complex_id = row[0].strip()
                        if complex_id and '/' in complex_id:
                            complex_ids.append(complex_id)
            
            print(f"Found {len(complex_ids)} IDs in {csv_file}")
            all_complex_ids.extend(complex_ids)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_ids = []
    for cid in all_complex_ids:
        if cid not in seen:
            seen.add(cid)
            unique_ids.append(cid)
    
    return unique_ids


def main():
    print("=" * 70)
    print("Extracting Protein-Ligand Pairs from Dataset")
    print("=" * 70)
    print()
    
    # Extract complex IDs from dataset
    print("Step 1: Reading dataset CSV files...")
    complex_ids = extract_complex_ids_from_all_datasets(max_per_file=25)
    print(f"Found {len(complex_ids)} unique complex IDs in dataset\n")
    
    # Get protein-ligand info for each
    print("Step 2: Fetching protein and ligand names from RCSB PDB...")
    print("(This may take a few minutes)\n")
    
    results = []
    failed = []
    
    for i, complex_id in enumerate(complex_ids, 1):
        pdb_id = complex_id.split('/')[0]
        print(f"[{i}/{len(complex_ids)}] {pdb_id}...", end=' ')
        
        protein, ligand = get_pdb_info_simple(pdb_id)
        
        if protein and protein != "Unknown":
            # If ligand is unknown, try to infer from common patterns
            if ligand == "Unknown":
                # Try to get ligand from non-polymer entities
                try:
                    url = f"https://data.rcsb.org/rest/v1/core/nonpolymer_entity/{pdb_id.upper()}"
                    resp = requests.get(url, timeout=5)
                    if resp.status_code == 200:
                        data = resp.json()
                        if data and len(data) > 0:
                            chem_ids = data[0].get('rcsb_nonpolymer_entity_container_identifiers', {}).get('chem_comp_ids', [])
                            if chem_ids:
                                ligand = chem_ids[0]
                except:
                    pass
            
            results.append({
                'complex_id': complex_id,
                'pdb_id': pdb_id,
                'protein': protein,
                'ligand': ligand if ligand != "Unknown" else "Ligand"
            })
            print(f"✓ {protein[:40]} + {ligand[:20]}")
        else:
            failed.append(complex_id)
            print("✗ Failed")
        
        time.sleep(0.2)  # Rate limiting
    
    print(f"\n✓ Successfully fetched {len(results)} pairs")
    print(f"✗ Failed to fetch {len(failed)} pairs\n")
    
    # Write to file
    output_file = "protein_ligand_pairs_from_dataset.txt"
    with open(output_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("Protein-Ligand Pairs GUARANTEED to be in Your Dataset\n")
        f.write("=" * 70 + "\n\n")
        f.write("These complex IDs are extracted directly from your dataset CSV files.\n")
        f.write("All pairs below are guaranteed to exist in your training data.\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("SIMPLE FORMAT (for copy-paste testing):\n")
        f.write("=" * 70 + "\n\n")
        
        for item in results:
            f.write(f"Protein: {item['protein']}\n")
            f.write(f"Ligand: {item['ligand']}\n")
            f.write("\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("DETAILED LIST WITH COMPLEX IDs:\n")
        f.write("=" * 70 + "\n\n")
        
        for item in results:
            f.write(f"Complex ID: {item['complex_id']}\n")
            f.write(f"PDB ID: {item['pdb_id']}\n")
            f.write(f"Protein: {item['protein']}\n")
            f.write(f"Ligand: {item['ligand']}\n")
            f.write("-" * 70 + "\n\n")
        
        if failed:
            f.write("\n" + "=" * 70 + "\n")
            f.write("Complex IDs that couldn't be resolved:\n")
            f.write("=" * 70 + "\n")
            for cid in failed[:20]:  # Show first 20
                f.write(f"{cid}\n")
    
    print(f"✓ Saved to {output_file}")
    print(f"\nSample pairs (first 10):")
    for item in results[:10]:
        print(f"  • {item['protein'][:50]} + {item['ligand'][:30]}")


if __name__ == "__main__":
    main()
