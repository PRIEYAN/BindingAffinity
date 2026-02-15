"""
Extract protein and ligand names for complexes in the dataset
Uses RCSB PDB API to get information
"""

import csv
import requests
import time
import json

def get_pdb_info(pdb_id: str):
    """Get protein and ligand information from RCSB PDB"""
    try:
        # Get entry information
        entry_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id.upper()}"
        response = requests.get(entry_url, timeout=10)
        
        if response.status_code != 200:
            return None, None
        
        entry_data = response.json()
        title = entry_data.get('struct', {}).get('title', '')
        
        # Get polymer entity (protein) information
        protein = "Unknown"
        try:
            polymer_url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id.upper()}/1"
            polymer_resp = requests.get(polymer_url, timeout=10)
            if polymer_resp.status_code == 200:
                polymer_data = polymer_resp.json()
                # Try to get protein name
                entity_name = polymer_data.get('rcsb_entity_source_organism', [{}])
                if entity_name and len(entity_name) > 0:
                    protein = entity_name[0].get('ncbi_scientific_name', '')
                if not protein:
                    # Try entity name
                    protein = polymer_data.get('rcsb_polymer_entity', {}).get('entity_poly', {}).get('rcsb_entity_poly_type', '')
        except:
            pass
        
        # Extract from title if still unknown
        if protein == "Unknown" or not protein:
            if 'with' in title.lower():
                parts = title.split('with')
                protein = parts[0].strip()[:50]
            elif '-' in title:
                protein = title.split('-')[0].strip()[:50]
            else:
                protein = title[:50] if title else pdb_id
        
        # Get ligand information (non-polymer entities)
        ligands = []
        try:
            nonpoly_url = f"https://data.rcsb.org/rest/v1/core/nonpolymer_entity/{pdb_id.upper()}"
            nonpoly_resp = requests.get(nonpoly_url, timeout=10)
            if nonpoly_resp.status_code == 200:
                nonpoly_data = nonpoly_resp.json()
                for entity in nonpoly_data:
                    chem_comp_id = entity.get('rcsb_nonpolymer_entity_container_identifiers', {}).get('chem_comp_ids', [])
                    if chem_comp_id:
                        ligands.extend(chem_comp_id)
        except:
            pass
        
        ligand = ligands[0] if ligands else "Unknown"
        
        return protein, ligand
        
    except Exception as e:
        print(f"Error for {pdb_id}: {e}")
        return None, None


def extract_complex_ids(csv_file: str, max_ids: int = 30):
    """Extract complex IDs from CSV file"""
    complex_ids = []
    try:
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for i, row in enumerate(reader):
                if i >= max_ids:
                    break
                if row:
                    complex_id = row[0].strip()
                    if complex_id and '/' in complex_id:
                        complex_ids.append(complex_id)
    except Exception as e:
        print(f"Error reading CSV: {e}")
    
    return complex_ids


def main():
    print("Extracting protein-ligand pairs from dataset...")
    
    # Extract complex IDs
    complex_ids = extract_complex_ids("final_train_features_true.csv", max_ids=30)
    print(f"Found {len(complex_ids)} complex IDs")
    
    results = []
    
    for i, complex_id in enumerate(complex_ids, 1):
        pdb_id = complex_id.split('/')[0]
        print(f"[{i}/{len(complex_ids)}] Fetching info for {pdb_id}...")
        
        protein, ligand = get_pdb_info(pdb_id)
        
        if protein and ligand:
            results.append({
                'complex_id': complex_id,
                'pdb_id': pdb_id,
                'protein': protein,
                'ligand': ligand
            })
            print(f"  ✓ {protein} + {ligand}")
        else:
            print(f"  ✗ Could not fetch info")
        
        time.sleep(0.3)  # Rate limiting
    
    # Write to text file
    output_file = "protein_ligand_pairs.txt"
    with open(output_file, 'w') as f:
        f.write("Protein-Ligand Pairs Available in Dataset\n")
        f.write("=" * 70 + "\n\n")
        
        for item in results:
            f.write(f"Complex ID: {item['complex_id']}\n")
            f.write(f"PDB ID: {item['pdb_id']}\n")
            f.write(f"Protein: {item['protein']}\n")
            f.write(f"Ligand: {item['ligand']}\n")
            f.write("-" * 70 + "\n\n")
        
        # Simple format for easy copy-paste
        f.write("\n\n" + "=" * 70 + "\n")
        f.write("SIMPLE FORMAT (for testing):\n")
        f.write("=" * 70 + "\n\n")
        for item in results:
            f.write(f"Protein: {item['protein']}\n")
            f.write(f"Ligand: {item['ligand']}\n")
            f.write("\n")
    
    print(f"\n✓ Saved {len(results)} pairs to {output_file}")
    print(f"\nSample pairs:")
    for item in results[:5]:
        print(f"  • {item['protein']} + {item['ligand']}")


if __name__ == "__main__":
    main()
