#!/usr/bin/env python3
"""
Get exact protein-ligand names for complex IDs in your dataset
Run this script when you have network access to query RCSB PDB
"""

import csv
import requests
import time
import json

def get_pdb_details(pdb_id: str):
    """Get detailed protein and ligand information from RCSB"""
    try:
        # Get entry info
        entry_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id.upper()}"
        entry_resp = requests.get(entry_url, timeout=10)
        
        if entry_resp.status_code != 200:
            return None, None, None
        
        entry_data = entry_resp.json()
        title = entry_data.get('struct', {}).get('title', '')
        
        # Get protein name from polymer entity
        protein = None
        try:
            polymer_url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id.upper()}/1"
            polymer_resp = requests.get(polymer_url, timeout=10)
            if polymer_resp.status_code == 200:
                polymer_data = polymer_resp.json()
                # Try multiple sources for protein name
                entity_source = polymer_data.get('rcsb_entity_source_organism', [{}])
                if entity_source and len(entity_source) > 0:
                    protein = entity_source[0].get('ncbi_scientific_name', '')
                
                if not protein:
                    # Try entity name
                    entity_name = polymer_data.get('rcsb_polymer_entity', {}).get('entity_poly', {})
                    protein = entity_name.get('rcsb_entity_poly_type', '')
                
                if not protein:
                    # Extract from title
                    if 'with' in title.lower():
                        protein = title.split('with')[0].strip()
                    else:
                        protein = title.split('-')[0].strip() if '-' in title else title[:50]
        except:
            # Fallback to title
            if 'with' in title.lower():
                protein = title.split('with')[0].strip()
            else:
                protein = title[:50] if title else pdb_id
        
        # Get ligand info - try multiple methods
        ligand = None
        
        # Method 1: Try nonpolymer entity endpoint
        try:
            nonpoly_url = f"https://data.rcsb.org/rest/v1/core/nonpolymer_entity/{pdb_id.upper()}"
            nonpoly_resp = requests.get(nonpoly_url, timeout=10)
            if nonpoly_resp.status_code == 200:
                nonpoly_data = nonpoly_resp.json()
                if isinstance(nonpoly_data, list) and len(nonpoly_data) > 0:
                    for entity in nonpoly_data:
                        chem_ids = entity.get('rcsb_nonpolymer_entity_container_identifiers', {}).get('chem_comp_ids', [])
                        if chem_ids:
                            ligand = chem_ids[0]
                            break
                elif isinstance(nonpoly_data, dict):
                    chem_ids = nonpoly_data.get('rcsb_nonpolymer_entity_container_identifiers', {}).get('chem_comp_ids', [])
                    if chem_ids:
                        ligand = chem_ids[0]
        except Exception as e:
            pass
        
        # Method 2: Try to extract from title if "with" is present (do this FIRST for better results)
        if not ligand and 'with' in title.lower():
            parts = title.split('with')
            if len(parts) > 1:
                potential_ligand = parts[1].strip()
                # Clean up common suffixes
                potential_ligand = potential_ligand.split(',')[0].split(';')[0].split('bound')[0].split('in complex')[0].strip()
                # Remove common words that aren't ligand names
                potential_ligand = potential_ligand.replace('inhibitor', '').replace('complex', '').replace('bound', '').strip()
                if len(potential_ligand) > 0 and len(potential_ligand) < 50:
                    # Check if it's not just a protein name
                    if potential_ligand.lower() not in ['protein', 'peptide', 'fragment', 'domain', 'subunit']:
                        ligand = potential_ligand
        
        # Method 3: Try ligand_expo endpoint
        if not ligand:
            try:
                ligand_url = f"https://data.rcsb.org/rest/v1/core/ligand/{pdb_id.upper()}"
                ligand_resp = requests.get(ligand_url, timeout=10)
                if ligand_resp.status_code == 200:
                    ligand_data = ligand_resp.json()
                    if isinstance(ligand_data, list) and len(ligand_data) > 0:
                        ligand = ligand_data[0].get('chemicalId', None)
                    elif isinstance(ligand_data, dict):
                        ligand = ligand_data.get('chemicalId', None)
            except:
                pass
        
        # Method 4: Try to get ligand from structure summary
        if not ligand:
            try:
                summary_url = f"https://data.rcsb.org/rest/v1/holdings/entry/{pdb_id.upper()}"
                summary_resp = requests.get(summary_url, timeout=10)
                if summary_resp.status_code == 200:
                    summary_data = summary_resp.json()
                    # Look for ligand information in various fields
                    if 'ligands' in summary_data:
                        ligands_list = summary_data['ligands']
                        if ligands_list and len(ligands_list) > 0:
                            ligand = ligands_list[0].get('chemicalId') or ligands_list[0].get('name')
            except:
                pass
        
        # If we got a chemical ID, try to get the common name
        if ligand and len(ligand) <= 3:  # Likely a chemical ID like "BEN", "ATP", etc.
            try:
                chem_url = f"https://data.rcsb.org/rest/v1/core/chemcomp/{ligand.upper()}"
                chem_resp = requests.get(chem_url, timeout=10)
                if chem_resp.status_code == 200:
                    chem_data = chem_resp.json()
                    # Try multiple name fields
                    ligand_name = (chem_data.get('name') or 
                                 chem_data.get('chem_comp', {}).get('name') or
                                 chem_data.get('three_letter_code') or
                                 ligand)
                    if ligand_name and ligand_name != ligand:
                        ligand = ligand_name
            except:
                pass
        
        # Clean protein name
        if protein:
            protein = protein.replace('Crystal structure of', '').replace('Structure of', '').strip()
            protein = protein.split(',')[0].split(';')[0].strip()
            if len(protein) > 60:
                protein = protein[:60]
        
        return protein, ligand, title
        
    except Exception as e:
        print(f"  Error: {e}")
        return None, None, None


def main():
    print("=" * 70)
    print("Getting Exact Protein-Ligand Names from RCSB PDB")
    print("=" * 70)
    print()
    
    # Read complex IDs from dataset
    complex_ids = []
    dataset_files = [
        "final_train_features_true.csv",
        "final_valid_features_true.csv",
        "test2013_features_true.csv",
        "coreset2016_features_true.csv"
    ]
    
    for csv_file in dataset_files:
        try:
            with open(csv_file, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for i, row in enumerate(reader):
                    if i >= 30:  # Limit per file
                        break
                    if row and row[0]:
                        cid = row[0].strip()
                        if cid and '/' in cid:
                            complex_ids.append(cid)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
    
    # Remove duplicates
    complex_ids = list(dict.fromkeys(complex_ids))
    print(f"Found {len(complex_ids)} unique complex IDs\n")
    
    # Get details for each
    results = []
    failed = []
    
    for i, complex_id in enumerate(complex_ids, 1):
        pdb_id = complex_id.split('/')[0]
        print(f"[{i}/{len(complex_ids)}] {pdb_id}...", end=' ')
        
        protein, ligand, title = get_pdb_details(pdb_id)
        
        # If we still don't have a ligand, try to extract from title one more time
        if protein and not ligand and title:
            # Look for common ligand patterns in title
            title_lower = title.lower()
            common_ligands = ['benzamidine', 'indinavir', 'saquinavir', 'ritonavir', 
                            'acetazolamide', 'methotrexate', 'staurosporine', 
                            'geldanamycin', 'estradiol', 'testosterone', 'dexamethasone',
                            'rosiglitazone', 'retinoic acid', 'calcitriol', 'remdesivir',
                            'rivaroxaban', 'apixaban', 'warfarin', 'heparin', 'aspirin',
                            'ibuprofen', 'naproxen', 'caffeine', 'atp', 'adp', 'nad',
                            'fad', 'coenzyme a', 'glucose', 'sucrose', 'lactose']
            for common_ligand in common_ligands:
                if common_ligand in title_lower:
                    ligand = common_ligand.title()
                    break
        
        if protein:
            results.append({
                'complex_id': complex_id,
                'pdb_id': pdb_id,
                'protein': protein,
                'ligand': ligand if ligand else 'Ligand',
                'title': title
            })
            if ligand and ligand != 'Ligand':
                print(f"✓ {protein[:40]} + {ligand[:30]}")
            else:
                print(f"✓ {protein[:40]} (ligand unknown)")
        else:
            failed.append(complex_id)
            print("✗ Failed")
        
        time.sleep(0.3)  # Rate limiting
    
    # Write results
    output_file = "protein_ligand_pairs_exact.txt"
    with open(output_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("EXACT Protein-Ligand Pairs from Your Dataset\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total pairs found: {len(results)}\n")
        f.write(f"Failed to resolve: {len(failed)}\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("SIMPLE FORMAT (for copy-paste testing):\n")
        f.write("=" * 70 + "\n\n")
        
        for item in results:
            f.write(f"Protein: {item['protein']}\n")
            f.write(f"Ligand: {item['ligand']}\n")
            f.write("\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("DETAILED LIST:\n")
        f.write("=" * 70 + "\n\n")
        
        for item in results:
            f.write(f"Complex ID: {item['complex_id']}\n")
            f.write(f"PDB ID: {item['pdb_id']}\n")
            f.write(f"Protein: {item['protein']}\n")
            f.write(f"Ligand: {item['ligand']}\n")
            if item.get('title'):
                f.write(f"Title: {item['title']}\n")
            f.write("-" * 70 + "\n\n")
    
    print(f"\n✓ Saved {len(results)} pairs to {output_file}")
    print(f"\nFirst 10 pairs:")
    for item in results[:10]:
        print(f"  • {item['protein']} + {item['ligand']}")


if __name__ == "__main__":
    main()
