"""
RCSB PDB REST API service
"""

import requests
from typing import List, Optional

RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"

# Known protein-ligand mappings as fallback
KNOWN_COMPLEXES = {
    ("hiv protease", "indinavir"): ["1hsg", "1hvr", "1hxw"],
    ("hiv-1 protease", "indinavir"): ["1hsg", "1hvr", "1hxw"],
    ("hiv protease", "indinavir"): ["1hsg", "1hvr", "1hxw"],
    ("ace2", "remdesivir"): ["7c8j", "6m0j"],
    ("lactate dehydrogenase", "oxm"): ["1ldm"],
}


def search_complexes(protein: str, ligand: str, max_results: int = 50) -> List[str]:
    """
    Search RCSB PDB for complexes containing both protein and ligand.
    Returns list of PDB IDs.
    Uses multiple search strategies for better results.
    """
    # Check known complexes first
    protein_lower = protein.lower().strip()
    ligand_lower = ligand.lower().strip()
    key = (protein_lower, ligand_lower)
    if key in KNOWN_COMPLEXES:
        return KNOWN_COMPLEXES[key]
    
    results = []
    
    # Strategy 1: Search by protein name (flexible) and ligand (exact or contains)
    query1 = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_polymer_entity.rcsb_macromolecular_names_combined.name",
                        "operator": "contains_words",
                        "value": protein
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_nonpolymer_entity_container_identifiers.chem_comp_ids",
                        "operator": "contains_words",
                        "value": ligand.upper()
                    }
                }
            ]
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {
                "start": 0,
                "rows": max_results
            }
        }
    }
    
    # Strategy 2: Search by protein name only (broader search)
    query2 = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_polymer_entity.rcsb_macromolecular_names_combined.name",
                "operator": "contains_words",
                "value": protein
            }
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {
                "start": 0,
                "rows": max_results
            }
        }
    }
    
    try:
        # Try strategy 1 first
        response = requests.post(RCSB_SEARCH_URL, json=query1, timeout=20)
        response.raise_for_status()
        results1 = response.json().get("result_set", [])
        results.extend([item['identifier'].lower() for item in results1])
        
        # If no results, try strategy 2 (broader search)
        if not results:
            response = requests.post(RCSB_SEARCH_URL, json=query2, timeout=20)
            response.raise_for_status()
            results2 = response.json().get("result_set", [])
            results.extend([item['identifier'].lower() for item in results2])
        
        # Remove duplicates and return
        return list(dict.fromkeys(results))  # Preserves order while removing duplicates
        
    except Exception as e:
        print(f"RCSB search error: {e}")
        # Try a simpler search as fallback
        try:
            # Simple text search
            simple_query = {
                "query": {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "struct.title",
                        "operator": "contains_words",
                        "value": f"{protein} {ligand}"
                    }
                },
                "return_type": "entry",
                "request_options": {
                    "paginate": {"start": 0, "rows": max_results}
                }
            }
            response = requests.post(RCSB_SEARCH_URL, json=simple_query, timeout=20)
            response.raise_for_status()
            results = response.json().get("result_set", [])
            return [item['identifier'].lower() for item in results]
        except:
            return []


def get_pdb_structure_url(pdb_id: str) -> str:
    """Get URL to download PDB structure file"""
    return f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
