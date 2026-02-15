"""
Flask backend for protein-ligand binding affinity prediction
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import warnings
warnings.filterwarnings('ignore')

from rcsb_service import search_complexes, get_pdb_structure_url
from gemini_service import select_best_complex
from predictor import predict_affinity, load_model
from dataset_lookup import filter_pdb_ids_by_dataset, get_dataset_complex_ids

app = Flask(__name__)
CORS(app)

# Load model at startup
try:
    print("Loading XGBoost model...")
    load_model()
    print("Model loaded successfully")
except Exception as e:
    print(f"Warning: Could not load model: {e}")

# Load dataset complex IDs at startup
try:
    print("Loading dataset complex IDs...")
    dataset_ids = get_dataset_complex_ids()
    print(f"Loaded {len(dataset_ids)} complex IDs from dataset")
except Exception as e:
    print(f"Warning: Could not load dataset IDs: {e}")


@app.route("/predict", methods=["POST"])
def predict():
    """Main prediction endpoint"""
    try:
        data = request.get_json()
        protein = data.get("protein", "").strip()
        ligand = data.get("ligand", "").strip()
        
        if not protein or not ligand:
            return jsonify({
                "success": False,
                "error": "Both protein and ligand names are required"
            }), 400
        
        # Step 1: Search RCSB PDB
        print(f"Searching RCSB for: {protein} + {ligand}")
        pdb_ids = search_complexes(protein, ligand)
        
        if not pdb_ids:
            # Try with simplified names (remove common prefixes/suffixes)
            protein_simple = protein.replace("HIV-1 ", "").replace("HIV ", "").replace("-1", "").strip()
            ligand_simple = ligand.split()[0] if ligand else ligand
            
            if protein_simple != protein or ligand_simple != ligand:
                print(f"Retrying with simplified names: {protein_simple} + {ligand_simple}")
                pdb_ids = search_complexes(protein_simple, ligand_simple)
            
            if not pdb_ids:
                return jsonify({
                    "success": False,
                    "error": f"No complexes found in RCSB PDB for {protein} + {ligand}. Try different protein/ligand names."
                }), 404
        
        # Step 1.5: Filter to only complexes that exist in dataset
        print(f"Filtering {len(pdb_ids)} RCSB results to dataset complexes...")
        pdb_ids_filtered = filter_pdb_ids_by_dataset(pdb_ids)
        print(f"Found {len(pdb_ids_filtered)} complexes in dataset")
        
        if not pdb_ids_filtered:
            # Get some sample complex IDs from dataset for user reference
            dataset_ids = list(get_dataset_complex_ids())[:5]
            return jsonify({
                "success": False,
                "error": f"Found complexes in RCSB PDB, but none are in your trained dataset. The complex IDs found ({', '.join(pdb_ids[:5])}) are not in your dataset files. Try different protein/ligand names that match complexes in your dataset.",
                "sample_dataset_ids": [cid.split('/')[0] for cid in dataset_ids]
            }), 404
        
        # Step 2: Use Gemini to select best complex (only from dataset complexes)
        # Convert filtered PDB IDs to complex ID format and verify they exist in dataset
        dataset_ids = get_dataset_complex_ids()
        valid_complex_ids = []
        for pid in pdb_ids_filtered:
            complex_id_candidate = f"{pid}/{pid}_cplx.pdb"
            if complex_id_candidate in dataset_ids:
                valid_complex_ids.append(complex_id_candidate)
        
        if not valid_complex_ids:
            return jsonify({
                "success": False,
                "error": "No valid complexes found in dataset for this protein-ligand pair."
            }), 404
        
        print(f"Using Gemini to select from {len(valid_complex_ids)} validated dataset complexes")
        # Extract PDB IDs for Gemini
        pdb_ids_for_gemini = [cid.split('/')[0] for cid in valid_complex_ids]
        complex_id = select_best_complex(protein, ligand, pdb_ids_for_gemini)
        
        # Verify and fix complex_id format
        if complex_id:
            # Ensure it's in the correct format
            if '/' not in complex_id or '_cplx.pdb' not in complex_id:
                pdb_id = complex_id.split('/')[0] if '/' in complex_id else complex_id
                complex_id = f"{pdb_id}/{pdb_id}_cplx.pdb"
            
            # Final verification
            if complex_id not in dataset_ids:
                complex_id = valid_complex_ids[0]
                print(f"Selected complex not in dataset, using first valid: {complex_id}")
        else:
            # Fallback to first valid complex
            complex_id = valid_complex_ids[0]
            print(f"Using first available complex: {complex_id}")
        
        # Step 3: Validate complex exists in dataset and predict
        print(f"Validating complex ID: {complex_id}")
        affinity, success = predict_affinity(complex_id)
        
        if not success:
            return jsonify({
                "success": False,
                "error": f"Complex ID '{complex_id}' not found in trained dataset",
                "complex_id": complex_id
            }), 404
        
        # Step 4: Get PDB structure URL
        pdb_id = complex_id.split('/')[0]
        structure_url = get_pdb_structure_url(pdb_id)
        
        return jsonify({
            "success": True,
            "complex_id": complex_id,
            "affinity": affinity,
            "protein": protein,
            "ligand": ligand,
            "pdb_id": pdb_id,
            "structure_url": structure_url
        })
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check"""
    try:
        load_model()
        model_loaded = True
    except:
        model_loaded = False
    
    return jsonify({
        "status": "ok",
        "model_loaded": model_loaded
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
