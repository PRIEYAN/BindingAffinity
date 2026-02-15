"""
Gemini API service for complex ID selection
"""

import os
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    try:
        from google import generativeai as genai
        GEMINI_AVAILABLE = True
    except ImportError:
        GEMINI_AVAILABLE = False


def get_gemini_api_key() -> Optional[str]:
    """Load GEMINI_API_KEY from .env file or environment"""
    if os.path.exists('.env'):
        with open('.env', 'r') as f:
            for line in f:
                if line.startswith('GEMINI_API_KEY='):
                    return line.split('=', 1)[1].strip().strip('"').strip("'")
    return os.getenv('GEMINI_API_KEY')


def select_best_complex(protein: str, ligand: str, pdb_ids: List[str]) -> Optional[str]:
    """
    Use Gemini 1.5 Pro to select the best matching complex from RCSB results.
    Returns complex ID in format: <pdbid>/<pdbid>_cplx.pdb
    """
    if not GEMINI_AVAILABLE:
        raise ImportError("Google Generative AI not installed")
    
    if not pdb_ids:
        return None
    
    api_key = get_gemini_api_key()
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found")
    
    genai.configure(api_key=api_key)
    # Try different model names
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
    except:
        try:
            model = genai.GenerativeModel('gemini-1.5-pro')
        except:
            model = genai.GenerativeModel('gemini-pro')
    
    prompt = f"""Given protein name "{protein}" and ligand name "{ligand}", select the most experimentally validated RCSB PDB complex where both coexist.

Available PDB IDs: {', '.join(pdb_ids[:20])}

Important:
- Select the complex that best matches both the protein and ligand
- Prefer experimentally determined structures (X-ray, NMR)
- Return ONLY the complex ID in format: <pdbid>/<pdbid>_cplx.pdb
- No explanation text, just the complex ID

Example output: 1abc/1abc_cplx.pdb"""

    try:
        response = model.generate_content(prompt)
        result = response.text.strip()
        
        # Clean response
        result = result.replace('```', '').replace('json', '').strip()
        if '\n' in result:
            result = result.split('\n')[0].strip()
        
        # Validate format
        if '/' in result and '_cplx.pdb' in result:
            pdb_id = result.split('/')[0]
            if pdb_id.lower() in [pid.lower() for pid in pdb_ids]:
                return result
        
        # Try to construct if just PDB ID
        if len(result) == 4 and result.isalnum() and result.lower() in [pid.lower() for pid in pdb_ids]:
            return f"{result.lower()}/{result.lower()}_cplx.pdb"
        
        # Retry once if format invalid
        response = model.generate_content(prompt + "\n\nIMPORTANT: Return format must be exactly: <pdbid>/<pdbid>_cplx.pdb")
        result = response.text.strip().replace('```', '').strip()
        if '/' in result and '_cplx.pdb' in result:
            pdb_id = result.split('/')[0]
            if pdb_id.lower() in [pid.lower() for pid in pdb_ids]:
                return result
        
        # Fallback to first result
        return f"{pdb_ids[0]}/{pdb_ids[0]}_cplx.pdb"
        
    except Exception as e:
        print(f"Gemini error: {e}")
        # Fallback to first result
        return f"{pdb_ids[0]}/{pdb_ids[0]}_cplx.pdb" if pdb_ids else None
