/**
 * Express REST API Server
 * Exposes endpoint to extract protein and ligand from natural language using Google Gemini API
 */

const express = require('express');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const app = express();

// Middleware
app.use(express.json());

// Validate environment variable
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
if (!GEMINI_API_KEY) {
    console.error('ERROR: GEMINI_API_KEY environment variable is not set');
    process.exit(1);
}

// Initialize Gemini
const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);

/**
 * Helper function to extract protein and ligand using Gemini NLP
 * @param {string} query - Natural language query from user
 * @returns {Promise<Object>} - Extracted protein, organism, and ligand information
 */
async function extractProteinLigand(query) {
    // Try newer model names, fallback to older ones
    let model;
    try {
        model = genAI.getGenerativeModel({ model: 'gemini-1.5-flash' });
    } catch (e) {
        try {
            model = genAI.getGenerativeModel({ model: 'gemini-1.5-pro' });
        } catch (e2) {
            model = genAI.getGenerativeModel({ model: 'gemini-pro' });
        }
    }
    
    const prompt = `You are a bioinformatics expert. Extract protein and ligand information from the following natural language query.

Query: "${query}"

Return ONLY a valid JSON object with this exact structure (no markdown, no code blocks, just pure JSON):
{
  "protein": "standard protein name (e.g., ACE2, lactase dehydrogenase, insulin)",
  "organism": "scientific organism name if mentioned (e.g., Homo sapiens, Escherichia coli) or null",
  "ligand": "standard ligand/drug name (e.g., Remdesivir, ATP, glucose)"
}

Rules:
- Use standard scientific names (e.g., "ACE2" not "ace2 protein")
- If organism is not mentioned, set "organism" to null
- Normalize protein names to common abbreviations or standard names
- Normalize ligand names to standard drug/compound names
- If information is missing, use null
- Return ONLY the JSON, nothing else`;

    try {
        const result = await model.generateContent(prompt);
        const response = await result.response;
        const text = response.text();
        
        // Clean the response - remove markdown code blocks if present
        let jsonText = text.trim();
        if (jsonText.startsWith('```json')) {
            jsonText = jsonText.replace(/```json\n?/g, '').replace(/```\n?/g, '').trim();
        } else if (jsonText.startsWith('```')) {
            jsonText = jsonText.replace(/```\n?/g, '').trim();
        }
        
        // Parse JSON
        const extracted = JSON.parse(jsonText);
        
        // Validate required fields
        if (!extracted.protein || !extracted.ligand) {
            throw new Error('Missing required fields: protein and ligand are required');
        }
        
        return {
            protein: extracted.protein.trim(),
            organism: extracted.organism ? extracted.organism.trim() : null,
            ligand: extracted.ligand.trim()
        };
    } catch (error) {
        console.error('API error:', error);
        // Don't expose internal API details to user
        const errorMsg = error.message || String(error);
        if (errorMsg.includes('404') || errorMsg.includes('not found')) {
            throw new Error('Processing service temporarily unavailable. Please try again.');
        }
        throw new Error(`Failed to extract protein/ligand information. Please rephrase your query.`);
    }
}

/**
 * Generate deterministic complex ID from protein and ligand
 * @param {string} protein - Protein name
 * @param {string} ligand - Ligand name
 * @returns {string} - Complex ID in format: <protein>_<ligand>_cplx
 */
function generateComplexId(protein, ligand) {
    // Normalize: remove spaces, special chars, convert to uppercase
    const normalize = (str) => str
        .replace(/[^a-zA-Z0-9]/g, '_')
        .replace(/_+/g, '_')
        .replace(/^_|_$/g, '')
        .toUpperCase();
    
    const proteinNorm = normalize(protein);
    const ligandNorm = normalize(ligand);
    
    return `${proteinNorm}_${ligandNorm}_cplx`;
}

/**
 * POST /generate-complex-id
 * Extracts protein and ligand from natural language query and generates complex ID
 */
app.post('/generate-complex-id', async (req, res) => {
    try {
        // Input validation
        const { query } = req.body;
        
        if (!query || typeof query !== 'string' || query.trim().length === 0) {
            return res.status(400).json({
                success: false,
                error: 'Invalid request: "query" field is required and must be a non-empty string'
            });
        }
        
        // Extract protein and ligand using Gemini
        const extracted = await extractProteinLigand(query.trim());
        
        // Generate complex ID
        const complexId = generateComplexId(extracted.protein, extracted.ligand);
        
        // Return response
        return res.json({
            success: true,
            protein: extracted.protein,
            organism: extracted.organism,
            ligand: extracted.ligand,
            complexId: complexId,
            source: 'AI-generated (not an official PDB ID)'
        });
        
    } catch (error) {
        console.error('Error in /generate-complex-id:', error);
        return res.status(500).json({
            success: false,
            error: error.message || 'Internal server error'
        });
    }
});

/**
 * Health check endpoint
 */
app.get('/health', (req, res) => {
    res.json({ status: 'ok', service: 'protein-ligand-complex-id-generator' });
});

// Start server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
    console.log(`GEMINI_API_KEY: ${GEMINI_API_KEY ? 'Set ✓' : 'Missing ✗'}`);
});
