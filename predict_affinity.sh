#!/bin/bash

###############################################################################
# Protein-Ligand Binding Affinity Prediction Pipeline
# 
# This script orchestrates the complete workflow:
# 1. Asks user for protein and ligand in natural language
# 2. Uses Google Gemini API (via Node.js server) to extract and normalize names
# 3. Generates a complex ID
# 4. Feeds the complex ID to run.py for affinity prediction
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SERVER_PORT=${PORT:-3000}
SERVER_URL="http://localhost:${SERVER_PORT}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}Protein-Ligand Binding Affinity Prediction Pipeline${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo -e "${RED}ERROR: Node.js is not installed. Please install Node.js first.${NC}"
    exit 1
fi

# Check if npm packages are installed
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}Installing Node.js dependencies...${NC}"
    npm install
    echo ""
fi

# Check if GEMINI_API_KEY is set, or try to load from .env file
if [ -z "$GEMINI_API_KEY" ]; then
    # Try to load from .env file if it exists
    if [ -f ".env" ]; then
        export $(grep -v '^#' .env | xargs)
    fi
    
    # Check again after loading .env
    if [ -z "$GEMINI_API_KEY" ]; then
        echo -e "${RED}ERROR: GEMINI_API_KEY environment variable is not set${NC}"
        echo -e "${YELLOW}Please set it with: export GEMINI_API_KEY='your-api-key'${NC}"
        echo -e "${YELLOW}Or create a .env file with: GEMINI_API_KEY='your-api-key'${NC}"
        exit 1
    fi
fi

# Check if Python model files exist
if [ ! -d "models" ] || [ ! -f "models/best_model_xgboost.pkl" ]; then
    echo -e "${RED}ERROR: Trained model not found in 'models/' directory${NC}"
    echo -e "${YELLOW}Please run training first: python3 train_binding_affinity.py${NC}"
    exit 1
fi

# Start the Node.js server in the background
echo -e "${BLUE}Starting API server...${NC}"
cd "$SCRIPT_DIR"
node server.js > server.log 2>&1 &
SERVER_PID=$!

# Wait for server to be ready
echo -e "${YELLOW}Waiting for server to start...${NC}"
sleep 3

# Check if server is running
if ! curl -s "${SERVER_URL}/health" > /dev/null; then
    echo -e "${RED}ERROR: Server failed to start. Check server.log for details.${NC}"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi

echo -e "${GREEN}Server is running (PID: $SERVER_PID)${NC}"
echo ""

# Get user input
echo -e "${BLUE}Enter protein and ligand information in natural language:${NC}"
echo -e "${YELLOW}Example: 'Human ACE2 protein with Remdesivir ligand'${NC}"
echo -e "${YELLOW}Example: 'lactate dehydrogenase and ATP'${NC}"
echo ""
read -p "Your query: " USER_QUERY

if [ -z "$USER_QUERY" ]; then
    echo -e "${RED}ERROR: No query provided${NC}"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi

echo ""
echo -e "${BLUE}Processing query...${NC}"

    # Call the API to extract protein and ligand
    RESPONSE=$(curl -s -X POST "${SERVER_URL}/generate-complex-id" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"$(echo "$USER_QUERY" | sed "s/\"/\\\\\"/g")\"}")

# Check if API call was successful
if echo "$RESPONSE" | grep -q '"success":true'; then
    # Extract values from JSON response
    PROTEIN=$(echo "$RESPONSE" | grep -o '"protein":"[^"]*"' | cut -d'"' -f4)
    LIGAND=$(echo "$RESPONSE" | grep -o '"ligand":"[^"]*"' | cut -d'"' -f4)
    COMPLEX_ID=$(echo "$RESPONSE" | grep -o '"complexId":"[^"]*"' | cut -d'"' -f4)
    ORGANISM=$(echo "$RESPONSE" | grep -o '"organism":"[^"]*"' | cut -d'"' -f4 2>/dev/null || echo "Not specified")
    
    echo -e "${GREEN}✓ Extraction successful!${NC}"
    echo ""
    echo -e "${BLUE}Extracted Information:${NC}"
    echo -e "  Protein: ${GREEN}${PROTEIN}${NC}"
    if [ "$ORGANISM" != "Not specified" ] && [ -n "$ORGANISM" ] && [ "$ORGANISM" != "null" ]; then
        echo -e "  Organism: ${GREEN}${ORGANISM}${NC}"
    fi
    echo -e "  Ligand: ${GREEN}${LIGAND}${NC}"
    echo -e "  Complex ID: ${GREEN}${COMPLEX_ID}${NC}"
    echo ""
    
    # Stop the server
    kill $SERVER_PID 2>/dev/null || true
    
    # Run the Python prediction script
    echo -e "${BLUE}Running affinity prediction...${NC}"
    echo ""
    python3 run.py "$COMPLEX_ID"
    PREDICTION_EXIT_CODE=$?
    
    if [ $PREDICTION_EXIT_CODE -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✓ Prediction completed successfully!${NC}"
    else
        echo ""
        echo -e "${YELLOW}Note: Complex ID '${COMPLEX_ID}' was not found in the dataset.${NC}"
        echo -e "${YELLOW}This is expected for AI-generated complex IDs.${NC}"
        echo -e "${YELLOW}To predict for this complex, you would need to compute the 16,800 features first.${NC}"
    fi
    
else
    # API call failed
    ERROR_MSG=$(echo "$RESPONSE" | grep -o '"error":"[^"]*"' | cut -d'"' -f4 || echo "Unknown error")
    echo -e "${RED}✗ API call failed: ${ERROR_MSG}${NC}"
    echo -e "${YELLOW}Full response: ${RESPONSE}${NC}"
    
    # Stop the server
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi
