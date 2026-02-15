# Binding Affinity AI - Premium macOS Dashboard

A modern, premium web application for predicting protein-ligand binding affinity using machine learning, featuring a beautiful macOS Sonoma-style interface with 3D molecule visualization.

## Features

✨ **Premium macOS Design**
- Glassmorphism UI with frosted glass effects
- Smooth animations and transitions
- Dark mode support
- Inter font typography
- Rounded corners and soft shadows

🔬 **Advanced Features**
- 3D molecule viewer (Three.js) with PDB structure rendering
- Animated affinity prediction graph
- RCSB PDB integration for structure search
- Gemini AI for intelligent complex selection
- XGBoost machine learning model

📊 **Dashboard**
- Sidebar navigation
- Prediction form with validation
- Results display with visualizations
- History tracking (coming soon)

## Tech Stack

**Backend:**
- Flask (Python)
- XGBoost (ML Model)
- Google Gemini AI
- RCSB PDB API

**Frontend:**
- React + TypeScript
- TailwindCSS
- Framer Motion
- Three.js
- Axios

## Quick Start

### 1. Backend Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set Gemini API key
export GEMINI_API_KEY='your-api-key-here'

# Run backend
python app.py
```

Backend runs on `http://localhost:5000`

### 2. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

Frontend runs on `http://localhost:5173`

### 3. Open Browser

Navigate to `http://localhost:5173`

## Project Structure

```
.
├── app.py                 # Flask backend
├── rcsb_service.py        # RCSB PDB API integration
├── gemini_service.py      # Gemini AI integration
├── predictor.py           # XGBoost model loading & prediction
├── models/                # Trained model files
├── final_train_features_true.csv  # Training dataset
├── frontend/
│   ├── src/
│   │   ├── components/    # React components
│   │   ├── pages/         # Page components
│   │   ├── services/      # API client
│   │   └── App.tsx        # Main app
│   └── package.json
└── requirements.txt
```

## API Endpoints

### POST `/predict`

Predict binding affinity for protein-ligand pair.

**Request:**
```json
{
  "protein": "ACE2",
  "ligand": "Remdesivir"
}
```

**Response:**
```json
{
  "success": true,
  "complex_id": "3rqw/3rqw_cplx.pdb",
  "affinity": 6.2345,
  "protein": "ACE2",
  "ligand": "Remdesivir",
  "pdb_id": "3rqw",
  "structure_url": "https://files.rcsb.org/download/3RQW.pdb"
}
```

## Environment Variables

**Backend:**
- `GEMINI_API_KEY` - Google Gemini API key (required)

**Frontend:**
- `VITE_API_URL` - Backend API URL (default: http://localhost:5000)

## Deployment

See `DEPLOYMENT.md` for AWS deployment instructions.

## Screenshots

The application features:
- Clean, modern macOS-style interface
- 3D interactive molecule viewer
- Animated prediction visualizations
- Smooth dark mode transitions

## License

MIT
