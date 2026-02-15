/**
 * API client for backend communication
 */

import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

export interface PredictionRequest {
  protein: string;
  ligand: string;
}

export interface PredictionResponse {
  success: boolean;
  complex_id?: string;
  affinity?: number;
  protein?: string;
  ligand?: string;
  pdb_id?: string;
  structure_url?: string;
  error?: string;
}

export const api = {
  async predict(protein: string, ligand: string): Promise<PredictionResponse> {
    try {
      const response = await axios.post<PredictionResponse>(
        `${API_BASE_URL}/predict`,
        { protein, ligand },
        {
          headers: { 'Content-Type': 'application/json' },
          timeout: 60000,
        }
      );
      return response.data;
    } catch (error: any) {
      if (error.response) {
        return error.response.data;
      } else if (error.request) {
        return {
          success: false,
          error: 'Cannot connect to server. Make sure the Flask backend is running.',
        };
      } else {
        return {
          success: false,
          error: error.message || 'An unexpected error occurred',
        };
      }
    }
  },

  async healthCheck(): Promise<boolean> {
    try {
      const response = await axios.get(`${API_BASE_URL}/health`, { timeout: 5000 });
      return response.status === 200;
    } catch {
      return false;
    }
  },
};
