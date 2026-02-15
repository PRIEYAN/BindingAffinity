/**
 * Predict page component
 */

import { useState } from 'react';
import { PredictionForm } from '../components/PredictionForm';
import { ResultCard } from '../components/ResultCard';
import { api, PredictionResponse } from '../services/api';

export function Predict() {
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handlePredict = async (protein: string, ligand: string) => {
    setIsLoading(true);
    setResult(null);

    try {
      const response = await api.predict(protein, ligand);
      setResult(response);
    } catch (error) {
      setResult({
        success: false,
        error: 'An unexpected error occurred',
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header Card */}
      <div className="glass-card p-6">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
          Predict Binding Affinity
        </h1>
        <p className="text-gray-600 dark:text-gray-400">
          Enter protein and ligand names to predict their binding affinity using AI-powered analysis
        </p>
      </div>

      {/* Prediction Form Card */}
      <div className="glass-card p-6">
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          Input Parameters
        </h2>
        <PredictionForm onSubmit={handlePredict} isLoading={isLoading} />
      </div>

      {/* Results */}
      {result && <ResultCard result={result} />}
    </div>
  );
}
