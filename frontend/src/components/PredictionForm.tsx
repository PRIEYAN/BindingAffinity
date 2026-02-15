/**
 * Prediction form component
 */

import { useState, FormEvent } from 'react';

interface PredictionFormProps {
  onSubmit: (protein: string, ligand: string) => void;
  isLoading: boolean;
}

export function PredictionForm({ onSubmit, isLoading }: PredictionFormProps) {
  const [protein, setProtein] = useState('');
  const [ligand, setLigand] = useState('');

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (protein.trim() && ligand.trim() && !isLoading) {
      onSubmit(protein.trim(), ligand.trim());
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div className="space-y-4">
        <div>
          <label htmlFor="protein" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Protein Name
          </label>
          <input
            id="protein"
            type="text"
            value={protein}
            onChange={(e) => setProtein(e.target.value)}
            placeholder="e.g., ACE2, HIV protease, lactase dehydrogenase"
            className="input-field"
            disabled={isLoading}
            required
          />
        </div>

        <div>
          <label htmlFor="ligand" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Ligand Name
          </label>
          <input
            id="ligand"
            type="text"
            value={ligand}
            onChange={(e) => setLigand(e.target.value)}
            placeholder="e.g., Remdesivir, Indinavir, ATP"
            className="input-field"
            disabled={isLoading}
            required
          />
        </div>
      </div>

      <button
        type="submit"
        className="btn-primary w-full"
        disabled={isLoading || !protein.trim() || !ligand.trim()}
      >
        {isLoading ? (
          <span className="flex items-center justify-center gap-2">
            <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
            Processing...
          </span>
        ) : (
          'Predict Affinity'
        )}
      </button>
    </form>
  );
}
