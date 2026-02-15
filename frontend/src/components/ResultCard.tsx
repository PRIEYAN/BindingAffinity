/**
 * Result card component with 3D viewer and animated graph
 */

import { motion } from 'framer-motion';
import { PredictionResponse } from '../services/api';
import { MoleculeViewer } from './MoleculeViewer';
import { AffinityGraph } from './AffinityGraph';

interface ResultCardProps {
  result: PredictionResponse;
}

export function ResultCard({ result }: ResultCardProps) {
  if (!result.success) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass-card p-6"
      >
        <div className="flex items-center gap-3 text-red-600 dark:text-red-400">
          <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <h3 className="text-lg font-semibold">Error</h3>
        </div>
        <p className="mt-2 text-gray-600 dark:text-gray-400">{result.error}</p>
        {result.complex_id && (
          <p className="mt-2 text-sm text-gray-500 dark:text-gray-500">
            Complex ID: {result.complex_id}
          </p>
        )}
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="space-y-6"
    >
      {/* Results Summary */}
      <div className="glass-card p-6">
        <h3 className="text-xl font-bold mb-4 text-gray-900 dark:text-white">
          Prediction Results
        </h3>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <p className="text-sm text-gray-500 dark:text-gray-400">Protein</p>
            <p className="text-lg font-semibold text-gray-900 dark:text-white">{result.protein}</p>
          </div>
          <div>
            <p className="text-sm text-gray-500 dark:text-gray-400">Ligand</p>
            <p className="text-lg font-semibold text-gray-900 dark:text-white">{result.ligand}</p>
          </div>
          <div>
            <p className="text-sm text-gray-500 dark:text-gray-400">Complex ID</p>
            <p className="text-lg font-mono text-gray-900 dark:text-white">{result.complex_id}</p>
          </div>
          <div>
            <p className="text-sm text-gray-500 dark:text-gray-400">Binding Affinity</p>
            <p className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              {result.affinity?.toFixed(4)}
            </p>
          </div>
        </div>
      </div>

      {/* 3D Molecule Viewer */}
      {result.pdb_id && (
        <div className="glass-card p-6">
          <h3 className="text-xl font-bold mb-4 text-gray-900 dark:text-white">
            3D Structure Viewer
          </h3>
          <div className="h-96 rounded-xl overflow-hidden">
            <MoleculeViewer pdbId={result.pdb_id} structureUrl={result.structure_url} />
          </div>
        </div>
      )}

      {/* Animated Affinity Graph */}
      {result.affinity && (
        <div className="glass-card p-6">
          <h3 className="text-xl font-bold mb-4 text-gray-900 dark:text-white">
            Affinity Visualization
          </h3>
          <AffinityGraph affinity={result.affinity} />
        </div>
      )}
    </motion.div>
  );
}
