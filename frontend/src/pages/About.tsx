/**
 * About page component
 */

export function About() {
  return (
    <div className="glass-card p-6">
      <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">
        About Binding Affinity AI
      </h1>
      <div className="space-y-4 text-gray-600 dark:text-gray-400">
        <p>
          This application uses machine learning to predict protein-ligand binding affinity.
        </p>
        <p>
          The model is trained on experimental data and uses XGBoost for predictions.
        </p>
        <p>
          Powered by RCSB PDB, Google Gemini AI, and advanced molecular modeling.
        </p>
      </div>
    </div>
  );
}
