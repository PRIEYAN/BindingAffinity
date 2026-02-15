/**
 * Animated affinity prediction graph
 */

import { motion } from 'framer-motion';
import { useEffect, useState } from 'react';

interface AffinityGraphProps {
  affinity: number;
}

export function AffinityGraph({ affinity }: AffinityGraphProps) {
  const [animatedValue, setAnimatedValue] = useState(0);

  useEffect(() => {
    const duration = 2000;
    const steps = 60;
    const increment = affinity / steps;
    let current = 0;
    let step = 0;

    const timer = setInterval(() => {
      step++;
      current += increment;
      if (step >= steps) {
        setAnimatedValue(affinity);
        clearInterval(timer);
      } else {
        setAnimatedValue(current);
      }
    }, duration / steps);

    return () => clearInterval(timer);
  }, [affinity]);

  // Normalize affinity to 0-100 scale (assuming range 0-15)
  const normalized = Math.min((animatedValue / 15) * 100, 100);
  const color = normalized > 70 ? 'from-green-500 to-emerald-500' : 
                normalized > 40 ? 'from-yellow-500 to-orange-500' : 
                'from-red-500 to-pink-500';

  return (
    <div className="space-y-4">
      {/* Bar Chart */}
      <div className="relative h-32 bg-gray-100/50 dark:bg-gray-800/50 rounded-xl p-4 overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${normalized}%` }}
          transition={{ duration: 2, ease: 'easeOut' }}
          className={`h-full bg-gradient-to-r ${color} rounded-lg shadow-lg`}
        />
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-3xl font-bold text-gray-900 dark:text-white drop-shadow-lg">
            {animatedValue.toFixed(2)}
          </span>
        </div>
      </div>

      {/* Scale Indicators */}
      <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400">
        <span>Low (0)</span>
        <span>Medium (7.5)</span>
        <span>High (15)</span>
      </div>

      {/* Interpretation */}
      <div className="text-center">
        <p className="text-sm text-gray-600 dark:text-gray-400">
          {normalized > 70 ? 'Strong Binding' : 
           normalized > 40 ? 'Moderate Binding' : 
           'Weak Binding'}
        </p>
      </div>
    </div>
  );
}
