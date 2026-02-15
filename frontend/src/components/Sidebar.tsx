/**
 * Sidebar navigation component
 */

import { Link, useLocation } from 'react-router-dom';

interface SidebarProps {
  darkMode: boolean;
  onToggleDarkMode: () => void;
}

export function Sidebar({ darkMode, onToggleDarkMode }: SidebarProps) {
  const location = useLocation();

  const navItems = [
    { path: '/', label: 'Predict', icon: '🔬' },
    { path: '/history', label: 'History', icon: '📊' },
    { path: '/about', label: 'About', icon: 'ℹ️' },
  ];

  return (
    <aside className="glass-sidebar fixed left-0 top-0 h-full w-64 p-6 z-10">
      <div className="flex flex-col h-full">
        {/* Logo/Title */}
        <div className="mb-8">
          <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-blue-800 dark:from-blue-400 dark:to-blue-600 bg-clip-text text-transparent">
            Binding Affinity AI
          </h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
            Protein-Ligand Prediction
          </p>
        </div>

        {/* Navigation */}
        <nav className="flex-1 space-y-2">
          {navItems.map((item) => {
            const isActive = location.pathname === item.path;
            return (
              <Link
                key={item.path}
                to={item.path}
                className={`
                  flex items-center gap-3 px-4 py-3 rounded-xl
                  transition-all duration-200
                  ${
                    isActive
                      ? 'bg-blue-500/10 text-blue-600 dark:text-blue-400 shadow-md'
                      : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100/50 dark:hover:bg-gray-800/50'
                  }
                `}
              >
                <span className="text-xl">{item.icon}</span>
                <span className="font-medium">{item.label}</span>
              </Link>
            );
          })}
        </nav>

        {/* Dark Mode Toggle */}
        <div className="mt-auto pt-6 border-t border-gray-200/50 dark:border-gray-700/50">
          <button
            onClick={onToggleDarkMode}
            className="w-full flex items-center justify-between px-4 py-3 rounded-xl bg-gray-100/50 dark:bg-gray-800/50 hover:bg-gray-200/50 dark:hover:bg-gray-700/50 transition-colors"
          >
            <span className="font-medium text-gray-700 dark:text-gray-300">
              {darkMode ? '🌙 Dark' : '☀️ Light'}
            </span>
            <span className="text-sm text-gray-500 dark:text-gray-400">Toggle</span>
          </button>
        </div>
      </div>
    </aside>
  );
}
