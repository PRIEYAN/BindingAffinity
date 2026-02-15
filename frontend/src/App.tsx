/**
 * Main App component with macOS-style dashboard
 */

import { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Sidebar } from './components/Sidebar';
import { Predict } from './pages/Predict';
import { History } from './pages/History';
import { About } from './pages/About';
import { api } from './services/api';

function App() {
  const [darkMode, setDarkMode] = useState(() => {
    const saved = localStorage.getItem('darkMode');
    return saved ? JSON.parse(saved) : false;
  });

  useEffect(() => {
    // Apply dark mode class
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
    localStorage.setItem('darkMode', JSON.stringify(darkMode));
  }, [darkMode]);

  useEffect(() => {
    // Check backend health
    api.healthCheck().then((isOnline) => {
      if (!isOnline) {
        console.warn('Backend server is not responding');
      }
    });
  }, []);

  return (
    <Router>
      <div className="min-h-screen">
        <Sidebar darkMode={darkMode} onToggleDarkMode={() => setDarkMode(!darkMode)} />
        
        <main className="ml-64 p-8">
          <Routes>
            <Route path="/" element={<Predict />} />
            <Route path="/history" element={<History />} />
            <Route path="/about" element={<About />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
