import React, { useState, useEffect } from 'react';
import StudentView from './pages/StudentView';
import DashboardView from './pages/DashboardView';
import { Shield, LayoutDashboard, GraduationCap, CheckCircle2 } from 'lucide-react';

// Edge Computing Architecture:
// In dev (port 3000): point directly to local FastAPI backend on port 8000
// In production/edge: both frontend and backend run on same local origin
const API_BASE_URL =
  process.env.REACT_APP_API_URL ||
  (typeof window !== 'undefined' && window.location.port === '3000'
    ? `${window.location.protocol}//${window.location.hostname}:8000`
    : '');

export default function App() {
  const [activeTab, setActiveTab] = useState('student');
  const [health, setHealth] = useState(null);

  useEffect(() => {
    let active = true;
    fetch(`${API_BASE_URL}/health`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (active && data) setHealth(data);
      })
      .catch(() => {});
    return () => {
      active = false;
    };
  }, []);

  return (
    <div className="min-h-screen flex flex-col bg-slate-950 text-slate-100 font-sans">
      {/* Header */}
      <header className="sticky top-0 z-50 bg-slate-900/90 backdrop-blur-md border-b border-slate-800">
        <div className="max-w-6xl mx-auto px-4 h-16 flex items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-emerald-600 rounded-xl text-white shadow-lg shadow-emerald-900/30">
              <Shield className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-bold text-lg tracking-tight text-white">VeriMath</span>
                <span className="text-[10px] font-mono uppercase tracking-wider bg-slate-800 text-emerald-400 px-2 py-0.5 rounded-full border border-slate-700">
                  Lean 4 Oracle
                </span>
                {health && (
                  <span
                    title={`Templates: ${health.supported_templates?.length || 0} | Cached: ${health.demo_cache_size || 0}`}
                    className={`hidden md:inline-flex items-center gap-1.5 text-[10px] font-mono px-2 py-0.5 rounded-full border ${
                      health.lean_installed
                        ? 'bg-emerald-950/60 text-emerald-300 border-emerald-700/60'
                        : 'bg-cyan-950/60 text-cyan-300 border-cyan-700/60'
                    }`}
                  >
                    <span className={`w-1.5 h-1.5 rounded-full ${health.lean_installed ? 'bg-emerald-400' : 'bg-cyan-400'}`}></span>
                    {health.lean_installed ? 'Lean 4 Kernel Live' : 'CAS + Formal Mode'}
                  </span>
                )}
              </div>
              <p className="text-[11px] text-slate-400 hidden sm:block">
                Zero-Trust Formal Verification Math Tutor
              </p>
            </div>
          </div>

          <nav className="flex items-center gap-1.5 bg-slate-950 p-1 rounded-xl border border-slate-800">
            <button
              type="button"
              onClick={() => setActiveTab('student')}
              className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all ${
                activeTab === 'student'
                  ? 'bg-emerald-600 text-white shadow'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <GraduationCap className="w-4 h-4" />
              <span>Student View</span>
            </button>

            <button
              type="button"
              onClick={() => setActiveTab('dashboard')}
              className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all ${
                activeTab === 'dashboard'
                  ? 'bg-emerald-600 text-white shadow'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <LayoutDashboard className="w-4 h-4" />
              <span>Teacher Dashboard</span>
            </button>
          </nav>
        </div>
      </header>

      {/* Main Container */}
      <main className="flex-1 max-w-6xl w-full mx-auto px-4 py-8">
        {activeTab === 'student' ? (
          <StudentView apiBaseUrl={API_BASE_URL} />
        ) : (
          <DashboardView apiBaseUrl={API_BASE_URL} />
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-900 bg-slate-950/80 py-6 text-center text-xs text-slate-500">
        <div className="max-w-6xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-500" />
            <span>Zero-Trust AI + Formal Verification Edge Tutor</span>
          </div>
          <div>
            Powered by <strong>Lean 4.36</strong>, <strong>Mathlib</strong>, <strong>Qwen2.5 &amp; SymPy CAS</strong> on <strong>Raspberry Pi 4 (8 GB)</strong>
          </div>
        </div>
      </footer>
    </div>
  );
}
