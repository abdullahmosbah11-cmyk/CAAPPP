import React, { useState } from 'react';
import ProblemInput from '../components/ProblemInput';
import StatusIndicator from '../components/StatusIndicator';
import ResultPanel from '../components/ResultPanel';
import LeanStatement from '../components/LeanStatement';
import RejectionPanel from '../components/RejectionPanel';
import StepGraderPanel from '../components/StepGraderPanel';
import { ShieldCheck, Cpu, Terminal, ListChecks } from 'lucide-react';

export default function StudentView({ apiBaseUrl }) {
  const [activeTab, setActiveTab] = useState('solve');
  const [problem, setProblem] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [currentStage, setCurrentStage] = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleSubmit = async (e, overrideProblem = null) => {
    if (e) e.preventDefault();
    const targetProblem = typeof overrideProblem === 'string' ? overrideProblem : problem;
    if (!targetProblem.trim() || isLoading) return;

    setIsLoading(true);
    setResult(null);
    setError(null);
    setCurrentStage(0);

    // Stage progression animation
    const interval = setInterval(() => {
      setCurrentStage((prev) => (prev < 4 ? prev + 1 : prev));
    }, 600);

    const cleanedProblem = targetProblem.replace(/\s+/g, ' ').trim();

    try {
      const response = await fetch(`${apiBaseUrl}/solve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          problem: cleanedProblem,
          student_id: 'student_k12_demo'
        }),
      });

      clearInterval(interval);
      setCurrentStage(5);

      if (!response.ok) {
        throw new Error(`Server returned HTTP ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      clearInterval(interval);
      setError(err.message || 'Network error connecting to VeriMath backend.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Intro hero banner */}
      <div className="bg-gradient-to-r from-slate-900 via-slate-900 to-emerald-950/40 border border-slate-800 rounded-2xl p-6 shadow-xl">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold text-slate-100 flex items-center gap-2">
              <span>VeriMath Interactive Assistant</span>
              <span className="text-xs bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 px-2.5 py-0.5 rounded-full font-mono">
                Edge Verified
              </span>
            </h1>
            <p className="text-sm text-slate-400 mt-1 max-w-2xl">
              Submit your mathematics problem. The local AI proposes a candidate answer, formalizes it, and submits it to the <strong>Lean 4 kernel</strong>. Only certified truths are presented.
            </p>
          </div>

          <div className="flex items-center gap-2 bg-slate-950/80 border border-slate-800 px-4 py-2 rounded-xl text-xs font-mono text-slate-300">
            <Cpu className="w-4 h-4 text-emerald-400" />
            <span>Raspberry Pi 4 • Local Kernel</span>
          </div>
        </div>

        {/* Mode Switcher Tabs */}
        <div className="flex items-center gap-2 mt-5 pt-4 border-t border-slate-800/80">
          <button
            type="button"
            onClick={() => setActiveTab('solve')}
            className={`px-4 py-2 rounded-xl text-xs font-semibold flex items-center gap-2 transition-colors ${
              activeTab === 'solve'
                ? 'bg-emerald-600 text-white shadow-lg shadow-emerald-950/40'
                : 'bg-slate-950/60 text-slate-400 hover:text-slate-200 border border-slate-800'
            }`}
          >
            <ShieldCheck className="w-4 h-4" />
            Solve &amp; Prove
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('steps')}
            className={`px-4 py-2 rounded-xl text-xs font-semibold flex items-center gap-2 transition-colors ${
              activeTab === 'steps'
                ? 'bg-emerald-600 text-white shadow-lg shadow-emerald-950/40'
                : 'bg-slate-950/60 text-slate-400 hover:text-slate-200 border border-slate-800'
            }`}
          >
            <ListChecks className="w-4 h-4" />
            Check My Steps
          </button>
        </div>
      </div>

      {activeTab === 'steps' ? (
        <StepGraderPanel apiBaseUrl={apiBaseUrl} initialProblem={problem} />
      ) : (
        <>
          {/* Input */}
          <ProblemInput
            apiBaseUrl={apiBaseUrl}
            problem={problem}
            setProblem={setProblem}
            onSubmit={handleSubmit}
            isLoading={isLoading}
          />

          {/* Pipeline Status Indicator */}
          <StatusIndicator isLoading={isLoading} currentStage={currentStage} />

          {/* Error state */}
          {error && (
            <div className="bg-rose-950/40 border border-rose-800 rounded-2xl p-5 text-rose-300 text-sm font-mono">
              ⚠️ {error.toLowerCase().includes('server') || error.toLowerCase().includes('http') ? error : `Connection error: ${error}`}
            </div>
          )}

          {/* Results */}
          {result && result.status === 'VERIFIED' && (
            <>
              <ResultPanel result={result} />
              <LeanStatement
                statement={result.lean_statement}
                templateId={result.template_id}
              />
            </>
          )}

          {/* Rejections & Unsupporteds */}
          {result && result.status !== 'VERIFIED' && (
            <RejectionPanel result={result} />
          )}
        </>
      )}
    </div>
  );
}
