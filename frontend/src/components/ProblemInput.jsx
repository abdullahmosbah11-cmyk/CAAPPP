import React, { useState } from 'react';
import { Send, Sparkles, X, Camera } from 'lucide-react';
import CameraScannerModal from './CameraScannerModal';

const PRESET_EXAMPLES = [
  { label: "Product Rule x² sin(x)", text: "Find the derivative of x² sin(x) with respect to x", kind: "template" },
  { label: "Related Rates Ladder", text: "A 10-foot ladder is leaning against a vertical wall. The bottom of the ladder slides away from the wall at a constant rate of 2 feet per second. How fast is the top of the ladder sliding down the wall at the exact moment when the bottom of the ladder is 6 feet away from the wall?", kind: "template" },
  { label: "Definite Integral ∫₀¹ x²", text: "Evaluate the definite integral of x² from 0 to 1", kind: "template" },
  { label: "Linear System (Generalized)", text: "Solve the system: 2x + 3y = 13, x - y = -1", kind: "generalized" },
  { label: "Complex Product (3+4i)(1-2i)", text: "Compute (3 + 4i) * (1 - 2i)", kind: "generalized" },
  { label: "Log Law: log(a·b)", text: "Prove that log(a * b) = log(a) + log(b)", kind: "generalized" },
  { label: "Rational Exp: 4^x = 8", text: "Solve 4^x = 8", kind: "generalized" },
  { label: "False Claim (Disprove!)", text: "Prove that (a + b)^2 = a^2 + b^2", kind: "disprove" },
  { label: "False Prime 91 (Disprove!)", text: "Prove that 91 is prime", kind: "disprove" },
];

export default function ProblemInput({ apiBaseUrl = '', problem, setProblem, onSubmit, isLoading }) {
  const [scannerOpen, setScannerOpen] = useState(false);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!isLoading && problem.trim()) {
        onSubmit(e);
      }
    }
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl backdrop-blur-sm">
      <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
        <label className="text-sm font-semibold text-slate-300 flex flex-wrap items-center gap-2">
          <span>Enter Math Problem</span>
          <span className="text-xs text-slate-500 font-normal">(Calculus, Algebra, Generalized Higher Math &amp; Claim Verification)</span>
        </label>
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={() => setScannerOpen(true)}
            disabled={isLoading}
            className="text-xs px-2.5 py-1 rounded-lg bg-indigo-950/60 hover:bg-indigo-900/60 text-indigo-300 border border-indigo-700/60 flex items-center gap-1.5 transition-colors"
          >
            <Camera className="w-3.5 h-3.5" />
            <span>Scan Photo / Camera</span>
          </button>
          <span className="text-xs text-slate-400 hidden md:flex items-center gap-1 font-mono">
            Press <kbd className="px-1.5 py-0.5 bg-slate-800 text-slate-300 rounded border border-slate-700 text-[10px]">Enter ↵</kbd> to solve
          </span>
        </div>
      </div>

      <form onSubmit={onSubmit} className="space-y-4">
        <div className="relative">
          <textarea
            value={problem}
            onChange={(e) => setProblem(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isLoading}
            placeholder="e.g. Find the derivative of x² sin(x), Compute (3 + 4i) * (1 - 2i), or scan a photo of handwritten math..."
            rows={3}
            className="w-full bg-slate-950/80 border border-slate-700/80 rounded-xl p-4 pr-10 text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent transition-all resize-none font-mono text-sm"
          />
          {problem && !isLoading && (
            <button
              type="button"
              onClick={() => setProblem('')}
              aria-label="Clear problem input"
              className="absolute top-3 right-3 p-1 rounded-lg text-slate-500 hover:text-slate-200 hover:bg-slate-800 transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>

        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pt-1">
          <div className="flex items-center gap-2 overflow-x-auto py-1 max-w-full sm:max-w-[75%] no-scrollbar">
            <span className="text-xs text-slate-500 flex items-center gap-1 shrink-0">
              <Sparkles className="w-3 h-3 text-emerald-400" /> Demo:
            </span>
            {PRESET_EXAMPLES.map((ex, idx) => (
              <button
                key={idx}
                type="button"
                onClick={() => setProblem(ex.text)}
                disabled={isLoading}
                className={`text-xs px-2.5 py-1 rounded-lg shrink-0 border transition-colors ${
                  ex.kind === 'disprove'
                    ? 'bg-rose-950/40 hover:bg-rose-900/50 text-rose-300 border-rose-800/60'
                    : ex.kind === 'generalized'
                      ? 'bg-indigo-950/40 hover:bg-indigo-900/50 text-indigo-300 border-indigo-800/60'
                      : 'bg-slate-800 hover:bg-slate-700 text-slate-300 border-slate-700'
                }`}
                title={ex.text}
              >
                {ex.label}
              </button>
            ))}
          </div>

          <button
            type="submit"
            disabled={isLoading || !problem.trim()}
            className="flex items-center justify-center gap-2 px-5 py-2.5 bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-800 disabled:text-slate-600 text-white font-medium rounded-xl transition-all shadow-lg shadow-emerald-900/20 shrink-0"
          >
            <Send className="w-4 h-4" />
            <span>{isLoading ? 'Verifying...' : 'Solve & Verify'}</span>
          </button>
        </div>
      </form>

      <CameraScannerModal
        isOpen={scannerOpen}
        onClose={() => setScannerOpen(false)}
        apiBaseUrl={apiBaseUrl}
        scanMode="question"
        onApplyScan={({ problemText, autoSubmit }) => {
          setProblem(problemText);
          if (autoSubmit && problemText) {
            onSubmit(null, problemText);
          }
        }}
      />
    </div>
  );
}
