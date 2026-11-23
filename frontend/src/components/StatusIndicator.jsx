import React from 'react';
import { CheckCircle2, Loader2 } from 'lucide-react';

const STAGES = [
  { id: 'classify', label: 'Domain Classification' },
  { id: 'solve', label: 'Candidate Solver' },
  { id: 'formalize', label: 'Lean 4 Formalizer' },
  { id: 'judge', label: 'AST & Faithfulness Judge' },
  { id: 'verify', label: 'Lean 4 Kernel Oracle' },
  { id: 'explain', label: 'Intuitive Explainer' },
];

export default function StatusIndicator({ isLoading, currentStage }) {
  if (!isLoading) return null;

  const activeIdx = currentStage ?? 0;

  return (
    <div className="bg-slate-900/90 border border-emerald-500/30 rounded-2xl p-5 my-6 shadow-2xl backdrop-blur-md">
      <div className="flex flex-wrap items-center justify-between gap-2 mb-4">
        <div className="flex items-center gap-2.5">
          <Loader2 className="w-5 h-5 text-emerald-400 animate-spin" />
          <span className="font-semibold text-slate-200 text-sm">
            Executing VeriMath Zero-Trust Pipeline...
          </span>
        </div>
        <span className="text-xs font-mono text-emerald-400 bg-emerald-950/60 border border-emerald-800 px-2.5 py-0.5 rounded-full">
          Kernel Verification Active
        </span>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-6 gap-2">
        {STAGES.map((stage, idx) => {
          const isDone = idx < activeIdx;
          const isCurrent = idx === activeIdx;
          return (
            <div
              key={stage.id}
              className={`flex flex-col p-2.5 rounded-xl border text-center transition-all ${
                isDone
                  ? 'bg-emerald-950/50 border-emerald-500/40 text-emerald-300'
                  : isCurrent
                    ? 'bg-cyan-950/50 border-cyan-400/60 text-cyan-200 shadow-sm shadow-cyan-900/40'
                    : 'bg-slate-950/40 border-slate-800 text-slate-600'
              }`}
            >
              <div className="flex items-center justify-center gap-1 text-[11px] font-mono mb-1">
                {isDone && <CheckCircle2 className="w-3 h-3 text-emerald-400" />}
                {isCurrent && <Loader2 className="w-3 h-3 text-cyan-400 animate-spin" />}
                <span>Step {idx + 1}</span>
              </div>
              <span className="text-xs font-medium truncate">{stage.label}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
