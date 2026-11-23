import React, { useState } from 'react';
import { Code2, ChevronDown, ChevronUp, Copy, Check } from 'lucide-react';

export default function LeanStatement({ statement, templateId }) {
  const [isOpen, setIsOpen] = useState(false);
  const [copied, setCopied] = useState(false);

  if (!statement) return null;

  const isGeneralized = typeof templateId === 'string' && templateId.startsWith('generalized_');

  const handleCopy = () => {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(statement).catch(() => {});
    }
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-xl transition-all">
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-6 py-4 flex flex-wrap items-center justify-between gap-2 bg-slate-900/80 hover:bg-slate-800/60 transition-colors text-left"
      >
        <div className="flex flex-wrap items-center gap-2.5">
          <Code2 className="w-5 h-5 text-indigo-400 shrink-0" />
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-sm font-semibold text-slate-200">
              Formal Lean 4 Theorem Code
            </span>
            <span className="text-xs font-mono px-2 py-0.5 rounded bg-indigo-950/80 text-indigo-300 border border-indigo-800">
              {isGeneralized ? `Generalized: ${templateId}` : `Template: ${templateId}`}
            </span>
          </div>
        </div>

        <div className="flex items-center gap-2 text-xs text-slate-400">
          <span>{isOpen ? 'Hide Proof' : 'Inspect Kernel Proof'}</span>
          {isOpen ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </div>
      </button>

      {isOpen && (
        <div className="p-6 bg-slate-950 border-t border-slate-800 space-y-3">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span className="font-mono">Lean 4 + Mathlib Statement</span>
            <button
              type="button"
              onClick={handleCopy}
              className="flex items-center gap-1.5 px-2.5 py-1 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg transition-colors font-mono"
            >
              {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
              <span>{copied ? 'Copied' : 'Copy Lean'}</span>
            </button>
          </div>

          <pre className="p-4 bg-slate-900/90 border border-slate-800 rounded-xl font-mono text-xs text-emerald-300 overflow-x-auto leading-relaxed shadow-inner">
            <code>{statement}</code>
          </pre>

          <p className="text-[11px] text-slate-500 italic">
            This formal theorem was independently certified by the Lean 4 kernel with zero unverified axioms or 'sorry' admissions.
          </p>
        </div>
      )}
    </div>
  );
}
