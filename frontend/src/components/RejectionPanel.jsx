import React, { useState } from 'react';
import { ShieldAlert, Ban, CheckCircle2, ChevronDown, ChevronRight, Terminal } from 'lucide-react';

export default function RejectionPanel({ result }) {
  const [showDisproof, setShowDisproof] = useState(false);

  if (!result || (result.status !== 'REJECTED' && result.status !== 'UNSUPPORTED' && result.status !== 'ERROR')) {
    return null;
  }

  const isUnsupported = result.status === 'UNSUPPORTED';
  const ce = result.counterexample && result.counterexample.found ? result.counterexample : null;

  return (
    <div className={`border rounded-2xl p-6 shadow-2xl space-y-4 ${
      isUnsupported
        ? 'bg-amber-950/20 border-amber-500/40'
        : 'bg-rose-950/20 border-rose-500/40'
    }`}>
      <div className="flex items-center gap-3">
        <div className={`p-2.5 rounded-xl border ${
          isUnsupported
            ? 'bg-amber-500/10 border-amber-500/30 text-amber-400'
            : 'bg-rose-500/10 border-rose-500/30 text-rose-400'
        }`}>
          {isUnsupported ? <Ban className="w-6 h-6" /> : <ShieldAlert className="w-6 h-6" />}
        </div>
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <h2 className={`text-lg font-bold ${isUnsupported ? 'text-amber-200' : 'text-rose-200'}`}>
              {isUnsupported
                ? 'Outside Verified Domain (Refusal)'
                : ce && ce.lean_verified
                  ? 'Mathematically Disproven (Lean 4 Counterexample)'
                  : 'Verification Failed (Hallucination Rejection)'}
            </h2>
            {ce && ce.lean_verified && (
              <span className="text-xs font-semibold px-2 py-0.5 bg-rose-500/20 text-rose-300 border border-rose-500/40 rounded-full font-mono flex items-center gap-1">
                <CheckCircle2 className="w-3 h-3" /> Counterexample Kernel-Verified
              </span>
            )}
          </div>
          <span className="text-xs text-slate-400">
            {isUnsupported
              ? 'VeriMath refuses to answer unverifiable inputs to guarantee absolute reliability.'
              : ce && ce.lean_verified
                ? 'A concrete counterexample was synthesized and formally proven in Lean 4 to refute this claim.'
                : 'The proposed answer could not be proved mathematically and has been safely intercepted.'}
          </span>
        </div>
      </div>

      {ce && (
        <div className="bg-slate-950/90 border border-rose-500/30 rounded-xl p-4 space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <span className="text-xs font-semibold uppercase tracking-wider text-rose-300">
              Verified Witness Counterexample
            </span>
            <span className="font-mono text-xs px-2.5 py-1 rounded-lg bg-rose-500/15 border border-rose-500/30 text-rose-200">
              {ce.witness_display}
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs font-mono">
            <div className="bg-slate-900/90 border border-slate-800 rounded-lg p-3">
              <div className="text-slate-400 mb-1">Evaluated Left Side</div>
              <div className="text-sm font-bold text-slate-100">{ce.lhs_value}</div>
            </div>
            <div className="bg-slate-900/90 border border-slate-800 rounded-lg p-3">
              <div className="text-slate-400 mb-1">Claimed / Right Side</div>
              <div className="text-sm font-bold text-rose-300">{ce.rhs_value}</div>
            </div>
          </div>

          <p className="text-xs text-slate-300 leading-relaxed">{ce.explanation}</p>

          {ce.disproof_lean_statement && (
            <div className="pt-1">
              <button
                type="button"
                onClick={() => setShowDisproof(!showDisproof)}
                className="text-xs font-mono text-slate-400 hover:text-slate-200 flex items-center gap-1.5 transition-colors"
              >
                {showDisproof ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
                <Terminal className="w-3.5 h-3.5 text-rose-400" />
                <span>{showDisproof ? 'Hide Lean 4 Counterexample Proof' : 'Show Lean 4 Counterexample Proof'}</span>
              </button>
              {showDisproof && (
                <pre className="mt-2 p-3 rounded-lg bg-slate-900 border border-slate-800 text-[11px] font-mono text-rose-200 overflow-x-auto whitespace-pre-wrap">
                  {ce.disproof_lean_statement}
                </pre>
              )}
            </div>
          )}
        </div>
      )}

      <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-4 text-sm text-slate-300 leading-relaxed font-mono">
        {result.rejection_reason || 'Verification was unsuccessful. Please check your problem syntax.'}
      </div>

      <div className="flex items-center justify-between text-xs text-slate-400 pt-2 border-t border-slate-800/80">
        <span className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-rose-500 inline-block animate-ping"></span>
          Student protected from unverified AI hallucinations.
        </span>
        <span className="font-mono text-slate-500">Latency: {result.elapsed_ms} ms</span>
      </div>
    </div>
  );
}
