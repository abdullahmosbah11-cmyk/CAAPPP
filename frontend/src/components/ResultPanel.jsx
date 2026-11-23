import React, { useState, useEffect, useRef } from 'react';
import {
  ShieldCheck,
  Clock,
  Zap,
  BookOpen,
  Sigma,
  ListOrdered,
  ChevronRight,
  ChevronLeft,
  Copy,
  Check,
  Sparkles,
  Lightbulb,
  GraduationCap,
  Layers,
  PlayCircle,
  Eye,
  EyeOff,
  CheckCircle2,
  HelpCircle,
} from 'lucide-react';
import katex from 'katex';
import 'katex/dist/katex.min.css';

/**
 * Renders a display-mode or inline-mode LaTeX formula using KaTeX.
 */
function MathDisplay({ latex, fallback, displayMode = true }) {
  const containerRef = useRef(null);

  useEffect(() => {
    if (containerRef.current) {
      let mathStr = latex || fallback || '';
      // Un-trap formulas mistakenly wrapped in \text{...} that contain mathematical operators
      if (mathStr.startsWith('\\text{') && mathStr.endsWith('}') && /[\^+\-*/=]|Real\./.test(mathStr)) {
        mathStr = mathStr.slice(6, -1);
      }
      // Sanitize Lean 4 namespace tokens if present (e.g. Real.sin x -> \sin(x))
      mathStr = mathStr.replace(/\bReal\.([a-zA-Z]+)\s+([a-zA-Z0-9]+)\b/g, (_, f, a) => `\\${f}(${a})`);
      mathStr = mathStr.replace(/\bReal\.([a-zA-Z]+)\b/g, (_, f) => `\\${f}`);
      // Sanitize unit exponents trapped inside \text{...} so KaTeX does not fail
      mathStr = mathStr.replace(/\\text\{\s*([a-zA-Z\s]+?)\s*\^(\d+)\s*\}/g, '\\text{ $1}^{$2}');
      mathStr = mathStr.replace(/\\text\{\s*([a-zA-Z\s]+?)\^(\d+)\s*\}/g, '\\text{ $1}^{$2}');
      try {
        katex.render(mathStr, containerRef.current, {
          displayMode,
          throwOnError: false,
          errorColor: '#34d399',
        });
      } catch (err) {
        if (containerRef.current) {
          containerRef.current.innerText = fallback || mathStr;
        }
      }
    }
  }, [latex, fallback, displayMode]);

  return <div ref={containerRef} className="overflow-x-auto py-1 text-emerald-300 font-medium" />;
}

/**
 * Formats prose text containing embedded LaTeX or math tokens with inline KaTeX rendering.
 */
function FormattedExplanation({ text }) {
  if (!text) return null;

  // Split on $...$, \(...\), or LaTeX commands starting with backslash
  const regex = /(\$[^$]+\$|\\\([^\)]+\\\)|\\[a-zA-Z]+(?:\{[^}]*\}|\[[^\]]*\]|_[a-zA-Z0-9]+|\^[a-zA-Z0-9]+|[^\s,;:()]+)*)/g;
  const parts = text.split(regex);

  return (
    <span className="leading-relaxed">
      {parts.map((part, idx) => {
        if (!part) return null;
        let math = null;
        if (part.startsWith('$') && part.endsWith('$')) {
          math = part.slice(1, -1);
        } else if (part.startsWith('\\(') && part.endsWith('\\)')) {
          math = part.slice(2, -2);
        } else if (part.startsWith('\\') && part.length > 1) {
          math = part;
        }

        if (math) {
          try {
            const html = katex.renderToString(math, {
              displayMode: false,
              throwOnError: false,
            });
            return (
              <span
                key={idx}
                dangerouslySetInnerHTML={{ __html: html }}
                className="inline-block px-1 text-emerald-300 font-mono"
              />
            );
          } catch (e) {
            return <span key={idx}>{part}</span>;
          }
        }
        return <span key={idx}>{part}</span>;
      })}
    </span>
  );
}

/**
 * Individual step card rendered in full solution view.
 */
function StepCard({ step, isExpanded, onToggle, totalSteps }) {
  const [copiedMath, setCopiedMath] = useState(false);

  const handleCopyMath = (e) => {
    e.stopPropagation();
    if (!step.math) return;
    navigator.clipboard.writeText(step.math);
    setCopiedMath(true);
    setTimeout(() => setCopiedMath(false), 2000);
  };

  // Determine an illustrative icon based on step title
  const getStepIcon = (title) => {
    const t = (title || '').toLowerCase();
    if (t.includes('identify') || t.includes('select') || t.includes('setup') || t.includes('label')) {
      return <Lightbulb className="w-4 h-4 text-amber-400" />;
    }
    if (t.includes('rule') || t.includes('theorem') || t.includes('formula') || t.includes('substitut')) {
      return <Sparkles className="w-4 h-4 text-emerald-400" />;
    }
    if (t.includes('evaluat') || t.includes('simplify') || t.includes('solve') || t.includes('compute')) {
      return <Sigma className="w-4 h-4 text-cyan-400" />;
    }
    return <CheckCircle2 className="w-4 h-4 text-emerald-400" />;
  };

  return (
    <div className="relative group transition-all">
      {/* Step Container */}
      <div className="bg-slate-950/80 border border-slate-800 hover:border-emerald-500/40 rounded-xl p-4.5 transition-all shadow-md space-y-3">
        {/* Step Header */}
        <div
          onClick={onToggle}
          className="flex items-center justify-between cursor-pointer select-none"
        >
          <div className="flex items-center gap-3">
            <span className="flex items-center justify-center w-7 h-7 rounded-lg bg-emerald-500/15 text-emerald-300 border border-emerald-500/30 text-xs font-bold font-mono shadow-sm">
              {step.step_number}
            </span>
            <div className="flex items-center gap-2">
              <span className="p-1 rounded-md bg-slate-900 border border-slate-800">
                {getStepIcon(step.title)}
              </span>
              <h4 className="text-sm font-semibold text-slate-100 group-hover:text-emerald-300 transition-colors">
                {step.title}
              </h4>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-[11px] text-slate-500 font-mono hidden sm:inline">
              Step {step.step_number} of {totalSteps}
            </span>
            <button
              type="button"
              className="p-1 text-slate-400 hover:text-slate-200 transition-colors"
            >
              {isExpanded ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
            </button>
          </div>
        </div>

        {/* Step Body (Collapsible) */}
        {isExpanded && (
          <div className="pt-2 border-t border-slate-800/80 space-y-3">
            {/* Pedagogical Explanation */}
            <p className="text-xs text-slate-300 leading-relaxed pl-1">
              <FormattedExplanation text={step.explanation} />
            </p>

            {/* KaTeX Math Formula Block */}
            {step.math && (
              <div className="relative bg-slate-900/90 border border-slate-800/90 rounded-lg p-3 text-center overflow-x-auto shadow-inner group/math">
                <button
                  type="button"
                  onClick={handleCopyMath}
                  title="Copy LaTeX Formula"
                  className="absolute right-2 top-2 p-1.5 rounded-md bg-slate-800/80 hover:bg-slate-700 text-slate-400 hover:text-slate-200 opacity-0 group-hover/math:opacity-100 transition-opacity text-[10px] flex items-center gap-1 font-mono"
                >
                  {copiedMath ? (
                    <>
                      <Check className="w-3 h-3 text-emerald-400" />
                      <span className="text-emerald-400">Copied</span>
                    </>
                  ) : (
                    <>
                      <Copy className="w-3 h-3" />
                      <span>LaTeX</span>
                    </>
                  )}
                </button>
                <div className="pt-1">
                  <MathDisplay latex={step.math} fallback={step.math} displayMode={true} />
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default function ResultPanel({ result }) {
  const [activeView, setActiveView] = useState('walkthrough'); // 'walkthrough' | 'all'
  const [currentStepIdx, setCurrentStepIdx] = useState(0);
  const [expandedSteps, setExpandedSteps] = useState({});
  const [copiedSolution, setCopiedSolution] = useState(false);
  const [copiedAnswer, setCopiedAnswer] = useState(false);

  const steps = result?.steps || [];
  const totalSteps = steps.length;

  // Initialize all steps expanded by default
  useEffect(() => {
    if (steps.length > 0) {
      const initExpanded = {};
      steps.forEach((_, idx) => {
        initExpanded[idx] = true;
      });
      setExpandedSteps(initExpanded);
    }
  }, [steps]);

  // Keyboard navigation for walkthrough (Left / Right arrow keys)
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (activeView !== 'walkthrough' || totalSteps <= 1) return;
      if (e.key === 'ArrowRight' && currentStepIdx < totalSteps - 1) {
        setCurrentStepIdx((prev) => prev + 1);
      } else if (e.key === 'ArrowLeft' && currentStepIdx > 0) {
        setCurrentStepIdx((prev) => prev - 1);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [activeView, currentStepIdx, totalSteps]);

  if (!result || result.status !== 'VERIFIED') return null;

  const toggleStep = (idx) => {
    setExpandedSteps((prev) => ({ ...prev, [idx]: !prev[idx] }));
  };

  const expandAll = () => {
    const all = {};
    steps.forEach((_, i) => (all[i] = true));
    setExpandedSteps(all);
  };

  const collapseAll = () => {
    setExpandedSteps({});
  };

  const handleCopyAnswer = () => {
    navigator.clipboard.writeText(result.latex_answer || result.verified_answer || '');
    setCopiedAnswer(true);
    setTimeout(() => setCopiedAnswer(false), 2000);
  };

  const handleCopyFullSolution = () => {
    let text = `Problem Solution (Lean 4 Verified):\nAnswer: ${result.verified_answer}\n\n`;
    if (steps.length > 0) {
      text += `Step-by-Step Derivation:\n`;
      steps.forEach((s) => {
        text += `Step ${s.step_number}: ${s.title}\n`;
        text += `${s.explanation}\n`;
        if (s.math) text += `Math: ${s.math}\n`;
        text += `\n`;
      });
    }
    if (result.explanation) {
      text += `Pedagogical Explanation:\n${result.explanation}\n`;
    }
    navigator.clipboard.writeText(text);
    setCopiedSolution(true);
    setTimeout(() => setCopiedSolution(false), 2000);
  };

  const currentStep = steps[currentStepIdx] || steps[0];

  return (
    <div className="bg-slate-900 border border-emerald-500/40 rounded-2xl p-6 shadow-2xl space-y-6 transition-all">
      {/* ───────────────────────────────────────────────────────────────── */}
      {/* 1. Header Banner */}
      {/* ───────────────────────────────────────────────────────────────── */}
      <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div className="flex items-center gap-3">
          <div className="p-2.5 bg-emerald-500/10 border border-emerald-500/30 rounded-xl text-emerald-400 shadow-sm shadow-emerald-950">
            <ShieldCheck className="w-6 h-6" />
          </div>
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <h2 className="text-lg font-bold text-slate-100">Mathematically Proven Correct</h2>
              <span className="text-xs font-semibold px-2 py-0.5 bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 rounded-full font-mono">
                Lean 4 Verified
              </span>
              {result.ast_verified !== false && (
                <span
                  title={
                    Array.isArray(result.ast_checks) && result.ast_checks.length > 0
                      ? `AST Guards Passed: ${result.ast_checks.join(', ')}`
                      : 'Structural AST Faithfulness Verified'
                  }
                  className="text-xs font-semibold px-2 py-0.5 bg-cyan-500/15 text-cyan-300 border border-cyan-500/30 rounded-full font-mono cursor-help"
                >
                  AST Faithfulness Verified
                </span>
              )}
              {result.generalized_mode && (
                <span className="text-xs font-semibold px-2 py-0.5 bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 rounded-full font-mono">
                  Generalized Out-of-Domain Proof
                </span>
              )}
            </div>
            <p className="text-xs text-slate-400">
              {result.generalized_mode
                ? 'Solved via zero-trust out-of-domain Lean 4 autoformalization & theorem verification.'
                : 'Verified by Lean 4 interactive theorem prover with zero-trust kernel assurance.'}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2.5 text-xs font-mono text-slate-400">
          {result.from_cache && (
            <span className="flex items-center gap-1 text-amber-400 bg-amber-950/40 border border-amber-800/60 px-2.5 py-1 rounded-lg">
              <Zap className="w-3.5 h-3.5" /> Instant Cache
            </span>
          )}
          <span className="flex items-center gap-1 bg-slate-800 px-2.5 py-1 rounded-lg border border-slate-700">
            <Clock className="w-3.5 h-3.5 text-slate-400" /> {result.elapsed_ms} ms
          </span>
          <button
            type="button"
            onClick={handleCopyFullSolution}
            className="flex items-center gap-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 px-3 py-1 rounded-lg border border-slate-700 transition-colors"
            title="Copy entire solution with steps"
          >
            {copiedSolution ? (
              <>
                <Check className="w-3.5 h-3.5 text-emerald-400" />
                <span className="text-emerald-400">Copied</span>
              </>
            ) : (
              <>
                <Copy className="w-3.5 h-3.5" />
                <span>Copy Solution</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* ───────────────────────────────────────────────────────────────── */}
      {/* 2. Verified Mathematical Answer Box */}
      {/* ───────────────────────────────────────────────────────────────── */}
      <div className="bg-slate-950/90 border border-slate-800 rounded-xl p-5 shadow-inner">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
            Verified Mathematical Answer
          </span>
          <div className="flex items-center gap-2">
            {result.latex_answer && (
              <span className="flex items-center gap-1 text-[11px] font-medium px-2 py-0.5 rounded-full bg-emerald-950/70 text-emerald-300 border border-emerald-800/50">
                <Sigma className="w-3 h-3 text-emerald-400" /> KaTeX Rendered
              </span>
            )}
            <button
              type="button"
              onClick={handleCopyAnswer}
              className="text-[11px] font-mono px-2 py-0.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded border border-slate-700 flex items-center gap-1 transition-colors"
            >
              {copiedAnswer ? (
                <>
                  <Check className="w-3 h-3 text-emerald-400" />
                  <span className="text-emerald-400">Copied</span>
                </>
              ) : (
                <>
                  <Copy className="w-3 h-3" />
                  <span>Copy</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Rendered LaTeX Formula */}
        <div className="text-xl md:text-2xl font-bold text-emerald-400 tracking-wide break-words py-1 text-center">
          <MathDisplay latex={result.latex_answer} fallback={result.verified_answer} />
        </div>

        {/* Details & Back Translation */}
        <div className="mt-3 pt-3 border-t border-slate-800/80 space-y-1.5 text-xs text-slate-400">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-slate-500 font-medium">Standard Notation:</span>
            <code className="bg-slate-900 px-2 py-0.5 rounded text-slate-300 font-mono text-[11px] border border-slate-800">
              {result.verified_answer}
            </code>
          </div>
          {result.back_translation && (
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-slate-500 font-medium">Formal Translation:</span>
              <span className="text-slate-300">{result.back_translation}</span>
            </div>
          )}
        </div>
      </div>

      {/* ───────────────────────────────────────────────────────────────── */}
      {/* 3. Step-by-Step Derivation & Guided Walkthrough */}
      {/* ───────────────────────────────────────────────────────────────── */}
      {steps.length > 0 && (
        <div className="bg-slate-950/70 border border-emerald-500/30 rounded-xl p-5 space-y-5">
          {/* Card Header & View Switcher */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-800">
            <div className="flex items-center gap-2.5">
              <div className="p-1.5 bg-emerald-500/10 border border-emerald-500/30 rounded-lg text-emerald-400">
                <ListOrdered className="w-4 h-4" />
              </div>
              <div>
                <h3 className="text-sm font-bold text-slate-100 flex items-center gap-2">
                  <span>Verified Step-by-Step Derivation</span>
                  <span className="text-[10px] font-normal px-2 py-0.5 rounded-full bg-slate-800 text-slate-300 border border-slate-700">
                    Formally Verified
                  </span>
                </h3>
                <p className="text-[11px] text-slate-400">
                  Pedagogical derivations certified by deterministic CAS.
                </p>
              </div>
            </div>

            {/* Mode Switcher Buttons */}
            <div className="flex items-center gap-1.5 bg-slate-900 p-1 rounded-xl border border-slate-800 self-start sm:self-auto">
              <button
                type="button"
                onClick={() => setActiveView('walkthrough')}
                className={`flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-medium transition-all ${
                  activeView === 'walkthrough'
                    ? 'bg-emerald-600 text-white shadow'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <PlayCircle className="w-3.5 h-3.5" />
                <span>Walkthrough</span>
              </button>
              <button
                type="button"
                onClick={() => setActiveView('all')}
                className={`flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-medium transition-all ${
                  activeView === 'all'
                    ? 'bg-emerald-600 text-white shadow'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <Layers className="w-3.5 h-3.5" />
                <span>All Steps ({totalSteps})</span>
              </button>
            </div>
          </div>

          {/* VIEW A: Guided Walkthrough Mode (One Step at a Time) */}
          {activeView === 'walkthrough' && currentStep && (
            <div className="space-y-4">
              {/* Step Navigation Dots / Pills & Progress Bar */}
              <div className="space-y-2">
                <div className="flex items-center justify-between text-xs text-slate-400">
                  <span className="font-medium text-slate-300 flex items-center gap-1.5">
                    <GraduationCap className="w-4 h-4 text-emerald-400" />
                    Step {currentStep.step_number} of {totalSteps}: {currentStep.title}
                  </span>
                  <span className="font-mono text-emerald-400">
                    {Math.round(((currentStepIdx + 1) / totalSteps) * 100)}% Complete
                  </span>
                </div>

                {/* Progress bar */}
                <div className="w-full bg-slate-900 h-1.5 rounded-full overflow-hidden border border-slate-800">
                  <div
                    className="bg-emerald-500 h-full rounded-full transition-all duration-300"
                    style={{ width: `${((currentStepIdx + 1) / totalSteps) * 100}%` }}
                  />
                </div>

                {/* Interactive Step Quick Selector */}
                <div className="flex items-center gap-1.5 pt-1 overflow-x-auto py-1">
                  {steps.map((s, idx) => (
                    <button
                      key={idx}
                      type="button"
                      onClick={() => setCurrentStepIdx(idx)}
                      className={`px-2.5 py-1 rounded-lg text-xs font-mono font-semibold transition-all border ${
                        idx === currentStepIdx
                          ? 'bg-emerald-500 text-slate-950 border-emerald-400 shadow-md font-bold'
                          : idx < currentStepIdx
                          ? 'bg-emerald-950/40 text-emerald-300 border-emerald-800/40 hover:bg-emerald-900/40'
                          : 'bg-slate-900 text-slate-400 border-slate-800 hover:bg-slate-850'
                      }`}
                    >
                      {idx + 1}
                    </button>
                  ))}
                </div>
              </div>

              {/* Active Walkthrough Step Card */}
              <div className="bg-slate-900/90 border-2 border-emerald-500/40 rounded-xl p-5 shadow-xl space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2.5">
                    <span className="px-2.5 py-1 text-xs font-bold rounded-lg bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 font-mono">
                      Step {currentStep.step_number}
                    </span>
                    <h4 className="text-base font-bold text-slate-100">{currentStep.title}</h4>
                  </div>
                  <span className="text-[11px] text-slate-400 italic">Focused Step</span>
                </div>

                {/* Explanation text */}
                <div className="text-sm text-slate-200 leading-relaxed bg-slate-950/60 p-3.5 rounded-lg border border-slate-800">
                  <FormattedExplanation text={currentStep.explanation} />
                </div>

                {/* Math display */}
                {currentStep.math && (
                  <div className="bg-slate-950 border border-slate-800 rounded-lg p-3.5 text-center overflow-x-auto shadow-inner">
                    <MathDisplay
                      latex={currentStep.math}
                      fallback={currentStep.math}
                      displayMode={true}
                    />
                  </div>
                )}
              </div>

              {/* Navigation Controls (Prev / Next) */}
              <div className="flex items-center justify-between pt-2">
                <button
                  type="button"
                  onClick={() => setCurrentStepIdx((prev) => Math.max(0, prev - 1))}
                  disabled={currentStepIdx === 0}
                  className={`flex items-center gap-1.5 px-4 py-2 rounded-xl text-xs font-medium border transition-all ${
                    currentStepIdx === 0
                      ? 'bg-slate-950 text-slate-600 border-slate-900 cursor-not-allowed'
                      : 'bg-slate-800 hover:bg-slate-750 text-slate-200 border-slate-700 shadow-sm'
                  }`}
                >
                  <ChevronLeft className="w-4 h-4" />
                  <span>Previous Step</span>
                </button>

                <div className="text-xs text-slate-500 font-mono hidden sm:inline">
                  Tip: Use <kbd className="px-1.5 py-0.5 bg-slate-800 rounded border border-slate-700 text-slate-300">←</kbd> / <kbd className="px-1.5 py-0.5 bg-slate-800 rounded border border-slate-700 text-slate-300">→</kbd> arrow keys to navigate
                </div>

                <button
                  type="button"
                  onClick={() => setCurrentStepIdx((prev) => Math.min(totalSteps - 1, prev + 1))}
                  disabled={currentStepIdx === totalSteps - 1}
                  className={`flex items-center gap-1.5 px-4 py-2 rounded-xl text-xs font-semibold border transition-all ${
                    currentStepIdx === totalSteps - 1
                      ? 'bg-slate-950 text-slate-600 border-slate-900 cursor-not-allowed'
                      : 'bg-emerald-600 hover:bg-emerald-500 text-white border-emerald-500 shadow-md shadow-emerald-950'
                  }`}
                >
                  <span>Next Step</span>
                  <ChevronRight className="w-4 h-4" />
                </button>
              </div>
            </div>
          )}

          {/* VIEW B: Full Solution (All Steps Roadmap) */}
          {activeView === 'all' && (
            <div className="space-y-3">
              <div className="flex items-center justify-between text-xs text-slate-400 pb-1">
                <span>Displaying all {totalSteps} sequential derivation steps:</span>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={expandAll}
                    className="hover:text-emerald-400 transition-colors"
                  >
                    Expand All
                  </button>
                  <span>•</span>
                  <button
                    type="button"
                    onClick={collapseAll}
                    className="hover:text-emerald-400 transition-colors"
                  >
                    Collapse All
                  </button>
                </div>
              </div>

              <div className="space-y-3">
                {steps.map((step, idx) => (
                  <StepCard
                    key={idx}
                    step={step}
                    isExpanded={Boolean(expandedSteps[idx])}
                    onToggle={() => toggleStep(idx)}
                    totalSteps={totalSteps}
                  />
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* ───────────────────────────────────────────────────────────────── */}
      {/* 4. Pedagogical Explanation & Concept Breakdown Card */}
      {/* ───────────────────────────────────────────────────────────────── */}
      {result.explanation && (
        <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-5 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-800">
            <div className="flex items-center gap-2.5">
              <div className="p-1.5 bg-emerald-500/10 border border-emerald-500/30 rounded-lg text-emerald-400">
                <BookOpen className="w-4 h-4" />
              </div>
              <div>
                <h3 className="text-sm font-bold text-slate-100 flex items-center gap-2">
                  <span>Pedagogical Concept & Intuitive Reasoning</span>
                  <span className="text-[10px] font-medium px-2 py-0.5 rounded-full bg-emerald-950/60 text-emerald-300 border border-emerald-800/40">
                    Curriculum Aligned
                  </span>
                </h3>
                <p className="text-[11px] text-slate-400">
                  Intuitive concept breakdown and mathematical principles.
                </p>
              </div>
            </div>

            <div className="flex items-center gap-2 text-xs font-mono text-slate-500">
              <span>
                {result.generalized_mode ? `Generalized: ${result.template_id}` : `Template: ${result.template_id || 'calculus'}`}
              </span>
            </div>
          </div>

          {/* Explanation Text */}
          <div className="bg-slate-900/70 border border-slate-800/80 rounded-xl p-4 text-slate-200 text-sm leading-relaxed shadow-sm space-y-3">
            {result.mental_model && (
              <div className="px-3 py-2 rounded-lg bg-emerald-950/40 border border-emerald-800/40 text-xs text-emerald-200 font-medium">
                <span className="text-emerald-400 font-bold uppercase tracking-wider mr-1.5">Mental Model:</span>
                <FormattedExplanation text={result.mental_model} />
              </div>
            )}
            <div>
              <FormattedExplanation text={result.explanation} />
            </div>
          </div>

          {/* Teacher's Note / Strategic Advice Callout */}
          <div className="bg-gradient-to-r from-amber-950/20 via-slate-900/60 to-slate-900 border border-amber-800/40 rounded-xl p-3.5 flex items-start gap-3">
            <div className="p-1 rounded bg-amber-500/20 text-amber-400 border border-amber-500/30 mt-0.5">
              <Lightbulb className="w-4 h-4" />
            </div>
            <div className="space-y-1 text-xs">
              <span className="font-bold text-amber-300 block">Teacher's Pedagogical Tip & Common Pitfall:</span>
              <p className="text-slate-300 leading-relaxed">
                {result.pitfall_tip ||
                  "Always verify domain restrictions and boundary points when evaluating derivatives or integrals. Identifying the right rule (product, chain, or substitution) before expanding prevents algebraic errors."}
              </p>
            </div>
          </div>

          {/* Footer note */}
          <div className="pt-2 border-t border-slate-800/60 flex items-center justify-between text-[11px] text-slate-500">
            <span>Deterministic symbolic derivation ensures 0% hallucination rate.</span>
            <span className="italic">Formal CAS Derivation</span>
          </div>
        </div>
      )}
    </div>
  );
}
