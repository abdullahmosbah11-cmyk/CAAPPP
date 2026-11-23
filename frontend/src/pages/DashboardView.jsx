import React, { useState, useEffect } from 'react';
import {
  RefreshCw, XCircle, Clock, Sparkles, ShieldCheck, Database, Layers, ShieldAlert, AlertTriangle
} from 'lucide-react';

export default function DashboardView({ apiBaseUrl }) {
  const [stats, setStats] = useState(null);
  const [recent, setRecent] = useState([]);
  const [failures, setFailures] = useState([]);
  const [connectors, setConnectors] = useState(null);
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(false);
  const [summaryLoading, setSummaryLoading] = useState(false);
  const [loadError, setLoadError] = useState(null);

  const fetchDashboardData = async () => {
    setLoading(true);
    setLoadError(null);
    try {
      const [statsRes, recentRes, failRes, connRes] = await Promise.all([
        fetch(`${apiBaseUrl}/dashboard/stats`),
        fetch(`${apiBaseUrl}/dashboard/recent?limit=15`),
        fetch(`${apiBaseUrl}/dashboard/failures`),
        fetch(`${apiBaseUrl}/connectors/status`),
      ]);

      if (!statsRes.ok && !recentRes.ok) {
        throw new Error(`Telemetry API unavailable (HTTP ${statsRes.status})`);
      }

      if (statsRes.ok) setStats(await statsRes.json());
      if (recentRes.ok) setRecent(await recentRes.json());
      if (failRes.ok) setFailures(await failRes.json());
      if (connRes.ok) setConnectors(await connRes.json());
    } catch (err) {
      setLoadError(err.message || 'Failed to reach VeriMath telemetry backend.');
    } finally {
      setLoading(false);
    }
  };

  const fetchGeminiSummary = async () => {
    setSummaryLoading(true);
    try {
      const res = await fetch(`${apiBaseUrl}/dashboard/summary`);
      if (res.ok) {
        const data = await res.json();
        setSummary(data.summary);
      }
    } catch (err) {
      console.error('Failed to fetch Gemini summary:', err);
    } finally {
      setSummaryLoading(false);
    }
  };

  useEffect(() => {
    fetchDashboardData();
  }, [apiBaseUrl]);

  const verdictBadgeClass = (verdict) => {
    if (verdict === 'VERIFIED') return 'bg-emerald-950 text-emerald-400 border border-emerald-800';
    if (verdict === 'UNSUPPORTED') return 'bg-amber-950 text-amber-300 border border-amber-800';
    return 'bg-rose-950 text-rose-400 border border-rose-800';
  };

  return (
    <div className="space-y-6">
      {/* Top Banner */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-100 flex flex-wrap items-center gap-2">
            <span>Classroom IoT Telemetry Dashboard</span>
            <span className="text-xs bg-indigo-500/20 text-indigo-400 border border-indigo-500/30 px-2.5 py-0.5 rounded-full font-mono">
              MQTT → SQLite Telemetry
            </span>
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Real-time monitoring of student submissions, formal Lean 4 verdicts, AST audits, and disproven false claims.
          </p>
        </div>

        <button
          type="button"
          onClick={fetchDashboardData}
          disabled={loading}
          className="flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-xl text-xs font-medium border border-slate-700 transition-colors shrink-0"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
          <span>Refresh Telemetry</span>
        </button>
      </div>

      {loadError && (
        <div className="bg-amber-950/30 border border-amber-500/40 rounded-2xl p-4 text-xs font-mono text-amber-200 flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
          <span>{loadError}</span>
        </div>
      )}

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg">
          <div className="flex items-center justify-between text-slate-400 text-xs mb-2">
            <span>Total Submissions</span>
            <Database className="w-4 h-4 text-slate-500" />
          </div>
          <div className="text-3xl font-bold font-mono text-slate-100">
            {stats ? stats.total_problems : '—'}
          </div>
          <div className="text-[11px] text-slate-500 mt-1">Logged via telemetry stream</div>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg">
          <div className="flex items-center justify-between text-slate-400 text-xs mb-2">
            <span>Verification Pass Rate</span>
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-3xl font-bold font-mono text-emerald-400">
            {stats ? `${(stats.pass_rate * 100).toFixed(1)}%` : '—'}
          </div>
          <div className="text-[11px] text-emerald-500/80 mt-1">
            {stats ? `${stats.verified_count} verified truths` : '—'}
          </div>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg">
          <div className="flex items-center justify-between text-slate-400 text-xs mb-2">
            <span>Claims Disproven / Rejected</span>
            <XCircle className="w-4 h-4 text-rose-400" />
          </div>
          <div className="text-3xl font-bold font-mono text-rose-400">
            {stats ? stats.rejected_count : '—'}
          </div>
          <div className="text-[11px] text-rose-500/80 mt-1">Lean kernel interceptions</div>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg">
          <div className="flex items-center justify-between text-slate-400 text-xs mb-2">
            <span>Avg Response Latency</span>
            <Clock className="w-4 h-4 text-indigo-400" />
          </div>
          <div className="text-3xl font-bold font-mono text-indigo-400">
            {stats ? `${Math.round(stats.avg_elapsed_ms)} ms` : '—'}
          </div>
          <div className="text-[11px] text-slate-500 mt-1">Cache target: ≤ 4 ms</div>
        </div>
      </div>

      {/* External Connectors Readiness */}
      {connectors && (
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-lg flex flex-wrap items-center justify-between gap-3 text-xs">
          <div className="font-semibold text-slate-300 flex items-center gap-2">
            <span>External Connectors &amp; Peripherals:</span>
          </div>
          <div className="flex flex-wrap items-center gap-2 font-mono">
            <span className="px-2.5 py-1 rounded-lg bg-emerald-950/60 text-emerald-300 border border-emerald-700/50">
              OCR Models: {(connectors.available_ocr_list || []).join(', ') || 'none'}
            </span>
            <span className="px-2.5 py-1 rounded-lg bg-cyan-950/60 text-cyan-300 border border-cyan-700/50">
              Cameras: browser WebRTC{(connectors.hardware_cameras || []).length ? ` + ${connectors.hardware_cameras.join(', ')}` : ''}
            </span>
            <span className={`px-2.5 py-1 rounded-lg border ${
              connectors.mqtt?.reachable
                ? 'bg-emerald-950/60 text-emerald-300 border-emerald-700/50'
                : 'bg-slate-950 text-slate-400 border-slate-700'
            }`}>
              MQTT ({connectors.mqtt?.host}:{connectors.mqtt?.port}): {connectors.mqtt?.reachable ? 'online' : 'offline-safe'}
            </span>
            <span className={`px-2.5 py-1 rounded-lg border ${
              connectors.firebase?.configured
                ? 'bg-emerald-950/60 text-emerald-300 border-emerald-700/50'
                : 'bg-slate-950 text-slate-400 border-slate-700'
            }`}>
              Firebase: {connectors.firebase?.configured ? 'connected' : 'local-only'}
            </span>
          </div>
        </div>
      )}

      {/* Gemini AI Session Analysis */}
      <div className="bg-slate-900 border border-indigo-500/30 rounded-2xl p-6 shadow-xl space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2 text-indigo-300 font-semibold text-sm">
            <Sparkles className="w-4 h-4 text-indigo-400" />
            <span>Teacher Pedagogical Insights</span>
          </div>
          <button
            type="button"
            onClick={fetchGeminiSummary}
            disabled={summaryLoading}
            className="flex items-center gap-1.5 px-3 py-1 bg-indigo-600/30 hover:bg-indigo-600/50 text-indigo-200 border border-indigo-500/40 rounded-lg text-xs transition-colors"
          >
            <RefreshCw className={`w-3 h-3 ${summaryLoading ? 'animate-spin' : ''}`} />
            <span>{summary ? 'Regenerate Analysis' : 'Analyze Session'}</span>
          </button>
        </div>

        <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-4 text-sm text-slate-300 leading-relaxed">
          {summary ? (
            <div className="whitespace-pre-line">{summary}</div>
          ) : (
            <span className="text-slate-500 italic">
              Click 'Analyze Session' to generate pedagogical insights across the classroom telemetry stream.
            </span>
          )}
        </div>
      </div>

      {/* Intercepted Hallucinations / Disproven False Claims */}
      <div className="bg-slate-900 border border-rose-500/20 rounded-2xl p-6 shadow-xl space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-rose-200 flex items-center gap-2">
            <ShieldAlert className="w-4 h-4 text-rose-400" />
            <span>Intercepted Hallucinations &amp; Disproven False Claims</span>
          </h3>
          <span className="text-xs font-mono text-slate-500">{failures.length} events</span>
        </div>

        {failures.length === 0 ? (
          <div className="bg-slate-950/70 border border-slate-800 rounded-xl p-4 text-xs text-slate-500 italic text-center">
            No rejected or disproven submissions recorded in the current session.
          </div>
        ) : (
          <div className="space-y-2">
            {failures.slice(0, 8).map((ev, i) => (
              <div
                key={ev.problem_id || i}
                className="bg-slate-950/80 border border-rose-900/40 rounded-xl px-4 py-3 flex flex-col sm:flex-row sm:items-center justify-between gap-2 text-xs"
              >
                <div className="text-slate-200 font-mono truncate max-w-xl">{ev.problem_text}</div>
                <div className="flex items-center gap-2 shrink-0">
                  <span className="font-mono text-slate-500">{ev.template_id || 'unknown'}</span>
                  <span className={`px-2 py-0.5 rounded-full text-[11px] font-semibold ${verdictBadgeClass(ev.lean_verdict)}`}>
                    {ev.lean_verdict}
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Recent Problems Table */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
        <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
          <Layers className="w-4 h-4 text-slate-400" />
          <span>Recent Student Problems Log</span>
        </h3>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead>
              <tr className="border-b border-slate-800 text-slate-500 uppercase tracking-wider">
                <th className="pb-3 px-2">Problem Statement</th>
                <th className="pb-3 px-2">Template / Domain</th>
                <th className="pb-3 px-2">Kernel Verdict</th>
                <th className="pb-3 px-2 text-right">Latency</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {recent.length === 0 ? (
                <tr>
                  <td colSpan={4} className="py-6 text-center text-slate-500 italic">
                    No classroom problems recorded yet.
                  </td>
                </tr>
              ) : (
                recent.map((ev, i) => (
                  <tr key={ev.problem_id || i} className="hover:bg-slate-800/30 transition-colors">
                    <td className="py-2.5 px-2 text-slate-300 font-sans truncate max-w-[280px]">
                      {ev.problem_text}
                    </td>
                    <td className="py-2.5 px-2 text-slate-400 font-mono">
                      {ev.template_id || 'none'}
                    </td>
                    <td className="py-2.5 px-2 font-semibold">
                      <span className={`px-2 py-0.5 rounded-full text-[11px] ${verdictBadgeClass(ev.lean_verdict)}`}>
                        {ev.lean_verdict}
                      </span>
                    </td>
                    <td className="py-2.5 px-2 text-right text-slate-400 font-mono">
                      {ev.from_cache ? 'cached • ' : ''}{ev.elapsed_ms} ms
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
