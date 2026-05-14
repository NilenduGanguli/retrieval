import { Activity, Coins, Cpu, Database, Timer } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

import { api } from '@/lib/api'
import type { AnalyticsSummary, QueryLogEntry } from '@/types'

function fmtTokens(n: number | null | undefined): string {
  if (n == null || isNaN(Number(n))) return '—'
  const v = Number(n)
  if (v >= 1_000_000) return (v / 1_000_000).toFixed(2) + 'M'
  if (v >= 1_000) return (v / 1_000).toFixed(1) + 'k'
  return String(Math.round(v))
}

export default function AnalyticsTab() {
  const [summary, setSummary] = useState<AnalyticsSummary | null>(null)
  const [queries, setQueries] = useState<QueryLogEntry[]>([])

  useEffect(() => {
    let alive = true
    const load = async () => {
      try {
        const [s, q] = await Promise.all([api.analyticsSummary(), api.recentQueries(60)])
        if (!alive) return
        setSummary(s)
        setQueries(q)
      } catch (e) {
        console.warn('analytics load failed', e)
      }
    }
    load()
    const t = setInterval(load, 8000)
    return () => {
      alive = false
      clearInterval(t)
    }
  }, [])

  const strategyData = Object.entries(summary?.strategy_mix || {}).map(([k, v]) => ({
    name: k.replace(/^use_/, ''),
    count: v,
  }))

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 lg:grid-cols-6 gap-3">
        <Stat label="Total queries" value={summary?.total_queries ?? '—'} icon={Activity} />
        <Stat label="Last 24h" value={summary?.queries_24h ?? '—'} icon={Timer} />
        <Stat
          label="p50"
          value={summary?.p50_latency_ms != null ? `${summary.p50_latency_ms.toFixed(0)} ms` : '—'}
          icon={Timer}
        />
        <Stat
          label="p95"
          value={summary?.p95_latency_ms != null ? `${summary.p95_latency_ms.toFixed(0)} ms` : '—'}
          icon={Timer}
        />
        <Stat
          label="tokens (total)"
          value={fmtTokens(summary?.token_totals?.total)}
          icon={Coins}
        />
        <Stat
          label="tokens (24h)"
          value={fmtTokens(summary?.token_totals?.total_24h)}
          icon={Coins}
        />
      </div>

      {summary?.token_totals && (summary.token_totals.prompt > 0 || summary.token_totals.completion > 0) && (
        <div className="card-soft p-3 flex flex-wrap items-center gap-4 text-xs">
          <span className="text-citi-blue uppercase tracking-wider text-[10px]">token breakdown</span>
          <span><span className="text-citi-blue">prompt</span>{' '}<span className="text-ink font-medium tabular-nums">{fmtTokens(summary.token_totals.prompt)}</span></span>
          <span><span className="text-citi-blue">completion</span>{' '}<span className="text-ink font-medium tabular-nums">{fmtTokens(summary.token_totals.completion)}</span></span>
          <span><span className="text-citi-blue">total</span>{' '}<span className="text-accent-dark font-semibold tabular-nums">{fmtTokens(summary.token_totals.total)}</span></span>
          {summary.token_totals.total > 0 && summary.total_queries > 0 && (
            <span className="ml-auto text-citi-blue">
              avg/query: <span className="text-ink tabular-nums">{Math.round(summary.token_totals.total / summary.total_queries)}</span>
            </span>
          )}
        </div>
      )}

      {/* Token spend by stage — aggregate across all logged queries */}
      {summary?.stage_token_breakdown && summary.stage_token_breakdown.length > 0 && (
        <div className="card p-4">
          <h3 className="text-sm font-medium mb-3 flex items-center gap-2">
            <Coins className="w-4 h-4 text-accent" />
            Tokens by pipeline stage (lifetime)
          </h3>
          <StageTokenBars rows={summary.stage_token_breakdown} />
        </div>
      )}

      <div className="grid grid-cols-12 gap-4">
        <div className="col-span-12 lg:col-span-7 card p-4">
          <h3 className="text-sm font-medium mb-3">Strategy mix (last 500 queries)</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={strategyData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f1f24" />
                <XAxis dataKey="name" stroke="#6b7280" tick={{ fontSize: 11 }} />
                <YAxis stroke="#6b7280" tick={{ fontSize: 11 }} />
                <Tooltip
                  contentStyle={{ background: '#16161a', border: '1px solid #1f1f24', borderRadius: 8 }}
                  labelStyle={{ color: '#e5e7eb' }}
                />
                <Bar dataKey="count" fill="#7c5cff" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="col-span-12 lg:col-span-5 card p-4">
          <h3 className="text-sm font-medium mb-3 flex items-center gap-2">
            <Database className="w-4 h-4 text-accent" />
            Top retrieved documents
          </h3>
          <div className="space-y-2">
            {(summary?.top_documents || []).map(d => (
              <div key={d.document_name} className="card-soft p-2 flex items-center justify-between">
                <span className="text-xs truncate flex-1 text-ink">{d.document_name}</span>
                <span className="chip text-[10px]">{d.appearances}</span>
              </div>
            ))}
            {!summary?.top_documents?.length && (
              <div className="text-xs text-citi-blue py-4 text-center">No queries yet.</div>
            )}
          </div>
        </div>
      </div>

      <div className="card overflow-hidden">
        <div className="px-4 py-2 border-b border-line flex items-center justify-between">
          <h3 className="text-sm font-medium">Recent queries</h3>
          <span className="text-xs text-citi-blue">{queries.length} shown</span>
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full text-xs">
            <thead className="bg-bg-soft text-citi-blue">
              <tr>
                <th className="text-left px-3 py-2 font-medium">When</th>
                <th className="text-left px-3 py-2 font-medium">Query</th>
                <th className="text-left px-3 py-2 font-medium">Strategy</th>
                <th className="text-right px-3 py-2 font-medium">Latency</th>
                <th className="text-right px-3 py-2 font-medium">Tokens</th>
                <th className="text-right px-3 py-2 font-medium">CRAG</th>
                <th className="text-right px-3 py-2 font-medium">#hits</th>
              </tr>
            </thead>
            <tbody>
              {queries.map(q => {
                const tok = (q as any).token_usage as { total?: number } | undefined
                return (
                  <tr key={q.id} className="border-t border-line/60 hover:bg-bg-soft/40">
                    <td className="px-3 py-2 text-citi-blue whitespace-nowrap">
                      {new Date(q.created_at).toLocaleTimeString()}
                    </td>
                    <td className="px-3 py-2 max-w-md truncate">{q.query_text}</td>
                    <td className="px-3 py-2">
                      <div className="flex flex-wrap gap-1">
                        {Object.entries(q.strategy)
                          .filter(([, v]) => v === true)
                          .map(([k]) => (
                            <span key={k} className="chip text-[10px]">{k}</span>
                          ))}
                      </div>
                    </td>
                    <td className="px-3 py-2 text-right text-ink">
                      {q.latency_ms.total != null ? `${Number(q.latency_ms.total).toFixed(0)} ms` : '—'}
                    </td>
                    <td className="px-3 py-2 text-right text-ink tabular-nums">
                      {fmtTokens(tok?.total)}
                    </td>
                    <td className="px-3 py-2 text-right">
                      {q.crag_confidence != null ? `${(q.crag_confidence * 100).toFixed(0)}%` : '—'}
                    </td>
                    <td className="px-3 py-2 text-right">{q.top_chunk_ids.length}</td>
                  </tr>
                )
              })}
              {!queries.length && (
                <tr>
                  <td colSpan={7} className="px-3 py-8 text-center text-citi-blue text-xs">
                    No queries logged yet.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

function Stat({
  label, value, icon: Icon,
}: {
  label: string; value: string | number; icon: React.ComponentType<{ className?: string }>
}) {
  return (
    <div className="card p-3">
      <div className="flex items-center gap-2 text-[10px] uppercase text-citi-blue mb-1">
        <Icon className="w-3 h-3" />
        {label}
      </div>
      <div className="text-lg font-semibold">{value}</div>
    </div>
  )
}

const STAGE_PALETTE: Record<string, string> = {
  rewrite:  'bg-fuchsia-500/70',
  hyde:     'bg-pink-500/70',
  rerank:   'bg-amber-500/70',
  crag:     'bg-red-500/70',
  generate: 'bg-emerald-500/70',
}

function StageTokenBars({
  rows,
}: {
  rows: Array<{ stage: string; total: number; prompt: number; completion: number; uses: number }>
}) {
  const max = Math.max(1, ...rows.map(r => r.total))
  return (
    <div className="space-y-2">
      {rows.map(r => {
        const pct = (r.total / max) * 100
        const color = STAGE_PALETTE[r.stage] ?? 'bg-slate-400/70'
        return (
          <div key={r.stage} className="flex items-center gap-3 text-xs">
            <div className="w-20 shrink-0 text-ink font-medium">{r.stage}</div>
            <div className="flex-1 h-3 rounded-md bg-bg-soft overflow-hidden border border-line">
              <div
                className={`h-full ${color} transition-all`}
                style={{ width: `${pct}%` }}
              />
            </div>
            <div className="w-44 shrink-0 text-right tabular-nums text-citi-blue">
              <span className="text-ink font-medium">{fmtTokens(r.total)}</span>{' '}
              <span className="opacity-70">({fmtTokens(r.prompt)} in / {fmtTokens(r.completion)} out)</span>
            </div>
            <div className="w-16 shrink-0 text-right text-citi-blue tabular-nums">{r.uses}× used</div>
          </div>
        )
      })}
    </div>
  )
}
