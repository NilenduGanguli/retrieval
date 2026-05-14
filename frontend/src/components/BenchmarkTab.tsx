import { FlaskConical, Play, Plus, Sparkles, Trash2, AlertCircle } from 'lucide-react'
import React, { useEffect, useRef, useState } from 'react'
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis, CartesianGrid } from 'recharts'

import { api, postSse } from '@/lib/api'
import { cn } from '@/lib/cn'
import type { BackendConfig, BenchRunResult, GoldenQuestion, Strategy } from '@/types'

import PipelineTracker, { applyStageEvent, type StagesMap } from './PipelineTracker'
import StrategyToggles from './StrategyToggles'
import TokenBadge from './TokenBadge'

const fmtPct = (v: number | null | undefined, digits = 1) =>
  v == null || isNaN(v as number) ? '—' : `${(Number(v) * 100).toFixed(digits)}%`

type Props = { config: BackendConfig | null }

export default function BenchmarkTab({ config }: Props) {
  const [questions, setQuestions] = useState<GoldenQuestion[]>([])
  const [newQ, setNewQ] = useState('')
  const [newChunkIds, setNewChunkIds] = useState('')
  const [running, setRunning] = useState(false)
  const [seeding, setSeeding] = useState(false)
  const [result, setResult] = useState<BenchRunResult | null>(null)
  const [history, setHistory] = useState<Array<{ id: number; label: string | null; metrics: Record<string, number>; n_questions: number; created_at: string }>>([])
  const [strategy, setStrategy] = useState<Strategy>({ hybrid: true, mmr: true, rewrite: null, hyde: null, rerank: null, crag: null, use_contextual: null, top_k: null })
  const [label, setLabel] = useState('')

  // live-stream state during a bench run
  const [progress, setProgress] = useState<{ processed: number; total: number } | null>(null)
  const [currentQuestion, setCurrentQuestion] = useState<{ index: number; id: number; text: string } | null>(null)
  const [stages, setStages] = useState<StagesMap>({})
  const [judging, setJudging] = useState<string | null>(null)
  const [perQuestion, setPerQuestion] = useState<Array<BenchRunResult['per_question'][number] & { stages?: StagesMap }>>([])
  const [expandedRow, setExpandedRow] = useState<number | null>(null)
  const [tokensSoFar, setTokensSoFar] = useState<{ prompt: number; completion: number; total: number }>({ prompt: 0, completion: 0, total: 0 })
  const [finalTokens, setFinalTokens] = useState<{ prompt: number; completion: number; total: number } | null>(null)
  const abortRef = useRef<(() => void) | null>(null)
  const stagesRef = useRef<StagesMap>({})
  useEffect(() => { stagesRef.current = stages }, [stages])
  useEffect(() => () => { abortRef.current?.() }, [])

  const reloadQuestions = async () => setQuestions(await api.listQuestions())
  const reloadRuns = async () => setHistory(await api.listRuns())

  useEffect(() => { reloadQuestions(); reloadRuns() }, [])

  async function addQ() {
    if (!newQ.trim()) return
    const ids = newChunkIds.split(/[, ]+/).filter(Boolean).map(s => parseInt(s, 10)).filter(n => !isNaN(n))
    await api.addQuestion({ question: newQ.trim(), ground_truth_chunk_ids: ids, tags: [] })
    setNewQ('')
    setNewChunkIds('')
    reloadQuestions()
  }

  async function deleteQ(id?: number) {
    if (id == null) return
    await api.deleteQuestion(id)
    reloadQuestions()
  }

  async function seedFromDocs() {
    setSeeding(true)
    try {
      await api.seedQuestions(2)
      await reloadQuestions()
    } finally {
      setSeeding(false)
    }
  }

  function run() {
    if (running) return
    setRunning(true)
    setResult(null)
    setPerQuestion([])
    setStages({})
    setCurrentQuestion(null)
    setJudging(null)
    setProgress({ processed: 0, total: questions.length })
    setTokensSoFar({ prompt: 0, completion: 0, total: 0 })
    setFinalTokens(null)

    let aggregateMetrics: Record<string, number> = {}
    let runId: number | null = null
    let nQuestions = 0

    abortRef.current?.()
    abortRef.current = postSse(
      '/api/bench/run',
      { label: label || null, strategy, question_ids: null },
      (event, data) => {
        try {
          if (event === 'start') {
            const p = JSON.parse(data)
            setProgress({ processed: 0, total: p.n_questions })
          } else if (event === 'question_start') {
            const p = JSON.parse(data)
            setCurrentQuestion({ index: p.index, id: p.id, text: p.question })
            setStages({})
            setJudging(null)
          } else if (event === 'stage') {
            const p = JSON.parse(data)
            setStages(prev => applyStageEvent(prev, p))
          } else if (event === 'judge_start') {
            const p = JSON.parse(data)
            setJudging(p.kind)
          } else if (event === 'judge_done') {
            setJudging(null)
          } else if (event === 'question_done') {
            const p = JSON.parse(data)
            // snapshot the per-stage tokens map at this exact moment
            setPerQuestion(prev => [...prev, { ...p, stages: { ...stagesRef.current } }])
          } else if (event === 'progress') {
            const p = JSON.parse(data)
            setProgress({ processed: p.processed, total: p.total })
            if (p.tokens_so_far) setTokensSoFar(p.tokens_so_far)
          } else if (event === 'done') {
            const p = JSON.parse(data)
            aggregateMetrics = p.metrics
            runId = p.run_id
            nQuestions = p.n_questions
            if (p.tokens) setFinalTokens(p.tokens)
          } else if (event === 'error') {
            console.warn('bench error:', data)
          }
        } catch (e) { /* ignore */ }
      },
      (err) => {
        console.warn('bench stream err:', err)
      },
    )

    // Watch for stream close (done_sentinel) — simplest: poll for runs change
    // We do it by listening: when SSE closes, abort handle becomes inert.
    // Instead use a timer that flips running off when we see done was received.
    const checkDone = window.setInterval(() => {
      if (runId !== null) {
        setRunning(false)
        setResult({
          run_id: runId,
          label: label || null,
          n_questions: nQuestions,
          metrics: aggregateMetrics,
          per_question: [],   // filled incrementally by setPerQuestion above
        })
        setCurrentQuestion(null)
        setStages({})
        setJudging(null)
        reloadRuns()
        window.clearInterval(checkDone)
      }
    }, 200)
  }

  const chart = history
    .slice()
    .reverse()
    .map((h, i) => ({
      i: i + 1,
      label: h.label || `run ${h.id}`,
      ...h.metrics,
    }))

  return (
    <div className="grid grid-cols-12 gap-6">
      <section className="col-span-12 lg:col-span-7 space-y-4">
        <div className="card p-4">
          <div className="flex items-center gap-2 mb-3">
            <FlaskConical className="w-4 h-4 text-accent" />
            <h2 className="font-medium text-sm">Run a benchmark batch</h2>
          </div>
          <p className="text-xs text-muted mb-3 leading-relaxed">
            Pulls every question in the golden set, runs the pipeline with the chosen strategy, and computes:
            recall@5, recall@10, MRR@10, nDCG@10, faithfulness, context precision.
            History saved to <code className="text-accent-soft">vector.bench_runs</code>.
          </p>
          {questions.length === 0 && (
            <div className="card-soft p-3 mb-3 flex items-start gap-2 border-amber-500/40 bg-amber-500/5">
              <AlertCircle className="w-4 h-4 text-amber-300 mt-0.5 shrink-0" />
              <div className="text-xs leading-relaxed">
                <p className="text-amber-200 font-medium mb-1">
                  No golden questions yet — the Run button stays disabled.
                </p>
                <p className="text-zinc-400">
                  Click <span className="text-accent-soft">✨ Seed from documents</span> on the right
                  to auto-generate a starter set from your ingested PDFs, or add questions
                  manually in the sidebar.
                </p>
              </div>
            </div>
          )}

          <LabelPicker
            value={label}
            onChange={setLabel}
            history={history}
            currentStrategy={strategy}
          />
          <div className="flex justify-end mt-3 items-center gap-3">
            {questions.length > 0 && (
              <span className="text-xs text-muted">
                will run {questions.length} question{questions.length === 1 ? '' : 's'}
              </span>
            )}
            <button
              onClick={run}
              disabled={running || questions.length === 0}
              title={questions.length === 0 ? 'Add at least one golden question first' : 'Run benchmark'}
              className={cn(
                'inline-flex items-center gap-2 px-4 py-1.5 rounded-md text-sm font-medium',
                'bg-accent text-white hover:bg-accent-soft disabled:opacity-40 disabled:cursor-not-allowed',
              )}
            >
              <Play className="w-4 h-4" />
              {running ? 'Running…' : 'Run'}
            </button>
          </div>

          {/* Live progress tracker */}
          {(running || progress) && (
            <div className="card-soft p-3 mt-4 space-y-3">
              {progress && (
                <div className="space-y-2">
                  <div className="flex items-center gap-3">
                    <span className="text-[11px] text-muted shrink-0 tabular-nums">
                      question {Math.min(progress.processed + (running ? 1 : 0), progress.total)} / {progress.total}
                    </span>
                    <div className="flex-1 h-2 rounded-full bg-bg-soft overflow-hidden">
                      <div
                        className="h-full bg-accent transition-all"
                        style={{ width: `${progress.total ? Math.min(100, ((progress.processed) / progress.total) * 100) : 0}%` }}
                      />
                    </div>
                    <span className="text-[11px] text-muted shrink-0 tabular-nums w-12 text-right">
                      {progress.total ? Math.round((progress.processed / progress.total) * 100) : 0}%
                    </span>
                  </div>
                  <div className="flex items-center justify-between gap-2 text-[11px] text-muted">
                    <span>tokens used so far:</span>
                    <TokenBadge usage={tokensSoFar} variant="accent" />
                  </div>
                </div>
              )}
              {currentQuestion && (
                <div className="text-xs">
                  <span className="text-muted">running</span>{' '}
                  <span className="text-zinc-300">#{currentQuestion.id}</span>{' '}
                  <span className="text-zinc-100 font-medium">{currentQuestion.text.slice(0, 100)}{currentQuestion.text.length > 100 ? '…' : ''}</span>
                </div>
              )}
              {Object.keys(stages).length > 0 && (
                <PipelineTracker stages={stages} />
              )}
              {judging && (
                <div className="inline-flex items-center gap-1.5 text-[11px] text-accent-soft">
                  <span className="w-2 h-2 rounded-full bg-accent animate-pulse-slow" />
                  judging {judging.replace('_', ' ')}…
                </div>
              )}
            </div>
          )}

          {result && (
            <>
              <div className="mt-4 grid grid-cols-3 gap-2">
                {Object.entries(result.metrics).map(([k, v]) => (
                  <div key={k} className="card-soft p-2">
                    <div className="text-[10px] uppercase text-muted">{k}</div>
                    <div className="text-lg font-semibold">{fmtPct(v, 1)}</div>
                  </div>
                ))}
              </div>
              {finalTokens && (
                <div className="mt-3 flex items-center justify-between gap-2 text-xs text-muted">
                  <span>aggregate token spend for run #{result.run_id}:</span>
                  <TokenBadge usage={finalTokens} size="md" variant="accent" />
                </div>
              )}
            </>
          )}
        </div>

        <div className="card p-4">
          <h3 className="text-sm font-medium mb-3">Metric trend over runs</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chart}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f1f24" />
                <XAxis dataKey="label" stroke="#6b7280" tick={{ fontSize: 10 }} />
                <YAxis
                  stroke="#6b7280"
                  tick={{ fontSize: 10 }}
                  domain={[0, 1]}
                  tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
                />
                <Tooltip
                  contentStyle={{ background: '#16161a', border: '1px solid #1f1f24', borderRadius: 8 }}
                  labelStyle={{ color: '#e5e7eb' }}
                  formatter={(value: number, name: string) => [
                    fmtPct(value, 1),
                    name,
                  ]}
                />
                <Line type="monotone" dataKey="recall@5" stroke="#7c5cff" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="mrr@10" stroke="#34d399" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="ndcg@10" stroke="#fbbf24" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="faithfulness" stroke="#f472b6" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="context_precision" stroke="#22d3ee" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="text-[10px] text-muted mt-2 flex flex-wrap gap-3">
            <Legend color="#7c5cff" label="recall@5" />
            <Legend color="#34d399" label="MRR@10" />
            <Legend color="#fbbf24" label="nDCG@10" />
            <Legend color="#f472b6" label="faithfulness" />
            <Legend color="#22d3ee" label="context precision" />
          </div>
        </div>

        {perQuestion.length > 0 && (
          <div className="card overflow-hidden">
            <div className="px-4 py-2 border-b border-line text-sm font-medium flex items-center justify-between">
              <span>Per-question detail</span>
              <span className="text-xs text-muted">
                {perQuestion.length}{progress?.total ? ` / ${progress.total}` : ''} questions
              </span>
            </div>
            <table className="min-w-full text-xs">
              <thead className="bg-bg-soft text-muted">
                <tr>
                  <th className="text-left px-3 py-2 font-medium">#</th>
                  <th className="text-left px-3 py-2 font-medium">Question</th>
                  <th className="text-right px-3 py-2 font-medium">R@5</th>
                  <th className="text-right px-3 py-2 font-medium">MRR</th>
                  <th className="text-right px-3 py-2 font-medium">nDCG</th>
                  <th className="text-right px-3 py-2 font-medium">faith</th>
                  <th className="text-right px-3 py-2 font-medium">ctx P</th>
                  <th className="text-right px-3 py-2 font-medium">tokens</th>
                </tr>
              </thead>
              <tbody>
                {perQuestion.map(q => {
                  const open = expandedRow === q.id
                  const qStages = (q as any).stages as StagesMap | undefined
                  return (
                    <React.Fragment key={q.id}>
                      <tr
                        onClick={() => setExpandedRow(open ? null : q.id)}
                        className="border-t border-line/60 cursor-pointer hover:bg-bg-soft/40"
                      >
                        <td className="px-3 py-2 text-muted">{q.id}</td>
                        <td className="px-3 py-2 truncate max-w-md">
                          <span className="text-muted mr-1">{open ? '▾' : '▸'}</span>
                          {q.question}
                        </td>
                        <td className="px-3 py-2 text-right">{fmtPct(q.metrics['recall@5'], 0)}</td>
                        <td className="px-3 py-2 text-right">{fmtPct(q.metrics['mrr@10'], 0)}</td>
                        <td className="px-3 py-2 text-right">{fmtPct(q.metrics['ndcg@10'], 0)}</td>
                        <td className="px-3 py-2 text-right">{fmtPct(q.metrics['faithfulness'], 0)}</td>
                        <td className="px-3 py-2 text-right">{fmtPct(q.metrics['context_precision'], 0)}</td>
                        <td className="px-3 py-2 text-right">
                          {(q as any).tokens
                            ? <TokenBadge usage={(q as any).tokens} variant="plain" size="sm" />
                            : <span className="text-muted">—</span>}
                        </td>
                      </tr>
                      {open && (
                        <tr className="bg-bg-soft/30">
                          <td colSpan={8} className="px-4 py-3">
                            <div className="space-y-2">
                              <div className="text-[10px] uppercase text-muted">
                                Per-stage tokens for this question
                              </div>
                              {qStages
                                ? <PipelineTracker stages={qStages} />
                                : <div className="text-xs text-muted">no stage breakdown captured</div>}
                            </div>
                          </td>
                        </tr>
                      )}
                    </React.Fragment>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <aside className="col-span-12 lg:col-span-5 space-y-4">
        <div className="card p-4">
          <h3 className="text-sm font-medium mb-3">Strategy for this run</h3>
          <StrategyToggles strategy={strategy} onChange={setStrategy} defaults={config?.defaults} />
        </div>

        <div className="card p-4">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-medium">Golden set ({questions.length})</h3>
            <button
              onClick={seedFromDocs}
              disabled={seeding}
              title="Auto-generate 2 questions per ingested document using the active LLM"
              className={cn(
                'inline-flex items-center gap-1.5 px-2 py-1 rounded-md text-[11px]',
                'bg-accent/15 text-accent-soft border border-accent/40 hover:bg-accent/25',
                'transition disabled:opacity-50',
              )}
            >
              <Sparkles className={cn('w-3 h-3', seeding && 'animate-spin')} />
              {seeding ? 'Seeding…' : 'Seed from documents'}
            </button>
          </div>
          <div className="space-y-2 max-h-[44vh] overflow-y-auto pr-1 mb-3">
            {questions.map(q => (
              <div key={q.id} className="card-soft p-2 group">
                <div className="flex justify-between gap-2">
                  <p className="text-xs flex-1">{q.question}</p>
                  <button
                    onClick={() => deleteQ(q.id)}
                    className="opacity-0 group-hover:opacity-100 text-muted hover:text-red-400 transition"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
                {q.ground_truth_chunk_ids.length > 0 && (
                  <div className="text-[10px] text-muted mt-1">
                    ground truth: [{q.ground_truth_chunk_ids.join(', ')}]
                  </div>
                )}
              </div>
            ))}
            {!questions.length && (
              <div className="text-xs text-muted text-center py-4">No golden questions yet.</div>
            )}
          </div>
          <div className="space-y-2">
            <textarea
              value={newQ}
              onChange={e => setNewQ(e.target.value)}
              placeholder="New question…"
              rows={2}
              className="w-full bg-bg-soft border border-line rounded-md p-2 text-xs resize-none"
            />
            <input
              value={newChunkIds}
              onChange={e => setNewChunkIds(e.target.value)}
              placeholder="Ground-truth chunk ids (e.g. 12, 47, 91)"
              className="w-full bg-bg-soft border border-line rounded-md p-2 text-xs"
            />
            <button
              onClick={addQ}
              className="inline-flex items-center gap-1 px-3 py-1 rounded-md text-xs
                         bg-bg-soft border border-line hover:border-accent transition"
            >
              <Plus className="w-3 h-3" />
              Add
            </button>
          </div>
        </div>
      </aside>
    </div>
  )
}

function Legend({ color, label }: { color: string; label: string }) {
  return (
    <span className="inline-flex items-center gap-1">
      <span className="w-2 h-2 rounded-sm" style={{ background: color }} />
      {label}
    </span>
  )
}

// Common strategy combos surfaced as quick-pick checkboxes
const PRESET_TAGS = [
  'hybrid',
  'no-hybrid',
  'rerank',
  'no-rerank',
  'contextual',
  'no-contextual',
  'hyde',
  'rewrite',
  'mmr',
  'crag',
]

function LabelPicker({
  value,
  onChange,
  history,
  currentStrategy,
}: {
  value: string
  onChange: (v: string) => void
  history: Array<{ label: string | null }>
  currentStrategy: Strategy
}) {
  // Recent past labels (deduped, max 8)
  const recentLabels = Array.from(
    new Set(history.map(h => h.label).filter((l): l is string => !!l)),
  ).slice(0, 8)

  // Selected tags currently in the value (split by "+")
  const selectedTags = value.split('+').map(s => s.trim()).filter(Boolean)
  const toggleTag = (tag: string) => {
    const set = new Set(selectedTags)
    if (set.has(tag)) set.delete(tag)
    else set.add(tag)
    onChange(Array.from(set).join('+'))
  }

  // Suggest a label from the current strategy toggles
  function suggestFromStrategy() {
    const tags: string[] = []
    if (currentStrategy.rerank ?? false) tags.push('rerank'); else tags.push('no-rerank')
    if ((currentStrategy.use_contextual ?? false)) tags.push('contextual'); else tags.push('no-contextual')
    if (currentStrategy.hybrid !== false) tags.push('hybrid'); else tags.push('no-hybrid')
    if (currentStrategy.rewrite) tags.push('rewrite')
    if (currentStrategy.hyde) tags.push('hyde')
    if (currentStrategy.crag) tags.push('crag')
    if (currentStrategy.mmr !== false) tags.push('mmr')
    onChange(tags.join('+'))
  }

  return (
    <div className="space-y-2">
      <label className="block">
        <div className="text-[10px] uppercase text-muted mb-1 flex items-center gap-2">
          <span>Run label</span>
          <button
            onClick={suggestFromStrategy}
            className="ml-auto text-[10px] text-accent-soft hover:text-accent"
          >
            ✨ suggest from current strategy
          </button>
        </div>
        <input
          value={value}
          onChange={e => onChange(e.target.value)}
          placeholder='tick presets below or type, e.g. "hybrid+rerank+contextual"'
          className="w-full bg-bg-soft border border-line rounded-md px-2 py-1.5 text-sm"
        />
      </label>

      <div>
        <div className="text-[10px] uppercase text-muted mb-1">Presets</div>
        <div className="flex flex-wrap gap-1.5">
          {PRESET_TAGS.map(tag => {
            const on = selectedTags.includes(tag)
            return (
              <label
                key={tag}
                className={cn(
                  'inline-flex items-center gap-1.5 px-2 py-1 rounded-md border text-[11px] cursor-pointer transition',
                  on
                    ? 'border-accent/60 bg-accent/15 text-accent-soft shadow-glow'
                    : 'border-line bg-bg-soft/40 text-zinc-400 hover:border-accent/40',
                )}
              >
                <input
                  type="checkbox"
                  checked={on}
                  onChange={() => toggleTag(tag)}
                  className="accent-[#7c5cff] w-3 h-3"
                />
                {tag}
              </label>
            )
          })}
        </div>
      </div>

      {recentLabels.length > 0 && (
        <div>
          <div className="text-[10px] uppercase text-muted mb-1">Past run labels</div>
          <div className="flex flex-wrap gap-1.5">
            {recentLabels.map(lbl => {
              const on = value === lbl
              return (
                <label
                  key={lbl}
                  className={cn(
                    'inline-flex items-center gap-1.5 px-2 py-1 rounded-md border text-[11px] cursor-pointer transition',
                    on
                      ? 'border-emerald-500/40 bg-emerald-500/10 text-emerald-300'
                      : 'border-line bg-bg-soft/40 text-zinc-400 hover:border-emerald-500/30',
                  )}
                >
                  <input
                    type="checkbox"
                    checked={on}
                    onChange={() => onChange(on ? '' : lbl)}
                    className="accent-emerald-500 w-3 h-3"
                  />
                  {lbl}
                </label>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}
