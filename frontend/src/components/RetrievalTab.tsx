import { ArrowRight, Clock, Coins, Cpu, Sparkles, Wand2, Layers, Shield } from 'lucide-react'
import React, { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'

import { postSse } from '@/lib/api'
import { cn } from '@/lib/cn'
import type { BackendConfig, ChunkHit, Citation, LatencyMs, Strategy } from '@/types'

import LatencyBar from './LatencyBar'
import PersistedDocuments from './PersistedDocuments'
import PipelineTracker, { applyStageEvent, type StagesMap } from './PipelineTracker'
import SourceCard from './SourceCard'
import StrategyToggles from './StrategyToggles'
import TokenBadge from './TokenBadge'
import TokenBar from './TokenBar'

type Props = { config: BackendConfig | null }

type Phase = 'idle' | 'retrieving' | 'streaming' | 'done' | 'error'

export default function RetrievalTab({ config }: Props) {
  const [query, setQuery] = useState('')
  const [strategy, setStrategy] = useState<Strategy>({
    hybrid: true, mmr: true,
    rewrite: null, hyde: null, rerank: null, crag: null, use_contextual: null,
    top_k: null,
  })
  const [phase, setPhase] = useState<Phase>('idle')
  const [meta, setMeta] = useState<{
    hits: ChunkHit[]
    latency_ms: LatencyMs
    strategy: Record<string, unknown>
    retrieval_ms: number
    rewritten_queries: string[]
    hyde_text: string | null
    crag_confidence: number | null
  } | null>(null)
  const [answer, setAnswer] = useState('')
  const [citations, setCitations] = useState<Citation[]>([])
  const [totalMs, setTotalMs] = useState<number | null>(null)
  const [genMs, setGenMs] = useState<number | null>(null)
  const [errMsg, setErrMsg] = useState<string | null>(null)
  const [stages, setStages] = useState<StagesMap>({})
  const [tokens, setTokens] = useState<{ prompt: number; completion: number; total: number } | null>(null)
  const abortRef = useRef<(() => void) | null>(null)
  const answerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    return () => {
      abortRef.current?.()
    }
  }, [])

  function send() {
    if (!query.trim() || phase === 'retrieving' || phase === 'streaming') return
    setPhase('retrieving')
    setMeta(null)
    setAnswer('')
    setCitations([])
    setTotalMs(null)
    setGenMs(null)
    setErrMsg(null)
    setStages({})
    setTokens(null)

    abortRef.current?.()
    abortRef.current = postSse(
      '/api/chat',
      { query, strategy, history: [] },
      (event, data) => {
        if (event === 'stage') {
          try {
            const p = JSON.parse(data)
            setStages(prev => applyStageEvent(prev, p))
          } catch { /* ignore */ }
        } else if (event === 'meta') {
          const m = JSON.parse(data)
          setMeta(m)
          setPhase('streaming')
        } else if (event === 'token') {
          setAnswer(prev => prev + data)
        } else if (event === 'done') {
          const d = JSON.parse(data)
          setCitations(d.citations || [])
          setGenMs(d.generate_ms)
          setTotalMs(d.total_ms)
          if (d.token_usage) setTokens(d.token_usage)
          setPhase('done')
        } else if (event === 'error') {
          setErrMsg(data)
          setPhase('error')
        }
      },
      (err) => {
        setErrMsg(String(err))
        setPhase('error')
      },
    )
  }

  useEffect(() => {
    if (answerRef.current) {
      answerRef.current.scrollTop = answerRef.current.scrollHeight
    }
  }, [answer])

  return (
    <div className="grid grid-cols-12 gap-6">
      {/* Left: Strategy + query input */}
      <aside className="col-span-12 lg:col-span-3 space-y-4">
        <div className="card p-4 space-y-3">
          <div className="flex items-center gap-2">
            <Wand2 className="w-4 h-4 text-accent" />
            <h2 className="font-medium text-sm">Strategy</h2>
          </div>
          <p className="text-xs text-muted leading-relaxed">
            Toggle each stage to demo its impact live. Defaults reflect server config.
          </p>
          <StrategyToggles
            strategy={strategy}
            onChange={setStrategy}
            defaults={config?.defaults}
          />
        </div>

        {config && (
          <div className="card-soft p-3 text-[11px] space-y-1 text-muted">
            <div className="flex items-center gap-1.5">
              <Cpu className="w-3 h-3" />
              <span>models</span>
            </div>
            <div><span className="text-zinc-400">emb:</span> {config.embedding_model}</div>
            <div><span className="text-zinc-400">gen:</span> {config.final_gen_model}</div>
            <div><span className="text-zinc-400">rerank:</span> {config.rerank_model}</div>
            <div><span className="text-zinc-400">fast:</span> {config.fast_model}</div>
          </div>
        )}

        <PersistedDocuments
          hits={meta?.hits ?? []}
          s3Enabled={config?.s3?.enabled ?? false}
        />
      </aside>

      {/* Center: chat + answer */}
      <section className="col-span-12 lg:col-span-6 space-y-4">
        <div className="card p-4 space-y-3">
          <div className="flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-accent" />
            <h2 className="font-medium text-sm">Ask anything from your corpus</h2>
          </div>
          <div className="flex gap-2">
            <textarea
              value={query}
              onChange={e => setQuery(e.target.value)}
              onKeyDown={e => {
                if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
                  e.preventDefault()
                  send()
                }
              }}
              placeholder="What does the document say about... (⌘/Ctrl+Enter to send)"
              rows={3}
              className="flex-1 bg-bg-soft border border-line rounded-lg p-3 text-sm
                         placeholder:text-muted focus:outline-none focus:border-accent/60
                         resize-none"
            />
            <button
              onClick={send}
              disabled={!query.trim() || phase === 'retrieving' || phase === 'streaming'}
              className={cn(
                'inline-flex items-center gap-2 px-4 rounded-lg text-sm font-medium transition',
                'bg-accent text-white hover:bg-accent-soft',
                'disabled:opacity-40 disabled:cursor-not-allowed',
              )}
            >
              <ArrowRight className="w-4 h-4" />
              Ask
            </button>
          </div>
          {meta && meta.rewritten_queries?.length > 1 && (
            <div className="flex flex-wrap gap-1.5 pt-1">
              <span className="text-[11px] text-muted">rewrites:</span>
              {meta.rewritten_queries.slice(1).map((q, i) => (
                <span key={i} className="chip-accent">{q}</span>
              ))}
            </div>
          )}
        </div>

        {/* Live pipeline tracker — visible during retrieval/streaming */}
        {(phase === 'retrieving' || phase === 'streaming' || phase === 'done') && (
          <PipelineTracker stages={stages} />
        )}

        {/* Answer card */}
        <div className="answer-card">
          {phase === 'idle' && (
            <div className="text-muted text-sm flex flex-col items-center justify-center h-full py-12">
              <Layers className="w-10 h-10 mb-2 opacity-30" />
              <span>Ask a question to see the pipeline work.</span>
            </div>
          )}

          {phase === 'retrieving' && Object.keys(stages).length === 0 && (
            <PipelineProgress message="Warming up…" />
          )}

          {(phase === 'streaming' || phase === 'done') && (
            <>
              <article ref={answerRef} className="answer-body">
                <AnswerMarkdown text={answer} citations={citations} hits={meta?.hits || []} />
                {phase === 'streaming' && (
                  <span className="inline-block w-2 h-4 bg-accent ml-1 animate-pulse-slow align-middle" />
                )}
              </article>

              {/* CRAG badge */}
              {meta?.crag_confidence != null && (
                <CragBadge confidence={meta.crag_confidence} />
              )}
            </>
          )}

          {phase === 'error' && (
            <div className="text-red-400 text-sm">
              <p className="font-medium">Pipeline error</p>
              <pre className="mt-2 text-xs whitespace-pre-wrap">{errMsg}</pre>
            </div>
          )}
        </div>

        {/* Latency breakdown (per-stage timings) */}
        {meta && (
          <div className="card p-4">
            <div className="flex items-center justify-between mb-2 gap-3 flex-wrap">
              <div className="flex items-center gap-2 text-sm">
                <Clock className="w-4 h-4 text-accent" />
                <span className="font-medium">Latency breakdown</span>
              </div>
              <span className="text-xs text-muted">
                total {totalMs?.toFixed(0) ?? '—'} ms
                {genMs != null && <> · generate {genMs.toFixed(0)} ms</>}
              </span>
            </div>
            <LatencyBar latency={{ ...meta.latency_ms, generate: genMs ?? 0 }} />
          </div>
        )}

        {/* Token breakdown (per-stage spend) — separate card so the user can
            read each axis without it competing with latency. */}
        {meta && (
          <div className="card p-4">
            <div className="flex items-center justify-between mb-2 gap-3 flex-wrap">
              <div className="flex items-center gap-2 text-sm">
                <Coins className="w-4 h-4 text-accent" />
                <span className="font-medium">Token breakdown</span>
              </div>
              <div className="flex items-center gap-2 text-xs text-muted">
                {tokens
                  ? <TokenBadge usage={tokens} variant="accent" label="total tokens" />
                  : <span>collecting…</span>}
              </div>
            </div>
            <TokenBar stages={stages} fallbackTotal={tokens?.total ?? undefined} />
          </div>
        )}
      </section>

      {/* Right: sources */}
      <aside className="col-span-12 lg:col-span-3 space-y-3">
        <div className="flex items-center gap-2 px-1">
          <Shield className="w-4 h-4 text-accent" />
          <h2 className="font-medium text-sm">Sources</h2>
          {meta && (
            <span className="chip ml-auto">{meta.hits.length}</span>
          )}
        </div>
        <div className="space-y-2 max-h-[calc(100vh-220px)] overflow-y-auto pr-1">
          {meta?.hits.map((h) => (
            <SourceCard key={h.chunk_id} hit={h} highlighted={citations.some(c => c.chunk_id === h.chunk_id)} />
          ))}
          {!meta && <div className="text-xs text-muted px-2 py-6 text-center">No retrieval yet.</div>}
        </div>
      </aside>
    </div>
  )
}

function PipelineProgress({ message }: { message: string }) {
  const stages = ['rewrite', 'hyde', 'embed', 'dense', 'sparse', 'fuse', 'rerank', 'mmr', 'crag']
  return (
    <div className="flex flex-col items-center justify-center py-12 gap-4 text-sm text-muted">
      <div className="flex items-center gap-1.5">
        {stages.map((s, i) => (
          <div
            key={s}
            className="w-2 h-2 rounded-full bg-accent/60 animate-pulse-slow"
            style={{ animationDelay: `${i * 90}ms` }}
          />
        ))}
      </div>
      <span>{message}</span>
    </div>
  )
}

function CitationChip({ n, hit }: { n: number; hit: ChunkHit | undefined }) {
  return (
    <button
      type="button"
      title={hit ? `${hit.document_name ?? 'unknown'} · p.${hit.page_number ?? '-'} · chunk #${hit.chunk_id}` : `[${n}]`}
      onClick={() => {
        if (!hit) return
        const el = document.getElementById(`source-${hit.chunk_id}`)
        el?.scrollIntoView({ behavior: 'smooth', block: 'center' })
        el?.classList.add('ring-2', 'ring-accent')
        setTimeout(() => el?.classList.remove('ring-2', 'ring-accent'), 1200)
      }}
      className="citation-marker"
    >
      {n}
    </button>
  )
}

/**
 * Replace inline [N] tokens (and chains like [1][3]) inside a markdown text
 * node with clickable citation chips. We walk react-markdown's text
 * children rather than injecting raw HTML so escaping always behaves.
 */
function withCitations(children: React.ReactNode, hits: ChunkHit[]): React.ReactNode {
  return React.Children.map(children, (child, ci) => {
    if (typeof child !== 'string') return child
    const matches = [...child.matchAll(/\[(\d+)\]/g)]
    if (matches.length === 0) return child
    const out: React.ReactNode[] = []
    let lastIdx = 0
    matches.forEach((m, i) => {
      const start = m.index ?? 0
      if (start > lastIdx) out.push(child.slice(lastIdx, start))
      const n = parseInt(m[1], 10)
      if (n >= 1 && n <= hits.length) {
        out.push(<CitationChip key={`c-${ci}-${i}-${start}`} n={n} hit={hits[n - 1]} />)
      } else {
        out.push(m[0])
      }
      lastIdx = start + m[0].length
    })
    if (lastIdx < child.length) out.push(child.slice(lastIdx))
    return out
  })
}

function AnswerMarkdown({
  text,
  hits,
}: {
  text: string
  citations: Citation[]
  hits: ChunkHit[]
}) {
  return (
    <ReactMarkdown
      components={{
        h1: ({ children }) => (
          <h1 className="text-lg font-semibold text-zinc-100 mt-5 mb-2 tracking-tight">
            {withCitations(children, hits)}
          </h1>
        ),
        h2: ({ children }) => (
          <h2 className="text-sm font-semibold uppercase tracking-wider text-accent-soft
                         mt-6 mb-2 pb-1 border-b border-accent/20">
            {withCitations(children, hits)}
          </h2>
        ),
        h3: ({ children }) => (
          <h3 className="text-sm font-semibold text-zinc-200 mt-4 mb-1.5">
            {withCitations(children, hits)}
          </h3>
        ),
        p: ({ children }) => (
          <p className="text-sm leading-relaxed text-zinc-200 my-2">
            {withCitations(children, hits)}
          </p>
        ),
        ul: ({ children }) => (
          <ul className="list-disc list-outside pl-5 my-2 space-y-1">
            {children}
          </ul>
        ),
        ol: ({ children }) => (
          <ol className="list-decimal list-outside pl-5 my-2 space-y-1">
            {children}
          </ol>
        ),
        li: ({ children }) => (
          <li className="text-sm leading-relaxed text-zinc-200 marker:text-muted">
            {withCitations(children, hits)}
          </li>
        ),
        strong: ({ children }) => (
          <strong className="text-zinc-50 font-semibold">{children}</strong>
        ),
        em: ({ children }) => (
          <em className="text-zinc-300">{children}</em>
        ),
        code: ({ children }) => (
          <code className="bg-bg-soft border border-line rounded px-1 py-0.5 text-[12px]
                           font-mono text-accent-soft">
            {children}
          </code>
        ),
        a: ({ href, children }) => (
          <a href={href} target="_blank" rel="noreferrer"
             className="text-accent-soft hover:underline">{children}</a>
        ),
        hr: () => <hr className="my-4 border-line" />,
      }}
    >
      {text}
    </ReactMarkdown>
  )
}

function CragBadge({ confidence }: { confidence: number }) {
  const tone =
    confidence >= 0.6
      ? 'border-emerald-500/40 bg-emerald-500/10 text-emerald-300'
      : confidence >= 0.2
      ? 'border-amber-500/40 bg-amber-500/10 text-amber-300'
      : 'border-red-500/40 bg-red-500/10 text-red-300'
  return (
    <div className={cn('mt-4 inline-flex items-center gap-2 text-xs px-2 py-1 rounded-md border', tone)}>
      <Shield className="w-3 h-3" />
      CRAG confidence {(confidence * 100).toFixed(0)}%
    </div>
  )
}
