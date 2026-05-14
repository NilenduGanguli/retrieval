import { Check, Coins, Loader2, MinusCircle } from 'lucide-react'

import { cn } from '@/lib/cn'

export type StageStatus = 'idle' | 'running' | 'done' | 'skipped'

export type StageTokens = { prompt?: number; completion?: number; total?: number }

export type StageState = {
  status: StageStatus
  ms?: number
  tokens?: StageTokens
  detail?: Record<string, unknown>
}

export type StagesMap = Record<string, StageState>

const ORDER: Array<{ key: string; label: string; hint: string; usesLLM: boolean }> = [
  { key: 'rewrite',  label: 'rewrite',  hint: 'multi-query paraphrase',         usesLLM: true  },
  { key: 'hyde',     label: 'hyde',     hint: 'hypothetical doc embeddings',    usesLLM: true  },
  { key: 'embed',    label: 'embed',    hint: 'encode query (Vertex)',          usesLLM: false },
  { key: 'dense',    label: 'dense',    hint: 'pgvector HNSW cosine',           usesLLM: false },
  { key: 'sparse',   label: 'sparse',   hint: 'Postgres FTS (BM25-ish)',        usesLLM: false },
  { key: 'fuse',     label: 'fuse',     hint: 'Reciprocal Rank Fusion',         usesLLM: false },
  { key: 'rerank',   label: 'rerank',   hint: 'LLM listwise (RankGPT)',         usesLLM: true  },
  { key: 'mmr',      label: 'mmr',      hint: 'diversify final top-K',          usesLLM: false },
  { key: 'crag',     label: 'crag',     hint: 'self-graded retrieval quality',  usesLLM: true  },
  { key: 'generate', label: 'generate', hint: 'streaming answer',               usesLLM: true  },
]

const fmtTok = (n?: number) => {
  if (n == null || !isFinite(n)) return null
  if (n >= 1000) return (n / 1000).toFixed(1) + 'k'
  return String(Math.round(n))
}

export default function PipelineTracker({ stages }: { stages: StagesMap }) {
  return (
    <div className="card-soft p-3">
      <div className="flex flex-wrap gap-1.5">
        {ORDER.map(({ key, label, hint, usesLLM }) => {
          const st = stages[key] ?? { status: 'idle' as StageStatus }
          const tokTotal = st.tokens?.total
          const tokIn = st.tokens?.prompt
          const tokOut = st.tokens?.completion
          const tooltip =
            hint +
            (st.ms != null ? ` · ${st.ms.toFixed(0)} ms` : '') +
            (tokTotal ? ` · ${tokTotal} tokens (${tokIn ?? 0} in / ${tokOut ?? 0} out)` : '')
          return (
            <div
              key={key}
              title={tooltip}
              className={cn(
                'inline-flex flex-col items-stretch px-2 py-1 rounded-md border text-[11px] transition gap-0.5',
                st.status === 'idle' &&
                  'border-line bg-bg-soft/40 text-citi-blue',
                st.status === 'running' &&
                  'border-accent/60 bg-accent/15 text-accent-dark shadow-glow',
                st.status === 'done' &&
                  'border-emerald-500/60 bg-emerald-500/15 text-emerald-700',
                st.status === 'skipped' &&
                  'border-line/40 bg-bg-soft/20 text-citi-blue line-through opacity-60',
              )}
            >
              <div className="inline-flex items-center gap-1.5">
                {st.status === 'running' && <Loader2 className="w-3 h-3 animate-spin" />}
                {st.status === 'done' && <Check className="w-3 h-3" />}
                {st.status === 'skipped' && <MinusCircle className="w-3 h-3" />}
                {st.status === 'idle' && <span className="w-2 h-2 rounded-full bg-slate-300" />}
                <span className="font-medium">{label}</span>
                {st.ms != null && (
                  <span className="text-[10px] tabular-nums opacity-70">
                    {st.ms < 1000 ? `${st.ms.toFixed(0)}ms` : `${(st.ms/1000).toFixed(1)}s`}
                  </span>
                )}
              </div>
              {/* second line: tokens when this stage uses an LLM and we have a count */}
              {usesLLM && st.status !== 'skipped' && (
                <div className="inline-flex items-center gap-1 text-[10px] tabular-nums opacity-80 self-end">
                  <Coins className="w-2.5 h-2.5" />
                  {tokTotal != null ? (
                    <span>{fmtTok(tokTotal)} <span className="opacity-60">({fmtTok(tokIn)}/{fmtTok(tokOut)})</span></span>
                  ) : (
                    <span className="opacity-40">—</span>
                  )}
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}

export function applyStageEvent(prev: StagesMap, payload: { stage: string; status: string; detail?: Record<string, unknown> }): StagesMap {
  const stage = payload.stage
  const status = payload.status
  const detail = payload.detail ?? {}
  const ms = typeof detail.ms === 'number' ? (detail.ms as number) : undefined
  const tokens = (detail.tokens as StageTokens | undefined) ?? undefined
  const next: StagesMap = { ...prev }
  const existing = next[stage] ?? { status: 'idle' as StageStatus }
  if (status === 'start') {
    next[stage] = { status: 'running', detail, tokens: existing.tokens }
  } else if (status === 'done') {
    next[stage] = { status: 'done', ms, tokens: tokens ?? existing.tokens, detail }
  } else if (status === 'skip') {
    next[stage] = { status: 'skipped', detail }
  } else if (status === 'running') {
    next[stage] = { status: 'running', detail, tokens: existing.tokens }
  }
  return next
}
