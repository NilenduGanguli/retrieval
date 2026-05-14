import { cn } from '@/lib/cn'

import type { StagesMap } from './PipelineTracker'

/**
 * Stacked horizontal bar that visualises *token spend per pipeline stage*,
 * mirroring the LatencyBar's design. Stages that don't make LLM calls
 * (dense, sparse, fuse, mmr) never contribute to this bar.
 */

type Seg = { key: string; label: string; tok: number; color: string }

const PALETTE: Record<string, string> = {
  rewrite:  'bg-fuchsia-500/70',
  hyde:     'bg-pink-500/70',
  rerank:   'bg-amber-500/70',
  crag:     'bg-red-500/70',
  generate: 'bg-emerald-500/70',
}

const ORDER = ['rewrite', 'hyde', 'rerank', 'crag', 'generate']

const fmt = (n: number) => {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(2) + 'M'
  if (n >= 1_000) return (n / 1_000).toFixed(1) + 'k'
  return String(Math.round(n))
}

export default function TokenBar({
  stages,
  fallbackTotal,
}: {
  stages: StagesMap
  /** Render this many tokens as a single 'generate' bar when no stage
   *  tokens are available (e.g. mid-stream before the final usage arrives). */
  fallbackTotal?: number
}) {
  const segs: Seg[] = ORDER.flatMap(key => {
    const st = stages[key]
    const tok = Number(st?.tokens?.total ?? 0)
    if (!tok) return []
    return [{ key, label: key, tok, color: PALETTE[key] ?? 'bg-slate-400/70' }]
  })

  if (segs.length === 0 && fallbackTotal) {
    segs.push({ key: 'generate', label: 'generate', tok: fallbackTotal, color: PALETTE.generate })
  }

  const total = segs.reduce((a, b) => a + b.tok, 0)

  if (total === 0) {
    return (
      <div className="text-xs text-citi-blue text-center py-2">
        No LLM stages ran for this query — embedding-only retrieval consumes no chat tokens.
      </div>
    )
  }

  return (
    <div className="space-y-2">
      <div className="flex h-3 rounded-md overflow-hidden border border-line">
        {segs.map(seg => {
          const w = (seg.tok / total) * 100
          if (w < 0.5) return null
          return (
            <div
              key={seg.key}
              className={cn(seg.color, 'transition')}
              style={{ width: `${w}%` }}
              title={`${seg.label}: ${fmt(seg.tok)} tokens`}
            />
          )
        })}
      </div>
      <div className="flex flex-wrap gap-2 text-[10px] text-citi-blue">
        {segs.map(seg => {
          const st = stages[seg.key]
          const tokIn = Number(st?.tokens?.prompt ?? 0)
          const tokOut = Number(st?.tokens?.completion ?? 0)
          return (
            <span key={seg.key} className="inline-flex items-center gap-1.5">
              <span className={cn('w-2 h-2 rounded-sm', seg.color)} />
              <span>{seg.label}</span>
              <span className="text-ink tabular-nums">{fmt(seg.tok)}</span>
              {(tokIn || tokOut) ? (
                <span className="opacity-70 tabular-nums">({fmt(tokIn)}/{fmt(tokOut)})</span>
              ) : null}
            </span>
          )
        })}
      </div>
    </div>
  )
}
