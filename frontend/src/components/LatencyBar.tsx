import { cn } from '@/lib/cn'
import type { LatencyMs } from '@/types'

const STAGES: Array<{ key: keyof LatencyMs; label: string; color: string }> = [
  { key: 'rewrite', label: 'rewrite', color: 'bg-fuchsia-500/70' },
  { key: 'hyde', label: 'hyde', color: 'bg-pink-500/70' },
  { key: 'embed', label: 'embed', color: 'bg-indigo-500/70' },
  { key: 'dense', label: 'dense', color: 'bg-blue-500/70' },
  { key: 'sparse', label: 'sparse', color: 'bg-cyan-500/70' },
  { key: 'fuse', label: 'fuse (RRF)', color: 'bg-teal-500/70' },
  { key: 'rerank', label: 'rerank', color: 'bg-amber-500/70' },
  { key: 'mmr', label: 'mmr', color: 'bg-orange-500/70' },
  { key: 'crag', label: 'crag', color: 'bg-red-500/70' },
  { key: 'generate', label: 'generate', color: 'bg-emerald-500/70' },
]

export default function LatencyBar({ latency }: { latency: LatencyMs }) {
  const segments = STAGES.map(s => ({
    ...s,
    ms: Math.max(0, Number(latency[s.key] || 0)),
  }))
  const total = segments.reduce((a, b) => a + b.ms, 0) || 1

  return (
    <div className="space-y-2">
      <div className="flex h-3 rounded-md overflow-hidden border border-line">
        {segments.map(seg => {
          const w = (seg.ms / total) * 100
          if (w < 0.5) return null
          return (
            <div
              key={seg.key as string}
              className={cn(seg.color, 'transition')}
              style={{ width: `${w}%` }}
              title={`${seg.label}: ${seg.ms.toFixed(1)} ms`}
            />
          )
        })}
      </div>
      <div className="flex flex-wrap gap-2 text-[10px] text-citi-blue">
        {segments
          .filter(s => s.ms > 0)
          .map(seg => (
            <span key={seg.key as string} className="inline-flex items-center gap-1.5">
              <span className={cn('w-2 h-2 rounded-sm', seg.color)} />
              <span>{seg.label}</span>
              <span className="text-ink">{seg.ms.toFixed(0)} ms</span>
            </span>
          ))}
      </div>
    </div>
  )
}
