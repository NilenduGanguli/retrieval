import * as Switch from '@radix-ui/react-switch'
import { Info } from 'lucide-react'

import { cn } from '@/lib/cn'
import type { BackendConfig, Strategy } from '@/types'

type Props = {
  strategy: Strategy
  onChange: (s: Strategy) => void
  defaults?: BackendConfig['defaults']
}

const FEATURES: Array<{
  key: keyof Strategy
  label: string
  hint: string
}> = [
  { key: 'rewrite', label: 'Query rewrite', hint: '3 paraphrased queries, fused via RRF' },
  { key: 'hyde', label: 'HyDE', hint: 'Embed an LLM-drafted hypothetical answer' },
  { key: 'use_contextual', label: 'Contextual chunks', hint: 'Use Anthropic-style augmented embeddings' },
  { key: 'rerank', label: 'Listwise rerank', hint: 'LLM re-orders top-N (RankGPT/RankZephyr)' },
  { key: 'crag', label: 'CRAG self-grader', hint: 'Confidence-grade retrieval before answering' },
  { key: 'hybrid', label: 'Hybrid (dense+BM25)', hint: 'Always on by default; flip off for dense-only A/B' },
  { key: 'mmr', label: 'MMR diversify', hint: 'Penalise near-duplicate chunks' },
]

export default function StrategyToggles({ strategy, onChange, defaults }: Props) {
  const valOf = (k: keyof Strategy): boolean => {
    const v = strategy[k]
    if (typeof v === 'boolean') return v
    if (defaults) {
      const dk = (k === 'use_contextual' ? 'contextual' : k) as keyof BackendConfig['defaults']
      const dv = defaults[dk]
      if (typeof dv === 'boolean') return dv
    }
    return false
  }
  const update = (k: keyof Strategy, v: boolean) => {
    onChange({ ...strategy, [k]: v })
  }

  return (
    <div className="space-y-2">
      {FEATURES.map(f => (
        <div
          key={f.key}
          className="flex items-center justify-between gap-2 p-2 rounded-lg
                     bg-bg-soft/60 border border-line/60 hover:border-accent/40 transition"
        >
          <div className="flex-1 min-w-0">
            <div className="text-xs font-medium flex items-center gap-1">
              {f.label}
              <span className="group relative">
                <Info className="w-3 h-3 text-muted" />
                <span className="absolute left-1/2 -translate-x-1/2 -top-1 -translate-y-full
                                 hidden group-hover:block z-10 whitespace-nowrap text-[10px]
                                 bg-bg-card border border-line rounded px-2 py-1 text-zinc-300">
                  {f.hint}
                </span>
              </span>
            </div>
          </div>
          <Switch.Root
            checked={valOf(f.key)}
            onCheckedChange={v => update(f.key, v)}
            className={cn(
              'w-9 h-5 rounded-full relative transition',
              valOf(f.key) ? 'bg-accent' : 'bg-zinc-700',
            )}
          >
            <Switch.Thumb
              className={cn(
                'block w-4 h-4 bg-white rounded-full shadow transform transition',
                'translate-x-0.5 data-[state=checked]:translate-x-[18px]',
              )}
            />
          </Switch.Root>
        </div>
      ))}

      <div className="pt-2">
        <label className="text-xs text-muted flex items-center justify-between mb-1">
          <span>top-K final</span>
          <span className="text-zinc-300">{strategy.top_k ?? defaults?.top_k ?? 8}</span>
        </label>
        <input
          type="range"
          min={1}
          max={20}
          value={strategy.top_k ?? defaults?.top_k ?? 8}
          onChange={e => onChange({ ...strategy, top_k: parseInt(e.target.value, 10) })}
          className="w-full accent-[#7c5cff]"
        />
      </div>
    </div>
  )
}
