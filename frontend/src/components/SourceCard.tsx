import { Copy, FileText, Hash } from 'lucide-react'
import { useState } from 'react'

import { cn } from '@/lib/cn'
import type { ChunkHit } from '@/types'

export default function SourceCard({ hit, highlighted }: { hit: ChunkHit; highlighted: boolean }) {
  const [open, setOpen] = useState(false)
  const [copied, setCopied] = useState(false)

  function copyId() {
    navigator.clipboard.writeText(String(hit.chunk_id)).catch(() => {})
    setCopied(true)
    window.setTimeout(() => setCopied(false), 1200)
  }

  return (
    <div
      id={`source-${hit.chunk_id}`}
      className={cn(
        'card-soft p-3 hover:border-accent/40 transition',
        highlighted && 'border-emerald-500/60',
      )}
    >
      {/* Rank + chunk_id header (chunk_id is now the headline) */}
      <div className="flex items-center justify-between gap-2 mb-2">
        <div className="flex items-center gap-1.5 min-w-0 flex-1">
          <span className="chip-accent text-[10px] shrink-0">#{hit.rank}</span>
          <button
            onClick={copyId}
            title="Copy chunk id"
            className={cn(
              'inline-flex items-center gap-1 px-1.5 py-0.5 rounded-md',
              'border text-[11px] font-mono tabular-nums transition shrink-0',
              copied
                ? 'border-emerald-500/60 bg-emerald-500/15 text-emerald-700'
                : 'border-accent/40 bg-accent/10 text-accent-dark hover:bg-accent/15',
            )}
          >
            <Hash className="w-3 h-3" />
            {hit.chunk_id}
            <Copy className="w-2.5 h-2.5 opacity-50" />
          </button>
        </div>
        <span className="text-[10px] text-citi-blue tabular-nums shrink-0">
          rrf {hit.score.toFixed(3)}
        </span>
      </div>

      {/* Document + page row */}
      <div className="flex items-center gap-1.5 text-[11px] text-ink mb-2 min-w-0">
        <FileText className="w-3 h-3 text-citi-blue shrink-0" />
        <span className="truncate font-medium">{hit.document_name || 'unknown'}</span>
        {hit.page_number != null && (
          <span className="chip text-[10px] shrink-0">page {hit.page_number}</span>
        )}
      </div>

      {/* Score breakdown */}
      {Object.keys(hit.score_breakdown).length > 0 && (
        <div className="flex flex-wrap items-center gap-1 mb-2">
          {Object.entries(hit.score_breakdown).map(([k, v]) => (
            <span key={k} className="chip text-[10px] text-citi-blue">
              {k.replace(/^dense_v\d+$/, 'dense')} {v.toFixed(3)}
            </span>
          ))}
        </div>
      )}

      {/* Body */}
      <button
        onClick={() => setOpen(o => !o)}
        className="text-left text-xs text-ink leading-relaxed w-full"
      >
        {open ? hit.content : hit.content.slice(0, 220) + (hit.content.length > 220 ? '…' : '')}
      </button>
      {hit.context && (
        <p className="mt-2 text-[10px] italic text-citi-blue border-l border-line pl-2">
          ctx: {hit.context.slice(0, 180)}{hit.context.length > 180 ? '…' : ''}
        </p>
      )}
    </div>
  )
}
