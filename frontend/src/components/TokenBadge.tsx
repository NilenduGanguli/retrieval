import { Coins } from 'lucide-react'

import { cn } from '@/lib/cn'

export type TokenLike = {
  prompt?: number | null
  completion?: number | null
  total?: number | null
} | null | undefined

function fmt(n: number | null | undefined): string {
  if (n == null || isNaN(Number(n))) return '0'
  const v = Number(n)
  if (v >= 1_000_000) return (v / 1_000_000).toFixed(2) + 'M'
  if (v >= 1_000) return (v / 1_000).toFixed(1) + 'k'
  return String(Math.round(v))
}

export default function TokenBadge({
  usage,
  size = 'sm',
  label = 'tokens',
  variant = 'soft',
  className,
}: {
  usage: TokenLike
  size?: 'sm' | 'md' | 'lg'
  label?: string
  variant?: 'soft' | 'accent' | 'plain'
  className?: string
}) {
  const total = Number(usage?.total ?? 0) || ((Number(usage?.prompt ?? 0) + Number(usage?.completion ?? 0)) || 0)
  const prompt = Number(usage?.prompt ?? 0)
  const completion = Number(usage?.completion ?? 0)
  const sizeCls =
    size === 'lg' ? 'text-sm px-2.5 py-1' :
    size === 'md' ? 'text-xs px-2 py-1' :
    'text-[11px] px-2 py-0.5'
  const variantCls =
    variant === 'accent' ? 'bg-accent/15 text-accent-dark border-accent/40' :
    variant === 'plain'  ? 'bg-transparent text-ink border-line' :
                           'bg-bg-soft/50 text-ink border-line'
  return (
    <span
      title={`${label}: ${fmt(total)}  (prompt ${fmt(prompt)} + completion ${fmt(completion)})`}
      className={cn(
        'inline-flex items-center gap-1.5 rounded-md border tabular-nums',
        sizeCls, variantCls, className,
      )}
    >
      <Coins className="w-3 h-3 opacity-80" />
      <span className="text-ink font-medium">{fmt(total)}</span>
      <span className="text-citi-blue text-[10px]">
        {fmt(prompt)} in / {fmt(completion)} out
      </span>
    </span>
  )
}
