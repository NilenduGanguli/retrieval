/**
 * Ingestion source selector.
 *
 * retrieval can ingest through either of two backends, chosen here by the user:
 *
 *   wega — POST /api/ingest/wega. The existing path: runs the local WEGA
 *          chunker in-process, or proxies to the ingest_remote service.
 *   des  — POST /api/ingest/des. document-enrichment-services: Azure Document
 *          Intelligence layout OCR + structure-aware chunking + gte-large
 *          embeddings, written straight into the same pgvector tables we read.
 *          retrieval only triggers it and follows its progress.
 *
 * The choice ONLY changes the POST target — both endpoints emit the identical
 * SSE contract (start / info / progress / done / error), so the Ingestion tab's
 * event handling, log, progress bar and chunk preview are shared verbatim.
 *
 * GET /api/ingest/sources advertises what the backend can actually do. Older
 * backends don't have that route at all, so a 404 (or any failure) degrades to
 * a WEGA-only list — today's behaviour, never a broken tab.
 */
import { RefreshCw } from 'lucide-react'
import { useCallback, useEffect, useMemo, useState } from 'react'

import { cn } from '@/lib/cn'

export type IngestSource = {
  id: string
  label: string
  /** false → the backend can't run this source right now (not configured, unreachable, …) */
  available: boolean
  /** what it points at when usable; WHY it isn't when it's disabled */
  detail: string | null
  /** advisory even when the source is usable — e.g. a pgvector schema mismatch */
  note: string | null
  noteTone: 'warn' | 'muted' | null
}

export const SOURCES_ENDPOINT = '/api/ingest/sources'
export const DEFAULT_SOURCE_ID = 'wega'

const KNOWN_LABELS: Record<string, string> = {
  wega: 'WEGA / Stellar',
  des: 'Document Enrichment Services',
}

const KNOWN_ENDPOINTS: Record<string, string> = {
  wega: '/api/ingest/wega',
  des: '/api/ingest/des',
}

/** The POST target for a source id. Unknown ids follow the same convention. */
export function ingestEndpoint(id: string): string {
  return KNOWN_ENDPOINTS[id] ?? `/api/ingest/${encodeURIComponent(id)}`
}

export function sourceLabel(sources: IngestSource[], id: string | null | undefined): string {
  if (!id) return KNOWN_LABELS[DEFAULT_SOURCE_ID]
  return sources.find(s => s.id === id)?.label ?? KNOWN_LABELS[id] ?? id
}

/** Fallback when /api/ingest/sources is missing or unreadable: wega only. */
export const FALLBACK_SOURCES: IngestSource[] = [
  {
    id: DEFAULT_SOURCE_ID,
    label: KNOWN_LABELS.wega,
    available: true,
    detail: null,
    note: null,
    noteTone: null,
  },
]

// ───────────────────────────────────────────────────────────────────────────
// Payload normalisation — tolerant on purpose. The endpoint may return an
// array, {sources: [...]} or {sources: {id: {...}}}, and may name the fields
// id/key, label/title, available/enabled, detail/reason/description.
// ───────────────────────────────────────────────────────────────────────────
function str(v: unknown): string | null {
  return typeof v === 'string' && v.trim() ? v.trim() : null
}

function bool(...vals: unknown[]): boolean | null {
  for (const v of vals) if (typeof v === 'boolean') return v
  return null
}

function normalizeOne(raw: unknown, fallbackId?: string): IngestSource | null {
  if (raw == null) return null
  if (typeof raw === 'string') {
    const id = str(raw)
    if (!id) return null
    return {
      id,
      label: KNOWN_LABELS[id] ?? id,
      available: true,
      detail: null,
      note: null,
      noteTone: null,
    }
  }
  if (typeof raw !== 'object') return null
  const o = raw as Record<string, unknown>
  const id = str(o.id) ?? str(o.key) ?? str(o.source) ?? str(o.name) ?? fallbackId ?? null
  if (!id) return null
  const label =
    str(o.label) ?? str(o.title) ?? str(o.display_name) ?? KNOWN_LABELS[id] ?? id
  const available = bool(o.available, o.enabled, o.ok, o.configured) ?? true

  // `detail` on the wire is "what it points at" (a URL, "local"/"remote");
  // the reason a source is down travels in `error`. Show the reason FIRST when
  // the option is disabled — a greyed-out control that only says
  // "http://localhost:8099" explains nothing.
  const target = str(o.detail) ?? str(o.description) ?? null
  const reason =
    str(o.error) ?? str(o.reason) ?? str(o.message) ?? (available ? null : str(o.status))
  let detail: string | null
  if (available) {
    detail = target
  } else {
    const enabled = bool(o.enabled, o.configured)
    const parts = [reason ?? (enabled === false ? 'not configured on this backend' : 'unreachable')]
    if (target) parts.push(target)
    detail = parts.join(' · ')
  }

  // DES writes its chunks straight into a pgvector schema. If that isn't the
  // schema retrieval queries, the ingest "succeeds" and nothing it produced is
  // ever searchable here — worth saying out loud, even while available=true.
  let note: string | null = null
  let noteTone: IngestSource['noteTone'] = null
  if (available && 'schema_match' in o) {
    const match = bool(o.schema_match)
    const vectorSchema = str(o.vector_schema)
    const expected = str(o.expected_schema)
    if (match === false) {
      note =
        `writes into schema ${vectorSchema ?? '(unknown)'}, but this service reads ` +
        `${expected ?? '(another schema)'} — chunks it ingests won't be retrievable here`
      noteTone = 'warn'
    } else if (match === null) {
      note = 'vector schema not reported — chunks may land in a schema this service does not read'
      noteTone = 'muted'
    }
  }

  return { id, label, available, detail, note, noteTone }
}

type NormalizedSources = { sources: IngestSource[]; defaultId: string | null }

export function normalizeSources(payload: unknown): NormalizedSources {
  let list: unknown[] = []
  let defaultId: string | null = null
  const seenKeys: string[] = []

  const fromRecord = (rec: Record<string, unknown>): void => {
    for (const [k, v] of Object.entries(rec)) {
      list.push(v)
      seenKeys.push(k)
    }
  }

  if (Array.isArray(payload)) {
    list = payload
  } else if (payload && typeof payload === 'object') {
    const o = payload as Record<string, unknown>
    defaultId = str(o.default) ?? str(o.default_source) ?? str(o.selected) ?? null
    const container = o.sources ?? o.ingest_sources ?? o.items
    if (Array.isArray(container)) {
      list = container
    } else if (container && typeof container === 'object') {
      fromRecord(container as Record<string, unknown>)
    } else {
      // Possibly the object itself is keyed by source id: {wega: {...}, des: {...}}
      const entries = Object.entries(o).filter(
        ([, v]) => v != null && typeof v === 'object' && !Array.isArray(v),
      )
      if (entries.length) {
        for (const [k, v] of entries) {
          list.push(v)
          seenKeys.push(k)
        }
      }
    }
  }

  const out: IngestSource[] = []
  const seenIds = new Set<string>()
  list.forEach((raw, i) => {
    const s = normalizeOne(raw, seenKeys[i])
    if (!s || seenIds.has(s.id)) return
    seenIds.add(s.id)
    out.push(s)
    const o = raw && typeof raw === 'object' ? (raw as Record<string, unknown>) : null
    if (!defaultId && o && o.default === true) defaultId = s.id
  })

  return { sources: out, defaultId }
}

// ───────────────────────────────────────────────────────────────────────────
// Hook
// ───────────────────────────────────────────────────────────────────────────
export type IngestSourcesState = {
  sources: IngestSource[]
  sourceId: string
  setSourceId: (id: string) => void
  selected: IngestSource | null
  loading: boolean
  /** true when the sources endpoint is absent/failed and we fell back to wega */
  degraded: boolean
  reload: () => void
}

function pickSource(
  sources: IngestSource[],
  current: string,
  defaultId: string | null,
): string {
  const usable = (id: string | null | undefined) =>
    id ? sources.find(s => s.id === id && s.available)?.id ?? null : null
  // Keep the user's choice, else wega (today's behaviour), else the backend's
  // stated default, else the first thing that actually works.
  return (
    usable(current) ??
    usable(DEFAULT_SOURCE_ID) ??
    usable(defaultId) ??
    sources.find(s => s.available)?.id ??
    sources[0]?.id ??
    DEFAULT_SOURCE_ID
  )
}

export function useIngestSources(): IngestSourcesState {
  const [sources, setSources] = useState<IngestSource[]>(FALLBACK_SOURCES)
  const [sourceId, setSourceId] = useState<string>(DEFAULT_SOURCE_ID)
  const [loading, setLoading] = useState(true)
  const [degraded, setDegraded] = useState(false)

  const load = useCallback(async () => {
    setLoading(true)
    try {
      // Deliberately raw fetch (not lib/api's json helper): a 404 here is an
      // expected outcome on an older backend, not an error worth throwing.
      const res = await fetch(SOURCES_ENDPOINT, { headers: { accept: 'application/json' } })
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
      const { sources: parsed, defaultId } = normalizeSources(await res.json())
      if (!parsed.length) throw new Error('empty source list')
      setSources(parsed)
      setDegraded(false)
      setSourceId(prev => pickSource(parsed, prev, defaultId))
    } catch {
      setSources(FALLBACK_SOURCES)
      setSourceId(DEFAULT_SOURCE_ID)
      setDegraded(true)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { void load() }, [load])

  const selected = useMemo(
    () => sources.find(s => s.id === sourceId) ?? null,
    [sources, sourceId],
  )

  return { sources, sourceId, setSourceId, selected, loading, degraded, reload: () => void load() }
}

// ───────────────────────────────────────────────────────────────────────────
// Control
// ───────────────────────────────────────────────────────────────────────────
export function IngestSourcePicker({
  state,
  className,
}: {
  state: IngestSourcesState
  className?: string
}) {
  const { sources, sourceId, setSourceId, loading, degraded, reload } = state
  const unavailable = sources.filter(s => !s.available)
  const noted = sources.filter(s => s.available && s.note)

  return (
    <div className={cn('space-y-1.5', className)}>
      <div className="flex items-center gap-1.5">
        <span className="text-[10px] uppercase tracking-wider text-citi-blue/80">
          Ingestion source
        </span>
        <button
          type="button"
          onClick={reload}
          disabled={loading}
          title="Re-check which ingestion backends are available"
          aria-label="Re-check ingestion sources"
          className="p-1 rounded hover:bg-bg-soft transition disabled:opacity-50"
        >
          <RefreshCw className={cn('w-3 h-3 text-citi-blue', loading && 'animate-spin')} />
        </button>
      </div>

      <div
        role="radiogroup"
        aria-label="Ingestion source"
        className="inline-flex flex-wrap items-center gap-1 rounded-lg border border-line bg-bg-soft p-1"
      >
        {sources.map(s => {
          const active = s.id === sourceId
          const disabled = !s.available
          return (
            <button
              key={s.id}
              type="button"
              role="radio"
              aria-checked={active}
              aria-disabled={disabled}
              disabled={disabled}
              onClick={() => setSourceId(s.id)}
              title={
                disabled
                  ? `${s.label} is unavailable${s.detail ? ` — ${s.detail}` : ''}`
                  : `${s.label}${s.detail ? ` — ${s.detail}` : ''}${s.note ? ` · ${s.note}` : ''}`
              }
              className={cn(
                'text-[11px] px-2.5 py-1 rounded-md border transition',
                active && !disabled
                  ? 'border-accent bg-accent text-white shadow-glow'
                  : 'border-transparent text-citi-blue hover:border-accent/60 hover:text-accent-dark',
                disabled && 'opacity-50 cursor-not-allowed hover:border-transparent hover:text-citi-blue',
              )}
            >
              {s.label}
            </button>
          )
        })}
      </div>

      {/* A greyed-out control with no reason is frustrating — always say why. */}
      {unavailable.map(s => (
        <div key={s.id} className="text-[11px] leading-snug">
          <span className="text-amber-700">{s.label} unavailable</span>
          {s.detail && <span className="text-citi-blue"> — {s.detail}</span>}
        </div>
      ))}

      {/* Usable, but with a caveat (e.g. it writes into a schema we don't read). */}
      {noted.map(s => (
        <div key={s.id} className="text-[11px] leading-snug">
          <span className={s.noteTone === 'warn' ? 'text-amber-700' : 'text-citi-blue'}>
            {s.label}
          </span>
          <span className={s.noteTone === 'warn' ? 'text-amber-700' : 'text-citi-blue'}>
            {' '}— {s.note}
          </span>
        </div>
      ))}

      {degraded && (
        <div className="text-[11px] text-citi-blue leading-snug">
          This backend doesn&rsquo;t advertise <code className="text-accent-dark">/api/ingest/sources</code> —
          {' '}using WEGA / Stellar.
        </div>
      )}
    </div>
  )
}
