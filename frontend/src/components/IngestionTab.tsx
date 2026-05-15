import { Coins, Database, FileUp, Layers3, RefreshCw, Sparkles, Trash2, Wand2 } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'

import { api, postSse, uploadSse } from '@/lib/api'
import { cn } from '@/lib/cn'
import type { DocumentSummary, HealthInfo } from '@/types'

import TokenBadge from './TokenBadge'

type Props = { health: HealthInfo | null; onChange: () => void }

type LogLine = { type: 'log' | 'info' | 'context' | 'done' | 'error' | 'start'; text: string; at: number; progress?: number }

export default function IngestionTab({ health, onChange }: Props) {
  const [docs, setDocs] = useState<DocumentSummary[] | null>(null)
  const [refreshing, setRefreshing] = useState(false)
  const [uploadLog, setUploadLog] = useState<LogLine[]>([])
  const [uploadProgress, setUploadProgress] = useState<number | null>(null)
  const [uploadStage, setUploadStage] = useState<string>('idle')
  const [uploadCounts, setUploadCounts] = useState<{ processed: number; total: number; failed: number } | null>(null)
  const [queue, setQueue] = useState<File[]>([])
  const [currentFile, setCurrentFile] = useState<string | null>(null)
  const [pendingDelete, setPendingDelete] = useState<string | null>(null)
  const [uploadTokens, setUploadTokens] = useState<{ prompt: number; completion: number; total: number } | null>(null)
  const [ctxTokens, setCtxTokens] = useState<{ prompt: number; completion: number; total: number } | null>(null)
  const [ctxLog, setCtxLog] = useState<LogLine[]>([])
  const [ctxProgress, setCtxProgress] = useState<number | null>(null)
  const [ctxDocument, setCtxDocument] = useState<string>('')
  const [ctxLimit, setCtxLimit] = useState(100)
  const fileRef = useRef<HTMLInputElement>(null)
  const abortRef = useRef<(() => void) | null>(null)
  const ctxPanelRef = useRef<HTMLDivElement>(null)
  const [ctxHighlight, setCtxHighlight] = useState(false)

  async function loadDocs() {
    setRefreshing(true)
    try {
      const d = await api.documents()
      setDocs(d)
    } finally {
      setRefreshing(false)
    }
  }
  useEffect(() => { loadDocs() }, [])

  // Top-bar contextualised badge dispatches `rag:focus-contextual` —
  // scroll the panel into view + flash it briefly.
  useEffect(() => {
    const handler = () => {
      requestAnimationFrame(() => {
        ctxPanelRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' })
        setCtxHighlight(true)
        window.setTimeout(() => setCtxHighlight(false), 1500)
      })
    }
    window.addEventListener('rag:focus-contextual', handler)
    return () => window.removeEventListener('rag:focus-contextual', handler)
  }, [])

  function uploadPdf(file: File, onComplete?: () => void) {
    setUploadProgress(0)
    setUploadStage('uploading')
    setUploadCounts(null)
    setUploadTokens(null)
    setCurrentFile(file.name)
    const form = new FormData()
    form.append('file', file)
    abortRef.current?.()
    abortRef.current = uploadSse(
      '/api/ingest/wega',
      form,
      (evt, raw) => {
        try {
          const data = JSON.parse(raw)
          if (evt === 'start') {
            const mode = data.mode || 'wega-stellar'
            setUploadStage('reading')
            setUploadLog(l => [...l, { type: 'info', text: `▶ ${mode} · ${data.filename || data.file}`, at: Date.now() }])
          }
          else if (evt === 'log') {
            const line = data.line || raw
            if (line.startsWith('reading')) setUploadStage('reading')
            else if (line.startsWith('extracted')) setUploadStage('extracted')
            else if (line.startsWith('created')) setUploadStage('chunking')
            setUploadLog(l => [...l, { type: 'log', text: line, at: Date.now() }])
          }
          else if (evt === 'progress') {
            setUploadStage('embedding')
            setUploadProgress(data.progress ?? null)
            setUploadCounts({
              processed: data.processed ?? 0,
              total: data.total ?? 0,
              failed: data.failed ?? 0,
            })
            if (data.tokens) setUploadTokens(data.tokens)
            setUploadLog(l => [
              ...l.slice(-300),
              {
                type: 'context',
                text: `· ${data.message} (${((data.progress ?? 0) * 100).toFixed(0)}%)`,
                at: Date.now(),
                progress: data.progress,
              },
            ])
          }
          else if (evt === 'done') {
            const summary = data.processed != null
              ? `done · ${data.processed}/${data.total} chunks · doc: ${data.document}`
              : `finished (rc=${data.returncode}) — ${data.file}`
            setUploadStage('done')
            setUploadLog(l => [...l, { type: 'done', text: summary, at: Date.now() }])
            setUploadProgress(1)
            if (data.processed != null) {
              setUploadCounts({ processed: data.processed, total: data.total, failed: data.failed ?? 0 })
            }
            if (data.tokens) setUploadTokens(data.tokens)
            loadDocs()
            onChange()
            onComplete?.()
          }
          else if (evt === 'error') {
            setUploadStage('error')
            setUploadLog(l => [...l, { type: 'error', text: data.message || raw, at: Date.now() }])
            onComplete?.()
          }
          else setUploadLog(l => [...l, { type: 'info', text: raw, at: Date.now() }])
        } catch {
          setUploadLog(l => [...l, { type: 'log', text: raw, at: Date.now() }])
        }
      },
      err => {
        setUploadLog(l => [...l, { type: 'error', text: String(err), at: Date.now() }])
        onComplete?.()
      },
    )
  }

  function enqueueFiles(files: FileList | File[]) {
    const incoming = Array.from(files).filter(f => f.size > 0)
    if (!incoming.length) return
    setUploadLog([])
    setQueue(prev => [...prev, ...incoming])
  }

  // Drive the queue: when nothing is in-flight and queue has items, kick off the next.
  useEffect(() => {
    if (currentFile != null) return
    if (queue.length === 0) return
    const [next, ...rest] = queue
    setQueue(rest)
    setUploadLog(l => [...l, { type: 'info', text: `─── ${next.name} (${(next.size/1024).toFixed(1)} KB) ───`, at: Date.now() }])
    uploadPdf(next, () => setCurrentFile(null))
  }, [queue, currentFile])

  // `ident` is either a document_id (UUID) — preferred — or a legacy document_name.
  async function handleDelete(ident: string) {
    setPendingDelete(ident)
    try {
      const r = await api.deleteDocument(ident)
      const label = r.document_name ?? ident
      setUploadLog(l => [...l, {
        type: 'done',
        text: `🗑 soft-deleted "${label}" · ${r.soft_deleted_chunks} chunks${r.s3_uri_removed ? ` · S3: ${r.s3_uri_removed}` : ''}`,
        at: Date.now(),
      }])
      await loadDocs()
      onChange()
    } catch (e) {
      setUploadLog(l => [...l, { type: 'error', text: `delete failed: ${e}`, at: Date.now() }])
    } finally {
      setPendingDelete(null)
    }
  }

  function runContextual() {
    setCtxLog([])
    setCtxProgress(0)
    setCtxTokens(null)
    abortRef.current?.()
    abortRef.current = postSse(
      `/api/ingest/contextual?document_name=${encodeURIComponent(ctxDocument)}&limit=${ctxLimit}&concurrency=4`,
      {},
      (evt, raw) => {
        try {
          const data = JSON.parse(raw)
          if (evt === 'context') {
            setCtxProgress(data.progress)
            if (data.tokens) setCtxTokens(data.tokens)
            const info = data.payload || {}
            setCtxLog(l => [
              ...l.slice(-200),
              {
                type: 'context',
                text: `[${info.document}] chunk #${info.chunk_id}${info.page != null ? ` p.${info.page}` : ''}`,
                at: Date.now(),
                progress: data.progress,
              },
            ])
          } else if (evt === 'done') {
            const stats = data.stats || data
            setCtxProgress(1)
            if (data.tokens) setCtxTokens(data.tokens)
            setCtxLog(l => [...l, { type: 'done', text: `done · processed=${stats.processed} skipped=${stats.skipped} failed=${stats.failed}`, at: Date.now() }])
            loadDocs()
            onChange()
          } else if (evt === 'error') {
            setCtxLog(l => [...l, { type: 'error', text: data.message || raw, at: Date.now() }])
          } else {
            setCtxLog(l => [...l, { type: 'info', text: raw, at: Date.now() }])
          }
        } catch {
          setCtxLog(l => [...l, { type: 'info', text: raw, at: Date.now() }])
        }
      },
      err => setCtxLog(l => [...l, { type: 'error', text: String(err), at: Date.now() }]),
    )
  }

  return (
    <div className="grid grid-cols-12 gap-6">
      <section className="col-span-12 lg:col-span-7 space-y-4">
        {/* Stats bar */}
        <div className="grid grid-cols-4 gap-3">
          <Stat label="Documents" value={health?.documents ?? '—'} />
          <Stat label="Chunks" value={(health?.chunks ?? 0).toLocaleString()} />
          <button
            type="button"
            onClick={() => window.dispatchEvent(new CustomEvent('rag:focus-contextual'))}
            title="Click to jump to the contextualisation panel"
            className="text-left"
          >
            <Stat
              label="Contextualised ↓"
              value={
                health
                  ? `${health.contextual_chunks.toLocaleString()} (${
                      health.chunks ? ((health.contextual_chunks / health.chunks) * 100).toFixed(0) : 0
                    }%)`
                  : '—'
              }
            />
          </button>
          <Stat label="Embedding dim" value={health?.embedding_dim ?? '—'} />
        </div>

        {/* WEGA ingest */}
        <div className="card p-5">
          <div className="flex items-center gap-2 mb-3">
            <FileUp className="w-4 h-4 text-accent" />
            <h2 className="font-medium text-sm">Ingest PDF (WEGA chunker)</h2>
          </div>
          <p className="text-xs text-citi-blue mb-3">
            Uploads the PDF to the server and runs your existing <code className="text-accent-dark">ingest.py</code>
            {' '}in-process. Stream of WEGA chunking → embedding → pgvector upsert.
          </p>
          <div className="flex items-center gap-2">
            <input
              ref={fileRef}
              type="file"
              accept="application/pdf"
              multiple
              className="block text-xs text-ink file:mr-3 file:py-1.5 file:px-3
                         file:rounded-md file:border file:border-line
                         file:bg-bg-soft file:text-ink hover:file:bg-bg-card"
              onChange={e => {
                if (e.target.files && e.target.files.length) {
                  enqueueFiles(e.target.files)
                  e.target.value = ''
                }
              }}
            />
            {uploadProgress != null && (
              <div className="flex-1 h-2 rounded-full bg-bg-soft overflow-hidden">
                <div className="h-full bg-accent transition-all" style={{ width: `${(uploadProgress * 100).toFixed(0)}%` }} />
              </div>
            )}
            {uploadProgress != null && (
              <span className="text-xs text-citi-blue tabular-nums">{((uploadProgress ?? 0) * 100).toFixed(0)}%</span>
            )}
          </div>
          {(currentFile || queue.length > 0) && (
            <div className="flex flex-wrap items-center gap-1.5 pt-2">
              {currentFile && (
                <span className="chip-accent text-[11px]">▶ {currentFile}</span>
              )}
              {queue.map((f, i) => (
                <span key={i} className="chip text-[11px]">⏳ {f.name}</span>
              ))}
            </div>
          )}
          {uploadStage !== 'idle' && (
            <IngestStages
              stage={uploadStage}
              counts={uploadCounts}
              className="mt-3"
            />
          )}
          {uploadTokens && (uploadTokens.total ?? 0) > 0 && (
            <div className="mt-3 card-soft p-2 flex items-center justify-between gap-3 text-xs">
              <div className="flex items-center gap-2 text-citi-blue">
                <Coins className="w-3.5 h-3.5 text-accent" />
                <span>Tokens consumed by this ingestion</span>
              </div>
              <TokenBadge usage={uploadTokens} variant="accent" size="md" />
            </div>
          )}
          {uploadLog.length > 0 && (
            <LogPanel lines={uploadLog} />
          )}
        </div>

        {/* Contextual gen */}
        <div
          ref={ctxPanelRef}
          className={cn(
            'card p-5 transition',
            ctxHighlight && 'ring-2 ring-accent shadow-glow',
          )}
        >
          <div className="flex items-center gap-2 mb-3">
            <Wand2 className="w-4 h-4 text-accent" />
            <h2 className="font-medium text-sm">
              Contextualise chunks{' '}
              <span className="text-citi-blue/70 font-normal">
                (Anthropic Contextual Retrieval)
              </span>
            </h2>
            <span className="chip-accent ml-2">+49% recall lift</span>
            {health && (
              <span className="chip ml-auto tabular-nums">
                {health.contextual_chunks}/{health.chunks} done
              </span>
            )}
          </div>
          <p className="text-xs text-citi-blue leading-relaxed mb-3">
            For each chunk, the LLM writes a ~50-100 token prefix that situates it inside the document, then we embed
            (prefix + chunk) and store the contextual embedding. This is the Anthropic Sept-2024 recipe.
          </p>
          <div className="grid grid-cols-3 gap-2 text-xs mb-3">
            <label className="col-span-2">
              <span className="block text-citi-blue mb-1">Document (blank = all docs)</span>
              <select
                value={ctxDocument}
                onChange={e => setCtxDocument(e.target.value)}
                className="w-full bg-bg-soft border border-line rounded-md px-2 py-1.5"
              >
                <option value="">— all documents —</option>
                {(docs || []).map(d => (
                  <option key={d.document_name} value={d.document_name}>
                    {d.document_name} ({d.chunk_count})
                  </option>
                ))}
              </select>
            </label>
            <label>
              <span className="block text-citi-blue mb-1">Batch size</span>
              <input
                type="number"
                value={ctxLimit}
                onChange={e => setCtxLimit(parseInt(e.target.value || '100', 10))}
                className="w-full bg-bg-soft border border-line rounded-md px-2 py-1.5"
              />
            </label>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={runContextual}
              className="inline-flex items-center gap-2 px-3 py-1.5 rounded-md text-sm
                         bg-accent text-white hover:bg-accent-soft transition"
            >
              <Sparkles className="w-4 h-4" />
              Generate context
            </button>
            {ctxProgress != null && (
              <div className="flex-1 h-2 rounded-full bg-bg-soft overflow-hidden">
                <div className="h-full bg-accent transition-all" style={{ width: `${(ctxProgress * 100).toFixed(0)}%` }} />
              </div>
            )}
            {ctxProgress != null && (
              <span className="text-xs text-citi-blue w-12 text-right">{(ctxProgress * 100).toFixed(0)}%</span>
            )}
          </div>
          {ctxTokens && (ctxTokens.total ?? 0) > 0 && (
            <div className="mt-3 card-soft p-2 flex items-center justify-between gap-3 text-xs">
              <div className="flex items-center gap-2 text-citi-blue">
                <Coins className="w-3.5 h-3.5 text-accent" />
                <span>Tokens consumed by context-prefix generation</span>
              </div>
              <TokenBadge usage={ctxTokens} variant="accent" size="md" />
            </div>
          )}
          {ctxLog.length > 0 && <LogPanel lines={ctxLog} />}
        </div>
      </section>

      {/* Document explorer */}
      <aside className="col-span-12 lg:col-span-5 space-y-3">
        <div className="flex items-center gap-2 px-1">
          <Database className="w-4 h-4 text-accent" />
          <h2 className="font-medium text-sm">Document explorer</h2>
          <button onClick={loadDocs} className="ml-auto p-1.5 rounded hover:bg-bg-soft transition">
            <RefreshCw className={cn('w-3.5 h-3.5 text-citi-blue', refreshing && 'animate-spin')} />
          </button>
        </div>
        <div className="space-y-2 max-h-[calc(100vh-180px)] overflow-y-auto pr-1">
          {(docs || []).map(d => {
            const ident = d.document_id ?? d.document_name
            return (
              <DocCard
                key={ident}
                doc={d}
                busy={pendingDelete === ident}
                onDelete={() => handleDelete(ident)}
              />
            )
          })}
          {(docs || []).length === 0 && (
            <div className="text-xs text-citi-blue px-2 py-4">No documents indexed yet.</div>
          )}
        </div>
      </aside>
    </div>
  )
}

function IngestStages({
  stage,
  counts,
  className,
}: {
  stage: string
  counts: { processed: number; total: number; failed: number } | null
  className?: string
}) {
  const order = ['uploading', 'reading', 'extracted', 'chunking', 'embedding', 'done']
  const idx = Math.max(0, order.indexOf(stage))
  const labels: Record<string, string> = {
    uploading: 'upload',
    reading: 'read pdf',
    extracted: 'extract text',
    chunking: 'chunk',
    embedding: 'embed → upsert',
    done: 'done',
    error: 'error',
  }
  return (
    <div className={cn('card-soft p-2 flex flex-wrap gap-1.5 text-[11px]', className)}>
      {order.map((s, i) => {
        const status =
          stage === 'error' ? (s === 'embedding' ? 'error' : i < idx ? 'done' : 'idle') :
          i < idx ? 'done' :
          i === idx ? 'running' :
          'idle'
        return (
          <span
            key={s}
            className={cn(
              'inline-flex items-center gap-1 px-2 py-0.5 rounded-md border transition',
              status === 'done' && 'border-emerald-500/60 bg-emerald-500/15 text-emerald-700',
              status === 'running' && 'border-accent/60 bg-accent/15 text-accent-dark shadow-glow',
              status === 'idle' && 'border-line bg-bg-soft/40 text-citi-blue',
              status === 'error' && 'border-red-500/40 bg-red-500/10 text-red-700',
            )}
          >
            {labels[s]}
            {s === 'embedding' && counts && counts.total > 0 && (
              <span className="text-[10px] tabular-nums opacity-80">
                {counts.processed}/{counts.total}
              </span>
            )}
          </span>
        )
      })}
    </div>
  )
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="card p-3">
      <div className="text-[10px] uppercase text-citi-blue">{label}</div>
      <div className="text-lg font-semibold mt-0.5">{value}</div>
    </div>
  )
}

function DocCard({
  doc, busy, onDelete,
}: {
  doc: DocumentSummary
  busy: boolean
  onDelete: () => void
}) {
  const cov = doc.contextual_coverage ?? 0
  return (
    <div className="card-soft p-3 group">
      <div className="flex items-start justify-between gap-2 mb-1">
        <div className="text-sm font-medium text-ink truncate min-w-0 flex-1">{doc.document_name}</div>
        <span className="chip text-[10px] shrink-0">
          <Layers3 className="w-3 h-3" />
          {doc.chunk_count}
        </span>
        <button
          onClick={() => {
            if (window.confirm(`Soft-delete "${doc.document_name}"? Chunks stay in DB marked deleted, S3 object is removed.`)) {
              onDelete()
            }
          }}
          disabled={busy}
          title="Soft delete (sets deleted_at, removes from S3)"
          className="opacity-0 group-hover:opacity-100 transition text-citi-blue hover:text-red-700 disabled:opacity-30 shrink-0"
        >
          <Trash2 className="w-3.5 h-3.5" />
        </button>
      </div>
      <div className="flex items-center gap-2 text-[11px] text-citi-blue mb-2 flex-wrap">
        {doc.first_page != null && doc.last_page != null && (
          <span>pp. {doc.first_page}-{doc.last_page}</span>
        )}
        {doc.total_tokens != null && <span>{(doc.total_tokens / 1000).toFixed(1)}k tok</span>}
        {doc.latest_job_id && <span className="truncate">job: {doc.latest_job_id.slice(0, 8)}</span>}
        {doc.document_id && (
          <span
            className="font-mono text-[10px] opacity-80 truncate"
            title={`document_id: ${doc.document_id}`}
          >
            id: {doc.document_id.slice(0, 8)}
          </span>
        )}
        {doc.sha256 && (
          <span
            className="font-mono text-[10px] opacity-80 truncate"
            title={`sha256: ${doc.sha256}`}
          >
            sha: {doc.sha256.slice(0, 8)}
          </span>
        )}
      </div>
      <div className="flex items-center gap-2">
        <div className="flex-1 h-1.5 rounded-full bg-bg-soft overflow-hidden">
          <div className="h-full bg-emerald-500/70 transition-all" style={{ width: `${(cov * 100).toFixed(0)}%` }} />
        </div>
        <span className="text-[10px] text-citi-blue w-10 text-right">
          {(cov * 100).toFixed(0)}%
        </span>
      </div>
    </div>
  )
}

function LogPanel({ lines }: { lines: LogLine[] }) {
  return (
    <div className="mt-3 bg-bg-soft border border-line rounded-md p-2 font-mono text-[11px]
                    max-h-56 overflow-y-auto leading-snug">
      {lines.map((l, i) => (
        <div
          key={i}
          className={cn(
            'whitespace-pre-wrap',
            l.type === 'error' && 'text-red-700',
            l.type === 'done' && 'text-emerald-700',
            l.type === 'context' && 'text-accent-dark',
            l.type === 'info' && 'text-citi-blue',
          )}
        >
          {l.text}
        </div>
      ))}
    </div>
  )
}
