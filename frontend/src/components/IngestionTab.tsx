import {
  Braces, Coins, Database, Download, Eye, FileText, FileUp, History, Layers,
  Layers3, Loader2, RefreshCw, Rows3, Sparkles, Table as TableIcon, Trash2,
  Wand2, X,
} from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'

import { api, postSse, uploadSse } from '@/lib/api'
import { cn } from '@/lib/cn'
import type { DocumentSummary, HealthInfo } from '@/types'

import {
  IngestSourcePicker,
  ingestEndpoint,
  sourceLabel,
  useIngestSources,
} from './IngestSourceSelector'
import TokenBadge from './TokenBadge'

type Props = { health: HealthInfo | null; onChange: () => void }

type LogLine = { type: 'log' | 'info' | 'context' | 'done' | 'error' | 'start'; text: string; at: number; progress?: number }

// One row per upload — captures the raw SSE event log so users can inspect
// what the backend (WEGA / Vertex / KYC) returned, plus the chunks that
// were just inserted. Kept in component state, never persisted.
type SessionEvent = { event: string; data: Record<string, any>; at: number }
type SessionChunk = {
  id: number
  content: string
  page_number: number | null
  token_count: number | null
  chunk_type: string | null
  context_text: string | null
}
type UploadSession = {
  key: string
  documentName: string
  documentId: string | null
  sha256: string | null
  // Which backend performed this ingestion. `source`/`sourceLabel` are what the
  // user picked; `mode` is what the backend reported on the `start` event
  // (wega-stellar / remote / local-vertex / des …) and wins when present.
  source: string
  sourceLabel: string
  mode: string | null
  startedAt: number
  finishedAt: number | null
  events: SessionEvent[]
  summary: Record<string, any> | null
  // Raw WEGA SDK chunker output (from ingest_core.py:chunker_result). Only
  // present on the remote-WEGA path; absent for local Vertex and the
  // subprocess path, in which case the UI falls back to the events stream.
  chunkerResult: unknown | null
  error: string | null
  chunks: SessionChunk[] | null
  chunksLoading: boolean
  chunksError: string | null
}

const MAX_SESSIONS = 5
const CHUNKS_FETCH_LIMIT = 500

function makeSessionKey(): string {
  return `s_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`
}

function triggerDownload(filename: string, content: string, mime: string): void {
  const blob = new Blob([content], { type: mime })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  setTimeout(() => URL.revokeObjectURL(url), 0)
}

function csvCell(v: unknown): string {
  if (v == null) return ''
  const s = String(v)
  if (/[",\n\r]/.test(s)) return `"${s.replace(/"/g, '""')}"`
  return s
}

function chunksToCsv(chunks: SessionChunk[]): string {
  const cols: (keyof SessionChunk)[] = [
    'id', 'page_number', 'token_count', 'chunk_type', 'context_text', 'content',
  ]
  const lines = [cols.join(',')]
  for (const c of chunks) lines.push(cols.map(k => csvCell(c[k])).join(','))
  return lines.join('\n')
}

function sanitizeFilename(name: string): string {
  return name.replace(/[^a-zA-Z0-9._-]+/g, '_').slice(0, 80) || 'upload'
}

// Compact tag for the source that produced a session — keeps a mixed
// wega/des history readable in the pills and the panel header.
function modeTag(s: { mode: string | null; source: string }): string {
  const m = (s.mode || s.source || '').toLowerCase()
  if (!m) return 'wega'
  if (m === 'wega-stellar') return 'wega'
  if (m === 'local-vertex') return 'vertex'
  return m
}

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
  // Which backend performs the ingestion. Only changes the POST target — the
  // SSE contract (and therefore everything below) is identical for both.
  const ingestSources = useIngestSources()
  const sourceIdRef = useRef(ingestSources.sourceId)
  useEffect(() => { sourceIdRef.current = ingestSources.sourceId }, [ingestSources.sourceId])
  const ctxPanelRef = useRef<HTMLDivElement>(null)
  const [ctxHighlight, setCtxHighlight] = useState(false)

  // Per-upload session capture — last MAX_SESSIONS uploads, newest first.
  const [sessions, setSessions] = useState<UploadSession[]>([])
  const [activeSessionKey, setActiveSessionKey] = useState<string | null>(null)
  const currentSessionRef = useRef<string | null>(null)

  function startSession(documentName: string, source: string, label: string): string {
    const key = makeSessionKey()
    currentSessionRef.current = key
    setActiveSessionKey(key)
    setSessions(prev => [
      {
        key,
        documentName,
        documentId: null,
        sha256: null,
        source,
        sourceLabel: label,
        mode: null,
        startedAt: Date.now(),
        finishedAt: null,
        events: [],
        summary: null,
        chunkerResult: null,
        error: null,
        chunks: null,
        chunksLoading: false,
        chunksError: null,
      },
      ...prev,
    ].slice(0, MAX_SESSIONS))
    return key
  }

  function updateSession(key: string, patch: (s: UploadSession) => UploadSession): void {
    setSessions(prev => prev.map(s => (s.key === key ? patch(s) : s)))
  }

  function recordEvent(key: string | null, event: string, data: Record<string, any>): void {
    if (!key) return
    updateSession(key, s => ({
      ...s,
      events: [...s.events, { event, data, at: Date.now() }],
    }))
  }

  // Fetch the chunks that were just inserted for `documentId`. Best-effort —
  // failures surface inline, don't abort the upload UX.
  async function fetchSessionChunks(key: string, documentId: string): Promise<void> {
    updateSession(key, s => ({ ...s, chunksLoading: true, chunksError: null }))
    try {
      const r = await api.documentChunks(documentId, CHUNKS_FETCH_LIMIT)
      updateSession(key, s => ({
        ...s,
        chunks: (r.chunks || []) as SessionChunk[],
        chunksLoading: false,
      }))
    } catch (e) {
      updateSession(key, s => ({
        ...s,
        chunksLoading: false,
        chunksError: String(e),
      }))
    }
  }

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

  function uploadPdf(file: File, source: string, onComplete?: () => void) {
    setUploadProgress(0)
    setUploadStage('uploading')
    setUploadCounts(null)
    setUploadTokens(null)
    setCurrentFile(file.name)
    const label = sourceLabel(ingestSources.sources, source)
    const sessionKey = startSession(file.name, source, label)
    const form = new FormData()
    form.append('file', file)
    abortRef.current?.()
    abortRef.current = uploadSse(
      // wega → /api/ingest/wega, des → /api/ingest/des. Same event contract.
      ingestEndpoint(source),
      form,
      (evt, raw) => {
        try {
          const data = JSON.parse(raw)
          recordEvent(sessionKey, evt, data)
          if (evt === 'start') {
            // `mode` is the backend's own name for the path it took
            // (wega-stellar / remote / local-vertex / des …). Fall back to the
            // source the user picked so a session is never unlabelled.
            const mode = data.mode || source
            setUploadStage('reading')
            updateSession(sessionKey, s => ({ ...s, mode }))
            setUploadLog(l => [...l, { type: 'info', text: `${mode} · ${data.filename || data.file}`, at: Date.now() }])
          }
          else if (
            evt === 'log' ||
            (evt === 'info' && (typeof data.line === 'string' || typeof data.message === 'string'))
          ) {
            // One log line per event. The in-process WEGA path names the event
            // `log`; the remote/DES paths name it `info` — same payload shape,
            // same stage inference, one handler.
            const line: string =
              (typeof data.line === 'string' ? data.line : null) ??
              (typeof data.message === 'string' ? data.message : null) ??
              raw
            if (line.startsWith('reading')) setUploadStage('reading')
            else if (line.startsWith('extracted')) setUploadStage('extracted')
            else if (line.startsWith('created')) setUploadStage('chunking')
            setUploadLog(l => [...l, { type: 'log', text: line, at: Date.now() }])
          }
          else if (evt === 'chunker_result') {
            // Raw WEGA SDK chunker output — store on the session so the
            // Response JSON tab can render this instead of the SSE log.
            updateSession(sessionKey, s => ({
              ...s,
              chunkerResult: data.chunker_result ?? data,
            }))
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
            updateSession(sessionKey, s => ({
              ...s,
              finishedAt: Date.now(),
              summary: data,
              documentId: data.document_id ?? s.documentId,
              sha256: data.sha256 ?? s.sha256,
              documentName: data.document || data.document_name || s.documentName,
            }))
            const docId = data.document_id ?? null
            if (docId) {
              // Fire-and-forget — UI shows a loading shimmer on the table tab.
              void fetchSessionChunks(sessionKey, docId)
            }
            loadDocs()
            onChange()
            onComplete?.()
          }
          else if (evt === 'error') {
            setUploadStage('error')
            setUploadLog(l => [...l, { type: 'error', text: data.message || raw, at: Date.now() }])
            updateSession(sessionKey, s => ({
              ...s,
              finishedAt: Date.now(),
              error: data.message || raw,
            }))
            onComplete?.()
          }
          else setUploadLog(l => [...l, { type: 'info', text: raw, at: Date.now() }])
        } catch {
          recordEvent(sessionKey, evt, { _raw: raw })
          setUploadLog(l => [...l, { type: 'log', text: raw, at: Date.now() }])
        }
      },
      err => {
        setUploadLog(l => [...l, { type: 'error', text: String(err), at: Date.now() }])
        updateSession(sessionKey, s => ({
          ...s,
          finishedAt: Date.now(),
          error: String(err),
        }))
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
    // Each file uses whichever source is selected when its turn comes up.
    const source = sourceIdRef.current
    setUploadLog(l => [...l, { type: 'info', text: `─── ${next.name} (${(next.size/1024).toFixed(1)} KB) → ${source} ───`, at: Date.now() }])
    uploadPdf(next, source, () => setCurrentFile(null))
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

        {/* Ingest — WEGA / Stellar or Document Enrichment Services */}
        <div className="card p-5">
          <div className="flex items-center gap-2 mb-3">
            <FileUp className="w-4 h-4 text-accent" />
            <h2 className="font-medium text-sm">Ingest PDF</h2>
          </div>
          <p className="text-xs text-citi-blue mb-3">
            {ingestSources.sourceId === 'des' ? (
              <>
                Hands the PDF to <code className="text-accent-dark">document-enrichment-services</code>, which runs
                Azure DI layout OCR → structure-aware chunking → gte-large embeddings and writes the chunks straight
                into the same pgvector tables. Its progress is streamed back here.
              </>
            ) : (
              <>
                Uploads the PDF to the server and runs your existing <code className="text-accent-dark">ingest.py</code>
                {' '}in-process. Stream of WEGA chunking → embedding → pgvector upsert.
              </>
            )}
          </p>
          <IngestSourcePicker state={ingestSources} className="mb-3" />
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
                <span className="chip-accent text-[11px] inline-flex items-center gap-1">
                  <Loader2 className="w-3 h-3 animate-spin" />
                  {currentFile}
                </span>
              )}
              {queue.map((f, i) => (
                <span key={i} className="chip text-[11px] inline-flex items-center gap-1">
                  <Layers className="w-3 h-3" />
                  {f.name}
                </span>
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

        {/* Upload session inspector — JSON response + inserted chunks table */}
        {sessions.length > 0 && (
          <UploadSessionPanel
            sessions={sessions}
            activeKey={activeSessionKey}
            onSelect={setActiveSessionKey}
            onRefetchChunks={fetchSessionChunks}
          />
        )}

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
      {/* Contextual-retrieval coverage — NOT ingestion progress. Shows the
          fraction of this doc's chunks that have a context_prefix row in
          chunk_context. 100% only after the user runs the contextual pass. */}
      <div
        className="space-y-1"
        title={
          cov >= 1
            ? `All ${doc.chunk_count} chunks have a contextual prefix embedded.`
            : `${Math.round(cov * doc.chunk_count)} of ${doc.chunk_count} chunks have a context prefix. ` +
              `Run the contextual pass from the Ingestion tab to fill the rest.`
        }
      >
        <div className="flex items-center justify-between text-[10px] uppercase tracking-wider text-citi-blue/80">
          <span>Contextual Coverage</span>
          <span className="tabular-nums normal-case text-citi-blue">
            {Math.round(cov * doc.chunk_count)}/{doc.chunk_count} · {(cov * 100).toFixed(0)}%
          </span>
        </div>
        <div className="h-1.5 rounded-full bg-bg-soft overflow-hidden">
          <div
            className={cn(
              'h-full transition-all',
              cov >= 1 ? 'bg-emerald-500/70' : 'bg-accent/70',
            )}
            style={{ width: `${(cov * 100).toFixed(0)}%` }}
          />
        </div>
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


// ───────────────────────────────────────────────────────────────────────────
// UploadSessionPanel — inspector for the last few upload sessions
//
// Two tabbed views per session:
//   1. Response  — the full JSON payload returned by /api/ingest/wega (the
//                  `done` event) plus every other SSE event captured during
//                  the upload. Downloadable as a single .json file.
//   2. Chunks    — paginated table of the chunk_embeddings rows that were
//                  just inserted (fetched once via /api/documents/{id}/chunks).
//                  Downloadable as .csv.
//
// Designed to stay compact: only the active session is rendered, headers
// are sticky, the table caps at a max height with internal scroll, and
// content previews are truncated with full text in a tooltip.
// ───────────────────────────────────────────────────────────────────────────
function UploadSessionPanel({
  sessions,
  activeKey,
  onSelect,
  onRefetchChunks,
}: {
  sessions: UploadSession[]
  activeKey: string | null
  onSelect: (key: string) => void
  onRefetchChunks: (key: string, documentId: string) => Promise<void>
}) {
  const active = useMemo(
    () => sessions.find(s => s.key === activeKey) ?? sessions[0] ?? null,
    [sessions, activeKey],
  )
  const [view, setView] = useState<'json' | 'chunks'>('json')
  const [previewOpen, setPreviewOpen] = useState(false)

  if (!active) return null

  const isRunning = active.finishedAt == null
  const summary = active.summary
  const docId = active.documentId
  const chunks = active.chunks ?? []
  const processed = (summary?.processed as number | undefined) ?? chunks.length

  // Two possible JSON shapes:
  //   1. WEGA chunker_result (preferred) — the structured SDK output emitted
  //      by ingest_core.py for the remote-WEGA path.
  //   2. Fallback SSE events log — every event we captured during the upload.
  //      Shown when the chunker_result event was never emitted (e.g. the
  //      local-Vertex or WEGA-subprocess paths).
  const hasChunkerResult = active.chunkerResult != null
  const jsonPayload = useMemo(() => {
    if (hasChunkerResult) {
      return {
        document_name: active.documentName,
        document_id: active.documentId,
        sha256: active.sha256,
        ingest_source: active.source,
        mode: active.mode,
        chunker_result: active.chunkerResult,
      }
    }
    return {
      document_name: active.documentName,
      document_id: active.documentId,
      sha256: active.sha256,
      ingest_source: active.source,
      mode: active.mode,
      started_at: new Date(active.startedAt).toISOString(),
      finished_at: active.finishedAt ? new Date(active.finishedAt).toISOString() : null,
      error: active.error,
      summary: active.summary,
      events: active.events,
      note: 'chunker_result not emitted by this path — falling back to SSE event log.',
    }
  }, [active, hasChunkerResult])

  const jsonText = useMemo(() => JSON.stringify(jsonPayload, null, 2), [jsonPayload])

  function downloadJson() {
    const base = sanitizeFilename(active.documentName.replace(/\.[^.]+$/, ''))
    const suffix = hasChunkerResult ? 'chunker.json' : 'ingest.json'
    triggerDownload(`${base}.${suffix}`, jsonText, 'application/json')
  }
  function downloadCsv() {
    if (!chunks.length) return
    const base = sanitizeFilename(active.documentName.replace(/\.[^.]+$/, ''))
    triggerDownload(`${base}.chunks.csv`, chunksToCsv(chunks), 'text/csv')
  }

  return (
    <div className="card p-5 space-y-3">
      <div className="flex items-center gap-2 flex-wrap">
        <History className="w-4 h-4 text-accent" />
        <h2 className="font-medium text-sm">Recent Upload Session</h2>
        <span
          className="chip text-[10px]"
          title={`Ingested via ${active.sourceLabel}${active.mode ? ` (mode: ${active.mode})` : ''}`}
        >
          {modeTag(active)}
        </span>
        {isRunning && (
          <span className="chip-accent text-[10px] inline-flex items-center gap-1">
            <Loader2 className="w-3 h-3 animate-spin" />
            running
          </span>
        )}
        {!isRunning && active.error && (
          <span className="chip text-[10px] border-red-300 bg-red-500/10 text-red-700">
            error
          </span>
        )}
        {!isRunning && !active.error && (
          <span className="chip-success text-[10px]">complete</span>
        )}
      </div>

      {/* Session pills — only rendered when there are multiple */}
      {sessions.length > 1 && (
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="text-[10px] uppercase tracking-wider text-citi-blue/80">
            sessions
          </span>
          {sessions.map(s => {
            const isActive = s.key === active.key
            return (
              <button
                key={s.key}
                onClick={() => onSelect(s.key)}
                className={cn(
                  'text-[11px] px-2 py-0.5 rounded-md border transition truncate max-w-[12rem]',
                  isActive
                    ? 'border-accent bg-accent text-white shadow-glow'
                    : 'border-line bg-bg-soft text-citi-blue hover:border-accent/60',
                )}
                title={`${s.documentName} · ${s.sourceLabel} · ${new Date(s.startedAt).toLocaleTimeString()}`}
              >
                <span className={cn('opacity-75 mr-1', !isActive && 'text-accent-dark')}>
                  {modeTag(s)}
                </span>
                {s.documentName}
              </button>
            )
          })}
        </div>
      )}

      {/* Identity strip */}
      <div className="card-soft p-2.5 space-y-1">
        <div className="flex items-center gap-2">
          <div className="text-sm font-medium text-ink truncate flex-1" title={active.documentName}>
            {active.documentName}
          </div>
          {docId && (
            <button
              onClick={() => setPreviewOpen(true)}
              title="Preview the source PDF inline"
              className="inline-flex items-center gap-1 px-2 py-0.5 text-[11px] rounded-md
                         border border-line bg-bg-soft text-citi-blue
                         hover:border-accent hover:text-accent-dark transition shrink-0"
            >
              <Eye className="w-3 h-3" />
              View
            </button>
          )}
        </div>
        <div className="flex flex-wrap items-center gap-2 text-[11px] text-citi-blue">
          {docId && (
            <span
              className="font-mono text-[10px] truncate"
              title={`document_id: ${docId}`}
            >
              id: {docId.slice(0, 8)}…
            </span>
          )}
          {active.sha256 && (
            <span
              className="font-mono text-[10px] truncate"
              title={`sha256: ${active.sha256}`}
            >
              sha: {active.sha256.slice(0, 8)}…
            </span>
          )}
          <span className="tabular-nums">{processed} chunks</span>
          <span>·</span>
          <span title={`Ingestion source: ${active.sourceLabel}${active.mode ? ` (mode: ${active.mode})` : ''}`}>
            via {active.sourceLabel}
          </span>
          <span>·</span>
          <span title={new Date(active.startedAt).toLocaleString()}>
            {new Date(active.startedAt).toLocaleTimeString()}
          </span>
        </div>
      </div>

      {/* Tab bar */}
      <div className="flex items-center gap-1 border-b border-line">
        <TabButton
          active={view === 'json'}
          onClick={() => setView('json')}
          icon={<Braces className="w-3.5 h-3.5" />}
          label={hasChunkerResult ? 'Chunker Result' : 'Response JSON'}
          badge={hasChunkerResult ? 'wega' : `${active.events.length}`}
        />
        <TabButton
          active={view === 'chunks'}
          onClick={() => setView('chunks')}
          icon={<TableIcon className="w-3.5 h-3.5" />}
          label="Inserted Chunks"
          badge={
            active.chunksLoading
              ? '…'
              : chunks.length > 0
                ? String(chunks.length)
                : (docId ? '0' : '—')
          }
        />
        <div className="ml-auto flex items-center gap-1.5">
          {view === 'json' && (
            <button
              onClick={downloadJson}
              className="inline-flex items-center gap-1 px-2 py-1 text-[11px] rounded-md
                         border border-line bg-bg-soft text-citi-blue
                         hover:border-accent hover:text-accent-dark transition"
            >
              <Download className="w-3 h-3" />
              Download JSON
            </button>
          )}
          {view === 'chunks' && (
            <>
              {docId && (
                <button
                  onClick={() => onRefetchChunks(active.key, docId)}
                  disabled={active.chunksLoading}
                  title="Refetch chunks from database"
                  className="inline-flex items-center gap-1 px-2 py-1 text-[11px] rounded-md
                             border border-line bg-bg-soft text-citi-blue
                             hover:border-accent hover:text-accent-dark transition
                             disabled:opacity-50"
                >
                  <RefreshCw className={cn('w-3 h-3', active.chunksLoading && 'animate-spin')} />
                  Refresh
                </button>
              )}
              <button
                onClick={downloadCsv}
                disabled={chunks.length === 0}
                className="inline-flex items-center gap-1 px-2 py-1 text-[11px] rounded-md
                           border border-line bg-bg-soft text-citi-blue
                           hover:border-accent hover:text-accent-dark transition
                           disabled:opacity-40 disabled:cursor-not-allowed"
              >
                <Download className="w-3 h-3" />
                Download CSV
              </button>
            </>
          )}
        </div>
      </div>

      {/* Tab content */}
      {view === 'json' ? (
        <SessionJsonView text={jsonText} />
      ) : (
        <SessionChunksView
          chunks={chunks}
          loading={active.chunksLoading}
          error={active.chunksError}
          hasDocumentId={!!docId}
        />
      )}

      {previewOpen && docId && (
        <SessionPreviewModal
          documentId={docId}
          displayName={active.documentName}
          onClose={() => setPreviewOpen(false)}
        />
      )}
    </div>
  )
}

function SessionPreviewModal({
  documentId, displayName, onClose,
}: {
  documentId: string
  displayName: string
  onClose: () => void
}) {
  // /api/documents/{ident}/view accepts either a UUID or a legacy name;
  // we always pass the UUID since it's unambiguous post-migration 005.
  const src = `/api/documents/${encodeURIComponent(documentId)}/view`
  const downloadHref = `/api/documents/${encodeURIComponent(documentId)}/download`
  return (
    <div
      className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4"
      onClick={onClose}
    >
      <div
        className="bg-bg border border-line rounded-lg shadow-2xl w-full max-w-5xl h-[85vh] flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-4 py-2.5 border-b border-line">
          <div className="flex items-center gap-2 text-sm min-w-0">
            <FileText className="w-4 h-4 text-accent shrink-0" />
            <span className="font-medium truncate" title={displayName}>{displayName}</span>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            <a
              href={downloadHref}
              className="text-xs px-2 py-1 rounded border border-line hover:border-accent/60 hover:text-accent transition inline-flex items-center gap-1"
            >
              <Download className="w-3 h-3" />
              Download
            </a>
            <button
              type="button"
              onClick={onClose}
              className="p-1 rounded hover:bg-bg-soft text-citi-blue hover:text-ink transition"
              aria-label="Close preview"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>
        <iframe
          src={src}
          title={`Preview of ${displayName}`}
          className="flex-1 w-full bg-white"
        />
      </div>
    </div>
  )
}

function TabButton({
  active, onClick, icon, label, badge,
}: {
  active: boolean
  onClick: () => void
  icon: React.ReactNode
  label: string
  badge?: string
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        '-mb-px inline-flex items-center gap-1.5 px-3 py-1.5 text-[12px] font-medium',
        'border-b-2 transition',
        active
          ? 'border-accent text-accent-dark'
          : 'border-transparent text-citi-blue hover:text-ink',
      )}
    >
      {icon}
      {label}
      {badge != null && (
        <span
          className={cn(
            'text-[10px] tabular-nums px-1.5 py-0 rounded-full',
            active ? 'bg-accent/15 text-accent-dark' : 'bg-bg-soft text-citi-blue',
          )}
        >
          {badge}
        </span>
      )}
    </button>
  )
}

function SessionJsonView({ text }: { text: string }) {
  return (
    <pre className="bg-bg-soft border border-line rounded-md p-3 font-mono text-[11px]
                    leading-relaxed text-ink overflow-auto max-h-[26rem] whitespace-pre">
      {text}
    </pre>
  )
}

function SessionChunksView({
  chunks, loading, error, hasDocumentId,
}: {
  chunks: SessionChunk[]
  loading: boolean
  error: string | null
  hasDocumentId: boolean
}) {
  if (loading) {
    return (
      <div className="flex items-center gap-2 text-xs text-citi-blue p-3">
        <Loader2 className="w-3.5 h-3.5 animate-spin" />
        Loading inserted chunks…
      </div>
    )
  }
  if (error) {
    return (
      <div className="text-xs text-red-700 bg-red-500/10 border border-red-300 rounded-md p-2 break-all">
        <strong>Error:</strong> {error}
      </div>
    )
  }
  if (!hasDocumentId) {
    return (
      <div className="text-xs text-citi-blue p-3">
        No <code className="text-accent-dark">document_id</code> on this session — the
        backend likely ingested before migration 005. Re-upload to capture chunks for download.
      </div>
    )
  }
  if (chunks.length === 0) {
    return (
      <div className="text-xs text-citi-blue p-3">No chunks returned.</div>
    )
  }
  return (
    <div className="border border-line rounded-md overflow-hidden">
      <div className="max-h-[26rem] overflow-auto">
        <table className="w-full text-[11px] text-left">
          <thead className="bg-bg-soft text-citi-blue uppercase tracking-wider sticky top-0">
            <tr>
              <Th className="w-12">id</Th>
              <Th className="w-12">pg</Th>
              <Th className="w-14">tok</Th>
              <Th className="w-20">type</Th>
              <Th>content</Th>
              <Th className="w-32">context</Th>
            </tr>
          </thead>
          <tbody>
            {chunks.map((c, i) => (
              <tr
                key={c.id}
                className={cn(
                  'align-top border-t border-line',
                  i % 2 === 1 && 'bg-bg-soft/40',
                )}
              >
                <Td className="font-mono tabular-nums">{c.id}</Td>
                <Td className="tabular-nums">{c.page_number ?? '—'}</Td>
                <Td className="tabular-nums">{c.token_count ?? '—'}</Td>
                <Td>{c.chunk_type ?? '—'}</Td>
                <Td>
                  <div
                    className="text-ink line-clamp-2 leading-snug"
                    title={c.content}
                  >
                    {c.content || '—'}
                  </div>
                </Td>
                <Td>
                  <div
                    className="text-citi-blue line-clamp-2 leading-snug"
                    title={c.context_text || ''}
                  >
                    {c.context_text || '—'}
                  </div>
                </Td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="bg-bg-soft border-t border-line px-3 py-1 text-[10px] text-citi-blue tabular-nums flex items-center gap-2">
        <Rows3 className="w-3 h-3" />
        {chunks.length} {chunks.length === 1 ? 'row' : 'rows'}
        {chunks.length === CHUNKS_FETCH_LIMIT && (
          <span className="text-amber-700">
            (capped at {CHUNKS_FETCH_LIMIT} — increase limit if needed)
          </span>
        )}
      </div>
    </div>
  )
}

function Th({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <th className={cn('px-2 py-1.5 font-semibold text-[10px]', className)}>{children}</th>
  )
}

function Td({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <td className={cn('px-2 py-1.5', className)}>{children}</td>
  )
}
