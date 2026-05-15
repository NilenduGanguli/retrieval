/**
 * KYC Intelligence — specialised tab for KYC document workflows.
 *
 * Three sub-sections:
 *   1. Upload + classify (2-pass: classify → type-specific extract → embed)
 *   2. Search (Owner / Entity + Universal keyword sub-tabs)
 *   3. Document Browser (group by owner, filter by category)
 */
import {
  Building2, ChevronDown, ChevronRight, FileText, Filter, IdCard, Layers,
  Loader2, RefreshCw, Search, ShieldCheck, Sparkles, Trash2, Upload,
} from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'

import { api, uploadSse } from '@/lib/api'
import { cn } from '@/lib/cn'

// ───────────────────────────────────────────────────────────────────────────
// Helpers
// ───────────────────────────────────────────────────────────────────────────

function docIcon(docType: string | null | undefined): string {
  const t = (docType || '').toLowerCase()
  if (t.includes('orbis') || t.includes('moody')) return '🔭'
  if (t.match(/d&b|dun|lexisnexis|aml|worldbase|market intelligence/)) return '🔎'
  if (t.match(/incorporation|registration|continuance|amendment|memorandum|articles/)) return '🏢'
  if (t.match(/bank|signature card|banking resolution|financial|statement/)) return '🏦'
  if (t.match(/passport|aadhaar|pan|driver|voter|national id/)) return '🪪'
  if (t.match(/gis|kyc|ownership|beneficial/)) return '📊'
  if (t.match(/resolution|power of attorney|notarial|board/)) return '⚖️'
  if (t.match(/agreement|contract|loan|guarantee/)) return '📝'
  if (t.match(/salary|payslip/)) return '💰'
  if (t.match(/utility|bill/)) return '🧾'
  if (t.match(/insurance/)) return '🛡️'
  if (t.match(/resignation|letter/)) return '✉️'
  if (t.match(/lodgement|form/)) return '📋'
  if (t.match(/tax/)) return '🧮'
  return '📄'
}

function ConfBadge({ score }: { score?: number | string | null }) {
  if (score == null || score === '') return null
  const v = typeof score === 'string' ? parseFloat(score) : score
  if (Number.isNaN(v)) return null
  const pct = `${Math.round(v * 100)}%`
  const tone =
    v >= 0.8 ? 'border-emerald-600 bg-emerald-500/15 text-emerald-800'
      : v >= 0.5 ? 'border-amber-600 bg-amber-500/15 text-amber-800'
        : 'border-slate-400 bg-slate-200 text-slate-800'
  return (
    <span className={cn('inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-medium border', tone)}>
      {v >= 0.8 ? '✓' : v >= 0.5 ? '~' : ''} {pct}
    </span>
  )
}

// ───────────────────────────────────────────────────────────────────────────
// Stage chip strip for ingest progress
// ───────────────────────────────────────────────────────────────────────────
type StageState = 'idle' | 'running' | 'done' | 'error'
type StageMap = Record<string, { state: StageState; detail?: string }>

const KYC_STAGE_ORDER = ['upload', 'ocr', 'classify', 'extract', 'embed', 'store'] as const
const KYC_STAGE_LABEL: Record<string, string> = {
  upload: 's3 upload', ocr: 'ocr', classify: 'classify',
  extract: 'extract', embed: 'embed', store: 'store',
}

function StageChips({ stages }: { stages: StageMap }) {
  return (
    <div className="card-soft p-2 flex flex-wrap gap-1.5 text-[11px]">
      {KYC_STAGE_ORDER.map((s) => {
        const st = stages[s]?.state ?? 'idle'
        return (
          <span
            key={s}
            className={cn(
              'inline-flex items-center gap-1 px-2 py-0.5 rounded-md border transition',
              st === 'done' && 'border-emerald-600 bg-emerald-500/15 text-emerald-800',
              st === 'running' && 'border-accent bg-accent/15 text-accent-dark shadow-glow',
              st === 'idle' && 'border-line bg-bg-soft text-citi-blue',
              st === 'error' && 'border-red-500 bg-red-500/10 text-red-700',
            )}
          >
            {KYC_STAGE_LABEL[s]}
            {stages[s]?.detail && (
              <span className="text-[10px] opacity-80 ml-0.5">· {stages[s]!.detail}</span>
            )}
          </span>
        )
      })}
    </div>
  )
}

// ───────────────────────────────────────────────────────────────────────────
// Main component
// ───────────────────────────────────────────────────────────────────────────

export default function KYCTab() {
  const [taxonomy, setTaxonomy] = useState<{ categories: Record<string, string[]>; all_doc_types: string[] } | null>(null)
  const [owners, setOwners] = useState<Array<{ owner: string; doc_count: number }>>([])
  const [allDocTypes, setAllDocTypes] = useState<string[]>([])

  // Ingest state
  const fileRef = useRef<HTMLInputElement>(null)
  const [queue, setQueue] = useState<File[]>([])
  const [current, setCurrent] = useState<string | null>(null)
  const [stages, setStages] = useState<StageMap>({})
  const [ingestLog, setIngestLog] = useState<Array<{ text: string; tone?: string }>>([])
  const abortRef = useRef<(() => void) | null>(null)

  // Search state — sub-tab toggle
  const [searchTab, setSearchTab] = useState<'owner' | 'universal'>('owner')

  // Owner-search state
  const [ownerInput, setOwnerInput] = useState('')
  const [selectedOwner, setSelectedOwner] = useState('')
  const [docTypeFilter, setDocTypeFilter] = useState('')
  const [ownerDocTypes, setOwnerDocTypes] = useState<string[]>([])
  const [listResults, setListResults] = useState<any[] | null>(null)
  const [extractResult, setExtractResult] = useState<any | null>(null)
  const [searching, setSearching] = useState(false)

  // Universal-search state
  const [keyword, setKeyword] = useState('')
  const [uResults, setUResults] = useState<any[] | null>(null)

  // Search errors (surfaced in UI so silent failures don't masquerade as "no data")
  const [listError, setListError] = useState<string | null>(null)
  const [uError, setUError] = useState<string | null>(null)

  // Browser state
  const [browseCategory, setBrowseCategory] = useState('All Categories')
  const [browseGroups, setBrowseGroups] = useState<Array<{ owner: string; docs: any[] }>>([])
  const [browseTotal, setBrowseTotal] = useState(0)
  const [expanded, setExpanded] = useState<Record<string, boolean>>({})
  const [browseLoading, setBrowseLoading] = useState(false)

  // ── Initial loads ───────────────────────────────────────────
  useEffect(() => {
    api.kycTaxonomy().then(setTaxonomy).catch(console.warn)
    refreshOwners()
    api.kycDocTypes().then((rows) => setAllDocTypes(rows.map(r => r.document_type))).catch(console.warn)
  }, [])

  async function refreshOwners() {
    try {
      const rows = await api.kycOwners()
      setOwners(rows)
    } catch (e) {
      console.warn('kyc owners failed', e)
    }
  }

  // ── Owner suggestions ──────────────────────────────────────
  const ownerSuggestions = useMemo(() => {
    if (!ownerInput || ownerInput.length < 2) return []
    const q = ownerInput.toLowerCase()
    const starts = owners.filter(o => o.owner.toLowerCase().startsWith(q))
    const contains = owners.filter(o => o.owner.toLowerCase().includes(q) && !o.owner.toLowerCase().startsWith(q))
    return [...starts, ...contains].slice(0, 8)
  }, [ownerInput, owners])

  // When selected owner changes, fetch its doc types
  useEffect(() => {
    if (selectedOwner) {
      api.kycDocTypes(selectedOwner)
        .then((rows) => setOwnerDocTypes(rows.map(r => r.document_type)))
        .catch(() => setOwnerDocTypes([]))
    } else {
      setOwnerDocTypes([])
    }
  }, [selectedOwner])

  // ── Ingestion driver ────────────────────────────────────────
  function uploadOne(file: File, onDone?: () => void) {
    setCurrent(file.name)
    setStages({})
    setIngestLog(l => [...l, { text: `─── ${file.name} (${(file.size / 1024).toFixed(1)} KB) ───` }])

    const form = new FormData()
    form.append('file', file)
    abortRef.current?.()
    abortRef.current = uploadSse(
      '/api/kyc/ingest',
      form,
      (evt, raw) => {
        try {
          const data = JSON.parse(raw)
          if (evt === 'start') {
            setIngestLog(l => [...l, { text: `▶ ${data.filename} (${(data.size_bytes / 1024).toFixed(1)} KB)`, tone: 'info' }])
          } else if (evt === 'stage') {
            const stage = data.stage
            const status = data.status
            let detail = ''
            if (stage === 'ocr' && status === 'done') detail = `${data.pages}p · ${data.chars}c`
            if (stage === 'classify' && status === 'done') detail = data.document_type
            if (stage === 'extract' && status === 'done') detail = `${data.fields} fields`
            if (stage === 'embed' && status === 'progress') detail = `${data.done}/${data.total}`
            if (stage === 'embed' && status === 'done') detail = `${data.embeddings}`
            if (stage === 'store' && status === 'done') detail = `${data.chunks} chunks`
            setStages(prev => ({
              ...prev,
              [stage]: {
                state: status === 'done' ? 'done' : (status === 'error' ? 'error' : 'running'),
                detail,
              },
            }))
            if (stage === 'classify' && status === 'done') {
              setIngestLog(l => [...l, {
                text: `${docIcon(data.document_type)} ${data.owner} → ${data.document_type} (${Math.round((data.confidence || 0) * 100)}%)`,
                tone: 'done',
              }])
            }
          } else if (evt === 'done') {
            setStages(prev => ({ ...prev, store: { state: 'done', detail: `${data.chunks} chunks` } }))
            setIngestLog(l => [...l, {
              text: `✓ ${data.owner} · ${data.document_type} · ${data.chunks} chunks → id ${data.kyc_document_id}`,
              tone: 'done',
            }])
            refreshOwners()
            api.kycDocTypes().then(rows => setAllDocTypes(rows.map(r => r.document_type))).catch(() => undefined)
            onDone?.()
          } else if (evt === 'error') {
            setIngestLog(l => [...l, { text: `✗ ${data.message || raw}`, tone: 'error' }])
            onDone?.()
          }
        } catch {
          setIngestLog(l => [...l, { text: raw }])
        }
      },
      (err) => {
        setIngestLog(l => [...l, { text: `✗ ${String(err)}`, tone: 'error' }])
        onDone?.()
      },
    )
  }

  function enqueue(files: FileList | File[]) {
    const arr = Array.from(files).filter(f => f.size > 0)
    if (!arr.length) return
    setQueue(prev => [...prev, ...arr])
  }

  // drive the queue
  useEffect(() => {
    if (current != null || queue.length === 0) return
    const [next, ...rest] = queue
    setQueue(rest)
    uploadOne(next, () => setCurrent(null))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [queue, current])

  // ── Owner search actions ────────────────────────────────────
  async function doListByOwner() {
    if (!selectedOwner && !ownerInput) return
    const o = selectedOwner || ownerInput
    setSearching(true)
    setListResults(null)
    setExtractResult(null)
    setListError(null)
    try {
      const r = await api.kycListByOwner(o, docTypeFilter || undefined)
      console.debug('[kyc] list-by-owner response', r)
      setListResults(Array.isArray(r?.results) ? r.results : [])
    } catch (e) {
      console.error('[kyc] list-by-owner failed', e)
      setListError(String(e))
      setListResults([])
    } finally {
      setSearching(false)
    }
  }

  async function doExtract() {
    const o = selectedOwner || ownerInput
    if (!o || !docTypeFilter) return
    setSearching(true)
    setListResults(null)
    setExtractResult(null)
    setListError(null)
    try {
      const r = await api.kycExtract(o, docTypeFilter)
      console.debug('[kyc] extract response', r)
      setExtractResult(r.result || { _empty: true })
    } catch (e) {
      console.error('[kyc] extract failed', e)
      setExtractResult({ _error: String(e) })
    } finally {
      setSearching(false)
    }
  }

  // ── Universal search ────────────────────────────────────────
  async function doUniversal() {
    if (!keyword.trim()) return
    setSearching(true)
    setUResults(null)
    setUError(null)
    try {
      const r = await api.kycUniversal(keyword.trim(), 8)
      console.debug('[kyc] universal-search response', r)
      setUResults(Array.isArray(r?.results) ? r.results : [])
    } catch (e) {
      console.error('[kyc] universal-search failed', e)
      setUError(String(e))
      setUResults([])
    } finally {
      setSearching(false)
    }
  }

  // ── Browse ──────────────────────────────────────────────────
  async function doBrowse() {
    setBrowseLoading(true)
    try {
      const r = await api.kycBrowse(browseCategory === 'All Categories' ? undefined : browseCategory)
      setBrowseGroups(r.groups)
      setBrowseTotal(r.total)
    } finally {
      setBrowseLoading(false)
    }
  }

  const totalDocsInOwners = owners.reduce((s, o) => s + o.doc_count, 0)
  const ownerDocTypesSet = useMemo(() => new Set(ownerDocTypes), [ownerDocTypes])

  // ───────────── RENDER ─────────────
  return (
    <div className="space-y-6">
      {/* HEADER STATS */}
      <div className="grid grid-cols-4 gap-3">
        <Stat label="ENTITIES"  value={owners.length} />
        <Stat label="DOCUMENTS" value={totalDocsInOwners} />
        <Stat label="DOC TYPES" value={allDocTypes.length} />
        <Stat label="CATEGORIES" value={taxonomy ? Object.keys(taxonomy.categories).length : '—'} />
      </div>

      {/* ── SECTION 1: UPLOAD ── */}
      <section className="card p-5 space-y-3">
        <div className="flex items-center gap-2">
          <Upload className="w-4 h-4 text-accent" />
          <h2 className="font-semibold text-sm text-ink">Document Upload & Classification</h2>
          <span className="chip-accent ml-2">2-pass: classify → extract</span>
        </div>
        <p className="text-xs text-citi-blue">
          Drop KYC PDFs (Orbis, D&amp;B, LexisNexis, COI, GIS, Bank Statements, IDs, …).
          Each file runs OCR → LLM classify → type-specific field extract → embed → pgvector.
        </p>

        <div className="flex items-center gap-2">
          <input
            ref={fileRef}
            type="file"
            accept="application/pdf"
            multiple
            className="block text-xs text-ink file:mr-3 file:py-1.5 file:px-3 file:rounded-md
                       file:border file:border-line file:bg-bg-soft file:text-ink hover:file:bg-bg-card"
            onChange={(e) => {
              if (e.target.files && e.target.files.length) {
                enqueue(e.target.files)
                e.target.value = ''
              }
            }}
          />
          {current && <span className="chip-accent text-[11px]">▶ {current}</span>}
          {queue.map((f, i) => (
            <span key={i} className="chip text-[11px]">⏳ {f.name}</span>
          ))}
        </div>

        {Object.keys(stages).length > 0 && <StageChips stages={stages} />}

        {ingestLog.length > 0 && (
          <div className="bg-bg-soft border border-line rounded-md p-2 font-mono text-[11px]
                          max-h-40 overflow-y-auto leading-snug">
            {ingestLog.slice(-200).map((l, i) => (
              <div
                key={i}
                className={cn(
                  'whitespace-pre-wrap',
                  l.tone === 'error' && 'text-red-700',
                  l.tone === 'done' && 'text-emerald-700',
                  l.tone === 'info' && 'text-citi-blue',
                  !l.tone && 'text-ink',
                )}
              >
                {l.text}
              </div>
            ))}
          </div>
        )}
      </section>

      {/* ── SECTION 2: SEARCH ── */}
      <section className="card p-5 space-y-3">
        <div className="flex items-center gap-2">
          <Search className="w-4 h-4 text-accent" />
          <h2 className="font-semibold text-sm text-ink">Document Search & Retrieval</h2>
        </div>

        <div className="inline-flex gap-1 p-1 rounded-lg bg-slate-100 border border-line">
          <button
            onClick={() => setSearchTab('owner')}
            className={cn(
              'inline-flex items-center gap-1.5 px-3 py-1 rounded-md text-xs font-medium transition',
              searchTab === 'owner' ? 'bg-accent text-white shadow-glow' : 'text-ink hover:text-citi-blue',
            )}
          >
            <Building2 className="w-3.5 h-3.5" />
            Owner / Entity
          </button>
          <button
            onClick={() => setSearchTab('universal')}
            className={cn(
              'inline-flex items-center gap-1.5 px-3 py-1 rounded-md text-xs font-medium transition',
              searchTab === 'universal' ? 'bg-accent text-white shadow-glow' : 'text-ink hover:text-citi-blue',
            )}
          >
            <IdCard className="w-3.5 h-3.5" />
            Universal Keyword
          </button>
        </div>

        {searchTab === 'owner' && (
          <div className="space-y-3">
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-[11px] font-semibold text-citi-blue uppercase tracking-wider">
                  Person / Company Name
                </label>
                <input
                  type="text"
                  value={selectedOwner || ownerInput}
                  onChange={(e) => {
                    setSelectedOwner('')
                    setOwnerInput(e.target.value)
                  }}
                  placeholder="Start typing a name…"
                  className="mt-1 w-full bg-white border border-line rounded-md px-2 py-1.5 text-sm text-ink
                             placeholder:text-citi-blue/60 focus:outline-none focus:border-accent focus:ring-2 focus:ring-accent/20"
                />
                {ownerSuggestions.length > 0 && !selectedOwner && (
                  <div className="mt-1 space-y-1">
                    {ownerSuggestions.map((s) => (
                      <button
                        key={s.owner}
                        onClick={() => { setSelectedOwner(s.owner); setOwnerInput(s.owner) }}
                        className="block w-full text-left px-2 py-1 rounded-md text-xs text-ink
                                   hover:bg-accent/10 hover:text-accent-dark transition"
                      >
                        {s.owner} <span className="text-citi-blue/70">({s.doc_count})</span>
                      </button>
                    ))}
                  </div>
                )}
              </div>
              <div>
                <label className="text-[11px] font-semibold text-citi-blue uppercase tracking-wider">
                  Document Type
                </label>
                <select
                  value={docTypeFilter}
                  onChange={(e) => setDocTypeFilter(e.target.value)}
                  className="mt-1 w-full bg-white border border-line rounded-md px-2 py-1.5 text-sm text-ink
                             focus:outline-none focus:border-accent focus:ring-2 focus:ring-accent/20"
                >
                  <option value="">— any type —</option>
                  {ownerDocTypes.length > 0 && (
                    <optgroup label={`📌 For ${selectedOwner || ownerInput}`}>
                      {ownerDocTypes.map((t) => <option key={t} value={t}>{t}</option>)}
                    </optgroup>
                  )}
                  {taxonomy && (
                    <optgroup label="All types">
                      {taxonomy.all_doc_types
                        .filter(t => !ownerDocTypesSet.has(t))
                        .map((t) => <option key={t} value={t}>{t}</option>)}
                    </optgroup>
                  )}
                </select>
              </div>
            </div>
            <div className="flex gap-2">
              <button
                onClick={doListByOwner}
                disabled={searching || (!selectedOwner && !ownerInput)}
                className="inline-flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium
                           bg-accent text-white hover:bg-accent-dark transition
                           disabled:opacity-40 disabled:cursor-not-allowed"
              >
                <FileText className="w-4 h-4" />
                List All Documents
              </button>
              <button
                onClick={doExtract}
                disabled={searching || !(selectedOwner || ownerInput) || !docTypeFilter}
                className="inline-flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium
                           bg-citi-blue text-white hover:bg-accent-dark transition
                           disabled:opacity-40 disabled:cursor-not-allowed"
              >
                <Sparkles className="w-4 h-4" />
                Detailed Extraction
              </button>
              {searching && <Loader2 className="w-4 h-4 animate-spin text-accent self-center" />}
            </div>

            {listError && (
              <div className="text-xs text-red-700 bg-red-500/10 border border-red-300 rounded-md p-2 break-all">
                <strong>Error:</strong> {listError}
              </div>
            )}

            {/* LIST results */}
            {listResults !== null && (
              <div className="space-y-2">
                {listResults.length === 0 ? (
                  <div className="text-xs text-citi-blue px-2 py-3">No documents found.</div>
                ) : (
                  <>
                    <div className="text-xs text-ink">
                      Found <strong>{listResults.length}</strong> document(s)
                    </div>
                    {listResults.map((d) => (
                      <DocItemCard key={d.id} doc={d} />
                    ))}
                  </>
                )}
              </div>
            )}

            {/* EXTRACT result */}
            {extractResult && !extractResult._empty && !extractResult._error && (
              <ExtractResultCard r={extractResult} />
            )}
            {extractResult?._empty && (
              <div className="text-xs text-citi-blue px-2 py-3">No matching document for that owner + type.</div>
            )}
            {extractResult?._error && (
              <div className="text-xs text-red-700 px-2 py-3">{extractResult._error}</div>
            )}
          </div>
        )}

        {searchTab === 'universal' && (
          <div className="space-y-3">
            <p className="text-xs text-citi-blue">
              Search for <strong>any value</strong> across all KYC documents — phone numbers, PAN, Aadhaar,
              addresses, account numbers, registration IDs, emails. Results show owner, name, type,
              and the exact field where the value was found.
            </p>
            <div className="flex gap-2">
              <input
                type="text"
                value={keyword}
                onChange={(e) => setKeyword(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter') doUniversal() }}
                placeholder='e.g.  ABCDE1234F  ·  +91-9876543210  ·  ACC-00123'
                className="flex-1 bg-white border border-line rounded-md px-3 py-1.5 text-sm text-ink
                           placeholder:text-citi-blue/60 focus:outline-none focus:border-accent focus:ring-2 focus:ring-accent/20"
              />
              <button
                onClick={doUniversal}
                disabled={searching || !keyword.trim()}
                className="inline-flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium
                           bg-accent text-white hover:bg-accent-dark transition
                           disabled:opacity-40 disabled:cursor-not-allowed"
              >
                <Search className="w-4 h-4" />
                Search
              </button>
              {searching && <Loader2 className="w-4 h-4 animate-spin text-accent self-center" />}
            </div>
            <div className="text-[10px] text-citi-blue/80">
              💡 Try: PAN · phone · street address · account number · registration ID · email · any keyword
            </div>

            {uError && (
              <div className="text-xs text-red-700 bg-red-500/10 border border-red-300 rounded-md p-2 break-all">
                <strong>Error:</strong> {uError}
              </div>
            )}

            {uResults !== null && (
              <div className="space-y-2">
                {uResults.length === 0 ? (
                  <div className="text-xs text-citi-blue px-2 py-3">No documents found for that keyword.</div>
                ) : (
                  <>
                    <div className="text-xs text-ink">
                      Found <strong>{uResults.length}</strong> document(s) matching <em>"{keyword}"</em>
                    </div>
                    {uResults.map((r, i) => (
                      <UniversalHitCard key={i} r={r} />
                    ))}
                  </>
                )}
              </div>
            )}
          </div>
        )}
      </section>

      {/* ── SECTION 3: BROWSE ── */}
      <section className="card p-5 space-y-3">
        <div className="flex items-center gap-2">
          <Layers className="w-4 h-4 text-accent" />
          <h2 className="font-semibold text-sm text-ink">Indexed Document Browser</h2>
        </div>
        <div className="flex items-center gap-2">
          <select
            value={browseCategory}
            onChange={(e) => setBrowseCategory(e.target.value)}
            className="bg-white border border-line rounded-md px-2 py-1.5 text-sm text-ink min-w-[14rem]
                       focus:outline-none focus:border-accent focus:ring-2 focus:ring-accent/20"
          >
            <option>All Categories</option>
            {taxonomy && Object.keys(taxonomy.categories).map((c) => <option key={c}>{c}</option>)}
          </select>
          <button
            onClick={doBrowse}
            disabled={browseLoading}
            className="inline-flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium
                       bg-accent text-white hover:bg-accent-dark transition disabled:opacity-40"
          >
            {browseLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <RefreshCw className="w-4 h-4" />}
            Show All Indexed Documents
          </button>
          {browseTotal > 0 && (
            <span className="chip ml-auto">{browseTotal} document{browseTotal === 1 ? '' : 's'}</span>
          )}
        </div>

        {browseGroups.length > 0 && (
          <div className="space-y-2">
            {browseGroups.map((g) => (
              <details
                key={g.owner}
                className="card-soft p-3 [&[open]_.chev]:rotate-90"
                onToggle={(e) => setExpanded((p) => ({ ...p, [g.owner]: (e.target as HTMLDetailsElement).open }))}
              >
                <summary className="flex items-center gap-2 cursor-pointer list-none">
                  <ChevronRight className="w-3.5 h-3.5 text-citi-blue chev transition" />
                  <span className="font-medium text-ink">👤 {g.owner}</span>
                  <span className="chip ml-2">{g.docs.length} doc{g.docs.length === 1 ? '' : 's'}</span>
                </summary>
                <div className="mt-2 space-y-1.5 pl-5">
                  {g.docs.map((d) => <DocItemCard key={d.id} doc={d} />)}
                </div>
              </details>
            ))}
          </div>
        )}
      </section>
    </div>
  )
}

// ───────── Subcomponents ──────────

function Stat({ label, value }: { label: string; value: number | string }) {
  return (
    <div className="card p-3">
      <div className="text-[10px] uppercase text-citi-blue tracking-wider font-semibold">{label}</div>
      <div className="text-lg font-semibold text-ink mt-0.5">{value}</div>
    </div>
  )
}

function DocItemCard({ doc }: { doc: any }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="border border-line rounded-md bg-bg-card p-2.5 hover:border-accent/60 transition">
      <div className="flex items-start gap-2">
        <span className="text-base shrink-0">{docIcon(doc.document_type)}</span>
        <div className="flex-1 min-w-0">
          <div className="text-sm font-medium text-ink truncate" title={doc.document_name}>
            {doc.document_name}
          </div>
          <div className="text-[11px] text-citi-blue truncate">
            {doc.owner ? <><strong>{doc.owner}</strong> · </> : null}
            {doc.document_type}
            {doc.report_date ? ` · ${doc.report_date}` : ''}
          </div>
        </div>
        <div className="flex items-center gap-1.5 shrink-0">
          {doc.source_platform && (
            <span className="chip text-[10px]">{doc.source_platform}</span>
          )}
          {doc.document_category && (
            <span className="chip-accent text-[10px]">{doc.document_category}</span>
          )}
          <ConfBadge score={doc.confidence_score} />
        </div>
      </div>
      {doc.extracted_data && Object.keys(doc.extracted_data).length > 0 && (
        <button
          onClick={() => setOpen((o) => !o)}
          className="mt-2 text-[10px] text-accent hover:text-accent-dark inline-flex items-center gap-1"
        >
          {open ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
          {open ? 'Hide' : 'Show'} extracted fields ({Object.keys(doc.extracted_data).length})
        </button>
      )}
      {open && doc.extracted_data && (
        <pre className="mt-2 text-[10px] font-mono bg-bg-soft border border-line rounded p-2 overflow-x-auto text-ink whitespace-pre-wrap">
          {JSON.stringify(doc.extracted_data, null, 2)}
        </pre>
      )}
    </div>
  )
}

function ExtractResultCard({ r }: { r: any }) {
  const cleanData = Object.fromEntries(
    Object.entries(r.data || {}).filter(([, v]) => v !== null && v !== '' && v !== 'null'),
  )
  return (
    <div className="card-soft p-3 space-y-2">
      <div className="flex items-start gap-2">
        <span className="text-xl">{docIcon(r.document_type)}</span>
        <div className="flex-1 min-w-0">
          <div className="text-sm font-semibold text-ink">{r.owner}</div>
          <div className="text-xs text-citi-blue">{r.document_name} · {r.document_type}</div>
        </div>
        <ConfBadge score={r.score} />
      </div>
      <div className="flex flex-wrap items-center gap-1.5 text-[10px]">
        {r.source_platform && <span className="chip">{r.source_platform}</span>}
        {r.report_date && <span className="chip">{r.report_date}</span>}
        <ConfBadge score={r.confidence_score} />
      </div>
      <details open className="text-[11px]">
        <summary className="cursor-pointer text-accent hover:text-accent-dark font-medium">
          📋 Extracted Data ({Object.keys(cleanData).length} fields)
        </summary>
        <pre className="mt-2 bg-bg-soft border border-line rounded p-2 overflow-x-auto font-mono text-ink whitespace-pre-wrap">
          {JSON.stringify(cleanData, null, 2)}
        </pre>
      </details>
    </div>
  )
}

function UniversalHitCard({ r }: { r: any }) {
  const cleanField = String(r.matched_field || 'content')
    .replace(/^ext_/, '')
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase())
  const srcBadge = r.match_source === 'metadata'
    ? <span className="chip-success text-[10px]">● Metadata Hit</span>
    : <span className="chip-accent text-[10px]">● Content Match</span>
  return (
    <div className="card-soft p-3 space-y-2">
      <div className="flex items-start gap-2">
        <span className="text-xl">{docIcon(r.document_type)}</span>
        <div className="flex-1 min-w-0">
          <div className="text-sm font-semibold text-ink">{r.owner || '—'}</div>
          <div className="text-xs text-citi-blue">{r.document_name} · {r.document_type}</div>
        </div>
        <div className="flex flex-col items-end gap-1 shrink-0">
          <ConfBadge score={r.relevance_score} />
          {srcBadge}
        </div>
      </div>
      <div className="border-l-2 border-accent pl-2">
        <div className="text-[10px] uppercase text-citi-blue font-semibold tracking-wider">
          🔎 Matched: {cleanField}
        </div>
        <div className="text-sm text-ink mt-0.5 break-all">{r.matched_value}</div>
      </div>
      <div className="flex flex-wrap gap-1.5 text-[10px]">
        {r.document_category && <span className="chip">{r.document_category}</span>}
        {r.confidence_score != null && <ConfBadge score={r.confidence_score} />}
      </div>
    </div>
  )
}
