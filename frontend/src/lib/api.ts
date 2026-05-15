import type {
  AnalyticsSummary,
  BackendConfig,
  BenchRunResult,
  DocumentSummary,
  GoldenQuestion,
  HealthInfo,
  QueryLogEntry,
  RetrieveResponse,
  Strategy,
} from '@/types'

const BASE = ''

async function json<T>(input: RequestInfo, init?: RequestInit): Promise<T> {
  const res = await fetch(input, init)
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}: ${await res.text()}`)
  return res.json() as Promise<T>
}

export const api = {
  health: () => json<HealthInfo>(`${BASE}/api/health`),
  config: () => json<BackendConfig>(`${BASE}/api/config`),

  documents: () => json<DocumentSummary[]>(`${BASE}/api/documents`),
  documentChunks: (ident: string, limit = 100) =>
    json<{ document_id: string | null; document_name: string | null; chunks: Array<any> }>(
      `${BASE}/api/documents/${encodeURIComponent(ident)}/chunks?limit=${limit}`,
    ),
  // ident is a document_id (UUID, preferred) or legacy document_name.
  deleteDocument: (ident: string) =>
    json<{ document_id: string | null; document_name: string | null; soft_deleted_chunks: number; s3_uri_removed: string | null }>(
      `${BASE}/api/documents/${encodeURIComponent(ident)}`,
      { method: 'DELETE' },
    ),

  retrieve: (query: string, strategy: Strategy) =>
    json<RetrieveResponse>(`${BASE}/api/retrieve`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ query, strategy }),
    }),

  analyticsSummary: () => json<AnalyticsSummary>(`${BASE}/api/analytics/summary`),
  recentQueries: (limit = 50) =>
    json<QueryLogEntry[]>(`${BASE}/api/analytics/queries?limit=${limit}`),

  // Benchmark
  listQuestions: () => json<GoldenQuestion[]>(`${BASE}/api/bench/questions`),
  addQuestion: (q: GoldenQuestion) =>
    json<GoldenQuestion>(`${BASE}/api/bench/questions`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(q),
    }),
  deleteQuestion: (qid: number) =>
    json<{ ok: boolean }>(`${BASE}/api/bench/questions/${qid}`, { method: 'DELETE' }),
  seedQuestions: (perDocument = 2) =>
    json<{ inserted: number; questions: Array<{ id: number; question: string; document_name: string; ground_truth_chunk_ids: number[] }> }>(
      `${BASE}/api/bench/seed-from-docs?per_document=${perDocument}`,
      { method: 'POST' },
    ),
  // Note: /api/bench/run is now an SSE stream. Use postSse() instead of api.runBench.
  listRuns: () =>
    json<Array<{
      id: number
      label: string | null
      strategy: Record<string, unknown>
      metrics: Record<string, number>
      n_questions: number
      created_at: string
    }>>(`${BASE}/api/bench/runs`),

  // KYC Intelligence ------------------------------------------------------
  kycTaxonomy: () => json<{ categories: Record<string, string[]>; all_doc_types: string[] }>(
    `${BASE}/api/kyc/taxonomy`,
  ),
  kycOwners: () => json<Array<{ owner: string; owner_normalized: string; doc_count: number }>>(
    `${BASE}/api/kyc/owners`,
  ),
  kycDocTypes: (owner?: string) =>
    json<Array<{ document_type: string; document_category: string | null; doc_count: number }>>(
      `${BASE}/api/kyc/doc-types${owner ? `?owner=${encodeURIComponent(owner)}` : ''}`,
    ),
  kycListByOwner: (owner: string, document_type?: string) =>
    json<{ owner: string; document_type: string | null; results: any[] }>(
      `${BASE}/api/kyc/list-by-owner`,
      {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ owner, document_type: document_type || null }),
      },
    ),
  kycExtract: (owner: string, document_type: string) =>
    json<{ owner: string; document_type: string; result: any }>(
      `${BASE}/api/kyc/extract`,
      {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ owner, document_type }),
      },
    ),
  kycUniversal: (keyword: string, top_k = 8) =>
    json<{ keyword: string; results: any[] }>(`${BASE}/api/kyc/universal-search`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ keyword, top_k }),
    }),
  kycBrowse: (category?: string) =>
    json<{ category: string | null; total: number; groups: Array<{ owner: string; docs: any[] }> }>(
      `${BASE}/api/kyc/browse${category ? `?category=${encodeURIComponent(category)}` : ''}`,
    ),
  kycDelete: (document_name: string) =>
    json<{ ok: boolean; id: number; s3_uri: string | null }>(
      `${BASE}/api/kyc/${encodeURIComponent(document_name)}`,
      { method: 'DELETE' },
    ),
}

// SSE helpers --------------------------------------------------------------
export type SseHandler = (event: string, data: string) => void

/**
 * Open a POST-with-body SSE stream. Native EventSource only supports GET, so
 * we use fetch + a manual parser. Returns a function to abort the stream.
 */
export function postSse(
  url: string,
  body: unknown,
  onEvent: SseHandler,
  onError?: (e: unknown) => void,
): () => void {
  const ctl = new AbortController()
  ;(async () => {
    try {
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'content-type': 'application/json', accept: 'text/event-stream' },
        body: JSON.stringify(body),
        signal: ctl.signal,
      })
      if (!res.body) throw new Error('no response body')
      const reader = res.body.getReader()
      const decoder = new TextDecoder('utf-8')
      let buf = ''
      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        buf += decoder.decode(value, { stream: true })
        // Normalise CRLF (sse-starlette emits \r\n) so the split below works
        buf = buf.replace(/\r\n/g, '\n')
        let idx: number
        while ((idx = buf.indexOf('\n\n')) >= 0) {
          const block = buf.slice(0, idx).trim()
          buf = buf.slice(idx + 2)
          if (!block) continue
          let evt = 'message'
          const dataLines: string[] = []
          for (const line of block.split('\n')) {
            if (line.startsWith('event:')) {
              evt = line.slice(6).trim()
            } else if (line.startsWith('data:')) {
              // SSE spec: strip the literal "data:" prefix and AT MOST one
              // leading space. Do NOT trim — that would eat the leading/
              // trailing whitespace of streamed text tokens and collapse
              // word boundaries (e.g. " Form" -> "Form" -> "thisForm").
              let payload = line.slice(5)
              if (payload.startsWith(' ')) payload = payload.slice(1)
              dataLines.push(payload)
            }
          }
          onEvent(evt, dataLines.join('\n'))
        }
      }
    } catch (err) {
      if ((err as DOMException)?.name === 'AbortError') return
      onError?.(err)
    }
  })()
  return () => ctl.abort()
}

/**
 * Same as postSse but for multipart/form-data (file upload).
 */
export function uploadSse(
  url: string,
  form: FormData,
  onEvent: SseHandler,
  onError?: (e: unknown) => void,
): () => void {
  const ctl = new AbortController()
  ;(async () => {
    try {
      const res = await fetch(url, {
        method: 'POST',
        body: form,
        signal: ctl.signal,
      })
      if (!res.body) throw new Error('no response body')
      const reader = res.body.getReader()
      const decoder = new TextDecoder('utf-8')
      let buf = ''
      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        buf += decoder.decode(value, { stream: true })
        // Normalise CRLF (sse-starlette emits \r\n) so the split below works
        buf = buf.replace(/\r\n/g, '\n')
        let idx: number
        while ((idx = buf.indexOf('\n\n')) >= 0) {
          const block = buf.slice(0, idx).trim()
          buf = buf.slice(idx + 2)
          if (!block) continue
          let evt = 'message'
          const dataLines: string[] = []
          for (const line of block.split('\n')) {
            if (line.startsWith('event:')) {
              evt = line.slice(6).trim()
            } else if (line.startsWith('data:')) {
              // SSE spec: strip the literal "data:" prefix and AT MOST one
              // leading space. Do NOT trim — that would eat the leading/
              // trailing whitespace of streamed text tokens and collapse
              // word boundaries (e.g. " Form" -> "Form" -> "thisForm").
              let payload = line.slice(5)
              if (payload.startsWith(' ')) payload = payload.slice(1)
              dataLines.push(payload)
            }
          }
          onEvent(evt, dataLines.join('\n'))
        }
      }
    } catch (err) {
      if ((err as DOMException)?.name === 'AbortError') return
      onError?.(err)
    }
  })()
  return () => ctl.abort()
}
