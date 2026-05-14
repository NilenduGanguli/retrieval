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
  documentChunks: (name: string, limit = 100) =>
    json<{ document_name: string; chunks: Array<any> }>(
      `${BASE}/api/documents/${encodeURIComponent(name)}/chunks?limit=${limit}`,
    ),
  deleteDocument: (name: string) =>
    json<{ document_name: string; soft_deleted_chunks: number; s3_uri_removed: string | null }>(
      `${BASE}/api/documents/${encodeURIComponent(name)}`,
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
