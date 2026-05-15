// API mirror types — keep in sync with backend/app/schemas.py

export type Strategy = {
  rewrite?: boolean | null
  hyde?: boolean | null
  use_contextual?: boolean | null
  hybrid?: boolean
  rerank?: boolean | null
  mmr?: boolean
  crag?: boolean | null
  top_k?: number | null
}

export type ChunkHit = {
  chunk_id: number
  document_name?: string | null
  page_number?: number | null
  content: string
  context?: string | null
  score: number
  score_breakdown: Record<string, number>
  rank: number
}

export type LatencyMs = {
  rewrite: number
  hyde: number
  embed: number
  dense: number
  sparse: number
  fuse: number
  rerank: number
  mmr: number
  crag: number
  generate: number
  total: number
}

export type RetrieveResponse = {
  query: string
  hits: ChunkHit[]
  latency_ms: LatencyMs
  strategy_used: Record<string, unknown>
  crag_confidence?: number | null
  rewritten_queries: string[]
  hyde_text?: string | null
}

export type Citation = {
  marker: number
  chunk_id: number
  span: [number, number]
}

export type DocumentSummary = {
  document_name: string
  chunk_count: number
  total_tokens?: number | null
  first_page?: number | null
  last_page?: number | null
  latest_job_id?: string | null
  contextual_coverage?: number | null
}

export type TokenCounts = {
  prompt: number
  completion: number
  total: number
}

export type AnalyticsSummary = {
  total_queries: number
  queries_24h: number
  avg_latency_ms?: number | null
  p50_latency_ms?: number | null
  p95_latency_ms?: number | null
  p99_latency_ms?: number | null
  avg_tokens?: number | null
  top_documents: { document_name: string; appearances: number }[]
  strategy_mix: Record<string, number>
  token_totals: { prompt: number; completion: number; total: number; total_24h: number }
  stage_token_breakdown: Array<{
    stage: string
    total: number
    prompt: number
    completion: number
    uses: number
  }>
}

export type QueryLogEntry = {
  id: number
  query_text: string
  strategy: Record<string, unknown>
  latency_ms: Record<string, number>
  top_chunk_ids: number[]
  answer_text?: string | null
  crag_confidence?: number | null
  created_at: string
}

export type BackendConfig = {
  embedding_model: string
  final_gen_model: string
  rerank_model: string
  fast_model: string
  contextual_model: string
  defaults: {
    rewrite: boolean
    hyde: boolean
    rerank: boolean
    crag: boolean
    contextual: boolean
    top_k: number
    rerank_topn: number
    mmr_lambda: number
  }
  s3?: {
    enabled: boolean
    endpoint?: string | null
    bucket: string
  }
  remote_ingest?: boolean
}

export type HealthInfo = {
  ok: boolean
  chunks: number
  documents: number
  contextual_chunks: number
  embedding_dim: number
  schema: string
  table: string
}

export type GoldenQuestion = {
  id?: number
  question: string
  ground_truth_chunk_ids: number[]
  ground_truth_answer?: string | null
  tags: string[]
}

export type BenchRunResult = {
  run_id: number
  label?: string | null
  n_questions: number
  metrics: Record<string, number>
  per_question: Array<{
    id: number
    question: string
    retrieved: number[]
    ground_truth: number[]
    metrics: Record<string, number>
    elapsed_ms: number
    answer: string
  }>
}

// ───────── KYC Intelligence ──────────────────────────────────
export type KYCTaxonomy = {
  categories: Record<string, string[]>
  all_doc_types: string[]
}

export type KYCDocument = {
  id: number
  document_name: string
  owner: string | null
  document_type: string | null
  document_category: string | null
  confidence_score: number | null
  source_platform: string | null
  report_date: string | null
  classification_signals: any
  extracted_data: Record<string, any>
  s3_uri: string | null
  created_at?: string | null
}

export type KYCOwner = {
  owner: string
  owner_normalized: string
  doc_count: number
}

export type KYCDocTypeRow = {
  document_type: string
  document_category: string | null
  doc_count: number
}

export type KYCExtraction = {
  owner: string
  document_name: string
  document_type: string
  document_category?: string | null
  score: number
  s3_uri?: string | null
  data: Record<string, any>
  stored_extracted_data?: Record<string, any>
  confidence_score?: number | null
  source_platform?: string | null
  report_date?: string | null
}

export type KYCUniversalHit = {
  kyc_document_id: number | null
  owner: string
  document_name: string
  document_type: string
  document_category: string | null
  confidence_score: number | null
  source_platform: string | null
  report_date: string | null
  s3_uri: string | null
  matched_field: string
  matched_value: string
  relevance_score: number
  match_source: 'metadata' | 'vector+llm' | string
}
