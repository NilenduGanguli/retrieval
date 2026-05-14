import { Download, Eye, FileText, HardDrive, X } from 'lucide-react'
import { useMemo, useState } from 'react'

import type { ChunkHit } from '@/types'

type Props = {
  hits: ChunkHit[]
  s3Enabled: boolean
}

/**
 * Lists the unique source documents that appear in the latest retrieval
 * result. Provides View (inline preview) + Download buttons.
 *
 * View opens an in-page overlay with an iframe pointed at the backend's
 * /api/documents/<name>/view endpoint (which streams the PDF from S3 with
 * an `inline` Content-Disposition).
 */
export default function PersistedDocuments({ hits, s3Enabled }: Props) {
  const [preview, setPreview] = useState<string | null>(null)

  const docs = useMemo(() => {
    const map = new Map<string, { name: string; pages: Set<number>; chunks: number }>()
    for (const h of hits) {
      if (!h.document_name) continue
      const cur = map.get(h.document_name) ?? {
        name: h.document_name,
        pages: new Set<number>(),
        chunks: 0,
      }
      if (h.page_number != null) cur.pages.add(h.page_number)
      cur.chunks += 1
      map.set(h.document_name, cur)
    }
    return Array.from(map.values()).sort((a, b) => b.chunks - a.chunks)
  }, [hits])

  if (docs.length === 0) {
    return (
      <div className="card-soft p-3 text-[11px] space-y-1 text-muted">
        <div className="flex items-center gap-1.5">
          <HardDrive className="w-3 h-3" />
          <span>persisted documents</span>
        </div>
        <div className="text-muted/80 italic">
          Run a retrieval to see the source documents here.
        </div>
      </div>
    )
  }

  return (
    <>
      <div className="card-soft p-3 text-[11px] space-y-2">
        <div className="flex items-center gap-1.5 text-muted">
          <HardDrive className="w-3 h-3" />
          <span>persisted documents</span>
          <span className="chip ml-auto">{docs.length}</span>
        </div>
        <ul className="space-y-1.5">
          {docs.map((d) => (
            <li key={d.name} className="rounded-md border border-line/60 bg-bg-soft/60 px-2 py-1.5">
              <div className="flex items-center gap-1.5 text-zinc-200">
                <FileText className="w-3 h-3 text-accent shrink-0" />
                <span className="truncate text-[12px]" title={d.name}>{d.name}</span>
              </div>
              <div className="flex items-center gap-2 mt-1 text-muted/90 text-[10px]">
                <span>{d.chunks} chunk{d.chunks > 1 ? 's' : ''}</span>
                {d.pages.size > 0 && <span>p. {[...d.pages].sort((a, b) => a - b).join(', ').slice(0, 30)}</span>}
              </div>
              <div className="flex gap-1.5 mt-1.5">
                <button
                  type="button"
                  onClick={() => setPreview(d.name)}
                  disabled={!s3Enabled}
                  className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px]
                             border border-line/60 hover:border-accent/60 hover:text-accent
                             disabled:opacity-40 disabled:cursor-not-allowed transition"
                  title={s3Enabled ? 'Preview inline' : 'S3 is disabled'}
                >
                  <Eye className="w-3 h-3" /> View
                </button>
                <a
                  href={s3Enabled ? `/api/documents/${encodeURIComponent(d.name)}/download` : undefined}
                  onClick={(e) => { if (!s3Enabled) e.preventDefault() }}
                  className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px]
                             border border-line/60 hover:border-accent/60 hover:text-accent transition
                             ${!s3Enabled ? 'opacity-40 cursor-not-allowed' : ''}`}
                  title={s3Enabled ? 'Download original' : 'S3 is disabled'}
                >
                  <Download className="w-3 h-3" /> Download
                </a>
              </div>
            </li>
          ))}
        </ul>
        {!s3Enabled && (
          <div className="text-[10px] text-muted/80 italic pt-1 border-t border-line/40">
            S3 disabled — set <code className="text-accent-soft">S3_ENABLED=true</code> in .env to enable view/download.
          </div>
        )}
      </div>

      {preview && (
        <DocumentPreviewModal
          name={preview}
          onClose={() => setPreview(null)}
        />
      )}
    </>
  )
}

function DocumentPreviewModal({ name, onClose }: { name: string; onClose: () => void }) {
  const src = `/api/documents/${encodeURIComponent(name)}/view`
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
          <div className="flex items-center gap-2 text-sm">
            <FileText className="w-4 h-4 text-accent" />
            <span className="font-medium truncate" title={name}>{name}</span>
          </div>
          <div className="flex items-center gap-2">
            <a
              href={`/api/documents/${encodeURIComponent(name)}/download`}
              className="text-xs px-2 py-1 rounded border border-line hover:border-accent/60 hover:text-accent transition inline-flex items-center gap-1"
            >
              <Download className="w-3 h-3" /> Download
            </a>
            <button
              type="button"
              onClick={onClose}
              className="p-1 rounded hover:bg-bg-soft text-muted hover:text-zinc-100 transition"
              aria-label="Close preview"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>
        <iframe
          src={src}
          title={`Preview of ${name}`}
          className="flex-1 w-full bg-zinc-900"
        />
      </div>
    </div>
  )
}
