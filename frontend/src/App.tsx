import * as Tabs from '@radix-ui/react-tabs'
import { useEffect, useState } from 'react'
import { Activity, BarChart3, Database, FlaskConical, MessageSquare, ShieldCheck } from 'lucide-react'

import { api } from '@/lib/api'
import { cn } from '@/lib/cn'
import type { BackendConfig, HealthInfo } from '@/types'

import CitiLogo from '@/components/CitiLogo'
import IngestionTab from '@/components/IngestionTab'
import KYCTab from '@/components/KYCTab'
import RetrievalTab from '@/components/RetrievalTab'
import AnalyticsTab from '@/components/AnalyticsTab'
import BenchmarkTab from '@/components/BenchmarkTab'

const TABS = [
  { value: 'retrieval', label: 'Retrieval', icon: MessageSquare },
  { value: 'ingestion', label: 'Ingestion', icon: Database },
  { value: 'kyc',       label: 'KYC',       icon: ShieldCheck },
  { value: 'analytics', label: 'Analytics', icon: BarChart3 },
  { value: 'benchmark', label: 'Benchmark', icon: FlaskConical },
]

export default function App() {
  const [config, setConfig] = useState<BackendConfig | null>(null)
  const [health, setHealth] = useState<HealthInfo | null>(null)
  const [tab, setTab] = useState('retrieval')

  useEffect(() => {
    let alive = true
    const load = async () => {
      try {
        const [c, h] = await Promise.all([api.config(), api.health()])
        if (!alive) return
        setConfig(c)
        setHealth(h)
      } catch (e) {
        console.warn('config/health load failed', e)
      }
    }
    load()
    const t = setInterval(load, 15000)
    return () => {
      alive = false
      clearInterval(t)
    }
  }, [])

  // Top-bar "contextualised" badge fires this event when clicked — switch
  // to the Ingestion tab so the user lands on the trigger panel.
  useEffect(() => {
    const handler = () => setTab('ingestion')
    window.addEventListener('rag:focus-contextual', handler)
    return () => window.removeEventListener('rag:focus-contextual', handler)
  }, [])

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header — white surface with strong Citi-blue accent band */}
      <header className="border-b border-line bg-white sticky top-0 z-20 shadow-cardLg">
        <div className="max-w-[1600px] mx-auto px-6 py-3 flex items-center gap-6">
          <div className="flex items-center gap-3 pr-4 border-r-2 border-line">
            <CitiLogo size={32} />
          </div>
          <div className="flex items-center gap-2">
            <span className="font-semibold tracking-tight text-citi-blue text-base">
              RAG <span className="heading-glow">Studio</span>
            </span>
            <span className="ml-2 text-xs text-citi-blue hidden sm:inline">
              hybrid · contextual · listwise rerank · CRAG
            </span>
          </div>

          <Tabs.Root value={tab} onValueChange={setTab} className="flex-1">
            <Tabs.List className="inline-flex gap-1 p-1 rounded-xl bg-slate-100 border border-line">
              {TABS.map(t => {
                const Icon = t.icon
                return (
                  <Tabs.Trigger
                    key={t.value}
                    value={t.value}
                    className={cn(
                      'inline-flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium transition',
                      'text-ink hover:text-citi-blue hover:bg-white',
                      'data-[state=active]:bg-accent data-[state=active]:text-white data-[state=active]:shadow-glow',
                    )}
                  >
                    <Icon className="w-4 h-4" />
                    <span>{t.label}</span>
                  </Tabs.Trigger>
                )
              })}
            </Tabs.List>
          </Tabs.Root>

          <div className="hidden md:flex items-center gap-3 text-xs">
            <HealthBadge health={health} />
          </div>
        </div>
        {/* Heavy Citi-blue accent band beneath the bar */}
        <div className="h-1 bg-gradient-to-r from-citi-blue via-accent to-accent-soft" />
      </header>

      <main className="flex-1 max-w-[1600px] mx-auto w-full px-6 py-6">
        {/* forceMount keeps each tab's React subtree mounted across switches so
            ingest progress, KYC search results, benchmark runs etc. don't get
            torn down when the user navigates away. Inactive panels are hidden
            via the data-[state=inactive]:hidden utility. */}
        <Tabs.Root value={tab} onValueChange={setTab}>
          <Tabs.Content
            value="retrieval"
            forceMount
            className="outline-none data-[state=inactive]:hidden"
          >
            <RetrievalTab config={config} />
          </Tabs.Content>
          <Tabs.Content
            value="ingestion"
            forceMount
            className="outline-none data-[state=inactive]:hidden"
          >
            <IngestionTab health={health} onChange={() => api.health().then(setHealth)} />
          </Tabs.Content>
          <Tabs.Content
            value="kyc"
            forceMount
            className="outline-none data-[state=inactive]:hidden"
          >
            <KYCTab />
          </Tabs.Content>
          <Tabs.Content
            value="analytics"
            forceMount
            className="outline-none data-[state=inactive]:hidden"
          >
            <AnalyticsTab />
          </Tabs.Content>
          <Tabs.Content
            value="benchmark"
            forceMount
            className="outline-none data-[state=inactive]:hidden"
          >
            <BenchmarkTab config={config} />
          </Tabs.Content>
        </Tabs.Root>
      </main>

      <footer className="border-t border-line text-[11px] text-ink px-6 py-2 flex items-center justify-between max-w-[1600px] mx-auto w-full bg-white">
        <span>
          {config ? (
            <>
              <span className="text-citi-blue font-semibold">emb:</span>{' '}
              <span className="text-ink">{config.embedding_model}</span> ·{' '}
              <span className="text-citi-blue font-semibold">gen:</span>{' '}
              <span className="text-ink">{config.final_gen_model}</span> ·{' '}
              <span className="text-citi-blue font-semibold">rerank:</span>{' '}
              <span className="text-ink">{config.rerank_model}</span>
            </>
          ) : (
            'loading…'
          )}
        </span>
        <span>
          {health
            ? `${health.documents} docs · ${health.chunks.toLocaleString()} chunks · dim ${health.embedding_dim}`
            : ''}
        </span>
      </footer>
    </div>
  )
}

function HealthBadge({ health }: { health: HealthInfo | null }) {
  if (!health) return <span className="chip">disconnected</span>
  const pct = health.chunks > 0
    ? Math.round((health.contextual_chunks / health.chunks) * 100)
    : 0
  const fullyDone = health.chunks > 0 && health.contextual_chunks === health.chunks
  return (
    <>
      <span className="chip-success">
        <Activity className="w-3 h-3" />
        live
      </span>
      <button
        type="button"
        onClick={() => {
          // Jump to the Ingestion tab and scroll to the contextualisation panel.
          // The custom event handler in IngestionTab listens for `rag:focus-contextual`.
          window.dispatchEvent(new CustomEvent('rag:focus-contextual'))
        }}
        title={
          fullyDone
            ? 'All chunks contextualised'
            : `Click to run Anthropic Contextual Retrieval on the remaining ${
                health.chunks - health.contextual_chunks
              } chunks`
        }
        className={cn(
          'chip transition hover:border-accent hover:text-accent-dark hover:shadow-glow cursor-pointer',
          fullyDone && 'border-emerald-600 text-emerald-800 bg-emerald-500/15',
        )}
      >
        {health.contextual_chunks}/{health.chunks} contextualised
        {!fullyDone && health.chunks > 0 && (
          <span className="ml-1 text-[10px] opacity-80">({pct}%)</span>
        )}
      </button>
    </>
  )
}
