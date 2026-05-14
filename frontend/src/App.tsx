import * as Tabs from '@radix-ui/react-tabs'
import { useEffect, useState } from 'react'
import { Activity, BarChart3, Database, FlaskConical, MessageSquare, Sparkles } from 'lucide-react'

import { api } from '@/lib/api'
import { cn } from '@/lib/cn'
import type { BackendConfig, HealthInfo } from '@/types'

import IngestionTab from '@/components/IngestionTab'
import RetrievalTab from '@/components/RetrievalTab'
import AnalyticsTab from '@/components/AnalyticsTab'
import BenchmarkTab from '@/components/BenchmarkTab'

const TABS = [
  { value: 'retrieval', label: 'Retrieval', icon: MessageSquare },
  { value: 'ingestion', label: 'Ingestion', icon: Database },
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

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="border-b border-line/80 backdrop-blur sticky top-0 z-20 bg-bg/70">
        <div className="max-w-[1600px] mx-auto px-6 py-3 flex items-center gap-6">
          <div className="flex items-center gap-2">
            <div className="relative">
              <Sparkles className="w-5 h-5 text-accent" />
              <span className="absolute -inset-1 rounded-full bg-accent/20 blur-md" />
            </div>
            <span className="font-semibold tracking-tight text-zinc-100">
              RAG <span className="heading-glow">Studio</span>
            </span>
            <span className="ml-2 text-xs text-muted hidden sm:inline">
              hybrid · contextual · listwise rerank · CRAG
            </span>
          </div>

          <Tabs.Root value={tab} onValueChange={setTab} className="flex-1">
            <Tabs.List className="inline-flex gap-1 p-1 rounded-xl bg-bg-soft border border-line">
              {TABS.map(t => {
                const Icon = t.icon
                return (
                  <Tabs.Trigger
                    key={t.value}
                    value={t.value}
                    className={cn(
                      'inline-flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm transition',
                      'text-zinc-400 hover:text-zinc-100',
                      'data-[state=active]:bg-bg-card data-[state=active]:text-zinc-100 data-[state=active]:shadow-glow',
                    )}
                  >
                    <Icon className="w-4 h-4" />
                    <span>{t.label}</span>
                  </Tabs.Trigger>
                )
              })}
            </Tabs.List>
          </Tabs.Root>

          <div className="hidden md:flex items-center gap-3 text-xs text-muted">
            <HealthBadge health={health} />
          </div>
        </div>
      </header>

      <main className="flex-1 max-w-[1600px] mx-auto w-full px-6 py-6">
        <Tabs.Root value={tab} onValueChange={setTab}>
          <Tabs.Content value="retrieval" className="outline-none">
            <RetrievalTab config={config} />
          </Tabs.Content>
          <Tabs.Content value="ingestion" className="outline-none">
            <IngestionTab health={health} onChange={() => api.health().then(setHealth)} />
          </Tabs.Content>
          <Tabs.Content value="analytics" className="outline-none">
            <AnalyticsTab />
          </Tabs.Content>
          <Tabs.Content value="benchmark" className="outline-none">
            <BenchmarkTab config={config} />
          </Tabs.Content>
        </Tabs.Root>
      </main>

      <footer className="border-t border-line/60 text-[11px] text-muted px-6 py-2 flex items-center justify-between max-w-[1600px] mx-auto w-full">
        <span>
          {config ? (
            <>
              <span className="text-zinc-400">emb:</span> {config.embedding_model} ·{' '}
              <span className="text-zinc-400">gen:</span> {config.final_gen_model} ·{' '}
              <span className="text-zinc-400">rerank:</span> {config.rerank_model}
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
  return (
    <>
      <span className="chip-success">
        <Activity className="w-3 h-3" />
        live
      </span>
      <span className="chip">
        {health.contextual_chunks}/{health.chunks} contextualised
      </span>
    </>
  )
}
