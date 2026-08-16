import { useState, useEffect, useRef, useCallback } from 'react'
import useSWR from 'swr'
import {
  fetchRepos,
  fetchRepoStats,
  fetchDependencies,
  fetchRepoInstruction,
  saveRepoInstruction,
  getWsBase,
  INDEXING_PROGRESS_WS_PATH,
  type StoredIndexedRepo,
  type IndexStats,
  type StoredDependency,
  type IndexingProgress,
  type InstructionResponse,
} from '../lib/api'
import { PageHeader } from '../components/layout/page-header'
import { BlockSkeleton, StatsGridSkeleton } from '../components/shared/page-skeletons'
import { StatsCard } from '../components/shared/stats-card'
import { TimeAgo } from '../components/shared/time-ago'
import { DataTable, type Column } from '../components/shared/data-table'
import { Modal } from '../components/shared/modal'
import { Tabs } from '../components/ui/tabs'
import { Card, CardContent } from '../components/ui/card'
import { formatNumber, formatDate } from '../lib/formatters'
import { Database, FileText, Clock, ExternalLink, Loader2, GraduationCap, Save, Bot, CheckCircle2 } from 'lucide-react'
import { useRouter } from '../router'

const tabItems = [
  { value: 'repos', label: 'Repositories' },
  { value: 'deps', label: 'Dependencies' },
]

function IndexingProgressBar({ progress }: { progress: IndexingProgress }) {
  const pct =
    progress.total_repos > 0
      ? Math.round((progress.indexed_repos / progress.total_repos) * 100)
      : 0

  return (
    <Card className="border-primary/20 bg-primary/5">
      <CardContent className="p-4 space-y-3">
        <div className="flex items-center gap-2">
          <Loader2 className="h-4 w-4 text-primary animate-spin" />
          <span className="text-sm font-medium">Indexing in progress</span>
          <span className="text-sm text-muted-foreground ml-auto">
            {progress.indexed_repos} / {progress.total_repos} repos ({pct}%)
          </span>
        </div>

        <div className="h-2 bg-primary/10 rounded-full overflow-hidden">
          <div
            className="h-full bg-primary rounded-full transition-all duration-500 ease-out"
            style={{ width: `${pct}%` }}
          />
        </div>

        <div className="flex items-center justify-between text-xs text-muted-foreground">
          {progress.current_repo && (
            <span>
              Current: <span className="font-mono font-medium text-foreground">{progress.current_repo}</span>
              {progress.current_repo_files > 0 && (
                <span className="ml-1">({formatNumber(progress.current_repo_files)} files)</span>
              )}
            </span>
          )}
          <span>{formatNumber(progress.total_files_indexed)} files indexed so far</span>
        </div>
      </CardContent>
    </Card>
  )
}

function useIndexingProgress() {
  const [progress, setProgress] = useState<IndexingProgress | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimer = useRef<ReturnType<typeof setTimeout>>(undefined)
  const mountedRef = useRef(true)
  const lastStatusRef = useRef<string | null>(null)

  const connect = useCallback(() => {
    if (!mountedRef.current) return
    const ws = new WebSocket(`${getWsBase()}${INDEXING_PROGRESS_WS_PATH}`)
    wsRef.current = ws

    ws.onmessage = (event) => {
      try {
        const data: IndexingProgress = JSON.parse(event.data)
        lastStatusRef.current = data.status
        setProgress(data)
      } catch { /* ignore malformed messages */ }
    }

    ws.onclose = () => {
      wsRef.current = null
      if (!mountedRef.current) return
      // Only reconnect if the last known status was "running" or we never
      // received data (null). When idle, there is nothing to stream.
      if (lastStatusRef.current === 'running' || lastStatusRef.current === null) {
        reconnectTimer.current = setTimeout(connect, 3000)
      }
    }

    ws.onerror = () => ws.close()
  }, [])

  useEffect(() => {
    mountedRef.current = true
    connect()
    return () => {
      mountedRef.current = false
      clearTimeout(reconnectTimer.current)
      wsRef.current?.close()
    }
  }, [connect])

  return progress
}

function RepoInstructionsEditor({ repo }: { repo: string }) {
  const { data, mutate } = useSWR<InstructionResponse>(
    ['repo-instruction', repo],
    () => fetchRepoInstruction(repo),
  )
  const [draft, setDraft] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)

  const serverText = data?.text ?? ''
  const text = draft ?? serverText
  const dirty = text !== serverText

  const handleSave = useCallback(async () => {
    setSaving(true)
    try {
      await saveRepoInstruction(repo, text)
      await mutate()
      setDraft(null)
      setSaved(true)
      setTimeout(() => setSaved(false), 3000)
    } finally {
      setSaving(false)
    }
  }, [repo, text, mutate])

  return (
    <div className="space-y-2 border-t pt-4">
      <div className="flex items-center gap-2">
        <Bot className="h-4 w-4 text-muted-foreground" />
        <p className="text-sm font-medium">Agent Instructions</p>
      </div>
      <p className="text-xs text-muted-foreground">
        Prepended to the agent's context for this repo, after the global instructions.
        Describe the repo's role (e.g. generated output vs source). Never overwrites the
        repo's own AGENTS.md.
      </p>
      <textarea
        value={text}
        onChange={e => setDraft(e.target.value)}
        placeholder="e.g. This repo holds generated SDK output. Do not edit here; fix the sdk-generator repo instead."
        className="w-full font-mono text-xs bg-muted/50 border rounded-md p-3 min-h-[120px] resize-y focus:outline-none focus:ring-2 focus:ring-primary/30"
        spellCheck={false}
      />
      <div className="flex items-center gap-3">
        <button
          onClick={handleSave}
          disabled={saving || !dirty}
          className="inline-flex items-center gap-1.5 rounded-md bg-green-600 text-white px-3 py-1.5 text-sm font-medium hover:bg-green-700 disabled:opacity-40 disabled:pointer-events-none transition-colors"
        >
          <Save className="h-4 w-4" />
          {saving ? 'Saving...' : 'Save'}
        </button>
        {saved && (
          <span className="text-xs text-green-600 inline-flex items-center gap-1">
            <CheckCircle2 className="h-3.5 w-3.5" /> Saved
          </span>
        )}
        {dirty && !saved && <span className="text-xs text-amber-600">Unsaved changes</span>}
      </div>
    </div>
  )
}

export default function ReposPage() {
  const { navigate } = useRouter()
  const [selectedRepo, setSelectedRepo] = useState<StoredIndexedRepo | null>(null)

  const { data: repos, isLoading: reposLoading } = useSWR<StoredIndexedRepo[]>(
    'repos',
    fetchRepos,
    { refreshInterval: 30000 },
  )

  const { data: stats, isLoading: statsLoading } = useSWR<IndexStats>(
    'repo-stats',
    fetchRepoStats,
    { refreshInterval: 30000 },
  )

  const { data: deps, isLoading: depsLoading } = useSWR<StoredDependency[]>(
    'dependencies',
    fetchDependencies,
    { refreshInterval: 60000 },
  )

  const reposData = Array.isArray(repos) ? repos : null
  const depsData = Array.isArray(deps) ? deps : null
  const statsData =
    stats &&
    typeof stats === 'object' &&
    typeof (stats as Partial<IndexStats>).repo_count === 'number' &&
    typeof (stats as Partial<IndexStats>).file_count === 'number'
      ? stats
      : null

  const indexingProgress = useIndexingProgress()

  const isIndexing = indexingProgress?.status === 'running'

  const repoColumns: Column<StoredIndexedRepo>[] = [
    {
      key: 'name',
      header: 'Name',
      render: row => <span className="font-medium">{row.name}</span>,
    },
    {
      key: 'path',
      header: 'Path',
      render: row => (
        <span className="text-sm font-mono text-muted-foreground">{row.path}</span>
      ),
    },
    {
      key: 'scm_url',
      header: 'GitHub URL',
      render: row =>
        row.scm_url ? (
          <a
            href={row.scm_url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-primary hover:underline inline-flex items-center gap-1 text-sm"
            onClick={e => e.stopPropagation()}
          >
            <ExternalLink className="h-3 w-3" />
            Link
          </a>
        ) : (
          <span className="text-muted-foreground text-sm">--</span>
        ),
    },
    {
      key: 'default_branch',
      header: 'Default Branch',
      render: row => <span className="text-sm">{row.default_branch}</span>,
    },
    {
      key: 'file_count',
      header: 'File Count',
      render: row => <span className="text-sm">{formatNumber(row.file_count)}</span>,
      sortable: true,
    },
    {
      key: 'last_indexed_at',
      header: 'Last Indexed',
      render: row => <TimeAgo date={row.last_indexed_at} />,
      sortable: true,
    },
  ]

  const depColumns: Column<StoredDependency>[] = [
    {
      key: 'upstream',
      header: 'Upstream',
      render: row => <span className="font-medium text-sm">{row.upstream}</span>,
    },
    {
      key: 'downstream',
      header: 'Downstream',
      render: row => <span className="font-medium text-sm">{row.downstream}</span>,
    },
    {
      key: 'dep_type',
      header: 'Type',
      render: row => <span className="text-sm capitalize">{row.dep_type}</span>,
    },
  ]

  return (
    <div className="space-y-6">
      <PageHeader title="Repositories" description="Repository index and dependencies" />

      {isIndexing && <IndexingProgressBar progress={indexingProgress!} />}

      {statsLoading && (
        <StatsGridSkeleton count={3} className="md:grid-cols-3" />
      )}

      {statsData && (
        <div className="grid gap-4 md:grid-cols-3">
          <StatsCard
            title="Total Repos"
            value={statsData.repo_count}
            icon={<Database className="h-4 w-4 text-primary" />}
            description="Indexed repositories"
          />
          <StatsCard
            title="Files Indexed"
            value={formatNumber(statsData.file_count)}
            icon={<FileText className="h-4 w-4 text-green-500" />}
            description="Total indexed files"
          />
          <StatsCard
            title="Last Indexed At"
            value={
              statsData.last_indexed_at
                ? formatDate(statsData.last_indexed_at)
                : '--'
            }
            icon={<Clock className="h-4 w-4 text-yellow-500" />}
            description="Most recent index run"
          />
        </div>
      )}

      <Tabs tabs={tabItems} defaultTab="repos">
        {activeTab =>
          activeTab === 'repos' ? (
            <>
              {reposLoading && <BlockSkeleton className="h-48 w-full" />}
              {reposData && (
                <DataTable
                  columns={repoColumns}
                  data={reposData}
                  keyFn={row => row.id}
                  emptyMessage="No repositories indexed yet"
                  onRowClick={row => setSelectedRepo(row)}
                />
              )}
            </>
          ) : (
            <>
              {depsLoading && <BlockSkeleton className="h-32 w-full" />}
              {depsData && (
                <DataTable
                  columns={depColumns}
                  data={depsData}
                  keyFn={row => row.id}
                  emptyMessage="No dependencies found. Dependencies are discovered when indexed repos import or depend on each other."
                />
              )}
            </>
          )
        }
      </Tabs>

      <Modal
        open={!!selectedRepo}
        onClose={() => setSelectedRepo(null)}
        title={selectedRepo?.name}
      >
        {selectedRepo && (
          <div className="space-y-4">
            <div className="grid gap-3 sm:grid-cols-2">
              <div>
                <p className="text-sm text-muted-foreground">Path</p>
                <p className="text-sm font-mono">{selectedRepo.path}</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Default Branch</p>
                <p className="text-sm">{selectedRepo.default_branch}</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">File Count</p>
                <p className="text-sm">{selectedRepo.file_count.toLocaleString()}</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">GitHub URL</p>
                {selectedRepo.scm_url ? (
                  <a
                    href={selectedRepo.scm_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-sm text-primary hover:underline inline-flex items-center gap-1"
                  >
                    <ExternalLink className="h-3 w-3" />
                    {selectedRepo.scm_url}
                  </a>
                ) : (
                  <p className="text-sm text-muted-foreground">--</p>
                )}
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Last Indexed</p>
                <p className="text-sm">{formatDate(selectedRepo.last_indexed_at)}</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Created</p>
                <p className="text-sm">{formatDate(selectedRepo.created_at)}</p>
              </div>
            </div>
            <button
              onClick={() => {
                setSelectedRepo(null)
                navigate(`/learning?repo=${encodeURIComponent(selectedRepo.name)}`)
              }}
              className="inline-flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-sm font-medium hover:bg-muted transition-colors"
            >
              <GraduationCap className="h-4 w-4" />
              View Learning
            </button>

            <RepoInstructionsEditor repo={selectedRepo.name} />
          </div>
        )}
      </Modal>
    </div>
  )
}
