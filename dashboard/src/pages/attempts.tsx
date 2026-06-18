import { useState } from 'react'
import useSWR from 'swr'
import {
  fetchAttempts,
  fetchAttemptDetail,
  fetchAttemptExecutionLog,
  type AttemptsResponse,
  type AttemptDetailResponse,
  type AttemptExecutionLogResponse,
} from '../lib/api'
import { PageHeader } from '../components/layout/page-header'
import { Select } from '../components/ui/select'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../components/ui/card'
import { Badge } from '../components/ui/badge'
import { BlockSkeleton, TableRowsSkeleton } from '../components/shared/page-skeletons'
import { DataTable, type Column } from '../components/shared/data-table'
import { StatusBadge } from '../components/shared/status-badge'
import { TimeAgo } from '../components/shared/time-ago'
import { ExternalLink, ChevronLeft, ChevronRight } from 'lucide-react'

const statusOptions = [
  { value: 'pending', label: 'Pending' },
  { value: 'success', label: 'Success' },
  { value: 'failed', label: 'Failed' },
  { value: 'merged', label: 'Merged' },
  { value: 'closed', label: 'Closed' },
  { value: 'cannot_fix', label: 'Cannot Fix' },
]

const sourceOptions = [
  { value: 'sentry', label: 'Sentry' },
  { value: 'linear', label: 'Linear' },
  { value: 'jira', label: 'Jira' },
  { value: 'github', label: 'GitHub' },
]

interface AttemptRow {
  id: number
  short_id: string
  source: string
  status: string
  pr_url: string | null
  attempted_at: string
  retry_count: number
}

type LogStream = 'stdout' | 'stderr' | 'events'

interface SelectedExecutionLog {
  executionId: number
  stream: LogStream
}

export default function AttemptsPage() {
  const [page, setPage] = useState(1)
  const [statusFilter, setStatusFilter] = useState('')
  const [sourceFilter, setSourceFilter] = useState('')
  const [selectedId, setSelectedId] = useState<number | null>(null)
  const [selectedLog, setSelectedLog] = useState<SelectedExecutionLog | null>(null)

  const { data, error, isLoading } = useSWR<AttemptsResponse>(
    ['attempts', page, statusFilter, sourceFilter],
    () =>
      fetchAttempts({
        page,
        per_page: 20,
        status: statusFilter || undefined,
        source: sourceFilter || undefined,
      }),
    { refreshInterval: 15000 },
  )

  const { data: detail, isLoading: detailLoading } = useSWR<AttemptDetailResponse>(
    selectedId ? ['attempt-detail', selectedId] : null,
    () => fetchAttemptDetail(selectedId!),
  )

  const {
    data: executionLog,
    error: executionLogError,
    isLoading: executionLogLoading,
  } = useSWR<AttemptExecutionLogResponse>(
    selectedId && selectedLog
      ? ['attempt-exec-log', selectedId, selectedLog.executionId, selectedLog.stream]
      : null,
    () =>
      fetchAttemptExecutionLog(
        selectedId!,
        selectedLog!.executionId,
        selectedLog!.stream,
      ),
  )

  const columns: Column<AttemptRow>[] = [
    {
      key: 'status',
      header: 'Status',
      render: row => <StatusBadge status={row.status} />,
    },
    {
      key: 'short_id',
      header: 'ID',
      render: row => (
        <button
          onClick={() => {
            setSelectedId(row.id === selectedId ? null : row.id)
            setSelectedLog(null)
          }}
          className="font-mono text-sm text-primary hover:underline"
        >
          {row.short_id}
        </button>
      ),
    },
    {
      key: 'source',
      header: 'Source',
      render: row => <span className="capitalize">{row.source}</span>,
    },
    {
      key: 'pr_url',
      header: 'PR',
      render: row =>
        row.pr_url ? (
          <a
            href={row.pr_url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-primary hover:text-primary/80 inline-flex items-center gap-1"
          >
            <ExternalLink className="h-3.5 w-3.5" />
            Link
          </a>
        ) : (
          <span className="text-muted-foreground">--</span>
        ),
    },
    {
      key: 'attempted_at',
      header: 'Time',
      render: row => <TimeAgo date={row.attempted_at} />,
      sortable: true,
    },
    {
      key: 'retry_count',
      header: 'Retries',
      render: row => <span>{row.retry_count}</span>,
    },
  ]

  const totalPages = data ? Math.ceil(data.total / data.per_page) : 0

  return (
    <div className="space-y-6">
      <PageHeader title="Attempts" description="Fix attempt history and details" />

      <div className="flex items-center gap-4">
        <Select
          value={statusFilter}
          onChange={v => {
            setStatusFilter(v)
            setPage(1)
          }}
          options={statusOptions}
          placeholder="All statuses"
          className="w-40"
        />
        <Select
          value={sourceFilter}
          onChange={v => {
            setSourceFilter(v)
            setPage(1)
          }}
          options={sourceOptions}
          placeholder="All sources"
          className="w-40"
        />
      </div>

      {error && (
        <div className="text-destructive text-sm">Failed to load attempts.</div>
      )}

      {isLoading && (
        <TableRowsSkeleton rows={6} />
      )}

      {data && (
        <>
          <Card>
            <CardContent className="p-4">
              <DataTable
                columns={columns}
                data={data.attempts}
                keyFn={row => row.id}
                emptyMessage="No attempts found"
                pageSize={0}
              />
            </CardContent>
          </Card>

          <div className="flex items-center justify-between">
            <p className="text-sm text-muted-foreground">
              Page {data.page} of {totalPages} ({data.total} total)
            </p>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setPage(p => Math.max(1, p - 1))}
                disabled={page <= 1}
                className="inline-flex items-center gap-1 px-3 py-1.5 text-sm border rounded-md hover:bg-muted disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <ChevronLeft className="h-4 w-4" />
                Previous
              </button>
              <button
                onClick={() => setPage(p => Math.min(totalPages, p + 1))}
                disabled={page >= totalPages}
                className="inline-flex items-center gap-1 px-3 py-1.5 text-sm border rounded-md hover:bg-muted disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Next
                <ChevronRight className="h-4 w-4" />
              </button>
            </div>
          </div>
        </>
      )}

      {selectedId && (
        <div className="space-y-4">
          {detailLoading && <BlockSkeleton className="h-48 w-full" />}

          {detail && (
            <>
              <Card>
                <CardHeader>
                  <CardTitle>Attempt Detail: {detail.attempt.short_id}</CardTitle>
                  <CardDescription>
                    {detail.attempt.source} - {detail.attempt.issue_id}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                    <div>
                      <p className="text-sm text-muted-foreground">Status</p>
                      <StatusBadge status={detail.attempt.status} />
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">Attempted At</p>
                      <TimeAgo date={detail.attempt.attempted_at} />
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">Repo</p>
                      <p className="text-sm">{detail.attempt.scm_repo || '--'}</p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">PR URL</p>
                      {detail.attempt.pr_url ? (
                        <a
                          href={detail.attempt.pr_url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-sm text-primary hover:underline inline-flex items-center gap-1"
                        >
                          <ExternalLink className="h-3.5 w-3.5" />
                          {detail.attempt.pr_url}
                        </a>
                      ) : (
                        <p className="text-sm text-muted-foreground">--</p>
                      )}
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">Retries</p>
                      <p className="text-sm">{detail.attempt.retry_count}</p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">Cascade Repo</p>
                      <p className="text-sm">{detail.attempt.cascade_repo || '--'}</p>
                    </div>
                    {detail.attempt.error_message && (
                      <div className="sm:col-span-2 lg:col-span-3">
                        <p className="text-sm text-muted-foreground">Error Message</p>
                        <p className="text-sm text-destructive">{detail.attempt.error_message}</p>
                      </div>
                    )}
                    {detail.attempt.issue_labels && detail.attempt.issue_labels.length > 0 && (
                      <div className="sm:col-span-2 lg:col-span-3">
                        <p className="text-sm text-muted-foreground mb-1">Labels</p>
                        <div className="flex flex-wrap gap-1">
                          {detail.attempt.issue_labels.map(label => (
                            <Badge key={label} variant="secondary">
                              {label}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>

              {detail.executions.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle>Executions</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="border-b">
                            <th className="text-left py-2 font-medium">Started</th>
                            <th className="text-left py-2 font-medium">Duration</th>
                            <th className="text-left py-2 font-medium">Exit Code</th>
                            <th className="text-left py-2 font-medium">Files Changed</th>
                            <th className="text-left py-2 font-medium">Lines +/-</th>
                            <th className="text-left py-2 font-medium">Logs</th>
                          </tr>
                        </thead>
                        <tbody>
                          {detail.executions.map(exec => (
                            <tr key={exec.id} className="border-b last:border-0">
                              <td className="py-2">
                                <TimeAgo date={exec.started_at} />
                              </td>
                              <td className="py-2">
                                {exec.duration_secs != null
                                  ? `${exec.duration_secs.toFixed(1)}s`
                                  : '--'}
                              </td>
                              <td className="py-2">
                                <span
                                  className={
                                    exec.exit_code === 0
                                      ? 'text-green-600'
                                      : 'text-red-600'
                                  }
                                >
                                  {exec.exit_code ?? '--'}
                                </span>
                              </td>
                              <td className="py-2">{exec.files_changed ?? '--'}</td>
                              <td className="py-2">
                                {exec.lines_added != null || exec.lines_removed != null
                                  ? `+${exec.lines_added ?? 0} / -${exec.lines_removed ?? 0}`
                                  : '--'}
                              </td>
                              <td className="py-2">
                                <div className="flex gap-2">
                                  <button
                                    onClick={() =>
                                      setSelectedLog({
                                        executionId: exec.id,
                                        stream: 'stdout',
                                      })
                                    }
                                    disabled={!exec.stdout_log_path && !exec.stdout_preview}
                                    className="text-xs px-2 py-1 border rounded hover:bg-muted disabled:opacity-40"
                                  >
                                    stdout
                                  </button>
                                  <button
                                    onClick={() =>
                                      setSelectedLog({
                                        executionId: exec.id,
                                        stream: 'stderr',
                                      })
                                    }
                                    disabled={!exec.stderr_log_path && !exec.stderr_preview}
                                    className="text-xs px-2 py-1 border rounded hover:bg-muted disabled:opacity-40"
                                  >
                                    stderr
                                  </button>
                                  <button
                                    onClick={() =>
                                      setSelectedLog({
                                        executionId: exec.id,
                                        stream: 'events',
                                      })
                                    }
                                    disabled={!exec.event_log_path}
                                    className="text-xs px-2 py-1 border rounded hover:bg-muted disabled:opacity-40"
                                  >
                                    events
                                  </button>
                                </div>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </CardContent>
                </Card>
              )}

              {selectedLog && (
                <Card>
                  <CardHeader>
                    <CardTitle>
                      Execution Log ({selectedLog.stream}) - #{selectedLog.executionId}
                    </CardTitle>
                    {executionLog?.path && (
                      <CardDescription>{executionLog.path}</CardDescription>
                    )}
                  </CardHeader>
                  <CardContent>
                    {executionLogLoading && <BlockSkeleton className="h-48 w-full" />}
                    {executionLogError && (
                      <p className="text-sm text-destructive">
                        Failed to load execution log.
                      </p>
                    )}
                    {executionLog && (
                      <div className="space-y-2">
                        {executionLog.truncated && (
                          <p className="text-xs text-muted-foreground">
                            Showing the most recent log content (truncated).
                          </p>
                        )}
                        <pre className="text-xs bg-muted rounded-md p-3 overflow-x-auto whitespace-pre-wrap break-words max-h-[480px] overflow-y-auto">
                          {executionLog.content || 'No log output captured.'}
                        </pre>
                      </div>
                    )}
                  </CardContent>
                </Card>
              )}

              {detail.reviews.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle>Reviews</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {detail.reviews.map(review => (
                        <div
                          key={review.id}
                          className="border rounded-md p-3"
                        >
                          <div className="flex items-center gap-2 mb-1">
                            <span className="font-medium text-sm">
                              {review.reviewer || 'Unknown'}
                            </span>
                            {review.review_state && (
                              <StatusBadge status={review.review_state} />
                            )}
                            {review.sentiment && (
                              <Badge variant="outline">{review.sentiment}</Badge>
                            )}
                          </div>
                          {review.actionable_feedback && (
                            <p className="text-sm text-muted-foreground">
                              {review.actionable_feedback}
                            </p>
                          )}
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              )}

              {detail.feedback && (
                <Card>
                  <CardHeader>
                    <CardTitle>Feedback Outcome</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid gap-3 sm:grid-cols-2">
                      <div>
                        <p className="text-sm text-muted-foreground">Outcome</p>
                        <StatusBadge status={detail.feedback.outcome} />
                      </div>
                      <div>
                        <p className="text-sm text-muted-foreground">Error Type</p>
                        <p className="text-sm">{detail.feedback.error_type || '--'}</p>
                      </div>
                      {detail.feedback.learnings && (
                        <div className="sm:col-span-2">
                          <p className="text-sm text-muted-foreground">Learnings</p>
                          <p className="text-sm">{detail.feedback.learnings}</p>
                        </div>
                      )}
                      {detail.feedback.keywords.length > 0 && (
                        <div className="sm:col-span-2">
                          <p className="text-sm text-muted-foreground mb-1">Keywords</p>
                          <div className="flex flex-wrap gap-1">
                            {detail.feedback.keywords.map(kw => (
                              <Badge key={kw} variant="secondary">
                                {kw}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              )}
            </>
          )}
        </div>
      )}
    </div>
  )
}
