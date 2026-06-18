import { Sentry } from './sentry'

const API_BASE = '/api';

export function getWsBase(): string {
  const loc = window.location;
  const proto = loc.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${proto}//${loc.host}`;
}


export interface Stats {
  total: number;
  pending: number;
  success: number;
  failed: number;
  merged: number;
  closed: number;
  cannot_fix: number;
  by_source: Record<string, SourceStats>;
}

export interface SourceStats {
  total: number;
  success: number;
  failed: number;
  merged: number;
  closed: number;
  cannot_fix: number;
}

export interface AttemptSummary {
  id: number;
  source: string;
  short_id: string;
  title: string;
  status: string;
  pr_url: string | null;
  attempted_at: string;
  retry_count: number;
}

export interface SourceSummary {
  name: string;
  total: number;
  success: number;
  failed: number;
  merged: number;
  success_rate: number;
}

export interface Overview {
  stats: Stats;
  success_rate: number;
  merge_rate: number;
  recent_attempts: AttemptSummary[];
  sources: SourceSummary[];
  time_savings?: TimeSavings | null;
  agent_spawns_today?: number;
}

export interface AttemptsResponse {
  attempts: AttemptSummary[];
  total: number;
  page: number;
  per_page: number;
}

export interface IssueSummary {
  id: number;
  source: string;
  issue_id: string;
  short_id: string | null;
  title: string | null;
  description: string | null;
  url: string | null;
  priority: string | null;
  status: string | null;
  labels: string[] | null;
  has_embedding: boolean;
  created_at: string;
  updated_at: string | null;
}

export interface IssuesResponse {
  issues: IssueSummary[];
  total: number;
  page: number;
  per_page: number;
}

export interface Health {
  status: string;
  version: string;
  uptime_secs: number;
  database: {
    status: string;
    error?: string;
  };
}

export interface RetriesResponse {
  retryable: AttemptSummary[];
  ready: AttemptSummary[];
  max_retries: number;
}

export interface SourceInfo {
  name: string;
  enabled: boolean;
  config: Record<string, unknown>;
}

export interface SourcesResponse {
  sources: SourceInfo[];
}


export interface AuthUser {
  id: number
  email: string
  name: string
  role: string
  avatar_url: string | null
}

export interface LoginResponse {
  user: AuthUser
}

export interface UserRecord {
  id: number
  email: string
  name: string
  role: string
  avatar_url: string | null
  created_at: string
  updated_at: string
}


export interface ActivityLogEntry {
  id: number;
  timestamp: string;
  activity_type: string;
  source: string | null;
  issue_id: string | null;
  short_id: string | null;
  message: string;
  metadata: Record<string, unknown> | null;
}

export interface ClaudeExecution {
  id: number;
  attempt_id: number | null;
  started_at: string;
  completed_at: string | null;
  duration_secs: number | null;
  exit_code: number | null;
  timed_out: boolean;
  stdout_preview: string | null;
  stderr_preview: string | null;
  stdout_log_path: string | null;
  stderr_log_path: string | null;
  event_log_path?: string | null;
  prompt_used: string | null;
  prompt_hash: string | null;
  model_version: string | null;
  working_directory: string | null;
  git_branch: string | null;
  git_commit_before: string | null;
  git_commit_after: string | null;
  files_changed: number | null;
  lines_added: number | null;
  lines_removed: number | null;
}

export interface PrReviewRecord {
  id: number;
  attempt_id: number | null;
  pr_url: string;
  reviewer: string | null;
  review_state: string | null;
  submitted_at: string | null;
  body: string | null;
  sentiment: string | null;
  actionable_feedback: string | null;
}

export interface FixAttemptDetail {
  id: number;
  issue_id: string;
  short_id: string;
  source: string;
  attempted_at: string;
  pr_url: string | null;
  scm_repo: string | null;
  scm_pr_number: number | null;
  status: string;
  error_message: string | null;
  merged_at: string | null;
  resolved_at: string | null;
  retry_count: number;
  last_retry_at: string | null;
  issue_labels?: string[];
  parent_attempt_id: number | null;
  cascade_repo: string | null;
}

export interface AttemptDetailResponse {
  attempt: FixAttemptDetail;
  executions: ClaudeExecution[];
  reviews: PrReviewRecord[];
  feedback: FixOutcome | null;
}

export interface AttemptExecutionLogResponse {
  attempt_id: number;
  execution_id: number;
  stream: 'stdout' | 'stderr' | 'events';
  path: string | null;
  content: string | null;
  truncated: boolean;
}

export interface AnalyticsSummary {
  success_rate: number;
  total_processed: number;
  total_successful: number;
  total_merged: number;
  avg_processing_time_secs: number | null;
  avg_time_to_merge_hours: number | null;
  most_common_error: string | null;
  success_rate_by_source: Record<string, number>;
  avg_time_to_pr_mins?: number | null;
  cost_estimate?: CostEstimate | null;
  mttr_trend?: MttrDataPoint[];
  repo_leaderboard?: RepoLeaderboardEntry[];
  commit_trend?: CommitTrendPoint[];
  support_rating?: SupportRatingSummary | null;
}

export interface CommitTrendPoint {
  period_start: string;
  commits: number;
}

export interface SupportReply {
  action_run_id: number;
  source: string;
  short_id: string;
  kind: string;
  detail?: string | null;
  created_at: string;
  response_time_mins?: number | null;
  rating?: number | null;
  note?: string | null;
  rated_by?: string | null;
}

export interface SupportRatingSummary {
  total_replies: number;
  rated_count: number;
  avg_rating: number | null;
  avg_response_time_mins: number | null;
  avg_rating_by_source: Record<string, number>;
}

export interface ProcessingMetric {
  id: number;
  timestamp: string;
  metric_name: string;
  metric_value: number;
  source: string | null;
  tags: Record<string, unknown> | null;
}

export interface ErrorPattern {
  id: number;
  pattern_hash: string;
  error_type: string | null;
  error_message: string | null;
  first_seen: string;
  last_seen: string;
  occurrence_count: number;
  sources: string[] | null;
  example_issue_ids: string[] | null;
  resolution_hints: string | null;
}

export interface PrRecord {
  id: number;
  pr_url: string;
  scm_repo: string;
  pr_number: number;
  attempt_id: number | null;
  issue_id: string | null;
  issue_source: string | null;
  title: string | null;
  description: string | null;
  author: string | null;
  head_branch: string | null;
  base_branch: string | null;
  status: string;
  created_at: string;
  updated_at: string | null;
  merged_at: string | null;
  closed_at: string | null;
  approvals_count: number;
  changes_requested_count: number;
  comments_count: number;
  last_review_at: string | null;
  time_to_first_review_mins: number | null;
  time_to_merge_mins: number | null;
  review_cycles: number;
  files_changed: number | null;
  lines_added: number | null;
  lines_removed: number | null;
}

export interface PrAnalytics {
  total: number;
  open: number;
  merged: number;
  closed: number;
  avg_time_to_first_review_mins: number | null;
  avg_time_to_merge_mins: number | null;
  avg_review_cycles: number | null;
  merge_rate: number | null;
  by_repo: Record<string, number>;
  avg_time_to_pr_mins?: number | null;
  rejection_reasons?: RejectionReason[];
}

export interface RejectionReason {
  category: string;
  count: number;
}

export interface CostEstimate {
  total_cost: number;
  avg_cost_per_fix: number;
  fix_count: number;
  cost_source: string;
  period: string;
}

export interface MttrDataPoint {
  period_start: string;
  mttr_minutes: number;
  sample_count: number;
}

export interface RepoLeaderboardEntry {
  repo: string;
  total: number;
  success_rate: number;
  merge_rate: number;
  avg_time_to_merge_mins: number | null;
}

export interface TimeSavings {
  merged_count: number;
  hours_saved: number;
  cost_saved: number;
  period: string;
}

export interface FixOutcome {
  id: number;
  attempt_id: number;
  source: string;
  issue_id: string;
  issue_text: string;
  prompt_used: string;
  outcome: string;
  error_type: string | null;
  learnings: string | null;
  keywords: string[];
}

export interface RegressionWatch {
  id: number;
  issue_type: string;
  issue_id: string;
  fix_attempt_id: number;
  status: string;
  pr_merged_at: string | null;
  monitoring_started_at: string | null;
  resolved_at: string | null;
  regressed_at: string | null;
  created_at: string;
}

export interface RegressionCheck {
  id: number;
  regression_watch_id: number;
  issue_still_exists: boolean;
  checked_at: string | null;
  check_details: string | null;
  created_at: string;
}

export interface PromptExperiment {
  id: number;
  experiment_name: string;
  variant: string;
  prompt_template: string;
  prompt_hash: string;
  created_at: string;
  active: boolean;
  success_count: number;
  failure_count: number;
  avg_time_to_merge: number | null;
  avg_review_score: number | null;
}

export interface CreatePromptExperimentRequest {
  experiment_name: string;
  variant: string;
  prompt_template: string;
  active?: boolean;
}

export interface StoredIndexedRepo {
  id: number;
  name: string;
  path: string;
  scm_url: string | null;
  default_branch: string;
  file_count: number;
  last_indexed_at: string;
  created_at: string;
}

export interface IndexStats {
  repo_count: number;
  file_count: number;
  last_indexed_at: string | null;
}

export interface StoredDependency {
  id: number;
  upstream: string;
  downstream: string;
  dep_type: string;
  created_at: string;
}

export interface IndexingProgress {
  status: string;
  total_repos: number;
  indexed_repos: number;
  current_repo: string | null;
  current_repo_files: number;
  total_files_indexed: number;
  started_at: string | null;
  updated_at: string | null;
}

export interface InferenceStats {
  total_attempts: number;
  with_feedback: number;
  correct: number;
  accuracy: number;
  by_confidence: {
    high: number;
    medium: number;
    low: number;
    none: number;
  };
}

export interface InferenceHistoryEntry {
  id: number;
  issue_id: string;
  issue_source: string;
  extracted_keywords: string | null;
  inferred_repo_name: string | null;
  confidence: string | null;
  inference_reason: string | null;
  was_correct: boolean | null;
  duration_ms: number | null;
  created_at: string;
}

export interface TelemetryWindowMetric {
  window: string;
  processed: number;
  successful: number;
  failed: number;
  merged: number;
  success_rate: number;
  error_rate: number;
  throughput_per_hour: number;
}

export interface TelemetryQueueMetrics {
  pending_attempts: number;
  retryable_attempts: number;
  ready_retries: number;
  open_prs: number;
  watches_awaiting_release: number;
  watches_monitoring: number;
  watches_resolved: number;
  watches_regressed: number;
}

export interface ProcessingTimeSummary {
  samples: number;
  avg_secs: number | null;
  p50_secs: number | null;
  p95_secs: number | null;
  p99_secs: number | null;
  max_secs: number | null;
}

export interface TelemetryProcessingTime {
  all_time: ProcessingTimeSummary;
  last_24h: ProcessingTimeSummary;
}

export interface SourceTelemetry {
  source: string;
  total: number;
  pending: number;
  success: number;
  failed: number;
  merged: number;
  closed: number;
  cannot_fix: number;
  retryable: number;
  success_rate: number;
}

export interface TelemetryTimeseriesPoint {
  bucket_start: string;
  total: number;
  pending: number;
  success: number;
  failed: number;
  merged: number;
  closed: number;
  cannot_fix: number;
}

export interface TelemetryOverview {
  generated_at: string;
  uptime_secs: number;
  windows: TelemetryWindowMetric[];
  queue: TelemetryQueueMetrics;
  processing_time: TelemetryProcessingTime;
  source_breakdown: SourceTelemetry[];
  top_errors: ErrorPattern[];
  activity_last_hour: Record<string, number>;
  metric_counts_last_24h: Record<string, number>;
  diagnostics?: Record<string, unknown> | null;
  pr_analytics: PrAnalytics;
  agent_spawns_today?: number;
  agent_spawns_this_week?: number;
}

export interface TelemetryTimeseries {
  period: string;
  bucket_minutes: number;
  generated_at: string;
  points: TelemetryTimeseriesPoint[];
}

export interface TelemetryPipelineTotals {
  fetched: number;
  matched: number;
  queued: number;
  processed: number;
  pr_created: number;
  retries_found: number;
  retries_executed: number;
  retries_failed: number;
  pr_status_checks: number;
  pr_status_merged: number;
  pr_status_closed: number;
  pr_status_errors: number;
  regression_watches_created: number;
  auto_resolved_on_merge: number;
  cascade_triggered: number;
  cascade_failed: number;
}

export interface TelemetryPipelineConversion {
  match_rate: number | null;
  queue_rate: number | null;
  processing_rate: number | null;
  pr_yield_rate: number | null;
}

export interface TelemetryPollLoad {
  poll_cycles: number;
  avg_cycle_secs: number | null;
  p95_cycle_secs: number | null;
  active_avg: number | null;
  active_max: number | null;
  pending_avg: number | null;
  pending_max: number | null;
  total_latest: number | null;
}

export interface TelemetryPipelineSource {
  source: string;
  fetched: number;
  matched: number;
  queued: number;
  processed: number;
  pr_created: number;
  retries_executed: number;
  retries_failed: number;
  match_rate: number | null;
  queue_rate: number | null;
  processing_rate: number | null;
  pr_yield_rate: number | null;
}

export interface TelemetryPipeline {
  generated_at: string;
  period: string;
  totals: TelemetryPipelineTotals;
  conversion: TelemetryPipelineConversion;
  poll_load: TelemetryPollLoad;
  per_source: TelemetryPipelineSource[];
}

export interface TelemetryLatencyByStatus {
  status: string;
  summary: ProcessingTimeSummary;
}

export interface TelemetryLatencyHistogramBucket {
  label: string;
  upper_bound_secs: number | null;
  count: number;
}

export interface TelemetryLatency {
  generated_at: string;
  period: string;
  overall: ProcessingTimeSummary;
  by_status: TelemetryLatencyByStatus[];
  histogram: TelemetryLatencyHistogramBucket[];
}


let onUnauthorized: (() => void) | null = null

export function setOnUnauthorized(cb: () => void) {
  onUnauthorized = cb
}

/**
 * Read the CSRF token from the readable `claudear_csrf` cookie. The backend
 * sets this cookie on every request and requires mutating requests (POST/PUT/
 * DELETE) to echo it back in the `x-csrf-token` header (double-submit cookie
 * pattern). Returns an empty string if the cookie isn't present yet.
 */
export function getCsrfToken(): string {
  const match = document.cookie.match(/(?:^|;\s*)claudear_csrf=([^;]*)/)
  return match ? decodeURIComponent(match[1]) : ''
}

async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(url)
  if (res.status === 401) {
    onUnauthorized?.()
    throw new Error('Unauthorized')
  }
  if (!res.ok) {
    const err = new Error(`Failed to fetch ${url}: ${res.status}`)
    Sentry.captureException(err, { tags: { 'api.url': url, 'api.method': 'GET', 'api.status': String(res.status) } })
    throw err
  }
  return res.json()
}

async function postJson<T>(url: string, body?: unknown): Promise<T> {
  Sentry.addBreadcrumb({ category: 'api', message: `POST ${url}`, level: 'info' })
  const res = await fetch(url, {
    method: 'POST',
    headers: {
      'x-csrf-token': getCsrfToken(),
      ...(body ? { 'Content-Type': 'application/json' } : {}),
    },
    body: body ? JSON.stringify(body) : undefined,
  })
  if (res.status === 401) {
    onUnauthorized?.()
    throw new Error('Unauthorized')
  }
  if (!res.ok) {
    const err = new Error(`Failed to post ${url}: ${res.status}`)
    Sentry.captureException(err, { tags: { 'api.url': url, 'api.method': 'POST', 'api.status': String(res.status) } })
    throw err
  }
  if (res.status === 204) return undefined as T
  return res.json()
}

async function putJson<T>(url: string, body: unknown): Promise<T> {
  Sentry.addBreadcrumb({ category: 'api', message: `PUT ${url}`, level: 'info' })
  const res = await fetch(url, {
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json',
      'x-csrf-token': getCsrfToken(),
    },
    body: JSON.stringify(body),
  })
  if (res.status === 401) {
    onUnauthorized?.()
    throw new Error('Unauthorized')
  }
  if (!res.ok) {
    const err = new Error(`Failed to put ${url}: ${res.status}`)
    Sentry.captureException(err, { tags: { 'api.url': url, 'api.method': 'PUT', 'api.status': String(res.status) } })
    throw err
  }
  return res.json()
}

async function deleteRequest(url: string): Promise<void> {
  Sentry.addBreadcrumb({ category: 'api', message: `DELETE ${url}`, level: 'info' })
  const res = await fetch(url, {
    method: 'DELETE',
    headers: { 'x-csrf-token': getCsrfToken() },
  })
  if (res.status === 401) {
    onUnauthorized?.()
    throw new Error('Unauthorized')
  }
  if (!res.ok) {
    const err = new Error(`Failed to delete ${url}: ${res.status}`)
    Sentry.captureException(err, { tags: { 'api.url': url, 'api.method': 'DELETE', 'api.status': String(res.status) } })
    throw err
  }
}

// Existing
export async function fetchOverview(): Promise<Overview> {
  return fetchJson(`${API_BASE}/stats/overview`);
}

export async function getStats(): Promise<Stats> {
  return fetchJson(`${API_BASE}/stats`);
}

export async function fetchAttempts(params?: {
  status?: string;
  source?: string;
  page?: number;
  per_page?: number;
}): Promise<AttemptsResponse> {
  const searchParams = new URLSearchParams();
  if (params?.status) searchParams.set('status', params.status);
  if (params?.source) searchParams.set('source', params.source);
  if (params?.page) searchParams.set('page', params.page.toString());
  if (params?.per_page) searchParams.set('per_page', params.per_page.toString());
  return fetchJson(`${API_BASE}/attempts?${searchParams}`);
}

export async function fetchIssues(params?: {
  source?: string;
  page?: number;
  per_page?: number;
}): Promise<IssuesResponse> {
  const searchParams = new URLSearchParams();
  if (params?.source) searchParams.set('source', params.source);
  if (params?.page) searchParams.set('page', params.page.toString());
  if (params?.per_page) searchParams.set('per_page', params.per_page.toString());
  return fetchJson(`${API_BASE}/issues?${searchParams}`);
}

export async function getHealth(): Promise<Health> {
  return fetchJson(`${API_BASE}/health`);
}

export async function getRetries(): Promise<RetriesResponse> {
  return fetchJson(`${API_BASE}/retries`);
}

export async function getSources(): Promise<SourcesResponse> {
  return fetchJson(`${API_BASE}/sources`);
}

// New fetchers

export async function fetchActivity(params?: {
  limit?: number;
  source?: string;
}): Promise<ActivityLogEntry[]> {
  const searchParams = new URLSearchParams();
  if (params?.limit) searchParams.set('limit', params.limit.toString());
  if (params?.source) searchParams.set('source', params.source);
  return fetchJson(`${API_BASE}/activity?${searchParams}`);
}

export async function fetchAttemptDetail(attemptId: number): Promise<AttemptDetailResponse> {
  return fetchJson(`${API_BASE}/attempts/${attemptId}/detail`);
}

export async function fetchAttemptExecutionLog(
  attemptId: number,
  executionId: number,
  stream: 'stdout' | 'stderr' | 'events',
): Promise<AttemptExecutionLogResponse> {
  return fetchJson(
    `${API_BASE}/attempts/${attemptId}/logs/${executionId}/${stream}`,
  );
}

export async function fetchAnalyticsSummary(): Promise<AnalyticsSummary> {
  return fetchJson(`${API_BASE}/analytics/summary`);
}

export async function fetchMetrics(params?: {
  name?: string;
  period?: string;
  limit?: number;
}): Promise<ProcessingMetric[]> {
  const searchParams = new URLSearchParams();
  if (params?.name) searchParams.set('name', params.name);
  if (params?.period) searchParams.set('period', params.period);
  if (params?.limit) searchParams.set('limit', params.limit.toString());
  return fetchJson(`${API_BASE}/metrics?${searchParams}`);
}

export async function fetchErrors(limit = 50): Promise<ErrorPattern[]> {
  return fetchJson(`${API_BASE}/errors?limit=${limit}`);
}

export async function fetchSupportReplies(): Promise<SupportReply[]> {
  return fetchJson(`${API_BASE}/support/replies`);
}

export async function rateReply(
  actionRunId: number,
  rating: number,
  note?: string,
): Promise<void> {
  await postJson(`${API_BASE}/support/replies/${actionRunId}/rating`, { rating, note });
}

export async function fetchPrs(params?: {
  status?: string;
  limit?: number;
}): Promise<PrRecord[]> {
  const searchParams = new URLSearchParams();
  if (params?.status) searchParams.set('status', params.status);
  if (params?.limit) searchParams.set('limit', params.limit.toString());
  return fetchJson(`${API_BASE}/prs?${searchParams}`);
}

export async function fetchPrAnalytics(): Promise<PrAnalytics> {
  return fetchJson(`${API_BASE}/prs/analytics`);
}

export async function fetchFeedback(params?: {
  source?: string;
  limit?: number;
}): Promise<FixOutcome[]> {
  const searchParams = new URLSearchParams();
  if (params?.source) searchParams.set('source', params.source);
  if (params?.limit) searchParams.set('limit', params.limit.toString());
  return fetchJson(`${API_BASE}/feedback?${searchParams}`);
}

export async function fetchRegressions(status?: string): Promise<RegressionWatch[]> {
  const searchParams = new URLSearchParams();
  if (status) searchParams.set('status', status);
  return fetchJson(`${API_BASE}/regressions?${searchParams}`);
}

export async function fetchRegressionChecks(watchId: number): Promise<RegressionCheck[]> {
  return fetchJson(`${API_BASE}/regressions/${watchId}/checks`);
}

export async function fetchExperiments(): Promise<PromptExperiment[]> {
  return fetchJson(`${API_BASE}/experiments`);
}

export async function createExperiment(
  data: CreatePromptExperimentRequest,
): Promise<PromptExperiment> {
  return postJson(`${API_BASE}/experiments`, data);
}

export async function updateExperiment(
  id: number,
  data: CreatePromptExperimentRequest,
): Promise<{ ok: boolean }> {
  return putJson(`${API_BASE}/experiments/${id}`, data);
}

export async function fetchRepos(): Promise<StoredIndexedRepo[]> {
  return fetchJson(`${API_BASE}/repos`);
}

export async function fetchRepoStats(): Promise<IndexStats> {
  return fetchJson(`${API_BASE}/repos/stats`);
}

export async function fetchDependencies(): Promise<StoredDependency[]> {
  return fetchJson(`${API_BASE}/repos/dependencies`);
}

export const INDEXING_PROGRESS_WS_PATH = '/api/repos/indexing-progress';

export async function fetchInferenceStats(): Promise<InferenceStats> {
  return fetchJson(`${API_BASE}/inference/stats`);
}

export async function fetchInferenceHistory(limit = 50): Promise<InferenceHistoryEntry[]> {
  return fetchJson(`${API_BASE}/inference/history?limit=${limit}`);
}

export async function fetchTelemetryOverview(): Promise<TelemetryOverview> {
  return fetchJson(`${API_BASE}/telemetry/overview`);
}

export async function fetchTelemetryTimeseries(params?: {
  period?: string;
  bucket_minutes?: number;
}): Promise<TelemetryTimeseries> {
  const searchParams = new URLSearchParams();
  if (params?.period) searchParams.set('period', params.period);
  if (params?.bucket_minutes) {
    searchParams.set('bucket_minutes', params.bucket_minutes.toString());
  }
  return fetchJson(`${API_BASE}/telemetry/timeseries?${searchParams}`);
}

export async function fetchTelemetryPipeline(params?: {
  period?: string;
}): Promise<TelemetryPipeline> {
  const searchParams = new URLSearchParams();
  if (params?.period) searchParams.set('period', params.period);
  return fetchJson(`${API_BASE}/telemetry/pipeline?${searchParams}`);
}

export async function fetchTelemetryLatency(params?: {
  period?: string;
}): Promise<TelemetryLatency> {
  const searchParams = new URLSearchParams();
  if (params?.period) searchParams.set('period', params.period);
  return fetchJson(`${API_BASE}/telemetry/latency?${searchParams}`);
}


export interface KnowledgeEntry {
  id: number
  value: string
  source_type: string
  confidence: number
  occurrence_count: number
  updated_at: string
}

export interface KnowledgeGroup {
  key: string
  label: string
  entries: KnowledgeEntry[]
}

export interface ReviewPatternSummary {
  total_patterns: number
  by_category: Record<string, number>
  promoted_count: number
}

export interface ReviewPattern {
  id: number
  scm_repo: string
  category: string
  pattern_text: string
  example_comments: string[]
  occurrence_count: number
  promoted_to_instruction: boolean
  created_at: string
  updated_at: string
}

export interface PromotedInstruction {
  id: number
  repo: string
  source_type: string
  instruction_text: string
  occurrence_count: number
  confidence: number
  is_active: boolean
  created_at: string
  updated_at: string
}

export interface StrategyFingerprint {
  id: number
  attempt_id: number
  files_explored: string[]
  tests_run: number
  tools_used: Record<string, number>
  fix_approach: string
  strategy_summary: string
  fix_quality_score: number | null
  created_at: string
}

export interface DiffAnalysisSummary {
  id: number
  attempt_id: number
  pr_url: string
  scm_repo: string
  pr_number: number
  files_changed: string[]
  file_types: Record<string, number>
  change_categories: string[]
  diff_summary: string
  created_at: string
}

export interface CrossRepoCorrelation {
  id: number
  repo_a: string
  repo_b: string
  correlation_count: number
  last_seen_at: string
  window_hours: number
}

export interface RepoLearningResponse {
  repo: string
  knowledge: KnowledgeGroup[]
  knowledge_total: number
  instructions: PromotedInstruction[]
  review_patterns: ReviewPattern[]
  review_pattern_summary: ReviewPatternSummary
  strategies: StrategyFingerprint[]
  diff_analyses: DiffAnalysisSummary[]
  correlations: CrossRepoCorrelation[]
}

export async function fetchRepoLearning(repo: string): Promise<RepoLearningResponse> {
  return fetchJson(`${API_BASE}/repos/${encodeURIComponent(repo)}/learning`)
}


export interface ConfigResponse {
  content: string
  path: string
}

export async function fetchConfig(): Promise<ConfigResponse> {
  return fetchJson(`${API_BASE}/config`)
}

export async function saveConfig(content: string): Promise<{ ok: boolean; message: string }> {
  return putJson(`${API_BASE}/config`, { content })
}


export async function login(email: string, password: string): Promise<LoginResponse> {
  return postJson(`${API_BASE}/auth/login`, { email, password })
}

export async function logout(): Promise<void> {
  await postJson(`${API_BASE}/auth/logout`)
}

export async function getMe(): Promise<AuthUser> {
  return fetchJson(`${API_BASE}/auth/me`)
}

export async function updateProfile(data: {
  name?: string; password?: string; current_password?: string
}): Promise<UserRecord> {
  return putJson(`${API_BASE}/auth/profile`, data)
}

export async function uploadAvatar(file: File): Promise<{ avatar_url: string }> {
  const form = new FormData()
  form.append('avatar', file)
  const res = await fetch(`${API_BASE}/auth/avatar`, { method: 'POST', body: form })
  if (res.status === 401) {
    onUnauthorized?.()
    throw new Error('Unauthorized')
  }
  if (!res.ok) throw new Error(`Failed to upload avatar: ${res.status}`)
  return res.json()
}


export async function fetchUsers(): Promise<UserRecord[]> {
  return fetchJson(`${API_BASE}/users`)
}

export async function getUser(id: number): Promise<UserRecord> {
  return fetchJson(`${API_BASE}/users/${id}`)
}

export async function createUser(data: {
  email: string; password: string; name: string; role: string
}): Promise<UserRecord> {
  return postJson(`${API_BASE}/users`, data)
}

export async function updateUser(id: number, data: {
  email?: string; password?: string; name?: string; role?: string
}): Promise<UserRecord> {
  return putJson(`${API_BASE}/users/${id}`, data)
}

export async function deleteUser(id: number): Promise<void> {
  return deleteRequest(`${API_BASE}/users/${id}`)
}
