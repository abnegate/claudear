//! Storage implementations for tracking fix attempts, embeddings, and analytics.

pub mod analytics;
#[cfg(feature = "sqlite")]
pub(crate) mod migrator;
#[cfg(feature = "sqlite")]
pub mod sqlite;
pub mod tenant;
pub mod types;
#[cfg(feature = "sqlite")]
pub mod vectorlite;

pub use analytics::{
    classify_error, compute_error_hash, AnalyticsService, TimePeriod, TrendAnalysis, TrendDirection,
};
#[cfg(feature = "sqlite")]
pub use sqlite::SqliteTracker;
pub use types::{
    ConfidenceBreakdown, DiagnosticCounts, IndexStats, IndexingProgress, InferenceHistoryEntry,
    InferenceStats, PurgeResult, StoredDependency, StoredIndexedRepo, StoredRepository, UserRow,
};
#[cfg(feature = "sqlite")]
pub use vectorlite::{is_vectorlite_available, try_load_vectorlite};

use chrono::{DateTime, Utc};
use claudear_core::error::Result;
use claudear_core::types::{
    ActivityLogEntry, AgentExecution, AnalyticsSummary, ErrorPattern, FixAttempt, FixAttemptStats,
    FixAttemptStatus, IssueEmbedding, PrAnalytics, PrRecord, PrReviewRecord, ProcessingMetric,
    PromptExperiment, QaKnowledgeEntry, QaMatch, RegressionCheck, RegressionWatch,
    RegressionWatchStatus, SimilarIssue,
};
use claudear_core::types::{CrossRepoCorrelation, FixOutcome};
use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

/// Maximum allowed length for PR URLs to prevent ReDoS and excessive memory usage.
const MAX_PR_URL_LENGTH: usize = 2048;

/// Compiled regex for parsing GitHub PR URLs into repo/PR number.
static PR_URL_REGEX: LazyLock<regex_lite::Regex> = LazyLock::new(|| {
    regex_lite::Regex::new(r"github\.com/([^/]+/[^/]+)/pull/(\d+)")
        .expect("PR URL regex should be valid")
});

/// Compiled regex for parsing GitLab MR URLs.
static MR_URL_REGEX: LazyLock<regex_lite::Regex> = LazyLock::new(|| {
    regex_lite::Regex::new(r"https?://[^/]+/(.+?)/-/merge_requests/(\d+)")
        .expect("MR URL regex should be valid")
});

/// Parse a GitHub PR or GitLab MR URL into `(repo, number)`.
pub fn parse_pr_url(url: &str) -> Option<(String, i64)> {
    if url.len() > MAX_PR_URL_LENGTH {
        return None;
    }
    if let Some(caps) = PR_URL_REGEX.captures(url) {
        let repo = caps.get(1)?.as_str().to_string();
        let pr_number: i64 = caps.get(2)?.as_str().parse().ok()?;
        return Some((repo, pr_number));
    }
    if let Some(caps) = MR_URL_REGEX.captures(url) {
        let project = caps.get(1)?.as_str().to_string();
        let mr_iid: i64 = caps.get(2)?.as_str().parse().ok()?;
        return Some((project, mr_iid));
    }
    None
}

/// Core fix attempt lifecycle methods (all required, no defaults).
pub trait AttemptTracker: Send + Sync {
    /// Check if an issue has already been attempted.
    fn has_attempted(&self, source: &str, issue_id: &str) -> Result<bool>;

    /// Get all attempted issue IDs for a source.
    fn get_attempted_issue_ids(&self, source: &str) -> Result<HashSet<String>>;

    /// Record a new fix attempt (pending status).
    fn record_attempt(&self, source: &str, issue_id: &str, short_id: &str) -> Result<()>;

    /// Record a new fix attempt with labels (pending status).
    fn record_attempt_with_labels(
        &self,
        source: &str,
        issue_id: &str,
        short_id: &str,
        labels: &[String],
    ) -> Result<()>;

    /// Update a fix attempt with success and PR URL.
    fn mark_success(&self, source: &str, issue_id: &str, pr_url: &str) -> Result<()>;

    /// Update a fix attempt with failure.
    fn mark_failed(&self, source: &str, issue_id: &str, error_message: &str) -> Result<()>;

    /// Mark a fix attempt as merged (PR was merged).
    fn mark_merged(&self, source: &str, issue_id: &str) -> Result<()>;

    /// Mark a fix attempt as closed (PR was closed without merging).
    fn mark_closed(&self, source: &str, issue_id: &str) -> Result<()>;

    /// Mark the issue as resolved on the remote source.
    fn mark_resolved(&self, source: &str, issue_id: &str) -> Result<()>;

    /// Get a specific fix attempt.
    fn get_attempt(&self, source: &str, issue_id: &str) -> Result<Option<FixAttempt>>;

    /// Get all fix attempts with a specific status.
    fn get_attempts_by_status(&self, status: FixAttemptStatus) -> Result<Vec<FixAttempt>>;

    /// Get all successful attempts that haven't been merged yet (PRs to monitor).
    fn get_pending_prs(&self) -> Result<Vec<FixAttempt>>;

    /// Get a fix attempt by PR URL.
    fn get_attempt_by_pr_url(&self, pr_url: &str) -> Result<Option<FixAttempt>>;

    /// Reset a failed attempt to allow retry.
    fn reset_attempt(&self, source: &str, issue_id: &str) -> Result<()>;

    /// Get statistics about fix attempts.
    fn get_stats(&self) -> Result<FixAttemptStats>;

    /// Increment retry count and set last retry timestamp.
    fn increment_retry(&self, source: &str, issue_id: &str) -> Result<()>;

    /// Mark an issue as cannot be fixed (max retries reached).
    fn mark_cannot_fix(&self, source: &str, issue_id: &str, reason: &str) -> Result<()>;

    /// Mark an issue as answered (it was a question, not a fix request).
    ///
    /// Default no-op; persistent trackers should set the attempt status to
    /// `answered` so it is not retried or re-polled.
    fn mark_answered(&self, source: &str, issue_id: &str, summary: &str) -> Result<()> {
        let _ = (source, issue_id, summary);
        Ok(())
    }

    /// Get issues that are eligible for retry (failed/closed with retry_count < max_retries).
    fn get_retryable_issues(&self, max_retries: u32) -> Result<Vec<FixAttempt>>;

    /// Prepare an issue for retry (reset status to pending, clear PR info).
    fn prepare_for_retry(&self, source: &str, issue_id: &str) -> Result<()>;
}

/// Activity logging, execution tracking, metrics, analytics, PR review,
/// error patterns, listing/pagination (all defaulted no-ops).
pub trait ActivityStore: Send + Sync {
    /// Record an activity log entry.
    fn record_activity(&self, _entry: &ActivityLogEntry) -> Result<i64> {
        Ok(0)
    }

    /// Record an action-pipeline run (verify verdict, reply post, etc.) for
    /// queryability/analytics. Default no-op; the SQLite tracker persists it.
    fn record_action_run(
        &self,
        _source: &str,
        _issue_id: &str,
        _short_id: &str,
        _action_kind: &str,
        _status: &str,
        _detail: &str,
    ) -> Result<()> {
        Ok(())
    }

    /// Get recent activity entries.
    fn get_recent_activities(&self, _limit: usize) -> Result<Vec<ActivityLogEntry>> {
        Ok(Vec::new())
    }

    /// Get recent activities with optional source filter.
    fn get_recent_activities_filtered(
        &self,
        _limit: usize,
        _source_filter: Option<&str>,
    ) -> Result<Vec<ActivityLogEntry>> {
        Ok(Vec::new())
    }

    /// Record an agent execution.
    fn record_execution(&self, _execution: &AgentExecution) -> Result<i64> {
        Ok(0)
    }

    /// Get agent executions for a given attempt ID.
    fn get_executions_for_attempt(&self, _attempt_id: i64) -> Result<Vec<AgentExecution>> {
        Ok(Vec::new())
    }

    /// Get a fix attempt by database ID.
    fn get_attempt_by_id(&self, _id: i64) -> Result<Option<FixAttempt>> {
        Ok(None)
    }

    /// Record a PR review.
    fn record_pr_review(&self, _review: &PrReviewRecord) -> Result<i64> {
        Ok(0)
    }

    /// Get PR reviews for a given attempt ID.
    fn get_reviews_for_attempt(&self, _attempt_id: i64) -> Result<Vec<PrReviewRecord>> {
        Ok(Vec::new())
    }

    /// Record an error pattern.
    fn record_error_pattern(&self, _pattern: &ErrorPattern) -> Result<i64> {
        Ok(0)
    }

    /// Get error patterns.
    fn get_error_patterns(&self, _limit: usize) -> Result<Vec<ErrorPattern>> {
        Ok(Vec::new())
    }

    /// Record a processing metric.
    fn record_metric(&self, _metric: &ProcessingMetric) -> Result<i64> {
        Ok(0)
    }

    /// Get processing metrics for a named metric.
    fn get_metrics(
        &self,
        _metric_name: &str,
        _since: Option<DateTime<Utc>>,
        _limit: usize,
    ) -> Result<Vec<ProcessingMetric>> {
        Ok(Vec::new())
    }

    /// Get analytics summary.
    fn get_analytics_summary(&self) -> Result<AnalyticsSummary> {
        Ok(AnalyticsSummary::default())
    }

    /// Get open PRs from the prs table.
    fn get_open_prs(&self) -> Result<Vec<claudear_core::types::PrRecord>> {
        Ok(Vec::new())
    }

    /// Get PR analytics.
    fn get_pr_analytics(&self) -> Result<PrAnalytics> {
        Ok(PrAnalytics::default())
    }

    /// Average time from issue attempt to PR creation in minutes.
    fn get_avg_time_to_pr(&self) -> Result<Option<f64>> {
        Ok(None)
    }

    /// Top rejection/review-change reason categories.
    fn get_rejection_reasons(
        &self,
        _limit: usize,
    ) -> Result<Vec<claudear_core::types::RejectionReason>> {
        Ok(Vec::new())
    }

    /// Count Claude agent spawns since a given ISO timestamp.
    fn get_agent_spawn_count(&self, _since_iso: &str) -> Result<i64> {
        Ok(0)
    }

    /// Compute cost estimate from Claude execution data.
    fn get_cost_estimate(
        &self,
        _since_iso: &str,
        _max_plan_monthly_cost: f64,
        _period_label: &str,
    ) -> Result<claudear_core::types::CostEstimate> {
        Ok(claudear_core::types::CostEstimate::default())
    }

    /// MTTR trend grouped by week.
    fn get_mttr_trend(&self, _weeks: usize) -> Result<Vec<claudear_core::types::MttrDataPoint>> {
        Ok(Vec::new())
    }

    /// Per-repository leaderboard.
    fn get_repo_leaderboard(&self) -> Result<Vec<claudear_core::types::RepoLeaderboardEntry>> {
        Ok(Vec::new())
    }

    /// Complexity-based engineering time savings estimate.
    fn get_complexity_time_savings(
        &self,
        _since_iso: &str,
        _hourly_rate: f64,
        _period_label: &str,
    ) -> Result<claudear_core::types::TimeSavings> {
        Ok(claudear_core::types::TimeSavings::default())
    }

    /// List all PRs with optional status and limit filters.
    fn list_prs(
        &self,
        _status: Option<&str>,
        _limit: usize,
    ) -> Result<Vec<claudear_core::types::PrRecord>> {
        Ok(Vec::new())
    }

    /// Upsert a PR record. Returns the row ID.
    fn upsert_pr(&self, _pr: &PrRecord) -> Result<i64> {
        Ok(0)
    }

    /// Get a PR record by URL.
    fn get_pr(&self, _pr_url: &str) -> Result<Option<PrRecord>> {
        Ok(None)
    }

    /// Update a PR's status.
    fn update_pr_status(&self, _pr_url: &str, _status: &str) -> Result<()> {
        Ok(())
    }

    /// List fix attempts with optional filters and pagination.
    fn list_attempts(
        &self,
        _status: Option<&str>,
        _source: Option<&str>,
        _limit: usize,
        _offset: usize,
    ) -> Result<Vec<FixAttempt>> {
        Ok(Vec::new())
    }

    /// Count fix attempts with optional filters.
    fn count_attempts(&self, _status: Option<&str>, _source: Option<&str>) -> Result<usize> {
        Ok(0)
    }

    /// List the most recent fix attempts.
    fn list_recent_attempts(&self, _limit: usize) -> Result<Vec<FixAttempt>> {
        Ok(Vec::new())
    }

    /// List fix attempts created since a timestamp.
    fn list_attempts_since(&self, _since: DateTime<Utc>) -> Result<Vec<FixAttempt>> {
        Ok(Vec::new())
    }

    /// Get the most recent merged attempt for a given SCM repo.
    fn get_most_recent_merged_attempt_for_repo(
        &self,
        _scm_repo: &str,
    ) -> Result<Option<FixAttempt>> {
        Ok(None)
    }

    /// Get a specific execution for an attempt by execution ID.
    fn get_execution_for_attempt(
        &self,
        _attempt_id: i64,
        _execution_id: i64,
    ) -> Result<Option<AgentExecution>> {
        Ok(None)
    }

    /// Record a batch of activity log entries. Returns count recorded.
    fn record_activities_batch(&self, _entries: &[ActivityLogEntry]) -> Result<usize> {
        Ok(0)
    }

    /// Prune old activity log entries. Returns count pruned.
    fn prune_old_activities(&self, _days_to_keep: i64) -> Result<usize> {
        Ok(0)
    }

    /// Purge all operational data (attempts, PRs, executions, reviews, activity logs,
    /// regressions, clusters, metrics) while preserving knowledge, inference, embeddings,
    /// code index, and learned patterns.
    ///
    /// Feedback outcomes are detached from attempts (attempt_id set to NULL) rather than
    /// deleted, so learnings and embeddings are retained.
    fn purge_operational_data(&self) -> Result<types::PurgeResult> {
        Ok(types::PurgeResult {
            fix_attempts: 0,
            prs: 0,
            pr_reviews: 0,
            pr_review_comments: 0,
            pr_review_states: 0,
            claude_executions: 0,
            strategy_fingerprints: 0,
            diff_analyses: 0,
            regression_watches: 0,
            release_tracking: 0,
            regression_checks: 0,
            qa_usage: 0,
            activity_log: 0,
            processing_metrics: 0,
            webhook_deliveries: 0,
            issue_clusters: 0,
            issue_cluster_members: 0,
            content_clusters: 0,
            severity_scores: 0,
            suppression_log: 0,
            eval_snapshots: 0,
            eval_deltas: 0,
            feedback_outcomes_detached: 0,
        })
    }

    /// Prune old processing metrics. Returns count pruned.
    fn prune_old_metrics(&self, _days_to_keep: i64) -> Result<usize> {
        Ok(0)
    }

    /// Get activity type counts since a timestamp.
    fn get_activity_type_counts_since(
        &self,
        _since: DateTime<Utc>,
    ) -> Result<HashMap<String, i64>> {
        Ok(HashMap::new())
    }

    /// Get metric counts for named metrics since a timestamp.
    fn get_metric_counts_since(
        &self,
        _metric_names: &[&str],
        _since: DateTime<Utc>,
    ) -> Result<HashMap<String, i64>> {
        Ok(HashMap::new())
    }

    /// Get metric sums for named metrics since a timestamp.
    fn get_metric_sums_since(
        &self,
        _metric_names: &[&str],
        _since: DateTime<Utc>,
    ) -> Result<HashMap<String, f64>> {
        Ok(HashMap::new())
    }

    /// Get metric sums grouped by source for named metrics since a timestamp.
    fn get_metric_sums_by_source_since(
        &self,
        _metric_names: &[&str],
        _since: DateTime<Utc>,
    ) -> Result<HashMap<(String, String), f64>> {
        Ok(HashMap::new())
    }

    /// Get diagnostic row counts for all major tables.
    fn get_diagnostic_counts(&self) -> Result<DiagnosticCounts> {
        Ok(DiagnosticCounts {
            fix_attempts: 0,
            fix_attempts_by_status: HashMap::new(),
            activity_log: 0,
            claude_executions: 0,
            pr_reviews: 0,
            pr_review_states: 0,
            issues: 0,
            similar_issues: 0,
            repositories: 0,
            repo_files: 0,
            inference_attempts: 0,
            error_patterns: 0,
            processing_metrics: 0,
            feedback_outcomes: 0,
            prs: 0,
            recent_fix_attempts: Vec::new(),
        })
    }

    /// Get the overall success rate (0.0-1.0).
    fn get_success_rate(&self) -> Result<f64> {
        Ok(0.0)
    }

    /// Record a cascade fix attempt. Returns the new attempt ID.
    fn record_cascade_attempt(
        &self,
        _source: &str,
        _issue_id: &str,
        _short_id: &str,
        _parent_attempt_id: i64,
        _cascade_repo: &str,
    ) -> Result<i64> {
        Ok(0)
    }

    /// Update a fix attempt with its PR details.
    fn update_attempt_pr(
        &self,
        _attempt_id: i64,
        _pr_url: &str,
        _scm_repo: &str,
        _pr_number: i64,
    ) -> Result<()> {
        Ok(())
    }

    /// Mark a cascade attempt as failed.
    fn mark_cascade_failed(&self, _attempt_id: i64, _error: &str) -> Result<()> {
        Ok(())
    }

    /// Save a PR review state.
    fn save_pr_review_state(&self, _state: &claudear_core::types::PrReviewState) -> Result<()> {
        Ok(())
    }

    /// Get all active PR review states.
    fn get_active_pr_review_states(&self) -> Result<Vec<claudear_core::types::PrReviewState>> {
        Ok(Vec::new())
    }

    /// Deactivate a PR review state.
    fn deactivate_pr_review_state(&self, _pr_url: &str) -> Result<()> {
        Ok(())
    }

    /// Record a PR review comment. Returns the row ID.
    fn record_pr_review_comment(
        &self,
        _pr_url: &str,
        _comment: &claudear_core::types::ReviewComment,
    ) -> Result<i64> {
        Ok(0)
    }

    /// Get all stored review comments for a PR.
    fn get_comments_for_pr(&self, _pr_url: &str) -> Result<Vec<types::StoredPrReviewComment>> {
        Ok(Vec::new())
    }

    /// Get fix attempts for a batch of (source, issue_id) keys.
    fn get_attempts_batch(&self, _keys: &[(&str, &str)]) -> Result<Vec<Option<FixAttempt>>> {
        Ok(Vec::new())
    }

    /// Record a release tracking entry. Returns the row ID.
    fn record_release_tracking(
        &self,
        _tracking: &claudear_core::types::ReleaseTracking,
    ) -> Result<i64> {
        Ok(0)
    }

    /// System 7: Update a PR's fix quality score.
    fn update_pr_fix_quality_score(&self, _pr_url: &str, _score: f64) -> Result<()> {
        Ok(())
    }
}

/// Q&A, feedback outcomes, learning systems, severity, clustering (all defaulted).
pub trait KnowledgeStore: Send + Sync {
    /// Store a feedback outcome.
    fn store_feedback_outcome(&self, _outcome: &FixOutcome) -> Result<i64> {
        Ok(0)
    }

    /// Get feedback outcomes with optional source filter.
    fn get_feedback_outcomes(
        &self,
        _source: Option<&str>,
        _limit: usize,
    ) -> Result<Vec<FixOutcome>> {
        Ok(Vec::new())
    }

    /// Get a feedback outcome by attempt ID.
    fn get_feedback_outcome_by_attempt(&self, _attempt_id: i64) -> Result<Option<FixOutcome>> {
        Ok(None)
    }

    /// System 1: Update learnings text on a feedback outcome.
    fn update_feedback_learnings(&self, _outcome_id: i64, _learnings: &str) -> Result<()> {
        Ok(())
    }

    /// Store a Q&A knowledge entry.
    fn store_qa_knowledge(&self, _entry: &QaKnowledgeEntry) -> Result<i64> {
        Ok(0)
    }

    /// Find semantically similar Q&A entries within a scoped source/repo.
    fn find_similar_qa_scoped(
        &self,
        _source: &str,
        _repo: Option<&str>,
        _question_norm: &str,
        _question_embedding: Option<&[f32]>,
        _threshold: f64,
        _limit: usize,
    ) -> Result<Vec<QaMatch>> {
        Ok(Vec::new())
    }

    /// Find semantically similar Q&A entries across all sources.
    fn find_similar_qa_global(
        &self,
        _question_norm: &str,
        _question_embedding: Option<&[f32]>,
        _threshold: f64,
        _limit: usize,
    ) -> Result<Vec<QaMatch>> {
        Ok(Vec::new())
    }

    /// Record usage of a Q&A entry for an attempt.
    fn record_qa_usage(
        &self,
        _attempt_id: i64,
        _qa_id: i64,
        _usage_type: &str,
        _similarity_score: f64,
    ) -> Result<i64> {
        Ok(0)
    }

    /// Update success/failure counters for a Q&A entry.
    fn update_qa_outcome_stats(&self, _qa_id: i64, _success: bool) -> Result<()> {
        Ok(())
    }

    /// Update Q&A outcome counters for all Q&A usage linked to an attempt.
    fn update_qa_outcome_stats_for_attempt(&self, _attempt_id: i64, _success: bool) -> Result<()> {
        Ok(())
    }

    /// System 2: Store a diff analysis for a merged PR.
    fn store_diff_analysis(&self, _analysis: &claudear_core::types::DiffAnalysis) -> Result<i64> {
        Ok(0)
    }

    /// System 2: Get diff analyses for a repo.
    fn get_diff_analyses_for_repo(
        &self,
        _repo: &str,
        _limit: usize,
    ) -> Result<Vec<claudear_core::types::DiffAnalysis>> {
        Ok(Vec::new())
    }

    /// System 3: Upsert a promoted instruction.
    fn upsert_promoted_instruction(
        &self,
        _instruction: &claudear_core::types::PromotedInstruction,
    ) -> Result<i64> {
        Ok(0)
    }

    /// System 3: Get active promoted instructions for a repo.
    fn get_promoted_instructions(
        &self,
        _repo: &str,
    ) -> Result<Vec<claudear_core::types::PromotedInstruction>> {
        Ok(Vec::new())
    }

    /// System 4: Upsert a repo knowledge entry.
    fn upsert_repo_knowledge(&self, _entry: &claudear_core::types::RepoKnowledge) -> Result<i64> {
        Ok(0)
    }

    /// System 4: Get all knowledge for a repo.
    fn get_repo_knowledge(&self, _repo: &str) -> Result<Vec<claudear_core::types::RepoKnowledge>> {
        Ok(Vec::new())
    }

    /// System 4: Get repo knowledge by key.
    fn get_repo_knowledge_by_key(
        &self,
        _repo: &str,
        _key: &str,
    ) -> Result<Vec<claudear_core::types::RepoKnowledge>> {
        Ok(Vec::new())
    }

    /// Return all known repository renames as (former_name, current_name) pairs.
    ///
    /// Reads `repo_knowledge` rows where `knowledge_key = "former_name"`.
    /// The `repo` column holds the current name; `knowledge_value` is the old name.
    fn get_all_repo_aliases(&self) -> Result<Vec<(String, String)>> {
        Ok(Vec::new())
    }

    /// System 5: Upsert a review pattern.
    fn upsert_review_pattern(&self, _pattern: &claudear_core::types::ReviewPattern) -> Result<i64> {
        Ok(0)
    }

    /// System 5: Get review patterns for a repo.
    fn get_review_patterns(
        &self,
        _repo: &str,
        _limit: usize,
    ) -> Result<Vec<claudear_core::types::ReviewPattern>> {
        Ok(Vec::new())
    }

    /// System 5: Get review patterns by category.
    fn get_review_patterns_by_category(
        &self,
        _repo: &str,
        _category: claudear_core::types::ReviewCategory,
    ) -> Result<Vec<claudear_core::types::ReviewPattern>> {
        Ok(Vec::new())
    }

    /// System 6: Store a strategy fingerprint.
    fn store_strategy_fingerprint(
        &self,
        _fingerprint: &claudear_core::types::StrategyFingerprint,
    ) -> Result<i64> {
        Ok(0)
    }

    /// System 6: Get successful strategies for a repo.
    fn get_successful_strategies(
        &self,
        _repo: &str,
        _limit: usize,
    ) -> Result<Vec<claudear_core::types::StrategyFingerprint>> {
        Ok(Vec::new())
    }

    /// System 8: Store an issue cluster.
    fn store_issue_cluster(&self, _cluster: &claudear_core::types::IssueCluster) -> Result<i64> {
        Ok(0)
    }

    /// System 8: Get active (unresolved) clusters for a source.
    fn get_active_clusters(
        &self,
        _source: &str,
    ) -> Result<Vec<claudear_core::types::IssueCluster>> {
        Ok(Vec::new())
    }

    /// System 8: Mark a cluster as resolved.
    fn update_cluster_resolution(
        &self,
        _cluster_id: i64,
        _resolved_by_issue_id: &str,
        _resolved_by_attempt_id: i64,
    ) -> Result<()> {
        Ok(())
    }

    /// System 8: Get recent issue arrivals within a time window.
    fn get_recent_issue_arrivals(
        &self,
        _source: &str,
        _window_minutes: i64,
    ) -> Result<Vec<(String, DateTime<Utc>)>> {
        Ok(Vec::new())
    }

    /// Store a content cluster detected by the prioritisation engine.
    fn store_content_cluster(
        &self,
        _cluster: &claudear_core::types::ContentCluster,
    ) -> Result<i64> {
        Ok(0)
    }

    /// Get active (unresolved) content clusters for a source.
    fn get_active_content_clusters(
        &self,
        _source: &str,
    ) -> Result<Vec<claudear_core::types::ContentCluster>> {
        Ok(Vec::new())
    }

    /// Resolve a content cluster by its database ID.
    fn resolve_content_cluster(&self, _cluster_id: i64) -> Result<()> {
        Ok(())
    }

    /// Store the severity score and blast radius for an issue.
    fn store_severity_score(
        &self,
        _source: &str,
        _issue_id: &str,
        _score: &claudear_core::types::SeverityScore,
        _blast_radius: claudear_core::types::BlastRadius,
    ) -> Result<()> {
        Ok(())
    }

    /// Upsert a cross-repo correlation record, incrementing the count.
    fn upsert_cross_repo_correlation(
        &self,
        _repo_a: &str,
        _repo_b: &str,
        _window_hours: i64,
    ) -> Result<CrossRepoCorrelation> {
        Ok(CrossRepoCorrelation {
            id: 0,
            repo_a: _repo_a.to_string(),
            repo_b: _repo_b.to_string(),
            correlation_count: 0,
            last_seen_at: Utc::now(),
            window_hours: _window_hours,
        })
    }

    /// Get cross-repo correlations above a minimum count and within max age.
    fn get_cross_repo_correlations(
        &self,
        _min_count: i64,
        _max_age_hours: i64,
    ) -> Result<Vec<CrossRepoCorrelation>> {
        Ok(Vec::new())
    }

    /// Record that an issue was suppressed by a rule.
    fn record_suppression(
        &self,
        _source: &str,
        _issue_id: &str,
        _rule_name: &str,
        _reason: &str,
    ) -> Result<()> {
        Ok(())
    }
}

/// Issue embeddings, vector search (all defaulted).
pub trait EmbeddingStore: Send + Sync {
    /// Store an issue embedding record. Returns the row ID.
    fn store_issue(&self, _issue: &IssueEmbedding) -> Result<i64> {
        Ok(0)
    }

    /// Store a single issue embedding. Returns the row ID.
    fn store_embedding(&self, _embedding: &IssueEmbedding) -> Result<i64> {
        Ok(0)
    }

    /// Store a batch of issue embeddings.
    fn store_embeddings_batch(&self, _embeddings: &[IssueEmbedding]) -> Result<()> {
        Ok(())
    }

    /// Get an embedding by source and issue ID.
    fn get_embedding(&self, _source: &str, _issue_id: &str) -> Result<Option<IssueEmbedding>> {
        Ok(None)
    }

    /// Get all embeddings with optional source filter and pagination.
    fn get_all_embeddings(
        &self,
        _source: Option<&str>,
        _limit: Option<usize>,
        _offset: Option<usize>,
    ) -> Result<Vec<IssueEmbedding>> {
        Ok(Vec::new())
    }

    /// Find similar issues by vector similarity. Returns None if vector search is unavailable.
    fn find_similar_issues_vector(
        &self,
        _query_embedding: &[f32],
        _source: &str,
        _exclude_issue_id: Option<&str>,
        _min_similarity: f64,
        _limit: usize,
    ) -> Result<Option<Vec<(IssueEmbedding, f64)>>> {
        Ok(None)
    }

    /// Find similar outcomes by vector similarity. Returns None if vector search is unavailable.
    fn find_similar_outcomes_vector(
        &self,
        _query_embedding: &[f32],
        _min_similarity: f64,
        _limit: usize,
    ) -> Result<Option<Vec<(FixOutcome, f64)>>> {
        Ok(None)
    }

    /// Search code chunks by vector similarity.
    fn search_code_chunks(
        &self,
        _query_embedding: &[f32],
        _repo_id: Option<i64>,
        _limit: usize,
    ) -> Result<Vec<claudear_core::types::CodeSearchResult>> {
        Ok(Vec::new())
    }

    /// Get recent fix attempts since a cutoff time.
    fn get_recent_attempts_since(&self, _since: &DateTime<Utc>) -> Result<Vec<FixAttempt>> {
        Ok(Vec::new())
    }

    /// List issues with optional source filter and pagination.
    fn list_issues(
        &self,
        _source: Option<&str>,
        _limit: usize,
        _offset: usize,
    ) -> Result<Vec<IssueEmbedding>> {
        Ok(Vec::new())
    }

    /// Count issues with optional source filter.
    fn count_issues(&self, _source: Option<&str>) -> Result<usize> {
        Ok(0)
    }

    /// Record an inference attempt. Returns the row ID.
    #[expect(clippy::too_many_arguments)]
    fn record_inference_attempt(
        &self,
        _issue_id: &str,
        _issue_source: &str,
        _extracted_filenames: &[String],
        _extracted_functions: &[String],
        _extracted_keywords: &[String],
        _inferred_repo_id: Option<i64>,
        _confidence: &str,
        _inference_reason: &str,
        _duration_ms: Option<u64>,
    ) -> Result<i64> {
        Ok(0)
    }

    /// Record feedback on an inference attempt.
    fn record_inference_feedback(
        &self,
        _inference_id: i64,
        _was_correct: bool,
        _actual_repo_id: Option<i64>,
        _feedback_source: &str,
    ) -> Result<()> {
        Ok(())
    }
}

/// Prompt experiment CRUD (all defaulted).
pub trait ExperimentStore: Send + Sync {
    /// Save a prompt experiment. Returns the row ID.
    fn save_experiment(&self, _experiment: &PromptExperiment) -> Result<i64> {
        Ok(0)
    }

    /// Update an existing prompt experiment configuration.
    fn update_experiment(
        &self,
        _experiment_id: i64,
        _experiment_name: &str,
        _variant: &str,
        _prompt_template: &str,
        _prompt_hash: &str,
        _active: bool,
    ) -> Result<bool> {
        Ok(false)
    }

    /// Update experiment stats after a result.
    fn update_experiment_stats(
        &self,
        _experiment_id: i64,
        _success: bool,
        _time_to_merge: Option<f64>,
    ) -> Result<()> {
        Ok(())
    }

    /// Get active prompt experiments.
    fn get_active_experiments(&self) -> Result<Vec<PromptExperiment>> {
        Ok(Vec::new())
    }
}

/// Evaluation snapshot and delta storage (all defaulted).
pub trait EvaluationStore: Send + Sync {
    /// Store an evaluation snapshot. Returns the row ID.
    fn store_eval_snapshot(
        &self,
        _attempt_id: Option<i64>,
        _phase: &str,
        _snapshot: &claudear_core::types::EvalSnapshot,
    ) -> Result<i64> {
        Ok(0)
    }

    /// Store an evaluation delta. Returns the row ID.
    fn store_eval_delta(
        &self,
        _attempt_id: Option<i64>,
        _repo: &str,
        _delta: &claudear_core::types::EvalDelta,
    ) -> Result<i64> {
        Ok(0)
    }
}

/// Webhook delivery deduplication (all defaulted).
pub trait WebhookStore: Send + Sync {
    /// Check if a delivery has been seen and record it. Returns true if already seen.
    fn check_and_record_delivery(&self, _delivery_id: &str, _source: &str) -> Result<bool> {
        Ok(false)
    }

    /// Cleanup old deliveries older than max_age_hours. Returns count removed.
    fn cleanup_old_deliveries(&self, _max_age_hours: u64) -> Result<usize> {
        Ok(0)
    }
}

/// Similar issue storage and lookup (all defaulted).
pub trait SimilarityStore: Send + Sync {
    /// Store a batch of similar issue records.
    fn store_similar_issues_batch(&self, _similar_issues: &[SimilarIssue]) -> Result<()> {
        Ok(())
    }

    /// Store a single similar issue relationship. Returns the row ID.
    fn store_similar_issue(&self, _similar: &SimilarIssue) -> Result<i64> {
        Ok(0)
    }

    /// Find similar issues for a given issue ID above a minimum score.
    fn find_similar_issues(
        &self,
        _issue_id: &str,
        _min_score: f64,
        _limit: usize,
    ) -> Result<Vec<SimilarIssue>> {
        Ok(Vec::new())
    }
}

/// Repo management, code indexing, dependencies, indexing progress (all defaulted).
pub trait RepoStore: Send + Sync {
    /// List indexed repositories.
    fn list_indexed_repos(&self) -> Result<Vec<StoredIndexedRepo>> {
        Ok(Vec::new())
    }

    /// Get index statistics.
    fn get_index_stats(&self) -> Result<IndexStats> {
        Ok(IndexStats {
            repo_count: 0,
            file_count: 0,
            last_indexed_at: None,
        })
    }

    /// Sync a single repository's files into storage.
    fn sync_repo_files(&self, _repo: &claudear_core::types::IndexedRepo) -> Result<()> {
        Ok(())
    }

    /// Get current indexing progress.
    fn get_indexing_progress(&self) -> Result<IndexingProgress> {
        Ok(IndexingProgress {
            status: "idle".to_string(),
            total_repos: 0,
            indexed_repos: 0,
            current_repo: None,
            current_repo_files: 0,
            total_files_indexed: 0,
            started_at: None,
            updated_at: None,
        })
    }

    /// Start indexing progress tracking.
    fn start_indexing_progress(&self, _total_repos: usize) -> Result<()> {
        Ok(())
    }

    /// Update indexing progress for a specific repo.
    fn update_indexing_progress(
        &self,
        _indexed_repos: usize,
        _current_repo: &str,
        _current_repo_files: usize,
        _total_files_indexed: usize,
    ) -> Result<()> {
        Ok(())
    }

    /// Mark indexing as complete.
    fn finish_indexing_progress(&self) -> Result<()> {
        Ok(())
    }

    /// Subscribe to real-time indexing progress updates via a watch channel.
    /// Default implementation returns a receiver that never changes (dead channel).
    fn subscribe_indexing_progress(&self) -> tokio::sync::watch::Receiver<IndexingProgress> {
        let (_, rx) = tokio::sync::watch::channel(IndexingProgress::default());
        rx
    }

    /// Add a dependency between two repositories. Creates the repos if they don't exist.
    fn add_dependency(&self, _upstream: &str, _downstream: &str, _dep_type: &str) -> Result<()> {
        Ok(())
    }

    /// Check if repo_a depends on repo_b.
    fn has_dependency(&self, _repo_a: &str, _repo_b: &str) -> Result<bool> {
        Ok(false)
    }

    /// List all dependencies.
    fn list_all_dependencies(&self) -> Result<Vec<StoredDependency>> {
        Ok(Vec::new())
    }

    /// Get inference statistics.
    fn get_inference_stats(&self) -> Result<InferenceStats> {
        Ok(InferenceStats {
            total_attempts: 0,
            with_feedback: 0,
            correct: 0,
            accuracy: 0.0,
            by_confidence: ConfidenceBreakdown {
                high: 0,
                medium: 0,
                low: 0,
                none: 0,
            },
        })
    }

    /// Get inference history.
    fn get_inference_history(&self, _limit: usize) -> Result<Vec<InferenceHistoryEntry>> {
        Ok(Vec::new())
    }

    /// Sync repositories from a RepoIndex into storage. Returns count synced.
    fn sync_from_index(
        &self,
        _index: &claudear_core::types::RepoIndex,
        _sync_files: bool,
    ) -> Result<usize> {
        Ok(0)
    }

    /// Get an indexed repository by name.
    fn get_indexed_repo(&self, _name: &str) -> Result<Option<StoredIndexedRepo>> {
        Ok(None)
    }

    /// Get or create a repository ID by name.
    fn get_or_create_repo_id(&self, _name: &str) -> Result<i64> {
        Ok(0)
    }

    /// Check if a file's content hash matches what's already indexed.
    fn code_chunk_hash_matches(
        &self,
        _repo_id: i64,
        _file_path: &str,
        _file_hash: &str,
    ) -> Result<bool> {
        Ok(false)
    }

    /// Delete all code symbols, chunks, and embeddings for a specific file.
    fn delete_code_data_for_file(&self, _repo_id: i64, _file_path: &str) -> Result<()> {
        Ok(())
    }

    /// Delete code chunks by IDs.
    fn delete_code_chunks_by_ids(&self, _chunk_ids: &[i64]) -> Result<()> {
        Ok(())
    }

    /// Remove code data for files that no longer exist in the repository.
    fn cleanup_stale_code_data(&self, _repo_id: i64, _current_paths: &[String]) -> Result<()> {
        Ok(())
    }

    /// Batch-save extracted code symbols.
    fn save_code_symbols(&self, _symbols: &[claudear_core::types::CodeSymbol]) -> Result<()> {
        Ok(())
    }

    /// Batch-save code chunks. Returns the assigned IDs.
    fn save_code_chunks(&self, _chunks: &[claudear_core::types::CodeChunk]) -> Result<Vec<i64>> {
        Ok(Vec::new())
    }

    /// Save embeddings for code chunks.
    fn save_code_chunk_embeddings(
        &self,
        _pairs: &[(i64, &[f32])],
        _model_name: &str,
    ) -> Result<()> {
        Ok(())
    }

    /// Get the embedding model name used for a repo's code chunks.
    /// Returns `None` if no embeddings exist for the repo.
    fn get_code_embedding_model(&self, _repo_id: i64) -> Result<Option<String>> {
        Ok(None)
    }

    /// Get a code-index metadata value for a repo.
    fn get_code_index_meta(&self, _repo_id: i64, _key: &str) -> Result<Option<String>> {
        Ok(None)
    }

    /// Set a code-index metadata value for a repo (upsert).
    fn set_code_index_meta(&self, _repo_id: i64, _key: &str, _value: &str) -> Result<()> {
        Ok(())
    }

    /// Delete all code data (symbols, chunks, embeddings) for a repo.
    fn delete_all_code_data_for_repo(&self, _repo_id: i64) -> Result<()> {
        Ok(())
    }

    /// Find code symbols by name substring.
    fn find_code_symbols(
        &self,
        _name: &str,
        _kind: Option<claudear_core::types::SymbolKind>,
        _repo_id: Option<i64>,
    ) -> Result<Vec<claudear_core::types::CodeSymbol>> {
        Ok(Vec::new())
    }
}

/// User management, sessions, channels, regression, experiments, diagnostics,
/// User CRUD, sessions, and channel cursors (all defaulted).
pub trait UserStore: Send + Sync {
    /// Retrieve a channel cursor value.
    fn get_channel_cursor(&self, _channel: &str, _cursor_key: &str) -> Result<Option<String>> {
        Ok(None)
    }

    /// Save a channel cursor value.
    fn set_channel_cursor(
        &self,
        _channel: &str,
        _cursor_key: &str,
        _cursor_value: &str,
    ) -> Result<()> {
        Ok(())
    }

    /// Create a new user.
    fn create_user(
        &self,
        _email: &str,
        _password_hash: &str,
        _name: &str,
        _role: &str,
    ) -> Result<i64> {
        Ok(0)
    }

    /// Get a user by email.
    fn get_user_by_email(&self, _email: &str) -> Result<Option<UserRow>> {
        Ok(None)
    }

    /// Get a user by ID.
    fn get_user_by_id(&self, _id: i64) -> Result<Option<UserRow>> {
        Ok(None)
    }

    /// List all users.
    fn list_users(&self) -> Result<Vec<UserRow>> {
        Ok(Vec::new())
    }

    /// Update user fields. Returns true if a row was modified.
    fn update_user(
        &self,
        _id: i64,
        _email: Option<&str>,
        _password_hash: Option<&str>,
        _name: Option<&str>,
        _role: Option<&str>,
        _avatar_url: Option<&str>,
    ) -> Result<bool> {
        Ok(false)
    }

    /// Delete a user by ID. Returns true if a row was deleted.
    fn delete_user(&self, _id: i64) -> Result<bool> {
        Ok(false)
    }

    /// Count total users.
    fn count_users(&self) -> Result<i64> {
        Ok(0)
    }

    /// Create a session for a user, returning the session token.
    fn create_session(&self, _user_id: i64, _expires_at: &str) -> Result<String> {
        Ok(String::new())
    }

    /// Get the user associated with a session token.
    fn get_session_user(&self, _token: &str) -> Result<Option<UserRow>> {
        Ok(None)
    }

    /// Delete a session by token.
    fn delete_session(&self, _token: &str) -> Result<()> {
        Ok(())
    }

    /// Cleanup expired and idle sessions, returning how many were removed.
    fn cleanup_expired_sessions(&self) -> Result<usize> {
        Ok(0)
    }

    /// Update the last_active_at timestamp for a session (idle timeout tracking).
    fn touch_session(&self, _token: &str) -> Result<()> {
        Ok(())
    }

    /// Delete all sessions for a user.
    fn delete_user_sessions(&self, _user_id: i64) -> Result<()> {
        Ok(())
    }
}

/// Regression watches and checks (all defaulted).
pub trait RegressionStore: Send + Sync {
    /// Get regression watches by status.
    fn get_regression_watches_by_status(
        &self,
        _status: RegressionWatchStatus,
    ) -> Result<Vec<RegressionWatch>> {
        Ok(Vec::new())
    }

    /// Get all regression watches.
    fn get_all_regression_watches(&self) -> Result<Vec<RegressionWatch>> {
        Ok(Vec::new())
    }

    /// Get regression checks for a watch.
    fn get_regression_checks(&self, _watch_id: i64) -> Result<Vec<RegressionCheck>> {
        Ok(Vec::new())
    }

    /// Create a regression watch. Returns the watch ID.
    fn create_regression_watch(&self, _watch: &RegressionWatch) -> Result<i64> {
        Ok(0)
    }

    /// Record a regression check. Returns the check ID.
    fn add_regression_check(&self, _check: &RegressionCheck) -> Result<i64> {
        Ok(0)
    }

    /// Update a regression watch's status.
    fn update_regression_watch_status(
        &self,
        _id: i64,
        _status: RegressionWatchStatus,
    ) -> Result<()> {
        Ok(())
    }

    /// Get a regression watch by ID.
    fn get_regression_watch(&self, _id: i64) -> Result<Option<RegressionWatch>> {
        Ok(None)
    }

    /// Record a regression check. Returns the check ID.
    fn record_regression_check(&self, _check: &RegressionCheck) -> Result<i64> {
        Ok(0)
    }
}

/// Chat sessions and messages (all defaulted).
pub trait ChatStore: Send + Sync {
    /// Create a new chat session.
    fn create_chat_session(&self, _id: &str, _repo_id: Option<i64>) -> Result<()> {
        Ok(())
    }

    /// Save a chat message to a session.
    fn save_chat_message(
        &self,
        _session_id: &str,
        _role: &str,
        _content: &str,
        _sources_json: Option<&str>,
    ) -> Result<i64> {
        Ok(0)
    }

    /// Get chat message history for a session.
    fn get_chat_history(
        &self,
        _session_id: &str,
        _limit: usize,
    ) -> Result<Vec<claudear_core::types::ChatMessage>> {
        Ok(Vec::new())
    }

    /// List all chat sessions.
    fn list_chat_sessions(&self) -> Result<Vec<claudear_core::types::ChatSession>> {
        Ok(Vec::new())
    }

    /// Delete a chat session and all its messages.
    fn delete_chat_session(&self, _session_id: &str) -> Result<()> {
        Ok(())
    }

    /// Cleanup expired chat sessions older than the given number of days.
    fn cleanup_expired_chat_sessions(&self, _max_age_days: u32) -> Result<usize> {
        Ok(0)
    }
}

/// Supertrait combining all 12 sub-traits. Auto-implemented for any type
/// that implements all sub-traits via the blanket impl below.
pub trait FixAttemptTracker:
    AttemptTracker
    + ActivityStore
    + KnowledgeStore
    + EmbeddingStore
    + ExperimentStore
    + EvaluationStore
    + WebhookStore
    + SimilarityStore
    + RepoStore
    + UserStore
    + RegressionStore
    + ChatStore
{
}

impl<T> FixAttemptTracker for T where
    T: AttemptTracker
        + ActivityStore
        + KnowledgeStore
        + EmbeddingStore
        + ExperimentStore
        + EvaluationStore
        + WebhookStore
        + SimilarityStore
        + RepoStore
        + UserStore
        + RegressionStore
        + ChatStore
{
}

/// A no-op tracker for use in tests that don't need persistence.
///
/// Implements all tracker traits with no-op defaults. Unlike `SqliteTracker`,
/// this is available without the `sqlite` feature flag.
pub struct NoopTracker;

impl AttemptTracker for NoopTracker {
    fn has_attempted(&self, _: &str, _: &str) -> Result<bool> {
        Ok(false)
    }
    fn get_attempted_issue_ids(&self, _: &str) -> Result<HashSet<String>> {
        Ok(HashSet::new())
    }
    fn record_attempt(&self, _: &str, _: &str, _: &str) -> Result<()> {
        Ok(())
    }
    fn record_attempt_with_labels(&self, _: &str, _: &str, _: &str, _: &[String]) -> Result<()> {
        Ok(())
    }
    fn mark_success(&self, _: &str, _: &str, _: &str) -> Result<()> {
        Ok(())
    }
    fn mark_failed(&self, _: &str, _: &str, _: &str) -> Result<()> {
        Ok(())
    }
    fn mark_merged(&self, _: &str, _: &str) -> Result<()> {
        Ok(())
    }
    fn mark_closed(&self, _: &str, _: &str) -> Result<()> {
        Ok(())
    }
    fn mark_resolved(&self, _: &str, _: &str) -> Result<()> {
        Ok(())
    }
    fn get_attempt(&self, _: &str, _: &str) -> Result<Option<FixAttempt>> {
        Ok(None)
    }
    fn get_attempts_by_status(&self, _: FixAttemptStatus) -> Result<Vec<FixAttempt>> {
        Ok(vec![])
    }
    fn get_pending_prs(&self) -> Result<Vec<FixAttempt>> {
        Ok(vec![])
    }
    fn get_attempt_by_pr_url(&self, _: &str) -> Result<Option<FixAttempt>> {
        Ok(None)
    }
    fn reset_attempt(&self, _: &str, _: &str) -> Result<()> {
        Ok(())
    }
    fn get_stats(&self) -> Result<FixAttemptStats> {
        Ok(FixAttemptStats {
            total: 0,
            pending: 0,
            success: 0,
            failed: 0,
            merged: 0,
            closed: 0,
            cannot_fix: 0,
            by_source: Default::default(),
        })
    }
    fn increment_retry(&self, _: &str, _: &str) -> Result<()> {
        Ok(())
    }
    fn mark_cannot_fix(&self, _: &str, _: &str, _: &str) -> Result<()> {
        Ok(())
    }
    fn get_retryable_issues(&self, _: u32) -> Result<Vec<FixAttempt>> {
        Ok(vec![])
    }
    fn prepare_for_retry(&self, _: &str, _: &str) -> Result<()> {
        Ok(())
    }
}

impl ActivityStore for NoopTracker {}
impl KnowledgeStore for NoopTracker {}
impl EmbeddingStore for NoopTracker {}
impl ExperimentStore for NoopTracker {}
impl EvaluationStore for NoopTracker {}
impl WebhookStore for NoopTracker {}
impl SimilarityStore for NoopTracker {}
impl RepoStore for NoopTracker {}
impl UserStore for NoopTracker {}
impl RegressionStore for NoopTracker {}
impl ChatStore for NoopTracker {}

#[cfg(test)]
mod tests {
    use super::*;

    /// Macro to generate a minimal mock struct that implements AttemptTracker
    /// (all 19 required methods) plus one additional sub-trait with empty body
    /// (relying on that trait's default methods).
    macro_rules! mock_for_subtrait {
        ($name:ident, $trait:ident) => {
            struct $name;

            impl AttemptTracker for $name {
                fn has_attempted(&self, _: &str, _: &str) -> Result<bool> {
                    Ok(false)
                }
                fn get_attempted_issue_ids(&self, _: &str) -> Result<HashSet<String>> {
                    Ok(HashSet::new())
                }
                fn record_attempt(&self, _: &str, _: &str, _: &str) -> Result<()> {
                    Ok(())
                }
                fn record_attempt_with_labels(
                    &self,
                    _: &str,
                    _: &str,
                    _: &str,
                    _: &[String],
                ) -> Result<()> {
                    Ok(())
                }
                fn mark_success(&self, _: &str, _: &str, _: &str) -> Result<()> {
                    Ok(())
                }
                fn mark_failed(&self, _: &str, _: &str, _: &str) -> Result<()> {
                    Ok(())
                }
                fn mark_merged(&self, _: &str, _: &str) -> Result<()> {
                    Ok(())
                }
                fn mark_closed(&self, _: &str, _: &str) -> Result<()> {
                    Ok(())
                }
                fn mark_resolved(&self, _: &str, _: &str) -> Result<()> {
                    Ok(())
                }
                fn get_attempt(&self, _: &str, _: &str) -> Result<Option<FixAttempt>> {
                    Ok(None)
                }
                fn get_attempts_by_status(&self, _: FixAttemptStatus) -> Result<Vec<FixAttempt>> {
                    Ok(vec![])
                }
                fn get_pending_prs(&self) -> Result<Vec<FixAttempt>> {
                    Ok(vec![])
                }
                fn get_attempt_by_pr_url(&self, _: &str) -> Result<Option<FixAttempt>> {
                    Ok(None)
                }
                fn reset_attempt(&self, _: &str, _: &str) -> Result<()> {
                    Ok(())
                }
                fn get_stats(&self) -> Result<FixAttemptStats> {
                    Ok(FixAttemptStats {
                        total: 0,
                        pending: 0,
                        success: 0,
                        failed: 0,
                        merged: 0,
                        closed: 0,
                        cannot_fix: 0,
                        by_source: Default::default(),
                    })
                }
                fn increment_retry(&self, _: &str, _: &str) -> Result<()> {
                    Ok(())
                }
                fn mark_cannot_fix(&self, _: &str, _: &str, _: &str) -> Result<()> {
                    Ok(())
                }
                fn get_retryable_issues(&self, _: u32) -> Result<Vec<FixAttempt>> {
                    Ok(vec![])
                }
                fn prepare_for_retry(&self, _: &str, _: &str) -> Result<()> {
                    Ok(())
                }
            }

            impl $trait for $name {}
        };
    }

    mod trait_structure {
        use super::*;

        /// A struct implementing all 8 sub-traits to prove the blanket impl
        /// auto-satisfies FixAttemptTracker.
        struct FullTracker;

        impl AttemptTracker for FullTracker {
            fn has_attempted(&self, _: &str, _: &str) -> Result<bool> {
                Ok(false)
            }
            fn get_attempted_issue_ids(&self, _: &str) -> Result<HashSet<String>> {
                Ok(HashSet::new())
            }
            fn record_attempt(&self, _: &str, _: &str, _: &str) -> Result<()> {
                Ok(())
            }
            fn record_attempt_with_labels(
                &self,
                _: &str,
                _: &str,
                _: &str,
                _: &[String],
            ) -> Result<()> {
                Ok(())
            }
            fn mark_success(&self, _: &str, _: &str, _: &str) -> Result<()> {
                Ok(())
            }
            fn mark_failed(&self, _: &str, _: &str, _: &str) -> Result<()> {
                Ok(())
            }
            fn mark_merged(&self, _: &str, _: &str) -> Result<()> {
                Ok(())
            }
            fn mark_closed(&self, _: &str, _: &str) -> Result<()> {
                Ok(())
            }
            fn mark_resolved(&self, _: &str, _: &str) -> Result<()> {
                Ok(())
            }
            fn get_attempt(&self, _: &str, _: &str) -> Result<Option<FixAttempt>> {
                Ok(None)
            }
            fn get_attempts_by_status(&self, _: FixAttemptStatus) -> Result<Vec<FixAttempt>> {
                Ok(vec![])
            }
            fn get_pending_prs(&self) -> Result<Vec<FixAttempt>> {
                Ok(vec![])
            }
            fn get_attempt_by_pr_url(&self, _: &str) -> Result<Option<FixAttempt>> {
                Ok(None)
            }
            fn reset_attempt(&self, _: &str, _: &str) -> Result<()> {
                Ok(())
            }
            fn get_stats(&self) -> Result<FixAttemptStats> {
                Ok(FixAttemptStats {
                    total: 0,
                    pending: 0,
                    success: 0,
                    failed: 0,
                    merged: 0,
                    closed: 0,
                    cannot_fix: 0,
                    by_source: Default::default(),
                })
            }
            fn increment_retry(&self, _: &str, _: &str) -> Result<()> {
                Ok(())
            }
            fn mark_cannot_fix(&self, _: &str, _: &str, _: &str) -> Result<()> {
                Ok(())
            }
            fn get_retryable_issues(&self, _: u32) -> Result<Vec<FixAttempt>> {
                Ok(vec![])
            }
            fn prepare_for_retry(&self, _: &str, _: &str) -> Result<()> {
                Ok(())
            }
        }
        impl ActivityStore for FullTracker {}
        impl KnowledgeStore for FullTracker {}
        impl EmbeddingStore for FullTracker {}
        impl ExperimentStore for FullTracker {}
        impl EvaluationStore for FullTracker {}
        impl WebhookStore for FullTracker {}
        impl SimilarityStore for FullTracker {}
        impl RepoStore for FullTracker {}
        impl UserStore for FullTracker {}
        impl RegressionStore for FullTracker {}
        impl ChatStore for FullTracker {}

        #[test]
        fn test_blanket_impl_auto_satisfies_fix_attempt_tracker() {
            let tracker = FullTracker;
            let _dyn_ref: &dyn FixAttemptTracker = &tracker;
        }

        #[test]
        fn test_each_subtrait_implementable_independently() {
            // Proves AttemptTracker + ActivityStore compiles without the other sub-traits.
            mock_for_subtrait!(ActivityOnly, ActivityStore);
            let t = ActivityOnly;
            assert_eq!(
                t.record_activity(&ActivityLogEntry::new("x", "y")).unwrap(),
                0
            );
        }

        #[test]
        fn test_noop_tracker_satisfies_fix_attempt_tracker() {
            let tracker = FullTracker;
            let dyn_ref: &dyn FixAttemptTracker = &tracker;
            // Exercise a method through the dyn ref to prove it works.
            assert!(!dyn_ref.has_attempted("s", "i").unwrap());
        }
    }

    mod activity_store_defaults {
        use super::*;

        mock_for_subtrait!(MockActivity, ActivityStore);

        #[test]
        fn test_default_record_activity_returns_zero() {
            let t = MockActivity;
            let entry = ActivityLogEntry::new("test", "message");
            assert_eq!(t.record_activity(&entry).unwrap(), 0);
        }

        #[test]
        fn test_default_get_recent_activities_returns_empty() {
            let t = MockActivity;
            assert!(t.get_recent_activities(100).unwrap().is_empty());
        }

        #[test]
        fn test_default_record_execution_returns_zero() {
            let t = MockActivity;
            let exec = AgentExecution::new();
            assert_eq!(t.record_execution(&exec).unwrap(), 0);
        }

        #[test]
        fn test_default_get_analytics_summary_returns_default() {
            let t = MockActivity;
            let summary = t.get_analytics_summary().unwrap();
            assert_eq!(summary.success_rate, 0.0);
        }

        #[test]
        fn test_default_get_open_prs_returns_empty() {
            let t = MockActivity;
            assert!(t.get_open_prs().unwrap().is_empty());
        }

        #[test]
        fn test_default_get_pr_analytics_returns_default() {
            let t = MockActivity;
            let analytics = t.get_pr_analytics().unwrap();
            assert_eq!(analytics.total, 0);
        }

        #[test]
        fn test_default_quality_score_update_succeeds() {
            let t = MockActivity;
            assert!(t.update_pr_fix_quality_score("url", 0.9).is_ok());
        }

        #[test]
        fn test_default_record_pr_review_returns_zero() {
            let t = MockActivity;
            let review = PrReviewRecord::new("https://github.com/org/repo/pull/1");
            assert_eq!(t.record_pr_review(&review).unwrap(), 0);
        }

        #[test]
        fn test_default_record_error_pattern_returns_zero() {
            let t = MockActivity;
            let pattern = ErrorPattern::new("hash123");
            assert_eq!(t.record_error_pattern(&pattern).unwrap(), 0);
        }

        #[test]
        fn test_default_record_metric_returns_zero() {
            let t = MockActivity;
            let metric = ProcessingMetric::new("queue_depth", 42.0);
            assert_eq!(t.record_metric(&metric).unwrap(), 0);
        }

        #[test]
        fn test_default_get_recent_activities_filtered_returns_empty() {
            let t = MockActivity;
            assert!(t
                .get_recent_activities_filtered(50, None)
                .unwrap()
                .is_empty());
            assert!(t
                .get_recent_activities_filtered(50, Some("linear"))
                .unwrap()
                .is_empty());
        }

        #[test]
        fn test_default_get_attempt_by_id_returns_none() {
            let t = MockActivity;
            assert!(t.get_attempt_by_id(1).unwrap().is_none());
            assert!(t.get_attempt_by_id(999).unwrap().is_none());
        }

        #[test]
        fn test_default_get_executions_for_attempt_returns_empty() {
            let t = MockActivity;
            assert!(t.get_executions_for_attempt(1).unwrap().is_empty());
        }

        #[test]
        fn test_default_get_reviews_for_attempt_returns_empty() {
            let t = MockActivity;
            assert!(t.get_reviews_for_attempt(1).unwrap().is_empty());
        }

        #[test]
        fn test_default_get_error_patterns_returns_empty() {
            let t = MockActivity;
            assert!(t.get_error_patterns(100).unwrap().is_empty());
        }

        #[test]
        fn test_default_get_metrics_returns_empty() {
            let t = MockActivity;
            assert!(t.get_metrics("queue_depth", None, 100).unwrap().is_empty());
            assert!(t
                .get_metrics("processing_time", Some(Utc::now()), 50)
                .unwrap()
                .is_empty());
        }

        #[test]
        fn test_default_get_avg_time_to_pr_returns_none() {
            let t = MockActivity;
            assert!(t.get_avg_time_to_pr().unwrap().is_none());
        }

        #[test]
        fn test_default_get_rejection_reasons_returns_empty() {
            let t = MockActivity;
            assert!(t.get_rejection_reasons(10).unwrap().is_empty());
        }

        #[test]
        fn test_default_get_agent_spawn_count_returns_zero() {
            let t = MockActivity;
            assert_eq!(t.get_agent_spawn_count("2024-01-01T00:00:00Z").unwrap(), 0);
        }

        #[test]
        fn test_default_get_cost_estimate_returns_default() {
            let t = MockActivity;
            let est = t
                .get_cost_estimate("2024-01-01T00:00:00Z", 100.0, "month")
                .unwrap();
            assert_eq!(est.total_cost, 0.0);
            assert_eq!(est.avg_cost_per_fix, 0.0);
            assert_eq!(est.fix_count, 0);
        }

        #[test]
        fn test_default_get_mttr_trend_returns_empty() {
            let t = MockActivity;
            assert!(t.get_mttr_trend(4).unwrap().is_empty());
        }

        #[test]
        fn test_default_get_repo_leaderboard_returns_empty() {
            let t = MockActivity;
            assert!(t.get_repo_leaderboard().unwrap().is_empty());
        }

        #[test]
        fn test_default_get_complexity_time_savings_returns_default() {
            let t = MockActivity;
            let savings = t
                .get_complexity_time_savings("2024-01-01T00:00:00Z", 150.0, "month")
                .unwrap();
            assert_eq!(savings.merged_count, 0);
            assert_eq!(savings.hours_saved, 0.0);
            assert_eq!(savings.cost_saved, 0.0);
        }

        #[test]
        fn test_default_list_prs_returns_empty() {
            let t = MockActivity;
            assert!(t.list_prs(None, 50).unwrap().is_empty());
            assert!(t.list_prs(Some("open"), 10).unwrap().is_empty());
        }
    }

    mod knowledge_store_defaults {
        use super::*;

        mock_for_subtrait!(MockKnowledge, KnowledgeStore);

        #[test]
        fn test_default_store_feedback_outcome_returns_zero() {
            let t = MockKnowledge;
            let outcome = FixOutcome {
                id: 0,
                attempt_id: 0,
                source: "s".into(),
                issue_id: "i".into(),
                issue_text: "t".into(),
                prompt_used: "p".into(),
                outcome: claudear_core::types::Outcome::Merged,
                error_type: None,
                learnings: None,
                keywords: vec![],
                embedding: None,
                created_at: Utc::now(),
            };
            assert_eq!(t.store_feedback_outcome(&outcome).unwrap(), 0);
        }

        #[test]
        fn test_default_get_feedback_outcomes_returns_empty() {
            let t = MockKnowledge;
            assert!(t.get_feedback_outcomes(None, 10).unwrap().is_empty());
            assert!(t
                .get_feedback_outcomes(Some("linear"), 10)
                .unwrap()
                .is_empty());
        }

        #[test]
        fn test_default_get_feedback_outcome_by_attempt_returns_none() {
            let t = MockKnowledge;
            assert!(t.get_feedback_outcome_by_attempt(999).unwrap().is_none());
        }

        #[test]
        fn test_default_store_qa_knowledge_returns_zero() {
            let t = MockKnowledge;
            let entry = QaKnowledgeEntry {
                id: 0,
                source: "s".into(),
                repo: None,
                issue_id: "i".into(),
                short_id: "S".into(),
                question_text: "q".into(),
                question_norm: "q".into(),
                question_embedding: None,
                answer_text: "a".into(),
                answer_norm: "a".into(),
                answer_embedding: None,
                channel: "c".into(),
                responder: None,
                correlation_id: "x".into(),
                asked_at: Utc::now(),
                answered_at: Utc::now(),
                success_count: 0,
                failure_count: 0,
                last_used_at: None,
                metadata: None,
            };
            assert_eq!(t.store_qa_knowledge(&entry).unwrap(), 0);
        }

        #[test]
        fn test_default_find_similar_qa_returns_empty() {
            let t = MockKnowledge;
            assert!(t
                .find_similar_qa_scoped("s", None, "q", None, 0.0, 10)
                .unwrap()
                .is_empty());
            assert!(t
                .find_similar_qa_global("q", None, 0.0, 10)
                .unwrap()
                .is_empty());
        }

        #[test]
        fn test_default_learning_methods_return_no_ops() {
            let t = MockKnowledge;
            assert!(t.update_feedback_learnings(1, "learnings").is_ok());
            assert!(t.get_diff_analyses_for_repo("repo", 10).unwrap().is_empty());
            assert!(t.get_promoted_instructions("repo").unwrap().is_empty());
            assert!(t.get_repo_knowledge("repo").unwrap().is_empty());
            assert!(t
                .get_repo_knowledge_by_key("repo", "key")
                .unwrap()
                .is_empty());
        }

        #[test]
        fn test_default_review_pattern_methods_return_no_ops() {
            let t = MockKnowledge;
            assert!(t.get_review_patterns("repo", 10).unwrap().is_empty());
            assert!(t
                .get_review_patterns_by_category(
                    "repo",
                    claudear_core::types::ReviewCategory::Other
                )
                .unwrap()
                .is_empty());
        }

        #[test]
        fn test_default_strategy_methods_return_no_ops() {
            let t = MockKnowledge;
            assert!(t.get_successful_strategies("repo", 10).unwrap().is_empty());
        }

        #[test]
        fn test_default_cluster_methods_return_no_ops() {
            let t = MockKnowledge;
            assert!(t.get_active_clusters("sentry").unwrap().is_empty());
            assert!(t.update_cluster_resolution(1, "issue", 1).is_ok());
            assert!(t
                .get_recent_issue_arrivals("sentry", 30)
                .unwrap()
                .is_empty());
        }

        #[test]
        fn test_default_record_qa_usage_returns_zero() {
            let t = MockKnowledge;
            assert_eq!(t.record_qa_usage(1, 2, "direct", 0.95).unwrap(), 0);
        }

        #[test]
        fn test_default_update_qa_outcome_stats_succeeds() {
            let t = MockKnowledge;
            assert!(t.update_qa_outcome_stats(1, true).is_ok());
            assert!(t.update_qa_outcome_stats(2, false).is_ok());
        }

        #[test]
        fn test_default_update_qa_outcome_stats_for_attempt_succeeds() {
            let t = MockKnowledge;
            assert!(t.update_qa_outcome_stats_for_attempt(1, true).is_ok());
            assert!(t.update_qa_outcome_stats_for_attempt(2, false).is_ok());
        }

        #[test]
        fn test_default_store_diff_analysis_returns_zero() {
            let t = MockKnowledge;
            let analysis = claudear_core::types::DiffAnalysis {
                id: 0,
                attempt_id: 1,
                pr_url: "https://github.com/org/repo/pull/1".into(),
                scm_repo: "org/repo".into(),
                pr_number: 1,
                files_changed: vec!["src/main.rs".into()],
                file_types: std::collections::HashMap::new(),
                change_categories: vec![],
                diff_summary: "Fixed a bug".into(),
                created_at: Utc::now(),
            };
            assert_eq!(t.store_diff_analysis(&analysis).unwrap(), 0);
        }

        #[test]
        fn test_default_upsert_promoted_instruction_returns_zero() {
            let t = MockKnowledge;
            let instruction = claudear_core::types::PromotedInstruction {
                id: 0,
                repo: "org/repo".into(),
                source_type: "qa".into(),
                instruction_text: "Always add tests".into(),
                occurrence_count: 5,
                confidence: 0.9,
                is_active: true,
                created_at: Utc::now(),
                updated_at: Utc::now(),
            };
            assert_eq!(t.upsert_promoted_instruction(&instruction).unwrap(), 0);
        }

        #[test]
        fn test_default_upsert_repo_knowledge_returns_zero() {
            let t = MockKnowledge;
            let entry = claudear_core::types::RepoKnowledge {
                id: 0,
                repo: "org/repo".into(),
                knowledge_key: "test_framework".into(),
                knowledge_value: "pytest".into(),
                source_type: "review".into(),
                confidence: 0.8,
                occurrence_count: 3,
                created_at: Utc::now(),
                updated_at: Utc::now(),
            };
            assert_eq!(t.upsert_repo_knowledge(&entry).unwrap(), 0);
        }

        #[test]
        fn test_default_upsert_review_pattern_returns_zero() {
            let t = MockKnowledge;
            let pattern = claudear_core::types::ReviewPattern {
                id: 0,
                scm_repo: "org/repo".into(),
                category: claudear_core::types::ReviewCategory::MissingTests,
                pattern_text: "Add unit tests for new functions".into(),
                example_comments: vec!["Please add tests".into()],
                occurrence_count: 4,
                promoted_to_instruction: false,
                created_at: Utc::now(),
                updated_at: Utc::now(),
            };
            assert_eq!(t.upsert_review_pattern(&pattern).unwrap(), 0);
        }

        #[test]
        fn test_default_store_strategy_fingerprint_returns_zero() {
            let t = MockKnowledge;
            let fp = claudear_core::types::StrategyFingerprint {
                id: 0,
                attempt_id: 1,
                files_explored: vec!["src/lib.rs".into()],
                tests_run: 5,
                tools_used: std::collections::HashMap::new(),
                fix_approach: "direct_edit".into(),
                strategy_summary: "Edited source directly".into(),
                fix_quality_score: None,
                created_at: Utc::now(),
            };
            assert_eq!(t.store_strategy_fingerprint(&fp).unwrap(), 0);
        }

        #[test]
        fn test_default_store_issue_cluster_returns_zero() {
            let t = MockKnowledge;
            let cluster = claudear_core::types::IssueCluster {
                id: 0,
                cluster_key: "TypeError::main".into(),
                source: "sentry".into(),
                issue_ids: vec!["SENTRY-1".into(), "SENTRY-2".into()],
                window_start: Utc::now(),
                window_end: Utc::now(),
                resolved_by_issue_id: None,
                resolved_by_attempt_id: None,
                status: "active".into(),
                created_at: Utc::now(),
            };
            assert_eq!(t.store_issue_cluster(&cluster).unwrap(), 0);
        }

        #[test]
        fn test_default_store_content_cluster_returns_zero() {
            let t = MockKnowledge;
            let cluster = claudear_core::types::ContentCluster {
                id: 0,
                cluster_key: "TypeError::app.main".into(),
                source: "sentry".into(),
                representative_issue_id: "SENTRY-1".into(),
                issue_ids: vec!["SENTRY-1".into(), "SENTRY-2".into()],
                error_type: Some("TypeError".into()),
                culprit: Some("app.main".into()),
                avg_similarity: 0.85,
                status: "active".into(),
                created_at: Utc::now(),
            };
            assert_eq!(t.store_content_cluster(&cluster).unwrap(), 0);
        }

        #[test]
        fn test_default_get_active_content_clusters_returns_empty() {
            let t = MockKnowledge;
            assert!(t.get_active_content_clusters("sentry").unwrap().is_empty());
        }

        #[test]
        fn test_default_resolve_content_cluster_succeeds() {
            let t = MockKnowledge;
            assert!(t.resolve_content_cluster(1).is_ok());
        }

        #[test]
        fn test_default_store_severity_score_succeeds() {
            let t = MockKnowledge;
            let score = claudear_core::types::SeverityScore::default();
            let blast = claudear_core::types::BlastRadius::default();
            assert!(t
                .store_severity_score("sentry", "SENTRY-1", &score, blast)
                .is_ok());
        }

        #[test]
        fn test_default_record_suppression_succeeds() {
            let t = MockKnowledge;
            assert!(t
                .record_suppression("sentry", "SENTRY-1", "flaky_test_rule", "Flaky test noise")
                .is_ok());
        }

        #[test]
        fn test_default_upsert_cross_repo_correlation_returns_valid_struct() {
            let t = MockKnowledge;
            let corr = t
                .upsert_cross_repo_correlation("org/repo-a", "org/repo-b", 24)
                .unwrap();
            assert_eq!(corr.id, 0);
            assert_eq!(corr.repo_a, "org/repo-a");
            assert_eq!(corr.repo_b, "org/repo-b");
            assert_eq!(corr.correlation_count, 0);
            assert_eq!(corr.window_hours, 24);
        }

        #[test]
        fn test_default_get_cross_repo_correlations_returns_empty() {
            let t = MockKnowledge;
            assert!(t.get_cross_repo_correlations(2, 168).unwrap().is_empty());
        }
    }

    mod embedding_store_defaults {
        use super::*;

        mock_for_subtrait!(MockEmbedding, EmbeddingStore);

        #[test]
        fn test_default_get_recent_attempts_since_returns_empty() {
            let t = MockEmbedding;
            let cutoff = Utc::now();
            assert!(t.get_recent_attempts_since(&cutoff).unwrap().is_empty());
        }
    }

    mod repo_store_defaults {
        use super::*;

        mock_for_subtrait!(MockRepo, RepoStore);

        #[test]
        fn test_default_index_stats_returns_zeros() {
            let t = MockRepo;
            let stats = t.get_index_stats().unwrap();
            assert_eq!(stats.repo_count, 0);
            assert_eq!(stats.file_count, 0);
            assert!(stats.last_indexed_at.is_none());
        }

        #[test]
        fn test_default_inference_stats_returns_zeros() {
            let t = MockRepo;
            let stats = t.get_inference_stats().unwrap();
            assert_eq!(stats.total_attempts, 0);
            assert_eq!(stats.accuracy, 0.0);
        }

        #[test]
        fn test_default_list_indexed_repos_returns_empty() {
            let t = MockRepo;
            assert!(t.list_indexed_repos().unwrap().is_empty());
        }

        #[test]
        fn test_default_get_indexing_progress_returns_idle_defaults() {
            let t = MockRepo;
            let progress = t.get_indexing_progress().unwrap();
            assert_eq!(progress.status, "idle");
            assert_eq!(progress.total_repos, 0);
            assert_eq!(progress.indexed_repos, 0);
            assert!(progress.current_repo.is_none());
            assert_eq!(progress.current_repo_files, 0);
            assert_eq!(progress.total_files_indexed, 0);
            assert!(progress.started_at.is_none());
            assert!(progress.updated_at.is_none());
        }

        #[test]
        fn test_default_subscribe_indexing_progress_returns_valid_receiver() {
            let t = MockRepo;
            let rx = t.subscribe_indexing_progress();
            let val = rx.borrow();
            assert_eq!(val.status, "idle");
            assert_eq!(val.total_repos, 0);
            assert_eq!(val.indexed_repos, 0);
        }

        #[test]
        fn test_default_list_all_dependencies_returns_empty() {
            let t = MockRepo;
            assert!(t.list_all_dependencies().unwrap().is_empty());
        }

        #[test]
        fn test_default_get_inference_history_returns_empty() {
            let t = MockRepo;
            assert!(t.get_inference_history(50).unwrap().is_empty());
        }

        #[test]
        fn test_default_has_dependency_returns_false() {
            let t = MockRepo;
            assert!(!t.has_dependency("repo_a", "repo_b").unwrap());
        }
    }

    mod user_store_defaults {
        use super::*;

        mock_for_subtrait!(MockUser, UserStore);

        #[test]
        fn test_default_channel_cursor_returns_none() {
            let t = MockUser;
            assert!(t.get_channel_cursor("ch", "key").unwrap().is_none());
            // set_channel_cursor should succeed silently
            assert!(t.set_channel_cursor("ch", "key", "val").is_ok());
        }
    }

    mod regression_store_defaults {
        use super::*;

        mock_for_subtrait!(MockRegression, RegressionStore);

        #[test]
        fn test_default_get_regression_watches_returns_empty() {
            let t = MockRegression;
            assert!(t
                .get_regression_watches_by_status(RegressionWatchStatus::AwaitingRelease)
                .unwrap()
                .is_empty());
            assert!(t.get_all_regression_watches().unwrap().is_empty());
        }

        #[test]
        fn test_default_get_regression_checks_returns_empty() {
            let t = MockRegression;
            assert!(t.get_regression_checks(1).unwrap().is_empty());
        }

        #[test]
        fn test_default_create_regression_watch_returns_zero() {
            let t = MockRegression;
            let watch = RegressionWatch {
                id: 0,
                issue_type: claudear_core::types::IssueType::LinearBug,
                issue_id: "I-1".into(),
                fix_attempt_id: 1,
                status: RegressionWatchStatus::AwaitingRelease,
                pr_merged_at: None,
                monitoring_started_at: None,
                resolved_at: None,
                regressed_at: None,
                created_at: Utc::now(),
            };
            assert_eq!(t.create_regression_watch(&watch).unwrap(), 0);
        }

        #[test]
        fn test_default_add_regression_check_returns_zero() {
            let t = MockRegression;
            let check = RegressionCheck {
                id: 0,
                regression_watch_id: 1,
                issue_still_exists: false,
                checked_at: Some(Utc::now()),
                check_details: None,
                created_at: Utc::now(),
            };
            assert_eq!(t.add_regression_check(&check).unwrap(), 0);
        }

        #[test]
        fn test_default_update_regression_watch_status_succeeds() {
            let t = MockRegression;
            assert!(t
                .update_regression_watch_status(1, RegressionWatchStatus::Monitoring)
                .is_ok());
        }

        #[test]
        fn test_default_get_regression_watch_returns_none() {
            let t = MockRegression;
            assert!(t.get_regression_watch(1).unwrap().is_none());
            assert!(t.get_regression_watch(999).unwrap().is_none());
        }

        #[test]
        fn test_default_record_regression_check_returns_zero() {
            let t = MockRegression;
            let check = RegressionCheck {
                id: 0,
                regression_watch_id: 1,
                issue_still_exists: true,
                checked_at: Some(Utc::now()),
                check_details: Some("Regression detected".into()),
                created_at: Utc::now(),
            };
            assert_eq!(t.record_regression_check(&check).unwrap(), 0);
        }
    }

    mod chat_store_defaults {
        use super::*;

        mock_for_subtrait!(MockChat, ChatStore);

        #[test]
        fn test_default_create_chat_session_succeeds() {
            let t = MockChat;
            assert!(t.create_chat_session("session-1", None).is_ok());
            assert!(t.create_chat_session("session-2", Some(42)).is_ok());
        }

        #[test]
        fn test_default_save_chat_message_returns_zero() {
            let t = MockChat;
            assert_eq!(
                t.save_chat_message("session-1", "user", "hello", None)
                    .unwrap(),
                0
            );
            assert_eq!(
                t.save_chat_message("session-1", "assistant", "hi", Some("[]"))
                    .unwrap(),
                0
            );
        }

        #[test]
        fn test_default_get_chat_history_returns_empty() {
            let t = MockChat;
            assert!(t.get_chat_history("session-1", 50).unwrap().is_empty());
        }

        #[test]
        fn test_default_list_chat_sessions_returns_empty() {
            let t = MockChat;
            assert!(t.list_chat_sessions().unwrap().is_empty());
        }

        #[test]
        fn test_default_delete_chat_session_succeeds() {
            let t = MockChat;
            assert!(t.delete_chat_session("session-1").is_ok());
        }

        #[test]
        fn test_default_cleanup_expired_chat_sessions_returns_zero() {
            let t = MockChat;
            assert_eq!(t.cleanup_expired_chat_sessions(30).unwrap(), 0);
        }
    }

    mod experiment_store_defaults {
        use super::*;

        mock_for_subtrait!(MockExperiment, ExperimentStore);

        #[test]
        fn test_default_save_experiment_returns_zero() {
            let t = MockExperiment;
            let experiment = PromptExperiment {
                id: 0,
                experiment_name: "test".into(),
                variant: "a".into(),
                prompt_template: "template".into(),
                prompt_hash: "hash".into(),
                active: true,
                success_count: 0,
                failure_count: 0,
                avg_time_to_merge: None,
                avg_review_score: None,
                created_at: Utc::now(),
            };
            assert_eq!(t.save_experiment(&experiment).unwrap(), 0);
        }

        #[test]
        fn test_default_update_experiment_returns_false() {
            let t = MockExperiment;
            assert!(!t
                .update_experiment(1, "test", "a", "template", "hash", true)
                .unwrap());
        }

        #[test]
        fn test_default_update_experiment_stats_succeeds() {
            let t = MockExperiment;
            assert!(t.update_experiment_stats(1, true, Some(30.0)).is_ok());
            assert!(t.update_experiment_stats(2, false, None).is_ok());
        }

        #[test]
        fn test_default_get_active_experiments_returns_empty() {
            let t = MockExperiment;
            assert!(t.get_active_experiments().unwrap().is_empty());
        }
    }

    mod evaluation_store_defaults {
        use super::*;

        mock_for_subtrait!(MockEvaluation, EvaluationStore);

        #[test]
        fn test_default_store_eval_snapshot_returns_zero() {
            let t = MockEvaluation;
            let snapshot = claudear_core::types::EvalSnapshot {
                category: claudear_core::types::EvalCategory::Test,
                tool_name: "cargo".into(),
                exit_code: 0,
                passed: 10,
                failed: 0,
                skipped: 0,
                warnings: 0,
                errors: 0,
                diagnostics: vec![],
                raw_output: String::new(),
                duration_secs: 1.0,
                line_coverage_pct: None,
                branch_coverage_pct: None,
            };
            assert_eq!(t.store_eval_snapshot(None, "pre", &snapshot).unwrap(), 0);
            assert_eq!(
                t.store_eval_snapshot(Some(1), "post", &snapshot).unwrap(),
                0
            );
        }

        #[test]
        fn test_default_store_eval_delta_returns_zero() {
            let t = MockEvaluation;
            let snapshot = claudear_core::types::EvalSnapshot {
                category: claudear_core::types::EvalCategory::Test,
                tool_name: "cargo".into(),
                exit_code: 0,
                passed: 10,
                failed: 0,
                skipped: 0,
                warnings: 0,
                errors: 0,
                diagnostics: vec![],
                raw_output: String::new(),
                duration_secs: 1.0,
                line_coverage_pct: None,
                branch_coverage_pct: None,
            };
            let delta = claudear_core::types::EvalDelta::compute(snapshot.clone(), snapshot);
            assert_eq!(t.store_eval_delta(None, "org/repo", &delta).unwrap(), 0);
        }
    }

    mod webhook_store_defaults {
        use super::*;

        mock_for_subtrait!(MockWebhook, WebhookStore);

        #[test]
        fn test_default_check_and_record_delivery_returns_false() {
            let t = MockWebhook;
            assert!(!t.check_and_record_delivery("delivery-1", "source").unwrap());
        }

        #[test]
        fn test_default_cleanup_old_deliveries_returns_zero() {
            let t = MockWebhook;
            assert_eq!(t.cleanup_old_deliveries(24).unwrap(), 0);
        }
    }

    mod similarity_store_defaults {
        use super::*;

        mock_for_subtrait!(MockSimilarity, SimilarityStore);

        #[test]
        fn test_default_store_similar_issues_batch_succeeds() {
            let t = MockSimilarity;
            assert!(t.store_similar_issues_batch(&[]).is_ok());
        }

        #[test]
        fn test_default_store_similar_issue_returns_zero() {
            let t = MockSimilarity;
            let similar = SimilarIssue {
                id: 0,
                source_issue_id: "ISSUE-1".into(),
                similar_issue_id: "ISSUE-2".into(),
                similarity_score: 0.85,
                computed_at: Utc::now(),
            };
            assert_eq!(t.store_similar_issue(&similar).unwrap(), 0);
        }

        #[test]
        fn test_default_find_similar_issues_returns_empty() {
            let t = MockSimilarity;
            assert!(t
                .find_similar_issues("ISSUE-1", 0.5, 10)
                .unwrap()
                .is_empty());
        }
    }

    mod user_store_extended {
        use super::*;

        mock_for_subtrait!(MockUserExt, UserStore);

        #[test]
        fn test_default_create_user_returns_zero() {
            let t = MockUserExt;
            assert_eq!(
                t.create_user("user@example.com", "hash", "User", "admin")
                    .unwrap(),
                0
            );
        }

        #[test]
        fn test_default_get_user_by_email_returns_none() {
            let t = MockUserExt;
            assert!(t.get_user_by_email("user@example.com").unwrap().is_none());
        }

        #[test]
        fn test_default_get_user_by_id_returns_none() {
            let t = MockUserExt;
            assert!(t.get_user_by_id(1).unwrap().is_none());
            assert!(t.get_user_by_id(999).unwrap().is_none());
        }

        #[test]
        fn test_default_list_users_returns_empty() {
            let t = MockUserExt;
            assert!(t.list_users().unwrap().is_empty());
        }

        #[test]
        fn test_default_update_user_returns_false() {
            let t = MockUserExt;
            assert!(!t
                .update_user(1, Some("new@email.com"), None, None, None, None)
                .unwrap());
            assert!(!t
                .update_user(
                    1,
                    None,
                    Some("hash"),
                    Some("Name"),
                    Some("user"),
                    Some("url")
                )
                .unwrap());
        }

        #[test]
        fn test_default_delete_user_returns_false() {
            let t = MockUserExt;
            assert!(!t.delete_user(1).unwrap());
        }

        #[test]
        fn test_default_count_users_returns_zero() {
            let t = MockUserExt;
            assert_eq!(t.count_users().unwrap(), 0);
        }

        #[test]
        fn test_default_create_session_returns_empty_string() {
            let t = MockUserExt;
            assert_eq!(t.create_session(1, "2025-12-31T23:59:59Z").unwrap(), "");
        }

        #[test]
        fn test_default_get_session_user_returns_none() {
            let t = MockUserExt;
            assert!(t.get_session_user("token123").unwrap().is_none());
        }

        #[test]
        fn test_default_delete_session_succeeds() {
            let t = MockUserExt;
            assert!(t.delete_session("token123").is_ok());
        }

        #[test]
        fn test_default_cleanup_expired_sessions_returns_zero() {
            let t = MockUserExt;
            assert_eq!(t.cleanup_expired_sessions().unwrap(), 0);
        }

        #[test]
        fn test_default_touch_session_succeeds() {
            let t = MockUserExt;
            assert!(t.touch_session("token123").is_ok());
        }

        #[test]
        fn test_default_delete_user_sessions_succeeds() {
            let t = MockUserExt;
            assert!(t.delete_user_sessions(1).is_ok());
        }
    }

    mod embedding_store_extended {
        use super::*;

        mock_for_subtrait!(MockEmbeddingExt, EmbeddingStore);

        fn test_issue_embedding() -> IssueEmbedding {
            IssueEmbedding {
                id: 0,
                source: "test".into(),
                issue_id: "issue-1".into(),
                short_id: None,
                title: None,
                description: None,
                url: None,
                priority: None,
                status: None,
                labels: None,
                embedding: None,
                embedding_model: None,
                created_at: Utc::now(),
                updated_at: None,
            }
        }

        #[test]
        fn test_default_store_issue_returns_zero() {
            let t = MockEmbeddingExt;
            let issue = test_issue_embedding();
            assert_eq!(t.store_issue(&issue).unwrap(), 0);
        }

        #[test]
        fn test_default_store_embedding_returns_zero() {
            let t = MockEmbeddingExt;
            let emb = test_issue_embedding();
            assert_eq!(t.store_embedding(&emb).unwrap(), 0);
        }

        #[test]
        fn test_default_store_embeddings_batch_succeeds() {
            let t = MockEmbeddingExt;
            assert!(t.store_embeddings_batch(&[]).is_ok());
        }

        #[test]
        fn test_default_get_embedding_returns_none() {
            let t = MockEmbeddingExt;
            assert!(t.get_embedding("source", "issue").unwrap().is_none());
        }

        #[test]
        fn test_default_get_all_embeddings_returns_empty() {
            let t = MockEmbeddingExt;
            assert!(t.get_all_embeddings(None, None, None).unwrap().is_empty());
            assert!(t
                .get_all_embeddings(Some("source"), Some(10), Some(0))
                .unwrap()
                .is_empty());
        }

        #[test]
        fn test_default_find_similar_issues_vector_returns_none() {
            let t = MockEmbeddingExt;
            let query = vec![0.1f32; 384];
            assert!(t
                .find_similar_issues_vector(&query, "source", None, 0.7, 10)
                .unwrap()
                .is_none());
        }

        #[test]
        fn test_default_find_similar_outcomes_vector_returns_none() {
            let t = MockEmbeddingExt;
            let query = vec![0.1f32; 384];
            assert!(t
                .find_similar_outcomes_vector(&query, 0.7, 10)
                .unwrap()
                .is_none());
        }

        #[test]
        fn test_default_search_code_chunks_returns_empty() {
            let t = MockEmbeddingExt;
            let query = vec![0.1f32; 384];
            assert!(t.search_code_chunks(&query, None, 10).unwrap().is_empty());
            assert!(t
                .search_code_chunks(&query, Some(1), 10)
                .unwrap()
                .is_empty());
        }

        #[test]
        fn test_default_list_issues_returns_empty() {
            let t = MockEmbeddingExt;
            assert!(t.list_issues(None, 10, 0).unwrap().is_empty());
            assert!(t.list_issues(Some("source"), 10, 0).unwrap().is_empty());
        }

        #[test]
        fn test_default_count_issues_returns_zero() {
            let t = MockEmbeddingExt;
            assert_eq!(t.count_issues(None).unwrap(), 0);
            assert_eq!(t.count_issues(Some("source")).unwrap(), 0);
        }

        #[test]
        fn test_default_record_inference_attempt_returns_zero() {
            let t = MockEmbeddingExt;
            assert_eq!(
                t.record_inference_attempt(
                    "issue",
                    "source",
                    &["file.rs".into()],
                    &["main".into()],
                    &["error".into()],
                    Some(1),
                    "high",
                    "matched by filename",
                    Some(42),
                )
                .unwrap(),
                0
            );
        }

        #[test]
        fn test_default_record_inference_feedback_succeeds() {
            let t = MockEmbeddingExt;
            assert!(t
                .record_inference_feedback(1, true, Some(1), "user")
                .is_ok());
            assert!(t.record_inference_feedback(2, false, None, "auto").is_ok());
        }
    }

    mod activity_store_extended {
        use super::*;

        mock_for_subtrait!(MockActivityExt, ActivityStore);

        #[test]
        fn test_default_upsert_pr_returns_zero() {
            let t = MockActivityExt;
            let pr = PrRecord::new("https://github.com/org/repo/pull/1", "org/repo", 1);
            assert_eq!(t.upsert_pr(&pr).unwrap(), 0);
        }

        #[test]
        fn test_default_get_pr_returns_none() {
            let t = MockActivityExt;
            assert!(t
                .get_pr("https://github.com/org/repo/pull/1")
                .unwrap()
                .is_none());
        }

        #[test]
        fn test_default_update_pr_status_succeeds() {
            let t = MockActivityExt;
            assert!(t
                .update_pr_status("https://github.com/org/repo/pull/1", "closed")
                .is_ok());
        }

        #[test]
        fn test_default_list_attempts_returns_empty() {
            let t = MockActivityExt;
            assert!(t.list_attempts(None, None, 50, 0).unwrap().is_empty());
            assert!(t
                .list_attempts(Some("success"), Some("linear"), 10, 0)
                .unwrap()
                .is_empty());
        }

        #[test]
        fn test_default_count_attempts_returns_zero() {
            let t = MockActivityExt;
            assert_eq!(t.count_attempts(None, None).unwrap(), 0);
            assert_eq!(t.count_attempts(Some("failed"), Some("sentry")).unwrap(), 0);
        }

        #[test]
        fn test_default_list_recent_attempts_returns_empty() {
            let t = MockActivityExt;
            assert!(t.list_recent_attempts(10).unwrap().is_empty());
        }

        #[test]
        fn test_default_list_attempts_since_returns_empty() {
            let t = MockActivityExt;
            assert!(t.list_attempts_since(Utc::now()).unwrap().is_empty());
        }

        #[test]
        fn test_default_get_most_recent_merged_attempt_for_repo_returns_none() {
            let t = MockActivityExt;
            assert!(t
                .get_most_recent_merged_attempt_for_repo("org/repo")
                .unwrap()
                .is_none());
        }

        #[test]
        fn test_default_get_execution_for_attempt_returns_none() {
            let t = MockActivityExt;
            assert!(t.get_execution_for_attempt(1, 1).unwrap().is_none());
        }

        #[test]
        fn test_default_record_activities_batch_returns_zero() {
            let t = MockActivityExt;
            assert_eq!(t.record_activities_batch(&[]).unwrap(), 0);
        }

        #[test]
        fn test_default_prune_old_activities_returns_zero() {
            let t = MockActivityExt;
            assert_eq!(t.prune_old_activities(30).unwrap(), 0);
        }

        #[test]
        fn test_default_prune_old_metrics_returns_zero() {
            let t = MockActivityExt;
            assert_eq!(t.prune_old_metrics(30).unwrap(), 0);
        }

        #[test]
        fn test_default_get_activity_type_counts_since_returns_empty() {
            let t = MockActivityExt;
            assert!(t
                .get_activity_type_counts_since(Utc::now())
                .unwrap()
                .is_empty());
        }

        #[test]
        fn test_default_get_metric_counts_since_returns_empty() {
            let t = MockActivityExt;
            assert!(t
                .get_metric_counts_since(&["queue_depth"], Utc::now())
                .unwrap()
                .is_empty());
        }

        #[test]
        fn test_default_get_metric_sums_since_returns_empty() {
            let t = MockActivityExt;
            assert!(t
                .get_metric_sums_since(&["processing_time"], Utc::now())
                .unwrap()
                .is_empty());
        }

        #[test]
        fn test_default_get_metric_sums_by_source_since_returns_empty() {
            let t = MockActivityExt;
            assert!(t
                .get_metric_sums_by_source_since(&["queue_depth"], Utc::now())
                .unwrap()
                .is_empty());
        }

        #[test]
        fn test_default_get_diagnostic_counts_returns_all_zeros() {
            let t = MockActivityExt;
            let counts = t.get_diagnostic_counts().unwrap();
            assert_eq!(counts.fix_attempts, 0);
            assert_eq!(counts.activity_log, 0);
            assert_eq!(counts.claude_executions, 0);
            assert_eq!(counts.pr_reviews, 0);
            assert_eq!(counts.pr_review_states, 0);
            assert_eq!(counts.issues, 0);
            assert_eq!(counts.similar_issues, 0);
            assert_eq!(counts.repositories, 0);
            assert_eq!(counts.repo_files, 0);
            assert_eq!(counts.inference_attempts, 0);
            assert_eq!(counts.error_patterns, 0);
            assert_eq!(counts.processing_metrics, 0);
            assert_eq!(counts.feedback_outcomes, 0);
            assert_eq!(counts.prs, 0);
            assert!(counts.recent_fix_attempts.is_empty());
            assert!(counts.fix_attempts_by_status.is_empty());
        }

        #[test]
        fn test_default_get_success_rate_returns_zero() {
            let t = MockActivityExt;
            let rate = t.get_success_rate().unwrap();
            assert!((rate - 0.0).abs() < f64::EPSILON);
        }

        #[test]
        fn test_default_record_cascade_attempt_returns_zero() {
            let t = MockActivityExt;
            assert_eq!(
                t.record_cascade_attempt("source", "issue", "short", 1, "org/repo")
                    .unwrap(),
                0
            );
        }

        #[test]
        fn test_default_update_attempt_pr_succeeds() {
            let t = MockActivityExt;
            assert!(t
                .update_attempt_pr(1, "https://github.com/org/repo/pull/1", "org/repo", 1)
                .is_ok());
        }

        #[test]
        fn test_default_mark_cascade_failed_succeeds() {
            let t = MockActivityExt;
            assert!(t.mark_cascade_failed(1, "build error").is_ok());
        }

        #[test]
        fn test_default_save_pr_review_state_succeeds() {
            let t = MockActivityExt;
            let state = claudear_core::types::PrReviewState {
                pr_url: "https://github.com/org/repo/pull/1".into(),
                repo: "org/repo".into(),
                pr_number: 1,
                issue_id: "LIN-1".into(),
                source: "linear".into(),
                last_review_id: None,
                last_review_time: None,
                last_comment_id: None,
                last_comment_time: None,
                is_active: true,
            };
            assert!(t.save_pr_review_state(&state).is_ok());
        }

        #[test]
        fn test_default_get_active_pr_review_states_returns_empty() {
            let t = MockActivityExt;
            assert!(t.get_active_pr_review_states().unwrap().is_empty());
        }

        #[test]
        fn test_default_deactivate_pr_review_state_succeeds() {
            let t = MockActivityExt;
            assert!(t
                .deactivate_pr_review_state("https://github.com/org/repo/pull/1")
                .is_ok());
        }

        #[test]
        fn test_default_get_comments_for_pr_returns_empty() {
            let t = MockActivityExt;
            assert!(t
                .get_comments_for_pr("https://github.com/org/repo/pull/1")
                .unwrap()
                .is_empty());
        }

        #[test]
        fn test_default_get_attempts_batch_returns_empty() {
            let t = MockActivityExt;
            assert!(t.get_attempts_batch(&[("s", "i")]).unwrap().is_empty());
        }

        #[test]
        fn test_default_record_release_tracking_returns_zero() {
            let t = MockActivityExt;
            let tracking = claudear_core::types::ReleaseTracking {
                id: 0,
                regression_watch_id: 1,
                release_version: "v1.0.0".into(),
                release_commit: "abc123".into(),
                released_at: None,
                created_at: Utc::now(),
            };
            assert_eq!(t.record_release_tracking(&tracking).unwrap(), 0);
        }

        #[test]
        fn test_default_record_pr_review_comment_returns_zero() {
            let t = MockActivityExt;
            let comment = claudear_core::types::ReviewComment {
                id: 1,
                path: "src/main.rs".into(),
                position: None,
                original_position: None,
                body: "LGTM".into(),
                user: claudear_core::types::ReviewUser {
                    id: 1,
                    login: "reviewer".into(),
                    user_type: Some("User".into()),
                },
                created_at: "2025-01-01T00:00:00Z".into(),
                updated_at: "2025-01-01T00:00:00Z".into(),
                html_url: "https://github.com/org/repo/pull/1#comment-1".into(),
                pull_request_review_id: None,
                line: None,
                start_line: None,
                side: None,
            };
            assert_eq!(
                t.record_pr_review_comment("https://github.com/org/repo/pull/1", &comment)
                    .unwrap(),
                0
            );
        }
    }

    mod repo_store_extended {
        use super::*;

        mock_for_subtrait!(MockRepoExt, RepoStore);

        #[test]
        fn test_default_get_or_create_repo_id_returns_zero() {
            let t = MockRepoExt;
            assert_eq!(t.get_or_create_repo_id("org/repo").unwrap(), 0);
        }

        #[test]
        fn test_default_code_chunk_hash_matches_returns_false() {
            let t = MockRepoExt;
            assert!(!t
                .code_chunk_hash_matches(1, "src/main.rs", "abc123")
                .unwrap());
        }

        #[test]
        fn test_default_delete_code_data_for_file_succeeds() {
            let t = MockRepoExt;
            assert!(t.delete_code_data_for_file(1, "src/main.rs").is_ok());
        }

        #[test]
        fn test_default_delete_code_chunks_by_ids_succeeds() {
            let t = MockRepoExt;
            assert!(t.delete_code_chunks_by_ids(&[1, 2, 3]).is_ok());
        }

        #[test]
        fn test_default_cleanup_stale_code_data_succeeds() {
            let t = MockRepoExt;
            assert!(t
                .cleanup_stale_code_data(1, &["src/main.rs".into()])
                .is_ok());
        }

        #[test]
        fn test_default_save_code_symbols_succeeds() {
            let t = MockRepoExt;
            assert!(t.save_code_symbols(&[]).is_ok());
        }

        #[test]
        fn test_default_save_code_chunks_returns_empty() {
            let t = MockRepoExt;
            assert!(t.save_code_chunks(&[]).unwrap().is_empty());
        }

        #[test]
        fn test_default_save_code_chunk_embeddings_succeeds() {
            let t = MockRepoExt;
            assert!(t.save_code_chunk_embeddings(&[], "model").is_ok());
        }

        #[test]
        fn test_default_get_code_embedding_model_returns_none() {
            let t = MockRepoExt;
            assert!(t.get_code_embedding_model(1).unwrap().is_none());
        }

        #[test]
        fn test_default_get_code_index_meta_returns_none() {
            let t = MockRepoExt;
            assert!(t.get_code_index_meta(1, "key").unwrap().is_none());
        }

        #[test]
        fn test_default_set_code_index_meta_succeeds() {
            let t = MockRepoExt;
            assert!(t.set_code_index_meta(1, "key", "value").is_ok());
        }

        #[test]
        fn test_default_delete_all_code_data_for_repo_succeeds() {
            let t = MockRepoExt;
            assert!(t.delete_all_code_data_for_repo(1).is_ok());
        }

        #[test]
        fn test_default_find_code_symbols_returns_empty() {
            let t = MockRepoExt;
            assert!(t.find_code_symbols("main", None, None).unwrap().is_empty());
        }

        #[test]
        fn test_default_get_indexed_repo_returns_none() {
            let t = MockRepoExt;
            assert!(t.get_indexed_repo("org/repo").unwrap().is_none());
        }

        #[test]
        fn test_default_sync_from_index_returns_zero() {
            let t = MockRepoExt;
            let index = claudear_core::types::RepoIndex::new();
            assert_eq!(t.sync_from_index(&index, false).unwrap(), 0);
        }

        #[test]
        fn test_default_start_indexing_progress_succeeds() {
            let t = MockRepoExt;
            assert!(t.start_indexing_progress(10).is_ok());
        }

        #[test]
        fn test_default_update_indexing_progress_succeeds() {
            let t = MockRepoExt;
            assert!(t.update_indexing_progress(5, "org/repo", 100, 500).is_ok());
        }

        #[test]
        fn test_default_finish_indexing_progress_succeeds() {
            let t = MockRepoExt;
            assert!(t.finish_indexing_progress().is_ok());
        }

        #[test]
        fn test_default_add_dependency_succeeds() {
            let t = MockRepoExt;
            assert!(t.add_dependency("upstream", "downstream", "npm").is_ok());
        }
    }

    // -------------------------------------------------------------------
    // parse_pr_url tests
    // -------------------------------------------------------------------

    #[test]
    fn test_parse_pr_url_github() {
        let result = parse_pr_url("https://github.com/owner/repo/pull/42");
        assert_eq!(result, Some(("owner/repo".to_string(), 42)));
    }

    #[test]
    fn test_parse_pr_url_github_large_number() {
        let result = parse_pr_url("https://github.com/my-org/my-repo/pull/12345");
        assert_eq!(result, Some(("my-org/my-repo".to_string(), 12345)));
    }

    #[test]
    fn test_parse_pr_url_gitlab_mr() {
        let result = parse_pr_url("https://gitlab.com/group/project/-/merge_requests/7");
        assert_eq!(result, Some(("group/project".to_string(), 7)));
    }

    #[test]
    fn test_parse_pr_url_gitlab_mr_nested_group() {
        let result = parse_pr_url("https://gitlab.example.com/org/sub/project/-/merge_requests/99");
        assert_eq!(result, Some(("org/sub/project".to_string(), 99)));
    }

    #[test]
    fn test_parse_pr_url_invalid() {
        assert!(parse_pr_url("https://example.com/something").is_none());
    }

    #[test]
    fn test_parse_pr_url_empty() {
        assert!(parse_pr_url("").is_none());
    }

    #[test]
    fn test_parse_pr_url_too_long() {
        let long_url = format!(
            "https://github.com/{}/pull/1",
            "a".repeat(MAX_PR_URL_LENGTH)
        );
        assert!(parse_pr_url(&long_url).is_none());
    }

    #[test]
    fn test_parse_pr_url_exactly_at_limit() {
        // Build a URL that is exactly MAX_PR_URL_LENGTH
        let base = "https://github.com/";
        let suffix = "/pull/1";
        let needed = MAX_PR_URL_LENGTH - base.len() - suffix.len();
        // Split needed into owner/repo
        let owner_len = needed / 2;
        let repo_len = needed - owner_len - 1; // -1 for the '/'
        let url = format!(
            "{}{}/{}{}",
            base,
            "a".repeat(owner_len),
            "b".repeat(repo_len),
            suffix
        );
        // Should still parse (at the limit, not over)
        assert!(url.len() <= MAX_PR_URL_LENGTH);
        let result = parse_pr_url(&url);
        assert!(result.is_some());
    }

    #[test]
    fn test_parse_pr_url_non_numeric_pr_number() {
        assert!(parse_pr_url("https://github.com/owner/repo/pull/abc").is_none());
    }

    #[test]
    fn test_parse_pr_url_pr_number_zero() {
        let result = parse_pr_url("https://github.com/owner/repo/pull/0");
        assert_eq!(result, Some(("owner/repo".to_string(), 0)));
    }

    #[test]
    fn test_parse_pr_url_gitlab_http() {
        let result = parse_pr_url("http://gitlab.internal/org/project/-/merge_requests/5");
        assert_eq!(result, Some(("org/project".to_string(), 5)));
    }
}
