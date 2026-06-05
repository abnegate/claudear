//! Core types shared across the application.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Maximum allowed length for issue IDs to prevent DoS attacks.
pub const MAX_ISSUE_ID_LENGTH: usize = 100;

/// Validate an issue ID for safety and sanity.
///
/// Returns `Ok(())` if the issue ID is valid, or `Err` with a description of the problem.
///
/// # Validation Rules
/// - Must not be empty
/// - Must not exceed `MAX_ISSUE_ID_LENGTH` (100) characters
/// - Must not contain path traversal sequences (`..`)
/// - Must not contain forward slashes (`/`)
/// - Must not contain backslashes (`\`)
/// - Must not contain null bytes
///
/// # Examples
/// ```
/// use claudear_core::types::validate_issue_id;
///
/// assert!(validate_issue_id("PROJ-123").is_ok());
/// assert!(validate_issue_id("abc123").is_ok());
/// assert!(validate_issue_id("").is_err());
/// assert!(validate_issue_id("../etc/passwd").is_err());
/// assert!(validate_issue_id("a/b").is_err());
/// ```
pub fn validate_issue_id(issue_id: &str) -> Result<(), String> {
    if issue_id.is_empty() {
        return Err("Issue ID cannot be empty".to_string());
    }

    if issue_id.len() > MAX_ISSUE_ID_LENGTH {
        return Err(format!(
            "Issue ID exceeds maximum length of {} characters",
            MAX_ISSUE_ID_LENGTH
        ));
    }

    if issue_id.contains("..") {
        return Err("Issue ID cannot contain path traversal sequences (..)".to_string());
    }

    if issue_id.contains('/') {
        return Err("Issue ID cannot contain forward slashes (/)".to_string());
    }

    if issue_id.contains('\\') {
        return Err("Issue ID cannot contain backslashes (\\)".to_string());
    }

    if issue_id.contains('\0') {
        return Err("Issue ID cannot contain null bytes".to_string());
    }

    Ok(())
}

/// Priority levels for issues.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum IssuePriority {
    #[default]
    None,
    Low,
    Medium,
    High,
    Critical,
}

impl std::fmt::Display for IssuePriority {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::Low => write!(f, "low"),
            Self::Medium => write!(f, "medium"),
            Self::High => write!(f, "high"),
            Self::Critical => write!(f, "critical"),
        }
    }
}

/// Status of an issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum IssueStatus {
    #[default]
    Open,
    InProgress,
    Resolved,
    Ignored,
}

impl std::fmt::Display for IssueStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Open => write!(f, "open"),
            Self::InProgress => write!(f, "in_progress"),
            Self::Resolved => write!(f, "resolved"),
            Self::Ignored => write!(f, "ignored"),
        }
    }
}

/// Unified issue representation across all sources.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Issue {
    /// Unique identifier from the source.
    pub id: String,
    /// Human-readable identifier (e.g., "PROJ-123", "SENTRY-ABC").
    pub short_id: String,
    /// Issue title.
    pub title: String,
    /// Issue description or error message.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// URL to view the issue in its source.
    pub url: String,
    /// Source service name.
    pub source: String,
    /// Priority level.
    pub priority: IssuePriority,
    /// Current status.
    pub status: IssueStatus,
    /// Additional metadata specific to the source.
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
    /// When the issue was first seen/created.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_at: Option<DateTime<Utc>>,
    /// When the issue was last updated.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<DateTime<Utc>>,
}

impl Issue {
    /// Create a new issue with required fields.
    pub fn new(
        id: impl Into<String>,
        short_id: impl Into<String>,
        title: impl Into<String>,
        url: impl Into<String>,
        source: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            short_id: short_id.into(),
            title: title.into(),
            description: None,
            url: url.into(),
            source: source.into(),
            priority: IssuePriority::default(),
            status: IssueStatus::default(),
            metadata: HashMap::new(),
            created_at: None,
            updated_at: None,
        }
    }

    /// Get a metadata value as a specific type.
    pub fn get_metadata<T: for<'de> Deserialize<'de>>(&self, key: &str) -> Option<T> {
        self.metadata
            .get(key)
            .and_then(|v| serde_json::from_value(v.clone()).ok())
    }

    /// Set a metadata value.
    pub fn set_metadata(&mut self, key: impl Into<String>, value: impl Serialize) {
        if let Ok(v) = serde_json::to_value(value) {
            self.metadata.insert(key.into(), v);
        }
    }
}

/// Priority for processing order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum MatchPriority {
    Low,
    #[default]
    Normal,
    High,
    Urgent,
}

/// Result of matching an issue against criteria.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchResult {
    /// Whether the issue matches.
    pub matches: bool,
    /// Human-readable reason for the match result.
    pub reason: String,
    /// Priority classification for processing order.
    pub priority: MatchPriority,
}

impl MatchResult {
    /// Create a matching result.
    pub fn matched(reason: impl Into<String>, priority: MatchPriority) -> Self {
        Self {
            matches: true,
            reason: reason.into(),
            priority,
        }
    }

    /// Create a non-matching result.
    pub fn not_matched(reason: impl Into<String>) -> Self {
        Self {
            matches: false,
            reason: reason.into(),
            priority: MatchPriority::Normal,
        }
    }
}

/// Result of a Claude fix attempt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentResult {
    /// Whether the fix was successful.
    pub success: bool,
    /// Raw output from Claude.
    pub output: String,
    /// Extracted PR URL if available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pr_url: Option<String>,
    /// Succinct changelog of what was changed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub changelog: Option<String>,
    /// Error message if failed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Structured blocking question emitted by Claude when it requires human input.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blocking_question: Option<BlockingQuestion>,
    /// Q&A knowledge IDs used while preparing this run.
    #[serde(default)]
    pub used_qa_ids: Vec<i64>,
    /// Confidence score (0-100) that the fix is correct.
    #[serde(default)]
    pub confidence: u8,
    /// Reasoning behind the confidence score.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence_reasoning: Option<String>,
    /// If Claude detected it's working in the wrong repository, the suggested correct repo.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wrong_repo: Option<String>,
}

/// Structured blocking question emitted by Claude.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockingQuestion {
    /// The actual question needing a human answer.
    pub question: String,
    /// Optional context Claude includes to help the responder.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
    /// Optional options Claude proposes.
    #[serde(default)]
    pub options: Vec<String>,
    /// Optional explanation of why the question is required.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub why: Option<String>,
}

/// Ask request used by notification channels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AskRequest {
    /// Correlation ID for cross-channel dedupe and reply matching.
    pub correlation_id: String,
    /// Source service name (e.g. linear, sentry).
    pub source: String,
    /// Optional target repository for scoped reuse.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repo: Option<String>,
    /// Source issue ID.
    pub issue_id: String,
    /// Human-readable issue key.
    pub short_id: String,
    /// Question payload.
    pub question: BlockingQuestion,
    /// Ask timestamp.
    pub asked_at: DateTime<Utc>,
    /// Optional desired Discord responder ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_discord_id: Option<String>,
    /// Optional desired email responder.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_email: Option<String>,
    /// Optional desired Slack responder ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_slack_id: Option<String>,
}

/// Delivery metadata for an ask message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AskDelivery {
    /// Channel name (discord/email/sms/push).
    pub channel: String,
    /// Channel-specific target identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
    /// Channel-specific message/thread ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message_id: Option<String>,
}

/// Reply captured from a notification channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AskReply {
    /// Correlation ID to link with ask.
    pub correlation_id: String,
    /// Channel where the reply was received.
    pub channel: String,
    /// Responder identity (channel user ID or email).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub responder: Option<String>,
    /// Raw reply text.
    pub answer: String,
    /// Reply timestamp.
    pub replied_at: DateTime<Utc>,
}

/// Status of a fix attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FixAttemptStatus {
    #[serde(rename = "pending")]
    Pending,
    #[serde(rename = "success")]
    Success,
    #[serde(rename = "failed")]
    Failed,
    /// PR was merged and issue was resolved.
    #[serde(rename = "merged")]
    Merged,
    /// PR was closed without merging.
    #[serde(rename = "closed")]
    Closed,
    /// Max retries reached, issue cannot be automatically fixed.
    #[serde(rename = "cannot_fix")]
    CannotFix,
    /// Input was a question and was answered (no PR/fix attempted).
    #[serde(rename = "answered")]
    Answered,
}

impl std::fmt::Display for FixAttemptStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pending => write!(f, "pending"),
            Self::Success => write!(f, "success"),
            Self::Failed => write!(f, "failed"),
            Self::Merged => write!(f, "merged"),
            Self::Closed => write!(f, "closed"),
            Self::CannotFix => write!(f, "cannot_fix"),
            Self::Answered => write!(f, "answered"),
        }
    }
}

impl std::str::FromStr for FixAttemptStatus {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "pending" => Ok(Self::Pending),
            "success" => Ok(Self::Success),
            "failed" => Ok(Self::Failed),
            "merged" => Ok(Self::Merged),
            "closed" => Ok(Self::Closed),
            "cannot_fix" => Ok(Self::CannotFix),
            "answered" => Ok(Self::Answered),
            _ => Err(format!("Unknown status: {}", s)),
        }
    }
}

/// Record of a fix attempt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixAttempt {
    pub id: i64,
    /// Issue ID from the source.
    pub issue_id: String,
    /// Human-readable issue ID.
    pub short_id: String,
    /// Source service name.
    pub source: String,
    /// When the attempt was made.
    pub attempted_at: DateTime<Utc>,
    /// PR URL if successful.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pr_url: Option<String>,
    /// GitHub repository (owner/repo format).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scm_repo: Option<String>,
    /// GitHub PR number.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scm_pr_number: Option<i64>,
    /// Current status.
    pub status: FixAttemptStatus,
    /// Error message if failed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    /// When the PR was merged (if merged).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub merged_at: Option<DateTime<Utc>>,
    /// When the issue was resolved on the source.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resolved_at: Option<DateTime<Utc>>,
    /// Number of retry attempts made.
    #[serde(default)]
    pub retry_count: u32,
    /// When the last retry was attempted.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_retry_at: Option<DateTime<Utc>>,
    /// Labels from the issue (for bug detection).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub issue_labels: Vec<String>,
    /// Parent attempt ID for cascade chains. NULL for root attempts.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_attempt_id: Option<i64>,
    /// Target repository for cascade attempts (e.g., "appwrite/appwrite").
    /// NULL for root attempts (original issue fix).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cascade_repo: Option<String>,
}

impl FixAttempt {
    /// Check if this fix attempt is for a bug (based on labels).
    ///
    /// Returns true if:
    /// - Source is "sentry" (all Sentry issues are bugs)
    /// - Issue has a label indicating it's a bug (e.g., "bug", "defect", "error")
    pub fn is_bug(&self) -> bool {
        // Sentry issues are always bugs
        if self.source == "sentry" {
            return true;
        }

        // Check for common bug labels
        const BUG_LABELS: &[&str] = &[
            "bug",
            "defect",
            "error",
            "fix",
            "hotfix",
            "regression",
            "issue",
            "problem",
            "incident",
            "crash",
            "broken",
        ];

        self.issue_labels.iter().any(|label| {
            let lower = label.to_lowercase();
            BUG_LABELS.iter().any(|bug_label| lower.contains(bug_label))
        })
    }
}

/// A first-class action that can be run against a payload (issue/conversation).
///
/// Actions chain: classification routes a bug/security report into `Verify`,
/// which on success starts `Resolve` (the fix pipeline); non-bug messages and
/// terminal states start `Reply`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionKind {
    /// Generate and post a grounded, human-sounding reply to the ticket.
    Reply,
    /// Attempt to reproduce a reported bug/security issue.
    Verify,
    /// Run the bug-resolution pipeline to create/watch/merge a PR.
    Resolve,
}

impl std::fmt::Display for ActionKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            ActionKind::Reply => "reply",
            ActionKind::Verify => "verify",
            ActionKind::Resolve => "resolve",
        };
        f.write_str(s)
    }
}

impl std::str::FromStr for ActionKind {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.trim().to_lowercase().as_str() {
            "reply" => Ok(ActionKind::Reply),
            "verify" => Ok(ActionKind::Verify),
            "resolve" => Ok(ActionKind::Resolve),
            other => Err(format!("unknown action: {other}")),
        }
    }
}

/// What kind of reply is being generated, which selects the framing the agent uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplyKind {
    /// General grounded answer to a non-bug message (question / feature request).
    Answer,
    /// The reported bug could not be reproduced; ask the reporter for repro steps.
    NeedRepro,
    /// A fix shipped and is live; let the reporter know.
    FixShipped,
}

impl ReplyKind {
    /// Stable lowercase identifier (for logging / persistence).
    pub fn as_str(&self) -> &'static str {
        match self {
            ReplyKind::Answer => "answer",
            ReplyKind::NeedRepro => "need_repro",
            ReplyKind::FixShipped => "fix_shipped",
        }
    }
}

/// Outcome of a `Verify` action: whether the agent reproduced the reported issue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifyResult {
    /// Whether the issue was reproduced as described.
    pub reproduced: bool,
    /// One-line summary of the verification verdict.
    #[serde(default)]
    pub summary: String,
    /// Evidence supporting the verdict (repro steps, failing output, etc.).
    #[serde(default)]
    pub evidence: String,
}

/// Statistics about fix attempts.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FixAttemptStats {
    pub total: usize,
    pub pending: usize,
    pub success: usize,
    pub failed: usize,
    /// PRs that were merged successfully.
    pub merged: usize,
    /// PRs that were closed without merging.
    pub closed: usize,
    /// Issues that reached max retries and cannot be fixed.
    pub cannot_fix: usize,
    pub by_source: HashMap<String, SourceStats>,
}

/// Per-source statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SourceStats {
    pub total: usize,
    pub success: usize,
    pub failed: usize,
    pub merged: usize,
    pub closed: usize,
    pub cannot_fix: usize,
}

/// Activity log entry for tracking operational events.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivityLogEntry {
    /// Database ID.
    pub id: i64,
    /// When the activity occurred.
    pub timestamp: DateTime<Utc>,
    /// Type of activity (e.g., 'issue_received', 'processing_started', 'pr_created', 'error').
    pub activity_type: String,
    /// Source service (e.g., 'linear', 'sentry').
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// Issue ID from the source.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub issue_id: Option<String>,
    /// Human-readable issue ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub short_id: Option<String>,
    /// Human-readable message describing the activity.
    pub message: String,
    /// Additional context as JSON.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

impl ActivityLogEntry {
    /// Create a new activity log entry.
    pub fn new(activity_type: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            id: 0,
            timestamp: Utc::now(),
            activity_type: activity_type.into(),
            source: None,
            issue_id: None,
            short_id: None,
            message: message.into(),
            metadata: None,
        }
    }

    /// Set the source for this activity.
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Set the issue ID for this activity.
    pub fn with_issue(mut self, issue_id: impl Into<String>, short_id: impl Into<String>) -> Self {
        self.issue_id = Some(issue_id.into());
        self.short_id = Some(short_id.into());
        self
    }

    /// Set metadata for this activity.
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = Some(metadata);
        self
    }
}

/// Claude execution record with detailed metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentExecution {
    /// Database ID.
    pub id: i64,
    /// Reference to fix_attempts table.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attempt_id: Option<i64>,
    /// When execution started.
    pub started_at: DateTime<Utc>,
    /// When execution completed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<DateTime<Utc>>,
    /// Duration in seconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_secs: Option<f64>,
    /// Process exit code.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i32>,
    /// Whether the process timed out.
    pub timed_out: bool,
    /// Preview of stdout (first/last N chars).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stdout_preview: Option<String>,
    /// Preview of stderr.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stderr_preview: Option<String>,
    /// Absolute path to the captured stdout log file.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stdout_log_path: Option<String>,
    /// Absolute path to the captured stderr log file.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stderr_log_path: Option<String>,
    /// Absolute path to the captured execution event log file (JSONL).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub event_log_path: Option<String>,
    /// The prompt sent to Claude.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_used: Option<String>,
    /// Hash of the prompt for grouping.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_hash: Option<String>,
    /// Claude model version used.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_version: Option<String>,
    /// Working directory path.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub working_directory: Option<String>,
    /// Git branch name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub git_branch: Option<String>,
    /// Git commit hash before execution.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub git_commit_before: Option<String>,
    /// Git commit hash after execution.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub git_commit_after: Option<String>,
    /// Number of files changed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub files_changed: Option<i32>,
    /// Lines added.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lines_added: Option<i32>,
    /// Lines removed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lines_removed: Option<i32>,
    /// Total cost in USD reported by Claude CLI.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_cost_usd: Option<f64>,
    /// Number of conversation turns.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_turns: Option<i64>,
    /// Claude session identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// API-side request duration in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_api_ms: Option<i64>,
    /// Input tokens used (non-cache).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<i64>,
    /// Output tokens generated.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<i64>,
    /// Tokens read from prompt cache.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<i64>,
    /// Tokens written to prompt cache.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<i64>,
    /// Provider name (e.g. "claude", "codex").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    /// A/B experiment name, if running under an experiment.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub experiment_name: Option<String>,
    /// Which experiment arm/variant was selected.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub experiment_variant: Option<String>,
}

/// Per-provider statistics from an A/B experiment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentProviderStats {
    pub provider: String,
    pub total_attempts: i64,
    pub success_count: i64,
    pub avg_cost: Option<f64>,
    pub avg_duration: Option<f64>,
    pub success_rate: f64,
}

impl AgentExecution {
    /// Create a new execution record with the start time set to now.
    pub fn new() -> Self {
        Self {
            id: 0,
            attempt_id: None,
            started_at: Utc::now(),
            completed_at: None,
            duration_secs: None,
            exit_code: None,
            timed_out: false,
            stdout_preview: None,
            stderr_preview: None,
            stdout_log_path: None,
            stderr_log_path: None,
            event_log_path: None,
            prompt_used: None,
            prompt_hash: None,
            model_version: None,
            working_directory: None,
            git_branch: None,
            git_commit_before: None,
            git_commit_after: None,
            files_changed: None,
            lines_added: None,
            lines_removed: None,
            total_cost_usd: None,
            num_turns: None,
            session_id: None,
            duration_api_ms: None,
            input_tokens: None,
            output_tokens: None,
            cache_read_input_tokens: None,
            cache_creation_input_tokens: None,
            provider: None,
            experiment_name: None,
            experiment_variant: None,
        }
    }

    /// Set the attempt ID.
    pub fn with_attempt_id(mut self, attempt_id: i64) -> Self {
        self.attempt_id = Some(attempt_id);
        self
    }

    /// Mark the execution as complete.
    pub fn complete(&mut self, exit_code: Option<i32>, timed_out: bool) {
        let now = Utc::now();
        self.completed_at = Some(now);
        self.duration_secs = Some((now - self.started_at).num_milliseconds() as f64 / 1000.0);
        self.exit_code = exit_code;
        self.timed_out = timed_out;
    }
}

impl Default for AgentExecution {
    fn default() -> Self {
        Self::new()
    }
}

/// PR review feedback for learning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrReviewRecord {
    /// Database ID.
    pub id: i64,
    /// Reference to fix_attempts table.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attempt_id: Option<i64>,
    /// PR URL.
    pub pr_url: String,
    /// Reviewer username.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reviewer: Option<String>,
    /// Review state (approved, changes_requested, commented).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub review_state: Option<String>,
    /// When the review was submitted.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub submitted_at: Option<DateTime<Utc>>,
    /// Review body/comment.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<String>,
    /// Computed sentiment (positive, negative, neutral).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sentiment: Option<String>,
    /// Extracted improvement suggestions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub actionable_feedback: Option<String>,
}

impl PrReviewRecord {
    /// Create a new PR review record.
    pub fn new(pr_url: impl Into<String>) -> Self {
        Self {
            id: 0,
            attempt_id: None,
            pr_url: pr_url.into(),
            reviewer: None,
            review_state: None,
            submitted_at: None,
            body: None,
            sentiment: None,
            actionable_feedback: None,
        }
    }
}

/// PR lifecycle tracking record.
///
/// Tracks comprehensive information about a PR from creation to merge/close.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrRecord {
    /// Database ID.
    pub id: i64,
    /// Full PR URL.
    pub pr_url: String,
    /// GitHub repository (owner/repo).
    pub scm_repo: String,
    /// PR number.
    pub pr_number: i64,

    /// Reference to fix_attempts table.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attempt_id: Option<i64>,
    /// Original issue ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub issue_id: Option<String>,
    /// Original issue source (linear, sentry).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub issue_source: Option<String>,

    /// PR title.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// PR description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// PR author.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub author: Option<String>,
    /// Head branch name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub head_branch: Option<String>,
    /// Base branch name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_branch: Option<String>,

    /// PR status: open, merged, closed.
    pub status: String,
    /// When the PR was created.
    pub created_at: DateTime<Utc>,
    /// When the PR was last updated.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<DateTime<Utc>>,
    /// When the PR was merged.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub merged_at: Option<DateTime<Utc>>,
    /// When the PR was closed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub closed_at: Option<DateTime<Utc>>,

    /// Number of approvals.
    pub approvals_count: i32,
    /// Number of changes_requested reviews.
    pub changes_requested_count: i32,
    /// Number of comments.
    pub comments_count: i32,
    /// When the last review was submitted.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_review_at: Option<DateTime<Utc>>,

    /// Minutes from PR creation to first review.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_to_first_review_mins: Option<i64>,
    /// Minutes from PR creation to merge.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_to_merge_mins: Option<i64>,
    /// Number of review cycles.
    pub review_cycles: i32,

    /// Number of files changed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub files_changed: Option<i64>,
    /// Lines added.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lines_added: Option<i64>,
    /// Lines removed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lines_removed: Option<i64>,
}

impl PrRecord {
    /// Create a new PR record.
    pub fn new(pr_url: impl Into<String>, scm_repo: impl Into<String>, pr_number: i64) -> Self {
        Self {
            id: 0,
            pr_url: pr_url.into(),
            scm_repo: scm_repo.into(),
            pr_number,
            attempt_id: None,
            issue_id: None,
            issue_source: None,
            title: None,
            description: None,
            author: None,
            head_branch: None,
            base_branch: None,
            status: "open".to_string(),
            created_at: Utc::now(),
            updated_at: None,
            merged_at: None,
            closed_at: None,
            approvals_count: 0,
            changes_requested_count: 0,
            comments_count: 0,
            last_review_at: None,
            time_to_first_review_mins: None,
            time_to_merge_mins: None,
            review_cycles: 0,
            files_changed: None,
            lines_added: None,
            lines_removed: None,
        }
    }

    /// Create a PR record with issue linkage.
    pub fn for_issue(
        pr_url: impl Into<String>,
        scm_repo: impl Into<String>,
        pr_number: i64,
        issue_source: impl Into<String>,
        issue_id: impl Into<String>,
    ) -> Self {
        let mut record = Self::new(pr_url, scm_repo, pr_number);
        record.issue_source = Some(issue_source.into());
        record.issue_id = Some(issue_id.into());
        record
    }
}

/// Aggregate PR analytics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PrAnalytics {
    /// Total number of PRs.
    pub total: i64,
    /// Number of open PRs.
    pub open: i64,
    /// Number of merged PRs.
    pub merged: i64,
    /// Number of closed PRs (without merge).
    pub closed: i64,

    /// Average time to first review in minutes.
    pub avg_time_to_first_review_mins: Option<f64>,
    /// Average time to merge in minutes.
    pub avg_time_to_merge_mins: Option<f64>,
    /// Average review cycles per PR.
    pub avg_review_cycles: Option<f64>,

    /// Merge rate (merged / (merged + closed)).
    pub merge_rate: Option<f64>,

    /// PRs by repository.
    pub by_repo: HashMap<String, i64>,
    /// Average time from issue attempt to PR creation in minutes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub avg_time_to_pr_mins: Option<f64>,
    /// PR rejection/review-change reason breakdown.
    #[serde(default)]
    pub rejection_reasons: Vec<RejectionReason>,
}

/// Issue embedding for similarity search.
#[derive(Debug, Clone)]
pub struct IssueEmbedding {
    /// Database ID.
    pub id: i64,
    /// Source service (e.g., 'linear', 'sentry').
    pub source: String,
    /// Issue ID from the source.
    pub issue_id: String,
    /// Human-readable issue ID.
    pub short_id: Option<String>,
    /// Issue title.
    pub title: Option<String>,
    /// Issue description or error message.
    pub description: Option<String>,
    /// URL to view the issue in its source.
    pub url: Option<String>,
    /// Priority level (none, low, medium, high, critical).
    pub priority: Option<String>,
    /// Current status (open, in_progress, resolved, ignored).
    pub status: Option<String>,
    /// Labels as JSON array text.
    pub labels: Option<String>,
    /// The embedding vector (serialized float32). None if issue stored without embedding.
    pub embedding: Option<Vec<f32>>,
    /// Model used to generate the embedding.
    pub embedding_model: Option<String>,
    /// When the embedding was created.
    pub created_at: DateTime<Utc>,
    /// When the issue was last updated.
    pub updated_at: Option<DateTime<Utc>>,
}

impl IssueEmbedding {
    /// Create a new issue embedding.
    pub fn new(
        source: impl Into<String>,
        issue_id: impl Into<String>,
        embedding: Vec<f32>,
    ) -> Self {
        Self {
            id: 0,
            source: source.into(),
            issue_id: issue_id.into(),
            short_id: None,
            title: None,
            description: None,
            url: None,
            priority: None,
            status: None,
            labels: None,
            embedding: Some(embedding),
            embedding_model: None,
            created_at: Utc::now(),
            updated_at: None,
        }
    }

    /// Create an IssueEmbedding from an Issue, storing content fields without an embedding.
    pub fn from_issue(issue: &Issue) -> Self {
        let labels: Vec<String> = issue.get_metadata("labels").unwrap_or_default();
        let labels_json = if labels.is_empty() {
            None
        } else {
            serde_json::to_string(&labels).ok()
        };
        Self {
            id: 0,
            source: issue.source.clone(),
            issue_id: issue.id.clone(),
            short_id: Some(issue.short_id.clone()),
            title: Some(issue.title.clone()),
            description: issue.description.clone(),
            url: Some(issue.url.clone()),
            priority: Some(issue.priority.to_string()),
            status: Some(issue.status.to_string()),
            labels: labels_json,
            embedding: None,
            embedding_model: None,
            created_at: issue.created_at.unwrap_or_else(Utc::now),
            updated_at: issue.updated_at,
        }
    }
}

/// Error pattern for recurring error analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorPattern {
    /// Database ID.
    pub id: i64,
    /// Hash of the normalized error (for deduplication).
    pub pattern_hash: String,
    /// Error type (build_failure, test_failure, timeout, claude_error).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_type: Option<String>,
    /// The error message.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    /// When first seen.
    pub first_seen: DateTime<Utc>,
    /// When last seen.
    pub last_seen: DateTime<Utc>,
    /// How many times this pattern occurred.
    pub occurrence_count: i32,
    /// JSON array of sources that hit this error.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sources: Option<Vec<String>>,
    /// JSON array of example issue IDs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub example_issue_ids: Option<Vec<String>>,
    /// Learned hints for fixing this error.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resolution_hints: Option<String>,
}

impl ErrorPattern {
    /// Create a new error pattern.
    pub fn new(pattern_hash: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            id: 0,
            pattern_hash: pattern_hash.into(),
            error_type: None,
            error_message: None,
            first_seen: now,
            last_seen: now,
            occurrence_count: 1,
            sources: None,
            example_issue_ids: None,
            resolution_hints: None,
        }
    }
}

/// Composite severity score produced by the prioritisation engine.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SeverityScore {
    /// Final weighted score (higher = more urgent).
    pub score: f64,
    /// Contribution from issue/match priority (0.0-1.0).
    pub severity_component: f64,
    /// Contribution from event frequency signals (0.0-1.0).
    pub frequency_component: f64,
    /// Contribution from regression risk signals (0.0-1.0).
    pub regression_component: f64,
    /// Contribution from blast radius classification (0.0-1.0).
    pub blast_radius_component: f64,
    /// Boost from belonging to a content cluster (0.0 or 1.0).
    pub cluster_boost: f64,
}

/// Blast radius classification for an issue.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default,
)]
#[serde(rename_all = "snake_case")]
pub enum BlastRadius {
    Cosmetic,
    Test,
    Peripheral,
    #[default]
    Core,
    Infrastructure,
    Critical,
}

impl std::fmt::Display for BlastRadius {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cosmetic => write!(f, "cosmetic"),
            Self::Test => write!(f, "test"),
            Self::Peripheral => write!(f, "peripheral"),
            Self::Core => write!(f, "core"),
            Self::Infrastructure => write!(f, "infrastructure"),
            Self::Critical => write!(f, "critical"),
        }
    }
}

/// A cluster of content-similar issues.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentCluster {
    /// Database ID (0 before storage).
    pub id: i64,
    /// Key identifying the cluster (e.g. "TypeError::app.main").
    pub cluster_key: String,
    /// Source service name.
    pub source: String,
    /// Issue ID of the representative (highest-scored) issue.
    pub representative_issue_id: String,
    /// All issue IDs in the cluster.
    pub issue_ids: Vec<String>,
    /// Shared error type (if any).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_type: Option<String>,
    /// Shared culprit (if any).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub culprit: Option<String>,
    /// Average pairwise similarity within the cluster.
    pub avg_similarity: f64,
    /// Status: "active" or "resolved".
    pub status: String,
    /// When the cluster was created.
    pub created_at: DateTime<Utc>,
}

/// A user-configured suppression rule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuppressionRule {
    /// Human-readable rule name.
    pub name: String,
    /// Which issue field to match against.
    pub field: SuppressionField,
    /// Pattern to match.
    pub pattern: String,
    /// How to match the pattern.
    #[serde(default)]
    pub match_mode: SuppressionMatchMode,
    /// Restrict this rule to specific sources (empty = all sources).
    #[serde(default)]
    pub sources: Vec<String>,
    /// Human-readable reason for suppression.
    #[serde(default)]
    pub reason: String,
}

/// Which issue field a suppression rule matches against.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SuppressionField {
    Title,
    Description,
    Source,
    Culprit,
    Filename,
    ErrorType,
    Project,
    Labels,
    Metadata(String),
}

/// How a suppression pattern is matched.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum SuppressionMatchMode {
    #[default]
    Contains,
    Exact,
    Regex,
}

/// Result of evaluating suppression rules against an issue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuppressionResult {
    /// Whether the issue was suppressed.
    pub suppressed: bool,
    /// Name of the matched rule (if any).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub matched_rule: Option<String>,
    /// Reason from the matched rule (if any).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// An issue after scoring and classification by the prioritisation engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrioritisedIssue {
    /// The original issue.
    pub issue: Issue,
    /// Criteria match result.
    pub match_result: MatchResult,
    /// Composite severity score.
    pub severity_score: SeverityScore,
    /// Blast radius classification.
    pub blast_radius: BlastRadius,
    /// Cluster key if the issue belongs to a content cluster.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cluster_key: Option<String>,
}

/// Processing metric for time-series data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingMetric {
    /// Database ID.
    pub id: i64,
    /// When the metric was recorded.
    pub timestamp: DateTime<Utc>,
    /// Metric name (queue_depth, processing_time, success_rate, etc.).
    pub metric_name: String,
    /// Metric value.
    pub metric_value: f64,
    /// Optional source for per-source metrics.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// Additional dimensions as JSON.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tags: Option<serde_json::Value>,
}

impl ProcessingMetric {
    /// Create a new processing metric.
    pub fn new(metric_name: impl Into<String>, metric_value: f64) -> Self {
        Self {
            id: 0,
            timestamp: Utc::now(),
            metric_name: metric_name.into(),
            metric_value,
            source: None,
            tags: None,
        }
    }

    /// Set the source for this metric.
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Set tags for this metric.
    pub fn with_tags(mut self, tags: serde_json::Value) -> Self {
        self.tags = Some(tags);
        self
    }
}

/// Prompt experiment for A/B testing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptExperiment {
    /// Database ID.
    pub id: i64,
    /// Experiment name.
    pub experiment_name: String,
    /// Variant (control, variant_a, etc.).
    pub variant: String,
    /// The prompt template.
    pub prompt_template: String,
    /// Hash of the prompt.
    pub prompt_hash: String,
    /// When the experiment was created.
    pub created_at: DateTime<Utc>,
    /// Whether this variant is active.
    pub active: bool,
    /// Number of successful outcomes.
    pub success_count: i32,
    /// Number of failed outcomes.
    pub failure_count: i32,
    /// Average time to merge (in hours).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub avg_time_to_merge: Option<f64>,
    /// Average review score.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub avg_review_score: Option<f64>,
}

impl PromptExperiment {
    /// Create a new prompt experiment.
    pub fn new(
        experiment_name: impl Into<String>,
        variant: impl Into<String>,
        prompt_template: impl Into<String>,
        prompt_hash: impl Into<String>,
    ) -> Self {
        Self {
            id: 0,
            experiment_name: experiment_name.into(),
            variant: variant.into(),
            prompt_template: prompt_template.into(),
            prompt_hash: prompt_hash.into(),
            created_at: Utc::now(),
            active: true,
            success_count: 0,
            failure_count: 0,
            avg_time_to_merge: None,
            avg_review_score: None,
        }
    }
}

/// Similar issue match.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarIssue {
    /// Database ID.
    pub id: i64,
    /// The source issue ID.
    pub source_issue_id: String,
    /// The similar issue ID.
    pub similar_issue_id: String,
    /// Similarity score (0.0 to 1.0).
    pub similarity_score: f64,
    /// When the similarity was computed.
    pub computed_at: DateTime<Utc>,
}

/// Stored Q&A knowledge entry used for semantic reuse.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QaKnowledgeEntry {
    pub id: i64,
    pub source: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repo: Option<String>,
    pub issue_id: String,
    pub short_id: String,
    pub question_text: String,
    pub question_norm: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub question_embedding: Option<Vec<f32>>,
    pub answer_text: String,
    pub answer_norm: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub answer_embedding: Option<Vec<f32>>,
    pub channel: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub responder: Option<String>,
    pub correlation_id: String,
    pub asked_at: DateTime<Utc>,
    pub answered_at: DateTime<Utc>,
    pub success_count: i64,
    pub failure_count: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_used_at: Option<DateTime<Utc>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

/// Similar Q&A match used during retrieval.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QaMatch {
    pub entry: QaKnowledgeEntry,
    pub semantic_similarity: f64,
    pub historical_success_rate: f64,
    pub final_score: f64,
}

impl SimilarIssue {
    /// Create a new similar issue record.
    pub fn new(
        source_issue_id: impl Into<String>,
        similar_issue_id: impl Into<String>,
        similarity_score: f64,
    ) -> Self {
        Self {
            id: 0,
            source_issue_id: source_issue_id.into(),
            similar_issue_id: similar_issue_id.into(),
            similarity_score,
            computed_at: Utc::now(),
        }
    }
}

/// Analytics summary statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AnalyticsSummary {
    /// Overall success rate (0.0 to 1.0).
    pub success_rate: f64,
    /// Total issues processed.
    pub total_processed: i64,
    /// Total successful fixes.
    pub total_successful: i64,
    /// Total merged PRs.
    pub total_merged: i64,
    /// Average processing time in seconds.
    pub avg_processing_time_secs: Option<f64>,
    /// Average time to merge in hours.
    pub avg_time_to_merge_hours: Option<f64>,
    /// Most common error type.
    pub most_common_error: Option<String>,
    /// Success rate by source.
    pub success_rate_by_source: HashMap<String, f64>,
    /// Average time from issue creation to PR creation in minutes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub avg_time_to_pr_mins: Option<f64>,
    /// Cost estimate breakdown.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost_estimate: Option<CostEstimate>,
    /// Weekly MTTR trend data.
    #[serde(default)]
    pub mttr_trend: Vec<MttrDataPoint>,
    /// Per-repo leaderboard.
    #[serde(default)]
    pub repo_leaderboard: Vec<RepoLeaderboardEntry>,
}

/// PR rejection/review-change reason category.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RejectionReason {
    pub category: String,
    pub count: i64,
}

/// Estimated cost breakdown for automated fixes.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CostEstimate {
    pub total_cost: f64,
    pub avg_cost_per_fix: f64,
    pub fix_count: i64,
    pub cost_source: String,
    pub period: String,
}

/// A single data point in the MTTR (mean time to resolve) trend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MttrDataPoint {
    pub period_start: String,
    pub mttr_minutes: f64,
    pub sample_count: i64,
}

/// Per-repository leaderboard entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepoLeaderboardEntry {
    pub repo: String,
    pub total: i64,
    pub success_rate: f64,
    pub merge_rate: f64,
    pub avg_time_to_merge_mins: Option<f64>,
}

/// Engineering time savings estimate.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TimeSavings {
    pub merged_count: i64,
    pub hours_saved: f64,
    pub cost_saved: f64,
    pub period: String,
}

/// Status of a regression watch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum RegressionWatchStatus {
    /// Waiting for the fix to be included in a release.
    #[default]
    AwaitingRelease,
    /// Release detected, actively monitoring for regressions.
    Monitoring,
    /// No regression detected after monitoring period, issue resolved.
    Resolved,
    /// Regression detected, fix failed.
    Regressed,
}

impl std::fmt::Display for RegressionWatchStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AwaitingRelease => write!(f, "awaiting_release"),
            Self::Monitoring => write!(f, "monitoring"),
            Self::Resolved => write!(f, "resolved"),
            Self::Regressed => write!(f, "regressed"),
        }
    }
}

impl std::str::FromStr for RegressionWatchStatus {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "awaiting_release" => Ok(Self::AwaitingRelease),
            "monitoring" => Ok(Self::Monitoring),
            "resolved" => Ok(Self::Resolved),
            "regressed" => Ok(Self::Regressed),
            _ => Err(format!("Unknown regression watch status: {}", s)),
        }
    }
}

/// Type of issue being watched for regression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IssueType {
    /// Issue originated from Sentry.
    SentryIssue,
    /// Issue from Linear marked as a bug.
    LinearBug,
    /// Issue from GitLab.
    GitLabIssue,
    /// Issue from Jira.
    JiraIssue,
}

impl IssueType {
    /// Get the source name for this issue type (used for retry lookups).
    pub fn source_name(&self) -> &'static str {
        match self {
            Self::SentryIssue => "sentry",
            Self::LinearBug => "linear",
            Self::GitLabIssue => "gitlab",
            Self::JiraIssue => "jira",
        }
    }
}

impl std::fmt::Display for IssueType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SentryIssue => write!(f, "sentry_issue"),
            Self::LinearBug => write!(f, "linear_bug"),
            Self::GitLabIssue => write!(f, "gitlab_issue"),
            Self::JiraIssue => write!(f, "jira_issue"),
        }
    }
}

impl std::str::FromStr for IssueType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "sentry_issue" => Ok(Self::SentryIssue),
            "linear_bug" => Ok(Self::LinearBug),
            "gitlab_issue" => Ok(Self::GitLabIssue),
            "jira_issue" => Ok(Self::JiraIssue),
            _ => Err(format!("Unknown issue type: {}", s)),
        }
    }
}

/// A watch for regression after a bug fix is merged.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionWatch {
    /// Database ID.
    pub id: i64,
    /// Type of issue (SentryIssue or LinearBug).
    pub issue_type: IssueType,
    /// Issue ID from the source.
    pub issue_id: String,
    /// Reference to the fix attempt.
    pub fix_attempt_id: i64,
    /// Current status of the watch.
    pub status: RegressionWatchStatus,
    /// When the PR was merged.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pr_merged_at: Option<DateTime<Utc>>,
    /// When regression monitoring started.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub monitoring_started_at: Option<DateTime<Utc>>,
    /// When the issue was resolved (after 24h of no regression).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resolved_at: Option<DateTime<Utc>>,
    /// When a regression was detected.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub regressed_at: Option<DateTime<Utc>>,
    /// When the watch was created.
    pub created_at: DateTime<Utc>,
}

impl RegressionWatch {
    /// Create a new regression watch.
    pub fn new(issue_type: IssueType, issue_id: impl Into<String>, fix_attempt_id: i64) -> Self {
        Self {
            id: 0,
            issue_type,
            issue_id: issue_id.into(),
            fix_attempt_id,
            status: RegressionWatchStatus::AwaitingRelease,
            pr_merged_at: None,
            monitoring_started_at: None,
            resolved_at: None,
            regressed_at: None,
            created_at: Utc::now(),
        }
    }
}

/// Tracking of a release that may contain a fix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReleaseTracking {
    /// Database ID.
    pub id: i64,
    /// Reference to the regression watch.
    pub regression_watch_id: i64,
    /// Release version/tag.
    pub release_version: String,
    /// Commit SHA of the release.
    pub release_commit: String,
    /// When the release was detected.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub released_at: Option<DateTime<Utc>>,
    /// When this tracking entry was created.
    pub created_at: DateTime<Utc>,
}

impl ReleaseTracking {
    /// Create a new release tracking entry.
    pub fn new(
        regression_watch_id: i64,
        release_version: impl Into<String>,
        release_commit: impl Into<String>,
    ) -> Self {
        Self {
            id: 0,
            regression_watch_id,
            release_version: release_version.into(),
            release_commit: release_commit.into(),
            released_at: Some(Utc::now()),
            created_at: Utc::now(),
        }
    }
}

/// A single regression check result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionCheck {
    /// Database ID.
    pub id: i64,
    /// Reference to the regression watch.
    pub regression_watch_id: i64,
    /// Whether the issue still exists (regression detected).
    pub issue_still_exists: bool,
    /// When the check was performed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checked_at: Option<DateTime<Utc>>,
    /// Detailed findings.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub check_details: Option<String>,
    /// When this check was created.
    pub created_at: DateTime<Utc>,
}

impl RegressionCheck {
    /// Create a new regression check.
    pub fn new(regression_watch_id: i64, issue_still_exists: bool) -> Self {
        Self {
            id: 0,
            regression_watch_id,
            issue_still_exists,
            checked_at: Some(Utc::now()),
            check_details: None,
            created_at: Utc::now(),
        }
    }
}

/// Category of file changes in a diff.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChangeCategory {
    Tests,
    Docs,
    Config,
    Dependencies,
    Migrations,
    NewCode,
    Modification,
}

impl std::fmt::Display for ChangeCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Tests => write!(f, "tests"),
            Self::Docs => write!(f, "docs"),
            Self::Config => write!(f, "config"),
            Self::Dependencies => write!(f, "dependencies"),
            Self::Migrations => write!(f, "migrations"),
            Self::NewCode => write!(f, "new_code"),
            Self::Modification => write!(f, "modification"),
        }
    }
}

/// Structured analysis of a PR diff.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffAnalysis {
    pub id: i64,
    pub attempt_id: i64,
    pub pr_url: String,
    pub scm_repo: String,
    pub pr_number: i64,
    pub files_changed: Vec<String>,
    pub file_types: HashMap<String, usize>,
    pub change_categories: Vec<ChangeCategory>,
    pub diff_summary: String,
    pub created_at: DateTime<Utc>,
}

/// A standing instruction promoted from repeated Q&A or review patterns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromotedInstruction {
    pub id: i64,
    pub repo: String,
    pub source_type: String,
    pub instruction_text: String,
    pub occurrence_count: i64,
    pub confidence: f64,
    pub is_active: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Per-repo accumulated knowledge entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepoKnowledge {
    pub id: i64,
    pub repo: String,
    pub knowledge_key: String,
    pub knowledge_value: String,
    pub source_type: String,
    pub confidence: f64,
    pub occurrence_count: i64,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Category of review feedback.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReviewCategory {
    MissingTests,
    StyleIssue,
    WrongApproach,
    Incomplete,
    Security,
    Performance,
    Documentation,
    Other,
}

impl std::fmt::Display for ReviewCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingTests => write!(f, "missing_tests"),
            Self::StyleIssue => write!(f, "style_issue"),
            Self::WrongApproach => write!(f, "wrong_approach"),
            Self::Incomplete => write!(f, "incomplete"),
            Self::Security => write!(f, "security"),
            Self::Performance => write!(f, "performance"),
            Self::Documentation => write!(f, "documentation"),
            Self::Other => write!(f, "other"),
        }
    }
}

impl ReviewCategory {
    pub fn parse(s: &str) -> Self {
        match s {
            "missing_tests" => Self::MissingTests,
            "style_issue" => Self::StyleIssue,
            "wrong_approach" => Self::WrongApproach,
            "incomplete" => Self::Incomplete,
            "security" => Self::Security,
            "performance" => Self::Performance,
            "documentation" => Self::Documentation,
            _ => Self::Other,
        }
    }

    /// Return all classified variants (excluding Other).
    pub fn classified_variants() -> &'static [ReviewCategory] {
        &[
            ReviewCategory::MissingTests,
            ReviewCategory::StyleIssue,
            ReviewCategory::WrongApproach,
            ReviewCategory::Incomplete,
            ReviewCategory::Security,
            ReviewCategory::Performance,
            ReviewCategory::Documentation,
        ]
    }
}

/// A classified review feedback pattern.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewPattern {
    pub id: i64,
    pub scm_repo: String,
    pub category: ReviewCategory,
    pub pattern_text: String,
    pub example_comments: Vec<String>,
    pub occurrence_count: i64,
    pub promoted_to_instruction: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// How Claude approached fixing an issue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyFingerprint {
    pub id: i64,
    pub attempt_id: i64,
    pub files_explored: Vec<String>,
    pub tests_run: i64,
    pub tools_used: HashMap<String, i64>,
    pub fix_approach: String,
    pub strategy_summary: String,
    pub fix_quality_score: Option<f64>,
    pub created_at: DateTime<Utc>,
}

/// Quality score for a fix based on merge velocity and review feedback.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixQualityScore {
    pub score: f64,
    pub merge_speed_component: f64,
    pub review_cycles_component: f64,
    pub approval_component: f64,
}

/// A cluster of correlated issues arriving in a time window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IssueCluster {
    pub id: i64,
    pub cluster_key: String,
    pub source: String,
    pub issue_ids: Vec<String>,
    pub window_start: DateTime<Utc>,
    pub window_end: DateTime<Utc>,
    pub resolved_by_issue_id: Option<String>,
    pub resolved_by_attempt_id: Option<i64>,
    pub status: String,
    pub created_at: DateTime<Utc>,
}

/// A member of an issue cluster.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IssueClusterMember {
    pub cluster_id: i64,
    pub issue_id: String,
    pub arrived_at: DateTime<Utc>,
}

/// Learnings extracted from Claude's execution log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedLearnings {
    pub root_cause: Option<String>,
    pub files_modified: Vec<String>,
    pub strategy_used: Option<String>,
    pub tests_added: bool,
    pub key_decisions: Vec<String>,
}

// --- Types relocated from feedback/outcomes.rs ---

/// The outcome of a fix attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Outcome {
    /// PR was merged successfully
    #[serde(rename = "merged")]
    Merged,
    /// PR was closed without merging
    #[serde(rename = "closed")]
    Closed,
    /// Fix attempt failed before creating PR
    #[serde(rename = "failed")]
    Failed,
    /// Issue could not be fixed after retries
    #[serde(rename = "cannot_fix")]
    CannotFix,
}

impl Outcome {
    /// Whether this outcome is considered successful.
    pub fn is_success(&self) -> bool {
        matches!(self, Outcome::Merged)
    }

    /// Parse from string.
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "merged" => Some(Outcome::Merged),
            "closed" => Some(Outcome::Closed),
            "failed" => Some(Outcome::Failed),
            "cannot_fix" | "cannotfix" => Some(Outcome::CannotFix),
            _ => None,
        }
    }

    /// Convert to string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Outcome::Merged => "merged",
            Outcome::Closed => "closed",
            Outcome::Failed => "failed",
            Outcome::CannotFix => "cannot_fix",
        }
    }
}

/// A recorded fix outcome with associated learnings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixOutcome {
    /// Unique identifier.
    pub id: i64,
    /// Associated fix attempt ID.
    pub attempt_id: i64,
    /// Source of the issue.
    pub source: String,
    /// Issue ID.
    pub issue_id: String,
    /// Issue title/description (for similarity matching).
    pub issue_text: String,
    /// Prompt that was used.
    pub prompt_used: String,
    /// Outcome of the fix.
    pub outcome: Outcome,
    /// Categorized error type (if failed).
    pub error_type: Option<String>,
    /// AI-generated learnings from this outcome.
    pub learnings: Option<String>,
    /// Keywords extracted from the issue (fallback for similarity when no embedding).
    pub keywords: Vec<String>,
    /// Embedding vector for semantic similarity (optional - computed async).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
    /// When this outcome was recorded.
    pub created_at: DateTime<Utc>,
}

impl FixOutcome {
    /// Create a new outcome from a fix attempt.
    pub fn from_attempt(
        attempt: &FixAttempt,
        issue: &Issue,
        prompt: &str,
        outcome: Outcome,
    ) -> Self {
        let description = issue.description.as_deref().unwrap_or("");
        let keywords = Self::extract_keywords(&issue.title, description);

        Self {
            id: 0, // Set by storage
            attempt_id: attempt.id,
            source: attempt.source.clone(),
            issue_id: attempt.issue_id.clone(),
            issue_text: format!("{}\n\n{}", issue.title, description),
            prompt_used: prompt.to_string(),
            outcome,
            error_type: attempt
                .error_message
                .as_ref()
                .map(|e| Self::categorize_error(e)),
            learnings: None,
            keywords,
            embedding: None, // Set async via set_embedding()
            created_at: Utc::now(),
        }
    }

    /// Set the embedding vector for this outcome.
    pub fn set_embedding(&mut self, embedding: Vec<f32>) {
        self.embedding = Some(embedding);
    }

    /// Extract keywords from title and description.
    pub fn extract_keywords(title: &str, description: &str) -> Vec<String> {
        let text = format!("{} {}", title, description).to_lowercase();

        let significant_words: Vec<&str> = text
            .split(|c: char| !c.is_alphanumeric() && c != '_')
            .filter(|w| w.len() > 3)
            .filter(|w| !is_common_word(w))
            .take(20)
            .collect();

        significant_words.into_iter().map(String::from).collect()
    }

    /// Categorize an error message into a type.
    pub fn categorize_error(error: &str) -> String {
        let error_lower = error.to_lowercase();

        if error_lower.contains("timeout") || error_lower.contains("timed out") {
            "timeout".to_string()
        } else if error_lower.contains("permission") || error_lower.contains("access denied") {
            "permission".to_string()
        } else if error_lower.contains("syntax") || error_lower.contains("parse") {
            "syntax".to_string()
        } else if error_lower.contains("test") && error_lower.contains("fail") {
            "test_failure".to_string()
        } else if error_lower.contains("build") && error_lower.contains("fail") {
            "build_failure".to_string()
        } else if error_lower.contains("not found") || error_lower.contains("missing") {
            "not_found".to_string()
        } else if error_lower.contains("conflict") {
            "conflict".to_string()
        } else {
            "unknown".to_string()
        }
    }

    /// Calculate semantic similarity score with another outcome (0.0 to 1.0).
    ///
    /// Uses cosine similarity on embeddings. Returns 0.0 if either outcome
    /// lacks an embedding.
    pub fn similarity(&self, other: &FixOutcome) -> f64 {
        if let (Some(ref self_emb), Some(ref other_emb)) = (&self.embedding, &other.embedding) {
            return cosine_similarity_f32(self_emb, other_emb) as f64;
        }

        0.0
    }
}

/// Check if a word is too common to be a useful keyword.
pub fn is_common_word(word: &str) -> bool {
    const COMMON_WORDS: &[&str] = &[
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "need",
        "dare",
        "this",
        "that",
        "these",
        "those",
        "what",
        "which",
        "who",
        "whom",
        "when",
        "where",
        "why",
        "how",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "than",
        "too",
        "very",
        "just",
        "also",
        "only",
        "now",
        "then",
        "here",
        "there",
        "with",
        "from",
        "into",
        "onto",
        "upon",
        "over",
        "under",
        "above",
        "below",
        "between",
        "among",
        "through",
        "during",
        "before",
        "after",
        "about",
        "against",
        "without",
        "within",
        "throughout",
        "around",
        "and",
        "but",
        "or",
        "nor",
        "for",
        "yet",
        "so",
        "because",
        "although",
        "while",
        "if",
        "unless",
        "until",
        "since",
        "once",
        "whereas",
        "error",
        "issue",
        "problem",
        "bug",
        "fix",
        "fixed",
        "fixing",
    ];

    COMMON_WORDS.contains(&word)
}

/// Cosine similarity between two f32 vectors.
pub fn cosine_similarity_f32(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let (dot_product, norm_a, norm_b) = a
        .iter()
        .zip(b.iter())
        .fold((0.0f32, 0.0f32, 0.0f32), |(dot, na, nb), (&x, &y)| {
            (dot + x * y, na + x * x, nb + y * y)
        });

    let norm_a = norm_a.sqrt();
    let norm_b = norm_b.sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a * norm_b)
}

// --- Types relocated from learning/cross_repo_correlator.rs ---

/// A detected correlation between two repos having concurrent issues.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossRepoCorrelation {
    pub id: i64,
    pub repo_a: String,
    pub repo_b: String,
    pub correlation_count: i64,
    pub last_seen_at: DateTime<Utc>,
    pub window_hours: i64,
}

// ── Code-index types (shared by storage + analysis) ──────────────────────────

/// Supported programming languages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Language {
    Rust,
    TypeScript,
    Tsx,
    JavaScript,
    Python,
    Go,
    Java,
    C,
    Cpp,
    Ruby,
    Php,
    Swift,
    Kotlin,
    CSharp,
    Dart,
    Yaml,
    Json,
    Dockerfile,
    Lua,
}

impl Language {
    /// Return the display name used in context text and DB storage.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Rust => "Rust",
            Self::TypeScript => "TypeScript",
            Self::Tsx => "TSX",
            Self::JavaScript => "JavaScript",
            Self::Python => "Python",
            Self::Go => "Go",
            Self::Java => "Java",
            Self::C => "C",
            Self::Cpp => "C++",
            Self::Ruby => "Ruby",
            Self::Php => "PHP",
            Self::Swift => "Swift",
            Self::Kotlin => "Kotlin",
            Self::CSharp => "C#",
            Self::Dart => "Dart",
            Self::Yaml => "YAML",
            Self::Json => "JSON",
            Self::Dockerfile => "Dockerfile",
            Self::Lua => "Lua",
        }
    }

    /// Infer language from a file extension.
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext {
            "rs" => Some(Self::Rust),
            "ts" => Some(Self::TypeScript),
            "tsx" => Some(Self::Tsx),
            "js" | "jsx" | "mjs" | "cjs" => Some(Self::JavaScript),
            "py" | "pyi" => Some(Self::Python),
            "go" => Some(Self::Go),
            "java" => Some(Self::Java),
            "c" | "h" => Some(Self::C),
            "cpp" | "cc" | "cxx" | "hpp" | "hxx" | "hh" => Some(Self::Cpp),
            "rb" => Some(Self::Ruby),
            "php" => Some(Self::Php),
            "swift" => Some(Self::Swift),
            "kt" | "kts" => Some(Self::Kotlin),
            "cs" => Some(Self::CSharp),
            "dart" => Some(Self::Dart),
            "yaml" | "yml" => Some(Self::Yaml),
            "json" => Some(Self::Json),
            "lua" => Some(Self::Lua),
            _ => None,
        }
    }

    /// Infer language from a filename (for extensionless files like `Dockerfile`).
    pub fn from_filename(name: &str) -> Option<Self> {
        match name {
            "Dockerfile" => Some(Self::Dockerfile),
            _ => None,
        }
    }
}

impl std::fmt::Display for Language {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Kind of extracted symbol.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SymbolKind {
    Function,
    Class,
    Method,
    Struct,
    Impl,
    Interface,
    Trait,
    Enum,
    Module,
    Constant,
}

impl SymbolKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Function => "function",
            Self::Class => "class",
            Self::Method => "method",
            Self::Struct => "struct",
            Self::Impl => "impl",
            Self::Interface => "interface",
            Self::Trait => "trait",
            Self::Enum => "enum",
            Self::Module => "module",
            Self::Constant => "constant",
        }
    }

    pub fn from_str_loose(s: &str) -> Option<Self> {
        match s {
            "function" => Some(Self::Function),
            "class" => Some(Self::Class),
            "method" => Some(Self::Method),
            "struct" => Some(Self::Struct),
            "impl" => Some(Self::Impl),
            "interface" => Some(Self::Interface),
            "trait" => Some(Self::Trait),
            "enum" => Some(Self::Enum),
            "module" => Some(Self::Module),
            "constant" => Some(Self::Constant),
            _ => None,
        }
    }
}

impl std::fmt::Display for SymbolKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// An extracted code symbol (function, class, struct, etc.).
#[derive(Debug, Clone)]
pub struct CodeSymbol {
    pub id: Option<i64>,
    pub repo_id: i64,
    pub file_path: String,
    pub symbol_name: String,
    pub symbol_kind: SymbolKind,
    pub parent_symbol: Option<String>,
    pub language: Language,
    pub start_line: usize,
    pub end_line: usize,
    pub signature: Option<String>,
}

/// A semantic code chunk for embedding.
#[derive(Debug, Clone)]
pub struct CodeChunk {
    pub id: Option<i64>,
    pub repo_id: i64,
    pub file_path: String,
    pub chunk_type: String,
    pub symbol_name: Option<String>,
    pub language: Language,
    pub start_line: usize,
    pub end_line: usize,
    pub chunk_text: String,
    pub context_text: String,
    pub file_hash: String,
    pub content_hash: Option<String>,
}

/// A code search result with similarity score.
#[derive(Debug, Clone)]
pub struct CodeSearchResult {
    pub chunk: CodeChunk,
    pub score: f64,
}

/// Statistics from a code indexing run.
#[derive(Debug, Clone, Default)]
pub struct CodeIndexStats {
    pub files_processed: usize,
    pub files_skipped: usize,
    pub files_failed: usize,
    pub symbols_extracted: usize,
    pub chunks_created: usize,
    pub embeddings_generated: usize,
}

impl std::fmt::Display for CodeIndexStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "processed={}, skipped={}, failed={}, symbols={}, chunks={}, embeddings={}",
            self.files_processed,
            self.files_skipped,
            self.files_failed,
            self.symbols_extracted,
            self.chunks_created,
            self.embeddings_generated,
        )
    }
}

/// Complexity metrics for a single function.
#[derive(Debug, Clone, Default)]
pub struct FunctionComplexity {
    pub name: String,
    pub start_line: u32,
    pub end_line: u32,
    pub cyclomatic: u32,
    pub line_count: u32,
    pub max_nesting: u32,
}

/// Complexity metrics for a single file.
#[derive(Debug, Clone, Default)]
pub struct FileComplexity {
    pub file_path: String,
    pub total_lines: u32,
    pub function_count: u32,
    pub functions: Vec<FunctionComplexity>,
    pub avg_cyclomatic: f64,
    pub max_cyclomatic: f64,
    pub avg_func_length: f64,
    pub max_func_length: f64,
    pub avg_nesting: f64,
    pub max_nesting: f64,
}

// ── Repository index types (shared by storage + analysis) ────────────────────

/// An indexed repository with file information.
#[derive(Debug, Clone)]
pub struct IndexedRepo {
    /// Repository name (e.g., "appwrite/cloud").
    pub name: String,
    /// Local filesystem path.
    pub path: std::path::PathBuf,
    /// GitHub URL inferred from org + name.
    pub scm_url: String,
    /// Relative file paths within the repository.
    pub files: Vec<String>,
    /// Default branch name.
    pub default_branch: String,
}

impl IndexedRepo {
    /// Create a new indexed repository.
    pub fn new(name: impl Into<String>, path: impl Into<std::path::PathBuf>) -> Self {
        let name = name.into();
        let scm_url = format!("https://github.com/{}", name);
        Self {
            name,
            path: path.into(),
            scm_url,
            files: Vec::new(),
            default_branch: "main".to_string(),
        }
    }

    /// Create a repository discovered via GitHub API.
    pub fn from_api(
        name: impl Into<String>,
        scm_url: impl Into<String>,
        default_branch: impl Into<String>,
        workspace: &std::path::Path,
    ) -> Self {
        let name = name.into();
        let repo_name = name.split('/').next_back().unwrap_or(&name);
        let path = workspace.join(repo_name);
        Self {
            name,
            path,
            scm_url: scm_url.into(),
            files: Vec::new(),
            default_branch: default_branch.into(),
        }
    }

    /// Set the SCM URL.
    pub fn with_scm_url(mut self, url: impl Into<String>) -> Self {
        self.scm_url = url.into();
        self
    }

    /// Set the default branch.
    pub fn with_default_branch(mut self, branch: impl Into<String>) -> Self {
        self.default_branch = branch.into();
        self
    }

    /// Check if this repo contains a file with the given name.
    pub fn has_file(&self, filename: &str) -> bool {
        let filename_lower = filename.to_lowercase();
        self.files.iter().any(|f| {
            f.to_lowercase().ends_with(&filename_lower)
                || f.to_lowercase().contains(&filename_lower)
        })
    }

    /// Find all files matching a query.
    pub fn find_files(&self, query: &str) -> Vec<&str> {
        let query_lower = query.to_lowercase();
        self.files
            .iter()
            .filter(|f| f.to_lowercase().contains(&query_lower))
            .map(|s| s.as_str())
            .collect()
    }
}

/// Index of discovered repositories.
#[derive(Debug, Default, Clone)]
pub struct RepoIndex {
    /// Indexed repositories keyed by name.
    repos: std::collections::HashMap<String, IndexedRepo>,
    /// File path to repository name mapping for fast lookups.
    file_index: std::collections::HashMap<String, String>,
    /// Known renames: old name -> current name.
    aliases: std::collections::HashMap<String, String>,
}

impl RepoIndex {
    /// Create a new empty index.
    pub fn new() -> Self {
        Self {
            repos: std::collections::HashMap::new(),
            file_index: std::collections::HashMap::new(),
            aliases: std::collections::HashMap::new(),
        }
    }

    /// Register a known repository rename so vendor-path lookups can resolve
    /// the old name to the current indexed repo.
    pub fn add_alias(&mut self, former_name: &str, current_name: &str) {
        self.aliases
            .insert(former_name.to_string(), current_name.to_string());
    }

    /// Add a repository to the index.
    pub fn add_repo(&mut self, repo: IndexedRepo) {
        // Index all files by full path for fast exact lookup
        for file in &repo.files {
            self.file_index.insert(file.clone(), repo.name.clone());
        }
        self.repos.insert(repo.name.clone(), repo);
    }

    /// Merge another RepoIndex into this one.
    pub fn merge(&mut self, other: Self) {
        for (_, repo) in other.repos {
            self.add_repo(repo);
        }
    }

    /// Find a repository by exact file path match.
    pub fn find_by_file(&self, filename: &str) -> Option<&IndexedRepo> {
        if let Some(repo_name) = self.file_index.get(filename) {
            return self.repos.get(repo_name);
        }
        if let Some(repo) = self.find_by_vendor_path(filename) {
            return Some(repo);
        }
        None
    }

    /// Extract repository name from vendor paths.
    pub fn find_by_vendor_path(&self, filename: &str) -> Option<&IndexedRepo> {
        let vendor_idx = filename.find("/vendor/")?;
        let after_vendor = &filename[vendor_idx + 8..];
        let parts: Vec<&str> = after_vendor.split('/').collect();
        if parts.len() >= 2 {
            let repo_name = format!("{}/{}", parts[0], parts[1]);

            // Direct lookup by current name
            if let Some(repo) = self.repos.get(&repo_name) {
                return Some(repo);
            }

            // Fall back to aliases (handles renames, e.g. utopia-php/framework -> utopia-php/http)
            if let Some(current_name) = self.aliases.get(&repo_name) {
                if let Some(repo) = self.repos.get(current_name) {
                    return Some(repo);
                }
            }
        }
        None
    }

    /// Search for files matching a query across all repositories.
    pub fn search_files(&self, query: &str) -> Vec<(&IndexedRepo, &str)> {
        let mut results = Vec::new();
        for repo in self.repos.values() {
            for file in repo.find_files(query) {
                results.push((repo, file));
            }
        }
        results
    }

    /// Get a repository by name.
    pub fn get(&self, name: &str) -> Option<&IndexedRepo> {
        self.repos.get(name)
    }

    /// Get a mutable reference to a repository by name.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut IndexedRepo> {
        self.repos.get_mut(name)
    }

    /// Remove and return a repository by name.
    pub fn remove(&mut self, name: &str) -> Option<IndexedRepo> {
        self.repos.remove(name)
    }

    /// Get all indexed repositories.
    pub fn list(&self) -> Vec<&IndexedRepo> {
        self.repos.values().collect()
    }

    /// Get the number of indexed repositories.
    pub fn len(&self) -> usize {
        self.repos.len()
    }

    /// Check if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.repos.is_empty()
    }

    /// Get total file count across all repositories.
    pub fn total_files(&self) -> usize {
        self.repos.values().map(|r| r.files.len()).sum()
    }
}

// ── SCM review types (shared by storage + integrations) ──────────────────────

/// A user who participated in a review.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewUser {
    pub id: i64,
    pub login: String,
    #[serde(rename = "type")]
    pub user_type: Option<String>,
}

/// An inline review comment on a PR/MR.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewComment {
    pub id: i64,
    pub path: String,
    pub position: Option<i64>,
    pub original_position: Option<i64>,
    pub body: String,
    pub user: ReviewUser,
    pub created_at: String,
    pub updated_at: String,
    pub html_url: String,
    pub pull_request_review_id: Option<i64>,
    pub line: Option<i64>,
    pub start_line: Option<i64>,
    pub side: Option<String>,
}

/// Persistent state for tracking review polling cursors on a single PR.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrReviewState {
    pub pr_url: String,
    pub repo: String,
    pub pr_number: i64,
    pub issue_id: String,
    pub source: String,
    pub last_review_id: Option<i64>,
    pub last_review_time: Option<String>,
    pub last_comment_id: Option<i64>,
    pub last_comment_time: Option<String>,
    pub is_active: bool,
}

impl PrReviewState {
    pub fn new(
        pr_url: impl Into<String>,
        repo: impl Into<String>,
        pr_number: i64,
        issue_id: impl Into<String>,
        source: impl Into<String>,
    ) -> Self {
        Self {
            pr_url: pr_url.into(),
            repo: repo.into(),
            pr_number,
            issue_id: issue_id.into(),
            source: source.into(),
            last_review_id: None,
            last_review_time: None,
            last_comment_id: None,
            last_comment_time: None,
            is_active: true,
        }
    }
}

// ── Evaluation types (shared by storage + analysis) ──────────────────────────

/// Category of evaluation tool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvalCategory {
    Test,
    Lint,
    StaticAnalysis,
    Coverage,
}

impl std::fmt::Display for EvalCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Test => write!(f, "test"),
            Self::Lint => write!(f, "lint"),
            Self::StaticAnalysis => write!(f, "static_analysis"),
            Self::Coverage => write!(f, "coverage"),
        }
    }
}

/// Severity of a diagnostic finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticSeverity {
    Error,
    Warning,
    Info,
}

/// A single diagnostic finding from a tool.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Diagnostic {
    pub file: String,
    pub line: Option<u32>,
    pub column: Option<u32>,
    pub severity: DiagnosticSeverity,
    pub code: Option<String>,
    pub message: String,
}

/// A snapshot of tool output at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalSnapshot {
    pub category: EvalCategory,
    pub tool_name: String,
    pub exit_code: i32,
    pub passed: u32,
    pub failed: u32,
    pub skipped: u32,
    pub warnings: u32,
    pub errors: u32,
    pub diagnostics: Vec<Diagnostic>,
    pub raw_output: String,
    pub duration_secs: f64,
    pub line_coverage_pct: Option<f64>,
    pub branch_coverage_pct: Option<f64>,
}

impl Default for EvalSnapshot {
    fn default() -> Self {
        Self {
            category: EvalCategory::Test,
            tool_name: String::new(),
            exit_code: -1,
            passed: 0,
            failed: 0,
            skipped: 0,
            warnings: 0,
            errors: 0,
            diagnostics: Vec::new(),
            raw_output: String::new(),
            duration_secs: 0.0,
            line_coverage_pct: None,
            branch_coverage_pct: None,
        }
    }
}

/// Computed delta between before and after snapshots.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalDelta {
    pub before: EvalSnapshot,
    pub after: EvalSnapshot,
    pub new_passes: i32,
    pub new_failures: i32,
    pub regressions: Vec<Diagnostic>,
    pub fixed: Vec<Diagnostic>,
    pub coverage_delta_pct: Option<f64>,
}

impl EvalDelta {
    pub fn compute(before: EvalSnapshot, after: EvalSnapshot) -> Self {
        let new_passes = after.passed as i32 - before.passed as i32;
        let new_failures = after.failed as i32 - before.failed as i32;

        let before_diags: std::collections::HashSet<_> = before.diagnostics.iter().collect();
        let after_diags: std::collections::HashSet<_> = after.diagnostics.iter().collect();

        let regressions: Vec<Diagnostic> = after
            .diagnostics
            .iter()
            .filter(|d| !before_diags.contains(d))
            .cloned()
            .collect();

        let fixed: Vec<Diagnostic> = before
            .diagnostics
            .iter()
            .filter(|d| !after_diags.contains(d))
            .cloned()
            .collect();

        let coverage_delta_pct = match (before.line_coverage_pct, after.line_coverage_pct) {
            (Some(b), Some(a)) => Some(a - b),
            _ => None,
        };

        Self {
            before,
            after,
            new_passes,
            new_failures,
            regressions,
            fixed,
            coverage_delta_pct,
        }
    }

    pub fn is_improvement(&self) -> bool {
        self.new_failures <= 0
            && self.regressions.is_empty()
            && self.coverage_delta_pct.is_none_or(|d| d >= 0.0)
    }
}

// ── Chat types (shared by storage + engine) ──────────────────────────────────

/// Role in a chat conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    User,
    Assistant,
}

impl ChatRole {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "user" => Some(Self::User),
            "assistant" => Some(Self::Assistant),
            _ => None,
        }
    }
}

/// A single message in a chat conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub id: Option<i64>,
    pub role: ChatRole,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sources_json: Option<String>,
    pub created_at: DateTime<Utc>,
}

/// A chat session with its message history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatSession {
    pub id: String,
    pub repo_id: Option<i64>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    #[serde(default)]
    pub messages: Vec<ChatMessage>,
}

/// Alias for backward compatibility (was `PrReviewComment` in github.rs).
pub type PrReviewComment = ReviewComment;

/// Normalize free-form text for exact-match fallback (lowercased, whitespace-collapsed).
pub fn normalize_text(input: &str) -> String {
    input
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Trait for Sentry HTTP client operations to enable testing.
#[async_trait::async_trait]
pub trait SentryHttpClient: Send + Sync {
    /// Perform a GET request with bearer auth.
    async fn get(&self, url: &str, auth_token: &str) -> crate::Result<crate::http::HttpResponse>;

    /// Perform a PUT request with bearer auth and JSON body.
    async fn put(
        &self,
        url: &str,
        auth_token: &str,
        body: serde_json::Value,
    ) -> crate::Result<crate::http::HttpResponse>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_issue_creation() {
        let issue = Issue::new(
            "123",
            "PROJ-123",
            "Fix bug",
            "https://example.com",
            "linear",
        );
        assert_eq!(issue.id, "123");
        assert_eq!(issue.short_id, "PROJ-123");
        assert_eq!(issue.source, "linear");
        assert_eq!(issue.priority, IssuePriority::None);
        assert_eq!(issue.status, IssueStatus::Open);
    }

    #[test]
    fn test_issue_metadata() {
        let mut issue = Issue::new(
            "123",
            "PROJ-123",
            "Fix bug",
            "https://example.com",
            "linear",
        );
        issue.set_metadata("count", 42i64);
        assert_eq!(issue.get_metadata::<i64>("count"), Some(42));
    }

    #[test]
    fn test_match_result() {
        let matched = MatchResult::matched("Matches criteria", MatchPriority::High);
        assert!(matched.matches);
        assert_eq!(matched.priority, MatchPriority::High);

        let not_matched = MatchResult::not_matched("Does not match");
        assert!(!not_matched.matches);
    }

    #[test]
    fn test_issue_serialization() {
        let issue = Issue::new(
            "123",
            "PROJ-123",
            "Fix bug",
            "https://example.com",
            "linear",
        );
        let json = serde_json::to_string(&issue).unwrap();
        let deserialized: Issue = serde_json::from_str(&json).unwrap();
        assert_eq!(issue.id, deserialized.id);
        assert_eq!(issue.short_id, deserialized.short_id);
    }

    #[test]
    fn test_priority_ordering() {
        assert!(IssuePriority::Critical > IssuePriority::High);
        assert!(IssuePriority::High > IssuePriority::Medium);
        assert!(IssuePriority::Medium > IssuePriority::Low);
        assert!(IssuePriority::Low > IssuePriority::None);
    }

    #[test]
    fn test_fix_attempt_status_parsing() {
        assert_eq!(
            "pending".parse::<FixAttemptStatus>().unwrap(),
            FixAttemptStatus::Pending
        );
        assert_eq!(
            "SUCCESS".parse::<FixAttemptStatus>().unwrap(),
            FixAttemptStatus::Success
        );
        assert_eq!(
            "Failed".parse::<FixAttemptStatus>().unwrap(),
            FixAttemptStatus::Failed
        );
        assert_eq!(
            "merged".parse::<FixAttemptStatus>().unwrap(),
            FixAttemptStatus::Merged
        );
        assert_eq!(
            "CLOSED".parse::<FixAttemptStatus>().unwrap(),
            FixAttemptStatus::Closed
        );
        assert_eq!(
            "cannot_fix".parse::<FixAttemptStatus>().unwrap(),
            FixAttemptStatus::CannotFix
        );
    }

    #[test]
    fn test_fix_attempt_status_parsing_invalid() {
        assert!("invalid".parse::<FixAttemptStatus>().is_err());
        assert!("".parse::<FixAttemptStatus>().is_err());
    }

    #[test]
    fn test_fix_attempt_status_display() {
        assert_eq!(FixAttemptStatus::Pending.to_string(), "pending");
        assert_eq!(FixAttemptStatus::Success.to_string(), "success");
        assert_eq!(FixAttemptStatus::Failed.to_string(), "failed");
        assert_eq!(FixAttemptStatus::Merged.to_string(), "merged");
        assert_eq!(FixAttemptStatus::Closed.to_string(), "closed");
        assert_eq!(FixAttemptStatus::CannotFix.to_string(), "cannot_fix");
    }

    #[test]
    fn test_issue_priority_display() {
        assert_eq!(IssuePriority::None.to_string(), "none");
        assert_eq!(IssuePriority::Low.to_string(), "low");
        assert_eq!(IssuePriority::Medium.to_string(), "medium");
        assert_eq!(IssuePriority::High.to_string(), "high");
        assert_eq!(IssuePriority::Critical.to_string(), "critical");
    }

    #[test]
    fn test_issue_status_display() {
        assert_eq!(IssueStatus::Open.to_string(), "open");
        assert_eq!(IssueStatus::InProgress.to_string(), "in_progress");
        assert_eq!(IssueStatus::Resolved.to_string(), "resolved");
        assert_eq!(IssueStatus::Ignored.to_string(), "ignored");
    }

    #[test]
    fn test_issue_priority_default() {
        assert_eq!(IssuePriority::default(), IssuePriority::None);
    }

    #[test]
    fn test_issue_status_default() {
        assert_eq!(IssueStatus::default(), IssueStatus::Open);
    }

    #[test]
    fn test_match_priority_default() {
        assert_eq!(MatchPriority::default(), MatchPriority::Normal);
    }

    #[test]
    fn test_match_priority_ordering() {
        assert!(MatchPriority::Urgent > MatchPriority::High);
        assert!(MatchPriority::High > MatchPriority::Normal);
        assert!(MatchPriority::Normal > MatchPriority::Low);
    }

    #[test]
    fn test_issue_with_description() {
        let mut issue = Issue::new("1", "T-1", "Title", "url", "src");
        issue.description = Some("Description text".to_string());
        assert_eq!(issue.description.as_deref(), Some("Description text"));
    }

    #[test]
    fn test_issue_metadata_string() {
        let mut issue = Issue::new("1", "T-1", "Title", "url", "src");
        issue.set_metadata("key", "value");
        assert_eq!(
            issue.get_metadata::<String>("key"),
            Some("value".to_string())
        );
    }

    #[test]
    fn test_issue_metadata_bool() {
        let mut issue = Issue::new("1", "T-1", "Title", "url", "src");
        issue.set_metadata("flag", true);
        assert_eq!(issue.get_metadata::<bool>("flag"), Some(true));
    }

    #[test]
    fn test_issue_metadata_vec() {
        let mut issue = Issue::new("1", "T-1", "Title", "url", "src");
        issue.set_metadata("tags", vec!["a", "b", "c"]);
        assert_eq!(
            issue.get_metadata::<Vec<String>>("tags"),
            Some(vec!["a".to_string(), "b".to_string(), "c".to_string()])
        );
    }

    #[test]
    fn test_issue_metadata_missing_key() {
        let issue = Issue::new("1", "T-1", "Title", "url", "src");
        assert_eq!(issue.get_metadata::<String>("nonexistent"), None);
    }

    #[test]
    fn test_issue_metadata_wrong_type() {
        let mut issue = Issue::new("1", "T-1", "Title", "url", "src");
        issue.set_metadata("number", 42i64);
        // Trying to get a number as a string should fail
        assert_eq!(issue.get_metadata::<String>("number"), None);
    }

    #[test]
    fn test_claude_result_success() {
        let result = AgentResult {
            success: true,
            output: "PR created".to_string(),
            pr_url: Some("https://github.com/test/pr/1".to_string()),
            changelog: None,
            error: None,
            blocking_question: None,
            used_qa_ids: Vec::new(),
            confidence: 0,
            confidence_reasoning: None,
            wrong_repo: None,
        };
        assert!(result.success);
        assert!(result.pr_url.is_some());
        assert!(result.error.is_none());
    }

    #[test]
    fn test_claude_result_failure() {
        let result = AgentResult {
            success: false,
            output: "".to_string(),
            pr_url: None,
            changelog: None,
            error: Some("Build failed".to_string()),
            blocking_question: None,
            used_qa_ids: Vec::new(),
            confidence: 0,
            confidence_reasoning: None,
            wrong_repo: None,
        };
        assert!(!result.success);
        assert!(result.pr_url.is_none());
        assert!(result.error.is_some());
    }

    #[test]
    fn test_fix_attempt_stats_default() {
        let stats = FixAttemptStats::default();
        assert_eq!(stats.total, 0);
        assert_eq!(stats.pending, 0);
        assert_eq!(stats.success, 0);
        assert_eq!(stats.failed, 0);
        assert_eq!(stats.merged, 0);
        assert_eq!(stats.closed, 0);
        assert_eq!(stats.cannot_fix, 0);
        assert!(stats.by_source.is_empty());
    }

    #[test]
    fn test_source_stats_default() {
        let stats = SourceStats::default();
        assert_eq!(stats.total, 0);
        assert_eq!(stats.success, 0);
        assert_eq!(stats.failed, 0);
        assert_eq!(stats.merged, 0);
        assert_eq!(stats.closed, 0);
        assert_eq!(stats.cannot_fix, 0);
    }

    #[test]
    fn test_issue_priority_serde() {
        let priority = IssuePriority::High;
        let json = serde_json::to_string(&priority).unwrap();
        assert_eq!(json, "\"high\"");
        let parsed: IssuePriority = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, IssuePriority::High);
    }

    #[test]
    fn test_issue_status_serde() {
        let status = IssueStatus::InProgress;
        let json = serde_json::to_string(&status).unwrap();
        assert_eq!(json, "\"in_progress\"");
        let parsed: IssueStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, IssueStatus::InProgress);
    }

    #[test]
    fn test_fix_attempt_status_serde() {
        let status = FixAttemptStatus::CannotFix;
        let json = serde_json::to_string(&status).unwrap();
        // Serde now matches Display/FromStr: "cannot_fix"
        assert_eq!(json, "\"cannot_fix\"");
        let parsed: FixAttemptStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, FixAttemptStatus::CannotFix);
    }

    #[test]
    fn test_fix_attempt_status_serde_roundtrip_all_variants() {
        for status in [
            FixAttemptStatus::Pending,
            FixAttemptStatus::Success,
            FixAttemptStatus::Failed,
            FixAttemptStatus::Merged,
            FixAttemptStatus::Closed,
            FixAttemptStatus::CannotFix,
        ] {
            let json = serde_json::to_string(&status).unwrap();
            // Verify serde serialization matches Display
            assert_eq!(json, format!("\"{}\"", status));
            // Verify round-trip through serde
            let parsed: FixAttemptStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, status);
            // Verify round-trip through FromStr
            let display_str = status.to_string();
            let from_str: FixAttemptStatus = display_str.parse().unwrap();
            assert_eq!(from_str, status);
        }
    }

    #[test]
    fn test_match_result_serde() {
        let result = MatchResult::matched("test", MatchPriority::High);
        let json = serde_json::to_string(&result).unwrap();
        let parsed: MatchResult = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.matches, result.matches);
        assert_eq!(parsed.reason, result.reason);
        assert_eq!(parsed.priority, result.priority);
    }

    #[test]
    fn test_issue_full_serde() {
        let mut issue = Issue::new("id", "short", "title", "url", "source");
        issue.description = Some("desc".to_string());
        issue.priority = IssuePriority::Critical;
        issue.status = IssueStatus::InProgress;
        issue.set_metadata("key", "value");
        issue.created_at = Some(chrono::Utc::now());

        let json = serde_json::to_string(&issue).unwrap();
        let parsed: Issue = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.id, issue.id);
        assert_eq!(parsed.short_id, issue.short_id);
        assert_eq!(parsed.title, issue.title);
        assert_eq!(parsed.description, issue.description);
        assert_eq!(parsed.url, issue.url);
        assert_eq!(parsed.source, issue.source);
        assert_eq!(parsed.priority, issue.priority);
        assert_eq!(parsed.status, issue.status);
    }

    #[test]
    fn test_fix_attempt_created_time() {
        let attempt = FixAttempt {
            id: 1,
            source: "linear".to_string(),
            issue_id: "123".to_string(),
            short_id: "LIN-123".to_string(),
            status: FixAttemptStatus::Pending,
            pr_url: None,
            scm_repo: None,
            scm_pr_number: None,
            error_message: None,
            attempted_at: chrono::Utc::now(),
            resolved_at: None,
            merged_at: None,
            retry_count: 0,
            last_retry_at: None,
            issue_labels: vec![],
            parent_attempt_id: None,
            cascade_repo: None,
        };

        // Verify fields
        assert_eq!(attempt.id, 1);
        assert_eq!(attempt.source, "linear");
        assert_eq!(attempt.retry_count, 0);
        assert!(attempt.resolved_at.is_none());
    }

    #[test]
    fn test_fix_attempt_with_pr() {
        let attempt = FixAttempt {
            id: 1,
            source: "linear".to_string(),
            issue_id: "123".to_string(),
            short_id: "LIN-123".to_string(),
            status: FixAttemptStatus::Success,
            pr_url: Some("https://github.com/org/repo/pull/42".to_string()),
            scm_repo: Some("org/repo".to_string()),
            scm_pr_number: Some(42),
            error_message: None,
            attempted_at: chrono::Utc::now(),
            resolved_at: None,
            merged_at: None,
            retry_count: 0,
            last_retry_at: None,
            issue_labels: vec![],
            parent_attempt_id: None,
            cascade_repo: None,
        };

        assert_eq!(
            attempt.pr_url,
            Some("https://github.com/org/repo/pull/42".to_string())
        );
        assert_eq!(attempt.scm_repo, Some("org/repo".to_string()));
        assert_eq!(attempt.scm_pr_number, Some(42));
    }

    #[test]
    fn test_fix_attempt_status_all_variants() {
        assert_eq!(FixAttemptStatus::Pending.to_string(), "pending");
        assert_eq!(FixAttemptStatus::Success.to_string(), "success");
        assert_eq!(FixAttemptStatus::Failed.to_string(), "failed");
        assert_eq!(FixAttemptStatus::Merged.to_string(), "merged");
        assert_eq!(FixAttemptStatus::Closed.to_string(), "closed");
        assert_eq!(FixAttemptStatus::CannotFix.to_string(), "cannot_fix");
    }

    #[test]
    fn test_fix_attempt_status_serde_all_variants() {
        let statuses = vec![
            FixAttemptStatus::Pending,
            FixAttemptStatus::Success,
            FixAttemptStatus::Failed,
            FixAttemptStatus::Merged,
            FixAttemptStatus::Closed,
            FixAttemptStatus::CannotFix,
        ];

        for status in statuses {
            let json = serde_json::to_string(&status).unwrap();
            let parsed: FixAttemptStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, status);
        }
    }

    #[test]
    fn test_issue_priority_serde_all_variants() {
        let priorities = vec![
            IssuePriority::None,
            IssuePriority::Low,
            IssuePriority::Medium,
            IssuePriority::High,
            IssuePriority::Critical,
        ];

        for priority in priorities {
            let json = serde_json::to_string(&priority).unwrap();
            let parsed: IssuePriority = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, priority);
        }
    }

    #[test]
    fn test_issue_status_serde_all_variants() {
        let statuses = vec![
            IssueStatus::Open,
            IssueStatus::InProgress,
            IssueStatus::Resolved,
            IssueStatus::Ignored,
        ];

        for status in statuses {
            let json = serde_json::to_string(&status).unwrap();
            let parsed: IssueStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, status);
        }
    }

    #[test]
    fn test_match_priority_serde_all_variants() {
        let priorities = vec![
            MatchPriority::Low,
            MatchPriority::Normal,
            MatchPriority::High,
            MatchPriority::Urgent,
        ];

        for priority in priorities {
            let json = serde_json::to_string(&priority).unwrap();
            let parsed: MatchPriority = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, priority);
        }
    }

    #[test]
    fn test_issue_metadata_number() {
        let mut issue = Issue::new("1", "T-1", "Title", "url", "src");
        issue.set_metadata("count", 42i64);
        assert_eq!(issue.get_metadata::<i64>("count"), Some(42));
    }

    #[test]
    fn test_issue_metadata_nested_object() {
        let mut issue = Issue::new("1", "T-1", "Title", "url", "src");
        issue.set_metadata("nested", serde_json::json!({"a": 1, "b": "test"}));

        let nested: serde_json::Value = issue.get_metadata("nested").unwrap();
        assert_eq!(nested["a"], 1);
        assert_eq!(nested["b"], "test");
    }

    #[test]
    fn test_issue_clone() {
        let mut original = Issue::new("id", "short", "title", "url", "source");
        original.description = Some("desc".to_string());
        original.priority = IssuePriority::High;
        original.set_metadata("key", "value");

        let cloned = original.clone();

        assert_eq!(cloned.id, original.id);
        assert_eq!(cloned.description, original.description);
        assert_eq!(cloned.priority, original.priority);
    }

    #[test]
    fn test_match_result_with_empty_reason() {
        let result = MatchResult::not_matched("");
        assert!(!result.matches);
        assert!(result.reason.is_empty());
    }

    #[test]
    #[expect(clippy::field_reassign_with_default)]
    fn test_fix_attempt_stats_with_data() {
        let mut stats = FixAttemptStats::default();
        stats.total = 100;
        stats.pending = 10;
        stats.success = 50;
        stats.failed = 20;
        stats.merged = 15;
        stats.closed = 3;
        stats.cannot_fix = 2;

        assert_eq!(stats.total, 100);
        assert_eq!(
            stats.pending
                + stats.success
                + stats.failed
                + stats.merged
                + stats.closed
                + stats.cannot_fix,
            100
        );
    }

    #[test]
    #[expect(clippy::field_reassign_with_default)]
    fn test_source_stats_with_data() {
        let mut stats = SourceStats::default();
        stats.total = 50;
        stats.success = 30;
        stats.failed = 10;
        stats.merged = 8;
        stats.closed = 2;
        stats.cannot_fix = 0;

        assert_eq!(stats.total, 50);
    }

    #[test]
    fn test_claude_result_empty_output() {
        let result = AgentResult {
            success: false,
            output: "".to_string(),
            pr_url: None,
            changelog: None,
            error: Some("No output".to_string()),
            blocking_question: None,
            used_qa_ids: Vec::new(),
            confidence: 0,
            confidence_reasoning: None,
            wrong_repo: None,
        };

        assert!(result.output.is_empty());
        assert!(result.error.is_some());
    }

    #[test]
    fn test_issue_with_updated_at() {
        let mut issue = Issue::new("1", "T-1", "Title", "url", "src");
        assert!(issue.updated_at.is_none());

        issue.updated_at = Some(chrono::Utc::now());
        assert!(issue.updated_at.is_some());
    }

    #[test]
    fn test_match_result_debug_format() {
        let result = MatchResult::matched("Test reason", MatchPriority::High);
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("matches"));
        assert!(debug_str.contains("priority"));
    }

    #[test]
    fn test_issue_debug_format() {
        let issue = Issue::new("1", "T-1", "Title", "url", "src");
        let debug_str = format!("{:?}", issue);
        assert!(debug_str.contains("Issue"));
    }

    #[test]
    fn test_regression_watch_status_display() {
        assert_eq!(
            RegressionWatchStatus::AwaitingRelease.to_string(),
            "awaiting_release"
        );
        assert_eq!(RegressionWatchStatus::Monitoring.to_string(), "monitoring");
        assert_eq!(RegressionWatchStatus::Resolved.to_string(), "resolved");
        assert_eq!(RegressionWatchStatus::Regressed.to_string(), "regressed");
    }

    #[test]
    fn test_regression_watch_status_from_str() {
        assert_eq!(
            "awaiting_release".parse::<RegressionWatchStatus>().unwrap(),
            RegressionWatchStatus::AwaitingRelease
        );
        assert_eq!(
            "monitoring".parse::<RegressionWatchStatus>().unwrap(),
            RegressionWatchStatus::Monitoring
        );
        assert_eq!(
            "resolved".parse::<RegressionWatchStatus>().unwrap(),
            RegressionWatchStatus::Resolved
        );
        assert_eq!(
            "regressed".parse::<RegressionWatchStatus>().unwrap(),
            RegressionWatchStatus::Regressed
        );
    }

    #[test]
    fn test_regression_watch_status_from_str_case_insensitive() {
        assert_eq!(
            "AWAITING_RELEASE".parse::<RegressionWatchStatus>().unwrap(),
            RegressionWatchStatus::AwaitingRelease
        );
        assert_eq!(
            "Monitoring".parse::<RegressionWatchStatus>().unwrap(),
            RegressionWatchStatus::Monitoring
        );
    }

    #[test]
    fn test_regression_watch_status_from_str_invalid() {
        assert!("invalid".parse::<RegressionWatchStatus>().is_err());
        assert!("".parse::<RegressionWatchStatus>().is_err());
    }

    #[test]
    fn test_regression_watch_status_default() {
        assert_eq!(
            RegressionWatchStatus::default(),
            RegressionWatchStatus::AwaitingRelease
        );
    }

    #[test]
    fn test_regression_watch_status_serde() {
        let status = RegressionWatchStatus::Monitoring;
        let json = serde_json::to_string(&status).unwrap();
        assert_eq!(json, "\"monitoring\"");
        let parsed: RegressionWatchStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, RegressionWatchStatus::Monitoring);
    }

    #[test]
    fn test_regression_watch_status_serde_all_variants() {
        let statuses = vec![
            RegressionWatchStatus::AwaitingRelease,
            RegressionWatchStatus::Monitoring,
            RegressionWatchStatus::Resolved,
            RegressionWatchStatus::Regressed,
        ];

        for status in statuses {
            let json = serde_json::to_string(&status).unwrap();
            let parsed: RegressionWatchStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, status);
        }
    }

    #[test]
    fn test_issue_type_display() {
        assert_eq!(IssueType::SentryIssue.to_string(), "sentry_issue");
        assert_eq!(IssueType::LinearBug.to_string(), "linear_bug");
    }

    #[test]
    fn test_issue_type_from_str() {
        assert_eq!(
            "sentry_issue".parse::<IssueType>().unwrap(),
            IssueType::SentryIssue
        );
        assert_eq!(
            "linear_bug".parse::<IssueType>().unwrap(),
            IssueType::LinearBug
        );
    }

    #[test]
    fn test_issue_type_from_str_case_insensitive() {
        assert_eq!(
            "SENTRY_ISSUE".parse::<IssueType>().unwrap(),
            IssueType::SentryIssue
        );
        assert_eq!(
            "Linear_Bug".parse::<IssueType>().unwrap(),
            IssueType::LinearBug
        );
    }

    #[test]
    fn test_issue_type_from_str_invalid() {
        assert!("invalid".parse::<IssueType>().is_err());
        assert!("github_issue".parse::<IssueType>().is_err());
    }

    #[test]
    fn test_issue_type_serde() {
        let issue_type = IssueType::SentryIssue;
        let json = serde_json::to_string(&issue_type).unwrap();
        assert_eq!(json, "\"sentry_issue\"");
        let parsed: IssueType = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, IssueType::SentryIssue);
    }

    #[test]
    fn test_issue_type_serde_all_variants() {
        let types = vec![IssueType::SentryIssue, IssueType::LinearBug];

        for issue_type in types {
            let json = serde_json::to_string(&issue_type).unwrap();
            let parsed: IssueType = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, issue_type);
        }
    }

    #[test]
    fn test_regression_watch_new() {
        let watch = RegressionWatch::new(IssueType::SentryIssue, "sentry-123", 1);

        assert_eq!(watch.issue_type, IssueType::SentryIssue);
        assert_eq!(watch.issue_id, "sentry-123");
        assert_eq!(watch.fix_attempt_id, 1);
        assert_eq!(watch.status, RegressionWatchStatus::AwaitingRelease);
        assert!(watch.pr_merged_at.is_none());
        assert!(watch.monitoring_started_at.is_none());
        assert!(watch.resolved_at.is_none());
        assert!(watch.regressed_at.is_none());
    }

    #[test]
    fn test_regression_watch_serde() {
        let watch = RegressionWatch::new(IssueType::LinearBug, "linear-456", 2);

        let json = serde_json::to_string(&watch).unwrap();
        let parsed: RegressionWatch = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.issue_type, watch.issue_type);
        assert_eq!(parsed.issue_id, watch.issue_id);
        assert_eq!(parsed.fix_attempt_id, watch.fix_attempt_id);
        assert_eq!(parsed.status, watch.status);
    }

    #[test]
    fn test_regression_watch_clone() {
        let watch = RegressionWatch::new(IssueType::SentryIssue, "sentry-789", 3);

        let cloned = watch.clone();
        assert_eq!(cloned.issue_type, watch.issue_type);
        assert_eq!(cloned.issue_id, watch.issue_id);
        assert_eq!(cloned.fix_attempt_id, watch.fix_attempt_id);
    }

    #[test]
    fn test_regression_watch_debug() {
        let watch = RegressionWatch::new(IssueType::SentryIssue, "sentry-123", 1);
        let debug_str = format!("{:?}", watch);
        assert!(debug_str.contains("RegressionWatch"));
    }

    #[test]
    fn test_release_tracking_new() {
        let tracking = ReleaseTracking::new(1, "v1.2.3", "abc123def");

        assert_eq!(tracking.regression_watch_id, 1);
        assert_eq!(tracking.release_version, "v1.2.3");
        assert_eq!(tracking.release_commit, "abc123def");
        assert!(tracking.released_at.is_some());
    }

    #[test]
    fn test_release_tracking_serde() {
        let tracking = ReleaseTracking::new(2, "v2.0.0", "def456abc");

        let json = serde_json::to_string(&tracking).unwrap();
        let parsed: ReleaseTracking = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.regression_watch_id, tracking.regression_watch_id);
        assert_eq!(parsed.release_version, tracking.release_version);
        assert_eq!(parsed.release_commit, tracking.release_commit);
    }

    #[test]
    fn test_release_tracking_clone() {
        let tracking = ReleaseTracking::new(3, "v3.0.0", "xyz789");

        let cloned = tracking.clone();
        assert_eq!(cloned.regression_watch_id, tracking.regression_watch_id);
        assert_eq!(cloned.release_version, tracking.release_version);
    }

    #[test]
    fn test_release_tracking_debug() {
        let tracking = ReleaseTracking::new(1, "v1.0.0", "commit123");
        let debug_str = format!("{:?}", tracking);
        assert!(debug_str.contains("ReleaseTracking"));
    }

    #[test]
    fn test_regression_check_new() {
        let check = RegressionCheck::new(1, false);

        assert_eq!(check.regression_watch_id, 1);
        assert!(!check.issue_still_exists);
        assert!(check.checked_at.is_some());
        assert!(check.check_details.is_none());
    }

    #[test]
    fn test_regression_check_with_details() {
        let mut check = RegressionCheck::new(2, true);
        check.check_details = Some("Issue reoccurred in production".to_string());

        assert!(check.issue_still_exists);
        assert_eq!(
            check.check_details,
            Some("Issue reoccurred in production".to_string())
        );
    }

    #[test]
    fn test_regression_check_serde() {
        let mut check = RegressionCheck::new(3, false);
        check.check_details = Some("No occurrences found".to_string());

        let json = serde_json::to_string(&check).unwrap();
        let parsed: RegressionCheck = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.regression_watch_id, check.regression_watch_id);
        assert_eq!(parsed.issue_still_exists, check.issue_still_exists);
        assert_eq!(parsed.check_details, check.check_details);
    }

    #[test]
    fn test_regression_check_clone() {
        let check = RegressionCheck::new(4, true);

        let cloned = check.clone();
        assert_eq!(cloned.regression_watch_id, check.regression_watch_id);
        assert_eq!(cloned.issue_still_exists, check.issue_still_exists);
    }

    #[test]
    fn test_regression_check_debug() {
        let check = RegressionCheck::new(1, false);
        let debug_str = format!("{:?}", check);
        assert!(debug_str.contains("RegressionCheck"));
    }

    #[test]
    fn test_regression_watch_status_transitions() {
        // Valid transition: AwaitingRelease -> Monitoring
        let mut watch = RegressionWatch::new(IssueType::SentryIssue, "issue-1", 1);
        assert_eq!(watch.status, RegressionWatchStatus::AwaitingRelease);

        watch.status = RegressionWatchStatus::Monitoring;
        watch.monitoring_started_at = Some(chrono::Utc::now());
        assert_eq!(watch.status, RegressionWatchStatus::Monitoring);

        // Valid transition: Monitoring -> Resolved
        watch.status = RegressionWatchStatus::Resolved;
        watch.resolved_at = Some(chrono::Utc::now());
        assert_eq!(watch.status, RegressionWatchStatus::Resolved);
    }

    #[test]
    fn test_regression_watch_status_regressed_transition() {
        // Valid transition: Monitoring -> Regressed
        let mut watch = RegressionWatch::new(IssueType::LinearBug, "issue-2", 2);
        watch.status = RegressionWatchStatus::Monitoring;
        watch.monitoring_started_at = Some(chrono::Utc::now());

        watch.status = RegressionWatchStatus::Regressed;
        watch.regressed_at = Some(chrono::Utc::now());
        assert_eq!(watch.status, RegressionWatchStatus::Regressed);
    }
    // Tests for validate_issue_id

    #[test]
    fn test_validate_issue_id_valid() {
        assert!(validate_issue_id("PROJ-123").is_ok());
        assert!(validate_issue_id("abc123").is_ok());
        assert!(validate_issue_id("simple_id").is_ok());
        assert!(validate_issue_id("ID-WITH-DASHES").is_ok());
        assert!(validate_issue_id("123456").is_ok());
        assert!(validate_issue_id("a").is_ok());
    }

    #[test]
    fn test_validate_issue_id_empty() {
        let result = validate_issue_id("");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("empty"));
    }

    #[test]
    fn test_validate_issue_id_too_long() {
        let long_id = "x".repeat(MAX_ISSUE_ID_LENGTH + 1);
        let result = validate_issue_id(&long_id);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("maximum length"));
    }

    #[test]
    fn test_validate_issue_id_at_max_length() {
        let max_id = "x".repeat(MAX_ISSUE_ID_LENGTH);
        assert!(validate_issue_id(&max_id).is_ok());
    }

    #[test]
    fn test_validate_issue_id_path_traversal() {
        assert!(validate_issue_id("..").is_err());
        assert!(validate_issue_id("../etc/passwd").is_err());
        assert!(validate_issue_id("foo..bar").is_err());
        assert!(validate_issue_id("a/../b").is_err());
    }

    #[test]
    fn test_validate_issue_id_forward_slash() {
        assert!(validate_issue_id("a/b").is_err());
        assert!(validate_issue_id("/leading").is_err());
        assert!(validate_issue_id("trailing/").is_err());
        assert!(validate_issue_id("path/to/something").is_err());
    }

    #[test]
    fn test_validate_issue_id_backslash() {
        assert!(validate_issue_id("a\\b").is_err());
        assert!(validate_issue_id("\\leading").is_err());
        assert!(validate_issue_id("windows\\path").is_err());
    }

    #[test]
    fn test_validate_issue_id_null_byte() {
        assert!(validate_issue_id("foo\0bar").is_err());
        assert!(validate_issue_id("\0").is_err());
    }

    #[test]
    fn test_validate_issue_id_unicode() {
        // Unicode should be allowed (for international issue IDs)
        assert!(validate_issue_id("项目-123").is_ok());
        assert!(validate_issue_id("задача-456").is_ok());
    }

    #[test]
    fn test_validate_issue_id_special_chars() {
        // These special chars should be allowed
        assert!(validate_issue_id("ID_with_underscore").is_ok());
        assert!(validate_issue_id("ID-with-dash").is_ok());
        assert!(validate_issue_id("ID.with.dot").is_ok());
        assert!(validate_issue_id("ID:with:colon").is_ok());
    }

    #[test]
    fn test_change_category_display() {
        assert_eq!(ChangeCategory::Tests.to_string(), "tests");
        assert_eq!(ChangeCategory::Docs.to_string(), "docs");
        assert_eq!(ChangeCategory::Config.to_string(), "config");
        assert_eq!(ChangeCategory::Dependencies.to_string(), "dependencies");
        assert_eq!(ChangeCategory::Migrations.to_string(), "migrations");
        assert_eq!(ChangeCategory::NewCode.to_string(), "new_code");
        assert_eq!(ChangeCategory::Modification.to_string(), "modification");
    }

    #[test]
    fn test_review_category_display() {
        assert_eq!(ReviewCategory::MissingTests.to_string(), "missing_tests");
        assert_eq!(ReviewCategory::StyleIssue.to_string(), "style_issue");
        assert_eq!(ReviewCategory::WrongApproach.to_string(), "wrong_approach");
        assert_eq!(ReviewCategory::Incomplete.to_string(), "incomplete");
        assert_eq!(ReviewCategory::Security.to_string(), "security");
        assert_eq!(ReviewCategory::Performance.to_string(), "performance");
        assert_eq!(ReviewCategory::Documentation.to_string(), "documentation");
        assert_eq!(ReviewCategory::Other.to_string(), "other");
    }

    #[test]
    fn test_review_category_parse_roundtrip() {
        let categories = vec![
            ReviewCategory::MissingTests,
            ReviewCategory::StyleIssue,
            ReviewCategory::WrongApproach,
            ReviewCategory::Incomplete,
            ReviewCategory::Security,
            ReviewCategory::Performance,
            ReviewCategory::Documentation,
            ReviewCategory::Other,
        ];
        for cat in categories {
            let display = cat.to_string();
            let parsed = ReviewCategory::parse(&display);
            assert_eq!(cat, parsed, "Round-trip failed for {:?}", cat);
        }
    }

    #[test]
    fn test_review_category_parse_unknown() {
        assert_eq!(
            ReviewCategory::parse("unknown_category"),
            ReviewCategory::Other
        );
        assert_eq!(ReviewCategory::parse(""), ReviewCategory::Other);
    }

    #[test]
    fn test_change_category_serde_roundtrip() {
        let cat = ChangeCategory::Tests;
        let json = serde_json::to_string(&cat).unwrap();
        let parsed: ChangeCategory = serde_json::from_str(&json).unwrap();
        assert_eq!(cat, parsed);
    }

    #[test]
    fn test_review_category_serde_roundtrip() {
        let cat = ReviewCategory::Security;
        let json = serde_json::to_string(&cat).unwrap();
        let parsed: ReviewCategory = serde_json::from_str(&json).unwrap();
        assert_eq!(cat, parsed);
    }

    #[test]
    fn test_diff_analysis_serde() {
        let analysis = DiffAnalysis {
            id: 1,
            attempt_id: 42,
            pr_url: "https://github.com/org/repo/pull/1".to_string(),
            scm_repo: "org/repo".to_string(),
            pr_number: 1,
            files_changed: vec!["src/main.rs".to_string()],
            file_types: {
                let mut m = std::collections::HashMap::new();
                m.insert("rs".to_string(), 1);
                m
            },
            change_categories: vec![ChangeCategory::Modification],
            diff_summary: "1 file changed".to_string(),
            created_at: chrono::Utc::now(),
        };
        let json = serde_json::to_string(&analysis).unwrap();
        let parsed: DiffAnalysis = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.attempt_id, 42);
        assert_eq!(parsed.files_changed.len(), 1);
        assert_eq!(parsed.change_categories[0], ChangeCategory::Modification);
    }

    #[test]
    fn test_fix_quality_score_serde() {
        let score = FixQualityScore {
            score: 0.87,
            merge_speed_component: 0.9,
            review_cycles_component: 0.8,
            approval_component: 1.0,
        };
        let json = serde_json::to_string(&score).unwrap();
        let parsed: FixQualityScore = serde_json::from_str(&json).unwrap();
        assert!((parsed.score - 0.87).abs() < f64::EPSILON);
    }

    #[test]
    fn test_strategy_fingerprint_serde() {
        let fp = StrategyFingerprint {
            id: 1,
            attempt_id: 10,
            files_explored: vec!["src/a.rs".to_string()],
            tests_run: 3,
            tools_used: {
                let mut m = std::collections::HashMap::new();
                m.insert("Read".to_string(), 5);
                m
            },
            fix_approach: "tdd".to_string(),
            strategy_summary: "test".to_string(),
            fix_quality_score: Some(0.9),
            created_at: chrono::Utc::now(),
        };
        let json = serde_json::to_string(&fp).unwrap();
        let parsed: StrategyFingerprint = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.fix_approach, "tdd");
        assert_eq!(*parsed.tools_used.get("Read").unwrap(), 5);
    }

    #[test]
    fn test_issue_cluster_serde() {
        let now = chrono::Utc::now();
        let cluster = IssueCluster {
            id: 1,
            cluster_key: "cluster_abc".to_string(),
            source: "sentry".to_string(),
            issue_ids: vec!["a".to_string(), "b".to_string()],
            window_start: now,
            window_end: now + chrono::Duration::minutes(15),
            resolved_by_issue_id: Some("a".to_string()),
            resolved_by_attempt_id: Some(42),
            status: "resolved".to_string(),
            created_at: now,
        };
        let json = serde_json::to_string(&cluster).unwrap();
        let parsed: IssueCluster = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.cluster_key, "cluster_abc");
        assert_eq!(parsed.issue_ids.len(), 2);
        assert_eq!(parsed.resolved_by_issue_id, Some("a".to_string()));
    }

    #[test]
    fn test_extracted_learnings_default() {
        let learnings = ExtractedLearnings {
            root_cause: None,
            files_modified: vec![],
            strategy_used: None,
            tests_added: false,
            key_decisions: vec![],
        };
        let json = serde_json::to_string(&learnings).unwrap();
        let parsed: ExtractedLearnings = serde_json::from_str(&json).unwrap();
        assert!(parsed.root_cause.is_none());
        assert!(parsed.files_modified.is_empty());
    }

    #[test]
    fn test_promoted_instruction_serde() {
        let instruction = PromotedInstruction {
            id: 1,
            repo: "org/repo".to_string(),
            source_type: "qa_promotion".to_string(),
            instruction_text: "Always add tests".to_string(),
            occurrence_count: 5,
            confidence: 0.85,
            is_active: true,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };
        let json = serde_json::to_string(&instruction).unwrap();
        let parsed: PromotedInstruction = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.instruction_text, "Always add tests");
        assert!(parsed.is_active);
    }

    #[test]
    fn test_validate_issue_id_exactly_max_length() {
        let id = "a".repeat(MAX_ISSUE_ID_LENGTH);
        assert!(validate_issue_id(&id).is_ok());
    }

    #[test]
    fn test_validate_issue_id_one_over_max() {
        let id = "a".repeat(MAX_ISSUE_ID_LENGTH + 1);
        assert!(validate_issue_id(&id).is_err());
    }

    #[test]
    fn test_validate_issue_id_embedded_null_byte() {
        assert!(validate_issue_id("abc\0def").is_err());
    }

    #[test]
    fn test_validate_issue_id_only_dots() {
        assert!(validate_issue_id("..").is_err());
    }

    #[test]
    fn test_validate_issue_id_contains_backslash() {
        assert!(validate_issue_id("a\\b").is_err());
    }

    #[test]
    fn test_validate_issue_id_single_char() {
        assert!(validate_issue_id("x").is_ok());
    }

    #[test]
    fn test_validate_issue_id_unicode_chars() {
        assert!(validate_issue_id("日本語").is_ok());
    }

    #[test]
    fn test_validate_issue_id_whitespace() {
        assert!(validate_issue_id("abc def").is_ok());
    }

    #[test]
    fn test_validate_issue_id_double_dot_in_middle() {
        assert!(validate_issue_id("foo..bar").is_err());
    }

    #[test]
    fn test_fix_attempt_is_bug_sentry_source() {
        let attempt = FixAttempt {
            id: 1,
            issue_id: "123".into(),
            short_id: "S-123".into(),
            source: "sentry".into(),
            attempted_at: Utc::now(),
            pr_url: None,
            scm_repo: None,
            scm_pr_number: None,
            status: FixAttemptStatus::Pending,
            error_message: None,
            merged_at: None,
            resolved_at: None,
            retry_count: 0,
            last_retry_at: None,
            issue_labels: vec![],
            parent_attempt_id: None,
            cascade_repo: None,
        };
        assert!(attempt.is_bug());
    }

    #[test]
    fn test_fix_attempt_is_bug_with_bug_label() {
        let attempt = FixAttempt {
            id: 1,
            issue_id: "123".into(),
            short_id: "P-123".into(),
            source: "linear".into(),
            attempted_at: Utc::now(),
            pr_url: None,
            scm_repo: None,
            scm_pr_number: None,
            status: FixAttemptStatus::Pending,
            error_message: None,
            merged_at: None,
            resolved_at: None,
            retry_count: 0,
            last_retry_at: None,
            issue_labels: vec!["bug".into()],
            parent_attempt_id: None,
            cascade_repo: None,
        };
        assert!(attempt.is_bug());
    }

    #[test]
    fn test_fix_attempt_is_bug_case_insensitive() {
        let attempt = FixAttempt {
            id: 1,
            issue_id: "1".into(),
            short_id: "T-1".into(),
            source: "linear".into(),
            attempted_at: Utc::now(),
            pr_url: None,
            scm_repo: None,
            scm_pr_number: None,
            status: FixAttemptStatus::Pending,
            error_message: None,
            merged_at: None,
            resolved_at: None,
            retry_count: 0,
            last_retry_at: None,
            issue_labels: vec!["BUG".into()],
            parent_attempt_id: None,
            cascade_repo: None,
        };
        assert!(attempt.is_bug());
    }

    #[test]
    fn test_fix_attempt_is_bug_label_substring() {
        let attempt = FixAttempt {
            id: 1,
            issue_id: "1".into(),
            short_id: "T-1".into(),
            source: "linear".into(),
            attempted_at: Utc::now(),
            pr_url: None,
            scm_repo: None,
            scm_pr_number: None,
            status: FixAttemptStatus::Pending,
            error_message: None,
            merged_at: None,
            resolved_at: None,
            retry_count: 0,
            last_retry_at: None,
            issue_labels: vec!["hotfix-urgent".into()],
            parent_attempt_id: None,
            cascade_repo: None,
        };
        assert!(attempt.is_bug());
    }

    #[test]
    fn test_fix_attempt_is_not_bug_feature_label() {
        let attempt = FixAttempt {
            id: 1,
            issue_id: "1".into(),
            short_id: "T-1".into(),
            source: "linear".into(),
            attempted_at: Utc::now(),
            pr_url: None,
            scm_repo: None,
            scm_pr_number: None,
            status: FixAttemptStatus::Pending,
            error_message: None,
            merged_at: None,
            resolved_at: None,
            retry_count: 0,
            last_retry_at: None,
            issue_labels: vec!["feature".into(), "enhancement".into()],
            parent_attempt_id: None,
            cascade_repo: None,
        };
        assert!(!attempt.is_bug());
    }

    #[test]
    fn test_fix_attempt_is_bug_empty_labels() {
        let attempt = FixAttempt {
            id: 1,
            issue_id: "1".into(),
            short_id: "T-1".into(),
            source: "linear".into(),
            attempted_at: Utc::now(),
            pr_url: None,
            scm_repo: None,
            scm_pr_number: None,
            status: FixAttemptStatus::Pending,
            error_message: None,
            merged_at: None,
            resolved_at: None,
            retry_count: 0,
            last_retry_at: None,
            issue_labels: vec![],
            parent_attempt_id: None,
            cascade_repo: None,
        };
        assert!(!attempt.is_bug());
    }

    #[test]
    fn test_fix_attempt_is_bug_all_bug_labels() {
        for label in [
            "bug",
            "defect",
            "error",
            "fix",
            "hotfix",
            "regression",
            "issue",
            "problem",
            "incident",
            "crash",
            "broken",
        ] {
            let attempt = FixAttempt {
                id: 1,
                issue_id: "1".into(),
                short_id: "T-1".into(),
                source: "linear".into(),
                attempted_at: Utc::now(),
                pr_url: None,
                scm_repo: None,
                scm_pr_number: None,
                status: FixAttemptStatus::Pending,
                error_message: None,
                merged_at: None,
                resolved_at: None,
                retry_count: 0,
                last_retry_at: None,
                issue_labels: vec![label.into()],
                parent_attempt_id: None,
                cascade_repo: None,
            };
            assert!(
                attempt.is_bug(),
                "Label '{}' should be detected as bug",
                label
            );
        }
    }

    #[test]
    fn test_claude_execution_complete() {
        let mut exec = AgentExecution::new();
        assert!(exec.completed_at.is_none());
        assert!(exec.duration_secs.is_none());
        exec.complete(Some(0), false);
        assert!(exec.completed_at.is_some());
        assert!(exec.duration_secs.is_some());
        assert_eq!(exec.exit_code, Some(0));
        assert!(!exec.timed_out);
    }

    #[test]
    fn test_claude_execution_complete_timeout() {
        let mut exec = AgentExecution::new();
        exec.complete(None, true);
        assert!(exec.timed_out);
        assert!(exec.exit_code.is_none());
        assert!(exec.completed_at.is_some());
    }

    #[test]
    fn test_claude_execution_complete_nonzero_exit() {
        let mut exec = AgentExecution::new();
        exec.complete(Some(1), false);
        assert_eq!(exec.exit_code, Some(1));
        assert!(!exec.timed_out);
    }

    #[test]
    fn test_claude_execution_with_attempt_id() {
        let exec = AgentExecution::new().with_attempt_id(42);
        assert_eq!(exec.attempt_id, Some(42));
    }

    #[test]
    fn test_claude_execution_default() {
        let exec = AgentExecution::default();
        assert_eq!(exec.id, 0);
        assert!(exec.attempt_id.is_none());
        assert!(!exec.timed_out);
    }

    #[test]
    fn test_claude_execution_duration_non_negative() {
        let mut exec = AgentExecution::new();
        exec.complete(Some(0), false);
        assert!(exec.duration_secs.unwrap() >= 0.0);
    }

    #[test]
    fn test_fix_attempt_status_display_roundtrip() {
        for status in [
            FixAttemptStatus::Pending,
            FixAttemptStatus::Success,
            FixAttemptStatus::Failed,
            FixAttemptStatus::Merged,
            FixAttemptStatus::Closed,
            FixAttemptStatus::CannotFix,
        ] {
            let parsed: FixAttemptStatus = status.to_string().parse().unwrap();
            assert_eq!(status, parsed);
        }
    }

    #[test]
    fn test_fix_attempt_status_serde_roundtrip() {
        for status in [
            FixAttemptStatus::Pending,
            FixAttemptStatus::Success,
            FixAttemptStatus::Failed,
            FixAttemptStatus::Merged,
            FixAttemptStatus::Closed,
            FixAttemptStatus::CannotFix,
        ] {
            let json = serde_json::to_string(&status).unwrap();
            let parsed: FixAttemptStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(status, parsed);
        }
    }

    #[test]
    fn test_regression_watch_status_display_roundtrip() {
        for status in [
            RegressionWatchStatus::AwaitingRelease,
            RegressionWatchStatus::Monitoring,
            RegressionWatchStatus::Resolved,
            RegressionWatchStatus::Regressed,
        ] {
            let parsed: RegressionWatchStatus = status.to_string().parse().unwrap();
            assert_eq!(status, parsed);
        }
    }

    #[test]
    fn test_regression_watch_status_parse_invalid() {
        assert!("unknown".parse::<RegressionWatchStatus>().is_err());
        assert!("".parse::<RegressionWatchStatus>().is_err());
    }

    #[test]
    fn test_regression_watch_status_default_value() {
        assert_eq!(
            RegressionWatchStatus::default(),
            RegressionWatchStatus::AwaitingRelease
        );
    }

    #[test]
    fn test_issue_type_source_name() {
        assert_eq!(IssueType::SentryIssue.source_name(), "sentry");
        assert_eq!(IssueType::LinearBug.source_name(), "linear");
    }

    #[test]
    fn test_issue_type_display_parse_roundtrip() {
        for issue_type in [IssueType::SentryIssue, IssueType::LinearBug] {
            let parsed: IssueType = issue_type.to_string().parse().unwrap();
            assert_eq!(issue_type, parsed);
        }
    }

    #[test]
    fn test_issue_type_parse_invalid() {
        assert!("unknown".parse::<IssueType>().is_err());
        assert!("".parse::<IssueType>().is_err());
    }

    #[test]
    fn test_change_category_display_all() {
        assert_eq!(ChangeCategory::Tests.to_string(), "tests");
        assert_eq!(ChangeCategory::Docs.to_string(), "docs");
        assert_eq!(ChangeCategory::Config.to_string(), "config");
        assert_eq!(ChangeCategory::Dependencies.to_string(), "dependencies");
        assert_eq!(ChangeCategory::Migrations.to_string(), "migrations");
        assert_eq!(ChangeCategory::NewCode.to_string(), "new_code");
        assert_eq!(ChangeCategory::Modification.to_string(), "modification");
    }

    #[test]
    fn test_change_category_serde_all_variants() {
        for cat in [
            ChangeCategory::Tests,
            ChangeCategory::Docs,
            ChangeCategory::Config,
            ChangeCategory::Dependencies,
            ChangeCategory::Migrations,
            ChangeCategory::NewCode,
            ChangeCategory::Modification,
        ] {
            let json = serde_json::to_string(&cat).unwrap();
            let parsed: ChangeCategory = serde_json::from_str(&json).unwrap();
            assert_eq!(cat, parsed);
        }
    }

    #[test]
    fn test_review_category_display_all() {
        assert_eq!(ReviewCategory::MissingTests.to_string(), "missing_tests");
        assert_eq!(ReviewCategory::StyleIssue.to_string(), "style_issue");
        assert_eq!(ReviewCategory::WrongApproach.to_string(), "wrong_approach");
        assert_eq!(ReviewCategory::Incomplete.to_string(), "incomplete");
        assert_eq!(ReviewCategory::Security.to_string(), "security");
        assert_eq!(ReviewCategory::Performance.to_string(), "performance");
        assert_eq!(ReviewCategory::Documentation.to_string(), "documentation");
        assert_eq!(ReviewCategory::Other.to_string(), "other");
    }

    #[test]
    fn test_review_category_parse_all_variants() {
        for cat in [
            ReviewCategory::MissingTests,
            ReviewCategory::StyleIssue,
            ReviewCategory::WrongApproach,
            ReviewCategory::Incomplete,
            ReviewCategory::Security,
            ReviewCategory::Performance,
            ReviewCategory::Documentation,
            ReviewCategory::Other,
        ] {
            let parsed = ReviewCategory::parse(&cat.to_string());
            assert_eq!(cat, parsed);
        }
    }

    #[test]
    fn test_review_category_parse_unknown_returns_other() {
        assert_eq!(ReviewCategory::parse("unknown"), ReviewCategory::Other);
        assert_eq!(ReviewCategory::parse(""), ReviewCategory::Other);
        assert_eq!(ReviewCategory::parse("foobar"), ReviewCategory::Other);
    }

    #[test]
    fn test_activity_log_entry_builder_chain() {
        let entry = ActivityLogEntry::new("test_type", "test message")
            .with_source("linear".to_string())
            .with_issue("issue-1".to_string(), "PROJ-1".to_string())
            .with_metadata(serde_json::json!({"key": "value"}));
        assert_eq!(entry.activity_type, "test_type");
        assert_eq!(entry.source, Some("linear".to_string()));
        assert_eq!(entry.issue_id, Some("issue-1".to_string()));
        assert!(entry.metadata.is_some());
    }

    #[test]
    fn test_activity_log_entry_minimal() {
        let entry = ActivityLogEntry::new("type", "msg");
        assert_eq!(entry.id, 0);
        assert!(entry.source.is_none());
        assert!(entry.issue_id.is_none());
        assert!(entry.metadata.is_none());
    }

    #[test]
    fn test_pr_record_new() {
        let pr = PrRecord::new("https://github.com/org/repo/pull/1", "org/repo", 1);
        assert_eq!(pr.status, "open");
        assert_eq!(pr.approvals_count, 0);
        assert_eq!(pr.review_cycles, 0);
        assert!(pr.merged_at.is_none());
    }

    #[test]
    fn test_pr_record_for_issue() {
        let pr = PrRecord::for_issue("url", "org/repo", 1, "linear", "issue-1");
        assert_eq!(pr.issue_source, Some("linear".to_string()));
        assert_eq!(pr.issue_id, Some("issue-1".to_string()));
    }

    #[test]
    fn test_error_pattern_new() {
        let pattern = ErrorPattern::new("abc123");
        assert_eq!(pattern.pattern_hash, "abc123");
        assert_eq!(pattern.occurrence_count, 1);
        assert!(pattern.error_type.is_none());
    }

    #[test]
    fn test_processing_metric_builder() {
        let metric = ProcessingMetric::new("queue_depth", 42.0)
            .with_source("linear")
            .with_tags(serde_json::json!({"env": "prod"}));
        assert_eq!(metric.metric_name, "queue_depth");
        assert_eq!(metric.metric_value, 42.0);
        assert!(metric.tags.is_some());
    }

    #[test]
    fn test_processing_metric_special_values() {
        let nan = ProcessingMetric::new("test", f64::NAN);
        assert!(nan.metric_value.is_nan());
        let inf = ProcessingMetric::new("test", f64::INFINITY);
        assert!(inf.metric_value.is_infinite());
    }

    #[test]
    fn test_similar_issue_new() {
        let si = SimilarIssue::new("a", "b", 0.95);
        assert_eq!(si.source_issue_id, "a");
        assert_eq!(si.similar_issue_id, "b");
        assert_eq!(si.id, 0);
    }

    #[test]
    fn test_issue_embedding_empty_vector() {
        let e = IssueEmbedding::new("s", "i", vec![]);
        assert!(e.embedding.as_ref().unwrap().is_empty());
    }

    #[test]
    fn test_prompt_experiment_new() {
        let exp = PromptExperiment::new("exp", "control", "tpl", "hash");
        assert!(exp.active);
        assert_eq!(exp.success_count, 0);
        assert_eq!(exp.failure_count, 0);
    }

    #[test]
    fn test_regression_watch_new_defaults() {
        let w = RegressionWatch::new(IssueType::SentryIssue, "i-1", 42);
        assert_eq!(w.status, RegressionWatchStatus::AwaitingRelease);
        assert!(w.pr_merged_at.is_none());
    }

    #[test]
    fn test_release_tracking_new_fields() {
        let rt = ReleaseTracking::new(1, "v1.0", "abc");
        assert!(rt.released_at.is_some());
    }

    #[test]
    fn test_regression_check_new_no_regression() {
        let c = RegressionCheck::new(1, false);
        assert!(!c.issue_still_exists);
        assert!(c.checked_at.is_some());
    }

    #[test]
    fn test_regression_check_regression_detected() {
        let c = RegressionCheck::new(1, true);
        assert!(c.issue_still_exists);
    }

    #[test]
    fn test_blocking_question_serde_full() {
        let q = BlockingQuestion {
            question: "API?".into(),
            context: Some("ctx".into()),
            options: vec!["REST".into(), "GraphQL".into()],
            why: Some("reason".into()),
        };
        let json = serde_json::to_string(&q).unwrap();
        let parsed: BlockingQuestion = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.options.len(), 2);
    }

    #[test]
    fn test_blocking_question_serde_minimal() {
        let q = BlockingQuestion {
            question: "Yes?".into(),
            context: None,
            options: vec![],
            why: None,
        };
        let json = serde_json::to_string(&q).unwrap();
        assert!(!json.contains("context"));
        assert!(!json.contains("why"));
    }

    #[test]
    fn test_match_result_not_matched_uses_normal_priority() {
        let r = MatchResult::not_matched("no match");
        assert!(!r.matches);
        assert_eq!(r.priority, MatchPriority::Normal);
    }

    #[test]
    fn test_issue_metadata_overwrite() {
        let mut issue = Issue::new("1", "T-1", "Title", "url", "src");
        issue.set_metadata("key", "first");
        issue.set_metadata("key", "second");
        assert_eq!(
            issue.get_metadata::<String>("key"),
            Some("second".to_string())
        );
    }

    #[test]
    fn test_pr_analytics_default() {
        let a = PrAnalytics::default();
        assert_eq!(a.total, 0);
        assert!(a.merge_rate.is_none());
        assert!(a.by_repo.is_empty());
    }

    #[test]
    fn test_analytics_summary_default() {
        let s = AnalyticsSummary::default();
        assert_eq!(s.success_rate, 0.0);
        assert!(s.most_common_error.is_none());
    }

    #[test]
    fn test_fix_quality_score_serde_roundtrip() {
        let score = FixQualityScore {
            score: 0.85,
            merge_speed_component: 0.9,
            review_cycles_component: 0.8,
            approval_component: 1.0,
        };
        let json = serde_json::to_string(&score).unwrap();
        let parsed: FixQualityScore = serde_json::from_str(&json).unwrap();
        assert!((parsed.score - 0.85).abs() < 0.001);
    }

    #[test]
    fn test_issue_cluster_serde_roundtrip() {
        let c = IssueCluster {
            id: 1,
            cluster_key: "k".into(),
            source: "sentry".into(),
            issue_ids: vec!["a".into(), "b".into()],
            window_start: Utc::now(),
            window_end: Utc::now(),
            resolved_by_issue_id: None,
            resolved_by_attempt_id: None,
            status: "active".into(),
            created_at: Utc::now(),
        };
        let json = serde_json::to_string(&c).unwrap();
        let parsed: IssueCluster = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.issue_ids.len(), 2);
    }

    #[test]
    fn test_strategy_fingerprint_serde_roundtrip() {
        let mut tools = HashMap::new();
        tools.insert("Read".into(), 5i64);
        let fp = StrategyFingerprint {
            id: 1,
            attempt_id: 42,
            files_explored: vec!["src/main.rs".into()],
            tests_run: 2,
            tools_used: tools,
            fix_approach: "tdd".into(),
            strategy_summary: "summary".into(),
            fix_quality_score: Some(0.9),
            created_at: Utc::now(),
        };
        let json = serde_json::to_string(&fp).unwrap();
        let parsed: StrategyFingerprint = serde_json::from_str(&json).unwrap();
        assert_eq!(*parsed.tools_used.get("Read").unwrap(), 5);
    }

    #[test]
    fn test_repo_knowledge_serde() {
        let k = RepoKnowledge {
            id: 1,
            repo: "org/repo".into(),
            knowledge_key: "test_pattern".into(),
            knowledge_value: "cargo test".into(),
            source_type: "diff".into(),
            confidence: 0.9,
            occurrence_count: 5,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        let json = serde_json::to_string(&k).unwrap();
        let parsed: RepoKnowledge = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.knowledge_key, "test_pattern");
    }

    #[test]
    fn test_review_pattern_serde() {
        let p = ReviewPattern {
            id: 1,
            scm_repo: "org/repo".into(),
            category: ReviewCategory::MissingTests,
            pattern_text: "Add tests".into(),
            example_comments: vec!["please add tests".into()],
            occurrence_count: 3,
            promoted_to_instruction: false,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        let json = serde_json::to_string(&p).unwrap();
        let parsed: ReviewPattern = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.category, ReviewCategory::MissingTests);
    }

    #[test]
    fn test_qa_match_serde() {
        let entry = QaKnowledgeEntry {
            id: 1,
            source: "l".into(),
            repo: None,
            issue_id: "i".into(),
            short_id: "I".into(),
            question_text: "Q".into(),
            question_norm: "q".into(),
            question_embedding: None,
            answer_text: "A".into(),
            answer_norm: "a".into(),
            answer_embedding: None,
            channel: "ch".into(),
            responder: None,
            correlation_id: "c".into(),
            asked_at: Utc::now(),
            answered_at: Utc::now(),
            success_count: 1,
            failure_count: 0,
            last_used_at: None,
            metadata: None,
        };
        let m = QaMatch {
            entry,
            semantic_similarity: 0.95,
            historical_success_rate: 1.0,
            final_score: 0.97,
        };
        let json = serde_json::to_string(&m).unwrap();
        let parsed: QaMatch = serde_json::from_str(&json).unwrap();
        assert!((parsed.final_score - 0.97).abs() < 0.001);
    }

    #[test]
    fn test_blast_radius_display_all_variants() {
        assert_eq!(BlastRadius::Cosmetic.to_string(), "cosmetic");
        assert_eq!(BlastRadius::Test.to_string(), "test");
        assert_eq!(BlastRadius::Peripheral.to_string(), "peripheral");
        assert_eq!(BlastRadius::Core.to_string(), "core");
        assert_eq!(BlastRadius::Infrastructure.to_string(), "infrastructure");
        assert_eq!(BlastRadius::Critical.to_string(), "critical");
    }

    #[test]
    fn test_blast_radius_default() {
        assert_eq!(BlastRadius::default(), BlastRadius::Core);
    }

    #[test]
    fn test_blast_radius_ordering() {
        assert!(BlastRadius::Critical > BlastRadius::Infrastructure);
        assert!(BlastRadius::Infrastructure > BlastRadius::Core);
        assert!(BlastRadius::Core > BlastRadius::Peripheral);
        assert!(BlastRadius::Peripheral > BlastRadius::Test);
        assert!(BlastRadius::Test > BlastRadius::Cosmetic);
    }

    #[test]
    fn test_blast_radius_serde_all_variants() {
        for br in [
            BlastRadius::Cosmetic,
            BlastRadius::Test,
            BlastRadius::Peripheral,
            BlastRadius::Core,
            BlastRadius::Infrastructure,
            BlastRadius::Critical,
        ] {
            let json = serde_json::to_string(&br).unwrap();
            let parsed: BlastRadius = serde_json::from_str(&json).unwrap();
            assert_eq!(br, parsed);
            // Verify snake_case serialization matches display
            assert_eq!(json, format!("\"{}\"", br));
        }
    }

    #[test]
    fn test_blast_radius_equality_and_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(BlastRadius::Core);
        set.insert(BlastRadius::Core);
        assert_eq!(set.len(), 1);
        set.insert(BlastRadius::Critical);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_severity_score_default() {
        let s = SeverityScore::default();
        assert_eq!(s.score, 0.0);
        assert_eq!(s.severity_component, 0.0);
        assert_eq!(s.frequency_component, 0.0);
        assert_eq!(s.regression_component, 0.0);
        assert_eq!(s.blast_radius_component, 0.0);
        assert_eq!(s.cluster_boost, 0.0);
    }

    #[test]
    fn test_severity_score_serde_roundtrip() {
        let s = SeverityScore {
            score: 0.95,
            severity_component: 0.8,
            frequency_component: 0.7,
            regression_component: 0.6,
            blast_radius_component: 0.5,
            cluster_boost: 1.0,
        };
        let json = serde_json::to_string(&s).unwrap();
        let parsed: SeverityScore = serde_json::from_str(&json).unwrap();
        assert!((parsed.score - 0.95).abs() < f64::EPSILON);
        assert!((parsed.cluster_boost - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_suppression_field_serde_simple_variants() {
        for field in [
            SuppressionField::Title,
            SuppressionField::Description,
            SuppressionField::Source,
            SuppressionField::Culprit,
            SuppressionField::Filename,
            SuppressionField::ErrorType,
            SuppressionField::Project,
            SuppressionField::Labels,
        ] {
            let json = serde_json::to_string(&field).unwrap();
            let parsed: SuppressionField = serde_json::from_str(&json).unwrap();
            assert_eq!(field, parsed);
        }
    }

    #[test]
    fn test_suppression_field_metadata_variant_serde() {
        let field = SuppressionField::Metadata("custom_key".to_string());
        let json = serde_json::to_string(&field).unwrap();
        let parsed: SuppressionField = serde_json::from_str(&json).unwrap();
        assert_eq!(field, parsed);
        if let SuppressionField::Metadata(key) = parsed {
            assert_eq!(key, "custom_key");
        } else {
            panic!("Expected Metadata variant");
        }
    }

    #[test]
    fn test_suppression_match_mode_default() {
        assert_eq!(
            SuppressionMatchMode::default(),
            SuppressionMatchMode::Contains
        );
    }

    #[test]
    fn test_suppression_match_mode_serde_all_variants() {
        for mode in [
            SuppressionMatchMode::Contains,
            SuppressionMatchMode::Exact,
            SuppressionMatchMode::Regex,
        ] {
            let json = serde_json::to_string(&mode).unwrap();
            let parsed: SuppressionMatchMode = serde_json::from_str(&json).unwrap();
            assert_eq!(mode, parsed);
        }
    }

    #[test]
    fn test_suppression_rule_serde_roundtrip() {
        let rule = SuppressionRule {
            name: "ignore-flaky".to_string(),
            field: SuppressionField::Title,
            pattern: "flaky test".to_string(),
            match_mode: SuppressionMatchMode::Contains,
            sources: vec!["sentry".to_string()],
            reason: "known flaky".to_string(),
        };
        let json = serde_json::to_string(&rule).unwrap();
        let parsed: SuppressionRule = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.name, "ignore-flaky");
        assert_eq!(parsed.field, SuppressionField::Title);
        assert_eq!(parsed.match_mode, SuppressionMatchMode::Contains);
        assert_eq!(parsed.sources.len(), 1);
    }

    #[test]
    fn test_suppression_rule_empty_sources() {
        let rule = SuppressionRule {
            name: "all-sources".to_string(),
            field: SuppressionField::ErrorType,
            pattern: "OutOfMemory".to_string(),
            match_mode: SuppressionMatchMode::Exact,
            sources: vec![],
            reason: "OOM not fixable".to_string(),
        };
        let json = serde_json::to_string(&rule).unwrap();
        let parsed: SuppressionRule = serde_json::from_str(&json).unwrap();
        assert!(parsed.sources.is_empty());
    }

    #[test]
    fn test_suppression_result_suppressed() {
        let r = SuppressionResult {
            suppressed: true,
            matched_rule: Some("rule-1".to_string()),
            reason: Some("known flaky".to_string()),
        };
        let json = serde_json::to_string(&r).unwrap();
        let parsed: SuppressionResult = serde_json::from_str(&json).unwrap();
        assert!(parsed.suppressed);
        assert_eq!(parsed.matched_rule, Some("rule-1".to_string()));
    }

    #[test]
    fn test_suppression_result_not_suppressed() {
        let r = SuppressionResult {
            suppressed: false,
            matched_rule: None,
            reason: None,
        };
        let json = serde_json::to_string(&r).unwrap();
        let parsed: SuppressionResult = serde_json::from_str(&json).unwrap();
        assert!(!parsed.suppressed);
        assert!(parsed.matched_rule.is_none());
    }

    #[test]
    fn test_content_cluster_serde_roundtrip() {
        let c = ContentCluster {
            id: 1,
            cluster_key: "TypeError::main".to_string(),
            source: "sentry".to_string(),
            representative_issue_id: "issue-1".to_string(),
            issue_ids: vec!["issue-1".into(), "issue-2".into()],
            error_type: Some("TypeError".to_string()),
            culprit: Some("app.main".to_string()),
            avg_similarity: 0.92,
            status: "active".to_string(),
            created_at: Utc::now(),
        };
        let json = serde_json::to_string(&c).unwrap();
        let parsed: ContentCluster = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.cluster_key, "TypeError::main");
        assert_eq!(parsed.issue_ids.len(), 2);
        assert!((parsed.avg_similarity - 0.92).abs() < f64::EPSILON);
    }

    #[test]
    fn test_prioritised_issue_serde_roundtrip() {
        let pi = PrioritisedIssue {
            issue: Issue::new("id", "short", "title", "url", "source"),
            match_result: MatchResult::matched("reason", MatchPriority::High),
            severity_score: SeverityScore::default(),
            blast_radius: BlastRadius::Core,
            cluster_key: Some("cluster-1".to_string()),
        };
        let json = serde_json::to_string(&pi).unwrap();
        let parsed: PrioritisedIssue = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.issue.id, "id");
        assert!(parsed.match_result.matches);
        assert_eq!(parsed.blast_radius, BlastRadius::Core);
        assert_eq!(parsed.cluster_key, Some("cluster-1".to_string()));
    }

    #[test]
    fn test_prioritised_issue_no_cluster_key() {
        let pi = PrioritisedIssue {
            issue: Issue::new("id", "short", "title", "url", "source"),
            match_result: MatchResult::not_matched("no match"),
            severity_score: SeverityScore::default(),
            blast_radius: BlastRadius::Cosmetic,
            cluster_key: None,
        };
        let json = serde_json::to_string(&pi).unwrap();
        assert!(!json.contains("cluster_key"));
    }

    #[test]
    fn test_issue_embedding_from_issue_basic() {
        let issue = Issue::new("id-1", "SHORT-1", "A title", "https://url", "linear");
        let emb = IssueEmbedding::from_issue(&issue);
        assert_eq!(emb.source, "linear");
        assert_eq!(emb.issue_id, "id-1");
        assert_eq!(emb.short_id, Some("SHORT-1".to_string()));
        assert_eq!(emb.title, Some("A title".to_string()));
        assert_eq!(emb.url, Some("https://url".to_string()));
        assert_eq!(emb.priority, Some("none".to_string()));
        assert_eq!(emb.status, Some("open".to_string()));
        assert!(emb.embedding.is_none());
        assert!(emb.labels.is_none());
    }

    #[test]
    fn test_issue_embedding_from_issue_with_labels() {
        let mut issue = Issue::new("id-2", "SHORT-2", "Title", "url", "sentry");
        issue.set_metadata("labels", vec!["bug", "urgent"]);
        let emb = IssueEmbedding::from_issue(&issue);
        assert!(emb.labels.is_some());
        let labels_str = emb.labels.unwrap();
        assert!(labels_str.contains("bug"));
        assert!(labels_str.contains("urgent"));
    }

    #[test]
    fn test_issue_embedding_from_issue_empty_labels() {
        let mut issue = Issue::new("id-3", "SHORT-3", "Title", "url", "sentry");
        issue.set_metadata("labels", Vec::<String>::new());
        let emb = IssueEmbedding::from_issue(&issue);
        assert!(emb.labels.is_none());
    }

    #[test]
    fn test_issue_embedding_from_issue_with_description() {
        let mut issue = Issue::new("id-4", "SHORT-4", "Title", "url", "linear");
        issue.description = Some("Detailed description".to_string());
        let emb = IssueEmbedding::from_issue(&issue);
        assert_eq!(emb.description, Some("Detailed description".to_string()));
    }

    #[test]
    fn test_issue_embedding_from_issue_with_priority_and_status() {
        let mut issue = Issue::new("id-5", "SHORT-5", "Title", "url", "linear");
        issue.priority = IssuePriority::Critical;
        issue.status = IssueStatus::InProgress;
        let emb = IssueEmbedding::from_issue(&issue);
        assert_eq!(emb.priority, Some("critical".to_string()));
        assert_eq!(emb.status, Some("in_progress".to_string()));
    }

    #[test]
    fn test_issue_embedding_from_issue_with_timestamps() {
        let now = Utc::now();
        let mut issue = Issue::new("id-6", "SHORT-6", "Title", "url", "linear");
        issue.created_at = Some(now);
        issue.updated_at = Some(now);
        let emb = IssueEmbedding::from_issue(&issue);
        assert_eq!(emb.created_at, now);
        assert_eq!(emb.updated_at, Some(now));
    }

    #[test]
    fn test_issue_embedding_new_with_vector() {
        let vec = vec![0.1, 0.2, 0.3, 0.4];
        let emb = IssueEmbedding::new("sentry", "issue-1", vec.clone());
        assert_eq!(emb.source, "sentry");
        assert_eq!(emb.issue_id, "issue-1");
        assert_eq!(emb.embedding, Some(vec));
        assert_eq!(emb.id, 0);
        assert!(emb.short_id.is_none());
        assert!(emb.title.is_none());
    }

    #[test]
    fn test_pr_review_record_new() {
        let r = PrReviewRecord::new("https://github.com/org/repo/pull/1");
        assert_eq!(r.id, 0);
        assert_eq!(r.pr_url, "https://github.com/org/repo/pull/1");
        assert!(r.attempt_id.is_none());
        assert!(r.reviewer.is_none());
        assert!(r.review_state.is_none());
        assert!(r.submitted_at.is_none());
        assert!(r.body.is_none());
        assert!(r.sentiment.is_none());
        assert!(r.actionable_feedback.is_none());
    }

    #[test]
    fn test_pr_review_record_serde() {
        let mut r = PrReviewRecord::new("https://github.com/org/repo/pull/1");
        r.reviewer = Some("alice".to_string());
        r.review_state = Some("approved".to_string());
        r.body = Some("LGTM".to_string());
        r.sentiment = Some("positive".to_string());
        let json = serde_json::to_string(&r).unwrap();
        let parsed: PrReviewRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.reviewer, Some("alice".to_string()));
        assert_eq!(parsed.review_state, Some("approved".to_string()));
    }

    #[test]
    fn test_ask_request_serde() {
        let q = BlockingQuestion {
            question: "Which API?".into(),
            context: None,
            options: vec!["REST".into()],
            why: None,
        };
        let req = AskRequest {
            correlation_id: "corr-1".into(),
            source: "linear".into(),
            repo: Some("org/repo".into()),
            issue_id: "issue-1".into(),
            short_id: "LIN-1".into(),
            question: q,
            asked_at: Utc::now(),
            target_discord_id: Some("123".into()),
            target_email: None,
            target_slack_id: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        let parsed: AskRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.correlation_id, "corr-1");
        assert_eq!(parsed.question.options.len(), 1);
    }

    #[test]
    fn test_ask_delivery_serde() {
        let d = AskDelivery {
            channel: "discord".into(),
            target: Some("user-123".into()),
            message_id: Some("msg-456".into()),
        };
        let json = serde_json::to_string(&d).unwrap();
        let parsed: AskDelivery = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.channel, "discord");
        assert_eq!(parsed.target, Some("user-123".into()));
    }

    #[test]
    fn test_ask_delivery_minimal() {
        let d = AskDelivery {
            channel: "email".into(),
            target: None,
            message_id: None,
        };
        let json = serde_json::to_string(&d).unwrap();
        assert!(!json.contains("target"));
        assert!(!json.contains("message_id"));
    }

    #[test]
    fn test_ask_reply_serde() {
        let r = AskReply {
            correlation_id: "corr-1".into(),
            channel: "discord".into(),
            responder: Some("user-1".into()),
            answer: "Use REST".into(),
            replied_at: Utc::now(),
        };
        let json = serde_json::to_string(&r).unwrap();
        let parsed: AskReply = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.answer, "Use REST");
        assert_eq!(parsed.correlation_id, "corr-1");
    }

    #[test]
    fn test_issue_type_display_all_variants() {
        assert_eq!(IssueType::SentryIssue.to_string(), "sentry_issue");
        assert_eq!(IssueType::LinearBug.to_string(), "linear_bug");
        assert_eq!(IssueType::GitLabIssue.to_string(), "gitlab_issue");
        assert_eq!(IssueType::JiraIssue.to_string(), "jira_issue");
    }

    #[test]
    fn test_issue_type_from_str_all_variants() {
        assert_eq!(
            "sentry_issue".parse::<IssueType>().unwrap(),
            IssueType::SentryIssue
        );
        assert_eq!(
            "linear_bug".parse::<IssueType>().unwrap(),
            IssueType::LinearBug
        );
        assert_eq!(
            "gitlab_issue".parse::<IssueType>().unwrap(),
            IssueType::GitLabIssue
        );
        assert_eq!(
            "jira_issue".parse::<IssueType>().unwrap(),
            IssueType::JiraIssue
        );
    }

    #[test]
    fn test_issue_type_source_name_all_variants() {
        assert_eq!(IssueType::SentryIssue.source_name(), "sentry");
        assert_eq!(IssueType::LinearBug.source_name(), "linear");
        assert_eq!(IssueType::GitLabIssue.source_name(), "gitlab");
        assert_eq!(IssueType::JiraIssue.source_name(), "jira");
    }

    #[test]
    fn test_issue_type_serde_all_four_variants() {
        for issue_type in [
            IssueType::SentryIssue,
            IssueType::LinearBug,
            IssueType::GitLabIssue,
            IssueType::JiraIssue,
        ] {
            let json = serde_json::to_string(&issue_type).unwrap();
            let parsed: IssueType = serde_json::from_str(&json).unwrap();
            assert_eq!(issue_type, parsed);
        }
    }

    #[test]
    fn test_issue_type_display_parse_roundtrip_all_variants() {
        for issue_type in [
            IssueType::SentryIssue,
            IssueType::LinearBug,
            IssueType::GitLabIssue,
            IssueType::JiraIssue,
        ] {
            let parsed: IssueType = issue_type.to_string().parse().unwrap();
            assert_eq!(issue_type, parsed);
        }
    }

    #[test]
    fn test_time_savings_default() {
        let ts = TimeSavings::default();
        assert_eq!(ts.merged_count, 0);
        assert_eq!(ts.hours_saved, 0.0);
        assert_eq!(ts.cost_saved, 0.0);
        assert!(ts.period.is_empty());
    }

    #[test]
    fn test_time_savings_serde() {
        let ts = TimeSavings {
            merged_count: 10,
            hours_saved: 50.5,
            cost_saved: 5050.0,
            period: "2026-01".to_string(),
        };
        let json = serde_json::to_string(&ts).unwrap();
        let parsed: TimeSavings = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.merged_count, 10);
        assert!((parsed.hours_saved - 50.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cost_estimate_default() {
        let ce = CostEstimate::default();
        assert_eq!(ce.total_cost, 0.0);
        assert_eq!(ce.avg_cost_per_fix, 0.0);
        assert_eq!(ce.fix_count, 0);
        assert!(ce.cost_source.is_empty());
        assert!(ce.period.is_empty());
    }

    #[test]
    fn test_cost_estimate_serde() {
        let ce = CostEstimate {
            total_cost: 123.45,
            avg_cost_per_fix: 12.34,
            fix_count: 10,
            cost_source: "claude_cli".to_string(),
            period: "2026-01".to_string(),
        };
        let json = serde_json::to_string(&ce).unwrap();
        let parsed: CostEstimate = serde_json::from_str(&json).unwrap();
        assert!((parsed.total_cost - 123.45).abs() < f64::EPSILON);
        assert_eq!(parsed.cost_source, "claude_cli");
    }

    #[test]
    fn test_mttr_data_point_serde() {
        let dp = MttrDataPoint {
            period_start: "2026-01-01".to_string(),
            mttr_minutes: 45.5,
            sample_count: 12,
        };
        let json = serde_json::to_string(&dp).unwrap();
        let parsed: MttrDataPoint = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.period_start, "2026-01-01");
        assert!((parsed.mttr_minutes - 45.5).abs() < f64::EPSILON);
        assert_eq!(parsed.sample_count, 12);
    }

    #[test]
    fn test_repo_leaderboard_entry_serde() {
        let entry = RepoLeaderboardEntry {
            repo: "org/repo".to_string(),
            total: 100,
            success_rate: 0.85,
            merge_rate: 0.72,
            avg_time_to_merge_mins: Some(120.0),
        };
        let json = serde_json::to_string(&entry).unwrap();
        let parsed: RepoLeaderboardEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.repo, "org/repo");
        assert!((parsed.success_rate - 0.85).abs() < f64::EPSILON);
    }

    #[test]
    fn test_rejection_reason_default() {
        let rr = RejectionReason::default();
        assert!(rr.category.is_empty());
        assert_eq!(rr.count, 0);
    }

    #[test]
    fn test_rejection_reason_serde() {
        let rr = RejectionReason {
            category: "missing_tests".to_string(),
            count: 5,
        };
        let json = serde_json::to_string(&rr).unwrap();
        let parsed: RejectionReason = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.category, "missing_tests");
        assert_eq!(parsed.count, 5);
    }

    #[test]
    fn test_issue_cluster_member_serde() {
        let m = IssueClusterMember {
            cluster_id: 42,
            issue_id: "issue-99".to_string(),
            arrived_at: Utc::now(),
        };
        let json = serde_json::to_string(&m).unwrap();
        let parsed: IssueClusterMember = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.cluster_id, 42);
        assert_eq!(parsed.issue_id, "issue-99");
    }

    #[test]
    fn test_review_category_classified_variants() {
        let variants = ReviewCategory::classified_variants();
        assert_eq!(variants.len(), 7);
        assert!(!variants.contains(&ReviewCategory::Other));
        assert!(variants.contains(&ReviewCategory::MissingTests));
        assert!(variants.contains(&ReviewCategory::StyleIssue));
        assert!(variants.contains(&ReviewCategory::WrongApproach));
        assert!(variants.contains(&ReviewCategory::Incomplete));
        assert!(variants.contains(&ReviewCategory::Security));
        assert!(variants.contains(&ReviewCategory::Performance));
        assert!(variants.contains(&ReviewCategory::Documentation));
    }

    #[test]
    fn test_claude_execution_serde_roundtrip() {
        let mut exec = AgentExecution::new();
        exec.attempt_id = Some(42);
        exec.model_version = Some("claude-3".to_string());
        exec.total_cost_usd = Some(0.05);
        exec.input_tokens = Some(1000);
        exec.output_tokens = Some(500);
        let json = serde_json::to_string(&exec).unwrap();
        let parsed: AgentExecution = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.attempt_id, Some(42));
        assert_eq!(parsed.model_version, Some("claude-3".to_string()));
        assert!((parsed.total_cost_usd.unwrap() - 0.05).abs() < f64::EPSILON);
    }

    #[test]
    fn test_claude_execution_new_defaults() {
        let exec = AgentExecution::new();
        assert_eq!(exec.id, 0);
        assert!(exec.attempt_id.is_none());
        assert!(exec.completed_at.is_none());
        assert!(exec.duration_secs.is_none());
        assert!(exec.exit_code.is_none());
        assert!(!exec.timed_out);
        assert!(exec.stdout_preview.is_none());
        assert!(exec.stderr_preview.is_none());
        assert!(exec.stdout_log_path.is_none());
        assert!(exec.stderr_log_path.is_none());
        assert!(exec.event_log_path.is_none());
        assert!(exec.prompt_used.is_none());
        assert!(exec.prompt_hash.is_none());
        assert!(exec.model_version.is_none());
        assert!(exec.working_directory.is_none());
        assert!(exec.git_branch.is_none());
        assert!(exec.git_commit_before.is_none());
        assert!(exec.git_commit_after.is_none());
        assert!(exec.files_changed.is_none());
        assert!(exec.lines_added.is_none());
        assert!(exec.lines_removed.is_none());
        assert!(exec.total_cost_usd.is_none());
        assert!(exec.num_turns.is_none());
        assert!(exec.session_id.is_none());
        assert!(exec.duration_api_ms.is_none());
        assert!(exec.input_tokens.is_none());
        assert!(exec.output_tokens.is_none());
        assert!(exec.cache_read_input_tokens.is_none());
        assert!(exec.cache_creation_input_tokens.is_none());
    }

    #[test]
    fn test_fix_attempt_serde_roundtrip() {
        let attempt = FixAttempt {
            id: 1,
            issue_id: "123".into(),
            short_id: "LIN-123".into(),
            source: "linear".into(),
            attempted_at: Utc::now(),
            pr_url: Some("https://github.com/org/repo/pull/42".into()),
            scm_repo: Some("org/repo".into()),
            scm_pr_number: Some(42),
            status: FixAttemptStatus::Success,
            error_message: None,
            merged_at: None,
            resolved_at: None,
            retry_count: 2,
            last_retry_at: Some(Utc::now()),
            issue_labels: vec!["bug".into(), "urgent".into()],
            parent_attempt_id: Some(0),
            cascade_repo: Some("org/other-repo".into()),
        };
        let json = serde_json::to_string(&attempt).unwrap();
        let parsed: FixAttempt = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.id, 1);
        assert_eq!(parsed.source, "linear");
        assert_eq!(parsed.retry_count, 2);
        assert_eq!(parsed.issue_labels.len(), 2);
        assert_eq!(parsed.cascade_repo, Some("org/other-repo".into()));
        assert_eq!(parsed.parent_attempt_id, Some(0));
    }

    #[test]
    fn test_claude_result_with_blocking_question() {
        let result = AgentResult {
            success: false,
            output: "Blocked".to_string(),
            pr_url: None,
            changelog: None,
            error: None,
            blocking_question: Some(BlockingQuestion {
                question: "Which database?".into(),
                context: Some("Need to choose".into()),
                options: vec!["PostgreSQL".into(), "MySQL".into()],
                why: Some("Architecture decision".into()),
            }),
            used_qa_ids: vec![1, 2, 3],
            confidence: 0,
            confidence_reasoning: None,
            wrong_repo: None,
        };
        let json = serde_json::to_string(&result).unwrap();
        let parsed: AgentResult = serde_json::from_str(&json).unwrap();
        assert!(parsed.blocking_question.is_some());
        let bq = parsed.blocking_question.unwrap();
        assert_eq!(bq.question, "Which database?");
        assert_eq!(bq.options.len(), 2);
        assert_eq!(parsed.used_qa_ids, vec![1, 2, 3]);
    }

    #[test]
    fn test_claude_result_serde_skip_none_fields() {
        let result = AgentResult {
            success: true,
            output: "ok".to_string(),
            pr_url: None,
            changelog: None,
            error: None,
            blocking_question: None,
            used_qa_ids: Vec::new(),
            confidence: 0,
            confidence_reasoning: None,
            wrong_repo: None,
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(!json.contains("pr_url"));
        assert!(!json.contains("error"));
        assert!(!json.contains("blocking_question"));
    }

    #[test]
    fn test_pr_analytics_serde_with_data() {
        let mut by_repo = HashMap::new();
        by_repo.insert("org/repo".to_string(), 25i64);
        let a = PrAnalytics {
            total: 50,
            open: 10,
            merged: 30,
            closed: 10,
            avg_time_to_first_review_mins: Some(30.0),
            avg_time_to_merge_mins: Some(120.0),
            avg_review_cycles: Some(1.5),
            merge_rate: Some(0.75),
            by_repo,
            avg_time_to_pr_mins: Some(15.0),
            rejection_reasons: vec![RejectionReason {
                category: "missing_tests".into(),
                count: 3,
            }],
        };
        let json = serde_json::to_string(&a).unwrap();
        let parsed: PrAnalytics = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.total, 50);
        assert_eq!(parsed.by_repo.get("org/repo"), Some(&25));
        assert_eq!(parsed.rejection_reasons.len(), 1);
    }

    #[test]
    fn test_analytics_summary_serde_with_data() {
        let mut sr_by_source = HashMap::new();
        sr_by_source.insert("sentry".to_string(), 0.9);
        let s = AnalyticsSummary {
            success_rate: 0.85,
            total_processed: 100,
            total_successful: 85,
            total_merged: 60,
            avg_processing_time_secs: Some(120.0),
            avg_time_to_merge_hours: Some(2.5),
            most_common_error: Some("build_failure".to_string()),
            success_rate_by_source: sr_by_source,
            avg_time_to_pr_mins: Some(10.0),
            cost_estimate: Some(CostEstimate {
                total_cost: 100.0,
                avg_cost_per_fix: 1.0,
                fix_count: 100,
                cost_source: "cli".into(),
                period: "2026-01".into(),
            }),
            mttr_trend: vec![MttrDataPoint {
                period_start: "2026-01-01".into(),
                mttr_minutes: 30.0,
                sample_count: 10,
            }],
            repo_leaderboard: vec![RepoLeaderboardEntry {
                repo: "org/repo".into(),
                total: 50,
                success_rate: 0.9,
                merge_rate: 0.8,
                avg_time_to_merge_mins: Some(60.0),
            }],
        };
        let json = serde_json::to_string(&s).unwrap();
        let parsed: AnalyticsSummary = serde_json::from_str(&json).unwrap();
        assert!((parsed.success_rate - 0.85).abs() < f64::EPSILON);
        assert_eq!(parsed.repo_leaderboard.len(), 1);
        assert!(parsed.cost_estimate.is_some());
    }

    #[test]
    fn test_extracted_learnings_populated() {
        let learnings = ExtractedLearnings {
            root_cause: Some("Null pointer dereference".to_string()),
            files_modified: vec!["src/main.rs".into(), "src/lib.rs".into()],
            strategy_used: Some("TDD approach".to_string()),
            tests_added: true,
            key_decisions: vec!["Used Option instead of unwrap".into()],
        };
        let json = serde_json::to_string(&learnings).unwrap();
        let parsed: ExtractedLearnings = serde_json::from_str(&json).unwrap();
        assert_eq!(
            parsed.root_cause,
            Some("Null pointer dereference".to_string())
        );
        assert_eq!(parsed.files_modified.len(), 2);
        assert!(parsed.tests_added);
        assert_eq!(parsed.key_decisions.len(), 1);
    }

    #[test]
    fn test_qa_knowledge_entry_full_serde() {
        let entry = QaKnowledgeEntry {
            id: 42,
            source: "linear".into(),
            repo: Some("org/repo".into()),
            issue_id: "issue-1".into(),
            short_id: "LIN-1".into(),
            question_text: "How to deploy?".into(),
            question_norm: "how to deploy".into(),
            question_embedding: Some(vec![0.1, 0.2, 0.3]),
            answer_text: "Run deploy.sh".into(),
            answer_norm: "run deploy.sh".into(),
            answer_embedding: Some(vec![0.4, 0.5, 0.6]),
            channel: "discord".into(),
            responder: Some("alice".into()),
            correlation_id: "corr-42".into(),
            asked_at: Utc::now(),
            answered_at: Utc::now(),
            success_count: 10,
            failure_count: 2,
            last_used_at: Some(Utc::now()),
            metadata: Some(serde_json::json!({"version": 2})),
        };
        let json = serde_json::to_string(&entry).unwrap();
        let parsed: QaKnowledgeEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.id, 42);
        assert_eq!(parsed.repo, Some("org/repo".into()));
        assert!(parsed.question_embedding.is_some());
        assert_eq!(parsed.success_count, 10);
        assert!(parsed.metadata.is_some());
    }

    #[test]
    fn test_error_pattern_serde_roundtrip() {
        let mut pattern = ErrorPattern::new("hash123");
        pattern.error_type = Some("build_failure".to_string());
        pattern.error_message = Some("compilation error".to_string());
        pattern.sources = Some(vec!["linear".into(), "sentry".into()]);
        pattern.example_issue_ids = Some(vec!["issue-1".into()]);
        pattern.resolution_hints = Some("Check Cargo.toml".to_string());
        let json = serde_json::to_string(&pattern).unwrap();
        let parsed: ErrorPattern = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.pattern_hash, "hash123");
        assert_eq!(parsed.error_type, Some("build_failure".to_string()));
        assert_eq!(parsed.sources.as_ref().unwrap().len(), 2);
    }

    #[test]
    fn test_pr_record_new_defaults() {
        let pr = PrRecord::new("https://github.com/org/repo/pull/5", "org/repo", 5);
        assert_eq!(pr.id, 0);
        assert_eq!(pr.pr_number, 5);
        assert_eq!(pr.status, "open");
        assert_eq!(pr.approvals_count, 0);
        assert_eq!(pr.changes_requested_count, 0);
        assert_eq!(pr.comments_count, 0);
        assert_eq!(pr.review_cycles, 0);
        assert!(pr.attempt_id.is_none());
        assert!(pr.issue_id.is_none());
        assert!(pr.issue_source.is_none());
        assert!(pr.title.is_none());
        assert!(pr.description.is_none());
        assert!(pr.author.is_none());
        assert!(pr.head_branch.is_none());
        assert!(pr.base_branch.is_none());
        assert!(pr.updated_at.is_none());
        assert!(pr.merged_at.is_none());
        assert!(pr.closed_at.is_none());
        assert!(pr.last_review_at.is_none());
        assert!(pr.time_to_first_review_mins.is_none());
        assert!(pr.time_to_merge_mins.is_none());
        assert!(pr.files_changed.is_none());
        assert!(pr.lines_added.is_none());
        assert!(pr.lines_removed.is_none());
    }

    #[test]
    fn test_pr_record_serde_roundtrip() {
        let mut pr = PrRecord::for_issue(
            "https://github.com/org/repo/pull/10",
            "org/repo",
            10,
            "sentry",
            "SENTRY-1",
        );
        pr.title = Some("Fix null pointer".to_string());
        pr.author = Some("bot".to_string());
        pr.head_branch = Some("fix/null-ptr".to_string());
        pr.base_branch = Some("main".to_string());
        pr.approvals_count = 2;
        pr.comments_count = 5;
        let json = serde_json::to_string(&pr).unwrap();
        let parsed: PrRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.pr_number, 10);
        assert_eq!(parsed.issue_source, Some("sentry".to_string()));
        assert_eq!(parsed.approvals_count, 2);
    }

    #[test]
    fn test_prompt_experiment_serde_roundtrip() {
        let mut exp = PromptExperiment::new("exp-1", "variant_a", "Fix: {{issue}}", "hash-abc");
        exp.success_count = 10;
        exp.failure_count = 2;
        exp.avg_time_to_merge = Some(3.5);
        exp.avg_review_score = Some(4.2);
        let json = serde_json::to_string(&exp).unwrap();
        let parsed: PromptExperiment = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.experiment_name, "exp-1");
        assert_eq!(parsed.variant, "variant_a");
        assert!(parsed.active);
        assert_eq!(parsed.success_count, 10);
        assert!((parsed.avg_time_to_merge.unwrap() - 3.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_similar_issue_serde_roundtrip() {
        let si = SimilarIssue::new("issue-a", "issue-b", 0.87);
        let json = serde_json::to_string(&si).unwrap();
        let parsed: SimilarIssue = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.source_issue_id, "issue-a");
        assert_eq!(parsed.similar_issue_id, "issue-b");
        assert!((parsed.similarity_score - 0.87).abs() < f64::EPSILON);
    }

    #[test]
    fn test_processing_metric_minimal() {
        let m = ProcessingMetric::new("latency", 0.5);
        assert_eq!(m.metric_name, "latency");
        assert_eq!(m.id, 0);
        assert!(m.source.is_none());
        assert!(m.tags.is_none());
    }

    #[test]
    fn test_processing_metric_serde_roundtrip() {
        let m = ProcessingMetric::new("queue_depth", 42.0)
            .with_source("linear")
            .with_tags(serde_json::json!({"env": "prod"}));
        let json = serde_json::to_string(&m).unwrap();
        let parsed: ProcessingMetric = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.metric_name, "queue_depth");
        assert_eq!(parsed.source, Some("linear".to_string()));
        assert!(parsed.tags.is_some());
    }

    #[test]
    fn test_fix_attempt_stats_serde_roundtrip() {
        let mut stats = FixAttemptStats {
            total: 100,
            success: 50,
            ..Default::default()
        };
        stats.by_source.insert(
            "sentry".to_string(),
            SourceStats {
                total: 50,
                success: 30,
                failed: 10,
                merged: 8,
                closed: 2,
                cannot_fix: 0,
            },
        );
        let json = serde_json::to_string(&stats).unwrap();
        let parsed: FixAttemptStats = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.total, 100);
        assert!(parsed.by_source.contains_key("sentry"));
        assert_eq!(parsed.by_source["sentry"].success, 30);
    }

    #[test]
    fn test_activity_log_entry_serde_roundtrip() {
        let entry = ActivityLogEntry::new("pr_created", "PR #42 created")
            .with_source("linear")
            .with_issue("issue-1", "LIN-1")
            .with_metadata(serde_json::json!({"pr_number": 42}));
        let json = serde_json::to_string(&entry).unwrap();
        let parsed: ActivityLogEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.activity_type, "pr_created");
        assert_eq!(parsed.message, "PR #42 created");
        assert_eq!(parsed.source, Some("linear".to_string()));
        assert_eq!(parsed.issue_id, Some("issue-1".to_string()));
        assert_eq!(parsed.short_id, Some("LIN-1".to_string()));
        assert!(parsed.metadata.is_some());
    }

    #[test]
    fn test_issue_deserialize_minimal_json() {
        let json = r#"{
            "id": "1",
            "short_id": "T-1",
            "title": "Title",
            "url": "https://example.com",
            "source": "test",
            "priority": "none",
            "status": "open"
        }"#;
        let issue: Issue = serde_json::from_str(json).unwrap();
        assert_eq!(issue.id, "1");
        assert!(issue.description.is_none());
        assert!(issue.metadata.is_empty());
        assert!(issue.created_at.is_none());
        assert!(issue.updated_at.is_none());
    }

    #[test]
    fn test_issue_metadata_default_empty() {
        let json = r#"{
            "id": "1",
            "short_id": "T-1",
            "title": "Title",
            "url": "url",
            "source": "src",
            "priority": "low",
            "status": "resolved"
        }"#;
        let issue: Issue = serde_json::from_str(json).unwrap();
        assert!(issue.metadata.is_empty());
    }

    #[test]
    fn test_fix_attempt_deserialize_with_defaults() {
        let json = serde_json::json!({
            "id": 1,
            "issue_id": "123",
            "short_id": "T-123",
            "source": "linear",
            "attempted_at": "2026-01-01T00:00:00Z",
            "status": "pending"
        });
        let attempt: FixAttempt = serde_json::from_value(json).unwrap();
        assert_eq!(attempt.retry_count, 0);
        assert!(attempt.issue_labels.is_empty());
        assert!(attempt.pr_url.is_none());
    }

    #[test]
    fn test_claude_result_deserialize_with_defaults() {
        let json = serde_json::json!({
            "success": true,
            "output": "done"
        });
        let result: AgentResult = serde_json::from_value(json).unwrap();
        assert!(result.success);
        assert!(result.used_qa_ids.is_empty());
        assert!(result.blocking_question.is_none());
    }

    #[test]
    fn test_blocking_question_deserialize_with_defaults() {
        let json = serde_json::json!({
            "question": "What API?"
        });
        let bq: BlockingQuestion = serde_json::from_value(json).unwrap();
        assert_eq!(bq.question, "What API?");
        assert!(bq.context.is_none());
        assert!(bq.options.is_empty());
        assert!(bq.why.is_none());
    }

    #[test]
    fn test_change_category_equality_and_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(ChangeCategory::Tests);
        set.insert(ChangeCategory::Tests);
        assert_eq!(set.len(), 1);
        set.insert(ChangeCategory::Docs);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_review_category_equality_and_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(ReviewCategory::Security);
        set.insert(ReviewCategory::Security);
        assert_eq!(set.len(), 1);
        set.insert(ReviewCategory::Performance);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_match_priority_serde_snake_case() {
        let json = serde_json::to_string(&MatchPriority::Low).unwrap();
        assert_eq!(json, "\"low\"");
        let json = serde_json::to_string(&MatchPriority::Normal).unwrap();
        assert_eq!(json, "\"normal\"");
        let json = serde_json::to_string(&MatchPriority::High).unwrap();
        assert_eq!(json, "\"high\"");
        let json = serde_json::to_string(&MatchPriority::Urgent).unwrap();
        assert_eq!(json, "\"urgent\"");
    }

    #[test]
    fn test_issue_priority_serde_snake_case() {
        assert_eq!(
            serde_json::to_string(&IssuePriority::None).unwrap(),
            "\"none\""
        );
        assert_eq!(
            serde_json::to_string(&IssuePriority::Low).unwrap(),
            "\"low\""
        );
        assert_eq!(
            serde_json::to_string(&IssuePriority::Medium).unwrap(),
            "\"medium\""
        );
        assert_eq!(
            serde_json::to_string(&IssuePriority::High).unwrap(),
            "\"high\""
        );
        assert_eq!(
            serde_json::to_string(&IssuePriority::Critical).unwrap(),
            "\"critical\""
        );
    }

    #[test]
    fn test_issue_skip_serializing_none_fields() {
        let issue = Issue::new("1", "T-1", "Title", "url", "src");
        let json = serde_json::to_string(&issue).unwrap();
        assert!(!json.contains("description"));
        assert!(!json.contains("created_at"));
        assert!(!json.contains("updated_at"));
    }

    #[test]
    fn test_fix_attempt_skip_serializing_none_fields() {
        let attempt = FixAttempt {
            id: 1,
            issue_id: "1".into(),
            short_id: "T-1".into(),
            source: "linear".into(),
            attempted_at: Utc::now(),
            pr_url: None,
            scm_repo: None,
            scm_pr_number: None,
            status: FixAttemptStatus::Pending,
            error_message: None,
            merged_at: None,
            resolved_at: None,
            retry_count: 0,
            last_retry_at: None,
            issue_labels: vec![],
            parent_attempt_id: None,
            cascade_repo: None,
        };
        let json = serde_json::to_string(&attempt).unwrap();
        assert!(!json.contains("pr_url"));
        assert!(!json.contains("scm_repo"));
        assert!(!json.contains("scm_pr_number"));
        assert!(!json.contains("error_message"));
        assert!(!json.contains("merged_at"));
        assert!(!json.contains("resolved_at"));
        assert!(!json.contains("last_retry_at"));
        assert!(!json.contains("issue_labels"));
        assert!(!json.contains("parent_attempt_id"));
        assert!(!json.contains("cascade_repo"));
    }

    #[test]
    fn test_validate_issue_id_error_messages() {
        let err = validate_issue_id("").unwrap_err();
        assert!(err.contains("empty"), "Expected 'empty' in: {}", err);

        let err = validate_issue_id(&"x".repeat(101)).unwrap_err();
        assert!(
            err.contains("maximum length"),
            "Expected 'maximum length' in: {}",
            err
        );

        let err = validate_issue_id("a..b").unwrap_err();
        assert!(
            err.contains("path traversal"),
            "Expected 'path traversal' in: {}",
            err
        );

        let err = validate_issue_id("a/b").unwrap_err();
        assert!(
            err.contains("forward slashes"),
            "Expected 'forward slashes' in: {}",
            err
        );

        let err = validate_issue_id("a\\b").unwrap_err();
        assert!(
            err.contains("backslashes"),
            "Expected 'backslashes' in: {}",
            err
        );

        let err = validate_issue_id("a\0b").unwrap_err();
        assert!(
            err.contains("null bytes"),
            "Expected 'null bytes' in: {}",
            err
        );
    }

    #[test]
    fn test_agent_execution_new_has_none_provider_fields() {
        let exec = AgentExecution::new();
        assert!(exec.provider.is_none());
        assert!(exec.experiment_name.is_none());
        assert!(exec.experiment_variant.is_none());
    }

    #[test]
    fn test_agent_execution_provider_field() {
        let mut exec = AgentExecution::new();
        exec.provider = Some("codex".to_string());
        assert_eq!(exec.provider.as_deref(), Some("codex"));
    }

    #[test]
    fn test_agent_execution_experiment_fields() {
        let mut exec = AgentExecution::new();
        exec.experiment_name = Some("claude-vs-codex".to_string());
        exec.experiment_variant = Some("claude".to_string());
        assert_eq!(exec.experiment_name.as_deref(), Some("claude-vs-codex"));
        assert_eq!(exec.experiment_variant.as_deref(), Some("claude"));
    }

    #[test]
    fn test_agent_execution_serialization_with_provider() {
        let mut exec = AgentExecution::new();
        exec.provider = Some("claude".to_string());
        exec.experiment_name = Some("test-exp".to_string());
        let json = serde_json::to_string(&exec).unwrap();
        assert!(json.contains("\"provider\":\"claude\""));
        assert!(json.contains("\"experiment_name\":\"test-exp\""));
    }

    #[test]
    fn test_agent_execution_serialization_skips_none_provider() {
        let exec = AgentExecution::new();
        let json = serde_json::to_string(&exec).unwrap();
        // skip_serializing_if = "Option::is_none" should omit these
        assert!(!json.contains("\"provider\""));
        assert!(!json.contains("\"experiment_name\""));
        assert!(!json.contains("\"experiment_variant\""));
    }

    #[test]
    fn test_experiment_provider_stats_serialization() {
        let stats = ExperimentProviderStats {
            provider: "claude".to_string(),
            total_attempts: 100,
            success_count: 85,
            avg_cost: Some(0.42),
            avg_duration: Some(120.5),
            success_rate: 0.85,
        };
        let json = serde_json::to_string(&stats).unwrap();
        assert!(json.contains("\"provider\":\"claude\""));
        assert!(json.contains("\"total_attempts\":100"));
        assert!(json.contains("\"success_count\":85"));
        assert!(json.contains("\"success_rate\":0.85"));
    }

    #[test]
    fn test_experiment_provider_stats_deserialization() {
        let json = r#"{
            "provider": "codex",
            "total_attempts": 50,
            "success_count": 30,
            "avg_cost": null,
            "avg_duration": 90.0,
            "success_rate": 0.6
        }"#;
        let stats: ExperimentProviderStats = serde_json::from_str(json).unwrap();
        assert_eq!(stats.provider, "codex");
        assert_eq!(stats.total_attempts, 50);
        assert_eq!(stats.success_count, 30);
        assert!(stats.avg_cost.is_none());
        assert_eq!(stats.avg_duration, Some(90.0));
        assert!((stats.success_rate - 0.6).abs() < f64::EPSILON);
    }

    #[test]
    fn test_experiment_provider_stats_zero_values() {
        let stats = ExperimentProviderStats {
            provider: "gemini".to_string(),
            total_attempts: 0,
            success_count: 0,
            avg_cost: None,
            avg_duration: None,
            success_rate: 0.0,
        };
        assert_eq!(stats.total_attempts, 0);
        assert_eq!(stats.success_count, 0);
        assert!(stats.avg_cost.is_none());
        assert_eq!(stats.success_rate, 0.0);
    }

    // --- AgentExecution provider/experiment field tests ---

    #[test]
    fn test_agent_execution_new_has_no_provider() {
        let exec = AgentExecution::new();
        assert!(exec.provider.is_none());
        assert!(exec.experiment_name.is_none());
        assert!(exec.experiment_variant.is_none());
    }

    #[test]
    fn test_agent_execution_complete_sets_duration() {
        let mut exec = AgentExecution::new();
        std::thread::sleep(std::time::Duration::from_millis(10));
        exec.complete(Some(0), false);
        assert!(exec.completed_at.is_some());
        assert!(exec.duration_secs.unwrap() >= 0.0);
        assert_eq!(exec.exit_code, Some(0));
        assert!(!exec.timed_out);
    }

    #[test]
    fn test_agent_execution_complete_timed_out() {
        let mut exec = AgentExecution::new();
        exec.complete(None, true);
        assert!(exec.timed_out);
        assert!(exec.exit_code.is_none());
    }

    #[test]
    fn test_agent_execution_with_attempt_id_builder() {
        let exec = AgentExecution::new().with_attempt_id(42);
        assert_eq!(exec.attempt_id, Some(42));
    }

    #[test]
    fn test_agent_execution_default_equals_new() {
        let a = AgentExecution::new();
        let b = AgentExecution::default();
        assert_eq!(a.id, b.id);
        assert_eq!(a.provider, b.provider);
        assert_eq!(a.timed_out, b.timed_out);
    }

    #[test]
    fn test_agent_execution_serialization_skips_none_fields() {
        let exec = AgentExecution::new();
        let json = serde_json::to_string(&exec).unwrap();
        // None fields should be skipped
        assert!(!json.contains("\"provider\""));
        assert!(!json.contains("\"experiment_name\""));
        assert!(!json.contains("\"experiment_variant\""));
        assert!(!json.contains("\"stdout_preview\""));
    }

    // --- AgentResult serialization tests ---

    #[test]
    fn test_agent_result_success_serialization() {
        let result = AgentResult {
            success: true,
            output: "Done".to_string(),
            pr_url: Some("https://github.com/org/repo/pull/1".to_string()),
            changelog: Some("- Fixed bug".to_string()),
            error: None,
            blocking_question: None,
            used_qa_ids: vec![1, 2, 3],
            confidence: 0,
            confidence_reasoning: None,
            wrong_repo: None,
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("\"success\":true"));
        assert!(json.contains("\"pr_url\""));
        assert!(!json.contains("\"error\""));
        assert!(!json.contains("\"blocking_question\""));
    }

    #[test]
    fn test_agent_result_failure_serialization() {
        let result = AgentResult {
            success: false,
            output: String::new(),
            pr_url: None,
            changelog: None,
            error: Some("Process exited with code 1".to_string()),
            blocking_question: None,
            used_qa_ids: Vec::new(),
            confidence: 0,
            confidence_reasoning: None,
            wrong_repo: None,
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("\"success\":false"));
        assert!(json.contains("\"error\""));
        assert!(!json.contains("\"pr_url\""));
    }

    #[test]
    fn test_agent_result_with_blocking_question() {
        let result = AgentResult {
            success: false,
            output: String::new(),
            pr_url: None,
            changelog: None,
            error: None,
            blocking_question: Some(BlockingQuestion {
                question: "Which database?".to_string(),
                context: Some("Multiple DBs found".to_string()),
                options: vec!["postgres".to_string(), "mysql".to_string()],
                why: Some("Need to know which DB to fix".to_string()),
            }),
            used_qa_ids: Vec::new(),
            confidence: 0,
            confidence_reasoning: None,
            wrong_repo: None,
        };
        let json = serde_json::to_string(&result).unwrap();
        let deser: AgentResult = serde_json::from_str(&json).unwrap();
        assert!(deser.blocking_question.is_some());
        let bq = deser.blocking_question.unwrap();
        assert_eq!(bq.question, "Which database?");
        assert_eq!(bq.options.len(), 2);
    }

    #[test]
    fn test_agent_result_roundtrip() {
        let original = AgentResult {
            success: true,
            output: "Fixed the bug".to_string(),
            pr_url: Some("https://github.com/a/b/pull/99".to_string()),
            changelog: Some("- Updated handler\n- Added test".to_string()),
            error: None,
            blocking_question: None,
            used_qa_ids: vec![5, 10],
            confidence: 85,
            confidence_reasoning: Some("Tests pass and fix is straightforward".to_string()),
            wrong_repo: None,
        };
        let json = serde_json::to_string(&original).unwrap();
        let deser: AgentResult = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.success, original.success);
        assert_eq!(deser.output, original.output);
        assert_eq!(deser.pr_url, original.pr_url);
        assert_eq!(deser.changelog, original.changelog);
        assert_eq!(deser.used_qa_ids, original.used_qa_ids);
    }

    #[test]
    fn test_agent_result_empty_used_qa_ids_default() {
        let json = r#"{"success":true,"output":"ok"}"#;
        let result: AgentResult = serde_json::from_str(json).unwrap();
        assert!(result.used_qa_ids.is_empty());
    }

    #[test]
    fn test_agent_result_confidence_defaults_when_missing() {
        let json = r#"{"success":true,"output":"done"}"#;
        let result: AgentResult = serde_json::from_str(json).unwrap();
        assert_eq!(result.confidence, 0);
        assert!(result.confidence_reasoning.is_none());
    }

    #[test]
    fn test_agent_result_confidence_roundtrip_values() {
        let original = AgentResult {
            success: true,
            output: "Fixed".to_string(),
            pr_url: Some("https://github.com/a/b/pull/1".to_string()),
            changelog: None,
            error: None,
            blocking_question: None,
            used_qa_ids: vec![],
            confidence: 92,
            confidence_reasoning: Some("All tests pass, fix is straightforward".to_string()),
            wrong_repo: None,
        };
        let json = serde_json::to_string(&original).unwrap();
        let deser: AgentResult = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.confidence, 92);
        assert_eq!(
            deser.confidence_reasoning.as_deref(),
            Some("All tests pass, fix is straightforward")
        );
    }

    #[test]
    fn test_agent_result_confidence_reasoning_skipped_when_none() {
        let result = AgentResult {
            success: true,
            output: "ok".to_string(),
            pr_url: None,
            changelog: None,
            error: None,
            blocking_question: None,
            used_qa_ids: vec![],
            confidence: 75,
            confidence_reasoning: None,
            wrong_repo: None,
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("\"confidence\":75"));
        assert!(
            !json.contains("confidence_reasoning"),
            "confidence_reasoning should be skipped when None"
        );
    }

    #[test]
    fn test_agent_result_confidence_reasoning_present_when_some() {
        let result = AgentResult {
            success: true,
            output: "ok".to_string(),
            pr_url: None,
            changelog: None,
            error: None,
            blocking_question: None,
            used_qa_ids: vec![],
            confidence: 60,
            confidence_reasoning: Some("Moderate certainty".to_string()),
            wrong_repo: None,
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("\"confidence\":60"));
        assert!(json.contains("\"confidence_reasoning\":\"Moderate certainty\""));
    }

    #[test]
    fn test_agent_result_confidence_boundary_max_u8() {
        let result = AgentResult {
            success: true,
            output: "ok".to_string(),
            pr_url: None,
            changelog: None,
            error: None,
            blocking_question: None,
            used_qa_ids: vec![],
            confidence: 100,
            confidence_reasoning: None,
            wrong_repo: None,
        };
        let json = serde_json::to_string(&result).unwrap();
        let deser: AgentResult = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.confidence, 100);
    }

    #[test]
    fn test_agent_result_confidence_over_255_rejected() {
        // u8 can hold 0-255, but JSON number 300 should fail deserialization
        let json = r#"{"success":true,"output":"ok","confidence":300}"#;
        let result = serde_json::from_str::<AgentResult>(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_agent_result_confidence_negative_rejected() {
        let json = r#"{"success":true,"output":"ok","confidence":-1}"#;
        let result = serde_json::from_str::<AgentResult>(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_agent_result_confidence_as_string_rejected() {
        let json = r#"{"success":true,"output":"ok","confidence":"high"}"#;
        let result = serde_json::from_str::<AgentResult>(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_agent_result_confidence_with_all_fields() {
        let result = AgentResult {
            success: true,
            output: "Fixed the auth bypass".to_string(),
            pr_url: Some("https://github.com/org/repo/pull/42".to_string()),
            changelog: Some("- Added input validation\n- Added unit test".to_string()),
            error: None,
            blocking_question: None,
            used_qa_ids: vec![1, 5],
            confidence: 95,
            confidence_reasoning: Some(
                "Exact reproduction test added, fix is minimal and targeted".to_string(),
            ),
            wrong_repo: None,
        };
        let json = serde_json::to_string(&result).unwrap();
        let deser: AgentResult = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.confidence, 95);
        assert_eq!(
            deser.pr_url.as_deref(),
            Some("https://github.com/org/repo/pull/42")
        );
        assert_eq!(deser.used_qa_ids, vec![1, 5]);
        assert!(deser.confidence_reasoning.unwrap().contains("minimal"));
    }

    #[test]
    fn test_agent_result_confidence_empty_reasoning_serialized() {
        let result = AgentResult {
            success: true,
            output: "ok".to_string(),
            pr_url: None,
            changelog: None,
            error: None,
            blocking_question: None,
            used_qa_ids: vec![],
            confidence: 40,
            confidence_reasoning: Some(String::new()),
            wrong_repo: None,
        };
        let json = serde_json::to_string(&result).unwrap();
        // Empty string is still Some(""), so it should be serialized
        assert!(json.contains("\"confidence_reasoning\":\"\""));
        let deser: AgentResult = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.confidence_reasoning.as_deref(), Some(""));
    }

    #[test]
    fn test_agent_result_confidence_float_rejected() {
        // serde_json rejects float values for u8 fields
        let json = r#"{"success":true,"output":"ok","confidence":85.9}"#;
        let result = serde_json::from_str::<AgentResult>(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_agent_result_confidence_zero_with_reasoning() {
        // Edge case: confidence=0 but reasoning still provided
        let result = AgentResult {
            success: false,
            output: "could not fix".to_string(),
            pr_url: None,
            changelog: None,
            error: Some("compilation failed".to_string()),
            blocking_question: None,
            used_qa_ids: vec![],
            confidence: 0,
            confidence_reasoning: Some("Code doesn't compile after changes".to_string()),
            wrong_repo: None,
        };
        let json = serde_json::to_string(&result).unwrap();
        let deser: AgentResult = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.confidence, 0);
        assert!(deser.confidence_reasoning.is_some());
    }
}
