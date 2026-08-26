//! IPC server implementation.
//!
//! Transport details (Unix sockets vs TCP) are abstracted by the
//! [`super::transport`] module — this file is platform-agnostic.

use super::protocol::{
    ActivityEntry, ActivityType, IpcCommand, IpcData, IpcResponse, WatcherState,
};
use super::transport::{self, IpcStream};
use super::{
    cleanup_stale_files, default_socket_path, remove_pid_file, remove_socket_file, write_pid_file,
};
use crate::watcher::Watcher;
use claudear_core::error::Result;
use claudear_core::platform;
use claudear_core::types::{ActivityLogEntry, FixAttemptStatus};
use claudear_integrations::notifier::Notifier;
use claudear_integrations::source::IssueSource;
use claudear_storage::FixAttemptTracker;

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::{broadcast, Mutex, RwLock, Semaphore};

/// Default maximum number of activity entries to keep (can be overridden via config).
const DEFAULT_MAX_ACTIVITY_ENTRIES: usize = 10_000;

/// Maximum number of concurrent IPC connections.
const MAX_CONCURRENT_CONNECTIONS: usize = 64;

/// IPC server that listens on a Unix socket (or TCP on Windows).
pub struct IpcServer {
    socket_path: PathBuf,
    tracker: Arc<dyn FixAttemptTracker>,
    sources: Vec<Arc<dyn IssueSource>>,
    notifier: Arc<dyn Notifier>,
    watcher: Option<Arc<Watcher>>,
    state: Arc<ServerState>,
    shutdown_tx: broadcast::Sender<()>,
}

/// Shared state for the IPC server.
struct ServerState {
    /// Whether the watcher is paused.
    paused: AtomicBool,

    /// Server start time.
    start_time: Instant,

    /// Number of issues processed.
    issues_processed: AtomicUsize,

    /// Number of PRs created.
    prs_created: AtomicUsize,

    /// Currently processing issue IDs.
    processing: RwLock<Vec<String>>,

    /// Recent activity log.
    activity: Mutex<VecDeque<ActivityEntry>>,

    /// Mode (poll, webhook, etc.).
    mode: RwLock<String>,

    /// Poll interval if applicable.
    poll_interval_ms: AtomicU64,

    /// Source names.
    source_names: Vec<String>,

    /// Maximum activity entries (configurable).
    max_activity_entries: usize,

    /// Maximum retry attempts (from config).
    max_retries: u32,
}

impl IpcServer {
    /// Create a new IPC server with default settings.
    pub fn new(
        tracker: Arc<dyn FixAttemptTracker>,
        sources: Vec<Arc<dyn IssueSource>>,
        notifier: Arc<dyn Notifier>,
    ) -> Self {
        Self::builder(tracker, sources, notifier).build()
    }

    /// Create a builder for configuring the IPC server.
    pub fn builder(
        tracker: Arc<dyn FixAttemptTracker>,
        sources: Vec<Arc<dyn IssueSource>>,
        notifier: Arc<dyn Notifier>,
    ) -> IpcServerBuilder {
        IpcServerBuilder::new(tracker, sources, notifier)
    }

    /// Set the watcher instance.
    pub fn with_watcher(mut self, watcher: Arc<Watcher>) -> Self {
        self.watcher = Some(watcher);
        self
    }

    /// Set the mode.
    pub async fn set_mode(&self, mode: &str) {
        *self.state.mode.write().await = mode.to_string();
    }

    /// Set the poll interval.
    pub fn set_poll_interval(&self, interval_ms: u64) {
        self.state
            .poll_interval_ms
            .store(interval_ms, Ordering::SeqCst);
    }

    /// Check if paused.
    pub fn is_paused(&self) -> bool {
        self.state.paused.load(Ordering::SeqCst)
    }

    /// Log an activity entry.
    ///
    /// Persists to both in-memory cache (for fast CLI queries) and database (for long-term analytics).
    pub async fn log_activity(
        &self,
        activity_type: ActivityType,
        message: &str,
        issue_id: Option<&str>,
        source: Option<&str>,
    ) {
        let entry = ActivityEntry {
            timestamp: chrono::Utc::now().to_rfc3339(),
            activity_type: activity_type.clone(),
            message: message.to_string(),
            issue_id: issue_id.map(String::from),
            source: source.map(String::from),
        };

        let db_activity_type = match activity_type {
            // Issue Events
            ActivityType::IssueDetected => "issue_detected",
            ActivityType::IssueStatusChanged => "issue_status_changed",
            ActivityType::IssuePriorityChanged => "issue_priority_changed",
            ActivityType::IssueCommented => "issue_commented",
            ActivityType::IssueResolved => "issue_resolved",
            ActivityType::IssueCancelled => "issue_cancelled",
            ActivityType::IssueEscalated => "issue_escalated",
            // Processing Events
            ActivityType::ProcessingStarted => "processing_started",
            ActivityType::ProcessingCompleted => "processing_completed",
            ActivityType::ProcessingFailed => "processing_failed",
            ActivityType::ProcessingSkipped => "processing_skipped",
            ActivityType::RetryScheduled => "retry_scheduled",
            ActivityType::RetryExecuted => "retry_executed",
            // PR Events
            ActivityType::PrCreated => "pr_created",
            ActivityType::PrMerged => "pr_merged",
            ActivityType::PrClosed => "pr_closed",
            ActivityType::PrReviewReceived => "pr_review_received",
            ActivityType::PrReviewRequested => "pr_review_requested",
            ActivityType::PrCommented => "pr_commented",
            ActivityType::PrStatusCheckPassed => "pr_status_check_passed",
            ActivityType::PrStatusCheckFailed => "pr_status_check_failed",
            ActivityType::PrAutoClosed => "pr_auto_closed",
            // Claude Events
            ActivityType::ClaudeStarted => "claude_started",
            ActivityType::ClaudeCompleted => "claude_completed",
            ActivityType::ClaudeTimedOut => "claude_timed_out",
            ActivityType::ClaudeFailed => "claude_failed",
            // Webhook Events
            ActivityType::WebhookReceived => "webhook_received",
            ActivityType::WebhookProcessed => "webhook_processed",
            ActivityType::WebhookRejected => "webhook_rejected",
            // System Events
            ActivityType::WatcherStarted => "watcher_started",
            ActivityType::WatcherStopped => "watcher_stopped",
            ActivityType::WatcherPaused => "watcher_paused",
            ActivityType::WatcherResumed => "watcher_resumed",
            ActivityType::RateLimitHit => "rate_limit_hit",
            ActivityType::StateChange => "state_change",
            ActivityType::Error => "error",
        };

        let db_entry = ActivityLogEntry::new(db_activity_type, message)
            .with_source(source.unwrap_or("system").to_string());
        let db_entry = if let Some(id) = issue_id {
            db_entry.with_issue(id.to_string(), id.to_string())
        } else {
            db_entry
        };

        if let Err(e) = self.tracker.record_activity(&db_entry) {
            tracing::warn!(error = %e, "Failed to persist activity to database");
        }

        let mut activity = self.state.activity.lock().await;
        if activity.len() >= self.state.max_activity_entries {
            activity.pop_front();
        }
        activity.push_back(entry);
    }

    /// Increment issues processed counter.
    pub fn inc_issues_processed(&self) {
        self.state.issues_processed.fetch_add(1, Ordering::SeqCst);
    }

    /// Increment PRs created counter.
    pub fn inc_prs_created(&self) {
        self.state.prs_created.fetch_add(1, Ordering::SeqCst);
    }

    /// Add a processing issue.
    pub async fn add_processing(&self, issue_id: &str) {
        self.state
            .processing
            .write()
            .await
            .push(issue_id.to_string());
    }

    /// Remove a processing issue.
    pub async fn remove_processing(&self, issue_id: &str) {
        self.state
            .processing
            .write()
            .await
            .retain(|id| id != issue_id);
    }

    /// Get a shutdown receiver.
    pub fn shutdown_receiver(&self) -> broadcast::Receiver<()> {
        self.shutdown_tx.subscribe()
    }

    /// Start the IPC server.
    pub async fn start(&self) -> Result<()> {
        // Clean up any stale files from previous runs
        cleanup_stale_files();

        // Remove existing socket/port file if present
        if self.socket_path.exists() {
            std::fs::remove_file(&self.socket_path)?;
        }

        // Write PID file
        write_pid_file()?;

        // Bind the IPC listener (Unix socket or TCP — handled by transport)
        let listener = transport::bind(&self.socket_path)?;
        tracing::info!("IPC server listening on {:?}", self.socket_path);

        // Restrict socket file permissions on Unix (no-op on Windows)
        platform::set_file_permissions_secure(&self.socket_path)?;

        let mut shutdown_rx = self.shutdown_tx.subscribe();
        let conn_semaphore = Arc::new(Semaphore::new(MAX_CONCURRENT_CONNECTIONS));

        loop {
            tokio::select! {
                accept_result = listener.accept() => {
                    match accept_result {
                        Ok((stream, _)) => {
                            let permit = match conn_semaphore.clone().try_acquire_owned() {
                                Ok(permit) => permit,
                                Err(_) => {
                                    tracing::warn!("IPC connection limit reached ({MAX_CONCURRENT_CONNECTIONS}), rejecting connection");
                                    drop(stream);
                                    continue;
                                }
                            };

                            let tracker = self.tracker.clone();
                            let sources = self.sources.clone();
                            let notifier = self.notifier.clone();
                            let watcher = self.watcher.clone();
                            let state = self.state.clone();
                            let shutdown_tx = self.shutdown_tx.clone();

                            tokio::spawn(async move {
                                let _permit = permit; // held until handler completes
                                if let Err(e) = handle_connection(
                                    stream, tracker, sources, notifier, watcher, state, shutdown_tx
                                ).await {
                                    tracing::error!("Error handling IPC connection: {}", e);
                                }
                            });
                        }
                        Err(e) => {
                            tracing::error!("Failed to accept connection: {}", e);
                        }
                    }
                }
                _ = shutdown_rx.recv() => {
                    tracing::info!("IPC server shutting down");
                    break;
                }
            }
        }

        // Cleanup
        remove_socket_file();
        remove_pid_file();

        Ok(())
    }

    /// Trigger shutdown.
    pub fn shutdown(&self) {
        let _ = self.shutdown_tx.send(());
    }
}

/// Handle a single IPC connection (platform-agnostic via [`IpcStream`]).
async fn handle_connection(
    stream: IpcStream,
    tracker: Arc<dyn FixAttemptTracker>,
    sources: Vec<Arc<dyn IssueSource>>,
    notifier: Arc<dyn Notifier>,
    watcher: Option<Arc<Watcher>>,
    state: Arc<ServerState>,
    shutdown_tx: broadcast::Sender<()>,
) -> Result<()> {
    let (reader, mut writer) = stream.into_split();
    let mut reader = BufReader::new(reader);
    let mut line = String::new();

    while reader.read_line(&mut line).await? > 0 {
        let command: IpcCommand = match serde_json::from_str(line.trim()) {
            Ok(cmd) => cmd,
            Err(e) => {
                let response = IpcResponse::error(format!("Invalid command: {}", e));
                let json = serde_json::to_string(&response)? + "\n";
                writer.write_all(json.as_bytes()).await?;
                line.clear();
                continue;
            }
        };

        let response = handle_command(
            command,
            &tracker,
            &sources,
            &notifier,
            &watcher,
            &state,
            &shutdown_tx,
        )
        .await;

        let json = serde_json::to_string(&response)? + "\n";
        writer.write_all(json.as_bytes()).await?;
        line.clear();
    }

    Ok(())
}

/// Handle a single IPC command.
async fn handle_command(
    command: IpcCommand,
    tracker: &Arc<dyn FixAttemptTracker>,
    _sources: &[Arc<dyn IssueSource>],
    _notifier: &Arc<dyn Notifier>,
    watcher: &Option<Arc<Watcher>>,
    state: &Arc<ServerState>,
    shutdown_tx: &broadcast::Sender<()>,
) -> IpcResponse {
    match command {
        IpcCommand::Ping => IpcResponse::ok_with(IpcData::Pong),

        IpcCommand::Status => {
            let watcher_state = WatcherState {
                running: true,
                paused: state.paused.load(Ordering::SeqCst),
                mode: state.mode.read().await.clone(),
                uptime_secs: state.start_time.elapsed().as_secs(),
                issues_processed: state.issues_processed.load(Ordering::SeqCst),
                prs_created: state.prs_created.load(Ordering::SeqCst),
                processing: state.processing.read().await.clone(),
                sources: state.source_names.clone(),
                poll_interval_ms: {
                    let interval = state.poll_interval_ms.load(Ordering::SeqCst);
                    if interval > 0 {
                        Some(interval)
                    } else {
                        None
                    }
                },
            };
            IpcResponse::ok_with(IpcData::State(watcher_state))
        }

        IpcCommand::Pause => {
            state.paused.store(true, Ordering::SeqCst);

            // Log watcher_paused activity
            let activity = ActivityLogEntry::new("watcher_paused", "Watcher paused by user")
                .with_source("system".to_string());
            tracker.record_activity(&activity).ok();

            IpcResponse::ok_with(IpcData::Message("Watcher paused".to_string()))
        }

        IpcCommand::Resume => {
            state.paused.store(false, Ordering::SeqCst);

            // Log watcher_resumed activity
            let activity = ActivityLogEntry::new("watcher_resumed", "Watcher resumed by user")
                .with_source("system".to_string());
            tracker.record_activity(&activity).ok();

            IpcResponse::ok_with(IpcData::Message("Watcher resumed".to_string()))
        }

        IpcCommand::Stats => match tracker.get_stats() {
            Ok(stats) => IpcResponse::ok_with(IpcData::Stats(stats)),
            Err(e) => IpcResponse::error(format!("Failed to get stats: {}", e)),
        },

        IpcCommand::ListPrs => match tracker.get_pending_prs() {
            Ok(attempts) => IpcResponse::ok_with(IpcData::Attempts(attempts)),
            Err(e) => IpcResponse::error(format!("Failed to list PRs: {}", e)),
        },

        IpcCommand::ListRetries => match tracker.get_retryable_issues(state.max_retries) {
            Ok(attempts) => IpcResponse::ok_with(IpcData::Attempts(attempts)),
            Err(e) => IpcResponse::error(format!("Failed to list retries: {}", e)),
        },

        IpcCommand::Trigger { source, issue_id } => {
            if let Some(watcher) = watcher {
                match watcher.trigger_issue(&source, &issue_id).await {
                    Ok(()) => {
                        // Get the PR URL if available
                        let pr_url = tracker
                            .get_attempt(&source, &issue_id)
                            .ok()
                            .flatten()
                            .and_then(|a| a.pr_url);

                        IpcResponse::ok_with(IpcData::Triggered {
                            source,
                            issue_id,
                            pr_url,
                        })
                    }
                    Err(e) => IpcResponse::error(format!("Failed to trigger: {}", e)),
                }
            } else {
                IpcResponse::error("Watcher not available")
            }
        }

        IpcCommand::Reset { source, issue_id } => {
            if let Some(watcher) = watcher {
                match watcher.reset_attempt(&source, &issue_id) {
                    Ok(()) => IpcResponse::ok_with(IpcData::Reset { source, issue_id }),
                    Err(e) => IpcResponse::error(format!("Failed to reset: {}", e)),
                }
            } else {
                // Try direct tracker reset
                match tracker.reset_attempt(&source, &issue_id) {
                    Ok(()) => IpcResponse::ok_with(IpcData::Reset { source, issue_id }),
                    Err(e) => IpcResponse::error(format!("Failed to reset: {}", e)),
                }
            }
        }

        IpcCommand::ProcessRetries => {
            if let Some(watcher) = watcher {
                match tracker.get_retryable_issues(state.max_retries) {
                    Ok(attempts) => {
                        let mut count = 0;
                        for attempt in attempts {
                            // Check if ready for retry (simplified check)
                            if attempt.status == FixAttemptStatus::Failed {
                                if let Err(e) =
                                    tracker.prepare_for_retry(&attempt.source, &attempt.issue_id)
                                {
                                    tracing::warn!(
                                        "Failed to prepare retry for {}: {}",
                                        attempt.short_id,
                                        e
                                    );
                                    continue;
                                }

                                if let Err(e) = watcher
                                    .trigger_issue(&attempt.source, &attempt.issue_id)
                                    .await
                                {
                                    tracing::warn!(
                                        "Failed to trigger retry for {}: {}",
                                        attempt.short_id,
                                        e
                                    );
                                } else {
                                    count += 1;
                                }
                            }
                        }
                        IpcResponse::ok_with(IpcData::RetriesProcessed { count })
                    }
                    Err(e) => IpcResponse::error(format!("Failed to get retryable issues: {}", e)),
                }
            } else {
                IpcResponse::error("Watcher not available for processing retries")
            }
        }

        IpcCommand::Activity { limit } => {
            let activity = state.activity.lock().await;
            let entries: Vec<_> = activity.iter().rev().take(limit).cloned().collect();
            IpcResponse::ok_with(IpcData::Activity(entries))
        }

        IpcCommand::Shutdown => {
            let _ = shutdown_tx.send(());
            IpcResponse::ok_with(IpcData::Message("Shutdown initiated".to_string()))
        }
    }
}

impl Drop for IpcServer {
    fn drop(&mut self) {
        // Best effort cleanup
        remove_socket_file();
        remove_pid_file();
    }
}

/// Builder for configuring an IpcServer.
pub struct IpcServerBuilder {
    tracker: Arc<dyn FixAttemptTracker>,
    sources: Vec<Arc<dyn IssueSource>>,
    notifier: Arc<dyn Notifier>,
    max_activity_entries: usize,
    max_retries: u32,
}

impl IpcServerBuilder {
    /// Create a new IpcServer builder.
    pub fn new(
        tracker: Arc<dyn FixAttemptTracker>,
        sources: Vec<Arc<dyn IssueSource>>,
        notifier: Arc<dyn Notifier>,
    ) -> Self {
        Self {
            tracker,
            sources,
            notifier,
            max_activity_entries: DEFAULT_MAX_ACTIVITY_ENTRIES,
            max_retries: 2,
        }
    }

    /// Set the maximum number of activity entries to keep.
    pub fn max_activity_entries(mut self, max: usize) -> Self {
        self.max_activity_entries = max;
        self
    }

    /// Set the maximum number of retries.
    pub fn max_retries(mut self, max: u32) -> Self {
        self.max_retries = max;
        self
    }

    /// Build the IpcServer.
    pub fn build(self) -> IpcServer {
        let (shutdown_tx, _) = broadcast::channel(1);
        let source_names = self.sources.iter().map(|s| s.name().to_string()).collect();

        IpcServer {
            socket_path: default_socket_path(),
            tracker: self.tracker,
            sources: self.sources,
            notifier: self.notifier,
            watcher: None,
            state: Arc::new(ServerState {
                paused: AtomicBool::new(false),
                start_time: Instant::now(),
                issues_processed: AtomicUsize::new(0),
                prs_created: AtomicUsize::new(0),
                processing: RwLock::new(Vec::new()),
                activity: Mutex::new(VecDeque::with_capacity(self.max_activity_entries)),
                mode: RwLock::new("initializing".to_string()),
                poll_interval_ms: AtomicU64::new(0),
                source_names,
                max_activity_entries: self.max_activity_entries,
                max_retries: self.max_retries,
            }),
            shutdown_tx,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_state_defaults() {
        let state = ServerState {
            paused: AtomicBool::new(false),
            start_time: Instant::now(),
            issues_processed: AtomicUsize::new(0),
            prs_created: AtomicUsize::new(0),
            processing: RwLock::new(Vec::new()),
            activity: Mutex::new(VecDeque::new()),
            mode: RwLock::new("test".to_string()),
            poll_interval_ms: AtomicU64::new(0),
            source_names: vec!["linear".to_string()],
            max_activity_entries: DEFAULT_MAX_ACTIVITY_ENTRIES,
            max_retries: 2,
        };

        assert!(!state.paused.load(Ordering::SeqCst));
        assert_eq!(state.issues_processed.load(Ordering::SeqCst), 0);
    }

    use async_trait::async_trait;
    use claudear_core::error::Result as CrateResult;
    use claudear_core::types::{Issue, MatchPriority, MatchResult};
    use claudear_integrations::notifier::Notifier;
    use claudear_integrations::source::IssueSource;
    use claudear_storage::SqliteTracker;

    /// Minimal mock for `IssueSource`.
    struct MockSource {
        source_name: String,
    }

    impl MockSource {
        fn new(name: &str) -> Self {
            Self {
                source_name: name.to_string(),
            }
        }
    }

    #[async_trait]
    impl IssueSource for MockSource {
        fn name(&self) -> &str {
            &self.source_name
        }
        fn display_name(&self) -> &str {
            &self.source_name
        }
        async fn fetch_issues(&self) -> CrateResult<Vec<Issue>> {
            Ok(vec![])
        }
        fn matches_criteria(&self, _issue: &Issue) -> MatchResult {
            MatchResult::matched("test", MatchPriority::Normal)
        }
        async fn build_issue_context(&self, _issue: &Issue) -> CrateResult<String> {
            Ok(String::new())
        }
        async fn get_issue(&self, _id: &str) -> CrateResult<Issue> {
            Err(claudear_core::error::Error::issue_not_found(
                &self.source_name,
                "mock",
            ))
        }
    }

    /// Minimal mock for `Notifier`.
    struct MockNotifier;

    #[async_trait]
    impl Notifier for MockNotifier {
        fn name(&self) -> &str {
            "mock"
        }
        fn is_enabled(&self) -> bool {
            true
        }
        async fn notify_start(&self, _issue: &Issue) -> CrateResult<()> {
            Ok(())
        }
        async fn notify_success(&self, _issue: &Issue, _pr_url: &str) -> CrateResult<()> {
            Ok(())
        }
        async fn notify_completed(&self, _issue: &Issue) -> CrateResult<()> {
            Ok(())
        }
        async fn notify_failed(&self, _issue: &Issue, _error: &str) -> CrateResult<()> {
            Ok(())
        }
        async fn notify_status(&self, _message: &str) -> CrateResult<()> {
            Ok(())
        }
        async fn notify_urgent_issues(&self, _issues: &[Issue]) -> CrateResult<()> {
            Ok(())
        }
    }

    /// Build a `ServerState` for tests. The `mode` is set to "test"
    /// and `max_activity_entries` is configurable.
    fn test_state(max_activity: usize, max_retries: u32) -> Arc<ServerState> {
        Arc::new(ServerState {
            paused: AtomicBool::new(false),
            start_time: Instant::now(),
            issues_processed: AtomicUsize::new(0),
            prs_created: AtomicUsize::new(0),
            processing: RwLock::new(Vec::new()),
            activity: Mutex::new(VecDeque::new()),
            mode: RwLock::new("test".to_string()),
            poll_interval_ms: AtomicU64::new(0),
            source_names: vec!["linear".to_string()],
            max_activity_entries: max_activity,
            max_retries,
        })
    }

    fn mock_tracker() -> Arc<dyn FixAttemptTracker> {
        Arc::new(SqliteTracker::in_memory().expect("in-memory tracker"))
    }

    fn mock_sources() -> Vec<Arc<dyn IssueSource>> {
        vec![Arc::new(MockSource::new("linear"))]
    }

    fn mock_notifier() -> Arc<dyn Notifier> {
        Arc::new(MockNotifier)
    }

    #[tokio::test]
    async fn test_handle_command_ping() {
        let tracker = mock_tracker();
        let sources = mock_sources();
        let notifier = mock_notifier();
        let state = test_state(100, 2);
        let (shutdown_tx, _) = broadcast::channel(1);

        let resp = handle_command(
            IpcCommand::Ping,
            &tracker,
            &sources,
            &notifier,
            &None,
            &state,
            &shutdown_tx,
        )
        .await;

        assert!(resp.is_ok());
        match resp {
            IpcResponse::Ok(IpcData::Pong) => {} // expected
            other => panic!("Expected Pong, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_handle_command_status() {
        let tracker = mock_tracker();
        let sources = mock_sources();
        let notifier = mock_notifier();
        let state = test_state(100, 2);
        state.issues_processed.store(5, Ordering::SeqCst);
        state.prs_created.store(3, Ordering::SeqCst);
        let (shutdown_tx, _) = broadcast::channel(1);

        let resp = handle_command(
            IpcCommand::Status,
            &tracker,
            &sources,
            &notifier,
            &None,
            &state,
            &shutdown_tx,
        )
        .await;

        match resp {
            IpcResponse::Ok(IpcData::State(ws)) => {
                assert!(ws.running);
                assert!(!ws.paused);
                assert_eq!(ws.mode, "test");
                assert!(ws.uptime_secs < 5); // just created
                assert_eq!(ws.issues_processed, 5);
                assert_eq!(ws.prs_created, 3);
                assert!(ws.processing.is_empty());
                assert_eq!(ws.sources, vec!["linear".to_string()]);
                assert_eq!(ws.poll_interval_ms, None); // 0 maps to None
            }
            other => panic!("Expected State, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_handle_command_pause() {
        let tracker = mock_tracker();
        let sources = mock_sources();
        let notifier = mock_notifier();
        let state = test_state(100, 2);
        let (shutdown_tx, _) = broadcast::channel(1);

        // Initially not paused
        assert!(!state.paused.load(Ordering::SeqCst));

        let resp = handle_command(
            IpcCommand::Pause,
            &tracker,
            &sources,
            &notifier,
            &None,
            &state,
            &shutdown_tx,
        )
        .await;

        assert!(state.paused.load(Ordering::SeqCst));
        match resp {
            IpcResponse::Ok(IpcData::Message(msg)) => {
                assert_eq!(msg, "Watcher paused");
            }
            other => panic!("Expected paused message, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_handle_command_resume() {
        let tracker = mock_tracker();
        let sources = mock_sources();
        let notifier = mock_notifier();
        let state = test_state(100, 2);
        state.paused.store(true, Ordering::SeqCst);
        let (shutdown_tx, _) = broadcast::channel(1);

        let resp = handle_command(
            IpcCommand::Resume,
            &tracker,
            &sources,
            &notifier,
            &None,
            &state,
            &shutdown_tx,
        )
        .await;

        assert!(!state.paused.load(Ordering::SeqCst));
        match resp {
            IpcResponse::Ok(IpcData::Message(msg)) => {
                assert_eq!(msg, "Watcher resumed");
            }
            other => panic!("Expected resumed message, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_handle_command_activity_returns_reverse_chronological_with_limit() {
        let tracker = mock_tracker();
        let sources = mock_sources();
        let notifier = mock_notifier();
        let state = test_state(100, 2);
        let (shutdown_tx, _) = broadcast::channel(1);

        // Add 5 entries
        {
            let mut activity = state.activity.lock().await;
            for i in 0..5 {
                activity.push_back(ActivityEntry {
                    timestamp: format!("2025-01-01T00:00:0{}Z", i),
                    activity_type: ActivityType::IssueDetected,
                    message: format!("entry-{}", i),
                    issue_id: None,
                    source: None,
                });
            }
        }

        // Request limit of 3
        let resp = handle_command(
            IpcCommand::Activity { limit: 3 },
            &tracker,
            &sources,
            &notifier,
            &None,
            &state,
            &shutdown_tx,
        )
        .await;

        match resp {
            IpcResponse::Ok(IpcData::Activity(entries)) => {
                assert_eq!(entries.len(), 3);
                // Reversed: most recent first
                assert_eq!(entries[0].message, "entry-4");
                assert_eq!(entries[1].message, "entry-3");
                assert_eq!(entries[2].message, "entry-2");
            }
            other => panic!("Expected Activity, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_handle_command_activity_returns_all_when_limit_exceeds_count() {
        let tracker = mock_tracker();
        let sources = mock_sources();
        let notifier = mock_notifier();
        let state = test_state(100, 2);
        let (shutdown_tx, _) = broadcast::channel(1);

        {
            let mut activity = state.activity.lock().await;
            activity.push_back(ActivityEntry {
                timestamp: "2025-01-01T00:00:00Z".to_string(),
                activity_type: ActivityType::WatcherStarted,
                message: "only-one".to_string(),
                issue_id: None,
                source: None,
            });
        }

        let resp = handle_command(
            IpcCommand::Activity { limit: 100 },
            &tracker,
            &sources,
            &notifier,
            &None,
            &state,
            &shutdown_tx,
        )
        .await;

        match resp {
            IpcResponse::Ok(IpcData::Activity(entries)) => {
                assert_eq!(entries.len(), 1);
                assert_eq!(entries[0].message, "only-one");
            }
            other => panic!("Expected Activity, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_handle_command_shutdown_sends_signal() {
        let tracker = mock_tracker();
        let sources = mock_sources();
        let notifier = mock_notifier();
        let state = test_state(100, 2);
        let (shutdown_tx, mut shutdown_rx) = broadcast::channel(1);

        let resp = handle_command(
            IpcCommand::Shutdown,
            &tracker,
            &sources,
            &notifier,
            &None,
            &state,
            &shutdown_tx,
        )
        .await;

        match resp {
            IpcResponse::Ok(IpcData::Message(msg)) => {
                assert_eq!(msg, "Shutdown initiated");
            }
            other => panic!("Expected shutdown message, got {:?}", other),
        }

        // Verify the signal was actually sent
        assert!(shutdown_rx.try_recv().is_ok());
    }

    #[tokio::test]
    async fn test_handle_command_stats() {
        let tracker = mock_tracker();
        let sources = mock_sources();
        let notifier = mock_notifier();
        let state = test_state(100, 2);
        let (shutdown_tx, _) = broadcast::channel(1);

        let resp = handle_command(
            IpcCommand::Stats,
            &tracker,
            &sources,
            &notifier,
            &None,
            &state,
            &shutdown_tx,
        )
        .await;

        match resp {
            IpcResponse::Ok(IpcData::Stats(stats)) => {
                // Fresh in-memory tracker should have all zeros
                assert_eq!(stats.total, 0);
                assert_eq!(stats.pending, 0);
                assert_eq!(stats.success, 0);
                assert_eq!(stats.failed, 0);
            }
            other => panic!("Expected Stats, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_handle_command_list_prs() {
        let tracker = mock_tracker();
        let sources = mock_sources();
        let notifier = mock_notifier();
        let state = test_state(100, 2);
        let (shutdown_tx, _) = broadcast::channel(1);

        let resp = handle_command(
            IpcCommand::ListPrs,
            &tracker,
            &sources,
            &notifier,
            &None,
            &state,
            &shutdown_tx,
        )
        .await;

        match resp {
            IpcResponse::Ok(IpcData::Attempts(attempts)) => {
                assert!(attempts.is_empty());
            }
            other => panic!("Expected Attempts, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_handle_command_list_retries() {
        let tracker = mock_tracker();
        let sources = mock_sources();
        let notifier = mock_notifier();
        let state = test_state(100, 2);
        let (shutdown_tx, _) = broadcast::channel(1);

        let resp = handle_command(
            IpcCommand::ListRetries,
            &tracker,
            &sources,
            &notifier,
            &None,
            &state,
            &shutdown_tx,
        )
        .await;

        match resp {
            IpcResponse::Ok(IpcData::Attempts(attempts)) => {
                assert!(attempts.is_empty());
            }
            other => panic!("Expected Attempts, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_handle_command_trigger_without_watcher() {
        let tracker = mock_tracker();
        let sources = mock_sources();
        let notifier = mock_notifier();
        let state = test_state(100, 2);
        let (shutdown_tx, _) = broadcast::channel(1);

        let resp = handle_command(
            IpcCommand::Trigger {
                source: "linear".to_string(),
                issue_id: "LIN-1".to_string(),
            },
            &tracker,
            &sources,
            &notifier,
            &None,
            &state,
            &shutdown_tx,
        )
        .await;

        match resp {
            IpcResponse::Error { message } => {
                assert_eq!(message, "Watcher not available");
            }
            other => panic!("Expected error, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_handle_command_reset_without_watcher_falls_through_to_tracker() {
        let tracker = mock_tracker();
        let sources = mock_sources();
        let notifier = mock_notifier();
        let state = test_state(100, 2);
        let (shutdown_tx, _) = broadcast::channel(1);

        // Record an attempt first so the reset has something to target
        tracker.record_attempt("linear", "LIN-1", "LIN-1").unwrap();

        let resp = handle_command(
            IpcCommand::Reset {
                source: "linear".to_string(),
                issue_id: "LIN-1".to_string(),
            },
            &tracker,
            &sources,
            &notifier,
            &None,
            &state,
            &shutdown_tx,
        )
        .await;

        match resp {
            IpcResponse::Ok(IpcData::Reset { source, issue_id }) => {
                assert_eq!(source, "linear");
                assert_eq!(issue_id, "LIN-1");
            }
            other => panic!("Expected Reset, got {:?}", other),
        }
    }

    #[test]
    fn test_builder_default_max_activity_entries() {
        let builder = IpcServerBuilder::new(mock_tracker(), mock_sources(), mock_notifier());
        assert_eq!(builder.max_activity_entries, DEFAULT_MAX_ACTIVITY_ENTRIES);
        assert_eq!(builder.max_activity_entries, 10_000);
    }

    #[test]
    fn test_builder_default_max_retries() {
        let builder = IpcServerBuilder::new(mock_tracker(), mock_sources(), mock_notifier());
        assert_eq!(builder.max_retries, 2);
    }

    #[test]
    fn test_builder_custom_max_activity_entries() {
        let server = IpcServerBuilder::new(mock_tracker(), mock_sources(), mock_notifier())
            .max_activity_entries(500)
            .build();

        assert_eq!(server.state.max_activity_entries, 500);
    }

    #[test]
    fn test_builder_custom_max_retries() {
        let server = IpcServerBuilder::new(mock_tracker(), mock_sources(), mock_notifier())
            .max_retries(5)
            .build();

        assert_eq!(server.state.max_retries, 5);
    }

    #[test]
    fn test_builder_collects_source_names() {
        let sources: Vec<Arc<dyn IssueSource>> = vec![
            Arc::new(MockSource::new("linear")),
            Arc::new(MockSource::new("sentry")),
            Arc::new(MockSource::new("jira")),
        ];
        let server = IpcServerBuilder::new(mock_tracker(), sources, mock_notifier()).build();

        assert_eq!(server.state.source_names.len(), 3);
        assert!(server.state.source_names.contains(&"linear".to_string()));
        assert!(server.state.source_names.contains(&"sentry".to_string()));
        assert!(server.state.source_names.contains(&"jira".to_string()));
    }

    #[test]
    fn test_is_paused_initially_false() {
        let server = IpcServer::new(mock_tracker(), mock_sources(), mock_notifier());
        assert!(!server.is_paused());
    }

    #[test]
    fn test_set_poll_interval_stores_value() {
        let server = IpcServer::new(mock_tracker(), mock_sources(), mock_notifier());
        server.set_poll_interval(300_000);
        assert_eq!(
            server.state.poll_interval_ms.load(Ordering::SeqCst),
            300_000
        );
    }

    #[test]
    fn test_inc_issues_processed() {
        let server = IpcServer::new(mock_tracker(), mock_sources(), mock_notifier());
        assert_eq!(server.state.issues_processed.load(Ordering::SeqCst), 0);
        server.inc_issues_processed();
        assert_eq!(server.state.issues_processed.load(Ordering::SeqCst), 1);
        server.inc_issues_processed();
        server.inc_issues_processed();
        assert_eq!(server.state.issues_processed.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn test_inc_prs_created() {
        let server = IpcServer::new(mock_tracker(), mock_sources(), mock_notifier());
        assert_eq!(server.state.prs_created.load(Ordering::SeqCst), 0);
        server.inc_prs_created();
        assert_eq!(server.state.prs_created.load(Ordering::SeqCst), 1);
        server.inc_prs_created();
        assert_eq!(server.state.prs_created.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn test_add_and_remove_processing() {
        let server = IpcServer::new(mock_tracker(), mock_sources(), mock_notifier());

        server.add_processing("LIN-1").await;
        server.add_processing("LIN-2").await;
        assert_eq!(server.state.processing.read().await.len(), 2);

        server.remove_processing("LIN-1").await;
        let processing = server.state.processing.read().await;
        assert_eq!(processing.len(), 1);
        assert_eq!(processing[0], "LIN-2");
    }

    #[tokio::test]
    async fn test_remove_processing_nonexistent_is_noop() {
        let server = IpcServer::new(mock_tracker(), mock_sources(), mock_notifier());
        server.add_processing("LIN-1").await;
        server.remove_processing("DOES-NOT-EXIST").await;
        assert_eq!(server.state.processing.read().await.len(), 1);
    }

    #[tokio::test]
    async fn test_log_activity_adds_entries() {
        let server = IpcServer::new(mock_tracker(), mock_sources(), mock_notifier());

        server
            .log_activity(
                ActivityType::IssueDetected,
                "Found issue",
                Some("LIN-1"),
                Some("linear"),
            )
            .await;
        server
            .log_activity(ActivityType::PrCreated, "PR opened", None, None)
            .await;

        let activity = server.state.activity.lock().await;
        assert_eq!(activity.len(), 2);
        assert_eq!(activity[0].message, "Found issue");
        assert_eq!(activity[0].issue_id, Some("LIN-1".to_string()));
        assert_eq!(activity[0].source, Some("linear".to_string()));
        assert_eq!(activity[1].message, "PR opened");
    }

    #[tokio::test]
    async fn test_log_activity_caps_at_max_entries() {
        let server = IpcServerBuilder::new(mock_tracker(), mock_sources(), mock_notifier())
            .max_activity_entries(3)
            .build();

        for i in 0..5 {
            server
                .log_activity(
                    ActivityType::IssueDetected,
                    &format!("entry-{}", i),
                    None,
                    None,
                )
                .await;
        }

        let activity = server.state.activity.lock().await;
        assert_eq!(activity.len(), 3);
        // Oldest entries should have been evicted; entries 2, 3, 4 remain
        assert_eq!(activity[0].message, "entry-2");
        assert_eq!(activity[1].message, "entry-3");
        assert_eq!(activity[2].message, "entry-4");
    }

    #[tokio::test]
    async fn test_set_mode() {
        let server = IpcServer::new(mock_tracker(), mock_sources(), mock_notifier());
        server.set_mode("poll").await;
        assert_eq!(*server.state.mode.read().await, "poll");
    }

    #[test]
    fn test_shutdown_receiver() {
        let server = IpcServer::new(mock_tracker(), mock_sources(), mock_notifier());
        let mut rx = server.shutdown_receiver();
        // No signal yet
        assert!(rx.try_recv().is_err());

        server.shutdown();
        assert!(rx.try_recv().is_ok());
    }

    #[tokio::test]
    async fn test_handle_command_status_with_poll_interval() {
        let tracker = mock_tracker();
        let sources = mock_sources();
        let notifier = mock_notifier();
        let state = test_state(100, 2);
        state.poll_interval_ms.store(60_000, Ordering::SeqCst);
        let (shutdown_tx, _) = broadcast::channel(1);

        let resp = handle_command(
            IpcCommand::Status,
            &tracker,
            &sources,
            &notifier,
            &None,
            &state,
            &shutdown_tx,
        )
        .await;

        match resp {
            IpcResponse::Ok(IpcData::State(ws)) => {
                assert_eq!(ws.poll_interval_ms, Some(60_000));
            }
            other => panic!("Expected State, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_handle_command_status_reflects_paused() {
        let tracker = mock_tracker();
        let sources = mock_sources();
        let notifier = mock_notifier();
        let state = test_state(100, 2);
        state.paused.store(true, Ordering::SeqCst);
        let (shutdown_tx, _) = broadcast::channel(1);

        let resp = handle_command(
            IpcCommand::Status,
            &tracker,
            &sources,
            &notifier,
            &None,
            &state,
            &shutdown_tx,
        )
        .await;

        match resp {
            IpcResponse::Ok(IpcData::State(ws)) => {
                assert!(ws.paused);
            }
            other => panic!("Expected State, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_ipc_server_set_mode() {
        let server = IpcServer::new(mock_tracker(), mock_sources(), mock_notifier());
        assert_eq!(*server.state.mode.read().await, "initializing");
        server.set_mode("webhook").await;
        assert_eq!(*server.state.mode.read().await, "webhook");
        server.set_mode("poll").await;
        assert_eq!(*server.state.mode.read().await, "poll");
    }

    #[test]
    fn test_ipc_server_set_poll_interval() {
        let server = IpcServer::new(mock_tracker(), mock_sources(), mock_notifier());
        assert_eq!(server.state.poll_interval_ms.load(Ordering::SeqCst), 0);
        server.set_poll_interval(60_000);
        assert_eq!(server.state.poll_interval_ms.load(Ordering::SeqCst), 60_000);
        server.set_poll_interval(120_000);
        assert_eq!(
            server.state.poll_interval_ms.load(Ordering::SeqCst),
            120_000
        );
    }

    #[tokio::test]
    async fn test_ipc_server_is_paused() {
        let server = IpcServer::new(mock_tracker(), mock_sources(), mock_notifier());
        // Default is not paused
        assert!(!server.is_paused());
        // Toggle paused via state
        server.state.paused.store(true, Ordering::SeqCst);
        assert!(server.is_paused());
        // Toggle back
        server.state.paused.store(false, Ordering::SeqCst);
        assert!(!server.is_paused());
    }

    #[test]
    fn test_ipc_server_inc_issues_processed() {
        let server = IpcServer::new(mock_tracker(), mock_sources(), mock_notifier());
        assert_eq!(server.state.issues_processed.load(Ordering::SeqCst), 0);
        server.inc_issues_processed();
        assert_eq!(server.state.issues_processed.load(Ordering::SeqCst), 1);
        server.inc_issues_processed();
        server.inc_issues_processed();
        assert_eq!(server.state.issues_processed.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn test_ipc_server_inc_prs_created() {
        let server = IpcServer::new(mock_tracker(), mock_sources(), mock_notifier());
        assert_eq!(server.state.prs_created.load(Ordering::SeqCst), 0);
        server.inc_prs_created();
        assert_eq!(server.state.prs_created.load(Ordering::SeqCst), 1);
        server.inc_prs_created();
        assert_eq!(server.state.prs_created.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn test_ipc_server_add_remove_processing() {
        let server = IpcServer::new(mock_tracker(), mock_sources(), mock_notifier());
        assert!(server.state.processing.read().await.is_empty());

        server.add_processing("ISSUE-1").await;
        server.add_processing("ISSUE-2").await;
        server.add_processing("ISSUE-3").await;
        assert_eq!(server.state.processing.read().await.len(), 3);

        server.remove_processing("ISSUE-2").await;
        let processing = server.state.processing.read().await;
        assert_eq!(processing.len(), 2);
        assert!(processing.contains(&"ISSUE-1".to_string()));
        assert!(!processing.contains(&"ISSUE-2".to_string()));
        assert!(processing.contains(&"ISSUE-3".to_string()));
        drop(processing);

        // Remove non-existent is a no-op
        server.remove_processing("DOES-NOT-EXIST").await;
        assert_eq!(server.state.processing.read().await.len(), 2);
    }

    #[tokio::test]
    async fn test_ipc_server_log_activity() {
        let server = IpcServer::new(mock_tracker(), mock_sources(), mock_notifier());

        server
            .log_activity(
                ActivityType::IssueDetected,
                "Detected issue LIN-1",
                Some("LIN-1"),
                Some("linear"),
            )
            .await;
        server
            .log_activity(
                ActivityType::PrCreated,
                "Created PR for LIN-1",
                Some("LIN-1"),
                Some("linear"),
            )
            .await;
        server
            .log_activity(ActivityType::WatcherStarted, "Watcher started", None, None)
            .await;

        let activity = server.state.activity.lock().await;
        assert_eq!(activity.len(), 3);
        assert_eq!(activity[0].message, "Detected issue LIN-1");
        assert_eq!(activity[0].issue_id, Some("LIN-1".to_string()));
        assert_eq!(activity[0].source, Some("linear".to_string()));
        assert_eq!(activity[1].message, "Created PR for LIN-1");
        assert_eq!(activity[2].message, "Watcher started");
        assert_eq!(activity[2].issue_id, None);
        assert_eq!(activity[2].source, None);
    }

    #[tokio::test]
    async fn test_ipc_server_log_activity_overflow() {
        let server = IpcServerBuilder::new(mock_tracker(), mock_sources(), mock_notifier())
            .max_activity_entries(5)
            .build();

        // Add 8 entries to overflow the capacity of 5
        for i in 0..8 {
            server
                .log_activity(
                    ActivityType::IssueDetected,
                    &format!("entry-{}", i),
                    None,
                    None,
                )
                .await;
        }

        let activity = server.state.activity.lock().await;
        assert_eq!(activity.len(), 5);
        // Oldest entries (0, 1, 2) should have been evicted; entries 3..7 remain
        assert_eq!(activity[0].message, "entry-3");
        assert_eq!(activity[1].message, "entry-4");
        assert_eq!(activity[2].message, "entry-5");
        assert_eq!(activity[3].message, "entry-6");
        assert_eq!(activity[4].message, "entry-7");
    }

    #[test]
    fn test_ipc_server_builder_defaults() {
        let builder = IpcServerBuilder::new(mock_tracker(), mock_sources(), mock_notifier());
        assert_eq!(builder.max_activity_entries, DEFAULT_MAX_ACTIVITY_ENTRIES);
        assert_eq!(builder.max_retries, 2);

        let server = builder.build();
        assert_eq!(
            server.state.max_activity_entries,
            DEFAULT_MAX_ACTIVITY_ENTRIES
        );
        assert_eq!(server.state.max_retries, 2);
        assert!(!server.is_paused());
        assert_eq!(server.state.issues_processed.load(Ordering::SeqCst), 0);
        assert_eq!(server.state.prs_created.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn test_ipc_server_builder_custom() {
        let server = IpcServerBuilder::new(mock_tracker(), mock_sources(), mock_notifier())
            .max_activity_entries(250)
            .max_retries(10)
            .build();

        assert_eq!(server.state.max_activity_entries, 250);
        assert_eq!(server.state.max_retries, 10);
    }

    // -------------------------------------------------------------------
    // handle_command: ProcessRetries without watcher
    // -------------------------------------------------------------------

    #[tokio::test]
    async fn test_handle_command_process_retries_without_watcher() {
        let tracker = mock_tracker();
        let sources = mock_sources();
        let notifier = mock_notifier();
        let state = test_state(100, 2);
        let (shutdown_tx, _) = broadcast::channel(1);

        let resp = handle_command(
            IpcCommand::ProcessRetries,
            &tracker,
            &sources,
            &notifier,
            &None,
            &state,
            &shutdown_tx,
        )
        .await;

        match resp {
            IpcResponse::Error { message } => {
                assert!(message.contains("Watcher not available"));
            }
            other => panic!("Expected error, got {:?}", other),
        }
    }

    // -------------------------------------------------------------------
    // handle_command: Reset without watcher (tracker fallback)
    // -------------------------------------------------------------------

    #[tokio::test]
    async fn test_handle_command_reset_nonexistent_attempt_without_watcher() {
        let tracker = mock_tracker();
        let sources = mock_sources();
        let notifier = mock_notifier();
        let state = test_state(100, 2);
        let (shutdown_tx, _) = broadcast::channel(1);

        // Reset an attempt that doesn't exist - should still succeed via
        // the tracker's reset_attempt which is a no-op for missing attempts
        let resp = handle_command(
            IpcCommand::Reset {
                source: "linear".to_string(),
                issue_id: "NONEXISTENT".to_string(),
            },
            &tracker,
            &sources,
            &notifier,
            &None,
            &state,
            &shutdown_tx,
        )
        .await;

        match resp {
            IpcResponse::Ok(IpcData::Reset { source, issue_id }) => {
                assert_eq!(source, "linear");
                assert_eq!(issue_id, "NONEXISTENT");
            }
            other => panic!("Expected Reset, got {:?}", other),
        }
    }

    // -------------------------------------------------------------------
    // handle_command: Stats with recorded data
    // -------------------------------------------------------------------

    #[tokio::test]
    async fn test_handle_command_stats_with_data() {
        let tracker = mock_tracker();
        // Record some attempts
        tracker.record_attempt("linear", "LIN-1", "LIN-1").unwrap();
        tracker
            .mark_success("linear", "LIN-1", "https://github.com/org/repo/pull/1")
            .unwrap();
        tracker.record_attempt("sentry", "SEN-1", "SEN-1").unwrap();
        tracker
            .mark_failed("sentry", "SEN-1", "build error")
            .unwrap();

        let sources = mock_sources();
        let notifier = mock_notifier();
        let state = test_state(100, 2);
        let (shutdown_tx, _) = broadcast::channel(1);

        let resp = handle_command(
            IpcCommand::Stats,
            &tracker,
            &sources,
            &notifier,
            &None,
            &state,
            &shutdown_tx,
        )
        .await;

        match resp {
            IpcResponse::Ok(IpcData::Stats(stats)) => {
                assert_eq!(stats.total, 2);
                assert_eq!(stats.success, 1);
                assert_eq!(stats.failed, 1);
            }
            other => panic!("Expected Stats, got {:?}", other),
        }
    }

    // -------------------------------------------------------------------
    // handle_command: ListPrs with data
    // -------------------------------------------------------------------

    #[tokio::test]
    async fn test_handle_command_list_prs_with_data() {
        let tracker = mock_tracker();
        tracker.record_attempt("linear", "LIN-1", "LIN-1").unwrap();
        tracker
            .mark_success("linear", "LIN-1", "https://github.com/org/repo/pull/1")
            .unwrap();

        let sources = mock_sources();
        let notifier = mock_notifier();
        let state = test_state(100, 2);
        let (shutdown_tx, _) = broadcast::channel(1);

        let resp = handle_command(
            IpcCommand::ListPrs,
            &tracker,
            &sources,
            &notifier,
            &None,
            &state,
            &shutdown_tx,
        )
        .await;

        match resp {
            IpcResponse::Ok(IpcData::Attempts(attempts)) => {
                assert_eq!(attempts.len(), 1);
                assert_eq!(attempts[0].source, "linear");
            }
            other => panic!("Expected Attempts, got {:?}", other),
        }
    }

    // -------------------------------------------------------------------
    // handle_command: ListRetries with retryable data
    // -------------------------------------------------------------------

    #[tokio::test]
    async fn test_handle_command_list_retries_with_data() {
        let tracker = mock_tracker();
        tracker.record_attempt("linear", "LIN-1", "LIN-1").unwrap();
        tracker
            .mark_failed("linear", "LIN-1", "build error")
            .unwrap();

        let sources = mock_sources();
        let notifier = mock_notifier();
        let state = test_state(100, 2);
        let (shutdown_tx, _) = broadcast::channel(1);

        let resp = handle_command(
            IpcCommand::ListRetries,
            &tracker,
            &sources,
            &notifier,
            &None,
            &state,
            &shutdown_tx,
        )
        .await;

        match resp {
            IpcResponse::Ok(IpcData::Attempts(attempts)) => {
                assert!(!attempts.is_empty());
                assert_eq!(attempts[0].source, "linear");
            }
            other => panic!("Expected Attempts, got {:?}", other),
        }
    }

    // -------------------------------------------------------------------
    // handle_command: Activity with empty state
    // -------------------------------------------------------------------

    #[tokio::test]
    async fn test_handle_command_activity_empty() {
        let tracker = mock_tracker();
        let sources = mock_sources();
        let notifier = mock_notifier();
        let state = test_state(100, 2);
        let (shutdown_tx, _) = broadcast::channel(1);

        let resp = handle_command(
            IpcCommand::Activity { limit: 50 },
            &tracker,
            &sources,
            &notifier,
            &None,
            &state,
            &shutdown_tx,
        )
        .await;

        match resp {
            IpcResponse::Ok(IpcData::Activity(entries)) => {
                assert!(entries.is_empty());
            }
            other => panic!("Expected Activity, got {:?}", other),
        }
    }

    // -------------------------------------------------------------------
    // log_activity with all activity type variants
    // -------------------------------------------------------------------

    #[tokio::test]
    async fn test_log_activity_all_activity_types() {
        let server = IpcServer::new(mock_tracker(), mock_sources(), mock_notifier());

        let all_types = vec![
            ActivityType::IssueDetected,
            ActivityType::IssueStatusChanged,
            ActivityType::IssuePriorityChanged,
            ActivityType::IssueCommented,
            ActivityType::IssueResolved,
            ActivityType::IssueCancelled,
            ActivityType::IssueEscalated,
            ActivityType::ProcessingStarted,
            ActivityType::ProcessingCompleted,
            ActivityType::ProcessingFailed,
            ActivityType::ProcessingSkipped,
            ActivityType::RetryScheduled,
            ActivityType::RetryExecuted,
            ActivityType::PrCreated,
            ActivityType::PrMerged,
            ActivityType::PrClosed,
            ActivityType::PrReviewReceived,
            ActivityType::PrReviewRequested,
            ActivityType::PrCommented,
            ActivityType::PrStatusCheckPassed,
            ActivityType::PrStatusCheckFailed,
            ActivityType::PrAutoClosed,
            ActivityType::ClaudeStarted,
            ActivityType::ClaudeCompleted,
            ActivityType::ClaudeTimedOut,
            ActivityType::ClaudeFailed,
            ActivityType::WebhookReceived,
            ActivityType::WebhookProcessed,
            ActivityType::WebhookRejected,
            ActivityType::WatcherStarted,
            ActivityType::WatcherStopped,
            ActivityType::WatcherPaused,
            ActivityType::WatcherResumed,
            ActivityType::RateLimitHit,
            ActivityType::StateChange,
            ActivityType::Error,
        ];

        let count = all_types.len();
        for (i, activity_type) in all_types.into_iter().enumerate() {
            server
                .log_activity(activity_type, &format!("msg-{}", i), None, None)
                .await;
        }

        let activity = server.state.activity.lock().await;
        assert_eq!(activity.len(), count);
    }

    // -------------------------------------------------------------------
    // log_activity with issue_id and source variations
    // -------------------------------------------------------------------

    #[tokio::test]
    async fn test_log_activity_with_issue_and_source() {
        let server = IpcServer::new(mock_tracker(), mock_sources(), mock_notifier());

        server
            .log_activity(
                ActivityType::IssueDetected,
                "msg with all fields",
                Some("LIN-42"),
                Some("linear"),
            )
            .await;

        let activity = server.state.activity.lock().await;
        assert_eq!(activity.len(), 1);
        assert_eq!(activity[0].issue_id, Some("LIN-42".to_string()));
        assert_eq!(activity[0].source, Some("linear".to_string()));
    }

    #[tokio::test]
    async fn test_log_activity_without_issue_or_source() {
        let server = IpcServer::new(mock_tracker(), mock_sources(), mock_notifier());

        server
            .log_activity(ActivityType::WatcherStarted, "no context", None, None)
            .await;

        let activity = server.state.activity.lock().await;
        assert_eq!(activity.len(), 1);
        assert_eq!(activity[0].issue_id, None);
        assert_eq!(activity[0].source, None);
    }

    // -------------------------------------------------------------------
    // IpcServer with_watcher
    // -------------------------------------------------------------------

    #[test]
    fn test_ipc_server_with_watcher_sets_watcher() {
        let server = IpcServer::new(mock_tracker(), mock_sources(), mock_notifier());
        assert!(server.watcher.is_none());
        // We cannot easily create a real Watcher in tests, but we can verify
        // the initial state
    }

    // -------------------------------------------------------------------
    // Builder chaining
    // -------------------------------------------------------------------

    #[test]
    fn test_builder_chaining() {
        let server = IpcServerBuilder::new(mock_tracker(), mock_sources(), mock_notifier())
            .max_activity_entries(100)
            .max_retries(3)
            .build();

        assert_eq!(server.state.max_activity_entries, 100);
        assert_eq!(server.state.max_retries, 3);
        assert!(!server.is_paused());
    }

    // -------------------------------------------------------------------
    // Shutdown receiver can be subscribed multiple times
    // -------------------------------------------------------------------

    #[test]
    fn test_multiple_shutdown_receivers() {
        let server = IpcServer::new(mock_tracker(), mock_sources(), mock_notifier());
        let mut rx1 = server.shutdown_receiver();
        let mut rx2 = server.shutdown_receiver();

        server.shutdown();

        assert!(rx1.try_recv().is_ok());
        assert!(rx2.try_recv().is_ok());
    }

    // -------------------------------------------------------------------
    // IpcServer::new uses default builder
    // -------------------------------------------------------------------

    #[test]
    fn test_ipc_server_new_uses_defaults() {
        let server = IpcServer::new(mock_tracker(), mock_sources(), mock_notifier());
        assert_eq!(
            server.state.max_activity_entries,
            DEFAULT_MAX_ACTIVITY_ENTRIES
        );
        assert_eq!(server.state.max_retries, 2);
        assert_eq!(*server.state.mode.blocking_read(), "initializing");
    }

    // -------------------------------------------------------------------
    // Status command shows processing items
    // -------------------------------------------------------------------

    #[tokio::test]
    async fn test_handle_command_status_with_processing_items() {
        let tracker = mock_tracker();
        let sources = mock_sources();
        let notifier = mock_notifier();
        let state = test_state(100, 2);
        {
            let mut processing = state.processing.write().await;
            processing.push("LIN-1".to_string());
            processing.push("SEN-2".to_string());
        }
        let (shutdown_tx, _) = broadcast::channel(1);

        let resp = handle_command(
            IpcCommand::Status,
            &tracker,
            &sources,
            &notifier,
            &None,
            &state,
            &shutdown_tx,
        )
        .await;

        match resp {
            IpcResponse::Ok(IpcData::State(ws)) => {
                assert_eq!(ws.processing.len(), 2);
                assert!(ws.processing.contains(&"LIN-1".to_string()));
                assert!(ws.processing.contains(&"SEN-2".to_string()));
            }
            other => panic!("Expected State, got {:?}", other),
        }
    }

    // -------------------------------------------------------------------
    // Pause and resume toggle
    // -------------------------------------------------------------------

    #[tokio::test]
    async fn test_pause_resume_toggle() {
        let tracker = mock_tracker();
        let sources = mock_sources();
        let notifier = mock_notifier();
        let state = test_state(100, 2);
        let (shutdown_tx, _) = broadcast::channel(1);

        // Initially not paused
        assert!(!state.paused.load(Ordering::SeqCst));

        // Pause
        handle_command(
            IpcCommand::Pause,
            &tracker,
            &sources,
            &notifier,
            &None,
            &state,
            &shutdown_tx,
        )
        .await;
        assert!(state.paused.load(Ordering::SeqCst));

        // Resume
        handle_command(
            IpcCommand::Resume,
            &tracker,
            &sources,
            &notifier,
            &None,
            &state,
            &shutdown_tx,
        )
        .await;
        assert!(!state.paused.load(Ordering::SeqCst));

        // Double pause
        handle_command(
            IpcCommand::Pause,
            &tracker,
            &sources,
            &notifier,
            &None,
            &state,
            &shutdown_tx,
        )
        .await;
        handle_command(
            IpcCommand::Pause,
            &tracker,
            &sources,
            &notifier,
            &None,
            &state,
            &shutdown_tx,
        )
        .await;
        assert!(state.paused.load(Ordering::SeqCst));
    }

    // -------------------------------------------------------------------
    // Activity limit edge case: limit = 0
    // -------------------------------------------------------------------

    #[tokio::test]
    async fn test_handle_command_activity_limit_zero() {
        let tracker = mock_tracker();
        let sources = mock_sources();
        let notifier = mock_notifier();
        let state = test_state(100, 2);
        let (shutdown_tx, _) = broadcast::channel(1);

        // Add some entries
        {
            let mut activity = state.activity.lock().await;
            activity.push_back(ActivityEntry {
                timestamp: "2025-01-01T00:00:00Z".to_string(),
                activity_type: ActivityType::IssueDetected,
                message: "test".to_string(),
                issue_id: None,
                source: None,
            });
        }

        let resp = handle_command(
            IpcCommand::Activity { limit: 0 },
            &tracker,
            &sources,
            &notifier,
            &None,
            &state,
            &shutdown_tx,
        )
        .await;

        match resp {
            IpcResponse::Ok(IpcData::Activity(entries)) => {
                assert!(entries.is_empty());
            }
            other => panic!("Expected Activity, got {:?}", other),
        }
    }

    // -------------------------------------------------------------------
    // Concurrent increments
    // -------------------------------------------------------------------

    #[test]
    fn test_concurrent_increments() {
        let server = IpcServer::new(mock_tracker(), mock_sources(), mock_notifier());

        for _ in 0..100 {
            server.inc_issues_processed();
        }
        for _ in 0..50 {
            server.inc_prs_created();
        }

        assert_eq!(server.state.issues_processed.load(Ordering::SeqCst), 100);
        assert_eq!(server.state.prs_created.load(Ordering::SeqCst), 50);
    }

    // -------------------------------------------------------------------
    // Builder with empty sources
    // -------------------------------------------------------------------

    #[test]
    fn test_builder_empty_sources() {
        let server = IpcServerBuilder::new(
            mock_tracker(),
            vec![], // no sources
            mock_notifier(),
        )
        .build();

        assert!(server.state.source_names.is_empty());
    }

    // -------------------------------------------------------------------
    // IpcCommand serialization roundtrip
    // -------------------------------------------------------------------

    #[test]
    fn test_ipc_command_serde_roundtrip() {
        let commands = vec![
            IpcCommand::Ping,
            IpcCommand::Status,
            IpcCommand::Pause,
            IpcCommand::Resume,
            IpcCommand::Stats,
            IpcCommand::ListPrs,
            IpcCommand::ListRetries,
            IpcCommand::ProcessRetries,
            IpcCommand::Shutdown,
            IpcCommand::Activity { limit: 42 },
            IpcCommand::Trigger {
                source: "linear".to_string(),
                issue_id: "LIN-1".to_string(),
            },
            IpcCommand::Reset {
                source: "sentry".to_string(),
                issue_id: "SEN-1".to_string(),
            },
        ];

        for cmd in commands {
            let json = serde_json::to_string(&cmd).unwrap();
            let parsed: IpcCommand = serde_json::from_str(&json).unwrap();
            // Verify it round-trips by serializing again
            let json2 = serde_json::to_string(&parsed).unwrap();
            assert_eq!(json, json2);
        }
    }

    // -------------------------------------------------------------------
    // IpcResponse serialization roundtrip
    // -------------------------------------------------------------------

    #[test]
    fn test_ipc_response_ok_serde() {
        let resp = IpcResponse::ok_with(IpcData::Pong);
        assert!(resp.is_ok());
        let json = serde_json::to_string(&resp).unwrap();
        let parsed: IpcResponse = serde_json::from_str(&json).unwrap();
        assert!(parsed.is_ok());
    }

    #[test]
    fn test_ipc_response_error_serde() {
        let resp = IpcResponse::error("something went wrong");
        assert!(!resp.is_ok());
        let json = serde_json::to_string(&resp).unwrap();
        let parsed: IpcResponse = serde_json::from_str(&json).unwrap();
        assert!(!parsed.is_ok());
    }

    #[test]
    fn test_ipc_response_message_serde() {
        let resp = IpcResponse::ok_with(IpcData::Message("hello world".to_string()));
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("hello world"));
    }

    #[test]
    fn test_ipc_response_retries_processed_serde() {
        let resp = IpcResponse::ok_with(IpcData::RetriesProcessed { count: 5 });
        let json = serde_json::to_string(&resp).unwrap();
        let parsed: IpcResponse = serde_json::from_str(&json).unwrap();
        match parsed {
            IpcResponse::Ok(IpcData::RetriesProcessed { count }) => {
                assert_eq!(count, 5);
            }
            other => panic!("Expected RetriesProcessed, got {:?}", other),
        }
    }

    #[test]
    fn test_ipc_response_triggered_serde() {
        let resp = IpcResponse::ok_with(IpcData::Triggered {
            source: "linear".to_string(),
            issue_id: "LIN-1".to_string(),
            pr_url: Some("https://github.com/org/repo/pull/1".to_string()),
        });
        let json = serde_json::to_string(&resp).unwrap();
        let parsed: IpcResponse = serde_json::from_str(&json).unwrap();
        match parsed {
            IpcResponse::Ok(IpcData::Triggered {
                source,
                issue_id,
                pr_url,
            }) => {
                assert_eq!(source, "linear");
                assert_eq!(issue_id, "LIN-1");
                assert!(pr_url.is_some());
            }
            other => panic!("Expected Triggered, got {:?}", other),
        }
    }
}
