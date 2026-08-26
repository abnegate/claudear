//! IPC client for communicating with the watcher daemon.
//!
//! Transport details are handled by [`super::transport`] — this file is
//! platform-agnostic.

use super::default_socket_path;
use super::protocol::{IpcCommand, IpcData, IpcResponse};
use super::transport;
use claudear_core::error::{Error, Result};

use std::path::PathBuf;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::time::timeout;

/// Default timeout for IPC operations.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// Client for communicating with the watcher daemon.
pub struct IpcClient {
    socket_path: PathBuf,
    timeout: Duration,
}

impl IpcClient {
    /// Create a new IPC client with the default socket path.
    pub fn new() -> Self {
        Self {
            socket_path: default_socket_path(),
            timeout: DEFAULT_TIMEOUT,
        }
    }

    /// Create a client with a custom socket/port file path.
    pub fn with_socket_path(socket_path: PathBuf) -> Self {
        Self {
            socket_path,
            timeout: DEFAULT_TIMEOUT,
        }
    }

    /// Set the timeout for operations.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Check if the daemon is running.
    pub fn is_daemon_running(&self) -> bool {
        transport::check_connection(&self.socket_path)
    }

    /// Send a command and receive a response.
    pub async fn send(&self, command: IpcCommand) -> Result<IpcResponse> {
        let result = timeout(self.timeout, self.send_internal(command)).await;

        match result {
            Ok(inner) => inner,
            Err(_) => Err(Error::Other("IPC request timed out".to_string())),
        }
    }

    async fn send_internal(&self, command: IpcCommand) -> Result<IpcResponse> {
        let stream = transport::connect(&self.socket_path)
            .await
            .map_err(|e| Error::Other(format!("Failed to connect to daemon: {}", e)))?;

        let (reader, mut writer) = stream.into_split();
        let mut reader = BufReader::new(reader);

        // Send command
        let json = serde_json::to_string(&command)? + "\n";
        writer.write_all(json.as_bytes()).await?;

        // Read response
        let mut line = String::new();
        reader.read_line(&mut line).await?;

        let response: IpcResponse = serde_json::from_str(line.trim())?;
        Ok(response)
    }

    /// Ping the daemon.
    pub async fn ping(&self) -> Result<bool> {
        match self.send(IpcCommand::Ping).await {
            Ok(IpcResponse::Ok(IpcData::Pong)) => Ok(true),
            Ok(_) => Ok(false),
            Err(_) => Ok(false),
        }
    }

    /// Get the daemon status.
    pub async fn status(&self) -> Result<IpcResponse> {
        self.send(IpcCommand::Status).await
    }

    /// Pause the watcher.
    pub async fn pause(&self) -> Result<IpcResponse> {
        self.send(IpcCommand::Pause).await
    }

    /// Resume the watcher.
    pub async fn resume(&self) -> Result<IpcResponse> {
        self.send(IpcCommand::Resume).await
    }

    /// Trigger a fix for an issue.
    pub async fn trigger(&self, source: &str, issue_id: &str) -> Result<IpcResponse> {
        self.send(IpcCommand::Trigger {
            source: source.to_string(),
            issue_id: issue_id.to_string(),
        })
        .await
    }

    /// Reset a failed attempt.
    pub async fn reset(&self, source: &str, issue_id: &str) -> Result<IpcResponse> {
        self.send(IpcCommand::Reset {
            source: source.to_string(),
            issue_id: issue_id.to_string(),
        })
        .await
    }

    /// Get statistics.
    pub async fn stats(&self) -> Result<IpcResponse> {
        self.send(IpcCommand::Stats).await
    }

    /// List pending PRs.
    pub async fn list_prs(&self) -> Result<IpcResponse> {
        self.send(IpcCommand::ListPrs).await
    }

    /// List retryable issues.
    pub async fn list_retries(&self) -> Result<IpcResponse> {
        self.send(IpcCommand::ListRetries).await
    }

    /// Process ready retries.
    pub async fn process_retries(&self) -> Result<IpcResponse> {
        self.send(IpcCommand::ProcessRetries).await
    }

    /// Get recent activity.
    pub async fn activity(&self, limit: usize) -> Result<IpcResponse> {
        self.send(IpcCommand::Activity { limit }).await
    }

    /// Request graceful shutdown.
    pub async fn shutdown(&self) -> Result<IpcResponse> {
        self.send(IpcCommand::Shutdown).await
    }
}

impl Default for IpcClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Print an IPC response in a human-readable format.
pub fn print_response(response: &IpcResponse) {
    match response {
        IpcResponse::Ok(data) => match data {
            IpcData::None => println!("OK"),
            IpcData::Pong => println!("Pong!"),
            IpcData::Message(msg) => println!("{}", msg),
            IpcData::State(state) => {
                println!("\nWatcher Status:");
                println!("  Running: {}", state.running);
                println!("  Paused:  {}", state.paused);
                println!("  Mode:    {}", state.mode);
                println!("  Uptime:  {}s", state.uptime_secs);
                println!("  Issues processed: {}", state.issues_processed);
                println!("  PRs created:      {}", state.prs_created);
                println!("  Sources: {}", state.sources.join(", "));
                if let Some(interval) = state.poll_interval_ms {
                    println!("  Poll interval: {}ms", interval);
                }
                if !state.processing.is_empty() {
                    println!("  Currently processing: {}", state.processing.join(", "));
                }
            }
            IpcData::Stats(stats) => {
                println!("\nFix Attempt Statistics:");
                println!("  Total:      {}", stats.total);
                println!("  Pending:    {}", stats.pending);
                println!("  Success:    {}", stats.success);
                println!("  Merged:     {}", stats.merged);
                println!("  Closed:     {}", stats.closed);
                println!("  Failed:     {}", stats.failed);
                println!("  Cannot Fix: {}", stats.cannot_fix);

                if !stats.by_source.is_empty() {
                    println!("\nBy Source:");
                    for (source, source_stats) in &stats.by_source {
                        println!(
                            "  {}: {} total, {} merged, {} failed",
                            source, source_stats.total, source_stats.merged, source_stats.failed
                        );
                    }
                }
            }
            IpcData::Attempts(attempts) => {
                if attempts.is_empty() {
                    println!("No attempts found.");
                } else {
                    for attempt in attempts {
                        println!(
                            "  [{}] {} - {:?} - {}",
                            attempt.source,
                            attempt.short_id,
                            attempt.status,
                            attempt.pr_url.as_deref().unwrap_or("N/A")
                        );
                    }
                }
            }
            IpcData::Triggered {
                source,
                issue_id,
                pr_url,
            } => {
                println!("Triggered fix for {}:{}", source, issue_id);
                if let Some(url) = pr_url {
                    println!("  PR: {}", url);
                }
            }
            IpcData::Reset { source, issue_id } => {
                println!("Reset {}:{}", source, issue_id);
            }
            IpcData::RetriesProcessed { count } => {
                println!("Processed {} retries", count);
            }
            IpcData::Activity(entries) => {
                if entries.is_empty() {
                    println!("No recent activity.");
                } else {
                    println!("\nRecent Activity:");
                    for entry in entries {
                        let source_info = entry
                            .source
                            .as_ref()
                            .map(|s| format!("[{}] ", s))
                            .unwrap_or_default();
                        let issue_info = entry
                            .issue_id
                            .as_ref()
                            .map(|id| format!("{}: ", id))
                            .unwrap_or_default();
                        println!(
                            "  {} {}{}{:?}: {}",
                            entry.timestamp,
                            source_info,
                            issue_info,
                            entry.activity_type,
                            entry.message
                        );
                    }
                }
            }
        },
        IpcResponse::Error { message } => {
            eprintln!("Error: {}", message);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_default() {
        let client = IpcClient::default();
        assert!(!client.is_daemon_running()); // Assuming no daemon in tests
    }

    #[test]
    fn test_client_with_timeout() {
        let client = IpcClient::new().with_timeout(Duration::from_secs(5));
        assert_eq!(client.timeout, Duration::from_secs(5));
    }

    #[test]
    fn test_client_with_socket_path() {
        let path = std::env::temp_dir().join("test-claudear.sock");
        let client = IpcClient::with_socket_path(path.clone());
        assert_eq!(client.socket_path, path);
    }

    // === Constructor tests ===

    #[test]
    fn test_new_uses_default_path() {
        let client = IpcClient::new();
        let expected = super::super::default_socket_path();
        assert_eq!(client.socket_path, expected);
    }

    #[test]
    fn test_with_socket_path_uses_custom_path() {
        let custom = std::env::temp_dir().join("custom-claudear.sock");
        let client = IpcClient::with_socket_path(custom.clone());
        assert_eq!(client.socket_path, custom);
    }

    #[test]
    fn test_new_uses_default_timeout() {
        let client = IpcClient::new();
        assert_eq!(client.timeout, Duration::from_secs(30));
    }

    #[test]
    fn test_with_socket_path_uses_default_timeout() {
        let client = IpcClient::with_socket_path(std::env::temp_dir().join("x.sock"));
        assert_eq!(client.timeout, Duration::from_secs(30));
    }

    // === with_timeout tests ===

    #[test]
    fn test_with_timeout_sets_custom_timeout() {
        let client = IpcClient::new().with_timeout(Duration::from_secs(120));
        assert_eq!(client.timeout, Duration::from_secs(120));
    }

    #[test]
    fn test_with_timeout_chained_with_socket_path() {
        let path = std::env::temp_dir().join("chained.sock");
        let client =
            IpcClient::with_socket_path(path.clone()).with_timeout(Duration::from_millis(500));
        assert_eq!(client.socket_path, path);
        assert_eq!(client.timeout, Duration::from_millis(500));
    }

    #[test]
    fn test_with_timeout_zero() {
        let client = IpcClient::new().with_timeout(Duration::from_secs(0));
        assert_eq!(client.timeout, Duration::from_secs(0));
    }

    // === is_daemon_running tests ===

    #[test]
    fn test_is_daemon_running_nonexistent_socket() {
        let client = IpcClient::with_socket_path(
            std::env::temp_dir().join("nonexistent-claudear-test.sock"),
        );
        assert!(!client.is_daemon_running());
    }

    #[test]
    fn test_is_daemon_running_path_is_regular_file() {
        // Create a temp file that is NOT a socket
        let tmp = std::env::temp_dir().join("claudear-test-not-a-socket.tmp");
        std::fs::write(&tmp, "not a socket").unwrap();
        let client = IpcClient::with_socket_path(tmp.clone());
        assert!(!client.is_daemon_running());
        let _ = std::fs::remove_file(&tmp);
    }

    // === send to non-existent socket tests ===

    #[tokio::test]
    async fn test_send_to_nonexistent_socket_returns_error() {
        let client =
            IpcClient::with_socket_path(std::env::temp_dir().join("claudear-no-such-socket.sock"));
        let result = client.send(IpcCommand::Ping).await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Failed to connect to daemon")
                || err_msg.contains("Failed to read port file"),
            "Unexpected error message: {}",
            err_msg
        );
    }

    #[tokio::test]
    async fn test_ping_nonexistent_socket_returns_false() {
        let client =
            IpcClient::with_socket_path(std::env::temp_dir().join("claudear-no-such-socket.sock"));
        let result = client.ping().await;
        assert!(result.is_ok());
        assert!(!result.unwrap());
    }

    #[tokio::test]
    async fn test_status_nonexistent_socket_returns_error() {
        let client =
            IpcClient::with_socket_path(std::env::temp_dir().join("claudear-no-such-socket.sock"));
        let result = client.status().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_send_trigger_nonexistent_socket_returns_error() {
        let client =
            IpcClient::with_socket_path(std::env::temp_dir().join("claudear-no-such-socket.sock"));
        let result = client.trigger("linear", "LIN-1").await;
        assert!(result.is_err());
    }

    // === print_response tests ===

    use super::super::protocol::{ActivityEntry, ActivityType, WatcherState};
    use claudear_core::types::{FixAttempt, FixAttemptStats, FixAttemptStatus};

    #[test]
    fn test_print_response_ok_none() {
        let response = IpcResponse::Ok(IpcData::None);
        print_response(&response);
    }

    #[test]
    fn test_print_response_pong() {
        let response = IpcResponse::Ok(IpcData::Pong);
        print_response(&response);
    }

    #[test]
    fn test_print_response_message() {
        let response = IpcResponse::Ok(IpcData::Message("hello".to_string()));
        print_response(&response);
    }

    #[test]
    fn test_print_response_error() {
        let response = IpcResponse::Error {
            message: "something went wrong".to_string(),
        };
        print_response(&response);
    }

    #[test]
    fn test_print_response_state() {
        let state = WatcherState {
            running: true,
            paused: false,
            mode: "poll".to_string(),
            uptime_secs: 3600,
            issues_processed: 42,
            prs_created: 7,
            processing: vec!["LIN-1".to_string(), "LIN-2".to_string()],
            sources: vec!["linear".to_string(), "sentry".to_string()],
            poll_interval_ms: Some(60_000),
        };
        let response = IpcResponse::Ok(IpcData::State(state));
        print_response(&response);
    }

    #[test]
    fn test_print_response_stats() {
        let stats = FixAttemptStats {
            total: 10,
            pending: 2,
            success: 5,
            failed: 2,
            merged: 1,
            closed: 0,
            cannot_fix: 0,
            by_source: std::collections::HashMap::new(),
        };
        let response = IpcResponse::Ok(IpcData::Stats(stats));
        print_response(&response);
    }

    #[test]
    fn test_print_response_attempts_empty() {
        let response = IpcResponse::Ok(IpcData::Attempts(vec![]));
        print_response(&response);
    }

    #[test]
    fn test_print_response_attempts() {
        let attempt = FixAttempt {
            id: 1,
            issue_id: "abc123".to_string(),
            short_id: "LIN-1".to_string(),
            source: "linear".to_string(),
            attempted_at: chrono::Utc::now(),
            pr_url: Some("https://github.com/org/repo/pull/42".to_string()),
            scm_repo: None,
            scm_pr_number: None,
            status: FixAttemptStatus::Success,
            error_message: None,
            merged_at: None,
            resolved_at: None,
            retry_count: 0,
            last_retry_at: None,
            issue_labels: vec![],
            parent_attempt_id: None,
            cascade_repo: None,
        };
        let response = IpcResponse::Ok(IpcData::Attempts(vec![attempt]));
        print_response(&response);
    }

    #[test]
    fn test_print_response_triggered() {
        // With pr_url
        let response = IpcResponse::Ok(IpcData::Triggered {
            source: "linear".to_string(),
            issue_id: "LIN-1".to_string(),
            pr_url: Some("https://github.com/org/repo/pull/99".to_string()),
        });
        print_response(&response);

        // Without pr_url
        let response = IpcResponse::Ok(IpcData::Triggered {
            source: "sentry".to_string(),
            issue_id: "SENTRY-42".to_string(),
            pr_url: None,
        });
        print_response(&response);
    }

    #[test]
    fn test_print_response_reset() {
        let response = IpcResponse::Ok(IpcData::Reset {
            source: "linear".to_string(),
            issue_id: "LIN-5".to_string(),
        });
        print_response(&response);
    }

    #[test]
    fn test_print_response_retries() {
        let response = IpcResponse::Ok(IpcData::RetriesProcessed { count: 3 });
        print_response(&response);
    }

    #[test]
    fn test_print_response_activity_empty() {
        let response = IpcResponse::Ok(IpcData::Activity(vec![]));
        print_response(&response);
    }

    #[test]
    fn test_print_response_activity() {
        let entries = vec![
            ActivityEntry {
                timestamp: "2025-01-01T00:00:00Z".to_string(),
                activity_type: ActivityType::IssueDetected,
                message: "Detected new issue".to_string(),
                issue_id: Some("LIN-1".to_string()),
                source: Some("linear".to_string()),
            },
            ActivityEntry {
                timestamp: "2025-01-01T00:01:00Z".to_string(),
                activity_type: ActivityType::PrCreated,
                message: "PR created".to_string(),
                issue_id: None,
                source: None,
            },
        ];
        let response = IpcResponse::Ok(IpcData::Activity(entries));
        print_response(&response);
    }

    // === Coverage tests for print_response with by_source entries ===

    #[test]
    fn test_print_response_stats_with_by_source() {
        use claudear_core::types::SourceStats;
        let mut by_source = std::collections::HashMap::new();
        by_source.insert(
            "linear".to_string(),
            SourceStats {
                total: 5,
                success: 3,
                failed: 1,
                merged: 2,
                closed: 0,
                cannot_fix: 0,
            },
        );
        by_source.insert(
            "sentry".to_string(),
            SourceStats {
                total: 3,
                success: 1,
                failed: 2,
                merged: 0,
                closed: 0,
                cannot_fix: 0,
            },
        );
        let stats = FixAttemptStats {
            total: 8,
            pending: 1,
            success: 4,
            failed: 3,
            merged: 2,
            closed: 0,
            cannot_fix: 0,
            by_source,
        };
        let response = IpcResponse::Ok(IpcData::Stats(stats));
        // Should not panic
        print_response(&response);
    }

    // === Coverage: state with no poll_interval and no processing ===

    #[test]
    fn test_print_response_state_no_poll_no_processing() {
        let state = WatcherState {
            running: false,
            paused: true,
            mode: "webhook".to_string(),
            uptime_secs: 0,
            issues_processed: 0,
            prs_created: 0,
            processing: vec![],
            sources: vec![],
            poll_interval_ms: None,
        };
        let response = IpcResponse::Ok(IpcData::State(state));
        print_response(&response);
    }

    // === Coverage: convenience methods that delegate to send ===

    #[tokio::test]
    async fn test_pause_nonexistent_socket_returns_error() {
        let client =
            IpcClient::with_socket_path(PathBuf::from("/tmp/claudear-no-such-socket.sock"));
        let result = client.pause().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_resume_nonexistent_socket_returns_error() {
        let client =
            IpcClient::with_socket_path(PathBuf::from("/tmp/claudear-no-such-socket.sock"));
        let result = client.resume().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_reset_nonexistent_socket_returns_error() {
        let client =
            IpcClient::with_socket_path(PathBuf::from("/tmp/claudear-no-such-socket.sock"));
        let result = client.reset("linear", "LIN-1").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_stats_nonexistent_socket_returns_error() {
        let client =
            IpcClient::with_socket_path(PathBuf::from("/tmp/claudear-no-such-socket.sock"));
        let result = client.stats().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_list_prs_nonexistent_socket_returns_error() {
        let client =
            IpcClient::with_socket_path(PathBuf::from("/tmp/claudear-no-such-socket.sock"));
        let result = client.list_prs().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_list_retries_nonexistent_socket_returns_error() {
        let client =
            IpcClient::with_socket_path(PathBuf::from("/tmp/claudear-no-such-socket.sock"));
        let result = client.list_retries().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_process_retries_nonexistent_socket_returns_error() {
        let client =
            IpcClient::with_socket_path(PathBuf::from("/tmp/claudear-no-such-socket.sock"));
        let result = client.process_retries().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_activity_nonexistent_socket_returns_error() {
        let client =
            IpcClient::with_socket_path(PathBuf::from("/tmp/claudear-no-such-socket.sock"));
        let result = client.activity(10).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_shutdown_nonexistent_socket_returns_error() {
        let client =
            IpcClient::with_socket_path(PathBuf::from("/tmp/claudear-no-such-socket.sock"));
        let result = client.shutdown().await;
        assert!(result.is_err());
    }
}
