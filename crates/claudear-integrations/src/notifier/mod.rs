//! Notification implementations.
//!
//! The notifier system provides a flexible abstraction for sending notifications
//! to various channels. All notifiers implement the `Notifier` trait.
//!
//! ## Available Notifiers
//!
//! - `ConsoleNotifier` - Prints to stdout (always enabled)
//! - `DiscordNotifier` - Sends Discord webhook messages
//! - `SlackNotifier` - Sends Slack Block Kit messages
//! - `EmailNotifier` - Sends email via SMTP
//! - `SmsNotifier` - Sends SMS via Twilio
//! - `PushNotifier` - Sends push notifications via Pushover
//!
//! ## Adding a New Notifier
//!
//! 1. Create a new module file (e.g., `slack.rs`)
//! 2. Implement the `Notifier` trait
//! 3. Export from this module
//! 4. Add configuration to `config.rs`
//!
//! Example:
//!
//! ```no_run
//! use async_trait::async_trait;
//! use claudear_integrations::notifier::Notifier;
//! use claudear_core::types::Issue;
//! use claudear_core::error::Result;
//!
//! pub struct SlackNotifier {
//!     webhook_url: String,
//! }
//!
//! #[async_trait]
//! impl Notifier for SlackNotifier {
//!     fn name(&self) -> &str { "slack" }
//!     fn is_enabled(&self) -> bool { !self.webhook_url.is_empty() }
//!
//!     async fn notify_start(&self, _issue: &Issue) -> Result<()> { Ok(()) }
//!     async fn notify_success(&self, _issue: &Issue, _pr_url: &str) -> Result<()> { Ok(()) }
//!     async fn notify_completed(&self, _issue: &Issue) -> Result<()> { Ok(()) }
//!     async fn notify_failed(&self, _issue: &Issue, _error: &str) -> Result<()> { Ok(()) }
//!     async fn notify_status(&self, _message: &str) -> Result<()> { Ok(()) }
//!     async fn notify_urgent_issues(&self, _issues: &[Issue]) -> Result<()> { Ok(()) }
//! }
//! ```

pub mod ask_orchestrator;
mod console;
mod discord;
mod email;
mod push;
mod slack;
mod sms;
mod telegram;
mod whatsapp;

pub use ask_orchestrator::send_to_all_and_wait_first_reply;
pub use console::ConsoleNotifier;
pub use discord::DiscordNotifier;
pub use email::EmailNotifier;
pub use push::PushNotifier;
pub use slack::SlackNotifier;
pub use sms::SmsNotifier;
pub use telegram::TelegramNotifier;
pub use whatsapp::WhatsAppNotifier;

use crate::reports::Report;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use claudear_core::error::Result;
use claudear_core::types::{AskDelivery, AskReply, AskRequest, Issue};
use std::sync::Arc;

/// Return the emoji for a given issue source.
pub(crate) fn get_source_emoji(source: &str) -> &'static str {
    match source.to_lowercase().as_str() {
        "linear" => "\u{1F4CB}",   // clipboard
        "sentry" => "\u{1F534}",   // red circle
        "github" => "\u{1F419}",   // octopus
        "jira" => "\u{1F3AB}",     // ticket
        "slack" => "\u{1F4AC}",    // speech balloon
        "whatsapp" => "\u{1F4F1}", // mobile phone
        "telegram" => "\u{2708}",  // airplane/paper plane
        _ => "\u{1F4CC}",          // pushpin
    }
}

/// Trait for notification services.
#[async_trait]
pub trait Notifier: Send + Sync {
    /// Unique name for this notifier.
    fn name(&self) -> &str;

    /// Whether this notifier is currently enabled/configured.
    fn is_enabled(&self) -> bool;

    /// Notify that processing has started for an issue.
    async fn notify_start(&self, issue: &Issue) -> Result<()>;

    /// Notify that a fix was successful with PR link.
    async fn notify_success(&self, issue: &Issue, pr_url: &str) -> Result<()>;

    /// Notify that processing completed but no PR was found.
    async fn notify_completed(&self, issue: &Issue) -> Result<()>;

    /// Notify that processing failed.
    async fn notify_failed(&self, issue: &Issue, error: &str) -> Result<()>;

    /// Send a general status message.
    async fn notify_status(&self, message: &str) -> Result<()>;

    /// Notify about multiple urgent issues detected.
    async fn notify_urgent_issues(&self, issues: &[Issue]) -> Result<()>;

    /// Notify that a PR was merged and issue was resolved.
    async fn notify_merged(&self, issue: &Issue, pr_url: &str) -> Result<()> {
        // Default implementation uses notify_status
        self.notify_status(&format!(
            "PR merged and issue resolved: {} - {}",
            issue.short_id, pr_url
        ))
        .await
    }

    /// Notify that a PR was closed without merging.
    async fn notify_closed(&self, issue: &Issue, pr_url: &str) -> Result<()> {
        self.notify_status(&format!(
            "PR closed without merging: {} - {}",
            issue.short_id, pr_url
        ))
        .await
    }

    /// Send a scheduled report.
    async fn notify_report(&self, report: &Report) -> Result<()> {
        // Default implementation formats as text and uses notify_status
        self.notify_status(&report.format_text()).await
    }

    /// Send a blocking question through this channel.
    async fn ask_question(
        &self,
        _issue: &Issue,
        _request: &AskRequest,
    ) -> Result<Option<AskDelivery>> {
        Ok(None)
    }

    /// Poll replies for a previously sent question.
    async fn poll_question_replies(
        &self,
        _request: &AskRequest,
        _since: DateTime<Utc>,
    ) -> Result<Vec<AskReply>> {
        Ok(Vec::new())
    }

    /// Whether this notifier can receive replies.
    fn supports_replies(&self) -> bool {
        false
    }

    /// Notify that a repo swap is happening for an issue.
    async fn notify_repo_swap(&self, issue: &Issue, from_repo: &str, to_repo: &str) -> Result<()> {
        self.notify_status(&format!(
            "Repo swap for {}: {} \u{2192} {}",
            issue.short_id, from_repo, to_repo
        ))
        .await
    }
}

/// Composite notifier that sends to multiple notifiers.
pub struct CompositeNotifier {
    notifiers: Vec<Arc<dyn Notifier>>,
}

impl CompositeNotifier {
    /// Create a new empty composite notifier.
    pub fn new() -> Self {
        Self { notifiers: vec![] }
    }

    /// Add a notifier to the composite.
    pub fn add(&mut self, notifier: Arc<dyn Notifier>) {
        if notifier.is_enabled() {
            self.notifiers.push(notifier);
        }
    }

    /// Check if any notifiers are enabled.
    pub fn is_enabled(&self) -> bool {
        !self.notifiers.is_empty()
    }

    async fn broadcast<F, Fut>(&self, f: F)
    where
        F: Fn(Arc<dyn Notifier>) -> Fut,
        Fut: std::future::Future<Output = Result<()>>,
    {
        let futures: Vec<_> = self
            .notifiers
            .iter()
            .map(|n| {
                let notifier = Arc::clone(n);
                f(notifier)
            })
            .collect();

        for result in futures::future::join_all(futures).await {
            if let Err(e) = result {
                tracing::error!("Notification error: {}", e);
            }
        }
    }
}

impl Default for CompositeNotifier {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Notifier for CompositeNotifier {
    fn name(&self) -> &str {
        "composite"
    }

    fn is_enabled(&self) -> bool {
        !self.notifiers.is_empty()
    }

    async fn notify_start(&self, issue: &Issue) -> Result<()> {
        let issue = issue.clone();
        self.broadcast(|n| {
            let issue = issue.clone();
            async move { n.notify_start(&issue).await }
        })
        .await;
        Ok(())
    }

    async fn notify_success(&self, issue: &Issue, pr_url: &str) -> Result<()> {
        let issue = issue.clone();
        let pr_url = pr_url.to_string();
        self.broadcast(|n| {
            let issue = issue.clone();
            let pr_url = pr_url.clone();
            async move { n.notify_success(&issue, &pr_url).await }
        })
        .await;
        Ok(())
    }

    async fn notify_completed(&self, issue: &Issue) -> Result<()> {
        let issue = issue.clone();
        self.broadcast(|n| {
            let issue = issue.clone();
            async move { n.notify_completed(&issue).await }
        })
        .await;
        Ok(())
    }

    async fn notify_failed(&self, issue: &Issue, error: &str) -> Result<()> {
        let issue = issue.clone();
        let error = error.to_string();
        self.broadcast(|n| {
            let issue = issue.clone();
            let error = error.clone();
            async move { n.notify_failed(&issue, &error).await }
        })
        .await;
        Ok(())
    }

    async fn notify_status(&self, message: &str) -> Result<()> {
        let message = message.to_string();
        self.broadcast(|n| {
            let message = message.clone();
            async move { n.notify_status(&message).await }
        })
        .await;
        Ok(())
    }

    async fn notify_urgent_issues(&self, issues: &[Issue]) -> Result<()> {
        let issues: Vec<Issue> = issues.to_vec();
        self.broadcast(|n| {
            let issues = issues.clone();
            async move { n.notify_urgent_issues(&issues).await }
        })
        .await;
        Ok(())
    }

    async fn notify_merged(&self, issue: &Issue, pr_url: &str) -> Result<()> {
        let issue = issue.clone();
        let pr_url = pr_url.to_string();
        self.broadcast(|n| {
            let issue = issue.clone();
            let pr_url = pr_url.clone();
            async move { n.notify_merged(&issue, &pr_url).await }
        })
        .await;
        Ok(())
    }

    async fn notify_closed(&self, issue: &Issue, pr_url: &str) -> Result<()> {
        let issue = issue.clone();
        let pr_url = pr_url.to_string();
        self.broadcast(|n| {
            let issue = issue.clone();
            let pr_url = pr_url.clone();
            async move { n.notify_closed(&issue, &pr_url).await }
        })
        .await;
        Ok(())
    }

    async fn notify_report(&self, report: &Report) -> Result<()> {
        let report = report.clone();
        self.broadcast(|n| {
            let report = report.clone();
            async move { n.notify_report(&report).await }
        })
        .await;
        Ok(())
    }

    async fn ask_question(
        &self,
        issue: &Issue,
        request: &AskRequest,
    ) -> Result<Option<AskDelivery>> {
        let issue = issue.clone();
        let request = request.clone();
        let futures: Vec<_> = self
            .notifiers
            .iter()
            .map(|n| {
                let notifier = Arc::clone(n);
                let issue = issue.clone();
                let request = request.clone();
                async move { notifier.ask_question(&issue, &request).await }
            })
            .collect();

        let mut first_delivery: Option<AskDelivery> = None;
        for result in futures::future::join_all(futures).await {
            match result {
                Ok(delivery) => {
                    if first_delivery.is_none() {
                        first_delivery = delivery;
                    }
                }
                Err(e) => tracing::error!("Notification error: {}", e),
            }
        }
        Ok(first_delivery)
    }

    async fn poll_question_replies(
        &self,
        request: &AskRequest,
        since: DateTime<Utc>,
    ) -> Result<Vec<AskReply>> {
        let request = request.clone();
        let futures: Vec<_> = self
            .notifiers
            .iter()
            .filter(|n| n.supports_replies())
            .map(|n| {
                let notifier = Arc::clone(n);
                let request = request.clone();
                async move { notifier.poll_question_replies(&request, since).await }
            })
            .collect();

        let mut replies = Vec::new();
        for result in futures::future::join_all(futures).await {
            match result {
                Ok(mut channel_replies) => replies.append(&mut channel_replies),
                Err(e) => tracing::error!("Notification error: {}", e),
            }
        }
        Ok(replies)
    }

    fn supports_replies(&self) -> bool {
        self.notifiers.iter().any(|n| n.supports_replies())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct MockNotifier {
        name: String,
        enabled: bool,
        call_count: AtomicUsize,
    }

    impl MockNotifier {
        fn new(name: &str, enabled: bool) -> Self {
            Self {
                name: name.to_string(),
                enabled,
                call_count: AtomicUsize::new(0),
            }
        }

        fn get_call_count(&self) -> usize {
            self.call_count.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl Notifier for MockNotifier {
        fn name(&self) -> &str {
            &self.name
        }

        fn is_enabled(&self) -> bool {
            self.enabled
        }

        async fn notify_start(&self, _issue: &Issue) -> Result<()> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        async fn notify_success(&self, _issue: &Issue, _pr_url: &str) -> Result<()> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        async fn notify_completed(&self, _issue: &Issue) -> Result<()> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        async fn notify_failed(&self, _issue: &Issue, _error: &str) -> Result<()> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        async fn notify_status(&self, _message: &str) -> Result<()> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        async fn notify_urgent_issues(&self, _issues: &[Issue]) -> Result<()> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    fn test_issue() -> Issue {
        Issue::new(
            "123",
            "TEST-123",
            "Test Issue",
            "https://example.com",
            "linear",
        )
    }

    #[test]
    fn test_composite_notifier_new() {
        let composite = CompositeNotifier::new();
        assert_eq!(composite.name(), "composite");
        assert!(!composite.is_enabled());
    }

    #[test]
    fn test_composite_notifier_default() {
        let composite = CompositeNotifier::default();
        assert!(!composite.is_enabled());
    }

    #[test]
    fn test_composite_notifier_add_enabled() {
        let mut composite = CompositeNotifier::new();
        let mock = Arc::new(MockNotifier::new("mock1", true));
        composite.add(mock);
        assert!(composite.is_enabled());
    }

    #[test]
    fn test_composite_notifier_add_disabled() {
        let mut composite = CompositeNotifier::new();
        let mock = Arc::new(MockNotifier::new("mock1", false));
        composite.add(mock);
        // Disabled notifiers shouldn't be added
        assert!(!composite.is_enabled());
    }

    #[test]
    fn test_composite_notifier_add_multiple() {
        let mut composite = CompositeNotifier::new();
        composite.add(Arc::new(MockNotifier::new("mock1", true)));
        composite.add(Arc::new(MockNotifier::new("mock2", true)));
        composite.add(Arc::new(MockNotifier::new("mock3", false))); // disabled
        assert!(composite.is_enabled());
        assert_eq!(composite.notifiers.len(), 2);
    }

    #[tokio::test]
    async fn test_composite_notify_start() {
        let mut composite = CompositeNotifier::new();
        let mock1 = Arc::new(MockNotifier::new("mock1", true));
        let mock2 = Arc::new(MockNotifier::new("mock2", true));
        let mock1_clone = Arc::clone(&mock1);
        let mock2_clone = Arc::clone(&mock2);
        composite.add(mock1);
        composite.add(mock2);

        let result = composite.notify_start(&test_issue()).await;
        assert!(result.is_ok());
        assert_eq!(mock1_clone.get_call_count(), 1);
        assert_eq!(mock2_clone.get_call_count(), 1);
    }

    #[tokio::test]
    async fn test_composite_notify_success() {
        let mut composite = CompositeNotifier::new();
        let mock = Arc::new(MockNotifier::new("mock", true));
        let mock_clone = Arc::clone(&mock);
        composite.add(mock);

        let result = composite
            .notify_success(&test_issue(), "https://github.com/org/repo/pull/1")
            .await;
        assert!(result.is_ok());
        assert_eq!(mock_clone.get_call_count(), 1);
    }

    #[tokio::test]
    async fn test_composite_notify_completed() {
        let mut composite = CompositeNotifier::new();
        let mock = Arc::new(MockNotifier::new("mock", true));
        let mock_clone = Arc::clone(&mock);
        composite.add(mock);

        let result = composite.notify_completed(&test_issue()).await;
        assert!(result.is_ok());
        assert_eq!(mock_clone.get_call_count(), 1);
    }

    #[tokio::test]
    async fn test_composite_notify_failed() {
        let mut composite = CompositeNotifier::new();
        let mock = Arc::new(MockNotifier::new("mock", true));
        let mock_clone = Arc::clone(&mock);
        composite.add(mock);

        let result = composite.notify_failed(&test_issue(), "Error").await;
        assert!(result.is_ok());
        assert_eq!(mock_clone.get_call_count(), 1);
    }

    #[tokio::test]
    async fn test_composite_notify_status() {
        let mut composite = CompositeNotifier::new();
        let mock = Arc::new(MockNotifier::new("mock", true));
        let mock_clone = Arc::clone(&mock);
        composite.add(mock);

        let result = composite.notify_status("Status message").await;
        assert!(result.is_ok());
        assert_eq!(mock_clone.get_call_count(), 1);
    }

    #[tokio::test]
    async fn test_composite_notify_urgent_issues() {
        let mut composite = CompositeNotifier::new();
        let mock = Arc::new(MockNotifier::new("mock", true));
        let mock_clone = Arc::clone(&mock);
        composite.add(mock);

        let issues = vec![test_issue(), test_issue()];
        let result = composite.notify_urgent_issues(&issues).await;
        assert!(result.is_ok());
        assert_eq!(mock_clone.get_call_count(), 1);
    }

    #[tokio::test]
    async fn test_composite_notify_merged() {
        let mut composite = CompositeNotifier::new();
        let mock = Arc::new(MockNotifier::new("mock", true));
        let mock_clone = Arc::clone(&mock);
        composite.add(mock);

        let result = composite
            .notify_merged(&test_issue(), "https://github.com/org/repo/pull/1")
            .await;
        assert!(result.is_ok());
        // notify_merged calls notify_status by default
        assert_eq!(mock_clone.get_call_count(), 1);
    }

    #[tokio::test]
    async fn test_composite_empty_broadcasts() {
        let composite = CompositeNotifier::new();

        // All should succeed even with no notifiers
        assert!(composite.notify_start(&test_issue()).await.is_ok());
        assert!(composite.notify_success(&test_issue(), "url").await.is_ok());
        assert!(composite.notify_completed(&test_issue()).await.is_ok());
        assert!(composite.notify_failed(&test_issue(), "err").await.is_ok());
        assert!(composite.notify_status("msg").await.is_ok());
        assert!(composite.notify_urgent_issues(&[]).await.is_ok());
    }

    #[tokio::test]
    async fn test_composite_broadcast_all_notifiers() {
        let mut composite = CompositeNotifier::new();
        let mocks: Vec<Arc<MockNotifier>> = (0..5)
            .map(|i| Arc::new(MockNotifier::new(&format!("mock{}", i), true)))
            .collect();

        for mock in &mocks {
            composite.add(Arc::clone(mock) as Arc<dyn Notifier>);
        }

        composite.notify_start(&test_issue()).await.unwrap();

        for mock in &mocks {
            assert_eq!(
                mock.get_call_count(),
                1,
                "Each notifier should be called once"
            );
        }
    }

    #[test]
    fn test_notifier_trait_name() {
        let mock = MockNotifier::new("test_notifier", true);
        assert_eq!(mock.name(), "test_notifier");
    }

    #[test]
    fn test_notifier_trait_is_enabled() {
        let enabled = MockNotifier::new("enabled", true);
        let disabled = MockNotifier::new("disabled", false);
        assert!(enabled.is_enabled());
        assert!(!disabled.is_enabled());
    }

    #[tokio::test]
    async fn test_composite_notify_report() {
        let mut composite = CompositeNotifier::new();
        let mock = Arc::new(MockNotifier::new("mock", true));
        let mock_clone = Arc::clone(&mock);
        composite.add(mock);

        let report = Report {
            period: "2024-01-01".to_string(),
            from: chrono::Utc::now(),
            to: chrono::Utc::now(),
            issues_attempted: 10,
            issues_succeeded: 8,
            issues_failed: 2,
            issues_cannot_fix: 0,
            success_rate: 80.0,
            failure_rate: 20.0,
            prs_created: 10,
            prs_merged: 5,
            prs_closed: 0,
            by_source: std::collections::HashMap::new(),
            pending_count: 0,
            retryable_count: 0,
        };

        let result = composite.notify_report(&report).await;
        assert!(result.is_ok());
        // notify_report default calls notify_status
        assert_eq!(mock_clone.get_call_count(), 1);
    }

    // Mock that returns errors to test error handling in broadcast
    struct FailingNotifier {
        name: String,
    }

    impl FailingNotifier {
        fn new(name: &str) -> Self {
            Self {
                name: name.to_string(),
            }
        }
    }

    #[async_trait]
    impl Notifier for FailingNotifier {
        fn name(&self) -> &str {
            &self.name
        }
        fn is_enabled(&self) -> bool {
            true
        }
        async fn notify_start(&self, _issue: &Issue) -> Result<()> {
            Err(claudear_core::error::Error::config("Notification failed"))
        }
        async fn notify_success(&self, _issue: &Issue, _pr_url: &str) -> Result<()> {
            Err(claudear_core::error::Error::config("Notification failed"))
        }
        async fn notify_completed(&self, _issue: &Issue) -> Result<()> {
            Err(claudear_core::error::Error::config("Notification failed"))
        }
        async fn notify_failed(&self, _issue: &Issue, _error: &str) -> Result<()> {
            Err(claudear_core::error::Error::config("Notification failed"))
        }
        async fn notify_status(&self, _message: &str) -> Result<()> {
            Err(claudear_core::error::Error::config("Notification failed"))
        }
        async fn notify_urgent_issues(&self, _issues: &[Issue]) -> Result<()> {
            Err(claudear_core::error::Error::config("Notification failed"))
        }
    }

    #[tokio::test]
    async fn test_composite_broadcast_handles_errors() {
        let mut composite = CompositeNotifier::new();
        composite.add(Arc::new(FailingNotifier::new("failing")));
        composite.add(Arc::new(MockNotifier::new("working", true)));

        // Should not panic even when one notifier fails
        let result = composite.notify_start(&test_issue()).await;
        // The composite itself returns Ok even if some notifiers fail
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_composite_all_notifiers_fail() {
        let mut composite = CompositeNotifier::new();
        composite.add(Arc::new(FailingNotifier::new("failing1")));
        composite.add(Arc::new(FailingNotifier::new("failing2")));

        // Should not panic even when all notifiers fail
        let result = composite.notify_start(&test_issue()).await;
        // The composite returns Ok even if all notifiers fail
        assert!(result.is_ok());
    }

    // Test the default trait implementation for notify_merged
    #[tokio::test]
    async fn test_default_notify_merged() {
        // MockNotifier doesn't override notify_merged, so it uses the default
        let notifier = MockNotifier::new("test", true);
        let issue = test_issue();

        // The default implementation calls notify_status
        let result = notifier
            .notify_merged(&issue, "https://github.com/org/repo/pull/1")
            .await;
        assert!(result.is_ok());
        assert_eq!(notifier.get_call_count(), 1); // notify_status was called
    }

    // Test the default trait implementation for notify_report
    #[tokio::test]
    async fn test_default_notify_report() {
        let notifier = MockNotifier::new("test", true);
        let report = Report {
            period: "2024-01-01".to_string(),
            from: chrono::Utc::now(),
            to: chrono::Utc::now(),
            issues_attempted: 5,
            issues_succeeded: 4,
            issues_failed: 1,
            issues_cannot_fix: 0,
            success_rate: 80.0,
            failure_rate: 20.0,
            prs_created: 5,
            prs_merged: 3,
            prs_closed: 0,
            by_source: std::collections::HashMap::new(),
            pending_count: 0,
            retryable_count: 0,
        };

        // The default implementation calls notify_status with formatted text
        let result = notifier.notify_report(&report).await;
        assert!(result.is_ok());
        assert_eq!(notifier.get_call_count(), 1); // notify_status was called
    }

    #[tokio::test]
    async fn test_composite_notify_merged_broadcasts() {
        let mut composite = CompositeNotifier::new();
        let mock1 = Arc::new(MockNotifier::new("mock1", true));
        let mock2 = Arc::new(MockNotifier::new("mock2", true));
        let mock1_clone = Arc::clone(&mock1);
        let mock2_clone = Arc::clone(&mock2);
        composite.add(mock1);
        composite.add(mock2);

        let result = composite
            .notify_merged(&test_issue(), "https://github.com/test")
            .await;
        assert!(result.is_ok());
        // Each notifier's notify_merged (which calls notify_status) should be called
        assert_eq!(mock1_clone.get_call_count(), 1);
        assert_eq!(mock2_clone.get_call_count(), 1);
    }

    #[tokio::test]
    async fn test_composite_notify_report_broadcasts() {
        let mut composite = CompositeNotifier::new();
        let mock1 = Arc::new(MockNotifier::new("mock1", true));
        let mock2 = Arc::new(MockNotifier::new("mock2", true));
        let mock1_clone = Arc::clone(&mock1);
        let mock2_clone = Arc::clone(&mock2);
        composite.add(mock1);
        composite.add(mock2);

        let report = Report {
            period: "2024-01-01".to_string(),
            from: chrono::Utc::now(),
            to: chrono::Utc::now(),
            issues_attempted: 1,
            issues_succeeded: 1,
            issues_failed: 0,
            issues_cannot_fix: 0,
            success_rate: 100.0,
            failure_rate: 0.0,
            prs_created: 1,
            prs_merged: 1,
            prs_closed: 0,
            by_source: std::collections::HashMap::new(),
            pending_count: 0,
            retryable_count: 0,
        };

        let result = composite.notify_report(&report).await;
        assert!(result.is_ok());
        // Each notifier should be called
        assert_eq!(mock1_clone.get_call_count(), 1);
        assert_eq!(mock2_clone.get_call_count(), 1);
    }

    #[test]
    fn test_composite_notifier_name_is_composite() {
        let composite = CompositeNotifier::new();
        assert_eq!(composite.name(), "composite");
    }

    #[tokio::test]
    async fn test_composite_with_only_disabled_notifiers() {
        let mut composite = CompositeNotifier::new();
        composite.add(Arc::new(MockNotifier::new("disabled1", false)));
        composite.add(Arc::new(MockNotifier::new("disabled2", false)));

        // No notifiers should be added
        assert!(!composite.is_enabled());
        assert!(composite.notifiers.is_empty());

        // Operations should still succeed
        assert!(composite.notify_start(&test_issue()).await.is_ok());
    }

    #[test]
    fn test_report_format_text() {
        let report = Report {
            period: "2024-01-01".to_string(),
            from: chrono::DateTime::parse_from_rfc3339("2024-01-01T00:00:00Z")
                .unwrap()
                .with_timezone(&chrono::Utc),
            to: chrono::DateTime::parse_from_rfc3339("2024-01-01T01:00:00Z")
                .unwrap()
                .with_timezone(&chrono::Utc),
            issues_attempted: 10,
            issues_succeeded: 8,
            issues_failed: 2,
            issues_cannot_fix: 0,
            success_rate: 80.0,
            failure_rate: 20.0,
            prs_created: 10,
            prs_merged: 5,
            prs_closed: 0,
            by_source: std::collections::HashMap::new(),
            pending_count: 0,
            retryable_count: 0,
        };

        let text = report.format_text();
        assert!(!text.is_empty());
        // Should contain some of the stats
        assert!(text.contains("10") || text.contains("8") || text.contains("5"));
    }

    #[test]
    fn test_get_source_emoji_linear() {
        assert_eq!(get_source_emoji("linear"), "\u{1F4CB}");
    }

    #[test]
    fn test_get_source_emoji_sentry() {
        assert_eq!(get_source_emoji("sentry"), "\u{1F534}");
    }

    #[test]
    fn test_get_source_emoji_github() {
        assert_eq!(get_source_emoji("github"), "\u{1F419}");
    }

    #[test]
    fn test_get_source_emoji_jira() {
        assert_eq!(get_source_emoji("jira"), "\u{1F3AB}");
    }

    #[test]
    fn test_get_source_emoji_slack() {
        assert_eq!(get_source_emoji("slack"), "\u{1F4AC}");
    }

    #[test]
    fn test_get_source_emoji_unknown_source() {
        assert_eq!(get_source_emoji("unknown"), "\u{1F4CC}");
    }

    #[test]
    fn test_get_source_emoji_empty_string() {
        assert_eq!(get_source_emoji(""), "\u{1F4CC}");
    }

    #[test]
    fn test_get_source_emoji_case_insensitive() {
        assert_eq!(get_source_emoji("Linear"), "\u{1F4CB}");
        assert_eq!(get_source_emoji("SENTRY"), "\u{1F534}");
        assert_eq!(get_source_emoji("GitHub"), "\u{1F419}");
        assert_eq!(get_source_emoji("JIRA"), "\u{1F3AB}");
        assert_eq!(get_source_emoji("Slack"), "\u{1F4AC}");
    }

    #[test]
    fn test_get_source_emoji_mixed_case() {
        assert_eq!(get_source_emoji("LiNeAr"), "\u{1F4CB}");
        assert_eq!(get_source_emoji("sEnTrY"), "\u{1F534}");
    }

    #[tokio::test]
    async fn test_default_ask_question_returns_none() {
        let notifier = MockNotifier::new("test", true);
        let issue = test_issue();
        let request = AskRequest {
            correlation_id: "corr-1".to_string(),
            source: "linear".to_string(),
            repo: None,
            issue_id: "123".to_string(),
            short_id: "TEST-123".to_string(),
            question: claudear_core::types::BlockingQuestion {
                question: "What should we do?".to_string(),
                context: None,
                options: vec![],
                why: None,
            },
            asked_at: chrono::Utc::now(),
            target_discord_id: None,
            target_email: None,
            target_slack_id: None,
        };

        let result = notifier.ask_question(&issue, &request).await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_default_poll_question_replies_returns_empty() {
        let notifier = MockNotifier::new("test", true);
        let request = AskRequest {
            correlation_id: "corr-1".to_string(),
            source: "linear".to_string(),
            repo: None,
            issue_id: "123".to_string(),
            short_id: "TEST-123".to_string(),
            question: claudear_core::types::BlockingQuestion {
                question: "What?".to_string(),
                context: None,
                options: vec![],
                why: None,
            },
            asked_at: chrono::Utc::now(),
            target_discord_id: None,
            target_email: None,
            target_slack_id: None,
        };

        let result = notifier
            .poll_question_replies(&request, chrono::Utc::now())
            .await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_default_supports_replies_false() {
        let notifier = MockNotifier::new("test", true);
        assert!(!notifier.supports_replies());
    }

    struct AskMockNotifier {
        name: String,
        delivery: Option<AskDelivery>,
        replies: Vec<AskReply>,
        _supports_replies: bool,
        should_fail: bool,
    }

    impl AskMockNotifier {
        fn with_delivery(name: &str, delivery: AskDelivery) -> Self {
            Self {
                name: name.to_string(),
                delivery: Some(delivery),
                replies: vec![],
                _supports_replies: false,
                should_fail: false,
            }
        }

        fn with_replies(name: &str, replies: Vec<AskReply>) -> Self {
            Self {
                name: name.to_string(),
                delivery: None,
                replies,
                _supports_replies: true,
                should_fail: false,
            }
        }

        fn no_delivery(name: &str) -> Self {
            Self {
                name: name.to_string(),
                delivery: None,
                replies: vec![],
                _supports_replies: false,
                should_fail: false,
            }
        }

        fn failing(name: &str) -> Self {
            Self {
                name: name.to_string(),
                delivery: None,
                replies: vec![],
                _supports_replies: true,
                should_fail: true,
            }
        }
    }

    #[async_trait]
    impl Notifier for AskMockNotifier {
        fn name(&self) -> &str {
            &self.name
        }
        fn is_enabled(&self) -> bool {
            true
        }
        async fn notify_start(&self, _issue: &Issue) -> Result<()> {
            Ok(())
        }
        async fn notify_success(&self, _issue: &Issue, _pr_url: &str) -> Result<()> {
            Ok(())
        }
        async fn notify_completed(&self, _issue: &Issue) -> Result<()> {
            Ok(())
        }
        async fn notify_failed(&self, _issue: &Issue, _error: &str) -> Result<()> {
            Ok(())
        }
        async fn notify_status(&self, _message: &str) -> Result<()> {
            Ok(())
        }
        async fn notify_urgent_issues(&self, _issues: &[Issue]) -> Result<()> {
            Ok(())
        }
        async fn ask_question(
            &self,
            _issue: &Issue,
            _request: &AskRequest,
        ) -> Result<Option<AskDelivery>> {
            if self.should_fail {
                return Err(claudear_core::error::Error::config("ask failed"));
            }
            Ok(self.delivery.clone())
        }
        async fn poll_question_replies(
            &self,
            _request: &AskRequest,
            _since: DateTime<Utc>,
        ) -> Result<Vec<AskReply>> {
            if self.should_fail {
                return Err(claudear_core::error::Error::config("poll failed"));
            }
            Ok(self.replies.clone())
        }
        fn supports_replies(&self) -> bool {
            self._supports_replies
        }
    }

    fn test_ask_request() -> AskRequest {
        AskRequest {
            correlation_id: "corr-123".to_string(),
            source: "linear".to_string(),
            repo: None,
            issue_id: "123".to_string(),
            short_id: "TEST-123".to_string(),
            question: claudear_core::types::BlockingQuestion {
                question: "Should we proceed?".to_string(),
                context: Some("context".to_string()),
                options: vec!["Yes".to_string(), "No".to_string()],
                why: Some("Need confirmation".to_string()),
            },
            asked_at: chrono::Utc::now(),
            target_discord_id: None,
            target_email: None,
            target_slack_id: None,
        }
    }

    #[tokio::test]
    async fn test_composite_ask_question_empty() {
        let composite = CompositeNotifier::new();
        let result = composite
            .ask_question(&test_issue(), &test_ask_request())
            .await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_composite_ask_question_no_delivery() {
        let mut composite = CompositeNotifier::new();
        composite.add(Arc::new(AskMockNotifier::no_delivery("mock1")));
        composite.add(Arc::new(AskMockNotifier::no_delivery("mock2")));

        let result = composite
            .ask_question(&test_issue(), &test_ask_request())
            .await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_composite_ask_question_returns_first_delivery() {
        let mut composite = CompositeNotifier::new();
        let delivery = AskDelivery {
            channel: "discord".to_string(),
            target: Some("user-1".to_string()),
            message_id: Some("msg-1".to_string()),
        };
        composite.add(Arc::new(AskMockNotifier::with_delivery(
            "discord", delivery,
        )));
        composite.add(Arc::new(AskMockNotifier::no_delivery("email")));

        let result = composite
            .ask_question(&test_issue(), &test_ask_request())
            .await;
        assert!(result.is_ok());
        let delivery = result.unwrap();
        assert!(delivery.is_some());
        let d = delivery.unwrap();
        assert_eq!(d.channel, "discord");
        assert_eq!(d.target, Some("user-1".to_string()));
        assert_eq!(d.message_id, Some("msg-1".to_string()));
    }

    #[tokio::test]
    async fn test_composite_ask_question_with_multiple_deliveries() {
        let mut composite = CompositeNotifier::new();
        let delivery1 = AskDelivery {
            channel: "discord".to_string(),
            target: Some("user-1".to_string()),
            message_id: Some("msg-1".to_string()),
        };
        let delivery2 = AskDelivery {
            channel: "slack".to_string(),
            target: Some("user-2".to_string()),
            message_id: Some("msg-2".to_string()),
        };
        composite.add(Arc::new(AskMockNotifier::with_delivery(
            "discord", delivery1,
        )));
        composite.add(Arc::new(AskMockNotifier::with_delivery("slack", delivery2)));

        let result = composite
            .ask_question(&test_issue(), &test_ask_request())
            .await;
        assert!(result.is_ok());
        // Should return the first delivery encountered
        let delivery = result.unwrap();
        assert!(delivery.is_some());
    }

    #[tokio::test]
    async fn test_composite_ask_question_handles_errors_gracefully() {
        let mut composite = CompositeNotifier::new();
        composite.add(Arc::new(AskMockNotifier::failing("failing")));
        let delivery = AskDelivery {
            channel: "discord".to_string(),
            target: None,
            message_id: None,
        };
        composite.add(Arc::new(AskMockNotifier::with_delivery(
            "discord", delivery,
        )));

        let result = composite
            .ask_question(&test_issue(), &test_ask_request())
            .await;
        // Should still return Ok even when one notifier fails
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_composite_ask_question_all_fail() {
        let mut composite = CompositeNotifier::new();
        composite.add(Arc::new(AskMockNotifier::failing("fail1")));
        composite.add(Arc::new(AskMockNotifier::failing("fail2")));

        let result = composite
            .ask_question(&test_issue(), &test_ask_request())
            .await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_composite_poll_replies_empty() {
        let composite = CompositeNotifier::new();
        let result = composite
            .poll_question_replies(&test_ask_request(), chrono::Utc::now())
            .await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_composite_poll_replies_no_reply_capable_notifiers() {
        let mut composite = CompositeNotifier::new();
        // MockNotifier does not support replies
        composite.add(Arc::new(MockNotifier::new("mock1", true)));
        composite.add(Arc::new(MockNotifier::new("mock2", true)));

        let result = composite
            .poll_question_replies(&test_ask_request(), chrono::Utc::now())
            .await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_composite_poll_replies_aggregates_from_multiple() {
        let mut composite = CompositeNotifier::new();
        let replies1 = vec![AskReply {
            correlation_id: "corr-123".to_string(),
            channel: "discord".to_string(),
            responder: Some("user-1".to_string()),
            answer: "Yes".to_string(),
            replied_at: chrono::Utc::now(),
        }];
        let replies2 = vec![AskReply {
            correlation_id: "corr-123".to_string(),
            channel: "slack".to_string(),
            responder: Some("user-2".to_string()),
            answer: "No".to_string(),
            replied_at: chrono::Utc::now(),
        }];

        composite.add(Arc::new(AskMockNotifier::with_replies("discord", replies1)));
        composite.add(Arc::new(AskMockNotifier::with_replies("slack", replies2)));

        let result = composite
            .poll_question_replies(&test_ask_request(), chrono::Utc::now())
            .await;
        assert!(result.is_ok());
        let replies = result.unwrap();
        assert_eq!(replies.len(), 2);
        // Verify both channels contributed
        let channels: Vec<&str> = replies.iter().map(|r| r.channel.as_str()).collect();
        assert!(channels.contains(&"discord"));
        assert!(channels.contains(&"slack"));
    }

    #[tokio::test]
    async fn test_composite_poll_replies_filters_non_reply_notifiers() {
        let mut composite = CompositeNotifier::new();

        // This notifier does NOT support replies (supports_replies = false)
        composite.add(Arc::new(AskMockNotifier::no_delivery("no-reply")));

        // This one DOES support replies
        let replies = vec![AskReply {
            correlation_id: "corr-123".to_string(),
            channel: "discord".to_string(),
            responder: Some("user-1".to_string()),
            answer: "Go ahead".to_string(),
            replied_at: chrono::Utc::now(),
        }];
        composite.add(Arc::new(AskMockNotifier::with_replies("discord", replies)));

        let result = composite
            .poll_question_replies(&test_ask_request(), chrono::Utc::now())
            .await;
        assert!(result.is_ok());
        let replies = result.unwrap();
        // Only the reply-capable notifier's replies should appear
        assert_eq!(replies.len(), 1);
        assert_eq!(replies[0].channel, "discord");
    }

    #[tokio::test]
    async fn test_composite_poll_replies_handles_errors() {
        let mut composite = CompositeNotifier::new();
        composite.add(Arc::new(AskMockNotifier::failing("failing")));
        let replies = vec![AskReply {
            correlation_id: "corr-123".to_string(),
            channel: "slack".to_string(),
            responder: None,
            answer: "Looks good".to_string(),
            replied_at: chrono::Utc::now(),
        }];
        composite.add(Arc::new(AskMockNotifier::with_replies("slack", replies)));

        let result = composite
            .poll_question_replies(&test_ask_request(), chrono::Utc::now())
            .await;
        assert!(result.is_ok());
        // Should still get replies from the working notifier
        let replies = result.unwrap();
        assert_eq!(replies.len(), 1);
        assert_eq!(replies[0].answer, "Looks good");
    }

    #[tokio::test]
    async fn test_composite_poll_replies_all_fail() {
        let mut composite = CompositeNotifier::new();
        composite.add(Arc::new(AskMockNotifier::failing("fail1")));
        composite.add(Arc::new(AskMockNotifier::failing("fail2")));

        let result = composite
            .poll_question_replies(&test_ask_request(), chrono::Utc::now())
            .await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_composite_supports_replies_empty() {
        let composite = CompositeNotifier::new();
        assert!(!composite.supports_replies());
    }

    #[test]
    fn test_composite_supports_replies_none_support() {
        let mut composite = CompositeNotifier::new();
        // MockNotifier doesn't support replies
        composite.add(Arc::new(MockNotifier::new("mock1", true)));
        composite.add(Arc::new(MockNotifier::new("mock2", true)));
        assert!(!composite.supports_replies());
    }

    #[test]
    fn test_composite_supports_replies_some_support() {
        let mut composite = CompositeNotifier::new();
        composite.add(Arc::new(MockNotifier::new("mock1", true)));
        composite.add(Arc::new(AskMockNotifier::with_replies(
            "reply-mock",
            vec![],
        )));
        assert!(composite.supports_replies());
    }

    #[test]
    fn test_composite_supports_replies_all_support() {
        let mut composite = CompositeNotifier::new();
        composite.add(Arc::new(AskMockNotifier::with_replies("r1", vec![])));
        composite.add(Arc::new(AskMockNotifier::with_replies("r2", vec![])));
        assert!(composite.supports_replies());
    }

    #[tokio::test]
    async fn test_composite_broadcast_errors_do_not_affect_other_notifiers() {
        let mut composite = CompositeNotifier::new();
        let working = Arc::new(MockNotifier::new("working", true));
        let working_clone = Arc::clone(&working);
        composite.add(Arc::new(FailingNotifier::new("failing")));
        composite.add(working);

        // All methods should succeed and the working notifier should still be called
        assert!(composite
            .notify_success(&test_issue(), "https://pr.url")
            .await
            .is_ok());
        assert_eq!(working_clone.get_call_count(), 1);

        assert!(composite.notify_completed(&test_issue()).await.is_ok());
        assert_eq!(working_clone.get_call_count(), 2);

        assert!(composite
            .notify_failed(&test_issue(), "error msg")
            .await
            .is_ok());
        assert_eq!(working_clone.get_call_count(), 3);

        assert!(composite.notify_status("status").await.is_ok());
        assert_eq!(working_clone.get_call_count(), 4);

        assert!(composite
            .notify_urgent_issues(&[test_issue()])
            .await
            .is_ok());
        assert_eq!(working_clone.get_call_count(), 5);

        assert!(composite
            .notify_merged(&test_issue(), "https://pr.url")
            .await
            .is_ok());
        assert_eq!(working_clone.get_call_count(), 6);
    }

    #[tokio::test]
    async fn test_composite_multiple_broadcasts_accumulate() {
        let mut composite = CompositeNotifier::new();
        let mock = Arc::new(MockNotifier::new("mock", true));
        let mock_clone = Arc::clone(&mock);
        composite.add(mock);

        composite.notify_start(&test_issue()).await.unwrap();
        composite.notify_start(&test_issue()).await.unwrap();
        composite.notify_start(&test_issue()).await.unwrap();

        assert_eq!(mock_clone.get_call_count(), 3);
    }

    #[tokio::test]
    async fn test_composite_all_fail_for_every_method() {
        let mut composite = CompositeNotifier::new();
        composite.add(Arc::new(FailingNotifier::new("fail1")));
        composite.add(Arc::new(FailingNotifier::new("fail2")));

        assert!(composite.notify_start(&test_issue()).await.is_ok());
        assert!(composite.notify_success(&test_issue(), "url").await.is_ok());
        assert!(composite.notify_completed(&test_issue()).await.is_ok());
        assert!(composite.notify_failed(&test_issue(), "err").await.is_ok());
        assert!(composite.notify_status("msg").await.is_ok());
        assert!(composite.notify_urgent_issues(&[]).await.is_ok());
        assert!(composite.notify_merged(&test_issue(), "url").await.is_ok());
    }

    #[tokio::test]
    async fn test_notifier_as_trait_object() {
        let mock: Arc<dyn Notifier> = Arc::new(MockNotifier::new("dyn_test", true));
        assert_eq!(mock.name(), "dyn_test");
        assert!(mock.is_enabled());
        assert!(!mock.supports_replies());
        assert!(mock.notify_start(&test_issue()).await.is_ok());
    }

    #[tokio::test]
    async fn test_composite_as_trait_object() {
        let mut composite = CompositeNotifier::new();
        composite.add(Arc::new(MockNotifier::new("inner", true)));

        let notifier: Arc<dyn Notifier> = Arc::new(composite);
        assert_eq!(notifier.name(), "composite");
        assert!(notifier.is_enabled());
        assert!(notifier.notify_start(&test_issue()).await.is_ok());
    }

    #[test]
    fn test_composite_is_enabled_reflects_internal_state() {
        let mut composite = CompositeNotifier::new();
        assert!(!composite.is_enabled());

        composite.add(Arc::new(MockNotifier::new("disabled", false)));
        assert!(!composite.is_enabled()); // disabled not added

        composite.add(Arc::new(MockNotifier::new("enabled", true)));
        assert!(composite.is_enabled());
    }

    #[tokio::test]
    async fn test_composite_broadcast_many_notifiers() {
        let mut composite = CompositeNotifier::new();
        let mocks: Vec<Arc<MockNotifier>> = (0..20)
            .map(|i| Arc::new(MockNotifier::new(&format!("mock{}", i), true)))
            .collect();

        for mock in &mocks {
            composite.add(Arc::clone(mock) as Arc<dyn Notifier>);
        }

        composite
            .notify_failed(&test_issue(), "test error")
            .await
            .unwrap();

        for (i, mock) in mocks.iter().enumerate() {
            assert_eq!(
                mock.get_call_count(),
                1,
                "Mock {} should have been called exactly once",
                i
            );
        }
    }

    #[tokio::test]
    async fn test_composite_poll_replies_multiple_per_notifier() {
        let mut composite = CompositeNotifier::new();
        let replies = vec![
            AskReply {
                correlation_id: "corr-1".to_string(),
                channel: "discord".to_string(),
                responder: Some("alice".to_string()),
                answer: "Yes".to_string(),
                replied_at: chrono::Utc::now(),
            },
            AskReply {
                correlation_id: "corr-1".to_string(),
                channel: "discord".to_string(),
                responder: Some("bob".to_string()),
                answer: "No".to_string(),
                replied_at: chrono::Utc::now(),
            },
            AskReply {
                correlation_id: "corr-1".to_string(),
                channel: "discord".to_string(),
                responder: Some("charlie".to_string()),
                answer: "Maybe".to_string(),
                replied_at: chrono::Utc::now(),
            },
        ];
        composite.add(Arc::new(AskMockNotifier::with_replies("discord", replies)));

        let result = composite
            .poll_question_replies(&test_ask_request(), chrono::Utc::now())
            .await;
        assert!(result.is_ok());
        let all_replies = result.unwrap();
        assert_eq!(all_replies.len(), 3);
    }

    #[tokio::test]
    async fn test_composite_ask_question_delivery_with_no_target() {
        let mut composite = CompositeNotifier::new();
        let delivery = AskDelivery {
            channel: "push".to_string(),
            target: None,
            message_id: None,
        };
        composite.add(Arc::new(AskMockNotifier::with_delivery("push", delivery)));

        let result = composite
            .ask_question(&test_issue(), &test_ask_request())
            .await;
        assert!(result.is_ok());
        let d = result.unwrap().unwrap();
        assert_eq!(d.channel, "push");
        assert!(d.target.is_none());
        assert!(d.message_id.is_none());
    }

    #[tokio::test]
    async fn test_default_notify_merged_formats_message() {
        // Create a notifier that captures the status message
        struct CapturingNotifier {
            last_message: std::sync::Mutex<Option<String>>,
        }

        #[async_trait]
        impl Notifier for CapturingNotifier {
            fn name(&self) -> &str {
                "capturing"
            }
            fn is_enabled(&self) -> bool {
                true
            }
            async fn notify_start(&self, _issue: &Issue) -> Result<()> {
                Ok(())
            }
            async fn notify_success(&self, _issue: &Issue, _pr_url: &str) -> Result<()> {
                Ok(())
            }
            async fn notify_completed(&self, _issue: &Issue) -> Result<()> {
                Ok(())
            }
            async fn notify_failed(&self, _issue: &Issue, _error: &str) -> Result<()> {
                Ok(())
            }
            async fn notify_status(&self, message: &str) -> Result<()> {
                *self.last_message.lock().unwrap() = Some(message.to_string());
                Ok(())
            }
            async fn notify_urgent_issues(&self, _issues: &[Issue]) -> Result<()> {
                Ok(())
            }
        }

        let notifier = CapturingNotifier {
            last_message: std::sync::Mutex::new(None),
        };

        let issue = test_issue();
        notifier
            .notify_merged(&issue, "https://github.com/org/repo/pull/42")
            .await
            .unwrap();

        let msg = notifier.last_message.lock().unwrap().clone().unwrap();
        assert!(msg.contains("TEST-123"));
        assert!(msg.contains("https://github.com/org/repo/pull/42"));
        assert!(msg.contains("PR merged"));
    }

    #[tokio::test]
    async fn test_composite_notify_urgent_issues_empty_list() {
        let mut composite = CompositeNotifier::new();
        let mock = Arc::new(MockNotifier::new("mock", true));
        let mock_clone = Arc::clone(&mock);
        composite.add(mock);

        let result = composite.notify_urgent_issues(&[]).await;
        assert!(result.is_ok());
        assert_eq!(mock_clone.get_call_count(), 1);
    }

    #[tokio::test]
    async fn test_composite_single_notifier_all_methods() {
        let mut composite = CompositeNotifier::new();
        let mock = Arc::new(MockNotifier::new("solo", true));
        let mock_clone = Arc::clone(&mock);
        composite.add(mock);

        composite.notify_start(&test_issue()).await.unwrap();
        composite
            .notify_success(&test_issue(), "pr-url")
            .await
            .unwrap();
        composite.notify_completed(&test_issue()).await.unwrap();
        composite.notify_failed(&test_issue(), "err").await.unwrap();
        composite.notify_status("msg").await.unwrap();
        composite.notify_urgent_issues(&[]).await.unwrap();
        composite
            .notify_merged(&test_issue(), "pr-url")
            .await
            .unwrap();

        // notify_merged default calls notify_status, so 7 total calls
        // (start + success + completed + failed + status + urgent + merged->status)
        assert_eq!(mock_clone.get_call_count(), 7);
    }

    #[tokio::test]
    async fn test_default_notify_closed() {
        let notifier = MockNotifier::new("test", true);
        let issue = test_issue();
        let result = notifier
            .notify_closed(&issue, "https://github.com/org/repo/pull/1")
            .await;
        assert!(result.is_ok());
        assert_eq!(notifier.get_call_count(), 1); // notify_status was called
    }

    #[tokio::test]
    async fn test_composite_notify_closed() {
        let mut composite = CompositeNotifier::new();
        let mock = Arc::new(MockNotifier::new("mock", true));
        let mock_clone = Arc::clone(&mock);
        composite.add(mock);

        let result = composite
            .notify_closed(&test_issue(), "https://github.com/org/repo/pull/1")
            .await;
        assert!(result.is_ok());
        assert_eq!(mock_clone.get_call_count(), 1);
    }

    #[tokio::test]
    async fn test_composite_notify_closed_broadcasts_to_all() {
        let mut composite = CompositeNotifier::new();
        let mock1 = Arc::new(MockNotifier::new("mock1", true));
        let mock2 = Arc::new(MockNotifier::new("mock2", true));
        let mock1_clone = Arc::clone(&mock1);
        let mock2_clone = Arc::clone(&mock2);
        composite.add(mock1);
        composite.add(mock2);

        composite
            .notify_closed(&test_issue(), "https://pr.url")
            .await
            .unwrap();
        assert_eq!(mock1_clone.get_call_count(), 1);
        assert_eq!(mock2_clone.get_call_count(), 1);
    }
}
