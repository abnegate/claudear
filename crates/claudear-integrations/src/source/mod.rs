//! Issue source implementations.

mod discord;
mod github;
mod gitlab;
mod helpscout;
mod jira;
mod linear;
pub mod sentry;
mod slack;
mod telegram;
mod whatsapp;

pub use discord::DiscordSource;
pub use github::GitHubSource;
pub use gitlab::GitLabSource;
pub(crate) use gitlab::{format_gitlab_context, gitlab_matches_criteria};
pub use helpscout::HelpScoutSource;
pub use jira::JiraSource;
pub use linear::LinearSource;
pub use sentry::SentrySource;
pub use slack::SlackSource;
pub use telegram::TelegramSource;
pub use whatsapp::{WhatsAppMessage, WhatsAppSource};

use async_trait::async_trait;
use claudear_core::error::Result;
use claudear_core::types::{Issue, MatchResult};
use std::collections::HashMap;
use std::sync::Arc;

/// Trait for issue sources (Linear, Sentry, GitHub, Jira, etc.)
#[async_trait]
pub trait IssueSource: Send + Sync {
    /// Unique name for this source.
    fn name(&self) -> &str;

    /// Human-readable display name.
    fn display_name(&self) -> &str;

    /// Fetch issues that should be considered for processing.
    async fn fetch_issues(&self) -> Result<Vec<Issue>>;

    /// Check if a specific issue matches the processing criteria.
    fn matches_criteria(&self, issue: &Issue) -> MatchResult;

    /// Build context string for Claude about an issue.
    async fn build_issue_context(&self, issue: &Issue) -> Result<String>;

    /// Fetch a single issue by ID.
    async fn get_issue(&self, issue_id: &str) -> Result<Issue>;

    /// Resolve/close an issue on the remote source.
    /// Called when a PR is merged and auto-resolve is enabled.
    async fn resolve_issue(&self, issue_id: &str) -> Result<()> {
        // Default implementation does nothing - not all sources support this
        let _ = issue_id;
        Ok(())
    }

    /// Add a comment to an issue on the remote source.
    async fn add_comment(&self, issue_id: &str, comment: &str) -> Result<()> {
        // Default implementation does nothing - not all sources support this
        let _ = (issue_id, comment);
        Ok(())
    }

    /// Get the current status of an issue from the remote source.
    /// Returns the raw status string from the source (e.g., "completed", "resolved", "ignored").
    async fn get_issue_status(&self, issue_id: &str) -> Result<String> {
        // Default implementation fetches the full issue and returns its status
        let issue = self.get_issue(issue_id).await?;
        Ok(issue.status.to_string())
    }

    /// Check if a status string represents a terminal state for this source.
    /// Terminal states are those where no further action is needed (e.g., completed, cancelled, resolved, ignored).
    fn is_terminal_status(&self, status: &str) -> bool {
        // Default implementation treats common terminal status names
        let s = status.to_lowercase();
        s == "completed"
            || s == "resolved"
            || s == "cancelled"
            || s == "canceled"
            || s == "ignored"
            || s == "closed"
            || s == "done"
    }

    /// Create a new issue on the remote source.
    async fn create_issue(
        &self,
        _title: &str,
        _description: &str,
        _labels: &[String],
    ) -> Result<Issue> {
        Err(claudear_core::error::Error::Other(
            "create_issue not supported by this source".into(),
        ))
    }

    /// Find an existing label by name, or create it if it doesn't exist.
    /// Returns the label ID.
    async fn find_or_create_label(&self, _name: &str) -> Result<String> {
        Err(claudear_core::error::Error::Other(
            "find_or_create_label not supported by this source".into(),
        ))
    }

    /// List open issues, optionally filtering by title substring.
    async fn list_open_issues(&self, _title_filter: &str) -> Result<Vec<Issue>> {
        Err(claudear_core::error::Error::Other(
            "list_open_issues not supported by this source".into(),
        ))
    }
}

/// Registry for available sources.
pub struct SourceRegistry {
    sources: HashMap<String, Arc<dyn IssueSource>>,
}

impl SourceRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            sources: HashMap::new(),
        }
    }

    /// Register a source.
    pub fn register(&mut self, source: Arc<dyn IssueSource>) {
        self.sources.insert(source.name().to_string(), source);
    }

    /// Get a source by name.
    pub fn get(&self, name: &str) -> Option<&Arc<dyn IssueSource>> {
        self.sources.get(name)
    }

    /// Get all registered sources.
    pub fn get_all(&self) -> Vec<&Arc<dyn IssueSource>> {
        self.sources.values().collect()
    }

    /// Check if a source is registered.
    pub fn has(&self, name: &str) -> bool {
        self.sources.contains_key(name)
    }

    /// Get all source names.
    pub fn names(&self) -> Vec<&str> {
        self.sources.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for SourceRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use claudear_core::types::{MatchPriority, MatchResult};

    struct MockSource {
        name: String,
        display: String,
    }

    impl MockSource {
        fn new(name: &str, display: &str) -> Self {
            Self {
                name: name.to_string(),
                display: display.to_string(),
            }
        }
    }

    #[async_trait]
    impl IssueSource for MockSource {
        fn name(&self) -> &str {
            &self.name
        }
        fn display_name(&self) -> &str {
            &self.display
        }
        async fn fetch_issues(&self) -> Result<Vec<Issue>> {
            Ok(vec![])
        }
        fn matches_criteria(&self, _issue: &Issue) -> MatchResult {
            MatchResult::matched("test", MatchPriority::Normal)
        }
        async fn build_issue_context(&self, _issue: &Issue) -> Result<String> {
            Ok(String::new())
        }
        async fn get_issue(&self, _id: &str) -> Result<Issue> {
            Err(claudear_core::error::Error::issue_not_found(
                &self.name, "test",
            ))
        }
    }

    #[test]
    fn test_source_registry_new() {
        let registry = SourceRegistry::new();
        assert!(registry.sources.is_empty());
        assert!(registry.get_all().is_empty());
    }

    #[test]
    fn test_source_registry_default() {
        let registry = SourceRegistry::default();
        assert!(registry.sources.is_empty());
    }

    #[test]
    fn test_source_registry_register() {
        let mut registry = SourceRegistry::new();
        let source = Arc::new(MockSource::new("test", "Test"));
        registry.register(source);
        assert!(registry.has("test"));
        assert!(!registry.has("other"));
    }

    #[test]
    fn test_source_registry_get() {
        let mut registry = SourceRegistry::new();
        let source = Arc::new(MockSource::new("linear", "Linear"));
        registry.register(source);

        let retrieved = registry.get("linear");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().name(), "linear");
        assert!(registry.get("sentry").is_none());
    }

    #[test]
    fn test_source_registry_get_all() {
        let mut registry = SourceRegistry::new();
        registry.register(Arc::new(MockSource::new("linear", "Linear")));
        registry.register(Arc::new(MockSource::new("sentry", "Sentry")));

        let all = registry.get_all();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_source_registry_has() {
        let mut registry = SourceRegistry::new();
        registry.register(Arc::new(MockSource::new("test", "Test")));

        assert!(registry.has("test"));
        assert!(!registry.has("nonexistent"));
    }

    #[test]
    fn test_source_registry_names() {
        let mut registry = SourceRegistry::new();
        registry.register(Arc::new(MockSource::new("a", "A")));
        registry.register(Arc::new(MockSource::new("b", "B")));
        registry.register(Arc::new(MockSource::new("c", "C")));

        let names = registry.names();
        assert_eq!(names.len(), 3);
        assert!(names.contains(&"a"));
        assert!(names.contains(&"b"));
        assert!(names.contains(&"c"));
    }

    #[test]
    fn test_source_registry_overwrite() {
        let mut registry = SourceRegistry::new();
        registry.register(Arc::new(MockSource::new("test", "Test1")));
        registry.register(Arc::new(MockSource::new("test", "Test2")));

        // Should overwrite, HashMap behavior
        assert_eq!(registry.get_all().len(), 1);
        assert_eq!(registry.get("test").unwrap().display_name(), "Test2");
    }

    #[test]
    fn test_source_registry_empty_names() {
        let registry = SourceRegistry::new();
        assert!(registry.names().is_empty());
    }

    #[tokio::test]
    async fn test_mock_source_fetch_issues() {
        let source = MockSource::new("test", "Test");
        let issues = source.fetch_issues().await.unwrap();
        assert!(issues.is_empty());
    }

    #[tokio::test]
    async fn test_mock_source_get_issue() {
        let source = MockSource::new("test", "Test");
        let result = source.get_issue("123").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_mock_source_build_context() {
        let source = MockSource::new("test", "Test");
        let issue = Issue::new("1", "T-1", "Test", "http://test.com", "test");
        let context = source.build_issue_context(&issue).await.unwrap();
        assert!(context.is_empty());
    }

    #[test]
    fn test_mock_source_matches_criteria() {
        let source = MockSource::new("test", "Test");
        let issue = Issue::new("1", "T-1", "Test", "http://test.com", "test");
        let result = source.matches_criteria(&issue);
        assert!(result.matches);
    }

    #[tokio::test]
    async fn test_default_resolve_issue() {
        let source = MockSource::new("test", "Test");
        // Default implementation should return Ok(())
        let result = source.resolve_issue("123").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_default_add_comment() {
        let source = MockSource::new("test", "Test");
        // Default implementation should return Ok(())
        let result = source.add_comment("123", "Test comment").await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_source_name_display() {
        let source = MockSource::new("sentry", "Sentry Errors");
        assert_eq!(source.name(), "sentry");
        assert_eq!(source.display_name(), "Sentry Errors");
    }
}
