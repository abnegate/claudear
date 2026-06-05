//! Webhook handlers and HTTP server.

mod configurator;
mod github;
mod gitlab;
mod helpscout;
mod jira;
mod linear;
mod linear_api;
mod sentry;
mod sentry_api;
mod slack;
mod telegram;
mod whatsapp;

pub use configurator::{print_setup_result, WebhookConfigurator, WebhookSetupResult};
pub use github::{GitHubWebhookHandler, WebhookAction};
pub use gitlab::{GitLabIssueWebhookHandler, GitLabMrWebhookHandler};
pub use helpscout::HelpScoutWebhookHandler;
pub use jira::JiraWebhookHandler;
pub use linear::LinearWebhookHandler;
pub use linear_api::{LinearApiClient, WebhookRegistration};
pub use sentry::SentryWebhookHandler;
pub(crate) use sentry::{map_priority as sentry_map_priority, map_status as sentry_map_status};
pub use sentry_api::{SentryApiClient, SentryWebhookRegistration};
pub use slack::SlackWebhookHandler;
pub use telegram::TelegramWebhookHandler;
pub use whatsapp::WhatsAppWebhookHandler;

use async_trait::async_trait;
use claudear_core::error::Result;
use claudear_core::types::{Issue, MatchResult};
use std::collections::HashMap;
use std::sync::Arc;

/// Trait for webhook handlers.
#[async_trait]
pub trait WebhookHandler: Send + Sync {
    /// Source name this handler is for.
    fn source_name(&self) -> &str;

    /// Verify the webhook signature/authenticity.
    fn verify_signature(&self, payload: &[u8], headers: &HashMap<String, String>) -> bool;

    /// Parse the webhook payload into an Issue.
    /// Returns None if this webhook should be ignored.
    async fn parse_payload(&self, payload: &serde_json::Value) -> Result<Option<Issue>>;

    /// Check if the parsed issue matches processing criteria.
    fn matches_criteria(&self, issue: &Issue) -> MatchResult;

    /// Build context for Claude from the issue.
    async fn build_issue_context(&self, issue: &Issue) -> Result<String>;
}

/// Registry for webhook handlers.
pub struct WebhookHandlerRegistry {
    handlers: HashMap<String, Arc<dyn WebhookHandler>>,
}

impl WebhookHandlerRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            handlers: HashMap::new(),
        }
    }

    /// Register a handler.
    pub fn register(&mut self, handler: Arc<dyn WebhookHandler>) {
        self.handlers
            .insert(handler.source_name().to_string(), handler);
    }

    /// Get a handler by source name.
    pub fn get(&self, source_name: &str) -> Option<&Arc<dyn WebhookHandler>> {
        self.handlers.get(source_name)
    }

    /// Get all registered handlers.
    pub fn get_all(&self) -> Vec<&Arc<dyn WebhookHandler>> {
        self.handlers.values().collect()
    }

    /// Check if a handler is registered.
    pub fn has(&self, source_name: &str) -> bool {
        self.handlers.contains_key(source_name)
    }
}

impl Default for WebhookHandlerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use claudear_core::error::Result;
    use claudear_core::types::{Issue, MatchResult};

    struct MockHandler {
        name: String,
    }

    impl MockHandler {
        fn new(name: &str) -> Self {
            Self {
                name: name.to_string(),
            }
        }
    }

    #[async_trait]
    impl WebhookHandler for MockHandler {
        fn source_name(&self) -> &str {
            &self.name
        }

        fn verify_signature(&self, _payload: &[u8], _headers: &HashMap<String, String>) -> bool {
            true
        }

        async fn parse_payload(&self, _payload: &serde_json::Value) -> Result<Option<Issue>> {
            Ok(None)
        }

        fn matches_criteria(&self, _issue: &Issue) -> MatchResult {
            MatchResult::matched("test", claudear_core::types::MatchPriority::Normal)
        }

        async fn build_issue_context(&self, _issue: &Issue) -> Result<String> {
            Ok(String::new())
        }
    }

    #[test]
    fn test_registry_new_is_empty() {
        let registry = WebhookHandlerRegistry::new();
        assert!(registry.get_all().is_empty());
        assert!(!registry.has("anything"));
    }

    #[test]
    fn test_registry_default_is_empty() {
        let registry = WebhookHandlerRegistry::default();
        assert!(registry.get_all().is_empty());
    }

    #[test]
    fn test_registry_register_and_get() {
        let mut registry = WebhookHandlerRegistry::new();
        registry.register(Arc::new(MockHandler::new("linear")));

        assert!(registry.has("linear"));
        assert!(registry.get("linear").is_some());
        assert_eq!(registry.get("linear").unwrap().source_name(), "linear");
    }

    #[test]
    fn test_registry_get_nonexistent() {
        let registry = WebhookHandlerRegistry::new();
        assert!(registry.get("nonexistent").is_none());
        assert!(!registry.has("nonexistent"));
    }

    #[test]
    fn test_registry_register_multiple() {
        let mut registry = WebhookHandlerRegistry::new();
        registry.register(Arc::new(MockHandler::new("linear")));
        registry.register(Arc::new(MockHandler::new("sentry")));
        registry.register(Arc::new(MockHandler::new("github")));

        assert_eq!(registry.get_all().len(), 3);
        assert!(registry.has("linear"));
        assert!(registry.has("sentry"));
        assert!(registry.has("github"));
    }

    #[test]
    fn test_registry_register_duplicate_overwrites() {
        let mut registry = WebhookHandlerRegistry::new();
        registry.register(Arc::new(MockHandler::new("linear")));
        registry.register(Arc::new(MockHandler::new("linear")));

        // Should still only have one handler
        assert_eq!(registry.get_all().len(), 1);
        assert!(registry.has("linear"));
    }

    #[test]
    fn test_registry_empty_source_name() {
        let mut registry = WebhookHandlerRegistry::new();
        registry.register(Arc::new(MockHandler::new("")));

        assert!(registry.has(""));
        assert!(registry.get("").is_some());
        assert_eq!(registry.get_all().len(), 1);
    }

    #[test]
    fn test_registry_has_case_sensitive() {
        let mut registry = WebhookHandlerRegistry::new();
        registry.register(Arc::new(MockHandler::new("Linear")));

        assert!(registry.has("Linear"));
        assert!(!registry.has("linear"));
        assert!(!registry.has("LINEAR"));
    }
}
