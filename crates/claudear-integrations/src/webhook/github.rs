//! GitHub webhook handler for PR review and pull request events.
//!
//! This handler processes `pull_request_review`, `pull_request_review_comment`,
//! `issue_comment` (PR conversation comments), and `pull_request` events from
//! GitHub webhooks to trigger review processing and detect PR merges/closes in
//! real-time instead of relying solely on polling.

use crate::scm::{is_skippable_bot, CodeReview, ReviewComment, ReviewUser, ReviewWatcher};
use claudear_config::config::GitHubConfig;
use claudear_core::error::Result;
use hmac::{Hmac, Mac};
use sha2::Sha256;
use std::collections::HashMap;
use std::sync::Arc;
use subtle::ConstantTimeEq;

/// Result of processing a GitHub webhook event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WebhookAction {
    /// The event was processed successfully (e.g. review forwarded to ReviewWatcher).
    Processed,
    /// The event was ignored (unrecognized event type, unwatched PR, etc.).
    Ignored,
    /// A pull request was closed (merged or closed without merge).
    PrClosed { pr_url: String, merged: bool },
}

impl WebhookAction {
    /// Returns `true` when the event was actively processed (not ignored).
    pub fn is_processed(&self) -> bool {
        !matches!(self, WebhookAction::Ignored)
    }
}

/// GitHub webhook handler for PR review and pull request events.
///
/// Unlike other webhook handlers, this one doesn't implement the `WebhookHandler` trait
/// because it doesn't create issues - it processes PR review events and integrates
/// directly with the ReviewWatcher.
pub struct GitHubWebhookHandler {
    config: GitHubConfig,
    review_watcher: Option<Arc<ReviewWatcher>>,
}

impl GitHubWebhookHandler {
    /// Create a new GitHub webhook handler.
    pub fn new(config: GitHubConfig, review_watcher: Option<Arc<ReviewWatcher>>) -> Self {
        Self {
            config,
            review_watcher,
        }
    }

    /// Get the source name.
    pub fn source_name(&self) -> &str {
        "github"
    }

    fn webhook_secret(&self) -> Option<&claudear_core::secret::SecretValue> {
        self.config
            .webhook_secret
            .as_ref()
            .filter(|s| !s.expose().is_empty())
            .or_else(|| {
                self.config
                    .app
                    .webhook_secret
                    .as_ref()
                    .filter(|s| !s.expose().is_empty())
            })
    }

    /// Verify the webhook signature using HMAC-SHA256.
    ///
    /// GitHub uses `x-hub-signature-256` header with format: `sha256=<hex>`
    pub fn verify_signature(&self, payload: &[u8], headers: &HashMap<String, String>) -> bool {
        let secret = match self.webhook_secret() {
            Some(s) => s,
            None => {
                tracing::error!(
                    source = "github",
                    "No webhook secret configured - rejecting request for security"
                );
                return false;
            }
        };

        // Headers are lowercased by the webhook server
        let signature = match headers.get("x-hub-signature-256") {
            Some(s) => s,
            None => {
                tracing::warn!(source = "github", "Missing x-hub-signature-256 header");
                return false;
            }
        };

        // Signature format: sha256=<hex>
        let signature_hex = match signature.strip_prefix("sha256=") {
            Some(hex) => hex,
            None => {
                tracing::warn!(
                    source = "github",
                    "Invalid signature format - expected sha256= prefix"
                );
                return false;
            }
        };

        let mut mac = match Hmac::<Sha256>::new_from_slice(secret.expose().as_bytes()) {
            Ok(m) => m,
            Err(_) => {
                tracing::error!(source = "github", "Failed to create HMAC from secret");
                return false;
            }
        };

        mac.update(payload);
        let expected = mac.finalize().into_bytes();
        let expected_hex = hex::encode(expected);

        signature_hex
            .as_bytes()
            .ct_eq(expected_hex.as_bytes())
            .into()
    }

    /// Check if this handler is enabled (has webhook secret and review watcher).
    pub fn is_enabled(&self) -> bool {
        self.webhook_secret().is_some() && self.review_watcher.is_some()
    }

    /// Get the event type from headers.
    /// Headers are expected to be lowercased by the webhook server.
    pub fn get_event_type(headers: &HashMap<String, String>) -> Option<&str> {
        headers.get("x-github-event").map(|s| s.as_str())
    }

    /// Process a webhook payload.
    ///
    /// SECURITY: This method verifies the webhook signature before processing.
    /// The raw_payload must be the exact bytes received from GitHub (before JSON parsing)
    /// to ensure signature verification is accurate.
    ///
    /// Returns a `WebhookAction` describing what happened, or an error if
    /// signature verification or processing failed.
    pub async fn process_webhook(
        &self,
        raw_payload: &[u8],
        payload: &serde_json::Value,
        headers: &HashMap<String, String>,
    ) -> Result<WebhookAction> {
        // CRITICAL: Verify signature before processing any webhook data
        if !self.verify_signature(raw_payload, headers) {
            tracing::warn!(
                source = "github",
                "Webhook signature verification failed - rejecting request"
            );
            return Err(claudear_core::error::Error::Webhook(
                "Invalid webhook signature".to_string(),
            ));
        }

        let event_type = match Self::get_event_type(headers) {
            Some(t) => t,
            None => {
                tracing::warn!(source = "github", "Missing x-github-event header");
                return Ok(WebhookAction::Ignored);
            }
        };

        match event_type {
            "pull_request_review" => self.handle_review_submitted(payload).await,
            "pull_request_review_comment" => self.handle_review_comment(payload).await,
            "issue_comment" => self.handle_issue_comment(payload).await,
            "pull_request" => self.handle_pull_request(payload).await,
            _ => {
                tracing::debug!(
                    source = "github",
                    event_type = %event_type,
                    "Ignoring unhandled event"
                );
                Ok(WebhookAction::Ignored)
            }
        }
    }

    /// Handle a pull_request_review.submitted event.
    async fn handle_review_submitted(&self, payload: &serde_json::Value) -> Result<WebhookAction> {
        let action = payload.get("action").and_then(|v| v.as_str()).unwrap_or("");

        if action != "submitted" {
            tracing::debug!(
                source = "github",
                action = %action,
                "Ignoring non-submitted review action"
            );
            return Ok(WebhookAction::Ignored);
        }

        let review = match payload.get("review") {
            Some(r) => r,
            None => {
                tracing::warn!(source = "github", "Missing review in payload");
                return Ok(WebhookAction::Ignored);
            }
        };

        let pr = match payload.get("pull_request") {
            Some(p) => p,
            None => {
                tracing::warn!(source = "github", "Missing pull_request in payload");
                return Ok(WebhookAction::Ignored);
            }
        };

        let pr_url = pr
            .get("html_url")
            .and_then(|v| v.as_str())
            .unwrap_or_default();

        let review_watcher = match &self.review_watcher {
            Some(rw) => rw,
            None => {
                tracing::debug!(
                    source = "github",
                    pr_url = %pr_url,
                    "ReviewWatcher not available, ignoring event"
                );
                return Ok(WebhookAction::Ignored);
            }
        };

        // Check if we're watching this PR
        let state = match review_watcher.get_state(pr_url) {
            Some(s) if s.is_active => s,
            _ => {
                tracing::debug!(
                    source = "github",
                    pr_url = %pr_url,
                    "PR not being watched, ignoring review"
                );
                return Ok(WebhookAction::Ignored);
            }
        };

        // Parse the review
        let pr_review = self.parse_review(review)?;

        // Skip bot reviews (unless the bot is in the allowed list)
        if is_skippable_bot(&pr_review.user, &self.config.allowed_bots) {
            tracing::debug!(
                source = "github",
                pr_url = %pr_url,
                reviewer = %pr_review.user.login,
                "Skipping bot review"
            );
            return Ok(WebhookAction::Ignored);
        }

        // Skip pending reviews
        if pr_review.state.to_uppercase() == "PENDING" {
            tracing::debug!(
                source = "github",
                pr_url = %pr_url,
                "Skipping pending review"
            );
            return Ok(WebhookAction::Ignored);
        }

        tracing::info!(
            source = "github",
            pr_url = %pr_url,
            reviewer = %pr_review.user.login,
            state = %pr_review.state,
            issue_id = %state.issue_id,
            "Received PR review via webhook"
        );

        let processed_events = review_watcher.check_for_pr(pr_url).await?;
        tracing::info!(
            source = "github",
            pr_url = %pr_url,
            events = processed_events.len(),
            "Processed review webhook through ReviewWatcher"
        );

        Ok(WebhookAction::Processed)
    }

    /// Handle a pull_request_review_comment.created event.
    async fn handle_review_comment(&self, payload: &serde_json::Value) -> Result<WebhookAction> {
        let action = payload.get("action").and_then(|v| v.as_str()).unwrap_or("");

        if action != "created" {
            tracing::debug!(
                source = "github",
                action = %action,
                "Ignoring non-created comment action"
            );
            return Ok(WebhookAction::Ignored);
        }

        let comment = match payload.get("comment") {
            Some(c) => c,
            None => {
                tracing::warn!(source = "github", "Missing comment in payload");
                return Ok(WebhookAction::Ignored);
            }
        };

        let pr = match payload.get("pull_request") {
            Some(p) => p,
            None => {
                tracing::warn!(source = "github", "Missing pull_request in payload");
                return Ok(WebhookAction::Ignored);
            }
        };

        let pr_url = pr
            .get("html_url")
            .and_then(|v| v.as_str())
            .unwrap_or_default();

        let review_watcher = match &self.review_watcher {
            Some(rw) => rw,
            None => {
                tracing::debug!(
                    source = "github",
                    pr_url = %pr_url,
                    "ReviewWatcher not available, ignoring event"
                );
                return Ok(WebhookAction::Ignored);
            }
        };

        // Check if we're watching this PR
        let state = match review_watcher.get_state(pr_url) {
            Some(s) if s.is_active => s,
            _ => {
                tracing::debug!(
                    source = "github",
                    pr_url = %pr_url,
                    "PR not being watched, ignoring comment"
                );
                return Ok(WebhookAction::Ignored);
            }
        };

        // Parse the comment
        let pr_comment = self.parse_comment(comment)?;

        // Skip bot comments (unless the bot is in the allowed list)
        if is_skippable_bot(&pr_comment.user, &self.config.allowed_bots) {
            tracing::debug!(
                source = "github",
                pr_url = %pr_url,
                author = %pr_comment.user.login,
                "Skipping bot comment"
            );
            return Ok(WebhookAction::Ignored);
        }

        tracing::info!(
            source = "github",
            pr_url = %pr_url,
            author = %pr_comment.user.login,
            path = %pr_comment.path,
            issue_id = %state.issue_id,
            "Received PR review comment via webhook"
        );

        let processed_events = review_watcher.check_for_pr(pr_url).await?;
        tracing::info!(
            source = "github",
            pr_url = %pr_url,
            events = processed_events.len(),
            "Processed review comment webhook through ReviewWatcher"
        );

        Ok(WebhookAction::Processed)
    }

    /// Handle an `issue_comment.created` event.
    ///
    /// GitHub fires this for comments on the PR *conversation* timeline (a plain
    /// "@claudear fix this" left outside a formal review or inline thread). Only
    /// comments on pull requests are relevant, so non-PR issue comments are
    /// ignored. When the comment mentions the review trigger on a watched PR, we
    /// re-poll via `check_for_pr`, which now picks up conversation comments.
    async fn handle_issue_comment(&self, payload: &serde_json::Value) -> Result<WebhookAction> {
        let action = payload.get("action").and_then(|v| v.as_str()).unwrap_or("");

        if action != "created" {
            tracing::debug!(
                source = "github",
                action = %action,
                "Ignoring non-created issue comment action"
            );
            return Ok(WebhookAction::Ignored);
        }

        let issue = match payload.get("issue") {
            Some(i) => i,
            None => {
                tracing::warn!(source = "github", "Missing issue in payload");
                return Ok(WebhookAction::Ignored);
            }
        };

        // Only PR conversation comments matter; a plain issue has no `pull_request`.
        let pr = match issue.get("pull_request") {
            Some(p) => p,
            None => {
                tracing::debug!(
                    source = "github",
                    "Issue comment is not on a pull request, ignoring"
                );
                return Ok(WebhookAction::Ignored);
            }
        };

        let pr_url = pr
            .get("html_url")
            .and_then(|v| v.as_str())
            .or_else(|| issue.get("html_url").and_then(|v| v.as_str()))
            .unwrap_or_default();

        let comment = match payload.get("comment") {
            Some(c) => c,
            None => {
                tracing::warn!(source = "github", "Missing comment in payload");
                return Ok(WebhookAction::Ignored);
            }
        };

        let review_watcher = match &self.review_watcher {
            Some(rw) => rw,
            None => {
                tracing::debug!(
                    source = "github",
                    pr_url = %pr_url,
                    "ReviewWatcher not available, ignoring event"
                );
                return Ok(WebhookAction::Ignored);
            }
        };

        // Only act on PRs we're actively watching.
        let state = match review_watcher.get_state(pr_url) {
            Some(s) if s.is_active => s,
            _ => {
                tracing::debug!(
                    source = "github",
                    pr_url = %pr_url,
                    "PR not being watched, ignoring issue comment"
                );
                return Ok(WebhookAction::Ignored);
            }
        };

        // Skip bot comments (unless the bot is in the allowed list).
        let user = ReviewUser {
            id: comment
                .get("user")
                .and_then(|u| u.get("id"))
                .and_then(|v| v.as_i64())
                .unwrap_or_default(),
            login: comment
                .get("user")
                .and_then(|u| u.get("login"))
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string(),
            user_type: comment
                .get("user")
                .and_then(|u| u.get("type"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
        };
        if is_skippable_bot(&user, &self.config.allowed_bots) {
            tracing::debug!(
                source = "github",
                pr_url = %pr_url,
                author = %user.login,
                "Skipping bot issue comment"
            );
            return Ok(WebhookAction::Ignored);
        }

        // Require the review trigger so unrelated PR chatter doesn't spin up a
        // re-poll. The polling path applies the same filter, so this only avoids
        // needless work; correctness does not depend on it.
        let trigger = self.config.review_trigger.to_lowercase();
        let body = comment
            .get("body")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        if !trigger.is_empty() && !body.to_lowercase().contains(&trigger) {
            tracing::debug!(
                source = "github",
                pr_url = %pr_url,
                "Issue comment does not mention review trigger, ignoring"
            );
            return Ok(WebhookAction::Ignored);
        }

        tracing::info!(
            source = "github",
            pr_url = %pr_url,
            author = %user.login,
            issue_id = %state.issue_id,
            "Received PR conversation comment via webhook"
        );

        let processed_events = review_watcher.check_for_pr(pr_url).await?;
        tracing::info!(
            source = "github",
            pr_url = %pr_url,
            events = processed_events.len(),
            "Processed issue comment webhook through ReviewWatcher"
        );

        Ok(WebhookAction::Processed)
    }

    /// Handle a pull_request.closed event (merged or closed without merge).
    async fn handle_pull_request(&self, payload: &serde_json::Value) -> Result<WebhookAction> {
        let action = payload.get("action").and_then(|v| v.as_str()).unwrap_or("");

        if action != "closed" {
            tracing::debug!(
                source = "github",
                action = %action,
                "Ignoring non-closed pull_request action"
            );
            return Ok(WebhookAction::Ignored);
        }

        let pr = match payload.get("pull_request") {
            Some(p) => p,
            None => {
                tracing::warn!(source = "github", "Missing pull_request in payload");
                return Ok(WebhookAction::Ignored);
            }
        };

        let pr_url = pr
            .get("html_url")
            .and_then(|v| v.as_str())
            .unwrap_or_default();

        if pr_url.is_empty() {
            tracing::warn!(
                source = "github",
                "Missing html_url in pull_request payload"
            );
            return Ok(WebhookAction::Ignored);
        }

        let merged = pr.get("merged").and_then(|v| v.as_bool()).unwrap_or(false);

        tracing::info!(
            source = "github",
            pr_url = %pr_url,
            merged = merged,
            "Received pull_request closed event via webhook"
        );

        Ok(WebhookAction::PrClosed {
            pr_url: pr_url.to_string(),
            merged,
        })
    }

    /// Parse a review from the webhook payload.
    fn parse_review(&self, review: &serde_json::Value) -> Result<CodeReview> {
        let user = review
            .get("user")
            .map(|u| ReviewUser {
                id: u.get("id").and_then(|v| v.as_i64()).unwrap_or(0),
                login: u
                    .get("login")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string(),
                user_type: u
                    .get("type")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string()),
            })
            .unwrap_or(ReviewUser {
                id: 0,
                login: "unknown".to_string(),
                user_type: None,
            });

        Ok(CodeReview {
            id: review.get("id").and_then(|v| v.as_i64()).unwrap_or(0),
            user,
            body: review
                .get("body")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            state: review
                .get("state")
                .and_then(|v| v.as_str())
                .unwrap_or("COMMENTED")
                .to_string(),
            submitted_at: review
                .get("submitted_at")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            html_url: review
                .get("html_url")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
        })
    }

    /// Parse a comment from the webhook payload.
    fn parse_comment(&self, comment: &serde_json::Value) -> Result<ReviewComment> {
        let user = comment
            .get("user")
            .map(|u| ReviewUser {
                id: u.get("id").and_then(|v| v.as_i64()).unwrap_or(0),
                login: u
                    .get("login")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string(),
                user_type: u
                    .get("type")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string()),
            })
            .unwrap_or(ReviewUser {
                id: 0,
                login: "unknown".to_string(),
                user_type: None,
            });

        Ok(ReviewComment {
            id: comment.get("id").and_then(|v| v.as_i64()).unwrap_or(0),
            path: comment
                .get("path")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string(),
            position: comment.get("position").and_then(|v| v.as_i64()),
            original_position: comment.get("original_position").and_then(|v| v.as_i64()),
            body: comment
                .get("body")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string(),
            user,
            created_at: comment
                .get("created_at")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string(),
            updated_at: comment
                .get("updated_at")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string(),
            html_url: comment
                .get("html_url")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string(),
            pull_request_review_id: comment
                .get("pull_request_review_id")
                .and_then(|v| v.as_i64()),
            start_line: comment.get("start_line").and_then(|v| v.as_i64()),
            line: comment.get("line").and_then(|v| v.as_i64()),
            side: comment
                .get("side")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use claudear_core::secret::SecretValue;

    fn create_test_config(webhook_secret: Option<&str>) -> GitHubConfig {
        GitHubConfig {
            token: Some("test_token".into()),
            poll_interval_ms: 60000,
            auto_resolve_on_merge: false,
            webhook_secret: webhook_secret.map(SecretValue::new),
            review_trigger: "@claudear".to_string(),
            use_ssh: false,
            ..Default::default()
        }
    }

    #[test]
    fn test_github_webhook_signature_verification_valid() {
        let config = create_test_config(Some("test_secret"));
        let handler = GitHubWebhookHandler::new(config, None);

        let payload = b"test payload";

        // Compute expected signature
        use hmac::{Hmac, Mac};
        use sha2::Sha256;
        let mut mac = Hmac::<Sha256>::new_from_slice(b"test_secret").unwrap();
        mac.update(payload);
        let expected = mac.finalize().into_bytes();
        let signature = format!("sha256={}", hex::encode(expected));

        let mut headers = HashMap::new();
        headers.insert("x-hub-signature-256".to_string(), signature);

        assert!(handler.verify_signature(payload, &headers));
    }

    #[test]
    fn test_github_webhook_signature_verification_invalid() {
        let config = create_test_config(Some("test_secret"));
        let handler = GitHubWebhookHandler::new(config, None);

        let payload = b"test payload";

        let mut headers = HashMap::new();
        headers.insert(
            "x-hub-signature-256".to_string(),
            "sha256=invalid".to_string(),
        );

        assert!(!handler.verify_signature(payload, &headers));
    }

    #[test]
    fn test_github_webhook_signature_missing_header() {
        let config = create_test_config(Some("test_secret"));
        let handler = GitHubWebhookHandler::new(config, None);

        let payload = b"test payload";
        let headers = HashMap::new();

        assert!(!handler.verify_signature(payload, &headers));
    }

    #[test]
    fn test_github_webhook_signature_no_secret() {
        let config = create_test_config(None);
        let handler = GitHubWebhookHandler::new(config, None);

        let payload = b"test payload";
        let mut headers = HashMap::new();
        headers.insert(
            "x-hub-signature-256".to_string(),
            "sha256=whatever".to_string(),
        );

        assert!(!handler.verify_signature(payload, &headers));
    }

    #[test]
    fn test_github_webhook_signature_falls_back_to_app_secret() {
        let mut config = create_test_config(None);
        config.app.webhook_secret = Some("app_secret".into());
        let handler = GitHubWebhookHandler::new(config, None);

        let payload = b"test payload";
        use hmac::{Hmac, Mac};
        use sha2::Sha256;
        let mut mac = Hmac::<Sha256>::new_from_slice(b"app_secret").unwrap();
        mac.update(payload);
        let expected = mac.finalize().into_bytes();
        let signature = format!("sha256={}", hex::encode(expected));

        let mut headers = HashMap::new();
        headers.insert("x-hub-signature-256".to_string(), signature);

        assert!(handler.verify_signature(payload, &headers));
    }

    #[test]
    fn test_get_event_type() {
        let mut headers = HashMap::new();
        assert!(GitHubWebhookHandler::get_event_type(&headers).is_none());

        headers.insert(
            "x-github-event".to_string(),
            "pull_request_review".to_string(),
        );
        assert_eq!(
            GitHubWebhookHandler::get_event_type(&headers),
            Some("pull_request_review")
        );

        // Headers are lowercased by the webhook server
        headers.clear();
        headers.insert(
            "x-github-event".to_string(),
            "pull_request_review_comment".to_string(),
        );
        assert_eq!(
            GitHubWebhookHandler::get_event_type(&headers),
            Some("pull_request_review_comment")
        );
    }

    #[test]
    fn test_is_enabled() {
        // No secret, no watcher
        let config = create_test_config(None);
        let handler = GitHubWebhookHandler::new(config, None);
        assert!(!handler.is_enabled());

        // Has secret, no watcher
        let config = create_test_config(Some("secret"));
        let handler = GitHubWebhookHandler::new(config, None);
        assert!(!handler.is_enabled());
    }

    #[test]
    fn test_parse_review() {
        let config = create_test_config(Some("secret"));
        let handler = GitHubWebhookHandler::new(config, None);

        let review_json = serde_json::json!({
            "id": 12345,
            "user": {
                "id": 1,
                "login": "reviewer",
                "type": "User"
            },
            "body": "LGTM!",
            "state": "APPROVED",
            "submitted_at": "2024-01-15T10:00:00Z",
            "html_url": "https://github.com/owner/repo/pull/1#pullrequestreview-12345"
        });

        let review = handler.parse_review(&review_json).unwrap();
        assert_eq!(review.id, 12345);
        assert_eq!(review.user.login, "reviewer");
        assert_eq!(review.state, "APPROVED");
        assert_eq!(review.body, Some("LGTM!".to_string()));
    }

    #[test]
    fn test_parse_comment() {
        let config = create_test_config(Some("secret"));
        let handler = GitHubWebhookHandler::new(config, None);

        let comment_json = serde_json::json!({
            "id": 67890,
            "user": {
                "id": 2,
                "login": "commenter",
                "type": "User"
            },
            "body": "Consider refactoring this",
            "path": "src/main.rs",
            "position": 10,
            "line": 42,
            "side": "RIGHT",
            "created_at": "2024-01-15T11:00:00Z",
            "updated_at": "2024-01-15T11:00:00Z",
            "html_url": "https://github.com/owner/repo/pull/1#discussion_r67890",
            "pull_request_review_id": 12345
        });

        let comment = handler.parse_comment(&comment_json).unwrap();
        assert_eq!(comment.id, 67890);
        assert_eq!(comment.user.login, "commenter");
        assert_eq!(comment.path, "src/main.rs");
        assert_eq!(comment.body, "Consider refactoring this");
        assert_eq!(comment.line, Some(42));
        assert_eq!(comment.pull_request_review_id, Some(12345));
    }

    // Async test infrastructure: MockScmProvider + helpers

    use crate::scm::{PrInfo, PrReviewState, PrStatus, RemoteRepo, ScmProvider};
    use async_trait::async_trait;
    use std::sync::Mutex;

    /// A minimal mock implementing `ScmProvider` for testing the webhook handler.
    /// Returns configurable reviews/comments; everything else returns Ok(defaults).
    struct MockScmProvider {
        reviews: Mutex<Vec<CodeReview>>,
        comments: Mutex<Vec<ReviewComment>>,
    }

    impl MockScmProvider {
        fn new() -> Self {
            Self {
                reviews: Mutex::new(Vec::new()),
                comments: Mutex::new(Vec::new()),
            }
        }

        fn with_reviews(reviews: Vec<CodeReview>) -> Self {
            Self {
                reviews: Mutex::new(reviews),
                comments: Mutex::new(Vec::new()),
            }
        }

        fn with_comments(comments: Vec<ReviewComment>) -> Self {
            Self {
                reviews: Mutex::new(Vec::new()),
                comments: Mutex::new(comments),
            }
        }
    }

    #[async_trait]
    impl ScmProvider for MockScmProvider {
        fn name(&self) -> &str {
            "github"
        }

        fn is_enabled(&self) -> bool {
            true
        }

        fn review_trigger(&self) -> &str {
            "@claudear"
        }

        async fn get_pr_status(
            &self,
            _project: &str,
            _number: i64,
        ) -> claudear_core::error::Result<PrStatus> {
            Ok(PrStatus::Open)
        }

        async fn get_pr_info(
            &self,
            _project: &str,
            _number: i64,
        ) -> claudear_core::error::Result<PrInfo> {
            Ok(PrInfo {
                head_branch: Some("feature".to_string()),
                base_branch: Some("main".to_string()),
                title: Some("Test PR".to_string()),
                author: Some("testuser".to_string()),
            })
        }

        async fn get_pr_diff(
            &self,
            _project: &str,
            _number: i64,
        ) -> claudear_core::error::Result<String> {
            Ok(String::new())
        }

        async fn get_reviews(
            &self,
            _project: &str,
            _number: i64,
        ) -> claudear_core::error::Result<Vec<CodeReview>> {
            Ok(self.reviews.lock().unwrap().clone())
        }

        async fn get_review_comments(
            &self,
            _project: &str,
            _number: i64,
        ) -> claudear_core::error::Result<Vec<ReviewComment>> {
            Ok(self.comments.lock().unwrap().clone())
        }

        async fn list_repos(&self, _org: &str) -> claudear_core::error::Result<Vec<RemoteRepo>> {
            Ok(Vec::new())
        }
    }

    /// Compute a valid HMAC-SHA256 signature for the given payload and secret.
    fn make_valid_signature(secret: &str, payload: &[u8]) -> String {
        use hmac::{Hmac, Mac};
        use sha2::Sha256;
        let mut mac = Hmac::<Sha256>::new_from_slice(secret.as_bytes()).unwrap();
        mac.update(payload);
        let result = mac.finalize().into_bytes();
        format!("sha256={}", hex::encode(result))
    }

    /// Build a headers map with the given event type and signature.
    fn make_headers(event_type: &str, signature: &str) -> HashMap<String, String> {
        let mut headers = HashMap::new();
        headers.insert("x-github-event".to_string(), event_type.to_string());
        headers.insert("x-hub-signature-256".to_string(), signature.to_string());
        headers
    }

    /// Helper to build a handler with a ReviewWatcher that is watching a specific PR.
    fn make_handler_watching_pr(
        pr_url: &str,
        provider: Arc<dyn ScmProvider>,
    ) -> GitHubWebhookHandler {
        let watcher = Arc::new(ReviewWatcher::new(provider));
        let state = PrReviewState::new(pr_url, "owner/repo", 1, "ISSUE-1", "linear");
        watcher.watch_pr(state);
        GitHubWebhookHandler::new(create_test_config(Some("test_secret")), Some(watcher))
    }

    /// Helper: minimal valid review-submitted payload.
    fn review_submitted_payload(pr_url: &str) -> serde_json::Value {
        serde_json::json!({
            "action": "submitted",
            "review": {
                "id": 100,
                "user": { "id": 1, "login": "reviewer", "type": "User" },
                "body": "Looks good",
                "state": "APPROVED",
                "submitted_at": "2024-01-15T10:00:00Z",
                "html_url": "https://github.com/owner/repo/pull/1#pullrequestreview-100"
            },
            "pull_request": {
                "html_url": pr_url,
                "number": 1
            }
        })
    }

    /// Helper: minimal valid comment-created payload.
    fn comment_created_payload(pr_url: &str) -> serde_json::Value {
        serde_json::json!({
            "action": "created",
            "comment": {
                "id": 200,
                "user": { "id": 2, "login": "commenter", "type": "User" },
                "body": "Nit: rename this",
                "path": "src/lib.rs",
                "position": 5,
                "line": 10,
                "side": "RIGHT",
                "created_at": "2024-01-15T11:00:00Z",
                "updated_at": "2024-01-15T11:00:00Z",
                "html_url": "https://github.com/owner/repo/pull/1#discussion_r200",
                "pull_request_review_id": 100
            },
            "pull_request": {
                "html_url": pr_url,
                "number": 1
            }
        })
    }

    // process_webhook integration tests (8 tests)

    #[tokio::test]
    async fn test_process_webhook_invalid_signature() {
        let handler = GitHubWebhookHandler::new(create_test_config(Some("test_secret")), None);
        let payload = b"{}";
        let headers = make_headers("pull_request_review", "sha256=badbadbadbad");

        let result = handler
            .process_webhook(payload, &serde_json::json!({}), &headers)
            .await;
        assert!(result.is_err(), "Invalid signature should return Err");
    }

    #[tokio::test]
    async fn test_process_webhook_missing_event_header() {
        let handler = GitHubWebhookHandler::new(create_test_config(Some("test_secret")), None);
        let payload = b"{}";
        let sig = make_valid_signature("test_secret", payload);
        // Only include signature header, no x-github-event
        let mut headers = HashMap::new();
        headers.insert("x-hub-signature-256".to_string(), sig);

        let result = handler
            .process_webhook(payload, &serde_json::json!({}), &headers)
            .await;
        assert!(
            !result.unwrap().is_processed(),
            "Missing event header should return Ok(false)"
        );
    }

    #[tokio::test]
    async fn test_process_webhook_unknown_event() {
        let handler = GitHubWebhookHandler::new(create_test_config(Some("test_secret")), None);
        let payload = b"{}";
        let sig = make_valid_signature("test_secret", payload);
        let headers = make_headers("push", &sig);

        let result = handler
            .process_webhook(payload, &serde_json::json!({}), &headers)
            .await;
        assert!(
            !result.unwrap().is_processed(),
            "Unknown event type should return Ok(false)"
        );
    }

    #[tokio::test]
    async fn test_process_webhook_review_submitted() {
        let pr_url = "https://github.com/owner/repo/pull/1";
        let mock = Arc::new(MockScmProvider::with_reviews(vec![CodeReview {
            id: 100,
            user: ReviewUser {
                id: 1,
                login: "reviewer".to_string(),
                user_type: Some("User".to_string()),
            },
            body: Some("Looks good".to_string()),
            state: "APPROVED".to_string(),
            submitted_at: Some("2024-01-15T10:00:00Z".to_string()),
            html_url: Some(
                "https://github.com/owner/repo/pull/1#pullrequestreview-100".to_string(),
            ),
        }]));
        let handler = make_handler_watching_pr(pr_url, mock);

        let payload_value = review_submitted_payload(pr_url);
        let payload_bytes = serde_json::to_vec(&payload_value).unwrap();
        let sig = make_valid_signature("test_secret", &payload_bytes);
        let headers = make_headers("pull_request_review", &sig);

        let result = handler
            .process_webhook(&payload_bytes, &payload_value, &headers)
            .await;
        assert!(
            result.unwrap().is_processed(),
            "Valid review submitted for watched PR should return Ok(true)"
        );
    }

    #[tokio::test]
    async fn test_process_webhook_review_no_watcher() {
        // Handler created with review_watcher=None
        let handler = GitHubWebhookHandler::new(create_test_config(Some("test_secret")), None);
        let pr_url = "https://github.com/owner/repo/pull/1";
        let payload_value = review_submitted_payload(pr_url);
        let payload_bytes = serde_json::to_vec(&payload_value).unwrap();
        let sig = make_valid_signature("test_secret", &payload_bytes);
        let headers = make_headers("pull_request_review", &sig);

        let result = handler
            .process_webhook(&payload_bytes, &payload_value, &headers)
            .await;
        assert!(
            !result.unwrap().is_processed(),
            "No watcher should return Ok(false)"
        );
    }

    #[tokio::test]
    async fn test_process_webhook_review_unwatched_pr() {
        // Watcher exists but the specific PR is NOT being watched
        let mock = Arc::new(MockScmProvider::new());
        let watcher = Arc::new(ReviewWatcher::new(mock));
        // Watch a different PR
        let other_state = PrReviewState::new(
            "https://github.com/owner/repo/pull/999",
            "owner/repo",
            999,
            "ISSUE-999",
            "linear",
        );
        watcher.watch_pr(other_state);
        let handler =
            GitHubWebhookHandler::new(create_test_config(Some("test_secret")), Some(watcher));

        let pr_url = "https://github.com/owner/repo/pull/1";
        let payload_value = review_submitted_payload(pr_url);
        let payload_bytes = serde_json::to_vec(&payload_value).unwrap();
        let sig = make_valid_signature("test_secret", &payload_bytes);
        let headers = make_headers("pull_request_review", &sig);

        let result = handler
            .process_webhook(&payload_bytes, &payload_value, &headers)
            .await;
        assert!(
            !result.unwrap().is_processed(),
            "Unwatched PR should return Ok(false)"
        );
    }

    #[tokio::test]
    async fn test_process_webhook_comment_created() {
        let pr_url = "https://github.com/owner/repo/pull/1";
        let mock = Arc::new(MockScmProvider::with_comments(vec![ReviewComment {
            id: 200,
            path: "src/lib.rs".to_string(),
            position: Some(5),
            original_position: None,
            body: "Nit: rename this".to_string(),
            user: ReviewUser {
                id: 2,
                login: "commenter".to_string(),
                user_type: Some("User".to_string()),
            },
            created_at: "2024-01-15T11:00:00Z".to_string(),
            updated_at: "2024-01-15T11:00:00Z".to_string(),
            html_url: "https://github.com/owner/repo/pull/1#discussion_r200".to_string(),
            pull_request_review_id: Some(100),
            start_line: None,
            line: Some(10),
            side: Some("RIGHT".to_string()),
        }]));
        let handler = make_handler_watching_pr(pr_url, mock);

        let payload_value = comment_created_payload(pr_url);
        let payload_bytes = serde_json::to_vec(&payload_value).unwrap();
        let sig = make_valid_signature("test_secret", &payload_bytes);
        let headers = make_headers("pull_request_review_comment", &sig);

        let result = handler
            .process_webhook(&payload_bytes, &payload_value, &headers)
            .await;
        assert!(
            result.unwrap().is_processed(),
            "Valid comment created for watched PR should return Ok(true)"
        );
    }

    #[tokio::test]
    async fn test_process_webhook_comment_unwatched_pr() {
        let mock = Arc::new(MockScmProvider::new());
        let watcher = Arc::new(ReviewWatcher::new(mock));
        // Watch a different PR
        let other_state = PrReviewState::new(
            "https://github.com/owner/repo/pull/999",
            "owner/repo",
            999,
            "ISSUE-999",
            "linear",
        );
        watcher.watch_pr(other_state);
        let handler =
            GitHubWebhookHandler::new(create_test_config(Some("test_secret")), Some(watcher));

        let pr_url = "https://github.com/owner/repo/pull/1";
        let payload_value = comment_created_payload(pr_url);
        let payload_bytes = serde_json::to_vec(&payload_value).unwrap();
        let sig = make_valid_signature("test_secret", &payload_bytes);
        let headers = make_headers("pull_request_review_comment", &sig);

        let result = handler
            .process_webhook(&payload_bytes, &payload_value, &headers)
            .await;
        assert!(
            !result.unwrap().is_processed(),
            "Comment on unwatched PR should return Ok(false)"
        );
    }

    /// Helper: an `issue_comment.created` payload on a pull request.
    fn issue_comment_payload(pr_url: &str, body: &str, is_pr: bool) -> serde_json::Value {
        let mut issue = serde_json::json!({
            "number": 1,
            "html_url": pr_url
        });
        if is_pr {
            issue["pull_request"] = serde_json::json!({ "html_url": pr_url });
        }
        serde_json::json!({
            "action": "created",
            "comment": {
                "id": 300,
                "user": { "id": 3, "login": "reviewer", "type": "User" },
                "body": body,
                "created_at": "2024-01-15T12:00:00Z",
                "updated_at": "2024-01-15T12:00:00Z",
                "html_url": "https://github.com/owner/repo/pull/1#issuecomment-300"
            },
            "issue": issue
        })
    }

    #[tokio::test]
    async fn test_issue_comment_on_watched_pr_with_trigger_is_processed() {
        let pr_url = "https://github.com/owner/repo/pull/1";
        let handler = make_handler_watching_pr(pr_url, Arc::new(MockScmProvider::new()));

        let payload_value =
            issue_comment_payload(pr_url, "@claudear please fix the shutdown race", true);
        let payload_bytes = serde_json::to_vec(&payload_value).unwrap();
        let sig = make_valid_signature("test_secret", &payload_bytes);
        let headers = make_headers("issue_comment", &sig);

        let result = handler
            .process_webhook(&payload_bytes, &payload_value, &headers)
            .await;
        assert!(
            result.unwrap().is_processed(),
            "Triggered PR conversation comment should be processed"
        );
    }

    #[tokio::test]
    async fn test_issue_comment_without_trigger_is_ignored() {
        let pr_url = "https://github.com/owner/repo/pull/1";
        let handler = make_handler_watching_pr(pr_url, Arc::new(MockScmProvider::new()));

        let payload_value = issue_comment_payload(pr_url, "LGTM, nice", true);
        let payload_bytes = serde_json::to_vec(&payload_value).unwrap();
        let sig = make_valid_signature("test_secret", &payload_bytes);
        let headers = make_headers("issue_comment", &sig);

        let result = handler
            .process_webhook(&payload_bytes, &payload_value, &headers)
            .await;
        assert!(
            !result.unwrap().is_processed(),
            "Untriggered PR conversation comment should be ignored"
        );
    }

    #[tokio::test]
    async fn test_issue_comment_on_plain_issue_is_ignored() {
        let pr_url = "https://github.com/owner/repo/pull/1";
        let handler = make_handler_watching_pr(pr_url, Arc::new(MockScmProvider::new()));

        // No `pull_request` field => a regular issue, not a PR.
        let payload_value = issue_comment_payload(pr_url, "@claudear fix this", false);
        let payload_bytes = serde_json::to_vec(&payload_value).unwrap();
        let sig = make_valid_signature("test_secret", &payload_bytes);
        let headers = make_headers("issue_comment", &sig);

        let result = handler
            .process_webhook(&payload_bytes, &payload_value, &headers)
            .await;
        assert!(
            !result.unwrap().is_processed(),
            "Comment on a non-PR issue should be ignored"
        );
    }

    // handle_review_submitted tests (5 tests)

    #[tokio::test]
    async fn test_handle_review_non_submitted_action() {
        let handler = GitHubWebhookHandler::new(create_test_config(Some("test_secret")), None);
        let payload = serde_json::json!({ "action": "edited" });

        let result = handler.handle_review_submitted(&payload).await;
        assert!(
            !result.unwrap().is_processed(),
            "Non-submitted action should return Ok(false)"
        );
    }

    #[tokio::test]
    async fn test_handle_review_missing_review_field() {
        let handler = GitHubWebhookHandler::new(create_test_config(Some("test_secret")), None);
        let payload = serde_json::json!({
            "action": "submitted",
            "pull_request": { "html_url": "https://github.com/owner/repo/pull/1" }
        });

        let result = handler.handle_review_submitted(&payload).await;
        assert!(
            !result.unwrap().is_processed(),
            "Missing review field should return Ok(false)"
        );
    }

    #[tokio::test]
    async fn test_handle_review_missing_pull_request() {
        let handler = GitHubWebhookHandler::new(create_test_config(Some("test_secret")), None);
        let payload = serde_json::json!({
            "action": "submitted",
            "review": {
                "id": 100,
                "user": { "id": 1, "login": "reviewer", "type": "User" },
                "state": "APPROVED"
            }
        });

        let result = handler.handle_review_submitted(&payload).await;
        assert!(
            !result.unwrap().is_processed(),
            "Missing pull_request field should return Ok(false)"
        );
    }

    #[tokio::test]
    async fn test_handle_review_bot_review() {
        let pr_url = "https://github.com/owner/repo/pull/1";
        let mock = Arc::new(MockScmProvider::new());
        let handler = make_handler_watching_pr(pr_url, mock);

        let payload = serde_json::json!({
            "action": "submitted",
            "review": {
                "id": 100,
                "user": { "id": 1, "login": "dependabot[bot]", "type": "Bot" },
                "body": "Automated review",
                "state": "APPROVED",
                "submitted_at": "2024-01-15T10:00:00Z"
            },
            "pull_request": {
                "html_url": pr_url,
                "number": 1
            }
        });

        let result = handler.handle_review_submitted(&payload).await;
        assert!(
            !result.unwrap().is_processed(),
            "Bot review should return Ok(false)"
        );
    }

    #[tokio::test]
    async fn test_handle_review_pending_review() {
        let pr_url = "https://github.com/owner/repo/pull/1";
        let mock = Arc::new(MockScmProvider::new());
        let handler = make_handler_watching_pr(pr_url, mock);

        let payload = serde_json::json!({
            "action": "submitted",
            "review": {
                "id": 100,
                "user": { "id": 1, "login": "reviewer", "type": "User" },
                "body": "",
                "state": "PENDING",
                "submitted_at": "2024-01-15T10:00:00Z"
            },
            "pull_request": {
                "html_url": pr_url,
                "number": 1
            }
        });

        let result = handler.handle_review_submitted(&payload).await;
        assert!(
            !result.unwrap().is_processed(),
            "Pending review should return Ok(false)"
        );
    }

    // handle_review_comment tests (3 tests)

    #[tokio::test]
    async fn test_handle_comment_non_created_action() {
        let handler = GitHubWebhookHandler::new(create_test_config(Some("test_secret")), None);
        let payload = serde_json::json!({ "action": "edited" });

        let result = handler.handle_review_comment(&payload).await;
        assert!(
            !result.unwrap().is_processed(),
            "Non-created action should return Ok(false)"
        );
    }

    #[tokio::test]
    async fn test_handle_comment_missing_comment_field() {
        let handler = GitHubWebhookHandler::new(create_test_config(Some("test_secret")), None);
        let payload = serde_json::json!({
            "action": "created",
            "pull_request": { "html_url": "https://github.com/owner/repo/pull/1" }
        });

        let result = handler.handle_review_comment(&payload).await;
        assert!(
            !result.unwrap().is_processed(),
            "Missing comment field should return Ok(false)"
        );
    }

    #[tokio::test]
    async fn test_handle_comment_bot_comment() {
        let pr_url = "https://github.com/owner/repo/pull/1";
        let mock = Arc::new(MockScmProvider::new());
        let handler = make_handler_watching_pr(pr_url, mock);

        let payload = serde_json::json!({
            "action": "created",
            "comment": {
                "id": 200,
                "user": { "id": 1, "login": "github-actions[bot]", "type": "Bot" },
                "body": "Automated comment",
                "path": "src/lib.rs",
                "position": 5,
                "line": 10,
                "side": "RIGHT",
                "created_at": "2024-01-15T11:00:00Z",
                "updated_at": "2024-01-15T11:00:00Z",
                "html_url": "https://github.com/owner/repo/pull/1#discussion_r200"
            },
            "pull_request": {
                "html_url": pr_url,
                "number": 1
            }
        });

        let result = handler.handle_review_comment(&payload).await;
        assert!(
            !result.unwrap().is_processed(),
            "Bot comment should return Ok(false)"
        );
    }

    // parse edge case tests (2 tests)

    #[test]
    fn test_parse_review_missing_fields() {
        let config = create_test_config(Some("secret"));
        let handler = GitHubWebhookHandler::new(config, None);

        // Minimal JSON: no user, no body, no state, no submitted_at, no html_url
        let review_json = serde_json::json!({});

        let review = handler.parse_review(&review_json).unwrap();
        assert_eq!(review.id, 0, "Missing id should default to 0");
        assert_eq!(review.user.id, 0, "Missing user should default id to 0");
        assert_eq!(
            review.user.login, "unknown",
            "Missing user should default login to 'unknown'"
        );
        assert!(
            review.user.user_type.is_none(),
            "Missing user should have no user_type"
        );
        assert!(review.body.is_none(), "Missing body should be None");
        assert_eq!(
            review.state, "COMMENTED",
            "Missing state should default to COMMENTED"
        );
        assert!(
            review.submitted_at.is_none(),
            "Missing submitted_at should be None"
        );
        assert!(review.html_url.is_none(), "Missing html_url should be None");
    }

    #[test]
    fn test_parse_comment_missing_fields() {
        let config = create_test_config(Some("secret"));
        let handler = GitHubWebhookHandler::new(config, None);

        // Minimal JSON: no fields at all
        let comment_json = serde_json::json!({});

        let comment = handler.parse_comment(&comment_json).unwrap();
        assert_eq!(comment.id, 0, "Missing id should default to 0");
        assert_eq!(
            comment.path, "",
            "Missing path should default to empty string"
        );
        assert!(
            comment.position.is_none(),
            "Missing position should be None"
        );
        assert!(
            comment.original_position.is_none(),
            "Missing original_position should be None"
        );
        assert_eq!(
            comment.body, "",
            "Missing body should default to empty string"
        );
        assert_eq!(comment.user.id, 0, "Missing user should default id to 0");
        assert_eq!(
            comment.user.login, "unknown",
            "Missing user should default login to 'unknown'"
        );
        assert!(
            comment.user.user_type.is_none(),
            "Missing user should have no user_type"
        );
        assert_eq!(
            comment.created_at, "",
            "Missing created_at should default to empty string"
        );
        assert_eq!(
            comment.updated_at, "",
            "Missing updated_at should default to empty string"
        );
        assert_eq!(
            comment.html_url, "",
            "Missing html_url should default to empty string"
        );
        assert!(
            comment.pull_request_review_id.is_none(),
            "Missing pull_request_review_id should be None"
        );
        assert!(
            comment.start_line.is_none(),
            "Missing start_line should be None"
        );
        assert!(comment.line.is_none(), "Missing line should be None");
        assert!(comment.side.is_none(), "Missing side should be None");
    }

    // HMAC signature verification — extended edge cases

    #[test]
    fn test_signature_empty_payload() {
        let config = create_test_config(Some("test_secret"));
        let handler = GitHubWebhookHandler::new(config, None);

        let payload = b"";
        let sig = make_valid_signature("test_secret", payload);
        let mut headers = HashMap::new();
        headers.insert("x-hub-signature-256".to_string(), sig);

        assert!(
            handler.verify_signature(payload, &headers),
            "Empty payload with valid signature should verify"
        );
    }

    #[test]
    fn test_signature_empty_secret_rejects() {
        // Empty webhook secrets are treated as missing and rejected.
        let config = create_test_config(Some(""));
        let handler = GitHubWebhookHandler::new(config, None);

        let payload = b"test payload";
        let sig = make_valid_signature("", payload);
        let mut headers = HashMap::new();
        headers.insert("x-hub-signature-256".to_string(), sig);

        assert!(!handler.verify_signature(payload, &headers));
    }

    #[test]
    fn test_signature_wrong_secret_rejects() {
        let config = create_test_config(Some("correct_secret"));
        let handler = GitHubWebhookHandler::new(config, None);

        let payload = b"test payload";
        // Sign with the wrong secret
        let sig = make_valid_signature("wrong_secret", payload);
        let mut headers = HashMap::new();
        headers.insert("x-hub-signature-256".to_string(), sig);

        assert!(
            !handler.verify_signature(payload, &headers),
            "Signature from wrong secret should not verify"
        );
    }

    #[test]
    fn test_signature_missing_sha256_prefix() {
        let config = create_test_config(Some("test_secret"));
        let handler = GitHubWebhookHandler::new(config, None);

        let payload = b"test payload";
        // Compute correct hex but omit the sha256= prefix
        let mut mac = Hmac::<Sha256>::new_from_slice(b"test_secret").unwrap();
        mac.update(payload);
        let hex_only = hex::encode(mac.finalize().into_bytes());

        let mut headers = HashMap::new();
        headers.insert("x-hub-signature-256".to_string(), hex_only);

        assert!(
            !handler.verify_signature(payload, &headers),
            "Signature without sha256= prefix should not verify"
        );
    }

    #[test]
    fn test_signature_wrong_prefix_sha1() {
        let config = create_test_config(Some("test_secret"));
        let handler = GitHubWebhookHandler::new(config, None);

        let payload = b"test payload";
        let mut mac = Hmac::<Sha256>::new_from_slice(b"test_secret").unwrap();
        mac.update(payload);
        let hex = hex::encode(mac.finalize().into_bytes());

        let mut headers = HashMap::new();
        // Use sha1= prefix instead of sha256=
        headers.insert("x-hub-signature-256".to_string(), format!("sha1={hex}"));

        assert!(
            !handler.verify_signature(payload, &headers),
            "Signature with sha1= prefix should not verify"
        );
    }

    #[test]
    fn test_signature_empty_signature_header() {
        let config = create_test_config(Some("test_secret"));
        let handler = GitHubWebhookHandler::new(config, None);

        let payload = b"test payload";
        let mut headers = HashMap::new();
        headers.insert("x-hub-signature-256".to_string(), String::new());

        assert!(
            !handler.verify_signature(payload, &headers),
            "Empty signature header should not verify"
        );
    }

    #[test]
    fn test_signature_sha256_prefix_only() {
        let config = create_test_config(Some("test_secret"));
        let handler = GitHubWebhookHandler::new(config, None);

        let payload = b"test payload";
        let mut headers = HashMap::new();
        headers.insert("x-hub-signature-256".to_string(), "sha256=".to_string());

        assert!(
            !handler.verify_signature(payload, &headers),
            "Signature with only sha256= prefix and no hex should not verify"
        );
    }

    #[test]
    fn test_signature_tampered_payload() {
        let config = create_test_config(Some("test_secret"));
        let handler = GitHubWebhookHandler::new(config, None);

        let original_payload = b"original payload";
        let sig = make_valid_signature("test_secret", original_payload);
        let mut headers = HashMap::new();
        headers.insert("x-hub-signature-256".to_string(), sig);

        let tampered_payload = b"tampered payload";
        assert!(
            !handler.verify_signature(tampered_payload, &headers),
            "Signature for different payload should not verify"
        );
    }

    #[test]
    fn test_signature_large_payload() {
        let config = create_test_config(Some("test_secret"));
        let handler = GitHubWebhookHandler::new(config, None);

        // Large payload (100KB)
        let payload = vec![b'x'; 100_000];
        let sig = make_valid_signature("test_secret", &payload);
        let mut headers = HashMap::new();
        headers.insert("x-hub-signature-256".to_string(), sig);

        assert!(
            handler.verify_signature(&payload, &headers),
            "Large payload with valid signature should verify"
        );
    }

    #[test]
    fn test_signature_unicode_payload() {
        let config = create_test_config(Some("test_secret"));
        let handler = GitHubWebhookHandler::new(config, None);

        let payload = "{ \"body\": \"LGTM! \u{1f44d}\" }".as_bytes();
        let sig = make_valid_signature("test_secret", payload);
        let mut headers = HashMap::new();
        headers.insert("x-hub-signature-256".to_string(), sig);

        assert!(
            handler.verify_signature(payload, &headers),
            "Payload with unicode/emoji should verify correctly"
        );
    }

    #[test]
    fn test_signature_case_sensitive_hex() {
        let config = create_test_config(Some("test_secret"));
        let handler = GitHubWebhookHandler::new(config, None);

        let payload = b"test payload";
        let mut mac = Hmac::<Sha256>::new_from_slice(b"test_secret").unwrap();
        mac.update(payload);
        let hex = hex::encode(mac.finalize().into_bytes()).to_uppercase();

        let mut headers = HashMap::new();
        headers.insert("x-hub-signature-256".to_string(), format!("sha256={hex}"));

        // Uppercase hex won't match lowercase via constant-time comparison
        assert!(
            !handler.verify_signature(payload, &headers),
            "Uppercase hex should not verify against lowercase computed hex"
        );
    }

    #[test]
    fn test_signature_with_extra_headers() {
        let config = create_test_config(Some("test_secret"));
        let handler = GitHubWebhookHandler::new(config, None);

        let payload = b"test payload";
        let sig = make_valid_signature("test_secret", payload);
        let mut headers = HashMap::new();
        headers.insert("x-hub-signature-256".to_string(), sig);
        headers.insert("x-github-event".to_string(), "push".to_string());
        headers.insert("x-github-delivery".to_string(), "abc-123".to_string());
        headers.insert("content-type".to_string(), "application/json".to_string());

        assert!(
            handler.verify_signature(payload, &headers),
            "Extra headers should not interfere with signature verification"
        );
    }

    // Event type routing — additional cases

    #[test]
    fn test_get_event_type_various_events() {
        let event_types = [
            "push",
            "issues",
            "issue_comment",
            "pull_request",
            "pull_request_review",
            "pull_request_review_comment",
            "check_suite",
            "check_run",
            "status",
            "ping",
            "create",
            "delete",
        ];

        for event_type in &event_types {
            let mut headers = HashMap::new();
            headers.insert("x-github-event".to_string(), event_type.to_string());
            assert_eq!(
                GitHubWebhookHandler::get_event_type(&headers),
                Some(*event_type),
                "Should return event type '{event_type}'"
            );
        }
    }

    #[test]
    fn test_get_event_type_empty_string() {
        let mut headers = HashMap::new();
        headers.insert("x-github-event".to_string(), String::new());
        assert_eq!(
            GitHubWebhookHandler::get_event_type(&headers),
            Some(""),
            "Empty event type string should still be returned"
        );
    }

    #[test]
    fn test_get_event_type_wrong_header_name() {
        let mut headers = HashMap::new();
        // The server lowercases headers, so X-GitHub-Event would never appear
        headers.insert(
            "X-GitHub-Event".to_string(),
            "pull_request_review".to_string(),
        );
        assert!(
            GitHubWebhookHandler::get_event_type(&headers).is_none(),
            "Case-sensitive lookup should not find uppercase header"
        );
    }

    // is_enabled — additional cases

    #[test]
    fn test_is_enabled_with_watcher_and_secret() {
        let mock = Arc::new(MockScmProvider::new());
        let watcher = Arc::new(ReviewWatcher::new(mock));
        let config = create_test_config(Some("secret"));
        let handler = GitHubWebhookHandler::new(config, Some(watcher));
        assert!(
            handler.is_enabled(),
            "Handler with secret and watcher should be enabled"
        );
    }

    #[test]
    fn test_is_enabled_empty_secret_with_watcher() {
        // Empty webhook secrets are treated as missing.
        let mock = Arc::new(MockScmProvider::new());
        let watcher = Arc::new(ReviewWatcher::new(mock));
        let config = create_test_config(Some(""));
        let handler = GitHubWebhookHandler::new(config, Some(watcher));
        assert!(!handler.is_enabled());
    }

    // source_name

    #[test]
    fn test_source_name() {
        let config = create_test_config(None);
        let handler = GitHubWebhookHandler::new(config, None);
        assert_eq!(handler.source_name(), "github");
    }

    // parse_review — extended edge cases

    #[test]
    fn test_parse_review_all_states() {
        let config = create_test_config(Some("secret"));
        let handler = GitHubWebhookHandler::new(config, None);

        let states = [
            "APPROVED",
            "CHANGES_REQUESTED",
            "COMMENTED",
            "DISMISSED",
            "PENDING",
        ];

        for state in &states {
            let review_json = serde_json::json!({
                "id": 1,
                "user": { "id": 1, "login": "rev", "type": "User" },
                "state": state,
            });

            let review = handler.parse_review(&review_json).unwrap();
            assert_eq!(
                review.state, *state,
                "State should be parsed correctly for {state}"
            );
        }
    }

    #[test]
    fn test_parse_review_null_body() {
        let config = create_test_config(Some("secret"));
        let handler = GitHubWebhookHandler::new(config, None);

        let review_json = serde_json::json!({
            "id": 1,
            "user": { "id": 1, "login": "rev", "type": "User" },
            "body": null,
            "state": "APPROVED",
        });

        let review = handler.parse_review(&review_json).unwrap();
        assert!(
            review.body.is_none(),
            "Explicit null body should be parsed as None"
        );
    }

    #[test]
    fn test_parse_review_empty_body() {
        let config = create_test_config(Some("secret"));
        let handler = GitHubWebhookHandler::new(config, None);

        let review_json = serde_json::json!({
            "id": 1,
            "user": { "id": 1, "login": "rev", "type": "User" },
            "body": "",
            "state": "COMMENTED",
        });

        let review = handler.parse_review(&review_json).unwrap();
        assert_eq!(
            review.body,
            Some(String::new()),
            "Empty string body should be Some(\"\")"
        );
    }

    #[test]
    fn test_parse_review_user_type_bot() {
        let config = create_test_config(Some("secret"));
        let handler = GitHubWebhookHandler::new(config, None);

        let review_json = serde_json::json!({
            "id": 1,
            "user": {
                "id": 41898282,
                "login": "github-actions[bot]",
                "type": "Bot"
            },
            "state": "COMMENTED",
        });

        let review = handler.parse_review(&review_json).unwrap();
        assert_eq!(review.user.login, "github-actions[bot]");
        assert_eq!(review.user.user_type, Some("Bot".to_string()));
    }

    #[test]
    fn test_parse_review_user_missing_type() {
        let config = create_test_config(Some("secret"));
        let handler = GitHubWebhookHandler::new(config, None);

        let review_json = serde_json::json!({
            "id": 1,
            "user": {
                "id": 42,
                "login": "someuser"
                // "type" field is absent
            },
            "state": "APPROVED",
        });

        let review = handler.parse_review(&review_json).unwrap();
        assert_eq!(review.user.login, "someuser");
        assert!(
            review.user.user_type.is_none(),
            "Missing type should result in None"
        );
    }

    #[test]
    fn test_parse_review_extra_fields_ignored() {
        let config = create_test_config(Some("secret"));
        let handler = GitHubWebhookHandler::new(config, None);

        let review_json = serde_json::json!({
            "id": 999,
            "user": {
                "id": 1,
                "login": "reviewer",
                "type": "User",
                "avatar_url": "https://avatars.githubusercontent.com/u/1",
                "gravatar_id": "",
                "site_admin": false
            },
            "body": "Nice work!",
            "state": "APPROVED",
            "submitted_at": "2024-01-15T10:00:00Z",
            "html_url": "https://github.com/owner/repo/pull/1#pullrequestreview-999",
            "node_id": "PRR_kwDOXXXXXX",
            "commit_id": "abc123def456",
            "author_association": "CONTRIBUTOR"
        });

        let review = handler.parse_review(&review_json).unwrap();
        assert_eq!(review.id, 999);
        assert_eq!(review.user.login, "reviewer");
        assert_eq!(review.state, "APPROVED");
        assert_eq!(review.body, Some("Nice work!".to_string()));
    }

    #[test]
    fn test_parse_review_id_as_large_number() {
        let config = create_test_config(Some("secret"));
        let handler = GitHubWebhookHandler::new(config, None);

        let review_json = serde_json::json!({
            "id": 2_147_483_648_i64, // Larger than i32::MAX
            "user": { "id": 1, "login": "rev", "type": "User" },
            "state": "APPROVED",
        });

        let review = handler.parse_review(&review_json).unwrap();
        assert_eq!(review.id, 2_147_483_648);
    }

    #[test]
    fn test_parse_review_negative_id() {
        let config = create_test_config(Some("secret"));
        let handler = GitHubWebhookHandler::new(config, None);

        let review_json = serde_json::json!({
            "id": -1,
            "user": { "id": -5, "login": "rev", "type": "User" },
            "state": "APPROVED",
        });

        let review = handler.parse_review(&review_json).unwrap();
        assert_eq!(review.id, -1);
        assert_eq!(review.user.id, -5);
    }

    #[test]
    fn test_parse_review_id_as_string_coercion() {
        let config = create_test_config(Some("secret"));
        let handler = GitHubWebhookHandler::new(config, None);

        // If id is provided as a string, as_i64() returns None, should default to 0
        let review_json = serde_json::json!({
            "id": "not-a-number",
            "user": { "id": 1, "login": "rev", "type": "User" },
            "state": "APPROVED",
        });

        let review = handler.parse_review(&review_json).unwrap();
        assert_eq!(review.id, 0, "Non-integer id should default to 0");
    }

    #[test]
    fn test_parse_review_user_empty_object() {
        let config = create_test_config(Some("secret"));
        let handler = GitHubWebhookHandler::new(config, None);

        // user exists but has no fields
        let review_json = serde_json::json!({
            "id": 1,
            "user": {},
            "state": "APPROVED",
        });

        let review = handler.parse_review(&review_json).unwrap();
        assert_eq!(review.user.id, 0);
        assert_eq!(review.user.login, "");
        assert!(review.user.user_type.is_none());
    }

    #[test]
    fn test_parse_review_user_null() {
        let config = create_test_config(Some("secret"));
        let handler = GitHubWebhookHandler::new(config, None);

        // user is null
        let review_json = serde_json::json!({
            "id": 1,
            "user": null,
            "state": "APPROVED",
        });

        let review = handler.parse_review(&review_json).unwrap();
        // null means .get("user") returns Some(Value::Null), but the map closure
        // on Value::Null will attempt to call .get("id") on null which returns None
        // so it falls to defaults within the map closure
        assert_eq!(review.user.id, 0);
        assert_eq!(review.user.login, "");
    }

    #[test]
    fn test_parse_review_multiline_body() {
        let config = create_test_config(Some("secret"));
        let handler = GitHubWebhookHandler::new(config, None);

        let body = "Line 1\nLine 2\n\n## Heading\n- bullet\n- bullet 2";
        let review_json = serde_json::json!({
            "id": 1,
            "user": { "id": 1, "login": "rev", "type": "User" },
            "body": body,
            "state": "CHANGES_REQUESTED",
        });

        let review = handler.parse_review(&review_json).unwrap();
        assert_eq!(review.body, Some(body.to_string()));
    }

    // parse_comment — extended edge cases

    #[test]
    fn test_parse_comment_all_fields_populated() {
        let config = create_test_config(Some("secret"));
        let handler = GitHubWebhookHandler::new(config, None);

        let comment_json = serde_json::json!({
            "id": 500,
            "user": { "id": 10, "login": "alice", "type": "User" },
            "body": "Please fix the formatting here",
            "path": "src/webhook/github.rs",
            "position": 15,
            "original_position": 12,
            "line": 100,
            "start_line": 95,
            "side": "LEFT",
            "created_at": "2024-06-15T08:30:00Z",
            "updated_at": "2024-06-15T09:00:00Z",
            "html_url": "https://github.com/owner/repo/pull/42#discussion_r500",
            "pull_request_review_id": 300
        });

        let comment = handler.parse_comment(&comment_json).unwrap();
        assert_eq!(comment.id, 500);
        assert_eq!(comment.user.id, 10);
        assert_eq!(comment.user.login, "alice");
        assert_eq!(comment.user.user_type, Some("User".to_string()));
        assert_eq!(comment.body, "Please fix the formatting here");
        assert_eq!(comment.path, "src/webhook/github.rs");
        assert_eq!(comment.position, Some(15));
        assert_eq!(comment.original_position, Some(12));
        assert_eq!(comment.line, Some(100));
        assert_eq!(comment.start_line, Some(95));
        assert_eq!(comment.side, Some("LEFT".to_string()));
        assert_eq!(comment.created_at, "2024-06-15T08:30:00Z");
        assert_eq!(comment.updated_at, "2024-06-15T09:00:00Z");
        assert_eq!(
            comment.html_url,
            "https://github.com/owner/repo/pull/42#discussion_r500"
        );
        assert_eq!(comment.pull_request_review_id, Some(300));
    }

    #[test]
    fn test_parse_comment_null_optional_fields() {
        let config = create_test_config(Some("secret"));
        let handler = GitHubWebhookHandler::new(config, None);

        let comment_json = serde_json::json!({
            "id": 1,
            "user": { "id": 1, "login": "u", "type": "User" },
            "body": "test",
            "path": "file.rs",
            "position": null,
            "original_position": null,
            "line": null,
            "start_line": null,
            "side": null,
            "pull_request_review_id": null,
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z",
            "html_url": "https://example.com/comment"
        });

        let comment = handler.parse_comment(&comment_json).unwrap();
        assert!(comment.position.is_none(), "Null position should be None");
        assert!(
            comment.original_position.is_none(),
            "Null original_position should be None"
        );
        assert!(comment.line.is_none(), "Null line should be None");
        assert!(
            comment.start_line.is_none(),
            "Null start_line should be None"
        );
        assert!(comment.side.is_none(), "Null side should be None");
        assert!(
            comment.pull_request_review_id.is_none(),
            "Null pull_request_review_id should be None"
        );
    }

    #[test]
    fn test_parse_comment_extra_fields_ignored() {
        let config = create_test_config(Some("secret"));
        let handler = GitHubWebhookHandler::new(config, None);

        let comment_json = serde_json::json!({
            "id": 1,
            "user": { "id": 1, "login": "u", "type": "User" },
            "body": "test",
            "path": "file.rs",
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z",
            "html_url": "https://example.com",
            "node_id": "IC_kwDOXXXXXX",
            "diff_hunk": "@@ -1,5 +1,5 @@",
            "commit_id": "abc123",
            "author_association": "OWNER",
            "reactions": { "+1": 2, "-1": 0 }
        });

        let comment = handler.parse_comment(&comment_json).unwrap();
        assert_eq!(comment.id, 1);
        assert_eq!(comment.body, "test");
    }

    #[test]
    fn test_parse_comment_bot_user() {
        let config = create_test_config(Some("secret"));
        let handler = GitHubWebhookHandler::new(config, None);

        let comment_json = serde_json::json!({
            "id": 1,
            "user": {
                "id": 41898282,
                "login": "github-actions[bot]",
                "type": "Bot"
            },
            "body": "CI passed",
            "path": "README.md",
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z",
            "html_url": "https://example.com"
        });

        let comment = handler.parse_comment(&comment_json).unwrap();
        assert_eq!(comment.user.user_type, Some("Bot".to_string()));
        assert_eq!(comment.user.login, "github-actions[bot]");
    }

    #[test]
    fn test_parse_comment_user_null() {
        let config = create_test_config(Some("secret"));
        let handler = GitHubWebhookHandler::new(config, None);

        let comment_json = serde_json::json!({
            "id": 1,
            "user": null,
            "body": "test",
            "path": "file.rs",
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z",
            "html_url": "https://example.com"
        });

        let comment = handler.parse_comment(&comment_json).unwrap();
        // null user -> .get("user") returns Some(Value::Null)
        // map closure on Value::Null -> .get("id") returns None -> defaults
        assert_eq!(comment.user.id, 0);
        assert_eq!(comment.user.login, "");
    }

    #[test]
    fn test_parse_comment_multiline_body() {
        let config = create_test_config(Some("secret"));
        let handler = GitHubWebhookHandler::new(config, None);

        let body = "This needs fixing:\n```rust\nfn main() {\n    todo!()\n}\n```\nPlease update.";
        let comment_json = serde_json::json!({
            "id": 1,
            "user": { "id": 1, "login": "u", "type": "User" },
            "body": body,
            "path": "src/main.rs",
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z",
            "html_url": "https://example.com"
        });

        let comment = handler.parse_comment(&comment_json).unwrap();
        assert_eq!(comment.body, body);
    }

    #[test]
    fn test_parse_comment_position_as_string_coercion() {
        let config = create_test_config(Some("secret"));
        let handler = GitHubWebhookHandler::new(config, None);

        // If position is provided as a string, as_i64() returns None
        let comment_json = serde_json::json!({
            "id": 1,
            "user": { "id": 1, "login": "u", "type": "User" },
            "body": "test",
            "path": "file.rs",
            "position": "10",
            "line": "42",
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z",
            "html_url": "https://example.com"
        });

        let comment = handler.parse_comment(&comment_json).unwrap();
        assert!(
            comment.position.is_none(),
            "String position should not coerce to i64"
        );
        assert!(
            comment.line.is_none(),
            "String line should not coerce to i64"
        );
    }

    // process_webhook — additional event routing tests

    #[tokio::test]
    async fn test_process_webhook_ping_event_ignored() {
        let handler = GitHubWebhookHandler::new(create_test_config(Some("test_secret")), None);
        let payload = b"{}";
        let sig = make_valid_signature("test_secret", payload);
        let headers = make_headers("ping", &sig);

        let result = handler
            .process_webhook(payload, &serde_json::json!({}), &headers)
            .await;
        assert!(
            !result.unwrap().is_processed(),
            "Ping event should be ignored"
        );
    }

    #[tokio::test]
    async fn test_process_webhook_issues_event_ignored() {
        let handler = GitHubWebhookHandler::new(create_test_config(Some("test_secret")), None);
        let payload = b"{}";
        let sig = make_valid_signature("test_secret", payload);
        let headers = make_headers("issues", &sig);

        let result = handler
            .process_webhook(payload, &serde_json::json!({}), &headers)
            .await;
        assert!(
            !result.unwrap().is_processed(),
            "Issues event should be ignored"
        );
    }

    #[tokio::test]
    async fn test_process_webhook_check_suite_event_ignored() {
        let handler = GitHubWebhookHandler::new(create_test_config(Some("test_secret")), None);
        let payload = b"{}";
        let sig = make_valid_signature("test_secret", payload);
        let headers = make_headers("check_suite", &sig);

        let result = handler
            .process_webhook(payload, &serde_json::json!({}), &headers)
            .await;
        assert!(
            !result.unwrap().is_processed(),
            "Check suite event should be ignored"
        );
    }

    #[tokio::test]
    async fn test_process_webhook_pull_request_closed_merged() {
        let handler = GitHubWebhookHandler::new(create_test_config(Some("test_secret")), None);
        let payload_value = serde_json::json!({
            "action": "closed",
            "pull_request": {
                "html_url": "https://github.com/owner/repo/pull/42",
                "merged": true
            }
        });
        let payload_bytes = serde_json::to_vec(&payload_value).unwrap();
        let sig = make_valid_signature("test_secret", &payload_bytes);
        let headers = make_headers("pull_request", &sig);

        let result = handler
            .process_webhook(&payload_bytes, &payload_value, &headers)
            .await
            .unwrap();
        assert_eq!(
            result,
            WebhookAction::PrClosed {
                pr_url: "https://github.com/owner/repo/pull/42".to_string(),
                merged: true
            }
        );
    }

    #[tokio::test]
    async fn test_process_webhook_pull_request_closed_not_merged() {
        let handler = GitHubWebhookHandler::new(create_test_config(Some("test_secret")), None);
        let payload_value = serde_json::json!({
            "action": "closed",
            "pull_request": {
                "html_url": "https://github.com/owner/repo/pull/42",
                "merged": false
            }
        });
        let payload_bytes = serde_json::to_vec(&payload_value).unwrap();
        let sig = make_valid_signature("test_secret", &payload_bytes);
        let headers = make_headers("pull_request", &sig);

        let result = handler
            .process_webhook(&payload_bytes, &payload_value, &headers)
            .await
            .unwrap();
        assert_eq!(
            result,
            WebhookAction::PrClosed {
                pr_url: "https://github.com/owner/repo/pull/42".to_string(),
                merged: false
            }
        );
    }

    #[tokio::test]
    async fn test_process_webhook_pull_request_opened_ignored() {
        let handler = GitHubWebhookHandler::new(create_test_config(Some("test_secret")), None);
        let payload_value = serde_json::json!({
            "action": "opened",
            "pull_request": {
                "html_url": "https://github.com/owner/repo/pull/42",
                "merged": false
            }
        });
        let payload_bytes = serde_json::to_vec(&payload_value).unwrap();
        let sig = make_valid_signature("test_secret", &payload_bytes);
        let headers = make_headers("pull_request", &sig);

        let result = handler
            .process_webhook(&payload_bytes, &payload_value, &headers)
            .await
            .unwrap();
        assert_eq!(result, WebhookAction::Ignored);
    }

    #[tokio::test]
    async fn test_process_webhook_pull_request_missing_pr_url_ignored() {
        let handler = GitHubWebhookHandler::new(create_test_config(Some("test_secret")), None);
        let payload_value = serde_json::json!({
            "action": "closed",
            "pull_request": {
                "merged": true
            }
        });
        let payload_bytes = serde_json::to_vec(&payload_value).unwrap();
        let sig = make_valid_signature("test_secret", &payload_bytes);
        let headers = make_headers("pull_request", &sig);

        let result = handler
            .process_webhook(&payload_bytes, &payload_value, &headers)
            .await
            .unwrap();
        assert_eq!(result, WebhookAction::Ignored);
    }

    #[tokio::test]
    async fn test_process_webhook_pull_request_missing_payload_ignored() {
        let handler = GitHubWebhookHandler::new(create_test_config(Some("test_secret")), None);
        let payload_value = serde_json::json!({
            "action": "closed"
        });
        let payload_bytes = serde_json::to_vec(&payload_value).unwrap();
        let sig = make_valid_signature("test_secret", &payload_bytes);
        let headers = make_headers("pull_request", &sig);

        let result = handler
            .process_webhook(&payload_bytes, &payload_value, &headers)
            .await
            .unwrap();
        assert_eq!(result, WebhookAction::Ignored);
    }

    #[tokio::test]
    async fn test_process_webhook_empty_event_type_ignored() {
        let handler = GitHubWebhookHandler::new(create_test_config(Some("test_secret")), None);
        let payload = b"{}";
        let sig = make_valid_signature("test_secret", payload);
        let headers = make_headers("", &sig);

        let result = handler
            .process_webhook(payload, &serde_json::json!({}), &headers)
            .await;
        assert!(
            !result.unwrap().is_processed(),
            "Empty event type should be ignored"
        );
    }

    #[tokio::test]
    async fn test_process_webhook_no_secret_configured() {
        // Handler with no webhook secret should reject all requests
        let handler = GitHubWebhookHandler::new(create_test_config(None), None);
        let payload = b"{}";
        // Even with a signature header, it should fail
        let mut headers = HashMap::new();
        headers.insert(
            "x-hub-signature-256".to_string(),
            "sha256=anything".to_string(),
        );
        headers.insert(
            "x-github-event".to_string(),
            "pull_request_review".to_string(),
        );

        let result = handler
            .process_webhook(payload, &serde_json::json!({}), &headers)
            .await;
        assert!(
            result.is_err(),
            "No webhook secret configured should result in signature verification error"
        );
    }

    // handle_review_submitted — additional cases

    #[tokio::test]
    async fn test_handle_review_missing_action() {
        let handler = GitHubWebhookHandler::new(create_test_config(Some("test_secret")), None);
        // No "action" field at all
        let payload = serde_json::json!({
            "review": { "id": 1 },
            "pull_request": { "html_url": "https://github.com/owner/repo/pull/1" }
        });

        let result = handler.handle_review_submitted(&payload).await;
        assert!(
            !result.unwrap().is_processed(),
            "Missing action should be treated as non-submitted"
        );
    }

    #[tokio::test]
    async fn test_handle_review_dismissed_action() {
        let handler = GitHubWebhookHandler::new(create_test_config(Some("test_secret")), None);
        let payload = serde_json::json!({ "action": "dismissed" });

        let result = handler.handle_review_submitted(&payload).await;
        assert!(
            !result.unwrap().is_processed(),
            "Dismissed action should return Ok(false)"
        );
    }

    #[tokio::test]
    async fn test_handle_review_changes_requested() {
        let pr_url = "https://github.com/owner/repo/pull/1";
        let mock = Arc::new(MockScmProvider::with_reviews(vec![CodeReview {
            id: 200,
            user: ReviewUser {
                id: 5,
                login: "reviewer2".to_string(),
                user_type: Some("User".to_string()),
            },
            body: Some("Please address the following...".to_string()),
            state: "CHANGES_REQUESTED".to_string(),
            submitted_at: Some("2024-03-01T10:00:00Z".to_string()),
            html_url: Some(format!("{pr_url}#pullrequestreview-200")),
        }]));
        let handler = make_handler_watching_pr(pr_url, mock);

        let payload = serde_json::json!({
            "action": "submitted",
            "review": {
                "id": 200,
                "user": { "id": 5, "login": "reviewer2", "type": "User" },
                "body": "Please address the following...",
                "state": "CHANGES_REQUESTED",
                "submitted_at": "2024-03-01T10:00:00Z"
            },
            "pull_request": {
                "html_url": pr_url,
                "number": 1
            }
        });

        let result = handler.handle_review_submitted(&payload).await;
        assert!(
            result.unwrap().is_processed(),
            "CHANGES_REQUESTED review on watched PR should return Ok(true)"
        );
    }

    #[tokio::test]
    async fn test_handle_review_commented_state() {
        let pr_url = "https://github.com/owner/repo/pull/1";
        let mock = Arc::new(MockScmProvider::with_reviews(vec![CodeReview {
            id: 300,
            user: ReviewUser {
                id: 7,
                login: "commenter".to_string(),
                user_type: Some("User".to_string()),
            },
            body: Some("Just a comment".to_string()),
            state: "COMMENTED".to_string(),
            submitted_at: Some("2024-03-01T12:00:00Z".to_string()),
            html_url: None,
        }]));
        let handler = make_handler_watching_pr(pr_url, mock);

        let payload = serde_json::json!({
            "action": "submitted",
            "review": {
                "id": 300,
                "user": { "id": 7, "login": "commenter", "type": "User" },
                "body": "Just a comment",
                "state": "COMMENTED",
                "submitted_at": "2024-03-01T12:00:00Z"
            },
            "pull_request": {
                "html_url": pr_url,
                "number": 1
            }
        });

        let result = handler.handle_review_submitted(&payload).await;
        assert!(
            result.unwrap().is_processed(),
            "COMMENTED review on watched PR should return Ok(true)"
        );
    }

    #[tokio::test]
    async fn test_handle_review_pending_is_case_insensitive() {
        let pr_url = "https://github.com/owner/repo/pull/1";
        let mock = Arc::new(MockScmProvider::new());
        let handler = make_handler_watching_pr(pr_url, mock);

        // Test lowercase "pending" — the code does .to_uppercase() before comparing
        let payload = serde_json::json!({
            "action": "submitted",
            "review": {
                "id": 100,
                "user": { "id": 1, "login": "reviewer", "type": "User" },
                "state": "pending",
                "submitted_at": "2024-01-15T10:00:00Z"
            },
            "pull_request": {
                "html_url": pr_url,
                "number": 1
            }
        });

        let result = handler.handle_review_submitted(&payload).await;
        assert!(
            !result.unwrap().is_processed(),
            "Lowercase 'pending' should also be skipped"
        );
    }

    #[tokio::test]
    async fn test_handle_review_pr_url_missing_html_url() {
        // Pull request has no html_url field — defaults to empty string
        let handler = GitHubWebhookHandler::new(create_test_config(Some("test_secret")), None);
        let payload = serde_json::json!({
            "action": "submitted",
            "review": {
                "id": 100,
                "user": { "id": 1, "login": "reviewer", "type": "User" },
                "state": "APPROVED",
                "submitted_at": "2024-01-15T10:00:00Z"
            },
            "pull_request": {
                "number": 1
            }
        });

        // No watcher, so returns Ok(false) — but should not panic
        let result = handler.handle_review_submitted(&payload).await;
        assert!(!result.unwrap().is_processed());
    }

    // handle_review_comment — additional cases

    #[tokio::test]
    async fn test_handle_comment_edited_action() {
        let handler = GitHubWebhookHandler::new(create_test_config(Some("test_secret")), None);
        let payload = serde_json::json!({ "action": "edited" });

        let result = handler.handle_review_comment(&payload).await;
        assert!(
            !result.unwrap().is_processed(),
            "Edited comment action should return Ok(false)"
        );
    }

    #[tokio::test]
    async fn test_handle_comment_deleted_action() {
        let handler = GitHubWebhookHandler::new(create_test_config(Some("test_secret")), None);
        let payload = serde_json::json!({ "action": "deleted" });

        let result = handler.handle_review_comment(&payload).await;
        assert!(
            !result.unwrap().is_processed(),
            "Deleted comment action should return Ok(false)"
        );
    }

    #[tokio::test]
    async fn test_handle_comment_missing_action() {
        let handler = GitHubWebhookHandler::new(create_test_config(Some("test_secret")), None);
        let payload = serde_json::json!({
            "comment": {
                "id": 1,
                "user": { "id": 1, "login": "u", "type": "User" },
                "body": "test",
                "path": "file.rs",
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z",
                "html_url": "https://example.com"
            },
            "pull_request": { "html_url": "https://github.com/owner/repo/pull/1" }
        });

        let result = handler.handle_review_comment(&payload).await;
        assert!(
            !result.unwrap().is_processed(),
            "Missing action should be treated as non-created"
        );
    }

    #[tokio::test]
    async fn test_handle_comment_missing_pull_request() {
        let handler = GitHubWebhookHandler::new(create_test_config(Some("test_secret")), None);
        let payload = serde_json::json!({
            "action": "created",
            "comment": {
                "id": 1,
                "user": { "id": 1, "login": "u", "type": "User" },
                "body": "test",
                "path": "file.rs",
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z",
                "html_url": "https://example.com"
            }
        });

        let result = handler.handle_review_comment(&payload).await;
        assert!(
            !result.unwrap().is_processed(),
            "Missing pull_request field should return Ok(false)"
        );
    }

    #[tokio::test]
    async fn test_handle_comment_no_watcher() {
        let handler = GitHubWebhookHandler::new(create_test_config(Some("test_secret")), None);
        let pr_url = "https://github.com/owner/repo/pull/1";
        let payload = comment_created_payload(pr_url);

        let result = handler.handle_review_comment(&payload).await;
        assert!(
            !result.unwrap().is_processed(),
            "No watcher should return Ok(false)"
        );
    }

    // constructor and handler struct tests

    #[test]
    fn test_new_handler_stores_config() {
        let config = create_test_config(Some("my_secret"));
        let handler = GitHubWebhookHandler::new(config, None);

        // Verify secret is stored by checking signature verification works
        let payload = b"data";
        let sig = make_valid_signature("my_secret", payload);
        let mut headers = HashMap::new();
        headers.insert("x-hub-signature-256".to_string(), sig);
        assert!(handler.verify_signature(payload, &headers));
    }

    #[test]
    fn test_new_handler_with_watcher() {
        let mock = Arc::new(MockScmProvider::new());
        let watcher = Arc::new(ReviewWatcher::new(mock));
        let config = create_test_config(Some("secret"));
        let handler = GitHubWebhookHandler::new(config, Some(watcher));
        assert!(handler.is_enabled());
    }

    #[test]
    fn test_new_handler_without_watcher() {
        let config = create_test_config(Some("secret"));
        let handler = GitHubWebhookHandler::new(config, None);
        assert!(!handler.is_enabled());
        assert!(handler.review_watcher.is_none());
    }

    // Full round-trip integration tests

    #[tokio::test]
    async fn test_full_roundtrip_review_approved() {
        let pr_url = "https://github.com/owner/repo/pull/5";
        let mock = Arc::new(MockScmProvider::with_reviews(vec![CodeReview {
            id: 777,
            user: ReviewUser {
                id: 42,
                login: "lead_reviewer".to_string(),
                user_type: Some("User".to_string()),
            },
            body: Some("Ship it!".to_string()),
            state: "APPROVED".to_string(),
            submitted_at: Some("2024-06-01T14:00:00Z".to_string()),
            html_url: Some(format!("{pr_url}#pullrequestreview-777")),
        }]));
        let handler = make_handler_watching_pr(pr_url, mock);

        let payload_value = serde_json::json!({
            "action": "submitted",
            "review": {
                "id": 777,
                "user": { "id": 42, "login": "lead_reviewer", "type": "User" },
                "body": "Ship it!",
                "state": "APPROVED",
                "submitted_at": "2024-06-01T14:00:00Z",
                "html_url": format!("{pr_url}#pullrequestreview-777")
            },
            "pull_request": {
                "html_url": pr_url,
                "number": 5
            }
        });

        let payload_bytes = serde_json::to_vec(&payload_value).unwrap();
        let sig = make_valid_signature("test_secret", &payload_bytes);
        let headers = make_headers("pull_request_review", &sig);

        let result = handler
            .process_webhook(&payload_bytes, &payload_value, &headers)
            .await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_processed());
    }

    #[tokio::test]
    async fn test_full_roundtrip_comment_on_watched_pr() {
        let pr_url = "https://github.com/owner/repo/pull/10";
        let mock = Arc::new(MockScmProvider::with_comments(vec![ReviewComment {
            id: 888,
            path: "lib/core.rs".to_string(),
            position: Some(20),
            original_position: Some(18),
            body: "Consider using an enum here".to_string(),
            user: ReviewUser {
                id: 99,
                login: "code_expert".to_string(),
                user_type: Some("User".to_string()),
            },
            created_at: "2024-06-01T15:00:00Z".to_string(),
            updated_at: "2024-06-01T15:00:00Z".to_string(),
            html_url: format!("{pr_url}#discussion_r888"),
            pull_request_review_id: Some(777),
            start_line: Some(18),
            line: Some(22),
            side: Some("RIGHT".to_string()),
        }]));
        let handler = make_handler_watching_pr(pr_url, mock);

        let payload_value = serde_json::json!({
            "action": "created",
            "comment": {
                "id": 888,
                "user": { "id": 99, "login": "code_expert", "type": "User" },
                "body": "Consider using an enum here",
                "path": "lib/core.rs",
                "position": 20,
                "line": 22,
                "start_line": 18,
                "side": "RIGHT",
                "created_at": "2024-06-01T15:00:00Z",
                "updated_at": "2024-06-01T15:00:00Z",
                "html_url": format!("{pr_url}#discussion_r888"),
                "pull_request_review_id": 777
            },
            "pull_request": {
                "html_url": pr_url,
                "number": 10
            }
        });

        let payload_bytes = serde_json::to_vec(&payload_value).unwrap();
        let sig = make_valid_signature("test_secret", &payload_bytes);
        let headers = make_headers("pull_request_review_comment", &sig);

        let result = handler
            .process_webhook(&payload_bytes, &payload_value, &headers)
            .await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_processed());
    }

    #[tokio::test]
    async fn test_process_webhook_signature_mismatch_returns_error() {
        let pr_url = "https://github.com/owner/repo/pull/1";
        let mock = Arc::new(MockScmProvider::new());
        let handler = make_handler_watching_pr(pr_url, mock);

        let payload_value = review_submitted_payload(pr_url);
        let payload_bytes = serde_json::to_vec(&payload_value).unwrap();
        // Use a wrong signature on purpose
        let headers = make_headers(
            "pull_request_review",
            "sha256=0000000000000000000000000000000000000000000000000000000000000000",
        );

        let result = handler
            .process_webhook(&payload_bytes, &payload_value, &headers)
            .await;
        assert!(result.is_err(), "Wrong signature should produce an error");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Invalid webhook signature"),
            "Error message should mention invalid signature, got: {err_msg}"
        );
    }

    // JSON payload edge cases

    #[tokio::test]
    async fn test_process_webhook_with_json_array_payload() {
        let handler = GitHubWebhookHandler::new(create_test_config(Some("test_secret")), None);
        let payload = b"[]";
        let sig = make_valid_signature("test_secret", payload);
        let headers = make_headers("pull_request_review", &sig);

        // JSON array is valid JSON but not an object — the handler should
        // handle this gracefully via .get() returning None
        let result = handler
            .process_webhook(payload, &serde_json::json!([]), &headers)
            .await;
        // action will be None -> unwrap_or("") -> not "submitted" -> Ok(false)
        assert!(!result.unwrap().is_processed());
    }

    #[tokio::test]
    async fn test_process_webhook_with_deeply_nested_payload() {
        let handler = GitHubWebhookHandler::new(create_test_config(Some("test_secret")), None);
        let payload_value = serde_json::json!({
            "action": "submitted",
            "review": {
                "id": 1,
                "user": {
                    "id": 1,
                    "login": "nested_user",
                    "type": "User",
                    "nested": {
                        "deeply": {
                            "nested": "value"
                        }
                    }
                },
                "state": "APPROVED"
            },
            "pull_request": {
                "html_url": "https://github.com/owner/repo/pull/1"
            }
        });
        let payload_bytes = serde_json::to_vec(&payload_value).unwrap();
        let sig = make_valid_signature("test_secret", &payload_bytes);
        let headers = make_headers("pull_request_review", &sig);

        // No watcher, so returns Ok(false), but should not crash
        let result = handler
            .process_webhook(&payload_bytes, &payload_value, &headers)
            .await;
        assert!(!result.unwrap().is_processed());
    }

    // Header validation edge cases

    #[test]
    fn test_verify_signature_with_only_unrelated_headers() {
        let config = create_test_config(Some("test_secret"));
        let handler = GitHubWebhookHandler::new(config, None);

        let payload = b"test";
        let mut headers = HashMap::new();
        headers.insert("content-type".to_string(), "application/json".to_string());
        headers.insert("x-github-event".to_string(), "push".to_string());
        // No x-hub-signature-256 header

        assert!(
            !handler.verify_signature(payload, &headers),
            "Missing signature header among other headers should fail"
        );
    }

    #[test]
    fn test_verify_signature_truncated_hex() {
        let config = create_test_config(Some("test_secret"));
        let handler = GitHubWebhookHandler::new(config, None);

        let payload = b"test payload";
        let full_sig = make_valid_signature("test_secret", payload);
        // Truncate the hex to half length
        let truncated = &full_sig[..full_sig.len() / 2];

        let mut headers = HashMap::new();
        headers.insert("x-hub-signature-256".to_string(), truncated.to_string());

        assert!(
            !handler.verify_signature(payload, &headers),
            "Truncated hex signature should not verify"
        );
    }

    #[test]
    fn test_verify_signature_with_newlines_in_hex() {
        let config = create_test_config(Some("test_secret"));
        let handler = GitHubWebhookHandler::new(config, None);

        let payload = b"test payload";
        let mut mac = Hmac::<Sha256>::new_from_slice(b"test_secret").unwrap();
        mac.update(payload);
        let hex = hex::encode(mac.finalize().into_bytes());
        // Insert a newline in the hex
        let bad_hex = format!("sha256={}\n", hex);

        let mut headers = HashMap::new();
        headers.insert("x-hub-signature-256".to_string(), bad_hex);

        assert!(
            !handler.verify_signature(payload, &headers),
            "Hex with trailing newline should not verify"
        );
    }

    #[test]
    fn test_verify_signature_with_spaces_in_hex() {
        let config = create_test_config(Some("test_secret"));
        let handler = GitHubWebhookHandler::new(config, None);

        let payload = b"test payload";
        let mut mac = Hmac::<Sha256>::new_from_slice(b"test_secret").unwrap();
        mac.update(payload);
        let hex = hex::encode(mac.finalize().into_bytes());
        let bad_hex = format!("sha256= {hex}");

        let mut headers = HashMap::new();
        headers.insert("x-hub-signature-256".to_string(), bad_hex);

        assert!(
            !handler.verify_signature(payload, &headers),
            "Hex with leading space should not verify"
        );
    }

    // Helper function tests

    #[test]
    fn test_make_valid_signature_produces_correct_format() {
        let sig = make_valid_signature("secret", b"payload");
        assert!(
            sig.starts_with("sha256="),
            "Signature should start with sha256="
        );
        let hex_part = sig.strip_prefix("sha256=").unwrap();
        assert_eq!(hex_part.len(), 64, "SHA-256 hex should be 64 characters");
        assert!(
            hex_part.chars().all(|c| c.is_ascii_hexdigit()),
            "Hex part should only contain hex digits"
        );
    }

    #[test]
    fn test_make_headers_produces_correct_headers() {
        let headers = make_headers("pull_request_review", "sha256=abc");
        assert_eq!(
            headers.get("x-github-event"),
            Some(&"pull_request_review".to_string())
        );
        assert_eq!(
            headers.get("x-hub-signature-256"),
            Some(&"sha256=abc".to_string())
        );
        assert_eq!(headers.len(), 2);
    }

    #[test]
    fn test_create_test_config_with_secret() {
        let config = create_test_config(Some("my_secret"));
        assert_eq!(
            config.webhook_secret.as_ref().map(|s| s.expose()),
            Some("my_secret")
        );
        assert_eq!(
            config.token.as_ref().map(|s| s.expose()),
            Some("test_token")
        );
        assert_eq!(config.review_trigger, "@claudear");
    }

    #[test]
    fn test_create_test_config_without_secret() {
        let config = create_test_config(None);
        assert!(config.webhook_secret.is_none());
    }
}
