//! HelpScout webhook handler.
//!
//! Handles conversation events (e.g. `convo.created`,
//! `convo.customer.reply.created`). HelpScout signs the raw request body with
//! HMAC-SHA1 (base64) in the `X-HelpScout-Signature` header.

use super::WebhookHandler;
use async_trait::async_trait;
use base64::Engine;
use claudear_config::config::HelpScoutConfig;
use claudear_core::error::Result;
use claudear_core::types::{Issue, MatchPriority, MatchResult};
use hmac::{Hmac, Mac};
use sha1::Sha1;
use std::collections::HashMap;
use subtle::ConstantTimeEq;

/// Webhook handler for HelpScout.
pub struct HelpScoutWebhookHandler {
    config: HelpScoutConfig,
}

impl HelpScoutWebhookHandler {
    /// Create a new HelpScout webhook handler.
    pub fn new(config: HelpScoutConfig) -> Self {
        Self { config }
    }

    /// Extract tag names from a conversation payload.
    fn extract_tags(data: &serde_json::Value) -> Vec<String> {
        data.get("tags")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|t| {
                        // Tags may be objects ({"tag":"bug"}) or bare strings.
                        t.get("tag")
                            .and_then(|s| s.as_str())
                            .or_else(|| t.as_str())
                            .map(|s| s.to_string())
                    })
                    .collect()
            })
            .unwrap_or_default()
    }
}

#[async_trait]
impl WebhookHandler for HelpScoutWebhookHandler {
    fn source_name(&self) -> &str {
        "helpscout"
    }

    fn verify_signature(&self, payload: &[u8], headers: &HashMap<String, String>) -> bool {
        let secret = match &self.config.webhook_secret {
            Some(s) => s,
            None => {
                tracing::error!(
                    source = "helpscout",
                    "No webhook secret configured - rejecting request for security"
                );
                return false;
            }
        };

        let signature = match headers.get("x-helpscout-signature") {
            Some(s) => s,
            None => return false,
        };

        let mut mac = match Hmac::<Sha1>::new_from_slice(secret.expose().as_bytes()) {
            Ok(m) => m,
            Err(_) => return false,
        };
        mac.update(payload);
        let expected =
            base64::engine::general_purpose::STANDARD.encode(mac.finalize().into_bytes());

        signature.as_bytes().ct_eq(expected.as_bytes()).into()
    }

    async fn parse_payload(&self, payload: &serde_json::Value) -> Result<Option<Issue>> {
        // The webhook body is the conversation object. Require an id.
        let id = match payload.get("id").and_then(|v| v.as_u64()) {
            Some(id) => id,
            None => return Ok(None),
        };

        let number = payload.get("number").and_then(|v| v.as_u64()).unwrap_or(id);
        let short_id = format!("HS-{number}");
        let title = payload
            .get("subject")
            .and_then(|v| v.as_str())
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("Conversation {number}"));
        let url = format!("https://secure.helpscout.net/conversation/{id}");

        let mut issue = Issue::new(id.to_string(), short_id, title, url, "helpscout");

        // Latest customer thread body, else the preview.
        let body = payload
            .get("_embedded")
            .and_then(|e| e.get("threads"))
            .and_then(|v| v.as_array())
            .and_then(|threads| {
                threads
                    .iter()
                    .filter(|t| {
                        t.get("type")
                            .and_then(|v| v.as_str())
                            .map(|t| t.eq_ignore_ascii_case("customer"))
                            .unwrap_or(false)
                    })
                    .filter_map(|t| t.get("body").and_then(|v| v.as_str()))
                    .rfind(|b| !b.trim().is_empty())
                    .map(|s| s.to_string())
            })
            .or_else(|| {
                payload
                    .get("preview")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
            });
        issue.description = body;

        let tags = Self::extract_tags(payload);
        if let Some(mailbox_id) = payload.get("mailboxId").and_then(|v| v.as_u64()) {
            issue.set_metadata("mailbox_id", mailbox_id.to_string());
        }
        issue.set_metadata("tags", &tags);
        issue.set_metadata("labels", &tags);
        if let Some(status) = payload.get("status").and_then(|v| v.as_str()) {
            issue.set_metadata("status", status);
        }
        if let Some(customer) = payload.get("primaryCustomer") {
            if let Some(email) = customer.get("email").and_then(|v| v.as_str()) {
                issue.set_metadata("customer_email", email);
            }
            if let Some(cid) = customer.get("id").and_then(|v| v.as_u64()) {
                issue.set_metadata("customer_id", cid.to_string());
            }
        }

        Ok(Some(issue))
    }

    fn matches_criteria(&self, issue: &Issue) -> MatchResult {
        if self.config.trigger_tags.is_empty() {
            return MatchResult::matched("No trigger tags configured", MatchPriority::Normal);
        }
        let tags: Vec<String> = issue.get_metadata("tags").unwrap_or_default();
        let matched = self
            .config
            .trigger_tags
            .iter()
            .any(|trigger| tags.iter().any(|t| t.eq_ignore_ascii_case(trigger)));
        if matched {
            MatchResult::matched("Matches trigger tags", MatchPriority::Normal)
        } else {
            MatchResult::not_matched("No matching trigger tags")
        }
    }

    async fn build_issue_context(&self, issue: &Issue) -> Result<String> {
        let mut ctx = format!("# HelpScout Conversation: {}\n\n", issue.short_id);
        ctx.push_str(&format!("**Subject:** {}\n", issue.title));
        ctx.push_str(&format!("**URL:** {}\n", issue.url));
        if let Some(email) = issue.get_metadata::<String>("customer_email") {
            ctx.push_str(&format!("**Customer:** {email}\n"));
        }
        let tags: Vec<String> = issue.get_metadata("tags").unwrap_or_default();
        if !tags.is_empty() {
            ctx.push_str(&format!("**Tags:** {}\n", tags.join(", ")));
        }
        if let Some(desc) = issue.description.as_ref() {
            ctx.push_str(&format!("\n## Customer message\n{desc}\n"));
        }
        Ok(ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(secret: Option<&str>) -> HelpScoutConfig {
        HelpScoutConfig {
            enabled: true,
            webhook_secret: secret.map(|s| s.into()),
            trigger_tags: vec!["bug".to_string()],
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn test_parse_payload_maps_conversation() {
        let handler = HelpScoutWebhookHandler::new(config(None));
        let payload = serde_json::json!({
            "id": 42,
            "number": 7,
            "subject": "App crashes",
            "status": "active",
            "mailboxId": 1,
            "preview": "Crashes on launch",
            "tags": [{"id": 1, "tag": "bug"}],
            "primaryCustomer": {"id": 99, "email": "a@b.com"}
        });
        let issue = handler.parse_payload(&payload).await.unwrap().unwrap();
        assert_eq!(issue.id, "42");
        assert_eq!(issue.short_id, "HS-7");
        assert_eq!(issue.title, "App crashes");
        assert_eq!(issue.description.as_deref(), Some("Crashes on launch"));
        assert_eq!(
            issue.get_metadata::<String>("mailbox_id").as_deref(),
            Some("1")
        );
        assert!(handler.matches_criteria(&issue).matches);
    }

    #[tokio::test]
    async fn test_parse_payload_without_id_is_ignored() {
        let handler = HelpScoutWebhookHandler::new(config(None));
        let payload = serde_json::json!({"subject": "no id"});
        assert!(handler.parse_payload(&payload).await.unwrap().is_none());
    }

    #[test]
    fn test_verify_signature_roundtrip() {
        let handler = HelpScoutWebhookHandler::new(config(Some("shhh")));
        let body = br#"{"id":42}"#;
        // Compute the expected base64 HMAC-SHA1 the same way HelpScout would.
        let mut mac = Hmac::<Sha1>::new_from_slice(b"shhh").unwrap();
        mac.update(body);
        let sig = base64::engine::general_purpose::STANDARD.encode(mac.finalize().into_bytes());

        let mut headers = HashMap::new();
        headers.insert("x-helpscout-signature".to_string(), sig);
        assert!(handler.verify_signature(body, &headers));

        // Wrong signature is rejected.
        let mut bad = HashMap::new();
        bad.insert("x-helpscout-signature".to_string(), "nope".to_string());
        assert!(!handler.verify_signature(body, &bad));
    }

    #[test]
    fn test_verify_signature_no_secret_rejects() {
        let handler = HelpScoutWebhookHandler::new(config(None));
        let headers = HashMap::new();
        assert!(!handler.verify_signature(b"{}", &headers));
    }
}
