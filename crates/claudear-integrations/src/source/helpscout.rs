//! HelpScout issue source adapter.
//!
//! Treats each HelpScout conversation as an incoming "payload" and each mailbox
//! as an "inbox". Uses the Mailbox API v2 with OAuth2 client-credentials.

use super::IssueSource;
use async_trait::async_trait;
use claudear_config::config::{HelpScoutConfig, ReplyAs};
use claudear_core::error::{Error, Result};
use claudear_core::http::{HttpClient, ReqwestHttpClient};
use claudear_core::types::{Issue, MatchPriority, MatchResult};
use serde::Deserialize;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Base URL for the HelpScout Mailbox API v2.
const HELPSCOUT_API_BASE: &str = "https://api.helpscout.net";

/// Refresh the access token this many seconds before it actually expires.
const TOKEN_EXPIRY_BUFFER_SECS: u64 = 60;

// ---- HelpScout API response shapes ----

#[derive(Debug, Deserialize)]
struct TokenResponse {
    access_token: String,
    #[serde(default)]
    expires_in: u64,
}

#[derive(Debug, Deserialize)]
struct ConversationsListResponse {
    #[serde(default, rename = "_embedded")]
    embedded: Option<ConversationsEmbedded>,
}

#[derive(Debug, Default, Deserialize)]
struct ConversationsEmbedded {
    #[serde(default)]
    conversations: Vec<HsConversation>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct HsConversation {
    id: u64,
    #[serde(default)]
    number: Option<u64>,
    #[serde(default)]
    subject: Option<String>,
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    mailbox_id: Option<u64>,
    #[serde(default)]
    preview: Option<String>,
    #[serde(default)]
    tags: Vec<HsTag>,
    #[serde(default)]
    primary_customer: Option<HsCustomer>,
    #[serde(default, rename = "_embedded")]
    embedded: Option<ThreadsEmbedded>,
}

#[derive(Debug, Default, Deserialize)]
struct HsTag {
    #[serde(default)]
    tag: String,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct HsCustomer {
    #[serde(default)]
    id: Option<u64>,
    #[serde(default)]
    email: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ThreadsListResponse {
    #[serde(default, rename = "_embedded")]
    embedded: Option<ThreadsEmbedded>,
}

#[derive(Debug, Default, Deserialize)]
struct ThreadsEmbedded {
    #[serde(default)]
    threads: Vec<HsThread>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct HsThread {
    #[serde(default, rename = "type")]
    thread_type: Option<String>,
    #[serde(default)]
    body: Option<String>,
}

/// HelpScout source: polls conversations from one or more mailboxes.
pub struct HelpScoutSource<H: HttpClient = ReqwestHttpClient> {
    config: HelpScoutConfig,
    http: H,
    /// Cached OAuth2 access token and the instant it should be refreshed at.
    token: Mutex<Option<(String, Instant)>>,
}

impl HelpScoutSource<ReqwestHttpClient> {
    /// Create a HelpScout source with the default HTTP client.
    pub fn new(config: HelpScoutConfig) -> Self {
        Self::with_http_client(config, ReqwestHttpClient::new())
    }
}

impl<H: HttpClient> HelpScoutSource<H> {
    /// Create a HelpScout source with a custom HTTP client (for testing).
    pub fn with_http_client(config: HelpScoutConfig, http: H) -> Self {
        Self {
            config,
            http,
            token: Mutex::new(None),
        }
    }

    /// Return a valid bearer token, fetching a new one if needed.
    async fn access_token(&self) -> Result<String> {
        // Fast path: a cached, non-expired token.
        if let Ok(guard) = self.token.lock() {
            if let Some((tok, refresh_at)) = guard.as_ref() {
                if Instant::now() < *refresh_at {
                    return Ok(tok.clone());
                }
            }
        }

        // Fetch a fresh token via client-credentials.
        let body = format!(
            "grant_type=client_credentials&client_id={}&client_secret={}",
            urlencoding(self.config.app_id.expose()),
            urlencoding(self.config.app_secret.expose()),
        );
        let resp = self
            .http
            .post(
                &format!("{HELPSCOUT_API_BASE}/v2/oauth2/token"),
                vec![(
                    "Content-Type",
                    "application/x-www-form-urlencoded".to_string(),
                )],
                &body,
            )
            .await?;
        if !resp.is_success() {
            return Err(Error::source(
                "helpscout",
                format!("token request failed ({}): {}", resp.status, resp.body),
            ));
        }
        let token: TokenResponse = resp.json()?;
        let ttl = token
            .expires_in
            .saturating_sub(TOKEN_EXPIRY_BUFFER_SECS)
            .max(1);
        if let Ok(mut guard) = self.token.lock() {
            *guard = Some((
                token.access_token.clone(),
                Instant::now() + Duration::from_secs(ttl),
            ));
        }
        Ok(token.access_token)
    }

    /// Authorized GET returning the raw response.
    async fn api_get(&self, path_and_query: &str) -> Result<claudear_core::http::HttpResponse> {
        let token = self.access_token().await?;
        self.http
            .get(
                &format!("{HELPSCOUT_API_BASE}{path_and_query}"),
                vec![("Authorization", format!("Bearer {token}"))],
            )
            .await
    }

    /// Fetch the threads for a conversation (used to build full context).
    async fn fetch_threads(&self, conversation_id: &str) -> Vec<HsThread> {
        match self
            .api_get(&format!("/v2/conversations/{conversation_id}/threads"))
            .await
        {
            Ok(resp) if resp.is_success() => resp
                .json::<ThreadsListResponse>()
                .ok()
                .and_then(|r| r.embedded)
                .map(|e| e.threads)
                .unwrap_or_default(),
            _ => Vec::new(),
        }
    }

    /// Map a HelpScout conversation to a generic `Issue`.
    fn map_conversation(&self, c: HsConversation) -> Issue {
        let tags: Vec<String> = c.tags.iter().map(|t| t.tag.clone()).collect();
        let number = c.number.unwrap_or(c.id);
        let short_id = format!("HS-{number}");
        let title = c
            .subject
            .clone()
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| format!("Conversation {number}"));
        let url = format!("https://secure.helpscout.net/conversation/{}", c.id);

        let mut issue = Issue::new(c.id.to_string(), short_id, title, url, "helpscout");

        // Prefer the latest customer thread body; fall back to the preview.
        let body = c
            .embedded
            .as_ref()
            .and_then(|e| latest_customer_body(&e.threads))
            .or_else(|| c.preview.clone());
        issue.description = body;

        if let Some(mailbox_id) = c.mailbox_id {
            issue.set_metadata("mailbox_id", mailbox_id.to_string());
        }
        issue.set_metadata("tags", &tags);
        // `labels` is the key the pipeline reads to populate FixAttempt.issue_labels
        // (used by the bug-detection heuristic).
        issue.set_metadata("labels", &tags);
        if let Some(status) = c.status.as_ref() {
            issue.set_metadata("status", status);
        }
        if let Some(customer) = c.primary_customer.as_ref() {
            if let Some(email) = customer.email.as_ref() {
                issue.set_metadata("customer_email", email);
            }
            if let Some(id) = customer.id {
                issue.set_metadata("customer_id", id.to_string());
            }
        }
        issue
    }
}

/// Return the body of the most recent customer-authored thread, if any.
fn latest_customer_body(threads: &[HsThread]) -> Option<String> {
    threads
        .iter()
        .filter(|t| {
            t.thread_type
                .as_deref()
                .map(|t| t.eq_ignore_ascii_case("customer"))
                .unwrap_or(false)
        })
        .filter_map(|t| t.body.clone())
        .rfind(|b| !b.trim().is_empty())
}

/// Minimal application/x-www-form-urlencoded encoder for token request values.
fn urlencoding(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Build a Claude-facing context string from a HelpScout conversation issue.
fn format_helpscout_context(issue: &Issue) -> String {
    let mut ctx = format!("# HelpScout Conversation: {}\n\n", issue.short_id);
    ctx.push_str(&format!("**Subject:** {}\n", issue.title));
    ctx.push_str(&format!("**URL:** {}\n", issue.url));
    if let Some(status) = issue.get_metadata::<String>("status") {
        ctx.push_str(&format!("**Status:** {status}\n"));
    }
    if let Some(email) = issue.get_metadata::<String>("customer_email") {
        ctx.push_str(&format!("**Customer:** {email}\n"));
    }
    let tags: Vec<String> = issue.get_metadata("tags").unwrap_or_default();
    if !tags.is_empty() {
        ctx.push_str(&format!("**Tags:** {}\n", tags.join(", ")));
    }
    ctx.push('\n');
    if let Some(desc) = issue.description.as_ref() {
        ctx.push_str(&format!("## Customer message\n{desc}\n"));
    }
    ctx
}

#[async_trait]
impl<H: HttpClient + 'static> IssueSource for HelpScoutSource<H> {
    fn name(&self) -> &str {
        "helpscout"
    }

    fn display_name(&self) -> &str {
        "HelpScout"
    }

    async fn fetch_issues(&self) -> Result<Vec<Issue>> {
        let status = if self.config.trigger_status.trim().is_empty() {
            "active"
        } else {
            self.config.trigger_status.as_str()
        };

        let mut issues = Vec::new();
        for mailbox in &self.config.mailbox_ids {
            let path = format!(
                "/v2/conversations?mailbox={}&status={}&embed=threads",
                urlencoding(mailbox),
                urlencoding(status),
            );
            let resp = self.api_get(&path).await?;
            if !resp.is_success() {
                return Err(Error::source(
                    "helpscout",
                    format!("list conversations failed ({}): {}", resp.status, resp.body),
                ));
            }
            let list: ConversationsListResponse = resp.json()?;
            if let Some(embedded) = list.embedded {
                for c in embedded.conversations {
                    issues.push(self.map_conversation(c));
                }
            }
        }
        Ok(issues)
    }

    fn matches_criteria(&self, issue: &Issue) -> MatchResult {
        // No trigger tags configured -> process everything in the mailbox.
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
        Ok(format_helpscout_context(issue))
    }

    async fn get_issue(&self, issue_id: &str) -> Result<Issue> {
        let resp = self
            .api_get(&format!("/v2/conversations/{issue_id}"))
            .await?;
        if resp.is_not_found() {
            return Err(Error::issue_not_found("helpscout", issue_id));
        }
        if !resp.is_success() {
            return Err(Error::source(
                "helpscout",
                format!("get conversation failed ({}): {}", resp.status, resp.body),
            ));
        }
        let mut conversation: HsConversation = resp.json()?;
        // Ensure we have the full thread history for context.
        if conversation
            .embedded
            .as_ref()
            .map(|e| e.threads.is_empty())
            .unwrap_or(true)
        {
            let threads = self.fetch_threads(issue_id).await;
            conversation.embedded = Some(ThreadsEmbedded { threads });
        }
        Ok(self.map_conversation(conversation))
    }

    async fn resolve_issue(&self, issue_id: &str) -> Result<()> {
        // HelpScout uses JSON-PATCH to update conversation status.
        let token = self.access_token().await?;
        let body = r#"[{"op":"replace","path":"/status","value":"closed"}]"#;
        let resp = self
            .http
            .patch(
                &format!("{HELPSCOUT_API_BASE}/v2/conversations/{issue_id}"),
                vec![
                    ("Authorization", format!("Bearer {token}")),
                    ("Content-Type", "application/json".to_string()),
                ],
                body,
            )
            .await?;
        if resp.is_success() {
            Ok(())
        } else {
            Err(Error::source(
                "helpscout",
                format!("close conversation failed ({}): {}", resp.status, resp.body),
            ))
        }
    }

    async fn add_comment(&self, issue_id: &str, comment: &str) -> Result<()> {
        let token = self.access_token().await?;
        let text =
            serde_json::to_string(comment).map_err(|e| Error::Other(format!("JSON error: {e}")))?;

        // Note (internal) is the safe default; reply is customer-facing.
        let (endpoint, body) = match self.config.reply_as {
            ReplyAs::Note => (
                format!("/v2/conversations/{issue_id}/notes"),
                format!("{{\"text\":{text}}}"),
            ),
            ReplyAs::Reply => {
                // A customer-facing reply requires the customer id.
                let issue = self.get_issue(issue_id).await?;
                let customer_id = issue.get_metadata::<String>("customer_id").ok_or_else(|| {
                    Error::source("helpscout", "conversation has no customer id for reply")
                })?;
                (
                    format!("/v2/conversations/{issue_id}/reply"),
                    format!("{{\"text\":{text},\"customer\":{{\"id\":{customer_id}}}}}"),
                )
            }
        };

        let resp = self
            .http
            .post(
                &format!("{HELPSCOUT_API_BASE}{endpoint}"),
                vec![
                    ("Authorization", format!("Bearer {token}")),
                    ("Content-Type", "application/json".to_string()),
                ],
                &body,
            )
            .await?;
        if resp.is_success() {
            Ok(())
        } else {
            Err(Error::source(
                "helpscout",
                format!("post reply failed ({}): {}", resp.status, resp.body),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use claudear_core::http::HttpResponse;
    use std::collections::HashMap;

    /// Mock HTTP client: returns canned responses keyed by a substring of the URL.
    struct MockHttpClient {
        responses: Mutex<Vec<(String, u16, String)>>,
        requests: Mutex<Vec<(String, String, String)>>, // (method, url, body)
    }

    impl MockHttpClient {
        fn new(responses: Vec<(&str, u16, &str)>) -> Self {
            Self {
                responses: Mutex::new(
                    responses
                        .into_iter()
                        .map(|(k, s, b)| (k.to_string(), s, b.to_string()))
                        .collect(),
                ),
                requests: Mutex::new(Vec::new()),
            }
        }

        fn respond(&self, url: &str) -> HttpResponse {
            let responses = self.responses.lock().unwrap();
            for (key, status, body) in responses.iter() {
                if url.contains(key.as_str()) {
                    return HttpResponse {
                        status: *status,
                        body: body.clone(),
                    };
                }
            }
            HttpResponse {
                status: 404,
                body: "{}".to_string(),
            }
        }
    }

    #[async_trait]
    impl HttpClient for MockHttpClient {
        async fn get(&self, url: &str, _headers: Vec<(&str, String)>) -> Result<HttpResponse> {
            self.requests
                .lock()
                .unwrap()
                .push(("GET".into(), url.into(), String::new()));
            Ok(self.respond(url))
        }

        async fn post(
            &self,
            url: &str,
            _headers: Vec<(&str, String)>,
            body: &str,
        ) -> Result<HttpResponse> {
            self.requests
                .lock()
                .unwrap()
                .push(("POST".into(), url.into(), body.into()));
            Ok(self.respond(url))
        }

        async fn patch(
            &self,
            url: &str,
            _headers: Vec<(&str, String)>,
            body: &str,
        ) -> Result<HttpResponse> {
            self.requests
                .lock()
                .unwrap()
                .push(("PATCH".into(), url.into(), body.into()));
            Ok(self.respond(url))
        }
    }

    fn test_config() -> HelpScoutConfig {
        HelpScoutConfig {
            enabled: true,
            app_id: "app".into(),
            app_secret: "secret".into(),
            mailbox_ids: vec!["1".to_string()],
            trigger_tags: vec!["bug".to_string()],
            trigger_status: "active".to_string(),
            reply_as: ReplyAs::Note,
            webhook_secret: None,
            max_issues_per_cycle: None,
            max_concurrent: None,
            poll_interval_ms: None,
        }
    }

    const TOKEN_BODY: &str =
        r#"{"access_token":"tok-123","expires_in":3600,"token_type":"bearer"}"#;

    #[tokio::test]
    async fn test_fetch_issues_maps_conversations() {
        let list = r#"{"_embedded":{"conversations":[
            {"id":42,"number":7,"subject":"Login crashes","status":"active","mailboxId":1,
             "preview":"It crashes on submit","tags":[{"id":1,"tag":"bug"}],
             "primaryCustomer":{"id":99,"email":"a@b.com"}}
        ]}}"#;
        let mock = MockHttpClient::new(vec![
            ("/v2/oauth2/token", 200, TOKEN_BODY),
            ("/v2/conversations?", 200, list),
        ]);
        let source = HelpScoutSource::with_http_client(test_config(), mock);

        let issues = source.fetch_issues().await.unwrap();
        assert_eq!(issues.len(), 1);
        let i = &issues[0];
        assert_eq!(i.id, "42");
        assert_eq!(i.short_id, "HS-7");
        assert_eq!(i.title, "Login crashes");
        assert_eq!(i.source, "helpscout");
        assert_eq!(i.description.as_deref(), Some("It crashes on submit"));
        assert_eq!(i.get_metadata::<String>("mailbox_id").as_deref(), Some("1"));
        assert_eq!(
            i.get_metadata::<String>("customer_email").as_deref(),
            Some("a@b.com")
        );
        assert_eq!(
            i.get_metadata::<Vec<String>>("labels"),
            Some(vec!["bug".to_string()])
        );
    }

    #[test]
    fn test_matches_criteria_by_tag() {
        let source = HelpScoutSource::with_http_client(test_config(), MockHttpClient::new(vec![]));
        let mut issue = Issue::new("1", "HS-1", "t", "u", "helpscout");
        issue.set_metadata("tags", vec!["bug".to_string(), "p1".to_string()]);
        assert!(source.matches_criteria(&issue).matches);

        let mut other = Issue::new("2", "HS-2", "t", "u", "helpscout");
        other.set_metadata("tags", vec!["question".to_string()]);
        assert!(!source.matches_criteria(&other).matches);
    }

    #[test]
    fn test_matches_criteria_no_trigger_tags_matches_all() {
        let mut cfg = test_config();
        cfg.trigger_tags = vec![];
        let source = HelpScoutSource::with_http_client(cfg, MockHttpClient::new(vec![]));
        let issue = Issue::new("1", "HS-1", "t", "u", "helpscout");
        assert!(source.matches_criteria(&issue).matches);
    }

    #[tokio::test]
    async fn test_add_comment_posts_note() {
        let mock = MockHttpClient::new(vec![
            ("/v2/oauth2/token", 200, TOKEN_BODY),
            ("/notes", 201, ""),
        ]);
        let source = HelpScoutSource::with_http_client(test_config(), mock);
        source
            .add_comment("42", "Thanks for the report!")
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn test_resolve_issue_closes_conversation() {
        let mock = MockHttpClient::new(vec![
            ("/v2/oauth2/token", 200, TOKEN_BODY),
            ("/v2/conversations/42", 200, ""),
        ]);
        let source = HelpScoutSource::with_http_client(test_config(), mock);
        source.resolve_issue("42").await.unwrap();
    }

    #[test]
    fn test_latest_customer_body_prefers_last_customer_thread() {
        let threads = vec![
            HsThread {
                thread_type: Some("customer".into()),
                body: Some("first".into()),
            },
            HsThread {
                thread_type: Some("note".into()),
                body: Some("internal".into()),
            },
            HsThread {
                thread_type: Some("customer".into()),
                body: Some("latest".into()),
            },
        ];
        assert_eq!(latest_customer_body(&threads).as_deref(), Some("latest"));
    }

    #[test]
    fn test_urlencoding_escapes_reserved() {
        assert_eq!(urlencoding("a b&c"), "a%20b%26c");
        assert_eq!(urlencoding("plain-123_.~"), "plain-123_.~");
    }

    #[test]
    fn test_format_context_includes_fields() {
        let mut issue = Issue::new("42", "HS-7", "Login crashes", "url", "helpscout");
        issue.description = Some("steps".into());
        issue.set_metadata("status", "active");
        issue.set_metadata("customer_email", "a@b.com");
        issue.set_metadata("tags", vec!["bug".to_string()]);
        let ctx = format_helpscout_context(&issue);
        assert!(ctx.contains("HS-7"));
        assert!(ctx.contains("Login crashes"));
        assert!(ctx.contains("a@b.com"));
        assert!(ctx.contains("bug"));
        assert!(ctx.contains("steps"));
        let _ = HashMap::<String, String>::new();
    }
}
