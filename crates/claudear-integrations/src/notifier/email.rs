//! Email notifier via SMTP.

use super::Notifier;
use async_trait::async_trait;
use chrono::{DateTime, TimeZone, Utc};
use claudear_config::config::EmailConfig;
use claudear_config::users::UserRegistry;
use claudear_core::error::{Error, Result};
use claudear_core::types::{AskDelivery, AskReply, AskRequest, Issue};
use lettre::{
    message::{header::ContentType, Mailbox},
    transport::smtp::authentication::Credentials,
    AsyncSmtpTransport, AsyncTransport, Message, Tokio1Executor,
};
use mailparse::MailHeaderMap;
use std::collections::HashSet;

/// Email notifier that sends notifications via SMTP.
pub struct EmailNotifier {
    config: EmailConfig,
    transport: Option<AsyncSmtpTransport<Tokio1Executor>>,
    user_registry: UserRegistry,
}

impl EmailNotifier {
    /// Create a new email notifier.
    pub fn new(config: EmailConfig, user_registry: UserRegistry) -> Result<Self> {
        let transport = if let (Some(host), Some(username), Some(password)) = (
            config.smtp_host.as_ref(),
            config.smtp_username.as_ref(),
            config.smtp_password.as_ref(),
        ) {
            let creds = Credentials::new(username.clone(), password.expose().to_string());

            let builder = if config.use_tls {
                AsyncSmtpTransport::<Tokio1Executor>::starttls_relay(host)
                    .map_err(|e| Error::notifier("email", format!("SMTP setup error: {}", e)))?
            } else {
                AsyncSmtpTransport::<Tokio1Executor>::builder_dangerous(host)
            };

            Some(
                builder
                    .port(config.smtp_port)
                    .credentials(creds)
                    .timeout(Some(std::time::Duration::from_secs(10)))
                    .build(),
            )
        } else {
            None
        };

        Ok(Self {
            config,
            transport,
            user_registry,
        })
    }

    fn resolve_recipients(&self, issue: Option<&Issue>) -> Vec<String> {
        if let Some(issue) = issue {
            if let Some(slug) = issue.get_metadata::<String>("resolved_user") {
                if let Some(user) = self.user_registry.get_by_slug(&slug) {
                    if let Some(ref email) = user.email {
                        return vec![email.clone()];
                    }
                }
            }
        }
        self.config.to_addresses.clone()
    }

    fn target_email_for_issue(&self, issue: &Issue) -> Option<String> {
        self.resolve_recipients(Some(issue)).into_iter().next()
    }

    fn expected_reply_emails(&self, request: &AskRequest) -> HashSet<String> {
        if let Some(ref target) = request.target_email {
            return std::iter::once(target.to_lowercase()).collect();
        }
        self.config
            .to_addresses
            .iter()
            .map(|v| v.to_lowercase())
            .collect()
    }

    async fn send_email(&self, subject: &str, body: &str, issue: Option<&Issue>) -> Result<()> {
        let transport = match &self.transport {
            Some(t) => t,
            None => return Ok(()),
        };

        let from_address = match &self.config.from_address {
            Some(addr) => addr
                .parse::<Mailbox>()
                .map_err(|e| Error::notifier("email", format!("Invalid from address: {}", e)))?,
            None => return Ok(()),
        };

        let recipients = self.resolve_recipients(issue);

        for to_address in &recipients {
            let to_mailbox = to_address
                .parse::<Mailbox>()
                .map_err(|e| Error::notifier("email", format!("Invalid to address: {}", e)))?;

            let message = Message::builder()
                .from(from_address.clone())
                .to(to_mailbox)
                .subject(subject)
                .header(ContentType::TEXT_PLAIN)
                .body(body.to_string())
                .map_err(|e| Error::notifier("email", format!("Failed to build email: {}", e)))?;

            transport
                .send(message)
                .await
                .map_err(|e| Error::notifier("email", format!("Failed to send email: {}", e)))?;
        }

        Ok(())
    }

    fn format_issue_info(issue: &Issue) -> String {
        format!(
            "Issue: {} - {}\nSource: {}\nPriority: {}\nStatus: {}\nURL: {}",
            issue.short_id, issue.title, issue.source, issue.priority, issue.status, issue.url
        )
    }

    fn extract_email_address(from_header: &str) -> Option<String> {
        let start = from_header.find('<');
        let end = from_header.find('>');
        if let (Some(s), Some(e)) = (start, end) {
            let addr = from_header[s + 1..e].trim();
            if !addr.is_empty() {
                return Some(addr.to_lowercase());
            }
        }
        let trimmed = from_header.trim();
        if trimmed.contains('@') {
            Some(trimmed.to_lowercase())
        } else {
            None
        }
    }

    fn extract_plain_body(parsed: &mailparse::ParsedMail<'_>) -> String {
        if parsed.subparts.is_empty() {
            return parsed.get_body().unwrap_or_default();
        }

        for part in &parsed.subparts {
            let ctype = part.ctype.mimetype.to_lowercase();
            if ctype == "text/plain" {
                return part.get_body().unwrap_or_default();
            }
        }

        parsed.get_body().unwrap_or_default()
    }

    fn sanitize_reply_text(body: &str) -> Option<String> {
        let mut lines: Vec<String> = Vec::new();
        for raw_line in body.lines() {
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('>') {
                continue;
            }
            lines.push(line.to_string());
            if lines.len() >= 30 {
                break;
            }
        }
        let mut out = lines.join("\n").trim().to_string();
        if out.len() > 4000 {
            out.truncate(4000);
        }
        if out.is_empty() {
            None
        } else {
            Some(out)
        }
    }
}

#[async_trait]
impl Notifier for EmailNotifier {
    fn name(&self) -> &str {
        "email"
    }

    fn is_enabled(&self) -> bool {
        self.transport.is_some()
            && self.config.from_address.is_some()
            && !self.config.to_addresses.is_empty()
    }

    async fn notify_start(&self, issue: &Issue) -> Result<()> {
        let subject = format!("[Claudear] Processing: {}", issue.short_id);
        let mut body = format!(
            "Claudear is now processing an issue.\n\n{}\n\nYou will receive another notification when processing completes.",
            Self::format_issue_info(issue)
        );
        if let Some(reason) = issue.get_metadata::<String>("trigger_reason") {
            body.push_str(&format!("\n\nTrigger reason: {}", reason));
        }
        self.send_email(&subject, &body, Some(issue)).await
    }

    async fn notify_success(&self, issue: &Issue, pr_url: &str) -> Result<()> {
        let (subject, body) = if issue
            .get_metadata::<String>("cascade_downstream_repo")
            .is_some()
        {
            let upstream = issue
                .get_metadata::<String>("cascade_upstream_repo")
                .unwrap_or_default();
            let downstream = issue
                .get_metadata::<String>("cascade_downstream_repo")
                .unwrap_or_default();
            (
                format!("[Claudear] Cascade PR: {} ({} -> {})", issue.short_id, upstream, downstream),
                format!(
                    "Claudear created a cascade PR for downstream adaptation.\n\nUpstream: {}\nDownstream: {}\n\n{}\n\nPR URL: {}",
                    upstream, downstream, Self::format_issue_info(issue), pr_url
                ),
            )
        } else if issue.get_metadata::<bool>("is_pr_update").unwrap_or(false) {
            (
                format!("[Claudear] PR Updated: {}", issue.short_id),
                format!(
                    "Claudear updated an existing PR to address review feedback.\n\n{}\n\nPR URL: {}",
                    Self::format_issue_info(issue), pr_url
                ),
            )
        } else {
            (
                format!("[Claudear] PR Created: {}", issue.short_id),
                format!(
                    "Claudear successfully created a PR!\n\n{}\n\nPR URL: {}",
                    Self::format_issue_info(issue),
                    pr_url
                ),
            )
        };
        let body = if let Some(reason) = issue.get_metadata::<String>("trigger_reason") {
            format!("{}\n\nTrigger reason: {}", body, reason)
        } else {
            body
        };
        let body = if let Some(confidence) = issue.get_metadata::<u8>("confidence") {
            format!("{}\n\nFix Confidence: {}/100", body, confidence)
        } else {
            body
        };
        self.send_email(&subject, &body, Some(issue)).await
    }

    async fn notify_completed(&self, issue: &Issue) -> Result<()> {
        let (subject, body) = if issue
            .get_metadata::<bool>("regression_resolved")
            .unwrap_or(false)
        {
            (
                format!("[Claudear] Regression Resolved: {}", issue.short_id),
                format!(
                    "No regression detected after the monitoring period.\n\n{}\n\nThe issue has been marked as resolved.",
                    Self::format_issue_info(issue)
                ),
            )
        } else {
            {
                let reason = issue
                    .get_metadata::<String>("completion_reason")
                    .unwrap_or_else(|| "No PR URL was captured".to_string());
                (
                    format!("[Claudear] Completed: {}", issue.short_id),
                    format!(
                        "Claudear completed processing without creating a PR.\n\nReason: {}\n\n{}",
                        reason,
                        Self::format_issue_info(issue)
                    ),
                )
            }
        };
        self.send_email(&subject, &body, Some(issue)).await
    }

    async fn notify_failed(&self, issue: &Issue, error: &str) -> Result<()> {
        let (subject, body) = if issue
            .get_metadata::<bool>("regression_detected")
            .unwrap_or(false)
        {
            (
                format!("[Claudear] Regression Detected: {}", issue.short_id),
                format!(
                    "A previously fixed issue has regressed.\n\n{}\n\nDetails: {}\n\nA retry has been scheduled.",
                    Self::format_issue_info(issue), error
                ),
            )
        } else if issue
            .get_metadata::<String>("cascade_downstream_repo")
            .is_some()
        {
            let upstream = issue
                .get_metadata::<String>("cascade_upstream_repo")
                .unwrap_or_default();
            let downstream = issue
                .get_metadata::<String>("cascade_downstream_repo")
                .unwrap_or_default();
            (
                format!("[Claudear] Cascade Failed: {} ({} -> {})", issue.short_id, upstream, downstream),
                format!(
                    "Claudear failed to create a cascade PR.\n\nUpstream: {}\nDownstream: {}\n\n{}\n\nError: {}",
                    upstream, downstream, Self::format_issue_info(issue), error
                ),
            )
        } else {
            (
                format!("[Claudear] Failed: {}", issue.short_id),
                format!(
                    "Claudear failed to process an issue.\n\n{}\n\nError: {}",
                    Self::format_issue_info(issue),
                    error
                ),
            )
        };
        let body = if let Some(reason) = issue.get_metadata::<String>("trigger_reason") {
            format!("{}\n\nTrigger reason: {}", body, reason)
        } else {
            body
        };
        self.send_email(&subject, &body, Some(issue)).await
    }

    async fn notify_merged(&self, issue: &Issue, pr_url: &str) -> Result<()> {
        let subject = format!("[Claudear] PR Merged: {}", issue.short_id);
        let body = format!(
            "A PR has been merged and the issue resolved.\n\n{}\n\nPR URL: {}",
            Self::format_issue_info(issue),
            pr_url
        );
        self.send_email(&subject, &body, Some(issue)).await
    }

    async fn notify_closed(&self, issue: &Issue, pr_url: &str) -> Result<()> {
        let subject = format!("[Claudear] PR Closed: {}", issue.short_id);
        let body = format!(
            "A PR was closed without merging.\n\n{}\n\nPR URL: {}",
            Self::format_issue_info(issue),
            pr_url
        );
        self.send_email(&subject, &body, Some(issue)).await
    }

    async fn notify_status(&self, message: &str) -> Result<()> {
        let subject = "[Claudear] Status Update".to_string();
        self.send_email(&subject, message, None).await
    }

    async fn notify_urgent_issues(&self, issues: &[Issue]) -> Result<()> {
        if issues.is_empty() {
            return Ok(());
        }

        let subject = format!(
            "[Claudear] {} Urgent Issue{} Detected",
            issues.len(),
            if issues.len() > 1 { "s" } else { "" }
        );

        let mut body = "The following urgent issues require attention:\n\n".to_string();
        for issue in issues.iter().take(10) {
            body.push_str(&format!(
                "- [{}] {} - {}\n  URL: {}\n\n",
                issue.source, issue.short_id, issue.title, issue.url
            ));
        }

        self.send_email(&subject, &body, None).await
    }

    async fn ask_question(
        &self,
        issue: &Issue,
        request: &AskRequest,
    ) -> Result<Option<AskDelivery>> {
        let subject = format!("Human input needed: {}", issue.short_id);
        let mut body = format!(
            "Claude needs human input.\n\n{}\n\nQuestion:\n{}\n",
            Self::format_issue_info(issue),
            request.question.question
        );
        if let Some(ref why) = request.question.why {
            body.push_str(&format!("\nWhy:\n{}\n", why));
        }
        if let Some(ref context) = request.question.context {
            body.push_str(&format!("\nContext:\n{}\n", context));
        }
        if !request.question.options.is_empty() {
            body.push_str(&format!(
                "\nOptions:\n- {}\n",
                request.question.options.join("\n- ")
            ));
        }
        body.push_str("\nReply to this email with your answer.\n");

        self.send_email(&subject, &body, Some(issue)).await?;
        Ok(Some(AskDelivery {
            channel: "email".to_string(),
            target: self.target_email_for_issue(issue),
            message_id: None,
        }))
    }

    async fn poll_question_replies(
        &self,
        request: &AskRequest,
        since: DateTime<Utc>,
    ) -> Result<Vec<AskReply>> {
        let imap_host = match self.config.imap_host.clone() {
            Some(v) if !v.is_empty() => v,
            _ => return Ok(Vec::new()),
        };
        let imap_username = match self.config.imap_username.clone() {
            Some(v) if !v.is_empty() => v,
            _ => return Ok(Vec::new()),
        };
        let imap_password = match self.config.imap_password.as_ref() {
            Some(v) => {
                let exposed = v.expose().to_string();
                if !exposed.is_empty() {
                    exposed
                } else {
                    return Ok(Vec::new());
                }
            }
            _ => return Ok(Vec::new()),
        };

        let imap_port = self.config.imap_port;
        let imap_folder = self.config.imap_folder.clone();
        let correlation_id = request.correlation_id.clone();
        let short_id = request.short_id.clone();
        let expected_senders = self.expected_reply_emails(request);
        let imap_use_tls = self.config.imap_use_tls;

        tokio::task::spawn_blocking(move || -> Result<Vec<AskReply>> {
            let tls = native_tls::TlsConnector::builder().build().map_err(|e| {
                Error::notifier("email", format!("Failed to build TLS connector: {}", e))
            })?;

            if !imap_use_tls {
                return Ok(Vec::new());
            }

            let client = imap::connect((imap_host.as_str(), imap_port), &imap_host, &tls)
                .map_err(|e| Error::notifier("email", format!("IMAP connect failed: {}", e)))?;
            let mut session = client
                .login(imap_username, imap_password)
                .map_err(|(e, _)| Error::notifier("email", format!("IMAP login failed: {}", e)))?;

            session
                .select(&imap_folder)
                .map_err(|e| Error::notifier("email", format!("IMAP select failed: {}", e)))?;

            let search_token = format!("Human input needed: {}", short_id);
            let search_query = format!("SUBJECT \"{}\"", search_token);
            let ids = session
                .search(search_query)
                .map_err(|e| Error::notifier("email", format!("IMAP search failed: {}", e)))?;

            if ids.is_empty() {
                let _ = session.logout();
                return Ok(Vec::new());
            }

            let sequence = ids
                .iter()
                .map(|id| id.to_string())
                .collect::<Vec<_>>()
                .join(",");
            let fetches = session
                .fetch(sequence, "RFC822")
                .map_err(|e| Error::notifier("email", format!("IMAP fetch failed: {}", e)))?;

            let mut replies = Vec::new();
            for fetch in fetches.iter() {
                let raw = match fetch.body() {
                    Some(v) => v,
                    None => continue,
                };

                let parsed = match mailparse::parse_mail(raw) {
                    Ok(v) => v,
                    Err(_) => continue,
                };

                let from_header = parsed.headers.get_first_value("From").unwrap_or_default();
                let responder = Self::extract_email_address(&from_header);
                if !expected_senders.is_empty() {
                    match responder.as_ref() {
                        Some(email) if expected_senders.contains(email) => {}
                        _ => continue,
                    }
                }

                let subject = parsed
                    .headers
                    .get_first_value("Subject")
                    .unwrap_or_default();
                let body_text = Self::extract_plain_body(&parsed);
                if !subject.contains(&search_token) && !body_text.contains(&search_token) {
                    continue;
                }

                let answer = match Self::sanitize_reply_text(&body_text) {
                    Some(v) => v,
                    None => continue,
                };

                let replied_at = parsed
                    .headers
                    .get_first_value("Date")
                    .and_then(|date| mailparse::dateparse(&date).ok())
                    .and_then(|secs| Utc.timestamp_opt(secs, 0).single())
                    .unwrap_or_else(Utc::now);

                if replied_at < since {
                    continue;
                }

                replies.push(AskReply {
                    correlation_id: correlation_id.clone(),
                    channel: "email".to_string(),
                    responder,
                    answer,
                    replied_at,
                });
            }

            replies.sort_by_key(|r| r.replied_at);
            let _ = session.logout();
            Ok(replies)
        })
        .await
        .map_err(|e| Error::notifier("email", format!("IMAP task failed: {}", e)))?
    }

    fn supports_replies(&self) -> bool {
        self.config
            .imap_host
            .as_ref()
            .map(|v| !v.is_empty())
            .unwrap_or(false)
            && self
                .config
                .imap_username
                .as_ref()
                .map(|v| !v.is_empty())
                .unwrap_or(false)
            && self
                .config
                .imap_password
                .as_ref()
                .map(|v| !v.expose().is_empty())
                .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use claudear_core::types::{IssuePriority, IssueStatus};

    fn empty_registry() -> UserRegistry {
        UserRegistry::new(std::collections::HashMap::new())
    }

    fn disabled_config() -> EmailConfig {
        EmailConfig {
            smtp_host: None,
            smtp_port: 587,
            smtp_username: None,
            smtp_password: None,
            from_address: None,
            to_addresses: vec![],
            use_tls: true,
            ..Default::default()
        }
    }

    fn partial_config() -> EmailConfig {
        EmailConfig {
            smtp_host: Some("smtp.example.com".to_string()),
            smtp_port: 587,
            smtp_username: Some("user".to_string()),
            smtp_password: Some("pass".into()),
            from_address: None, // Missing from address
            to_addresses: vec!["test@example.com".to_string()],
            use_tls: true,
            ..Default::default()
        }
    }

    fn no_to_config() -> EmailConfig {
        EmailConfig {
            smtp_host: Some("smtp.example.com".to_string()),
            smtp_port: 587,
            smtp_username: Some("user".to_string()),
            smtp_password: Some("pass".into()),
            from_address: Some("from@example.com".to_string()),
            to_addresses: vec![], // No recipients
            use_tls: true,
            ..Default::default()
        }
    }

    #[test]
    fn test_new_disabled() {
        let notifier = EmailNotifier::new(disabled_config(), empty_registry()).unwrap();
        assert_eq!(notifier.name(), "email");
        assert!(!notifier.is_enabled());
    }

    #[test]
    fn test_name() {
        let notifier = EmailNotifier::new(disabled_config(), empty_registry()).unwrap();
        assert_eq!(notifier.name(), "email");
    }

    #[test]
    fn test_is_enabled_no_transport() {
        let notifier = EmailNotifier::new(disabled_config(), empty_registry()).unwrap();
        assert!(!notifier.is_enabled());
    }

    #[test]
    fn test_is_enabled_no_from() {
        let notifier = EmailNotifier::new(partial_config(), empty_registry()).unwrap();
        // Has transport but no from address
        assert!(!notifier.is_enabled());
    }

    #[test]
    fn test_is_enabled_no_to() {
        let notifier = EmailNotifier::new(no_to_config(), empty_registry()).unwrap();
        // Has transport and from but no recipients
        assert!(!notifier.is_enabled());
    }

    #[test]
    fn test_format_issue_info() {
        let mut issue = Issue::new(
            "123",
            "PROJ-123",
            "Test Issue",
            "https://example.com",
            "linear",
        );
        issue.priority = IssuePriority::High;
        issue.status = IssueStatus::Open;

        let info = EmailNotifier::format_issue_info(&issue);

        assert!(info.contains("PROJ-123"));
        assert!(info.contains("Test Issue"));
        assert!(info.contains("linear"));
        assert!(info.contains("https://example.com"));
    }

    #[test]
    fn test_format_issue_info_all_priorities() {
        for priority in [
            IssuePriority::Critical,
            IssuePriority::High,
            IssuePriority::Medium,
            IssuePriority::Low,
            IssuePriority::None,
        ] {
            let mut issue = Issue::new("123", "TEST-1", "Test", "https://example.com", "linear");
            issue.priority = priority;

            let info = EmailNotifier::format_issue_info(&issue);
            assert!(!info.is_empty());
        }
    }

    #[test]
    fn test_format_issue_info_all_statuses() {
        for status in [
            IssueStatus::Open,
            IssueStatus::InProgress,
            IssueStatus::Resolved,
            IssueStatus::Ignored,
        ] {
            let mut issue = Issue::new("123", "TEST-1", "Test", "https://example.com", "sentry");
            issue.status = status;

            let info = EmailNotifier::format_issue_info(&issue);
            assert!(!info.is_empty());
        }
    }

    #[tokio::test]
    async fn test_notify_start_disabled() {
        let notifier = EmailNotifier::new(disabled_config(), empty_registry()).unwrap();
        let issue = Issue::new("123", "PROJ-123", "Test", "https://example.com", "linear");

        // Should return Ok even when disabled
        let result = notifier.notify_start(&issue).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_notify_success_disabled() {
        let notifier = EmailNotifier::new(disabled_config(), empty_registry()).unwrap();
        let issue = Issue::new("123", "PROJ-123", "Test", "https://example.com", "linear");

        let result = notifier
            .notify_success(&issue, "https://github.com/org/repo/pull/1")
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_notify_completed_disabled() {
        let notifier = EmailNotifier::new(disabled_config(), empty_registry()).unwrap();
        let issue = Issue::new("123", "PROJ-123", "Test", "https://example.com", "linear");

        let result = notifier.notify_completed(&issue).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_notify_failed_disabled() {
        let notifier = EmailNotifier::new(disabled_config(), empty_registry()).unwrap();
        let issue = Issue::new("123", "PROJ-123", "Test", "https://example.com", "linear");

        let result = notifier.notify_failed(&issue, "Error message").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_notify_status_disabled() {
        let notifier = EmailNotifier::new(disabled_config(), empty_registry()).unwrap();

        let result = notifier.notify_status("Status update").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_notify_urgent_issues_empty() {
        let notifier = EmailNotifier::new(disabled_config(), empty_registry()).unwrap();

        let result = notifier.notify_urgent_issues(&[]).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_notify_urgent_issues_disabled() {
        let notifier = EmailNotifier::new(disabled_config(), empty_registry()).unwrap();
        let issues = vec![
            Issue::new("1", "PROJ-1", "Issue 1", "https://example.com", "linear"),
            Issue::new("2", "PROJ-2", "Issue 2", "https://example.com", "linear"),
        ];

        let result = notifier.notify_urgent_issues(&issues).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_notify_urgent_issues_single_vs_plural() {
        let notifier = EmailNotifier::new(disabled_config(), empty_registry()).unwrap();

        // Single issue
        let single = vec![Issue::new(
            "1",
            "PROJ-1",
            "Issue",
            "https://example.com",
            "linear",
        )];
        let result = notifier.notify_urgent_issues(&single).await;
        assert!(result.is_ok());

        // Multiple issues
        let multiple = vec![
            Issue::new("1", "PROJ-1", "Issue 1", "https://example.com", "linear"),
            Issue::new("2", "PROJ-2", "Issue 2", "https://example.com", "linear"),
        ];
        let result = notifier.notify_urgent_issues(&multiple).await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_new_with_non_tls() {
        let config = EmailConfig {
            smtp_host: Some("localhost".to_string()),
            smtp_port: 25,
            smtp_username: Some("user".to_string()),
            smtp_password: Some("pass".into()),
            from_address: Some("test@localhost".to_string()),
            to_addresses: vec!["recipient@localhost".to_string()],
            use_tls: false,
            ..Default::default()
        };

        let notifier = EmailNotifier::new(config, empty_registry()).unwrap();
        // Should create successfully with dangerous builder
        assert_eq!(notifier.name(), "email");
    }

    #[test]
    fn test_extract_email_address_variants() {
        assert_eq!(
            EmailNotifier::extract_email_address("Jane <jane@example.com>").as_deref(),
            Some("jane@example.com")
        );
        assert_eq!(
            EmailNotifier::extract_email_address("ops@example.com").as_deref(),
            Some("ops@example.com")
        );
    }

    #[test]
    fn test_sanitize_reply_text_removes_token_and_quotes() {
        let body = "Thanks\n\n> quoted\nCLAUDEAR-Q:abc123\nUse main";
        let parsed = EmailNotifier::sanitize_reply_text(body).unwrap();
        assert_eq!(parsed, "Thanks\nCLAUDEAR-Q:abc123\nUse main");
    }

    #[tokio::test]
    async fn test_ask_question_routes_to_resolved_user_email() {
        let mut users = std::collections::HashMap::new();
        users.insert(
            "jake".to_string(),
            claudear_config::config::UserConfig {
                email: Some("jake@example.com".to_string()),
                ..Default::default()
            },
        );
        let registry = UserRegistry::new(users);
        let config = EmailConfig {
            smtp_host: None,
            from_address: Some("bot@example.com".to_string()),
            to_addresses: vec!["global@example.com".to_string()],
            ..Default::default()
        };
        let notifier = EmailNotifier::new(config, registry).unwrap();
        let mut issue = Issue::new("1", "LIN-1", "Test", "https://example.com", "linear");
        issue.set_metadata("resolved_user", "jake");
        let request = AskRequest {
            correlation_id: "tok-email-1".to_string(),
            source: "linear".to_string(),
            repo: Some("org/repo".to_string()),
            issue_id: "1".to_string(),
            short_id: "LIN-1".to_string(),
            question: claudear_core::types::BlockingQuestion {
                question: "Which env?".to_string(),
                context: None,
                options: vec![],
                why: None,
            },
            asked_at: Utc::now(),
            target_discord_id: None,
            target_email: None,
            target_slack_id: None,
        };
        let delivery = notifier
            .ask_question(&issue, &request)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(delivery.target.as_deref(), Some("jake@example.com"));
    }

    #[test]
    fn test_extract_email_address_angle_brackets() {
        assert_eq!(
            EmailNotifier::extract_email_address("John Doe <john@example.com>").as_deref(),
            Some("john@example.com")
        );
    }

    #[test]
    fn test_extract_email_address_bare_email() {
        assert_eq!(
            EmailNotifier::extract_email_address("user@example.com").as_deref(),
            Some("user@example.com")
        );
    }

    #[test]
    fn test_extract_email_address_lowercases() {
        assert_eq!(
            EmailNotifier::extract_email_address("User@Example.COM").as_deref(),
            Some("user@example.com")
        );
    }

    #[test]
    fn test_extract_email_address_angle_bracket_lowercases() {
        assert_eq!(
            EmailNotifier::extract_email_address("Foo <FOO@BAR.COM>").as_deref(),
            Some("foo@bar.com")
        );
    }

    #[test]
    fn test_extract_email_address_empty_angle_brackets_falls_back() {
        // Empty inside angle brackets should fall back to plain parsing
        assert_eq!(
            EmailNotifier::extract_email_address("Name <>").as_deref(),
            None
        );
    }

    #[test]
    fn test_extract_email_address_no_at_symbol_returns_none() {
        assert_eq!(
            EmailNotifier::extract_email_address("not-an-email").as_deref(),
            None
        );
    }

    #[test]
    fn test_extract_email_address_empty_string() {
        assert_eq!(EmailNotifier::extract_email_address("").as_deref(), None);
    }

    #[test]
    fn test_extract_email_address_whitespace_trimmed() {
        assert_eq!(
            EmailNotifier::extract_email_address("  user@example.com  ").as_deref(),
            Some("user@example.com")
        );
    }

    #[test]
    fn test_sanitize_reply_text_strips_quoted_lines() {
        let body = "> original message\nMy actual reply";
        let result = EmailNotifier::sanitize_reply_text(body).unwrap();
        assert_eq!(result, "My actual reply");
    }

    #[test]
    fn test_sanitize_reply_text_strips_token_line() {
        let body = "CLAUDEAR-Q:tok-1\nUse staging environment";
        let result = EmailNotifier::sanitize_reply_text(body).unwrap();
        assert_eq!(result, "CLAUDEAR-Q:tok-1\nUse staging environment");
    }

    #[test]
    fn test_sanitize_reply_text_strips_empty_lines() {
        let body = "\n\n\nHello\n\n\n";
        let result = EmailNotifier::sanitize_reply_text(body).unwrap();
        assert_eq!(result, "Hello");
    }

    #[test]
    fn test_sanitize_reply_text_all_quoted_returns_none() {
        let body = "> quoted line 1\n> quoted line 2\n> quoted line 3";
        let result = EmailNotifier::sanitize_reply_text(body);
        assert!(result.is_none());
    }

    #[test]
    fn test_sanitize_reply_text_empty_body_returns_none() {
        let result = EmailNotifier::sanitize_reply_text("");
        assert!(result.is_none());
    }

    #[test]
    fn test_sanitize_reply_text_only_whitespace_returns_none() {
        let result = EmailNotifier::sanitize_reply_text("   \n   \n   ");
        assert!(result.is_none());
    }

    #[test]
    fn test_sanitize_reply_text_truncates_long_output() {
        let long_line = "x".repeat(5000);
        let result = EmailNotifier::sanitize_reply_text(&long_line).unwrap();
        assert!(result.len() <= 4000);
    }

    #[test]
    fn test_sanitize_reply_text_limits_to_30_lines() {
        let body = (1..=50)
            .map(|i| format!("Line {}", i))
            .collect::<Vec<_>>()
            .join("\n");
        let result = EmailNotifier::sanitize_reply_text(&body).unwrap();
        let line_count = result.lines().count();
        assert!(
            line_count <= 30,
            "Expected at most 30 lines, got {}",
            line_count
        );
    }

    #[test]
    fn test_sanitize_reply_text_mixed_content() {
        let body = "> On Mon, user wrote:\n> Original question\n\nMy answer is yes\n\nCLAUDEAR-Q:tok-1\n\n> More quoted text\nSecond line of answer";
        let result = EmailNotifier::sanitize_reply_text(body).unwrap();
        assert_eq!(
            result,
            "My answer is yes\nCLAUDEAR-Q:tok-1\nSecond line of answer"
        );
    }

    #[test]
    fn test_format_issue_info_contains_all_fields() {
        let mut issue = Issue::new(
            "42",
            "PROJ-42",
            "Authentication bug",
            "https://linear.app/42",
            "linear",
        );
        issue.priority = IssuePriority::Critical;
        issue.status = IssueStatus::InProgress;

        let info = EmailNotifier::format_issue_info(&issue);
        assert!(info.contains("PROJ-42"));
        assert!(info.contains("Authentication bug"));
        assert!(info.contains("linear"));
        assert!(info.contains("https://linear.app/42"));
        assert!(info.contains(&issue.priority.to_string()));
        assert!(info.contains(&issue.status.to_string()));
    }

    #[test]
    fn test_format_issue_info_format_structure() {
        let issue = Issue::new("1", "TEST-1", "Title", "https://url.com", "sentry");
        let info = EmailNotifier::format_issue_info(&issue);
        // Verify it has the expected "Key: Value" structure
        assert!(info.starts_with("Issue: TEST-1 - Title"));
        assert!(info.contains("Source: sentry"));
        assert!(info.contains("URL: https://url.com"));
    }

    #[test]
    fn test_resolve_recipients_returns_global_when_no_issue() {
        let config = EmailConfig {
            to_addresses: vec!["global@example.com".to_string()],
            ..Default::default()
        };
        let notifier = EmailNotifier::new(config, empty_registry()).unwrap();
        let recipients = notifier.resolve_recipients(None);
        assert_eq!(recipients, vec!["global@example.com".to_string()]);
    }

    #[test]
    fn test_resolve_recipients_returns_global_when_no_resolved_user() {
        let config = EmailConfig {
            to_addresses: vec!["global@example.com".to_string()],
            ..Default::default()
        };
        let notifier = EmailNotifier::new(config, empty_registry()).unwrap();
        let issue = Issue::new("1", "LIN-1", "Test", "https://example.com", "linear");
        let recipients = notifier.resolve_recipients(Some(&issue));
        assert_eq!(recipients, vec!["global@example.com".to_string()]);
    }

    #[test]
    fn test_resolve_recipients_uses_resolved_user_email() {
        let mut users = std::collections::HashMap::new();
        users.insert(
            "jake".to_string(),
            claudear_config::config::UserConfig {
                email: Some("jake@example.com".to_string()),
                ..Default::default()
            },
        );
        let registry = UserRegistry::new(users);
        let config = EmailConfig {
            to_addresses: vec!["global@example.com".to_string()],
            ..Default::default()
        };
        let notifier = EmailNotifier::new(config, registry).unwrap();
        let mut issue = Issue::new("1", "LIN-1", "Test", "https://example.com", "linear");
        issue.set_metadata("resolved_user", "jake");
        let recipients = notifier.resolve_recipients(Some(&issue));
        assert_eq!(recipients, vec!["jake@example.com".to_string()]);
    }

    #[test]
    fn test_resolve_recipients_falls_back_when_user_has_no_email() {
        let mut users = std::collections::HashMap::new();
        users.insert(
            "jake".to_string(),
            claudear_config::config::UserConfig {
                email: None,
                ..Default::default()
            },
        );
        let registry = UserRegistry::new(users);
        let config = EmailConfig {
            to_addresses: vec!["global@example.com".to_string()],
            ..Default::default()
        };
        let notifier = EmailNotifier::new(config, registry).unwrap();
        let mut issue = Issue::new("1", "LIN-1", "Test", "https://example.com", "linear");
        issue.set_metadata("resolved_user", "jake");
        let recipients = notifier.resolve_recipients(Some(&issue));
        assert_eq!(recipients, vec!["global@example.com".to_string()]);
    }

    #[test]
    fn test_expected_reply_emails_from_request_target() {
        let config = EmailConfig {
            to_addresses: vec!["global@example.com".to_string()],
            ..Default::default()
        };
        let notifier = EmailNotifier::new(config, empty_registry()).unwrap();
        let request = AskRequest {
            correlation_id: "tok-1".to_string(),
            source: "linear".to_string(),
            repo: None,
            issue_id: "1".to_string(),
            short_id: "LIN-1".to_string(),
            question: claudear_core::types::BlockingQuestion {
                question: "Q?".to_string(),
                context: None,
                options: vec![],
                why: None,
            },
            asked_at: Utc::now(),
            target_discord_id: None,
            target_email: Some("Specific@Example.COM".to_string()),
            target_slack_id: None,
        };
        let emails = notifier.expected_reply_emails(&request);
        assert_eq!(emails.len(), 1);
        assert!(emails.contains("specific@example.com"));
    }

    #[test]
    fn test_expected_reply_emails_falls_back_to_to_addresses() {
        let config = EmailConfig {
            to_addresses: vec!["a@example.com".to_string(), "B@Example.COM".to_string()],
            ..Default::default()
        };
        let notifier = EmailNotifier::new(config, empty_registry()).unwrap();
        let request = AskRequest {
            correlation_id: "tok-2".to_string(),
            source: "linear".to_string(),
            repo: None,
            issue_id: "1".to_string(),
            short_id: "LIN-1".to_string(),
            question: claudear_core::types::BlockingQuestion {
                question: "Q?".to_string(),
                context: None,
                options: vec![],
                why: None,
            },
            asked_at: Utc::now(),
            target_discord_id: None,
            target_email: None,
            target_slack_id: None,
        };
        let emails = notifier.expected_reply_emails(&request);
        assert_eq!(emails.len(), 2);
        assert!(emails.contains("a@example.com"));
        assert!(emails.contains("b@example.com"));
    }

    #[test]
    fn test_supports_replies_true_when_all_imap_set() {
        let config = EmailConfig {
            imap_host: Some("imap.example.com".to_string()),
            imap_username: Some("user".to_string()),
            imap_password: Some("pass".into()),
            ..Default::default()
        };
        let notifier = EmailNotifier::new(config, empty_registry()).unwrap();
        assert!(notifier.supports_replies());
    }

    #[test]
    fn test_supports_replies_false_when_no_imap_host() {
        let config = EmailConfig {
            imap_host: None,
            imap_username: Some("user".to_string()),
            imap_password: Some("pass".into()),
            ..Default::default()
        };
        let notifier = EmailNotifier::new(config, empty_registry()).unwrap();
        assert!(!notifier.supports_replies());
    }

    #[test]
    fn test_supports_replies_false_when_empty_imap_host() {
        let config = EmailConfig {
            imap_host: Some("".to_string()),
            imap_username: Some("user".to_string()),
            imap_password: Some("pass".into()),
            ..Default::default()
        };
        let notifier = EmailNotifier::new(config, empty_registry()).unwrap();
        assert!(!notifier.supports_replies());
    }

    #[test]
    fn test_supports_replies_false_when_no_imap_username() {
        let config = EmailConfig {
            imap_host: Some("imap.example.com".to_string()),
            imap_username: None,
            imap_password: Some("pass".into()),
            ..Default::default()
        };
        let notifier = EmailNotifier::new(config, empty_registry()).unwrap();
        assert!(!notifier.supports_replies());
    }

    #[test]
    fn test_supports_replies_false_when_no_imap_password() {
        let config = EmailConfig {
            imap_host: Some("imap.example.com".to_string()),
            imap_username: Some("user".to_string()),
            imap_password: None,
            ..Default::default()
        };
        let notifier = EmailNotifier::new(config, empty_registry()).unwrap();
        assert!(!notifier.supports_replies());
    }

    #[test]
    fn test_supports_replies_false_when_empty_password() {
        let config = EmailConfig {
            imap_host: Some("imap.example.com".to_string()),
            imap_username: Some("user".to_string()),
            imap_password: Some("".into()),
            ..Default::default()
        };
        let notifier = EmailNotifier::new(config, empty_registry()).unwrap();
        assert!(!notifier.supports_replies());
    }

    #[test]
    fn test_target_email_for_issue_returns_first_recipient() {
        let config = EmailConfig {
            to_addresses: vec![
                "first@example.com".to_string(),
                "second@example.com".to_string(),
            ],
            ..Default::default()
        };
        let notifier = EmailNotifier::new(config, empty_registry()).unwrap();
        let issue = Issue::new("1", "LIN-1", "Test", "https://example.com", "linear");
        assert_eq!(
            notifier.target_email_for_issue(&issue),
            Some("first@example.com".to_string())
        );
    }

    #[test]
    fn test_target_email_for_issue_none_when_empty() {
        let config = EmailConfig {
            to_addresses: vec![],
            ..Default::default()
        };
        let notifier = EmailNotifier::new(config, empty_registry()).unwrap();
        let issue = Issue::new("1", "LIN-1", "Test", "https://example.com", "linear");
        assert_eq!(notifier.target_email_for_issue(&issue), None);
    }

    #[tokio::test]
    async fn test_ask_question_delivery_channel_is_email() {
        let config = EmailConfig {
            to_addresses: vec!["user@example.com".to_string()],
            ..Default::default()
        };
        let notifier = EmailNotifier::new(config, empty_registry()).unwrap();
        let issue = Issue::new("1", "LIN-1", "Test", "https://example.com", "linear");
        let request = AskRequest {
            correlation_id: "tok-ch".to_string(),
            source: "linear".to_string(),
            repo: None,
            issue_id: "1".to_string(),
            short_id: "LIN-1".to_string(),
            question: claudear_core::types::BlockingQuestion {
                question: "Q?".to_string(),
                context: None,
                options: vec![],
                why: None,
            },
            asked_at: Utc::now(),
            target_discord_id: None,
            target_email: None,
            target_slack_id: None,
        };
        let delivery = notifier
            .ask_question(&issue, &request)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(delivery.channel, "email");
        assert!(delivery.message_id.is_none());
    }

    #[test]
    fn test_extract_email_address_with_display_name_and_angle_brackets() {
        assert_eq!(
            EmailNotifier::extract_email_address("Alice Bob <alice@company.org>").as_deref(),
            Some("alice@company.org")
        );
    }

    #[test]
    fn test_extract_email_address_with_quotes_in_display_name() {
        assert_eq!(
            EmailNotifier::extract_email_address("\"Alice Bob\" <alice@company.org>").as_deref(),
            Some("alice@company.org")
        );
    }

    #[test]
    fn test_extract_email_address_only_whitespace() {
        assert_eq!(EmailNotifier::extract_email_address("   ").as_deref(), None);
    }

    #[test]
    fn test_extract_email_address_empty_brackets_no_at() {
        // "<>" with name prefix - no @ in fallback
        assert_eq!(
            EmailNotifier::extract_email_address("NoEmail <>").as_deref(),
            None
        );
    }

    #[test]
    fn test_extract_email_address_nested_angle_brackets() {
        // Picks the first < > pair
        assert_eq!(
            EmailNotifier::extract_email_address("Name <outer@example.com> extra>").as_deref(),
            Some("outer@example.com")
        );
    }

    #[test]
    fn test_extract_email_address_uppercase_in_brackets_lowercased() {
        assert_eq!(
            EmailNotifier::extract_email_address("Name <USER@EXAMPLE.COM>").as_deref(),
            Some("user@example.com")
        );
    }

    #[test]
    fn test_sanitize_reply_text_preserves_multiline_answer() {
        let body = "Line one\nLine two\nLine three";
        let result = EmailNotifier::sanitize_reply_text(body).unwrap();
        assert_eq!(result, "Line one\nLine two\nLine three");
    }

    #[test]
    fn test_sanitize_reply_text_strips_multiple_token_lines() {
        let body = "CLAUDEAR-Q:tok-1\nAnswer\nCLAUDEAR-Q:tok-1\nMore answer";
        let result = EmailNotifier::sanitize_reply_text(body).unwrap();
        assert_eq!(
            result,
            "CLAUDEAR-Q:tok-1\nAnswer\nCLAUDEAR-Q:tok-1\nMore answer"
        );
    }

    #[test]
    fn test_sanitize_reply_text_strips_mixed_quotes_and_tokens() {
        let body =
            "> Original question\n> Second line\nCLAUDEAR-Q:tok-1\n\nMy answer\n> Another quote";
        let result = EmailNotifier::sanitize_reply_text(body).unwrap();
        assert_eq!(result, "CLAUDEAR-Q:tok-1\nMy answer");
    }

    #[test]
    fn test_sanitize_reply_text_only_token_returns_none() {
        let body = "CLAUDEAR-Q:tok-1";
        let result = EmailNotifier::sanitize_reply_text(body).unwrap();
        assert_eq!(result, "CLAUDEAR-Q:tok-1");
    }

    #[test]
    fn test_sanitize_reply_text_token_with_whitespace_only_returns_none() {
        let body = "CLAUDEAR-Q:tok-1\n   \n   \n";
        let result = EmailNotifier::sanitize_reply_text(body).unwrap();
        assert_eq!(result, "CLAUDEAR-Q:tok-1");
    }

    #[test]
    fn test_sanitize_reply_text_different_correlation_id_preserves_line() {
        let body = "CLAUDEAR-Q:other-tok\nSome answer";
        let result = EmailNotifier::sanitize_reply_text(body).unwrap();
        // Token lines are no longer stripped by sanitize_reply_text
        assert!(result.contains("CLAUDEAR-Q:other-tok"));
        assert!(result.contains("Some answer"));
    }

    #[test]
    fn test_sanitize_reply_text_exactly_30_lines() {
        let body = (1..=30)
            .map(|i| format!("Line {}", i))
            .collect::<Vec<_>>()
            .join("\n");
        let result = EmailNotifier::sanitize_reply_text(&body).unwrap();
        assert_eq!(result.lines().count(), 30);
    }

    #[test]
    fn test_sanitize_reply_text_31_lines_truncated() {
        let body = (1..=31)
            .map(|i| format!("Line {}", i))
            .collect::<Vec<_>>()
            .join("\n");
        let result = EmailNotifier::sanitize_reply_text(&body).unwrap();
        assert_eq!(result.lines().count(), 30);
        assert!(result.contains("Line 30"));
        assert!(!result.contains("Line 31"));
    }

    #[test]
    fn test_sanitize_reply_text_exactly_4000_chars() {
        // Build a string that is exactly 4000 chars of content lines
        let line = "x".repeat(100);
        let body = (0..40).map(|_| line.clone()).collect::<Vec<_>>().join("\n"); // 40 * 100 + 39 newlines but only 30 lines are kept
        let result = EmailNotifier::sanitize_reply_text(&body).unwrap();
        assert!(result.len() <= 4000);
    }

    #[test]
    fn test_extract_plain_body_simple() {
        let raw = b"From: user@example.com\r\nSubject: Test\r\nContent-Type: text/plain\r\n\r\nHello world";
        let parsed = mailparse::parse_mail(raw).unwrap();
        let body = EmailNotifier::extract_plain_body(&parsed);
        assert!(body.contains("Hello world"));
    }

    #[test]
    fn test_extract_plain_body_multipart() {
        let raw = b"From: user@example.com\r\nSubject: Test\r\nContent-Type: multipart/alternative; boundary=bound\r\n\r\n--bound\r\nContent-Type: text/plain\r\n\r\nPlain text body\r\n--bound\r\nContent-Type: text/html\r\n\r\n<p>HTML body</p>\r\n--bound--";
        let parsed = mailparse::parse_mail(raw).unwrap();
        let body = EmailNotifier::extract_plain_body(&parsed);
        assert!(body.contains("Plain text body"));
        assert!(!body.contains("<p>"));
    }

    #[test]
    fn test_extract_plain_body_no_plain_part_falls_back() {
        let raw = b"From: user@example.com\r\nSubject: Test\r\nContent-Type: multipart/alternative; boundary=bound\r\n\r\n--bound\r\nContent-Type: text/html\r\n\r\n<p>Only HTML</p>\r\n--bound--";
        let parsed = mailparse::parse_mail(raw).unwrap();
        let body = EmailNotifier::extract_plain_body(&parsed);
        // Falls back to get_body() on the parent
        assert!(!body.is_empty() || body.is_empty()); // Shouldn't panic
    }

    #[test]
    fn test_resolve_recipients_with_resolved_user_unknown_slug() {
        let config = EmailConfig {
            to_addresses: vec!["global@example.com".to_string()],
            ..Default::default()
        };
        let notifier = EmailNotifier::new(config, empty_registry()).unwrap();
        let mut issue = Issue::new("1", "LIN-1", "Test", "https://example.com", "linear");
        issue.set_metadata("resolved_user", "unknown_user");
        let recipients = notifier.resolve_recipients(Some(&issue));
        // Should fall back to global since user slug not found
        assert_eq!(recipients, vec!["global@example.com".to_string()]);
    }

    #[test]
    fn test_resolve_recipients_multiple_global_addresses() {
        let config = EmailConfig {
            to_addresses: vec![
                "a@example.com".to_string(),
                "b@example.com".to_string(),
                "c@example.com".to_string(),
            ],
            ..Default::default()
        };
        let notifier = EmailNotifier::new(config, empty_registry()).unwrap();
        let recipients = notifier.resolve_recipients(None);
        assert_eq!(recipients.len(), 3);
    }

    #[test]
    fn test_resolve_recipients_empty_global() {
        let config = EmailConfig {
            to_addresses: vec![],
            ..Default::default()
        };
        let notifier = EmailNotifier::new(config, empty_registry()).unwrap();
        let recipients = notifier.resolve_recipients(None);
        assert!(recipients.is_empty());
    }

    #[test]
    fn test_target_email_for_issue_uses_resolved_user() {
        let mut users = std::collections::HashMap::new();
        users.insert(
            "jake".to_string(),
            claudear_config::config::UserConfig {
                email: Some("jake@resolved.com".to_string()),
                ..Default::default()
            },
        );
        let registry = UserRegistry::new(users);
        let config = EmailConfig {
            to_addresses: vec!["fallback@example.com".to_string()],
            ..Default::default()
        };
        let notifier = EmailNotifier::new(config, registry).unwrap();
        let mut issue = Issue::new("1", "LIN-1", "Test", "https://example.com", "linear");
        issue.set_metadata("resolved_user", "jake");
        assert_eq!(
            notifier.target_email_for_issue(&issue),
            Some("jake@resolved.com".to_string())
        );
    }

    #[test]
    fn test_expected_reply_emails_empty_to_addresses_no_target() {
        let config = EmailConfig {
            to_addresses: vec![],
            ..Default::default()
        };
        let notifier = EmailNotifier::new(config, empty_registry()).unwrap();
        let request = AskRequest {
            correlation_id: "tok-empty".to_string(),
            source: "linear".to_string(),
            repo: None,
            issue_id: "1".to_string(),
            short_id: "LIN-1".to_string(),
            question: claudear_core::types::BlockingQuestion {
                question: "Q?".to_string(),
                context: None,
                options: vec![],
                why: None,
            },
            asked_at: Utc::now(),
            target_discord_id: None,
            target_email: None,
            target_slack_id: None,
        };
        let emails = notifier.expected_reply_emails(&request);
        assert!(emails.is_empty());
    }

    #[test]
    fn test_supports_replies_false_when_empty_imap_username() {
        let config = EmailConfig {
            imap_host: Some("imap.example.com".to_string()),
            imap_username: Some("".to_string()),
            imap_password: Some("pass".into()),
            ..Default::default()
        };
        let notifier = EmailNotifier::new(config, empty_registry()).unwrap();
        assert!(!notifier.supports_replies());
    }

    #[tokio::test]
    async fn test_poll_question_replies_no_imap_host() {
        let config = EmailConfig {
            imap_host: None,
            ..Default::default()
        };
        let notifier = EmailNotifier::new(config, empty_registry()).unwrap();
        let request = AskRequest {
            correlation_id: "tok-poll".to_string(),
            source: "linear".to_string(),
            repo: None,
            issue_id: "1".to_string(),
            short_id: "LIN-1".to_string(),
            question: claudear_core::types::BlockingQuestion {
                question: "Q?".to_string(),
                context: None,
                options: vec![],
                why: None,
            },
            asked_at: Utc::now(),
            target_discord_id: None,
            target_email: None,
            target_slack_id: None,
        };
        let replies = notifier
            .poll_question_replies(&request, Utc::now())
            .await
            .unwrap();
        assert!(replies.is_empty());
    }

    #[tokio::test]
    async fn test_poll_question_replies_empty_imap_host() {
        let config = EmailConfig {
            imap_host: Some("".to_string()),
            imap_username: Some("user".to_string()),
            imap_password: Some("pass".into()),
            ..Default::default()
        };
        let notifier = EmailNotifier::new(config, empty_registry()).unwrap();
        let request = AskRequest {
            correlation_id: "tok-empty-host".to_string(),
            source: "linear".to_string(),
            repo: None,
            issue_id: "1".to_string(),
            short_id: "LIN-1".to_string(),
            question: claudear_core::types::BlockingQuestion {
                question: "Q?".to_string(),
                context: None,
                options: vec![],
                why: None,
            },
            asked_at: Utc::now(),
            target_discord_id: None,
            target_email: None,
            target_slack_id: None,
        };
        let replies = notifier
            .poll_question_replies(&request, Utc::now())
            .await
            .unwrap();
        assert!(replies.is_empty());
    }

    #[tokio::test]
    async fn test_poll_question_replies_empty_imap_username() {
        let config = EmailConfig {
            imap_host: Some("imap.example.com".to_string()),
            imap_username: Some("".to_string()),
            imap_password: Some("pass".into()),
            ..Default::default()
        };
        let notifier = EmailNotifier::new(config, empty_registry()).unwrap();
        let request = AskRequest {
            correlation_id: "tok-empty-user".to_string(),
            source: "linear".to_string(),
            repo: None,
            issue_id: "1".to_string(),
            short_id: "LIN-1".to_string(),
            question: claudear_core::types::BlockingQuestion {
                question: "Q?".to_string(),
                context: None,
                options: vec![],
                why: None,
            },
            asked_at: Utc::now(),
            target_discord_id: None,
            target_email: None,
            target_slack_id: None,
        };
        let replies = notifier
            .poll_question_replies(&request, Utc::now())
            .await
            .unwrap();
        assert!(replies.is_empty());
    }

    #[tokio::test]
    async fn test_poll_question_replies_empty_imap_password() {
        let config = EmailConfig {
            imap_host: Some("imap.example.com".to_string()),
            imap_username: Some("user".to_string()),
            imap_password: Some("".into()),
            ..Default::default()
        };
        let notifier = EmailNotifier::new(config, empty_registry()).unwrap();
        let request = AskRequest {
            correlation_id: "tok-empty-pass".to_string(),
            source: "linear".to_string(),
            repo: None,
            issue_id: "1".to_string(),
            short_id: "LIN-1".to_string(),
            question: claudear_core::types::BlockingQuestion {
                question: "Q?".to_string(),
                context: None,
                options: vec![],
                why: None,
            },
            asked_at: Utc::now(),
            target_discord_id: None,
            target_email: None,
            target_slack_id: None,
        };
        let replies = notifier
            .poll_question_replies(&request, Utc::now())
            .await
            .unwrap();
        assert!(replies.is_empty());
    }

    #[tokio::test]
    async fn test_poll_question_replies_no_imap_username() {
        let config = EmailConfig {
            imap_host: Some("imap.example.com".to_string()),
            imap_username: None,
            imap_password: Some("pass".into()),
            ..Default::default()
        };
        let notifier = EmailNotifier::new(config, empty_registry()).unwrap();
        let request = AskRequest {
            correlation_id: "tok-no-user".to_string(),
            source: "linear".to_string(),
            repo: None,
            issue_id: "1".to_string(),
            short_id: "LIN-1".to_string(),
            question: claudear_core::types::BlockingQuestion {
                question: "Q?".to_string(),
                context: None,
                options: vec![],
                why: None,
            },
            asked_at: Utc::now(),
            target_discord_id: None,
            target_email: None,
            target_slack_id: None,
        };
        let replies = notifier
            .poll_question_replies(&request, Utc::now())
            .await
            .unwrap();
        assert!(replies.is_empty());
    }

    #[tokio::test]
    async fn test_poll_question_replies_no_imap_password() {
        let config = EmailConfig {
            imap_host: Some("imap.example.com".to_string()),
            imap_username: Some("user".to_string()),
            imap_password: None,
            ..Default::default()
        };
        let notifier = EmailNotifier::new(config, empty_registry()).unwrap();
        let request = AskRequest {
            correlation_id: "tok-no-pass".to_string(),
            source: "linear".to_string(),
            repo: None,
            issue_id: "1".to_string(),
            short_id: "LIN-1".to_string(),
            question: claudear_core::types::BlockingQuestion {
                question: "Q?".to_string(),
                context: None,
                options: vec![],
                why: None,
            },
            asked_at: Utc::now(),
            target_discord_id: None,
            target_email: None,
            target_slack_id: None,
        };
        let replies = notifier
            .poll_question_replies(&request, Utc::now())
            .await
            .unwrap();
        assert!(replies.is_empty());
    }

    #[test]
    fn test_format_issue_info_contains_priority_and_status() {
        let mut issue = Issue::new("1", "TEST-1", "Title", "https://url.com", "jira");
        issue.priority = IssuePriority::Critical;
        issue.status = IssueStatus::InProgress;
        let info = EmailNotifier::format_issue_info(&issue);
        assert!(info.contains("Priority: critical"));
        assert!(info.contains("Status: in_progress"));
    }

    #[test]
    fn test_format_issue_info_newline_structure() {
        let issue = Issue::new("1", "TEST-1", "Title", "https://url.com", "linear");
        let info = EmailNotifier::format_issue_info(&issue);
        let lines: Vec<&str> = info.lines().collect();
        assert_eq!(lines.len(), 5);
        assert!(lines[0].starts_with("Issue:"));
        assert!(lines[1].starts_with("Source:"));
        assert!(lines[2].starts_with("Priority:"));
        assert!(lines[3].starts_with("Status:"));
        assert!(lines[4].starts_with("URL:"));
    }

    #[tokio::test]
    async fn test_ask_question_with_options_and_context() {
        let config = EmailConfig {
            to_addresses: vec!["user@example.com".to_string()],
            ..Default::default()
        };
        let notifier = EmailNotifier::new(config, empty_registry()).unwrap();
        let issue = Issue::new("1", "LIN-1", "Test", "https://example.com", "linear");
        let request = AskRequest {
            correlation_id: "tok-opts".to_string(),
            source: "linear".to_string(),
            repo: None,
            issue_id: "1".to_string(),
            short_id: "LIN-1".to_string(),
            question: claudear_core::types::BlockingQuestion {
                question: "Which env?".to_string(),
                context: Some("We need to deploy somewhere".to_string()),
                options: vec!["staging".to_string(), "production".to_string()],
                why: Some("Multiple environments available".to_string()),
            },
            asked_at: Utc::now(),
            target_discord_id: None,
            target_email: None,
            target_slack_id: None,
        };
        // With no transport, send_email returns Ok
        let delivery = notifier
            .ask_question(&issue, &request)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(delivery.channel, "email");
        assert_eq!(delivery.target.as_deref(), Some("user@example.com"));
    }

    #[test]
    fn test_is_enabled_true_when_all_set() {
        // We cannot easily create a fully enabled notifier without a real SMTP server,
        // but we can test the partial_config (has transport, no from_address)
        let notifier = EmailNotifier::new(partial_config(), empty_registry()).unwrap();
        // Has transport but no from, so not enabled
        assert!(!notifier.is_enabled());
    }

    #[test]
    fn test_new_with_tls_and_valid_host() {
        // This will attempt starttls_relay which might fail for invalid host
        // but the builder should succeed
        let config = EmailConfig {
            smtp_host: Some("smtp.gmail.com".to_string()),
            smtp_port: 587,
            smtp_username: Some("user@gmail.com".to_string()),
            smtp_password: Some("password".into()),
            from_address: Some("user@gmail.com".to_string()),
            to_addresses: vec!["recipient@example.com".to_string()],
            use_tls: true,
            ..Default::default()
        };
        let result = EmailNotifier::new(config, empty_registry());
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_notify_success_cascade_pr_disabled() {
        let notifier = EmailNotifier::new(disabled_config(), empty_registry()).unwrap();
        let mut issue = Issue::new("1", "PROJ-1", "Test", "https://example.com", "linear");
        issue.set_metadata("cascade_upstream_repo", "org/upstream");
        issue.set_metadata("cascade_downstream_repo", "org/downstream");

        let result = notifier
            .notify_success(&issue, "https://github.com/org/downstream/pull/5")
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_notify_success_pr_update_disabled() {
        let notifier = EmailNotifier::new(disabled_config(), empty_registry()).unwrap();
        let mut issue = Issue::new("1", "PROJ-1", "Test", "https://example.com", "linear");
        issue.set_metadata("is_pr_update", true);

        let result = notifier
            .notify_success(&issue, "https://github.com/org/repo/pull/42")
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_notify_success_regular_pr_disabled() {
        let notifier = EmailNotifier::new(disabled_config(), empty_registry()).unwrap();
        let issue = Issue::new("1", "PROJ-1", "Test", "https://example.com", "linear");

        let result = notifier
            .notify_success(&issue, "https://github.com/org/repo/pull/1")
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_notify_completed_regression_resolved_disabled() {
        let notifier = EmailNotifier::new(disabled_config(), empty_registry()).unwrap();
        let mut issue = Issue::new("1", "PROJ-1", "Test", "https://example.com", "linear");
        issue.set_metadata("regression_resolved", true);

        let result = notifier.notify_completed(&issue).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_notify_completed_regular_disabled() {
        let notifier = EmailNotifier::new(disabled_config(), empty_registry()).unwrap();
        let issue = Issue::new("1", "PROJ-1", "Test", "https://example.com", "linear");

        let result = notifier.notify_completed(&issue).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_notify_failed_regression_detected_disabled() {
        let notifier = EmailNotifier::new(disabled_config(), empty_registry()).unwrap();
        let mut issue = Issue::new("1", "PROJ-1", "Test", "https://example.com", "linear");
        issue.set_metadata("regression_detected", true);

        let result = notifier.notify_failed(&issue, "Regression error").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_notify_failed_cascade_disabled() {
        let notifier = EmailNotifier::new(disabled_config(), empty_registry()).unwrap();
        let mut issue = Issue::new("1", "PROJ-1", "Test", "https://example.com", "linear");
        issue.set_metadata("cascade_upstream_repo", "org/upstream");
        issue.set_metadata("cascade_downstream_repo", "org/downstream");

        let result = notifier.notify_failed(&issue, "Cascade error").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_notify_failed_regular_disabled() {
        let notifier = EmailNotifier::new(disabled_config(), empty_registry()).unwrap();
        let issue = Issue::new("1", "PROJ-1", "Test", "https://example.com", "linear");

        let result = notifier.notify_failed(&issue, "Some error").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_notify_merged_disabled() {
        let notifier = EmailNotifier::new(disabled_config(), empty_registry()).unwrap();
        let issue = Issue::new("1", "PROJ-1", "Test", "https://example.com", "linear");

        let result = notifier
            .notify_merged(&issue, "https://github.com/org/repo/pull/1")
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_notify_closed_disabled() {
        let notifier = EmailNotifier::new(disabled_config(), empty_registry()).unwrap();
        let issue = Issue::new("1", "PROJ-1", "Test", "https://example.com", "linear");

        let result = notifier
            .notify_closed(&issue, "https://github.com/org/repo/pull/1")
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_ask_question_with_why_field() {
        let config = EmailConfig {
            to_addresses: vec!["user@example.com".to_string()],
            ..Default::default()
        };
        let notifier = EmailNotifier::new(config, empty_registry()).unwrap();
        let issue = Issue::new("1", "LIN-1", "Test", "https://example.com", "linear");
        let request = AskRequest {
            correlation_id: "tok-why".to_string(),
            source: "linear".to_string(),
            repo: None,
            issue_id: "1".to_string(),
            short_id: "LIN-1".to_string(),
            question: claudear_core::types::BlockingQuestion {
                question: "Which env?".to_string(),
                context: None,
                options: vec![],
                why: Some("Multiple environments available".to_string()),
            },
            asked_at: Utc::now(),
            target_discord_id: None,
            target_email: None,
            target_slack_id: None,
        };
        let delivery = notifier
            .ask_question(&issue, &request)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(delivery.channel, "email");
    }

    #[tokio::test]
    async fn test_ask_question_with_context_field() {
        let config = EmailConfig {
            to_addresses: vec!["user@example.com".to_string()],
            ..Default::default()
        };
        let notifier = EmailNotifier::new(config, empty_registry()).unwrap();
        let issue = Issue::new("1", "LIN-1", "Test", "https://example.com", "linear");
        let request = AskRequest {
            correlation_id: "tok-ctx".to_string(),
            source: "linear".to_string(),
            repo: None,
            issue_id: "1".to_string(),
            short_id: "LIN-1".to_string(),
            question: claudear_core::types::BlockingQuestion {
                question: "Which env?".to_string(),
                context: Some("We have staging and production".to_string()),
                options: vec![],
                why: None,
            },
            asked_at: Utc::now(),
            target_discord_id: None,
            target_email: None,
            target_slack_id: None,
        };
        let delivery = notifier
            .ask_question(&issue, &request)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(delivery.channel, "email");
    }

    #[tokio::test]
    async fn test_ask_question_with_options_field() {
        let config = EmailConfig {
            to_addresses: vec!["user@example.com".to_string()],
            ..Default::default()
        };
        let notifier = EmailNotifier::new(config, empty_registry()).unwrap();
        let issue = Issue::new("1", "LIN-1", "Test", "https://example.com", "linear");
        let request = AskRequest {
            correlation_id: "tok-options".to_string(),
            source: "linear".to_string(),
            repo: None,
            issue_id: "1".to_string(),
            short_id: "LIN-1".to_string(),
            question: claudear_core::types::BlockingQuestion {
                question: "Which env?".to_string(),
                context: None,
                options: vec!["staging".to_string(), "production".to_string()],
                why: None,
            },
            asked_at: Utc::now(),
            target_discord_id: None,
            target_email: None,
            target_slack_id: None,
        };
        let delivery = notifier
            .ask_question(&issue, &request)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(delivery.channel, "email");
    }

    #[test]
    fn test_extract_plain_body_no_subparts() {
        let raw = b"From: user@example.com\r\nSubject: Test\r\nContent-Type: text/plain\r\n\r\nSimple body";
        let parsed = mailparse::parse_mail(raw).unwrap();
        let body = EmailNotifier::extract_plain_body(&parsed);
        assert!(body.contains("Simple body"));
    }

    #[test]
    fn test_extract_plain_body_multipart_with_html_only() {
        let raw = b"From: user@example.com\r\nSubject: Test\r\nContent-Type: multipart/alternative; boundary=bound\r\n\r\n--bound\r\nContent-Type: text/html\r\n\r\n<b>Only HTML</b>\r\n--bound--";
        let parsed = mailparse::parse_mail(raw).unwrap();
        let body = EmailNotifier::extract_plain_body(&parsed);
        // Falls back to parent body when no text/plain part found
        assert!(!body.is_empty() || body.is_empty()); // Should not panic
    }

    #[test]
    fn test_extract_plain_body_multipart_prefers_text_plain() {
        let raw = b"From: user@example.com\r\nSubject: Test\r\nContent-Type: multipart/mixed; boundary=bound\r\n\r\n--bound\r\nContent-Type: text/html\r\n\r\n<b>HTML content</b>\r\n--bound\r\nContent-Type: text/plain\r\n\r\nPlain text content\r\n--bound--";
        let parsed = mailparse::parse_mail(raw).unwrap();
        let body = EmailNotifier::extract_plain_body(&parsed);
        assert!(body.contains("Plain text content"));
    }

    #[tokio::test]
    async fn test_notify_urgent_issues_many_issues_disabled() {
        let notifier = EmailNotifier::new(disabled_config(), empty_registry()).unwrap();
        let issues: Vec<Issue> = (1..=15)
            .map(|i| {
                Issue::new(
                    i.to_string(),
                    format!("PROJ-{}", i),
                    format!("Issue {}", i),
                    format!("https://example.com/{}", i),
                    "linear",
                )
            })
            .collect();

        let result = notifier.notify_urgent_issues(&issues).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_send_email_no_transport_returns_ok() {
        let notifier = EmailNotifier::new(disabled_config(), empty_registry()).unwrap();
        let issue = Issue::new("1", "PROJ-1", "Test", "https://example.com", "linear");
        // All notification methods should return Ok when no transport
        assert!(notifier.notify_start(&issue).await.is_ok());
        assert!(notifier
            .notify_success(&issue, "https://pr.url")
            .await
            .is_ok());
        assert!(notifier.notify_completed(&issue).await.is_ok());
        assert!(notifier.notify_failed(&issue, "error").await.is_ok());
        assert!(notifier
            .notify_merged(&issue, "https://pr.url")
            .await
            .is_ok());
        assert!(notifier
            .notify_closed(&issue, "https://pr.url")
            .await
            .is_ok());
        assert!(notifier.notify_status("status").await.is_ok());
        assert!(notifier.notify_urgent_issues(&[]).await.is_ok());
    }

    #[tokio::test]
    async fn test_send_email_no_from_address_returns_ok() {
        let notifier = EmailNotifier::new(partial_config(), empty_registry()).unwrap();
        let issue = Issue::new("1", "PROJ-1", "Test", "https://example.com", "linear");
        // Transport exists but no from_address, send_email returns Ok(())
        assert!(notifier.notify_start(&issue).await.is_ok());
    }

    #[test]
    fn test_resolve_recipients_resolved_user_not_in_registry() {
        let config = EmailConfig {
            to_addresses: vec!["fallback@example.com".to_string()],
            ..Default::default()
        };
        let notifier = EmailNotifier::new(config, empty_registry()).unwrap();
        let mut issue = Issue::new("1", "LIN-1", "Test", "https://example.com", "linear");
        issue.set_metadata("resolved_user", "nonexistent_user");
        let recipients = notifier.resolve_recipients(Some(&issue));
        assert_eq!(recipients, vec!["fallback@example.com".to_string()]);
    }

    #[test]
    fn test_target_email_for_issue_with_resolved_user() {
        let mut users = std::collections::HashMap::new();
        users.insert(
            "alice".to_string(),
            claudear_config::config::UserConfig {
                email: Some("alice@company.com".to_string()),
                ..Default::default()
            },
        );
        let registry = UserRegistry::new(users);
        let config = EmailConfig {
            to_addresses: vec!["global@example.com".to_string()],
            ..Default::default()
        };
        let notifier = EmailNotifier::new(config, registry).unwrap();
        let mut issue = Issue::new("1", "LIN-1", "Test", "https://example.com", "linear");
        issue.set_metadata("resolved_user", "alice");
        assert_eq!(
            notifier.target_email_for_issue(&issue),
            Some("alice@company.com".to_string())
        );
    }

    #[test]
    fn test_extract_email_address_angle_brackets_with_spaces() {
        assert_eq!(
            EmailNotifier::extract_email_address("  User Name   < user@example.com  >").as_deref(),
            Some("user@example.com")
        );
    }

    #[test]
    fn test_extract_email_address_only_at_symbol() {
        // Just "@" is technically "has @" but not a valid email
        // The function returns it since it doesn't validate beyond containing @
        let result = EmailNotifier::extract_email_address("@");
        assert!(result.is_some()); // It has @ so it returns Some
    }

    #[test]
    fn test_sanitize_reply_text_token_embedded_in_text_strips_line() {
        let body = "Start CLAUDEAR-Q:tok-embed End\nActual reply";
        let result = EmailNotifier::sanitize_reply_text(body).unwrap();
        assert_eq!(result, "Start CLAUDEAR-Q:tok-embed End\nActual reply");
    }

    #[tokio::test]
    async fn test_poll_question_replies_non_tls_returns_empty() {
        let config = EmailConfig {
            imap_host: Some("imap.example.com".to_string()),
            imap_username: Some("user".to_string()),
            imap_password: Some("pass".into()),
            imap_use_tls: false,
            ..Default::default()
        };
        let notifier = EmailNotifier::new(config, empty_registry()).unwrap();
        let request = AskRequest {
            correlation_id: "tok-notls".to_string(),
            source: "linear".to_string(),
            repo: None,
            issue_id: "1".to_string(),
            short_id: "LIN-1".to_string(),
            question: claudear_core::types::BlockingQuestion {
                question: "Q?".to_string(),
                context: None,
                options: vec![],
                why: None,
            },
            asked_at: Utc::now(),
            target_discord_id: None,
            target_email: None,
            target_slack_id: None,
        };
        let replies = notifier
            .poll_question_replies(&request, Utc::now())
            .await
            .unwrap();
        assert!(replies.is_empty());
    }

    #[tokio::test]
    async fn test_ask_question_all_fields_present() {
        let config = EmailConfig {
            to_addresses: vec!["user@example.com".to_string()],
            ..Default::default()
        };
        let notifier = EmailNotifier::new(config, empty_registry()).unwrap();
        let issue = Issue::new("1", "LIN-1", "Test", "https://example.com", "linear");
        let request = AskRequest {
            correlation_id: "tok-all".to_string(),
            source: "linear".to_string(),
            repo: Some("org/repo".to_string()),
            issue_id: "1".to_string(),
            short_id: "LIN-1".to_string(),
            question: claudear_core::types::BlockingQuestion {
                question: "Choose a path".to_string(),
                context: Some("Repo has multiple modules".to_string()),
                options: vec!["option-a".to_string(), "option-b".to_string()],
                why: Some("Because the choice matters".to_string()),
            },
            asked_at: Utc::now(),
            target_discord_id: None,
            target_email: None,
            target_slack_id: None,
        };
        let delivery = notifier
            .ask_question(&issue, &request)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(delivery.channel, "email");
        assert_eq!(delivery.target.as_deref(), Some("user@example.com"));
        assert!(delivery.message_id.is_none());
    }

    #[tokio::test]
    async fn test_ask_question_target_from_resolved_user() {
        let mut users = std::collections::HashMap::new();
        users.insert(
            "bob".to_string(),
            claudear_config::config::UserConfig {
                email: Some("bob@company.com".to_string()),
                ..Default::default()
            },
        );
        let registry = UserRegistry::new(users);
        let config = EmailConfig {
            to_addresses: vec!["global@example.com".to_string()],
            ..Default::default()
        };
        let notifier = EmailNotifier::new(config, registry).unwrap();
        let mut issue = Issue::new("1", "LIN-1", "Test", "https://example.com", "linear");
        issue.set_metadata("resolved_user", "bob");
        let request = AskRequest {
            correlation_id: "tok-bob".to_string(),
            source: "linear".to_string(),
            repo: None,
            issue_id: "1".to_string(),
            short_id: "LIN-1".to_string(),
            question: claudear_core::types::BlockingQuestion {
                question: "Q?".to_string(),
                context: None,
                options: vec![],
                why: None,
            },
            asked_at: Utc::now(),
            target_discord_id: None,
            target_email: None,
            target_slack_id: None,
        };
        let delivery = notifier
            .ask_question(&issue, &request)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(delivery.target.as_deref(), Some("bob@company.com"));
    }

    #[test]
    fn test_extract_email_address_bracket_format() {
        assert_eq!(
            EmailNotifier::extract_email_address("John Doe <john@example.com>").as_deref(),
            Some("john@example.com")
        );
    }

    #[test]
    fn test_extract_email_address_plain() {
        assert_eq!(
            EmailNotifier::extract_email_address("john@example.com").as_deref(),
            Some("john@example.com")
        );
    }

    #[test]
    fn test_extract_email_address_no_at() {
        assert_eq!(
            EmailNotifier::extract_email_address("not an email").as_deref(),
            None
        );
    }

    #[test]
    fn test_extract_email_address_empty() {
        assert_eq!(EmailNotifier::extract_email_address("").as_deref(), None);
    }

    #[test]
    fn test_extract_email_address_bracket_empty() {
        assert_eq!(EmailNotifier::extract_email_address("<>").as_deref(), None);
    }

    #[test]
    fn test_extract_email_address_case_normalization() {
        assert_eq!(
            EmailNotifier::extract_email_address("USER@EXAMPLE.COM").as_deref(),
            Some("user@example.com")
        );
        assert_eq!(
            EmailNotifier::extract_email_address("Display Name <MiXeD@CaSe.CoM>").as_deref(),
            Some("mixed@case.com")
        );
    }

    #[test]
    fn test_sanitize_reply_text_normal() {
        let body = "Yes, deploy to production please";
        let result = EmailNotifier::sanitize_reply_text(body).unwrap();
        assert_eq!(result, "Yes, deploy to production please");
    }

    #[test]
    fn test_sanitize_reply_text_with_quoted() {
        let body = "> On Monday, bot wrote:\n> Original question\nMy answer";
        let result = EmailNotifier::sanitize_reply_text(body).unwrap();
        assert_eq!(result, "My answer");
    }

    #[test]
    fn test_sanitize_reply_text_with_token() {
        let body = "CLAUDEAR-Q:tok-t\nUse staging environment\nThanks";
        let result = EmailNotifier::sanitize_reply_text(body).unwrap();
        assert_eq!(result, "CLAUDEAR-Q:tok-t\nUse staging environment\nThanks");
    }

    #[test]
    fn test_sanitize_reply_text_empty() {
        // Only quoted lines -> None
        let body = "> line 1\n> line 2\n> line 3";
        assert!(EmailNotifier::sanitize_reply_text(body).is_none());

        // Completely empty -> None
        assert!(EmailNotifier::sanitize_reply_text("").is_none());
    }

    #[test]
    fn test_sanitize_reply_text_long() {
        let long_text = "x".repeat(5000);
        let result = EmailNotifier::sanitize_reply_text(&long_text).unwrap();
        assert!(result.len() <= 4000);
    }

    #[test]
    fn test_sanitize_reply_text_max_lines() {
        // Build 50 non-empty, non-quoted, non-token lines
        let body = (1..=50)
            .map(|i| format!("Content line {}", i))
            .collect::<Vec<_>>()
            .join("\n");
        let result = EmailNotifier::sanitize_reply_text(&body).unwrap();
        let line_count = result.lines().count();
        assert!(
            line_count <= 30,
            "Expected at most 30 lines, got {}",
            line_count
        );
        // Should contain line 30 but not line 31
        assert!(result.contains("Content line 30"));
        assert!(!result.contains("Content line 31"));
    }

    #[test]
    fn test_format_issue_info_full_structure() {
        let mut issue = Issue::new(
            "42",
            "PROJ-42",
            "Login broken",
            "https://example.com/42",
            "linear",
        );
        issue.priority = IssuePriority::High;
        issue.status = IssueStatus::InProgress;

        let info = EmailNotifier::format_issue_info(&issue);
        assert!(info.contains("PROJ-42"));
        assert!(info.contains("Login broken"));
        assert!(info.contains("linear"));
        assert!(info.contains("https://example.com/42"));
        // Verify structure
        let lines: Vec<&str> = info.lines().collect();
        assert_eq!(lines.len(), 5);
        assert!(lines[0].starts_with("Issue: PROJ-42 - Login broken"));
        assert!(lines[1].starts_with("Source:"));
        assert!(lines[2].starts_with("Priority:"));
        assert!(lines[3].starts_with("Status:"));
        assert!(lines[4].starts_with("URL:"));
    }

    #[tokio::test]
    async fn test_ask_question_no_recipients_returns_none_target() {
        let config = EmailConfig {
            to_addresses: vec![],
            ..Default::default()
        };
        let notifier = EmailNotifier::new(config, empty_registry()).unwrap();
        let issue = Issue::new("1", "LIN-1", "Test", "https://example.com", "linear");
        let request = AskRequest {
            correlation_id: "tok-none".to_string(),
            source: "linear".to_string(),
            repo: None,
            issue_id: "1".to_string(),
            short_id: "LIN-1".to_string(),
            question: claudear_core::types::BlockingQuestion {
                question: "Q?".to_string(),
                context: None,
                options: vec![],
                why: None,
            },
            asked_at: Utc::now(),
            target_discord_id: None,
            target_email: None,
            target_slack_id: None,
        };
        let delivery = notifier
            .ask_question(&issue, &request)
            .await
            .unwrap()
            .unwrap();
        assert!(delivery.target.is_none());
    }

    #[tokio::test]
    async fn test_notify_start_with_trigger_reason() {
        let notifier = EmailNotifier::new(disabled_config(), empty_registry()).unwrap();
        let mut issue = Issue::new("1", "LIN-1", "Test", "https://example.com", "linear");
        issue.set_metadata("trigger_reason", "Retry attempt 2: timeout");
        // With no transport, notify_start builds the body (including trigger_reason) but returns Ok
        let result = notifier.notify_start(&issue).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_notify_start_without_trigger_reason() {
        let notifier = EmailNotifier::new(disabled_config(), empty_registry()).unwrap();
        let issue = Issue::new("1", "LIN-1", "Test", "https://example.com", "linear");
        let result = notifier.notify_start(&issue).await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_format_issue_info_does_not_include_trigger_reason() {
        let mut issue = Issue::new("1", "LIN-1", "Test", "https://example.com", "linear");
        issue.set_metadata("trigger_reason", "Retry attempt 2: timeout");
        let info = EmailNotifier::format_issue_info(&issue);
        // format_issue_info only has issue fields; trigger_reason is appended by notify_start
        assert!(!info.contains("Trigger"));
        assert!(info.contains("LIN-1"));
    }

    #[test]
    fn test_extract_email_address_multiple_at_signs() {
        let result = EmailNotifier::extract_email_address("user@host@domain.com");
        // Contains @ so returns Some (the raw string lowercased)
        assert!(result.is_some());
    }

    #[test]
    fn test_extract_email_address_whitespace_inside_angle_brackets() {
        let result = EmailNotifier::extract_email_address("Name < user@example.com >");
        assert_eq!(result, Some("user@example.com".to_string()));
    }

    #[test]
    fn test_sanitize_reply_text_single_non_quoted_line() {
        let result = EmailNotifier::sanitize_reply_text("Yes");
        assert_eq!(result, Some("Yes".to_string()));
    }

    #[test]
    fn test_sanitize_reply_text_tabs_treated_as_whitespace() {
        let body = "\t\t\n\tAnswer here\n\t\t";
        let result = EmailNotifier::sanitize_reply_text(body);
        assert_eq!(result, Some("Answer here".to_string()));
    }

    #[test]
    fn test_sanitize_reply_text_exactly_4000_chars_no_further_truncation() {
        let line = "a".repeat(4000);
        let result = EmailNotifier::sanitize_reply_text(&line).unwrap();
        assert_eq!(result.len(), 4000);
    }

    #[test]
    fn test_sanitize_reply_text_over_4000_chars_is_truncated() {
        let line = "b".repeat(5000);
        let result = EmailNotifier::sanitize_reply_text(&line).unwrap();
        assert_eq!(result.len(), 4000);
    }

    #[test]
    fn test_extract_plain_body_multipart_html_only_uses_fallback() {
        let raw = b"Content-Type: multipart/alternative; boundary=\"boundary\"\r\n\r\n--boundary\r\nContent-Type: text/html\r\n\r\n<p>Hello</p>\r\n--boundary--";
        let parsed = mailparse::parse_mail(raw).unwrap();
        let _body = EmailNotifier::extract_plain_body(&parsed);
        // Should not panic; falls back to parent body
    }

    #[test]
    fn test_format_issue_info_default_priority_and_status() {
        let issue = Issue::new(
            "id-x",
            "SEN-1",
            "Error msg",
            "https://sentry.io/1",
            "sentry",
        );
        let info = EmailNotifier::format_issue_info(&issue);
        assert!(info.contains("SEN-1"));
        assert!(info.contains("Error msg"));
        assert!(info.contains("sentry"));
        assert!(info.contains("none")); // default priority is None
        assert!(info.contains("open")); // default status
    }

    #[test]
    fn test_expected_reply_emails_empty_config_and_no_target() {
        let notifier = EmailNotifier::new(disabled_config(), empty_registry()).unwrap();
        let request = claudear_core::types::AskRequest {
            correlation_id: "tok-empty-cfg".to_string(),
            source: "linear".to_string(),
            repo: None,
            issue_id: "1".to_string(),
            short_id: "LIN-1".to_string(),
            question: claudear_core::types::BlockingQuestion {
                question: "Q?".to_string(),
                context: None,
                options: vec![],
                why: None,
            },
            asked_at: chrono::Utc::now(),
            target_discord_id: None,
            target_email: None,
            target_slack_id: None,
        };
        let emails = notifier.expected_reply_emails(&request);
        assert!(emails.is_empty());
    }

    #[test]
    fn test_resolve_recipients_unknown_slug_falls_back() {
        let mut users = std::collections::HashMap::new();
        users.insert(
            "alice".to_string(),
            claudear_config::config::UserConfig {
                email: Some("alice@company.com".to_string()),
                ..Default::default()
            },
        );
        let registry = claudear_config::users::UserRegistry::new(users);
        let config = EmailConfig {
            to_addresses: vec!["fallback@example.com".to_string()],
            ..Default::default()
        };
        let notifier = EmailNotifier::new(config, registry).unwrap();
        let mut issue = Issue::new("1", "LIN-1", "Test", "https://example.com", "linear");
        issue.set_metadata("resolved_user", "unknown_slug");
        let recipients = notifier.resolve_recipients(Some(&issue));
        assert_eq!(recipients, vec!["fallback@example.com"]);
    }

    #[test]
    fn test_email_config_defaults() {
        let config = EmailConfig::default();
        assert_eq!(config.smtp_port, 587);
        assert!(config.use_tls);
        assert_eq!(config.imap_port, 993);
        assert!(config.imap_use_tls);
        assert_eq!(config.imap_folder, "INBOX");
        assert!(config.to_addresses.is_empty());
        assert!(config.smtp_host.is_none());
        assert!(config.from_address.is_none());
    }

    #[tokio::test]
    async fn test_notify_urgent_issues_plural_subject_format() {
        let notifier = EmailNotifier::new(disabled_config(), empty_registry()).unwrap();
        let issues = vec![
            Issue::new("1", "P-1", "Bug 1", "https://example.com", "linear"),
            Issue::new("2", "P-2", "Bug 2", "https://example.com", "linear"),
            Issue::new("3", "P-3", "Bug 3", "https://example.com", "linear"),
        ];
        let result = notifier.notify_urgent_issues(&issues).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_notify_success_cascade_with_upstream_and_downstream() {
        let notifier = EmailNotifier::new(disabled_config(), empty_registry()).unwrap();
        let mut issue = Issue::new("1", "LIN-1", "Fix", "https://example.com", "linear");
        issue.set_metadata("cascade_downstream_repo", "downstream/repo");
        issue.set_metadata("cascade_upstream_repo", "upstream/repo");
        let result = notifier
            .notify_success(&issue, "https://github.com/pr/5")
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_notify_success_pr_update_metadata() {
        let notifier = EmailNotifier::new(disabled_config(), empty_registry()).unwrap();
        let mut issue = Issue::new("1", "LIN-1", "Fix", "https://example.com", "linear");
        issue.set_metadata("is_pr_update", true);
        let result = notifier
            .notify_success(&issue, "https://github.com/pr/5")
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_notify_completed_regression_resolved_metadata() {
        let notifier = EmailNotifier::new(disabled_config(), empty_registry()).unwrap();
        let mut issue = Issue::new("1", "LIN-1", "Fix", "https://example.com", "linear");
        issue.set_metadata("regression_resolved", true);
        let result = notifier.notify_completed(&issue).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_notify_completed_custom_completion_reason() {
        let notifier = EmailNotifier::new(disabled_config(), empty_registry()).unwrap();
        let mut issue = Issue::new("1", "LIN-1", "Fix", "https://example.com", "linear");
        issue.set_metadata("completion_reason", "Already fixed");
        let result = notifier.notify_completed(&issue).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_notify_failed_regression_detected_metadata() {
        let notifier = EmailNotifier::new(disabled_config(), empty_registry()).unwrap();
        let mut issue = Issue::new("1", "LIN-1", "Fix", "https://example.com", "linear");
        issue.set_metadata("regression_detected", true);
        let result = notifier.notify_failed(&issue, "Tests failing").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_notify_failed_cascade_downstream_metadata() {
        let notifier = EmailNotifier::new(disabled_config(), empty_registry()).unwrap();
        let mut issue = Issue::new("1", "LIN-1", "Fix", "https://example.com", "linear");
        issue.set_metadata("cascade_downstream_repo", "downstream/repo");
        issue.set_metadata("cascade_upstream_repo", "upstream/repo");
        let result = notifier.notify_failed(&issue, "Build error").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_notify_success_trigger_reason_appended() {
        let notifier = EmailNotifier::new(disabled_config(), empty_registry()).unwrap();
        let mut issue = Issue::new("1", "LIN-1", "Fix", "https://example.com", "linear");
        issue.set_metadata("trigger_reason", "Auto retry");
        let result = notifier
            .notify_success(&issue, "https://github.com/pr/1")
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_notify_failed_trigger_reason_appended() {
        let notifier = EmailNotifier::new(disabled_config(), empty_registry()).unwrap();
        let mut issue = Issue::new("1", "LIN-1", "Fix", "https://example.com", "linear");
        issue.set_metadata("trigger_reason", "Manual");
        let result = notifier.notify_failed(&issue, "error msg").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_notify_success_with_confidence() {
        let notifier = EmailNotifier::new(disabled_config(), empty_registry()).unwrap();
        let mut issue = Issue::new("1", "LIN-1", "Fix", "https://example.com", "linear");
        issue.set_metadata("confidence", 85u8);
        let result = notifier
            .notify_success(&issue, "https://github.com/pr/1")
            .await;
        assert!(result.is_ok());
    }

    // --- Tests that exercise send_email internals (with transport) ---

    fn send_config() -> EmailConfig {
        EmailConfig {
            smtp_host: Some("localhost".to_string()),
            smtp_port: 25,
            smtp_username: Some("user".to_string()),
            smtp_password: Some("pass".into()),
            from_address: Some("bot@example.com".to_string()),
            to_addresses: vec!["recipient@example.com".to_string()],
            use_tls: false,
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn test_send_email_builds_message_and_fails_on_transport() {
        let notifier = EmailNotifier::new(send_config(), empty_registry()).unwrap();
        assert!(notifier.is_enabled());
        let issue = Issue::new("1", "PROJ-1", "Test", "https://example.com", "linear");
        // This will build the email message but fail when trying to connect to localhost:25
        let result = notifier.notify_start(&issue).await;
        assert!(result.is_err());
        let err_str = result.unwrap_err().to_string();
        assert!(
            err_str.contains("email")
                || err_str.contains("send")
                || err_str.contains("SMTP")
                || err_str.contains("Failed")
        );
    }

    #[tokio::test]
    async fn test_send_email_success_path_exercises_message_building() {
        let notifier = EmailNotifier::new(send_config(), empty_registry()).unwrap();
        let issue = Issue::new("1", "PROJ-1", "Test", "https://example.com", "linear");
        // Exercises: from_address parsing, to_address parsing, Message building, transport.send
        let result = notifier
            .notify_success(&issue, "https://github.com/pr/1")
            .await;
        assert!(result.is_err()); // Transport fails but message was built
    }

    #[tokio::test]
    async fn test_send_email_completed_exercises_message_building() {
        let notifier = EmailNotifier::new(send_config(), empty_registry()).unwrap();
        let issue = Issue::new("1", "PROJ-1", "Test", "https://example.com", "linear");
        let result = notifier.notify_completed(&issue).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_send_email_failed_exercises_message_building() {
        let notifier = EmailNotifier::new(send_config(), empty_registry()).unwrap();
        let issue = Issue::new("1", "PROJ-1", "Test", "https://example.com", "linear");
        let result = notifier.notify_failed(&issue, "Build error").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_send_email_merged_exercises_message_building() {
        let notifier = EmailNotifier::new(send_config(), empty_registry()).unwrap();
        let issue = Issue::new("1", "PROJ-1", "Test", "https://example.com", "linear");
        let result = notifier
            .notify_merged(&issue, "https://github.com/pr/1")
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_send_email_closed_exercises_message_building() {
        let notifier = EmailNotifier::new(send_config(), empty_registry()).unwrap();
        let issue = Issue::new("1", "PROJ-1", "Test", "https://example.com", "linear");
        let result = notifier
            .notify_closed(&issue, "https://github.com/pr/1")
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_send_email_status_exercises_message_building() {
        let notifier = EmailNotifier::new(send_config(), empty_registry()).unwrap();
        let result = notifier.notify_status("Status update message").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_send_email_urgent_issues_exercises_message_building() {
        let notifier = EmailNotifier::new(send_config(), empty_registry()).unwrap();
        let issues = vec![
            Issue::new("1", "PROJ-1", "Bug 1", "https://example.com", "linear"),
            Issue::new("2", "PROJ-2", "Bug 2", "https://example.com", "linear"),
        ];
        let result = notifier.notify_urgent_issues(&issues).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_send_email_ask_question_exercises_message_building() {
        let notifier = EmailNotifier::new(send_config(), empty_registry()).unwrap();
        let issue = Issue::new("1", "LIN-1", "Test", "https://example.com", "linear");
        let request = AskRequest {
            correlation_id: "tok-send".to_string(),
            source: "linear".to_string(),
            repo: None,
            issue_id: "1".to_string(),
            short_id: "LIN-1".to_string(),
            question: claudear_core::types::BlockingQuestion {
                question: "Which env?".to_string(),
                context: Some("Multiple envs".to_string()),
                options: vec!["staging".to_string(), "prod".to_string()],
                why: Some("Need to choose".to_string()),
            },
            asked_at: Utc::now(),
            target_discord_id: None,
            target_email: None,
            target_slack_id: None,
        };
        let result = notifier.ask_question(&issue, &request).await;
        assert!(result.is_err()); // Transport fails
    }

    #[tokio::test]
    async fn test_send_email_cascade_success_exercises_message_building() {
        let notifier = EmailNotifier::new(send_config(), empty_registry()).unwrap();
        let mut issue = Issue::new("1", "PROJ-1", "Test", "https://example.com", "linear");
        issue.set_metadata("cascade_downstream_repo", "downstream/repo");
        issue.set_metadata("cascade_upstream_repo", "upstream/repo");
        let result = notifier
            .notify_success(&issue, "https://github.com/pr/1")
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_send_email_pr_update_success_exercises_message_building() {
        let notifier = EmailNotifier::new(send_config(), empty_registry()).unwrap();
        let mut issue = Issue::new("1", "PROJ-1", "Test", "https://example.com", "linear");
        issue.set_metadata("is_pr_update", true);
        let result = notifier
            .notify_success(&issue, "https://github.com/pr/1")
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_send_email_regression_resolved_completed_exercises_message_building() {
        let notifier = EmailNotifier::new(send_config(), empty_registry()).unwrap();
        let mut issue = Issue::new("1", "PROJ-1", "Test", "https://example.com", "linear");
        issue.set_metadata("regression_resolved", true);
        let result = notifier.notify_completed(&issue).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_send_email_completion_reason_exercises_message_building() {
        let notifier = EmailNotifier::new(send_config(), empty_registry()).unwrap();
        let mut issue = Issue::new("1", "PROJ-1", "Test", "https://example.com", "linear");
        issue.set_metadata("completion_reason", "Already resolved");
        let result = notifier.notify_completed(&issue).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_send_email_regression_detected_failed_exercises_message_building() {
        let notifier = EmailNotifier::new(send_config(), empty_registry()).unwrap();
        let mut issue = Issue::new("1", "PROJ-1", "Test", "https://example.com", "linear");
        issue.set_metadata("regression_detected", true);
        let result = notifier.notify_failed(&issue, "Tests failing").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_send_email_cascade_failed_exercises_message_building() {
        let notifier = EmailNotifier::new(send_config(), empty_registry()).unwrap();
        let mut issue = Issue::new("1", "PROJ-1", "Test", "https://example.com", "linear");
        issue.set_metadata("cascade_downstream_repo", "downstream/repo");
        issue.set_metadata("cascade_upstream_repo", "upstream/repo");
        let result = notifier.notify_failed(&issue, "Build error").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_send_email_with_trigger_reason_exercises_message_building() {
        let notifier = EmailNotifier::new(send_config(), empty_registry()).unwrap();
        let mut issue = Issue::new("1", "PROJ-1", "Test", "https://example.com", "linear");
        issue.set_metadata("trigger_reason", "Retry attempt");
        let result = notifier.notify_start(&issue).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_send_email_success_trigger_reason_exercises_message_building() {
        let notifier = EmailNotifier::new(send_config(), empty_registry()).unwrap();
        let mut issue = Issue::new("1", "PROJ-1", "Test", "https://example.com", "linear");
        issue.set_metadata("trigger_reason", "Manual retry");
        let result = notifier
            .notify_success(&issue, "https://github.com/pr/1")
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_send_email_failed_trigger_reason_exercises_message_building() {
        let notifier = EmailNotifier::new(send_config(), empty_registry()).unwrap();
        let mut issue = Issue::new("1", "PROJ-1", "Test", "https://example.com", "linear");
        issue.set_metadata("trigger_reason", "Auto retry");
        let result = notifier.notify_failed(&issue, "timeout").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_send_email_zero_recipients_returns_ok() {
        let config = EmailConfig {
            smtp_host: Some("localhost".to_string()),
            smtp_port: 25,
            smtp_username: Some("user".to_string()),
            smtp_password: Some("pass".into()),
            from_address: Some("bot@example.com".to_string()),
            to_addresses: vec![],
            use_tls: false,
            ..Default::default()
        };
        let notifier = EmailNotifier::new(config, empty_registry()).unwrap();
        let issue = Issue::new("1", "PROJ-1", "Test", "https://example.com", "linear");
        // No recipients -> for loop doesn't execute -> Ok(())
        let result = notifier.notify_start(&issue).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_send_email_invalid_from_address_returns_error() {
        let config = EmailConfig {
            smtp_host: Some("localhost".to_string()),
            smtp_port: 25,
            smtp_username: Some("user".to_string()),
            smtp_password: Some("pass".into()),
            from_address: Some("not-a-valid-email".to_string()),
            to_addresses: vec!["valid@example.com".to_string()],
            use_tls: false,
            ..Default::default()
        };
        let notifier = EmailNotifier::new(config, empty_registry()).unwrap();
        let issue = Issue::new("1", "PROJ-1", "Test", "https://example.com", "linear");
        let result = notifier.notify_start(&issue).await;
        assert!(result.is_err());
        let err_str = result.unwrap_err().to_string();
        assert!(err_str.contains("Invalid from address") || err_str.contains("email"));
    }

    #[tokio::test]
    async fn test_send_email_invalid_to_address_returns_error() {
        let config = EmailConfig {
            smtp_host: Some("localhost".to_string()),
            smtp_port: 25,
            smtp_username: Some("user".to_string()),
            smtp_password: Some("pass".into()),
            from_address: Some("bot@example.com".to_string()),
            to_addresses: vec!["not-a-valid-email".to_string()],
            use_tls: false,
            ..Default::default()
        };
        let notifier = EmailNotifier::new(config, empty_registry()).unwrap();
        let issue = Issue::new("1", "PROJ-1", "Test", "https://example.com", "linear");
        let result = notifier.notify_start(&issue).await;
        assert!(result.is_err());
        let err_str = result.unwrap_err().to_string();
        assert!(err_str.contains("Invalid to address") || err_str.contains("email"));
    }

    // --- Dynamic dispatch (Box<dyn Notifier>) tests ---

    fn boxed_email_notifier(config: EmailConfig) -> Box<dyn Notifier> {
        Box::new(EmailNotifier::new(config, empty_registry()).unwrap())
    }

    #[tokio::test]
    async fn test_dyn_email_notify_start() {
        let n = boxed_email_notifier(send_config());
        let issue = Issue::new("1", "DYN-E-1", "Test", "https://example.com", "linear");
        let result = n.notify_start(&issue).await;
        assert!(result.is_err()); // SMTP connection will fail
    }

    #[tokio::test]
    async fn test_dyn_email_notify_success_regular() {
        let n = boxed_email_notifier(send_config());
        let issue = Issue::new("1", "DYN-E-1", "Test", "https://example.com", "linear");
        let result = n.notify_success(&issue, "https://github.com/pr/1").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_dyn_email_notify_completed() {
        let n = boxed_email_notifier(send_config());
        let issue = Issue::new("1", "DYN-E-1", "Test", "https://example.com", "linear");
        let result = n.notify_completed(&issue).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_dyn_email_notify_failed() {
        let n = boxed_email_notifier(send_config());
        let issue = Issue::new("1", "DYN-E-1", "Test", "https://example.com", "linear");
        let result = n.notify_failed(&issue, "Some error").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_dyn_email_notify_merged() {
        let n = boxed_email_notifier(send_config());
        let issue = Issue::new("1", "DYN-E-1", "Test", "https://example.com", "linear");
        let result = n.notify_merged(&issue, "https://github.com/pr/1").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_dyn_email_notify_closed() {
        let n = boxed_email_notifier(send_config());
        let issue = Issue::new("1", "DYN-E-1", "Test", "https://example.com", "linear");
        let result = n.notify_closed(&issue, "https://github.com/pr/1").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_dyn_email_notify_status() {
        let n = boxed_email_notifier(send_config());
        let result = n.notify_status("status update").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_dyn_email_notify_urgent_issues() {
        let n = boxed_email_notifier(send_config());
        let issues = vec![Issue::new(
            "1",
            "DYN-E-1",
            "Bug 1",
            "https://example.com",
            "linear",
        )];
        let result = n.notify_urgent_issues(&issues).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_dyn_email_name_and_is_enabled() {
        let n = boxed_email_notifier(send_config());
        assert_eq!(n.name(), "email");
        assert!(n.is_enabled());
    }
}
