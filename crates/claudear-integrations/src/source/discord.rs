//! Discord message source adapter.
//!
//! Polls a Discord channel for human messages and converts them into issues.

use super::IssueSource;
use crate::discord::{DiscordClient, DiscordMessage};
use async_trait::async_trait;
use claudear_config::config::DiscordConfig;
use claudear_core::error::{Error, Result};
use claudear_core::types::{Issue, MatchPriority, MatchResult};
use std::sync::RwLock;

/// Discord channel polling source that converts messages into issues.
pub struct DiscordSource {
    config: DiscordConfig,
    /// Last seen message ID for incremental polling. `None` means first poll (seed).
    last_seen_id: RwLock<Option<String>>,
    /// Reusable Discord API client, created once during construction.
    /// `None` when bot_token is not configured.
    client: Option<DiscordClient>,
}

impl DiscordSource {
    /// Create a new Discord source from config.
    pub fn new(config: DiscordConfig) -> Self {
        let client = config
            .bot_token
            .as_ref()
            .map(|s| s.expose())
            .and_then(|token| DiscordClient::new(token).ok());
        Self {
            config,
            last_seen_id: RwLock::new(None),
            client,
        }
    }

    /// Get the channel ID to listen on (listen_channel_id or fallback to channel_id).
    fn listen_channel_id(&self) -> Option<&str> {
        self.config
            .listen_channel_id
            .as_deref()
            .or(self.config.channel_id.as_deref())
    }

    /// Check if a message is from a bot (excludes webhook messages, which
    /// appear as bot-authored but are user-triggered via webhook URLs).
    fn is_bot_message(msg: &DiscordMessage) -> bool {
        if msg.webhook_id.is_some() {
            return false;
        }
        msg.author.as_ref().is_some_and(|a| a.bot)
    }

    /// Check if a message is one of our own notifications (e.g. ask questions,
    /// success/failure alerts). These are sent via webhook and should not be
    /// treated as new issues.
    fn is_own_notification(msg: &DiscordMessage) -> bool {
        // Ask question messages have an embed with "Input needed:" title.
        if msg.embeds.iter().any(|e| {
            e.title
                .as_ref()
                .is_some_and(|t| t.contains("Input needed:"))
        }) {
            return true;
        }
        // Webhook messages with Claudear embeds are our notifications.
        // Their text content is either empty or only user mentions (<@123>).
        if msg.webhook_id.is_some() {
            let stripped = Self::strip_mentions(&msg.content);
            if stripped.trim().is_empty() {
                return true;
            }
        }
        false
    }

    fn normalize_id(id: &str) -> &str {
        id.trim_start_matches(['&', '!', '@'])
    }

    fn mentions_user(msg: &DiscordMessage, user_id: &str) -> bool {
        let user_id = Self::normalize_id(user_id);
        if user_id.is_empty() {
            return false;
        }
        if msg.mentions.iter().any(|u| u.id == user_id) {
            return true;
        }
        msg.content.contains(&format!("<@{}>", user_id))
            || msg.content.contains(&format!("<@!{}>", user_id))
    }

    fn mentions_role(msg: &DiscordMessage, role_id: &str) -> bool {
        let role_id = Self::normalize_id(role_id);
        if role_id.is_empty() {
            return false;
        }
        msg.content.contains(&format!("<@&{}>", role_id))
    }

    fn passes_mention_gate(&self, msg: &DiscordMessage) -> bool {
        let user_gate = self.config.bot_id.as_deref().filter(|s| !s.is_empty());
        let role_gate = self.config.bot_role_id.as_deref().filter(|s| !s.is_empty());
        match (user_gate, role_gate) {
            // No gate configured: ingest everything.
            (None, None) => true,
            (user_id, role_id) => {
                user_id.is_some_and(|id| Self::mentions_user(msg, id))
                    || role_id.is_some_and(|id| Self::mentions_role(msg, id))
            }
        }
    }

    /// Strip Discord user mentions (<@123>, <@!123>) from text.
    fn strip_mentions(content: &str) -> String {
        let mut result = content.to_string();
        while let Some(start) = result.find("<@") {
            if let Some(end) = result[start..].find('>') {
                result.replace_range(start..start + end + 1, "");
            } else {
                break;
            }
        }
        result
    }

    /// Extract a title from message content (first line, max 100 chars).
    fn extract_title(content: &str) -> String {
        let first_line = content.lines().next().unwrap_or(content);
        if first_line.len() > 100 {
            format!("{}...", &first_line[..first_line.floor_char_boundary(97)])
        } else {
            first_line.to_string()
        }
    }

    /// Build a Discord message URL.
    fn message_url(&self, channel_id: &str, message_id: &str) -> String {
        match &self.config.guild_id {
            Some(guild_id) => format!(
                "https://discord.com/channels/{}/{}/{}",
                guild_id, channel_id, message_id
            ),
            None => format!(
                "https://discord.com/channels/@me/{}/{}",
                channel_id, message_id
            ),
        }
    }

    /// Convert a Discord message to an Issue.
    fn message_to_issue(&self, msg: &DiscordMessage) -> Issue {
        let short_id = format!("DISCORD-{}", msg.id.chars().take(8).collect::<String>());
        let title = Self::extract_title(&msg.content);
        let url = self.message_url(&msg.channel_id, &msg.id);

        let mut issue = Issue::new(&msg.id, &short_id, &title, &url, "discord");
        issue.description = Some(msg.content.clone());

        if let Some(ref author) = msg.author {
            issue.set_metadata("author_username", &author.username);
            issue.set_metadata("author_id", &author.id);
        }
        issue.set_metadata("channel_id", &msg.channel_id);

        // When this message is a reply, record the parent it points at so the
        // engine can reconstruct the reply-chain conversation as grounding
        // context. Only the parent id is available on the payload; the parent's
        // content is resolved later (from our DB for Claudear's own answers, or
        // by fetching from Discord for other users' messages).
        if let Some(ref reference) = msg.message_reference {
            if let Some(ref parent_id) = reference.message_id {
                issue.set_metadata("reply_to_message_id", parent_id);
                let parent_channel = reference
                    .channel_id
                    .clone()
                    .unwrap_or_else(|| msg.channel_id.clone());
                issue.set_metadata("reply_to_channel_id", &parent_channel);
            }
        }

        issue
    }
}

#[async_trait]
impl IssueSource for DiscordSource {
    fn name(&self) -> &str {
        "discord"
    }

    fn display_name(&self) -> &str {
        "Discord Messages"
    }

    async fn fetch_issues(&self) -> Result<Vec<Issue>> {
        let client = self
            .client
            .as_ref()
            .ok_or_else(|| Error::config("Discord bot_token is required for source polling"))?;

        let channel_id = self
            .listen_channel_id()
            .ok_or_else(|| {
                Error::config(
                    "Discord listen_channel_id or channel_id is required for source polling",
                )
            })?
            .to_string();

        let last_seen = self
            .last_seen_id
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();

        match last_seen {
            None => {
                // First poll: seed the cursor with the latest message, return no issues
                let messages = client.list_channel_messages(&channel_id, 1).await?;
                if let Some(latest) = messages.first() {
                    let mut lock = self.last_seen_id.write().unwrap_or_else(|e| e.into_inner());
                    *lock = Some(latest.id.clone());
                    tracing::info!(message_id = %latest.id, "Discord source seeded cursor");
                }
                Ok(vec![])
            }
            Some(after_id) => {
                let messages = client
                    .list_channel_messages_after(&channel_id, &after_id, 100)
                    .await?;

                if messages.is_empty() {
                    return Ok(vec![]);
                }

                // Update cursor to the latest message (before filtering, so the
                // cursor always advances even when every message is filtered out).
                if let Some(latest) = messages.last() {
                    let mut lock = self.last_seen_id.write().unwrap_or_else(|e| e.into_inner());
                    *lock = Some(latest.id.clone());
                }

                let debug = self.config.debug_logging;

                if !debug {
                    let issues: Vec<Issue> = messages
                        .iter()
                        .filter(|msg| !Self::is_bot_message(msg))
                        .filter(|msg| !Self::is_own_notification(msg))
                        .filter(|msg| !msg.content.trim().is_empty())
                        .filter(|msg| self.passes_mention_gate(msg))
                        .map(|msg| self.message_to_issue(msg))
                        .collect();
                    return Ok(issues);
                }

                // log during debug mode
                let mut issues = Vec::new();
                for msg in &messages {
                    let author = msg
                        .author
                        .as_ref()
                        .map(|a| a.username.as_str())
                        .unwrap_or("<unknown>");

                    if Self::is_bot_message(msg) {
                        tracing::info!(id = %msg.id, author, "Discord message ignored: authored by a bot");
                        continue;
                    }
                    if Self::is_own_notification(msg) {
                        tracing::info!(id = %msg.id, author, "Discord message ignored: our own notification");
                        continue;
                    }
                    if msg.content.trim().is_empty() {
                        tracing::info!(id = %msg.id, author, "Discord message ignored: empty content");
                        continue;
                    }
                    if !self.passes_mention_gate(msg) {
                        tracing::info!(
                            id = %msg.id,
                            author,
                            mentioned_users = ?msg.mentions.iter().map(|u| u.id.as_str()).collect::<Vec<_>>(),
                            bot_id = ?self.config.bot_id,
                            bot_role_id = ?self.config.bot_role_id,
                            "Discord message ignored: bot not mentioned (user or role)"
                        );
                        continue;
                    }
                    tracing::info!(
                        id = %msg.id,
                        author,
                        mentioned_users = ?msg.mentions.iter().map(|u| u.id.as_str()).collect::<Vec<_>>(),
                        "Discord message accepted: bot mentioned"
                    );
                    issues.push(self.message_to_issue(msg));
                }

                Ok(issues)
            }
        }
    }

    fn matches_criteria(&self, _issue: &Issue) -> MatchResult {
        // All Discord messages that pass filtering are valid issues
        MatchResult::matched("discord_message", MatchPriority::Normal)
    }

    async fn build_issue_context(&self, issue: &Issue) -> Result<String> {
        let mut context = format!("Discord Message Issue: {}\n", issue.title);

        if let Some(ref desc) = issue.description {
            context.push_str(&format!("\nMessage:\n{}\n", desc));
        }

        if let Some(author) = issue.get_metadata::<String>("author_username") {
            context.push_str(&format!("\nAuthor: {}\n", author));
        }

        context.push_str(&format!("\nURL: {}\n", issue.url));

        Ok(context)
    }

    async fn create_issue(
        &self,
        title: &str,
        description: &str,
        _labels: &[String],
    ) -> Result<Issue> {
        let channel_id = self
            .listen_channel_id()
            .ok_or_else(|| Error::config("Discord channel_id is required to create an issue"))?
            .to_string();

        let content = if description.is_empty() {
            title.to_string()
        } else {
            format!("{}\n\n{}", title, description)
        };

        // Prefer webhook URL: messages posted via webhook have webhook_id set,
        // which bypasses the is_bot_message filter. This makes them appear as
        // user-posted messages to the daemon's poll_issues.
        if let Some(ref webhook_url) = self.config.webhook_url {
            let url = format!("{}?wait=true", webhook_url.expose());
            let http = reqwest::Client::new();
            let resp = http
                .post(&url)
                .json(&serde_json::json!({ "content": content }))
                .send()
                .await
                .map_err(|e| Error::Other(format!("Failed to post Discord webhook: {}", e)))?;

            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                return Err(Error::Other(format!(
                    "Discord webhook returned {}: {}",
                    status, body
                )));
            }

            let msg: crate::discord::DiscordMessage = resp.json().await.map_err(|e| {
                Error::Other(format!("Failed to parse Discord webhook response: {}", e))
            })?;

            return Ok(self.message_to_issue(&msg));
        }

        // Fallback: use bot token to send message directly.
        let client = self.client.as_ref().ok_or_else(|| {
            Error::config("Discord bot_token or webhook_url is required to create an issue")
        })?;

        let params = crate::discord::CreateMessageParams::text(content);
        let msg = client
            .send_message(&channel_id, params)
            .await
            .map_err(|e| Error::Other(format!("Failed to create Discord issue: {}", e)))?;

        Ok(self.message_to_issue(&msg))
    }

    async fn get_issue(&self, issue_id: &str) -> Result<Issue> {
        let client = self
            .client
            .as_ref()
            .ok_or_else(|| Error::config("Discord bot_token is required to fetch an issue"))?;

        let channel_id = self
            .listen_channel_id()
            .ok_or_else(|| Error::config("Discord channel_id is required to fetch an issue"))?
            .to_string();

        let msg = client
            .get_message(&channel_id, issue_id)
            .await
            .map_err(|_| Error::issue_not_found("discord", issue_id))?;

        Ok(self.message_to_issue(&msg))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discord::DiscordUser;

    fn make_config() -> DiscordConfig {
        DiscordConfig {
            bot_token: Some("test-token".into()),
            channel_id: Some("chan-123".to_string()),
            source_enabled: true,
            listen_channel_id: None,
            guild_id: Some("guild-456".to_string()),
            ..Default::default()
        }
    }

    fn make_message(id: &str, content: &str, bot: bool) -> DiscordMessage {
        DiscordMessage {
            id: id.to_string(),
            channel_id: "chan-123".to_string(),
            author: Some(DiscordUser {
                id: "user-1".to_string(),
                username: "testuser".to_string(),
                discriminator: "0".to_string(),
                avatar: None,
                bot,
            }),
            content: content.to_string(),
            timestamp: "2024-01-01T00:00:00Z".to_string(),
            message_reference: None,
            thread: None,
            webhook_id: None,
            embeds: vec![],
            mentions: vec![],
        }
    }

    #[test]
    fn test_extract_title_short() {
        assert_eq!(DiscordSource::extract_title("Short title"), "Short title");
    }

    #[test]
    fn test_extract_title_multiline() {
        assert_eq!(
            DiscordSource::extract_title("First line\nSecond line\nThird"),
            "First line"
        );
    }

    #[test]
    fn test_extract_title_long() {
        let long = "a".repeat(150);
        let title = DiscordSource::extract_title(&long);
        assert_eq!(title.len(), 100);
        assert!(title.ends_with("..."));
    }

    #[test]
    fn test_is_bot_message() {
        let bot_msg = make_message("1", "hello", true);
        let human_msg = make_message("2", "hello", false);
        let no_author = DiscordMessage {
            id: "3".to_string(),
            channel_id: "c".to_string(),
            author: None,
            content: "hello".to_string(),
            timestamp: "2024-01-01T00:00:00Z".to_string(),
            message_reference: None,
            thread: None,
            webhook_id: None,
            embeds: vec![],
            mentions: vec![],
        };
        // Webhook messages have author.bot=true but should NOT be filtered
        let mut webhook_msg = make_message("4", "hello", true);
        webhook_msg.webhook_id = Some("wh-123".to_string());

        assert!(DiscordSource::is_bot_message(&bot_msg));
        assert!(!DiscordSource::is_bot_message(&human_msg));
        assert!(!DiscordSource::is_bot_message(&no_author));
        assert!(!DiscordSource::is_bot_message(&webhook_msg));
    }

    #[test]
    fn test_message_url_with_guild() {
        let source = DiscordSource::new(make_config());
        let url = source.message_url("chan-123", "msg-789");
        assert_eq!(
            url,
            "https://discord.com/channels/guild-456/chan-123/msg-789"
        );
    }

    #[test]
    fn test_message_url_without_guild() {
        let mut config = make_config();
        config.guild_id = None;
        let source = DiscordSource::new(config);
        let url = source.message_url("chan-123", "msg-789");
        assert_eq!(url, "https://discord.com/channels/@me/chan-123/msg-789");
    }

    #[test]
    fn test_message_to_issue() {
        let source = DiscordSource::new(make_config());
        let msg = make_message("123456789", "Fix the login bug\nMore details here", false);
        let issue = source.message_to_issue(&msg);

        assert_eq!(issue.id, "123456789");
        assert_eq!(issue.short_id, "DISCORD-12345678");
        assert_eq!(issue.title, "Fix the login bug");
        assert_eq!(issue.source, "discord");
        assert_eq!(
            issue.description.as_deref(),
            Some("Fix the login bug\nMore details here")
        );
        assert_eq!(
            issue.get_metadata::<String>("author_username"),
            Some("testuser".to_string())
        );
        assert_eq!(
            issue.get_metadata::<String>("author_id"),
            Some("user-1".to_string())
        );
        assert_eq!(
            issue.get_metadata::<String>("channel_id"),
            Some("chan-123".to_string())
        );
    }

    #[test]
    fn test_message_to_issue_captures_reply_pointer() {
        let source = DiscordSource::new(make_config());
        let mut msg = make_message("999", "what about Y?", false);
        msg.message_reference = Some(crate::discord::DiscordMessageReference {
            message_id: Some("parent-42".to_string()),
            channel_id: Some("thread-7".to_string()),
            guild_id: None,
            fail_if_not_exists: None,
        });
        let issue = source.message_to_issue(&msg);
        assert_eq!(
            issue.get_metadata::<String>("reply_to_message_id"),
            Some("parent-42".to_string())
        );
        assert_eq!(
            issue.get_metadata::<String>("reply_to_channel_id"),
            Some("thread-7".to_string())
        );
    }

    #[test]
    fn test_message_to_issue_reply_channel_falls_back_to_message_channel() {
        let source = DiscordSource::new(make_config());
        let mut msg = make_message("999", "follow up", false);
        msg.message_reference = Some(crate::discord::DiscordMessageReference {
            message_id: Some("parent-42".to_string()),
            channel_id: None,
            guild_id: None,
            fail_if_not_exists: None,
        });
        let issue = source.message_to_issue(&msg);
        // No reference channel: falls back to the message's own channel.
        assert_eq!(
            issue.get_metadata::<String>("reply_to_channel_id"),
            Some("chan-123".to_string())
        );
    }

    #[test]
    fn test_message_to_issue_no_reply_pointer_when_not_reply() {
        let source = DiscordSource::new(make_config());
        let msg = make_message("999", "not a reply", false);
        let issue = source.message_to_issue(&msg);
        assert!(issue
            .get_metadata::<String>("reply_to_message_id")
            .is_none());
    }

    #[test]
    fn test_listen_channel_id_fallback() {
        let source = DiscordSource::new(make_config());
        assert_eq!(source.listen_channel_id(), Some("chan-123"));
    }

    #[test]
    fn test_listen_channel_id_explicit() {
        let mut config = make_config();
        config.listen_channel_id = Some("listen-789".to_string());
        let source = DiscordSource::new(config);
        assert_eq!(source.listen_channel_id(), Some("listen-789"));
    }

    #[test]
    fn test_name_and_display_name() {
        let source = DiscordSource::new(make_config());
        assert_eq!(source.name(), "discord");
        assert_eq!(source.display_name(), "Discord Messages");
    }

    #[test]
    fn test_matches_criteria() {
        let source = DiscordSource::new(make_config());
        let issue = Issue::new("1", "D-1", "Test", "http://test.com", "discord");
        let result = source.matches_criteria(&issue);
        assert!(result.matches);
    }

    // --- Bot-mention gate ---

    #[test]
    fn test_mentions_user_via_structured_mentions() {
        // The authoritative path: Discord's parsed `mentions` array contains
        // the bot, even if the raw content has no `<@id>` token.
        let mut msg = make_message("1", "how do I paginate?", false);
        msg.mentions = vec![DiscordUser {
            id: "999".to_string(),
            username: "claudear".to_string(),
            discriminator: "0".to_string(),
            avatar: None,
            bot: true,
        }];
        assert!(DiscordSource::mentions_user(&msg, "999"));
        // A different user mentioned is not the bot.
        assert!(!DiscordSource::mentions_user(&msg, "111"));
    }

    #[test]
    fn test_mentions_user_content_fallback() {
        // Fallback when `mentions` wasn't populated: scan raw content for the
        // mention token in both forms.
        let plain = make_message("1", "hey <@999> how do I paginate?", false);
        let nick = make_message("2", "<@!999> help", false);
        let other = make_message("3", "hey <@111> look", false);
        let none = make_message("4", "how do I paginate?", false);

        assert!(DiscordSource::mentions_user(&plain, "999"));
        assert!(DiscordSource::mentions_user(&nick, "999"));
        assert!(!DiscordSource::mentions_user(&other, "999"));
        assert!(!DiscordSource::mentions_user(&none, "999"));
        // Empty user id never matches.
        assert!(!DiscordSource::mentions_user(&plain, ""));
        // A role mention (<@&999>) must NOT count as a user mention.
        let role = make_message("5", "<@&999> hello", false);
        assert!(!DiscordSource::mentions_user(&role, "999"));
    }

    #[test]
    fn test_mentions_role() {
        // Role mentions live only in the raw content as `<@&ID>`.
        let role = make_message("1", "<@&777> what adapters does vcs support?", false);
        let user = make_message("2", "<@777> hi", false);
        let none = make_message("3", "no mention", false);

        assert!(DiscordSource::mentions_role(&role, "777"));
        // A leading `&` sigil pasted from `<@&ID>` is tolerated.
        assert!(DiscordSource::mentions_role(&role, "&777"));
        // A user mention must NOT count as a role mention.
        assert!(!DiscordSource::mentions_role(&user, "777"));
        assert!(!DiscordSource::mentions_role(&none, "777"));
        assert!(!DiscordSource::mentions_role(&role, ""));
    }

    #[test]
    fn test_passes_mention_gate_accepts_user_or_role() {
        // Both a user id and a role id are configured: either mention passes.
        let mut config = make_config();
        config.bot_id = Some("999".to_string());
        config.bot_role_id = Some("777".to_string());
        let source = DiscordSource::new(config);

        let user_tag = make_message("1", "<@999> what is query not equal syntax?", false);
        let role_tag = make_message("2", "<@&777> what is query not equal syntax?", false);
        let untagged = make_message("3", "what is query not equal syntax?", false);
        // Mentioning the role's number as a *user* must not pass.
        let wrong_form = make_message("4", "<@777> hi", false);

        assert!(source.passes_mention_gate(&user_tag));
        assert!(source.passes_mention_gate(&role_tag));
        assert!(!source.passes_mention_gate(&untagged));
        assert!(!source.passes_mention_gate(&wrong_form));
    }

    #[test]
    fn test_passes_mention_gate_role_only() {
        // Only a role id configured (no bot_id): role mention passes, user
        // mention of that number does not.
        let mut config = make_config();
        config.bot_id = None;
        config.bot_role_id = Some("777".to_string());
        let source = DiscordSource::new(config);

        assert!(source.passes_mention_gate(&make_message("1", "<@&777> help", false)));
        assert!(!source.passes_mention_gate(&make_message("2", "<@777> help", false)));
        assert!(!source.passes_mention_gate(&make_message("3", "no tag", false)));
    }

    #[test]
    fn test_passes_mention_gate_requires_tag_when_bot_id_set() {
        let mut config = make_config();
        config.bot_id = Some("999".to_string());
        let source = DiscordSource::new(config);

        let tagged = make_message("1", "<@999> what is query not equal syntax?", false);
        let untagged = make_message("2", "what is query not equal syntax?", false);

        assert!(source.passes_mention_gate(&tagged));
        assert!(!source.passes_mention_gate(&untagged));
    }

    #[test]
    fn test_passes_mention_gate_allows_all_when_unset() {
        // Default config has neither bot_id nor bot_role_id → legacy
        // "reply to all" behaviour.
        let source = DiscordSource::new(make_config());
        let untagged = make_message("1", "no mention here", false);
        assert!(source.passes_mention_gate(&untagged));
    }

    #[tokio::test]
    async fn test_build_issue_context() {
        let source = DiscordSource::new(make_config());
        let mut issue = Issue::new(
            "1",
            "DISCORD-1",
            "Fix login",
            "https://discord.com/channels/g/c/1",
            "discord",
        );
        issue.description = Some("Fix the login bug please".to_string());
        issue.set_metadata("author_username", "alice");

        let context = source.build_issue_context(&issue).await.unwrap();
        assert!(context.contains("Fix login"));
        assert!(context.contains("Fix the login bug please"));
        assert!(context.contains("alice"));
        assert!(context.contains("https://discord.com/channels/g/c/1"));
    }

    #[tokio::test]
    async fn test_get_issue_no_client_returns_error() {
        let mut config = make_config();
        config.bot_token = None;
        let source = DiscordSource::new(config);
        let result = source.get_issue("123").await;
        assert!(result.is_err());
    }

    #[test]
    fn test_seed_behavior_initial_state() {
        let source = DiscordSource::new(make_config());
        assert!(source.last_seen_id.read().unwrap().is_none());
    }
}
