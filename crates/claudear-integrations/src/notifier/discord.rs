//! Discord webhook notifier.

use super::Notifier;
use crate::ask_reply_inbox;
use crate::discord::{CreateMessageParams, DiscordClient, DiscordMessageReference, MessageEmbed};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use claudear_config::config::DiscordConfig;
use claudear_config::users::UserRegistry;
use claudear_core::error::{Error, Result};
use claudear_core::http::HttpResponse;
use claudear_core::types::{AskDelivery, AskReply, AskRequest, Issue};
use serde::Serialize;

/// Trait for HTTP client used by Discord notifier.
#[async_trait]
pub trait DiscordWebhookClient: Send + Sync {
    async fn post_json(&self, url: &str, body: &serde_json::Value) -> Result<HttpResponse>;
}

/// Real HTTP client using reqwest.
pub struct ReqwestDiscordWebhookClient {
    client: reqwest::Client,
}

impl ReqwestDiscordWebhookClient {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .connect_timeout(std::time::Duration::from_secs(10))
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
        }
    }
}

impl Default for ReqwestDiscordWebhookClient {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl DiscordWebhookClient for ReqwestDiscordWebhookClient {
    async fn post_json(&self, url: &str, body: &serde_json::Value) -> Result<HttpResponse> {
        let response = self.client.post(url).json(body).send().await?;

        let status = response.status().as_u16();
        let body = response.text().await.unwrap_or_default();

        Ok(HttpResponse { status, body })
    }
}

/// Metadata returned after successfully sending a Discord message.
struct SentMessageInfo {
    message_id: String,
    channel_id: String,
}

/// Discord webhook notifier.
pub struct DiscordNotifier<H: DiscordWebhookClient = ReqwestDiscordWebhookClient> {
    config: DiscordConfig,
    http: H,
    user_registry: UserRegistry,
    /// Reusable Discord bot API client, created once during construction.
    bot_client: Option<DiscordClient>,
}

#[derive(Debug, Serialize)]
pub(crate) struct DiscordMessage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) embeds: Option<Vec<DiscordEmbed>>,
}

#[derive(Debug, Serialize)]
pub(crate) struct DiscordEmbed {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) color: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) fields: Option<Vec<DiscordField>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) footer: Option<DiscordFooter>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) timestamp: Option<String>,
}

#[derive(Debug, Serialize)]
pub(crate) struct DiscordField {
    pub(crate) name: String,
    pub(crate) value: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) inline: Option<bool>,
}

#[derive(Debug, Serialize)]
pub(crate) struct DiscordFooter {
    pub(crate) text: String,
}

impl DiscordNotifier<ReqwestDiscordWebhookClient> {
    pub fn new(config: DiscordConfig, user_registry: UserRegistry) -> Self {
        let bot_client = config
            .bot_token
            .as_ref()
            .map(|s| s.expose())
            .filter(|t| !t.is_empty())
            .and_then(|token| DiscordClient::new(token).ok());
        Self {
            config,
            http: ReqwestDiscordWebhookClient::new(),
            user_registry,
            bot_client,
        }
    }
}

/// Maximum lengths for user-controlled fields to prevent unbounded memory allocation.
const MAX_SHORT_ID_LENGTH: usize = 64;
const MAX_SOURCE_LENGTH: usize = 32;
const MAX_URL_LENGTH: usize = 2000;
const MAX_DESCRIPTION_LENGTH: usize = 2048;

/// Truncate a string to the specified maximum length, adding "..." if truncated.
fn truncate_string(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else if max_len > 3 {
        format!("{}...", &s[..s.floor_char_boundary(max_len - 3)])
    } else {
        s[..s.floor_char_boundary(max_len)].to_string()
    }
}

impl<H: DiscordWebhookClient> DiscordNotifier<H> {
    /// Create a new Discord notifier with a custom HTTP client.
    pub fn with_http_client(config: DiscordConfig, http: H) -> Self {
        let bot_client = config
            .bot_token
            .as_ref()
            .map(|s| s.expose())
            .filter(|t| !t.is_empty())
            .and_then(|token| DiscordClient::new(token).ok());
        Self {
            config,
            http,
            user_registry: UserRegistry::new(std::collections::HashMap::new()),
            bot_client,
        }
    }

    /// Create a new Discord notifier with a custom HTTP client and user registry.
    pub fn with_http_client_and_registry(
        config: DiscordConfig,
        http: H,
        user_registry: UserRegistry,
    ) -> Self {
        let bot_client = config
            .bot_token
            .as_ref()
            .map(|s| s.expose())
            .filter(|t| !t.is_empty())
            .and_then(|token| DiscordClient::new(token).ok());
        Self {
            config,
            http,
            user_registry,
            bot_client,
        }
    }

    async fn send(&self, message: DiscordMessage) -> Result<Option<SentMessageInfo>> {
        if let Some(ref webhook_url) = self.config.webhook_url {
            let body = serde_json::to_value(&message)?;
            let url_with_wait = if webhook_url.expose().contains('?') {
                format!("{}&wait=true", webhook_url.expose())
            } else {
                format!("{}?wait=true", webhook_url.expose())
            };
            let response = self.http.post_json(&url_with_wait, &body).await?;

            if response.status < 200 || response.status >= 300 {
                return Err(Error::notifier(
                    "discord",
                    format!("Webhook error: {}", response.body),
                ));
            }

            let info = serde_json::from_str::<serde_json::Value>(&response.body)
                .ok()
                .and_then(|v| {
                    let id = v.get("id")?.as_str()?.to_string();
                    let channel_id = v.get("channel_id")?.as_str()?.to_string();
                    Some(SentMessageInfo {
                        message_id: id,
                        channel_id,
                    })
                });
            return Ok(info);
        }

        if let Some(ref client) = self.bot_client {
            if let Some(ref channel_id) = self.config.channel_id {
                if !channel_id.is_empty() {
                    let params = Self::to_create_message_params(&message);
                    let sent = client.send_message(channel_id, params).await?;
                    return Ok(Some(SentMessageInfo {
                        message_id: sent.id,
                        channel_id: sent.channel_id,
                    }));
                }
            }
        }

        Ok(None)
    }

    /// Send a message, preferring the channel the issue originated from.
    ///
    /// When a bot client is available and the issue carries an originating
    /// `channel_id` (e.g. a Discord question), the message is posted there.
    /// When `reply_to_origin` is true, it is sent as a native Discord reply to
    /// the originating message (`issue.id`), so the answer is threaded under the
    /// question. Replies require the bot path; the webhook fallback ignores the
    /// reply and posts a standalone message.
    async fn send_to_issue_channel(
        &self,
        issue: &Issue,
        message: DiscordMessage,
        reply_to_origin: bool,
    ) -> Result<Option<SentMessageInfo>> {
        let origin_channel = issue
            .get_metadata::<String>("channel_id")
            .filter(|c| !c.is_empty());
        if let (Some(client), Some(channel_id)) = (self.bot_client.as_ref(), origin_channel) {
            let mut params = Self::to_create_message_params(&message);
            if reply_to_origin {
                // Reply to the original Discord message so the answer is
                // visibly threaded to the question it addresses.
                params.message_reference = origin_reply_reference(issue, &channel_id);
            }
            let sent = client.send_message(&channel_id, params).await?;
            return Ok(Some(SentMessageInfo {
                message_id: sent.id,
                channel_id: sent.channel_id,
            }));
        }
        self.send(message).await
    }

    fn has_bot_channel(&self) -> bool {
        self.config
            .bot_token
            .as_ref()
            .map(|v| !v.expose().is_empty())
            .unwrap_or(false)
            && self
                .config
                .channel_id
                .as_ref()
                .map(|v| !v.is_empty())
                .unwrap_or(false)
    }

    /// Returns true when at least one concrete delivery path exists:
    /// a non-empty webhook URL, or a successfully constructed bot client
    /// paired with a non-empty channel ID.
    fn has_delivery_path(&self) -> bool {
        let has_webhook = self
            .config
            .webhook_url
            .as_ref()
            .map(|v| !v.expose().is_empty())
            .unwrap_or(false);

        let has_bot = self.bot_client.is_some()
            && self
                .config
                .channel_id
                .as_ref()
                .map(|v| !v.is_empty())
                .unwrap_or(false);

        has_webhook || has_bot
    }

    fn to_create_message_params(msg: &DiscordMessage) -> CreateMessageParams {
        let content = msg.content.clone().unwrap_or_default();
        let embeds = msg.embeds.as_ref().map(|embeds| {
            embeds
                .iter()
                .map(|e| {
                    let mut embed = MessageEmbed::new();
                    if let Some(ref title) = e.title {
                        embed = embed.title(title.clone());
                    }
                    if let Some(ref description) = e.description {
                        embed = embed.description(description.clone());
                    }
                    if let Some(ref url) = e.url {
                        embed = embed.url(url.clone());
                    }
                    if let Some(color) = e.color {
                        embed = embed.color(color);
                    }
                    if let Some(ref fields) = e.fields {
                        for f in fields {
                            embed = embed.field(&f.name, &f.value, f.inline.unwrap_or(false));
                        }
                    }
                    if let Some(ref footer) = e.footer {
                        embed = embed.footer(&footer.text);
                    }
                    if let Some(ref ts) = e.timestamp {
                        embed = embed.timestamp(ts.clone());
                    }
                    embed
                })
                .collect()
        });

        CreateMessageParams {
            content: if content.len() > 2000 {
                format!("{}...", &content[..content.floor_char_boundary(1997)])
            } else {
                content
            },
            tts: None,
            embeds,
            message_reference: None,
        }
    }

    fn get_user_mention(&self) -> Option<String> {
        self.config.user_id.as_ref().map(|id| format!("<@{}>", id))
    }

    fn get_user_mention_for_issue(&self, issue: &Issue) -> Option<String> {
        // Check for resolved user first
        if let Some(slug) = issue.get_metadata::<String>("resolved_user") {
            if let Some(user) = self.user_registry.get_by_slug(&slug) {
                if let Some(ref discord_id) = user.discord_id {
                    return Some(format!("<@{}>", discord_id));
                }
            }
        }
        // Fall back to global config
        self.config.user_id.as_ref().map(|id| format!("<@{}>", id))
    }

    fn get_target_discord_id_for_issue(&self, issue: &Issue) -> Option<String> {
        if let Some(slug) = issue.get_metadata::<String>("resolved_user") {
            if let Some(user) = self.user_registry.get_by_slug(&slug) {
                if let Some(ref discord_id) = user.discord_id {
                    return Some(discord_id.clone());
                }
            }
        }
        self.config.user_id.clone()
    }

    fn extract_reply_text(content: &str) -> Option<String> {
        let answer = content.trim();
        if answer.is_empty() {
            None
        } else {
            Some(answer.to_string())
        }
    }
}

// Re-export the shared emoji function for backward compatibility within this module.
pub(crate) use super::get_source_emoji;

/// Return the current UTC timestamp in RFC 3339 format.
pub(crate) fn timestamp() -> String {
    chrono::Utc::now().to_rfc3339()
}

/// Build the Discord message for a "processing started" notification.
pub(crate) fn build_start_message(issue: &Issue, mention: Option<String>) -> DiscordMessage {
    let emoji = get_source_emoji(&issue.source);
    let short_id = truncate_string(&issue.short_id, MAX_SHORT_ID_LENGTH);
    let title = truncate_string(&issue.title, MAX_DESCRIPTION_LENGTH);
    let url = truncate_string(&issue.url, MAX_URL_LENGTH);
    let source = truncate_string(&issue.source, MAX_SOURCE_LENGTH);

    let mut fields = vec![
        DiscordField {
            name: "Source".to_string(),
            value: source,
            inline: Some(true),
        },
        DiscordField {
            name: "Priority".to_string(),
            value: issue.priority.to_string(),
            inline: Some(true),
        },
        DiscordField {
            name: "Status".to_string(),
            value: issue.status.to_string(),
            inline: Some(true),
        },
    ];
    if let Some(reason) = issue.get_metadata::<String>("trigger_reason") {
        fields.push(DiscordField {
            name: "Trigger".to_string(),
            value: reason,
            inline: Some(false),
        });
    }

    DiscordMessage {
        content: mention.map(|m| m.to_string()),
        embeds: Some(vec![DiscordEmbed {
            title: Some(format!("{} Processing: {}", emoji, short_id)),
            description: Some(title),
            url: Some(url),
            color: Some(0x3498db), // Blue
            fields: Some(fields),
            footer: Some(DiscordFooter {
                text: "Claudear".to_string(),
            }),
            timestamp: Some(timestamp()),
        }]),
    }
}

/// Maximum number of message chunks to send for a single answer.
const MAX_ANSWER_CHUNKS: usize = 5;

/// Split `text` into UTF-8-safe chunks of at most `max_len` bytes, preferring to
/// break on newline boundaries. At most `max_chunks` are produced; if content
/// remains beyond the cap, a truncation marker is appended to the last chunk.
pub(crate) fn chunk_text(text: &str, max_len: usize, max_chunks: usize) -> Vec<String> {
    let text = text.trim();
    if text.is_empty() || max_len == 0 || max_chunks == 0 {
        return Vec::new();
    }

    let mut chunks: Vec<String> = Vec::new();
    let mut remaining = text;

    while !remaining.is_empty() && chunks.len() < max_chunks {
        if remaining.len() <= max_len {
            chunks.push(remaining.to_string());
            remaining = "";
            break;
        }

        // Largest char boundary <= max_len.
        let mut split = max_len;
        while split > 0 && !remaining.is_char_boundary(split) {
            split -= 1;
        }
        // Prefer breaking on the last newline in the first half..split for cleaner output.
        if let Some(nl) = remaining[..split].rfind('\n') {
            if nl > max_len / 2 {
                split = nl + 1;
            }
        }
        if split == 0 {
            // Pathological: a single char wider than max_len shouldn't happen, but
            // guard against an infinite loop.
            split = remaining.len().min(max_len.max(1));
            while split < remaining.len() && !remaining.is_char_boundary(split) {
                split += 1;
            }
        }

        chunks.push(remaining[..split].to_string());
        remaining = remaining[split..].trim_start_matches('\n');
    }

    if !remaining.is_empty() {
        let marker = "\n…(answer truncated)";
        if let Some(last) = chunks.last_mut() {
            let budget = max_len.saturating_sub(marker.len());
            if last.len() > budget {
                let mut cut = budget;
                while cut > 0 && !last.is_char_boundary(cut) {
                    cut -= 1;
                }
                last.truncate(cut);
            }
            last.push_str(marker);
        }
    }

    chunks
}

/// Build the Discord message(s) carrying a RAG-grounded answer. Long answers are
/// split across multiple embeds/messages.
/// Build a native-reply reference targeting the message that originated `issue`,
/// so a reply is threaded under the original question.
///
/// Returns `None` when the issue carries no usable message id (nothing to reply
/// to), in which case the caller posts a standalone message instead.
pub(crate) fn origin_reply_reference(
    issue: &Issue,
    channel_id: &str,
) -> Option<DiscordMessageReference> {
    if issue.id.is_empty() {
        return None;
    }
    Some(DiscordMessageReference {
        message_id: Some(issue.id.clone()),
        channel_id: Some(channel_id.to_string()),
        guild_id: None,
        // Deliver even if the original question was deleted in the meantime.
        fail_if_not_exists: Some(false),
    })
}

pub(crate) fn build_answer_messages(
    issue: &Issue,
    answer: &str,
    mention: Option<String>,
) -> Vec<DiscordMessage> {
    let short_id = truncate_string(&issue.short_id, MAX_SHORT_ID_LENGTH);
    let chunks = chunk_text(answer, MAX_DESCRIPTION_LENGTH, MAX_ANSWER_CHUNKS);
    let total = chunks.len();

    chunks
        .into_iter()
        .enumerate()
        .map(|(i, chunk)| {
            let title = if i == 0 {
                Some(format!("\u{1F4AC} Answer: {}", short_id))
            } else {
                None
            };
            let footer_text = if total > 1 {
                format!("Claudear \u{00B7} {}/{}", i + 1, total)
            } else {
                "Claudear".to_string()
            };
            DiscordMessage {
                content: if i == 0 { mention.clone() } else { None },
                embeds: Some(vec![DiscordEmbed {
                    title,
                    description: Some(chunk),
                    url: if i == 0 {
                        Some(truncate_string(&issue.url, MAX_URL_LENGTH))
                    } else {
                        None
                    },
                    color: Some(0x9b59b6), // Purple
                    fields: None,
                    footer: Some(DiscordFooter { text: footer_text }),
                    timestamp: Some(timestamp()),
                }]),
            }
        })
        .collect()
}

/// Build the Discord message for a "PR created" notification.
pub(crate) fn build_success_message(
    issue: &Issue,
    pr_url: &str,
    mention: Option<String>,
) -> DiscordMessage {
    let emoji = get_source_emoji(&issue.source);
    let short_id = truncate_string(&issue.short_id, MAX_SHORT_ID_LENGTH);
    let title = truncate_string(&issue.title, MAX_DESCRIPTION_LENGTH);
    let issue_url = truncate_string(&issue.url, MAX_URL_LENGTH);
    let pr_url_truncated = truncate_string(pr_url, MAX_URL_LENGTH);
    let source = truncate_string(&issue.source, MAX_SOURCE_LENGTH);

    let mut fields = vec![
        DiscordField {
            name: "Source".to_string(),
            value: format!("{} {}", emoji, source),
            inline: Some(true),
        },
        DiscordField {
            name: "Issue".to_string(),
            value: format!("[{}]({})", short_id, issue_url),
            inline: Some(true),
        },
        DiscordField {
            name: if issue.get_metadata::<bool>("is_pr_update").unwrap_or(false) {
                "Updated PR".to_string()
            } else {
                "PR Link".to_string()
            },
            value: format!("[View PR]({})", pr_url_truncated),
            inline: Some(false),
        },
    ];

    if let Some(changelog) = issue.get_metadata::<String>("changelog") {
        fields.push(DiscordField {
            name: "Changes".to_string(),
            value: truncate_string(&changelog, 1000),
            inline: Some(false),
        });
    }
    if let Some(reason) = issue.get_metadata::<String>("trigger_reason") {
        fields.push(DiscordField {
            name: "Trigger".to_string(),
            value: reason,
            inline: Some(false),
        });
    }
    if let Some(confidence) = issue.get_metadata::<u8>("confidence") {
        let mut conf_value = format!("{}/100", confidence);
        if let Some(reasoning) = issue.get_metadata::<String>("confidence_reasoning") {
            conf_value.push_str(&format!("\n{}", truncate_string(&reasoning, 900)));
        }
        fields.push(DiscordField {
            name: "Fix Confidence".to_string(),
            value: conf_value,
            inline: Some(true),
        });
    }

    DiscordMessage {
        content: mention.map(|m| m.to_string()),
        embeds: Some(vec![DiscordEmbed {
            title: Some(
                if issue.get_metadata::<bool>("is_pr_update").unwrap_or(false) {
                    format!("\u{270F}\u{FE0F} PR Updated: {}", short_id)
                } else {
                    format!("\u{2705} PR Created: {}", short_id)
                },
            ),
            description: Some(title),
            url: Some(pr_url_truncated.clone()),
            color: Some(0x2ecc71), // Green
            fields: Some(fields),
            footer: Some(DiscordFooter {
                text: "Claudear".to_string(),
            }),
            timestamp: Some(timestamp()),
        }]),
    }
}

/// Build the Discord message for a "completed without PR" notification.
pub(crate) fn build_completed_message(issue: &Issue, mention: Option<String>) -> DiscordMessage {
    let emoji = get_source_emoji(&issue.source);
    let short_id = truncate_string(&issue.short_id, MAX_SHORT_ID_LENGTH);
    let title = truncate_string(&issue.title, MAX_DESCRIPTION_LENGTH);
    let url = truncate_string(&issue.url, MAX_URL_LENGTH);
    let source = truncate_string(&issue.source, MAX_SOURCE_LENGTH);

    let reason = issue
        .get_metadata::<String>("completion_reason")
        .unwrap_or_else(|| "Claude completed but no PR URL was captured".to_string());
    let reason_display = truncate_string(&reason, 1000);

    DiscordMessage {
        content: mention.map(|m| m.to_string()),
        embeds: Some(vec![DiscordEmbed {
            title: Some(format!("\u{2714}\u{FE0F} Completed: {}", short_id)),
            description: Some(title),
            url: Some(url),
            color: Some(0x9b59b6), // Purple
            fields: Some(vec![
                DiscordField {
                    name: "Source".to_string(),
                    value: format!("{} {}", emoji, source),
                    inline: Some(true),
                },
                DiscordField {
                    name: "Reason".to_string(),
                    value: reason_display,
                    inline: Some(false),
                },
            ]),
            footer: Some(DiscordFooter {
                text: "Claudear".to_string(),
            }),
            timestamp: Some(timestamp()),
        }]),
    }
}

/// Build the Discord message for a "failed" notification.
pub(crate) fn build_failed_message(
    issue: &Issue,
    error: &str,
    mention: Option<String>,
) -> DiscordMessage {
    let emoji = get_source_emoji(&issue.source);
    let short_id = truncate_string(&issue.short_id, MAX_SHORT_ID_LENGTH);
    let title = truncate_string(&issue.title, MAX_DESCRIPTION_LENGTH);
    let url = truncate_string(&issue.url, MAX_URL_LENGTH);
    let source = truncate_string(&issue.source, MAX_SOURCE_LENGTH);
    let error_display = truncate_string(error, 1000);

    let mut fields = vec![
        DiscordField {
            name: "Source".to_string(),
            value: format!("{} {}", emoji, source),
            inline: Some(true),
        },
        DiscordField {
            name: "Error".to_string(),
            value: error_display,
            inline: Some(false),
        },
    ];
    if let Some(reason) = issue.get_metadata::<String>("trigger_reason") {
        fields.push(DiscordField {
            name: "Trigger".to_string(),
            value: reason,
            inline: Some(false),
        });
    }

    DiscordMessage {
        content: mention.map(|m| m.to_string()),
        embeds: Some(vec![DiscordEmbed {
            title: Some(format!("\u{274C} Failed: {}", short_id)),
            description: Some(title),
            url: Some(url),
            color: Some(0xe74c3c), // Red
            fields: Some(fields),
            footer: Some(DiscordFooter {
                text: "Claudear".to_string(),
            }),
            timestamp: Some(timestamp()),
        }]),
    }
}

/// Build the Discord message for a status notification.
pub(crate) fn build_status_message(message: &str) -> DiscordMessage {
    let message_truncated = truncate_string(message, MAX_DESCRIPTION_LENGTH);

    DiscordMessage {
        content: None,
        embeds: Some(vec![DiscordEmbed {
            title: None,
            description: Some(message_truncated),
            url: None,
            color: Some(0x9b59b6), // Purple
            fields: None,
            footer: Some(DiscordFooter {
                text: "Claudear".to_string(),
            }),
            timestamp: Some(timestamp()),
        }]),
    }
}

/// Build the Discord message for a "PR merged" notification.
pub(crate) fn build_merged_message(
    issue: &Issue,
    pr_url: &str,
    mention: Option<String>,
) -> DiscordMessage {
    let emoji = get_source_emoji(&issue.source);
    let short_id = truncate_string(&issue.short_id, MAX_SHORT_ID_LENGTH);
    let title = truncate_string(&issue.title, MAX_DESCRIPTION_LENGTH);
    let issue_url = truncate_string(&issue.url, MAX_URL_LENGTH);
    let pr_url_truncated = truncate_string(pr_url, MAX_URL_LENGTH);
    let source = truncate_string(&issue.source, MAX_SOURCE_LENGTH);

    DiscordMessage {
        content: mention.map(|m| m.to_string()),
        embeds: Some(vec![DiscordEmbed {
            title: Some(format!("\u{1F389} PR Merged: {}", short_id)),
            description: Some(title),
            url: Some(pr_url_truncated.clone()),
            color: Some(0x1abc9c), // Teal
            fields: Some(vec![
                DiscordField {
                    name: "Source".to_string(),
                    value: format!("{} {}", emoji, source),
                    inline: Some(true),
                },
                DiscordField {
                    name: "Issue".to_string(),
                    value: format!("[{}]({})", short_id, issue_url),
                    inline: Some(true),
                },
                DiscordField {
                    name: "PR Link".to_string(),
                    value: format!("[View PR]({})", pr_url_truncated),
                    inline: Some(false),
                },
            ]),
            footer: Some(DiscordFooter {
                text: "Claudear".to_string(),
            }),
            timestamp: Some(timestamp()),
        }]),
    }
}

/// Build the Discord message for a "PR closed" notification.
pub(crate) fn build_closed_message(
    issue: &Issue,
    pr_url: &str,
    mention: Option<String>,
) -> DiscordMessage {
    let emoji = get_source_emoji(&issue.source);
    let short_id = truncate_string(&issue.short_id, MAX_SHORT_ID_LENGTH);
    let title = truncate_string(&issue.title, MAX_DESCRIPTION_LENGTH);
    let pr_url_truncated = truncate_string(pr_url, MAX_URL_LENGTH);
    let source = truncate_string(&issue.source, MAX_SOURCE_LENGTH);

    DiscordMessage {
        content: mention.map(|m| m.to_string()),
        embeds: Some(vec![DiscordEmbed {
            title: Some(format!("\u{1F6AB} PR Closed: {}", short_id)),
            description: Some(title),
            url: Some(pr_url_truncated.clone()),
            color: Some(0x95a5a6), // Grey
            fields: Some(vec![
                DiscordField {
                    name: "Source".to_string(),
                    value: format!("{} {}", emoji, source),
                    inline: Some(true),
                },
                DiscordField {
                    name: "PR Link".to_string(),
                    value: format!("[View PR]({})", pr_url_truncated),
                    inline: Some(true),
                },
                DiscordField {
                    name: "Note".to_string(),
                    value: "PR was closed without merging".to_string(),
                    inline: Some(false),
                },
            ]),
            footer: Some(DiscordFooter {
                text: "Claudear".to_string(),
            }),
            timestamp: Some(timestamp()),
        }]),
    }
}

/// Build the Discord message for a cascade PR creation.
pub(crate) fn build_cascade_success_message(
    issue: &Issue,
    pr_url: &str,
    mention: Option<String>,
) -> DiscordMessage {
    let short_id = truncate_string(&issue.short_id, MAX_SHORT_ID_LENGTH);
    let pr_url_truncated = truncate_string(pr_url, MAX_URL_LENGTH);
    let upstream = issue
        .get_metadata::<String>("cascade_upstream_repo")
        .unwrap_or_default();
    let downstream = issue
        .get_metadata::<String>("cascade_downstream_repo")
        .unwrap_or_default();
    let upstream_pr_url = issue
        .get_metadata::<String>("cascade_upstream_pr_url")
        .unwrap_or_default();
    let original_issue_short_id = issue
        .get_metadata::<String>("cascade_original_issue_short_id")
        .unwrap_or_default();

    let mut fields = vec![
        DiscordField {
            name: "Upstream".to_string(),
            value: upstream,
            inline: Some(true),
        },
        DiscordField {
            name: "Downstream".to_string(),
            value: downstream.clone(),
            inline: Some(true),
        },
    ];

    if !original_issue_short_id.is_empty() {
        fields.push(DiscordField {
            name: "Original Issue".to_string(),
            value: original_issue_short_id,
            inline: Some(true),
        });
    }

    if !upstream_pr_url.is_empty() {
        fields.push(DiscordField {
            name: "Upstream PR".to_string(),
            value: format!(
                "[View PR]({})",
                truncate_string(&upstream_pr_url, MAX_URL_LENGTH)
            ),
            inline: Some(false),
        });
    }

    fields.push(DiscordField {
        name: "Cascade PR".to_string(),
        value: format!("[View PR]({})", pr_url_truncated),
        inline: Some(false),
    });

    if let Some(confidence) = issue.get_metadata::<u8>("confidence") {
        let mut conf_value = format!("{}/100", confidence);
        if let Some(reasoning) = issue.get_metadata::<String>("confidence_reasoning") {
            conf_value.push_str(&format!("\n{}", truncate_string(&reasoning, 900)));
        }
        fields.push(DiscordField {
            name: "Fix Confidence".to_string(),
            value: conf_value,
            inline: Some(true),
        });
    }

    DiscordMessage {
        content: mention.map(|m| m.to_string()),
        embeds: Some(vec![DiscordEmbed {
            title: Some(format!("\u{1F517} Cascade PR: {}", short_id)),
            description: Some(format!("Downstream adaptation for {}", downstream)),
            url: Some(pr_url_truncated),
            color: Some(0x3498db), // Blue
            fields: Some(fields),
            footer: Some(DiscordFooter {
                text: "Claudear \u{2014} Cascade".to_string(),
            }),
            timestamp: Some(timestamp()),
        }]),
    }
}

/// Build the Discord message for a cascade failure.
pub(crate) fn build_cascade_failed_message(
    issue: &Issue,
    error: &str,
    mention: Option<String>,
) -> DiscordMessage {
    let short_id = truncate_string(&issue.short_id, MAX_SHORT_ID_LENGTH);
    let error_display = truncate_string(error, 1000);
    let upstream = issue
        .get_metadata::<String>("cascade_upstream_repo")
        .unwrap_or_default();
    let downstream = issue
        .get_metadata::<String>("cascade_downstream_repo")
        .unwrap_or_default();
    let upstream_pr_url = issue
        .get_metadata::<String>("cascade_upstream_pr_url")
        .unwrap_or_default();
    let original_issue_short_id = issue
        .get_metadata::<String>("cascade_original_issue_short_id")
        .unwrap_or_default();

    let mut fields = vec![
        DiscordField {
            name: "Upstream".to_string(),
            value: upstream,
            inline: Some(true),
        },
        DiscordField {
            name: "Downstream".to_string(),
            value: downstream.clone(),
            inline: Some(true),
        },
    ];

    if !original_issue_short_id.is_empty() {
        fields.push(DiscordField {
            name: "Original Issue".to_string(),
            value: original_issue_short_id,
            inline: Some(true),
        });
    }

    if !upstream_pr_url.is_empty() {
        fields.push(DiscordField {
            name: "Upstream PR".to_string(),
            value: format!(
                "[View PR]({})",
                truncate_string(&upstream_pr_url, MAX_URL_LENGTH)
            ),
            inline: Some(false),
        });
    }

    fields.push(DiscordField {
        name: "Error".to_string(),
        value: error_display,
        inline: Some(false),
    });

    DiscordMessage {
        content: mention.map(|m| m.to_string()),
        embeds: Some(vec![DiscordEmbed {
            title: Some(format!("\u{26A0}\u{FE0F} Cascade Failed: {}", short_id)),
            description: Some(format!("Failed to adapt {}", downstream)),
            url: None,
            color: Some(0xe67e22), // Dark Orange
            fields: Some(fields),
            footer: Some(DiscordFooter {
                text: "Claudear \u{2014} Cascade".to_string(),
            }),
            timestamp: Some(timestamp()),
        }]),
    }
}

/// Build the Discord message for a regression detection.
pub(crate) fn build_regression_detected_message(
    issue: &Issue,
    error: &str,
    mention: Option<String>,
) -> DiscordMessage {
    let emoji = get_source_emoji(&issue.source);
    let short_id = truncate_string(&issue.short_id, MAX_SHORT_ID_LENGTH);
    let url = truncate_string(&issue.url, MAX_URL_LENGTH);
    let source = truncate_string(&issue.source, MAX_SOURCE_LENGTH);
    let error_display = truncate_string(error, 1000);

    DiscordMessage {
        content: mention.map(|m| m.to_string()),
        embeds: Some(vec![DiscordEmbed {
            title: Some(format!("\u{1F4C9} Regression Detected: {}", short_id)),
            description: Some("A previously fixed issue has regressed".to_string()),
            url: if url.is_empty() { None } else { Some(url) },
            color: Some(0xe74c3c), // Red
            fields: Some(vec![
                DiscordField {
                    name: "Source".to_string(),
                    value: format!("{} {}", emoji, source),
                    inline: Some(true),
                },
                DiscordField {
                    name: "Details".to_string(),
                    value: error_display,
                    inline: Some(false),
                },
                DiscordField {
                    name: "Action".to_string(),
                    value: "Retry has been scheduled".to_string(),
                    inline: Some(false),
                },
            ]),
            footer: Some(DiscordFooter {
                text: "Claudear \u{2014} Regression Monitor".to_string(),
            }),
            timestamp: Some(timestamp()),
        }]),
    }
}

/// Build the Discord message for a regression resolved (final check passed).
pub(crate) fn build_regression_resolved_message(
    issue: &Issue,
    mention: Option<String>,
) -> DiscordMessage {
    let emoji = get_source_emoji(&issue.source);
    let short_id = truncate_string(&issue.short_id, MAX_SHORT_ID_LENGTH);
    let url = truncate_string(&issue.url, MAX_URL_LENGTH);
    let source = truncate_string(&issue.source, MAX_SOURCE_LENGTH);

    DiscordMessage {
        content: mention.map(|m| m.to_string()),
        embeds: Some(vec![DiscordEmbed {
            title: Some(format!("\u{2705} Regression Resolved: {}", short_id)),
            description: Some("No regression detected after monitoring period".to_string()),
            url: if url.is_empty() { None } else { Some(url) },
            color: Some(0x2ecc71), // Green
            fields: Some(vec![
                DiscordField {
                    name: "Source".to_string(),
                    value: format!("{} {}", emoji, source),
                    inline: Some(true),
                },
                DiscordField {
                    name: "Status".to_string(),
                    value: "Issue resolved after final check".to_string(),
                    inline: Some(false),
                },
            ]),
            footer: Some(DiscordFooter {
                text: "Claudear \u{2014} Regression Monitor".to_string(),
            }),
            timestamp: Some(timestamp()),
        }]),
    }
}

/// Build the Discord message for an "urgent issues" notification.
///
/// Returns `None` when the issue list is empty (nothing to send).
pub(crate) fn build_urgent_issues_message(
    issues: &[Issue],
    mention: Option<String>,
) -> Option<DiscordMessage> {
    if issues.is_empty() {
        return None;
    }

    let fields: Vec<DiscordField> = issues
        .iter()
        .take(10)
        .map(|issue| {
            let emoji = get_source_emoji(&issue.source);
            let short_id = truncate_string(&issue.short_id, MAX_SHORT_ID_LENGTH);
            let title = truncate_string(&issue.title, 50);
            let url = truncate_string(&issue.url, MAX_URL_LENGTH);
            DiscordField {
                name: format!("{} {}", emoji, short_id),
                value: format!("[{}]({})", title, url),
                inline: Some(true),
            }
        })
        .collect();

    Some(DiscordMessage {
        content: mention.map(|m| m.to_string()),
        embeds: Some(vec![DiscordEmbed {
            title: Some(format!(
                "\u{1F6A8} {} Urgent Issue{} Detected",
                issues.len(),
                if issues.len() > 1 { "s" } else { "" }
            )),
            description: Some("The following issues require attention:".to_string()),
            url: None,
            color: Some(0xf39c12), // Orange
            fields: Some(fields),
            footer: Some(DiscordFooter {
                text: "Claudear".to_string(),
            }),
            timestamp: Some(timestamp()),
        }]),
    })
}

/// Build the Discord message for a human-in-the-loop question.
pub(crate) fn build_ask_question_message(
    issue: &Issue,
    request: &AskRequest,
    mention: Option<String>,
) -> DiscordMessage {
    // Mention in content (outside embed so it pings the user).
    // Reply detection uses Discord's native reply feature (message_reference).
    let content = mention;

    let mut fields = Vec::new();

    if let Some(ref why) = request.question.why {
        fields.push(DiscordField {
            name: "Why".to_string(),
            value: why.clone(),
            inline: Some(false),
        });
    }

    if let Some(ref ctx) = request.question.context {
        fields.push(DiscordField {
            name: "Context".to_string(),
            value: truncate_string(ctx, 400).to_string(),
            inline: Some(false),
        });
    }

    if !request.question.options.is_empty() {
        let options_text = request
            .question
            .options
            .iter()
            .enumerate()
            .map(|(i, opt)| format!("**{}**. {}", i + 1, opt))
            .collect::<Vec<_>>()
            .join("\n");
        fields.push(DiscordField {
            name: "Options".to_string(),
            value: options_text,
            inline: Some(false),
        });
    }

    DiscordMessage {
        content,
        embeds: Some(vec![DiscordEmbed {
            title: Some(format!("\u{2753} Input needed: {}", issue.short_id)),
            description: Some(request.question.question.clone()),
            url: None,
            color: Some(0xf59e0b), // Amber
            fields: if fields.is_empty() {
                None
            } else {
                Some(fields)
            },
            footer: Some(DiscordFooter {
                text: "Claudear".to_string(),
            }),
            timestamp: Some(timestamp()),
        }]),
    }
}

#[async_trait]
impl<H: DiscordWebhookClient + 'static> Notifier for DiscordNotifier<H> {
    fn name(&self) -> &str {
        "discord"
    }

    fn is_enabled(&self) -> bool {
        self.config.webhook_url.is_some() || self.has_bot_channel()
    }

    async fn notify_start(&self, issue: &Issue) -> Result<()> {
        let mention = self.get_user_mention_for_issue(issue);
        let _ = self.send(build_start_message(issue, mention)).await?;
        Ok(())
    }

    async fn notify_success(&self, issue: &Issue, pr_url: &str) -> Result<()> {
        let mention = self.get_user_mention_for_issue(issue);
        if issue
            .get_metadata::<String>("cascade_downstream_repo")
            .is_some()
        {
            let _ = self
                .send(build_cascade_success_message(issue, pr_url, mention))
                .await?;
        } else {
            let _ = self
                .send(build_success_message(issue, pr_url, mention))
                .await?;
        }
        Ok(())
    }

    async fn notify_completed(&self, issue: &Issue) -> Result<()> {
        let mention = self.get_user_mention_for_issue(issue);
        if issue
            .get_metadata::<bool>("regression_resolved")
            .unwrap_or(false)
        {
            let _ = self
                .send(build_regression_resolved_message(issue, mention))
                .await?;
        } else {
            let _ = self.send(build_completed_message(issue, mention)).await?;
        }
        Ok(())
    }

    async fn notify_failed(&self, issue: &Issue, error: &str) -> Result<()> {
        let mention = self.get_user_mention_for_issue(issue);
        if issue
            .get_metadata::<bool>("regression_detected")
            .unwrap_or(false)
        {
            let _ = self
                .send(build_regression_detected_message(issue, error, mention))
                .await?;
        } else if issue
            .get_metadata::<String>("cascade_downstream_repo")
            .is_some()
        {
            let _ = self
                .send(build_cascade_failed_message(issue, error, mention))
                .await?;
        } else {
            let _ = self
                .send(build_failed_message(issue, error, mention))
                .await?;
        }
        Ok(())
    }

    async fn notify_merged(&self, issue: &Issue, pr_url: &str) -> Result<()> {
        let mention = self.get_user_mention_for_issue(issue);
        let _ = self
            .send(build_merged_message(issue, pr_url, mention))
            .await?;
        Ok(())
    }

    async fn notify_closed(&self, issue: &Issue, pr_url: &str) -> Result<()> {
        let mention = self.get_user_mention_for_issue(issue);
        let _ = self
            .send(build_closed_message(issue, pr_url, mention))
            .await?;
        Ok(())
    }

    async fn notify_status(&self, message: &str) -> Result<()> {
        let _ = self.send(build_status_message(message)).await?;
        Ok(())
    }

    async fn notify_answer(&self, issue: &Issue, answer: &str) -> Result<()> {
        if !self.has_delivery_path() {
            return Err(Error::notifier(
                "discord",
                "No delivery path configured: need either a webhook URL or a bot token with channel ID",
            ));
        }
        let mention = self.get_user_mention_for_issue(issue);
        // Reply to the original question with the first message so the answer is
        // threaded to it; any continuation chunks follow as normal messages.
        for (i, message) in build_answer_messages(issue, answer, mention)
            .into_iter()
            .enumerate()
        {
            let _ = self.send_to_issue_channel(issue, message, i == 0).await?;
        }
        Ok(())
    }

    async fn notify_urgent_issues(&self, issues: &[Issue]) -> Result<()> {
        let mention = self.get_user_mention();
        if let Some(message) = build_urgent_issues_message(issues, mention) {
            let _ = self.send(message).await?;
        }
        Ok(())
    }

    async fn ask_question(
        &self,
        issue: &Issue,
        request: &AskRequest,
    ) -> Result<Option<AskDelivery>> {
        if !self.has_delivery_path() {
            return Err(Error::notifier(
                "discord",
                "No delivery path configured: need either a webhook URL or a bot token with channel ID",
            ));
        }

        let mention = self.get_user_mention_for_issue(issue);
        let sent_info = self
            .send(build_ask_question_message(issue, request, mention))
            .await?;

        let message_id = sent_info.as_ref().map(|info| info.message_id.clone());
        if let Some(ref info) = sent_info {
            ask_reply_inbox::remember_ask_delivery_id(
                "discord",
                &request.correlation_id,
                info.message_id.clone(),
            );
            ask_reply_inbox::remember_ask_poll_channel(
                "discord",
                &request.correlation_id,
                info.channel_id.clone(),
            );
        }

        Ok(Some(AskDelivery {
            channel: "discord".to_string(),
            target: self.get_target_discord_id_for_issue(issue),
            message_id,
        }))
    }

    async fn poll_question_replies(
        &self,
        request: &AskRequest,
        _since: DateTime<Utc>,
    ) -> Result<Vec<AskReply>> {
        let client = match self.bot_client.as_ref() {
            Some(c) => c,
            None => return Ok(Vec::new()),
        };

        // Prefer the channel the webhook actually posted to (fixes M2: webhook
        // may target a different channel than self.config.channel_id). Fall back
        // to the configured channel_id when no remembered channel exists.
        let poll_channel = ask_reply_inbox::ask_poll_channel("discord", &request.correlation_id);
        let channel_id = match poll_channel
            .as_deref()
            .or(self.config.channel_id.as_deref())
        {
            Some(v) if !v.is_empty() => v,
            _ => return Ok(Vec::new()),
        };

        // Use remembered delivery IDs from the inbox (set during ask_question).
        let remembered_ids: std::collections::HashSet<String> =
            ask_reply_inbox::ask_delivery_ids("discord", &request.correlation_id)
                .into_iter()
                .collect();

        // When we have a known ask message ID, fetch only messages posted after
        // it (fixes the 50-message window limit). Otherwise fall back to recent
        // history and scan by embed title.
        let messages = if let Some(after_id) = remembered_ids.iter().min() {
            client
                .list_channel_messages_after(channel_id, after_id, 100)
                .await?
        } else {
            client.list_channel_messages(channel_id, 100).await?
        };

        // Build the full set of ask message IDs: start with remembered IDs,
        // then also scan fetched messages for the embed title pattern as a
        // fallback (covers the case where remembered IDs are empty).
        let mut ask_message_ids = remembered_ids;
        if ask_message_ids.is_empty() {
            let ask_prefix = format!("\u{2753} Input needed: {}", request.short_id);
            for m in &messages {
                if m.embeds
                    .iter()
                    .any(|e| e.title.as_ref().is_some_and(|t| t.starts_with(&ask_prefix)))
                {
                    ask_message_ids.insert(m.id.clone());
                }
            }
        }

        let reply_pairs: Vec<(String, AskReply)> = messages
            .into_iter()
            .filter_map(|message| {
                if ask_message_ids.contains(&message.id) {
                    return None;
                }

                let author = message.author?;

                let parsed = DateTime::parse_from_rfc3339(&message.timestamp)
                    .ok()
                    .map(|dt| dt.with_timezone(&Utc))?;

                // Only accept Discord replies (message_reference) to an ask message.
                let is_reply_to_ask = message
                    .message_reference
                    .as_ref()
                    .and_then(|r| r.message_id.as_ref())
                    .map(|mid| ask_message_ids.contains(mid))
                    .unwrap_or(false);

                if !is_reply_to_ask {
                    return None;
                }

                let answer = Self::extract_reply_text(&message.content)?;
                Some((
                    message.id.clone(),
                    AskReply {
                        correlation_id: request.correlation_id.clone(),
                        channel: "discord".to_string(),
                        responder: Some(author.id),
                        answer,
                        replied_at: parsed,
                    },
                ))
            })
            .collect();

        for (msg_id, _) in &reply_pairs {
            let emojis = [
                "\u{1F389}",
                "\u{1F49C}",
                "\u{2728}",
                "\u{1F31F}",
                "\u{1F64C}",
                "\u{1F4AA}",
                "\u{1F525}",
                "\u{1F680}",
                "\u{1F929}",
                "\u{1F496}",
            ];
            let hash: usize = msg_id
                .bytes()
                .fold(0usize, |acc, b| acc.wrapping_add(b as usize));
            let idx = hash % emojis.len();
            if let Err(e) = client.add_reaction(channel_id, msg_id, emojis[idx]).await {
                tracing::debug!(error = %e, "Failed to react to reply message");
            }
        }

        let mut replies: Vec<AskReply> = reply_pairs.into_iter().map(|(_, r)| r).collect();
        replies.sort_by_key(|r| r.replied_at);
        Ok(replies)
    }

    fn supports_replies(&self) -> bool {
        self.has_bot_channel()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_registry() -> claudear_config::users::UserRegistry {
        claudear_config::users::UserRegistry::new(std::collections::HashMap::new())
    }

    #[test]
    fn test_source_emoji() {
        assert_eq!(get_source_emoji("linear"), "\u{1F4CB}");
        assert_eq!(get_source_emoji("sentry"), "\u{1F534}");
        assert_eq!(get_source_emoji("github"), "\u{1F419}");
        assert_eq!(get_source_emoji("unknown"), "\u{1F4CC}");
    }

    #[test]
    fn test_chunk_text_short_single_chunk() {
        let chunks = chunk_text("hello world", 2048, 5);
        assert_eq!(chunks, vec!["hello world".to_string()]);
    }

    #[test]
    fn test_chunk_text_empty() {
        assert!(chunk_text("", 2048, 5).is_empty());
        assert!(chunk_text("   ", 2048, 5).is_empty());
    }

    #[test]
    fn test_chunk_text_splits_on_length() {
        let text = "a".repeat(50);
        let chunks = chunk_text(&text, 20, 5);
        assert!(chunks.len() >= 3);
        for c in &chunks {
            assert!(c.len() <= 20);
        }
        // Concatenation preserves all content (no truncation within cap).
        assert_eq!(chunks.concat(), text);
    }

    #[test]
    fn test_chunk_text_respects_max_chunks_and_marks_truncation() {
        let text = "a".repeat(1000);
        let chunks = chunk_text(&text, 20, 2);
        assert_eq!(chunks.len(), 2);
        assert!(chunks.last().unwrap().contains("truncated"));
    }

    #[test]
    fn test_build_answer_messages_first_has_title_and_mention() {
        let issue = Issue::new("123", "DISCORD-123", "Q", "https://x/y", "discord");
        let msgs = build_answer_messages(&issue, "a short answer", Some("<@1>".to_string()));
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].content.as_deref(), Some("<@1>"));
        let embed = &msgs[0].embeds.as_ref().unwrap()[0];
        assert!(embed.title.as_ref().unwrap().contains("DISCORD-123"));
        assert_eq!(embed.description.as_deref(), Some("a short answer"));
    }

    #[test]
    fn test_build_answer_messages_multichunk_only_first_has_mention() {
        let issue = Issue::new("123", "DISCORD-123", "Q", "https://x/y", "discord");
        let long = "b".repeat(MAX_DESCRIPTION_LENGTH * 2 + 100);
        let msgs = build_answer_messages(&issue, &long, Some("<@1>".to_string()));
        assert!(msgs.len() >= 2);
        assert_eq!(msgs[0].content.as_deref(), Some("<@1>"));
        assert!(msgs[1].content.is_none());
        // Continuation embeds carry no title.
        assert!(msgs[1].embeds.as_ref().unwrap()[0].title.is_none());
    }

    // --- Native reply reference (threading answers to the question) ---

    #[test]
    fn test_origin_reply_reference_targets_original_message() {
        let issue = Issue::new(
            "1511974915572502629",
            "DISCORD-15119749",
            "what is query not equal syntax?",
            "https://discord/x",
            "discord",
        );
        let r = origin_reply_reference(&issue, "1471462861338312775")
            .expect("reference should be built for a non-empty message id");

        // Replies to the originating Discord message, in its channel.
        assert_eq!(r.message_id.as_deref(), Some("1511974915572502629"));
        assert_eq!(r.channel_id.as_deref(), Some("1471462861338312775"));
        // Must not fail delivery if the question was deleted meanwhile.
        assert_eq!(r.fail_if_not_exists, Some(false));
    }

    #[test]
    fn test_origin_reply_reference_none_without_message_id() {
        let mut issue = Issue::new("", "DISCORD-1", "q", "u", "discord");
        issue.id.clear();
        assert!(origin_reply_reference(&issue, "chan").is_none());
    }

    #[test]
    fn test_reply_reference_serializes_required_fields() {
        let issue = Issue::new("99", "DISCORD-9", "q", "u", "discord");
        let r = origin_reply_reference(&issue, "chan").unwrap();
        let json = serde_json::to_string(&r).unwrap();
        // The reply target and graceful-degrade flag must reach Discord.
        assert!(json.contains("\"message_id\":\"99\""));
        assert!(json.contains("\"channel_id\":\"chan\""));
        assert!(json.contains("\"fail_if_not_exists\":false"));
    }

    #[test]
    fn test_user_mention() {
        let config_with_id = DiscordConfig {
            webhook_url: Some("https://example.com".into()),
            user_id: Some("123456".to_string()),
            ..Default::default()
        };
        let notifier = DiscordNotifier::new(config_with_id, empty_registry());
        assert_eq!(notifier.get_user_mention(), Some("<@123456>".to_string()));

        let config_without_id = DiscordConfig {
            webhook_url: Some("https://example.com".into()),
            user_id: None,
            ..Default::default()
        };
        let notifier = DiscordNotifier::new(config_without_id, empty_registry());
        assert_eq!(notifier.get_user_mention(), None);
    }

    #[test]
    fn test_is_enabled() {
        let enabled_config = DiscordConfig {
            webhook_url: Some("https://example.com".into()),
            user_id: None,
            ..Default::default()
        };
        let notifier = DiscordNotifier::new(enabled_config, empty_registry());
        assert!(notifier.is_enabled());

        let disabled_config = DiscordConfig {
            webhook_url: None,
            user_id: None,
            ..Default::default()
        };
        let notifier = DiscordNotifier::new(disabled_config, empty_registry());
        assert!(!notifier.is_enabled());
    }

    #[test]
    fn test_is_enabled_with_bot_channel_only() {
        let config = DiscordConfig {
            webhook_url: None,
            user_id: None,
            bot_token: Some("bot-token".into()),
            channel_id: Some("channel-123".to_string()),
            ..Default::default()
        };
        let notifier = DiscordNotifier::new(config, empty_registry());
        assert!(notifier.is_enabled());
    }

    #[test]
    fn test_is_enabled_false_with_only_bot_token() {
        let config = DiscordConfig {
            webhook_url: None,
            user_id: None,
            bot_token: Some("bot-token".into()),
            channel_id: None,
            ..Default::default()
        };
        let notifier = DiscordNotifier::new(config, empty_registry());
        assert!(!notifier.is_enabled());
    }

    #[test]
    fn test_is_enabled_false_with_only_channel_id() {
        let config = DiscordConfig {
            webhook_url: None,
            user_id: None,
            bot_token: None,
            channel_id: Some("channel-123".to_string()),
            ..Default::default()
        };
        let notifier = DiscordNotifier::new(config, empty_registry());
        assert!(!notifier.is_enabled());
    }

    #[test]
    fn test_is_enabled_false_with_empty_bot_token() {
        let config = DiscordConfig {
            webhook_url: None,
            user_id: None,
            bot_token: Some("".into()),
            channel_id: Some("channel-123".to_string()),
            ..Default::default()
        };
        let notifier = DiscordNotifier::new(config, empty_registry());
        assert!(!notifier.is_enabled());
    }

    #[test]
    fn test_notifier_name() {
        let config = DiscordConfig::default();
        let notifier = DiscordNotifier::new(config, empty_registry());
        assert_eq!(notifier.name(), "discord");
    }

    #[test]
    fn test_source_emoji_case_insensitive() {
        assert_eq!(get_source_emoji("LINEAR"), "\u{1F4CB}");
        assert_eq!(get_source_emoji("Linear"), "\u{1F4CB}");
        assert_eq!(get_source_emoji("SENTRY"), "\u{1F534}");
        assert_eq!(get_source_emoji("GitHub"), "\u{1F419}");
    }

    #[test]
    fn test_source_emoji_jira() {
        assert_eq!(get_source_emoji("jira"), "\u{1F3AB}");
        assert_eq!(get_source_emoji("JIRA"), "\u{1F3AB}");
    }

    #[test]
    fn test_timestamp_format() {
        let ts = timestamp();
        // Should be valid RFC3339
        assert!(ts.contains("T"));
        assert!(ts.contains("+") || ts.contains("Z"));
    }

    #[test]
    fn test_discord_message_serialization() {
        let message = DiscordMessage {
            content: Some("Test message".to_string()),
            embeds: None,
        };
        let json = serde_json::to_string(&message).unwrap();
        assert!(json.contains("Test message"));
        // embeds should be skipped because it's None
        assert!(!json.contains("embeds"));
    }

    #[test]
    fn test_discord_embed_serialization() {
        let embed = DiscordEmbed {
            title: Some("Test Title".to_string()),
            description: Some("Test Description".to_string()),
            url: Some("https://example.com".to_string()),
            color: Some(0xFF0000),
            fields: None,
            footer: None,
            timestamp: None,
        };
        let json = serde_json::to_string(&embed).unwrap();
        assert!(json.contains("Test Title"));
        assert!(json.contains("Test Description"));
        assert!(json.contains("https://example.com"));
        // Optional fields should be skipped
        assert!(!json.contains("fields"));
        assert!(!json.contains("footer"));
        assert!(!json.contains("timestamp"));
    }

    #[test]
    fn test_discord_field_serialization() {
        let field = DiscordField {
            name: "Field Name".to_string(),
            value: "Field Value".to_string(),
            inline: Some(true),
        };
        let json = serde_json::to_string(&field).unwrap();
        assert!(json.contains("Field Name"));
        assert!(json.contains("Field Value"));
        assert!(json.contains("true"));
    }

    #[test]
    fn test_discord_field_serialization_no_inline() {
        let field = DiscordField {
            name: "Field Name".to_string(),
            value: "Field Value".to_string(),
            inline: None,
        };
        let json = serde_json::to_string(&field).unwrap();
        assert!(!json.contains("inline"));
    }

    #[test]
    fn test_discord_footer_serialization() {
        let footer = DiscordFooter {
            text: "Footer Text".to_string(),
        };
        let json = serde_json::to_string(&footer).unwrap();
        assert!(json.contains("Footer Text"));
    }

    #[tokio::test]
    async fn test_notify_status_disabled() {
        let config = DiscordConfig {
            webhook_url: None, // Disabled
            user_id: None,
            ..Default::default()
        };
        let notifier = DiscordNotifier::new(config, empty_registry());

        // Should return Ok without actually sending
        let result = notifier.notify_status("Test status").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_notify_start_disabled() {
        let config = DiscordConfig {
            webhook_url: None,
            user_id: None,
            ..Default::default()
        };
        let notifier = DiscordNotifier::new(config, empty_registry());

        let issue = Issue::new(
            "123",
            "TEST-123",
            "Test Issue",
            "https://example.com",
            "linear",
        );
        let result = notifier.notify_start(&issue).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_notify_success_disabled() {
        let config = DiscordConfig {
            webhook_url: None,
            user_id: None,
            ..Default::default()
        };
        let notifier = DiscordNotifier::new(config, empty_registry());

        let issue = Issue::new(
            "123",
            "TEST-123",
            "Test Issue",
            "https://example.com",
            "linear",
        );
        let result = notifier
            .notify_success(&issue, "https://github.com/org/repo/pull/1")
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_notify_failed_disabled() {
        let config = DiscordConfig {
            webhook_url: None,
            user_id: None,
            ..Default::default()
        };
        let notifier = DiscordNotifier::new(config, empty_registry());

        let issue = Issue::new(
            "123",
            "TEST-123",
            "Test Issue",
            "https://example.com",
            "linear",
        );
        let result = notifier.notify_failed(&issue, "Test error").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_notify_completed_disabled() {
        let config = DiscordConfig {
            webhook_url: None,
            user_id: None,
            ..Default::default()
        };
        let notifier = DiscordNotifier::new(config, empty_registry());

        let issue = Issue::new(
            "123",
            "TEST-123",
            "Test Issue",
            "https://example.com",
            "linear",
        );
        let result = notifier.notify_completed(&issue).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_notify_urgent_issues_empty() {
        let config = DiscordConfig {
            webhook_url: Some("https://example.com".into()),
            user_id: None,
            ..Default::default()
        };
        let notifier = DiscordNotifier::new(config, empty_registry());

        // Empty list should return Ok without sending
        let result = notifier.notify_urgent_issues(&[]).await;
        assert!(result.is_ok());
    }

    // Mock-based tests for HTTP-dependent functionality

    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    /// Mock Discord webhook client for testing.
    struct MockDiscordWebhookClient {
        response_status: u16,
        response_body: String,
        call_count: AtomicUsize,
        last_calls: Mutex<Vec<(String, serde_json::Value)>>,
    }

    impl MockDiscordWebhookClient {
        fn new(status: u16, body: &str) -> Self {
            Self {
                response_status: status,
                response_body: body.to_string(),
                call_count: AtomicUsize::new(0),
                last_calls: Mutex::new(Vec::new()),
            }
        }

        fn success() -> Self {
            Self::new(204, "")
        }

        fn error(status: u16, body: &str) -> Self {
            Self::new(status, body)
        }

        fn get_call_count(&self) -> usize {
            self.call_count.load(Ordering::SeqCst)
        }

        fn get_last_call(&self) -> Option<(String, serde_json::Value)> {
            self.last_calls.lock().unwrap().last().cloned()
        }
    }

    #[async_trait]
    impl DiscordWebhookClient for MockDiscordWebhookClient {
        async fn post_json(&self, url: &str, body: &serde_json::Value) -> Result<HttpResponse> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            self.last_calls
                .lock()
                .unwrap()
                .push((url.to_string(), body.clone()));

            Ok(HttpResponse {
                status: self.response_status,
                body: self.response_body.clone(),
            })
        }
    }

    fn enabled_config() -> DiscordConfig {
        DiscordConfig {
            webhook_url: Some("https://discord.com/api/webhooks/123/abc".into()),
            user_id: None,
            ..Default::default()
        }
    }

    fn enabled_config_with_user() -> DiscordConfig {
        DiscordConfig {
            webhook_url: Some("https://discord.com/api/webhooks/123/abc".into()),
            user_id: Some("987654321".to_string()),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn test_send_webhook_success() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);
        let issue = Issue::new(
            "123",
            "PROJ-123",
            "Test Issue",
            "https://example.com",
            "linear",
        );

        let result = notifier.notify_start(&issue).await;

        assert!(result.is_ok());
        assert_eq!(notifier.http.get_call_count(), 1);
    }

    #[tokio::test]
    async fn test_send_webhook_sends_to_correct_url() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);
        let issue = Issue::new(
            "123",
            "PROJ-123",
            "Test Issue",
            "https://example.com",
            "linear",
        );

        notifier.notify_start(&issue).await.unwrap();

        let (url, _) = notifier.http.get_last_call().unwrap();
        assert_eq!(url, "https://discord.com/api/webhooks/123/abc?wait=true");
    }

    #[tokio::test]
    async fn test_send_webhook_error_response() {
        let mock = MockDiscordWebhookClient::error(400, "Bad Request");
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);
        let issue = Issue::new(
            "123",
            "PROJ-123",
            "Test Issue",
            "https://example.com",
            "linear",
        );

        let result = notifier.notify_start(&issue).await;

        assert!(result.is_err());
        let err_str = result.unwrap_err().to_string();
        assert!(err_str.contains("Webhook error"));
        assert!(err_str.contains("Bad Request"));
    }

    #[tokio::test]
    async fn test_send_webhook_server_error() {
        let mock = MockDiscordWebhookClient::error(500, "Internal Server Error");
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);

        let result = notifier.notify_status("Test").await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_notify_start_sends_embed() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);
        let issue = Issue::new(
            "123",
            "PROJ-123",
            "Test Issue Title",
            "https://example.com",
            "linear",
        );

        notifier.notify_start(&issue).await.unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        assert!(body["embeds"].is_array());
        let embed = &body["embeds"][0];
        assert!(embed["title"].as_str().unwrap().contains("PROJ-123"));
        assert_eq!(embed["description"], "Test Issue Title");
    }

    #[tokio::test]
    async fn test_notify_start_with_user_mention() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config_with_user(), mock);
        let issue = Issue::new(
            "123",
            "PROJ-123",
            "Test Issue",
            "https://example.com",
            "linear",
        );

        notifier.notify_start(&issue).await.unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        let content = body["content"].as_str().unwrap();
        assert!(content.contains("<@987654321>"));
    }

    #[tokio::test]
    async fn test_notify_success_sends_correct_embed() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);
        let issue = Issue::new(
            "123",
            "PROJ-123",
            "Test Issue",
            "https://example.com",
            "linear",
        );

        notifier
            .notify_success(&issue, "https://github.com/org/repo/pull/42")
            .await
            .unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        let embed = &body["embeds"][0];
        assert!(embed["title"].as_str().unwrap().contains("PR Created"));
        assert_eq!(embed["url"], "https://github.com/org/repo/pull/42");
        assert_eq!(embed["color"], 0x2ecc71); // Green
    }

    #[tokio::test]
    async fn test_notify_completed_sends_correct_embed() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);
        let issue = Issue::new(
            "123",
            "PROJ-123",
            "Test Issue",
            "https://example.com",
            "linear",
        );

        notifier.notify_completed(&issue).await.unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        let embed = &body["embeds"][0];
        assert!(embed["title"].as_str().unwrap().contains("Completed"));
        assert_eq!(embed["color"], 0x9b59b6); // Purple
    }

    #[tokio::test]
    async fn test_notify_failed_sends_correct_embed() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);
        let issue = Issue::new(
            "123",
            "PROJ-123",
            "Test Issue",
            "https://example.com",
            "linear",
        );

        notifier
            .notify_failed(&issue, "Build failed with exit code 1")
            .await
            .unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        let embed = &body["embeds"][0];
        assert!(embed["title"].as_str().unwrap().contains("Failed"));
        assert_eq!(embed["color"], 0xe74c3c); // Red
                                              // Check error field
        let fields = embed["fields"].as_array().unwrap();
        let error_field = fields.iter().find(|f| f["name"] == "Error").unwrap();
        assert!(error_field["value"]
            .as_str()
            .unwrap()
            .contains("Build failed"));
    }

    #[tokio::test]
    async fn test_notify_failed_truncates_long_error() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);
        let issue = Issue::new(
            "123",
            "PROJ-123",
            "Test Issue",
            "https://example.com",
            "linear",
        );

        let long_error = "x".repeat(2000);
        notifier.notify_failed(&issue, &long_error).await.unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        let fields = body["embeds"][0]["fields"].as_array().unwrap();
        let error_field = fields.iter().find(|f| f["name"] == "Error").unwrap();
        let error_value = error_field["value"].as_str().unwrap();
        assert!(error_value.len() <= 1010);
        assert!(error_value.ends_with("..."));
    }

    #[tokio::test]
    async fn test_notify_status_sends_embed() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);

        notifier.notify_status("System is healthy").await.unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        let embed = &body["embeds"][0];
        assert_eq!(embed["description"], "System is healthy");
    }

    #[tokio::test]
    async fn test_notify_urgent_issues_sends_embed() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);
        let issues = vec![
            Issue::new("1", "PROJ-1", "Issue 1", "https://example.com/1", "linear"),
            Issue::new("2", "PROJ-2", "Issue 2", "https://example.com/2", "sentry"),
        ];

        notifier.notify_urgent_issues(&issues).await.unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        let embed = &body["embeds"][0];
        assert!(embed["title"].as_str().unwrap().contains("2 Urgent Issues"));
        assert_eq!(embed["color"], 0xf39c12); // Orange
    }

    #[tokio::test]
    async fn test_notify_urgent_issues_with_user_mention() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config_with_user(), mock);
        let issues = vec![Issue::new(
            "1",
            "PROJ-1",
            "Issue",
            "https://example.com",
            "linear",
        )];

        notifier.notify_urgent_issues(&issues).await.unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        let content = body["content"].as_str().unwrap();
        assert!(content.contains("<@987654321>"));
    }

    #[tokio::test]
    async fn test_notify_urgent_issues_truncates_long_title() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);
        let long_title = "x".repeat(100);
        let issues = vec![Issue::new(
            "1",
            "PROJ-1",
            &long_title,
            "https://example.com",
            "linear",
        )];

        notifier.notify_urgent_issues(&issues).await.unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        let fields = body["embeds"][0]["fields"].as_array().unwrap();
        let field_value = fields[0]["value"].as_str().unwrap();
        // Title should be truncated (47 chars + "...")
        assert!(field_value.contains("..."));
    }

    #[tokio::test]
    async fn test_notify_urgent_issues_limits_to_ten() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);
        let issues: Vec<Issue> = (1..=20)
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

        notifier.notify_urgent_issues(&issues).await.unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        let fields = body["embeds"][0]["fields"].as_array().unwrap();
        assert_eq!(fields.len(), 10);
    }

    #[tokio::test]
    async fn test_notify_urgent_issues_single_item_grammar() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);
        let issues = vec![Issue::new(
            "1",
            "PROJ-1",
            "Issue",
            "https://example.com",
            "linear",
        )];

        notifier.notify_urgent_issues(&issues).await.unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        let title = body["embeds"][0]["title"].as_str().unwrap();
        // Should use singular "Issue" not "Issues"
        assert!(title.contains("1 Urgent Issue Detected"));
        assert!(!title.contains("Issues"));
    }

    #[test]
    fn test_with_http_client() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);

        assert!(notifier.is_enabled());
        assert_eq!(notifier.name(), "discord");
    }

    #[test]
    fn test_reqwest_discord_webhook_client_default() {
        let client = ReqwestDiscordWebhookClient::default();
        assert!(std::mem::size_of_val(&client) > 0);
    }

    #[test]
    fn test_http_response_fields() {
        let response = HttpResponse {
            status: 201,
            body: "Created".to_string(),
        };
        assert_eq!(response.status, 201);
        assert_eq!(response.body, "Created");
    }

    #[tokio::test]
    async fn test_source_specific_embeds() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);

        // Test linear source
        let linear_issue = Issue::new(
            "1",
            "LIN-1",
            "Linear Issue",
            "https://linear.app/1",
            "linear",
        );
        notifier.notify_start(&linear_issue).await.unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        let fields = body["embeds"][0]["fields"].as_array().unwrap();
        let source_field = fields.iter().find(|f| f["name"] == "Source").unwrap();
        assert_eq!(source_field["value"], "linear");
    }

    #[tokio::test]
    async fn test_embed_has_timestamp() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);
        let issue = Issue::new("1", "PROJ-1", "Test", "https://example.com", "linear");

        notifier.notify_start(&issue).await.unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        let timestamp = body["embeds"][0]["timestamp"].as_str().unwrap();
        // Should be RFC3339 format
        assert!(timestamp.contains("T"));
    }

    #[tokio::test]
    async fn test_embed_has_footer() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);
        let issue = Issue::new("1", "PROJ-1", "Test", "https://example.com", "linear");

        notifier.notify_start(&issue).await.unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        let footer = body["embeds"][0]["footer"]["text"].as_str().unwrap();
        assert_eq!(footer, "Claudear");
    }

    #[tokio::test]
    async fn test_notify_start_with_resolved_user_mention() {
        let mock = MockDiscordWebhookClient::success();
        let mut users = std::collections::HashMap::new();
        users.insert(
            "jake".to_string(),
            claudear_config::config::UserConfig {
                discord_id: Some("111222333".to_string()),
                ..Default::default()
            },
        );
        let registry = claudear_config::users::UserRegistry::new(users);
        let config = DiscordConfig {
            webhook_url: Some("https://discord.com/api/webhooks/123/abc".into()),
            user_id: None,
            ..Default::default()
        };
        let notifier = DiscordNotifier::with_http_client_and_registry(config, mock, registry);
        let mut issue = Issue::new("1", "LIN-1", "Test", "https://example.com", "linear");
        issue.set_metadata("resolved_user", "jake");
        notifier.notify_start(&issue).await.unwrap();
        let (_, body) = notifier.http.get_last_call().unwrap();
        let content = body["content"].as_str().unwrap();
        assert!(content.contains("<@111222333>"));
    }

    #[tokio::test]
    async fn test_resolved_user_overrides_global_user_id() {
        let mock = MockDiscordWebhookClient::success();
        let mut users = std::collections::HashMap::new();
        users.insert(
            "jake".to_string(),
            claudear_config::config::UserConfig {
                discord_id: Some("111222333".to_string()),
                ..Default::default()
            },
        );
        let registry = claudear_config::users::UserRegistry::new(users);
        let config = DiscordConfig {
            webhook_url: Some("https://discord.com/api/webhooks/123/abc".into()),
            user_id: Some("999999999".to_string()),
            ..Default::default()
        };
        let notifier = DiscordNotifier::with_http_client_and_registry(config, mock, registry);
        let mut issue = Issue::new("1", "LIN-1", "Test", "https://example.com", "linear");
        issue.set_metadata("resolved_user", "jake");
        notifier.notify_start(&issue).await.unwrap();
        let (_, body) = notifier.http.get_last_call().unwrap();
        let content = body["content"].as_str().unwrap();
        assert!(content.contains("<@111222333>"));
        assert!(!content.contains("<@999999999>"));
    }

    #[tokio::test]
    async fn test_fallback_to_global_when_no_resolved_user() {
        let mock = MockDiscordWebhookClient::success();
        let registry = claudear_config::users::UserRegistry::new(std::collections::HashMap::new());
        let config = DiscordConfig {
            webhook_url: Some("https://discord.com/api/webhooks/123/abc".into()),
            user_id: Some("999999999".to_string()),
            ..Default::default()
        };
        let notifier = DiscordNotifier::with_http_client_and_registry(config, mock, registry);
        let issue = Issue::new("1", "LIN-1", "Test", "https://example.com", "linear");
        notifier.notify_start(&issue).await.unwrap();
        let (_, body) = notifier.http.get_last_call().unwrap();
        let content = body["content"].as_str().unwrap();
        assert!(content.contains("<@999999999>"));
    }

    #[tokio::test]
    async fn test_ask_question_uses_resolved_user_target() {
        let mock = MockDiscordWebhookClient::success();
        let mut users = std::collections::HashMap::new();
        users.insert(
            "jake".to_string(),
            claudear_config::config::UserConfig {
                discord_id: Some("111222333".to_string()),
                ..Default::default()
            },
        );
        let registry = claudear_config::users::UserRegistry::new(users);
        let config = DiscordConfig {
            webhook_url: Some("https://discord.com/api/webhooks/123/abc".into()),
            user_id: Some("999999999".to_string()),
            ..Default::default()
        };
        let notifier = DiscordNotifier::with_http_client_and_registry(config, mock, registry);
        let mut issue = Issue::new("1", "LIN-1", "Test", "https://example.com", "linear");
        issue.set_metadata("resolved_user", "jake");

        let request = claudear_core::types::AskRequest {
            correlation_id: "tok-1".to_string(),
            source: "linear".to_string(),
            repo: Some("org/repo".to_string()),
            issue_id: issue.id.clone(),
            short_id: issue.short_id.clone(),
            question: claudear_core::types::BlockingQuestion {
                question: "Choose target branch?".to_string(),
                context: None,
                options: vec![],
                why: None,
            },
            asked_at: chrono::Utc::now(),
            target_discord_id: None,
            target_email: None,
            target_slack_id: None,
        };
        let delivery = notifier
            .ask_question(&issue, &request)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(delivery.target.as_deref(), Some("111222333"));
    }

    #[tokio::test]
    async fn test_ask_question_falls_back_to_global_target() {
        let mock = MockDiscordWebhookClient::success();
        let registry = claudear_config::users::UserRegistry::new(std::collections::HashMap::new());
        let config = DiscordConfig {
            webhook_url: Some("https://discord.com/api/webhooks/123/abc".into()),
            user_id: Some("999999999".to_string()),
            ..Default::default()
        };
        let notifier = DiscordNotifier::with_http_client_and_registry(config, mock, registry);
        let issue = Issue::new("1", "LIN-1", "Test", "https://example.com", "linear");
        let request = claudear_core::types::AskRequest {
            correlation_id: "tok-2".to_string(),
            source: "linear".to_string(),
            repo: Some("org/repo".to_string()),
            issue_id: issue.id.clone(),
            short_id: issue.short_id.clone(),
            question: claudear_core::types::BlockingQuestion {
                question: "Pick env?".to_string(),
                context: None,
                options: vec![],
                why: None,
            },
            asked_at: chrono::Utc::now(),
            target_discord_id: None,
            target_email: None,
            target_slack_id: None,
        };
        let delivery = notifier
            .ask_question(&issue, &request)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(delivery.target.as_deref(), Some("999999999"));
    }

    #[test]
    fn test_extract_reply_text() {
        let content = "Use main branch";
        let parsed =
            DiscordNotifier::<ReqwestDiscordWebhookClient>::extract_reply_text(content).unwrap();
        assert_eq!(parsed, "Use main branch");
    }

    #[test]
    fn test_truncate_string_short_unchanged() {
        assert_eq!(truncate_string("hello", 10), "hello");
    }

    #[test]
    fn test_truncate_string_exact_length_unchanged() {
        assert_eq!(truncate_string("hello", 5), "hello");
    }

    #[test]
    fn test_truncate_string_over_limit_adds_ellipsis() {
        let result = truncate_string("hello world", 8);
        assert!(result.ends_with("..."));
        assert!(result.len() <= 8);
    }

    #[test]
    fn test_truncate_string_very_small_max_no_room_for_ellipsis() {
        // When max_len <= 3, no room for ellipsis so just truncate
        let result = truncate_string("hello", 3);
        assert_eq!(result.len(), 3);
        assert!(!result.contains("..."));
    }

    #[test]
    fn test_truncate_string_max_len_zero() {
        let result = truncate_string("hello", 0);
        assert_eq!(result, "");
    }

    #[test]
    fn test_truncate_string_empty_input() {
        assert_eq!(truncate_string("", 10), "");
    }

    #[test]
    fn test_truncate_string_with_known_constants() {
        let long_id = "x".repeat(100);
        let result = truncate_string(&long_id, MAX_SHORT_ID_LENGTH);
        assert!(result.len() <= MAX_SHORT_ID_LENGTH);
        assert!(result.ends_with("..."));

        let long_source = "y".repeat(50);
        let result = truncate_string(&long_source, MAX_SOURCE_LENGTH);
        assert!(result.len() <= MAX_SOURCE_LENGTH);
        assert!(result.ends_with("..."));
    }

    #[test]
    fn test_truncate_string_description_length() {
        let long_desc = "z".repeat(3000);
        let result = truncate_string(&long_desc, MAX_DESCRIPTION_LENGTH);
        assert!(result.len() <= MAX_DESCRIPTION_LENGTH);
        assert!(result.ends_with("..."));
    }

    #[test]
    fn test_truncate_string_url_length() {
        let long_url = format!("https://example.com/{}", "a".repeat(2500));
        let result = truncate_string(&long_url, MAX_URL_LENGTH);
        assert!(result.len() <= MAX_URL_LENGTH);
        assert!(result.ends_with("..."));
    }

    #[test]
    fn test_extract_reply_text_empty_string() {
        let result = DiscordNotifier::<ReqwestDiscordWebhookClient>::extract_reply_text("");
        assert!(result.is_none());
    }

    #[test]
    fn test_extract_reply_text_whitespace_only() {
        let result =
            DiscordNotifier::<ReqwestDiscordWebhookClient>::extract_reply_text("   \n\t  ");
        assert!(result.is_none());
    }

    #[test]
    fn test_extract_reply_text_trims_whitespace() {
        let result =
            DiscordNotifier::<ReqwestDiscordWebhookClient>::extract_reply_text("  yes  ").unwrap();
        assert_eq!(result, "yes");
    }

    #[test]
    fn test_supports_replies_true_when_both_set() {
        let config = DiscordConfig {
            webhook_url: Some("https://example.com".into()),
            user_id: None,
            bot_token: Some("bot-token".into()),
            channel_id: Some("channel-123".to_string()),
            ..Default::default()
        };
        let notifier = DiscordNotifier::new(config, empty_registry());
        assert!(notifier.supports_replies());
    }

    #[test]
    fn test_supports_replies_false_when_no_bot_token() {
        let config = DiscordConfig {
            webhook_url: Some("https://example.com".into()),
            user_id: None,
            bot_token: None,
            channel_id: Some("channel-123".to_string()),
            ..Default::default()
        };
        let notifier = DiscordNotifier::new(config, empty_registry());
        assert!(!notifier.supports_replies());
    }

    #[test]
    fn test_supports_replies_false_when_no_channel_id() {
        let config = DiscordConfig {
            webhook_url: Some("https://example.com".into()),
            user_id: None,
            bot_token: Some("bot-token".into()),
            channel_id: None,
            ..Default::default()
        };
        let notifier = DiscordNotifier::new(config, empty_registry());
        assert!(!notifier.supports_replies());
    }

    #[test]
    fn test_supports_replies_false_when_empty_bot_token() {
        let config = DiscordConfig {
            webhook_url: Some("https://example.com".into()),
            user_id: None,
            bot_token: Some("".into()),
            channel_id: Some("channel-123".to_string()),
            ..Default::default()
        };
        let notifier = DiscordNotifier::new(config, empty_registry());
        assert!(!notifier.supports_replies());
    }

    #[test]
    fn test_supports_replies_false_when_empty_channel_id() {
        let config = DiscordConfig {
            webhook_url: Some("https://example.com".into()),
            user_id: None,
            bot_token: Some("bot-token".into()),
            channel_id: Some("".to_string()),
            ..Default::default()
        };
        let notifier = DiscordNotifier::new(config, empty_registry());
        assert!(!notifier.supports_replies());
    }

    #[test]
    fn test_get_user_mention_for_issue_no_resolved_user_no_global() {
        let config = DiscordConfig {
            webhook_url: Some("https://example.com".into()),
            user_id: None,
            ..Default::default()
        };
        let notifier = DiscordNotifier::new(config, empty_registry());
        let issue = Issue::new("1", "LIN-1", "Test", "https://example.com", "linear");
        assert_eq!(notifier.get_user_mention_for_issue(&issue), None);
    }

    #[test]
    fn test_get_user_mention_for_issue_resolved_user_no_discord_id() {
        let mut users = std::collections::HashMap::new();
        users.insert(
            "jake".to_string(),
            claudear_config::config::UserConfig {
                discord_id: None,
                ..Default::default()
            },
        );
        let registry = claudear_config::users::UserRegistry::new(users);
        let config = DiscordConfig {
            webhook_url: Some("https://example.com".into()),
            user_id: Some("fallback".to_string()),
            ..Default::default()
        };
        let notifier = DiscordNotifier::with_http_client_and_registry(
            config,
            ReqwestDiscordWebhookClient::new(),
            registry,
        );
        let mut issue = Issue::new("1", "LIN-1", "Test", "https://example.com", "linear");
        issue.set_metadata("resolved_user", "jake");
        // Falls back to global because resolved user has no discord_id
        assert_eq!(
            notifier.get_user_mention_for_issue(&issue),
            Some("<@fallback>".to_string())
        );
    }

    #[tokio::test]
    async fn test_ask_question_includes_options_and_context() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);
        let issue = Issue::new("1", "LIN-1", "Test Issue", "https://example.com", "linear");
        let request = AskRequest {
            correlation_id: "tok-opts".to_string(),
            source: "linear".to_string(),
            repo: None,
            issue_id: "1".to_string(),
            short_id: "LIN-1".to_string(),
            question: claudear_core::types::BlockingQuestion {
                question: "Pick a branch".to_string(),
                context: Some("We need a target for the PR".to_string()),
                options: vec!["main".to_string(), "develop".to_string()],
                why: Some("Multiple branches available".to_string()),
            },
            asked_at: chrono::Utc::now(),
            target_discord_id: None,
            target_email: None,
            target_slack_id: None,
        };
        notifier.ask_question(&issue, &request).await.unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        // No mention and no correlation tag => content field is absent
        assert!(body.get("content").is_none() || body["content"].is_null());
        let embed = &body["embeds"][0];
        assert_eq!(embed["description"].as_str().unwrap(), "Pick a branch");
        let fields = embed["fields"].as_array().unwrap();
        assert!(fields.iter().any(|f| f["name"] == "Why"
            && f["value"]
                .as_str()
                .unwrap()
                .contains("Multiple branches available")));
        assert!(fields.iter().any(|f| f["name"] == "Context"
            && f["value"]
                .as_str()
                .unwrap()
                .contains("We need a target for the PR")));
        assert!(fields
            .iter()
            .any(|f| f["name"] == "Options" && f["value"].as_str().unwrap().contains("main")));
    }

    #[tokio::test]
    async fn test_ask_question_delivery_channel() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);
        let issue = Issue::new("1", "LIN-1", "Test", "https://example.com", "linear");
        let request = AskRequest {
            correlation_id: "tok-ch".to_string(),
            source: "linear".to_string(),
            repo: None,
            issue_id: "1".to_string(),
            short_id: "LIN-1".to_string(),
            question: claudear_core::types::BlockingQuestion {
                question: "Question?".to_string(),
                context: None,
                options: vec![],
                why: None,
            },
            asked_at: chrono::Utc::now(),
            target_discord_id: None,
            target_email: None,
            target_slack_id: None,
        };
        let delivery = notifier
            .ask_question(&issue, &request)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(delivery.channel, "discord");
        assert!(delivery.message_id.is_none());
    }

    // --- Additional tests for coverage ---

    fn make_ask_request(
        correlation_id: &str,
        question: &str,
        context: Option<&str>,
        options: Vec<&str>,
        why: Option<&str>,
        target_discord_id: Option<&str>,
    ) -> AskRequest {
        AskRequest {
            correlation_id: correlation_id.to_string(),
            source: "linear".to_string(),
            repo: None,
            issue_id: "1".to_string(),
            short_id: "LIN-1".to_string(),
            question: claudear_core::types::BlockingQuestion {
                question: question.to_string(),
                context: context.map(|s| s.to_string()),
                options: options.into_iter().map(|s| s.to_string()).collect(),
                why: why.map(|s| s.to_string()),
            },
            asked_at: chrono::Utc::now(),
            target_discord_id: target_discord_id.map(|s| s.to_string()),
            target_email: None,
            target_slack_id: None,
        }
    }

    #[tokio::test]
    async fn test_notify_start_sentry_source_uses_red_circle_emoji() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);
        let issue = Issue::new(
            "1",
            "SENTRY-1",
            "Sentry Error",
            "https://sentry.io/1",
            "sentry",
        );

        notifier.notify_start(&issue).await.unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        let title = body["embeds"][0]["title"].as_str().unwrap();
        assert!(title.contains("\u{1F534}"));
        assert!(title.contains("SENTRY-1"));
    }

    #[tokio::test]
    async fn test_notify_start_github_source_uses_octopus_emoji() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);
        let issue = Issue::new(
            "1",
            "GH-42",
            "GitHub Issue",
            "https://github.com/1",
            "github",
        );

        notifier.notify_start(&issue).await.unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        let title = body["embeds"][0]["title"].as_str().unwrap();
        assert!(title.contains("\u{1F419}"));
    }

    #[tokio::test]
    async fn test_notify_start_jira_source_uses_ticket_emoji() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);
        let issue = Issue::new("1", "JIRA-99", "Jira Ticket", "https://jira.com/1", "jira");

        notifier.notify_start(&issue).await.unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        let title = body["embeds"][0]["title"].as_str().unwrap();
        assert!(title.contains("\u{1F3AB}"));
    }

    #[tokio::test]
    async fn test_notify_start_unknown_source_uses_pushpin_emoji() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);
        let issue = Issue::new("1", "X-1", "Unknown", "https://example.com", "custom");

        notifier.notify_start(&issue).await.unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        let title = body["embeds"][0]["title"].as_str().unwrap();
        assert!(title.contains("\u{1F4CC}"));
    }

    #[tokio::test]
    async fn test_notify_start_embed_has_priority_and_status_fields() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);
        let issue = Issue::new("1", "P-1", "Test", "https://example.com", "linear");

        notifier.notify_start(&issue).await.unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        let fields = body["embeds"][0]["fields"].as_array().unwrap();
        let priority_field = fields.iter().find(|f| f["name"] == "Priority").unwrap();
        assert_eq!(priority_field["value"], "none");
        let status_field = fields.iter().find(|f| f["name"] == "Status").unwrap();
        assert_eq!(status_field["value"], "open");
    }

    #[tokio::test]
    async fn test_notify_start_blue_color() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);
        let issue = Issue::new("1", "P-1", "Test", "https://example.com", "linear");

        notifier.notify_start(&issue).await.unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        assert_eq!(body["embeds"][0]["color"], 0x3498db);
    }

    #[tokio::test]
    async fn test_notify_start_no_mention_when_no_user_id() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);
        let issue = Issue::new("1", "P-1", "Test", "https://example.com", "linear");

        notifier.notify_start(&issue).await.unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        assert!(body["content"].is_null());
    }

    #[tokio::test]
    async fn test_notify_success_with_user_mention() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config_with_user(), mock);
        let issue = Issue::new("1", "P-1", "Test", "https://example.com", "linear");

        notifier
            .notify_success(&issue, "https://github.com/pr/1")
            .await
            .unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        let content = body["content"].as_str().unwrap();
        assert!(content.contains("<@987654321>"));
        assert_eq!(content, "<@987654321>");
    }

    #[tokio::test]
    async fn test_notify_success_embed_fields_contain_source_and_issue_link() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);
        let issue = Issue::new(
            "1",
            "LIN-5",
            "Fix bug",
            "https://linear.app/issue/5",
            "linear",
        );

        notifier
            .notify_success(&issue, "https://github.com/org/repo/pull/99")
            .await
            .unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        let fields = body["embeds"][0]["fields"].as_array().unwrap();

        let source_field = fields.iter().find(|f| f["name"] == "Source").unwrap();
        assert!(source_field["value"].as_str().unwrap().contains("linear"));

        let issue_field = fields.iter().find(|f| f["name"] == "Issue").unwrap();
        let issue_val = issue_field["value"].as_str().unwrap();
        assert!(issue_val.contains("LIN-5"));
        assert!(issue_val.contains("https://linear.app/issue/5"));

        let pr_field = fields.iter().find(|f| f["name"] == "PR Link").unwrap();
        let pr_val = pr_field["value"].as_str().unwrap();
        assert!(pr_val.contains("https://github.com/org/repo/pull/99"));
    }

    #[tokio::test]
    async fn test_notify_success_no_content_when_no_user() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);
        let issue = Issue::new("1", "P-1", "Test", "https://example.com", "linear");

        notifier
            .notify_success(&issue, "https://github.com/pr/1")
            .await
            .unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        assert!(body["content"].is_null());
    }

    #[tokio::test]
    async fn test_notify_completed_with_user_mention() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config_with_user(), mock);
        let issue = Issue::new("1", "P-1", "Test", "https://example.com", "linear");

        notifier.notify_completed(&issue).await.unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        let content = body["content"].as_str().unwrap();
        assert!(content.contains("<@987654321>"));
        assert_eq!(content, "<@987654321>");
    }

    #[tokio::test]
    async fn test_notify_completed_has_reason_field() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);
        let issue = Issue::new("1", "P-1", "Test", "https://example.com", "linear");

        notifier.notify_completed(&issue).await.unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        let fields = body["embeds"][0]["fields"].as_array().unwrap();
        let reason_field = fields.iter().find(|f| f["name"] == "Reason").unwrap();
        assert!(reason_field["value"]
            .as_str()
            .unwrap()
            .contains("no PR URL was captured"));
    }

    #[tokio::test]
    async fn test_notify_completed_has_source_field_with_emoji() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);
        let issue = Issue::new("1", "P-1", "Test", "https://example.com", "sentry");

        notifier.notify_completed(&issue).await.unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        let fields = body["embeds"][0]["fields"].as_array().unwrap();
        let source_field = fields.iter().find(|f| f["name"] == "Source").unwrap();
        let val = source_field["value"].as_str().unwrap();
        assert!(val.contains("\u{1F534}"));
        assert!(val.contains("sentry"));
    }

    #[tokio::test]
    async fn test_notify_failed_with_user_mention() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config_with_user(), mock);
        let issue = Issue::new("1", "P-1", "Test", "https://example.com", "linear");

        notifier.notify_failed(&issue, "oops").await.unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        let content = body["content"].as_str().unwrap();
        assert!(content.contains("<@987654321>"));
        assert_eq!(content, "<@987654321>");
    }

    #[tokio::test]
    async fn test_notify_failed_has_source_field() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);
        let issue = Issue::new("1", "P-1", "Test", "https://example.com", "github");

        notifier.notify_failed(&issue, "error").await.unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        let fields = body["embeds"][0]["fields"].as_array().unwrap();
        let source_field = fields.iter().find(|f| f["name"] == "Source").unwrap();
        let val = source_field["value"].as_str().unwrap();
        assert!(val.contains("\u{1F419}"));
        assert!(val.contains("github"));
    }

    #[tokio::test]
    async fn test_notify_status_no_content_field() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);

        notifier.notify_status("All clear").await.unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        assert!(body["content"].is_null());
    }

    #[tokio::test]
    async fn test_notify_status_truncates_long_message() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);
        let long_msg = "z".repeat(3000);

        notifier.notify_status(&long_msg).await.unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        let desc = body["embeds"][0]["description"].as_str().unwrap();
        assert!(desc.len() <= MAX_DESCRIPTION_LENGTH);
        assert!(desc.ends_with("..."));
    }

    #[tokio::test]
    async fn test_notify_status_purple_color() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);

        notifier.notify_status("test").await.unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        assert_eq!(body["embeds"][0]["color"], 0x9b59b6);
    }

    #[tokio::test]
    async fn test_notify_urgent_issues_mixed_sources() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);
        let issues = vec![
            Issue::new("1", "LIN-1", "Linear Bug", "https://linear.app/1", "linear"),
            Issue::new(
                "2",
                "SEN-2",
                "Sentry Error",
                "https://sentry.io/2",
                "sentry",
            ),
            Issue::new(
                "3",
                "GH-3",
                "GitHub Issue",
                "https://github.com/3",
                "github",
            ),
        ];

        notifier.notify_urgent_issues(&issues).await.unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        let fields = body["embeds"][0]["fields"].as_array().unwrap();
        assert_eq!(fields.len(), 3);

        let field0_name = fields[0]["name"].as_str().unwrap();
        assert!(field0_name.contains("\u{1F4CB}")); // linear clipboard
        let field1_name = fields[1]["name"].as_str().unwrap();
        assert!(field1_name.contains("\u{1F534}")); // sentry red circle
        let field2_name = fields[2]["name"].as_str().unwrap();
        assert!(field2_name.contains("\u{1F419}")); // github octopus
    }

    #[tokio::test]
    async fn test_notify_urgent_issues_description_text() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);
        let issues = vec![Issue::new(
            "1",
            "P-1",
            "Test",
            "https://example.com",
            "linear",
        )];

        notifier.notify_urgent_issues(&issues).await.unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        let desc = body["embeds"][0]["description"].as_str().unwrap();
        assert!(desc.contains("require attention"));
    }

    #[tokio::test]
    async fn test_send_boundary_status_199_is_error() {
        let mock = MockDiscordWebhookClient::new(199, "Not OK");
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);

        let result = notifier.notify_status("test").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_send_boundary_status_200_is_success() {
        let mock = MockDiscordWebhookClient::new(200, "OK");
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);

        let result = notifier.notify_status("test").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_send_boundary_status_299_is_success() {
        let mock = MockDiscordWebhookClient::new(299, "OK");
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);

        let result = notifier.notify_status("test").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_send_boundary_status_300_is_error() {
        let mock = MockDiscordWebhookClient::new(300, "Redirect");
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);

        let result = notifier.notify_status("test").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_ask_question_without_mention_no_user() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);
        let issue = Issue::new("1", "LIN-1", "Test", "https://example.com", "linear");
        let request = make_ask_request("tok-1", "Which branch?", None, vec![], None, None);

        notifier.ask_question(&issue, &request).await.unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        // No mention and no correlation tag => content field is absent
        assert!(body.get("content").is_none() || body["content"].is_null());
        let embed = &body["embeds"][0];
        assert_eq!(embed["description"].as_str().unwrap(), "Which branch?");
    }

    #[tokio::test]
    async fn test_ask_question_with_user_mention() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config_with_user(), mock);
        let issue = Issue::new("1", "LIN-1", "Test", "https://example.com", "linear");
        let request = make_ask_request("tok-2", "Pick env?", None, vec![], None, None);

        notifier.ask_question(&issue, &request).await.unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        let content = body["content"].as_str().unwrap();
        assert!(content.starts_with("<@987654321>"));
    }

    #[tokio::test]
    async fn test_ask_question_with_why_field() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);
        let issue = Issue::new("1", "LIN-1", "Test", "https://example.com", "linear");
        let request = make_ask_request(
            "tok-3",
            "Which DB?",
            None,
            vec![],
            Some("Multiple databases found"),
            None,
        );

        notifier.ask_question(&issue, &request).await.unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        let embed = &body["embeds"][0];
        let fields = embed["fields"].as_array().unwrap();
        assert!(fields
            .iter()
            .any(|f| f["name"] == "Why" && f["value"] == "Multiple databases found"));
    }

    #[tokio::test]
    async fn test_ask_question_with_context_field() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);
        let issue = Issue::new("1", "LIN-1", "Test", "https://example.com", "linear");
        let request = make_ask_request(
            "tok-4",
            "Which target?",
            Some("The repo has multiple deploy targets"),
            vec![],
            None,
            None,
        );

        notifier.ask_question(&issue, &request).await.unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        let embed = &body["embeds"][0];
        let fields = embed["fields"].as_array().unwrap();
        assert!(fields.iter().any(
            |f| f["name"] == "Context" && f["value"] == "The repo has multiple deploy targets"
        ));
    }

    #[tokio::test]
    async fn test_ask_question_truncates_long_context() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);
        let issue = Issue::new("1", "LIN-1", "Test", "https://example.com", "linear");
        let long_context = "x".repeat(600);
        let request = make_ask_request(
            "tok-5",
            "Question?",
            Some(&long_context),
            vec![],
            None,
            None,
        );

        notifier.ask_question(&issue, &request).await.unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        let embed = &body["embeds"][0];
        let fields = embed["fields"].as_array().unwrap();
        let ctx_field = fields.iter().find(|f| f["name"] == "Context").unwrap();
        let ctx_value = ctx_field["value"].as_str().unwrap();
        assert!(ctx_value.contains("..."));
        assert!(ctx_value.len() <= 403); // 400 chars + "..."
    }

    #[tokio::test]
    async fn test_ask_question_with_options() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);
        let issue = Issue::new("1", "LIN-1", "Test", "https://example.com", "linear");
        let request = make_ask_request(
            "tok-6",
            "Pick one",
            None,
            vec!["alpha", "beta", "gamma"],
            None,
            None,
        );

        notifier.ask_question(&issue, &request).await.unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        let embed = &body["embeds"][0];
        let fields = embed["fields"].as_array().unwrap();
        let opts_field = fields.iter().find(|f| f["name"] == "Options").unwrap();
        let opts_value = opts_field["value"].as_str().unwrap();
        assert!(opts_value.contains("alpha"));
        assert!(opts_value.contains("beta"));
        assert!(opts_value.contains("gamma"));
    }

    #[tokio::test]
    async fn test_ask_question_empty_options_not_shown() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);
        let issue = Issue::new("1", "LIN-1", "Test", "https://example.com", "linear");
        let request = make_ask_request("tok-7", "Free text answer?", None, vec![], None, None);

        notifier.ask_question(&issue, &request).await.unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        let embeds = body["embeds"].as_array().unwrap();
        let embed = &embeds[0];
        // No fields when no options/why/context
        assert!(embed.get("fields").is_none() || embed["fields"].is_null());
    }

    #[tokio::test]
    async fn test_ask_question_uses_embed_format() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);
        let issue = Issue::new("1", "LIN-1", "Test", "https://example.com", "linear");
        let request = make_ask_request("tok-8", "Confirm?", None, vec![], None, None);

        notifier.ask_question(&issue, &request).await.unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        // No mention and no correlation tag => content field is absent
        assert!(body.get("content").is_none() || body["content"].is_null());
        let embeds = body["embeds"].as_array().unwrap();
        assert_eq!(embeds.len(), 1);
        assert!(embeds[0]["title"].as_str().unwrap().contains("LIN-1"));
        assert_eq!(embeds[0]["description"].as_str().unwrap(), "Confirm?");
    }

    #[tokio::test]
    async fn test_ask_question_disabled_webhook_returns_error() {
        let config = DiscordConfig {
            webhook_url: None,
            user_id: None,
            ..Default::default()
        };
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(config, mock);
        let issue = Issue::new("1", "LIN-1", "Test", "https://example.com", "linear");
        let request = make_ask_request("tok-9", "Confirm?", None, vec![], None, None);

        let result = notifier.ask_question(&issue, &request).await;
        assert!(result.is_err());
        let err_str = result.unwrap_err().to_string();
        assert!(err_str.contains("No delivery path configured"));
    }

    #[tokio::test]
    async fn test_poll_question_replies_no_bot_token_returns_empty() {
        let config = DiscordConfig {
            webhook_url: Some("https://discord.com/api/webhooks/123/abc".into()),
            user_id: None,
            bot_token: None,
            channel_id: Some("channel-123".to_string()),
            ..Default::default()
        };
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(config, mock);
        let request = make_ask_request("tok-10", "Question?", None, vec![], None, None);

        let replies = notifier
            .poll_question_replies(&request, chrono::Utc::now())
            .await
            .unwrap();
        assert!(replies.is_empty());
    }

    #[tokio::test]
    async fn test_poll_question_replies_empty_bot_token_returns_empty() {
        let config = DiscordConfig {
            webhook_url: Some("https://discord.com/api/webhooks/123/abc".into()),
            user_id: None,
            bot_token: Some("".into()),
            channel_id: Some("channel-123".to_string()),
            ..Default::default()
        };
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(config, mock);
        let request = make_ask_request("tok-11", "Question?", None, vec![], None, None);

        let replies = notifier
            .poll_question_replies(&request, chrono::Utc::now())
            .await
            .unwrap();
        assert!(replies.is_empty());
    }

    #[tokio::test]
    async fn test_poll_question_replies_no_channel_id_returns_empty() {
        let config = DiscordConfig {
            webhook_url: Some("https://discord.com/api/webhooks/123/abc".into()),
            user_id: None,
            bot_token: Some("valid-token".into()),
            channel_id: None,
            ..Default::default()
        };
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(config, mock);
        let request = make_ask_request("tok-12", "Question?", None, vec![], None, None);

        let replies = notifier
            .poll_question_replies(&request, chrono::Utc::now())
            .await
            .unwrap();
        assert!(replies.is_empty());
    }

    #[tokio::test]
    async fn test_poll_question_replies_empty_channel_id_returns_empty() {
        let config = DiscordConfig {
            webhook_url: Some("https://discord.com/api/webhooks/123/abc".into()),
            user_id: None,
            bot_token: Some("valid-token".into()),
            channel_id: Some("".to_string()),
            ..Default::default()
        };
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(config, mock);
        let request = make_ask_request("tok-13", "Question?", None, vec![], None, None);

        let replies = notifier
            .poll_question_replies(&request, chrono::Utc::now())
            .await
            .unwrap();
        assert!(replies.is_empty());
    }

    #[test]
    fn test_get_target_discord_id_for_issue_with_resolved_user() {
        let mut users = std::collections::HashMap::new();
        users.insert(
            "alice".to_string(),
            claudear_config::config::UserConfig {
                discord_id: Some("alice-discord-id".to_string()),
                ..Default::default()
            },
        );
        let registry = claudear_config::users::UserRegistry::new(users);
        let config = DiscordConfig {
            webhook_url: Some("https://example.com".into()),
            user_id: Some("global-id".to_string()),
            ..Default::default()
        };
        let notifier = DiscordNotifier::with_http_client_and_registry(
            config,
            MockDiscordWebhookClient::success(),
            registry,
        );
        let mut issue = Issue::new("1", "P-1", "Test", "https://example.com", "linear");
        issue.set_metadata("resolved_user", "alice");

        assert_eq!(
            notifier.get_target_discord_id_for_issue(&issue),
            Some("alice-discord-id".to_string())
        );
    }

    #[test]
    fn test_get_target_discord_id_for_issue_fallback_to_global() {
        let config = DiscordConfig {
            webhook_url: Some("https://example.com".into()),
            user_id: Some("global-id".to_string()),
            ..Default::default()
        };
        let notifier =
            DiscordNotifier::with_http_client(config, MockDiscordWebhookClient::success());
        let issue = Issue::new("1", "P-1", "Test", "https://example.com", "linear");

        assert_eq!(
            notifier.get_target_discord_id_for_issue(&issue),
            Some("global-id".to_string())
        );
    }

    #[test]
    fn test_get_target_discord_id_for_issue_resolved_user_no_discord() {
        let mut users = std::collections::HashMap::new();
        users.insert(
            "bob".to_string(),
            claudear_config::config::UserConfig {
                discord_id: None,
                ..Default::default()
            },
        );
        let registry = claudear_config::users::UserRegistry::new(users);
        let config = DiscordConfig {
            webhook_url: Some("https://example.com".into()),
            user_id: Some("global-id".to_string()),
            ..Default::default()
        };
        let notifier = DiscordNotifier::with_http_client_and_registry(
            config,
            MockDiscordWebhookClient::success(),
            registry,
        );
        let mut issue = Issue::new("1", "P-1", "Test", "https://example.com", "linear");
        issue.set_metadata("resolved_user", "bob");

        assert_eq!(
            notifier.get_target_discord_id_for_issue(&issue),
            Some("global-id".to_string())
        );
    }

    #[test]
    fn test_get_target_discord_id_for_issue_no_user_at_all() {
        let config = DiscordConfig {
            webhook_url: Some("https://example.com".into()),
            user_id: None,
            ..Default::default()
        };
        let notifier =
            DiscordNotifier::with_http_client(config, MockDiscordWebhookClient::success());
        let issue = Issue::new("1", "P-1", "Test", "https://example.com", "linear");

        assert_eq!(notifier.get_target_discord_id_for_issue(&issue), None);
    }

    #[tokio::test]
    async fn test_notify_start_truncates_long_short_id() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);
        let long_id = "X".repeat(200);
        let issue = Issue::new("1", &long_id, "Test", "https://example.com", "linear");

        notifier.notify_start(&issue).await.unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        let title = body["embeds"][0]["title"].as_str().unwrap();
        assert!(title.len() < 200);
    }

    #[tokio::test]
    async fn test_notify_start_truncates_long_description() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);
        let long_title = "D".repeat(3000);
        let issue = Issue::new("1", "P-1", &long_title, "https://example.com", "linear");

        notifier.notify_start(&issue).await.unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        let desc = body["embeds"][0]["description"].as_str().unwrap();
        assert!(desc.len() <= MAX_DESCRIPTION_LENGTH);
        assert!(desc.ends_with("..."));
    }

    #[tokio::test]
    async fn test_ask_question_with_all_fields() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config_with_user(), mock);
        let issue = Issue::new("1", "LIN-1", "Test", "https://example.com", "linear");
        let request = make_ask_request(
            "tok-all",
            "Select deployment target",
            Some("Found staging and prod"),
            vec!["staging", "production"],
            Some("Need to know before PR"),
            None,
        );

        let delivery = notifier
            .ask_question(&issue, &request)
            .await
            .unwrap()
            .unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        let content = body["content"].as_str().unwrap();
        assert!(content.contains("<@987654321>"));
        assert!(!content.contains("[CLAUDEAR-Q:"));
        let embed = &body["embeds"][0];
        assert_eq!(
            embed["description"].as_str().unwrap(),
            "Select deployment target"
        );
        let fields = embed["fields"].as_array().unwrap();
        assert!(fields
            .iter()
            .any(|f| f["name"] == "Why" && f["value"] == "Need to know before PR"));
        assert!(fields
            .iter()
            .any(|f| f["name"] == "Context" && f["value"] == "Found staging and prod"));
        assert!(fields
            .iter()
            .any(|f| f["name"] == "Options" && f["value"].as_str().unwrap().contains("staging")));
        assert_eq!(delivery.channel, "discord");
    }

    #[tokio::test]
    async fn test_notify_success_truncates_long_pr_url() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);
        let issue = Issue::new("1", "P-1", "Test", "https://example.com", "linear");
        let long_pr_url = format!("https://github.com/{}", "a".repeat(2500));

        notifier.notify_success(&issue, &long_pr_url).await.unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        let url = body["embeds"][0]["url"].as_str().unwrap();
        assert!(url.len() <= MAX_URL_LENGTH);
    }

    #[tokio::test]
    async fn test_notify_failed_short_error_not_truncated() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);
        let issue = Issue::new("1", "P-1", "Test", "https://example.com", "linear");

        notifier
            .notify_failed(&issue, "Compilation error on line 42")
            .await
            .unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        let fields = body["embeds"][0]["fields"].as_array().unwrap();
        let error_field = fields.iter().find(|f| f["name"] == "Error").unwrap();
        assert_eq!(
            error_field["value"].as_str().unwrap(),
            "Compilation error on line 42"
        );
    }

    #[test]
    fn test_with_http_client_and_registry_creates_valid_notifier() {
        let mut users = std::collections::HashMap::new();
        users.insert(
            "test".to_string(),
            claudear_config::config::UserConfig {
                discord_id: Some("test-discord".to_string()),
                ..Default::default()
            },
        );
        let registry = claudear_config::users::UserRegistry::new(users);
        let mock = MockDiscordWebhookClient::success();
        let notifier =
            DiscordNotifier::with_http_client_and_registry(enabled_config(), mock, registry);

        assert!(notifier.is_enabled());
        assert_eq!(notifier.name(), "discord");
    }

    #[tokio::test]
    async fn test_send_no_webhook_url_returns_ok_without_calling_http() {
        let mock = MockDiscordWebhookClient::success();
        let config = DiscordConfig {
            webhook_url: None,
            user_id: None,
            ..Default::default()
        };
        let notifier = DiscordNotifier::with_http_client(config, mock);
        let issue = Issue::new("1", "P-1", "Test", "https://example.com", "linear");

        let result = notifier.notify_start(&issue).await;
        assert!(result.is_ok());
        assert_eq!(notifier.http.get_call_count(), 0);
    }

    #[tokio::test]
    async fn test_notify_urgent_issues_fields_have_inline_true() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);
        let issues = vec![Issue::new(
            "1",
            "P-1",
            "Test",
            "https://example.com",
            "linear",
        )];

        notifier.notify_urgent_issues(&issues).await.unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        let fields = body["embeds"][0]["fields"].as_array().unwrap();
        for field in fields {
            assert_eq!(field["inline"], true);
        }
    }

    #[tokio::test]
    async fn test_notify_urgent_issues_field_value_is_markdown_link() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);
        let issues = vec![Issue::new(
            "1",
            "P-1",
            "Fix memory leak",
            "https://example.com/issue/1",
            "linear",
        )];

        notifier.notify_urgent_issues(&issues).await.unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        let fields = body["embeds"][0]["fields"].as_array().unwrap();
        let value = fields[0]["value"].as_str().unwrap();
        assert!(value.starts_with('['));
        assert!(value.contains("]("));
        assert!(value.contains("https://example.com/issue/1"));
    }

    #[tokio::test]
    async fn test_ask_question_target_from_resolved_user_registry() {
        let mut users = std::collections::HashMap::new();
        users.insert(
            "charlie".to_string(),
            claudear_config::config::UserConfig {
                discord_id: Some("charlie-discord".to_string()),
                ..Default::default()
            },
        );
        let registry = claudear_config::users::UserRegistry::new(users);
        let config = DiscordConfig {
            webhook_url: Some("https://discord.com/api/webhooks/123/abc".into()),
            user_id: Some("fallback-id".to_string()),
            ..Default::default()
        };
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client_and_registry(config, mock, registry);
        let mut issue = Issue::new("1", "LIN-1", "Test", "https://example.com", "linear");
        issue.set_metadata("resolved_user", "charlie");
        let request = make_ask_request("tok-resolved", "Confirm?", None, vec![], None, None);

        let delivery = notifier
            .ask_question(&issue, &request)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(delivery.target.as_deref(), Some("charlie-discord"));
    }

    #[tokio::test]
    async fn test_ask_question_http_error_propagates() {
        let mock = MockDiscordWebhookClient::error(500, "Server Error");
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);
        let issue = Issue::new("1", "LIN-1", "Test", "https://example.com", "linear");
        let request = make_ask_request("tok-err", "Confirm?", None, vec![], None, None);

        let result = notifier.ask_question(&issue, &request).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_truncate_string_at_char_boundary_with_multibyte() {
        let s = "abcdefghij\u{00E9}klm"; // e-acute is 2 bytes
        let result = truncate_string(s, 12);
        assert!(result.len() <= 12);
        assert!(result.ends_with("..."));
    }

    #[test]
    fn test_truncate_string_emoji_boundary() {
        let s = "hello \u{1F600} world"; // grinning face is 4 bytes
        let result = truncate_string(s, 10);
        assert!(result.len() <= 10);
    }

    #[tokio::test]
    async fn test_notify_start_embed_url_matches_issue_url() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);
        let issue = Issue::new(
            "1",
            "P-1",
            "Test",
            "https://linear.app/team/issue/P-1",
            "linear",
        );

        notifier.notify_start(&issue).await.unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        let url = body["embeds"][0]["url"].as_str().unwrap();
        assert_eq!(url, "https://linear.app/team/issue/P-1");
    }

    #[tokio::test]
    async fn test_notify_completed_embed_url_matches_issue_url() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);
        let issue = Issue::new(
            "1",
            "P-1",
            "Test",
            "https://linear.app/team/issue/P-1",
            "linear",
        );

        notifier.notify_completed(&issue).await.unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        let url = body["embeds"][0]["url"].as_str().unwrap();
        assert_eq!(url, "https://linear.app/team/issue/P-1");
    }

    #[tokio::test]
    async fn test_notify_failed_embed_url_matches_issue_url() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);
        let issue = Issue::new(
            "1",
            "P-1",
            "Test",
            "https://linear.app/team/issue/P-1",
            "linear",
        );

        notifier.notify_failed(&issue, "err").await.unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        let url = body["embeds"][0]["url"].as_str().unwrap();
        assert_eq!(url, "https://linear.app/team/issue/P-1");
    }

    #[tokio::test]
    async fn test_notify_status_has_no_title() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);

        notifier.notify_status("All good").await.unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        assert!(body["embeds"][0]["title"].is_null());
    }

    #[tokio::test]
    async fn test_notify_status_has_no_url() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);

        notifier.notify_status("All good").await.unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        assert!(body["embeds"][0]["url"].is_null());
    }

    #[tokio::test]
    async fn test_notify_status_has_no_fields() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);

        notifier.notify_status("All good").await.unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        assert!(body["embeds"][0]["fields"].is_null());
    }

    #[tokio::test]
    async fn test_ask_question_minimal_has_embed_with_no_fields() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);
        let issue = Issue::new("1", "LIN-1", "Test", "https://example.com", "linear");
        let request = make_ask_request("tok-no-fields", "Confirm?", None, vec![], None, None);

        notifier.ask_question(&issue, &request).await.unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        let embed = &body["embeds"][0];
        assert_eq!(embed["description"].as_str().unwrap(), "Confirm?");
        // No why/context/options → fields should be null or empty
        assert!(embed["fields"].is_null() || embed["fields"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_notify_urgent_issues_orange_color() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);
        let issues = vec![Issue::new(
            "1",
            "P-1",
            "Test",
            "https://example.com",
            "linear",
        )];

        notifier.notify_urgent_issues(&issues).await.unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        assert_eq!(body["embeds"][0]["color"], 0xf39c12);
    }

    #[tokio::test]
    async fn test_notify_urgent_issues_no_url_in_embed() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);
        let issues = vec![Issue::new(
            "1",
            "P-1",
            "Test",
            "https://example.com",
            "linear",
        )];

        notifier.notify_urgent_issues(&issues).await.unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        assert!(body["embeds"][0]["url"].is_null());
    }

    // --- Synchronous tests for standalone build_* helpers ---

    fn test_issue() -> Issue {
        Issue::new(
            "42",
            "PROJ-42",
            "Fix the widget",
            "https://example.com/issue/42",
            "linear",
        )
    }

    #[test]
    fn test_build_start_message_with_mention() {
        let issue = test_issue();
        let msg = build_start_message(&issue, Some("<@12345>".to_string()));

        assert_eq!(msg.content.as_deref(), Some("<@12345>"));
        let embeds = msg.embeds.as_ref().unwrap();
        assert_eq!(embeds.len(), 1);
        let embed = &embeds[0];
        assert!(embed
            .title
            .as_ref()
            .unwrap()
            .contains("Processing: PROJ-42"));
        assert_eq!(embed.description.as_deref(), Some("Fix the widget"));
        assert_eq!(embed.url.as_deref(), Some("https://example.com/issue/42"));
        assert_eq!(embed.color, Some(0x3498db));
        let fields = embed.fields.as_ref().unwrap();
        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0].name, "Source");
        assert_eq!(fields[0].value, "linear");
        assert_eq!(fields[1].name, "Priority");
        assert_eq!(fields[2].name, "Status");
        assert_eq!(embed.footer.as_ref().unwrap().text, "Claudear");
        assert!(embed.timestamp.is_some());
    }

    #[test]
    fn test_build_start_message_without_mention() {
        let issue = test_issue();
        let msg = build_start_message(&issue, None);

        assert!(msg.content.is_none());
        assert!(msg.embeds.is_some());
    }

    #[test]
    fn test_build_success_message_fields() {
        let issue = test_issue();
        let msg = build_success_message(
            &issue,
            "https://github.com/org/repo/pull/99",
            Some("<@user>".to_string()),
        );

        assert_eq!(msg.content.as_deref(), Some("<@user>"));
        let embed = &msg.embeds.as_ref().unwrap()[0];
        assert!(embed
            .title
            .as_ref()
            .unwrap()
            .contains("PR Created: PROJ-42"));
        assert_eq!(embed.color, Some(0x2ecc71));
        assert_eq!(
            embed.url.as_deref(),
            Some("https://github.com/org/repo/pull/99")
        );
        let fields = embed.fields.as_ref().unwrap();
        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0].name, "Source");
        assert!(fields[0].value.contains("linear"));
        assert_eq!(fields[1].name, "Issue");
        assert!(fields[1].value.contains("PROJ-42"));
        assert_eq!(fields[2].name, "PR Link");
        assert!(fields[2]
            .value
            .contains("https://github.com/org/repo/pull/99"));
    }

    #[test]
    fn test_build_success_message_without_mention() {
        let issue = test_issue();
        let msg = build_success_message(&issue, "https://pr.url", None);

        assert!(msg.content.is_none());
    }

    #[test]
    fn test_build_completed_message_fields() {
        let issue = test_issue();
        let msg = build_completed_message(&issue, Some("<@u>".to_string()));

        assert_eq!(msg.content.as_deref(), Some("<@u>"));
        let embed = &msg.embeds.as_ref().unwrap()[0];
        assert!(embed.title.as_ref().unwrap().contains("Completed: PROJ-42"));
        assert_eq!(embed.color, Some(0x9b59b6));
        let fields = embed.fields.as_ref().unwrap();
        assert_eq!(fields.len(), 2);
        assert_eq!(fields[0].name, "Source");
        assert_eq!(fields[1].name, "Reason");
        assert!(fields[1].value.contains("no PR URL was captured"));
    }

    #[test]
    fn test_build_completed_message_without_mention() {
        let issue = test_issue();
        let msg = build_completed_message(&issue, None);

        assert!(msg.content.is_none());
    }

    #[test]
    fn test_build_failed_message_fields() {
        let issue = test_issue();
        let msg = build_failed_message(
            &issue,
            "Build failed with exit code 1",
            Some("<@u>".to_string()),
        );

        assert_eq!(msg.content.as_deref(), Some("<@u>"));
        let embed = &msg.embeds.as_ref().unwrap()[0];
        assert!(embed.title.as_ref().unwrap().contains("Failed: PROJ-42"));
        assert_eq!(embed.color, Some(0xe74c3c));
        let fields = embed.fields.as_ref().unwrap();
        assert_eq!(fields.len(), 2);
        assert_eq!(fields[0].name, "Source");
        assert_eq!(fields[1].name, "Error");
        assert_eq!(fields[1].value, "Build failed with exit code 1");
    }

    #[test]
    fn test_build_failed_message_truncates_long_error() {
        let issue = test_issue();
        let long_error = "x".repeat(2000);
        let msg = build_failed_message(&issue, &long_error, None);

        let fields = msg.embeds.as_ref().unwrap()[0].fields.as_ref().unwrap();
        let error_value = &fields[1].value;
        assert!(error_value.len() <= 1003);
        assert!(error_value.ends_with("..."));
    }

    #[test]
    fn test_build_status_message_fields() {
        let msg = build_status_message("System is healthy");

        assert!(msg.content.is_none());
        let embed = &msg.embeds.as_ref().unwrap()[0];
        assert!(embed.title.is_none());
        assert_eq!(embed.description.as_deref(), Some("System is healthy"));
        assert!(embed.url.is_none());
        assert_eq!(embed.color, Some(0x9b59b6));
        assert!(embed.fields.is_none());
        assert_eq!(embed.footer.as_ref().unwrap().text, "Claudear");
    }

    #[test]
    fn test_build_status_message_truncates_long_text() {
        let long_msg = "z".repeat(3000);
        let msg = build_status_message(&long_msg);

        let desc = msg.embeds.as_ref().unwrap()[0]
            .description
            .as_ref()
            .unwrap();
        assert!(desc.len() <= MAX_DESCRIPTION_LENGTH);
        assert!(desc.ends_with("..."));
    }

    #[test]
    fn test_build_urgent_issues_message_empty_returns_none() {
        let result = build_urgent_issues_message(&[], None);
        assert!(result.is_none());
    }

    #[test]
    fn test_build_urgent_issues_message_single() {
        let issues = vec![test_issue()];
        let msg = build_urgent_issues_message(&issues, Some("<@u>".to_string())).unwrap();

        assert_eq!(msg.content.as_deref(), Some("<@u>"));
        let embed = &msg.embeds.as_ref().unwrap()[0];
        assert!(embed
            .title
            .as_ref()
            .unwrap()
            .contains("1 Urgent Issue Detected"));
        assert!(!embed.title.as_ref().unwrap().contains("Issues"));
        assert_eq!(embed.color, Some(0xf39c12));
        let fields = embed.fields.as_ref().unwrap();
        assert_eq!(fields.len(), 1);
        assert!(fields[0].name.contains("PROJ-42"));
    }

    #[test]
    fn test_build_urgent_issues_message_plural() {
        let issues = vec![
            Issue::new("1", "P-1", "Issue 1", "https://example.com/1", "linear"),
            Issue::new("2", "P-2", "Issue 2", "https://example.com/2", "sentry"),
        ];
        let msg = build_urgent_issues_message(&issues, None).unwrap();

        assert!(msg.content.is_none());
        let title = msg.embeds.as_ref().unwrap()[0].title.as_ref().unwrap();
        assert!(title.contains("2 Urgent Issues Detected"));
    }

    #[test]
    fn test_build_urgent_issues_message_limits_to_ten() {
        let issues: Vec<Issue> = (1..=15)
            .map(|i| {
                Issue::new(
                    i.to_string(),
                    format!("P-{}", i),
                    format!("Issue {}", i),
                    format!("https://example.com/{}", i),
                    "linear",
                )
            })
            .collect();
        let msg = build_urgent_issues_message(&issues, None).unwrap();

        let fields = msg.embeds.as_ref().unwrap()[0].fields.as_ref().unwrap();
        assert_eq!(fields.len(), 10);
    }

    #[test]
    fn test_build_ask_question_message_minimal() {
        let issue = test_issue();
        let request = make_ask_request("tok-1", "Which branch?", None, vec![], None, None);
        let msg = build_ask_question_message(&issue, &request, None);

        // Content should be empty (no mention, no correlation tag)
        assert!(msg.content.is_none() || msg.content.as_ref().unwrap().is_empty());

        let embeds = msg.embeds.as_ref().unwrap();
        assert_eq!(embeds.len(), 1);
        let embed = &embeds[0];
        assert!(embed.title.as_ref().unwrap().contains("PROJ-42"));
        assert_eq!(embed.description.as_ref().unwrap(), "Which branch?");
        assert!(embed.fields.is_none());
    }

    #[test]
    fn test_build_ask_question_message_with_all_fields() {
        let issue = test_issue();
        let request = make_ask_request(
            "tok-all",
            "Select target",
            Some("Found staging and prod"),
            vec!["staging", "production"],
            Some("Need to know before PR"),
            None,
        );
        let msg = build_ask_question_message(&issue, &request, Some("<@987654321>".to_string()));

        let content = msg.content.as_ref().unwrap();
        assert!(content.starts_with("<@987654321>"));
        assert!(!content.contains("[CLAUDEAR-Q:"));

        let embeds = msg.embeds.as_ref().unwrap();
        let embed = &embeds[0];
        assert_eq!(embed.description.as_ref().unwrap(), "Select target");

        let fields = embed.fields.as_ref().unwrap();
        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0].name, "Why");
        assert_eq!(fields[0].value, "Need to know before PR");
        assert_eq!(fields[1].name, "Context");
        assert!(fields[1].value.contains("Found staging and prod"));
        assert_eq!(fields[2].name, "Options");
        assert!(fields[2].value.contains("staging"));
        assert!(fields[2].value.contains("production"));
    }

    #[test]
    fn test_build_ask_question_message_truncates_long_context() {
        let issue = test_issue();
        let long_context = "x".repeat(600);
        let request = make_ask_request(
            "tok-ctx",
            "Question?",
            Some(&long_context),
            vec![],
            None,
            None,
        );
        let msg = build_ask_question_message(&issue, &request, None);

        let embeds = msg.embeds.as_ref().unwrap();
        let fields = embeds[0].fields.as_ref().unwrap();
        let ctx_field = fields.iter().find(|f| f.name == "Context").unwrap();
        assert!(ctx_field.value.contains("..."));
    }

    #[test]
    fn test_to_create_message_params_preserves_content() {
        let msg = DiscordMessage {
            content: Some("Hello world".to_string()),
            embeds: None,
        };
        let params = DiscordNotifier::<MockDiscordWebhookClient>::to_create_message_params(&msg);
        assert_eq!(params.content, "Hello world");
        assert!(params.embeds.is_none());
    }

    #[test]
    fn test_to_create_message_params_defaults_empty_content() {
        let msg = DiscordMessage {
            content: None,
            embeds: None,
        };
        let params = DiscordNotifier::<MockDiscordWebhookClient>::to_create_message_params(&msg);
        assert_eq!(params.content, "");
    }

    #[test]
    fn test_to_create_message_params_preserves_embeds() {
        let msg = DiscordMessage {
            content: Some("text".to_string()),
            embeds: Some(vec![DiscordEmbed {
                title: Some("My Title".to_string()),
                description: Some("My Desc".to_string()),
                url: Some("https://example.com".to_string()),
                color: Some(0xFF0000),
                fields: Some(vec![DiscordField {
                    name: "Field1".to_string(),
                    value: "Value1".to_string(),
                    inline: Some(true),
                }]),
                footer: Some(DiscordFooter {
                    text: "Footer".to_string(),
                }),
                timestamp: Some("2024-01-01T00:00:00Z".to_string()),
            }]),
        };
        let params = DiscordNotifier::<MockDiscordWebhookClient>::to_create_message_params(&msg);
        assert_eq!(params.content, "text");

        let embeds = params.embeds.unwrap();
        assert_eq!(embeds.len(), 1);
        let embed = &embeds[0];
        assert_eq!(embed.title.as_deref(), Some("My Title"));
        assert_eq!(embed.description.as_deref(), Some("My Desc"));
        assert_eq!(embed.url.as_deref(), Some("https://example.com"));
        assert_eq!(embed.color, Some(0xFF0000));
        let fields = embed.fields.as_ref().unwrap();
        assert_eq!(fields.len(), 1);
        assert_eq!(fields[0].name, "Field1");
        assert_eq!(fields[0].value, "Value1");
        assert_eq!(fields[0].inline, Some(true));
        assert_eq!(embed.footer.as_ref().unwrap().text, "Footer");
        assert_eq!(embed.timestamp.as_deref(), Some("2024-01-01T00:00:00Z"));
    }

    #[tokio::test]
    async fn test_send_prefers_webhook_when_both_configured() {
        let mock = MockDiscordWebhookClient::success();
        let config = DiscordConfig {
            webhook_url: Some("https://discord.com/api/webhooks/123/abc".into()),
            user_id: None,
            bot_token: Some("bot-token".into()),
            channel_id: Some("channel-123".to_string()),
            ..Default::default()
        };
        let notifier = DiscordNotifier::with_http_client(config, mock);
        let issue = Issue::new("1", "P-1", "Test", "https://example.com", "linear");

        notifier.notify_start(&issue).await.unwrap();

        // Should have used the webhook (mock was called), not the bot API
        assert_eq!(notifier.http.get_call_count(), 1);
        let (url, _) = notifier.http.get_last_call().unwrap();
        assert_eq!(url, "https://discord.com/api/webhooks/123/abc?wait=true");
    }

    #[test]
    fn test_build_merged_message_basic() {
        let issue = Issue::new(
            "1",
            "BUG-123",
            "Fix login bug",
            "https://example.com/1",
            "github",
        );
        let msg = build_merged_message(&issue, "https://github.com/org/repo/pull/42", None);
        assert!(msg.content.is_none());
        let embed = &msg.embeds.as_ref().unwrap()[0];
        assert!(embed.title.as_ref().unwrap().contains("PR Merged"));
        assert!(embed.title.as_ref().unwrap().contains("BUG-123"));
        assert_eq!(embed.color, Some(0x1abc9c));
        let fields = embed.fields.as_ref().unwrap();
        assert!(fields.iter().any(|f| f.name == "PR Link"));
        assert!(fields.iter().any(|f| f.name == "Source"));
        assert!(fields.iter().any(|f| f.name == "Issue"));
    }

    #[test]
    fn test_build_merged_message_with_mention() {
        let issue = Issue::new("1", "BUG-123", "Test", "https://example.com/1", "linear");
        let msg = build_merged_message(&issue, "https://pr.url", Some("<@user>".to_string()));
        assert_eq!(msg.content, Some("<@user>".to_string()));
    }

    #[test]
    fn test_build_closed_message_basic() {
        let issue = Issue::new(
            "1",
            "BUG-456",
            "Broken feature",
            "https://example.com/1",
            "sentry",
        );
        let msg = build_closed_message(&issue, "https://github.com/org/repo/pull/99", None);
        let embed = &msg.embeds.as_ref().unwrap()[0];
        assert!(embed.title.as_ref().unwrap().contains("PR Closed"));
        assert!(embed.title.as_ref().unwrap().contains("BUG-456"));
        assert_eq!(embed.color, Some(0x95a5a6));
        let fields = embed.fields.as_ref().unwrap();
        assert!(fields
            .iter()
            .any(|f| f.name == "Note" && f.value.contains("closed without merging")));
    }

    #[test]
    fn test_build_closed_message_with_mention() {
        let issue = Issue::new("1", "BUG-456", "Test", "https://example.com/1", "github");
        let msg = build_closed_message(&issue, "https://pr.url", Some("<@admin>".to_string()));
        assert_eq!(msg.content, Some("<@admin>".to_string()));
    }

    #[test]
    fn test_build_cascade_success_message_basic() {
        let mut issue = Issue::new(
            "1",
            "CASCADE-1",
            "Fix cascade",
            "https://example.com/1",
            "github",
        );
        issue.set_metadata("cascade_upstream_repo", "org/upstream");
        issue.set_metadata("cascade_downstream_repo", "org/downstream");
        issue.set_metadata(
            "cascade_upstream_pr_url",
            "https://github.com/org/upstream/pull/5",
        );
        issue.set_metadata("cascade_original_issue_short_id", "LIN-42");
        let msg =
            build_cascade_success_message(&issue, "https://github.com/org/downstream/pull/1", None);
        let embed = &msg.embeds.as_ref().unwrap()[0];
        assert!(embed.title.as_ref().unwrap().contains("Cascade PR"));
        assert_eq!(embed.color, Some(0x3498db));
        let fields = embed.fields.as_ref().unwrap();
        assert!(fields
            .iter()
            .any(|f| f.name == "Upstream" && f.value == "org/upstream"));
        assert!(fields
            .iter()
            .any(|f| f.name == "Downstream" && f.value == "org/downstream"));
        assert!(fields
            .iter()
            .any(|f| f.name == "Original Issue" && f.value == "LIN-42"));
        assert!(fields
            .iter()
            .any(|f| f.name == "Upstream PR" && f.value.contains("org/upstream/pull/5")));
        assert!(fields.iter().any(|f| f.name == "Cascade PR"));
        assert!(embed.footer.as_ref().unwrap().text.contains("Cascade"));
    }

    #[test]
    fn test_build_cascade_success_message_no_metadata() {
        let issue = Issue::new(
            "1",
            "CASCADE-1",
            "Fix cascade",
            "https://example.com/1",
            "github",
        );
        let msg = build_cascade_success_message(&issue, "https://pr.url", None);
        let embed = &msg.embeds.as_ref().unwrap()[0];
        // Should not panic, upstream/downstream default to empty strings
        let fields = embed.fields.as_ref().unwrap();
        assert!(fields.iter().any(|f| f.name == "Upstream"));
        // No Original Issue or Upstream PR fields when metadata is missing
        assert!(!fields.iter().any(|f| f.name == "Original Issue"));
        assert!(!fields.iter().any(|f| f.name == "Upstream PR"));
    }

    #[test]
    fn test_build_cascade_failed_message_basic() {
        let mut issue = Issue::new(
            "1",
            "CASCADE-2",
            "Fail cascade",
            "https://example.com/1",
            "github",
        );
        issue.set_metadata("cascade_upstream_repo", "org/upstream");
        issue.set_metadata("cascade_downstream_repo", "org/downstream");
        issue.set_metadata(
            "cascade_upstream_pr_url",
            "https://github.com/org/upstream/pull/5",
        );
        issue.set_metadata("cascade_original_issue_short_id", "LIN-42");
        let msg = build_cascade_failed_message(&issue, "build compilation failed", None);
        let embed = &msg.embeds.as_ref().unwrap()[0];
        assert!(embed.title.as_ref().unwrap().contains("Cascade Failed"));
        assert_eq!(embed.color, Some(0xe67e22));
        assert!(embed.url.is_none());
        let fields = embed.fields.as_ref().unwrap();
        assert!(fields
            .iter()
            .any(|f| f.name == "Error" && f.value.contains("build compilation failed")));
        assert!(fields
            .iter()
            .any(|f| f.name == "Original Issue" && f.value == "LIN-42"));
        assert!(fields
            .iter()
            .any(|f| f.name == "Upstream PR" && f.value.contains("org/upstream/pull/5")));
    }

    #[test]
    fn test_build_cascade_failed_message_with_mention() {
        let issue = Issue::new("1", "CASCADE-2", "Test", "https://example.com/1", "github");
        let msg = build_cascade_failed_message(&issue, "error", Some("<@dev>".to_string()));
        assert_eq!(msg.content, Some("<@dev>".to_string()));
    }

    #[test]
    fn test_build_regression_detected_message_basic() {
        let issue = Issue::new(
            "1",
            "REG-1",
            "Login broken",
            "https://example.com/1",
            "sentry",
        );
        let msg = build_regression_detected_message(&issue, "Login endpoint returns 500", None);
        let embed = &msg.embeds.as_ref().unwrap()[0];
        assert!(embed
            .title
            .as_ref()
            .unwrap()
            .contains("Regression Detected"));
        assert!(embed.title.as_ref().unwrap().contains("REG-1"));
        assert_eq!(embed.color, Some(0xe74c3c));
        assert!(embed.description.as_ref().unwrap().contains("regressed"));
        let fields = embed.fields.as_ref().unwrap();
        assert!(fields
            .iter()
            .any(|f| f.name == "Details" && f.value.contains("Login endpoint")));
        assert!(fields
            .iter()
            .any(|f| f.name == "Action" && f.value.contains("scheduled")));
        assert!(embed
            .footer
            .as_ref()
            .unwrap()
            .text
            .contains("Regression Monitor"));
    }

    #[test]
    fn test_build_regression_detected_message_empty_url() {
        let issue = Issue::new("1", "REG-1", "Test", "", "sentry");
        let msg = build_regression_detected_message(&issue, "error", None);
        let embed = &msg.embeds.as_ref().unwrap()[0];
        assert!(embed.url.is_none());
    }

    #[test]
    fn test_build_regression_detected_message_with_url() {
        let issue = Issue::new("1", "REG-1", "Test", "https://example.com/issue", "sentry");
        let msg = build_regression_detected_message(&issue, "error", None);
        let embed = &msg.embeds.as_ref().unwrap()[0];
        assert!(embed.url.is_some());
    }

    #[test]
    fn test_build_regression_resolved_message_basic() {
        let issue = Issue::new(
            "1",
            "REG-2",
            "Login fixed",
            "https://example.com/2",
            "github",
        );
        let msg = build_regression_resolved_message(&issue, None);
        let embed = &msg.embeds.as_ref().unwrap()[0];
        assert!(embed
            .title
            .as_ref()
            .unwrap()
            .contains("Regression Resolved"));
        assert!(embed.title.as_ref().unwrap().contains("REG-2"));
        assert_eq!(embed.color, Some(0x2ecc71));
        assert!(embed
            .description
            .as_ref()
            .unwrap()
            .contains("No regression detected"));
        let fields = embed.fields.as_ref().unwrap();
        assert!(fields.iter().any(|f| f.name == "Status"));
        assert!(embed
            .footer
            .as_ref()
            .unwrap()
            .text
            .contains("Regression Monitor"));
    }

    #[test]
    fn test_build_regression_resolved_message_with_mention() {
        let issue = Issue::new("1", "REG-2", "Test", "https://example.com/2", "github");
        let msg = build_regression_resolved_message(&issue, Some("<@user>".to_string()));
        assert_eq!(msg.content, Some("<@user>".to_string()));
    }

    #[test]
    fn test_build_regression_resolved_message_empty_url() {
        let issue = Issue::new("1", "REG-2", "Test", "", "github");
        let msg = build_regression_resolved_message(&issue, None);
        let embed = &msg.embeds.as_ref().unwrap()[0];
        assert!(embed.url.is_none());
    }

    #[test]
    fn test_build_urgent_issues_message_description_content() {
        let issues = vec![Issue::new(
            "1",
            "URG-1",
            "Critical",
            "https://example.com/1",
            "sentry",
        )];
        let msg = build_urgent_issues_message(&issues, None).unwrap();
        let embed = &msg.embeds.as_ref().unwrap()[0];
        assert!(embed
            .description
            .as_ref()
            .unwrap()
            .contains("require attention"));
    }

    // --- Tests for notify_merged message building ---

    #[tokio::test]
    async fn test_notify_merged_sends_correct_embed() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);
        let issue = Issue::new("1", "PROJ-1", "Fix bug", "https://example.com/1", "linear");

        notifier
            .notify_merged(&issue, "https://github.com/org/repo/pull/55")
            .await
            .unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        let embed = &body["embeds"][0];
        assert!(embed["title"].as_str().unwrap().contains("PR Merged"));
        assert!(embed["title"].as_str().unwrap().contains("PROJ-1"));
        assert_eq!(embed["color"], 0x1abc9c); // Teal
        assert_eq!(embed["url"], "https://github.com/org/repo/pull/55");
        let fields = embed["fields"].as_array().unwrap();
        let pr_field = fields.iter().find(|f| f["name"] == "PR Link").unwrap();
        assert!(pr_field["value"].as_str().unwrap().contains("View PR"));
        let issue_field = fields.iter().find(|f| f["name"] == "Issue").unwrap();
        assert!(issue_field["value"].as_str().unwrap().contains("PROJ-1"));
    }

    #[tokio::test]
    async fn test_notify_merged_with_user_mention() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config_with_user(), mock);
        let issue = Issue::new("1", "P-1", "Test", "https://example.com", "linear");

        notifier
            .notify_merged(&issue, "https://github.com/pr/1")
            .await
            .unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        let content = body["content"].as_str().unwrap();
        assert!(content.contains("<@987654321>"));
    }

    // --- Tests for notify_closed message building ---

    #[tokio::test]
    async fn test_notify_closed_sends_correct_embed() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);
        let issue = Issue::new("1", "PROJ-1", "Fix bug", "https://example.com/1", "linear");

        notifier
            .notify_closed(&issue, "https://github.com/org/repo/pull/56")
            .await
            .unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        let embed = &body["embeds"][0];
        assert!(embed["title"].as_str().unwrap().contains("PR Closed"));
        assert!(embed["title"].as_str().unwrap().contains("PROJ-1"));
        assert_eq!(embed["color"], 0x95a5a6); // Grey
        assert_eq!(embed["url"], "https://github.com/org/repo/pull/56");
        let fields = embed["fields"].as_array().unwrap();
        let note_field = fields.iter().find(|f| f["name"] == "Note").unwrap();
        assert!(note_field["value"]
            .as_str()
            .unwrap()
            .contains("closed without merging"));
    }

    #[tokio::test]
    async fn test_notify_closed_with_user_mention() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config_with_user(), mock);
        let issue = Issue::new("1", "P-1", "Test", "https://example.com", "linear");

        notifier
            .notify_closed(&issue, "https://github.com/pr/1")
            .await
            .unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        let content = body["content"].as_str().unwrap();
        assert!(content.contains("<@987654321>"));
    }

    // --- Tests for cascade success message ---

    #[tokio::test]
    async fn test_notify_success_cascade_sends_cascade_embed() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);
        let mut issue = Issue::new("1", "LIN-1", "Fix", "https://example.com", "linear");
        issue.set_metadata("cascade_downstream_repo", "downstream/repo");
        issue.set_metadata("cascade_upstream_repo", "upstream/repo");
        issue.set_metadata(
            "cascade_upstream_pr_url",
            "https://github.com/upstream/pr/1",
        );
        issue.set_metadata("cascade_original_issue_short_id", "ORIG-1");

        notifier
            .notify_success(&issue, "https://github.com/downstream/repo/pull/10")
            .await
            .unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        let embed = &body["embeds"][0];
        assert!(embed["title"].as_str().unwrap().contains("Cascade PR"));
        assert_eq!(embed["color"], 0x3498db); // Blue
        let fields = embed["fields"].as_array().unwrap();
        let upstream_field = fields.iter().find(|f| f["name"] == "Upstream").unwrap();
        assert_eq!(upstream_field["value"], "upstream/repo");
        let downstream_field = fields.iter().find(|f| f["name"] == "Downstream").unwrap();
        assert_eq!(downstream_field["value"], "downstream/repo");
        let orig_field = fields
            .iter()
            .find(|f| f["name"] == "Original Issue")
            .unwrap();
        assert_eq!(orig_field["value"], "ORIG-1");
        let upstream_pr = fields.iter().find(|f| f["name"] == "Upstream PR").unwrap();
        assert!(upstream_pr["value"]
            .as_str()
            .unwrap()
            .contains("https://github.com/upstream/pr/1"));
        let cascade_pr = fields.iter().find(|f| f["name"] == "Cascade PR").unwrap();
        assert!(cascade_pr["value"]
            .as_str()
            .unwrap()
            .contains("downstream/repo/pull/10"));
        let footer = embed["footer"]["text"].as_str().unwrap();
        assert!(footer.contains("Cascade"));
    }

    // --- Tests for cascade failed message ---

    #[tokio::test]
    async fn test_notify_failed_cascade_sends_cascade_failed_embed() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);
        let mut issue = Issue::new("1", "LIN-1", "Fix", "https://example.com", "linear");
        issue.set_metadata("cascade_downstream_repo", "downstream/repo");
        issue.set_metadata("cascade_upstream_repo", "upstream/repo");
        issue.set_metadata(
            "cascade_upstream_pr_url",
            "https://github.com/upstream/pr/2",
        );
        issue.set_metadata("cascade_original_issue_short_id", "ORIG-2");

        notifier
            .notify_failed(&issue, "Adaptation failed")
            .await
            .unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        let embed = &body["embeds"][0];
        assert!(embed["title"].as_str().unwrap().contains("Cascade Failed"));
        assert_eq!(embed["color"], 0xe67e22); // Dark Orange
        let desc = embed["description"].as_str().unwrap();
        assert!(desc.contains("Failed to adapt"));
        assert!(desc.contains("downstream/repo"));
        let fields = embed["fields"].as_array().unwrap();
        let error_field = fields.iter().find(|f| f["name"] == "Error").unwrap();
        assert!(error_field["value"]
            .as_str()
            .unwrap()
            .contains("Adaptation failed"));
        let footer = embed["footer"]["text"].as_str().unwrap();
        assert!(footer.contains("Cascade"));
    }

    // --- Tests for regression detected message ---

    #[tokio::test]
    async fn test_notify_failed_regression_sends_regression_embed() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);
        let mut issue = Issue::new("1", "SEN-1", "Error", "https://sentry.io/1", "sentry");
        issue.set_metadata("regression_detected", true);

        notifier
            .notify_failed(&issue, "Tests failing again")
            .await
            .unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        let embed = &body["embeds"][0];
        assert!(embed["title"]
            .as_str()
            .unwrap()
            .contains("Regression Detected"));
        assert_eq!(embed["color"], 0xe74c3c); // Red
        let desc = embed["description"].as_str().unwrap();
        assert!(desc.contains("previously fixed issue has regressed"));
        let fields = embed["fields"].as_array().unwrap();
        let details = fields.iter().find(|f| f["name"] == "Details").unwrap();
        assert!(details["value"]
            .as_str()
            .unwrap()
            .contains("Tests failing again"));
        let action = fields.iter().find(|f| f["name"] == "Action").unwrap();
        assert!(action["value"]
            .as_str()
            .unwrap()
            .contains("Retry has been scheduled"));
        let footer = embed["footer"]["text"].as_str().unwrap();
        assert!(footer.contains("Regression Monitor"));
    }

    // --- Tests for regression resolved message ---

    #[tokio::test]
    async fn test_notify_completed_regression_resolved_sends_resolved_embed() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);
        let mut issue = Issue::new("1", "SEN-1", "Error", "https://sentry.io/1", "sentry");
        issue.set_metadata("regression_resolved", true);

        notifier.notify_completed(&issue).await.unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        let embed = &body["embeds"][0];
        assert!(embed["title"]
            .as_str()
            .unwrap()
            .contains("Regression Resolved"));
        assert_eq!(embed["color"], 0x2ecc71); // Green
        let desc = embed["description"].as_str().unwrap();
        assert!(desc.contains("No regression detected"));
        let footer = embed["footer"]["text"].as_str().unwrap();
        assert!(footer.contains("Regression Monitor"));
    }

    // --- Tests for is_pr_update path in success message ---

    #[tokio::test]
    async fn test_notify_success_pr_update_sends_updated_title() {
        let mock = MockDiscordWebhookClient::success();
        let notifier = DiscordNotifier::with_http_client(enabled_config(), mock);
        let mut issue = Issue::new("1", "LIN-1", "Fix", "https://example.com", "linear");
        issue.set_metadata("is_pr_update", true);

        notifier
            .notify_success(&issue, "https://github.com/org/repo/pull/77")
            .await
            .unwrap();

        let (_, body) = notifier.http.get_last_call().unwrap();
        let embed = &body["embeds"][0];
        assert!(embed["title"].as_str().unwrap().contains("PR Updated"));
        let fields = embed["fields"].as_array().unwrap();
        let pr_field = fields.iter().find(|f| f["name"] == "Updated PR").unwrap();
        assert!(pr_field["value"].as_str().unwrap().contains("View PR"));
    }

    // --- Test to_create_message_params content truncation ---

    #[test]
    fn test_to_create_message_params_truncates_long_content() {
        let long_content = "x".repeat(2500);
        let msg = DiscordMessage {
            content: Some(long_content.clone()),
            embeds: None,
        };
        let params = DiscordNotifier::<MockDiscordWebhookClient>::to_create_message_params(&msg);
        assert!(params.content.len() <= 2000);
        assert!(params.content.ends_with("..."));
    }

    #[test]
    fn test_to_create_message_params_short_content_unchanged() {
        let msg = DiscordMessage {
            content: Some("short message".to_string()),
            embeds: None,
        };
        let params = DiscordNotifier::<MockDiscordWebhookClient>::to_create_message_params(&msg);
        assert_eq!(params.content, "short message");
    }

    #[test]
    fn test_to_create_message_params_none_content_is_empty() {
        let msg = DiscordMessage {
            content: None,
            embeds: None,
        };
        let params = DiscordNotifier::<MockDiscordWebhookClient>::to_create_message_params(&msg);
        assert_eq!(params.content, "");
    }

    #[test]
    fn test_to_create_message_params_with_embeds() {
        let msg = DiscordMessage {
            content: Some("hello".to_string()),
            embeds: Some(vec![DiscordEmbed {
                title: Some("Title".to_string()),
                description: Some("Desc".to_string()),
                url: Some("https://example.com".to_string()),
                color: Some(0xFF0000),
                fields: Some(vec![DiscordField {
                    name: "F1".to_string(),
                    value: "V1".to_string(),
                    inline: Some(true),
                }]),
                footer: Some(DiscordFooter {
                    text: "Footer".to_string(),
                }),
                timestamp: Some("2024-01-01T00:00:00Z".to_string()),
            }]),
        };
        let params = DiscordNotifier::<MockDiscordWebhookClient>::to_create_message_params(&msg);
        assert_eq!(params.content, "hello");
        assert!(params.embeds.is_some());
        let embeds = params.embeds.unwrap();
        assert_eq!(embeds.len(), 1);
    }

    #[test]
    fn test_to_create_message_params_embed_without_optional_fields() {
        let msg = DiscordMessage {
            content: None,
            embeds: Some(vec![DiscordEmbed {
                title: None,
                description: None,
                url: None,
                color: None,
                fields: None,
                footer: None,
                timestamp: None,
            }]),
        };
        let params = DiscordNotifier::<MockDiscordWebhookClient>::to_create_message_params(&msg);
        assert!(params.embeds.is_some());
    }

    // --- Test cascade success without optional fields ---

    #[test]
    fn test_build_cascade_success_message_without_optional_metadata() {
        let issue = Issue::new("1", "LIN-1", "Fix", "https://example.com", "linear");
        let msg = build_cascade_success_message(
            &issue,
            "https://github.com/repo/pull/1",
            Some("<@user>".to_string()),
        );
        let embed = &msg.embeds.as_ref().unwrap()[0];
        assert!(embed.title.as_ref().unwrap().contains("Cascade PR"));
        // Without original_issue_short_id set, it should be empty string
        let fields = embed.fields.as_ref().unwrap();
        // No "Original Issue" field when the value is empty
        let orig = fields.iter().find(|f| f.name == "Original Issue");
        assert!(orig.is_none());
    }

    #[test]
    fn test_build_cascade_failed_message_without_optional_metadata() {
        let issue = Issue::new("1", "LIN-1", "Fix", "https://example.com", "linear");
        let msg = build_cascade_failed_message(&issue, "Some error", Some("<@user>".to_string()));
        let embed = &msg.embeds.as_ref().unwrap()[0];
        assert!(embed.title.as_ref().unwrap().contains("Cascade Failed"));
        assert_eq!(embed.color, Some(0xe67e22));
        let fields = embed.fields.as_ref().unwrap();
        let error_field = fields.iter().find(|f| f.name == "Error").unwrap();
        assert_eq!(error_field.value, "Some error");
    }

    // --- Test build functions directly ---

    #[test]
    fn test_build_start_message_fields() {
        let issue = Issue::new("1", "LIN-1", "Test Issue", "https://linear.app/1", "linear");
        let msg = build_start_message(&issue, Some("<@user>".to_string()));
        assert_eq!(msg.content, Some("<@user>".to_string()));
        let embed = &msg.embeds.as_ref().unwrap()[0];
        assert!(embed.title.as_ref().unwrap().contains("Processing"));
        assert_eq!(embed.color, Some(0x3498db));
        let fields = embed.fields.as_ref().unwrap();
        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0].name, "Source");
        assert_eq!(fields[1].name, "Priority");
        assert_eq!(fields[2].name, "Status");
    }

    #[test]
    fn test_build_merged_message_fields() {
        let issue = Issue::new("1", "LIN-1", "Fix", "https://linear.app/1", "linear");
        let msg = build_merged_message(&issue, "https://github.com/org/repo/pull/1", None);
        assert!(msg.content.is_none());
        let embed = &msg.embeds.as_ref().unwrap()[0];
        assert!(embed.title.as_ref().unwrap().contains("PR Merged"));
        assert_eq!(embed.color, Some(0x1abc9c)); // Teal
        let fields = embed.fields.as_ref().unwrap();
        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0].name, "Source");
        assert_eq!(fields[1].name, "Issue");
        assert_eq!(fields[2].name, "PR Link");
    }

    #[test]
    fn test_build_closed_message_fields() {
        let issue = Issue::new("1", "LIN-1", "Fix", "https://linear.app/1", "linear");
        let msg = build_closed_message(&issue, "https://github.com/org/repo/pull/1", None);
        let embed = &msg.embeds.as_ref().unwrap()[0];
        assert!(embed.title.as_ref().unwrap().contains("PR Closed"));
        assert_eq!(embed.color, Some(0x95a5a6)); // Grey
        let fields = embed.fields.as_ref().unwrap();
        assert_eq!(fields.len(), 3);
        let note = fields.iter().find(|f| f.name == "Note").unwrap();
        assert!(note.value.contains("closed without merging"));
    }

    #[test]
    fn test_build_status_message_truncation() {
        let long_msg = "z".repeat(3000);
        let msg = build_status_message(&long_msg);
        assert!(msg.content.is_none());
        let embed = &msg.embeds.as_ref().unwrap()[0];
        assert!(embed.description.as_ref().unwrap().len() <= MAX_DESCRIPTION_LENGTH);
        assert!(embed.description.as_ref().unwrap().ends_with("..."));
        assert_eq!(embed.color, Some(0x9b59b6));
    }

    #[test]
    fn test_build_regression_detected_message_empty_url_becomes_none() {
        let mut issue = Issue::new("1", "SEN-1", "Error", "", "sentry");
        issue.set_metadata("regression_detected", true);
        let msg = build_regression_detected_message(&issue, "fail", None);
        let embed = &msg.embeds.as_ref().unwrap()[0];
        assert!(embed.url.is_none()); // empty URL should become None
    }

    #[test]
    fn test_build_regression_resolved_message_empty_url_becomes_none() {
        let issue = Issue::new("1", "SEN-1", "Error", "", "sentry");
        let msg = build_regression_resolved_message(&issue, None);
        let embed = &msg.embeds.as_ref().unwrap()[0];
        // Empty URL becomes None
        assert!(embed.url.is_none());
    }

    #[test]
    fn test_build_regression_resolved_message_with_url() {
        let issue = Issue::new("1", "SEN-1", "Error", "https://sentry.io/1", "sentry");
        let msg = build_regression_resolved_message(&issue, Some("<@user>".to_string()));
        assert_eq!(msg.content, Some("<@user>".to_string()));
        let embed = &msg.embeds.as_ref().unwrap()[0];
        assert!(embed
            .title
            .as_ref()
            .unwrap()
            .contains("Regression Resolved"));
        assert_eq!(embed.color, Some(0x2ecc71));
        assert_eq!(embed.url, Some("https://sentry.io/1".to_string()));
        let fields = embed.fields.as_ref().unwrap();
        let status_field = fields.iter().find(|f| f.name == "Status").unwrap();
        assert!(status_field.value.contains("resolved after final check"));
    }

    #[test]
    fn test_build_start_message_with_trigger_reason() {
        let mut issue = Issue::new("1", "LIN-1", "Test", "https://linear.app/1", "linear");
        issue.set_metadata("trigger_reason", "Retry attempt 2: timeout error");
        let msg = build_start_message(&issue, None);
        let embed = &msg.embeds.as_ref().unwrap()[0];
        let fields = embed.fields.as_ref().unwrap();
        let trigger = fields.iter().find(|f| f.name == "Trigger").unwrap();
        assert_eq!(trigger.value, "Retry attempt 2: timeout error");
        assert_eq!(trigger.inline, Some(false));
    }

    #[test]
    fn test_build_start_message_without_trigger_reason() {
        let issue = Issue::new("1", "LIN-1", "Test", "https://linear.app/1", "linear");
        let msg = build_start_message(&issue, None);
        let embed = &msg.embeds.as_ref().unwrap()[0];
        let fields = embed.fields.as_ref().unwrap();
        assert!(fields.iter().all(|f| f.name != "Trigger"));
    }

    #[test]
    fn test_build_success_message_with_trigger_reason() {
        let mut issue = Issue::new("1", "LIN-1", "Test", "https://linear.app/1", "linear");
        issue.set_metadata("trigger_reason", "Review feedback received");
        let msg = build_success_message(&issue, "https://github.com/pr/1", None);
        let embed = &msg.embeds.as_ref().unwrap()[0];
        let fields = embed.fields.as_ref().unwrap();
        let trigger = fields.iter().find(|f| f.name == "Trigger").unwrap();
        assert_eq!(trigger.value, "Review feedback received");
    }

    #[test]
    fn test_build_failed_message_with_trigger_reason() {
        let mut issue = Issue::new("1", "LIN-1", "Test", "https://linear.app/1", "linear");
        issue.set_metadata("trigger_reason", "Manual trigger");
        let msg = build_failed_message(&issue, "some error", None);
        let embed = &msg.embeds.as_ref().unwrap()[0];
        let fields = embed.fields.as_ref().unwrap();
        let trigger = fields.iter().find(|f| f.name == "Trigger").unwrap();
        assert_eq!(trigger.value, "Manual trigger");
    }

    // === Coverage: build_success_message with changelog metadata ===

    #[test]
    fn test_build_success_message_changelog_field() {
        let mut issue = Issue::new("1", "PROJ-1", "Test", "https://example.com", "linear");
        issue.set_metadata("changelog", "Fixed auth bug");
        let msg = build_success_message(&issue, "https://github.com/pr/1", None);
        let embed = &msg.embeds.as_ref().unwrap()[0];
        let fields = embed.fields.as_ref().unwrap();
        assert!(fields
            .iter()
            .any(|f| f.name == "Changes" && f.value.contains("Fixed auth bug")));
    }

    // === Coverage: build_completed_message with custom completion_reason ===

    #[test]
    fn test_build_completed_message_custom_reason() {
        let mut issue = Issue::new("1", "PROJ-1", "Test", "https://example.com", "linear");
        issue.set_metadata("completion_reason", "Already fixed in previous release");
        let msg = build_completed_message(&issue, None);
        let embed = &msg.embeds.as_ref().unwrap()[0];
        let fields = embed.fields.as_ref().unwrap();
        assert!(fields
            .iter()
            .any(|f| f.name == "Reason" && f.value.contains("Already fixed")));
    }

    // === Coverage: confidence field in success messages ===

    #[test]
    fn test_build_success_message_with_confidence() {
        let mut issue = Issue::new("1", "LIN-1", "Test", "https://linear.app/1", "linear");
        issue.set_metadata("confidence", 85u8);
        let msg = build_success_message(&issue, "https://github.com/pr/1", None);
        let embed = &msg.embeds.as_ref().unwrap()[0];
        let fields = embed.fields.as_ref().unwrap();
        let conf = fields.iter().find(|f| f.name == "Fix Confidence").unwrap();
        assert_eq!(conf.value, "85/100");
        assert_eq!(conf.inline, Some(true));
    }

    #[test]
    fn test_build_success_message_with_confidence_and_reasoning() {
        let mut issue = Issue::new("1", "LIN-1", "Test", "https://linear.app/1", "linear");
        issue.set_metadata("confidence", 72u8);
        issue.set_metadata("confidence_reasoning", "Simple null check fix".to_string());
        let msg = build_success_message(&issue, "https://github.com/pr/1", None);
        let embed = &msg.embeds.as_ref().unwrap()[0];
        let fields = embed.fields.as_ref().unwrap();
        let conf = fields.iter().find(|f| f.name == "Fix Confidence").unwrap();
        assert!(conf.value.contains("72/100"));
        assert!(conf.value.contains("Simple null check fix"));
    }

    #[test]
    fn test_build_success_message_without_confidence() {
        let issue = Issue::new("1", "LIN-1", "Test", "https://linear.app/1", "linear");
        let msg = build_success_message(&issue, "https://github.com/pr/1", None);
        let embed = &msg.embeds.as_ref().unwrap()[0];
        let fields = embed.fields.as_ref().unwrap();
        assert!(!fields.iter().any(|f| f.name == "Fix Confidence"));
    }

    #[test]
    fn test_build_cascade_success_message_with_confidence() {
        let mut issue = Issue::new("1", "LIN-1", "Test", "https://linear.app/1", "linear");
        issue.set_metadata("cascade_downstream_repo", "org/downstream".to_string());
        issue.set_metadata("cascade_upstream_repo", "org/upstream".to_string());
        issue.set_metadata("confidence", 90u8);
        issue.set_metadata("confidence_reasoning", "Straightforward port".to_string());
        let msg = build_cascade_success_message(&issue, "https://github.com/pr/2", None);
        let embed = &msg.embeds.as_ref().unwrap()[0];
        let fields = embed.fields.as_ref().unwrap();
        let conf = fields.iter().find(|f| f.name == "Fix Confidence").unwrap();
        assert!(conf.value.contains("90/100"));
        assert!(conf.value.contains("Straightforward port"));
    }
}
