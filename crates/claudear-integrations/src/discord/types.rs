//! Discord API types for thread management.

use serde::{Deserialize, Serialize};

/// A Discord user.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscordUser {
    /// User ID.
    pub id: String,
    /// Username.
    pub username: String,
    /// Discriminator (legacy, may be 0).
    #[serde(default)]
    pub discriminator: String,
    /// Avatar hash.
    pub avatar: Option<String>,
    /// Whether the user is a bot.
    #[serde(default)]
    pub bot: bool,
}

/// A Discord channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscordChannel {
    /// Channel ID.
    pub id: String,
    /// Channel type.
    #[serde(rename = "type")]
    pub channel_type: u8,
    /// Guild ID (if applicable).
    pub guild_id: Option<String>,
    /// Channel name.
    pub name: Option<String>,
    /// Parent channel ID (for threads).
    pub parent_id: Option<String>,
}

/// A Discord thread.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscordThread {
    /// Thread channel ID.
    pub id: String,
    /// Thread type (11 = public, 12 = private).
    #[serde(rename = "type")]
    pub thread_type: u8,
    /// Guild ID.
    pub guild_id: Option<String>,
    /// Thread name.
    pub name: String,
    /// Parent channel ID.
    pub parent_id: Option<String>,
    /// Owner ID.
    pub owner_id: Option<String>,
    /// Whether the thread is archived.
    #[serde(default)]
    pub archived: bool,
    /// Whether the thread is locked.
    #[serde(default)]
    pub locked: bool,
    /// Message count.
    pub message_count: Option<u32>,
    /// Member count.
    pub member_count: Option<u32>,
}

impl DiscordThread {
    /// Check if the thread is still active (not archived or locked).
    pub fn is_active(&self) -> bool {
        !self.archived && !self.locked
    }
}

/// A Discord message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscordMessage {
    /// Message ID.
    pub id: String,
    /// Channel ID.
    pub channel_id: String,
    /// Message author.
    pub author: Option<DiscordUser>,
    /// Message content.
    pub content: String,
    /// Timestamp.
    pub timestamp: String,
    /// Reference to another message when this is a reply.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message_reference: Option<DiscordMessageReference>,
    /// Thread associated with this message (if started).
    pub thread: Option<DiscordThread>,
    /// Webhook ID if the message was sent via a webhook.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub webhook_id: Option<String>,
    /// Message embeds.
    #[serde(default)]
    pub embeds: Vec<DiscordEmbed>,
}

/// A Discord embed object (subset of fields used for identification).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscordEmbed {
    /// Embed title.
    #[serde(default)]
    pub title: Option<String>,
    /// Embed description.
    #[serde(default)]
    pub description: Option<String>,
}

/// A Discord message reference (used for replies).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscordMessageReference {
    /// Referenced message ID.
    pub message_id: Option<String>,
    /// Referenced channel ID.
    pub channel_id: Option<String>,
    /// Referenced guild ID.
    pub guild_id: Option<String>,
    /// When `Some(false)`, sending still succeeds even if the referenced
    /// message no longer exists (Discord defaults this to `true`, which would
    /// otherwise fail the whole send if the original message was deleted).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fail_if_not_exists: Option<bool>,
}

/// Parameters for creating a thread.
#[derive(Debug, Clone, Serialize)]
pub struct CreateThreadParams {
    /// Thread name (1-100 characters).
    pub name: String,
    /// Auto-archive duration in minutes (60, 1440, 4320, 10080).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auto_archive_duration: Option<u32>,
    /// Thread type (11 = public, 12 = private).
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub thread_type: Option<u8>,
    /// Whether to send a rate limit error rather than being slow.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rate_limit_per_user: Option<u32>,
}

impl CreateThreadParams {
    /// Create params for a public thread.
    pub fn public(name: impl Into<String>) -> Self {
        let name = name.into();
        let name = if name.len() > 100 {
            name[..name.floor_char_boundary(100)].to_string()
        } else {
            name
        };
        Self {
            name,
            auto_archive_duration: Some(10080), // 7 days
            thread_type: Some(11),              // Public thread
            rate_limit_per_user: None,
        }
    }

    /// Create params for a private thread.
    pub fn private(name: impl Into<String>) -> Self {
        let name = name.into();
        let name = if name.len() > 100 {
            name[..name.floor_char_boundary(100)].to_string()
        } else {
            name
        };
        Self {
            name,
            auto_archive_duration: Some(10080), // 7 days
            thread_type: Some(12),              // Private thread
            rate_limit_per_user: None,
        }
    }

    /// Set auto-archive duration.
    pub fn with_auto_archive(mut self, minutes: u32) -> Self {
        self.auto_archive_duration = Some(minutes);
        self
    }
}

/// Parameters for creating a message.
#[derive(Debug, Clone, Serialize)]
pub struct CreateMessageParams {
    /// Message content (up to 2000 characters).
    pub content: String,
    /// Whether this is a TTS message.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tts: Option<bool>,
    /// Embeds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embeds: Option<Vec<MessageEmbed>>,
    /// Reference to another message (for replies).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message_reference: Option<DiscordMessageReference>,
}

impl CreateMessageParams {
    /// Create simple text message.
    pub fn text(content: impl Into<String>) -> Self {
        let content = content.into();
        let content = if content.len() > 2000 {
            format!("{}...", &content[..content.floor_char_boundary(1997)])
        } else {
            content
        };
        Self {
            content,
            tts: None,
            embeds: None,
            message_reference: None,
        }
    }

    /// Create message with embed.
    pub fn with_embed(content: impl Into<String>, embed: MessageEmbed) -> Self {
        let content = content.into();
        let content = if content.len() > 2000 {
            format!("{}...", &content[..content.floor_char_boundary(1997)])
        } else {
            content
        };
        Self {
            content,
            tts: None,
            embeds: Some(vec![embed]),
            message_reference: None,
        }
    }

    /// Set this message as a reply to another message.
    pub fn replying_to(mut self, message_id: impl Into<String>) -> Self {
        self.message_reference = Some(DiscordMessageReference {
            message_id: Some(message_id.into()),
            channel_id: None,
            guild_id: None,
            // Still deliver even if the referenced message was deleted.
            fail_if_not_exists: Some(false),
        });
        self
    }
}

/// A Discord embed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageEmbed {
    /// Embed title.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Embed description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Embed URL.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// Embed color.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub color: Option<u32>,
    /// Embed fields.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fields: Option<Vec<EmbedField>>,
    /// Footer.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub footer: Option<EmbedFooter>,
    /// Timestamp.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<String>,
}

impl MessageEmbed {
    /// Create a simple embed.
    pub fn new() -> Self {
        Self {
            title: None,
            description: None,
            url: None,
            color: None,
            fields: None,
            footer: None,
            timestamp: None,
        }
    }

    /// Set title.
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Set description.
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Set URL.
    pub fn url(mut self, url: impl Into<String>) -> Self {
        self.url = Some(url.into());
        self
    }

    /// Set color.
    pub fn color(mut self, color: u32) -> Self {
        self.color = Some(color);
        self
    }

    /// Add a field.
    pub fn field(
        mut self,
        name: impl Into<String>,
        value: impl Into<String>,
        inline: bool,
    ) -> Self {
        let field = EmbedField {
            name: name.into(),
            value: value.into(),
            inline: Some(inline),
        };
        self.fields.get_or_insert_with(Vec::new).push(field);
        self
    }

    /// Set footer.
    pub fn footer(mut self, text: impl Into<String>) -> Self {
        self.footer = Some(EmbedFooter { text: text.into() });
        self
    }

    /// Set timestamp.
    pub fn timestamp(mut self, ts: impl Into<String>) -> Self {
        self.timestamp = Some(ts.into());
        self
    }
}

impl Default for MessageEmbed {
    fn default() -> Self {
        Self::new()
    }
}

/// An embed field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedField {
    /// Field name.
    pub name: String,
    /// Field value.
    pub value: String,
    /// Whether inline.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inline: Option<bool>,
}

/// An embed footer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedFooter {
    /// Footer text.
    pub text: String,
}

/// Thread state for tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadState {
    /// Thread ID.
    pub thread_id: String,
    /// Thread name.
    pub thread_name: String,
    /// Parent channel ID.
    pub channel_id: String,
    /// Associated PR URL.
    pub pr_url: String,
    /// Associated issue ID.
    pub issue_id: String,
    /// Issue source.
    pub source: String,
    /// Created timestamp.
    pub created_at: String,
    /// Whether the thread is still active.
    pub is_active: bool,
    /// Last message ID in thread.
    pub last_message_id: Option<String>,
}

impl ThreadState {
    /// Create a new thread state.
    pub fn new(
        thread_id: impl Into<String>,
        thread_name: impl Into<String>,
        channel_id: impl Into<String>,
        pr_url: impl Into<String>,
        issue_id: impl Into<String>,
        source: impl Into<String>,
    ) -> Self {
        Self {
            thread_id: thread_id.into(),
            thread_name: thread_name.into(),
            channel_id: channel_id.into(),
            pr_url: pr_url.into(),
            issue_id: issue_id.into(),
            source: source.into(),
            created_at: chrono::Utc::now().to_rfc3339(),
            is_active: true,
            last_message_id: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_thread_params_public() {
        let params = CreateThreadParams::public("Test Thread");
        assert_eq!(params.name, "Test Thread");
        assert_eq!(params.thread_type, Some(11));
        assert_eq!(params.auto_archive_duration, Some(10080));
    }

    #[test]
    fn test_create_thread_params_private() {
        let params = CreateThreadParams::private("Private Thread");
        assert_eq!(params.name, "Private Thread");
        assert_eq!(params.thread_type, Some(12));
    }

    #[test]
    fn test_create_thread_params_truncates_long_name() {
        let long_name = "a".repeat(150);
        let params = CreateThreadParams::public(&long_name);
        assert_eq!(params.name.len(), 100);
    }

    #[test]
    fn test_create_message_params_text() {
        let params = CreateMessageParams::text("Hello");
        assert_eq!(params.content, "Hello");
        assert!(params.embeds.is_none());
    }

    #[test]
    fn test_create_message_params_truncates() {
        let long_content = "a".repeat(3000);
        let params = CreateMessageParams::text(&long_content);
        assert_eq!(params.content.len(), 2000);
        assert!(params.content.ends_with("..."));
    }

    #[test]
    fn test_message_embed_builder() {
        let embed = MessageEmbed::new()
            .title("Test")
            .description("Desc")
            .color(0xFF0000)
            .field("Field1", "Value1", true)
            .footer("Footer");

        assert_eq!(embed.title, Some("Test".to_string()));
        assert_eq!(embed.description, Some("Desc".to_string()));
        assert_eq!(embed.color, Some(0xFF0000));
        assert!(embed.fields.is_some());
        assert_eq!(embed.fields.as_ref().unwrap().len(), 1);
        assert!(embed.footer.is_some());
    }

    #[test]
    fn test_thread_state_new() {
        let state = ThreadState::new(
            "123456",
            "PR: Fix Bug",
            "channel123",
            "https://github.com/user/repo/pull/1",
            "issue-1",
            "linear",
        );

        assert_eq!(state.thread_id, "123456");
        assert_eq!(state.thread_name, "PR: Fix Bug");
        assert!(state.is_active);
        assert!(state.last_message_id.is_none());
    }

    #[test]
    fn test_discord_thread_is_active() {
        let active_thread = DiscordThread {
            id: "123".to_string(),
            thread_type: 11,
            guild_id: None,
            name: "Test".to_string(),
            parent_id: None,
            owner_id: None,
            archived: false,
            locked: false,
            message_count: None,
            member_count: None,
        };
        assert!(active_thread.is_active());

        let archived_thread = DiscordThread {
            archived: true,
            ..active_thread.clone()
        };
        assert!(!archived_thread.is_active());

        let locked_thread = DiscordThread {
            locked: true,
            ..active_thread
        };
        assert!(!locked_thread.is_active());
    }

    #[test]
    fn test_create_thread_params_with_auto_archive() {
        let params = CreateThreadParams::public("Test").with_auto_archive(60);
        assert_eq!(params.auto_archive_duration, Some(60));
    }

    #[test]
    fn test_create_message_params_with_embed() {
        let embed = MessageEmbed::new().title("Title").description("Desc");
        let params = CreateMessageParams::with_embed("Message", embed);
        assert!(params.embeds.is_some());
        assert_eq!(params.embeds.unwrap().len(), 1);
    }

    #[test]
    fn test_message_embed_url() {
        let embed = MessageEmbed::new().url("https://example.com");
        assert_eq!(embed.url, Some("https://example.com".to_string()));
    }

    #[test]
    fn test_message_embed_timestamp() {
        let embed = MessageEmbed::new().timestamp("2024-01-01T00:00:00Z");
        assert_eq!(embed.timestamp, Some("2024-01-01T00:00:00Z".to_string()));
    }

    #[test]
    fn test_message_embed_multiple_fields() {
        let embed = MessageEmbed::new()
            .field("F1", "V1", true)
            .field("F2", "V2", false)
            .field("F3", "V3", true);
        let fields = embed.fields.unwrap();
        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0].name, "F1");
        assert!(fields[0].inline.unwrap());
        assert!(!fields[1].inline.unwrap());
    }

    #[test]
    fn test_thread_state_set_last_message() {
        let mut state = ThreadState::new("t1", "Thread", "c1", "https://pr", "i1", "linear");
        assert!(state.last_message_id.is_none());
        state.last_message_id = Some("msg123".to_string());
        assert_eq!(state.last_message_id.unwrap(), "msg123");
    }

    #[test]
    fn test_thread_state_deactivate() {
        let mut state = ThreadState::new("t1", "Thread", "c1", "https://pr", "i1", "linear");
        assert!(state.is_active);
        state.is_active = false;
        assert!(!state.is_active);
    }

    #[test]
    fn test_discord_user_serialization() {
        let user = DiscordUser {
            id: "123".to_string(),
            username: "testuser".to_string(),
            discriminator: "0001".to_string(),
            avatar: Some("abc123".to_string()),
            bot: false,
        };
        let json = serde_json::to_string(&user).unwrap();
        assert!(json.contains("testuser"));
        assert!(json.contains("0001"));
    }

    #[test]
    fn test_discord_user_bot_flag() {
        let bot_user = DiscordUser {
            id: "456".to_string(),
            username: "bot".to_string(),
            discriminator: "0000".to_string(),
            avatar: None,
            bot: true,
        };
        assert!(bot_user.bot);
    }

    #[test]
    fn test_discord_channel_serialization() {
        let channel = DiscordChannel {
            id: "123".to_string(),
            channel_type: 0,
            guild_id: Some("guild123".to_string()),
            name: Some("general".to_string()),
            parent_id: None,
        };
        let json = serde_json::to_string(&channel).unwrap();
        assert!(json.contains("general"));
        assert!(json.contains("guild123"));
    }

    #[test]
    fn test_discord_message_serialization() {
        let message = DiscordMessage {
            id: "msg1".to_string(),
            channel_id: "chan1".to_string(),
            author: Some(DiscordUser {
                id: "user1".to_string(),
                username: "author".to_string(),
                discriminator: "0".to_string(),
                avatar: None,
                bot: false,
            }),
            content: "Hello world".to_string(),
            timestamp: "2024-01-01T00:00:00Z".to_string(),
            message_reference: None,
            thread: None,
            webhook_id: None,
            embeds: vec![],
        };
        let json = serde_json::to_string(&message).unwrap();
        assert!(json.contains("Hello world"));
        assert!(json.contains("author"));
    }

    #[test]
    fn test_embed_field_serialization() {
        let field = EmbedField {
            name: "Test".to_string(),
            value: "Value".to_string(),
            inline: Some(true),
        };
        let json = serde_json::to_string(&field).unwrap();
        assert!(json.contains("Test"));
        assert!(json.contains("Value"));
        assert!(json.contains("true"));
    }

    #[test]
    fn test_embed_footer_serialization() {
        let footer = EmbedFooter {
            text: "Footer text".to_string(),
        };
        let json = serde_json::to_string(&footer).unwrap();
        assert!(json.contains("Footer text"));
    }

    #[test]
    fn test_message_embed_default() {
        let embed = MessageEmbed::default();
        assert!(embed.title.is_none());
        assert!(embed.description.is_none());
        assert!(embed.color.is_none());
    }

    #[test]
    fn test_create_thread_params_exactly_100_chars() {
        let name = "a".repeat(100);
        let params = CreateThreadParams::public(&name);
        assert_eq!(params.name.len(), 100);
    }

    #[test]
    fn test_create_message_params_exactly_2000_chars() {
        let content = "a".repeat(2000);
        let params = CreateMessageParams::text(&content);
        assert_eq!(params.content.len(), 2000);
    }

    #[test]
    fn test_discord_user_deserialize_full() {
        let json = r#"{
            "id": "123456789",
            "username": "testuser",
            "discriminator": "1234",
            "avatar": "abcdef123456",
            "bot": false
        }"#;
        let user: DiscordUser = serde_json::from_str(json).unwrap();
        assert_eq!(user.id, "123456789");
        assert_eq!(user.username, "testuser");
        assert_eq!(user.discriminator, "1234");
        assert_eq!(user.avatar, Some("abcdef123456".to_string()));
        assert!(!user.bot);
    }

    #[test]
    fn test_discord_user_deserialize_missing_optional_fields() {
        let json = r#"{
            "id": "123",
            "username": "minimal_user",
            "avatar": null
        }"#;
        let user: DiscordUser = serde_json::from_str(json).unwrap();
        assert_eq!(user.id, "123");
        assert_eq!(user.username, "minimal_user");
        // discriminator should default to empty string via #[serde(default)]
        assert_eq!(user.discriminator, "");
        assert!(user.avatar.is_none());
        // bot should default to false via #[serde(default)]
        assert!(!user.bot);
    }

    #[test]
    fn test_discord_user_deserialize_bot_true() {
        let json = r#"{
            "id": "999",
            "username": "webhook-bot",
            "discriminator": "0000",
            "avatar": null,
            "bot": true
        }"#;
        let user: DiscordUser = serde_json::from_str(json).unwrap();
        assert!(user.bot);
        assert_eq!(user.discriminator, "0000");
    }

    #[test]
    fn test_discord_channel_deserialize_full() {
        let json = r#"{
            "id": "chan-001",
            "type": 0,
            "guild_id": "guild-123",
            "name": "general",
            "parent_id": "category-456"
        }"#;
        let channel: DiscordChannel = serde_json::from_str(json).unwrap();
        assert_eq!(channel.id, "chan-001");
        assert_eq!(channel.channel_type, 0);
        assert_eq!(channel.guild_id, Some("guild-123".to_string()));
        assert_eq!(channel.name, Some("general".to_string()));
        assert_eq!(channel.parent_id, Some("category-456".to_string()));
    }

    #[test]
    fn test_discord_channel_deserialize_missing_optional_fields() {
        let json = r#"{
            "id": "chan-002",
            "type": 4
        }"#;
        let channel: DiscordChannel = serde_json::from_str(json).unwrap();
        assert_eq!(channel.id, "chan-002");
        assert_eq!(channel.channel_type, 4);
        assert!(channel.guild_id.is_none());
        assert!(channel.name.is_none());
        assert!(channel.parent_id.is_none());
    }

    #[test]
    fn test_discord_thread_deserialize_full() {
        let json = r#"{
            "id": "thread-001",
            "type": 11,
            "guild_id": "guild-1",
            "name": "Discussion Thread",
            "parent_id": "chan-1",
            "owner_id": "user-1",
            "archived": false,
            "locked": false,
            "message_count": 42,
            "member_count": 5
        }"#;
        let thread: DiscordThread = serde_json::from_str(json).unwrap();
        assert_eq!(thread.id, "thread-001");
        assert_eq!(thread.thread_type, 11);
        assert_eq!(thread.guild_id, Some("guild-1".to_string()));
        assert_eq!(thread.name, "Discussion Thread");
        assert_eq!(thread.parent_id, Some("chan-1".to_string()));
        assert_eq!(thread.owner_id, Some("user-1".to_string()));
        assert!(!thread.archived);
        assert!(!thread.locked);
        assert_eq!(thread.message_count, Some(42));
        assert_eq!(thread.member_count, Some(5));
    }

    #[test]
    fn test_discord_thread_deserialize_missing_optional_fields() {
        let json = r#"{
            "id": "thread-002",
            "type": 12,
            "name": "Bare Thread"
        }"#;
        let thread: DiscordThread = serde_json::from_str(json).unwrap();
        assert_eq!(thread.id, "thread-002");
        assert_eq!(thread.thread_type, 12);
        assert!(thread.guild_id.is_none());
        assert_eq!(thread.name, "Bare Thread");
        assert!(thread.parent_id.is_none());
        assert!(thread.owner_id.is_none());
        // archived and locked default to false
        assert!(!thread.archived);
        assert!(!thread.locked);
        assert!(thread.message_count.is_none());
        assert!(thread.member_count.is_none());
    }

    #[test]
    fn test_discord_message_deserialize_minimal() {
        let json = r#"{
            "id": "msg-001",
            "channel_id": "chan-1",
            "content": "Hello, world!",
            "timestamp": "2024-06-15T12:00:00Z"
        }"#;
        let msg: DiscordMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.id, "msg-001");
        assert_eq!(msg.channel_id, "chan-1");
        assert!(msg.author.is_none());
        assert_eq!(msg.content, "Hello, world!");
        assert_eq!(msg.timestamp, "2024-06-15T12:00:00Z");
        assert!(msg.message_reference.is_none());
        assert!(msg.thread.is_none());
        assert!(msg.webhook_id.is_none());
    }

    #[test]
    fn test_discord_message_deserialize_with_message_reference() {
        let json = r#"{
            "id": "msg-002",
            "channel_id": "chan-1",
            "content": "This is a reply",
            "timestamp": "2024-06-15T13:00:00Z",
            "message_reference": {
                "message_id": "msg-001",
                "channel_id": "chan-1",
                "guild_id": "guild-1"
            }
        }"#;
        let msg: DiscordMessage = serde_json::from_str(json).unwrap();
        assert!(msg.message_reference.is_some());
        let reference = msg.message_reference.unwrap();
        assert_eq!(reference.message_id, Some("msg-001".to_string()));
        assert_eq!(reference.channel_id, Some("chan-1".to_string()));
        assert_eq!(reference.guild_id, Some("guild-1".to_string()));
    }

    #[test]
    fn test_discord_message_deserialize_with_thread() {
        let json = r#"{
            "id": "msg-003",
            "channel_id": "chan-1",
            "content": "Thread starter",
            "timestamp": "2024-06-15T14:00:00Z",
            "thread": {
                "id": "thread-100",
                "type": 11,
                "name": "Spawned Thread",
                "archived": false,
                "locked": false
            }
        }"#;
        let msg: DiscordMessage = serde_json::from_str(json).unwrap();
        assert!(msg.thread.is_some());
        let thread = msg.thread.unwrap();
        assert_eq!(thread.id, "thread-100");
        assert_eq!(thread.name, "Spawned Thread");
        assert!(thread.is_active());
    }

    #[test]
    fn test_discord_message_deserialize_with_webhook_id() {
        let json = r#"{
            "id": "msg-004",
            "channel_id": "chan-1",
            "content": "Webhook message",
            "timestamp": "2024-06-15T15:00:00Z",
            "webhook_id": "webhook-555"
        }"#;
        let msg: DiscordMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.webhook_id, Some("webhook-555".to_string()));
    }

    #[test]
    fn test_discord_message_reference_deserialize_all_none() {
        let json = r#"{
            "message_id": null,
            "channel_id": null,
            "guild_id": null
        }"#;
        let reference: DiscordMessageReference = serde_json::from_str(json).unwrap();
        assert!(reference.message_id.is_none());
        assert!(reference.channel_id.is_none());
        assert!(reference.guild_id.is_none());
    }

    #[test]
    fn test_discord_thread_is_active_when_both_archived_and_locked() {
        let thread = DiscordThread {
            id: "t-1".to_string(),
            thread_type: 11,
            guild_id: None,
            name: "Dead Thread".to_string(),
            parent_id: None,
            owner_id: None,
            archived: true,
            locked: true,
            message_count: None,
            member_count: None,
        };
        assert!(!thread.is_active());
    }

    #[test]
    fn test_create_thread_params_private_long_name_truncation() {
        let long_name = "x".repeat(200);
        let params = CreateThreadParams::private(&long_name);
        assert_eq!(params.name.len(), 100);
        assert_eq!(params.thread_type, Some(12));
        // All 100 chars should be 'x'
        assert!(params.name.chars().all(|c| c == 'x'));
    }

    #[test]
    fn test_create_message_params_with_embed_long_content_truncation() {
        let long_content = "b".repeat(3000);
        let embed = MessageEmbed::new().title("Test Embed");
        let params = CreateMessageParams::with_embed(&long_content, embed);
        assert_eq!(params.content.len(), 2000);
        assert!(params.content.ends_with("..."));
        // The first 1997 chars should be 'b', followed by "..."
        assert!(params.content[..1997].chars().all(|c| c == 'b'));
        // Verify embed is still present
        assert!(params.embeds.is_some());
        assert_eq!(
            params.embeds.as_ref().unwrap()[0].title,
            Some("Test Embed".to_string())
        );
    }

    #[test]
    fn test_message_embed_with_empty_strings() {
        let embed = MessageEmbed::new()
            .title("")
            .description("")
            .url("")
            .footer("");
        assert_eq!(embed.title, Some("".to_string()));
        assert_eq!(embed.description, Some("".to_string()));
        assert_eq!(embed.url, Some("".to_string()));
        assert_eq!(embed.footer.unwrap().text, "");
    }

    #[test]
    fn test_thread_state_serialization_roundtrip() {
        let state = ThreadState {
            thread_id: "t-100".to_string(),
            thread_name: "Roundtrip Thread".to_string(),
            channel_id: "c-200".to_string(),
            pr_url: "https://github.com/org/repo/pull/42".to_string(),
            issue_id: "PROJ-123".to_string(),
            source: "jira".to_string(),
            created_at: "2024-06-15T10:30:00Z".to_string(),
            is_active: true,
            last_message_id: Some("msg-999".to_string()),
        };
        let json = serde_json::to_string(&state).unwrap();
        let deserialized: ThreadState = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.thread_id, "t-100");
        assert_eq!(deserialized.thread_name, "Roundtrip Thread");
        assert_eq!(deserialized.channel_id, "c-200");
        assert_eq!(deserialized.pr_url, "https://github.com/org/repo/pull/42");
        assert_eq!(deserialized.issue_id, "PROJ-123");
        assert_eq!(deserialized.source, "jira");
        assert_eq!(deserialized.created_at, "2024-06-15T10:30:00Z");
        assert!(deserialized.is_active);
        assert_eq!(deserialized.last_message_id, Some("msg-999".to_string()));
    }

    #[test]
    fn test_thread_state_serialization_roundtrip_inactive_no_last_message() {
        let state = ThreadState {
            thread_id: "t-200".to_string(),
            thread_name: "Inactive".to_string(),
            channel_id: "c-300".to_string(),
            pr_url: "https://github.com/org/repo/pull/99".to_string(),
            issue_id: "LIN-456".to_string(),
            source: "linear".to_string(),
            created_at: "2024-01-01T00:00:00Z".to_string(),
            is_active: false,
            last_message_id: None,
        };
        let json = serde_json::to_string(&state).unwrap();
        let deserialized: ThreadState = serde_json::from_str(&json).unwrap();
        assert!(!deserialized.is_active);
        assert!(deserialized.last_message_id.is_none());
    }

    #[test]
    fn test_create_message_params_text_skips_optional_fields_in_json() {
        let params = CreateMessageParams::text("Hello");
        let json = serde_json::to_string(&params).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        let obj = value.as_object().unwrap();
        // tts and embeds should not be present due to skip_serializing_if
        assert!(!obj.contains_key("tts"), "tts should not appear in JSON");
        assert!(
            !obj.contains_key("embeds"),
            "embeds should not appear in JSON"
        );
        assert!(obj.contains_key("content"));
        assert_eq!(obj["content"], "Hello");
    }

    #[test]
    fn test_create_thread_params_public_skips_rate_limit_in_json() {
        let params = CreateThreadParams::public("My Thread");
        let json = serde_json::to_string(&params).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        let obj = value.as_object().unwrap();
        // rate_limit_per_user should not be present
        assert!(
            !obj.contains_key("rate_limit_per_user"),
            "rate_limit_per_user should not appear in JSON"
        );
        // But name, auto_archive_duration, and type should be present
        assert_eq!(obj["name"], "My Thread");
        assert_eq!(obj["auto_archive_duration"], 10080);
        assert_eq!(obj["type"], 11);
    }

    #[test]
    fn test_message_embed_new_serializes_to_mostly_empty_json() {
        let embed = MessageEmbed::new();
        let json = serde_json::to_string(&embed).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        let obj = value.as_object().unwrap();
        // All fields are Option with skip_serializing_if, so should be empty
        assert!(
            !obj.contains_key("title"),
            "title should not appear in JSON"
        );
        assert!(
            !obj.contains_key("description"),
            "description should not appear in JSON"
        );
        assert!(!obj.contains_key("url"), "url should not appear in JSON");
        assert!(
            !obj.contains_key("color"),
            "color should not appear in JSON"
        );
        assert!(
            !obj.contains_key("fields"),
            "fields should not appear in JSON"
        );
        assert!(
            !obj.contains_key("footer"),
            "footer should not appear in JSON"
        );
        assert!(
            !obj.contains_key("timestamp"),
            "timestamp should not appear in JSON"
        );
        // The JSON should just be "{}"
        assert_eq!(json, "{}");
    }

    #[test]
    fn test_discord_message_without_reference_skips_it_in_json() {
        let message = DiscordMessage {
            id: "msg-1".to_string(),
            channel_id: "chan-1".to_string(),
            author: None,
            content: "No reference".to_string(),
            timestamp: "2024-01-01T00:00:00Z".to_string(),
            message_reference: None,
            thread: None,
            webhook_id: None,
            embeds: vec![],
        };
        let json = serde_json::to_string(&message).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        let obj = value.as_object().unwrap();
        assert!(
            !obj.contains_key("message_reference"),
            "message_reference should not appear when None"
        );
        assert!(
            !obj.contains_key("webhook_id"),
            "webhook_id should not appear when None"
        );
    }

    #[test]
    fn test_discord_message_with_reference_includes_it_in_json() {
        let message = DiscordMessage {
            id: "msg-2".to_string(),
            channel_id: "chan-1".to_string(),
            author: None,
            content: "With reference".to_string(),
            timestamp: "2024-01-01T00:00:00Z".to_string(),
            message_reference: Some(DiscordMessageReference {
                message_id: Some("msg-1".to_string()),
                channel_id: Some("chan-1".to_string()),
                guild_id: None,
                fail_if_not_exists: None,
            }),
            thread: None,
            webhook_id: Some("wh-123".to_string()),
            embeds: vec![],
        };
        let json = serde_json::to_string(&message).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        let obj = value.as_object().unwrap();
        assert!(
            obj.contains_key("message_reference"),
            "message_reference should appear when Some"
        );
        assert!(
            obj.contains_key("webhook_id"),
            "webhook_id should appear when Some"
        );
        let ref_obj = obj["message_reference"].as_object().unwrap();
        assert_eq!(ref_obj["message_id"], "msg-1");
    }

    #[test]
    fn test_embed_field_without_inline_skips_it_in_json() {
        let field = EmbedField {
            name: "Key".to_string(),
            value: "Val".to_string(),
            inline: None,
        };
        let json = serde_json::to_string(&field).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        let obj = value.as_object().unwrap();
        assert!(
            !obj.contains_key("inline"),
            "inline should not appear when None"
        );
    }

    #[test]
    fn test_discord_user_deserialize_roundtrip() {
        let user = DiscordUser {
            id: "789".to_string(),
            username: "roundtrip".to_string(),
            discriminator: "5678".to_string(),
            avatar: Some("hash".to_string()),
            bot: true,
        };
        let json = serde_json::to_string(&user).unwrap();
        let deserialized: DiscordUser = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.id, user.id);
        assert_eq!(deserialized.username, user.username);
        assert_eq!(deserialized.discriminator, user.discriminator);
        assert_eq!(deserialized.avatar, user.avatar);
        assert_eq!(deserialized.bot, user.bot);
    }

    #[test]
    fn test_discord_channel_deserialize_roundtrip() {
        let channel = DiscordChannel {
            id: "c-1".to_string(),
            channel_type: 2,
            guild_id: Some("g-1".to_string()),
            name: Some("voice".to_string()),
            parent_id: Some("cat-1".to_string()),
        };
        let json = serde_json::to_string(&channel).unwrap();
        let deserialized: DiscordChannel = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.id, channel.id);
        assert_eq!(deserialized.channel_type, channel.channel_type);
        assert_eq!(deserialized.guild_id, channel.guild_id);
        assert_eq!(deserialized.name, channel.name);
        assert_eq!(deserialized.parent_id, channel.parent_id);
    }

    #[test]
    fn test_discord_thread_deserialize_roundtrip() {
        let thread = DiscordThread {
            id: "thr-1".to_string(),
            thread_type: 12,
            guild_id: Some("g-2".to_string()),
            name: "Private Thread".to_string(),
            parent_id: Some("c-5".to_string()),
            owner_id: Some("u-10".to_string()),
            archived: true,
            locked: false,
            message_count: Some(100),
            member_count: Some(10),
        };
        let json = serde_json::to_string(&thread).unwrap();
        let deserialized: DiscordThread = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.id, thread.id);
        assert_eq!(deserialized.thread_type, thread.thread_type);
        assert_eq!(deserialized.guild_id, thread.guild_id);
        assert_eq!(deserialized.name, thread.name);
        assert_eq!(deserialized.parent_id, thread.parent_id);
        assert_eq!(deserialized.owner_id, thread.owner_id);
        assert_eq!(deserialized.archived, thread.archived);
        assert_eq!(deserialized.locked, thread.locked);
        assert_eq!(deserialized.message_count, thread.message_count);
        assert_eq!(deserialized.member_count, thread.member_count);
    }

    #[test]
    fn test_discord_message_deserialize_full_roundtrip() {
        let message = DiscordMessage {
            id: "msg-rt".to_string(),
            channel_id: "chan-rt".to_string(),
            author: Some(DiscordUser {
                id: "u-rt".to_string(),
                username: "roundtripper".to_string(),
                discriminator: "9999".to_string(),
                avatar: Some("avhash".to_string()),
                bot: false,
            }),
            content: "Roundtrip test".to_string(),
            timestamp: "2024-12-25T00:00:00Z".to_string(),
            message_reference: Some(DiscordMessageReference {
                message_id: Some("msg-orig".to_string()),
                channel_id: Some("chan-rt".to_string()),
                guild_id: Some("guild-rt".to_string()),
                fail_if_not_exists: None,
            }),
            thread: None,
            webhook_id: Some("wh-rt".to_string()),
            embeds: vec![],
        };
        let json = serde_json::to_string(&message).unwrap();
        let deserialized: DiscordMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.id, "msg-rt");
        assert_eq!(deserialized.content, "Roundtrip test");
        assert!(deserialized.author.is_some());
        assert_eq!(
            deserialized.author.as_ref().unwrap().username,
            "roundtripper"
        );
        assert!(deserialized.message_reference.is_some());
        assert_eq!(
            deserialized.message_reference.as_ref().unwrap().message_id,
            Some("msg-orig".to_string())
        );
        assert_eq!(deserialized.webhook_id, Some("wh-rt".to_string()));
    }

    #[test]
    fn test_create_thread_params_private_exactly_100_chars() {
        let name = "z".repeat(100);
        let params = CreateThreadParams::private(&name);
        assert_eq!(params.name.len(), 100);
        assert_eq!(params.thread_type, Some(12));
    }

    #[test]
    fn test_create_message_with_embed_exactly_2000_chars() {
        let content = "c".repeat(2000);
        let embed = MessageEmbed::new().title("Embed");
        let params = CreateMessageParams::with_embed(&content, embed);
        // Exactly 2000 should not truncate
        assert_eq!(params.content.len(), 2000);
        assert!(!params.content.ends_with("..."));
    }

    #[test]
    fn test_create_message_with_embed_2001_chars_truncates() {
        let content = "d".repeat(2001);
        let embed = MessageEmbed::new().title("Embed");
        let params = CreateMessageParams::with_embed(&content, embed);
        assert_eq!(params.content.len(), 2000);
        assert!(params.content.ends_with("..."));
    }

    #[test]
    fn test_message_embed_partial_serialization() {
        let embed = MessageEmbed::new().title("Only Title").color(0x00FF00);
        let json = serde_json::to_string(&embed).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        let obj = value.as_object().unwrap();
        assert!(obj.contains_key("title"));
        assert!(obj.contains_key("color"));
        assert!(!obj.contains_key("description"));
        assert!(!obj.contains_key("url"));
        assert!(!obj.contains_key("fields"));
        assert!(!obj.contains_key("footer"));
        assert!(!obj.contains_key("timestamp"));
        assert_eq!(obj["title"], "Only Title");
        assert_eq!(obj["color"], 0x00FF00);
    }

    #[test]
    fn test_message_embed_deserialize_from_json() {
        let json = r#"{
            "title": "Deserialized Embed",
            "description": "From JSON",
            "color": 16711680,
            "fields": [
                {"name": "F1", "value": "V1", "inline": true}
            ],
            "footer": {"text": "foot"}
        }"#;
        let embed: MessageEmbed = serde_json::from_str(json).unwrap();
        assert_eq!(embed.title, Some("Deserialized Embed".to_string()));
        assert_eq!(embed.description, Some("From JSON".to_string()));
        assert_eq!(embed.color, Some(16711680));
        assert!(embed.url.is_none());
        assert!(embed.timestamp.is_none());
        let fields = embed.fields.unwrap();
        assert_eq!(fields.len(), 1);
        assert_eq!(fields[0].name, "F1");
        assert_eq!(fields[0].value, "V1");
        assert_eq!(fields[0].inline, Some(true));
        assert_eq!(embed.footer.unwrap().text, "foot");
    }

    #[test]
    fn test_message_embed_deserialize_empty_json() {
        let json = "{}";
        let embed: MessageEmbed = serde_json::from_str(json).unwrap();
        assert!(embed.title.is_none());
        assert!(embed.description.is_none());
        assert!(embed.url.is_none());
        assert!(embed.color.is_none());
        assert!(embed.fields.is_none());
        assert!(embed.footer.is_none());
        assert!(embed.timestamp.is_none());
    }

    #[test]
    fn test_discord_message_reference_deserialize_partial() {
        let json = r#"{"message_id": "msg-only"}"#;
        let reference: DiscordMessageReference = serde_json::from_str(json).unwrap();
        assert_eq!(reference.message_id, Some("msg-only".to_string()));
        assert!(reference.channel_id.is_none());
        assert!(reference.guild_id.is_none());
    }

    #[test]
    fn test_create_thread_params_private_short_name_no_truncation() {
        let name = "Short";
        let params = CreateThreadParams::private(name);
        assert_eq!(params.name, "Short");
        assert_eq!(params.name.len(), 5);
    }

    #[test]
    fn test_create_thread_params_serialization_with_rate_limit() {
        let mut params = CreateThreadParams::public("Rate Limited");
        params.rate_limit_per_user = Some(10);
        let json = serde_json::to_string(&params).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        let obj = value.as_object().unwrap();
        assert!(
            obj.contains_key("rate_limit_per_user"),
            "rate_limit_per_user should appear when Some"
        );
        assert_eq!(obj["rate_limit_per_user"], 10);
    }
}
