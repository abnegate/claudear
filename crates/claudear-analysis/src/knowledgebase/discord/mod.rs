use crate::feedback::EmbeddingClient;
use chrono::{DateTime, Utc};
use claudear_core::error::Result;
use claudear_core::types::{
    DiscordChannelKind, DiscordIndexStats, DiscordMessageChunk, DiscordSearchResult,
};
use claudear_storage::FixAttemptTracker;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
pub const DISCORD_INDEX_VERSION: &str = "1";
pub const DISCORD_MESSAGE_CHUNK_SIZE: i16 = 1500;
/// Default number of chunks embedded per batch.
const DEFAULT_BATCH_SIZE: usize = 32;

#[derive(Debug, Clone)]
pub struct DiscordMessageInput {
    pub message_id: String,
    pub channel_id: String, // channel / thread id
    pub guild_id: String,
    pub channel_name: String,
    pub is_thread: bool,
    pub author: String,
    pub content: String,
    pub timestamp: String,
    pub reply_to: Option<String>, // referenced message id, if a reply
}

pub struct DiscordIndexer {
    tracker: Arc<dyn FixAttemptTracker>,
    embedding_client: Arc<EmbeddingClient>,
    batch_size: usize,
}

impl DiscordIndexer {
    pub fn new(
        tracker: Arc<dyn FixAttemptTracker>,
        embedding_client: Arc<EmbeddingClient>,
    ) -> Self {
        Self {
            tracker,
            embedding_client,
            batch_size: DEFAULT_BATCH_SIZE,
        }
    }

    pub fn with_config(
        tracker: Arc<dyn FixAttemptTracker>,
        embedding_client: Arc<EmbeddingClient>,
        batch_size: usize,
    ) -> Self {
        Self {
            tracker,
            embedding_client,
            batch_size: if batch_size == 0 {
                DEFAULT_BATCH_SIZE
            } else {
                batch_size
            },
        }
    }

    pub async fn prepare(&self) -> Result<bool> {
        let mut needs_full_reindex = false;

        if let Ok(Some(stored_model)) = self.tracker.get_discord_message_embedding_model() {
            if stored_model != self.embedding_client.model() {
                needs_full_reindex = true;
            }
        }
        if let Ok(Some(stored_ver)) = self.tracker.get_discord_message_index_meta("index_version") {
            if stored_ver != DISCORD_INDEX_VERSION {
                needs_full_reindex = true;
            }
        }

        if needs_full_reindex {
            tracing::info!(
                "Discord index version or embedding model changed — forcing full re-index"
            );
            self.tracker.delete_all_discord_message_data()?;
        }

        self.tracker
            .set_discord_message_index_meta("index_version", DISCORD_INDEX_VERSION)?;
        Ok(needs_full_reindex)
    }

    pub async fn index(
        &self,
        channel_id: &str,
        is_thread: bool,
        inputs: Vec<DiscordMessageInput>,
    ) -> Result<DiscordIndexStats> {
        tracing::info!(
            channel = %channel_id,
            is_thread,
            messages = inputs.len(),
            "Starting Discord channel indexing"
        );

        // The global version/model freshness gate runs once per orchestrator pass
        // in `prepare()` (called before any channel is indexed) — not here — so a
        // forced full re-index can also reset every channel's cursor before the
        // backfill/incremental path is chosen. See `prepare`.
        let mut stats = DiscordIndexStats::default();

        // grouping by 10min window buckets
        let mut buckets: Vec<Vec<DiscordMessageInput>> = vec![Vec::new()];
        let mut participants = HashMap::from([(0, HashSet::<String>::new())]);
        let mut last_message: Option<DiscordMessageInput> = None;
        for message in inputs {
            if let Some(prev) = &last_message {
                if Self::more_than_10_minutes_apart(&message.timestamp, &prev.timestamp) {
                    buckets.push(Vec::new());
                    participants.insert(buckets.len() - 1, HashSet::new());
                }
            }

            buckets.last_mut().unwrap().push(message.clone());
            let participant_set = participants.get_mut(&(buckets.len() - 1)).unwrap();
            participant_set.insert(message.author.clone());
            last_message = Some(message);
        }

        tracing::debug!(
            channel = %channel_id,
            buckets = buckets.iter().filter(|b| !b.is_empty()).count(),
            "Grouped Discord messages into conversation windows"
        );

        let channel_kind = if is_thread {
            DiscordChannelKind::Thread
        } else {
            DiscordChannelKind::Channel
        };

        let mut pending: Vec<DiscordMessageChunk> = Vec::new();

        for (idx, messages) in buckets.iter().enumerate() {
            if messages.is_empty() {
                continue;
            }

            // Render the conversation window into the text we embed/search.
            let chunk_text = messages
                .iter()
                .map(|m| format!("{}: {}", m.author, m.content))
                .collect::<Vec<_>>()
                .join("\n");

            // Hash first, then skip the expensive embed/insert if it already exists.
            let content_hash = sha256_hex(&chunk_text);
            if self
                .tracker
                .discord_chunk_hash_matches(channel_id, &content_hash)?
            {
                stats.messages_skipped += messages.len();
                continue;
            }

            let mut participant_vec: Vec<String> =
                participants.get(&idx).unwrap().iter().cloned().collect();
            participant_vec.sort();

            let first = messages.first().unwrap();
            let last = messages.last().unwrap();
            let context_text = format!(
                "Discord #{} — participants: {}",
                first.channel_name,
                participant_vec.join(", ")
            );

            pending.push(DiscordMessageChunk {
                id: None,
                guild_id: Some(first.guild_id.clone()),
                channel_id: channel_id.to_string(),
                channel_kind,
                start_message_id: first.message_id.clone(),
                end_message_id: last.message_id.clone(),
                participant_ids: Some(participant_vec),
                start_message_time: first.timestamp.clone(),
                end_message_time: last.timestamp.clone(),
                chunk_text,
                context_text,
                content_hash: Some(content_hash),
            });

            stats.messages_processed += messages.len();

            if pending.len() >= self.batch_size {
                self.flush(&mut pending).await?;
            }
        }

        self.flush(&mut pending).await?;

        self.tracker
            .set_discord_message_index_meta("index_version", DISCORD_INDEX_VERSION)?;

        tracing::info!(channel = %channel_id, %stats, "Discord channel indexing complete");
        Ok(stats)
    }

    async fn flush(&self, chunks: &mut Vec<DiscordMessageChunk>) -> Result<()> {
        if chunks.is_empty() {
            return Ok(());
        }

        tracing::debug!(chunks = chunks.len(), "Flushing Discord chunk batch");
        let chunk_ids = self.tracker.save_discord_chunks(chunks)?;

        // Content-hash dedup: embed each unique text only once.
        let mut unique_texts: Vec<String> = Vec::new();
        let mut hash_to_embed_idx: HashMap<String, usize> = HashMap::new();
        let mut chunk_to_embed_idx: Vec<usize> = Vec::with_capacity(chunks.len());

        for chunk in chunks.iter() {
            let key = chunk.content_hash.as_deref().unwrap_or("");
            if let Some(&idx) = hash_to_embed_idx.get(key) {
                chunk_to_embed_idx.push(idx);
            } else {
                let idx = unique_texts.len();
                unique_texts.push(format!("{}\n\n{}", chunk.context_text, chunk.chunk_text));
                hash_to_embed_idx.insert(key.to_string(), idx);
                chunk_to_embed_idx.push(idx);
            }
        }

        let unique_refs: Vec<&str> = unique_texts.iter().map(|s| s.as_str()).collect();
        match self.embedding_client.embed_batch(&unique_refs).await {
            Ok(unique_embeddings) => {
                let pairs: Vec<(i64, &[f32])> = chunk_ids
                    .iter()
                    .zip(chunk_to_embed_idx.iter())
                    .filter_map(|(&id, &embed_idx)| {
                        unique_embeddings.get(embed_idx).map(|e| (id, e.as_slice()))
                    })
                    .collect();

                let model_name = self.embedding_client.model();
                self.tracker
                    .save_discord_chunk_embeddings(&pairs, model_name)?;
                tracing::debug!(embeddings = pairs.len(), "Saved Discord chunk embeddings");
            }
            Err(e) => {
                tracing::warn!(error = %e, "Failed to embed Discord chunks");
                if let Err(del_err) = self
                    .tracker
                    .delete_discord_message_chunks_by_ids(&chunk_ids)
                {
                    tracing::warn!(error = %del_err, "Failed to delete unembedded Discord chunks");
                }
                // Propagate the failure so callers don't advance their cursor past
                // these messages — otherwise a transient embedding outage would
                // permanently skip them (the chunks were just deleted).
                chunks.clear();
                return Err(e);
            }
        }

        chunks.clear();
        Ok(())
    }

    fn more_than_10_minutes_apart(ts1: &str, ts2: &str) -> bool {
        let t1 = DateTime::parse_from_rfc3339(ts1)
            .unwrap()
            .with_timezone(&Utc);

        let t2 = DateTime::parse_from_rfc3339(ts2)
            .unwrap()
            .with_timezone(&Utc);

        (t2 - t1).num_minutes().abs() > 10
    }
}

pub struct DiscordSearchService {
    tracker: Arc<dyn FixAttemptTracker>,
    embedding_client: Arc<EmbeddingClient>,
}

impl DiscordSearchService {
    pub fn new(
        tracker: Arc<dyn FixAttemptTracker>,
        embedding_client: Arc<EmbeddingClient>,
    ) -> Self {
        Self {
            tracker,
            embedding_client,
        }
    }

    pub async fn search(
        &self,
        query: &str,
        channel_id: Option<&str>,
        limit: usize,
    ) -> Result<Vec<DiscordSearchResult>> {
        let embedding = self.embedding_client.embed(query).await?;
        self.tracker
            .search_discord_message_chunks(&embedding, channel_id, limit)
    }
}

pub fn format_discord_search_context(results: &[DiscordSearchResult]) -> String {
    use std::fmt::Write;

    if results.is_empty() {
        return String::new();
    }

    let mut context = String::from("\n\n## Relevant Discussions from Discord\n\n");
    context.push_str(
        "The following Discord conversations were found to be semantically relevant to this issue. ",
    );
    context.push_str("Use them to understand prior discussion and inform your approach:\n\n");

    for (i, result) in results.iter().enumerate() {
        let chunk = &result.chunk;
        let _ = writeln!(
            context,
            "### {}. {} (Similarity: {:.0}%)",
            i + 1,
            discord_span_links(chunk),
            result.score * 100.0,
        );

        if let Some(participants) = chunk.participant_ids.as_ref().filter(|p| !p.is_empty()) {
            let _ = writeln!(context, "**Participants:** {}", participants.join(", "));
        }

        let _ = writeln!(
            context,
            "**When:** {} – {}",
            chunk.start_message_time, chunk.end_message_time,
        );

        // Truncate long conversations, respecting UTF-8 char boundaries.
        let text = if chunk.chunk_text.len() > 2000 {
            let mut end = 2000;
            while !chunk.chunk_text.is_char_boundary(end) {
                end -= 1;
            }
            format!("{}...\n(truncated)", &chunk.chunk_text[..end])
        } else {
            chunk.chunk_text.clone()
        };

        let _ = writeln!(context, "```\n{}\n```\n", text);
    }

    context
}

/// Build a Discord jump URL to a single message. Falls back to the `@me`
/// (DM) path when the guild id is missing so we always emit a clickable link.
fn discord_jump_url(guild_id: Option<&str>, channel_id: &str, message_id: &str) -> String {
    match guild_id.filter(|g| !g.is_empty()) {
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

/// Render a chunk's channel label with jump links spanning its window: `from`
/// (first message) and `to` (last message). Collapses to a single link when the
/// window is one message.
fn discord_span_links(chunk: &DiscordMessageChunk) -> String {
    let guild = chunk.guild_id.as_deref();
    let start = discord_jump_url(guild, &chunk.channel_id, &chunk.start_message_id);
    if chunk.end_message_id == chunk.start_message_id {
        format!("[Channel `{}`]({})", chunk.channel_id, start)
    } else {
        let end = discord_jump_url(guild, &chunk.channel_id, &chunk.end_message_id);
        format!(
            "Channel `{}` — [from]({}) → [to]({})",
            chunk.channel_id, start, end
        )
    }
}

/// Build a compact Discord-markdown block of jump links to the retrieved
/// discussions, for appending to an outgoing notification so readers can open
/// the referenced conversations. Each entry links the start and end of the
/// conversation window. Returns an empty string when there are no results.
pub fn format_discord_reference_links(results: &[DiscordSearchResult]) -> String {
    use std::fmt::Write;

    if results.is_empty() {
        return String::new();
    }

    let mut out = String::from("\n\n\u{1F4CE} **Referenced Discord discussions**\n");
    for result in results {
        let _ = writeln!(
            out,
            "- {} ({:.0}%)",
            discord_span_links(&result.chunk),
            result.score * 100.0,
        );
    }
    out
}

fn sha256_hex(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Try to create an EmbeddingClient; returns `None` when the ONNX model
    /// cannot be downloaded (common in CI). Tests that need one early-return.
    fn try_embedding_client() -> Option<Arc<EmbeddingClient>> {
        crate::feedback::EmbeddingClient::new(crate::feedback::EmbeddingConfig {
            pool_size: 1,
            ..Default::default()
        })
        .ok()
        .map(Arc::new)
    }

    fn input(id: &str, author: &str, ts: &str) -> DiscordMessageInput {
        DiscordMessageInput {
            message_id: id.to_string(),
            channel_id: "chan1".to_string(),
            guild_id: "guild1".to_string(),
            channel_name: "general".to_string(),
            is_thread: false,
            author: author.to_string(),
            content: format!("message {id}"),
            timestamp: ts.to_string(),
            reply_to: None,
        }
    }

    // ---- pure helpers ---------------------------------------------------

    #[test]
    fn test_sha256_hex_deterministic_and_distinct() {
        let h = sha256_hex("alice: hi\nbob: hey");
        assert_eq!(h.len(), 64);
        assert_eq!(h, sha256_hex("alice: hi\nbob: hey"));
        assert_ne!(h, sha256_hex("alice: hi\nbob: hey!"));
    }

    #[test]
    fn test_more_than_10_minutes_apart() {
        let a = "2024-01-01T10:00:00Z";
        let within = "2024-01-01T10:05:00Z";
        let beyond = "2024-01-01T10:20:00Z";

        assert!(!DiscordIndexer::more_than_10_minutes_apart(a, within));
        assert!(DiscordIndexer::more_than_10_minutes_apart(a, beyond));
        // Order-independent (uses abs).
        assert!(DiscordIndexer::more_than_10_minutes_apart(beyond, a));
        // Exactly equal is not "more than".
        assert!(!DiscordIndexer::more_than_10_minutes_apart(a, a));
    }

    // ---- format_discord_search_context ----------------------------------

    #[test]
    fn test_format_context_empty_is_empty_string() {
        assert!(format_discord_search_context(&[]).is_empty());
    }

    #[test]
    fn test_format_context_renders_fields() {
        let chunk = DiscordMessageChunk {
            id: Some(1),
            guild_id: Some("g".to_string()),
            channel_id: "chan1".to_string(),
            channel_kind: DiscordChannelKind::Channel,
            start_message_id: "1".to_string(),
            end_message_id: "2".to_string(),
            participant_ids: Some(vec!["alice".to_string(), "bob".to_string()]),
            start_message_time: "2024-01-01T10:00:00Z".to_string(),
            end_message_time: "2024-01-01T10:05:00Z".to_string(),
            chunk_text: "alice: hi\nbob: hey".to_string(),
            context_text: "ctx".to_string(),
            content_hash: Some("h".to_string()),
        };
        let out = format_discord_search_context(&[DiscordSearchResult { chunk, score: 0.95 }]);

        assert!(out.contains("## Relevant Discussions from Discord"));
        assert!(out.contains("chan1"));
        assert!(out.contains("95%"));
        assert!(out.contains("alice, bob"));
        assert!(out.contains("alice: hi"));
        // A multi-message window links both the first and last message.
        assert!(out.contains("[from](https://discord.com/channels/g/chan1/1)"));
        assert!(out.contains("[to](https://discord.com/channels/g/chan1/2)"));
    }

    #[test]
    fn test_format_context_without_guild_uses_me_fallback() {
        let chunk = DiscordMessageChunk {
            id: Some(1),
            guild_id: None,
            channel_id: "chan1".to_string(),
            channel_kind: DiscordChannelKind::Channel,
            start_message_id: "1".to_string(),
            end_message_id: "1".to_string(),
            participant_ids: None,
            start_message_time: "2024-01-01T10:00:00Z".to_string(),
            end_message_time: "2024-01-01T10:05:00Z".to_string(),
            chunk_text: "hi".to_string(),
            context_text: "ctx".to_string(),
            content_hash: Some("h".to_string()),
        };
        let out = format_discord_search_context(&[DiscordSearchResult { chunk, score: 0.5 }]);

        // Missing guild id => still linkable via the @me path; single message
        // collapses to one link.
        assert!(out.contains("[Channel `chan1`](https://discord.com/channels/@me/chan1/1)"));
    }

    #[test]
    fn test_reference_links_empty_is_empty_string() {
        assert!(format_discord_reference_links(&[]).is_empty());
    }

    #[test]
    fn test_reference_links_span_and_score() {
        let chunk = DiscordMessageChunk {
            id: Some(1),
            guild_id: Some("g".to_string()),
            channel_id: "chan1".to_string(),
            channel_kind: DiscordChannelKind::Channel,
            start_message_id: "10".to_string(),
            end_message_id: "20".to_string(),
            participant_ids: None,
            start_message_time: "2024-01-01T10:00:00Z".to_string(),
            end_message_time: "2024-01-01T10:05:00Z".to_string(),
            chunk_text: "hi".to_string(),
            context_text: "ctx".to_string(),
            content_hash: Some("h".to_string()),
        };
        let out = format_discord_reference_links(&[DiscordSearchResult { chunk, score: 0.87 }]);

        assert!(out.contains("Referenced Discord discussions"));
        assert!(out.contains("[from](https://discord.com/channels/g/chan1/10)"));
        assert!(out.contains("[to](https://discord.com/channels/g/chan1/20)"));
        assert!(out.contains("87%"));
    }

    // ---- end-to-end index (needs embedding model + sqlite) --------------

    #[tokio::test]
    async fn test_index_creates_then_dedups_chunks() {
        let Some(embedding_client) = try_embedding_client() else {
            return;
        };
        let tracker: Arc<dyn FixAttemptTracker> =
            Arc::new(claudear_storage::SqliteTracker::in_memory().unwrap());
        let indexer = DiscordIndexer::new(tracker, embedding_client);

        // Two messages within 10 min (one bucket) + a third after a >10 min gap
        // (second bucket) => two chunks covering all three messages.
        let inputs = vec![
            input("1", "alice", "2024-01-01T10:00:00Z"),
            input("2", "bob", "2024-01-01T10:05:00Z"),
            input("3", "alice", "2024-01-01T10:20:00Z"),
        ];

        let stats = indexer.index("chan1", false, inputs.clone()).await.unwrap();
        assert_eq!(stats.messages_processed, 3);
        assert_eq!(stats.messages_skipped, 0);

        // Re-indexing identical input => every chunk hash already exists => skip.
        let stats2 = indexer.index("chan1", false, inputs).await.unwrap();
        assert_eq!(stats2.messages_processed, 0);
        assert_eq!(stats2.messages_skipped, 3);
    }

    #[tokio::test]
    async fn test_prepare_wipes_on_version_change() {
        let Some(embedding_client) = try_embedding_client() else {
            return;
        };
        let tracker: Arc<dyn FixAttemptTracker> =
            Arc::new(claudear_storage::SqliteTracker::in_memory().unwrap());
        let indexer = DiscordIndexer::new(tracker.clone(), embedding_client);

        // Index some data and register a channel that has finished backfill.
        indexer
            .index(
                "chan1",
                false,
                vec![input("1", "alice", "2024-01-01T10:00:00Z")],
            )
            .await
            .unwrap();
        tracker
            .set_discord_channel_backfill("chan1", true, Some("1"), "2024-01-01T00:00:00Z")
            .unwrap();

        // Simulate a pipeline-version change since the last run.
        tracker
            .set_discord_message_index_meta("index_version", "stale")
            .unwrap();

        // prepare() must detect the change, wipe data, and reset channel cursors.
        let wiped = indexer.prepare().await.unwrap();
        assert!(wiped);
        assert!(!tracker.discord_chunk_hash_matches("chan1", "any").unwrap());
        let (complete, cursor) = tracker.get_discord_channel_backfill("chan1").unwrap();
        assert!(!complete, "backfill flag reset so channel re-backfills");
        assert_eq!(cursor, None);
        assert_eq!(
            tracker
                .get_discord_message_index_meta("index_version")
                .unwrap()
                .as_deref(),
            Some(DISCORD_INDEX_VERSION)
        );

        // A second prepare() with matching version is a no-op (no wipe).
        assert!(!indexer.prepare().await.unwrap());
    }

    #[tokio::test]
    async fn test_index_then_search_roundtrip() {
        let Some(embedding_client) = try_embedding_client() else {
            return;
        };
        let tracker: Arc<dyn FixAttemptTracker> =
            Arc::new(claudear_storage::SqliteTracker::in_memory().unwrap());
        let indexer = DiscordIndexer::new(tracker.clone(), embedding_client.clone());

        let inputs = vec![
            input("1", "alice", "2024-01-01T10:00:00Z"),
            input("2", "bob", "2024-01-01T10:01:00Z"),
        ];
        indexer.index("chan1", false, inputs).await.unwrap();

        let search = DiscordSearchService::new(tracker, embedding_client);
        // Tolerant of vectorlite being unavailable (search returns empty then),
        // exactly like the code-chunk vector tests.
        let results = search.search("message", Some("chan1"), 5).await.unwrap();
        if let Some(top) = results.first() {
            assert_eq!(top.chunk.channel_id, "chan1");
        }
    }
}
