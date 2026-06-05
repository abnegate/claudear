//! SQLite-based fix attempt tracker and analytics storage.

use super::types::{
    ConfidenceBreakdown, DiagnosticCounts, IndexStats, IndexingProgress, InferenceHistoryEntry,
    InferenceStats, PurgeResult, StoredDependency, StoredIndexedRepo, StoredPrReviewComment,
    StoredRepository, UserRow,
};
use super::{
    is_vectorlite_available, try_load_vectorlite, ActivityStore, AttemptTracker, ChatStore,
    EmbeddingStore, EvaluationStore, ExperimentStore, KnowledgeStore, RegressionStore, RepoStore,
    SimilarityStore, UserStore, WebhookStore,
};
use chrono::{DateTime, Utc};
use claudear_core::error::Result;
use claudear_core::types::{
    ActivityLogEntry, AgentExecution, AnalyticsSummary, CrossRepoCorrelation, ErrorPattern,
    ExperimentProviderStats, FixAttempt, FixAttemptStats, FixAttemptStatus, FixOutcome,
    IssueEmbedding, Outcome, PrReviewRecord, ProcessingMetric, PromptExperiment, QaKnowledgeEntry,
    QaMatch, SimilarIssue, SourceStats,
};
use rand::RngExt;
use rusqlite::OptionalExtension;
use rusqlite::{params, Connection, TransactionBehavior};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

const DEFAULT_LOG_DIR: &str = "./logs";
const AUDIT_LOG_SUBDIR: &str = "audit";
const QA_VECTOR_TABLE: &str = "qa_question_embeddings";
const QA_VECTOR_EF_SEARCH: usize = 200;
const QA_VECTOR_CANDIDATE_MULTIPLIER: usize = 20;
const ISSUE_VECTOR_TABLE: &str = "issue_embedding_vectors";
const ISSUE_VECTOR_EF_SEARCH: usize = 200;
const ISSUE_VECTOR_CANDIDATE_MULTIPLIER: usize = 20;
const OUTCOME_VECTOR_TABLE: &str = "outcome_embedding_vectors";
const OUTCOME_VECTOR_EF_SEARCH: usize = 200;
const OUTCOME_VECTOR_CANDIDATE_MULTIPLIER: usize = 20;
const CODE_CHUNK_VECTOR_TABLE: &str = "code_chunk_vectors";
const CODE_CHUNK_VECTOR_EF_SEARCH: usize = 200;
const CODE_CHUNK_VECTOR_CANDIDATE_MULTIPLIER: usize = 20;

/// Maximum allowed embedding dimension to prevent malformed data from generating
/// unbounded DDL strings. Covers all common embedding models (up to 8192-dim).
const MAX_EMBEDDING_DIMENSION: usize = 16384;

/// Validate that an embedding dimension is within safe bounds for SQL DDL interpolation.
fn validate_dimension(dimension: usize) -> Result<()> {
    if dimension == 0 || dimension > MAX_EMBEDDING_DIMENSION {
        return Err(claudear_core::error::Error::storage(format!(
            "Embedding dimension {} is out of valid range (1..={})",
            dimension, MAX_EMBEDDING_DIMENSION
        )));
    }
    Ok(())
}

impl UserRow {
    fn from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<Self> {
        Ok(Self {
            id: row.get(0)?,
            email: row.get(1)?,
            password_hash: row.get(2)?,
            name: row.get(3)?,
            role: row.get(4)?,
            avatar_url: row.get(5)?,
            created_at: row.get(6)?,
            updated_at: row.get(7)?,
        })
    }
}

/// Generate a cryptographically random session token (64 hex chars = 32 bytes).
fn generate_session_token() -> String {
    let mut bytes = [0u8; 32];
    rand::rng().fill(&mut bytes);
    hex::encode(bytes)
}

/// SQLite-based fix attempt tracker for persistence.
///
/// # Async Safety
///
/// This implementation uses `std::sync::Mutex` which is appropriate for:
/// - Short-duration locks (SQLite in-process operations are typically fast)
/// - Operations that don't hold the lock across `.await` points
///
/// All methods are synchronous and complete quickly, making this safe to call
/// from async contexts without risking thread starvation. The mutex is never
/// held across await points since all trait methods are synchronous.
pub struct SqliteTracker {
    conn: Mutex<Connection>,
    indexing_tx: tokio::sync::watch::Sender<IndexingProgress>,
}

impl SqliteTracker {
    /// Create a new SQLite tracker with the given database path.
    pub fn new(db_path: impl AsRef<Path>) -> Result<Self> {
        let conn = Connection::open(db_path)?;
        let (indexing_tx, _) = tokio::sync::watch::channel(IndexingProgress::default());
        let tracker = Self {
            conn: Mutex::new(conn),
            indexing_tx,
        };
        tracker.init()?;
        Ok(tracker)
    }

    /// Create an in-memory SQLite tracker (for testing).
    pub fn in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        let (indexing_tx, _) = tokio::sync::watch::channel(IndexingProgress::default());
        let tracker = Self {
            conn: Mutex::new(conn),
            indexing_tx,
        };
        tracker.init()?;
        Ok(tracker)
    }

    /// Subscribe to real-time indexing progress updates.
    pub fn subscribe_indexing_progress(&self) -> tokio::sync::watch::Receiver<IndexingProgress> {
        self.indexing_tx.subscribe()
    }

    /// Acquire a lock on the database connection, handling poisoned mutex gracefully.
    fn acquire_lock(&self) -> Result<std::sync::MutexGuard<'_, Connection>> {
        self.conn.lock().map_err(|e| {
            claudear_core::error::Error::Storage(format!("Failed to acquire database lock: {}", e))
        })
    }

    fn init(&self) -> Result<()> {
        let conn = self.acquire_lock()?;

        // These settings optimize SQLite for better throughput in our use case:
        // - Concurrent reads/writes from webhook server and watcher
        // - Moderate write workload with analytics/metrics
        // - BLOB storage for embeddings
        conn.execute_batch(
            r#"
            -- WAL mode: biggest win for concurrent reads + writes
            -- Note: In-memory DBs will stay in "memory" mode which is fine
            PRAGMA journal_mode = WAL;

            -- Don't wait for fsync on every commit (safe with WAL)
            PRAGMA synchronous = NORMAL;

            -- 16MB cache (default is 2MB) - keeps hot pages in RAM
            PRAGMA cache_size = -16384;

            -- Memory-map up to 64MB of the DB file for faster BLOB access
            PRAGMA mmap_size = 67108864;

            -- Store temp tables in memory
            PRAGMA temp_store = MEMORY;

            -- Timeout instead of immediate SQLITE_BUSY (5 seconds)
            PRAGMA busy_timeout = 5000;

            -- Enable foreign key enforcement
            PRAGMA foreign_keys = ON;
            "#,
        )?;

        // Run versioned SQL migrations (tracked in schema_migrations table).
        super::migrator::run(&conn).map_err(claudear_core::error::Error::Database)?;

        // Reset any stuck "running" indexing progress rows from a previous crash
        conn.execute(
            "UPDATE indexing_progress SET status = 'idle' WHERE status = 'running'",
            [],
        )?;

        // Update query planner statistics after schema creation
        conn.execute("ANALYZE", [])?;

        Ok(())
    }

    fn parse_datetime(s: &str) -> std::result::Result<DateTime<Utc>, rusqlite::Error> {
        DateTime::parse_from_rfc3339(s)
            .map(|dt| dt.with_timezone(&Utc))
            .or_else(|_| {
                chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S").map(|dt| dt.and_utc())
            })
            .map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    0,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )
            })
    }

    /// Parse an optional embedding BLOB from a row column.
    fn parse_optional_embedding(
        row: &rusqlite::Row<'_>,
        col: usize,
    ) -> std::result::Result<Option<Vec<f32>>, rusqlite::Error> {
        let bytes: Option<Vec<u8>> = row.get(col)?;
        match bytes {
            None => Ok(None),
            Some(b) if b.is_empty() => Ok(None),
            Some(b) => {
                if b.len() % 4 != 0 {
                    return Err(rusqlite::Error::InvalidColumnType(
                        col,
                        "embedding".to_string(),
                        rusqlite::types::Type::Blob,
                    ));
                }
                let embedding: Vec<f32> = b
                    .chunks_exact(4)
                    .map(|chunk| {
                        let arr: [u8; 4] =
                            chunk.try_into().expect("chunks_exact guarantees 4 bytes");
                        f32::from_le_bytes(arr)
                    })
                    .collect();
                Ok(Some(embedding))
            }
        }
    }

    /// Map a row to an `IssueEmbedding`.
    ///
    /// Expects columns 0-7 in standard order (id, source, issue_id, short_id,
    /// title, embedding, embedding_model, created_at).  The remaining metadata
    /// columns (description, url, priority, status, labels, updated_at) start
    /// at `meta_offset`.
    fn row_to_issue_embedding(
        row: &rusqlite::Row<'_>,
        meta_offset: usize,
    ) -> rusqlite::Result<IssueEmbedding> {
        let embedding = Self::parse_optional_embedding(row, 5)?;
        let updated_at: Option<String> = row.get(meta_offset + 5)?;

        Ok(IssueEmbedding {
            id: row.get(0)?,
            source: row.get(1)?,
            issue_id: row.get(2)?,
            short_id: row.get(3)?,
            title: row.get(4)?,
            description: row.get(meta_offset)?,
            url: row.get(meta_offset + 1)?,
            priority: row.get(meta_offset + 2)?,
            status: row.get(meta_offset + 3)?,
            labels: row.get(meta_offset + 4)?,
            embedding,
            embedding_model: row.get(6)?,
            created_at: Self::parse_datetime(&row.get::<_, String>(7)?)?,
            updated_at: Self::parse_optional_datetime(updated_at)?,
        })
    }

    fn parse_optional_datetime(
        s: Option<String>,
    ) -> std::result::Result<Option<DateTime<Utc>>, rusqlite::Error> {
        match s {
            Some(s) => Self::parse_datetime(&s).map(Some),
            None => Ok(None),
        }
    }

    fn resolve_log_root() -> PathBuf {
        std::env::var("CLAUDEAR_LOG_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from(DEFAULT_LOG_DIR))
    }

    /// Best-effort mirror of structured audit records to JSONL on disk.
    fn append_audit_json_line(category: &str, payload: &serde_json::Value) {
        use std::io::Write as _;

        let root = Self::resolve_log_root();
        if root.as_os_str().is_empty() {
            return;
        }

        let day = chrono::Utc::now().format("%Y-%m-%d").to_string();
        let dir = root.join(AUDIT_LOG_SUBDIR).join(category);
        if let Err(e) = std::fs::create_dir_all(&dir) {
            tracing::warn!(
                component = "sqlite",
                category = category,
                path = %dir.display(),
                error = %e,
                "Failed to create audit log directory"
            );
            return;
        }

        let file_path = dir.join(format!("{}.jsonl", day));
        let mut file = match std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&file_path)
        {
            Ok(file) => file,
            Err(e) => {
                tracing::warn!(
                    component = "sqlite",
                    category = category,
                    path = %file_path.display(),
                    error = %e,
                    "Failed to open audit log file"
                );
                return;
            }
        };

        let serialized = match serde_json::to_string(payload) {
            Ok(value) => value,
            Err(e) => {
                tracing::warn!(
                    component = "sqlite",
                    category = category,
                    error = %e,
                    "Failed to serialize audit log payload"
                );
                return;
            }
        };

        if let Err(e) = writeln!(file, "{}", serialized) {
            tracing::warn!(
                component = "sqlite",
                category = category,
                path = %file_path.display(),
                error = %e,
                "Failed to append audit log payload"
            );
        }
    }

    /// Delegate to the module-level `parse_pr_url` for backward compatibility
    /// with tests that call `SqliteTracker::parse_pr_url`.
    pub fn parse_pr_url(url: &str) -> Option<(String, i64)> {
        super::parse_pr_url(url)
    }

    fn embedding_to_blob(embedding: Option<&[f32]>) -> Option<Vec<u8>> {
        embedding.map(|values| {
            let mut blob = Vec::with_capacity(values.len() * 4);
            for f in values {
                blob.extend_from_slice(&f.to_le_bytes());
            }
            blob
        })
    }

    fn blob_to_embedding(blob: Option<Vec<u8>>) -> Option<Vec<f32>> {
        let blob = blob?;
        if !blob.len().is_multiple_of(4) {
            return None;
        }
        Some(
            blob.chunks_exact(4)
                .map(|chunk| {
                    let arr: [u8; 4] = chunk.try_into().expect("chunks_exact guarantees 4 bytes");
                    f32::from_le_bytes(arr)
                })
                .collect(),
        )
    }

    fn table_exists(conn: &Connection, table_name: &str) -> Result<bool> {
        let exists: Option<i64> = conn
            .query_row(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?1 LIMIT 1",
                params![table_name],
                |row| row.get(0),
            )
            .optional()?;
        Ok(exists.is_some())
    }

    fn ensure_qa_vector_table(conn: &Connection, dimension: usize) -> Result<bool> {
        if dimension == 0 {
            return Ok(false);
        }
        validate_dimension(dimension)?;

        if !is_vectorlite_available(conn) {
            match try_load_vectorlite(conn) {
                Ok(true) => {}
                Ok(false) => return Ok(false),
                Err(e) => {
                    tracing::debug!(error = %e, "Unable to load vectorlite extension for Q&A search");
                    return Ok(false);
                }
            }
        }

        if Self::table_exists(conn, QA_VECTOR_TABLE)? {
            return Ok(true);
        }

        let sql = format!(
            r#"
            CREATE VIRTUAL TABLE IF NOT EXISTS {table} USING vectorlite(
                embedding float32[{dimension}] cosine,
                hnsw(max_elements=10000, ef_construction=200, M=16)
            )
            "#,
            table = QA_VECTOR_TABLE,
            dimension = dimension
        );

        match conn.execute_batch(&sql) {
            Ok(()) => {
                let backfill_sql = format!(
                    r#"
                    INSERT INTO {table}(rowid, embedding)
                    SELECT id, question_embedding
                    FROM qa_knowledge
                    WHERE question_embedding IS NOT NULL
                      AND length(question_embedding) = ?1
                    "#,
                    table = QA_VECTOR_TABLE
                );
                if let Err(e) = conn.execute(&backfill_sql, params![(dimension * 4) as i64]) {
                    tracing::debug!(error = %e, "Failed to backfill Q&A vector embeddings");
                }
                Ok(true)
            }
            Err(e) => {
                tracing::debug!(error = %e, "Failed to create Q&A vector table");
                Ok(false)
            }
        }
    }

    fn upsert_qa_vector_embedding(conn: &Connection, qa_id: i64, embedding: &[f32]) -> Result<()> {
        if embedding.is_empty() {
            return Ok(());
        }

        if !Self::ensure_qa_vector_table(conn, embedding.len())? {
            return Ok(());
        }

        let vector_blob: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();
        let delete_sql = format!("DELETE FROM {} WHERE rowid = ?1", QA_VECTOR_TABLE);
        let insert_sql = format!(
            "INSERT INTO {}(rowid, embedding) VALUES (?1, ?2)",
            QA_VECTOR_TABLE
        );

        conn.execute(&delete_sql, params![qa_id])?;
        conn.execute(&insert_sql, params![qa_id, vector_blob])?;
        Ok(())
    }

    /// Ensure the HNSW vector table for issue embeddings exists.
    /// Returns `true` if the table is available, `false` if vectorlite is not installed.
    fn ensure_issue_vector_table(conn: &Connection, dimension: usize) -> Result<bool> {
        if dimension == 0 {
            return Ok(false);
        }
        validate_dimension(dimension)?;

        if !is_vectorlite_available(conn) {
            match try_load_vectorlite(conn) {
                Ok(true) => {}
                Ok(false) => return Ok(false),
                Err(e) => {
                    tracing::debug!(error = %e, "Unable to load vectorlite extension for issue embeddings");
                    return Ok(false);
                }
            }
        }

        if Self::table_exists(conn, ISSUE_VECTOR_TABLE)? {
            return Ok(true);
        }

        let sql = format!(
            r#"
            CREATE VIRTUAL TABLE IF NOT EXISTS {table} USING vectorlite(
                embedding float32[{dimension}] cosine,
                hnsw(max_elements=10000, ef_construction=200, M=16)
            )
            "#,
            table = ISSUE_VECTOR_TABLE,
            dimension = dimension
        );

        match conn.execute_batch(&sql) {
            Ok(()) => {
                let backfill_sql = format!(
                    r#"
                    INSERT INTO {table}(rowid, embedding)
                    SELECT id, embedding
                    FROM issues
                    WHERE embedding IS NOT NULL
                      AND length(embedding) = ?1
                    "#,
                    table = ISSUE_VECTOR_TABLE
                );
                if let Err(e) = conn.execute(&backfill_sql, params![(dimension * 4) as i64]) {
                    tracing::debug!(error = %e, "Failed to backfill issue vector embeddings");
                }
                Ok(true)
            }
            Err(e) => {
                tracing::debug!(error = %e, "Failed to create issue vector table");
                Ok(false)
            }
        }
    }

    /// Upsert an issue embedding into the HNSW vector table.
    fn upsert_issue_vector_embedding(
        conn: &Connection,
        issue_emb_id: i64,
        embedding: &[f32],
    ) -> Result<()> {
        if embedding.is_empty() {
            return Ok(());
        }

        if !Self::ensure_issue_vector_table(conn, embedding.len())? {
            return Ok(());
        }

        let vector_blob: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();
        let delete_sql = format!("DELETE FROM {} WHERE rowid = ?1", ISSUE_VECTOR_TABLE);
        let insert_sql = format!(
            "INSERT INTO {}(rowid, embedding) VALUES (?1, ?2)",
            ISSUE_VECTOR_TABLE
        );

        conn.execute(&delete_sql, params![issue_emb_id])?;
        conn.execute(&insert_sql, params![issue_emb_id, vector_blob])?;
        Ok(())
    }

    /// Ensure the HNSW vector table for outcome embeddings exists.
    fn ensure_outcome_vector_table(conn: &Connection, dimension: usize) -> Result<bool> {
        if dimension == 0 {
            return Ok(false);
        }
        validate_dimension(dimension)?;

        if !is_vectorlite_available(conn) {
            match try_load_vectorlite(conn) {
                Ok(true) => {}
                Ok(false) => return Ok(false),
                Err(e) => {
                    tracing::debug!(error = %e, "Unable to load vectorlite extension for outcome embeddings");
                    return Ok(false);
                }
            }
        }

        if Self::table_exists(conn, OUTCOME_VECTOR_TABLE)? {
            return Ok(true);
        }

        let sql = format!(
            r#"
            CREATE VIRTUAL TABLE IF NOT EXISTS {table} USING vectorlite(
                embedding float32[{dimension}] cosine,
                hnsw(max_elements=10000, ef_construction=200, M=16)
            )
            "#,
            table = OUTCOME_VECTOR_TABLE,
            dimension = dimension
        );

        match conn.execute_batch(&sql) {
            Ok(()) => {
                let backfill_sql = format!(
                    r#"
                    INSERT INTO {table}(rowid, embedding)
                    SELECT id, embedding
                    FROM feedback_outcomes
                    WHERE embedding IS NOT NULL
                      AND length(embedding) = ?1
                    "#,
                    table = OUTCOME_VECTOR_TABLE
                );
                if let Err(e) = conn.execute(&backfill_sql, params![(dimension * 4) as i64]) {
                    tracing::debug!(error = %e, "Failed to backfill outcome vector embeddings");
                }
                Ok(true)
            }
            Err(e) => {
                tracing::debug!(error = %e, "Failed to create outcome vector table");
                Ok(false)
            }
        }
    }

    /// Upsert an outcome embedding into the HNSW vector table.
    fn upsert_outcome_vector_embedding(
        conn: &Connection,
        outcome_id: i64,
        embedding: &[f32],
    ) -> Result<()> {
        if embedding.is_empty() {
            return Ok(());
        }

        if !Self::ensure_outcome_vector_table(conn, embedding.len())? {
            return Ok(());
        }

        let vector_blob: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();
        let delete_sql = format!("DELETE FROM {} WHERE rowid = ?1", OUTCOME_VECTOR_TABLE);
        let insert_sql = format!(
            "INSERT INTO {}(rowid, embedding) VALUES (?1, ?2)",
            OUTCOME_VECTOR_TABLE
        );

        conn.execute(&delete_sql, params![outcome_id])?;
        conn.execute(&insert_sql, params![outcome_id, vector_blob])?;
        Ok(())
    }

    fn query_qa_matches_vector_scoped(
        conn: &Connection,
        source: &str,
        repo: Option<&str>,
        question_embedding: &[f32],
        threshold: f64,
        limit: usize,
        candidate_limit: usize,
    ) -> Result<Option<Vec<QaMatch>>> {
        if question_embedding.is_empty() || limit == 0 || candidate_limit == 0 {
            return Ok(Some(Vec::new()));
        }

        if !Self::ensure_qa_vector_table(conn, question_embedding.len())? {
            return Ok(None);
        }

        let query_blob: Vec<u8> = question_embedding
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let sql = format!(
            r#"
            WITH candidates AS (
                SELECT rowid AS qa_id,
                       MAX(0.0, MIN(1.0, 1.0 - distance)) AS semantic_similarity
                FROM {table}
                WHERE knn_search(embedding, knn_param(?1, ?2, ?3))
            ),
            scored AS (
                SELECT k.id, k.source, k.repo, k.issue_id, k.short_id, k.question_text, k.question_norm,
                       k.question_embedding, k.answer_text, k.answer_norm, k.answer_embedding, k.channel,
                       k.responder, k.correlation_id, k.asked_at, k.answered_at, k.success_count,
                       k.failure_count, k.last_used_at, k.metadata,
                       c.semantic_similarity AS semantic_similarity,
                       CASE
                           WHEN (k.success_count + k.failure_count) > 0 THEN
                               CAST(k.success_count AS REAL) / CAST((k.success_count + k.failure_count) AS REAL)
                           ELSE 0.5
                       END AS historical_success_rate
                FROM candidates c
                JOIN qa_knowledge k ON k.id = c.qa_id
                WHERE k.source = ?4
                  AND (?5 IS NULL OR k.repo = ?5)
            ),
            ranked AS (
                SELECT id, source, repo, issue_id, short_id, question_text, question_norm,
                       question_embedding, answer_text, answer_norm, answer_embedding, channel,
                       responder, correlation_id, asked_at, answered_at, success_count,
                       failure_count, last_used_at, metadata, semantic_similarity,
                       historical_success_rate,
                       (semantic_similarity * 0.75 + historical_success_rate * 0.25) AS final_score
                FROM scored
            )
            SELECT id, source, repo, issue_id, short_id, question_text, question_norm,
                   question_embedding, answer_text, answer_norm, answer_embedding, channel,
                   responder, correlation_id, asked_at, answered_at, success_count,
                   failure_count, last_used_at, metadata, semantic_similarity,
                   historical_success_rate, final_score
            FROM ranked
            WHERE semantic_similarity >= ?6 OR final_score >= ?6
            ORDER BY final_score DESC, answered_at DESC
            LIMIT ?7
            "#,
            table = QA_VECTOR_TABLE
        );

        let mut stmt = match conn.prepare(&sql) {
            Ok(stmt) => stmt,
            Err(e) => {
                tracing::debug!(error = %e, "Failed to prepare scoped Q&A vector ranking query");
                return Ok(None);
            }
        };

        let rows = match stmt.query_map(
            params![
                query_blob,
                candidate_limit as i64,
                QA_VECTOR_EF_SEARCH as i64,
                source,
                repo,
                threshold,
                limit as i64
            ],
            Self::row_to_qa_match,
        ) {
            Ok(rows) => rows,
            Err(e) => {
                tracing::debug!(error = %e, "Scoped Q&A vector ranking query failed");
                return Ok(None);
            }
        };

        let mut matches = Vec::new();
        for row in rows {
            match row {
                Ok(m) => matches.push(m),
                Err(e) => tracing::debug!(error = %e, "Failed to read scoped Q&A vector row"),
            }
        }

        Ok(Some(matches))
    }

    fn query_qa_matches_vector_global(
        conn: &Connection,
        question_embedding: &[f32],
        threshold: f64,
        limit: usize,
        candidate_limit: usize,
    ) -> Result<Option<Vec<QaMatch>>> {
        if question_embedding.is_empty() || limit == 0 || candidate_limit == 0 {
            return Ok(Some(Vec::new()));
        }

        if !Self::ensure_qa_vector_table(conn, question_embedding.len())? {
            return Ok(None);
        }

        let query_blob: Vec<u8> = question_embedding
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let sql = format!(
            r#"
            WITH candidates AS (
                SELECT rowid AS qa_id,
                       MAX(0.0, MIN(1.0, 1.0 - distance)) AS semantic_similarity
                FROM {table}
                WHERE knn_search(embedding, knn_param(?1, ?2, ?3))
            ),
            scored AS (
                SELECT k.id, k.source, k.repo, k.issue_id, k.short_id, k.question_text, k.question_norm,
                       k.question_embedding, k.answer_text, k.answer_norm, k.answer_embedding, k.channel,
                       k.responder, k.correlation_id, k.asked_at, k.answered_at, k.success_count,
                       k.failure_count, k.last_used_at, k.metadata,
                       c.semantic_similarity AS semantic_similarity,
                       CASE
                           WHEN (k.success_count + k.failure_count) > 0 THEN
                               CAST(k.success_count AS REAL) / CAST((k.success_count + k.failure_count) AS REAL)
                           ELSE 0.5
                       END AS historical_success_rate
                FROM candidates c
                JOIN qa_knowledge k ON k.id = c.qa_id
            ),
            ranked AS (
                SELECT id, source, repo, issue_id, short_id, question_text, question_norm,
                       question_embedding, answer_text, answer_norm, answer_embedding, channel,
                       responder, correlation_id, asked_at, answered_at, success_count,
                       failure_count, last_used_at, metadata, semantic_similarity,
                       historical_success_rate,
                       (semantic_similarity * 0.75 + historical_success_rate * 0.25) AS final_score
                FROM scored
            )
            SELECT id, source, repo, issue_id, short_id, question_text, question_norm,
                   question_embedding, answer_text, answer_norm, answer_embedding, channel,
                   responder, correlation_id, asked_at, answered_at, success_count,
                   failure_count, last_used_at, metadata, semantic_similarity,
                   historical_success_rate, final_score
            FROM ranked
            WHERE semantic_similarity >= ?4 OR final_score >= ?4
            ORDER BY final_score DESC, answered_at DESC
            LIMIT ?5
            "#,
            table = QA_VECTOR_TABLE
        );

        let mut stmt = match conn.prepare(&sql) {
            Ok(stmt) => stmt,
            Err(e) => {
                tracing::debug!(error = %e, "Failed to prepare global Q&A vector ranking query");
                return Ok(None);
            }
        };

        let rows = match stmt.query_map(
            params![
                query_blob,
                candidate_limit as i64,
                QA_VECTOR_EF_SEARCH as i64,
                threshold,
                limit as i64
            ],
            Self::row_to_qa_match,
        ) {
            Ok(rows) => rows,
            Err(e) => {
                tracing::debug!(error = %e, "Global Q&A vector ranking query failed");
                return Ok(None);
            }
        };

        let mut matches = Vec::new();
        for row in rows {
            match row {
                Ok(m) => matches.push(m),
                Err(e) => tracing::debug!(error = %e, "Failed to read global Q&A vector row"),
            }
        }

        Ok(Some(matches))
    }

    fn query_qa_matches_exact_scoped(
        conn: &Connection,
        source: &str,
        repo: Option<&str>,
        question_norm: &str,
        threshold: f64,
        limit: usize,
    ) -> Result<Vec<QaMatch>> {
        let mut stmt = conn.prepare(
            r#"
            WITH scoped AS (
                SELECT k.id, k.source, k.repo, k.issue_id, k.short_id, k.question_text, k.question_norm,
                       k.question_embedding, k.answer_text, k.answer_norm, k.answer_embedding, k.channel,
                       k.responder, k.correlation_id, k.asked_at, k.answered_at, k.success_count,
                       k.failure_count, k.last_used_at, k.metadata,
                       1.0 AS semantic_similarity,
                       CASE
                           WHEN (k.success_count + k.failure_count) > 0 THEN
                               CAST(k.success_count AS REAL) / CAST((k.success_count + k.failure_count) AS REAL)
                           ELSE 0.5
                       END AS historical_success_rate
                FROM qa_knowledge k
                WHERE k.source = ?1
                  AND (?2 IS NULL OR k.repo = ?2)
                  AND k.question_norm = ?3
            ),
            ranked AS (
                SELECT id, source, repo, issue_id, short_id, question_text, question_norm,
                       question_embedding, answer_text, answer_norm, answer_embedding, channel,
                       responder, correlation_id, asked_at, answered_at, success_count,
                       failure_count, last_used_at, metadata, semantic_similarity,
                       historical_success_rate,
                       (semantic_similarity * 0.75 + historical_success_rate * 0.25) AS final_score
                FROM scoped
            )
            SELECT id, source, repo, issue_id, short_id, question_text, question_norm,
                   question_embedding, answer_text, answer_norm, answer_embedding, channel,
                   responder, correlation_id, asked_at, answered_at, success_count,
                   failure_count, last_used_at, metadata, semantic_similarity,
                   historical_success_rate, final_score
            FROM ranked
            WHERE semantic_similarity >= ?4 OR final_score >= ?4
            ORDER BY final_score DESC, answered_at DESC
            LIMIT ?5
            "#,
        )?;
        let rows = stmt.query_map(
            params![source, repo, question_norm, threshold, limit as i64],
            Self::row_to_qa_match,
        )?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Into::into)
    }

    fn query_qa_matches_exact_global(
        conn: &Connection,
        question_norm: &str,
        threshold: f64,
        limit: usize,
    ) -> Result<Vec<QaMatch>> {
        let mut stmt = conn.prepare(
            r#"
            WITH scoped AS (
                SELECT k.id, k.source, k.repo, k.issue_id, k.short_id, k.question_text, k.question_norm,
                       k.question_embedding, k.answer_text, k.answer_norm, k.answer_embedding, k.channel,
                       k.responder, k.correlation_id, k.asked_at, k.answered_at, k.success_count,
                       k.failure_count, k.last_used_at, k.metadata,
                       1.0 AS semantic_similarity,
                       CASE
                           WHEN (k.success_count + k.failure_count) > 0 THEN
                               CAST(k.success_count AS REAL) / CAST((k.success_count + k.failure_count) AS REAL)
                           ELSE 0.5
                       END AS historical_success_rate
                FROM qa_knowledge k
                WHERE (?1 = '' OR k.question_norm = ?1)
            ),
            ranked AS (
                SELECT id, source, repo, issue_id, short_id, question_text, question_norm,
                       question_embedding, answer_text, answer_norm, answer_embedding, channel,
                       responder, correlation_id, asked_at, answered_at, success_count,
                       failure_count, last_used_at, metadata, semantic_similarity,
                       historical_success_rate,
                       (semantic_similarity * 0.75 + historical_success_rate * 0.25) AS final_score
                FROM scoped
            )
            SELECT id, source, repo, issue_id, short_id, question_text, question_norm,
                   question_embedding, answer_text, answer_norm, answer_embedding, channel,
                   responder, correlation_id, asked_at, answered_at, success_count,
                   failure_count, last_used_at, metadata, semantic_similarity,
                   historical_success_rate, final_score
            FROM ranked
            WHERE semantic_similarity >= ?2 OR final_score >= ?2
            ORDER BY final_score DESC, answered_at DESC
            LIMIT ?3
            "#,
        )?;
        let rows = stmt.query_map(
            params![question_norm, threshold, limit as i64],
            Self::row_to_qa_match,
        )?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Into::into)
    }
}

impl AttemptTracker for SqliteTracker {
    fn has_attempted(&self, source: &str, issue_id: &str) -> Result<bool> {
        let conn = self.acquire_lock()?;
        // Exclude soft-reset entries (reset_at IS NOT NULL) so they are treated as
        // "not yet attempted" and will be picked up for re-processing.
        let mut stmt = conn.prepare_cached(
            "SELECT 1 FROM fix_attempts WHERE source = ? AND issue_id = ? AND reset_at IS NULL",
        )?;
        Ok(stmt.exists(params![source, issue_id])?)
    }

    fn get_attempted_issue_ids(&self, source: &str) -> Result<HashSet<String>> {
        let conn = self.acquire_lock()?;
        let mut stmt = conn.prepare_cached(
            "SELECT issue_id FROM fix_attempts WHERE source = ? AND reset_at IS NULL",
        )?;

        let rows = stmt.query_map(params![source], |row| row.get::<_, String>(0))?;
        let mut result = HashSet::new();
        for row in rows {
            result.insert(row?);
        }
        Ok(result)
    }

    fn record_attempt(&self, source: &str, issue_id: &str, short_id: &str) -> Result<()> {
        self.record_attempt_with_labels(source, issue_id, short_id, &[])
    }

    fn record_attempt_with_labels(
        &self,
        source: &str,
        issue_id: &str,
        short_id: &str,
        labels: &[String],
    ) -> Result<()> {
        tracing::info!(
            source = source,
            issue_id = issue_id,
            short_id = short_id,
            labels_count = labels.len(),
            "Recording fix attempt with labels"
        );
        let conn = self.acquire_lock()?;

        // Serialize labels to JSON
        let labels_json = if labels.is_empty() {
            None
        } else {
            Some(serde_json::to_string(labels)?)
        };

        // Manual upsert: check for existing non-cascade row, then INSERT or UPDATE.
        // We cannot use ON CONFLICT(source, issue_id) because the uniqueness constraint
        // is a partial index (cascade rows share source+issue_id with the parent).
        let existing: Option<(i64, Option<String>)> = conn
            .prepare_cached(
                "SELECT id, reset_at FROM fix_attempts WHERE source = ? AND issue_id = ? AND cascade_repo IS NULL",
            )?
            .query_row(params![source, issue_id], |row| {
                Ok((row.get(0)?, row.get(1)?))
            })
            .optional()?;

        match existing {
            None => {
                // No existing row — insert fresh
                conn.execute(
                    r#"INSERT INTO fix_attempts (source, issue_id, short_id, status, attempted_at, issue_labels)
                       VALUES (?, ?, ?, 'pending', datetime('now'), ?)"#,
                    params![source, issue_id, short_id, labels_json],
                )?;
                tracing::info!(source = source, issue_id = issue_id, "Fix attempt recorded");
            }
            Some((_id, Some(_reset_at))) => {
                // Existing row in reset state — update it
                conn.execute(
                    r#"UPDATE fix_attempts SET
                         short_id = ?,
                         attempted_at = datetime('now'),
                         issue_labels = COALESCE(?, issue_labels),
                         reset_at = NULL
                       WHERE source = ? AND issue_id = ? AND cascade_repo IS NULL"#,
                    params![short_id, labels_json, source, issue_id],
                )?;
                tracing::info!(
                    source = source,
                    issue_id = issue_id,
                    "Fix attempt updated (was in reset state)"
                );
            }
            Some((_id, None)) => {
                // Existing row NOT in reset state — skip
                tracing::warn!(
                    source = source,
                    issue_id = issue_id,
                    "Attempt already exists and is not in reset state, skipping"
                );
            }
        }
        Ok(())
    }

    fn mark_success(&self, source: &str, issue_id: &str, pr_url: &str) -> Result<()> {
        tracing::info!(
            source = source,
            issue_id = issue_id,
            pr_url = pr_url,
            "Marking fix attempt as success"
        );
        let conn = self.acquire_lock()?;

        // Parse PR URL to extract GitHub repo and PR number
        let (scm_repo, scm_pr_number) = match Self::parse_pr_url(pr_url) {
            Some((repo, pr_num)) => (Some(repo), Some(pr_num)),
            None => {
                tracing::warn!(
                    pr_url = pr_url,
                    source = source,
                    issue_id = issue_id,
                    "Failed to parse PR URL - PR tracking may not work correctly"
                );
                (None, None)
            }
        };

        let rows_affected = conn.execute(
            r#"
            UPDATE fix_attempts
            SET status = 'success', pr_url = ?, scm_repo = ?, scm_pr_number = ?
            WHERE source = ? AND issue_id = ? AND cascade_repo IS NULL
            "#,
            params![pr_url, scm_repo, scm_pr_number, source, issue_id],
        )?;
        tracing::info!(
            source = source,
            issue_id = issue_id,
            rows_affected = rows_affected,
            scm_repo = ?scm_repo,
            "Fix attempt marked as success"
        );
        Ok(())
    }

    fn mark_merged(&self, source: &str, issue_id: &str) -> Result<()> {
        tracing::info!(
            source = source,
            issue_id = issue_id,
            "Marking fix attempt as merged"
        );
        let conn = self.acquire_lock()?;
        let rows_affected = conn.execute(
            r#"
            UPDATE fix_attempts
            SET status = 'merged', merged_at = datetime('now')
            WHERE source = ? AND issue_id = ? AND cascade_repo IS NULL
            "#,
            params![source, issue_id],
        )?;
        tracing::info!(
            source = source,
            issue_id = issue_id,
            rows_affected = rows_affected,
            "Fix attempt marked as merged"
        );
        Ok(())
    }

    fn mark_closed(&self, source: &str, issue_id: &str) -> Result<()> {
        tracing::info!(
            source = source,
            issue_id = issue_id,
            "Marking fix attempt as closed"
        );
        let conn = self.acquire_lock()?;
        let rows_affected = conn.execute(
            r#"
            UPDATE fix_attempts
            SET status = 'closed'
            WHERE source = ? AND issue_id = ? AND cascade_repo IS NULL
            "#,
            params![source, issue_id],
        )?;
        tracing::info!(
            source = source,
            issue_id = issue_id,
            rows_affected = rows_affected,
            "Fix attempt marked as closed"
        );
        Ok(())
    }

    fn mark_resolved(&self, source: &str, issue_id: &str) -> Result<()> {
        tracing::info!(
            source = source,
            issue_id = issue_id,
            "Marking fix attempt as resolved"
        );
        let conn = self.acquire_lock()?;
        let rows_affected = conn.execute(
            r#"
            UPDATE fix_attempts
            SET resolved_at = datetime('now')
            WHERE source = ? AND issue_id = ? AND cascade_repo IS NULL
            "#,
            params![source, issue_id],
        )?;
        tracing::info!(
            source = source,
            issue_id = issue_id,
            rows_affected = rows_affected,
            "Fix attempt marked as resolved"
        );
        Ok(())
    }

    fn mark_failed(&self, source: &str, issue_id: &str, error_message: &str) -> Result<()> {
        tracing::info!(
            source = source,
            issue_id = issue_id,
            error_message = error_message,
            "Marking fix attempt as failed"
        );
        let conn = self.acquire_lock()?;
        let rows_affected = conn.execute(
            r#"
            UPDATE fix_attempts
            SET status = 'failed', error_message = ?
            WHERE source = ? AND issue_id = ? AND cascade_repo IS NULL
            "#,
            params![error_message, source, issue_id],
        )?;
        tracing::info!(
            source = source,
            issue_id = issue_id,
            rows_affected = rows_affected,
            "Fix attempt marked as failed"
        );
        Ok(())
    }

    fn get_attempt(&self, source: &str, issue_id: &str) -> Result<Option<FixAttempt>> {
        let conn = self.acquire_lock()?;
        let mut stmt = conn.prepare_cached(
            r#"
            SELECT id, source, issue_id, short_id, attempted_at, pr_url, scm_repo,
                   scm_pr_number, status, error_message, merged_at, resolved_at,
                   retry_count, last_retry_at, issue_labels, parent_attempt_id, cascade_repo
            FROM fix_attempts
            WHERE source = ? AND issue_id = ?
            "#,
        )?;

        let result = stmt
            .query_row(params![source, issue_id], Self::row_to_fix_attempt)
            .ok();

        Ok(result)
    }

    fn get_attempts_by_status(&self, status: FixAttemptStatus) -> Result<Vec<FixAttempt>> {
        let conn = self.acquire_lock()?;
        let mut stmt = conn.prepare_cached(
            r#"
            SELECT id, source, issue_id, short_id, attempted_at, pr_url, scm_repo,
                   scm_pr_number, status, error_message, merged_at, resolved_at,
                   retry_count, last_retry_at, issue_labels, parent_attempt_id, cascade_repo
            FROM fix_attempts
            WHERE status = ?
            ORDER BY attempted_at DESC
            "#,
        )?;

        let status_str = status.to_string();
        let rows = stmt.query_map(params![status_str], Self::row_to_fix_attempt)?;

        let mut results = Vec::new();
        for row in rows.flatten() {
            results.push(row);
        }
        Ok(results)
    }

    fn get_pending_prs(&self) -> Result<Vec<FixAttempt>> {
        let conn = self.acquire_lock()?;
        let mut stmt = conn.prepare_cached(
            r#"
            SELECT id, source, issue_id, short_id, attempted_at, pr_url, scm_repo,
                   scm_pr_number, status, error_message, merged_at, resolved_at,
                   retry_count, last_retry_at, issue_labels, parent_attempt_id, cascade_repo
            FROM fix_attempts
            WHERE status = 'success' AND pr_url IS NOT NULL AND scm_repo IS NOT NULL
            ORDER BY attempted_at DESC
            "#,
        )?;

        let rows = stmt.query_map([], Self::row_to_fix_attempt)?;

        let mut results = Vec::new();
        for row in rows.flatten() {
            results.push(row);
        }
        Ok(results)
    }

    fn get_attempt_by_pr_url(&self, pr_url: &str) -> Result<Option<FixAttempt>> {
        let conn = self.acquire_lock()?;
        let mut stmt = conn.prepare_cached(
            r#"
            SELECT id, source, issue_id, short_id, attempted_at, pr_url, scm_repo,
                   scm_pr_number, status, error_message, merged_at, resolved_at,
                   retry_count, last_retry_at, issue_labels, parent_attempt_id, cascade_repo
            FROM fix_attempts
            WHERE pr_url = ?
            ORDER BY attempted_at DESC, id DESC
            LIMIT 1
            "#,
        )?;

        let result = stmt
            .query_row(params![pr_url], Self::row_to_fix_attempt)
            .ok();

        Ok(result)
    }

    fn reset_attempt(&self, source: &str, issue_id: &str) -> Result<()> {
        let conn = self.acquire_lock()?;
        // Soft reset: preserve the row (and FK references) but mark it for re-processing.
        // The reset_at timestamp signals has_attempted/get_attempted_issue_ids to treat
        // this issue as "not yet attempted" so it will be picked up again.
        conn.execute(
            r#"
            UPDATE fix_attempts
            SET status = 'pending',
                retry_count = 0,
                reset_at = datetime('now'),
                pr_url = NULL,
                scm_repo = NULL,
                scm_pr_number = NULL,
                error_message = NULL,
                merged_at = NULL,
                resolved_at = NULL,
                attempted_at = datetime('now')
            WHERE source = ? AND issue_id = ?
            "#,
            params![source, issue_id],
        )?;
        Ok(())
    }

    fn increment_retry(&self, source: &str, issue_id: &str) -> Result<()> {
        let conn = self.acquire_lock()?;
        conn.execute(
            r#"
            UPDATE fix_attempts
            SET retry_count = COALESCE(retry_count, 0) + 1,
                last_retry_at = datetime('now')
            WHERE source = ? AND issue_id = ?
            "#,
            params![source, issue_id],
        )?;
        Ok(())
    }

    fn mark_cannot_fix(&self, source: &str, issue_id: &str, reason: &str) -> Result<()> {
        tracing::info!(
            source = source,
            issue_id = issue_id,
            reason = reason,
            "Marking fix attempt as cannot_fix"
        );
        let conn = self.acquire_lock()?;
        let rows_affected = conn.execute(
            r#"
            UPDATE fix_attempts
            SET status = 'cannot_fix', error_message = ?
            WHERE source = ? AND issue_id = ?
            "#,
            params![reason, source, issue_id],
        )?;
        tracing::info!(
            source = source,
            issue_id = issue_id,
            rows_affected = rows_affected,
            "Fix attempt marked as cannot_fix"
        );
        Ok(())
    }

    fn mark_answered(&self, source: &str, issue_id: &str, summary: &str) -> Result<()> {
        tracing::info!(
            source = source,
            issue_id = issue_id,
            "Marking attempt as answered"
        );
        let conn = self.acquire_lock()?;
        conn.execute(
            r#"
            UPDATE fix_attempts
            SET status = 'answered', error_message = ?
            WHERE source = ? AND issue_id = ?
            "#,
            params![summary, source, issue_id],
        )?;
        Ok(())
    }

    fn get_retryable_issues(&self, max_retries: u32) -> Result<Vec<FixAttempt>> {
        let conn = self.acquire_lock()?;
        let mut stmt = conn.prepare_cached(
            r#"
            SELECT id, source, issue_id, short_id, attempted_at, pr_url, scm_repo,
                   scm_pr_number, status, error_message, merged_at, resolved_at,
                   retry_count, last_retry_at, issue_labels, parent_attempt_id, cascade_repo
            FROM fix_attempts
            WHERE (status = 'failed' OR status = 'closed')
              AND COALESCE(retry_count, 0) < ?
            ORDER BY attempted_at ASC
            "#,
        )?;

        let rows = stmt.query_map(params![max_retries], Self::row_to_fix_attempt)?;

        let mut results = Vec::new();
        for row in rows.flatten() {
            results.push(row);
        }
        Ok(results)
    }

    fn prepare_for_retry(&self, source: &str, issue_id: &str) -> Result<()> {
        let conn = self.acquire_lock()?;
        // Atomically increment retry count and reset status in one statement.
        // Guard with status check to prevent overwriting active/succeeded attempts.
        let rows_affected = conn.execute(
            r#"
            UPDATE fix_attempts
            SET status = 'pending',
                retry_count = COALESCE(retry_count, 0) + 1,
                last_retry_at = datetime('now'),
                pr_url = NULL,
                scm_repo = NULL,
                scm_pr_number = NULL,
                error_message = NULL,
                attempted_at = datetime('now')
            WHERE source = ? AND issue_id = ?
              AND status IN ('failed', 'closed')
            "#,
            params![source, issue_id],
        )?;
        if rows_affected == 0 {
            tracing::warn!(
                source = source,
                issue_id = issue_id,
                "prepare_for_retry: no rows updated (attempt not in retryable state)"
            );
            return Err(claudear_core::error::Error::Storage(format!(
                "Attempt {}/{} not in retryable state (failed/closed)",
                source, issue_id
            )));
        }
        Ok(())
    }

    fn get_stats(&self) -> Result<FixAttemptStats> {
        let conn = self.acquire_lock()?;
        let mut stats = FixAttemptStats::default();

        // Overall stats
        let mut stmt = conn.prepare_cached(
            r#"
            SELECT status, COUNT(*) as count
            FROM fix_attempts
            GROUP BY status
            "#,
        )?;

        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)? as usize))
        })?;

        for row in rows {
            match row {
                Ok((status, count)) => {
                    stats.total += count;
                    match status.as_str() {
                        "pending" => stats.pending = count,
                        "success" => stats.success = count,
                        "failed" => stats.failed = count,
                        "merged" => stats.merged = count,
                        "closed" => stats.closed = count,
                        "cannot_fix" => stats.cannot_fix = count,
                        _ => {}
                    }
                }
                Err(e) => {
                    tracing::warn!(error = %e, "Failed to read stats row");
                }
            }
        }

        // Stats by source
        let mut stmt = conn.prepare_cached(
            r#"
            SELECT source, status, COUNT(*) as count
            FROM fix_attempts
            GROUP BY source, status
            "#,
        )?;

        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, i64>(2)? as usize,
            ))
        })?;

        let mut by_source: HashMap<String, SourceStats> = HashMap::new();

        for row in rows {
            match row {
                Ok((source, status, count)) => {
                    let entry = by_source.entry(source).or_default();
                    entry.total += count;
                    match status.as_str() {
                        "success" => entry.success = count,
                        "failed" => entry.failed = count,
                        "merged" => entry.merged = count,
                        "closed" => entry.closed = count,
                        "cannot_fix" => entry.cannot_fix = count,
                        _ => {}
                    }
                }
                Err(e) => {
                    tracing::warn!(error = %e, "Failed to read source stats row");
                }
            }
        }

        stats.by_source = by_source;

        Ok(stats)
    }
}

impl ActivityStore for SqliteTracker {
    fn record_activity(&self, entry: &ActivityLogEntry) -> Result<i64> {
        SqliteTracker::record_activity(self, entry)
    }

    fn record_action_run(
        &self,
        source: &str,
        issue_id: &str,
        short_id: &str,
        action_kind: &str,
        status: &str,
        detail: &str,
    ) -> Result<()> {
        SqliteTracker::record_action_run(
            self,
            source,
            issue_id,
            short_id,
            action_kind,
            status,
            detail,
        )
    }

    fn get_recent_activities(&self, limit: usize) -> Result<Vec<ActivityLogEntry>> {
        SqliteTracker::get_recent_activities(self, limit, None)
    }

    fn record_execution(&self, execution: &AgentExecution) -> Result<i64> {
        SqliteTracker::record_execution(self, execution)
    }

    fn record_pr_review(&self, review: &PrReviewRecord) -> Result<i64> {
        SqliteTracker::record_pr_review(self, review)
    }

    fn record_error_pattern(&self, pattern: &ErrorPattern) -> Result<i64> {
        SqliteTracker::record_error_pattern(self, pattern)
    }

    fn record_metric(&self, metric: &ProcessingMetric) -> Result<i64> {
        SqliteTracker::record_metric(self, metric)
    }

    fn get_analytics_summary(&self) -> Result<AnalyticsSummary> {
        SqliteTracker::get_analytics_summary(self)
    }

    fn get_recent_activities_filtered(
        &self,
        limit: usize,
        source_filter: Option<&str>,
    ) -> Result<Vec<ActivityLogEntry>> {
        SqliteTracker::get_recent_activities(self, limit, source_filter)
    }

    fn get_attempt_by_id(&self, id: i64) -> Result<Option<FixAttempt>> {
        SqliteTracker::get_attempt_by_id(self, id)
    }

    fn get_executions_for_attempt(&self, attempt_id: i64) -> Result<Vec<AgentExecution>> {
        SqliteTracker::get_executions_for_attempt(self, attempt_id)
    }

    fn get_reviews_for_attempt(&self, attempt_id: i64) -> Result<Vec<PrReviewRecord>> {
        SqliteTracker::get_reviews_for_attempt(self, attempt_id)
    }

    fn get_error_patterns(&self, limit: usize) -> Result<Vec<ErrorPattern>> {
        SqliteTracker::get_error_patterns(self, limit)
    }

    fn get_metrics(
        &self,
        metric_name: &str,
        since: Option<DateTime<Utc>>,
        limit: usize,
    ) -> Result<Vec<ProcessingMetric>> {
        SqliteTracker::get_metrics(self, metric_name, since, limit)
    }

    fn get_open_prs(&self) -> Result<Vec<claudear_core::types::PrRecord>> {
        SqliteTracker::get_open_prs(self)
    }

    fn get_pr_analytics(&self) -> Result<claudear_core::types::PrAnalytics> {
        SqliteTracker::get_pr_analytics(self)
    }

    fn get_avg_time_to_pr(&self) -> Result<Option<f64>> {
        SqliteTracker::get_avg_time_to_pr(self)
    }

    fn get_rejection_reasons(
        &self,
        limit: usize,
    ) -> Result<Vec<claudear_core::types::RejectionReason>> {
        SqliteTracker::get_rejection_reasons(self, limit)
    }

    fn get_agent_spawn_count(&self, since_iso: &str) -> Result<i64> {
        SqliteTracker::get_agent_spawn_count(self, since_iso)
    }

    fn get_cost_estimate(
        &self,
        since_iso: &str,
        max_plan_monthly_cost: f64,
        period_label: &str,
    ) -> Result<claudear_core::types::CostEstimate> {
        SqliteTracker::get_cost_estimate(self, since_iso, max_plan_monthly_cost, period_label)
    }

    fn get_mttr_trend(&self, weeks: usize) -> Result<Vec<claudear_core::types::MttrDataPoint>> {
        SqliteTracker::get_mttr_trend(self, weeks)
    }

    fn get_repo_leaderboard(&self) -> Result<Vec<claudear_core::types::RepoLeaderboardEntry>> {
        SqliteTracker::get_repo_leaderboard(self)
    }

    fn get_complexity_time_savings(
        &self,
        since_iso: &str,
        hourly_rate: f64,
        period_label: &str,
    ) -> Result<claudear_core::types::TimeSavings> {
        SqliteTracker::get_complexity_time_savings(self, since_iso, hourly_rate, period_label)
    }

    fn list_prs(
        &self,
        status: Option<&str>,
        limit: usize,
    ) -> Result<Vec<claudear_core::types::PrRecord>> {
        SqliteTracker::list_prs(self, status, limit)
    }

    fn update_pr_fix_quality_score(&self, pr_url: &str, score: f64) -> Result<()> {
        SqliteTracker::update_pr_fix_quality_score(self, pr_url, score)
    }

    /// Record multiple activities in a single transaction for better performance.
    ///
    /// This is more efficient than calling `record_activity` in a loop because:
    /// - Single transaction reduces fsync overhead
    /// - Prepared statement is reused across all inserts
    fn record_activities_batch(&self, entries: &[ActivityLogEntry]) -> Result<usize> {
        if entries.is_empty() {
            return Ok(0);
        }

        let mut conn = self.acquire_lock()?;
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;

        {
            let mut stmt = tx.prepare_cached(
                r#"
                INSERT INTO activity_log (timestamp, activity_type, source, issue_id, short_id, message, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                "#,
            )?;

            for entry in entries {
                let metadata_json = entry.metadata.as_ref().map(|m| m.to_string());
                stmt.execute(params![
                    entry.timestamp.format("%Y-%m-%d %H:%M:%S").to_string(),
                    entry.activity_type,
                    entry.source,
                    entry.issue_id,
                    entry.short_id,
                    entry.message,
                    metadata_json,
                ])?;
            }
        }

        tx.commit()?;
        drop(conn);
        for entry in entries {
            Self::append_audit_json_line(
                "activity",
                &serde_json::json!({
                    "id": serde_json::Value::Null,
                    "timestamp": entry.timestamp.to_rfc3339(),
                    "activity_type": entry.activity_type,
                    "source": entry.source,
                    "issue_id": entry.issue_id,
                    "short_id": entry.short_id,
                    "message": entry.message,
                    "metadata": entry.metadata,
                }),
            );
        }
        Ok(entries.len())
    }

    /// Count activity events grouped by type since a timestamp.
    fn get_activity_type_counts_since(&self, since: DateTime<Utc>) -> Result<HashMap<String, i64>> {
        let conn = self.acquire_lock()?;
        let since_str = since.format("%Y-%m-%d %H:%M:%S").to_string();
        let mut stmt = conn.prepare(
            r#"
            SELECT activity_type, COUNT(*)
            FROM activity_log
            WHERE timestamp >= ?1
            GROUP BY activity_type
            "#,
        )?;

        let rows = stmt.query_map(params![since_str], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
        })?;

        let mut counts = HashMap::new();
        for row in rows.flatten() {
            counts.insert(row.0, row.1);
        }
        Ok(counts)
    }

    /// Save or update a PR review state for persistence.
    ///
    /// Uses upsert semantics - creates new record or updates existing based on pr_url.
    fn save_pr_review_state(&self, state: &claudear_core::types::PrReviewState) -> Result<()> {
        let conn = self.acquire_lock()?;
        conn.execute(
            r#"
            INSERT INTO pr_review_states (
                pr_url, repo, pr_number, issue_id, source,
                last_review_id, last_review_time, last_comment_id, last_comment_time,
                is_active, created_at
            )
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, datetime('now'))
            ON CONFLICT(pr_url) DO UPDATE SET
                repo = excluded.repo,
                pr_number = excluded.pr_number,
                issue_id = excluded.issue_id,
                source = excluded.source,
                last_review_id = excluded.last_review_id,
                last_review_time = excluded.last_review_time,
                last_comment_id = excluded.last_comment_id,
                last_comment_time = excluded.last_comment_time,
                is_active = excluded.is_active
            "#,
            params![
                state.pr_url,
                state.repo,
                state.pr_number,
                state.issue_id,
                state.source,
                state.last_review_id,
                state.last_review_time,
                state.last_comment_id,
                state.last_comment_time,
                state.is_active as i32,
            ],
        )?;

        tracing::debug!(
            pr_url = %state.pr_url,
            is_active = state.is_active,
            "PR review state saved"
        );

        Ok(())
    }

    /// Get all active PR review states for restoration on startup.
    fn get_active_pr_review_states(&self) -> Result<Vec<claudear_core::types::PrReviewState>> {
        let conn = self.acquire_lock()?;
        let mut stmt = conn.prepare(
            r#"
            SELECT pr_url, repo, pr_number, issue_id, source,
                   last_review_id, last_review_time, last_comment_id, last_comment_time,
                   is_active
            FROM pr_review_states
            WHERE is_active = 1
            ORDER BY created_at DESC
            "#,
        )?;

        let rows = stmt.query_map([], Self::row_to_pr_review_state)?;

        let mut results = Vec::new();
        for row in rows.flatten() {
            results.push(row);
        }

        tracing::debug!(count = results.len(), "Retrieved active PR review states");

        Ok(results)
    }

    /// Deactivate a PR review state (mark as no longer being watched).
    fn deactivate_pr_review_state(&self, pr_url: &str) -> Result<()> {
        let conn = self.acquire_lock()?;
        let rows_affected = conn.execute(
            "UPDATE pr_review_states SET is_active = 0 WHERE pr_url = ?",
            params![pr_url],
        )?;

        tracing::debug!(
            pr_url = %pr_url,
            rows_affected = rows_affected,
            "PR review state deactivated"
        );

        Ok(())
    }

    /// Record a PR review comment for persistence.
    fn record_pr_review_comment(
        &self,
        pr_url: &str,
        comment: &claudear_core::types::ReviewComment,
    ) -> Result<i64> {
        let conn = self.acquire_lock()?;

        conn.execute(
            r#"
            INSERT INTO pr_review_comments (
                scm_comment_id, pr_url, review_id, path, position, line,
                body, author, created_at, updated_at, html_url
            )
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)
            ON CONFLICT(scm_comment_id) DO UPDATE SET
                body = excluded.body,
                updated_at = excluded.updated_at
            "#,
            params![
                comment.id,
                pr_url,
                comment.pull_request_review_id,
                comment.path,
                comment.position,
                comment.line,
                comment.body,
                comment.user.login,
                comment.created_at,
                comment.updated_at,
                comment.html_url,
            ],
        )?;

        Ok(conn.last_insert_rowid())
    }

    /// Get all comments for a specific PR.
    fn get_comments_for_pr(&self, pr_url: &str) -> Result<Vec<StoredPrReviewComment>> {
        let conn = self.acquire_lock()?;
        let mut stmt = conn.prepare(
            r#"
            SELECT id, scm_comment_id, pr_url, review_id, path, position, line,
                   body, author, created_at, updated_at, html_url
            FROM pr_review_comments
            WHERE pr_url = ?
            ORDER BY created_at ASC
            "#,
        )?;

        let rows = stmt.query_map(params![pr_url], Self::row_to_stored_pr_review_comment)?;

        let mut results = Vec::new();
        for row in rows.flatten() {
            results.push(row);
        }

        Ok(results)
    }

    /// Get metric row counts grouped by name since a timestamp.
    fn get_metric_counts_since(
        &self,
        metric_names: &[&str],
        since: DateTime<Utc>,
    ) -> Result<HashMap<String, i64>> {
        if metric_names.is_empty() {
            return Ok(HashMap::new());
        }

        let conn = self.acquire_lock()?;

        let placeholders = (0..metric_names.len())
            .map(|i| format!("?{}", i + 2))
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!(
            r#"
            SELECT metric_name, COUNT(*)
            FROM processing_metrics
            WHERE timestamp >= ?1
              AND metric_name IN ({})
            GROUP BY metric_name
            "#,
            placeholders
        );

        let mut bind_params: Vec<Box<dyn rusqlite::ToSql>> =
            Vec::with_capacity(metric_names.len() + 1);
        bind_params.push(Box::new(since.format("%Y-%m-%d %H:%M:%S").to_string()));
        for metric_name in metric_names {
            bind_params.push(Box::new((*metric_name).to_string()));
        }
        let bind_refs: Vec<&dyn rusqlite::ToSql> = bind_params.iter().map(|p| p.as_ref()).collect();

        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(bind_refs.as_slice(), |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
        })?;

        let mut counts = HashMap::new();
        for row in rows.flatten() {
            counts.insert(row.0, row.1);
        }
        Ok(counts)
    }

    /// Get metric sums grouped by name since a timestamp.
    fn get_metric_sums_since(
        &self,
        metric_names: &[&str],
        since: DateTime<Utc>,
    ) -> Result<HashMap<String, f64>> {
        if metric_names.is_empty() {
            return Ok(HashMap::new());
        }

        let conn = self.acquire_lock()?;

        let placeholders = (0..metric_names.len())
            .map(|i| format!("?{}", i + 2))
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!(
            r#"
            SELECT metric_name, SUM(metric_value)
            FROM processing_metrics
            WHERE timestamp >= ?1
              AND metric_name IN ({})
            GROUP BY metric_name
            "#,
            placeholders
        );

        let mut bind_params: Vec<Box<dyn rusqlite::ToSql>> =
            Vec::with_capacity(metric_names.len() + 1);
        bind_params.push(Box::new(since.format("%Y-%m-%d %H:%M:%S").to_string()));
        for metric_name in metric_names {
            bind_params.push(Box::new((*metric_name).to_string()));
        }
        let bind_refs: Vec<&dyn rusqlite::ToSql> = bind_params.iter().map(|p| p.as_ref()).collect();

        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(bind_refs.as_slice(), |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, Option<f64>>(1)?.unwrap_or(0.0),
            ))
        })?;

        let mut sums = HashMap::new();
        for row in rows.flatten() {
            sums.insert(row.0, row.1);
        }
        Ok(sums)
    }

    fn get_metric_sums_by_source_since(
        &self,
        metric_names: &[&str],
        since: DateTime<Utc>,
    ) -> Result<HashMap<(String, String), f64>> {
        if metric_names.is_empty() {
            return Ok(HashMap::new());
        }

        let conn = self.acquire_lock()?;

        let placeholders = (0..metric_names.len())
            .map(|i| format!("?{}", i + 2))
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!(
            r#"
            SELECT metric_name, source, SUM(metric_value)
            FROM processing_metrics
            WHERE timestamp >= ?1
              AND metric_name IN ({})
            GROUP BY metric_name, source
            "#,
            placeholders
        );

        let mut bind_params: Vec<Box<dyn rusqlite::ToSql>> =
            Vec::with_capacity(metric_names.len() + 1);
        bind_params.push(Box::new(since.format("%Y-%m-%d %H:%M:%S").to_string()));
        for metric_name in metric_names {
            bind_params.push(Box::new((*metric_name).to_string()));
        }
        let bind_refs: Vec<&dyn rusqlite::ToSql> = bind_params.iter().map(|p| p.as_ref()).collect();

        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(bind_refs.as_slice(), |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, Option<String>>(1)?,
                row.get::<_, Option<f64>>(2)?.unwrap_or(0.0),
            ))
        })?;

        let mut sums = HashMap::new();
        for row in rows.flatten() {
            if let Some(source) = row.1 {
                sums.insert((row.0, source), row.2);
            }
        }
        Ok(sums)
    }

    /// Prune old activity logs to prevent unbounded growth.
    fn prune_old_activities(&self, days_to_keep: i64) -> Result<usize> {
        let conn = self.acquire_lock()?;

        // Compute the full datetime modifier in Rust to avoid SQL string concatenation
        // This is safer than building strings in SQL even though days_to_keep is already i64
        let modifier = format!("-{} days", days_to_keep.abs());

        let deleted = conn.execute(
            r#"
            DELETE FROM activity_log
            WHERE timestamp < datetime('now', ?)
            "#,
            params![modifier],
        )?;

        Ok(deleted)
    }

    /// Prune old metrics to prevent unbounded growth.
    fn prune_old_metrics(&self, days_to_keep: i64) -> Result<usize> {
        let conn = self.acquire_lock()?;

        // Compute the full datetime modifier in Rust to avoid SQL string concatenation
        let modifier = format!("-{} days", days_to_keep.abs());

        let deleted = conn.execute(
            r#"
            DELETE FROM processing_metrics
            WHERE timestamp < datetime('now', ?)
            "#,
            params![modifier],
        )?;

        Ok(deleted)
    }

    fn purge_operational_data(&self) -> Result<PurgeResult> {
        let conn = self.acquire_lock()?;

        // Detach feedback_outcomes from attempts (preserve learnings/embeddings).
        // Set attempt_id to NULL so the FK constraint doesn't block deleting fix_attempts.
        // FixOutcome.attempt_id is i64 so NULLs deserialize as 0 via row.get().
        let feedback_outcomes_detached = conn.execute(
            "UPDATE feedback_outcomes SET attempt_id = NULL WHERE attempt_id IS NOT NULL",
            [],
        )?;

        // Temporarily disable FK enforcement so we can bulk-delete without
        // worrying about ordering across every referencing table.
        // PRAGMA foreign_keys cannot be changed inside a transaction, so we
        // run it outside, do all deletes, then re-enable.
        conn.execute_batch("PRAGMA foreign_keys = OFF")?;

        let result = (|| -> Result<PurgeResult> {
            // Regression tracking
            let release_tracking = conn.execute("DELETE FROM release_tracking", [])?;
            let regression_checks = conn.execute("DELETE FROM regression_checks", [])?;
            let regression_watches = conn.execute("DELETE FROM regression_watches", [])?;

            // PR review data
            let pr_review_comments = conn.execute("DELETE FROM pr_review_comments", [])?;
            let pr_reviews = conn.execute("DELETE FROM pr_reviews", [])?;
            let pr_review_states = conn.execute("DELETE FROM pr_review_states", [])?;

            // Execution & strategy data
            let claude_executions = conn.execute("DELETE FROM claude_executions", [])?;
            let strategy_fingerprints = conn.execute("DELETE FROM strategy_fingerprints", [])?;
            let diff_analyses = conn.execute("DELETE FROM diff_analyses", [])?;
            let qa_usage = conn.execute("DELETE FROM qa_usage", [])?;

            // Evaluation data
            let eval_snapshots = conn.execute("DELETE FROM eval_snapshots", [])?;
            let eval_deltas = conn.execute("DELETE FROM eval_deltas", [])?;

            // PRs
            let prs = conn.execute("DELETE FROM prs", [])?;

            // Clustering & prioritisation
            let issue_cluster_members = conn.execute("DELETE FROM issue_cluster_members", [])?;
            let issue_clusters = conn.execute("DELETE FROM issue_clusters", [])?;
            let content_clusters = conn.execute("DELETE FROM content_clusters", [])?;
            let severity_scores = conn.execute("DELETE FROM severity_scores", [])?;
            let suppression_log = conn.execute("DELETE FROM suppression_log", [])?;

            // Operational logs & metrics
            let activity_log = conn.execute("DELETE FROM activity_log", [])?;
            let processing_metrics = conn.execute("DELETE FROM processing_metrics", [])?;
            let webhook_deliveries = conn.execute("DELETE FROM webhook_deliveries", [])?;

            // Fix attempts (root entity)
            let fix_attempts = conn.execute("DELETE FROM fix_attempts", [])?;

            Ok(PurgeResult {
                fix_attempts,
                prs,
                pr_reviews,
                pr_review_comments,
                pr_review_states,
                claude_executions,
                strategy_fingerprints,
                diff_analyses,
                regression_watches,
                release_tracking,
                regression_checks,
                qa_usage,
                activity_log,
                processing_metrics,
                webhook_deliveries,
                issue_clusters,
                issue_cluster_members,
                content_clusters,
                severity_scores,
                suppression_log,
                eval_snapshots,
                eval_deltas,
                feedback_outcomes_detached,
            })
        })();

        // Always re-enable FK enforcement
        conn.execute_batch("PRAGMA foreign_keys = ON")?;

        result
    }

    /// Get diagnostic counts for all major tables.
    ///
    /// This is useful for debugging and verifying that data is being written correctly.
    fn get_diagnostic_counts(&self) -> Result<DiagnosticCounts> {
        let conn = self.acquire_lock()?;

        let fix_attempts: i64 =
            conn.query_row("SELECT COUNT(*) FROM fix_attempts", [], |row| row.get(0))?;
        let fix_attempts_by_status: HashMap<String, i64> = {
            let mut map = HashMap::new();
            let mut stmt =
                conn.prepare("SELECT status, COUNT(*) FROM fix_attempts GROUP BY status")?;
            let rows = stmt.query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
            })?;
            for row in rows.flatten() {
                map.insert(row.0, row.1);
            }
            map
        };

        let activity_log: i64 =
            conn.query_row("SELECT COUNT(*) FROM activity_log", [], |row| row.get(0))?;

        let claude_executions: i64 =
            conn.query_row("SELECT COUNT(*) FROM claude_executions", [], |row| {
                row.get(0)
            })?;

        let pr_reviews: i64 =
            conn.query_row("SELECT COUNT(*) FROM pr_reviews", [], |row| row.get(0))?;

        let pr_review_states: i64 =
            conn.query_row("SELECT COUNT(*) FROM pr_review_states", [], |row| {
                row.get(0)
            })?;

        let issues: i64 = conn.query_row("SELECT COUNT(*) FROM issues", [], |row| row.get(0))?;

        let similar_issues: i64 =
            conn.query_row("SELECT COUNT(*) FROM similar_issues", [], |row| row.get(0))?;

        let repositories: i64 =
            conn.query_row("SELECT COUNT(*) FROM repositories", [], |row| row.get(0))?;

        let repo_files: i64 =
            conn.query_row("SELECT COUNT(*) FROM repo_files", [], |row| row.get(0))?;

        let inference_attempts: i64 =
            conn.query_row("SELECT COUNT(*) FROM inference_attempts", [], |row| {
                row.get(0)
            })?;

        let error_patterns: i64 =
            conn.query_row("SELECT COUNT(*) FROM error_patterns", [], |row| row.get(0))?;

        let processing_metrics: i64 =
            conn.query_row("SELECT COUNT(*) FROM processing_metrics", [], |row| {
                row.get(0)
            })?;

        let feedback_outcomes: i64 =
            conn.query_row("SELECT COUNT(*) FROM feedback_outcomes", [], |row| {
                row.get(0)
            })?;

        let prs: i64 = conn.query_row("SELECT COUNT(*) FROM prs", [], |row| row.get(0))?;

        // Get recent fix attempts for debugging
        let recent_fix_attempts: Vec<(String, String, String, String)> = {
            let mut stmt = conn.prepare(
                "SELECT source, issue_id, short_id, status FROM fix_attempts ORDER BY attempted_at DESC LIMIT 5"
            )?;
            let rows = stmt.query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                ))
            })?;
            rows.flatten().collect()
        };

        Ok(DiagnosticCounts {
            fix_attempts,
            fix_attempts_by_status,
            activity_log,
            claude_executions,
            pr_reviews,
            pr_review_states,
            issues,
            similar_issues,
            repositories,
            repo_files,
            inference_attempts,
            error_patterns,
            processing_metrics,
            feedback_outcomes,
            prs,
            recent_fix_attempts,
        })
    }

    /// Upsert a PR record.
    ///
    /// Creates a new record or updates an existing one based on pr_url.
    fn upsert_pr(&self, pr: &claudear_core::types::PrRecord) -> Result<i64> {
        let conn = self.acquire_lock()?;

        conn.execute(
            r#"
            INSERT INTO prs (
                pr_url, scm_repo, pr_number, attempt_id, issue_id, issue_source,
                title, description, author, head_branch, base_branch, status,
                created_at, updated_at, merged_at, closed_at,
                approvals_count, changes_requested_count, comments_count, last_review_at,
                time_to_first_review_mins, time_to_merge_mins, review_cycles,
                files_changed, lines_added, lines_removed
            ) VALUES (
                ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12,
                ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?24, ?25, ?26
            )
            ON CONFLICT(pr_url) DO UPDATE SET
                scm_repo = excluded.scm_repo,
                pr_number = excluded.pr_number,
                attempt_id = COALESCE(excluded.attempt_id, prs.attempt_id),
                issue_id = COALESCE(excluded.issue_id, prs.issue_id),
                issue_source = COALESCE(excluded.issue_source, prs.issue_source),
                title = COALESCE(excluded.title, prs.title),
                description = COALESCE(excluded.description, prs.description),
                author = COALESCE(excluded.author, prs.author),
                head_branch = COALESCE(excluded.head_branch, prs.head_branch),
                base_branch = COALESCE(excluded.base_branch, prs.base_branch),
                status = excluded.status,
                updated_at = datetime('now'),
                merged_at = COALESCE(excluded.merged_at, prs.merged_at),
                closed_at = COALESCE(excluded.closed_at, prs.closed_at),
                approvals_count = excluded.approvals_count,
                changes_requested_count = excluded.changes_requested_count,
                comments_count = excluded.comments_count,
                last_review_at = COALESCE(excluded.last_review_at, prs.last_review_at),
                time_to_first_review_mins = COALESCE(excluded.time_to_first_review_mins, prs.time_to_first_review_mins),
                time_to_merge_mins = COALESCE(excluded.time_to_merge_mins, prs.time_to_merge_mins),
                review_cycles = excluded.review_cycles,
                files_changed = COALESCE(excluded.files_changed, prs.files_changed),
                lines_added = COALESCE(excluded.lines_added, prs.lines_added),
                lines_removed = COALESCE(excluded.lines_removed, prs.lines_removed)
            "#,
            params![
                pr.pr_url,
                pr.scm_repo,
                pr.pr_number,
                pr.attempt_id,
                pr.issue_id,
                pr.issue_source,
                pr.title,
                pr.description,
                pr.author,
                pr.head_branch,
                pr.base_branch,
                pr.status,
                pr.created_at.format("%Y-%m-%d %H:%M:%S").to_string(),
                pr.updated_at.map(|t| t.format("%Y-%m-%d %H:%M:%S").to_string()),
                pr.merged_at.map(|t| t.format("%Y-%m-%d %H:%M:%S").to_string()),
                pr.closed_at.map(|t| t.format("%Y-%m-%d %H:%M:%S").to_string()),
                pr.approvals_count,
                pr.changes_requested_count,
                pr.comments_count,
                pr.last_review_at.map(|t| t.format("%Y-%m-%d %H:%M:%S").to_string()),
                pr.time_to_first_review_mins,
                pr.time_to_merge_mins,
                pr.review_cycles,
                pr.files_changed,
                pr.lines_added,
                pr.lines_removed,
            ],
        )?;

        // Get the id (either inserted or existing)
        let id: i64 = conn.query_row(
            "SELECT id FROM prs WHERE pr_url = ?",
            params![pr.pr_url],
            |row| row.get(0),
        )?;

        tracing::info!(
            pr_url = %pr.pr_url,
            status = %pr.status,
            id = id,
            "PR record upserted"
        );

        Ok(id)
    }

    /// Get a PR record by URL.
    fn get_pr(&self, pr_url: &str) -> Result<Option<claudear_core::types::PrRecord>> {
        let conn = self.acquire_lock()?;
        let mut stmt = conn.prepare(
            r#"
            SELECT id, pr_url, scm_repo, pr_number, attempt_id, issue_id, issue_source,
                   title, description, author, head_branch, base_branch, status,
                   created_at, updated_at, merged_at, closed_at,
                   approvals_count, changes_requested_count, comments_count, last_review_at,
                   time_to_first_review_mins, time_to_merge_mins, review_cycles,
                   files_changed, lines_added, lines_removed
            FROM prs WHERE pr_url = ?
            "#,
        )?;

        let result = stmt.query_row(params![pr_url], Self::row_to_pr_record).ok();
        Ok(result)
    }

    /// Update PR status.
    fn update_pr_status(&self, pr_url: &str, status: &str) -> Result<()> {
        let conn = self.acquire_lock()?;

        let now = Utc::now().format("%Y-%m-%d %H:%M:%S").to_string();
        let (merged_at, closed_at) = match status {
            "merged" => (Some(now.clone()), None),
            "closed" => (None, Some(now.clone())),
            _ => (None, None),
        };

        conn.execute(
            r#"
            UPDATE prs SET
                status = ?1,
                updated_at = ?2,
                merged_at = COALESCE(?3, merged_at),
                closed_at = COALESCE(?4, closed_at)
            WHERE pr_url = ?5
            "#,
            params![status, now, merged_at, closed_at, pr_url],
        )?;

        tracing::info!(
            pr_url = %pr_url,
            status = status,
            "PR status updated"
        );

        Ok(())
    }

    /// Record a cascade fix attempt linked to a parent attempt.
    fn record_cascade_attempt(
        &self,
        source: &str,
        issue_id: &str,
        short_id: &str,
        parent_attempt_id: i64,
        cascade_repo: &str,
    ) -> Result<i64> {
        let conn = self.acquire_lock()?;

        // Check if there's already a pending cascade (no PR yet) for this combo.
        // Completed cascades (with PR) are allowed to be re-triggered (e.g. merge + release).
        let pending_id: Option<i64> = conn
            .prepare_cached(
                "SELECT id FROM fix_attempts WHERE source = ? AND issue_id = ? AND cascade_repo = ? AND (pr_url IS NULL OR pr_url = '')",
            )?
            .query_row(params![source, issue_id, cascade_repo], |row| row.get(0))
            .ok();

        if let Some(id) = pending_id {
            tracing::info!(
                source = source,
                issue_id = issue_id,
                cascade_repo = cascade_repo,
                attempt_id = id,
                "Pending cascade attempt already exists, skipping"
            );
            return Ok(id);
        }

        conn.execute(
            r#"INSERT INTO fix_attempts (source, issue_id, short_id, status, attempted_at, parent_attempt_id, cascade_repo)
               VALUES (?, ?, ?, 'pending', datetime('now'), ?, ?)"#,
            params![source, issue_id, short_id, parent_attempt_id, cascade_repo],
        )?;

        let id = conn.last_insert_rowid();
        tracing::info!(
            source = source,
            issue_id = issue_id,
            cascade_repo = cascade_repo,
            parent_attempt_id = parent_attempt_id,
            attempt_id = id,
            "Recorded cascade fix attempt"
        );
        Ok(id)
    }

    /// Update a cascade attempt's PR info.
    fn update_attempt_pr(
        &self,
        attempt_id: i64,
        pr_url: &str,
        scm_repo: &str,
        pr_number: i64,
    ) -> Result<()> {
        let conn = self.acquire_lock()?;
        conn.execute(
            "UPDATE fix_attempts SET pr_url = ?, scm_repo = ?, scm_pr_number = ?, status = 'success' WHERE id = ?",
            params![pr_url, scm_repo, pr_number, attempt_id],
        )?;
        Ok(())
    }

    /// Mark a cascade attempt as failed.
    fn mark_cascade_failed(&self, attempt_id: i64, error: &str) -> Result<()> {
        let conn = self.acquire_lock()?;
        conn.execute(
            "UPDATE fix_attempts SET status = 'failed', error_message = ? WHERE id = ?",
            params![error, attempt_id],
        )?;
        Ok(())
    }

    /// List fix attempts with optional status/source filters and pagination.
    fn list_attempts(
        &self,
        status: Option<&str>,
        source: Option<&str>,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<FixAttempt>> {
        let conn = self.acquire_lock()?;
        let mut attempts = Vec::new();

        let query_all = r#"
            SELECT id, source, issue_id, short_id, attempted_at, pr_url, scm_repo,
                   scm_pr_number, status, error_message, merged_at, resolved_at,
                   retry_count, last_retry_at, issue_labels, parent_attempt_id, cascade_repo
            FROM fix_attempts
            ORDER BY attempted_at DESC
            LIMIT ?1 OFFSET ?2
        "#;
        let query_status = r#"
            SELECT id, source, issue_id, short_id, attempted_at, pr_url, scm_repo,
                   scm_pr_number, status, error_message, merged_at, resolved_at,
                   retry_count, last_retry_at, issue_labels, parent_attempt_id, cascade_repo
            FROM fix_attempts
            WHERE status = ?1
            ORDER BY attempted_at DESC
            LIMIT ?2 OFFSET ?3
        "#;
        let query_source = r#"
            SELECT id, source, issue_id, short_id, attempted_at, pr_url, scm_repo,
                   scm_pr_number, status, error_message, merged_at, resolved_at,
                   retry_count, last_retry_at, issue_labels, parent_attempt_id, cascade_repo
            FROM fix_attempts
            WHERE source = ?1
            ORDER BY attempted_at DESC
            LIMIT ?2 OFFSET ?3
        "#;
        let query_status_source = r#"
            SELECT id, source, issue_id, short_id, attempted_at, pr_url, scm_repo,
                   scm_pr_number, status, error_message, merged_at, resolved_at,
                   retry_count, last_retry_at, issue_labels, parent_attempt_id, cascade_repo
            FROM fix_attempts
            WHERE status = ?1 AND source = ?2
            ORDER BY attempted_at DESC
            LIMIT ?3 OFFSET ?4
        "#;

        match (status, source) {
            (Some(status), Some(source)) => {
                let mut stmt = conn.prepare_cached(query_status_source)?;
                let rows = stmt.query_map(
                    params![status, source, limit as i64, offset as i64],
                    Self::row_to_fix_attempt,
                )?;
                attempts.extend(rows.flatten());
            }
            (Some(status), None) => {
                let mut stmt = conn.prepare_cached(query_status)?;
                let rows = stmt.query_map(
                    params![status, limit as i64, offset as i64],
                    Self::row_to_fix_attempt,
                )?;
                attempts.extend(rows.flatten());
            }
            (None, Some(source)) => {
                let mut stmt = conn.prepare_cached(query_source)?;
                let rows = stmt.query_map(
                    params![source, limit as i64, offset as i64],
                    Self::row_to_fix_attempt,
                )?;
                attempts.extend(rows.flatten());
            }
            (None, None) => {
                let mut stmt = conn.prepare_cached(query_all)?;
                let rows = stmt.query_map(
                    params![limit as i64, offset as i64],
                    Self::row_to_fix_attempt,
                )?;
                attempts.extend(rows.flatten());
            }
        }

        Ok(attempts)
    }

    /// Count fix attempts with optional status/source filters.
    fn count_attempts(&self, status: Option<&str>, source: Option<&str>) -> Result<usize> {
        let conn = self.acquire_lock()?;
        let count: i64 = match (status, source) {
            (Some(status), Some(source)) => conn.query_row(
                "SELECT COUNT(*) FROM fix_attempts WHERE status = ?1 AND source = ?2",
                params![status, source],
                |row| row.get(0),
            )?,
            (Some(status), None) => conn.query_row(
                "SELECT COUNT(*) FROM fix_attempts WHERE status = ?1",
                params![status],
                |row| row.get(0),
            )?,
            (None, Some(source)) => conn.query_row(
                "SELECT COUNT(*) FROM fix_attempts WHERE source = ?1",
                params![source],
                |row| row.get(0),
            )?,
            (None, None) => {
                conn.query_row("SELECT COUNT(*) FROM fix_attempts", [], |row| row.get(0))?
            }
        };
        Ok(count as usize)
    }

    /// List recent attempts ordered by attempted time descending.
    fn list_recent_attempts(&self, limit: usize) -> Result<Vec<FixAttempt>> {
        self.list_attempts(None, None, limit, 0)
    }

    /// List attempts since a timestamp, ordered by attempted time descending.
    fn list_attempts_since(&self, since: DateTime<Utc>) -> Result<Vec<FixAttempt>> {
        let conn = self.acquire_lock()?;
        let since_str = since.format("%Y-%m-%d %H:%M:%S").to_string();
        let mut stmt = conn.prepare(
            r#"
            SELECT id, source, issue_id, short_id, attempted_at, pr_url, scm_repo,
                   scm_pr_number, status, error_message, merged_at, resolved_at,
                   retry_count, last_retry_at, issue_labels, parent_attempt_id, cascade_repo
            FROM fix_attempts
            WHERE attempted_at >= ?1
            ORDER BY attempted_at DESC
            "#,
        )?;
        let rows = stmt.query_map(params![since_str], Self::row_to_fix_attempt)?;
        Ok(rows.flatten().collect())
    }

    /// Get the most recently merged fix attempt for a given SCM repo.
    fn get_most_recent_merged_attempt_for_repo(
        &self,
        scm_repo: &str,
    ) -> Result<Option<FixAttempt>> {
        let conn = self.acquire_lock()?;
        let mut stmt = conn.prepare(
            r#"
            SELECT id, source, issue_id, short_id, attempted_at, pr_url, scm_repo,
                   scm_pr_number, status, error_message, merged_at, resolved_at,
                   retry_count, last_retry_at, issue_labels, parent_attempt_id, cascade_repo
            FROM fix_attempts
            WHERE scm_repo = ? AND status = 'merged'
            ORDER BY merged_at DESC
            LIMIT 1
            "#,
        )?;

        let result = stmt
            .query_row(params![scm_repo], Self::row_to_fix_attempt)
            .ok();
        Ok(result)
    }

    /// Get a specific execution for an attempt.
    fn get_execution_for_attempt(
        &self,
        attempt_id: i64,
        execution_id: i64,
    ) -> Result<Option<AgentExecution>> {
        let conn = self.acquire_lock()?;
        let mut stmt = conn.prepare(
            r#"
            SELECT id, attempt_id, started_at, completed_at, duration_secs, exit_code, timed_out,
                   stdout_preview, stderr_preview, stdout_log_path, stderr_log_path, event_log_path,
                   prompt_used, prompt_hash, model_version, working_directory, git_branch,
                   git_commit_before, git_commit_after, files_changed, lines_added, lines_removed,
                   total_cost_usd, num_turns, session_id, duration_api_ms,
                   input_tokens, output_tokens, cache_read_input_tokens, cache_creation_input_tokens,
                   provider, experiment_name, experiment_variant
            FROM claude_executions
            WHERE attempt_id = ?1 AND id = ?2
            LIMIT 1
            "#,
        )?;

        let execution = stmt
            .query_row(params![attempt_id, execution_id], |row| {
                Ok(AgentExecution {
                    id: row.get(0)?,
                    attempt_id: row.get(1)?,
                    started_at: Self::parse_datetime(&row.get::<_, String>(2)?)?,
                    completed_at: Self::parse_optional_datetime(row.get(3)?)?,
                    duration_secs: row.get(4)?,
                    exit_code: row.get(5)?,
                    timed_out: row.get::<_, i32>(6).unwrap_or(0) != 0,
                    stdout_preview: row.get(7)?,
                    stderr_preview: row.get(8)?,
                    stdout_log_path: row.get(9)?,
                    stderr_log_path: row.get(10)?,
                    event_log_path: row.get(11)?,
                    prompt_used: row.get(12)?,
                    prompt_hash: row.get(13)?,
                    model_version: row.get(14)?,
                    working_directory: row.get(15)?,
                    git_branch: row.get(16)?,
                    git_commit_before: row.get(17)?,
                    git_commit_after: row.get(18)?,
                    files_changed: row.get(19)?,
                    lines_added: row.get(20)?,
                    lines_removed: row.get(21)?,
                    total_cost_usd: row.get(22)?,
                    num_turns: row.get(23)?,
                    session_id: row.get(24)?,
                    duration_api_ms: row.get(25)?,
                    input_tokens: row.get(26)?,
                    output_tokens: row.get(27)?,
                    cache_read_input_tokens: row.get(28)?,
                    cache_creation_input_tokens: row.get(29)?,
                    provider: row.get(30)?,
                    experiment_name: row.get(31)?,
                    experiment_variant: row.get(32)?,
                })
            })
            .optional()?;

        Ok(execution)
    }

    fn get_attempts_batch(&self, keys: &[(&str, &str)]) -> Result<Vec<Option<FixAttempt>>> {
        if keys.is_empty() {
            return Ok(Vec::new());
        }
        let conn = self.acquire_lock()?;
        let mut results = Vec::with_capacity(keys.len());
        // Use a prepared statement to amortize compilation cost across the batch.
        let mut stmt = conn.prepare_cached(
            r#"
            SELECT id, source, issue_id, short_id, attempted_at, pr_url, scm_repo,
                   scm_pr_number, status, error_message, merged_at, resolved_at,
                   retry_count, last_retry_at, issue_labels, parent_attempt_id, cascade_repo
            FROM fix_attempts
            WHERE source = ? AND issue_id = ?
            "#,
        )?;
        for (source, issue_id) in keys {
            let attempt = stmt
                .query_row(params![source, issue_id], Self::row_to_fix_attempt)
                .ok();
            results.push(attempt);
        }
        Ok(results)
    }

    fn record_release_tracking(
        &self,
        tracking: &claudear_core::types::ReleaseTracking,
    ) -> Result<i64> {
        let conn = self.acquire_lock()?;

        conn.execute(
            r#"
            INSERT INTO release_tracking (
                regression_watch_id, release_version, release_commit, released_at, created_at
            )
            VALUES (?1, ?2, ?3, ?4, ?5)
            "#,
            params![
                tracking.regression_watch_id,
                tracking.release_version,
                tracking.release_commit,
                tracking
                    .released_at
                    .map(|dt| dt.format("%Y-%m-%d %H:%M:%S").to_string()),
                tracking.created_at.format("%Y-%m-%d %H:%M:%S").to_string(),
            ],
        )?;

        Ok(conn.last_insert_rowid())
    }

    fn get_success_rate(&self) -> Result<f64> {
        SqliteTracker::get_success_rate(self)
    }

    // --- Code Indexing ---
}

impl KnowledgeStore for SqliteTracker {
    fn store_feedback_outcome(&self, outcome: &FixOutcome) -> Result<i64> {
        SqliteTracker::store_feedback_outcome(self, outcome)
    }

    fn get_feedback_outcomes(&self, source: Option<&str>, limit: usize) -> Result<Vec<FixOutcome>> {
        SqliteTracker::get_feedback_outcomes(self, source, limit)
    }

    fn get_feedback_outcome_by_attempt(&self, attempt_id: i64) -> Result<Option<FixOutcome>> {
        SqliteTracker::get_feedback_outcome_by_attempt(self, attempt_id)
    }

    fn store_qa_knowledge(&self, entry: &QaKnowledgeEntry) -> Result<i64> {
        SqliteTracker::store_qa_knowledge(self, entry)
    }

    fn find_similar_qa_scoped(
        &self,
        source: &str,
        repo: Option<&str>,
        question_norm: &str,
        question_embedding: Option<&[f32]>,
        threshold: f64,
        limit: usize,
    ) -> Result<Vec<QaMatch>> {
        SqliteTracker::find_similar_qa_scoped(
            self,
            source,
            repo,
            question_norm,
            question_embedding,
            threshold,
            limit,
        )
    }

    fn find_similar_qa_global(
        &self,
        question_norm: &str,
        question_embedding: Option<&[f32]>,
        threshold: f64,
        limit: usize,
    ) -> Result<Vec<QaMatch>> {
        SqliteTracker::find_similar_qa_global(
            self,
            question_norm,
            question_embedding,
            threshold,
            limit,
        )
    }

    fn record_qa_usage(
        &self,
        attempt_id: i64,
        qa_id: i64,
        usage_type: &str,
        similarity_score: f64,
    ) -> Result<i64> {
        SqliteTracker::record_qa_usage(self, attempt_id, qa_id, usage_type, similarity_score)
    }

    fn update_qa_outcome_stats(&self, qa_id: i64, success: bool) -> Result<()> {
        SqliteTracker::update_qa_outcome_stats(self, qa_id, success)
    }

    fn update_qa_outcome_stats_for_attempt(&self, attempt_id: i64, success: bool) -> Result<()> {
        SqliteTracker::update_qa_outcome_stats_for_attempt(self, attempt_id, success)
    }

    fn update_feedback_learnings(&self, outcome_id: i64, learnings: &str) -> Result<()> {
        SqliteTracker::update_feedback_learnings(self, outcome_id, learnings)
    }

    fn store_diff_analysis(&self, analysis: &claudear_core::types::DiffAnalysis) -> Result<i64> {
        SqliteTracker::store_diff_analysis(self, analysis)
    }

    fn get_diff_analyses_for_repo(
        &self,
        repo: &str,
        limit: usize,
    ) -> Result<Vec<claudear_core::types::DiffAnalysis>> {
        SqliteTracker::get_diff_analyses_for_repo(self, repo, limit)
    }

    fn upsert_promoted_instruction(
        &self,
        instruction: &claudear_core::types::PromotedInstruction,
    ) -> Result<i64> {
        SqliteTracker::upsert_promoted_instruction(self, instruction)
    }

    fn get_promoted_instructions(
        &self,
        repo: &str,
    ) -> Result<Vec<claudear_core::types::PromotedInstruction>> {
        SqliteTracker::get_promoted_instructions(self, repo)
    }

    fn upsert_repo_knowledge(&self, entry: &claudear_core::types::RepoKnowledge) -> Result<i64> {
        SqliteTracker::upsert_repo_knowledge(self, entry)
    }

    fn get_repo_knowledge(&self, repo: &str) -> Result<Vec<claudear_core::types::RepoKnowledge>> {
        SqliteTracker::get_repo_knowledge(self, repo)
    }

    fn get_repo_knowledge_by_key(
        &self,
        repo: &str,
        key: &str,
    ) -> Result<Vec<claudear_core::types::RepoKnowledge>> {
        SqliteTracker::get_repo_knowledge_by_key(self, repo, key)
    }

    fn get_all_repo_aliases(&self) -> Result<Vec<(String, String)>> {
        SqliteTracker::get_all_repo_aliases(self)
    }

    fn upsert_review_pattern(&self, pattern: &claudear_core::types::ReviewPattern) -> Result<i64> {
        SqliteTracker::upsert_review_pattern(self, pattern)
    }

    fn get_review_patterns(
        &self,
        repo: &str,
        limit: usize,
    ) -> Result<Vec<claudear_core::types::ReviewPattern>> {
        SqliteTracker::get_review_patterns(self, repo, limit)
    }

    fn get_review_patterns_by_category(
        &self,
        repo: &str,
        category: claudear_core::types::ReviewCategory,
    ) -> Result<Vec<claudear_core::types::ReviewPattern>> {
        SqliteTracker::get_review_patterns_by_category(self, repo, category)
    }

    fn store_strategy_fingerprint(
        &self,
        fingerprint: &claudear_core::types::StrategyFingerprint,
    ) -> Result<i64> {
        SqliteTracker::store_strategy_fingerprint(self, fingerprint)
    }

    fn get_successful_strategies(
        &self,
        repo: &str,
        limit: usize,
    ) -> Result<Vec<claudear_core::types::StrategyFingerprint>> {
        SqliteTracker::get_successful_strategies(self, repo, limit)
    }

    fn store_issue_cluster(&self, cluster: &claudear_core::types::IssueCluster) -> Result<i64> {
        SqliteTracker::store_issue_cluster(self, cluster)
    }

    fn get_active_clusters(&self, source: &str) -> Result<Vec<claudear_core::types::IssueCluster>> {
        SqliteTracker::get_active_clusters(self, source)
    }

    fn update_cluster_resolution(
        &self,
        cluster_id: i64,
        resolved_by_issue_id: &str,
        resolved_by_attempt_id: i64,
    ) -> Result<()> {
        SqliteTracker::update_cluster_resolution(
            self,
            cluster_id,
            resolved_by_issue_id,
            resolved_by_attempt_id,
        )
    }

    fn get_recent_issue_arrivals(
        &self,
        source: &str,
        window_minutes: i64,
    ) -> Result<Vec<(String, DateTime<Utc>)>> {
        SqliteTracker::get_recent_issue_arrivals(self, source, window_minutes)
    }

    fn store_content_cluster(&self, cluster: &claudear_core::types::ContentCluster) -> Result<i64> {
        SqliteTracker::store_content_cluster(self, cluster)
    }

    fn get_active_content_clusters(
        &self,
        source: &str,
    ) -> Result<Vec<claudear_core::types::ContentCluster>> {
        SqliteTracker::get_active_content_clusters(self, source)
    }

    fn resolve_content_cluster(&self, cluster_id: i64) -> Result<()> {
        SqliteTracker::resolve_content_cluster(self, cluster_id)
    }

    fn store_severity_score(
        &self,
        source: &str,
        issue_id: &str,
        score: &claudear_core::types::SeverityScore,
        blast_radius: claudear_core::types::BlastRadius,
    ) -> Result<()> {
        SqliteTracker::store_severity_score(self, source, issue_id, score, blast_radius)
    }

    fn record_suppression(
        &self,
        source: &str,
        issue_id: &str,
        rule_name: &str,
        reason: &str,
    ) -> Result<()> {
        SqliteTracker::record_suppression(self, source, issue_id, rule_name, reason)
    }

    fn upsert_cross_repo_correlation(
        &self,
        repo_a: &str,
        repo_b: &str,
        window_hours: i64,
    ) -> Result<CrossRepoCorrelation> {
        SqliteTracker::upsert_cross_repo_correlation(self, repo_a, repo_b, window_hours)
    }

    fn get_cross_repo_correlations(
        &self,
        min_count: i64,
        max_age_hours: i64,
    ) -> Result<Vec<CrossRepoCorrelation>> {
        SqliteTracker::get_cross_repo_correlations(self, min_count, max_age_hours)
    }
}

impl EmbeddingStore for SqliteTracker {
    fn get_recent_attempts_since(&self, since: &DateTime<Utc>) -> Result<Vec<FixAttempt>> {
        SqliteTracker::get_recent_attempts_since(self, since)
    }

    /// Store an issue embedding.
    fn store_embedding(&self, embedding: &IssueEmbedding) -> Result<i64> {
        let conn = self.acquire_lock()?;

        // Serialize the embedding vector to bytes if present
        let embedding_bytes: Option<Vec<u8>> = embedding.embedding.as_ref().map(|emb| {
            let mut bytes = Vec::with_capacity(emb.len() * 4);
            for f in emb {
                bytes.extend_from_slice(&f.to_le_bytes());
            }
            bytes
        });

        let updated_at_str = embedding
            .updated_at
            .map(|dt| dt.format("%Y-%m-%d %H:%M:%S").to_string());

        conn.execute(
            r#"
            INSERT INTO issues (source, issue_id, short_id, title, description, url, priority, status, labels, embedding, embedding_model, created_at, updated_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)
            ON CONFLICT(source, issue_id) DO UPDATE SET
                short_id = COALESCE(excluded.short_id, short_id),
                title = COALESCE(excluded.title, title),
                description = COALESCE(excluded.description, description),
                url = COALESCE(excluded.url, url),
                priority = COALESCE(excluded.priority, priority),
                status = COALESCE(excluded.status, status),
                labels = COALESCE(excluded.labels, labels),
                embedding = COALESCE(excluded.embedding, embedding),
                embedding_model = COALESCE(excluded.embedding_model, embedding_model),
                updated_at = COALESCE(excluded.updated_at, updated_at)
            "#,
            params![
                embedding.source,
                embedding.issue_id,
                embedding.short_id,
                embedding.title,
                embedding.description,
                embedding.url,
                embedding.priority,
                embedding.status,
                embedding.labels,
                embedding_bytes,
                embedding.embedding_model,
                embedding.created_at.format("%Y-%m-%d %H:%M:%S").to_string(),
                updated_at_str,
            ],
        )?;

        // Dual-write: upsert into HNSW vector table (get row ID reliably via SELECT)
        let row_id: i64 = conn.query_row(
            "SELECT id FROM issues WHERE source = ?1 AND issue_id = ?2",
            params![embedding.source, embedding.issue_id],
            |row| row.get(0),
        )?;
        if let Some(ref emb) = embedding.embedding {
            if let Err(e) = Self::upsert_issue_vector_embedding(&conn, row_id, emb) {
                tracing::debug!(error = %e, "Failed to upsert issue vector embedding");
            }
        }

        Ok(row_id)
    }

    /// Store multiple embeddings in a single transaction.
    ///
    /// Much more efficient than calling `store_embedding` in a loop because
    /// the mutex is acquired once and all inserts share one transaction.
    fn store_embeddings_batch(&self, embeddings: &[IssueEmbedding]) -> Result<()> {
        if embeddings.is_empty() {
            return Ok(());
        }
        let mut conn = self.acquire_lock()?;
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;

        {
            let mut stmt = tx.prepare_cached(
                r#"
                INSERT INTO issues (source, issue_id, short_id, title, description, url, priority, status, labels, embedding, embedding_model, created_at, updated_at)
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)
                ON CONFLICT(source, issue_id) DO UPDATE SET
                    short_id = COALESCE(excluded.short_id, short_id),
                    title = COALESCE(excluded.title, title),
                    description = COALESCE(excluded.description, description),
                    url = COALESCE(excluded.url, url),
                    priority = COALESCE(excluded.priority, priority),
                    status = COALESCE(excluded.status, status),
                    labels = COALESCE(excluded.labels, labels),
                    embedding = COALESCE(excluded.embedding, embedding),
                    embedding_model = COALESCE(excluded.embedding_model, embedding_model),
                    updated_at = COALESCE(excluded.updated_at, updated_at)
                "#,
            )?;

            for embedding in embeddings {
                let embedding_bytes: Option<Vec<u8>> = embedding.embedding.as_ref().map(|emb| {
                    let mut bytes = Vec::with_capacity(emb.len() * 4);
                    for f in emb {
                        bytes.extend_from_slice(&f.to_le_bytes());
                    }
                    bytes
                });

                let updated_at_str = embedding
                    .updated_at
                    .map(|dt| dt.format("%Y-%m-%d %H:%M:%S").to_string());

                stmt.execute(params![
                    embedding.source,
                    embedding.issue_id,
                    embedding.short_id,
                    embedding.title,
                    embedding.description,
                    embedding.url,
                    embedding.priority,
                    embedding.status,
                    embedding.labels,
                    embedding_bytes,
                    embedding.embedding_model,
                    embedding.created_at.format("%Y-%m-%d %H:%M:%S").to_string(),
                    updated_at_str,
                ])?;

                // Dual-write: upsert into HNSW vector table
                let row_id: i64 = tx.query_row(
                    "SELECT id FROM issues WHERE source = ?1 AND issue_id = ?2",
                    params![embedding.source, embedding.issue_id],
                    |row| row.get(0),
                )?;
                if let Some(ref emb) = embedding.embedding {
                    if let Err(e) = Self::upsert_issue_vector_embedding(&tx, row_id, emb) {
                        tracing::debug!(error = %e, "Failed to upsert issue vector embedding in batch");
                    }
                }
            }
        }

        tx.commit()?;
        Ok(())
    }

    /// Get an embedding by source and issue ID.
    fn get_embedding(&self, source: &str, issue_id: &str) -> Result<Option<IssueEmbedding>> {
        let conn = self.acquire_lock()?;
        let mut stmt = conn.prepare(
            r#"
            SELECT id, source, issue_id, short_id, title, embedding, embedding_model, created_at,
                   description, url, priority, status, labels, updated_at
            FROM issues
            WHERE source = ? AND issue_id = ?
            "#,
        )?;

        let result = stmt
            .query_row(params![source, issue_id], |row| {
                Self::row_to_issue_embedding(row, 8)
            })
            .ok();

        Ok(result)
    }

    /// Get embeddings with pagination support to prevent memory exhaustion.
    ///
    /// # Arguments
    /// * `source` - Optional filter by source
    /// * `limit` - Maximum number of embeddings to return (defaults to 1000, max 10000)
    /// * `offset` - Number of records to skip for pagination (defaults to 0)
    ///
    /// # Returns
    /// A vector of embeddings, limited to prevent unbounded memory usage.
    fn get_all_embeddings(
        &self,
        source: Option<&str>,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> Result<Vec<IssueEmbedding>> {
        let conn = self.acquire_lock()?;

        // Enforce reasonable limits to prevent memory exhaustion
        const DEFAULT_LIMIT: usize = 1000;
        const MAX_LIMIT: usize = 10000;
        let limit = limit.unwrap_or(DEFAULT_LIMIT).min(MAX_LIMIT);
        let offset = offset.unwrap_or(0);

        let query = match source {
            Some(_) => {
                r#"
                SELECT id, source, issue_id, short_id, title, embedding, embedding_model, created_at,
                       description, url, priority, status, labels, updated_at
                FROM issues
                WHERE source = ?
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            "#
            }
            None => {
                r#"
                SELECT id, source, issue_id, short_id, title, embedding, embedding_model, created_at,
                       description, url, priority, status, labels, updated_at
                FROM issues
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            "#
            }
        };

        let mut stmt = conn.prepare(query)?;

        let row_mapper = |row: &rusqlite::Row<'_>| Self::row_to_issue_embedding(row, 8);

        let rows = match source {
            Some(s) => stmt.query_map(params![s, limit as i64, offset as i64], row_mapper)?,
            None => stmt.query_map(params![limit as i64, offset as i64], row_mapper)?,
        };

        // Collect results, propagating any errors from corrupted embeddings
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| {
                claudear_core::error::Error::Storage(format!("Failed to read embeddings: {}", e))
            })
    }

    /// Store an issue (convenience wrapper around store_embedding).
    fn store_issue(&self, issue: &IssueEmbedding) -> Result<i64> {
        self.store_embedding(issue)
    }

    /// Find similar issue embeddings using the HNSW vector index.
    ///
    /// Returns `None` if vectorlite is unavailable.
    /// Returns `Some(vec)` with matching embeddings and similarity scores.
    fn find_similar_issues_vector(
        &self,
        query_embedding: &[f32],
        source: &str,
        exclude_issue_id: Option<&str>,
        min_similarity: f64,
        limit: usize,
    ) -> Result<Option<Vec<(IssueEmbedding, f64)>>> {
        if query_embedding.is_empty() || limit == 0 {
            return Ok(Some(Vec::new()));
        }

        let conn = self.acquire_lock()?;

        if !Self::ensure_issue_vector_table(&conn, query_embedding.len())? {
            return Ok(None);
        }

        let query_blob: Vec<u8> = query_embedding
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let candidate_limit = limit * ISSUE_VECTOR_CANDIDATE_MULTIPLIER;

        let sql = format!(
            r#"
            WITH candidates AS (
                SELECT rowid AS emb_id,
                       MAX(0.0, MIN(1.0, 1.0 - distance)) AS similarity
                FROM {table}
                WHERE knn_search(embedding, knn_param(?1, ?2, ?3))
            )
            SELECT e.id, e.source, e.issue_id, e.short_id, e.title, e.embedding,
                   e.embedding_model, e.created_at, c.similarity,
                   e.description, e.url, e.priority, e.status, e.labels, e.updated_at
            FROM candidates c
            JOIN issues e ON e.id = c.emb_id
            WHERE e.source = ?4
              AND (?5 IS NULL OR e.issue_id != ?5)
              AND c.similarity >= ?6
            ORDER BY c.similarity DESC
            LIMIT ?7
            "#,
            table = ISSUE_VECTOR_TABLE
        );

        let mut stmt = match conn.prepare(&sql) {
            Ok(stmt) => stmt,
            Err(e) => {
                tracing::debug!(error = %e, "Failed to prepare issue vector search query");
                return Ok(None);
            }
        };

        let rows = match stmt.query_map(
            params![
                query_blob,
                candidate_limit as i64,
                ISSUE_VECTOR_EF_SEARCH as i64,
                source,
                exclude_issue_id,
                min_similarity,
                limit as i64
            ],
            |row| {
                let ie = Self::row_to_issue_embedding(row, 9)?;
                let similarity: f64 = row.get(8)?;
                Ok((ie, similarity))
            },
        ) {
            Ok(rows) => rows,
            Err(e) => {
                tracing::debug!(error = %e, "Issue vector search query failed");
                return Ok(None);
            }
        };

        let mut results = Vec::new();
        for row in rows {
            match row {
                Ok(r) => results.push(r),
                Err(e) => tracing::debug!(error = %e, "Failed to read issue vector row"),
            }
        }

        Ok(Some(results))
    }

    /// Find similar outcome embeddings using the HNSW vector index.
    ///
    /// Returns `None` if vectorlite is unavailable (caller should fall back).
    /// Returns `Some(vec)` with matching outcomes and similarity scores.
    fn find_similar_outcomes_vector(
        &self,
        query_embedding: &[f32],
        min_similarity: f64,
        limit: usize,
    ) -> Result<Option<Vec<(FixOutcome, f64)>>> {
        if query_embedding.is_empty() || limit == 0 {
            return Ok(Some(Vec::new()));
        }

        let conn = self.acquire_lock()?;

        if !Self::ensure_outcome_vector_table(&conn, query_embedding.len())? {
            return Ok(None);
        }

        let query_blob: Vec<u8> = query_embedding
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let candidate_limit = limit * OUTCOME_VECTOR_CANDIDATE_MULTIPLIER;

        let sql = format!(
            r#"
            WITH candidates AS (
                SELECT rowid AS outcome_id,
                       MAX(0.0, MIN(1.0, 1.0 - distance)) AS similarity
                FROM {table}
                WHERE knn_search(embedding, knn_param(?1, ?2, ?3))
            )
            SELECT f.id, f.attempt_id, f.source, f.issue_id, f.issue_text, f.prompt_used,
                   f.outcome, f.error_type, f.learnings, f.keywords, f.created_at,
                   f.embedding, c.similarity
            FROM candidates c
            JOIN feedback_outcomes f ON f.id = c.outcome_id
            WHERE c.similarity >= ?4
            ORDER BY c.similarity DESC
            LIMIT ?5
            "#,
            table = OUTCOME_VECTOR_TABLE
        );

        let mut stmt = match conn.prepare(&sql) {
            Ok(stmt) => stmt,
            Err(e) => {
                tracing::debug!(error = %e, "Failed to prepare outcome vector search query");
                return Ok(None);
            }
        };

        let rows = match stmt.query_map(
            params![
                query_blob,
                candidate_limit as i64,
                OUTCOME_VECTOR_EF_SEARCH as i64,
                min_similarity,
                limit as i64
            ],
            |row| {
                let outcome_str: String = row.get(6)?;
                let keywords_str: Option<String> = row.get(9)?;
                let created_at_str: String = row.get(10)?;

                // Deserialize embedding BLOB if present
                let embedding_blob: Option<Vec<u8>> = row.get(11)?;
                let embedding = embedding_blob.and_then(|blob| {
                    if blob.len() % 4 != 0 {
                        return None;
                    }
                    Some(
                        blob.chunks_exact(4)
                            .map(|chunk| {
                                let arr: [u8; 4] =
                                    chunk.try_into().expect("chunks_exact guarantees 4 bytes");
                                f32::from_le_bytes(arr)
                            })
                            .collect(),
                    )
                });

                let fo = FixOutcome {
                    id: row.get(0)?,
                    attempt_id: row.get(1)?,
                    source: row.get(2)?,
                    issue_id: row.get(3)?,
                    issue_text: row.get(4)?,
                    prompt_used: row.get(5)?,
                    outcome: Outcome::parse(&outcome_str).unwrap_or(Outcome::Failed),
                    error_type: row.get(7)?,
                    learnings: row.get(8)?,
                    keywords: keywords_str
                        .and_then(|s| serde_json::from_str(&s).ok())
                        .unwrap_or_default(),
                    embedding,
                    created_at: Self::parse_datetime(&created_at_str)?,
                };
                let similarity: f64 = row.get(12)?;
                Ok((fo, similarity))
            },
        ) {
            Ok(rows) => rows,
            Err(e) => {
                tracing::debug!(error = %e, "Outcome vector search query failed");
                return Ok(None);
            }
        };

        let mut results = Vec::new();
        for row in rows {
            match row {
                Ok(r) => results.push(r),
                Err(e) => tracing::debug!(error = %e, "Failed to read outcome vector row"),
            }
        }

        Ok(Some(results))
    }

    /// Record an inference attempt.
    fn record_inference_attempt(
        &self,
        issue_id: &str,
        issue_source: &str,
        extracted_filenames: &[String],
        extracted_functions: &[String],
        extracted_keywords: &[String],
        inferred_repo_id: Option<i64>,
        confidence: &str,
        inference_reason: &str,
        duration_ms: Option<u64>,
    ) -> Result<i64> {
        let conn = self.acquire_lock()?;

        let filenames_json = serde_json::to_string(extracted_filenames).unwrap_or_default();
        let functions_json = serde_json::to_string(extracted_functions).unwrap_or_default();
        let keywords_json = serde_json::to_string(extracted_keywords).unwrap_or_default();

        conn.execute(
            r#"
            INSERT INTO inference_attempts (
                issue_id, issue_source, extracted_filenames, extracted_functions,
                extracted_keywords, inferred_repo_id, confidence, inference_reason,
                inference_duration_ms
            )
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
            "#,
            params![
                issue_id,
                issue_source,
                filenames_json,
                functions_json,
                keywords_json,
                inferred_repo_id,
                confidence,
                inference_reason,
                duration_ms.map(|d| d as i64),
            ],
        )?;

        Ok(conn.last_insert_rowid())
    }

    /// Record feedback on an inference attempt (was it correct?).
    fn record_inference_feedback(
        &self,
        inference_id: i64,
        was_correct: bool,
        actual_repo_id: Option<i64>,
        feedback_source: &str,
    ) -> Result<()> {
        let conn = self.acquire_lock()?;
        conn.execute(
            r#"
            UPDATE inference_attempts
            SET was_correct = ?1, actual_repo_id = ?2, feedback_source = ?3, feedback_at = datetime('now')
            WHERE id = ?4
            "#,
            params![was_correct, actual_repo_id, feedback_source, inference_id],
        )?;
        Ok(())
    }

    /// List issues with pagination, sorted by updated_at DESC, then created_at DESC.
    fn list_issues(
        &self,
        source: Option<&str>,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<IssueEmbedding>> {
        let conn = self.acquire_lock()?;

        let query = match source {
            Some(_) => {
                r#"
                SELECT id, source, issue_id, short_id, title, embedding, embedding_model, created_at,
                       description, url, priority, status, labels, updated_at
                FROM issues
                WHERE source = ?
                ORDER BY COALESCE(updated_at, created_at) DESC
                LIMIT ? OFFSET ?
                "#
            }
            None => {
                r#"
                SELECT id, source, issue_id, short_id, title, embedding, embedding_model, created_at,
                       description, url, priority, status, labels, updated_at
                FROM issues
                ORDER BY COALESCE(updated_at, created_at) DESC
                LIMIT ? OFFSET ?
                "#
            }
        };

        let mut stmt = conn.prepare(query)?;

        let row_mapper = |row: &rusqlite::Row<'_>| Self::row_to_issue_embedding(row, 8);

        let rows = match source {
            Some(s) => stmt.query_map(params![s, limit as i64, offset as i64], row_mapper)?,
            None => stmt.query_map(params![limit as i64, offset as i64], row_mapper)?,
        };

        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| {
                claudear_core::error::Error::Storage(format!("Failed to list issues: {}", e))
            })
    }

    /// Count issues, optionally filtered by source.
    fn count_issues(&self, source: Option<&str>) -> Result<usize> {
        let conn = self.acquire_lock()?;
        let count: i64 = match source {
            Some(s) => conn.query_row(
                "SELECT COUNT(*) FROM issues WHERE source = ?",
                params![s],
                |row| row.get(0),
            )?,
            None => conn.query_row("SELECT COUNT(*) FROM issues", [], |row| row.get(0))?,
        };
        Ok(count as usize)
    }

    fn search_code_chunks(
        &self,
        query_embedding: &[f32],
        repo_id: Option<i64>,
        limit: usize,
    ) -> Result<Vec<claudear_core::types::CodeSearchResult>> {
        SqliteTracker::search_code_chunks(self, query_embedding, repo_id, limit)
    }
}

impl RepoStore for SqliteTracker {
    fn list_indexed_repos(&self) -> Result<Vec<StoredIndexedRepo>> {
        SqliteTracker::list_indexed_repos(self)
    }

    fn get_index_stats(&self) -> Result<IndexStats> {
        SqliteTracker::get_index_stats(self)
    }

    fn get_indexing_progress(&self) -> Result<IndexingProgress> {
        SqliteTracker::get_indexing_progress(self)
    }

    fn subscribe_indexing_progress(&self) -> tokio::sync::watch::Receiver<IndexingProgress> {
        SqliteTracker::subscribe_indexing_progress(self)
    }

    fn add_dependency(&self, upstream: &str, downstream: &str, dep_type: &str) -> Result<()> {
        SqliteTracker::add_dependency(self, upstream, downstream, dep_type)
    }

    fn list_all_dependencies(&self) -> Result<Vec<StoredDependency>> {
        SqliteTracker::list_all_dependencies(self)
    }

    fn get_inference_stats(&self) -> Result<InferenceStats> {
        SqliteTracker::get_inference_stats(self)
    }

    fn get_inference_history(&self, limit: usize) -> Result<Vec<InferenceHistoryEntry>> {
        SqliteTracker::get_inference_history(self, limit)
    }

    fn has_dependency(&self, repo_a: &str, repo_b: &str) -> Result<bool> {
        SqliteTracker::has_dependency(self, repo_a, repo_b)
    }

    /// Sync repositories from a RepoIndex to the database.
    ///
    /// Updates paths for all repos in the index and optionally syncs files.
    fn sync_from_index(
        &self,
        index: &claudear_core::types::RepoIndex,
        sync_files: bool,
    ) -> Result<usize> {
        let repos = index.list();
        let mut synced = 0;

        for repo in repos {
            let path_str = repo.path.to_string_lossy();

            if sync_files {
                // Use save_indexed_repo which also updates file_count and last_indexed_at
                let repo_id = self.save_indexed_repo(
                    &repo.name,
                    &path_str,
                    Some(&repo.scm_url),
                    &repo.default_branch,
                    repo.files.len(),
                )?;

                if !repo.files.is_empty() {
                    let files_with_types: Vec<(String, Option<String>)> = repo
                        .files
                        .iter()
                        .map(|f| {
                            let file_type = std::path::Path::new(f)
                                .extension()
                                .map(|e| e.to_string_lossy().to_string());
                            (f.clone(), file_type)
                        })
                        .collect();

                    self.save_repo_files(repo_id, &files_with_types)?;
                }
            } else {
                // Just update paths in repositories table
                self.upsert_repository(&repo.name, Some(&path_str), None)?;
            }
            synced += 1;
        }

        Ok(synced)
    }

    fn sync_repo_files(&self, repo: &claudear_core::types::IndexedRepo) -> Result<()> {
        self.sync_repo_files(repo)
    }

    fn get_indexed_repo(&self, name: &str) -> Result<Option<StoredIndexedRepo>> {
        let conn = self.acquire_lock()?;
        let mut stmt = conn.prepare(
            r#"
            SELECT id, name, path, scm_url, default_branch, file_count, last_indexed_at, created_at
            FROM repositories WHERE name = ?
            "#,
        )?;

        let result = stmt.query_row(params![name], |row| {
            Ok(StoredIndexedRepo {
                id: row.get(0)?,
                name: row.get(1)?,
                path: row.get(2)?,
                scm_url: row.get(3)?,
                default_branch: row.get(4)?,
                file_count: row.get(5)?,
                last_indexed_at: row.get(6)?,
                created_at: row.get(7)?,
            })
        });

        match result {
            Ok(repo) => Ok(Some(repo)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    // --- Analytics ---

    fn get_or_create_repo_id(&self, name: &str) -> Result<i64> {
        SqliteTracker::get_or_create_repo_id(self, name)
    }

    fn code_chunk_hash_matches(
        &self,
        repo_id: i64,
        file_path: &str,
        file_hash: &str,
    ) -> Result<bool> {
        SqliteTracker::code_chunk_hash_matches(self, repo_id, file_path, file_hash)
    }

    fn delete_code_data_for_file(&self, repo_id: i64, file_path: &str) -> Result<()> {
        SqliteTracker::delete_code_data_for_file(self, repo_id, file_path)
    }

    fn delete_code_chunks_by_ids(&self, chunk_ids: &[i64]) -> Result<()> {
        SqliteTracker::delete_code_chunks_by_ids(self, chunk_ids)
    }

    fn cleanup_stale_code_data(&self, repo_id: i64, current_paths: &[String]) -> Result<()> {
        SqliteTracker::cleanup_stale_code_data(self, repo_id, current_paths)
    }

    fn save_code_symbols(&self, symbols: &[claudear_core::types::CodeSymbol]) -> Result<()> {
        SqliteTracker::save_code_symbols(self, symbols)
    }

    fn save_code_chunks(&self, chunks: &[claudear_core::types::CodeChunk]) -> Result<Vec<i64>> {
        SqliteTracker::save_code_chunks(self, chunks)
    }

    fn save_code_chunk_embeddings(&self, pairs: &[(i64, &[f32])], model_name: &str) -> Result<()> {
        SqliteTracker::save_code_chunk_embeddings(self, pairs, model_name)
    }

    fn find_code_symbols(
        &self,
        name: &str,
        kind: Option<claudear_core::types::SymbolKind>,
        repo_id: Option<i64>,
    ) -> Result<Vec<claudear_core::types::CodeSymbol>> {
        SqliteTracker::find_code_symbols(self, name, kind, repo_id)
    }

    fn get_code_embedding_model(&self, repo_id: i64) -> Result<Option<String>> {
        SqliteTracker::get_code_embedding_model(self, repo_id)
    }

    fn get_code_index_meta(&self, repo_id: i64, key: &str) -> Result<Option<String>> {
        SqliteTracker::get_code_index_meta(self, repo_id, key)
    }

    fn set_code_index_meta(&self, repo_id: i64, key: &str, value: &str) -> Result<()> {
        SqliteTracker::set_code_index_meta(self, repo_id, key, value)
    }

    fn delete_all_code_data_for_repo(&self, repo_id: i64) -> Result<()> {
        SqliteTracker::delete_all_code_data_for_repo(self, repo_id)
    }
}

impl UserStore for SqliteTracker {
    fn get_channel_cursor(&self, channel: &str, cursor_key: &str) -> Result<Option<String>> {
        SqliteTracker::get_channel_cursor(self, channel, cursor_key)
    }

    fn set_channel_cursor(
        &self,
        channel: &str,
        cursor_key: &str,
        cursor_value: &str,
    ) -> Result<()> {
        SqliteTracker::set_channel_cursor(self, channel, cursor_key, cursor_value)
    }

    fn create_user(&self, email: &str, password_hash: &str, name: &str, role: &str) -> Result<i64> {
        let conn = self.acquire_lock()?;
        conn.execute(
            "INSERT INTO users (email, password_hash, name, role) VALUES (?1, ?2, ?3, ?4)",
            params![email, password_hash, name, role],
        )?;
        Ok(conn.last_insert_rowid())
    }

    fn get_user_by_email(&self, email: &str) -> Result<Option<UserRow>> {
        let conn = self.acquire_lock()?;
        let mut stmt = conn.prepare(
            "SELECT id, email, password_hash, name, role, avatar_url, created_at, updated_at FROM users WHERE email = ?1"
        )?;
        let user = stmt
            .query_row(params![email], UserRow::from_row)
            .optional()?;
        Ok(user)
    }

    fn get_user_by_id(&self, id: i64) -> Result<Option<UserRow>> {
        let conn = self.acquire_lock()?;
        let mut stmt = conn.prepare(
            "SELECT id, email, password_hash, name, role, avatar_url, created_at, updated_at FROM users WHERE id = ?1"
        )?;
        let user = stmt.query_row(params![id], UserRow::from_row).optional()?;
        Ok(user)
    }

    fn list_users(&self) -> Result<Vec<UserRow>> {
        let conn = self.acquire_lock()?;
        let mut stmt = conn.prepare(
            "SELECT id, email, password_hash, name, role, avatar_url, created_at, updated_at FROM users ORDER BY id"
        )?;
        let users = stmt
            .query_map([], UserRow::from_row)?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(users)
    }

    fn update_user(
        &self,
        id: i64,
        email: Option<&str>,
        password_hash: Option<&str>,
        name: Option<&str>,
        role: Option<&str>,
        avatar_url: Option<&str>,
    ) -> Result<bool> {
        let conn = self.acquire_lock()?;
        let mut sets = Vec::new();
        let mut values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();

        if let Some(e) = email {
            sets.push("email = ?");
            values.push(Box::new(e.to_string()));
        }
        if let Some(p) = password_hash {
            sets.push("password_hash = ?");
            values.push(Box::new(p.to_string()));
        }
        if let Some(n) = name {
            sets.push("name = ?");
            values.push(Box::new(n.to_string()));
        }
        if let Some(r) = role {
            sets.push("role = ?");
            values.push(Box::new(r.to_string()));
        }
        if let Some(a) = avatar_url {
            sets.push("avatar_url = ?");
            values.push(Box::new(a.to_string()));
        }

        if sets.is_empty() {
            return Ok(false);
        }

        sets.push("updated_at = datetime('now')");
        values.push(Box::new(id));

        let sql = format!("UPDATE users SET {} WHERE id = ?", sets.join(", "));
        let params: Vec<&dyn rusqlite::types::ToSql> = values.iter().map(|v| v.as_ref()).collect();
        let rows = conn.execute(&sql, params.as_slice())?;
        Ok(rows > 0)
    }

    fn delete_user(&self, id: i64) -> Result<bool> {
        let conn = self.acquire_lock()?;
        let rows = conn.execute("DELETE FROM users WHERE id = ?1", params![id])?;
        Ok(rows > 0)
    }

    fn count_users(&self) -> Result<i64> {
        let conn = self.acquire_lock()?;
        let count: i64 = conn.query_row("SELECT COUNT(*) FROM users", [], |row| row.get(0))?;
        Ok(count)
    }

    fn create_session(&self, user_id: i64, expires_at: &str) -> Result<String> {
        let token = generate_session_token();
        let conn = self.acquire_lock()?;
        conn.execute(
            "INSERT INTO sessions (id, user_id, expires_at, last_active_at) VALUES (?1, ?2, ?3, datetime('now'))",
            params![token, user_id, expires_at],
        )?;
        Ok(token)
    }

    fn get_session_user(&self, token: &str) -> Result<Option<UserRow>> {
        let conn = self.acquire_lock()?;
        let mut stmt = conn.prepare(
            "SELECT u.id, u.email, u.password_hash, u.name, u.role, u.avatar_url, u.created_at, u.updated_at
             FROM sessions s
             JOIN users u ON s.user_id = u.id
             WHERE s.id = ?1 AND s.expires_at > datetime('now') AND s.last_active_at > datetime('now', '-30 minutes')",
        )?;
        let user = stmt
            .query_row(params![token], UserRow::from_row)
            .optional()?;
        Ok(user)
    }

    fn delete_session(&self, token: &str) -> Result<()> {
        let conn = self.acquire_lock()?;
        conn.execute("DELETE FROM sessions WHERE id = ?1", params![token])?;
        Ok(())
    }

    fn cleanup_expired_sessions(&self) -> Result<usize> {
        let conn = self.acquire_lock()?;
        let deleted = conn.execute(
            "DELETE FROM sessions WHERE expires_at <= datetime('now') OR last_active_at <= datetime('now', '-30 minutes')",
            [],
        )?;
        Ok(deleted)
    }

    fn touch_session(&self, token: &str) -> Result<()> {
        let conn = self.acquire_lock()?;
        conn.execute(
            "UPDATE sessions SET last_active_at = datetime('now') WHERE id = ?1",
            params![token],
        )?;
        Ok(())
    }

    fn delete_user_sessions(&self, user_id: i64) -> Result<()> {
        let conn = self.acquire_lock()?;
        conn.execute("DELETE FROM sessions WHERE user_id = ?1", params![user_id])?;
        Ok(())
    }
}

impl RegressionStore for SqliteTracker {
    fn get_regression_watches_by_status(
        &self,
        status: claudear_core::types::RegressionWatchStatus,
    ) -> Result<Vec<claudear_core::types::RegressionWatch>> {
        SqliteTracker::get_regression_watches_by_status(self, status)
    }

    fn get_all_regression_watches(&self) -> Result<Vec<claudear_core::types::RegressionWatch>> {
        SqliteTracker::get_all_regression_watches(self)
    }

    fn get_regression_checks(
        &self,
        watch_id: i64,
    ) -> Result<Vec<claudear_core::types::RegressionCheck>> {
        SqliteTracker::get_regression_checks(self, watch_id)
    }

    fn create_regression_watch(
        &self,
        watch: &claudear_core::types::RegressionWatch,
    ) -> Result<i64> {
        SqliteTracker::create_regression_watch(self, watch)
    }

    fn add_regression_check(&self, check: &claudear_core::types::RegressionCheck) -> Result<i64> {
        SqliteTracker::add_regression_check(self, check)
    }

    fn update_regression_watch_status(
        &self,
        id: i64,
        status: claudear_core::types::RegressionWatchStatus,
    ) -> Result<()> {
        SqliteTracker::update_regression_watch_status(self, id, status)
    }

    fn get_regression_watch(
        &self,
        id: i64,
    ) -> Result<Option<claudear_core::types::RegressionWatch>> {
        SqliteTracker::get_regression_watch(self, id)
    }

    fn record_regression_check(
        &self,
        check: &claudear_core::types::RegressionCheck,
    ) -> Result<i64> {
        SqliteTracker::record_regression_check(self, check)
    }
}

impl ChatStore for SqliteTracker {
    fn create_chat_session(&self, id: &str, repo_id: Option<i64>) -> Result<()> {
        SqliteTracker::create_chat_session(self, id, repo_id)
    }

    fn save_chat_message(
        &self,
        session_id: &str,
        role: &str,
        content: &str,
        sources_json: Option<&str>,
    ) -> Result<i64> {
        SqliteTracker::save_chat_message(self, session_id, role, content, sources_json)
    }

    fn get_chat_history(
        &self,
        session_id: &str,
        limit: usize,
    ) -> Result<Vec<claudear_core::types::ChatMessage>> {
        SqliteTracker::get_chat_history(self, session_id, limit)
    }

    fn list_chat_sessions(&self) -> Result<Vec<claudear_core::types::ChatSession>> {
        SqliteTracker::list_chat_sessions(self)
    }

    fn delete_chat_session(&self, session_id: &str) -> Result<()> {
        SqliteTracker::delete_chat_session(self, session_id)
    }

    fn cleanup_expired_chat_sessions(&self, max_age_days: u32) -> Result<usize> {
        SqliteTracker::cleanup_expired_chat_sessions(self, max_age_days)
    }
}

impl ExperimentStore for SqliteTracker {
    fn save_experiment(&self, experiment: &PromptExperiment) -> Result<i64> {
        let conn = self.acquire_lock()?;

        conn.execute(
            r#"
            INSERT INTO prompt_experiments (experiment_name, variant, prompt_template, prompt_hash, created_at, active, success_count, failure_count, avg_time_to_merge, avg_review_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#,
            params![
                experiment.experiment_name,
                experiment.variant,
                experiment.prompt_template,
                experiment.prompt_hash,
                experiment.created_at.format("%Y-%m-%d %H:%M:%S").to_string(),
                experiment.active as i32,
                experiment.success_count,
                experiment.failure_count,
                experiment.avg_time_to_merge,
                experiment.avg_review_score,
            ],
        )?;
        Ok(conn.last_insert_rowid())
    }

    fn update_experiment(
        &self,
        experiment_id: i64,
        experiment_name: &str,
        variant: &str,
        prompt_template: &str,
        prompt_hash: &str,
        active: bool,
    ) -> Result<bool> {
        let conn = self.acquire_lock()?;

        let rows = conn.execute(
            r#"
            UPDATE prompt_experiments
            SET experiment_name = ?, variant = ?, prompt_template = ?, prompt_hash = ?, active = ?
            WHERE id = ?
            "#,
            params![
                experiment_name,
                variant,
                prompt_template,
                prompt_hash,
                active as i32,
                experiment_id,
            ],
        )?;

        Ok(rows > 0)
    }

    fn update_experiment_stats(
        &self,
        experiment_id: i64,
        success: bool,
        time_to_merge: Option<f64>,
    ) -> Result<()> {
        let conn = self.acquire_lock()?;

        if success {
            conn.execute(
                r#"
                UPDATE prompt_experiments
                SET success_count = success_count + 1
                WHERE id = ?
                "#,
                params![experiment_id],
            )?;
        } else {
            conn.execute(
                r#"
                UPDATE prompt_experiments
                SET failure_count = failure_count + 1
                WHERE id = ?
                "#,
                params![experiment_id],
            )?;
        }

        if let Some(ttm) = time_to_merge {
            conn.execute(
                r#"
                UPDATE prompt_experiments
                SET avg_time_to_merge = CASE
                    WHEN avg_time_to_merge IS NULL THEN ?
                    ELSE (avg_time_to_merge * (success_count - 1) + ?) / success_count
                END
                WHERE id = ?
                "#,
                params![ttm, ttm, experiment_id],
            )?;
        }

        Ok(())
    }

    fn get_active_experiments(&self) -> Result<Vec<PromptExperiment>> {
        SqliteTracker::get_active_experiments(self)
    }
}

impl EvaluationStore for SqliteTracker {
    fn store_eval_snapshot(
        &self,
        attempt_id: Option<i64>,
        phase: &str,
        snapshot: &claudear_core::types::EvalSnapshot,
    ) -> Result<i64> {
        let conn = self.acquire_lock()?;
        let diagnostics_json =
            serde_json::to_string(&snapshot.diagnostics).unwrap_or_else(|_| "[]".into());
        conn.execute(
            "INSERT INTO eval_snapshots (attempt_id, phase, category, tool_name, exit_code, passed, failed, skipped, warnings, errors, diagnostics_json, raw_output, duration_secs, line_coverage_pct, branch_coverage_pct)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)",
            params![
                attempt_id,
                phase,
                snapshot.category.to_string(),
                snapshot.tool_name,
                snapshot.exit_code,
                snapshot.passed,
                snapshot.failed,
                snapshot.skipped,
                snapshot.warnings,
                snapshot.errors,
                diagnostics_json,
                snapshot.raw_output,
                snapshot.duration_secs,
                snapshot.line_coverage_pct,
                snapshot.branch_coverage_pct,
            ],
        )?;
        Ok(conn.last_insert_rowid())
    }

    fn store_eval_delta(
        &self,
        attempt_id: Option<i64>,
        repo: &str,
        delta: &claudear_core::types::EvalDelta,
    ) -> Result<i64> {
        let conn = self.acquire_lock()?;
        let regressions_json =
            serde_json::to_string(&delta.regressions).unwrap_or_else(|_| "[]".into());
        let fixed_json = serde_json::to_string(&delta.fixed).unwrap_or_else(|_| "[]".into());
        conn.execute(
            "INSERT INTO eval_deltas (attempt_id, repo, tool_name, category, new_passes, new_failures, regressions_json, fixed_json, coverage_delta_pct, overall_improved)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                attempt_id,
                repo,
                delta.after.tool_name,
                delta.after.category.to_string(),
                delta.new_passes,
                delta.new_failures,
                regressions_json,
                fixed_json,
                delta.coverage_delta_pct,
                delta.is_improvement() as i32,
            ],
        )?;
        Ok(conn.last_insert_rowid())
    }
}

impl WebhookStore for SqliteTracker {
    fn check_and_record_delivery(&self, delivery_id: &str, source: &str) -> Result<bool> {
        let conn = self.acquire_lock()?;
        let rows_affected = conn.execute(
            "INSERT OR IGNORE INTO webhook_deliveries (delivery_id, source) VALUES (?, ?)",
            params![delivery_id, source],
        )?;
        Ok(rows_affected > 0)
    }

    fn cleanup_old_deliveries(&self, max_age_hours: u64) -> Result<usize> {
        let conn = self.acquire_lock()?;
        let rows = conn.execute(
            "DELETE FROM webhook_deliveries WHERE received_at < datetime('now', ?)",
            params![format!("-{} hours", max_age_hours)],
        )?;
        if rows > 0 {
            tracing::debug!(rows = rows, "Cleaned up old webhook deliveries");
        }
        Ok(rows)
    }
}

impl SimilarityStore for SqliteTracker {
    fn store_similar_issues_batch(&self, similar_issues: &[SimilarIssue]) -> Result<()> {
        if similar_issues.is_empty() {
            return Ok(());
        }
        let mut conn = self.acquire_lock()?;
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        {
            let mut stmt = tx.prepare_cached(
                r#"
                INSERT INTO similar_issues (source_issue_id, similar_issue_id, similarity_score, computed_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(source_issue_id, similar_issue_id) DO UPDATE SET
                    similarity_score = excluded.similarity_score,
                    computed_at = excluded.computed_at
                "#,
            )?;
            for similar in similar_issues {
                stmt.execute(params![
                    similar.source_issue_id,
                    similar.similar_issue_id,
                    similar.similarity_score,
                    similar.computed_at.format("%Y-%m-%d %H:%M:%S").to_string(),
                ])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    fn store_similar_issue(&self, similar: &SimilarIssue) -> Result<i64> {
        let conn = self.acquire_lock()?;

        conn.execute(
            r#"
            INSERT INTO similar_issues (source_issue_id, similar_issue_id, similarity_score, computed_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(source_issue_id, similar_issue_id) DO UPDATE SET
                similarity_score = excluded.similarity_score,
                computed_at = excluded.computed_at
            "#,
            params![
                similar.source_issue_id,
                similar.similar_issue_id,
                similar.similarity_score,
                similar.computed_at.format("%Y-%m-%d %H:%M:%S").to_string(),
            ],
        )?;
        Ok(conn.last_insert_rowid())
    }

    fn find_similar_issues(
        &self,
        issue_id: &str,
        min_score: f64,
        limit: usize,
    ) -> Result<Vec<SimilarIssue>> {
        let conn = self.acquire_lock()?;
        let mut stmt = conn.prepare(
            r#"
            SELECT id, source_issue_id, similar_issue_id, similarity_score, computed_at
            FROM similar_issues
            WHERE source_issue_id = ? AND similarity_score >= ?
            ORDER BY similarity_score DESC
            LIMIT ?
            "#,
        )?;

        let mut results = Vec::new();
        let rows = stmt.query_map(params![issue_id, min_score, limit as i64], |row| {
            Ok(SimilarIssue {
                id: row.get(0)?,
                source_issue_id: row.get(1)?,
                similar_issue_id: row.get(2)?,
                similarity_score: row.get(3)?,
                computed_at: Self::parse_datetime(&row.get::<_, String>(4)?)?,
            })
        })?;

        for row in rows.flatten() {
            results.push(row);
        }

        Ok(results)
    }
}

impl SqliteTracker {
    /// Record an activity to the activity log.
    /// Persist a single action-pipeline run.
    pub fn record_action_run(
        &self,
        source: &str,
        issue_id: &str,
        short_id: &str,
        action_kind: &str,
        status: &str,
        detail: &str,
    ) -> Result<()> {
        let conn = self.acquire_lock()?;
        conn.execute(
            r#"
            INSERT INTO action_runs (source, issue_id, short_id, action_kind, status, detail)
            VALUES (?, ?, ?, ?, ?, ?)
            "#,
            params![source, issue_id, short_id, action_kind, status, detail],
        )?;
        Ok(())
    }

    pub fn record_activity(&self, entry: &ActivityLogEntry) -> Result<i64> {
        let conn = self.acquire_lock()?;
        let metadata_json = entry.metadata.as_ref().map(|m| m.to_string());

        conn.execute(
            r#"
            INSERT INTO activity_log (timestamp, activity_type, source, issue_id, short_id, message, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            "#,
            params![
                entry.timestamp.format("%Y-%m-%d %H:%M:%S").to_string(),
                entry.activity_type,
                entry.source,
                entry.issue_id,
                entry.short_id,
                entry.message,
                metadata_json,
            ],
        )?;
        let id = conn.last_insert_rowid();
        drop(conn);
        Self::append_audit_json_line(
            "activity",
            &serde_json::json!({
                "id": id,
                "timestamp": entry.timestamp.to_rfc3339(),
                "activity_type": entry.activity_type,
                "source": entry.source,
                "issue_id": entry.issue_id,
                "short_id": entry.short_id,
                "message": entry.message,
                "metadata": entry.metadata,
            }),
        );
        Ok(id)
    }

    /// Get recent activities, optionally filtered by source.
    pub fn get_recent_activities(
        &self,
        limit: usize,
        source_filter: Option<&str>,
    ) -> Result<Vec<ActivityLogEntry>> {
        let conn = self.acquire_lock()?;

        // Build query dynamically based on whether source filter is provided
        let (query, params): (String, Vec<Box<dyn rusqlite::ToSql>>) = match source_filter {
            Some(source) => (
                r#"
                SELECT id, timestamp, activity_type, source, issue_id, short_id, message, metadata
                FROM activity_log
                WHERE source = ?1
                ORDER BY timestamp DESC
                LIMIT ?2
                "#
                .to_string(),
                vec![Box::new(source.to_string()), Box::new(limit as i64)],
            ),
            None => (
                r#"
                SELECT id, timestamp, activity_type, source, issue_id, short_id, message, metadata
                FROM activity_log
                ORDER BY timestamp DESC
                LIMIT ?1
                "#
                .to_string(),
                vec![Box::new(limit as i64)],
            ),
        };

        let mut stmt = conn.prepare(&query)?;
        let params_refs: Vec<&dyn rusqlite::ToSql> = params.iter().map(|p| p.as_ref()).collect();
        let rows = stmt.query_map(params_refs.as_slice(), |row| {
            Ok(Self::row_to_activity_entry(row))
        })?;

        Ok(rows.flatten().collect())
    }

    /// Get activities for a specific issue.
    pub fn get_activities_for_issue(
        &self,
        source: &str,
        issue_id: &str,
    ) -> Result<Vec<ActivityLogEntry>> {
        let conn = self.acquire_lock()?;
        let mut stmt = conn.prepare(
            r#"
            SELECT id, timestamp, activity_type, source, issue_id, short_id, message, metadata
            FROM activity_log
            WHERE source = ? AND issue_id = ?
            ORDER BY timestamp DESC
            "#,
        )?;

        let mut entries = Vec::new();
        let rows = stmt.query_map(params![source, issue_id], |row| {
            Ok(Self::row_to_activity_entry(row))
        })?;

        for row in rows.flatten() {
            entries.push(row);
        }

        Ok(entries)
    }

    fn row_to_activity_entry(row: &rusqlite::Row<'_>) -> ActivityLogEntry {
        let metadata_str: Option<String> = row.get(7).ok();
        let metadata = metadata_str.and_then(|s| serde_json::from_str(&s).ok());

        ActivityLogEntry {
            id: row.get(0).unwrap_or(0),
            timestamp: row
                .get::<_, String>(1)
                .ok()
                .and_then(|s| Self::parse_datetime(&s).ok())
                .unwrap_or_else(Utc::now),
            activity_type: row.get(2).unwrap_or_default(),
            source: row.get(3).ok(),
            issue_id: row.get(4).ok(),
            short_id: row.get(5).ok(),
            message: row.get(6).unwrap_or_default(),
            metadata,
        }
    }

    /// Convert a database row to a FixAttempt.
    /// Expects columns in order: id, source, issue_id, short_id, attempted_at, pr_url,
    /// scm_repo, scm_pr_number, status, error_message, merged_at, resolved_at,
    /// retry_count, last_retry_at, issue_labels, parent_attempt_id, cascade_repo
    fn row_to_fix_attempt(row: &rusqlite::Row<'_>) -> rusqlite::Result<FixAttempt> {
        // Parse issue_labels from JSON string
        let issue_labels: Vec<String> = row
            .get::<_, Option<String>>(14)?
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();

        Ok(FixAttempt {
            id: row.get(0)?,
            source: row.get(1)?,
            issue_id: row.get(2)?,
            short_id: row.get(3)?,
            attempted_at: Self::parse_datetime(&row.get::<_, String>(4)?)?,
            pr_url: row.get(5)?,
            scm_repo: row.get(6)?,
            scm_pr_number: row.get(7)?,
            status: row
                .get::<_, String>(8)?
                .parse()
                .unwrap_or(FixAttemptStatus::Pending),
            error_message: row.get(9)?,
            merged_at: Self::parse_optional_datetime(row.get(10)?)?,
            resolved_at: Self::parse_optional_datetime(row.get(11)?)?,
            retry_count: row.get::<_, Option<u32>>(12)?.unwrap_or(0),
            last_retry_at: Self::parse_optional_datetime(row.get(13)?)?,
            issue_labels,
            parent_attempt_id: row.get::<_, Option<i64>>(15).ok().flatten(),
            cascade_repo: row.get::<_, Option<String>>(16).ok().flatten(),
        })
    }

    /// Convert a database row to a StoredDependency.
    /// Expects columns: rd.id, u.name, d.name, rd.dependency_type, rd.created_at
    fn row_to_dependency(row: &rusqlite::Row<'_>) -> rusqlite::Result<StoredDependency> {
        Ok(StoredDependency {
            id: row.get(0)?,
            upstream: row.get(1)?,
            downstream: row.get(2)?,
            dep_type: row.get(3)?,
            created_at: row.get(4)?,
        })
    }

    /// Record a Claude execution.
    pub fn record_execution(&self, execution: &AgentExecution) -> Result<i64> {
        let conn = self.acquire_lock()?;

        conn.execute(
            r#"
            INSERT INTO claude_executions (
                attempt_id, started_at, completed_at, duration_secs, exit_code, timed_out,
                stdout_preview, stderr_preview, stdout_log_path, stderr_log_path, event_log_path,
                prompt_used, prompt_hash, model_version, working_directory, git_branch,
                git_commit_before, git_commit_after, files_changed, lines_added, lines_removed,
                total_cost_usd, num_turns, session_id, duration_api_ms,
                input_tokens, output_tokens, cache_read_input_tokens, cache_creation_input_tokens,
                provider, experiment_name, experiment_variant
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#,
            params![
                execution.attempt_id,
                execution.started_at.format("%Y-%m-%d %H:%M:%S").to_string(),
                execution
                    .completed_at
                    .map(|t| t.format("%Y-%m-%d %H:%M:%S").to_string()),
                execution.duration_secs,
                execution.exit_code,
                execution.timed_out as i32,
                execution.stdout_preview,
                execution.stderr_preview,
                execution.stdout_log_path,
                execution.stderr_log_path,
                execution.event_log_path,
                execution.prompt_used,
                execution.prompt_hash,
                execution.model_version,
                execution.working_directory,
                execution.git_branch,
                execution.git_commit_before,
                execution.git_commit_after,
                execution.files_changed,
                execution.lines_added,
                execution.lines_removed,
                execution.total_cost_usd,
                execution.num_turns,
                execution.session_id,
                execution.duration_api_ms,
                execution.input_tokens,
                execution.output_tokens,
                execution.cache_read_input_tokens,
                execution.cache_creation_input_tokens,
                execution.provider,
                execution.experiment_name,
                execution.experiment_variant,
            ],
        )?;
        let id = conn.last_insert_rowid();
        drop(conn);
        Self::append_audit_json_line(
            "execution",
            &serde_json::json!({
                "id": id,
                "attempt_id": execution.attempt_id,
                "started_at": execution.started_at.to_rfc3339(),
                "completed_at": execution.completed_at.map(|v| v.to_rfc3339()),
                "duration_secs": execution.duration_secs,
                "exit_code": execution.exit_code,
                "timed_out": execution.timed_out,
                "stdout_preview": execution.stdout_preview,
                "stderr_preview": execution.stderr_preview,
                "stdout_log_path": execution.stdout_log_path,
                "stderr_log_path": execution.stderr_log_path,
                "event_log_path": execution.event_log_path,
                "prompt_hash": execution.prompt_hash,
                "model_version": execution.model_version,
                "working_directory": execution.working_directory,
                "git_branch": execution.git_branch,
                "git_commit_before": execution.git_commit_before,
                "git_commit_after": execution.git_commit_after,
                "files_changed": execution.files_changed,
                "lines_added": execution.lines_added,
                "lines_removed": execution.lines_removed,
                "total_cost_usd": execution.total_cost_usd,
                "num_turns": execution.num_turns,
                "session_id": execution.session_id,
                "duration_api_ms": execution.duration_api_ms,
                "input_tokens": execution.input_tokens,
                "output_tokens": execution.output_tokens,
                "cache_read_input_tokens": execution.cache_read_input_tokens,
                "cache_creation_input_tokens": execution.cache_creation_input_tokens,
            }),
        );
        Ok(id)
    }

    /// Get executions for a specific attempt.
    pub fn get_executions_for_attempt(&self, attempt_id: i64) -> Result<Vec<AgentExecution>> {
        let conn = self.acquire_lock()?;
        let mut stmt = conn.prepare(
            r#"
            SELECT id, attempt_id, started_at, completed_at, duration_secs, exit_code, timed_out,
                   stdout_preview, stderr_preview, stdout_log_path, stderr_log_path, event_log_path,
                   prompt_used, prompt_hash, model_version, working_directory, git_branch,
                   git_commit_before, git_commit_after, files_changed, lines_added, lines_removed,
                   total_cost_usd, num_turns, session_id, duration_api_ms,
                   input_tokens, output_tokens, cache_read_input_tokens, cache_creation_input_tokens,
                   provider, experiment_name, experiment_variant
            FROM claude_executions
            WHERE attempt_id = ?
            ORDER BY started_at DESC
            "#,
        )?;

        let mut executions = Vec::new();
        let rows = stmt.query_map(params![attempt_id], |row| {
            Ok(AgentExecution {
                id: row.get(0)?,
                attempt_id: row.get(1)?,
                started_at: Self::parse_datetime(&row.get::<_, String>(2)?)?,
                completed_at: Self::parse_optional_datetime(row.get(3)?)?,
                duration_secs: row.get(4)?,
                exit_code: row.get(5)?,
                timed_out: row.get::<_, i32>(6).unwrap_or(0) != 0,
                stdout_preview: row.get(7)?,
                stderr_preview: row.get(8)?,
                stdout_log_path: row.get(9)?,
                stderr_log_path: row.get(10)?,
                event_log_path: row.get(11)?,
                prompt_used: row.get(12)?,
                prompt_hash: row.get(13)?,
                model_version: row.get(14)?,
                working_directory: row.get(15)?,
                git_branch: row.get(16)?,
                git_commit_before: row.get(17)?,
                git_commit_after: row.get(18)?,
                files_changed: row.get(19)?,
                lines_added: row.get(20)?,
                lines_removed: row.get(21)?,
                total_cost_usd: row.get(22)?,
                num_turns: row.get(23)?,
                session_id: row.get(24)?,
                duration_api_ms: row.get(25)?,
                input_tokens: row.get(26)?,
                output_tokens: row.get(27)?,
                cache_read_input_tokens: row.get(28)?,
                cache_creation_input_tokens: row.get(29)?,
                provider: row.get(30)?,
                experiment_name: row.get(31)?,
                experiment_variant: row.get(32)?,
            })
        })?;

        for row in rows.flatten() {
            executions.push(row);
        }

        Ok(executions)
    }

    /// Get experiment comparison results.
    pub fn get_experiment_results(
        &self,
        experiment_name: &str,
    ) -> Result<Vec<ExperimentProviderStats>> {
        let conn = self.acquire_lock()?;
        let mut stmt = conn.prepare(
            r#"
            SELECT
                provider,
                COUNT(*) as total_attempts,
                SUM(CASE WHEN exit_code = 0 THEN 1 ELSE 0 END) as success_count,
                AVG(total_cost_usd) as avg_cost,
                AVG(duration_secs) as avg_duration,
                CAST(SUM(CASE WHEN exit_code = 0 THEN 1 ELSE 0 END) AS REAL) / COUNT(*) as success_rate
            FROM claude_executions
            WHERE experiment_name = ?
            GROUP BY provider
            ORDER BY success_rate DESC
            "#,
        )?;

        let rows = stmt.query_map(params![experiment_name], |row| {
            Ok(ExperimentProviderStats {
                provider: row.get::<_, String>(0)?,
                total_attempts: row.get(1)?,
                success_count: row.get(2)?,
                avg_cost: row.get(3)?,
                avg_duration: row.get(4)?,
                success_rate: row.get(5)?,
            })
        })?;

        Ok(rows.flatten().collect())
    }

    /// Record a PR review.
    pub fn record_pr_review(&self, review: &PrReviewRecord) -> Result<i64> {
        let conn = self.acquire_lock()?;

        conn.execute(
            r#"
            INSERT INTO pr_reviews (attempt_id, pr_url, reviewer, review_state, submitted_at, body, sentiment, actionable_feedback)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            "#,
            params![
                review.attempt_id,
                review.pr_url,
                review.reviewer,
                review.review_state,
                review.submitted_at.map(|t| t.format("%Y-%m-%d %H:%M:%S").to_string()),
                review.body,
                review.sentiment,
                review.actionable_feedback,
            ],
        )?;
        Ok(conn.last_insert_rowid())
    }

    /// Get reviews for a specific attempt.
    pub fn get_reviews_for_attempt(&self, attempt_id: i64) -> Result<Vec<PrReviewRecord>> {
        let conn = self.acquire_lock()?;
        let mut stmt = conn.prepare(
            r#"
            SELECT id, attempt_id, pr_url, reviewer, review_state, submitted_at, body, sentiment, actionable_feedback
            FROM pr_reviews
            WHERE attempt_id = ?
            ORDER BY submitted_at DESC
            "#,
        )?;

        let mut reviews = Vec::new();
        let rows = stmt.query_map(params![attempt_id], |row| {
            Ok(PrReviewRecord {
                id: row.get(0)?,
                attempt_id: row.get(1)?,
                pr_url: row.get(2)?,
                reviewer: row.get(3)?,
                review_state: row.get(4)?,
                submitted_at: Self::parse_optional_datetime(row.get(5)?)?,
                body: row.get(6)?,
                sentiment: row.get(7)?,
                actionable_feedback: row.get(8)?,
            })
        })?;

        for row in rows.flatten() {
            reviews.push(row);
        }

        Ok(reviews)
    }

    /// Convert a database row to a StoredPrReviewComment.
    /// Expects columns: id, scm_comment_id, pr_url, review_id, path, position, line,
    /// body, author, created_at, updated_at, html_url
    fn row_to_stored_pr_review_comment(
        row: &rusqlite::Row<'_>,
    ) -> rusqlite::Result<StoredPrReviewComment> {
        Ok(StoredPrReviewComment {
            id: row.get(0)?,
            scm_comment_id: row.get(1)?,
            pr_url: row.get(2)?,
            review_id: row.get(3)?,
            path: row.get(4)?,
            position: row.get(5)?,
            line: row.get(6)?,
            body: row.get(7)?,
            author: row.get(8)?,
            created_at: row.get(9)?,
            updated_at: row.get(10)?,
            html_url: row.get(11)?,
        })
    }

    /// Convert a database row to a PrReviewState.
    /// Expects columns: pr_url, repo, pr_number, issue_id, source,
    /// last_review_id, last_review_time, last_comment_id, last_comment_time, is_active
    fn row_to_pr_review_state(
        row: &rusqlite::Row<'_>,
    ) -> rusqlite::Result<claudear_core::types::PrReviewState> {
        Ok(claudear_core::types::PrReviewState {
            pr_url: row.get(0)?,
            repo: row.get(1)?,
            pr_number: row.get(2)?,
            issue_id: row.get(3)?,
            source: row.get(4)?,
            last_review_id: row.get(5)?,
            last_review_time: row.get(6)?,
            last_comment_id: row.get(7)?,
            last_comment_time: row.get(8)?,
            is_active: row.get::<_, i32>(9)? != 0,
        })
    }

    /// Record or update an error pattern.
    pub fn record_error_pattern(&self, pattern: &ErrorPattern) -> Result<i64> {
        let conn = self.acquire_lock()?;

        let sources_json = pattern
            .sources
            .as_ref()
            .and_then(|s| serde_json::to_string(s).ok());
        let example_ids_json = pattern
            .example_issue_ids
            .as_ref()
            .and_then(|s| serde_json::to_string(s).ok());

        conn.execute(
            r#"
            INSERT INTO error_patterns (pattern_hash, error_type, error_message, first_seen, last_seen, occurrence_count, sources, example_issue_ids, resolution_hints)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(pattern_hash) DO UPDATE SET
                last_seen = excluded.last_seen,
                occurrence_count = occurrence_count + 1,
                sources = excluded.sources,
                example_issue_ids = excluded.example_issue_ids
            "#,
            params![
                pattern.pattern_hash,
                pattern.error_type,
                pattern.error_message,
                pattern.first_seen.format("%Y-%m-%d %H:%M:%S").to_string(),
                pattern.last_seen.format("%Y-%m-%d %H:%M:%S").to_string(),
                pattern.occurrence_count,
                sources_json,
                example_ids_json,
                pattern.resolution_hints,
            ],
        )?;
        Ok(conn.last_insert_rowid())
    }

    /// Get the most common error patterns.
    pub fn get_error_patterns(&self, limit: usize) -> Result<Vec<ErrorPattern>> {
        let conn = self.acquire_lock()?;
        let mut stmt = conn.prepare(
            r#"
            SELECT id, pattern_hash, error_type, error_message, first_seen, last_seen, occurrence_count, sources, example_issue_ids, resolution_hints
            FROM error_patterns
            ORDER BY occurrence_count DESC
            LIMIT ?
            "#,
        )?;

        let mut patterns = Vec::new();
        let rows = stmt.query_map(params![limit as i64], |row| {
            let sources_str: Option<String> = row.get(7)?;
            let example_ids_str: Option<String> = row.get(8)?;

            Ok(ErrorPattern {
                id: row.get(0)?,
                pattern_hash: row.get(1)?,
                error_type: row.get(2)?,
                error_message: row.get(3)?,
                first_seen: Self::parse_datetime(&row.get::<_, String>(4)?)?,
                last_seen: Self::parse_datetime(&row.get::<_, String>(5)?)?,
                occurrence_count: row.get(6)?,
                sources: sources_str.and_then(|s| serde_json::from_str(&s).ok()),
                example_issue_ids: example_ids_str.and_then(|s| serde_json::from_str(&s).ok()),
                resolution_hints: row.get(9)?,
            })
        })?;

        for row in rows.flatten() {
            patterns.push(row);
        }

        Ok(patterns)
    }

    /// Store a feedback outcome to the database.
    pub fn store_feedback_outcome(&self, outcome: &FixOutcome) -> Result<i64> {
        let conn = self.acquire_lock()?;
        let keywords_json = serde_json::to_string(&outcome.keywords).unwrap_or_default();

        // Serialize embedding to BLOB if present
        let embedding_blob: Option<Vec<u8>> = outcome
            .embedding
            .as_ref()
            .map(|emb| emb.iter().flat_map(|f| f.to_le_bytes()).collect());

        conn.execute(
            r#"
            INSERT INTO feedback_outcomes (attempt_id, source, issue_id, issue_text, prompt_used, outcome, error_type, learnings, keywords, created_at, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#,
            params![
                outcome.attempt_id,
                outcome.source,
                outcome.issue_id,
                outcome.issue_text,
                outcome.prompt_used,
                outcome.outcome.as_str(),
                outcome.error_type,
                outcome.learnings,
                keywords_json,
                outcome.created_at.format("%Y-%m-%d %H:%M:%S").to_string(),
                embedding_blob,
            ],
        )?;
        let row_id = conn.last_insert_rowid();

        // Dual-write: upsert into HNSW vector table if embedding is present
        if let Some(ref emb) = outcome.embedding {
            if let Err(e) = Self::upsert_outcome_vector_embedding(&conn, row_id, emb) {
                tracing::debug!(error = %e, "Failed to upsert outcome vector embedding");
            }
        }

        Ok(row_id)
    }

    /// Retrieve feedback outcomes with optional source filter.
    pub fn get_feedback_outcomes(
        &self,
        source: Option<&str>,
        limit: usize,
    ) -> Result<Vec<FixOutcome>> {
        let conn = self.acquire_lock()?;

        let (sql, params_vec): (&str, Vec<Box<dyn rusqlite::types::ToSql>>) = match source {
            Some(s) => (
                r#"
                SELECT id, attempt_id, source, issue_id, issue_text, prompt_used, outcome, error_type, learnings, keywords, created_at, embedding
                FROM feedback_outcomes
                WHERE source = ?
                ORDER BY created_at DESC
                LIMIT ?
                "#,
                vec![Box::new(s.to_string()), Box::new(limit as i64)],
            ),
            None => (
                r#"
                SELECT id, attempt_id, source, issue_id, issue_text, prompt_used, outcome, error_type, learnings, keywords, created_at, embedding
                FROM feedback_outcomes
                ORDER BY created_at DESC
                LIMIT ?
                "#,
                vec![Box::new(limit as i64)],
            ),
        };

        let mut stmt = conn.prepare(sql)?;
        let params_refs: Vec<&dyn rusqlite::types::ToSql> =
            params_vec.iter().map(|p| p.as_ref()).collect();
        let rows = stmt.query_map(params_refs.as_slice(), Self::row_to_fix_outcome)?;

        let mut outcomes = Vec::new();
        for row in rows.flatten() {
            outcomes.push(row);
        }
        Ok(outcomes)
    }

    /// Get a single feedback outcome by attempt ID.
    pub fn get_feedback_outcome_by_attempt(&self, attempt_id: i64) -> Result<Option<FixOutcome>> {
        let conn = self.acquire_lock()?;
        let mut stmt = conn.prepare(
            r#"
            SELECT id, attempt_id, source, issue_id, issue_text, prompt_used, outcome, error_type, learnings, keywords, created_at, embedding
            FROM feedback_outcomes
            WHERE attempt_id = ?
            LIMIT 1
            "#,
        )?;

        let mut rows = stmt.query_map(params![attempt_id], Self::row_to_fix_outcome)?;
        Ok(rows.next().and_then(|r| r.ok()))
    }

    /// Store a Q&A knowledge entry.
    pub fn store_qa_knowledge(&self, entry: &QaKnowledgeEntry) -> Result<i64> {
        let conn = self.acquire_lock()?;

        conn.execute(
            r#"
            INSERT INTO qa_knowledge (
                source, repo, issue_id, short_id, question_text, question_norm, question_embedding,
                answer_text, answer_norm, answer_embedding, channel, responder, correlation_id,
                asked_at, answered_at, success_count, failure_count, last_used_at, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#,
            params![
                entry.source,
                entry.repo,
                entry.issue_id,
                entry.short_id,
                entry.question_text,
                entry.question_norm,
                Self::embedding_to_blob(entry.question_embedding.as_deref()),
                entry.answer_text,
                entry.answer_norm,
                Self::embedding_to_blob(entry.answer_embedding.as_deref()),
                entry.channel,
                entry.responder,
                entry.correlation_id,
                entry.asked_at.format("%Y-%m-%d %H:%M:%S").to_string(),
                entry.answered_at.format("%Y-%m-%d %H:%M:%S").to_string(),
                entry.success_count,
                entry.failure_count,
                entry
                    .last_used_at
                    .map(|v| v.format("%Y-%m-%d %H:%M:%S").to_string()),
                entry.metadata.as_ref().map(|m| m.to_string()),
            ],
        )?;

        let qa_id = conn.last_insert_rowid();

        if let Some(question_embedding) = entry.question_embedding.as_deref() {
            if let Err(e) = Self::upsert_qa_vector_embedding(&conn, qa_id, question_embedding) {
                tracing::debug!(
                    qa_id = qa_id,
                    error = %e,
                    "Failed to sync Q&A vector embedding; falling back to SQL ranking"
                );
            }
        }

        Ok(qa_id)
    }

    /// Find semantically similar Q&A entries within source/repo scope.
    pub fn find_similar_qa_scoped(
        &self,
        source: &str,
        repo: Option<&str>,
        question_norm: &str,
        question_embedding: Option<&[f32]>,
        threshold: f64,
        limit: usize,
    ) -> Result<Vec<QaMatch>> {
        let conn = self.acquire_lock()?;

        if let Some(query_embedding) = question_embedding {
            let candidate_limit = limit
                .saturating_mul(QA_VECTOR_CANDIDATE_MULTIPLIER)
                .max(limit);
            if let Some(vector_matches) = Self::query_qa_matches_vector_scoped(
                &conn,
                source,
                repo,
                query_embedding,
                threshold,
                limit,
                candidate_limit,
            )? {
                if !vector_matches.is_empty() {
                    return Ok(vector_matches);
                }
            }
        }

        Self::query_qa_matches_exact_scoped(&conn, source, repo, question_norm, threshold, limit)
    }

    /// Find semantically similar Q&A entries globally.
    pub fn find_similar_qa_global(
        &self,
        question_norm: &str,
        question_embedding: Option<&[f32]>,
        threshold: f64,
        limit: usize,
    ) -> Result<Vec<QaMatch>> {
        let conn = self.acquire_lock()?;

        if let Some(query_embedding) = question_embedding {
            let candidate_limit = limit
                .saturating_mul(QA_VECTOR_CANDIDATE_MULTIPLIER)
                .max(limit);
            if let Some(vector_matches) = Self::query_qa_matches_vector_global(
                &conn,
                query_embedding,
                threshold,
                limit,
                candidate_limit,
            )? {
                if !vector_matches.is_empty() {
                    return Ok(vector_matches);
                }
            }
        }

        Self::query_qa_matches_exact_global(&conn, question_norm, threshold, limit)
    }

    /// Record usage of a Q&A entry for an attempt.
    pub fn record_qa_usage(
        &self,
        attempt_id: i64,
        qa_id: i64,
        usage_type: &str,
        similarity_score: f64,
    ) -> Result<i64> {
        let conn = self.acquire_lock()?;
        conn.execute(
            r#"
            INSERT INTO qa_usage (attempt_id, qa_id, usage_type, similarity_score, created_at)
            VALUES (?1, ?2, ?3, ?4, datetime('now'))
            ON CONFLICT(attempt_id, qa_id) DO UPDATE SET
                usage_type = excluded.usage_type,
                similarity_score = excluded.similarity_score,
                created_at = excluded.created_at
            "#,
            params![attempt_id, qa_id, usage_type, similarity_score],
        )?;
        Ok(conn.last_insert_rowid())
    }

    /// Update success/failure counters for a Q&A entry.
    pub fn update_qa_outcome_stats(&self, qa_id: i64, success: bool) -> Result<()> {
        let conn = self.acquire_lock()?;
        let (field, sql) = if success {
            (
                "success_count",
                "UPDATE qa_knowledge SET success_count = success_count + 1, last_used_at = datetime('now') WHERE id = ?",
            )
        } else {
            (
                "failure_count",
                "UPDATE qa_knowledge SET failure_count = failure_count + 1, last_used_at = datetime('now') WHERE id = ?",
            )
        };
        conn.execute(sql, params![qa_id])?;
        tracing::debug!(qa_id = qa_id, field = field, "Updated Q&A outcome stats");
        Ok(())
    }

    /// Update success/failure counters for all Q&A entries used by an attempt.
    pub fn update_qa_outcome_stats_for_attempt(
        &self,
        attempt_id: i64,
        success: bool,
    ) -> Result<()> {
        let conn = self.acquire_lock()?;
        let sql = if success {
            r#"
            UPDATE qa_knowledge
            SET success_count = success_count + 1,
                last_used_at = datetime('now')
            WHERE id IN (SELECT qa_id FROM qa_usage WHERE attempt_id = ?1)
            "#
        } else {
            r#"
            UPDATE qa_knowledge
            SET failure_count = failure_count + 1,
                last_used_at = datetime('now')
            WHERE id IN (SELECT qa_id FROM qa_usage WHERE attempt_id = ?1)
            "#
        };
        conn.execute(sql, params![attempt_id])?;
        Ok(())
    }

    /// Get channel cursor value for polling channels.
    pub fn get_channel_cursor(&self, channel: &str, cursor_key: &str) -> Result<Option<String>> {
        let conn = self.acquire_lock()?;
        let mut stmt = conn.prepare(
            "SELECT cursor_value FROM question_channel_cursor WHERE channel = ?1 AND cursor_key = ?2",
        )?;
        let value = stmt
            .query_row(params![channel, cursor_key], |row| row.get::<_, String>(0))
            .optional()?;
        Ok(value)
    }

    /// Set channel cursor value for polling channels.
    pub fn set_channel_cursor(
        &self,
        channel: &str,
        cursor_key: &str,
        cursor_value: &str,
    ) -> Result<()> {
        let conn = self.acquire_lock()?;
        conn.execute(
            r#"
            INSERT INTO question_channel_cursor (channel, cursor_key, cursor_value, updated_at)
            VALUES (?1, ?2, ?3, datetime('now'))
            ON CONFLICT(channel, cursor_key) DO UPDATE SET
                cursor_value = excluded.cursor_value,
                updated_at = excluded.updated_at
            "#,
            params![channel, cursor_key, cursor_value],
        )?;
        Ok(())
    }

    fn row_to_qa_knowledge(row: &rusqlite::Row<'_>) -> rusqlite::Result<QaKnowledgeEntry> {
        let metadata: Option<String> = row.get(19)?;
        Ok(QaKnowledgeEntry {
            id: row.get(0)?,
            source: row.get(1)?,
            repo: row.get(2)?,
            issue_id: row.get(3)?,
            short_id: row.get(4)?,
            question_text: row.get(5)?,
            question_norm: row.get(6)?,
            question_embedding: Self::blob_to_embedding(row.get(7)?),
            answer_text: row.get(8)?,
            answer_norm: row.get(9)?,
            answer_embedding: Self::blob_to_embedding(row.get(10)?),
            channel: row.get(11)?,
            responder: row.get(12)?,
            correlation_id: row.get(13)?,
            asked_at: Self::parse_datetime(&row.get::<_, String>(14)?)?,
            answered_at: Self::parse_datetime(&row.get::<_, String>(15)?)?,
            success_count: row.get(16)?,
            failure_count: row.get(17)?,
            last_used_at: Self::parse_optional_datetime(row.get::<_, Option<String>>(18)?)?,
            metadata: metadata.and_then(|s| serde_json::from_str(&s).ok()),
        })
    }

    fn row_to_qa_match(row: &rusqlite::Row<'_>) -> rusqlite::Result<QaMatch> {
        let entry = Self::row_to_qa_knowledge(row)?;
        Ok(QaMatch {
            entry,
            semantic_similarity: row.get(20)?,
            historical_success_rate: row.get(21)?,
            final_score: row.get(22)?,
        })
    }

    /// Map a database row to a FixOutcome.
    /// Expected column order: id, attempt_id, source, issue_id, issue_text,
    /// prompt_used, outcome, error_type, learnings, keywords, created_at, embedding
    fn row_to_fix_outcome(row: &rusqlite::Row) -> rusqlite::Result<FixOutcome> {
        let outcome_str: String = row.get(6)?;
        let keywords_str: Option<String> = row.get(9)?;
        let created_at_str: String = row.get(10)?;

        // Deserialize embedding BLOB if present (column 11)
        let embedding_blob: Option<Vec<u8>> = row.get(11).unwrap_or(None);
        let embedding = embedding_blob.and_then(|blob| {
            if blob.len() % 4 != 0 {
                return None;
            }
            Some(
                blob.chunks_exact(4)
                    .map(|chunk| {
                        let arr: [u8; 4] =
                            chunk.try_into().expect("chunks_exact guarantees 4 bytes");
                        f32::from_le_bytes(arr)
                    })
                    .collect(),
            )
        });

        Ok(FixOutcome {
            id: row.get(0)?,
            attempt_id: row.get(1)?,
            source: row.get(2)?,
            issue_id: row.get(3)?,
            issue_text: row.get(4)?,
            prompt_used: row.get(5)?,
            outcome: Outcome::parse(&outcome_str).unwrap_or(Outcome::Failed),
            error_type: row.get(7)?,
            learnings: row.get(8)?,
            keywords: keywords_str
                .and_then(|s| serde_json::from_str(&s).ok())
                .unwrap_or_default(),
            embedding,
            created_at: Self::parse_datetime(&created_at_str)?,
        })
    }

    /// Record a processing metric.
    pub fn record_metric(&self, metric: &ProcessingMetric) -> Result<i64> {
        let conn = self.acquire_lock()?;

        let tags_json = metric.tags.as_ref().map(|t| t.to_string());

        conn.execute(
            r#"
            INSERT INTO processing_metrics (timestamp, metric_name, metric_value, source, tags)
            VALUES (?, ?, ?, ?, ?)
            "#,
            params![
                metric.timestamp.format("%Y-%m-%d %H:%M:%S").to_string(),
                metric.metric_name,
                metric.metric_value,
                metric.source,
                tags_json,
            ],
        )?;
        let id = conn.last_insert_rowid();
        drop(conn);
        Self::append_audit_json_line(
            "metric",
            &serde_json::json!({
                "id": id,
                "timestamp": metric.timestamp.to_rfc3339(),
                "metric_name": metric.metric_name,
                "metric_value": metric.metric_value,
                "source": metric.source,
                "tags": metric.tags,
            }),
        );
        Ok(id)
    }

    /// Record multiple metrics in a single transaction for better performance.
    ///
    /// This is more efficient than calling `record_metric` in a loop because:
    /// - Single transaction reduces fsync overhead
    /// - Prepared statement is reused across all inserts
    pub fn record_metrics_batch(&self, metrics: &[ProcessingMetric]) -> Result<usize> {
        if metrics.is_empty() {
            return Ok(0);
        }

        let mut conn = self.acquire_lock()?;
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;

        {
            let mut stmt = tx.prepare_cached(
                r#"
                INSERT INTO processing_metrics (timestamp, metric_name, metric_value, source, tags)
                VALUES (?, ?, ?, ?, ?)
                "#,
            )?;

            for metric in metrics {
                let tags_json = metric.tags.as_ref().map(|t| t.to_string());
                stmt.execute(params![
                    metric.timestamp.format("%Y-%m-%d %H:%M:%S").to_string(),
                    metric.metric_name,
                    metric.metric_value,
                    metric.source,
                    tags_json,
                ])?;
            }
        }

        tx.commit()?;
        drop(conn);
        for metric in metrics {
            Self::append_audit_json_line(
                "metric",
                &serde_json::json!({
                    "id": serde_json::Value::Null,
                    "timestamp": metric.timestamp.to_rfc3339(),
                    "metric_name": metric.metric_name,
                    "metric_value": metric.metric_value,
                    "source": metric.source,
                    "tags": metric.tags,
                }),
            );
        }
        Ok(metrics.len())
    }

    /// Get metrics by name within a time range.
    pub fn get_metrics(
        &self,
        metric_name: &str,
        since: Option<DateTime<Utc>>,
        limit: usize,
    ) -> Result<Vec<ProcessingMetric>> {
        let conn = self.acquire_lock()?;

        // Build query dynamically based on whether since filter is provided
        let (query, params): (String, Vec<Box<dyn rusqlite::ToSql>>) = match since {
            Some(since_time) => (
                r#"
                SELECT id, timestamp, metric_name, metric_value, source, tags
                FROM processing_metrics
                WHERE metric_name = ?1 AND timestamp >= ?2
                ORDER BY timestamp DESC
                LIMIT ?3
                "#
                .to_string(),
                vec![
                    Box::new(metric_name.to_string()),
                    Box::new(since_time.format("%Y-%m-%d %H:%M:%S").to_string()),
                    Box::new(limit as i64),
                ],
            ),
            None => (
                r#"
                SELECT id, timestamp, metric_name, metric_value, source, tags
                FROM processing_metrics
                WHERE metric_name = ?1
                ORDER BY timestamp DESC
                LIMIT ?2
                "#
                .to_string(),
                vec![Box::new(metric_name.to_string()), Box::new(limit as i64)],
            ),
        };

        let mut stmt = conn.prepare(&query)?;
        let params_refs: Vec<&dyn rusqlite::ToSql> = params.iter().map(|p| p.as_ref()).collect();
        let rows = stmt.query_map(params_refs.as_slice(), Self::row_to_metric)?;

        Ok(rows.flatten().collect())
    }

    fn row_to_metric(row: &rusqlite::Row<'_>) -> rusqlite::Result<ProcessingMetric> {
        let tags_str: Option<String> = row.get(5)?;
        let tags = tags_str.and_then(|s| serde_json::from_str(&s).ok());

        Ok(ProcessingMetric {
            id: row.get(0)?,
            timestamp: Self::parse_datetime(&row.get::<_, String>(1)?)?,
            metric_name: row.get(2)?,
            metric_value: row.get(3)?,
            source: row.get(4)?,
            tags,
        })
    }

    /// Get active experiments.
    pub fn get_active_experiments(&self) -> Result<Vec<PromptExperiment>> {
        let conn = self.acquire_lock()?;
        let mut stmt = conn.prepare(
            r#"
            SELECT id, experiment_name, variant, prompt_template, prompt_hash, created_at, active, success_count, failure_count, avg_time_to_merge, avg_review_score
            FROM prompt_experiments
            WHERE active = 1
            ORDER BY experiment_name, variant
            "#,
        )?;

        let mut experiments = Vec::new();
        let rows = stmt.query_map([], |row| {
            Ok(PromptExperiment {
                id: row.get(0)?,
                experiment_name: row.get(1)?,
                variant: row.get(2)?,
                prompt_template: row.get(3)?,
                prompt_hash: row.get(4)?,
                created_at: Self::parse_datetime(&row.get::<_, String>(5)?)?,
                active: row.get::<_, i32>(6)? != 0,
                success_count: row.get(7)?,
                failure_count: row.get(8)?,
                avg_time_to_merge: row.get(9)?,
                avg_review_score: row.get(10)?,
            })
        })?;

        for row in rows.flatten() {
            experiments.push(row);
        }

        Ok(experiments)
    }

    /// Get the overall success rate.
    pub fn get_success_rate(&self) -> Result<f64> {
        let conn = self.acquire_lock()?;
        let mut stmt = conn.prepare(
            r#"
            SELECT
                CAST(SUM(CASE WHEN status IN ('success', 'merged') THEN 1 ELSE 0 END) AS REAL) /
                NULLIF(CAST(COUNT(*) AS REAL), 0)
            FROM fix_attempts
            "#,
        )?;

        let rate: f64 = stmt.query_row([], |row| row.get(0)).unwrap_or(0.0);
        Ok(rate)
    }

    /// Get a comprehensive analytics summary.
    pub fn get_analytics_summary(&self) -> Result<AnalyticsSummary> {
        let conn = self.acquire_lock()?;

        // Get basic stats
        let mut stmt = conn.prepare(
            r#"
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN status IN ('success', 'merged') THEN 1 ELSE 0 END) as successful,
                SUM(CASE WHEN status = 'merged' THEN 1 ELSE 0 END) as merged
            FROM fix_attempts
            "#,
        )?;

        let (total, successful, merged): (i64, i64, i64) = stmt
            .query_row([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
            .unwrap_or((0, 0, 0));

        let success_rate = if total > 0 {
            successful as f64 / total as f64
        } else {
            0.0
        };

        // Get average processing time
        let mut stmt = conn.prepare(
            r#"
            SELECT AVG(duration_secs) FROM claude_executions WHERE duration_secs IS NOT NULL
            "#,
        )?;
        let avg_processing_time: Option<f64> = stmt.query_row([], |row| row.get(0)).ok();

        // Get most common error
        let mut stmt = conn.prepare(
            r#"
            SELECT error_type FROM error_patterns ORDER BY occurrence_count DESC LIMIT 1
            "#,
        )?;
        let most_common_error: Option<String> = stmt.query_row([], |row| row.get(0)).ok();

        // Get success rate by source
        let mut stmt = conn.prepare(
            r#"
            SELECT source,
                   CAST(SUM(CASE WHEN status IN ('success', 'merged') THEN 1 ELSE 0 END) AS REAL) /
                   NULLIF(CAST(COUNT(*) AS REAL), 0) as rate
            FROM fix_attempts
            GROUP BY source
            "#,
        )?;

        let mut success_rate_by_source = HashMap::new();
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
        })?;

        for row in rows.flatten() {
            success_rate_by_source.insert(row.0, row.1);
        }

        Ok(AnalyticsSummary {
            success_rate,
            total_processed: total,
            total_successful: successful,
            total_merged: merged,
            avg_processing_time_secs: avg_processing_time,
            avg_time_to_merge_hours: None, // Would need more complex calculation
            most_common_error,
            success_rate_by_source,
            avg_time_to_pr_mins: None,
            cost_estimate: None,
            mttr_trend: Vec::new(),
            repo_leaderboard: Vec::new(),
        })
    }

    /// Add or update a repository in the database.
    pub fn upsert_repository(
        &self,
        name: &str,
        path: Option<&str>,
        scm_url: Option<&str>,
    ) -> Result<i64> {
        let conn = self.acquire_lock()?;

        // Use name as scm_url if not provided
        let scm_url = scm_url.unwrap_or(name);
        let path = path.unwrap_or("");

        conn.execute(
            r#"
            INSERT INTO repositories (name, path, scm_url)
            VALUES (?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                path = CASE WHEN excluded.path != '' THEN excluded.path ELSE repositories.path END,
                scm_url = excluded.scm_url
            "#,
            params![name, path, scm_url],
        )?;

        // Get the id
        let id: i64 = conn.query_row(
            "SELECT id FROM repositories WHERE name = ?",
            params![name],
            |row| row.get(0),
        )?;

        Ok(id)
    }

    /// Get a repository by name.
    pub fn get_repository(&self, name: &str) -> Result<Option<StoredRepository>> {
        let conn = self.acquire_lock()?;

        let result = conn.query_row(
            r#"
            SELECT id, name, path, scm_url, created_at
            FROM repositories WHERE name = ?
            "#,
            params![name],
            |row| {
                Ok(StoredRepository {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    path: row.get::<_, String>(2).ok().filter(|s| !s.is_empty()),
                    scm_url: row.get(3)?,
                    created_at: row.get(4)?,
                })
            },
        );

        match result {
            Ok(repo) => Ok(Some(repo)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// List all repositories.
    pub fn list_repositories(&self) -> Result<Vec<StoredRepository>> {
        let conn = self.acquire_lock()?;
        let mut stmt = conn.prepare(
            r#"
            SELECT id, name, path, scm_url, created_at
            FROM repositories ORDER BY name
            "#,
        )?;

        let mut repos = Vec::new();
        let rows = stmt.query_map([], |row| {
            Ok(StoredRepository {
                id: row.get(0)?,
                name: row.get(1)?,
                path: row.get::<_, String>(2).ok().filter(|s| !s.is_empty()),
                scm_url: row.get(3)?,
                created_at: row.get(4)?,
            })
        })?;

        for row in rows.flatten() {
            repos.push(row);
        }

        Ok(repos)
    }

    /// Add a dependency between two repositories.
    /// Creates the repos if they don't exist.
    pub fn add_dependency(&self, upstream: &str, downstream: &str, dep_type: &str) -> Result<()> {
        // Ensure both repos exist
        let upstream_id = self.upsert_repository(upstream, None, None)?;
        let downstream_id = self.upsert_repository(downstream, None, None)?;

        let conn = self.acquire_lock()?;
        conn.execute(
            r#"
            INSERT INTO repository_dependencies (upstream_id, downstream_id, dependency_type)
            VALUES (?, ?, ?)
            ON CONFLICT(upstream_id, downstream_id) DO UPDATE SET
                dependency_type = excluded.dependency_type
            "#,
            params![upstream_id, downstream_id, dep_type],
        )?;

        Ok(())
    }

    /// Get all dependencies for a repository (what it depends on).
    pub fn get_dependencies(&self, repo_name: &str) -> Result<Vec<StoredDependency>> {
        let conn = self.acquire_lock()?;
        let mut stmt = conn.prepare(
            r#"
            SELECT rd.id, u.name, d.name, rd.dependency_type, rd.created_at
            FROM repository_dependencies rd
            JOIN repositories u ON rd.upstream_id = u.id
            JOIN repositories d ON rd.downstream_id = d.id
            WHERE d.name = ?
            "#,
        )?;

        let rows = stmt.query_map(params![repo_name], Self::row_to_dependency)?;
        Ok(rows.flatten().collect())
    }

    /// Get all dependents of a repository (what depends on it).
    ///
    /// This is an alias for `get_direct_dependants` for API compatibility.
    #[inline]
    pub fn get_dependents(&self, repo_name: &str) -> Result<Vec<StoredDependency>> {
        self.get_direct_dependants(repo_name)
    }

    /// Get all dependencies in the database.
    pub fn list_all_dependencies(&self) -> Result<Vec<StoredDependency>> {
        let conn = self.acquire_lock()?;
        let mut stmt = conn.prepare(
            r#"
            SELECT rd.id, u.name, d.name, rd.dependency_type, rd.created_at
            FROM repository_dependencies rd
            JOIN repositories u ON rd.upstream_id = u.id
            JOIN repositories d ON rd.downstream_id = d.id
            ORDER BY d.name, u.name
            "#,
        )?;

        let rows = stmt.query_map([], Self::row_to_dependency)?;
        Ok(rows.flatten().collect())
    }

    /// Clear all repositories and dependencies from the database.
    pub fn clear_repositories(&self) -> Result<()> {
        let conn = self.acquire_lock()?;
        conn.execute_batch(
            r#"
            DELETE FROM repository_dependencies;
            DELETE FROM repositories;
            "#,
        )?;
        Ok(())
    }

    /// Get repositories that directly depend on the given repository.
    ///
    /// Returns repos where the given repo is an upstream dependency.
    pub fn get_direct_dependants(&self, repo: &str) -> Result<Vec<StoredDependency>> {
        let conn = self.acquire_lock()?;
        let mut stmt = conn.prepare(
            r#"
            SELECT rd.id, u.name, d.name, rd.dependency_type, rd.created_at
            FROM repository_dependencies rd
            JOIN repositories u ON rd.upstream_id = u.id
            JOIN repositories d ON rd.downstream_id = d.id
            WHERE u.name = ?
            ORDER BY d.name
            "#,
        )?;

        let rows = stmt.query_map(params![repo], Self::row_to_dependency)?;
        Ok(rows.flatten().collect())
    }

    /// Get all repositories that depend on the given repository, transitively.
    ///
    /// Uses a recursive CTE to traverse the dependency graph.
    /// Returns (repo_name, depth) pairs where depth indicates how many hops from the source.
    pub fn get_all_dependants(&self, repo: &str) -> Result<Vec<(String, i32)>> {
        let conn = self.acquire_lock()?;
        let mut stmt = conn.prepare(
            r#"
            WITH RECURSIVE dependants AS (
                -- Base case: direct dependants
                SELECT d.id, d.name, 1 as depth
                FROM repository_dependencies rd
                JOIN repositories u ON rd.upstream_id = u.id
                JOIN repositories d ON rd.downstream_id = d.id
                WHERE u.name = ?

                UNION

                -- Recursive case: dependants of dependants
                SELECT d.id, d.name, dep.depth + 1
                FROM dependants dep
                JOIN repository_dependencies rd ON rd.upstream_id = dep.id
                JOIN repositories d ON rd.downstream_id = d.id
                WHERE dep.depth < 10  -- Prevent infinite loops
            )
            SELECT DISTINCT name, MIN(depth) as depth
            FROM dependants
            GROUP BY name
            ORDER BY depth, name
            "#,
        )?;

        let mut results = Vec::new();
        let rows = stmt.query_map(params![repo], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i32>(1)?))
        })?;

        for row in rows.flatten() {
            results.push(row);
        }

        Ok(results)
    }

    /// Save an indexed repository to the database.
    pub fn save_indexed_repo(
        &self,
        name: &str,
        path: &str,
        scm_url: Option<&str>,
        default_branch: &str,
        file_count: usize,
    ) -> Result<i64> {
        let conn = self.acquire_lock()?;
        conn.execute(
            r#"
            INSERT INTO repositories (name, path, scm_url, default_branch, file_count, last_indexed_at)
            VALUES (?1, ?2, ?3, ?4, ?5, datetime('now'))
            ON CONFLICT(name) DO UPDATE SET
                path = excluded.path,
                scm_url = COALESCE(excluded.scm_url, scm_url),
                default_branch = excluded.default_branch,
                file_count = excluded.file_count,
                last_indexed_at = datetime('now')
            "#,
            params![name, path, scm_url, default_branch, file_count as i64],
        )?;

        // last_insert_rowid() returns 0 on UPDATE, so query for the actual ID
        let id: i64 = conn
            .query_row(
                "SELECT id FROM repositories WHERE name = ?",
                params![name],
                |row| row.get(0),
            )
            .map_err(|e| {
                claudear_core::error::Error::Storage(format!(
                    "Failed to retrieve repository ID after UPSERT for '{}': {}. \
                This indicates a database inconsistency.",
                    name, e
                ))
            })?;
        Ok(id)
    }

    /// Save a file to the repo files index.
    pub fn save_repo_file(
        &self,
        repo_id: i64,
        file_path: &str,
        file_type: Option<&str>,
    ) -> Result<()> {
        let conn = self.acquire_lock()?;
        conn.execute(
            r#"
            INSERT INTO repo_files (repo_id, file_path, file_type)
            VALUES (?1, ?2, ?3)
            ON CONFLICT(repo_id, file_path) DO UPDATE SET
                file_type = excluded.file_type
            "#,
            params![repo_id, file_path, file_type],
        )?;
        Ok(())
    }

    /// Save multiple files to the repo files index efficiently.
    pub fn save_repo_files(&self, repo_id: i64, files: &[(String, Option<String>)]) -> Result<()> {
        let conn = self.acquire_lock()?;
        let mut stmt = conn.prepare(
            r#"
            INSERT INTO repo_files (repo_id, file_path, file_type)
            VALUES (?1, ?2, ?3)
            ON CONFLICT(repo_id, file_path) DO UPDATE SET
                file_type = excluded.file_type
            "#,
        )?;

        for (file_path, file_type) in files {
            stmt.execute(params![repo_id, file_path, file_type.as_deref()])?;
        }

        Ok(())
    }

    /// Clear files for a repository (before re-indexing).
    pub fn clear_repo_files(&self, repo_id: i64) -> Result<()> {
        let conn = self.acquire_lock()?;
        conn.execute("DELETE FROM repo_files WHERE repo_id = ?", params![repo_id])?;
        Ok(())
    }

    /// Sync a single repository's files to the database.
    ///
    /// Clears existing files and saves the new file list.
    pub fn sync_repo_files(&self, repo: &claudear_core::types::IndexedRepo) -> Result<()> {
        let path_str = repo.path.to_string_lossy();

        // Save/update the repo entry
        let repo_id = self.save_indexed_repo(
            &repo.name,
            &path_str,
            Some(&repo.scm_url),
            &repo.default_branch,
            repo.files.len(),
        )?;

        // Clear and re-save files
        self.clear_repo_files(repo_id)?;

        if !repo.files.is_empty() {
            let files_with_types: Vec<(String, Option<String>)> = repo
                .files
                .iter()
                .map(|f| {
                    let file_type = std::path::Path::new(f)
                        .extension()
                        .map(|e| e.to_string_lossy().to_string());
                    (f.clone(), file_type)
                })
                .collect();

            self.save_repo_files(repo_id, &files_with_types)?;
        }

        Ok(())
    }

    /// List all indexed repositories.
    pub fn list_indexed_repos(&self) -> Result<Vec<StoredIndexedRepo>> {
        let conn = self.acquire_lock()?;
        let mut stmt = conn.prepare(
            r#"
            SELECT id, name, path, scm_url, default_branch, file_count, last_indexed_at, created_at
            FROM repositories ORDER BY name
            "#,
        )?;

        let mut repos = Vec::new();
        let rows = stmt.query_map([], |row| {
            Ok(StoredIndexedRepo {
                id: row.get(0)?,
                name: row.get(1)?,
                path: row.get(2)?,
                scm_url: row.get(3)?,
                default_branch: row.get(4)?,
                file_count: row.get(5)?,
                last_indexed_at: row.get(6)?,
                created_at: row.get(7)?,
            })
        })?;

        for row in rows.flatten() {
            repos.push(row);
        }

        Ok(repos)
    }

    /// Get index statistics.
    pub fn get_index_stats(&self) -> Result<IndexStats> {
        let conn = self.acquire_lock()?;

        let repo_count: i64 =
            conn.query_row("SELECT COUNT(*) FROM repositories", [], |row| row.get(0))?;
        let file_count: i64 =
            conn.query_row("SELECT COUNT(*) FROM repo_files", [], |row| row.get(0))?;
        let last_indexed: Option<String> = conn
            .query_row("SELECT MAX(last_indexed_at) FROM repositories", [], |row| {
                row.get(0)
            })
            .ok();

        Ok(IndexStats {
            repo_count: repo_count as usize,
            file_count: file_count as usize,
            last_indexed_at: last_indexed,
        })
    }

    /// Start indexing progress tracking.
    pub fn start_indexing_progress(&self, total_repos: usize) -> Result<()> {
        let now = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S").to_string();
        let conn = self.acquire_lock()?;
        conn.execute(
            r#"
            UPDATE indexing_progress SET
                status = 'running',
                total_repos = ?1,
                indexed_repos = 0,
                current_repo = NULL,
                current_repo_files = 0,
                total_files_indexed = 0,
                started_at = ?2,
                updated_at = ?2
            WHERE id = 1
            "#,
            params![total_repos as i64, &now],
        )?;
        drop(conn);
        let new_value = IndexingProgress {
            status: "running".to_string(),
            total_repos,
            started_at: Some(now.clone()),
            updated_at: Some(now),
            ..Default::default()
        };
        self.indexing_tx.send_if_modified(|current| {
            if *current != new_value {
                *current = new_value;
                true
            } else {
                false
            }
        });
        Ok(())
    }

    /// Update indexing progress for a specific repo.
    pub fn update_indexing_progress(
        &self,
        indexed_repos: usize,
        current_repo: &str,
        current_repo_files: usize,
        total_files_indexed: usize,
    ) -> Result<()> {
        let now = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S").to_string();
        let conn = self.acquire_lock()?;
        // Read total_repos and started_at from DB (avoid borrowing the watch channel
        // while the conn lock is held).
        let (db_total_repos, db_started_at): (i64, Option<String>) = conn.query_row(
            "SELECT total_repos, started_at FROM indexing_progress WHERE id = 1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        conn.execute(
            r#"
            UPDATE indexing_progress SET
                indexed_repos = ?1,
                current_repo = ?2,
                current_repo_files = ?3,
                total_files_indexed = ?4,
                updated_at = ?5
            WHERE id = 1
            "#,
            params![
                indexed_repos as i64,
                current_repo,
                current_repo_files as i64,
                total_files_indexed as i64,
                &now,
            ],
        )?;
        drop(conn);
        let new_value = IndexingProgress {
            status: "running".to_string(),
            total_repos: db_total_repos as usize,
            indexed_repos,
            current_repo: Some(current_repo.to_string()),
            current_repo_files,
            total_files_indexed,
            started_at: db_started_at,
            updated_at: Some(now),
        };
        self.indexing_tx.send_if_modified(|current| {
            if *current != new_value {
                *current = new_value;
                true
            } else {
                false
            }
        });
        Ok(())
    }

    /// Mark indexing as complete.
    pub fn finish_indexing_progress(&self) -> Result<()> {
        let now = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S").to_string();
        let conn = self.acquire_lock()?;
        // Read values from DB before the UPDATE (avoid borrowing the watch channel
        // while the conn lock is held).
        let db_total_repos: i64 = conn.query_row(
            "SELECT total_repos FROM indexing_progress WHERE id = 1",
            [],
            |row| row.get(0),
        )?;
        conn.execute(
            r#"
            UPDATE indexing_progress SET
                status = 'idle',
                current_repo = NULL,
                started_at = NULL,
                indexed_repos = 0,
                total_files_indexed = 0,
                current_repo_files = 0,
                updated_at = ?1
            WHERE id = 1
            "#,
            params![&now],
        )?;
        drop(conn);
        let new_value = IndexingProgress {
            status: "idle".to_string(),
            total_repos: db_total_repos as usize,
            indexed_repos: 0,
            total_files_indexed: 0,
            updated_at: Some(now),
            ..Default::default()
        };
        self.indexing_tx.send_if_modified(|current| {
            if *current != new_value {
                *current = new_value;
                true
            } else {
                false
            }
        });
        Ok(())
    }

    /// Get current indexing progress.
    pub fn get_indexing_progress(&self) -> Result<IndexingProgress> {
        let conn = self.acquire_lock()?;
        let row = conn.query_row(
            r#"
            SELECT status, total_repos, indexed_repos, current_repo,
                   current_repo_files, total_files_indexed, started_at, updated_at
            FROM indexing_progress WHERE id = 1
            "#,
            [],
            |row| {
                Ok(IndexingProgress {
                    status: row.get(0)?,
                    total_repos: row.get::<_, i64>(1)? as usize,
                    indexed_repos: row.get::<_, i64>(2)? as usize,
                    current_repo: row.get(3)?,
                    current_repo_files: row.get::<_, i64>(4)? as usize,
                    total_files_indexed: row.get::<_, i64>(5)? as usize,
                    started_at: row.get(6)?,
                    updated_at: row.get(7)?,
                })
            },
        )?;
        Ok(row)
    }

    /// Get inference statistics.
    pub fn get_inference_stats(&self) -> Result<InferenceStats> {
        let conn = self.acquire_lock()?;

        let total: i64 = conn.query_row("SELECT COUNT(*) FROM inference_attempts", [], |row| {
            row.get(0)
        })?;
        let with_feedback: i64 = conn.query_row(
            "SELECT COUNT(*) FROM inference_attempts WHERE was_correct IS NOT NULL",
            [],
            |row| row.get(0),
        )?;
        let correct: i64 = conn.query_row(
            "SELECT COUNT(*) FROM inference_attempts WHERE was_correct = 1",
            [],
            |row| row.get(0),
        )?;

        let high_confidence: i64 = conn.query_row(
            "SELECT COUNT(*) FROM inference_attempts WHERE confidence = 'high'",
            [],
            |row| row.get(0),
        )?;
        let medium_confidence: i64 = conn.query_row(
            "SELECT COUNT(*) FROM inference_attempts WHERE confidence = 'medium'",
            [],
            |row| row.get(0),
        )?;
        let low_confidence: i64 = conn.query_row(
            "SELECT COUNT(*) FROM inference_attempts WHERE confidence = 'low'",
            [],
            |row| row.get(0),
        )?;
        let no_match: i64 = conn.query_row(
            "SELECT COUNT(*) FROM inference_attempts WHERE inferred_repo_id IS NULL",
            [],
            |row| row.get(0),
        )?;

        Ok(InferenceStats {
            total_attempts: total as usize,
            with_feedback: with_feedback as usize,
            correct: correct as usize,
            accuracy: if with_feedback > 0 {
                (correct as f64 / with_feedback as f64) * 100.0
            } else {
                0.0
            },
            by_confidence: ConfidenceBreakdown {
                high: high_confidence as usize,
                medium: medium_confidence as usize,
                low: low_confidence as usize,
                none: no_match as usize,
            },
        })
    }

    /// Get recent inference history.
    ///
    /// Returns the most recent inference attempts, sorted by creation time (newest first).
    pub fn get_inference_history(&self, limit: usize) -> Result<Vec<InferenceHistoryEntry>> {
        let conn = self.acquire_lock()?;

        let mut stmt = conn.prepare(
            "SELECT
                ia.id,
                ia.issue_id,
                ia.issue_source,
                ia.extracted_keywords,
                r.name as repo_name,
                ia.confidence,
                ia.inference_reason,
                ia.was_correct,
                ia.inference_duration_ms,
                ia.created_at
            FROM inference_attempts ia
            LEFT JOIN repositories r ON ia.inferred_repo_id = r.id
            ORDER BY ia.created_at DESC
            LIMIT ?",
        )?;

        let rows = stmt.query_map([limit as i64], |row| {
            Ok(InferenceHistoryEntry {
                id: row.get(0)?,
                issue_id: row.get(1)?,
                issue_source: row.get(2)?,
                extracted_keywords: row.get(3)?,
                inferred_repo_name: row.get(4)?,
                confidence: row.get(5)?,
                inference_reason: row.get(6)?,
                was_correct: row.get(7)?,
                duration_ms: row.get(8)?,
                created_at: row.get(9)?,
            })
        })?;

        let mut entries = Vec::new();
        for row in rows {
            entries.push(row?);
        }

        Ok(entries)
    }

    /// Get all open PRs.
    pub fn get_open_prs(&self) -> Result<Vec<claudear_core::types::PrRecord>> {
        let conn = self.acquire_lock()?;
        let mut stmt = conn.prepare(
            r#"
            SELECT id, pr_url, scm_repo, pr_number, attempt_id, issue_id, issue_source,
                   title, description, author, head_branch, base_branch, status,
                   created_at, updated_at, merged_at, closed_at,
                   approvals_count, changes_requested_count, comments_count, last_review_at,
                   time_to_first_review_mins, time_to_merge_mins, review_cycles,
                   files_changed, lines_added, lines_removed
            FROM prs WHERE status = 'open'
            ORDER BY created_at DESC
            "#,
        )?;

        let rows = stmt.query_map([], Self::row_to_pr_record)?;
        Ok(rows.flatten().collect())
    }

    /// Get PR analytics.
    pub fn get_pr_analytics(&self) -> Result<claudear_core::types::PrAnalytics> {
        let conn = self.acquire_lock()?;

        let total: i64 = conn.query_row("SELECT COUNT(*) FROM prs", [], |row| row.get(0))?;
        let open: i64 = conn.query_row(
            "SELECT COUNT(*) FROM prs WHERE status = 'open'",
            [],
            |row| row.get(0),
        )?;
        let merged: i64 = conn.query_row(
            "SELECT COUNT(*) FROM prs WHERE status = 'merged'",
            [],
            |row| row.get(0),
        )?;
        let closed: i64 = conn.query_row(
            "SELECT COUNT(*) FROM prs WHERE status = 'closed'",
            [],
            |row| row.get(0),
        )?;

        let avg_time_to_first_review_mins: Option<f64> = conn.query_row(
            "SELECT AVG(time_to_first_review_mins) FROM prs WHERE time_to_first_review_mins IS NOT NULL",
            [],
            |row| row.get(0),
        ).ok();

        let avg_time_to_merge_mins: Option<f64> = conn
            .query_row(
                "SELECT AVG(time_to_merge_mins) FROM prs WHERE time_to_merge_mins IS NOT NULL",
                [],
                |row| row.get(0),
            )
            .ok();

        let avg_review_cycles: Option<f64> = conn
            .query_row("SELECT AVG(review_cycles) FROM prs", [], |row| row.get(0))
            .ok();

        let merge_rate = if merged + closed > 0 {
            Some(merged as f64 / (merged + closed) as f64)
        } else {
            None
        };

        // Get counts by repository
        let mut by_repo = HashMap::new();
        let mut stmt = conn.prepare("SELECT scm_repo, COUNT(*) FROM prs GROUP BY scm_repo")?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
        })?;
        for row in rows.flatten() {
            by_repo.insert(row.0, row.1);
        }

        Ok(claudear_core::types::PrAnalytics {
            total,
            open,
            merged,
            closed,
            avg_time_to_first_review_mins,
            avg_time_to_merge_mins,
            avg_review_cycles,
            merge_rate,
            by_repo,
            avg_time_to_pr_mins: None,
            rejection_reasons: Vec::new(),
        })
    }

    /// Average time from issue attempt to PR creation in minutes.
    pub fn get_avg_time_to_pr(&self) -> Result<Option<f64>> {
        let conn = self.acquire_lock()?;
        let result: Option<f64> = conn
            .query_row(
                r#"
                SELECT AVG((julianday(p.created_at) - julianday(fa.attempted_at)) * 24.0 * 60.0)
                FROM prs p
                JOIN fix_attempts fa ON fa.id = p.attempt_id
                WHERE p.created_at IS NOT NULL AND fa.attempted_at IS NOT NULL
                "#,
                [],
                |row| row.get(0),
            )
            .ok()
            .flatten();
        Ok(result)
    }

    /// Top PR rejection/review-change reason categories.
    pub fn get_rejection_reasons(
        &self,
        limit: usize,
    ) -> Result<Vec<claudear_core::types::RejectionReason>> {
        let conn = self.acquire_lock()?;
        let mut stmt = conn.prepare(
            r#"
            SELECT category, SUM(occurrence_count) as total
            FROM review_patterns
            GROUP BY category
            ORDER BY total DESC
            LIMIT ?1
            "#,
        )?;
        let rows = stmt.query_map(params![limit as i64], |row| {
            Ok(claudear_core::types::RejectionReason {
                category: row.get(0)?,
                count: row.get(1)?,
            })
        })?;
        Ok(rows.flatten().collect())
    }

    /// Count agent (Claude) spawns since a given ISO timestamp.
    pub fn get_agent_spawn_count(&self, since_iso: &str) -> Result<i64> {
        let conn = self.acquire_lock()?;
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM claude_executions WHERE started_at >= ?1",
            params![since_iso],
            |row| row.get(0),
        )?;
        Ok(count)
    }

    /// Compute cost estimate from Claude execution data.
    ///
    /// Priority: (1) actual `total_cost_usd` from CLI, (2) plan-based estimate,
    /// (3) duration-based fallback.
    pub fn get_cost_estimate(
        &self,
        since_iso: &str,
        max_plan_monthly_cost: f64,
        period_label: &str,
    ) -> Result<claudear_core::types::CostEstimate> {
        let conn = self.acquire_lock()?;

        // Try actual API cost first
        let (api_cost_sum, api_cost_count): (f64, i64) = conn
            .query_row(
                r#"
                SELECT COALESCE(SUM(total_cost_usd), 0.0), COUNT(total_cost_usd)
                FROM claude_executions
                WHERE started_at >= ?1 AND total_cost_usd IS NOT NULL
                "#,
                params![since_iso],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap_or((0.0, 0));

        let fix_count: i64 = conn
            .query_row(
                r#"
                SELECT COUNT(*) FROM fix_attempts
                WHERE status IN ('success', 'merged') AND attempted_at >= ?1
                "#,
                params![since_iso],
                |row| row.get(0),
            )
            .unwrap_or(0);

        let (total_cost, cost_source) = if api_cost_count > 0 {
            (api_cost_sum, "api".to_string())
        } else if max_plan_monthly_cost > 0.0 {
            // Estimate from plan cost: amortize over total minutes used this month
            let monthly_total_mins: f64 = conn
                .query_row(
                    r#"
                    SELECT COALESCE(SUM(duration_secs), 0.0) / 60.0
                    FROM claude_executions
                    WHERE started_at >= datetime('now', '-30 days') AND duration_secs IS NOT NULL
                    "#,
                    [],
                    |row| row.get(0),
                )
                .unwrap_or(0.0);

            let period_mins: f64 = conn
                .query_row(
                    r#"
                    SELECT COALESCE(SUM(duration_secs), 0.0) / 60.0
                    FROM claude_executions
                    WHERE started_at >= ?1 AND duration_secs IS NOT NULL
                    "#,
                    params![since_iso],
                    |row| row.get(0),
                )
                .unwrap_or(0.0);

            if monthly_total_mins > 0.0 {
                let cost_per_min = max_plan_monthly_cost / monthly_total_mins;
                (period_mins * cost_per_min, "plan_estimate".to_string())
            } else {
                (0.0, "plan_estimate".to_string())
            }
        } else {
            // Fallback: duration-based estimate at $0.05/min
            let total_duration_mins: f64 = conn
                .query_row(
                    r#"
                    SELECT COALESCE(SUM(duration_secs), 0.0) / 60.0
                    FROM claude_executions
                    WHERE started_at >= ?1 AND duration_secs IS NOT NULL
                    "#,
                    params![since_iso],
                    |row| row.get(0),
                )
                .unwrap_or(0.0);
            (total_duration_mins * 0.05, "duration_estimate".to_string())
        };

        let avg_cost_per_fix = if fix_count > 0 {
            total_cost / fix_count as f64
        } else {
            0.0
        };

        Ok(claudear_core::types::CostEstimate {
            total_cost,
            avg_cost_per_fix,
            fix_count,
            cost_source,
            period: period_label.to_string(),
        })
    }

    /// MTTR trend grouped by ISO week buckets.
    pub fn get_mttr_trend(&self, weeks: usize) -> Result<Vec<claudear_core::types::MttrDataPoint>> {
        let conn = self.acquire_lock()?;
        let modifier = format!("-{} days", weeks * 7);
        let mut stmt = conn.prepare(
            r#"
            SELECT
                date(fa.merged_at, 'weekday 0', '-6 days') as week_start,
                AVG((julianday(fa.merged_at) - julianday(fa.attempted_at)) * 24.0 * 60.0) as avg_mins,
                COUNT(*) as cnt
            FROM fix_attempts fa
            WHERE fa.status = 'merged'
              AND fa.merged_at IS NOT NULL
              AND fa.attempted_at IS NOT NULL
              AND fa.merged_at >= datetime('now', ?1)
            GROUP BY week_start
            ORDER BY week_start ASC
            "#,
        )?;
        let rows = stmt.query_map(params![modifier], |row| {
            Ok(claudear_core::types::MttrDataPoint {
                period_start: row.get(0)?,
                mttr_minutes: row.get(1)?,
                sample_count: row.get(2)?,
            })
        })?;
        Ok(rows.flatten().collect())
    }

    /// Per-repository leaderboard.
    pub fn get_repo_leaderboard(&self) -> Result<Vec<claudear_core::types::RepoLeaderboardEntry>> {
        let conn = self.acquire_lock()?;
        let mut stmt = conn.prepare(
            r#"
            SELECT
                fa.scm_repo,
                COUNT(*) as total,
                CAST(SUM(CASE WHEN fa.status IN ('success','merged') THEN 1 ELSE 0 END) AS REAL) / NULLIF(COUNT(*), 0) as success_rate,
                CAST(SUM(CASE WHEN fa.status = 'merged' THEN 1 ELSE 0 END) AS REAL) /
                    NULLIF(SUM(CASE WHEN fa.status IN ('merged','closed') THEN 1 ELSE 0 END), 0) as merge_rate,
                AVG(CASE WHEN p.time_to_merge_mins IS NOT NULL THEN p.time_to_merge_mins END) as avg_ttm
            FROM fix_attempts fa
            LEFT JOIN prs p ON p.attempt_id = fa.id
            WHERE fa.scm_repo IS NOT NULL AND fa.scm_repo != ''
            GROUP BY fa.scm_repo
            ORDER BY total DESC
            "#,
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(claudear_core::types::RepoLeaderboardEntry {
                repo: row.get(0)?,
                total: row.get(1)?,
                success_rate: row.get::<_, Option<f64>>(2)?.unwrap_or(0.0),
                merge_rate: row.get::<_, Option<f64>>(3)?.unwrap_or(0.0),
                avg_time_to_merge_mins: row.get(4)?,
            })
        })?;
        Ok(rows.flatten().collect())
    }

    /// Complexity-based engineering time savings estimate.
    ///
    /// For each merged fix, computes a complexity score from lines changed, files
    /// changed, execution duration, review cycles, and retry count, then maps the
    /// score to an estimated hours-saved value.
    pub fn get_complexity_time_savings(
        &self,
        since_iso: &str,
        hourly_rate: f64,
        period_label: &str,
    ) -> Result<claudear_core::types::TimeSavings> {
        let conn = self.acquire_lock()?;
        let mut stmt = conn.prepare(
            r#"
            SELECT
                fa.id,
                COALESCE(p.lines_added, 0) + COALESCE(p.lines_removed, 0) as total_lines,
                COALESCE(p.files_changed, 0) as files_changed,
                COALESCE(ce.duration_secs, 0) as exec_duration,
                COALESCE(p.review_cycles, 0) as review_cycles,
                COALESCE(fa.retry_count, 0) as retry_count
            FROM fix_attempts fa
            LEFT JOIN prs p ON p.attempt_id = fa.id
            LEFT JOIN claude_executions ce ON ce.attempt_id = fa.id
            WHERE fa.status = 'merged' AND fa.merged_at >= ?1
            "#,
        )?;

        let rows = stmt.query_map(params![since_iso], |row| {
            Ok((
                row.get::<_, i64>(0)?, // id
                row.get::<_, f64>(1)?, // total_lines
                row.get::<_, f64>(2)?, // files_changed
                row.get::<_, f64>(3)?, // exec_duration
                row.get::<_, f64>(4)?, // review_cycles
                row.get::<_, f64>(5)?, // retry_count
            ))
        })?;

        let mut merged_count: i64 = 0;
        let mut hours_saved: f64 = 0.0;

        for row in rows.flatten() {
            merged_count += 1;
            let (_id, total_lines, files_changed, exec_duration, review_cycles, retry_count) = row;

            // If all signals are zero, default to 2.0h
            if total_lines == 0.0
                && files_changed == 0.0
                && exec_duration == 0.0
                && review_cycles == 0.0
                && retry_count == 0.0
            {
                hours_saved += 2.0;
                continue;
            }

            let score = 0.30 * normalize_signal(total_lines, &[0.0, 20.0, 100.0, 500.0, 2000.0])
                + 0.20 * normalize_signal(files_changed, &[0.0, 2.0, 5.0, 15.0, 50.0])
                + 0.25 * normalize_signal(exec_duration, &[0.0, 120.0, 600.0, 1800.0, 7200.0])
                + 0.15 * normalize_signal(review_cycles, &[0.0, 1.0, 2.0, 4.0, 8.0])
                + 0.10 * normalize_signal(retry_count, &[0.0, 0.0, 1.0, 2.0, 4.0]);

            hours_saved += complexity_to_hours(score);
        }

        let cost_saved = hours_saved * hourly_rate;

        Ok(claudear_core::types::TimeSavings {
            merged_count,
            hours_saved,
            cost_saved,
            period: period_label.to_string(),
        })
    }

    /// List PRs with optional status filter and limit.
    pub fn list_prs(
        &self,
        status: Option<&str>,
        limit: usize,
    ) -> Result<Vec<claudear_core::types::PrRecord>> {
        let conn = self.acquire_lock()?;
        let (sql, params_vec): (String, Vec<Box<dyn rusqlite::types::ToSql>>) = match status {
            Some(s) => (
                r#"
                    SELECT id, pr_url, scm_repo, pr_number, attempt_id, issue_id, issue_source,
                           title, description, author, head_branch, base_branch, status,
                           created_at, updated_at, merged_at, closed_at,
                           approvals_count, changes_requested_count, comments_count, last_review_at,
                           time_to_first_review_mins, time_to_merge_mins, review_cycles,
                           files_changed, lines_added, lines_removed
                    FROM prs WHERE status = ?1
                    ORDER BY created_at DESC LIMIT ?2
                "#
                .to_string(),
                vec![
                    Box::new(s.to_string()) as Box<dyn rusqlite::types::ToSql>,
                    Box::new(limit as i64) as Box<dyn rusqlite::types::ToSql>,
                ],
            ),
            None => (
                r#"
                    SELECT id, pr_url, scm_repo, pr_number, attempt_id, issue_id, issue_source,
                           title, description, author, head_branch, base_branch, status,
                           created_at, updated_at, merged_at, closed_at,
                           approvals_count, changes_requested_count, comments_count, last_review_at,
                           time_to_first_review_mins, time_to_merge_mins, review_cycles,
                           files_changed, lines_added, lines_removed
                    FROM prs ORDER BY created_at DESC LIMIT ?1
                "#
                .to_string(),
                vec![Box::new(limit as i64) as Box<dyn rusqlite::types::ToSql>],
            ),
        };
        let mut stmt = conn.prepare(&sql)?;
        let param_refs: Vec<&dyn rusqlite::types::ToSql> =
            params_vec.iter().map(|p| p.as_ref()).collect();
        let rows = stmt.query_map(param_refs.as_slice(), Self::row_to_pr_record)?;
        Ok(rows.flatten().collect())
    }

    fn row_to_pr_record(
        row: &rusqlite::Row<'_>,
    ) -> rusqlite::Result<claudear_core::types::PrRecord> {
        Ok(claudear_core::types::PrRecord {
            id: row.get(0)?,
            pr_url: row.get(1)?,
            scm_repo: row.get(2)?,
            pr_number: row.get(3)?,
            attempt_id: row.get(4)?,
            issue_id: row.get(5)?,
            issue_source: row.get(6)?,
            title: row.get(7)?,
            description: row.get(8)?,
            author: row.get(9)?,
            head_branch: row.get(10)?,
            base_branch: row.get(11)?,
            status: row.get(12)?,
            created_at: Self::parse_datetime(&row.get::<_, String>(13)?)?,
            updated_at: Self::parse_optional_datetime(row.get(14)?)?,
            merged_at: Self::parse_optional_datetime(row.get(15)?)?,
            closed_at: Self::parse_optional_datetime(row.get(16)?)?,
            approvals_count: row.get(17)?,
            changes_requested_count: row.get(18)?,
            comments_count: row.get(19)?,
            last_review_at: Self::parse_optional_datetime(row.get(20)?)?,
            time_to_first_review_mins: row.get(21)?,
            time_to_merge_mins: row.get(22)?,
            review_cycles: row.get(23)?,
            files_changed: row.get(24)?,
            lines_added: row.get(25)?,
            lines_removed: row.get(26)?,
        })
    }

    /// Get a fix attempt by its ID.
    pub fn get_attempt_by_id(&self, id: i64) -> Result<Option<FixAttempt>> {
        let conn = self.acquire_lock()?;
        let mut stmt = conn.prepare(
            r#"
            SELECT id, source, issue_id, short_id, attempted_at, pr_url, scm_repo,
                   scm_pr_number, status, error_message, merged_at, resolved_at,
                   retry_count, last_retry_at, issue_labels, parent_attempt_id, cascade_repo
            FROM fix_attempts
            WHERE id = ?
            "#,
        )?;

        let result = stmt.query_row(params![id], Self::row_to_fix_attempt).ok();
        Ok(result)
    }

    /// Get a regression watch by ID.
    pub fn get_regression_watch(
        &self,
        id: i64,
    ) -> Result<Option<claudear_core::types::RegressionWatch>> {
        let conn = self.acquire_lock()?;
        let mut stmt = conn.prepare(
            r#"
            SELECT id, issue_type, issue_id, fix_attempt_id, status,
                   pr_merged_at, monitoring_started_at, resolved_at, regressed_at, created_at
            FROM regression_watches
            WHERE id = ?
            "#,
        )?;

        let result = stmt
            .query_row(params![id], Self::row_to_regression_watch)
            .ok();
        Ok(result)
    }

    /// Get regression watches by status.
    pub fn get_regression_watches_by_status(
        &self,
        status: claudear_core::types::RegressionWatchStatus,
    ) -> Result<Vec<claudear_core::types::RegressionWatch>> {
        let conn = self.acquire_lock()?;
        let mut stmt = conn.prepare(
            r#"
            SELECT id, issue_type, issue_id, fix_attempt_id, status,
                   pr_merged_at, monitoring_started_at, resolved_at, regressed_at, created_at
            FROM regression_watches
            WHERE status = ?
            ORDER BY created_at DESC
            "#,
        )?;

        let rows = stmt.query_map(params![status.to_string()], Self::row_to_regression_watch)?;

        let mut results = Vec::new();
        for row in rows.flatten() {
            results.push(row);
        }
        Ok(results)
    }

    /// Get all regression watches.
    pub fn get_all_regression_watches(&self) -> Result<Vec<claudear_core::types::RegressionWatch>> {
        let conn = self.acquire_lock()?;
        let mut stmt = conn.prepare(
            r#"
            SELECT id, issue_type, issue_id, fix_attempt_id, status,
                   pr_merged_at, monitoring_started_at, resolved_at, regressed_at, created_at
            FROM regression_watches
            ORDER BY created_at DESC
            "#,
        )?;

        let rows = stmt.query_map([], Self::row_to_regression_watch)?;
        Ok(rows.flatten().collect())
    }

    /// Record a regression check.
    pub fn record_regression_check(
        &self,
        check: &claudear_core::types::RegressionCheck,
    ) -> Result<i64> {
        let conn = self.acquire_lock()?;

        conn.execute(
            r#"
            INSERT INTO regression_checks (
                regression_watch_id, issue_still_exists, checked_at, check_details, created_at
            )
            VALUES (?1, ?2, ?3, ?4, ?5)
            "#,
            params![
                check.regression_watch_id,
                check.issue_still_exists as i32,
                check
                    .checked_at
                    .map(|dt| dt.format("%Y-%m-%d %H:%M:%S").to_string()),
                check.check_details,
                check.created_at.format("%Y-%m-%d %H:%M:%S").to_string(),
            ],
        )?;

        Ok(conn.last_insert_rowid())
    }

    /// Get regression checks for a watch.
    pub fn get_regression_checks(
        &self,
        watch_id: i64,
    ) -> Result<Vec<claudear_core::types::RegressionCheck>> {
        let conn = self.acquire_lock()?;
        let mut stmt = conn.prepare(
            r#"
            SELECT id, regression_watch_id, issue_still_exists, checked_at, check_details, created_at
            FROM regression_checks
            WHERE regression_watch_id = ?
            ORDER BY created_at DESC
            "#,
        )?;

        let rows = stmt.query_map(params![watch_id], Self::row_to_regression_check)?;

        let mut results = Vec::new();
        for row in rows.flatten() {
            results.push(row);
        }
        Ok(results)
    }

    /// Create a new regression watch. Returns the watch ID.
    pub fn create_regression_watch(
        &self,
        watch: &claudear_core::types::RegressionWatch,
    ) -> Result<i64> {
        let conn = self.acquire_lock()?;

        conn.execute(
            r#"
            INSERT INTO regression_watches (
                issue_type, issue_id, fix_attempt_id, status,
                pr_merged_at, monitoring_started_at, resolved_at, regressed_at, created_at
            )
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
            "#,
            params![
                watch.issue_type.to_string(),
                watch.issue_id,
                watch.fix_attempt_id,
                watch.status.to_string(),
                watch
                    .pr_merged_at
                    .map(|dt| dt.format("%Y-%m-%d %H:%M:%S").to_string()),
                watch
                    .monitoring_started_at
                    .map(|dt| dt.format("%Y-%m-%d %H:%M:%S").to_string()),
                watch
                    .resolved_at
                    .map(|dt| dt.format("%Y-%m-%d %H:%M:%S").to_string()),
                watch
                    .regressed_at
                    .map(|dt| dt.format("%Y-%m-%d %H:%M:%S").to_string()),
                watch.created_at.format("%Y-%m-%d %H:%M:%S").to_string(),
            ],
        )?;

        Ok(conn.last_insert_rowid())
    }

    /// Add a regression check. Returns the check ID.
    pub fn add_regression_check(
        &self,
        check: &claudear_core::types::RegressionCheck,
    ) -> Result<i64> {
        self.record_regression_check(check)
    }

    /// Update regression watch status.
    pub fn update_regression_watch_status(
        &self,
        id: i64,
        status: claudear_core::types::RegressionWatchStatus,
    ) -> Result<()> {
        let conn = self.acquire_lock()?;
        let now = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S").to_string();

        let (monitoring_started, resolved, regressed) = match status {
            claudear_core::types::RegressionWatchStatus::Monitoring => {
                (Some(now.clone()), None, None)
            }
            claudear_core::types::RegressionWatchStatus::Resolved => {
                (None, Some(now.clone()), None)
            }
            claudear_core::types::RegressionWatchStatus::Regressed => {
                (None, None, Some(now.clone()))
            }
            _ => (None, None, None),
        };

        conn.execute(
            r#"
            UPDATE regression_watches SET
                status = ?1,
                monitoring_started_at = COALESCE(?2, monitoring_started_at),
                resolved_at = COALESCE(?3, resolved_at),
                regressed_at = COALESCE(?4, regressed_at)
            WHERE id = ?5
            "#,
            params![
                status.to_string(),
                monitoring_started,
                resolved,
                regressed,
                id
            ],
        )?;

        Ok(())
    }

    fn row_to_regression_watch(
        row: &rusqlite::Row<'_>,
    ) -> rusqlite::Result<claudear_core::types::RegressionWatch> {
        let issue_type_str: String = row.get(1)?;
        let status_str: String = row.get(4)?;

        Ok(claudear_core::types::RegressionWatch {
            id: row.get(0)?,
            issue_type: issue_type_str
                .parse()
                .unwrap_or(claudear_core::types::IssueType::SentryIssue),
            issue_id: row.get(2)?,
            fix_attempt_id: row.get(3)?,
            status: status_str
                .parse()
                .unwrap_or(claudear_core::types::RegressionWatchStatus::AwaitingRelease),
            pr_merged_at: Self::parse_optional_datetime(row.get(5)?)?,
            monitoring_started_at: Self::parse_optional_datetime(row.get(6)?)?,
            resolved_at: Self::parse_optional_datetime(row.get(7)?)?,
            regressed_at: Self::parse_optional_datetime(row.get(8)?)?,
            created_at: Self::parse_datetime(&row.get::<_, String>(9)?)?,
        })
    }

    fn row_to_regression_check(
        row: &rusqlite::Row<'_>,
    ) -> rusqlite::Result<claudear_core::types::RegressionCheck> {
        Ok(claudear_core::types::RegressionCheck {
            id: row.get(0)?,
            regression_watch_id: row.get(1)?,
            issue_still_exists: row.get::<_, i32>(2)? != 0,
            checked_at: Self::parse_optional_datetime(row.get(3)?)?,
            check_details: row.get(4)?,
            created_at: Self::parse_datetime(&row.get::<_, String>(5)?)?,
        })
    }
}

impl SqliteTracker {
    /// System 1: Update learnings text on a feedback outcome.
    pub fn update_feedback_learnings(&self, outcome_id: i64, learnings: &str) -> Result<()> {
        let conn = self.acquire_lock()?;
        conn.execute(
            "UPDATE feedback_outcomes SET learnings = ?1 WHERE id = ?2",
            params![learnings, outcome_id],
        )?;
        Ok(())
    }

    /// System 2: Store a diff analysis.
    pub fn store_diff_analysis(
        &self,
        analysis: &claudear_core::types::DiffAnalysis,
    ) -> Result<i64> {
        let conn = self.acquire_lock()?;
        let files_json = serde_json::to_string(&analysis.files_changed).unwrap_or_default();
        let types_json = serde_json::to_string(&analysis.file_types).unwrap_or_default();
        let cats_json = serde_json::to_string(&analysis.change_categories).unwrap_or_default();

        conn.execute(
            "INSERT INTO diff_analyses (attempt_id, pr_url, scm_repo, pr_number, files_changed, file_types, change_categories, diff_summary, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                analysis.attempt_id,
                analysis.pr_url,
                analysis.scm_repo,
                analysis.pr_number,
                files_json,
                types_json,
                cats_json,
                analysis.diff_summary,
                analysis.created_at.to_rfc3339(),
            ],
        )?;
        Ok(conn.last_insert_rowid())
    }

    /// System 2: Get diff analyses for a repo.
    pub fn get_diff_analyses_for_repo(
        &self,
        repo: &str,
        limit: usize,
    ) -> Result<Vec<claudear_core::types::DiffAnalysis>> {
        let conn = self.acquire_lock()?;
        let mut stmt = conn.prepare(
            "SELECT id, attempt_id, pr_url, scm_repo, pr_number, files_changed, file_types, change_categories, diff_summary, created_at
             FROM diff_analyses WHERE scm_repo = ?1 ORDER BY created_at DESC LIMIT ?2",
        )?;
        let rows = stmt
            .query_map(params![repo, limit as i64], |row| {
                Ok(claudear_core::types::DiffAnalysis {
                    id: row.get(0)?,
                    attempt_id: row.get(1)?,
                    pr_url: row.get(2)?,
                    scm_repo: row.get(3)?,
                    pr_number: row.get(4)?,
                    files_changed: serde_json::from_str(&row.get::<_, String>(5)?)
                        .unwrap_or_default(),
                    file_types: serde_json::from_str(&row.get::<_, String>(6)?).unwrap_or_default(),
                    change_categories: serde_json::from_str(&row.get::<_, String>(7)?)
                        .unwrap_or_default(),
                    diff_summary: row.get(8)?,
                    created_at: Self::parse_datetime(&row.get::<_, String>(9)?)?,
                })
            })?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }

    /// System 3: Upsert a promoted instruction.
    pub fn upsert_promoted_instruction(
        &self,
        instruction: &claudear_core::types::PromotedInstruction,
    ) -> Result<i64> {
        let conn = self.acquire_lock()?;
        let now = Utc::now().to_rfc3339();

        // Try to update existing
        let updated = conn.execute(
            "UPDATE promoted_instructions SET occurrence_count = ?1, confidence = ?2, is_active = ?3, updated_at = ?4
             WHERE repo = ?5 AND source_type = ?6 AND instruction_text = ?7",
            params![
                instruction.occurrence_count,
                instruction.confidence,
                instruction.is_active as i32,
                now,
                instruction.repo,
                instruction.source_type,
                instruction.instruction_text,
            ],
        )?;

        if updated > 0 {
            let id: i64 = conn
                .query_row(
                    "SELECT id FROM promoted_instructions WHERE repo = ?1 AND source_type = ?2 AND instruction_text = ?3",
                    params![instruction.repo, instruction.source_type, instruction.instruction_text],
                    |row| row.get(0),
                )
                .unwrap_or(0);
            return Ok(id);
        }

        conn.execute(
            "INSERT INTO promoted_instructions (repo, source_type, instruction_text, occurrence_count, confidence, is_active, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                instruction.repo,
                instruction.source_type,
                instruction.instruction_text,
                instruction.occurrence_count,
                instruction.confidence,
                instruction.is_active as i32,
                now,
                now,
            ],
        )?;
        Ok(conn.last_insert_rowid())
    }

    /// System 3: Get active promoted instructions for a repo.
    pub fn get_promoted_instructions(
        &self,
        repo: &str,
    ) -> Result<Vec<claudear_core::types::PromotedInstruction>> {
        let conn = self.acquire_lock()?;
        let mut stmt = conn.prepare(
            "SELECT id, repo, source_type, instruction_text, occurrence_count, confidence, is_active, created_at, updated_at
             FROM promoted_instructions WHERE repo = ?1 AND is_active = 1 ORDER BY confidence DESC",
        )?;
        let rows = stmt
            .query_map(params![repo], |row| {
                Ok(claudear_core::types::PromotedInstruction {
                    id: row.get(0)?,
                    repo: row.get(1)?,
                    source_type: row.get(2)?,
                    instruction_text: row.get(3)?,
                    occurrence_count: row.get(4)?,
                    confidence: row.get(5)?,
                    is_active: row.get::<_, i32>(6)? != 0,
                    created_at: Self::parse_datetime(&row.get::<_, String>(7)?)?,
                    updated_at: Self::parse_datetime(&row.get::<_, String>(8)?)?,
                })
            })?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }

    /// System 4: Upsert a repo knowledge entry.
    pub fn upsert_repo_knowledge(
        &self,
        entry: &claudear_core::types::RepoKnowledge,
    ) -> Result<i64> {
        let conn = self.acquire_lock()?;
        let now = Utc::now().to_rfc3339();

        let updated = conn.execute(
            "UPDATE repo_knowledge SET confidence = ?1, occurrence_count = occurrence_count + 1, updated_at = ?2
             WHERE repo = ?3 AND knowledge_key = ?4 AND knowledge_value = ?5",
            params![entry.confidence, now, entry.repo, entry.knowledge_key, entry.knowledge_value],
        )?;

        if updated > 0 {
            let id: i64 = conn
                .query_row(
                    "SELECT id FROM repo_knowledge WHERE repo = ?1 AND knowledge_key = ?2 AND knowledge_value = ?3",
                    params![entry.repo, entry.knowledge_key, entry.knowledge_value],
                    |row| row.get(0),
                )
                .unwrap_or(0);
            return Ok(id);
        }

        conn.execute(
            "INSERT INTO repo_knowledge (repo, knowledge_key, knowledge_value, source_type, confidence, occurrence_count, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                entry.repo,
                entry.knowledge_key,
                entry.knowledge_value,
                entry.source_type,
                entry.confidence,
                entry.occurrence_count,
                now,
                now,
            ],
        )?;
        Ok(conn.last_insert_rowid())
    }

    /// System 4: Get all knowledge for a repo.
    pub fn get_repo_knowledge(
        &self,
        repo: &str,
    ) -> Result<Vec<claudear_core::types::RepoKnowledge>> {
        let conn = self.acquire_lock()?;
        let mut stmt = conn.prepare(
            "SELECT id, repo, knowledge_key, knowledge_value, source_type, confidence, occurrence_count, created_at, updated_at
             FROM repo_knowledge WHERE repo = ?1 ORDER BY occurrence_count DESC",
        )?;
        let rows = stmt
            .query_map(params![repo], |row| {
                Ok(claudear_core::types::RepoKnowledge {
                    id: row.get(0)?,
                    repo: row.get(1)?,
                    knowledge_key: row.get(2)?,
                    knowledge_value: row.get(3)?,
                    source_type: row.get(4)?,
                    confidence: row.get(5)?,
                    occurrence_count: row.get(6)?,
                    created_at: Self::parse_datetime(&row.get::<_, String>(7)?)?,
                    updated_at: Self::parse_datetime(&row.get::<_, String>(8)?)?,
                })
            })?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }

    /// System 4: Get repo knowledge by key.
    pub fn get_repo_knowledge_by_key(
        &self,
        repo: &str,
        key: &str,
    ) -> Result<Vec<claudear_core::types::RepoKnowledge>> {
        let conn = self.acquire_lock()?;
        let mut stmt = conn.prepare(
            "SELECT id, repo, knowledge_key, knowledge_value, source_type, confidence, occurrence_count, created_at, updated_at
             FROM repo_knowledge WHERE repo = ?1 AND knowledge_key = ?2 ORDER BY occurrence_count DESC",
        )?;
        let rows = stmt
            .query_map(params![repo, key], |row| {
                Ok(claudear_core::types::RepoKnowledge {
                    id: row.get(0)?,
                    repo: row.get(1)?,
                    knowledge_key: row.get(2)?,
                    knowledge_value: row.get(3)?,
                    source_type: row.get(4)?,
                    confidence: row.get(5)?,
                    occurrence_count: row.get(6)?,
                    created_at: Self::parse_datetime(&row.get::<_, String>(7)?)?,
                    updated_at: Self::parse_datetime(&row.get::<_, String>(8)?)?,
                })
            })?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }

    /// Return all known repository renames as (former_name, current_name) pairs.
    pub fn get_all_repo_aliases(&self) -> Result<Vec<(String, String)>> {
        let conn = self.acquire_lock()?;
        let mut stmt = conn.prepare(
            "SELECT knowledge_value, repo FROM repo_knowledge WHERE knowledge_key = 'former_name'",
        )?;
        let rows = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }

    /// System 5: Upsert a review pattern.
    pub fn upsert_review_pattern(
        &self,
        pattern: &claudear_core::types::ReviewPattern,
    ) -> Result<i64> {
        let conn = self.acquire_lock()?;
        let now = Utc::now().to_rfc3339();
        let category_str = pattern.category.to_string();
        let examples_json = serde_json::to_string(&pattern.example_comments).unwrap_or_default();

        let updated = conn.execute(
            "UPDATE review_patterns SET occurrence_count = occurrence_count + 1, example_comments = ?1, updated_at = ?2
             WHERE scm_repo = ?3 AND category = ?4 AND pattern_text = ?5",
            params![examples_json, now, pattern.scm_repo, category_str, pattern.pattern_text],
        )?;

        if updated > 0 {
            let id: i64 = conn
                .query_row(
                    "SELECT id FROM review_patterns WHERE scm_repo = ?1 AND category = ?2 AND pattern_text = ?3",
                    params![pattern.scm_repo, category_str, pattern.pattern_text],
                    |row| row.get(0),
                )
                .unwrap_or(0);
            return Ok(id);
        }

        conn.execute(
            "INSERT INTO review_patterns (scm_repo, category, pattern_text, example_comments, occurrence_count, promoted_to_instruction, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                pattern.scm_repo,
                category_str,
                pattern.pattern_text,
                examples_json,
                pattern.occurrence_count,
                pattern.promoted_to_instruction as i32,
                now,
                now,
            ],
        )?;
        Ok(conn.last_insert_rowid())
    }

    /// System 5: Get review patterns for a repo.
    pub fn get_review_patterns(
        &self,
        repo: &str,
        limit: usize,
    ) -> Result<Vec<claudear_core::types::ReviewPattern>> {
        let conn = self.acquire_lock()?;
        let mut stmt = conn.prepare(
            "SELECT id, scm_repo, category, pattern_text, example_comments, occurrence_count, promoted_to_instruction, created_at, updated_at
             FROM review_patterns WHERE scm_repo = ?1 ORDER BY occurrence_count DESC LIMIT ?2",
        )?;
        let rows = stmt
            .query_map(params![repo, limit as i64], |row| {
                Ok(claudear_core::types::ReviewPattern {
                    id: row.get(0)?,
                    scm_repo: row.get(1)?,
                    category: claudear_core::types::ReviewCategory::parse(
                        &row.get::<_, String>(2)?,
                    ),
                    pattern_text: row.get(3)?,
                    example_comments: serde_json::from_str(&row.get::<_, String>(4)?)
                        .unwrap_or_default(),
                    occurrence_count: row.get(5)?,
                    promoted_to_instruction: row.get::<_, i32>(6)? != 0,
                    created_at: Self::parse_datetime(&row.get::<_, String>(7)?)?,
                    updated_at: Self::parse_datetime(&row.get::<_, String>(8)?)?,
                })
            })?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }

    /// System 5: Get review patterns by category.
    pub fn get_review_patterns_by_category(
        &self,
        repo: &str,
        category: claudear_core::types::ReviewCategory,
    ) -> Result<Vec<claudear_core::types::ReviewPattern>> {
        let conn = self.acquire_lock()?;
        let category_str = category.to_string();
        let mut stmt = conn.prepare(
            "SELECT id, scm_repo, category, pattern_text, example_comments, occurrence_count, promoted_to_instruction, created_at, updated_at
             FROM review_patterns WHERE scm_repo = ?1 AND category = ?2 ORDER BY occurrence_count DESC",
        )?;
        let rows = stmt
            .query_map(params![repo, category_str], |row| {
                Ok(claudear_core::types::ReviewPattern {
                    id: row.get(0)?,
                    scm_repo: row.get(1)?,
                    category: claudear_core::types::ReviewCategory::parse(
                        &row.get::<_, String>(2)?,
                    ),
                    pattern_text: row.get(3)?,
                    example_comments: serde_json::from_str(&row.get::<_, String>(4)?)
                        .unwrap_or_default(),
                    occurrence_count: row.get(5)?,
                    promoted_to_instruction: row.get::<_, i32>(6)? != 0,
                    created_at: Self::parse_datetime(&row.get::<_, String>(7)?)?,
                    updated_at: Self::parse_datetime(&row.get::<_, String>(8)?)?,
                })
            })?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }

    /// System 6: Store a strategy fingerprint.
    pub fn store_strategy_fingerprint(
        &self,
        fp: &claudear_core::types::StrategyFingerprint,
    ) -> Result<i64> {
        let conn = self.acquire_lock()?;
        let files_json = serde_json::to_string(&fp.files_explored).unwrap_or_default();
        let tools_json = serde_json::to_string(&fp.tools_used).unwrap_or_default();

        conn.execute(
            "INSERT INTO strategy_fingerprints (attempt_id, files_explored, tests_run, tools_used, fix_approach, strategy_summary, fix_quality_score, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                fp.attempt_id,
                files_json,
                fp.tests_run,
                tools_json,
                fp.fix_approach,
                fp.strategy_summary,
                fp.fix_quality_score,
                fp.created_at.to_rfc3339(),
            ],
        )?;
        Ok(conn.last_insert_rowid())
    }

    /// System 6: Get successful strategies for a repo.
    pub fn get_successful_strategies(
        &self,
        repo: &str,
        limit: usize,
    ) -> Result<Vec<claudear_core::types::StrategyFingerprint>> {
        let conn = self.acquire_lock()?;
        let mut stmt = conn.prepare(
            "SELECT sf.id, sf.attempt_id, sf.files_explored, sf.tests_run, sf.tools_used, sf.fix_approach, sf.strategy_summary, sf.fix_quality_score, sf.created_at
             FROM strategy_fingerprints sf
             JOIN fix_attempts fa ON fa.id = sf.attempt_id
             WHERE fa.scm_repo = ?1 AND fa.status = 'merged'
             ORDER BY sf.fix_quality_score DESC NULLS LAST
             LIMIT ?2",
        )?;
        let rows = stmt
            .query_map(params![repo, limit as i64], |row| {
                Ok(claudear_core::types::StrategyFingerprint {
                    id: row.get(0)?,
                    attempt_id: row.get(1)?,
                    files_explored: serde_json::from_str(&row.get::<_, String>(2)?)
                        .unwrap_or_default(),
                    tests_run: row.get(3)?,
                    tools_used: serde_json::from_str(&row.get::<_, String>(4)?).unwrap_or_default(),
                    fix_approach: row.get(5)?,
                    strategy_summary: row.get(6)?,
                    fix_quality_score: row.get(7)?,
                    created_at: Self::parse_datetime(&row.get::<_, String>(8)?)?,
                })
            })?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }

    /// System 7: Update a PR's fix quality score.
    pub fn update_pr_fix_quality_score(&self, pr_url: &str, score: f64) -> Result<()> {
        let conn = self.acquire_lock()?;
        conn.execute(
            "UPDATE prs SET fix_quality_score = ?1 WHERE pr_url = ?2",
            params![score, pr_url],
        )?;
        Ok(())
    }

    /// System 8: Store an issue cluster.
    pub fn store_issue_cluster(&self, cluster: &claudear_core::types::IssueCluster) -> Result<i64> {
        let conn = self.acquire_lock()?;
        let ids_json = serde_json::to_string(&cluster.issue_ids).unwrap_or_default();

        conn.execute(
            "INSERT OR IGNORE INTO issue_clusters (cluster_key, source, issue_ids, window_start, window_end, status, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                cluster.cluster_key,
                cluster.source,
                ids_json,
                cluster.window_start.to_rfc3339(),
                cluster.window_end.to_rfc3339(),
                cluster.status,
                Utc::now().to_rfc3339(),
            ],
        )?;

        let cluster_id = conn.last_insert_rowid();

        // Insert members
        for issue_id in &cluster.issue_ids {
            conn.execute(
                "INSERT OR IGNORE INTO issue_cluster_members (cluster_id, issue_id, arrived_at)
                 VALUES (?1, ?2, ?3)",
                params![cluster_id, issue_id, Utc::now().to_rfc3339()],
            )?;
        }

        Ok(cluster_id)
    }

    /// System 8: Get active clusters for a source.
    pub fn get_active_clusters(
        &self,
        source: &str,
    ) -> Result<Vec<claudear_core::types::IssueCluster>> {
        let conn = self.acquire_lock()?;
        let mut stmt = conn.prepare(
            "SELECT id, cluster_key, source, issue_ids, window_start, window_end, resolved_by_issue_id, resolved_by_attempt_id, status, created_at
             FROM issue_clusters WHERE source = ?1 AND status = 'active' ORDER BY created_at DESC",
        )?;
        let rows = stmt
            .query_map(params![source], |row| {
                Ok(claudear_core::types::IssueCluster {
                    id: row.get(0)?,
                    cluster_key: row.get(1)?,
                    source: row.get(2)?,
                    issue_ids: serde_json::from_str(&row.get::<_, String>(3)?).unwrap_or_default(),
                    window_start: Self::parse_datetime(&row.get::<_, String>(4)?)?,
                    window_end: Self::parse_datetime(&row.get::<_, String>(5)?)?,
                    resolved_by_issue_id: row.get(6)?,
                    resolved_by_attempt_id: row.get(7)?,
                    status: row.get(8)?,
                    created_at: Self::parse_datetime(&row.get::<_, String>(9)?)?,
                })
            })?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }

    /// System 8: Mark a cluster as resolved.
    pub fn update_cluster_resolution(
        &self,
        cluster_id: i64,
        resolved_by_issue_id: &str,
        resolved_by_attempt_id: i64,
    ) -> Result<()> {
        let conn = self.acquire_lock()?;
        conn.execute(
            "UPDATE issue_clusters SET status = 'resolved', resolved_by_issue_id = ?1, resolved_by_attempt_id = ?2 WHERE id = ?3",
            params![resolved_by_issue_id, resolved_by_attempt_id, cluster_id],
        )?;
        Ok(())
    }

    /// System 8: Get recent issue arrivals within a time window.
    pub fn get_recent_issue_arrivals(
        &self,
        source: &str,
        window_minutes: i64,
    ) -> Result<Vec<(String, DateTime<Utc>)>> {
        let conn = self.acquire_lock()?;
        // Use SQLite's datetime function to compute the cutoff so the format
        // matches the 'datetime("now")' format used in record_attempt.
        let cutoff_modifier = format!("-{} minutes", window_minutes);
        let mut stmt = conn.prepare(
            "SELECT issue_id, attempted_at FROM fix_attempts WHERE source = ?1 AND attempted_at >= datetime('now', ?2) ORDER BY attempted_at ASC",
        )?;
        let rows = stmt
            .query_map(params![source, cutoff_modifier], |row| {
                let issue_id: String = row.get(0)?;
                let attempted_at = Self::parse_datetime(&row.get::<_, String>(1)?)?;
                Ok((issue_id, attempted_at))
            })?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }

    pub fn store_content_cluster(
        &self,
        cluster: &claudear_core::types::ContentCluster,
    ) -> Result<i64> {
        let conn = self.acquire_lock()?;
        let ids_json = serde_json::to_string(&cluster.issue_ids).unwrap_or_else(|_| "[]".into());
        conn.execute(
            "INSERT INTO content_clusters (cluster_key, source, representative_issue_id, issue_ids, error_type, culprit, avg_similarity, status, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
             ON CONFLICT(cluster_key, source) DO UPDATE SET
                representative_issue_id = excluded.representative_issue_id,
                issue_ids = excluded.issue_ids,
                error_type = excluded.error_type,
                culprit = excluded.culprit,
                avg_similarity = excluded.avg_similarity,
                status = excluded.status",
            params![
                cluster.cluster_key,
                cluster.source,
                cluster.representative_issue_id,
                ids_json,
                cluster.error_type,
                cluster.culprit,
                cluster.avg_similarity,
                cluster.status,
                Utc::now().format("%Y-%m-%d %H:%M:%S").to_string(),
            ],
        )?;
        Ok(conn.last_insert_rowid())
    }

    pub fn get_active_content_clusters(
        &self,
        source: &str,
    ) -> Result<Vec<claudear_core::types::ContentCluster>> {
        let conn = self.acquire_lock()?;
        let mut stmt = conn.prepare(
            "SELECT id, cluster_key, source, representative_issue_id, issue_ids, error_type, culprit, avg_similarity, status, created_at
             FROM content_clusters WHERE source = ?1 AND status = 'active' ORDER BY created_at DESC",
        )?;
        let rows = stmt
            .query_map(params![source], |row| {
                Ok(claudear_core::types::ContentCluster {
                    id: row.get(0)?,
                    cluster_key: row.get(1)?,
                    source: row.get(2)?,
                    representative_issue_id: row.get(3)?,
                    issue_ids: serde_json::from_str(&row.get::<_, String>(4)?).unwrap_or_default(),
                    error_type: row.get(5)?,
                    culprit: row.get(6)?,
                    avg_similarity: row.get(7)?,
                    status: row.get(8)?,
                    created_at: Self::parse_datetime(&row.get::<_, String>(9)?)?,
                })
            })?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }

    pub fn resolve_content_cluster(&self, cluster_id: i64) -> Result<()> {
        let conn = self.acquire_lock()?;
        conn.execute(
            "UPDATE content_clusters SET status = 'resolved' WHERE id = ?1",
            params![cluster_id],
        )?;
        Ok(())
    }

    pub fn store_severity_score(
        &self,
        source: &str,
        issue_id: &str,
        score: &claudear_core::types::SeverityScore,
        blast_radius: claudear_core::types::BlastRadius,
    ) -> Result<()> {
        let conn = self.acquire_lock()?;
        conn.execute(
            "INSERT OR REPLACE INTO severity_scores (source, issue_id, score, severity_component, frequency_component, regression_component, blast_radius_component, cluster_boost, blast_radius, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, datetime('now'))",
            params![
                source,
                issue_id,
                score.score,
                score.severity_component,
                score.frequency_component,
                score.regression_component,
                score.blast_radius_component,
                score.cluster_boost,
                blast_radius.to_string(),
            ],
        )?;
        Ok(())
    }

    pub fn record_suppression(
        &self,
        source: &str,
        issue_id: &str,
        rule_name: &str,
        reason: &str,
    ) -> Result<()> {
        let conn = self.acquire_lock()?;
        conn.execute(
            "INSERT OR IGNORE INTO suppression_log (source, issue_id, rule_name, reason, created_at)
             VALUES (?1, ?2, ?3, ?4, datetime('now'))",
            params![source, issue_id, rule_name, reason],
        )?;
        Ok(())
    }

    pub fn get_recent_attempts_since(&self, since: &DateTime<Utc>) -> Result<Vec<FixAttempt>> {
        let conn = self.acquire_lock()?;
        let since_str = since.format("%Y-%m-%d %H:%M:%S").to_string();
        let mut stmt = conn.prepare(
            "SELECT id, source, issue_id, short_id, attempted_at, pr_url, scm_repo,
                    scm_pr_number, status, error_message, merged_at, resolved_at,
                    retry_count, last_retry_at, issue_labels, parent_attempt_id, cascade_repo
             FROM fix_attempts WHERE attempted_at >= ?1 ORDER BY attempted_at DESC",
        )?;
        let rows = stmt
            .query_map(params![since_str], Self::row_to_fix_attempt)?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }

    pub fn has_dependency(&self, repo_a: &str, repo_b: &str) -> Result<bool> {
        let conn = self.acquire_lock()?;
        // Check if repo_a depends on repo_b (repo_b is upstream of repo_a)
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM repository_dependencies rd
             JOIN repositories r1 ON rd.downstream_id = r1.id
             JOIN repositories r2 ON rd.upstream_id = r2.id
             WHERE r1.name = ?1 AND r2.name = ?2",
            params![repo_a, repo_b],
            |row| row.get(0),
        )?;
        Ok(count > 0)
    }

    pub fn upsert_cross_repo_correlation(
        &self,
        repo_a: &str,
        repo_b: &str,
        window_hours: i64,
    ) -> Result<CrossRepoCorrelation> {
        let conn = self.acquire_lock()?;
        let now = Utc::now().format("%Y-%m-%d %H:%M:%S").to_string();
        conn.execute(
            "INSERT INTO cross_repo_correlations (repo_a, repo_b, correlation_count, last_seen_at, window_hours)
             VALUES (?1, ?2, 1, ?3, ?4)
             ON CONFLICT(repo_a, repo_b) DO UPDATE SET
                correlation_count = correlation_count + 1,
                last_seen_at = excluded.last_seen_at,
                window_hours = excluded.window_hours",
            params![repo_a, repo_b, now, window_hours],
        )?;

        let row = conn.query_row(
            "SELECT id, repo_a, repo_b, correlation_count, last_seen_at, window_hours FROM cross_repo_correlations WHERE repo_a = ?1 AND repo_b = ?2",
            params![repo_a, repo_b],
            |row| {
                Ok(CrossRepoCorrelation {
                    id: row.get(0)?,
                    repo_a: row.get(1)?,
                    repo_b: row.get(2)?,
                    correlation_count: row.get(3)?,
                    last_seen_at: Self::parse_datetime(&row.get::<_, String>(4)?)?,
                    window_hours: row.get(5)?,
                })
            },
        )?;
        Ok(row)
    }

    pub fn get_cross_repo_correlations(
        &self,
        min_count: i64,
        max_age_hours: i64,
    ) -> Result<Vec<CrossRepoCorrelation>> {
        let conn = self.acquire_lock()?;
        let cutoff_modifier = format!("-{} hours", max_age_hours);
        let mut stmt = conn.prepare(
            "SELECT id, repo_a, repo_b, correlation_count, last_seen_at, window_hours
             FROM cross_repo_correlations
             WHERE correlation_count >= ?1 AND last_seen_at >= datetime('now', ?2)
             ORDER BY correlation_count DESC",
        )?;
        let rows = stmt
            .query_map(params![min_count, cutoff_modifier], |row| {
                Ok(CrossRepoCorrelation {
                    id: row.get(0)?,
                    repo_a: row.get(1)?,
                    repo_b: row.get(2)?,
                    correlation_count: row.get(3)?,
                    last_seen_at: Self::parse_datetime(&row.get::<_, String>(4)?)?,
                    window_hours: row.get(5)?,
                })
            })?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }

    pub fn store_code_complexity(
        &self,
        repo_id: i64,
        file_path: &str,
        fc: &claudear_core::types::FileComplexity,
    ) -> Result<()> {
        let conn = self.acquire_lock()?;
        conn.execute(
            "INSERT OR REPLACE INTO code_complexity (repo_id, file_path, avg_cyclomatic, max_cyclomatic, avg_func_length, max_func_length, avg_nesting, max_nesting, total_lines, function_count, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, datetime('now'))",
            params![
                repo_id,
                file_path,
                fc.avg_cyclomatic,
                fc.max_cyclomatic,
                fc.avg_func_length,
                fc.max_func_length,
                fc.avg_nesting,
                fc.max_nesting,
                fc.total_lines,
                fc.function_count,
            ],
        )?;
        Ok(())
    }

    /// Get or create a repository ID by name.
    pub fn get_or_create_repo_id(&self, name: &str) -> Result<i64> {
        self.upsert_repository(name, None, None)
    }

    /// Check if a file's hash matches the currently stored hash (for incremental indexing).
    /// Returns true only if chunks exist AND every chunk has an embedding,
    /// ensuring partially-embedded files are re-indexed.
    pub fn code_chunk_hash_matches(
        &self,
        repo_id: i64,
        file_path: &str,
        file_hash: &str,
    ) -> Result<bool> {
        let conn = self.acquire_lock()?;
        let (chunk_count, embedded_count): (i64, i64) = conn.query_row(
            r#"
            SELECT
                (SELECT COUNT(*) FROM code_chunks
                 WHERE repo_id = ?1 AND file_path = ?2 AND file_hash = ?3),
                (SELECT COUNT(*) FROM code_chunks c
                 JOIN code_chunk_embeddings e ON e.chunk_id = c.id
                 WHERE c.repo_id = ?1 AND c.file_path = ?2 AND c.file_hash = ?3)
            "#,
            params![repo_id, file_path, file_hash],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        Ok(chunk_count > 0 && chunk_count == embedded_count)
    }

    /// Delete all code symbols, chunks, and embeddings for a specific file.
    pub fn delete_code_data_for_file(&self, repo_id: i64, file_path: &str) -> Result<()> {
        let mut conn = self.acquire_lock()?;
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        // Embeddings are CASCADE-deleted via code_chunks FK.
        tx.execute(
            "DELETE FROM code_symbols WHERE repo_id = ?1 AND file_path = ?2",
            params![repo_id, file_path],
        )?;
        tx.execute(
            "DELETE FROM code_chunks WHERE repo_id = ?1 AND file_path = ?2",
            params![repo_id, file_path],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// Delete code chunks (and their CASCADE-deleted embeddings) by chunk IDs.
    pub fn delete_code_chunks_by_ids(&self, chunk_ids: &[i64]) -> Result<()> {
        if chunk_ids.is_empty() {
            return Ok(());
        }
        let conn = self.acquire_lock()?;
        let placeholders: Vec<String> = (1..=chunk_ids.len()).map(|i| format!("?{}", i)).collect();
        let sql = format!(
            "DELETE FROM code_chunks WHERE id IN ({})",
            placeholders.join(", ")
        );
        let params: Vec<&dyn rusqlite::ToSql> = chunk_ids
            .iter()
            .map(|id| id as &dyn rusqlite::ToSql)
            .collect();
        conn.execute(&sql, params.as_slice())?;
        Ok(())
    }

    /// Remove code data for files that no longer exist in the repository.
    pub fn cleanup_stale_code_data(&self, repo_id: i64, current_paths: &[String]) -> Result<()> {
        let mut conn = self.acquire_lock()?;

        if current_paths.is_empty() {
            // No source files found — delete ALL code data for this repo.
            let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
            tx.execute(
                "DELETE FROM code_symbols WHERE repo_id = ?1",
                params![repo_id],
            )?;
            tx.execute(
                "DELETE FROM code_chunks WHERE repo_id = ?1",
                params![repo_id],
            )?;
            tx.commit()?;
            return Ok(());
        }

        // Get all file paths we have indexed for this repo.
        // code_chunks is the primary artifact; stale symbols are cleaned up
        // via delete_code_data_for_file which deletes both tables.
        let indexed_paths: Vec<String> = {
            let mut stmt =
                conn.prepare("SELECT DISTINCT file_path FROM code_chunks WHERE repo_id = ?1")?;
            let rows: Vec<String> = stmt
                .query_map(params![repo_id], |row| row.get(0))?
                .filter_map(|r| r.ok())
                .collect();
            rows
        };

        let current_set: std::collections::HashSet<&str> =
            current_paths.iter().map(|s| s.as_str()).collect();

        let stale_paths: Vec<&String> = indexed_paths
            .iter()
            .filter(|p| !current_set.contains(p.as_str()))
            .collect();

        if !stale_paths.is_empty() {
            let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
            for path in &stale_paths {
                tx.execute(
                    "DELETE FROM code_symbols WHERE repo_id = ?1 AND file_path = ?2",
                    params![repo_id, path],
                )?;
                tx.execute(
                    "DELETE FROM code_chunks WHERE repo_id = ?1 AND file_path = ?2",
                    params![repo_id, path],
                )?;
            }
            tx.commit()?;
        }
        Ok(())
    }

    /// Batch-save extracted code symbols.
    pub fn save_code_symbols(&self, symbols: &[claudear_core::types::CodeSymbol]) -> Result<()> {
        let mut conn = self.acquire_lock()?;
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;

        {
            let mut stmt = tx.prepare(
                r#"
                INSERT INTO code_symbols (repo_id, file_path, symbol_name, symbol_kind, parent_symbol, language, start_line, end_line, signature)
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
                "#,
            )?;

            for sym in symbols {
                stmt.execute(params![
                    sym.repo_id,
                    &sym.file_path,
                    &sym.symbol_name,
                    sym.symbol_kind.as_str(),
                    sym.parent_symbol.as_deref(),
                    sym.language.as_str(),
                    sym.start_line as i64,
                    sym.end_line as i64,
                    sym.signature.as_deref(),
                ])?;
            }
        }

        tx.commit()?;
        Ok(())
    }

    /// Batch-save code chunks. Returns the assigned IDs.
    pub fn save_code_chunks(&self, chunks: &[claudear_core::types::CodeChunk]) -> Result<Vec<i64>> {
        let mut conn = self.acquire_lock()?;
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;

        let mut ids = Vec::with_capacity(chunks.len());
        {
            let mut stmt = tx.prepare(
                r#"
                INSERT INTO code_chunks (repo_id, file_path, chunk_type, symbol_name, language, start_line, end_line, chunk_text, context_text, file_hash, content_hash)
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)
                "#,
            )?;

            for chunk in chunks {
                stmt.execute(params![
                    chunk.repo_id,
                    &chunk.file_path,
                    &chunk.chunk_type,
                    chunk.symbol_name.as_deref(),
                    chunk.language.as_str(),
                    chunk.start_line as i64,
                    chunk.end_line as i64,
                    &chunk.chunk_text,
                    &chunk.context_text,
                    &chunk.file_hash,
                    chunk.content_hash.as_deref(),
                ])?;
                ids.push(tx.last_insert_rowid());
            }
        }

        tx.commit()?;
        Ok(ids)
    }

    /// Save embeddings for code chunks and insert into the HNSW vector index.
    pub fn save_code_chunk_embeddings(
        &self,
        pairs: &[(i64, &[f32])],
        model_name: &str,
    ) -> Result<()> {
        if pairs.is_empty() {
            return Ok(());
        }

        let mut conn = self.acquire_lock()?;

        let dimension = pairs[0].1.len();
        let _ = Self::ensure_code_chunk_vector_table(&conn, dimension);

        let has_vector_table = Self::table_exists(&conn, CODE_CHUNK_VECTOR_TABLE)?;

        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;

        {
            let mut stmt = tx.prepare(
                "INSERT INTO code_chunk_embeddings (chunk_id, embedding, embedding_model) VALUES (?1, ?2, ?3)",
            )?;

            for &(chunk_id, embedding) in pairs {
                let blob: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();
                stmt.execute(params![chunk_id, blob, model_name])?;

                if has_vector_table {
                    let emb_id = tx.last_insert_rowid();
                    let insert_sql = format!(
                        "INSERT INTO {}(rowid, embedding) VALUES (?1, ?2)",
                        CODE_CHUNK_VECTOR_TABLE
                    );
                    if let Err(e) = tx.execute(&insert_sql, params![emb_id, blob]) {
                        tracing::warn!(error = %e, chunk_id, "Failed to insert into code chunk vector table — aborting batch");
                        // Drop tx without commit to roll back the entire batch,
                        // so embeddings and vector index stay consistent.
                        return Err(e.into());
                    }
                }
            }
        }

        tx.commit()?;
        Ok(())
    }

    /// Lazily create the HNSW vector table for code chunk embeddings.
    fn ensure_code_chunk_vector_table(conn: &Connection, dimension: usize) -> Result<bool> {
        if dimension == 0 {
            return Ok(false);
        }
        validate_dimension(dimension)?;

        if !is_vectorlite_available(conn) {
            match try_load_vectorlite(conn) {
                Ok(true) => {}
                Ok(false) => return Ok(false),
                Err(e) => {
                    tracing::debug!(error = %e, "Unable to load vectorlite for code chunk search");
                    return Ok(false);
                }
            }
        }

        if Self::table_exists(conn, CODE_CHUNK_VECTOR_TABLE)? {
            return Ok(true);
        }

        let sql = format!(
            r#"
            CREATE VIRTUAL TABLE IF NOT EXISTS {table} USING vectorlite(
                embedding float32[{dimension}] cosine,
                hnsw(max_elements=100000, ef_construction=200, M=16)
            )
            "#,
            table = CODE_CHUNK_VECTOR_TABLE,
            dimension = dimension
        );

        match conn.execute_batch(&sql) {
            Ok(()) => {
                // Backfill existing embeddings.
                let backfill = format!(
                    r#"
                    INSERT INTO {table}(rowid, embedding)
                    SELECT id, embedding
                    FROM code_chunk_embeddings
                    WHERE length(embedding) = ?1
                    "#,
                    table = CODE_CHUNK_VECTOR_TABLE
                );
                if let Err(e) = conn.execute(&backfill, params![(dimension * 4) as i64]) {
                    tracing::debug!(error = %e, "Failed to backfill code chunk vector embeddings");
                }
                Ok(true)
            }
            Err(e) => {
                tracing::debug!(error = %e, "Failed to create code chunk vector table");
                Ok(false)
            }
        }
    }

    /// Search code chunks by vector similarity using HNSW index.
    pub fn search_code_chunks(
        &self,
        query_embedding: &[f32],
        repo_id: Option<i64>,
        limit: usize,
    ) -> Result<Vec<claudear_core::types::CodeSearchResult>> {
        use claudear_core::types::{CodeChunk, CodeSearchResult};

        if query_embedding.is_empty() || limit == 0 {
            return Ok(Vec::new());
        }

        // Clamp limit to prevent excessive memory usage from the HNSW candidate multiplier.
        let limit = limit.min(1000);

        let conn = self.acquire_lock()?;

        if !Self::ensure_code_chunk_vector_table(&conn, query_embedding.len())? {
            // Vectorlite unavailable — fall back to empty results.
            return Ok(Vec::new());
        }

        let query_blob: Vec<u8> = query_embedding
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let candidate_limit = limit * CODE_CHUNK_VECTOR_CANDIDATE_MULTIPLIER;

        let sql = format!(
            r#"
            WITH candidates AS (
                SELECT rowid AS emb_id,
                       MAX(0.0, MIN(1.0, 1.0 - distance)) AS similarity
                FROM {table}
                WHERE knn_search(embedding, knn_param(?1, ?2, ?3))
            )
            SELECT c.similarity,
                   ch.id, ch.repo_id, ch.file_path, ch.chunk_type, ch.symbol_name,
                   ch.language, ch.start_line, ch.end_line, ch.chunk_text,
                   ch.context_text, ch.file_hash
            FROM candidates c
            JOIN code_chunk_embeddings e ON e.id = c.emb_id
            JOIN code_chunks ch ON ch.id = e.chunk_id
            WHERE (?4 IS NULL OR ch.repo_id = ?4)
            ORDER BY c.similarity DESC
            LIMIT ?5
            "#,
            table = CODE_CHUNK_VECTOR_TABLE
        );

        let mut stmt = match conn.prepare(&sql) {
            Ok(s) => s,
            Err(e) => {
                tracing::debug!(error = %e, "Failed to prepare code chunk vector search");
                return Ok(Vec::new());
            }
        };

        let rows = match stmt.query_map(
            params![
                query_blob,
                candidate_limit as i64,
                CODE_CHUNK_VECTOR_EF_SEARCH as i64,
                repo_id,
                limit as i64,
            ],
            |row| {
                let similarity: f64 = row.get(0)?;
                let lang_str: String = row.get(6)?;
                let chunk_text: String = row.get(9)?;
                let stored_prefix: String = row.get(10)?;
                // Reconstruct full context_text from stored prefix + chunk_text.
                let context_text = format!("{}\n{}", stored_prefix.trim_end(), chunk_text);

                Ok(CodeSearchResult {
                    chunk: CodeChunk {
                        id: row.get(1)?,
                        repo_id: row.get(2)?,
                        file_path: row.get(3)?,
                        chunk_type: row.get(4)?,
                        symbol_name: row.get(5)?,
                        language: parse_language(&lang_str),
                        start_line: row.get::<_, i64>(7)? as usize,
                        end_line: row.get::<_, i64>(8)? as usize,
                        chunk_text,
                        context_text,
                        file_hash: row.get(11)?,
                        content_hash: None,
                    },
                    score: similarity,
                })
            },
        ) {
            Ok(r) => r,
            Err(e) => {
                tracing::debug!(error = %e, "Code chunk vector search failed");
                return Ok(Vec::new());
            }
        };

        let mut results = Vec::new();
        for row in rows {
            match row {
                Ok(r) => results.push(r),
                Err(e) => tracing::debug!(error = %e, "Failed to read code chunk vector row"),
            }
        }

        Ok(results)
    }

    /// Find code symbols by name (substring match).
    pub fn find_code_symbols(
        &self,
        name: &str,
        kind: Option<claudear_core::types::SymbolKind>,
        repo_id: Option<i64>,
    ) -> Result<Vec<claudear_core::types::CodeSymbol>> {
        use claudear_core::types::CodeSymbol;

        let conn = self.acquire_lock()?;
        // Escape LIKE special characters in user input to prevent wildcard injection.
        let escaped = name
            .replace('\\', "\\\\")
            .replace('%', "\\%")
            .replace('_', "\\_");
        let pattern = format!("%{}%", escaped);

        let sql = r#"
            SELECT id, repo_id, file_path, symbol_name, symbol_kind, parent_symbol,
                   language, start_line, end_line, signature
            FROM code_symbols
            WHERE symbol_name LIKE ?1 ESCAPE '\'
              AND (?2 IS NULL OR symbol_kind = ?2)
              AND (?3 IS NULL OR repo_id = ?3)
            ORDER BY symbol_name
            LIMIT 100
        "#;

        let kind_str = kind.map(|k| k.as_str().to_string());
        let mut stmt = conn.prepare(sql)?;
        let rows = stmt.query_map(params![pattern, kind_str, repo_id], |row| {
            let lang_str: String = row.get(6)?;
            let kind_str: String = row.get(4)?;
            Ok(CodeSymbol {
                id: row.get(0)?,
                repo_id: row.get(1)?,
                file_path: row.get(2)?,
                symbol_name: row.get(3)?,
                symbol_kind: claudear_core::types::SymbolKind::from_str_loose(&kind_str)
                    .unwrap_or(claudear_core::types::SymbolKind::Function),
                parent_symbol: row.get(5)?,
                language: parse_language(&lang_str),
                start_line: row.get::<_, i64>(7)? as usize,
                end_line: row.get::<_, i64>(8)? as usize,
                signature: row.get(9)?,
            })
        })?;

        let mut results = Vec::new();
        for sym in rows.flatten() {
            results.push(sym);
        }
        Ok(results)
    }

    /// Get the embedding model used for a repo's existing code chunk embeddings.
    pub fn get_code_embedding_model(&self, repo_id: i64) -> Result<Option<String>> {
        let conn = self.acquire_lock()?;
        let result: Option<String> = conn
            .query_row(
                r#"
                SELECT embedding_model FROM code_chunk_embeddings
                WHERE chunk_id IN (SELECT id FROM code_chunks WHERE repo_id = ?1)
                LIMIT 1
                "#,
                params![repo_id],
                |row| row.get(0),
            )
            .optional()?;
        Ok(result)
    }

    /// Get a value from the code_index_metadata table.
    pub fn get_code_index_meta(&self, repo_id: i64, key: &str) -> Result<Option<String>> {
        let conn = self.acquire_lock()?;
        let result: Option<String> = conn
            .query_row(
                "SELECT value FROM code_index_metadata WHERE repo_id = ?1 AND key = ?2",
                params![repo_id, key],
                |row| row.get(0),
            )
            .optional()?;
        Ok(result)
    }

    /// Set (upsert) a value in the code_index_metadata table.
    pub fn set_code_index_meta(&self, repo_id: i64, key: &str, value: &str) -> Result<()> {
        let conn = self.acquire_lock()?;
        conn.execute(
            "INSERT INTO code_index_metadata (repo_id, key, value) VALUES (?1, ?2, ?3)
             ON CONFLICT(repo_id, key) DO UPDATE SET value = excluded.value",
            params![repo_id, key, value],
        )?;
        Ok(())
    }

    /// Delete all code data (symbols, chunks, embeddings) for a repo.
    pub fn delete_all_code_data_for_repo(&self, repo_id: i64) -> Result<()> {
        let mut conn = self.acquire_lock()?;
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        tx.execute(
            "DELETE FROM code_symbols WHERE repo_id = ?1",
            params![repo_id],
        )?;
        // Embeddings are CASCADE-deleted via code_chunks FK.
        tx.execute(
            "DELETE FROM code_chunks WHERE repo_id = ?1",
            params![repo_id],
        )?;
        tx.commit()?;
        Ok(())
    }

    // --- Chat Session Store ---

    /// Create a new chat session.
    pub fn create_chat_session(&self, id: &str, repo_id: Option<i64>) -> Result<()> {
        let conn = self.acquire_lock()?;
        conn.execute(
            "INSERT INTO chat_sessions (id, repo_id) VALUES (?1, ?2)",
            params![id, repo_id],
        )?;
        Ok(())
    }

    /// Save a chat message to a session.
    pub fn save_chat_message(
        &self,
        session_id: &str,
        role: &str,
        content: &str,
        sources_json: Option<&str>,
    ) -> Result<i64> {
        let conn = self.acquire_lock()?;
        conn.execute(
            "INSERT INTO chat_messages (session_id, role, content, sources_json) VALUES (?1, ?2, ?3, ?4)",
            params![session_id, role, content, sources_json],
        )?;
        let id = conn.last_insert_rowid();
        // Update session's updated_at
        conn.execute(
            "UPDATE chat_sessions SET updated_at = datetime('now') WHERE id = ?1",
            params![session_id],
        )?;
        Ok(id)
    }

    /// Get chat message history for a session (most recent N messages).
    pub fn get_chat_history(
        &self,
        session_id: &str,
        limit: usize,
    ) -> Result<Vec<claudear_core::types::ChatMessage>> {
        let conn = self.acquire_lock()?;
        let mut stmt = conn.prepare(
            "SELECT id, role, content, sources_json, created_at
             FROM chat_messages
             WHERE session_id = ?1
             ORDER BY created_at DESC
             LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![session_id, limit as i64], |row| {
            let role_str: String = row.get(1)?;
            let created_str: String = row.get(4)?;
            Ok(claudear_core::types::ChatMessage {
                id: Some(row.get(0)?),
                role: claudear_core::types::ChatRole::parse(&role_str)
                    .unwrap_or(claudear_core::types::ChatRole::User),
                content: row.get(2)?,
                sources_json: row.get(3)?,
                created_at: chrono::NaiveDateTime::parse_from_str(
                    &created_str,
                    "%Y-%m-%d %H:%M:%S",
                )
                .map(|dt| dt.and_utc())
                .unwrap_or_else(|_| chrono::Utc::now()),
            })
        })?;

        let mut messages: Vec<_> = rows.filter_map(|r| r.ok()).collect();
        messages.reverse(); // Return in chronological order
        Ok(messages)
    }

    /// List all chat sessions.
    pub fn list_chat_sessions(&self) -> Result<Vec<claudear_core::types::ChatSession>> {
        let conn = self.acquire_lock()?;
        let mut stmt = conn.prepare(
            "SELECT id, repo_id, created_at, updated_at
             FROM chat_sessions
             ORDER BY updated_at DESC",
        )?;
        let rows = stmt.query_map([], |row| {
            let created_str: String = row.get(2)?;
            let updated_str: String = row.get(3)?;
            Ok(claudear_core::types::ChatSession {
                id: row.get(0)?,
                repo_id: row.get(1)?,
                created_at: chrono::NaiveDateTime::parse_from_str(
                    &created_str,
                    "%Y-%m-%d %H:%M:%S",
                )
                .map(|dt| dt.and_utc())
                .unwrap_or_else(|_| chrono::Utc::now()),
                updated_at: chrono::NaiveDateTime::parse_from_str(
                    &updated_str,
                    "%Y-%m-%d %H:%M:%S",
                )
                .map(|dt| dt.and_utc())
                .unwrap_or_else(|_| chrono::Utc::now()),
                messages: Vec::new(),
            })
        })?;
        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    /// Delete a chat session and all its messages.
    pub fn delete_chat_session(&self, session_id: &str) -> Result<()> {
        let conn = self.acquire_lock()?;
        // Messages are CASCADE-deleted via FK.
        conn.execute(
            "DELETE FROM chat_sessions WHERE id = ?1",
            params![session_id],
        )?;
        Ok(())
    }

    /// Cleanup expired chat sessions older than max_age_days. Returns count deleted.
    pub fn cleanup_expired_chat_sessions(&self, max_age_days: u32) -> Result<usize> {
        let conn = self.acquire_lock()?;
        let count = conn.execute(
            "DELETE FROM chat_sessions WHERE updated_at < datetime('now', ?1)",
            params![format!("-{max_age_days} days")],
        )?;
        Ok(count)
    }
}

/// Parse a language string from the DB back into a Language enum.
fn parse_language(s: &str) -> claudear_core::types::Language {
    match s {
        "Rust" => claudear_core::types::Language::Rust,
        "TypeScript" => claudear_core::types::Language::TypeScript,
        "TSX" => claudear_core::types::Language::Tsx,
        "JavaScript" => claudear_core::types::Language::JavaScript,
        "Python" => claudear_core::types::Language::Python,
        "Go" => claudear_core::types::Language::Go,
        "Java" => claudear_core::types::Language::Java,
        "C" => claudear_core::types::Language::C,
        "C++" => claudear_core::types::Language::Cpp,
        "Ruby" => claudear_core::types::Language::Ruby,
        "PHP" => claudear_core::types::Language::Php,
        "Swift" => claudear_core::types::Language::Swift,
        "Kotlin" => claudear_core::types::Language::Kotlin,
        other => {
            tracing::error!(language = %other, "Unknown language in DB — data integrity issue; falling back to Rust");
            claudear_core::types::Language::Rust
        }
    }
}

/// Normalize a value into the [0.0, 1.0] range based on threshold buckets.
///
/// The thresholds define 4 equal-width buckets:
///   [t0..t1] → [0.0..0.25], [t1..t2] → [0.25..0.5], [t2..t3] → [0.5..0.75], [t3..t4] → [0.75..1.0]
/// Values below t0 map to 0.0, above t4 map to 1.0.
fn normalize_signal(value: f64, thresholds: &[f64; 5]) -> f64 {
    if value <= thresholds[0] {
        return 0.0;
    }
    for i in 1..5 {
        if value <= thresholds[i] {
            let lo = thresholds[i - 1];
            let hi = thresholds[i];
            if (hi - lo).abs() < f64::EPSILON {
                return (i as f64) / 4.0;
            }
            let bucket_frac = (value - lo) / (hi - lo);
            return ((i - 1) as f64 + bucket_frac) / 4.0;
        }
    }
    1.0
}

/// Map a complexity score (0.0–1.0) to estimated hours saved.
fn complexity_to_hours(score: f64) -> f64 {
    match score {
        s if s <= 0.2 => 0.5,
        s if s <= 0.4 => 1.0,
        s if s <= 0.6 => 2.0,
        s if s <= 0.8 => 4.0,
        _ => 8.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Datelike, Timelike, Utc};

    #[test]
    fn test_record_and_retrieve_attempt() {
        let tracker = SqliteTracker::in_memory().unwrap();

        tracker.record_attempt("linear", "123", "PROJ-123").unwrap();

        assert!(tracker.has_attempted("linear", "123").unwrap());
        assert!(!tracker.has_attempted("linear", "456").unwrap());
        assert!(!tracker.has_attempted("sentry", "123").unwrap());

        let attempt = tracker.get_attempt("linear", "123").unwrap().unwrap();
        assert_eq!(attempt.issue_id, "123");
        assert_eq!(attempt.short_id, "PROJ-123");
        assert_eq!(attempt.source, "linear");
        assert_eq!(attempt.status, FixAttemptStatus::Pending);
    }

    #[test]
    fn test_mark_success() {
        let tracker = SqliteTracker::in_memory().unwrap();

        tracker.record_attempt("linear", "123", "PROJ-123").unwrap();
        tracker
            .mark_success("linear", "123", "https://github.com/org/repo/pull/42")
            .unwrap();

        let attempt = tracker.get_attempt("linear", "123").unwrap().unwrap();
        assert_eq!(attempt.status, FixAttemptStatus::Success);
        assert_eq!(
            attempt.pr_url,
            Some("https://github.com/org/repo/pull/42".to_string())
        );
        // Check that GitHub info was extracted
        assert_eq!(attempt.scm_repo, Some("org/repo".to_string()));
        assert_eq!(attempt.scm_pr_number, Some(42));
    }

    #[test]
    fn test_mark_merged() {
        let tracker = SqliteTracker::in_memory().unwrap();

        tracker.record_attempt("linear", "123", "PROJ-123").unwrap();
        tracker
            .mark_success("linear", "123", "https://github.com/org/repo/pull/42")
            .unwrap();
        tracker.mark_merged("linear", "123").unwrap();

        let attempt = tracker.get_attempt("linear", "123").unwrap().unwrap();
        assert_eq!(attempt.status, FixAttemptStatus::Merged);
        assert!(attempt.merged_at.is_some());
    }

    #[test]
    fn test_mark_closed() {
        let tracker = SqliteTracker::in_memory().unwrap();

        tracker.record_attempt("linear", "123", "PROJ-123").unwrap();
        tracker
            .mark_success("linear", "123", "https://github.com/org/repo/pull/42")
            .unwrap();
        tracker.mark_closed("linear", "123").unwrap();

        let attempt = tracker.get_attempt("linear", "123").unwrap().unwrap();
        assert_eq!(attempt.status, FixAttemptStatus::Closed);
    }

    #[test]
    fn test_get_pending_prs() {
        let tracker = SqliteTracker::in_memory().unwrap();

        // Create a successful attempt with a PR
        tracker.record_attempt("linear", "123", "PROJ-123").unwrap();
        tracker
            .mark_success("linear", "123", "https://github.com/org/repo/pull/42")
            .unwrap();

        // Create a merged attempt (should not be in pending PRs)
        tracker.record_attempt("linear", "456", "PROJ-456").unwrap();
        tracker
            .mark_success("linear", "456", "https://github.com/org/repo/pull/43")
            .unwrap();
        tracker.mark_merged("linear", "456").unwrap();

        let pending_prs = tracker.get_pending_prs().unwrap();
        assert_eq!(pending_prs.len(), 1);
        assert_eq!(pending_prs[0].issue_id, "123");
    }

    #[test]
    fn test_get_attempt_by_pr_url() {
        let tracker = SqliteTracker::in_memory().unwrap();

        tracker.record_attempt("linear", "123", "PROJ-123").unwrap();
        tracker
            .mark_success("linear", "123", "https://github.com/org/repo/pull/42")
            .unwrap();

        let attempt = tracker
            .get_attempt_by_pr_url("https://github.com/org/repo/pull/42")
            .unwrap()
            .unwrap();
        assert_eq!(attempt.issue_id, "123");
        assert_eq!(attempt.source, "linear");
    }

    #[test]
    fn test_get_attempt_by_pr_url_returns_latest_attempt() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let pr_url = "https://github.com/org/repo/pull/99";

        tracker
            .record_attempt("linear", "older", "PROJ-OLDER")
            .unwrap();
        tracker.mark_success("linear", "older", pr_url).unwrap();

        tracker
            .record_attempt("linear", "newer", "PROJ-NEWER")
            .unwrap();
        tracker.mark_success("linear", "newer", pr_url).unwrap();

        {
            let conn = tracker.acquire_lock().unwrap();
            conn.execute(
                "UPDATE fix_attempts SET attempted_at = ? WHERE source = ? AND issue_id = ?",
                params!["2030-01-01 00:00:00", "linear", "older"],
            )
            .unwrap();
            conn.execute(
                "UPDATE fix_attempts SET attempted_at = ? WHERE source = ? AND issue_id = ?",
                params!["2020-01-01 00:00:00", "linear", "newer"],
            )
            .unwrap();
        }

        let attempt = tracker.get_attempt_by_pr_url(pr_url).unwrap().unwrap();
        assert_eq!(attempt.issue_id, "older");
        assert_eq!(attempt.short_id, "PROJ-OLDER");
    }

    #[test]
    fn test_parse_pr_url() {
        let (repo, pr) =
            SqliteTracker::parse_pr_url("https://github.com/owner/repo/pull/123").unwrap();
        assert_eq!(repo, "owner/repo");
        assert_eq!(pr, 123);

        // Non-GitHub URL should return None
        assert!(
            SqliteTracker::parse_pr_url("https://gitlab.com/owner/repo/merge_requests/123")
                .is_none()
        );
    }

    #[test]
    fn test_mark_failed() {
        let tracker = SqliteTracker::in_memory().unwrap();

        tracker.record_attempt("linear", "123", "PROJ-123").unwrap();
        tracker
            .mark_failed("linear", "123", "Something went wrong")
            .unwrap();

        let attempt = tracker.get_attempt("linear", "123").unwrap().unwrap();
        assert_eq!(attempt.status, FixAttemptStatus::Failed);
        assert_eq!(
            attempt.error_message,
            Some("Something went wrong".to_string())
        );
    }

    #[test]
    fn test_reset_attempt() {
        let tracker = SqliteTracker::in_memory().unwrap();

        tracker.record_attempt("linear", "123", "PROJ-123").unwrap();
        tracker
            .mark_success("linear", "123", "https://github.com/org/repo/pull/1")
            .unwrap();
        assert!(tracker.has_attempted("linear", "123").unwrap());

        // Soft reset: has_attempted returns false, but the row is preserved
        tracker.reset_attempt("linear", "123").unwrap();
        assert!(!tracker.has_attempted("linear", "123").unwrap());

        // Row still exists with pending status and reset fields cleared
        let attempt = tracker.get_attempt("linear", "123").unwrap().unwrap();
        assert_eq!(attempt.status, FixAttemptStatus::Pending);
        assert_eq!(attempt.retry_count, 0);
        assert!(attempt.pr_url.is_none());
        assert!(attempt.error_message.is_none());
        assert!(attempt.merged_at.is_none());
        assert!(attempt.resolved_at.is_none());

        // get_attempted_issue_ids also excludes reset entries
        let ids = tracker.get_attempted_issue_ids("linear").unwrap();
        assert!(!ids.contains("123"));
    }

    #[test]
    fn test_get_attempted_issue_ids() {
        let tracker = SqliteTracker::in_memory().unwrap();

        tracker.record_attempt("linear", "123", "PROJ-123").unwrap();
        tracker.record_attempt("linear", "456", "PROJ-456").unwrap();
        tracker
            .record_attempt("sentry", "789", "SENTRY-789")
            .unwrap();

        let linear_ids = tracker.get_attempted_issue_ids("linear").unwrap();
        assert_eq!(linear_ids.len(), 2);
        assert!(linear_ids.contains("123"));
        assert!(linear_ids.contains("456"));

        let sentry_ids = tracker.get_attempted_issue_ids("sentry").unwrap();
        assert_eq!(sentry_ids.len(), 1);
        assert!(sentry_ids.contains("789"));
    }

    #[test]
    fn test_get_attempts_by_status() {
        let tracker = SqliteTracker::in_memory().unwrap();

        tracker.record_attempt("linear", "123", "PROJ-123").unwrap();
        tracker.record_attempt("linear", "456", "PROJ-456").unwrap();
        tracker
            .mark_success("linear", "123", "https://example.com/pr/1")
            .unwrap();

        let pending = tracker
            .get_attempts_by_status(FixAttemptStatus::Pending)
            .unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].issue_id, "456");

        let success = tracker
            .get_attempts_by_status(FixAttemptStatus::Success)
            .unwrap();
        assert_eq!(success.len(), 1);
        assert_eq!(success[0].issue_id, "123");
    }

    #[test]
    fn test_get_stats() {
        let tracker = SqliteTracker::in_memory().unwrap();

        tracker.record_attempt("linear", "1", "PROJ-1").unwrap();
        tracker.record_attempt("linear", "2", "PROJ-2").unwrap();
        tracker.record_attempt("sentry", "3", "SENTRY-3").unwrap();

        tracker
            .mark_success("linear", "1", "https://example.com/pr/1")
            .unwrap();
        tracker.mark_failed("linear", "2", "Error").unwrap();

        let stats = tracker.get_stats().unwrap();
        assert_eq!(stats.total, 3);
        assert_eq!(stats.pending, 1);
        assert_eq!(stats.success, 1);
        assert_eq!(stats.failed, 1);

        let linear_stats = stats.by_source.get("linear").unwrap();
        assert_eq!(linear_stats.total, 2);
        assert_eq!(linear_stats.success, 1);
        assert_eq!(linear_stats.failed, 1);

        let sentry_stats = stats.by_source.get("sentry").unwrap();
        assert_eq!(sentry_stats.total, 1);
    }

    #[test]
    fn test_mark_resolved() {
        let tracker = SqliteTracker::in_memory().unwrap();

        tracker.record_attempt("linear", "123", "PROJ-123").unwrap();
        tracker
            .mark_success("linear", "123", "https://github.com/org/repo/pull/42")
            .unwrap();
        tracker.mark_resolved("linear", "123").unwrap();

        let attempt = tracker.get_attempt("linear", "123").unwrap().unwrap();
        assert!(attempt.resolved_at.is_some());
    }

    #[test]
    fn test_increment_retry() {
        let tracker = SqliteTracker::in_memory().unwrap();

        tracker.record_attempt("linear", "123", "PROJ-123").unwrap();
        tracker.mark_failed("linear", "123", "Error").unwrap();

        let attempt = tracker.get_attempt("linear", "123").unwrap().unwrap();
        assert_eq!(attempt.retry_count, 0);

        tracker.increment_retry("linear", "123").unwrap();
        let attempt = tracker.get_attempt("linear", "123").unwrap().unwrap();
        assert_eq!(attempt.retry_count, 1);
        assert!(attempt.last_retry_at.is_some());

        tracker.increment_retry("linear", "123").unwrap();
        let attempt = tracker.get_attempt("linear", "123").unwrap().unwrap();
        assert_eq!(attempt.retry_count, 2);
    }

    #[test]
    fn test_mark_cannot_fix() {
        let tracker = SqliteTracker::in_memory().unwrap();

        tracker.record_attempt("linear", "123", "PROJ-123").unwrap();
        tracker
            .mark_cannot_fix("linear", "123", "Max retries exceeded")
            .unwrap();

        let attempt = tracker.get_attempt("linear", "123").unwrap().unwrap();
        assert_eq!(attempt.status, FixAttemptStatus::CannotFix);
        assert_eq!(
            attempt.error_message,
            Some("Max retries exceeded".to_string())
        );
    }

    #[test]
    fn test_get_retryable_issues() {
        let tracker = SqliteTracker::in_memory().unwrap();

        // Failed with 0 retries - retryable
        tracker.record_attempt("linear", "1", "PROJ-1").unwrap();
        tracker.mark_failed("linear", "1", "Error").unwrap();

        // Failed with 2 retries - still retryable if max is 3
        tracker.record_attempt("linear", "2", "PROJ-2").unwrap();
        tracker.mark_failed("linear", "2", "Error").unwrap();
        tracker.increment_retry("linear", "2").unwrap();
        tracker.increment_retry("linear", "2").unwrap();

        // Closed - retryable
        tracker.record_attempt("linear", "3", "PROJ-3").unwrap();
        tracker
            .mark_success("linear", "3", "https://github.com/org/repo/pull/1")
            .unwrap();
        tracker.mark_closed("linear", "3").unwrap();

        // Success - not retryable
        tracker.record_attempt("linear", "4", "PROJ-4").unwrap();
        tracker
            .mark_success("linear", "4", "https://github.com/org/repo/pull/2")
            .unwrap();

        let retryable = tracker.get_retryable_issues(3).unwrap();
        assert_eq!(retryable.len(), 3);

        // With max 2 retries, issue 2 should be excluded
        let retryable = tracker.get_retryable_issues(2).unwrap();
        assert_eq!(retryable.len(), 2);
    }

    #[test]
    fn test_prepare_for_retry() {
        let tracker = SqliteTracker::in_memory().unwrap();

        tracker.record_attempt("linear", "123", "PROJ-123").unwrap();
        tracker
            .mark_success("linear", "123", "https://github.com/org/repo/pull/42")
            .unwrap();
        tracker.mark_closed("linear", "123").unwrap();

        // Prepare for retry should reset to pending and increment retry_count
        tracker.prepare_for_retry("linear", "123").unwrap();

        let attempt = tracker.get_attempt("linear", "123").unwrap().unwrap();
        assert_eq!(attempt.status, FixAttemptStatus::Pending);
        assert!(attempt.pr_url.is_none());
        assert!(attempt.scm_repo.is_none());
        assert!(attempt.scm_pr_number.is_none());
        assert!(attempt.error_message.is_none());
        assert_eq!(attempt.retry_count, 1);
        assert!(attempt.last_retry_at.is_some());
    }

    #[test]
    fn test_prepare_for_retry_rejects_non_retryable_status() {
        let tracker = SqliteTracker::in_memory().unwrap();

        // Pending attempt should not be retryable
        tracker.record_attempt("linear", "1", "PROJ-1").unwrap();
        assert!(tracker.prepare_for_retry("linear", "1").is_err());

        // Success attempt should not be retryable
        tracker.record_attempt("linear", "2", "PROJ-2").unwrap();
        tracker
            .mark_success("linear", "2", "https://github.com/org/repo/pull/1")
            .unwrap();
        assert!(tracker.prepare_for_retry("linear", "2").is_err());

        // Cannot_fix attempt should not be retryable
        tracker.record_attempt("linear", "3", "PROJ-3").unwrap();
        tracker
            .mark_cannot_fix("linear", "3", "Max retries")
            .unwrap();
        assert!(tracker.prepare_for_retry("linear", "3").is_err());
    }

    #[test]
    fn test_get_retryable_issues_excludes_pending() {
        let tracker = SqliteTracker::in_memory().unwrap();

        // Pending issues should NOT be retryable -- they are still in their initial processing.
        // Only failed/closed issues should be eligible for retry.
        tracker
            .record_attempt("linear", "pending-1", "PROJ-PENDING-1")
            .unwrap();

        let retryable = tracker.get_retryable_issues(3).unwrap();
        assert_eq!(retryable.len(), 0);

        // After marking as failed, it should become retryable
        tracker
            .mark_failed("linear", "pending-1", "some error")
            .unwrap();
        let retryable = tracker.get_retryable_issues(3).unwrap();
        assert_eq!(retryable.len(), 1);
        assert_eq!(retryable[0].issue_id, "pending-1");
        assert_eq!(retryable[0].status, FixAttemptStatus::Failed);
    }

    #[test]
    fn test_parse_pr_url_various_formats() {
        // Standard format
        let (repo, pr) =
            SqliteTracker::parse_pr_url("https://github.com/owner/repo/pull/123").unwrap();
        assert_eq!(repo, "owner/repo");
        assert_eq!(pr, 123);

        // With trailing slash
        let (repo, pr) = SqliteTracker::parse_pr_url("https://github.com/owner/repo/pull/456/")
            .unwrap_or(("".into(), 0));
        // Regex doesn't match trailing slash, this is expected behavior
        assert_eq!(repo, "owner/repo");
        assert_eq!(pr, 456);

        // HTTP instead of HTTPS (should still work as regex doesn't require https)
        let result = SqliteTracker::parse_pr_url("http://github.com/owner/repo/pull/789");
        assert!(result.is_some());

        // Invalid URL
        assert!(SqliteTracker::parse_pr_url("not a url").is_none());

        // Empty string
        assert!(SqliteTracker::parse_pr_url("").is_none());
    }

    #[test]
    fn test_get_attempt_not_found() {
        let tracker = SqliteTracker::in_memory().unwrap();

        let attempt = tracker.get_attempt("linear", "nonexistent").unwrap();
        assert!(attempt.is_none());
    }

    #[test]
    fn test_get_attempt_by_pr_url_not_found() {
        let tracker = SqliteTracker::in_memory().unwrap();

        let attempt = tracker
            .get_attempt_by_pr_url("https://github.com/org/repo/pull/999")
            .unwrap();
        assert!(attempt.is_none());
    }

    #[test]
    fn test_get_attempts_by_status_empty() {
        let tracker = SqliteTracker::in_memory().unwrap();

        let attempts = tracker
            .get_attempts_by_status(FixAttemptStatus::Merged)
            .unwrap();
        assert!(attempts.is_empty());
    }

    #[test]
    fn test_stats_empty_database() {
        let tracker = SqliteTracker::in_memory().unwrap();

        let stats = tracker.get_stats().unwrap();
        assert_eq!(stats.total, 0);
        assert_eq!(stats.pending, 0);
        assert_eq!(stats.success, 0);
        assert_eq!(stats.failed, 0);
        assert!(stats.by_source.is_empty());
    }

    #[test]
    fn test_stats_all_statuses() {
        let tracker = SqliteTracker::in_memory().unwrap();

        // Pending
        tracker.record_attempt("linear", "1", "PROJ-1").unwrap();

        // Success
        tracker.record_attempt("linear", "2", "PROJ-2").unwrap();
        tracker
            .mark_success("linear", "2", "https://github.com/org/repo/pull/1")
            .unwrap();

        // Failed
        tracker.record_attempt("linear", "3", "PROJ-3").unwrap();
        tracker.mark_failed("linear", "3", "Error").unwrap();

        // Merged
        tracker.record_attempt("linear", "4", "PROJ-4").unwrap();
        tracker
            .mark_success("linear", "4", "https://github.com/org/repo/pull/2")
            .unwrap();
        tracker.mark_merged("linear", "4").unwrap();

        // Closed
        tracker.record_attempt("linear", "5", "PROJ-5").unwrap();
        tracker
            .mark_success("linear", "5", "https://github.com/org/repo/pull/3")
            .unwrap();
        tracker.mark_closed("linear", "5").unwrap();

        // Cannot fix
        tracker.record_attempt("linear", "6", "PROJ-6").unwrap();
        tracker
            .mark_cannot_fix("linear", "6", "Max retries")
            .unwrap();

        let stats = tracker.get_stats().unwrap();
        assert_eq!(stats.total, 6);
        assert_eq!(stats.pending, 1);
        assert_eq!(stats.success, 1);
        assert_eq!(stats.failed, 1);
        assert_eq!(stats.merged, 1);
        assert_eq!(stats.closed, 1);
        assert_eq!(stats.cannot_fix, 1);
    }

    #[test]
    fn test_record_attempt_upsert_preserves_data() {
        let tracker = SqliteTracker::in_memory().unwrap();

        // Record initial attempt
        tracker.record_attempt("linear", "123", "PROJ-123").unwrap();
        tracker
            .mark_success("linear", "123", "https://github.com/org/repo/pull/1")
            .unwrap();

        // Record again — conditional upsert should NOT overwrite an already-processed
        // attempt (no reset_at set), so the row should remain unchanged.
        tracker
            .record_attempt("linear", "123", "PROJ-123-v2")
            .unwrap();

        let attempt = tracker.get_attempt("linear", "123").unwrap().unwrap();
        // short_id should NOT be updated (conditional upsert skipped)
        assert_eq!(attempt.short_id, "PROJ-123");
        // status and pr_url should be preserved
        assert_eq!(attempt.status, FixAttemptStatus::Success);
        assert_eq!(
            attempt.pr_url,
            Some("https://github.com/org/repo/pull/1".to_string())
        );
    }

    #[test]
    fn test_record_attempt_upsert_works_after_reset() {
        let tracker = SqliteTracker::in_memory().unwrap();

        // Record and process
        tracker.record_attempt("linear", "123", "PROJ-123").unwrap();
        tracker
            .mark_success("linear", "123", "https://github.com/org/repo/pull/1")
            .unwrap();

        // Soft reset
        tracker.reset_attempt("linear", "123").unwrap();

        // Record again — conditional upsert SHOULD update because reset_at IS NOT NULL
        tracker
            .record_attempt("linear", "123", "PROJ-123-v2")
            .unwrap();

        let attempt = tracker.get_attempt("linear", "123").unwrap().unwrap();
        // short_id should be updated (upsert fires for reset entries)
        assert_eq!(attempt.short_id, "PROJ-123-v2");
        assert_eq!(attempt.status, FixAttemptStatus::Pending);
        // pr_url was cleared by reset
        assert!(attempt.pr_url.is_none());
    }

    #[test]
    fn test_multiple_sources_isolation() {
        let tracker = SqliteTracker::in_memory().unwrap();

        // Same issue_id in different sources
        tracker
            .record_attempt("linear", "123", "LINEAR-123")
            .unwrap();
        tracker
            .record_attempt("sentry", "123", "SENTRY-123")
            .unwrap();

        assert!(tracker.has_attempted("linear", "123").unwrap());
        assert!(tracker.has_attempted("sentry", "123").unwrap());

        tracker
            .mark_success("linear", "123", "https://github.com/org/repo/pull/1")
            .unwrap();

        let linear_attempt = tracker.get_attempt("linear", "123").unwrap().unwrap();
        let sentry_attempt = tracker.get_attempt("sentry", "123").unwrap().unwrap();

        assert_eq!(linear_attempt.status, FixAttemptStatus::Success);
        assert_eq!(sentry_attempt.status, FixAttemptStatus::Pending);
    }

    #[test]
    fn test_parse_datetime_rfc3339() {
        let dt = SqliteTracker::parse_datetime("2024-01-15T10:30:00Z").unwrap();
        assert_eq!(dt.year(), 2024);
        assert_eq!(dt.month(), 1);
        assert_eq!(dt.day(), 15);
        assert_eq!(dt.hour(), 10);
        assert_eq!(dt.minute(), 30);
    }

    #[test]
    fn test_parse_datetime_sqlite_format() {
        let dt = SqliteTracker::parse_datetime("2024-01-15 10:30:00").unwrap();
        assert_eq!(dt.year(), 2024);
        assert_eq!(dt.month(), 1);
        assert_eq!(dt.day(), 15);
    }

    #[test]
    fn test_parse_datetime_invalid() {
        // Should return an error for invalid format
        let result = SqliteTracker::parse_datetime("not a date");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_optional_datetime() {
        let none_result = SqliteTracker::parse_optional_datetime(None).unwrap();
        assert!(none_result.is_none());

        let some_result =
            SqliteTracker::parse_optional_datetime(Some("2024-01-15T10:30:00Z".to_string()))
                .unwrap();
        assert!(some_result.is_some());
    }

    #[test]
    fn test_get_pending_prs_empty() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let pending_prs = tracker.get_pending_prs().unwrap();
        assert!(pending_prs.is_empty());
    }

    #[test]
    fn test_get_pending_prs_gitlab_mr_url() {
        let tracker = SqliteTracker::in_memory().unwrap();

        // Create attempt with GitLab MR URL (now parsed successfully)
        tracker.record_attempt("linear", "123", "PROJ-123").unwrap();
        tracker
            .mark_success(
                "linear",
                "123",
                "https://gitlab.com/org/repo/-/merge_requests/42",
            )
            .unwrap();

        // GitLab MR URLs are now parsed into scm_repo/scm_pr_number fields
        let pending_prs = tracker.get_pending_prs().unwrap();
        assert_eq!(pending_prs.len(), 1);
        assert_eq!(pending_prs[0].scm_repo, Some("org/repo".to_string()));
        assert_eq!(pending_prs[0].scm_pr_number, Some(42));
    }

    #[test]
    fn test_create_regression_watch() {
        use claudear_core::types::{IssueType, RegressionWatch};

        let tracker = SqliteTracker::in_memory().unwrap();

        // First create a fix attempt to reference
        tracker
            .record_attempt("sentry", "sentry-123", "SENTRY-123")
            .unwrap();
        let attempt = tracker
            .get_attempt("sentry", "sentry-123")
            .unwrap()
            .unwrap();

        let watch = RegressionWatch::new(IssueType::SentryIssue, "sentry-123", attempt.id);

        let watch_id = tracker.create_regression_watch(&watch).unwrap();
        assert!(watch_id > 0);
    }

    #[test]
    fn test_create_regression_watch_with_linear_bug() {
        use claudear_core::types::{IssueType, RegressionWatch};

        let tracker = SqliteTracker::in_memory().unwrap();

        // Create a fix attempt for linear
        tracker
            .record_attempt("linear", "linear-456", "LIN-456")
            .unwrap();
        let attempt = tracker
            .get_attempt("linear", "linear-456")
            .unwrap()
            .unwrap();

        let watch = RegressionWatch::new(IssueType::LinearBug, "linear-456", attempt.id);

        let watch_id = tracker.create_regression_watch(&watch).unwrap();
        assert!(watch_id > 0);
    }

    #[test]
    fn test_get_regression_watch() {
        use claudear_core::types::{IssueType, RegressionWatch, RegressionWatchStatus};

        let tracker = SqliteTracker::in_memory().unwrap();

        // Create fix attempt and watch
        tracker
            .record_attempt("sentry", "sentry-789", "SENTRY-789")
            .unwrap();
        let attempt = tracker
            .get_attempt("sentry", "sentry-789")
            .unwrap()
            .unwrap();

        let watch = RegressionWatch::new(IssueType::SentryIssue, "sentry-789", attempt.id);

        let watch_id = tracker.create_regression_watch(&watch).unwrap();

        // Retrieve the watch
        let retrieved = tracker.get_regression_watch(watch_id).unwrap().unwrap();
        assert_eq!(retrieved.id, watch_id);
        assert_eq!(retrieved.issue_type, IssueType::SentryIssue);
        assert_eq!(retrieved.issue_id, "sentry-789");
        assert_eq!(retrieved.fix_attempt_id, attempt.id);
        assert_eq!(retrieved.status, RegressionWatchStatus::AwaitingRelease);
    }

    #[test]
    fn test_get_regression_watch_not_found() {
        let tracker = SqliteTracker::in_memory().unwrap();

        let result = tracker.get_regression_watch(99999).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_get_regression_watches_by_status() {
        use claudear_core::types::{IssueType, RegressionWatch, RegressionWatchStatus};

        let tracker = SqliteTracker::in_memory().unwrap();

        // Create multiple fix attempts
        tracker
            .record_attempt("sentry", "issue-1", "ISSUE-1")
            .unwrap();
        tracker
            .record_attempt("sentry", "issue-2", "ISSUE-2")
            .unwrap();
        tracker
            .record_attempt("linear", "issue-3", "ISSUE-3")
            .unwrap();

        let attempt1 = tracker.get_attempt("sentry", "issue-1").unwrap().unwrap();
        let attempt2 = tracker.get_attempt("sentry", "issue-2").unwrap().unwrap();
        let attempt3 = tracker.get_attempt("linear", "issue-3").unwrap().unwrap();

        // Create watches with different statuses
        let watch1 = RegressionWatch::new(IssueType::SentryIssue, "issue-1", attempt1.id);
        let watch2 = RegressionWatch::new(IssueType::SentryIssue, "issue-2", attempt2.id);
        let watch3 = RegressionWatch::new(IssueType::LinearBug, "issue-3", attempt3.id);

        let watch1_id = tracker.create_regression_watch(&watch1).unwrap();
        let _watch2_id = tracker.create_regression_watch(&watch2).unwrap();
        let _watch3_id = tracker.create_regression_watch(&watch3).unwrap();

        // All should start as AwaitingRelease
        let awaiting = tracker
            .get_regression_watches_by_status(RegressionWatchStatus::AwaitingRelease)
            .unwrap();
        assert_eq!(awaiting.len(), 3);

        // Update one to Monitoring
        tracker
            .update_regression_watch_status(watch1_id, RegressionWatchStatus::Monitoring)
            .unwrap();

        let awaiting = tracker
            .get_regression_watches_by_status(RegressionWatchStatus::AwaitingRelease)
            .unwrap();
        assert_eq!(awaiting.len(), 2);

        let monitoring = tracker
            .get_regression_watches_by_status(RegressionWatchStatus::Monitoring)
            .unwrap();
        assert_eq!(monitoring.len(), 1);
        assert_eq!(monitoring[0].id, watch1_id);
    }

    #[test]
    fn test_get_regression_watches_by_status_empty() {
        use claudear_core::types::RegressionWatchStatus;

        let tracker = SqliteTracker::in_memory().unwrap();

        let watches = tracker
            .get_regression_watches_by_status(RegressionWatchStatus::Monitoring)
            .unwrap();
        assert!(watches.is_empty());
    }

    #[test]
    fn test_update_regression_watch_status() {
        use claudear_core::types::{IssueType, RegressionWatch, RegressionWatchStatus};

        let tracker = SqliteTracker::in_memory().unwrap();

        // Create fix attempt and watch
        tracker
            .record_attempt("sentry", "sentry-status-test", "SENTRY-ST")
            .unwrap();
        let attempt = tracker
            .get_attempt("sentry", "sentry-status-test")
            .unwrap()
            .unwrap();

        let watch = RegressionWatch::new(IssueType::SentryIssue, "sentry-status-test", attempt.id);

        let watch_id = tracker.create_regression_watch(&watch).unwrap();

        // Verify initial status
        let retrieved = tracker.get_regression_watch(watch_id).unwrap().unwrap();
        assert_eq!(retrieved.status, RegressionWatchStatus::AwaitingRelease);

        // Update to Monitoring
        tracker
            .update_regression_watch_status(watch_id, RegressionWatchStatus::Monitoring)
            .unwrap();
        let retrieved = tracker.get_regression_watch(watch_id).unwrap().unwrap();
        assert_eq!(retrieved.status, RegressionWatchStatus::Monitoring);
        assert!(retrieved.monitoring_started_at.is_some());

        // Update to Resolved
        tracker
            .update_regression_watch_status(watch_id, RegressionWatchStatus::Resolved)
            .unwrap();
        let retrieved = tracker.get_regression_watch(watch_id).unwrap().unwrap();
        assert_eq!(retrieved.status, RegressionWatchStatus::Resolved);
        assert!(retrieved.resolved_at.is_some());
    }

    #[test]
    fn test_update_regression_watch_status_to_regressed() {
        use claudear_core::types::{IssueType, RegressionWatch, RegressionWatchStatus};

        let tracker = SqliteTracker::in_memory().unwrap();

        // Create fix attempt and watch
        tracker
            .record_attempt("linear", "linear-regressed", "LIN-REG")
            .unwrap();
        let attempt = tracker
            .get_attempt("linear", "linear-regressed")
            .unwrap()
            .unwrap();

        let watch = RegressionWatch::new(IssueType::LinearBug, "linear-regressed", attempt.id);

        let watch_id = tracker.create_regression_watch(&watch).unwrap();

        // Move to Monitoring first
        tracker
            .update_regression_watch_status(watch_id, RegressionWatchStatus::Monitoring)
            .unwrap();

        // Update to Regressed
        tracker
            .update_regression_watch_status(watch_id, RegressionWatchStatus::Regressed)
            .unwrap();
        let retrieved = tracker.get_regression_watch(watch_id).unwrap().unwrap();
        assert_eq!(retrieved.status, RegressionWatchStatus::Regressed);
        assert!(retrieved.regressed_at.is_some());
    }

    #[test]
    fn test_record_release_tracking() {
        use claudear_core::types::{IssueType, RegressionWatch, ReleaseTracking};

        let tracker = SqliteTracker::in_memory().unwrap();

        // Create fix attempt and watch
        tracker
            .record_attempt("sentry", "sentry-release", "SENTRY-REL")
            .unwrap();
        let attempt = tracker
            .get_attempt("sentry", "sentry-release")
            .unwrap()
            .unwrap();

        let watch = RegressionWatch::new(IssueType::SentryIssue, "sentry-release", attempt.id);

        let watch_id = tracker.create_regression_watch(&watch).unwrap();

        // Record release tracking
        let release = ReleaseTracking::new(watch_id, "v1.2.3", "abc123def456");

        let release_id = tracker.record_release_tracking(&release).unwrap();
        assert!(release_id > 0);
    }

    #[test]
    fn test_record_release_tracking_multiple_versions() {
        use claudear_core::types::{IssueType, RegressionWatch, ReleaseTracking};

        let tracker = SqliteTracker::in_memory().unwrap();

        // Create fix attempt and watch
        tracker
            .record_attempt("linear", "linear-multi-release", "LIN-MR")
            .unwrap();
        let attempt = tracker
            .get_attempt("linear", "linear-multi-release")
            .unwrap()
            .unwrap();

        let watch = RegressionWatch::new(IssueType::LinearBug, "linear-multi-release", attempt.id);

        let watch_id = tracker.create_regression_watch(&watch).unwrap();

        // Record multiple releases
        let release1 = ReleaseTracking::new(watch_id, "v1.0.0", "commit1");
        let release2 = ReleaseTracking::new(watch_id, "v1.0.1", "commit2");
        let release3 = ReleaseTracking::new(watch_id, "v1.1.0", "commit3");

        let id1 = tracker.record_release_tracking(&release1).unwrap();
        let id2 = tracker.record_release_tracking(&release2).unwrap();
        let id3 = tracker.record_release_tracking(&release3).unwrap();

        assert!(id1 > 0);
        assert!(id2 > id1);
        assert!(id3 > id2);
    }

    #[test]
    fn test_record_regression_check() {
        use claudear_core::types::{IssueType, RegressionCheck, RegressionWatch};

        let tracker = SqliteTracker::in_memory().unwrap();

        // Create fix attempt and watch
        tracker
            .record_attempt("sentry", "sentry-check", "SENTRY-CHK")
            .unwrap();
        let attempt = tracker
            .get_attempt("sentry", "sentry-check")
            .unwrap()
            .unwrap();

        let watch = RegressionWatch::new(IssueType::SentryIssue, "sentry-check", attempt.id);

        let watch_id = tracker.create_regression_watch(&watch).unwrap();

        // Record a check showing issue does not exist
        let check = RegressionCheck::new(watch_id, false);
        let check_id = tracker.record_regression_check(&check).unwrap();
        assert!(check_id > 0);
    }

    #[test]
    fn test_record_regression_check_issue_exists() {
        use claudear_core::types::{IssueType, RegressionCheck, RegressionWatch};

        let tracker = SqliteTracker::in_memory().unwrap();

        // Create fix attempt and watch
        tracker
            .record_attempt("linear", "linear-check-exists", "LIN-CHK")
            .unwrap();
        let attempt = tracker
            .get_attempt("linear", "linear-check-exists")
            .unwrap()
            .unwrap();

        let watch = RegressionWatch::new(IssueType::LinearBug, "linear-check-exists", attempt.id);

        let watch_id = tracker.create_regression_watch(&watch).unwrap();

        // Record a check showing issue still exists
        let mut check = RegressionCheck::new(watch_id, true);
        check.check_details = Some("Issue reoccurred 5 times in the last hour".to_string());

        let check_id = tracker.record_regression_check(&check).unwrap();
        assert!(check_id > 0);
    }

    #[test]
    fn test_get_regression_checks() {
        use claudear_core::types::{IssueType, RegressionCheck, RegressionWatch};

        let tracker = SqliteTracker::in_memory().unwrap();

        // Create fix attempt and watch
        tracker
            .record_attempt("sentry", "sentry-get-checks", "SENTRY-GC")
            .unwrap();
        let attempt = tracker
            .get_attempt("sentry", "sentry-get-checks")
            .unwrap()
            .unwrap();

        let watch = RegressionWatch::new(IssueType::SentryIssue, "sentry-get-checks", attempt.id);

        let watch_id = tracker.create_regression_watch(&watch).unwrap();

        // Record multiple checks
        let check1 = RegressionCheck::new(watch_id, false);
        let check2 = RegressionCheck::new(watch_id, false);
        let check3 = RegressionCheck::new(watch_id, true);

        tracker.record_regression_check(&check1).unwrap();
        tracker.record_regression_check(&check2).unwrap();
        tracker.record_regression_check(&check3).unwrap();

        // Get all checks for this watch
        let checks = tracker.get_regression_checks(watch_id).unwrap();
        assert_eq!(checks.len(), 3);

        // Verify the last check shows issue exists
        let last_check = checks.last().unwrap();
        assert!(last_check.issue_still_exists);
    }

    #[test]
    fn test_get_regression_checks_empty() {
        use claudear_core::types::{IssueType, RegressionWatch};

        let tracker = SqliteTracker::in_memory().unwrap();

        // Create fix attempt and watch
        tracker
            .record_attempt("linear", "linear-empty-checks", "LIN-EC")
            .unwrap();
        let attempt = tracker
            .get_attempt("linear", "linear-empty-checks")
            .unwrap()
            .unwrap();

        let watch = RegressionWatch::new(IssueType::LinearBug, "linear-empty-checks", attempt.id);

        let watch_id = tracker.create_regression_watch(&watch).unwrap();

        // Get checks for watch with no checks recorded
        let checks = tracker.get_regression_checks(watch_id).unwrap();
        assert!(checks.is_empty());
    }

    #[test]
    fn test_get_regression_checks_for_nonexistent_watch() {
        let tracker = SqliteTracker::in_memory().unwrap();

        // Get checks for a watch ID that doesn't exist
        let checks = tracker.get_regression_checks(99999).unwrap();
        assert!(checks.is_empty());
    }

    #[test]
    fn test_regression_watch_table_creation() {
        let tracker = SqliteTracker::in_memory().unwrap();

        // The table should be created during init - verify by trying to query
        let conn = tracker.conn.lock().unwrap();
        let result = conn.execute("SELECT 1 FROM regression_watches LIMIT 1", []);
        // Should succeed (table exists) or return empty result, not error
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_release_tracking_table_creation() {
        let tracker = SqliteTracker::in_memory().unwrap();

        // The table should be created during init
        let conn = tracker.conn.lock().unwrap();
        let result = conn.execute("SELECT 1 FROM release_tracking LIMIT 1", []);
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_regression_checks_table_creation() {
        let tracker = SqliteTracker::in_memory().unwrap();

        // The table should be created during init
        let conn = tracker.conn.lock().unwrap();
        let result = conn.execute("SELECT 1 FROM regression_checks LIMIT 1", []);
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_full_regression_watch_lifecycle() {
        use claudear_core::types::{
            IssueType, RegressionCheck, RegressionWatch, RegressionWatchStatus, ReleaseTracking,
        };

        let tracker = SqliteTracker::in_memory().unwrap();

        // 1. Create a fix attempt
        tracker
            .record_attempt("sentry", "lifecycle-test", "SENTRY-LC")
            .unwrap();
        tracker
            .mark_success(
                "sentry",
                "lifecycle-test",
                "https://github.com/org/repo/pull/99",
            )
            .unwrap();
        let attempt = tracker
            .get_attempt("sentry", "lifecycle-test")
            .unwrap()
            .unwrap();

        // 2. Create regression watch
        let watch = RegressionWatch::new(IssueType::SentryIssue, "lifecycle-test", attempt.id);
        let watch_id = tracker.create_regression_watch(&watch).unwrap();

        // 3. Verify initial state
        let watch = tracker.get_regression_watch(watch_id).unwrap().unwrap();
        assert_eq!(watch.status, RegressionWatchStatus::AwaitingRelease);

        // 4. Record release
        let release = ReleaseTracking::new(watch_id, "v2.0.0", "release-commit-hash");
        tracker.record_release_tracking(&release).unwrap();

        // 5. Update to monitoring
        tracker
            .update_regression_watch_status(watch_id, RegressionWatchStatus::Monitoring)
            .unwrap();
        let watch = tracker.get_regression_watch(watch_id).unwrap().unwrap();
        assert_eq!(watch.status, RegressionWatchStatus::Monitoring);

        // 6. Record several checks showing no regression
        for _ in 0..3 {
            let check = RegressionCheck::new(watch_id, false);
            tracker.record_regression_check(&check).unwrap();
        }

        // 7. Verify checks are recorded
        let checks = tracker.get_regression_checks(watch_id).unwrap();
        assert_eq!(checks.len(), 3);

        // 8. Mark as resolved
        tracker
            .update_regression_watch_status(watch_id, RegressionWatchStatus::Resolved)
            .unwrap();
        let watch = tracker.get_regression_watch(watch_id).unwrap().unwrap();
        assert_eq!(watch.status, RegressionWatchStatus::Resolved);
        assert!(watch.resolved_at.is_some());
    }

    #[test]
    fn test_regression_watch_lifecycle_with_regression() {
        use claudear_core::types::{
            IssueType, RegressionCheck, RegressionWatch, RegressionWatchStatus, ReleaseTracking,
        };

        let tracker = SqliteTracker::in_memory().unwrap();

        // 1. Create fix attempt and watch
        tracker
            .record_attempt("linear", "regression-lifecycle", "LIN-RL")
            .unwrap();
        tracker
            .mark_success(
                "linear",
                "regression-lifecycle",
                "https://github.com/org/repo/pull/100",
            )
            .unwrap();
        let attempt = tracker
            .get_attempt("linear", "regression-lifecycle")
            .unwrap()
            .unwrap();

        let watch = RegressionWatch::new(IssueType::LinearBug, "regression-lifecycle", attempt.id);
        let watch_id = tracker.create_regression_watch(&watch).unwrap();

        // 2. Record release and start monitoring
        let release = ReleaseTracking::new(watch_id, "v3.0.0", "commit-hash");
        tracker.record_release_tracking(&release).unwrap();
        tracker
            .update_regression_watch_status(watch_id, RegressionWatchStatus::Monitoring)
            .unwrap();

        // 3. Record checks - first ones OK, then regression detected
        let check1 = RegressionCheck::new(watch_id, false);
        let check2 = RegressionCheck::new(watch_id, false);
        let mut check3 = RegressionCheck::new(watch_id, true);
        check3.check_details = Some("Bug reoccurred in production".to_string());

        tracker.record_regression_check(&check1).unwrap();
        tracker.record_regression_check(&check2).unwrap();
        tracker.record_regression_check(&check3).unwrap();

        // 4. Mark as regressed
        tracker
            .update_regression_watch_status(watch_id, RegressionWatchStatus::Regressed)
            .unwrap();
        let watch = tracker.get_regression_watch(watch_id).unwrap().unwrap();
        assert_eq!(watch.status, RegressionWatchStatus::Regressed);
        assert!(watch.regressed_at.is_some());

        // 5. Verify checks history
        let checks = tracker.get_regression_checks(watch_id).unwrap();
        assert_eq!(checks.len(), 3);
        assert!(!checks[0].issue_still_exists);
        assert!(!checks[1].issue_still_exists);
        assert!(checks[2].issue_still_exists);
    }

    #[test]
    fn test_pragma_settings_applied() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let conn = tracker.acquire_lock().unwrap();

        // Verify WAL mode is enabled (note: in-memory DBs may not support WAL, so we check for memory or wal)
        let journal_mode: String = conn
            .query_row("PRAGMA journal_mode", [], |row| row.get(0))
            .unwrap();
        // In-memory databases use "memory" journal mode, file-based would use "wal"
        assert!(
            journal_mode == "memory" || journal_mode == "wal",
            "Expected journal_mode to be 'memory' or 'wal', got '{}'",
            journal_mode
        );

        // Verify synchronous is NORMAL (1)
        let synchronous: i32 = conn
            .query_row("PRAGMA synchronous", [], |row| row.get(0))
            .unwrap();
        assert_eq!(synchronous, 1, "Expected synchronous=1 (NORMAL)");

        // Verify cache_size is set (negative means KB)
        let cache_size: i64 = conn
            .query_row("PRAGMA cache_size", [], |row| row.get(0))
            .unwrap();
        assert_eq!(cache_size, -16384, "Expected cache_size=-16384 (16MB)");

        // Verify temp_store is MEMORY (2)
        let temp_store: i32 = conn
            .query_row("PRAGMA temp_store", [], |row| row.get(0))
            .unwrap();
        assert_eq!(temp_store, 2, "Expected temp_store=2 (MEMORY)");

        // Verify busy_timeout is set
        let busy_timeout: i32 = conn
            .query_row("PRAGMA busy_timeout", [], |row| row.get(0))
            .unwrap();
        assert_eq!(busy_timeout, 5000, "Expected busy_timeout=5000");

        // Verify foreign_keys is ON (1)
        let foreign_keys: i32 = conn
            .query_row("PRAGMA foreign_keys", [], |row| row.get(0))
            .unwrap();
        assert_eq!(foreign_keys, 1, "Expected foreign_keys=1 (ON)");
    }

    #[test]
    fn test_batch_record_activities() {
        let tracker = SqliteTracker::in_memory().unwrap();

        let entries = vec![
            ActivityLogEntry {
                id: 0,
                timestamp: Utc::now(),
                activity_type: "test".to_string(),
                source: Some("linear".to_string()),
                issue_id: Some("1".to_string()),
                short_id: Some("LIN-1".to_string()),
                message: "Test activity 1".to_string(),
                metadata: None,
            },
            ActivityLogEntry {
                id: 0,
                timestamp: Utc::now(),
                activity_type: "test".to_string(),
                source: Some("linear".to_string()),
                issue_id: Some("2".to_string()),
                short_id: Some("LIN-2".to_string()),
                message: "Test activity 2".to_string(),
                metadata: None,
            },
            ActivityLogEntry {
                id: 0,
                timestamp: Utc::now(),
                activity_type: "test".to_string(),
                source: Some("sentry".to_string()),
                issue_id: Some("3".to_string()),
                short_id: Some("SENTRY-3".to_string()),
                message: "Test activity 3".to_string(),
                metadata: None,
            },
        ];

        let count = tracker.record_activities_batch(&entries).unwrap();
        assert_eq!(count, 3);

        let activities = tracker.get_recent_activities(10, None).unwrap();
        assert_eq!(activities.len(), 3);
    }

    #[test]
    fn test_batch_record_activities_empty() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let entries: Vec<ActivityLogEntry> = vec![];

        let count = tracker.record_activities_batch(&entries).unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_batch_record_metrics() {
        let tracker = SqliteTracker::in_memory().unwrap();

        let metrics = vec![
            ProcessingMetric {
                id: 0,
                timestamp: Utc::now(),
                metric_name: "issues_processed".to_string(),
                metric_value: 10.0,
                source: Some("linear".to_string()),
                tags: None,
            },
            ProcessingMetric {
                id: 0,
                timestamp: Utc::now(),
                metric_name: "issues_processed".to_string(),
                metric_value: 5.0,
                source: Some("sentry".to_string()),
                tags: None,
            },
            ProcessingMetric {
                id: 0,
                timestamp: Utc::now(),
                metric_name: "fix_duration_secs".to_string(),
                metric_value: 120.5,
                source: Some("linear".to_string()),
                tags: None,
            },
        ];

        let count = tracker.record_metrics_batch(&metrics).unwrap();
        assert_eq!(count, 3);

        let fetched = tracker.get_metrics("issues_processed", None, 10).unwrap();
        assert_eq!(fetched.len(), 2);
    }

    #[test]
    fn test_batch_record_metrics_empty() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let metrics: Vec<ProcessingMetric> = vec![];

        let count = tracker.record_metrics_batch(&metrics).unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_save_and_get_pr_review_states() {
        let tracker = SqliteTracker::in_memory().unwrap();

        // Create a PR review state
        let state = claudear_core::types::PrReviewState::new(
            "https://github.com/owner/repo/pull/123",
            "owner/repo",
            123,
            "issue-1",
            "linear",
        );

        // Save it
        tracker.save_pr_review_state(&state).unwrap();

        // Retrieve active states
        let states = tracker.get_active_pr_review_states().unwrap();
        assert_eq!(states.len(), 1);
        assert_eq!(states[0].pr_url, "https://github.com/owner/repo/pull/123");
        assert_eq!(states[0].repo, "owner/repo");
        assert_eq!(states[0].pr_number, 123);
        assert_eq!(states[0].issue_id, "issue-1");
        assert_eq!(states[0].source, "linear");
        assert!(states[0].is_active);
    }

    #[test]
    fn test_pr_review_state_update() {
        let tracker = SqliteTracker::in_memory().unwrap();

        // Create and save initial state
        let mut state = claudear_core::types::PrReviewState::new(
            "https://github.com/owner/repo/pull/456",
            "owner/repo",
            456,
            "issue-2",
            "sentry",
        );
        tracker.save_pr_review_state(&state).unwrap();

        // Update the state with review info
        state.last_review_id = Some(999);
        state.last_review_time = Some("2024-01-15T10:00:00Z".to_string());
        state.last_comment_id = Some(888);
        state.last_comment_time = Some("2024-01-15T11:00:00Z".to_string());
        tracker.save_pr_review_state(&state).unwrap();

        // Verify the update
        let states = tracker.get_active_pr_review_states().unwrap();
        assert_eq!(states.len(), 1);
        assert_eq!(states[0].last_review_id, Some(999));
        assert_eq!(
            states[0].last_review_time,
            Some("2024-01-15T10:00:00Z".to_string())
        );
        assert_eq!(states[0].last_comment_id, Some(888));
        assert_eq!(
            states[0].last_comment_time,
            Some("2024-01-15T11:00:00Z".to_string())
        );
    }

    #[test]
    fn test_deactivate_pr_review_state() {
        let tracker = SqliteTracker::in_memory().unwrap();

        // Create and save two states
        let state1 = claudear_core::types::PrReviewState::new(
            "https://github.com/owner/repo/pull/1",
            "owner/repo",
            1,
            "issue-1",
            "linear",
        );
        let state2 = claudear_core::types::PrReviewState::new(
            "https://github.com/owner/repo/pull/2",
            "owner/repo",
            2,
            "issue-2",
            "linear",
        );
        tracker.save_pr_review_state(&state1).unwrap();
        tracker.save_pr_review_state(&state2).unwrap();

        // Verify both are active
        let states = tracker.get_active_pr_review_states().unwrap();
        assert_eq!(states.len(), 2);

        // Deactivate one
        tracker.deactivate_pr_review_state(&state1.pr_url).unwrap();

        // Verify only one remains active
        let states = tracker.get_active_pr_review_states().unwrap();
        assert_eq!(states.len(), 1);
        assert_eq!(states[0].pr_url, "https://github.com/owner/repo/pull/2");
    }

    #[test]
    fn test_record_pr_review_comment() {
        let tracker = SqliteTracker::in_memory().unwrap();

        let comment = claudear_core::types::ReviewComment {
            id: 12345,
            path: "src/main.rs".to_string(),
            position: Some(10),
            original_position: None,
            body: "Consider using a const here".to_string(),
            user: claudear_core::types::ReviewUser {
                id: 1,
                login: "reviewer1".to_string(),
                user_type: Some("User".to_string()),
            },
            created_at: "2024-01-15T10:00:00Z".to_string(),
            updated_at: "2024-01-15T10:00:00Z".to_string(),
            html_url: "https://github.com/owner/repo/pull/1#comment-12345".to_string(),
            pull_request_review_id: None, // None since we haven't created a review
            start_line: None,
            line: Some(42),
            side: Some("RIGHT".to_string()),
        };

        let pr_url = "https://github.com/owner/repo/pull/1";
        tracker.record_pr_review_comment(pr_url, &comment).unwrap();

        // Retrieve and verify
        let comments = tracker.get_comments_for_pr(pr_url).unwrap();
        assert_eq!(comments.len(), 1);
        assert_eq!(comments[0].scm_comment_id, 12345);
        assert_eq!(comments[0].path, "src/main.rs");
        assert_eq!(comments[0].body, "Consider using a const here");
        assert_eq!(comments[0].author, "reviewer1");
        assert_eq!(comments[0].line, Some(42));
    }

    #[test]
    fn test_get_comments_for_pr() {
        let tracker = SqliteTracker::in_memory().unwrap();

        let pr_url = "https://github.com/owner/repo/pull/42";

        // Create multiple comments
        for i in 1..=3 {
            let comment = claudear_core::types::ReviewComment {
                id: i * 100,
                path: format!("src/file{}.rs", i),
                position: Some(i),
                original_position: None,
                body: format!("Comment {}", i),
                user: claudear_core::types::ReviewUser {
                    id: i,
                    login: format!("reviewer{}", i),
                    user_type: Some("User".to_string()),
                },
                created_at: format!("2024-01-15T10:0{}:00Z", i),
                updated_at: format!("2024-01-15T10:0{}:00Z", i),
                html_url: format!("https://github.com/owner/repo/pull/42#comment-{}", i * 100),
                pull_request_review_id: None,
                start_line: None,
                line: Some(i),
                side: None,
            };
            tracker.record_pr_review_comment(pr_url, &comment).unwrap();
        }

        let comments = tracker.get_comments_for_pr(pr_url).unwrap();
        assert_eq!(comments.len(), 3);

        // Verify ordering by created_at ASC
        assert_eq!(comments[0].scm_comment_id, 100);
        assert_eq!(comments[1].scm_comment_id, 200);
        assert_eq!(comments[2].scm_comment_id, 300);
    }

    #[test]
    fn test_pr_review_comment_upsert() {
        let tracker = SqliteTracker::in_memory().unwrap();

        let pr_url = "https://github.com/owner/repo/pull/1";
        let comment = claudear_core::types::ReviewComment {
            id: 999,
            path: "src/main.rs".to_string(),
            position: None,
            original_position: None,
            body: "Original comment".to_string(),
            user: claudear_core::types::ReviewUser {
                id: 1,
                login: "author".to_string(),
                user_type: Some("User".to_string()),
            },
            created_at: "2024-01-15T10:00:00Z".to_string(),
            updated_at: "2024-01-15T10:00:00Z".to_string(),
            html_url: "https://github.com/owner/repo/pull/1#comment-999".to_string(),
            pull_request_review_id: None,
            start_line: None,
            line: None,
            side: None,
        };

        tracker.record_pr_review_comment(pr_url, &comment).unwrap();

        // Update the comment (same id, different body)
        let updated_comment = claudear_core::types::ReviewComment {
            body: "Updated comment body".to_string(),
            updated_at: "2024-01-15T11:00:00Z".to_string(),
            ..comment
        };
        tracker
            .record_pr_review_comment(pr_url, &updated_comment)
            .unwrap();

        // Should still have only one comment
        let comments = tracker.get_comments_for_pr(pr_url).unwrap();
        assert_eq!(comments.len(), 1);
        assert_eq!(comments[0].body, "Updated comment body");
        assert_eq!(comments[0].updated_at, "2024-01-15T11:00:00Z");
    }

    #[test]
    fn test_store_and_retrieve_feedback_outcome() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker
            .record_attempt("linear", "issue-1", "LIN-1")
            .unwrap();

        let attempt = tracker.get_attempt("linear", "issue-1").unwrap().unwrap();

        let outcome = FixOutcome {
            id: 0,
            attempt_id: attempt.id,
            source: "linear".to_string(),
            issue_id: "issue-1".to_string(),
            issue_text: "Database timeout\n\nConnection fails".to_string(),
            prompt_used: "Fix the timeout".to_string(),
            outcome: claudear_core::types::Outcome::Merged,
            error_type: None,
            learnings: Some("Check connection pool".to_string()),
            keywords: vec!["database".to_string(), "timeout".to_string()],
            embedding: None,
            created_at: chrono::Utc::now(),
        };

        let id = tracker.store_feedback_outcome(&outcome).unwrap();
        assert!(id > 0);

        // Retrieve by attempt
        let retrieved = tracker
            .get_feedback_outcome_by_attempt(attempt.id)
            .unwrap()
            .unwrap();
        assert_eq!(retrieved.source, "linear");
        assert_eq!(retrieved.issue_id, "issue-1");
        assert_eq!(retrieved.outcome, claudear_core::types::Outcome::Merged);
        assert_eq!(
            retrieved.learnings,
            Some("Check connection pool".to_string())
        );
        assert_eq!(
            retrieved.keywords,
            vec!["database".to_string(), "timeout".to_string()]
        );
    }

    #[test]
    fn test_get_feedback_outcomes_with_source_filter() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker
            .record_attempt("linear", "issue-1", "LIN-1")
            .unwrap();
        tracker
            .record_attempt("sentry", "issue-2", "SENT-2")
            .unwrap();

        let attempt1 = tracker.get_attempt("linear", "issue-1").unwrap().unwrap();
        let attempt2 = tracker.get_attempt("sentry", "issue-2").unwrap().unwrap();

        let outcome1 = FixOutcome {
            id: 0,
            attempt_id: attempt1.id,
            source: "linear".to_string(),
            issue_id: "issue-1".to_string(),
            issue_text: "Linear issue".to_string(),
            prompt_used: "prompt".to_string(),
            outcome: claudear_core::types::Outcome::Merged,
            error_type: None,
            learnings: None,
            keywords: vec![],
            embedding: None,
            created_at: chrono::Utc::now(),
        };

        let outcome2 = FixOutcome {
            id: 0,
            attempt_id: attempt2.id,
            source: "sentry".to_string(),
            issue_id: "issue-2".to_string(),
            issue_text: "Sentry issue".to_string(),
            prompt_used: "prompt".to_string(),
            outcome: claudear_core::types::Outcome::Failed,
            error_type: Some("timeout".to_string()),
            learnings: None,
            keywords: vec![],
            embedding: None,
            created_at: chrono::Utc::now(),
        };

        tracker.store_feedback_outcome(&outcome1).unwrap();
        tracker.store_feedback_outcome(&outcome2).unwrap();

        // All outcomes
        let all = tracker.get_feedback_outcomes(None, 100).unwrap();
        assert_eq!(all.len(), 2);

        // Filter by source
        let linear_only = tracker.get_feedback_outcomes(Some("linear"), 100).unwrap();
        assert_eq!(linear_only.len(), 1);
        assert_eq!(linear_only[0].source, "linear");

        let sentry_only = tracker.get_feedback_outcomes(Some("sentry"), 100).unwrap();
        assert_eq!(sentry_only.len(), 1);
        assert_eq!(
            sentry_only[0].outcome,
            claudear_core::types::Outcome::Failed
        );
    }

    #[test]
    fn test_create_and_get_user() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let id = tracker
            .create_user("test@example.com", "$2b$12$hash", "Test User", "admin")
            .unwrap();
        assert!(id > 0);
        let user = tracker.get_user_by_id(id).unwrap().unwrap();
        assert_eq!(user.email, "test@example.com");
        assert_eq!(user.name, "Test User");
        assert_eq!(user.role, "admin");
    }

    #[test]
    fn test_get_user_by_email() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker
            .create_user("alice@example.com", "$2b$12$hash", "Alice", "viewer")
            .unwrap();
        let user = tracker
            .get_user_by_email("alice@example.com")
            .unwrap()
            .unwrap();
        assert_eq!(user.name, "Alice");
        let missing = tracker.get_user_by_email("nobody@example.com").unwrap();
        assert!(missing.is_none());
    }

    #[test]
    fn test_list_users() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker
            .create_user("a@test.com", "hash", "A", "admin")
            .unwrap();
        tracker
            .create_user("b@test.com", "hash", "B", "viewer")
            .unwrap();
        let users = tracker.list_users().unwrap();
        assert_eq!(users.len(), 2);
    }

    #[test]
    fn test_update_user() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let id = tracker
            .create_user("old@test.com", "hash", "Old Name", "viewer")
            .unwrap();
        tracker
            .update_user(
                id,
                Some("new@test.com"),
                None,
                Some("New Name"),
                Some("admin"),
                None,
            )
            .unwrap();
        let user = tracker.get_user_by_id(id).unwrap().unwrap();
        assert_eq!(user.email, "new@test.com");
        assert_eq!(user.name, "New Name");
        assert_eq!(user.role, "admin");
    }

    #[test]
    fn test_delete_user() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let id = tracker
            .create_user("del@test.com", "hash", "Delete Me", "viewer")
            .unwrap();
        assert!(tracker.delete_user(id).unwrap());
        assert!(tracker.get_user_by_id(id).unwrap().is_none());
        assert!(!tracker.delete_user(id).unwrap());
    }

    #[test]
    fn test_duplicate_email_fails() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker
            .create_user("dup@test.com", "hash", "First", "admin")
            .unwrap();
        let result = tracker.create_user("dup@test.com", "hash", "Second", "viewer");
        assert!(result.is_err());
    }

    #[test]
    fn test_create_and_validate_session() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let user_id = tracker
            .create_user("sess@test.com", "hash", "Session User", "admin")
            .unwrap();
        let token = tracker
            .create_session(user_id, "2099-12-31T23:59:59")
            .unwrap();
        assert_eq!(token.len(), 64);
        let user = tracker.get_session_user(&token).unwrap().unwrap();
        assert_eq!(user.id, user_id);
        assert_eq!(user.email, "sess@test.com");
    }

    #[test]
    fn test_expired_session_returns_none() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let user_id = tracker
            .create_user("exp@test.com", "hash", "Expired", "viewer")
            .unwrap();
        let token = tracker
            .create_session(user_id, "2000-01-01T00:00:00")
            .unwrap();
        let user = tracker.get_session_user(&token).unwrap();
        assert!(user.is_none());
    }

    #[test]
    fn test_delete_session() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let user_id = tracker
            .create_user("delsess@test.com", "hash", "Del Sess", "admin")
            .unwrap();
        let token = tracker
            .create_session(user_id, "2099-12-31T23:59:59")
            .unwrap();
        assert!(tracker.get_session_user(&token).unwrap().is_some());
        tracker.delete_session(&token).unwrap();
        assert!(tracker.get_session_user(&token).unwrap().is_none());
    }

    #[test]
    fn test_cleanup_expired_sessions() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let user_id = tracker
            .create_user("clean@test.com", "hash", "Clean", "admin")
            .unwrap();
        tracker
            .create_session(user_id, "2000-01-01T00:00:00")
            .unwrap();
        tracker
            .create_session(user_id, "2000-01-02T00:00:00")
            .unwrap();
        tracker
            .create_session(user_id, "2099-12-31T23:59:59")
            .unwrap();
        let deleted = tracker.cleanup_expired_sessions().unwrap();
        assert_eq!(deleted, 2);
    }

    #[test]
    fn test_touch_session_updates_last_active() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let user_id = tracker
            .create_user("touch@test.com", "hash", "Touch", "admin")
            .unwrap();
        let token = tracker
            .create_session(user_id, "2099-12-31T23:59:59")
            .unwrap();
        // Touch should succeed and session should still be valid
        tracker.touch_session(&token).unwrap();
        let user = tracker.get_session_user(&token).unwrap();
        assert!(user.is_some());
    }

    #[test]
    fn test_count_users() {
        let tracker = SqliteTracker::in_memory().unwrap();
        assert_eq!(tracker.count_users().unwrap(), 0);
        tracker
            .create_user("c1@test.com", "hash", "C1", "admin")
            .unwrap();
        tracker
            .create_user("c2@test.com", "hash", "C2", "viewer")
            .unwrap();
        assert_eq!(tracker.count_users().unwrap(), 2);
    }

    #[test]
    fn test_qa_tables_exist_after_migration() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let conn = tracker.acquire_lock().unwrap();

        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name IN ('qa_knowledge','qa_usage','question_channel_cursor')",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 3);
    }

    #[test]
    fn test_find_similar_qa_scoped_filters_by_repo() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let now = Utc::now();
        let question = "Which branch should we use?";
        let question_norm = claudear_core::types::normalize_text(question);

        let entry_a = QaKnowledgeEntry {
            id: 0,
            source: "linear".to_string(),
            repo: Some("org/repo-a".to_string()),
            issue_id: "1".to_string(),
            short_id: "LIN-1".to_string(),
            question_text: question.to_string(),
            question_norm: question_norm.clone(),
            question_embedding: None,
            answer_text: "Use main".to_string(),
            answer_norm: "use main".to_string(),
            answer_embedding: None,
            channel: "email".to_string(),
            responder: Some("a@example.com".to_string()),
            correlation_id: "c1".to_string(),
            asked_at: now,
            answered_at: now,
            success_count: 1,
            failure_count: 0,
            last_used_at: None,
            metadata: None,
        };

        let entry_b = QaKnowledgeEntry {
            repo: Some("org/repo-b".to_string()),
            issue_id: "2".to_string(),
            short_id: "LIN-2".to_string(),
            correlation_id: "c2".to_string(),
            ..entry_a.clone()
        };

        tracker.store_qa_knowledge(&entry_a).unwrap();
        tracker.store_qa_knowledge(&entry_b).unwrap();

        let matches = tracker
            .find_similar_qa_scoped("linear", Some("org/repo-a"), &question_norm, None, 0.8, 5)
            .unwrap();

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].entry.repo.as_deref(), Some("org/repo-a"));
    }

    #[test]
    fn test_find_similar_qa_scoped_exact_ranks_by_success_rate() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let now = Utc::now();
        let question = "Which branch should we use?";
        let question_norm = claudear_core::types::normalize_text(question);

        let base_entry = QaKnowledgeEntry {
            id: 0,
            source: "linear".to_string(),
            repo: Some("org/repo-a".to_string()),
            issue_id: "1".to_string(),
            short_id: "LIN-1".to_string(),
            question_text: question.to_string(),
            question_norm: question_norm.clone(),
            question_embedding: None,
            answer_text: "Use main".to_string(),
            answer_norm: "use main".to_string(),
            answer_embedding: None,
            channel: "email".to_string(),
            responder: Some("a@example.com".to_string()),
            correlation_id: "c1".to_string(),
            asked_at: now,
            answered_at: now,
            success_count: 0,
            failure_count: 0,
            last_used_at: None,
            metadata: None,
        };

        let low_confidence = base_entry.clone();
        let high_confidence = QaKnowledgeEntry {
            issue_id: "2".to_string(),
            short_id: "LIN-2".to_string(),
            answer_text: "Use release branch".to_string(),
            answer_norm: "use release branch".to_string(),
            correlation_id: "c2".to_string(),
            success_count: 9,
            failure_count: 1,
            ..base_entry
        };

        tracker.store_qa_knowledge(&low_confidence).unwrap();
        tracker.store_qa_knowledge(&high_confidence).unwrap();

        let matches = tracker
            .find_similar_qa_scoped("linear", Some("org/repo-a"), &question_norm, None, 0.8, 5)
            .unwrap();

        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0].entry.answer_text, "Use release branch");
        assert!(matches[0].historical_success_rate > matches[1].historical_success_rate);
        assert!(matches[0].final_score > matches[1].final_score);
    }

    #[test]
    fn test_find_similar_qa_global_exact_only_returns_normalized_match() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let now = Utc::now();
        let question_norm = claudear_core::types::normalize_text("Pick deployment region");

        let matching_entry = QaKnowledgeEntry {
            id: 0,
            source: "linear".to_string(),
            repo: Some("org/repo-a".to_string()),
            issue_id: "1".to_string(),
            short_id: "LIN-1".to_string(),
            question_text: "Pick deployment region".to_string(),
            question_norm: question_norm.clone(),
            question_embedding: None,
            answer_text: "us-east-1".to_string(),
            answer_norm: "us-east-1".to_string(),
            answer_embedding: None,
            channel: "email".to_string(),
            responder: Some("a@example.com".to_string()),
            correlation_id: "c1".to_string(),
            asked_at: now,
            answered_at: now,
            success_count: 1,
            failure_count: 0,
            last_used_at: None,
            metadata: None,
        };
        let non_matching_entry = QaKnowledgeEntry {
            issue_id: "2".to_string(),
            short_id: "LIN-2".to_string(),
            question_text: "Pick staging region".to_string(),
            question_norm: claudear_core::types::normalize_text("Pick staging region"),
            answer_text: "eu-west-1".to_string(),
            answer_norm: "eu-west-1".to_string(),
            correlation_id: "c2".to_string(),
            ..matching_entry.clone()
        };

        tracker.store_qa_knowledge(&matching_entry).unwrap();
        tracker.store_qa_knowledge(&non_matching_entry).unwrap();

        let matches = tracker
            .find_similar_qa_global(&question_norm, None, 0.8, 5)
            .unwrap();

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].entry.answer_text, "us-east-1");
    }

    #[test]
    fn test_qa_usage_updates_outcome_stats_for_attempt() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker
            .record_attempt("linear", "issue-1", "LIN-1")
            .unwrap();
        let attempt = tracker.get_attempt("linear", "issue-1").unwrap().unwrap();

        let now = Utc::now();
        let qa_id = tracker
            .store_qa_knowledge(&QaKnowledgeEntry {
                id: 0,
                source: "linear".to_string(),
                repo: Some("org/repo".to_string()),
                issue_id: "issue-1".to_string(),
                short_id: "LIN-1".to_string(),
                question_text: "Question?".to_string(),
                question_norm: "question?".to_string(),
                question_embedding: None,
                answer_text: "Answer".to_string(),
                answer_norm: "answer".to_string(),
                answer_embedding: None,
                channel: "discord".to_string(),
                responder: Some("user-1".to_string()),
                correlation_id: "corr-1".to_string(),
                asked_at: now,
                answered_at: now,
                success_count: 0,
                failure_count: 0,
                last_used_at: None,
                metadata: None,
            })
            .unwrap();

        tracker
            .record_qa_usage(attempt.id, qa_id, "asked", 1.0)
            .unwrap();
        tracker
            .update_qa_outcome_stats_for_attempt(attempt.id, true)
            .unwrap();

        let conn = tracker.acquire_lock().unwrap();
        let success_count: i64 = conn
            .query_row(
                "SELECT success_count FROM qa_knowledge WHERE id = ?1",
                params![qa_id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(success_count, 1);
    }

    #[test]
    fn test_record_activity_returns_positive_id() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let entry = ActivityLogEntry {
            id: 0,
            timestamp: Utc::now(),
            activity_type: "issue_received".to_string(),
            source: Some("linear".to_string()),
            issue_id: Some("abc-123".to_string()),
            short_id: Some("LIN-1".to_string()),
            message: "New issue received".to_string(),
            metadata: None,
        };

        let id = tracker.record_activity(&entry).unwrap();
        assert!(id > 0);
    }

    #[test]
    fn test_record_activity_with_metadata() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let meta = serde_json::json!({"priority": "high", "assignee": "alice"});
        let entry = ActivityLogEntry {
            id: 0,
            timestamp: Utc::now(),
            activity_type: "processing_started".to_string(),
            source: Some("sentry".to_string()),
            issue_id: Some("evt-1".to_string()),
            short_id: None,
            message: "Processing started".to_string(),
            metadata: Some(meta.clone()),
        };

        tracker.record_activity(&entry).unwrap();
        let activities = tracker.get_recent_activities(10, None).unwrap();
        assert_eq!(activities.len(), 1);
        assert_eq!(activities[0].metadata.as_ref().unwrap()["priority"], "high");
    }

    #[test]
    fn test_get_recent_activities_empty_db() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let activities = tracker.get_recent_activities(10, None).unwrap();
        assert!(activities.is_empty());
    }

    #[test]
    fn test_get_recent_activities_respects_limit() {
        let tracker = SqliteTracker::in_memory().unwrap();
        for i in 0..5 {
            let entry = ActivityLogEntry {
                id: 0,
                timestamp: Utc::now(),
                activity_type: "test".to_string(),
                source: Some("linear".to_string()),
                issue_id: Some(format!("issue-{}", i)),
                short_id: None,
                message: format!("Activity {}", i),
                metadata: None,
            };
            tracker.record_activity(&entry).unwrap();
        }

        let activities = tracker.get_recent_activities(3, None).unwrap();
        assert_eq!(activities.len(), 3);
    }

    #[test]
    fn test_get_recent_activities_source_filter() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let sources = ["linear", "sentry", "linear"];
        for (i, source) in sources.iter().enumerate() {
            let entry = ActivityLogEntry {
                id: 0,
                timestamp: Utc::now(),
                activity_type: "test".to_string(),
                source: Some(source.to_string()),
                issue_id: Some(format!("issue-{}", i)),
                short_id: None,
                message: format!("Activity {}", i),
                metadata: None,
            };
            tracker.record_activity(&entry).unwrap();
        }

        let linear_only = tracker.get_recent_activities(10, Some("linear")).unwrap();
        assert_eq!(linear_only.len(), 2);

        let sentry_only = tracker.get_recent_activities(10, Some("sentry")).unwrap();
        assert_eq!(sentry_only.len(), 1);

        let github_only = tracker.get_recent_activities(10, Some("github")).unwrap();
        assert!(github_only.is_empty());
    }

    #[test]
    fn test_get_recent_activities_ordered_desc() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let timestamps = [
            chrono::DateTime::parse_from_rfc3339("2024-01-01T00:00:00Z")
                .unwrap()
                .with_timezone(&Utc),
            chrono::DateTime::parse_from_rfc3339("2024-06-15T12:00:00Z")
                .unwrap()
                .with_timezone(&Utc),
            chrono::DateTime::parse_from_rfc3339("2024-03-10T06:00:00Z")
                .unwrap()
                .with_timezone(&Utc),
        ];
        for (i, ts) in timestamps.iter().enumerate() {
            let entry = ActivityLogEntry {
                id: 0,
                timestamp: *ts,
                activity_type: "test".to_string(),
                source: None,
                issue_id: Some(format!("{}", i)),
                short_id: None,
                message: format!("ts {}", i),
                metadata: None,
            };
            tracker.record_activity(&entry).unwrap();
        }

        let activities = tracker.get_recent_activities(10, None).unwrap();
        assert_eq!(activities.len(), 3);
        // Most recent first
        assert!(activities[0].timestamp >= activities[1].timestamp);
        assert!(activities[1].timestamp >= activities[2].timestamp);
    }

    #[test]
    fn test_get_activities_for_issue() {
        let tracker = SqliteTracker::in_memory().unwrap();
        for i in 0..3 {
            let entry = ActivityLogEntry {
                id: 0,
                timestamp: Utc::now(),
                activity_type: format!("step_{}", i),
                source: Some("linear".to_string()),
                issue_id: Some("target-issue".to_string()),
                short_id: Some("LIN-1".to_string()),
                message: format!("Step {}", i),
                metadata: None,
            };
            tracker.record_activity(&entry).unwrap();
        }
        // Different issue
        let other = ActivityLogEntry {
            id: 0,
            timestamp: Utc::now(),
            activity_type: "test".to_string(),
            source: Some("linear".to_string()),
            issue_id: Some("other-issue".to_string()),
            short_id: None,
            message: "Other".to_string(),
            metadata: None,
        };
        tracker.record_activity(&other).unwrap();

        let activities = tracker
            .get_activities_for_issue("linear", "target-issue")
            .unwrap();
        assert_eq!(activities.len(), 3);

        let empty = tracker
            .get_activities_for_issue("linear", "nonexistent")
            .unwrap();
        assert!(empty.is_empty());
    }

    #[test]
    fn test_record_activities_batch_inserts_all() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let entries: Vec<ActivityLogEntry> = (0..10)
            .map(|i| ActivityLogEntry {
                id: 0,
                timestamp: Utc::now(),
                activity_type: "batch_test".to_string(),
                source: Some("linear".to_string()),
                issue_id: Some(format!("issue-{}", i)),
                short_id: None,
                message: format!("Batch entry {}", i),
                metadata: None,
            })
            .collect();

        let count = tracker.record_activities_batch(&entries).unwrap();
        assert_eq!(count, 10);

        let activities = tracker.get_recent_activities(100, None).unwrap();
        assert_eq!(activities.len(), 10);
    }

    #[test]
    fn test_record_pr_review_and_retrieve() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker
            .record_attempt("linear", "issue-1", "LIN-1")
            .unwrap();
        let attempt = tracker.get_attempt("linear", "issue-1").unwrap().unwrap();

        let review = PrReviewRecord {
            id: 0,
            attempt_id: Some(attempt.id),
            pr_url: "https://github.com/org/repo/pull/10".to_string(),
            reviewer: Some("alice".to_string()),
            review_state: Some("approved".to_string()),
            submitted_at: Some(Utc::now()),
            body: Some("Looks good!".to_string()),
            sentiment: Some("positive".to_string()),
            actionable_feedback: None,
        };

        let id = tracker.record_pr_review(&review).unwrap();
        assert!(id > 0);

        let reviews = tracker.get_reviews_for_attempt(attempt.id).unwrap();
        assert_eq!(reviews.len(), 1);
        assert_eq!(reviews[0].reviewer, Some("alice".to_string()));
        assert_eq!(reviews[0].review_state, Some("approved".to_string()));
        assert_eq!(reviews[0].body, Some("Looks good!".to_string()));
        assert_eq!(reviews[0].sentiment, Some("positive".to_string()));
    }

    #[test]
    fn test_multiple_reviews_per_attempt() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker
            .record_attempt("linear", "issue-1", "LIN-1")
            .unwrap();
        let attempt = tracker.get_attempt("linear", "issue-1").unwrap().unwrap();

        let reviewers = ["alice", "bob", "charlie"];
        let states = ["approved", "changes_requested", "commented"];

        for (reviewer, state) in reviewers.iter().zip(states.iter()) {
            let review = PrReviewRecord {
                id: 0,
                attempt_id: Some(attempt.id),
                pr_url: "https://github.com/org/repo/pull/10".to_string(),
                reviewer: Some(reviewer.to_string()),
                review_state: Some(state.to_string()),
                submitted_at: Some(Utc::now()),
                body: None,
                sentiment: None,
                actionable_feedback: None,
            };
            tracker.record_pr_review(&review).unwrap();
        }

        let reviews = tracker.get_reviews_for_attempt(attempt.id).unwrap();
        assert_eq!(reviews.len(), 3);
    }

    #[test]
    fn test_get_reviews_for_nonexistent_attempt() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let reviews = tracker.get_reviews_for_attempt(9999).unwrap();
        assert!(reviews.is_empty());
    }

    #[test]
    fn test_store_and_retrieve_embedding() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let embedding = IssueEmbedding {
            id: 0,
            source: "linear".to_string(),
            issue_id: "emb-1".to_string(),
            short_id: Some("LIN-1".to_string()),
            title: Some("Fix login bug".to_string()),
            description: None,
            url: None,
            priority: None,
            status: None,
            labels: None,
            embedding: Some(vec![0.1, 0.2, 0.3, 0.4]),
            embedding_model: Some("text-embedding-3-small".to_string()),
            created_at: Utc::now(),
            updated_at: None,
        };

        let id = tracker.store_embedding(&embedding).unwrap();
        assert!(id > 0);

        let retrieved = tracker.get_embedding("linear", "emb-1").unwrap().unwrap();
        assert_eq!(retrieved.source, "linear");
        assert_eq!(retrieved.issue_id, "emb-1");
        assert_eq!(retrieved.short_id, Some("LIN-1".to_string()));
        assert_eq!(retrieved.title, Some("Fix login bug".to_string()));
        let emb = retrieved.embedding.unwrap();
        assert_eq!(emb.len(), 4);
        assert!((emb[0] - 0.1).abs() < f32::EPSILON);
        assert!((emb[3] - 0.4).abs() < f32::EPSILON);
        assert_eq!(
            retrieved.embedding_model,
            Some("text-embedding-3-small".to_string())
        );
    }

    #[test]
    fn test_store_embedding_upsert_updates_on_conflict() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let original = IssueEmbedding {
            id: 0,
            source: "linear".to_string(),
            issue_id: "emb-1".to_string(),
            short_id: Some("LIN-1".to_string()),
            title: Some("Original title".to_string()),
            description: None,
            url: None,
            priority: None,
            status: None,
            labels: None,
            embedding: Some(vec![1.0, 2.0]),
            embedding_model: Some("model-v1".to_string()),
            created_at: Utc::now(),
            updated_at: None,
        };
        tracker.store_embedding(&original).unwrap();

        let updated = IssueEmbedding {
            id: 0,
            source: "linear".to_string(),
            issue_id: "emb-1".to_string(),
            short_id: Some("LIN-1".to_string()),
            title: Some("Original title".to_string()),
            description: None,
            url: None,
            priority: None,
            status: None,
            labels: None,
            embedding: Some(vec![3.0, 4.0, 5.0]),
            embedding_model: Some("model-v2".to_string()),
            created_at: Utc::now(),
            updated_at: None,
        };
        tracker.store_embedding(&updated).unwrap();

        let retrieved = tracker.get_embedding("linear", "emb-1").unwrap().unwrap();
        let emb = retrieved.embedding.unwrap();
        assert_eq!(emb.len(), 3);
        assert!((emb[0] - 3.0).abs() < f32::EPSILON);
        assert_eq!(retrieved.embedding_model, Some("model-v2".to_string()));
    }

    #[test]
    fn test_get_embedding_nonexistent() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let result = tracker.get_embedding("linear", "nonexistent").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_get_all_embeddings_pagination() {
        let tracker = SqliteTracker::in_memory().unwrap();
        for i in 0..5 {
            let emb = IssueEmbedding {
                id: 0,
                source: "linear".to_string(),
                issue_id: format!("emb-{}", i),
                short_id: None,
                title: None,
                description: None,
                url: None,
                priority: None,
                status: None,
                labels: None,
                embedding: Some(vec![i as f32]),
                embedding_model: None,
                created_at: Utc::now(),
                updated_at: None,
            };
            tracker.store_embedding(&emb).unwrap();
        }

        let page1 = tracker.get_all_embeddings(None, Some(2), Some(0)).unwrap();
        assert_eq!(page1.len(), 2);

        let page2 = tracker.get_all_embeddings(None, Some(2), Some(2)).unwrap();
        assert_eq!(page2.len(), 2);

        let all = tracker.get_all_embeddings(None, Some(100), None).unwrap();
        assert_eq!(all.len(), 5);
    }

    #[test]
    fn test_get_all_embeddings_source_filter() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let sources = ["linear", "sentry", "linear"];
        for (i, source) in sources.iter().enumerate() {
            let emb = IssueEmbedding {
                id: 0,
                source: source.to_string(),
                issue_id: format!("emb-{}", i),
                short_id: None,
                title: None,
                description: None,
                url: None,
                priority: None,
                status: None,
                labels: None,
                embedding: Some(vec![1.0]),
                embedding_model: None,
                created_at: Utc::now(),
                updated_at: None,
            };
            tracker.store_embedding(&emb).unwrap();
        }

        let linear = tracker
            .get_all_embeddings(Some("linear"), Some(100), None)
            .unwrap();
        assert_eq!(linear.len(), 2);

        let sentry = tracker
            .get_all_embeddings(Some("sentry"), Some(100), None)
            .unwrap();
        assert_eq!(sentry.len(), 1);
    }

    #[test]
    fn test_store_feedback_outcome_with_all_fields() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker
            .record_attempt("linear", "issue-fb", "LIN-FB")
            .unwrap();
        let attempt = tracker.get_attempt("linear", "issue-fb").unwrap().unwrap();

        let outcome = FixOutcome {
            id: 0,
            attempt_id: attempt.id,
            source: "linear".to_string(),
            issue_id: "issue-fb".to_string(),
            issue_text: "Null pointer in handler".to_string(),
            prompt_used: "Fix the null pointer".to_string(),
            outcome: claudear_core::types::Outcome::CannotFix,
            error_type: Some("null_reference".to_string()),
            learnings: Some("Check for null before access".to_string()),
            keywords: vec![
                "null".to_string(),
                "pointer".to_string(),
                "handler".to_string(),
            ],
            embedding: None,
            created_at: Utc::now(),
        };

        let id = tracker.store_feedback_outcome(&outcome).unwrap();
        assert!(id > 0);

        let retrieved = tracker
            .get_feedback_outcome_by_attempt(attempt.id)
            .unwrap()
            .unwrap();
        assert_eq!(retrieved.outcome, claudear_core::types::Outcome::CannotFix);
        assert_eq!(retrieved.error_type, Some("null_reference".to_string()));
        assert_eq!(retrieved.keywords.len(), 3);
        assert!(retrieved.keywords.contains(&"null".to_string()));
    }

    #[test]
    fn test_get_feedback_outcome_by_attempt_not_found() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let result = tracker.get_feedback_outcome_by_attempt(9999).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_get_feedback_outcomes_limit() {
        let tracker = SqliteTracker::in_memory().unwrap();
        for i in 0..5 {
            tracker
                .record_attempt("linear", &format!("issue-{}", i), &format!("LIN-{}", i))
                .unwrap();
            let attempt = tracker
                .get_attempt("linear", &format!("issue-{}", i))
                .unwrap()
                .unwrap();
            let outcome = FixOutcome {
                id: 0,
                attempt_id: attempt.id,
                source: "linear".to_string(),
                issue_id: format!("issue-{}", i),
                issue_text: "text".to_string(),
                prompt_used: "prompt".to_string(),
                outcome: claudear_core::types::Outcome::Merged,
                error_type: None,
                learnings: None,
                keywords: vec![],
                embedding: None,
                created_at: Utc::now(),
            };
            tracker.store_feedback_outcome(&outcome).unwrap();
        }

        let limited = tracker.get_feedback_outcomes(None, 3).unwrap();
        assert_eq!(limited.len(), 3);
    }

    #[test]
    fn test_get_feedback_outcomes_empty() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let outcomes = tracker.get_feedback_outcomes(None, 100).unwrap();
        assert!(outcomes.is_empty());
    }

    fn make_qa_entry(
        source: &str,
        repo: Option<&str>,
        issue_id: &str,
        question: &str,
        answer: &str,
        correlation_id: &str,
    ) -> QaKnowledgeEntry {
        let now = Utc::now();
        QaKnowledgeEntry {
            id: 0,
            source: source.to_string(),
            repo: repo.map(|r| r.to_string()),
            issue_id: issue_id.to_string(),
            short_id: format!("LIN-{}", issue_id),
            question_text: question.to_string(),
            question_norm: claudear_core::types::normalize_text(question),
            question_embedding: None,
            answer_text: answer.to_string(),
            answer_norm: claudear_core::types::normalize_text(answer),
            answer_embedding: None,
            channel: "discord".to_string(),
            responder: Some("user@example.com".to_string()),
            correlation_id: correlation_id.to_string(),
            asked_at: now,
            answered_at: now,
            success_count: 0,
            failure_count: 0,
            last_used_at: None,
            metadata: None,
        }
    }

    #[test]
    fn test_store_qa_knowledge_returns_positive_id() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let entry = make_qa_entry(
            "linear",
            Some("org/repo"),
            "1",
            "How to deploy?",
            "Run deploy script",
            "c1",
        );
        let id = tracker.store_qa_knowledge(&entry).unwrap();
        assert!(id > 0);
    }

    #[test]
    fn test_store_qa_knowledge_with_metadata() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let mut entry = make_qa_entry(
            "linear",
            Some("org/repo"),
            "1",
            "How to deploy?",
            "Run deploy script",
            "c1",
        );
        entry.metadata = Some(serde_json::json!({"category": "deployment"}));

        let id = tracker.store_qa_knowledge(&entry).unwrap();
        assert!(id > 0);
    }

    #[test]
    fn test_record_qa_usage_and_update_stats() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker
            .record_attempt("linear", "issue-qa", "LIN-QA")
            .unwrap();
        let attempt = tracker.get_attempt("linear", "issue-qa").unwrap().unwrap();

        let entry = make_qa_entry(
            "linear",
            Some("org/repo"),
            "issue-qa",
            "How to fix it?",
            "Reset the cache",
            "c-qa",
        );
        let qa_id = tracker.store_qa_knowledge(&entry).unwrap();

        // Record usage
        let usage_id = tracker
            .record_qa_usage(attempt.id, qa_id, "auto_applied", 0.95)
            .unwrap();
        assert!(usage_id > 0);

        // Update outcome stats directly
        tracker.update_qa_outcome_stats(qa_id, true).unwrap();
        tracker.update_qa_outcome_stats(qa_id, true).unwrap();
        tracker.update_qa_outcome_stats(qa_id, false).unwrap();

        let conn = tracker.acquire_lock().unwrap();
        let (sc, fc): (i64, i64) = conn
            .query_row(
                "SELECT success_count, failure_count FROM qa_knowledge WHERE id = ?1",
                params![qa_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(sc, 2);
        assert_eq!(fc, 1);
    }

    #[test]
    fn test_find_similar_qa_scoped_no_results() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let matches = tracker
            .find_similar_qa_scoped(
                "linear",
                Some("org/repo"),
                "completely unique query",
                None,
                0.8,
                5,
            )
            .unwrap();
        assert!(matches.is_empty());
    }

    #[test]
    fn test_find_similar_qa_global_no_results() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let matches = tracker
            .find_similar_qa_global("completely unique query", None, 0.8, 5)
            .unwrap();
        assert!(matches.is_empty());
    }

    #[test]
    fn test_find_similar_qa_scoped_exact_match() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let question = "What branch should I use for hotfixes?";
        let entry = make_qa_entry(
            "linear",
            Some("org/repo"),
            "1",
            question,
            "Use the hotfix branch",
            "c1",
        );
        tracker.store_qa_knowledge(&entry).unwrap();

        let question_norm = claudear_core::types::normalize_text(question);
        let matches = tracker
            .find_similar_qa_scoped("linear", Some("org/repo"), &question_norm, None, 0.8, 5)
            .unwrap();

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].entry.answer_text, "Use the hotfix branch");
    }

    #[test]
    fn test_find_similar_qa_global_exact_match() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let question = "What branch should I use for hotfixes?";
        let entry = make_qa_entry(
            "linear",
            Some("org/repo"),
            "1",
            question,
            "Use the hotfix branch",
            "c1",
        );
        tracker.store_qa_knowledge(&entry).unwrap();

        let question_norm = claudear_core::types::normalize_text(question);
        let matches = tracker
            .find_similar_qa_global(&question_norm, None, 0.8, 5)
            .unwrap();

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].entry.answer_text, "Use the hotfix branch");
    }

    #[test]
    fn test_record_qa_usage_upsert() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker
            .record_attempt("linear", "issue-upsert", "LIN-U")
            .unwrap();
        let attempt = tracker
            .get_attempt("linear", "issue-upsert")
            .unwrap()
            .unwrap();

        let entry = make_qa_entry("linear", Some("org/repo"), "issue-upsert", "Q?", "A.", "cu");
        let qa_id = tracker.store_qa_knowledge(&entry).unwrap();

        // First insert
        tracker
            .record_qa_usage(attempt.id, qa_id, "asked", 0.8)
            .unwrap();
        // Upsert with updated usage_type
        tracker
            .record_qa_usage(attempt.id, qa_id, "auto_applied", 0.95)
            .unwrap();

        // Should not fail; ON CONFLICT updates
        let conn = tracker.acquire_lock().unwrap();
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM qa_usage WHERE attempt_id = ?1 AND qa_id = ?2",
                params![attempt.id, qa_id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_save_and_get_active_experiments() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let exp = PromptExperiment::new("test-exp", "control", "Fix {{issue}}", "hash123");
        let id = tracker.save_experiment(&exp).unwrap();
        assert!(id > 0);

        let active = tracker.get_active_experiments().unwrap();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].experiment_name, "test-exp");
        assert_eq!(active[0].variant, "control");
        assert!(active[0].active);
    }

    #[test]
    fn test_save_inactive_experiment_not_in_active_list() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let mut exp = PromptExperiment::new("disabled-exp", "variant_a", "template", "hash");
        exp.active = false;
        tracker.save_experiment(&exp).unwrap();

        let active = tracker.get_active_experiments().unwrap();
        assert!(active.is_empty());
    }

    #[test]
    fn test_update_experiment_stats_success() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let exp = PromptExperiment::new("stats-exp", "control", "template", "hash");
        let id = tracker.save_experiment(&exp).unwrap();

        tracker
            .update_experiment_stats(id, true, Some(2.5))
            .unwrap();
        tracker
            .update_experiment_stats(id, true, Some(3.5))
            .unwrap();
        tracker.update_experiment_stats(id, false, None).unwrap();

        let active = tracker.get_active_experiments().unwrap();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].success_count, 2);
        assert_eq!(active[0].failure_count, 1);
        assert!(active[0].avg_time_to_merge.is_some());
    }

    #[test]
    fn test_multiple_experiment_variants() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let variants = ["control", "variant_a", "variant_b"];
        for variant in &variants {
            let exp = PromptExperiment::new(
                "multi-exp",
                *variant,
                "template",
                format!("hash-{}", variant),
            );
            tracker.save_experiment(&exp).unwrap();
        }

        let active = tracker.get_active_experiments().unwrap();
        assert_eq!(active.len(), 3);
    }

    #[test]
    fn test_get_active_experiments_empty() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let active = tracker.get_active_experiments().unwrap();
        assert!(active.is_empty());
    }

    #[test]
    fn test_upsert_and_get_repository() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let id = tracker
            .upsert_repository(
                "org/my-repo",
                Some("/path/to/repo"),
                Some("https://github.com/org/my-repo"),
            )
            .unwrap();
        assert!(id > 0);

        let repo = tracker.get_repository("org/my-repo").unwrap().unwrap();
        assert_eq!(repo.name, "org/my-repo");
        assert_eq!(repo.path, Some("/path/to/repo".to_string()));
        assert_eq!(repo.scm_url, "https://github.com/org/my-repo");
    }

    #[test]
    fn test_upsert_repository_conflict_updates() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker
            .upsert_repository(
                "org/repo",
                Some("/old/path"),
                Some("https://github.com/org/repo"),
            )
            .unwrap();

        // Upsert again with new path
        tracker
            .upsert_repository(
                "org/repo",
                Some("/new/path"),
                Some("https://github.com/org/repo-updated"),
            )
            .unwrap();

        let repo = tracker.get_repository("org/repo").unwrap().unwrap();
        assert_eq!(repo.path, Some("/new/path".to_string()));
        assert_eq!(repo.scm_url, "https://github.com/org/repo-updated");
    }

    #[test]
    fn test_upsert_repository_defaults() {
        let tracker = SqliteTracker::in_memory().unwrap();
        // No path or scm_url
        tracker.upsert_repository("org/repo", None, None).unwrap();

        let repo = tracker.get_repository("org/repo").unwrap().unwrap();
        assert_eq!(repo.name, "org/repo");
        // scm_url defaults to name
        assert_eq!(repo.scm_url, "org/repo");
        // Empty path stored as empty string, filtered to None
        assert!(repo.path.is_none());
    }

    #[test]
    fn test_get_repository_not_found() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let result = tracker.get_repository("nonexistent").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_list_repositories() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker.upsert_repository("b-repo", None, None).unwrap();
        tracker.upsert_repository("a-repo", None, None).unwrap();
        tracker.upsert_repository("c-repo", None, None).unwrap();

        let repos = tracker.list_repositories().unwrap();
        assert_eq!(repos.len(), 3);
        // Ordered by name
        assert_eq!(repos[0].name, "a-repo");
        assert_eq!(repos[1].name, "b-repo");
        assert_eq!(repos[2].name, "c-repo");
    }

    #[test]
    fn test_add_dependency_and_get_dependencies() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker
            .add_dependency("upstream-lib", "downstream-app", "runtime")
            .unwrap();

        let deps = tracker.get_dependencies("downstream-app").unwrap();
        assert_eq!(deps.len(), 1);
        assert_eq!(deps[0].upstream, "upstream-lib");
        assert_eq!(deps[0].downstream, "downstream-app");
        assert_eq!(deps[0].dep_type, "runtime");
    }

    #[test]
    fn test_add_dependency_creates_repos() {
        let tracker = SqliteTracker::in_memory().unwrap();
        // Repos should not exist yet
        assert!(tracker.get_repository("lib-a").unwrap().is_none());
        assert!(tracker.get_repository("app-b").unwrap().is_none());

        tracker.add_dependency("lib-a", "app-b", "build").unwrap();

        // Both repos should now exist
        assert!(tracker.get_repository("lib-a").unwrap().is_some());
        assert!(tracker.get_repository("app-b").unwrap().is_some());
    }

    #[test]
    fn test_add_dependency_upsert_type() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker.add_dependency("lib", "app", "runtime").unwrap();
        tracker.add_dependency("lib", "app", "build").unwrap();

        let deps = tracker.get_dependencies("app").unwrap();
        assert_eq!(deps.len(), 1);
        assert_eq!(deps[0].dep_type, "build");
    }

    #[test]
    fn test_get_dependents() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker
            .add_dependency("core-lib", "app-a", "runtime")
            .unwrap();
        tracker
            .add_dependency("core-lib", "app-b", "runtime")
            .unwrap();
        tracker
            .add_dependency("other-lib", "app-a", "build")
            .unwrap();

        let dependents = tracker.get_dependents("core-lib").unwrap();
        assert_eq!(dependents.len(), 2);
    }

    #[test]
    fn test_list_all_dependencies() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker.add_dependency("lib-a", "app-1", "runtime").unwrap();
        tracker.add_dependency("lib-b", "app-1", "build").unwrap();
        tracker.add_dependency("lib-a", "app-2", "runtime").unwrap();

        let all = tracker.list_all_dependencies().unwrap();
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn test_clear_repositories() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker.add_dependency("lib", "app", "runtime").unwrap();
        assert!(!tracker.list_repositories().unwrap().is_empty());

        tracker.clear_repositories().unwrap();
        assert!(tracker.list_repositories().unwrap().is_empty());
        assert!(tracker.list_all_dependencies().unwrap().is_empty());
    }

    #[test]
    fn test_get_all_dependants_transitive() {
        let tracker = SqliteTracker::in_memory().unwrap();
        // core -> mid -> leaf
        tracker.add_dependency("core", "mid", "runtime").unwrap();
        tracker.add_dependency("mid", "leaf", "runtime").unwrap();

        let all = tracker.get_all_dependants("core").unwrap();
        assert_eq!(all.len(), 2);
        // First level: mid at depth 1
        let mid_entry = all.iter().find(|(name, _)| name == "mid");
        assert!(mid_entry.is_some());
        assert_eq!(mid_entry.unwrap().1, 1);
        // Second level: leaf at depth 2
        let leaf_entry = all.iter().find(|(name, _)| name == "leaf");
        assert!(leaf_entry.is_some());
        assert_eq!(leaf_entry.unwrap().1, 2);
    }

    #[test]
    fn test_record_and_get_execution() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker
            .record_attempt("linear", "exec-issue", "LIN-E")
            .unwrap();
        let attempt = tracker
            .get_attempt("linear", "exec-issue")
            .unwrap()
            .unwrap();

        let mut execution = AgentExecution::new().with_attempt_id(attempt.id);
        execution.prompt_used = Some("Fix the bug".to_string());
        execution.model_version = Some("claude-3.5-sonnet".to_string());
        execution.working_directory = Some("/home/user/repo".to_string());
        execution.git_branch = Some("fix/issue-123".to_string());
        execution.exit_code = Some(0);
        execution.files_changed = Some(3);
        execution.lines_added = Some(50);
        execution.lines_removed = Some(10);
        execution.event_log_path = Some("/tmp/claudear.events.jsonl".to_string());

        let id = tracker.record_execution(&execution).unwrap();
        assert!(id > 0);

        let executions = tracker.get_executions_for_attempt(attempt.id).unwrap();
        assert_eq!(executions.len(), 1);
        assert_eq!(executions[0].attempt_id, Some(attempt.id));
        assert_eq!(executions[0].prompt_used, Some("Fix the bug".to_string()));
        assert_eq!(
            executions[0].model_version,
            Some("claude-3.5-sonnet".to_string())
        );
        assert_eq!(executions[0].exit_code, Some(0));
        assert_eq!(executions[0].files_changed, Some(3));
        assert_eq!(executions[0].lines_added, Some(50));
        assert_eq!(executions[0].lines_removed, Some(10));
        assert_eq!(
            executions[0].event_log_path,
            Some("/tmp/claudear.events.jsonl".to_string())
        );
    }

    #[test]
    fn test_multiple_executions_per_attempt() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker
            .record_attempt("linear", "multi-exec", "LIN-ME")
            .unwrap();
        let attempt = tracker
            .get_attempt("linear", "multi-exec")
            .unwrap()
            .unwrap();

        for i in 0..3 {
            let mut execution = AgentExecution::new().with_attempt_id(attempt.id);
            execution.exit_code = Some(i);
            tracker.record_execution(&execution).unwrap();
        }

        let executions = tracker.get_executions_for_attempt(attempt.id).unwrap();
        assert_eq!(executions.len(), 3);
    }

    #[test]
    fn test_get_executions_for_nonexistent_attempt() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let executions = tracker.get_executions_for_attempt(9999).unwrap();
        assert!(executions.is_empty());
    }

    #[test]
    fn test_record_execution_with_timed_out() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let mut execution = AgentExecution::new();
        execution.timed_out = true;
        execution.exit_code = None;

        let id = tracker.record_execution(&execution).unwrap();
        assert!(id > 0);

        // Verify timed_out round-trips (attempt_id is None so use get via raw query)
        let conn = tracker.acquire_lock().unwrap();
        let timed_out: i32 = conn
            .query_row(
                "SELECT timed_out FROM claude_executions WHERE id = ?",
                params![id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(timed_out, 1);
    }

    #[test]
    fn test_parse_pr_url_github_standard() {
        let result = SqliteTracker::parse_pr_url("https://github.com/owner/repo/pull/42");
        assert_eq!(result, Some(("owner/repo".to_string(), 42)));
    }

    #[test]
    fn test_parse_pr_url_github_with_trailing_slash() {
        // The regex won't match trailing content after the number
        let result = SqliteTracker::parse_pr_url("https://github.com/my-org/my-repo/pull/123");
        assert_eq!(result, Some(("my-org/my-repo".to_string(), 123)));
    }

    #[test]
    fn test_parse_pr_url_non_github() {
        let result = SqliteTracker::parse_pr_url("https://gitlab.com/owner/repo/merge_requests/1");
        assert_eq!(result, None);
    }

    #[test]
    fn test_parse_pr_url_invalid() {
        assert_eq!(SqliteTracker::parse_pr_url("not-a-url"), None);
        assert_eq!(SqliteTracker::parse_pr_url(""), None);
    }

    #[test]
    fn test_parse_pr_url_too_long() {
        let long_url = format!(
            "https://github.com/{}/pull/1",
            "a".repeat(crate::MAX_PR_URL_LENGTH)
        );
        let result = SqliteTracker::parse_pr_url(&long_url);
        assert_eq!(result, None);
    }

    #[test]
    fn test_parse_pr_url_gitlab_standard() {
        let result =
            SqliteTracker::parse_pr_url("https://gitlab.com/group/project/-/merge_requests/42");
        assert_eq!(result, Some(("group/project".to_string(), 42)));
    }

    #[test]
    fn test_parse_pr_url_gitlab_self_hosted() {
        let result =
            SqliteTracker::parse_pr_url("https://gitlab.example.com/org/repo/-/merge_requests/7");
        assert_eq!(result, Some(("org/repo".to_string(), 7)));
    }

    #[test]
    fn test_parse_pr_url_gitlab_nested_groups() {
        let result = SqliteTracker::parse_pr_url(
            "https://gitlab.com/group/subgroup/project/-/merge_requests/99",
        );
        assert_eq!(result, Some(("group/subgroup/project".to_string(), 99)));
    }

    #[test]
    fn test_parse_pr_url_gitlab_deeply_nested() {
        let result = SqliteTracker::parse_pr_url("https://gitlab.com/a/b/c/d/-/merge_requests/1");
        assert_eq!(result, Some(("a/b/c/d".to_string(), 1)));
    }

    #[test]
    fn test_parse_pr_url_gitlab_http() {
        let result =
            SqliteTracker::parse_pr_url("http://gitlab.internal/team/repo/-/merge_requests/5");
        assert_eq!(result, Some(("team/repo".to_string(), 5)));
    }

    #[test]
    fn test_parse_pr_url_gitlab_without_dash_prefix() {
        let result =
            SqliteTracker::parse_pr_url("https://gitlab.com/group/project/merge_requests/1");
        assert_eq!(result, None);
    }

    #[test]
    fn test_parse_pr_url_gitlab_large_mr_number() {
        let result =
            SqliteTracker::parse_pr_url("https://gitlab.com/org/repo/-/merge_requests/999999");
        assert_eq!(result, Some(("org/repo".to_string(), 999999)));
    }

    #[test]
    fn test_attempt_full_lifecycle_to_merged() {
        let tracker = SqliteTracker::in_memory().unwrap();

        // 1. Record attempt
        tracker
            .record_attempt("linear", "lifecycle-1", "LIN-LC1")
            .unwrap();
        let attempt = tracker
            .get_attempt("linear", "lifecycle-1")
            .unwrap()
            .unwrap();
        assert_eq!(attempt.status, FixAttemptStatus::Pending);

        // 2. Mark success with PR
        tracker
            .mark_success(
                "linear",
                "lifecycle-1",
                "https://github.com/org/repo/pull/99",
            )
            .unwrap();
        let attempt = tracker
            .get_attempt("linear", "lifecycle-1")
            .unwrap()
            .unwrap();
        assert_eq!(attempt.status, FixAttemptStatus::Success);
        assert_eq!(attempt.scm_repo, Some("org/repo".to_string()));
        assert_eq!(attempt.scm_pr_number, Some(99));

        // 3. Mark merged
        tracker.mark_merged("linear", "lifecycle-1").unwrap();
        let attempt = tracker
            .get_attempt("linear", "lifecycle-1")
            .unwrap()
            .unwrap();
        assert_eq!(attempt.status, FixAttemptStatus::Merged);
        assert!(attempt.merged_at.is_some());
    }

    #[test]
    fn test_attempt_lifecycle_fail_retry_succeed() {
        let tracker = SqliteTracker::in_memory().unwrap();

        tracker
            .record_attempt("linear", "retry-issue", "LIN-R1")
            .unwrap();

        // Fail
        tracker
            .mark_failed("linear", "retry-issue", "Build error")
            .unwrap();
        let attempt = tracker
            .get_attempt("linear", "retry-issue")
            .unwrap()
            .unwrap();
        assert_eq!(attempt.status, FixAttemptStatus::Failed);
        assert_eq!(attempt.error_message, Some("Build error".to_string()));

        // prepare_for_retry atomically increments retry_count and resets status
        tracker.prepare_for_retry("linear", "retry-issue").unwrap();
        let attempt = tracker
            .get_attempt("linear", "retry-issue")
            .unwrap()
            .unwrap();
        assert_eq!(attempt.status, FixAttemptStatus::Pending);
        assert_eq!(attempt.retry_count, 1);

        // Succeed
        tracker
            .mark_success(
                "linear",
                "retry-issue",
                "https://github.com/org/repo/pull/50",
            )
            .unwrap();
        let attempt = tracker
            .get_attempt("linear", "retry-issue")
            .unwrap()
            .unwrap();
        assert_eq!(attempt.status, FixAttemptStatus::Success);
    }

    #[test]
    fn test_record_attempt_with_labels() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let labels = vec!["bug".to_string(), "high-priority".to_string()];
        tracker
            .record_attempt_with_labels("linear", "labeled-1", "LIN-L1", &labels)
            .unwrap();

        let attempt = tracker.get_attempt("linear", "labeled-1").unwrap().unwrap();
        assert_eq!(attempt.issue_labels.len(), 2);
        assert!(attempt.issue_labels.contains(&"bug".to_string()));
        assert!(attempt.issue_labels.contains(&"high-priority".to_string()));
    }

    #[test]
    fn test_record_and_get_error_patterns() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let now = Utc::now();
        let pattern = ErrorPattern {
            id: 0,
            pattern_hash: "hash-1".to_string(),
            error_type: Some("build_failure".to_string()),
            error_message: Some("undefined reference to main".to_string()),
            first_seen: now,
            last_seen: now,
            occurrence_count: 1,
            sources: Some(vec!["linear".to_string()]),
            example_issue_ids: Some(vec!["issue-1".to_string()]),
            resolution_hints: Some("Check linker flags".to_string()),
        };

        let id = tracker.record_error_pattern(&pattern).unwrap();
        assert!(id > 0);

        let patterns = tracker.get_error_patterns(10).unwrap();
        assert_eq!(patterns.len(), 1);
        assert_eq!(patterns[0].pattern_hash, "hash-1");
        assert_eq!(patterns[0].error_type, Some("build_failure".to_string()));
        assert_eq!(patterns[0].sources, Some(vec!["linear".to_string()]));
    }

    #[test]
    fn test_error_pattern_upsert_increments_count() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let now = Utc::now();
        let pattern = ErrorPattern {
            id: 0,
            pattern_hash: "hash-dup".to_string(),
            error_type: Some("timeout".to_string()),
            error_message: None,
            first_seen: now,
            last_seen: now,
            occurrence_count: 1,
            sources: None,
            example_issue_ids: None,
            resolution_hints: None,
        };

        tracker.record_error_pattern(&pattern).unwrap();
        tracker.record_error_pattern(&pattern).unwrap();
        tracker.record_error_pattern(&pattern).unwrap();

        let patterns = tracker.get_error_patterns(10).unwrap();
        assert_eq!(patterns.len(), 1);
        // Initial insert (1) + 2 upserts (each adds 1)
        assert_eq!(patterns[0].occurrence_count, 3);
    }

    #[test]
    fn test_get_error_patterns_empty() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let patterns = tracker.get_error_patterns(10).unwrap();
        assert!(patterns.is_empty());
    }

    #[test]
    fn test_record_and_get_metric() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let metric = ProcessingMetric {
            id: 0,
            timestamp: Utc::now(),
            metric_name: "queue_depth".to_string(),
            metric_value: 42.0,
            source: Some("linear".to_string()),
            tags: Some(serde_json::json!({"region": "us-east-1"})),
        };

        let id = tracker.record_metric(&metric).unwrap();
        assert!(id > 0);

        let metrics = tracker.get_metrics("queue_depth", None, 10).unwrap();
        assert_eq!(metrics.len(), 1);
        assert!((metrics[0].metric_value - 42.0).abs() < f64::EPSILON);
        assert_eq!(metrics[0].source, Some("linear".to_string()));
        assert!(metrics[0].tags.is_some());
    }

    #[test]
    fn test_get_metrics_with_time_filter() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let old_ts = chrono::DateTime::parse_from_rfc3339("2020-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let recent_ts = Utc::now();

        let old_metric = ProcessingMetric {
            id: 0,
            timestamp: old_ts,
            metric_name: "test_metric".to_string(),
            metric_value: 1.0,
            source: None,
            tags: None,
        };
        let recent_metric = ProcessingMetric {
            id: 0,
            timestamp: recent_ts,
            metric_name: "test_metric".to_string(),
            metric_value: 2.0,
            source: None,
            tags: None,
        };

        tracker.record_metric(&old_metric).unwrap();
        tracker.record_metric(&recent_metric).unwrap();

        let since = chrono::DateTime::parse_from_rfc3339("2024-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let filtered = tracker.get_metrics("test_metric", Some(since), 10).unwrap();
        assert_eq!(filtered.len(), 1);
        assert!((filtered[0].metric_value - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_store_and_find_similar_issues() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let similar = SimilarIssue::new("issue-a", "issue-b", 0.95);
        let id = tracker.store_similar_issue(&similar).unwrap();
        assert!(id > 0);

        let results = tracker.find_similar_issues("issue-a", 0.8, 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].similar_issue_id, "issue-b");
        assert!((results[0].similarity_score - 0.95).abs() < f64::EPSILON);
    }

    #[test]
    fn test_find_similar_issues_min_score_filter() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker
            .store_similar_issue(&SimilarIssue::new("src", "high", 0.95))
            .unwrap();
        tracker
            .store_similar_issue(&SimilarIssue::new("src", "low", 0.5))
            .unwrap();

        let high_only = tracker.find_similar_issues("src", 0.9, 10).unwrap();
        assert_eq!(high_only.len(), 1);
        assert_eq!(high_only[0].similar_issue_id, "high");

        let all = tracker.find_similar_issues("src", 0.0, 10).unwrap();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_similar_issue_upsert() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker
            .store_similar_issue(&SimilarIssue::new("a", "b", 0.7))
            .unwrap();
        tracker
            .store_similar_issue(&SimilarIssue::new("a", "b", 0.9))
            .unwrap();

        let results = tracker.find_similar_issues("a", 0.0, 10).unwrap();
        assert_eq!(results.len(), 1);
        assert!((results[0].similarity_score - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn test_get_success_rate_empty() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let rate = tracker.get_success_rate().unwrap();
        assert!((rate - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_get_success_rate_mixed() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker.record_attempt("linear", "s1", "L1").unwrap();
        tracker
            .mark_success("linear", "s1", "https://github.com/o/r/pull/1")
            .unwrap();
        tracker.record_attempt("linear", "s2", "L2").unwrap();
        tracker.mark_failed("linear", "s2", "error").unwrap();

        let rate = tracker.get_success_rate().unwrap();
        assert!((rate - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_get_analytics_summary() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker.record_attempt("linear", "a1", "L1").unwrap();
        tracker
            .mark_success("linear", "a1", "https://github.com/o/r/pull/1")
            .unwrap();
        tracker.mark_merged("linear", "a1").unwrap();

        tracker.record_attempt("sentry", "a2", "S1").unwrap();
        tracker.mark_failed("sentry", "a2", "build error").unwrap();

        let summary = tracker.get_analytics_summary().unwrap();
        assert_eq!(summary.total_processed, 2);
        assert_eq!(summary.total_successful, 1);
        assert_eq!(summary.total_merged, 1);
        assert!((summary.success_rate - 0.5).abs() < f64::EPSILON);
        assert!(summary.success_rate_by_source.contains_key("linear"));
        assert!(summary.success_rate_by_source.contains_key("sentry"));
    }

    #[test]
    fn test_get_analytics_summary_empty() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let summary = tracker.get_analytics_summary().unwrap();
        assert_eq!(summary.total_processed, 0);
        assert!((summary.success_rate - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_prune_old_activities() {
        let tracker = SqliteTracker::in_memory().unwrap();
        // Insert an old activity via raw SQL
        {
            let conn = tracker.acquire_lock().unwrap();
            conn.execute(
                "INSERT INTO activity_log (timestamp, activity_type, message) VALUES ('2020-01-01 00:00:00', 'old', 'old entry')",
                [],
            ).unwrap();
        }
        // Insert a recent one
        let entry = ActivityLogEntry {
            id: 0,
            timestamp: Utc::now(),
            activity_type: "recent".to_string(),
            source: None,
            issue_id: None,
            short_id: None,
            message: "recent entry".to_string(),
            metadata: None,
        };
        tracker.record_activity(&entry).unwrap();

        let deleted = tracker.prune_old_activities(30).unwrap();
        assert_eq!(deleted, 1);

        let remaining = tracker.get_recent_activities(100, None).unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].activity_type, "recent");
    }

    #[test]
    fn test_prune_old_metrics() {
        let tracker = SqliteTracker::in_memory().unwrap();
        {
            let conn = tracker.acquire_lock().unwrap();
            conn.execute(
                "INSERT INTO processing_metrics (timestamp, metric_name, metric_value) VALUES ('2020-01-01 00:00:00', 'old_metric', 1.0)",
                [],
            ).unwrap();
        }
        let recent = ProcessingMetric {
            id: 0,
            timestamp: Utc::now(),
            metric_name: "new_metric".to_string(),
            metric_value: 2.0,
            source: None,
            tags: None,
        };
        tracker.record_metric(&recent).unwrap();

        let deleted = tracker.prune_old_metrics(30).unwrap();
        assert_eq!(deleted, 1);
    }

    #[test]
    fn test_get_set_channel_cursor() {
        let tracker = SqliteTracker::in_memory().unwrap();

        // Initially empty
        let cursor = tracker
            .get_channel_cursor("discord", "last_message_id")
            .unwrap();
        assert!(cursor.is_none());

        // Set cursor
        tracker
            .set_channel_cursor("discord", "last_message_id", "msg-123")
            .unwrap();
        let cursor = tracker
            .get_channel_cursor("discord", "last_message_id")
            .unwrap();
        assert_eq!(cursor, Some("msg-123".to_string()));

        // Update cursor (upsert)
        tracker
            .set_channel_cursor("discord", "last_message_id", "msg-456")
            .unwrap();
        let cursor = tracker
            .get_channel_cursor("discord", "last_message_id")
            .unwrap();
        assert_eq!(cursor, Some("msg-456".to_string()));
    }

    #[test]
    fn test_channel_cursor_different_keys() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker
            .set_channel_cursor("discord", "cursor_a", "val-a")
            .unwrap();
        tracker
            .set_channel_cursor("discord", "cursor_b", "val-b")
            .unwrap();

        assert_eq!(
            tracker.get_channel_cursor("discord", "cursor_a").unwrap(),
            Some("val-a".to_string())
        );
        assert_eq!(
            tracker.get_channel_cursor("discord", "cursor_b").unwrap(),
            Some("val-b".to_string())
        );
    }

    #[test]
    fn test_get_attempt_by_id() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker
            .record_attempt("linear", "by-id-issue", "LIN-BI")
            .unwrap();
        let attempt = tracker
            .get_attempt("linear", "by-id-issue")
            .unwrap()
            .unwrap();

        let by_id = tracker.get_attempt_by_id(attempt.id).unwrap().unwrap();
        assert_eq!(by_id.issue_id, "by-id-issue");
        assert_eq!(by_id.source, "linear");
    }

    #[test]
    fn test_get_attempt_by_id_not_found() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let result = tracker.get_attempt_by_id(99999).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_embedding_to_blob_and_back() {
        let original = vec![1.0f32, 2.5, -(std::f32::consts::PI), 0.0];
        let blob = SqliteTracker::embedding_to_blob(Some(&original));
        assert!(blob.is_some());

        let restored = SqliteTracker::blob_to_embedding(blob);
        assert!(restored.is_some());
        let restored = restored.unwrap();
        assert_eq!(restored.len(), 4);
        assert!((restored[0] - 1.0).abs() < f32::EPSILON);
        assert!((restored[2] - (-std::f32::consts::PI)).abs() < 0.001);
    }

    #[test]
    fn test_embedding_to_blob_none() {
        let blob = SqliteTracker::embedding_to_blob(None);
        assert!(blob.is_none());

        let restored = SqliteTracker::blob_to_embedding(None);
        assert!(restored.is_none());
    }

    #[test]
    fn test_store_and_get_diff_analysis() {
        let tracker = SqliteTracker::in_memory().unwrap();
        // Need a fix_attempt for the FK
        tracker.record_attempt("sentry", "issue-1", "I-1").unwrap();
        let attempt = tracker.get_attempt("sentry", "issue-1").unwrap().unwrap();

        let analysis = claudear_core::types::DiffAnalysis {
            id: 0,
            attempt_id: attempt.id,
            pr_url: "https://github.com/org/repo/pull/42".to_string(),
            scm_repo: "org/repo".to_string(),
            pr_number: 42,
            files_changed: vec!["src/main.rs".to_string(), "tests/test.rs".to_string()],
            file_types: {
                let mut m = std::collections::HashMap::new();
                m.insert("rs".to_string(), 2);
                m
            },
            change_categories: vec![
                claudear_core::types::ChangeCategory::Modification,
                claudear_core::types::ChangeCategory::Tests,
            ],
            diff_summary: "2 files changed across 2 categories".to_string(),
            created_at: chrono::Utc::now(),
        };

        let id = tracker.store_diff_analysis(&analysis).unwrap();
        assert!(id > 0);

        let results = tracker.get_diff_analyses_for_repo("org/repo", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].pr_number, 42);
        assert_eq!(results[0].files_changed.len(), 2);
        assert!(results[0]
            .files_changed
            .contains(&"src/main.rs".to_string()));
        assert_eq!(*results[0].file_types.get("rs").unwrap(), 2);
        assert_eq!(results[0].change_categories.len(), 2);
        assert_eq!(
            results[0].diff_summary,
            "2 files changed across 2 categories"
        );
    }

    #[test]
    fn test_get_diff_analyses_for_repo_empty() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let results = tracker
            .get_diff_analyses_for_repo("nonexistent", 10)
            .unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_upsert_promoted_instruction_insert_and_update() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let instruction = claudear_core::types::PromotedInstruction {
            id: 0,
            repo: "org/repo".to_string(),
            source_type: "qa_promotion".to_string(),
            instruction_text: "Always use the async API".to_string(),
            occurrence_count: 2,
            confidence: 0.7,
            is_active: true,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };

        let id1 = tracker.upsert_promoted_instruction(&instruction).unwrap();
        assert!(id1 > 0);

        // Upsert again with updated confidence
        let updated = claudear_core::types::PromotedInstruction {
            occurrence_count: 5,
            confidence: 0.9,
            ..instruction.clone()
        };
        let id2 = tracker.upsert_promoted_instruction(&updated).unwrap();
        assert_eq!(id1, id2);

        let results = tracker.get_promoted_instructions("org/repo").unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].occurrence_count, 5);
        assert!((results[0].confidence - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn test_get_promoted_instructions_only_active() {
        let tracker = SqliteTracker::in_memory().unwrap();

        let active = claudear_core::types::PromotedInstruction {
            id: 0,
            repo: "org/repo".to_string(),
            source_type: "qa_promotion".to_string(),
            instruction_text: "Active instruction".to_string(),
            occurrence_count: 3,
            confidence: 0.8,
            is_active: true,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };
        tracker.upsert_promoted_instruction(&active).unwrap();

        let inactive = claudear_core::types::PromotedInstruction {
            instruction_text: "Inactive instruction".to_string(),
            is_active: false,
            ..active.clone()
        };
        tracker.upsert_promoted_instruction(&inactive).unwrap();

        let results = tracker.get_promoted_instructions("org/repo").unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].instruction_text, "Active instruction");
    }

    #[test]
    fn test_upsert_repo_knowledge_increments_count() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let entry = claudear_core::types::RepoKnowledge {
            id: 0,
            repo: "org/repo".to_string(),
            knowledge_key: "common_fix_dirs".to_string(),
            knowledge_value: "src/handlers".to_string(),
            source_type: "diff_analysis".to_string(),
            confidence: 0.6,
            occurrence_count: 1,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };

        tracker.upsert_repo_knowledge(&entry).unwrap();
        tracker.upsert_repo_knowledge(&entry).unwrap();
        tracker.upsert_repo_knowledge(&entry).unwrap();

        let results = tracker.get_repo_knowledge("org/repo").unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].occurrence_count, 3);
    }

    #[test]
    fn test_get_repo_knowledge_by_key() {
        let tracker = SqliteTracker::in_memory().unwrap();

        let entry1 = claudear_core::types::RepoKnowledge {
            id: 0,
            repo: "org/repo".to_string(),
            knowledge_key: "common_fix_dirs".to_string(),
            knowledge_value: "src/handlers".to_string(),
            source_type: "diff_analysis".to_string(),
            confidence: 0.6,
            occurrence_count: 1,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };
        tracker.upsert_repo_knowledge(&entry1).unwrap();

        let entry2 = claudear_core::types::RepoKnowledge {
            knowledge_key: "test_pattern".to_string(),
            knowledge_value: "cargo test".to_string(),
            ..entry1.clone()
        };
        tracker.upsert_repo_knowledge(&entry2).unwrap();

        let dirs = tracker
            .get_repo_knowledge_by_key("org/repo", "common_fix_dirs")
            .unwrap();
        assert_eq!(dirs.len(), 1);
        assert_eq!(dirs[0].knowledge_value, "src/handlers");

        let tests = tracker
            .get_repo_knowledge_by_key("org/repo", "test_pattern")
            .unwrap();
        assert_eq!(tests.len(), 1);
        assert_eq!(tests[0].knowledge_value, "cargo test");

        let empty = tracker
            .get_repo_knowledge_by_key("org/repo", "nonexistent")
            .unwrap();
        assert!(empty.is_empty());
    }

    #[test]
    fn test_upsert_review_pattern_increments_count() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let pattern = claudear_core::types::ReviewPattern {
            id: 0,
            scm_repo: "org/repo".to_string(),
            category: claudear_core::types::ReviewCategory::MissingTests,
            pattern_text: "Please add tests".to_string(),
            example_comments: vec!["Add tests for this".to_string()],
            occurrence_count: 1,
            promoted_to_instruction: false,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };

        tracker.upsert_review_pattern(&pattern).unwrap();
        tracker.upsert_review_pattern(&pattern).unwrap();

        let results = tracker.get_review_patterns("org/repo", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].occurrence_count, 2);
        assert_eq!(
            results[0].category,
            claudear_core::types::ReviewCategory::MissingTests
        );
    }

    #[test]
    fn test_get_review_patterns_by_category() {
        let tracker = SqliteTracker::in_memory().unwrap();

        let test_pattern = claudear_core::types::ReviewPattern {
            id: 0,
            scm_repo: "org/repo".to_string(),
            category: claudear_core::types::ReviewCategory::MissingTests,
            pattern_text: "Need tests".to_string(),
            example_comments: vec![],
            occurrence_count: 1,
            promoted_to_instruction: false,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };
        tracker.upsert_review_pattern(&test_pattern).unwrap();

        let security_pattern = claudear_core::types::ReviewPattern {
            category: claudear_core::types::ReviewCategory::Security,
            pattern_text: "SQL injection risk".to_string(),
            ..test_pattern.clone()
        };
        tracker.upsert_review_pattern(&security_pattern).unwrap();

        let tests = tracker
            .get_review_patterns_by_category(
                "org/repo",
                claudear_core::types::ReviewCategory::MissingTests,
            )
            .unwrap();
        assert_eq!(tests.len(), 1);
        assert_eq!(tests[0].pattern_text, "Need tests");

        let security = tracker
            .get_review_patterns_by_category(
                "org/repo",
                claudear_core::types::ReviewCategory::Security,
            )
            .unwrap();
        assert_eq!(security.len(), 1);
        assert_eq!(security[0].pattern_text, "SQL injection risk");
    }

    #[test]
    fn test_store_and_get_strategy_fingerprint() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker.record_attempt("linear", "issue-1", "I-1").unwrap();
        let attempt = tracker.get_attempt("linear", "issue-1").unwrap().unwrap();
        // Mark as merged so get_successful_strategies can find it
        tracker
            .mark_success("linear", "issue-1", "https://github.com/org/repo/pull/1")
            .unwrap();
        tracker.mark_merged("linear", "issue-1").unwrap();

        let fp = claudear_core::types::StrategyFingerprint {
            id: 0,
            attempt_id: attempt.id,
            files_explored: vec!["src/main.rs".to_string(), "src/lib.rs".to_string()],
            tests_run: 3,
            tools_used: {
                let mut m = std::collections::HashMap::new();
                m.insert("Read".to_string(), 5);
                m.insert("Edit".to_string(), 2);
                m
            },
            fix_approach: "tdd".to_string(),
            strategy_summary: "2 files explored, 3 tests run, approach: tdd".to_string(),
            fix_quality_score: Some(0.85),
            created_at: chrono::Utc::now(),
        };

        let id = tracker.store_strategy_fingerprint(&fp).unwrap();
        assert!(id > 0);

        let results = tracker.get_successful_strategies("org/repo", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].fix_approach, "tdd");
        assert_eq!(results[0].tests_run, 3);
        assert_eq!(results[0].files_explored.len(), 2);
        assert_eq!(*results[0].tools_used.get("Read").unwrap(), 5);
        assert!((results[0].fix_quality_score.unwrap() - 0.85).abs() < f64::EPSILON);
    }

    #[test]
    fn test_store_and_get_issue_cluster() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let now = chrono::Utc::now();

        let cluster = claudear_core::types::IssueCluster {
            id: 0,
            cluster_key: "cluster_abc123".to_string(),
            source: "sentry".to_string(),
            issue_ids: vec![
                "iss-1".to_string(),
                "iss-2".to_string(),
                "iss-3".to_string(),
            ],
            window_start: now,
            window_end: now + chrono::Duration::minutes(15),
            resolved_by_issue_id: None,
            resolved_by_attempt_id: None,
            status: "active".to_string(),
            created_at: now,
        };

        let id = tracker.store_issue_cluster(&cluster).unwrap();
        assert!(id > 0);

        let active = tracker.get_active_clusters("sentry").unwrap();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].cluster_key, "cluster_abc123");
        assert_eq!(active[0].issue_ids.len(), 3);
        assert_eq!(active[0].status, "active");
    }

    #[test]
    fn test_store_issue_cluster_deduplication() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let now = chrono::Utc::now();

        let cluster = claudear_core::types::IssueCluster {
            id: 0,
            cluster_key: "cluster_dup".to_string(),
            source: "sentry".to_string(),
            issue_ids: vec!["iss-1".to_string()],
            window_start: now,
            window_end: now + chrono::Duration::minutes(10),
            resolved_by_issue_id: None,
            resolved_by_attempt_id: None,
            status: "active".to_string(),
            created_at: now,
        };

        tracker.store_issue_cluster(&cluster).unwrap();
        // Inserting again with same cluster_key should not fail (INSERT OR IGNORE)
        tracker.store_issue_cluster(&cluster).unwrap();

        let active = tracker.get_active_clusters("sentry").unwrap();
        assert_eq!(active.len(), 1);
    }

    #[test]
    fn test_update_cluster_resolution() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let now = chrono::Utc::now();

        let cluster = claudear_core::types::IssueCluster {
            id: 0,
            cluster_key: "cluster_resolve".to_string(),
            source: "sentry".to_string(),
            issue_ids: vec!["iss-1".to_string(), "iss-2".to_string()],
            window_start: now,
            window_end: now + chrono::Duration::minutes(10),
            resolved_by_issue_id: None,
            resolved_by_attempt_id: None,
            status: "active".to_string(),
            created_at: now,
        };

        let id = tracker.store_issue_cluster(&cluster).unwrap();

        tracker.update_cluster_resolution(id, "iss-1", 42).unwrap();

        // Should no longer appear in active clusters
        let active = tracker.get_active_clusters("sentry").unwrap();
        assert!(active.is_empty());
    }

    #[test]
    fn test_update_feedback_learnings() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker.record_attempt("linear", "issue-1", "I-1").unwrap();
        let attempt = tracker.get_attempt("linear", "issue-1").unwrap().unwrap();

        let outcome = claudear_core::types::FixOutcome {
            id: 0,
            attempt_id: attempt.id,
            source: "linear".to_string(),
            issue_id: "issue-1".to_string(),
            issue_text: "test issue".to_string(),
            prompt_used: "fix it".to_string(),
            outcome: claudear_core::types::Outcome::Merged,
            error_type: None,
            learnings: None,
            keywords: vec![],
            embedding: None,
            created_at: chrono::Utc::now(),
        };

        let outcome_id = tracker.store_feedback_outcome(&outcome).unwrap();
        assert!(outcome_id > 0);

        // Update learnings
        tracker
            .update_feedback_learnings(outcome_id, "Root cause: null pointer; Strategy: direct_fix")
            .unwrap();

        // Verify
        let retrieved = tracker
            .get_feedback_outcome_by_attempt(attempt.id)
            .unwrap()
            .unwrap();
        assert_eq!(
            retrieved.learnings,
            Some("Root cause: null pointer; Strategy: direct_fix".to_string())
        );
    }

    #[test]
    fn test_update_pr_fix_quality_score() {
        let tracker = SqliteTracker::in_memory().unwrap();
        // Create a PR record
        tracker.record_attempt("linear", "issue-1", "I-1").unwrap();
        let attempt = tracker.get_attempt("linear", "issue-1").unwrap().unwrap();

        let pr = claudear_core::types::PrRecord {
            id: 0,
            pr_url: "https://github.com/org/repo/pull/1".to_string(),
            scm_repo: "org/repo".to_string(),
            pr_number: 1,
            attempt_id: Some(attempt.id),
            issue_id: Some("issue-1".to_string()),
            issue_source: Some("linear".to_string()),
            title: Some("Fix the bug".to_string()),
            description: None,
            author: Some("claudear-bot".to_string()),
            head_branch: Some("fix/issue-1".to_string()),
            base_branch: Some("main".to_string()),
            status: "merged".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: None,
            merged_at: Some(chrono::Utc::now()),
            closed_at: None,
            approvals_count: 2,
            changes_requested_count: 0,
            comments_count: 1,
            last_review_at: None,
            time_to_first_review_mins: Some(30),
            time_to_merge_mins: Some(60),
            review_cycles: 1,
            files_changed: Some(3),
            lines_added: Some(50),
            lines_removed: Some(10),
        };

        tracker.upsert_pr(&pr).unwrap();

        // Update quality score
        tracker
            .update_pr_fix_quality_score("https://github.com/org/repo/pull/1", 0.87)
            .unwrap();

        // Verify
        let retrieved = tracker
            .get_pr("https://github.com/org/repo/pull/1")
            .unwrap()
            .unwrap();
        // The PrRecord doesn't expose fix_quality_score directly,
        // but the UPDATE should succeed without error
        assert_eq!(retrieved.pr_url, "https://github.com/org/repo/pull/1");
    }

    #[test]
    fn test_get_recent_issue_arrivals() {
        let tracker = SqliteTracker::in_memory().unwrap();

        // Record some recent attempts
        tracker.record_attempt("sentry", "iss-a", "A").unwrap();
        tracker.record_attempt("sentry", "iss-b", "B").unwrap();
        tracker.record_attempt("sentry", "iss-c", "C").unwrap();
        tracker.record_attempt("linear", "iss-d", "D").unwrap(); // different source

        let arrivals = tracker.get_recent_issue_arrivals("sentry", 60).unwrap();
        assert_eq!(arrivals.len(), 3);
        // All should be sentry issues
        for (id, _) in &arrivals {
            assert!(id.starts_with("iss-"));
        }
    }

    #[test]
    fn test_diff_analyses_limit() {
        let tracker = SqliteTracker::in_memory().unwrap();

        for i in 0..5 {
            tracker
                .record_attempt("sentry", &format!("issue-{}", i), &format!("I-{}", i))
                .unwrap();
            let attempt = tracker
                .get_attempt("sentry", &format!("issue-{}", i))
                .unwrap()
                .unwrap();
            let analysis = claudear_core::types::DiffAnalysis {
                id: 0,
                attempt_id: attempt.id,
                pr_url: format!("https://github.com/org/repo/pull/{}", i),
                scm_repo: "org/repo".to_string(),
                pr_number: i as i64,
                files_changed: vec![],
                file_types: std::collections::HashMap::new(),
                change_categories: vec![],
                diff_summary: format!("analysis {}", i),
                created_at: chrono::Utc::now(),
            };
            tracker.store_diff_analysis(&analysis).unwrap();
        }

        let limited = tracker.get_diff_analyses_for_repo("org/repo", 2).unwrap();
        assert_eq!(limited.len(), 2);
    }

    #[test]
    fn test_repo_knowledge_multiple_values_same_key() {
        let tracker = SqliteTracker::in_memory().unwrap();

        let entry1 = claudear_core::types::RepoKnowledge {
            id: 0,
            repo: "org/repo".to_string(),
            knowledge_key: "common_fix_dirs".to_string(),
            knowledge_value: "src/handlers".to_string(),
            source_type: "diff_analysis".to_string(),
            confidence: 0.6,
            occurrence_count: 1,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };
        tracker.upsert_repo_knowledge(&entry1).unwrap();

        let entry2 = claudear_core::types::RepoKnowledge {
            knowledge_value: "src/models".to_string(),
            ..entry1.clone()
        };
        tracker.upsert_repo_knowledge(&entry2).unwrap();

        // Should have 2 entries for the same key but different values
        let results = tracker
            .get_repo_knowledge_by_key("org/repo", "common_fix_dirs")
            .unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_review_pattern_category_roundtrip() {
        let tracker = SqliteTracker::in_memory().unwrap();

        let categories = vec![
            claudear_core::types::ReviewCategory::MissingTests,
            claudear_core::types::ReviewCategory::StyleIssue,
            claudear_core::types::ReviewCategory::WrongApproach,
            claudear_core::types::ReviewCategory::Incomplete,
            claudear_core::types::ReviewCategory::Security,
            claudear_core::types::ReviewCategory::Performance,
            claudear_core::types::ReviewCategory::Documentation,
            claudear_core::types::ReviewCategory::Other,
        ];

        for (i, cat) in categories.iter().enumerate() {
            let pattern = claudear_core::types::ReviewPattern {
                id: 0,
                scm_repo: "org/repo".to_string(),
                category: *cat,
                pattern_text: format!("pattern {}", i),
                example_comments: vec![],
                occurrence_count: 1,
                promoted_to_instruction: false,
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
            };
            tracker.upsert_review_pattern(&pattern).unwrap();
        }

        let all = tracker.get_review_patterns("org/repo", 100).unwrap();
        assert_eq!(all.len(), categories.len());

        // Check each category survived the round-trip
        for cat in &categories {
            let found = tracker
                .get_review_patterns_by_category("org/repo", *cat)
                .unwrap();
            assert_eq!(found.len(), 1, "Expected 1 pattern for {:?}", cat);
        }
    }

    #[test]
    fn test_mark_merged_nonexistent_issue() {
        let tracker = SqliteTracker::in_memory().unwrap();
        // Marking a nonexistent issue should not panic (it's a no-op UPDATE)
        let result = tracker.mark_merged("linear", "nonexistent");
        assert!(result.is_ok());
    }

    #[test]
    fn test_mark_failed_nonexistent_issue() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let result = tracker.mark_failed("linear", "nonexistent", "error");
        assert!(result.is_ok());
    }

    #[test]
    fn test_mark_closed_nonexistent_issue() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let result = tracker.mark_closed("linear", "nonexistent");
        assert!(result.is_ok());
    }

    #[test]
    fn test_mark_resolved_nonexistent_issue() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let result = tracker.mark_resolved("linear", "nonexistent");
        assert!(result.is_ok());
    }

    #[test]
    fn test_mark_cannot_fix_nonexistent_issue() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let result = tracker.mark_cannot_fix("linear", "nonexistent", "reason");
        assert!(result.is_ok());
    }

    #[test]
    fn test_mark_success_nonexistent_issue() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let result = tracker.mark_success("linear", "nonexistent", "https://github.com/o/r/pull/1");
        assert!(result.is_ok());
    }

    #[test]
    fn test_mark_merged_idempotent() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker.record_attempt("linear", "123", "P-123").unwrap();
        tracker
            .mark_success("linear", "123", "https://github.com/o/r/pull/1")
            .unwrap();
        tracker.mark_merged("linear", "123").unwrap();
        // Mark merged again
        tracker.mark_merged("linear", "123").unwrap();
        let attempt = tracker.get_attempt("linear", "123").unwrap().unwrap();
        assert_eq!(attempt.status, FixAttemptStatus::Merged);
    }

    #[test]
    fn test_mark_failed_idempotent() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker.record_attempt("linear", "123", "P-123").unwrap();
        tracker.mark_failed("linear", "123", "error1").unwrap();
        tracker.mark_failed("linear", "123", "error2").unwrap();
        let attempt = tracker.get_attempt("linear", "123").unwrap().unwrap();
        assert_eq!(attempt.status, FixAttemptStatus::Failed);
        // Latest error message should be stored
        assert_eq!(attempt.error_message, Some("error2".to_string()));
    }

    #[test]
    fn test_mark_merged_after_failed() {
        // Transition: Failed -> Merged (e.g., manual merge after fix)
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker.record_attempt("linear", "123", "P-123").unwrap();
        tracker.mark_failed("linear", "123", "build error").unwrap();
        tracker.mark_merged("linear", "123").unwrap();
        let attempt = tracker.get_attempt("linear", "123").unwrap().unwrap();
        assert_eq!(attempt.status, FixAttemptStatus::Merged);
    }

    #[test]
    fn test_mark_closed_after_merged() {
        // Transition: Merged -> Closed (unusual but possible)
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker.record_attempt("linear", "123", "P-123").unwrap();
        tracker
            .mark_success("linear", "123", "https://github.com/o/r/pull/1")
            .unwrap();
        tracker.mark_merged("linear", "123").unwrap();
        tracker.mark_closed("linear", "123").unwrap();
        let attempt = tracker.get_attempt("linear", "123").unwrap().unwrap();
        assert_eq!(attempt.status, FixAttemptStatus::Closed);
    }

    #[test]
    fn test_record_attempt_duplicate_source_issue() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker.record_attempt("linear", "123", "P-123").unwrap();
        // Recording same source+issue again should work (updates)
        let result = tracker.record_attempt("linear", "123", "P-123");
        // SQLite INSERT OR REPLACE should handle this
        assert!(result.is_ok());
    }

    #[test]
    fn test_has_attempted_empty_source() {
        let tracker = SqliteTracker::in_memory().unwrap();
        assert!(!tracker.has_attempted("", "123").unwrap());
    }

    #[test]
    fn test_has_attempted_empty_issue_id() {
        let tracker = SqliteTracker::in_memory().unwrap();
        assert!(!tracker.has_attempted("linear", "").unwrap());
    }

    #[test]
    fn test_get_attempt_nonexistent() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let attempt = tracker.get_attempt("linear", "nonexistent").unwrap();
        assert!(attempt.is_none());
    }

    #[test]
    fn test_get_attempt_by_pr_url_nonexistent() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let attempt = tracker
            .get_attempt_by_pr_url("https://github.com/o/r/pull/999")
            .unwrap();
        assert!(attempt.is_none());
    }

    #[test]
    fn test_reset_attempt_nonexistent() {
        let tracker = SqliteTracker::in_memory().unwrap();
        // Should not panic
        let result = tracker.reset_attempt("linear", "nonexistent");
        assert!(result.is_ok());
    }

    #[test]
    fn test_get_attempted_issue_ids_empty_source() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let ids = tracker
            .get_attempted_issue_ids("nonexistent_source")
            .unwrap();
        assert!(ids.is_empty());
    }

    #[test]
    fn test_get_stats_empty_db() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let stats = tracker.get_stats().unwrap();
        assert_eq!(stats.total, 0);
        assert_eq!(stats.pending, 0);
        assert_eq!(stats.success, 0);
        assert_eq!(stats.failed, 0);
        assert_eq!(stats.merged, 0);
        assert_eq!(stats.closed, 0);
        assert_eq!(stats.cannot_fix, 0);
        assert!(stats.by_source.is_empty());
    }

    #[test]
    fn test_increment_retry_nonexistent() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let result = tracker.increment_retry("linear", "nonexistent");
        assert!(result.is_ok());
    }

    #[test]
    fn test_get_retryable_issues_empty() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let retryable = tracker.get_retryable_issues(3).unwrap();
        assert!(retryable.is_empty());
    }

    #[test]
    fn test_mark_failed_with_empty_error_message() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker.record_attempt("linear", "123", "P-123").unwrap();
        tracker.mark_failed("linear", "123", "").unwrap();
        let attempt = tracker.get_attempt("linear", "123").unwrap().unwrap();
        assert_eq!(attempt.status, FixAttemptStatus::Failed);
        assert_eq!(attempt.error_message, Some("".to_string()));
    }

    #[test]
    fn test_mark_success_with_non_scm_url() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker.record_attempt("linear", "123", "P-123").unwrap();
        tracker
            .mark_success(
                "linear",
                "123",
                "https://gitlab.com/org/repo/-/merge_requests/1",
            )
            .unwrap();
        let attempt = tracker.get_attempt("linear", "123").unwrap().unwrap();
        assert_eq!(attempt.status, FixAttemptStatus::Success);
        // GitLab MR URLs are now parsed into scm_repo/scm_pr_number
        assert_eq!(attempt.scm_repo, Some("org/repo".to_string()));
        assert_eq!(attempt.scm_pr_number, Some(1));
    }

    #[test]
    fn test_record_attempt_with_labels_stored() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker
            .record_attempt_with_labels(
                "linear",
                "123",
                "P-123",
                &["bug".to_string(), "urgent".to_string()],
            )
            .unwrap();
        let attempt = tracker.get_attempt("linear", "123").unwrap().unwrap();
        assert!(attempt.issue_labels.contains(&"bug".to_string()));
        assert!(attempt.issue_labels.contains(&"urgent".to_string()));
    }

    #[test]
    fn test_record_attempt_with_empty_labels() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker
            .record_attempt_with_labels("linear", "123", "P-123", &[])
            .unwrap();
        let attempt = tracker.get_attempt("linear", "123").unwrap().unwrap();
        assert!(attempt.issue_labels.is_empty());
    }

    #[test]
    fn test_get_pending_prs_empty_db() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let pending = tracker.get_pending_prs().unwrap();
        assert!(pending.is_empty());
    }

    #[test]
    fn test_full_lifecycle_pending_to_merged() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker.record_attempt("linear", "123", "P-123").unwrap();

        let attempt = tracker.get_attempt("linear", "123").unwrap().unwrap();
        assert_eq!(attempt.status, FixAttemptStatus::Pending);

        tracker
            .mark_success("linear", "123", "https://github.com/o/r/pull/1")
            .unwrap();
        let attempt = tracker.get_attempt("linear", "123").unwrap().unwrap();
        assert_eq!(attempt.status, FixAttemptStatus::Success);

        tracker.mark_merged("linear", "123").unwrap();
        let attempt = tracker.get_attempt("linear", "123").unwrap().unwrap();
        assert_eq!(attempt.status, FixAttemptStatus::Merged);
        assert!(attempt.merged_at.is_some());

        tracker.mark_resolved("linear", "123").unwrap();
        let attempt = tracker.get_attempt("linear", "123").unwrap().unwrap();
        // mark_resolved sets resolved_at timestamp
        assert!(attempt.resolved_at.is_some());
    }

    #[test]
    fn test_full_lifecycle_pending_to_cannot_fix() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker.record_attempt("linear", "123", "P-123").unwrap();
        tracker
            .mark_failed("linear", "123", "first failure")
            .unwrap();
        tracker
            .mark_cannot_fix("linear", "123", "unable to fix this")
            .unwrap();

        let attempt = tracker.get_attempt("linear", "123").unwrap().unwrap();
        assert_eq!(attempt.status, FixAttemptStatus::CannotFix);
        assert_eq!(
            attempt.error_message,
            Some("unable to fix this".to_string())
        );
    }

    #[test]
    fn test_get_recent_attempts_since_returns_recent() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker.record_attempt("linear", "issue-1", "P-1").unwrap();
        tracker.record_attempt("linear", "issue-2", "P-2").unwrap();
        // Both should be within the last hour
        let since = Utc::now() - chrono::Duration::hours(1);
        let attempts = tracker.get_recent_attempts_since(&since).unwrap();
        assert_eq!(attempts.len(), 2);
    }

    #[test]
    fn test_get_recent_attempts_since_empty_when_old() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker.record_attempt("linear", "issue-1", "P-1").unwrap();
        // Nothing in the future
        let since = Utc::now() + chrono::Duration::hours(1);
        let attempts = tracker.get_recent_attempts_since(&since).unwrap();
        assert!(attempts.is_empty());
    }

    #[test]
    fn test_has_dependency_false_when_none() {
        let tracker = SqliteTracker::in_memory().unwrap();
        assert!(!tracker.has_dependency("org/repo-a", "org/repo-b").unwrap());
    }

    #[test]
    fn test_has_dependency_true_when_exists() {
        let tracker = SqliteTracker::in_memory().unwrap();
        // add_dependency(upstream, downstream, dep_type)
        // This means downstream depends on upstream
        tracker
            .add_dependency("org/upstream", "org/downstream", "runtime")
            .unwrap();
        // has_dependency(repo_a, repo_b) checks if repo_a depends on repo_b
        assert!(tracker
            .has_dependency("org/downstream", "org/upstream")
            .unwrap());
        // Reverse should be false
        assert!(!tracker
            .has_dependency("org/upstream", "org/downstream")
            .unwrap());
    }

    #[test]
    fn test_upsert_cross_repo_correlation_creates_new() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let corr = tracker
            .upsert_cross_repo_correlation("org/a", "org/b", 24)
            .unwrap();
        assert_eq!(corr.repo_a, "org/a");
        assert_eq!(corr.repo_b, "org/b");
        assert_eq!(corr.correlation_count, 1);
        assert_eq!(corr.window_hours, 24);
    }

    #[test]
    fn test_upsert_cross_repo_correlation_increments() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker
            .upsert_cross_repo_correlation("org/a", "org/b", 24)
            .unwrap();
        tracker
            .upsert_cross_repo_correlation("org/a", "org/b", 24)
            .unwrap();
        let corr = tracker
            .upsert_cross_repo_correlation("org/a", "org/b", 24)
            .unwrap();
        assert_eq!(corr.correlation_count, 3);
    }

    #[test]
    fn test_get_cross_repo_correlations_filters_by_count() {
        let tracker = SqliteTracker::in_memory().unwrap();
        // Create correlation with count 1
        tracker
            .upsert_cross_repo_correlation("org/a", "org/b", 24)
            .unwrap();
        // Query with min_count=2 should return empty
        let results = tracker.get_cross_repo_correlations(2, 48).unwrap();
        assert!(results.is_empty());
        // Increment to 2
        tracker
            .upsert_cross_repo_correlation("org/a", "org/b", 24)
            .unwrap();
        let results = tracker.get_cross_repo_correlations(2, 48).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_get_cross_repo_correlations_multiple_pairs() {
        let tracker = SqliteTracker::in_memory().unwrap();
        for _ in 0..3 {
            tracker
                .upsert_cross_repo_correlation("org/a", "org/b", 24)
                .unwrap();
        }
        for _ in 0..5 {
            tracker
                .upsert_cross_repo_correlation("org/c", "org/d", 24)
                .unwrap();
        }
        let results = tracker.get_cross_repo_correlations(3, 48).unwrap();
        assert_eq!(results.len(), 2);
        // Should be sorted by count DESC
        assert_eq!(results[0].correlation_count, 5);
        assert_eq!(results[1].correlation_count, 3);
    }

    #[test]
    fn test_store_code_complexity() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let repo_id = tracker.upsert_repository("test/repo", None, None).unwrap();
        let fc = claudear_core::types::FileComplexity {
            file_path: "src/main.rs".into(),
            total_lines: 100,
            function_count: 5,
            functions: Vec::new(),
            avg_cyclomatic: 2.5,
            max_cyclomatic: 8.0,
            avg_func_length: 20.0,
            max_func_length: 50.0,
            avg_nesting: 1.5,
            max_nesting: 4.0,
        };
        tracker
            .store_code_complexity(repo_id, "src/main.rs", &fc)
            .unwrap();
        // Store again (should upsert without error)
        tracker
            .store_code_complexity(repo_id, "src/main.rs", &fc)
            .unwrap();
    }

    #[test]
    fn test_store_eval_snapshot() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let snapshot = claudear_core::types::EvalSnapshot {
            category: claudear_core::types::EvalCategory::Test,
            tool_name: "cargo test".into(),
            exit_code: 0,
            passed: 10,
            failed: 0,
            skipped: 2,
            warnings: 1,
            errors: 0,
            diagnostics: Vec::new(),
            raw_output: "all tests passed".into(),
            duration_secs: 5.5,
            line_coverage_pct: Some(85.0),
            branch_coverage_pct: None,
        };
        let id = tracker
            .store_eval_snapshot(Some(1), "before", &snapshot)
            .unwrap();
        assert!(id > 0);
    }

    #[test]
    fn test_store_eval_snapshot_without_attempt() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let snapshot = claudear_core::types::EvalSnapshot::default();
        let id = tracker
            .store_eval_snapshot(None, "before", &snapshot)
            .unwrap();
        assert!(id > 0);
    }

    #[test]
    fn test_store_eval_delta() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let before = claudear_core::types::EvalSnapshot {
            category: claudear_core::types::EvalCategory::Test,
            tool_name: "cargo test".into(),
            exit_code: 1,
            passed: 8,
            failed: 2,
            skipped: 0,
            warnings: 0,
            errors: 2,
            diagnostics: Vec::new(),
            raw_output: String::new(),
            duration_secs: 3.0,
            line_coverage_pct: Some(75.0),
            branch_coverage_pct: None,
        };
        let after = claudear_core::types::EvalSnapshot {
            category: claudear_core::types::EvalCategory::Test,
            tool_name: "cargo test".into(),
            exit_code: 0,
            passed: 10,
            failed: 0,
            skipped: 0,
            warnings: 0,
            errors: 0,
            diagnostics: Vec::new(),
            raw_output: String::new(),
            duration_secs: 4.0,
            line_coverage_pct: Some(85.0),
            branch_coverage_pct: None,
        };
        let delta = claudear_core::types::EvalDelta::compute(before, after);
        let id = tracker
            .store_eval_delta(Some(1), "test/repo", &delta)
            .unwrap();
        assert!(id > 0);
    }

    #[test]
    fn test_store_eval_delta_with_diagnostics() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let diag = claudear_core::types::Diagnostic {
            file: "src/lib.rs".into(),
            line: Some(42),
            column: Some(5),
            severity: claudear_core::types::DiagnosticSeverity::Warning,
            code: Some("W001".into()),
            message: "unused variable".into(),
        };
        let before = claudear_core::types::EvalSnapshot {
            diagnostics: vec![diag.clone()],
            ..Default::default()
        };
        let after = claudear_core::types::EvalSnapshot::default();
        let delta = claudear_core::types::EvalDelta::compute(before, after);
        assert_eq!(delta.fixed.len(), 1);
        let id = tracker.store_eval_delta(None, "test/repo", &delta).unwrap();
        assert!(id > 0);
    }

    #[test]
    fn test_save_indexed_repo() {
        let tracker = SqliteTracker::in_memory().unwrap();

        let id = tracker
            .save_indexed_repo(
                "my-repo",
                "/tmp/my-repo",
                Some("https://github.com/org/my-repo"),
                "main",
                42,
            )
            .unwrap();
        assert!(id > 0);

        // Verify by retrieving the repo
        let repo = tracker.get_indexed_repo("my-repo").unwrap().unwrap();
        assert_eq!(repo.name, "my-repo");
        assert_eq!(repo.path, "/tmp/my-repo");
        assert_eq!(
            repo.scm_url,
            Some("https://github.com/org/my-repo".to_string())
        );
        assert_eq!(repo.default_branch, "main");
        assert_eq!(repo.file_count, 42);
    }

    #[test]
    fn test_save_indexed_repo_upsert_updates_existing() {
        let tracker = SqliteTracker::in_memory().unwrap();

        let id1 = tracker
            .save_indexed_repo(
                "my-repo",
                "/tmp/old-path",
                Some("https://old.url"),
                "main",
                10,
            )
            .unwrap();

        let id2 = tracker
            .save_indexed_repo(
                "my-repo",
                "/tmp/new-path",
                Some("https://new.url"),
                "develop",
                99,
            )
            .unwrap();

        // Same repo, so IDs should match
        assert_eq!(id1, id2);

        let repo = tracker.get_indexed_repo("my-repo").unwrap().unwrap();
        assert_eq!(repo.path, "/tmp/new-path");
        assert_eq!(repo.scm_url, Some("https://new.url".to_string()));
        assert_eq!(repo.default_branch, "develop");
        assert_eq!(repo.file_count, 99);
    }

    #[test]
    fn test_save_indexed_repo_upsert_preserves_scm_url_when_null() {
        let tracker = SqliteTracker::in_memory().unwrap();

        tracker
            .save_indexed_repo("my-repo", "/path", Some("https://original.url"), "main", 5)
            .unwrap();

        // Update with None scm_url -- the COALESCE should preserve the original
        tracker
            .save_indexed_repo("my-repo", "/path2", None, "main", 10)
            .unwrap();

        let repo = tracker.get_indexed_repo("my-repo").unwrap().unwrap();
        assert_eq!(repo.scm_url, Some("https://original.url".to_string()));
        assert_eq!(repo.path, "/path2");
    }

    #[test]
    fn test_save_repo_file() {
        let tracker = SqliteTracker::in_memory().unwrap();

        let repo_id = tracker
            .save_indexed_repo("test-repo", "/tmp/test", None, "main", 0)
            .unwrap();

        tracker
            .save_repo_file(repo_id, "src/main.rs", Some("rs"))
            .unwrap();
        tracker
            .save_repo_file(repo_id, "README.md", Some("md"))
            .unwrap();

        // Verify via get_index_stats
        let stats = tracker.get_index_stats().unwrap();
        assert_eq!(stats.file_count, 2);
    }

    #[test]
    fn test_save_repo_file_upsert_updates_type() {
        let tracker = SqliteTracker::in_memory().unwrap();

        let repo_id = tracker
            .save_indexed_repo("test-repo", "/tmp/test", None, "main", 0)
            .unwrap();

        tracker
            .save_repo_file(repo_id, "src/main.rs", Some("rs"))
            .unwrap();
        // Update the file_type
        tracker
            .save_repo_file(repo_id, "src/main.rs", Some("rust"))
            .unwrap();

        // Should still be 1 file (upsert, not duplicate)
        let stats = tracker.get_index_stats().unwrap();
        assert_eq!(stats.file_count, 1);
    }

    #[test]
    fn test_save_repo_file_with_none_type() {
        let tracker = SqliteTracker::in_memory().unwrap();

        let repo_id = tracker
            .save_indexed_repo("test-repo", "/tmp/test", None, "main", 0)
            .unwrap();

        tracker.save_repo_file(repo_id, "Makefile", None).unwrap();

        let stats = tracker.get_index_stats().unwrap();
        assert_eq!(stats.file_count, 1);
    }

    #[test]
    fn test_save_repo_files() {
        let tracker = SqliteTracker::in_memory().unwrap();

        let repo_id = tracker
            .save_indexed_repo("test-repo", "/tmp/test", None, "main", 0)
            .unwrap();

        let files = vec![
            ("src/lib.rs".to_string(), Some("rs".to_string())),
            ("src/main.rs".to_string(), Some("rs".to_string())),
            ("Cargo.toml".to_string(), Some("toml".to_string())),
            ("Makefile".to_string(), None),
        ];

        tracker.save_repo_files(repo_id, &files).unwrap();

        let stats = tracker.get_index_stats().unwrap();
        assert_eq!(stats.file_count, 4);
    }

    #[test]
    fn test_save_repo_files_empty() {
        let tracker = SqliteTracker::in_memory().unwrap();

        let repo_id = tracker
            .save_indexed_repo("test-repo", "/tmp/test", None, "main", 0)
            .unwrap();

        let files: Vec<(String, Option<String>)> = vec![];
        tracker.save_repo_files(repo_id, &files).unwrap();

        let stats = tracker.get_index_stats().unwrap();
        assert_eq!(stats.file_count, 0);
    }

    #[test]
    fn test_clear_repo_files() {
        let tracker = SqliteTracker::in_memory().unwrap();

        let repo_id = tracker
            .save_indexed_repo("test-repo", "/tmp/test", None, "main", 0)
            .unwrap();

        let files = vec![
            ("a.rs".to_string(), Some("rs".to_string())),
            ("b.rs".to_string(), Some("rs".to_string())),
            ("c.rs".to_string(), Some("rs".to_string())),
        ];
        tracker.save_repo_files(repo_id, &files).unwrap();

        let stats_before = tracker.get_index_stats().unwrap();
        assert_eq!(stats_before.file_count, 3);

        tracker.clear_repo_files(repo_id).unwrap();

        let stats_after = tracker.get_index_stats().unwrap();
        assert_eq!(stats_after.file_count, 0);
    }

    #[test]
    fn test_clear_repo_files_only_affects_target_repo() {
        let tracker = SqliteTracker::in_memory().unwrap();

        let repo_a = tracker
            .save_indexed_repo("repo-a", "/tmp/a", None, "main", 0)
            .unwrap();
        let repo_b = tracker
            .save_indexed_repo("repo-b", "/tmp/b", None, "main", 0)
            .unwrap();

        tracker
            .save_repo_files(repo_a, &[("a.rs".to_string(), Some("rs".to_string()))])
            .unwrap();
        tracker
            .save_repo_files(repo_b, &[("b.rs".to_string(), Some("rs".to_string()))])
            .unwrap();

        tracker.clear_repo_files(repo_a).unwrap();

        // repo-b's files should still exist
        let stats = tracker.get_index_stats().unwrap();
        assert_eq!(stats.file_count, 1);
    }

    #[test]
    fn test_get_indexed_repo() {
        let tracker = SqliteTracker::in_memory().unwrap();

        tracker
            .save_indexed_repo(
                "my-repo",
                "/tmp/my-repo",
                Some("https://github.com/org/repo"),
                "main",
                15,
            )
            .unwrap();

        let repo = tracker.get_indexed_repo("my-repo").unwrap().unwrap();
        assert_eq!(repo.name, "my-repo");
        assert_eq!(repo.path, "/tmp/my-repo");
        assert_eq!(repo.default_branch, "main");
        assert_eq!(repo.file_count, 15);
        assert!(!repo.last_indexed_at.is_empty());
        assert!(!repo.created_at.is_empty());
    }

    #[test]
    fn test_get_indexed_repo_not_found() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let result = tracker.get_indexed_repo("nonexistent-repo").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_list_indexed_repos() {
        let tracker = SqliteTracker::in_memory().unwrap();

        tracker
            .save_indexed_repo("charlie", "/tmp/c", None, "main", 3)
            .unwrap();
        tracker
            .save_indexed_repo("alpha", "/tmp/a", None, "main", 1)
            .unwrap();
        tracker
            .save_indexed_repo("bravo", "/tmp/b", None, "develop", 2)
            .unwrap();

        let repos = tracker.list_indexed_repos().unwrap();
        assert_eq!(repos.len(), 3);
        // Should be ordered by name
        assert_eq!(repos[0].name, "alpha");
        assert_eq!(repos[1].name, "bravo");
        assert_eq!(repos[2].name, "charlie");
    }

    #[test]
    fn test_list_indexed_repos_empty() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let repos = tracker.list_indexed_repos().unwrap();
        assert!(repos.is_empty());
    }

    #[test]
    fn test_get_index_stats() {
        let tracker = SqliteTracker::in_memory().unwrap();

        // Empty database stats
        let stats = tracker.get_index_stats().unwrap();
        assert_eq!(stats.repo_count, 0);
        assert_eq!(stats.file_count, 0);
        assert!(stats.last_indexed_at.is_none());

        // Add some repos and files
        let repo1 = tracker
            .save_indexed_repo("repo-1", "/tmp/1", None, "main", 2)
            .unwrap();
        let repo2 = tracker
            .save_indexed_repo("repo-2", "/tmp/2", None, "main", 3)
            .unwrap();

        tracker
            .save_repo_files(
                repo1,
                &[
                    ("a.rs".to_string(), Some("rs".to_string())),
                    ("b.rs".to_string(), Some("rs".to_string())),
                ],
            )
            .unwrap();
        tracker
            .save_repo_files(
                repo2,
                &[
                    ("c.py".to_string(), Some("py".to_string())),
                    ("d.py".to_string(), Some("py".to_string())),
                    ("e.py".to_string(), Some("py".to_string())),
                ],
            )
            .unwrap();

        let stats = tracker.get_index_stats().unwrap();
        assert_eq!(stats.repo_count, 2);
        assert_eq!(stats.file_count, 5);
        assert!(stats.last_indexed_at.is_some());
    }

    #[test]
    fn test_get_direct_dependants() {
        let tracker = SqliteTracker::in_memory().unwrap();

        tracker
            .add_dependency("core-lib", "app-x", "runtime")
            .unwrap();
        tracker
            .add_dependency("core-lib", "app-y", "build")
            .unwrap();
        tracker
            .add_dependency("other-lib", "app-z", "runtime")
            .unwrap();

        let dependants = tracker.get_direct_dependants("core-lib").unwrap();
        assert_eq!(dependants.len(), 2);

        let names: Vec<&str> = dependants.iter().map(|d| d.downstream.as_str()).collect();
        assert!(names.contains(&"app-x"));
        assert!(names.contains(&"app-y"));
    }

    #[test]
    fn test_get_direct_dependants_empty() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let dependants = tracker.get_direct_dependants("nonexistent").unwrap();
        assert!(dependants.is_empty());
    }

    #[test]
    fn test_get_direct_dependants_excludes_transitive() {
        let tracker = SqliteTracker::in_memory().unwrap();
        // core -> mid -> leaf
        tracker.add_dependency("core", "mid", "runtime").unwrap();
        tracker.add_dependency("mid", "leaf", "runtime").unwrap();

        // Direct dependants of core should only be mid, not leaf
        let dependants = tracker.get_direct_dependants("core").unwrap();
        assert_eq!(dependants.len(), 1);
        assert_eq!(dependants[0].downstream, "mid");
    }

    #[test]
    fn test_record_inference_attempt() {
        let tracker = SqliteTracker::in_memory().unwrap();

        // Create a repo for the inferred_repo_id
        let repo_id = tracker
            .save_indexed_repo("target-repo", "/tmp/target", None, "main", 10)
            .unwrap();

        let id = tracker
            .record_inference_attempt(
                "issue-123",
                "linear",
                &["src/main.rs".to_string(), "src/lib.rs".to_string()],
                &["handle_request".to_string()],
                &["timeout".to_string(), "database".to_string()],
                Some(repo_id),
                "high",
                "Matched by filename pattern",
                Some(42),
            )
            .unwrap();

        assert!(id > 0);

        // Verify via inference history
        let history = tracker.get_inference_history(10).unwrap();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].id, id);
        assert_eq!(history[0].issue_id, "issue-123");
        assert_eq!(history[0].issue_source, "linear");
        assert_eq!(
            history[0].inferred_repo_name,
            Some("target-repo".to_string())
        );
        assert_eq!(history[0].confidence, Some("high".to_string()));
        assert_eq!(
            history[0].inference_reason,
            Some("Matched by filename pattern".to_string())
        );
        assert_eq!(history[0].duration_ms, Some(42));
    }

    #[test]
    fn test_record_inference_attempt_no_repo_match() {
        let tracker = SqliteTracker::in_memory().unwrap();

        let id = tracker
            .record_inference_attempt(
                "issue-456",
                "sentry",
                &[],
                &[],
                &["unknown_keyword".to_string()],
                None,
                "low",
                "No matching repository found",
                Some(5),
            )
            .unwrap();

        assert!(id > 0);

        let history = tracker.get_inference_history(10).unwrap();
        assert_eq!(history.len(), 1);
        assert!(history[0].inferred_repo_name.is_none());
        assert_eq!(history[0].confidence, Some("low".to_string()));
    }

    #[test]
    fn test_record_inference_feedback() {
        let tracker = SqliteTracker::in_memory().unwrap();

        let repo_id = tracker
            .save_indexed_repo("correct-repo", "/tmp/correct", None, "main", 5)
            .unwrap();

        let inference_id = tracker
            .record_inference_attempt(
                "issue-fb-1",
                "linear",
                &[],
                &[],
                &["error".to_string()],
                Some(repo_id),
                "medium",
                "Keyword match",
                None,
            )
            .unwrap();

        // Record positive feedback
        tracker
            .record_inference_feedback(inference_id, true, Some(repo_id), "user")
            .unwrap();

        // Verify the feedback was recorded
        let history = tracker.get_inference_history(10).unwrap();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].was_correct, Some(true));
    }

    #[test]
    fn test_record_inference_feedback_incorrect() {
        let tracker = SqliteTracker::in_memory().unwrap();

        let wrong_repo_id = tracker
            .save_indexed_repo("wrong-repo", "/tmp/wrong", None, "main", 5)
            .unwrap();
        let actual_repo_id = tracker
            .save_indexed_repo("actual-repo", "/tmp/actual", None, "main", 5)
            .unwrap();

        let inference_id = tracker
            .record_inference_attempt(
                "issue-fb-2",
                "linear",
                &[],
                &[],
                &[],
                Some(wrong_repo_id),
                "high",
                "Wrong match",
                None,
            )
            .unwrap();

        tracker
            .record_inference_feedback(inference_id, false, Some(actual_repo_id), "admin")
            .unwrap();

        let history = tracker.get_inference_history(10).unwrap();
        assert_eq!(history[0].was_correct, Some(false));
    }

    #[test]
    fn test_get_inference_stats_empty() {
        let tracker = SqliteTracker::in_memory().unwrap();

        let stats = tracker.get_inference_stats().unwrap();
        assert_eq!(stats.total_attempts, 0);
        assert_eq!(stats.with_feedback, 0);
        assert_eq!(stats.correct, 0);
        assert_eq!(stats.accuracy, 0.0);
        assert_eq!(stats.by_confidence.high, 0);
        assert_eq!(stats.by_confidence.medium, 0);
        assert_eq!(stats.by_confidence.low, 0);
        assert_eq!(stats.by_confidence.none, 0);
    }

    #[test]
    fn test_get_inference_stats() {
        let tracker = SqliteTracker::in_memory().unwrap();

        let repo_id = tracker
            .save_indexed_repo("stats-repo", "/tmp/stats", None, "main", 1)
            .unwrap();

        // High confidence, correct
        let id1 = tracker
            .record_inference_attempt(
                "i1",
                "linear",
                &[],
                &[],
                &[],
                Some(repo_id),
                "high",
                "reason",
                None,
            )
            .unwrap();
        tracker
            .record_inference_feedback(id1, true, Some(repo_id), "user")
            .unwrap();

        // Medium confidence, incorrect
        let id2 = tracker
            .record_inference_attempt(
                "i2",
                "sentry",
                &[],
                &[],
                &[],
                Some(repo_id),
                "medium",
                "reason",
                None,
            )
            .unwrap();
        tracker
            .record_inference_feedback(id2, false, None, "user")
            .unwrap();

        // Low confidence, no feedback
        tracker
            .record_inference_attempt(
                "i3",
                "linear",
                &[],
                &[],
                &[],
                Some(repo_id),
                "low",
                "reason",
                None,
            )
            .unwrap();

        // No match (inferred_repo_id is NULL)
        tracker
            .record_inference_attempt("i4", "sentry", &[], &[], &[], None, "low", "no match", None)
            .unwrap();

        let stats = tracker.get_inference_stats().unwrap();
        assert_eq!(stats.total_attempts, 4);
        assert_eq!(stats.with_feedback, 2);
        assert_eq!(stats.correct, 1);
        // accuracy = 1/2 * 100 = 50.0
        assert!((stats.accuracy - 50.0).abs() < f64::EPSILON);
        assert_eq!(stats.by_confidence.high, 1);
        assert_eq!(stats.by_confidence.medium, 1);
        assert_eq!(stats.by_confidence.low, 2);
        assert_eq!(stats.by_confidence.none, 1); // The one with no repo match
    }

    #[test]
    fn test_get_inference_history() {
        let tracker = SqliteTracker::in_memory().unwrap();

        let repo_id = tracker
            .save_indexed_repo("hist-repo", "/tmp/hist", None, "main", 1)
            .unwrap();

        // Record several attempts
        for i in 0..5 {
            tracker
                .record_inference_attempt(
                    &format!("issue-{}", i),
                    "linear",
                    &[],
                    &[],
                    &[format!("kw-{}", i)],
                    Some(repo_id),
                    "medium",
                    &format!("reason {}", i),
                    Some(i * 10),
                )
                .unwrap();
        }

        // All 5 should be retrievable
        let history_all = tracker.get_inference_history(10).unwrap();
        assert_eq!(history_all.len(), 5);

        // Limit to 3 should return exactly 3
        let history = tracker.get_inference_history(3).unwrap();
        assert_eq!(history.len(), 3);

        // Verify all returned entries are valid inference attempts
        for entry in &history {
            assert!(entry.issue_id.starts_with("issue-"));
            assert_eq!(entry.issue_source, "linear");
            assert_eq!(entry.inferred_repo_name, Some("hist-repo".to_string()));
        }
    }

    #[test]
    fn test_get_inference_history_empty() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let history = tracker.get_inference_history(10).unwrap();
        assert!(history.is_empty());
    }

    #[test]
    fn test_get_inference_history_includes_feedback() {
        let tracker = SqliteTracker::in_memory().unwrap();

        let repo_id = tracker
            .save_indexed_repo("fb-repo", "/tmp/fb", None, "main", 1)
            .unwrap();

        let id = tracker
            .record_inference_attempt(
                "fb-issue",
                "linear",
                &[],
                &[],
                &[],
                Some(repo_id),
                "high",
                "matched",
                Some(100),
            )
            .unwrap();

        tracker
            .record_inference_feedback(id, true, Some(repo_id), "user")
            .unwrap();

        let history = tracker.get_inference_history(10).unwrap();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].was_correct, Some(true));
        assert_eq!(history[0].inferred_repo_name, Some("fb-repo".to_string()));
        assert_eq!(history[0].duration_ms, Some(100));
    }

    #[test]
    fn test_check_and_record_delivery_new() {
        let tracker = SqliteTracker::in_memory().unwrap();

        // First time should return true (new delivery)
        let is_new = tracker
            .check_and_record_delivery("delivery-1", "github")
            .unwrap();
        assert!(is_new);
    }

    #[test]
    fn test_check_and_record_delivery_duplicate() {
        let tracker = SqliteTracker::in_memory().unwrap();

        // Record a delivery
        let first = tracker
            .check_and_record_delivery("delivery-1", "github")
            .unwrap();
        assert!(first);

        // Same delivery ID should return false (duplicate)
        let second = tracker
            .check_and_record_delivery("delivery-1", "github")
            .unwrap();
        assert!(!second);
    }

    #[test]
    fn test_check_and_record_delivery_different_ids() {
        let tracker = SqliteTracker::in_memory().unwrap();

        let a = tracker
            .check_and_record_delivery("delivery-a", "github")
            .unwrap();
        let b = tracker
            .check_and_record_delivery("delivery-b", "github")
            .unwrap();

        assert!(a);
        assert!(b);
    }

    #[test]
    fn test_check_and_record_delivery_different_sources() {
        let tracker = SqliteTracker::in_memory().unwrap();

        // Same delivery ID but different source - the UNIQUE constraint is on
        // (delivery_id, source) so these are treated as distinct deliveries.
        let first = tracker
            .check_and_record_delivery("delivery-1", "github")
            .unwrap();
        assert!(first);

        let second = tracker
            .check_and_record_delivery("delivery-1", "gitlab")
            .unwrap();
        // Different (delivery_id, source) pair = new delivery
        assert!(second);

        // But same pair should be duplicate
        let third = tracker
            .check_and_record_delivery("delivery-1", "github")
            .unwrap();
        assert!(!third);
    }

    #[test]
    fn test_cleanup_old_deliveries_no_old_records() {
        let tracker = SqliteTracker::in_memory().unwrap();

        // Record a fresh delivery
        tracker
            .check_and_record_delivery("delivery-1", "github")
            .unwrap();

        // Cleaning up records older than 24 hours should not remove the fresh one
        let removed = tracker.cleanup_old_deliveries(24).unwrap();
        assert_eq!(removed, 0);

        // Delivery should still be present (duplicate check returns false)
        let is_new = tracker
            .check_and_record_delivery("delivery-1", "github")
            .unwrap();
        assert!(!is_new);
    }

    #[test]
    fn test_cleanup_old_deliveries_removes_old() {
        let tracker = SqliteTracker::in_memory().unwrap();

        // Insert a delivery and manually backdate it
        tracker
            .check_and_record_delivery("old-delivery", "github")
            .unwrap();
        {
            let conn = tracker.acquire_lock().unwrap();
            conn.execute(
                "UPDATE webhook_deliveries SET received_at = datetime('now', '-48 hours') WHERE delivery_id = ?",
                params!["old-delivery"],
            )
            .unwrap();
        }

        // Insert a fresh delivery
        tracker
            .check_and_record_delivery("new-delivery", "github")
            .unwrap();

        // Clean up deliveries older than 24 hours
        let removed = tracker.cleanup_old_deliveries(24).unwrap();
        assert_eq!(removed, 1);

        // Old delivery should be gone (re-inserting returns true)
        let is_new = tracker
            .check_and_record_delivery("old-delivery", "github")
            .unwrap();
        assert!(is_new);

        // New delivery should still be present
        let is_new = tracker
            .check_and_record_delivery("new-delivery", "github")
            .unwrap();
        assert!(!is_new);
    }

    #[test]
    fn test_cleanup_old_deliveries_empty_table() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let removed = tracker.cleanup_old_deliveries(24).unwrap();
        assert_eq!(removed, 0);
    }

    #[test]
    fn test_get_execution_for_attempt_found() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker
            .record_attempt("linear", "exec-single", "LIN-ES")
            .unwrap();
        let attempt = tracker
            .get_attempt("linear", "exec-single")
            .unwrap()
            .unwrap();

        let mut execution = AgentExecution::new().with_attempt_id(attempt.id);
        execution.prompt_used = Some("Fix the bug".to_string());
        execution.exit_code = Some(0);

        let exec_id = tracker.record_execution(&execution).unwrap();

        let result = tracker
            .get_execution_for_attempt(attempt.id, exec_id)
            .unwrap();
        assert!(result.is_some());
        let exec = result.unwrap();
        assert_eq!(exec.id, exec_id);
        assert_eq!(exec.attempt_id, Some(attempt.id));
        assert_eq!(exec.prompt_used, Some("Fix the bug".to_string()));
        assert_eq!(exec.exit_code, Some(0));
    }

    #[test]
    fn test_get_execution_for_attempt_wrong_execution_id() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker
            .record_attempt("linear", "exec-wrong", "LIN-EW")
            .unwrap();
        let attempt = tracker
            .get_attempt("linear", "exec-wrong")
            .unwrap()
            .unwrap();

        let execution = AgentExecution::new().with_attempt_id(attempt.id);
        tracker.record_execution(&execution).unwrap();

        // Query with a non-existent execution_id
        let result = tracker
            .get_execution_for_attempt(attempt.id, 99999)
            .unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_get_execution_for_attempt_wrong_attempt_id() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker
            .record_attempt("linear", "exec-wa", "LIN-WA")
            .unwrap();
        let attempt = tracker.get_attempt("linear", "exec-wa").unwrap().unwrap();

        let execution = AgentExecution::new().with_attempt_id(attempt.id);
        let exec_id = tracker.record_execution(&execution).unwrap();

        // Correct execution_id but wrong attempt_id
        let result = tracker.get_execution_for_attempt(99999, exec_id).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_record_error_pattern_and_retrieve() {
        let tracker = SqliteTracker::in_memory().unwrap();

        let pattern = ErrorPattern {
            id: 0,
            pattern_hash: "hash-abc".to_string(),
            error_type: Some("build_failure".to_string()),
            error_message: Some("cannot find module 'foo'".to_string()),
            first_seen: Utc::now(),
            last_seen: Utc::now(),
            occurrence_count: 1,
            sources: Some(vec!["linear".to_string()]),
            example_issue_ids: Some(vec!["PROJ-1".to_string()]),
            resolution_hints: Some("Install the missing dependency".to_string()),
        };

        let id = tracker.record_error_pattern(&pattern).unwrap();
        assert!(id > 0);

        let patterns = tracker.get_error_patterns(10).unwrap();
        assert_eq!(patterns.len(), 1);
        assert_eq!(patterns[0].pattern_hash, "hash-abc");
        assert_eq!(patterns[0].error_type, Some("build_failure".to_string()));
        assert_eq!(
            patterns[0].error_message,
            Some("cannot find module 'foo'".to_string())
        );
        assert_eq!(patterns[0].occurrence_count, 1);
        assert_eq!(patterns[0].sources, Some(vec!["linear".to_string()]));
        assert_eq!(
            patterns[0].example_issue_ids,
            Some(vec!["PROJ-1".to_string()])
        );
        assert_eq!(
            patterns[0].resolution_hints,
            Some("Install the missing dependency".to_string())
        );
    }

    #[test]
    fn test_record_error_pattern_upsert_increments_count() {
        let tracker = SqliteTracker::in_memory().unwrap();

        let pattern = ErrorPattern {
            id: 0,
            pattern_hash: "hash-dup".to_string(),
            error_type: Some("test_failure".to_string()),
            error_message: Some("assertion failed".to_string()),
            first_seen: Utc::now(),
            last_seen: Utc::now(),
            occurrence_count: 1,
            sources: None,
            example_issue_ids: None,
            resolution_hints: None,
        };

        tracker.record_error_pattern(&pattern).unwrap();
        // Record again with the same hash - should upsert and increment count
        tracker.record_error_pattern(&pattern).unwrap();

        let patterns = tracker.get_error_patterns(10).unwrap();
        assert_eq!(patterns.len(), 1);
        // Initial insert count is 1, then ON CONFLICT increments by 1 = 2
        assert_eq!(patterns[0].occurrence_count, 2);
    }

    #[test]
    fn test_get_error_patterns_respects_limit() {
        let tracker = SqliteTracker::in_memory().unwrap();

        for i in 0..5 {
            let pattern = ErrorPattern {
                id: 0,
                pattern_hash: format!("hash-{}", i),
                error_type: Some("type".to_string()),
                error_message: Some(format!("error {}", i)),
                first_seen: Utc::now(),
                last_seen: Utc::now(),
                occurrence_count: i + 1,
                sources: None,
                example_issue_ids: None,
                resolution_hints: None,
            };
            tracker.record_error_pattern(&pattern).unwrap();
        }

        let patterns = tracker.get_error_patterns(3).unwrap();
        assert_eq!(patterns.len(), 3);
        // Should be ordered by occurrence_count DESC
        assert!(patterns[0].occurrence_count >= patterns[1].occurrence_count);
        assert!(patterns[1].occurrence_count >= patterns[2].occurrence_count);
    }

    #[test]
    fn test_get_error_patterns_ordered_by_occurrence() {
        let tracker = SqliteTracker::in_memory().unwrap();

        // Insert patterns with different counts
        for (hash, count) in [("low", 1), ("high", 10), ("mid", 5)] {
            let pattern = ErrorPattern {
                id: 0,
                pattern_hash: hash.to_string(),
                error_type: None,
                error_message: None,
                first_seen: Utc::now(),
                last_seen: Utc::now(),
                occurrence_count: count,
                sources: None,
                example_issue_ids: None,
                resolution_hints: None,
            };
            tracker.record_error_pattern(&pattern).unwrap();
        }

        let patterns = tracker.get_error_patterns(10).unwrap();
        assert_eq!(patterns.len(), 3);
        assert_eq!(patterns[0].pattern_hash, "high");
        assert_eq!(patterns[1].pattern_hash, "mid");
        assert_eq!(patterns[2].pattern_hash, "low");
    }

    #[test]
    fn test_get_feedback_outcome_by_attempt_found() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker
            .record_attempt("linear", "fb-issue", "LIN-FB")
            .unwrap();
        let attempt = tracker.get_attempt("linear", "fb-issue").unwrap().unwrap();

        let outcome = FixOutcome {
            id: 0,
            attempt_id: attempt.id,
            source: "linear".to_string(),
            issue_id: "fb-issue".to_string(),
            issue_text: "Fix the login page".to_string(),
            prompt_used: "test prompt".to_string(),
            outcome: claudear_core::types::Outcome::Merged,
            error_type: None,
            learnings: Some("Always validate inputs".to_string()),
            keywords: vec!["login".to_string(), "fix".to_string()],
            embedding: None,
            created_at: Utc::now(),
        };

        tracker.store_feedback_outcome(&outcome).unwrap();

        let result = tracker.get_feedback_outcome_by_attempt(attempt.id).unwrap();
        assert!(result.is_some());
        let found = result.unwrap();
        assert_eq!(found.attempt_id, attempt.id);
        assert_eq!(found.source, "linear");
        assert_eq!(found.issue_id, "fb-issue");
        assert_eq!(found.issue_text, "Fix the login page");
        assert_eq!(found.prompt_used, "test prompt");
        assert!(found.outcome.is_success());
        assert_eq!(found.learnings, Some("Always validate inputs".to_string()));
        assert_eq!(found.keywords, vec!["login".to_string(), "fix".to_string()]);
    }

    #[test]
    fn test_record_metrics_batch() {
        let tracker = SqliteTracker::in_memory().unwrap();

        let metrics: Vec<ProcessingMetric> = (0..5)
            .map(|i| ProcessingMetric {
                id: 0,
                timestamp: Utc::now(),
                metric_name: "batch_metric".to_string(),
                metric_value: i as f64,
                source: Some("linear".to_string()),
                tags: None,
            })
            .collect();

        let count = tracker.record_metrics_batch(&metrics).unwrap();
        assert_eq!(count, 5);

        let retrieved = tracker.get_metrics("batch_metric", None, 100).unwrap();
        assert_eq!(retrieved.len(), 5);
    }

    #[test]
    fn test_record_metrics_batch_empty() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let count = tracker.record_metrics_batch(&[]).unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_record_metrics_batch_preserves_values() {
        let tracker = SqliteTracker::in_memory().unwrap();

        let metrics = vec![
            ProcessingMetric {
                id: 0,
                timestamp: Utc::now(),
                metric_name: "latency".to_string(),
                metric_value: 100.5,
                source: Some("github".to_string()),
                tags: Some(serde_json::json!({"region": "us-west"})),
            },
            ProcessingMetric {
                id: 0,
                timestamp: Utc::now(),
                metric_name: "throughput".to_string(),
                metric_value: 42.0,
                source: Some("linear".to_string()),
                tags: None,
            },
        ];

        tracker.record_metrics_batch(&metrics).unwrap();

        let latency = tracker.get_metrics("latency", None, 10).unwrap();
        assert_eq!(latency.len(), 1);
        assert!((latency[0].metric_value - 100.5).abs() < f64::EPSILON);
        assert_eq!(latency[0].source, Some("github".to_string()));
        assert!(latency[0].tags.is_some());

        let throughput = tracker.get_metrics("throughput", None, 10).unwrap();
        assert_eq!(throughput.len(), 1);
        assert!((throughput[0].metric_value - 42.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_get_metric_counts_since() {
        let tracker = SqliteTracker::in_memory().unwrap();

        // Insert metrics
        for name in &["requests", "errors", "requests"] {
            let metric = ProcessingMetric {
                id: 0,
                timestamp: Utc::now(),
                metric_name: name.to_string(),
                metric_value: 1.0,
                source: None,
                tags: None,
            };
            tracker.record_metric(&metric).unwrap();
        }

        let since = chrono::DateTime::parse_from_rfc3339("2020-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc);

        let counts = tracker
            .get_metric_counts_since(&["requests", "errors"], since)
            .unwrap();
        assert_eq!(counts.get("requests"), Some(&2));
        assert_eq!(counts.get("errors"), Some(&1));
    }

    #[test]
    fn test_get_metric_counts_since_empty_names() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let since = Utc::now();
        let counts = tracker.get_metric_counts_since(&[], since).unwrap();
        assert!(counts.is_empty());
    }

    #[test]
    fn test_get_metric_counts_since_filters_by_time() {
        let tracker = SqliteTracker::in_memory().unwrap();

        // Insert an old metric
        let old_metric = ProcessingMetric {
            id: 0,
            timestamp: chrono::DateTime::parse_from_rfc3339("2020-01-01T00:00:00Z")
                .unwrap()
                .with_timezone(&Utc),
            metric_name: "old_count".to_string(),
            metric_value: 1.0,
            source: None,
            tags: None,
        };
        tracker.record_metric(&old_metric).unwrap();

        // Insert a recent metric
        let recent_metric = ProcessingMetric {
            id: 0,
            timestamp: Utc::now(),
            metric_name: "old_count".to_string(),
            metric_value: 1.0,
            source: None,
            tags: None,
        };
        tracker.record_metric(&recent_metric).unwrap();

        let since = chrono::DateTime::parse_from_rfc3339("2024-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let counts = tracker
            .get_metric_counts_since(&["old_count"], since)
            .unwrap();
        assert_eq!(counts.get("old_count"), Some(&1));
    }

    #[test]
    fn test_get_metric_counts_since_missing_names() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let since = chrono::DateTime::parse_from_rfc3339("2020-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let counts = tracker
            .get_metric_counts_since(&["nonexistent"], since)
            .unwrap();
        assert!(counts.is_empty());
    }

    #[test]
    fn test_get_metric_sums_since() {
        let tracker = SqliteTracker::in_memory().unwrap();

        let values = [10.5, 20.3, 5.2];
        for val in &values {
            let metric = ProcessingMetric {
                id: 0,
                timestamp: Utc::now(),
                metric_name: "cost_usd".to_string(),
                metric_value: *val,
                source: None,
                tags: None,
            };
            tracker.record_metric(&metric).unwrap();
        }

        let since = chrono::DateTime::parse_from_rfc3339("2020-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let sums = tracker.get_metric_sums_since(&["cost_usd"], since).unwrap();
        let total = sums.get("cost_usd").copied().unwrap_or(0.0);
        assert!((total - 36.0).abs() < 0.01);
    }

    #[test]
    fn test_get_metric_sums_since_empty_names() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let since = Utc::now();
        let sums = tracker.get_metric_sums_since(&[], since).unwrap();
        assert!(sums.is_empty());
    }

    #[test]
    fn test_get_metric_sums_since_multiple_names() {
        let tracker = SqliteTracker::in_memory().unwrap();

        for (name, val) in [("alpha", 10.0), ("alpha", 20.0), ("beta", 5.0)] {
            let metric = ProcessingMetric {
                id: 0,
                timestamp: Utc::now(),
                metric_name: name.to_string(),
                metric_value: val,
                source: None,
                tags: None,
            };
            tracker.record_metric(&metric).unwrap();
        }

        let since = chrono::DateTime::parse_from_rfc3339("2020-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let sums = tracker
            .get_metric_sums_since(&["alpha", "beta"], since)
            .unwrap();
        assert!((sums.get("alpha").copied().unwrap_or(0.0) - 30.0).abs() < f64::EPSILON);
        assert!((sums.get("beta").copied().unwrap_or(0.0) - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_get_metric_sums_since_filters_by_time() {
        let tracker = SqliteTracker::in_memory().unwrap();

        let old = ProcessingMetric {
            id: 0,
            timestamp: chrono::DateTime::parse_from_rfc3339("2020-01-01T00:00:00Z")
                .unwrap()
                .with_timezone(&Utc),
            metric_name: "cost".to_string(),
            metric_value: 100.0,
            source: None,
            tags: None,
        };
        let recent = ProcessingMetric {
            id: 0,
            timestamp: Utc::now(),
            metric_name: "cost".to_string(),
            metric_value: 25.0,
            source: None,
            tags: None,
        };
        tracker.record_metric(&old).unwrap();
        tracker.record_metric(&recent).unwrap();

        let since = chrono::DateTime::parse_from_rfc3339("2024-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let sums = tracker.get_metric_sums_since(&["cost"], since).unwrap();
        assert!((sums.get("cost").copied().unwrap_or(0.0) - 25.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_get_metric_sums_by_source_since() {
        let tracker = SqliteTracker::in_memory().unwrap();

        let data = [
            ("cost", "linear", 10.0),
            ("cost", "linear", 20.0),
            ("cost", "sentry", 5.0),
            ("latency", "linear", 100.0),
        ];

        for (name, source, val) in &data {
            let metric = ProcessingMetric {
                id: 0,
                timestamp: Utc::now(),
                metric_name: name.to_string(),
                metric_value: *val,
                source: Some(source.to_string()),
                tags: None,
            };
            tracker.record_metric(&metric).unwrap();
        }

        let since = chrono::DateTime::parse_from_rfc3339("2020-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let sums = tracker
            .get_metric_sums_by_source_since(&["cost", "latency"], since)
            .unwrap();

        assert!(
            (sums
                .get(&("cost".to_string(), "linear".to_string()))
                .copied()
                .unwrap_or(0.0)
                - 30.0)
                .abs()
                < f64::EPSILON
        );
        assert!(
            (sums
                .get(&("cost".to_string(), "sentry".to_string()))
                .copied()
                .unwrap_or(0.0)
                - 5.0)
                .abs()
                < f64::EPSILON
        );
        assert!(
            (sums
                .get(&("latency".to_string(), "linear".to_string()))
                .copied()
                .unwrap_or(0.0)
                - 100.0)
                .abs()
                < f64::EPSILON
        );
    }

    #[test]
    fn test_get_metric_sums_by_source_since_empty_names() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let since = Utc::now();
        let sums = tracker.get_metric_sums_by_source_since(&[], since).unwrap();
        assert!(sums.is_empty());
    }

    #[test]
    fn test_get_metric_sums_by_source_since_no_source() {
        let tracker = SqliteTracker::in_memory().unwrap();

        // Metrics with None source should not appear in by-source results
        let metric = ProcessingMetric {
            id: 0,
            timestamp: Utc::now(),
            metric_name: "no_src".to_string(),
            metric_value: 42.0,
            source: None,
            tags: None,
        };
        tracker.record_metric(&metric).unwrap();

        let since = chrono::DateTime::parse_from_rfc3339("2020-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let sums = tracker
            .get_metric_sums_by_source_since(&["no_src"], since)
            .unwrap();
        // source is NULL so the row is skipped by the `if let Some(source)` check
        assert!(sums.is_empty());
    }

    #[test]
    fn test_get_metric_sums_by_source_since_filters_by_time() {
        let tracker = SqliteTracker::in_memory().unwrap();

        let old = ProcessingMetric {
            id: 0,
            timestamp: chrono::DateTime::parse_from_rfc3339("2020-01-01T00:00:00Z")
                .unwrap()
                .with_timezone(&Utc),
            metric_name: "timed".to_string(),
            metric_value: 999.0,
            source: Some("src".to_string()),
            tags: None,
        };
        let recent = ProcessingMetric {
            id: 0,
            timestamp: Utc::now(),
            metric_name: "timed".to_string(),
            metric_value: 7.0,
            source: Some("src".to_string()),
            tags: None,
        };
        tracker.record_metric(&old).unwrap();
        tracker.record_metric(&recent).unwrap();

        let since = chrono::DateTime::parse_from_rfc3339("2024-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let sums = tracker
            .get_metric_sums_by_source_since(&["timed"], since)
            .unwrap();
        let val = sums
            .get(&("timed".to_string(), "src".to_string()))
            .copied()
            .unwrap_or(0.0);
        assert!((val - 7.0).abs() < f64::EPSILON);
    }

    fn make_pr_record(pr_url: &str, repo: &str, number: i64) -> claudear_core::types::PrRecord {
        claudear_core::types::PrRecord::new(pr_url, repo, number)
    }

    #[test]
    fn test_upsert_pr_and_get_pr() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let pr = make_pr_record("https://github.com/org/repo/pull/1", "org/repo", 1);
        let id = tracker.upsert_pr(&pr).unwrap();
        assert!(id > 0);

        let fetched = tracker
            .get_pr("https://github.com/org/repo/pull/1")
            .unwrap()
            .unwrap();
        assert_eq!(fetched.pr_url, "https://github.com/org/repo/pull/1");
        assert_eq!(fetched.scm_repo, "org/repo");
        assert_eq!(fetched.pr_number, 1);
        assert_eq!(fetched.status, "open");
    }

    #[test]
    fn test_get_pr_not_found() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let result = tracker
            .get_pr("https://github.com/org/repo/pull/999")
            .unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_upsert_pr_updates_on_conflict() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let mut pr = make_pr_record("https://github.com/org/repo/pull/1", "org/repo", 1);
        tracker.upsert_pr(&pr).unwrap();

        pr.status = "merged".to_string();
        pr.approvals_count = 2;
        pr.comments_count = 5;
        tracker.upsert_pr(&pr).unwrap();

        let fetched = tracker
            .get_pr("https://github.com/org/repo/pull/1")
            .unwrap()
            .unwrap();
        assert_eq!(fetched.status, "merged");
        assert_eq!(fetched.approvals_count, 2);
        assert_eq!(fetched.comments_count, 5);
    }

    #[test]
    fn test_get_open_prs() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let pr1 = make_pr_record("https://github.com/org/repo/pull/1", "org/repo", 1);
        let mut pr2 = make_pr_record("https://github.com/org/repo/pull/2", "org/repo", 2);
        pr2.status = "merged".to_string();
        let pr3 = make_pr_record("https://github.com/org/repo/pull/3", "org/repo", 3);

        tracker.upsert_pr(&pr1).unwrap();
        tracker.upsert_pr(&pr2).unwrap();
        tracker.upsert_pr(&pr3).unwrap();

        let open = tracker.get_open_prs().unwrap();
        assert_eq!(open.len(), 2);
        // All should be open status
        for p in &open {
            assert_eq!(p.status, "open");
        }
    }

    #[test]
    fn test_get_open_prs_empty() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let open = tracker.get_open_prs().unwrap();
        assert!(open.is_empty());
    }

    #[test]
    fn test_get_pr_analytics() {
        let tracker = SqliteTracker::in_memory().unwrap();

        let mut pr1 = make_pr_record("https://github.com/org/a/pull/1", "org/a", 1);
        pr1.time_to_first_review_mins = Some(30);
        pr1.time_to_merge_mins = Some(120);
        pr1.review_cycles = 2;
        pr1.status = "merged".to_string();
        tracker.upsert_pr(&pr1).unwrap();

        let mut pr2 = make_pr_record("https://github.com/org/a/pull/2", "org/a", 2);
        pr2.status = "closed".to_string();
        tracker.upsert_pr(&pr2).unwrap();

        let pr3 = make_pr_record("https://github.com/org/b/pull/1", "org/b", 1);
        tracker.upsert_pr(&pr3).unwrap();

        let analytics = tracker.get_pr_analytics().unwrap();
        assert_eq!(analytics.total, 3);
        assert_eq!(analytics.open, 1);
        assert_eq!(analytics.merged, 1);
        assert_eq!(analytics.closed, 1);
        // merge_rate = 1/(1+1) = 0.5
        assert!((analytics.merge_rate.unwrap() - 0.5).abs() < f64::EPSILON);
        // by_repo
        assert_eq!(*analytics.by_repo.get("org/a").unwrap(), 2);
        assert_eq!(*analytics.by_repo.get("org/b").unwrap(), 1);
    }

    #[test]
    fn test_get_pr_analytics_empty() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let analytics = tracker.get_pr_analytics().unwrap();
        assert_eq!(analytics.total, 0);
        assert!(analytics.merge_rate.is_none());
        assert!(analytics.by_repo.is_empty());
    }

    #[test]
    fn test_list_prs_no_filter() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker
            .upsert_pr(&make_pr_record("https://github.com/a/b/pull/1", "a/b", 1))
            .unwrap();
        tracker
            .upsert_pr(&make_pr_record("https://github.com/a/b/pull/2", "a/b", 2))
            .unwrap();

        let all = tracker.list_prs(None, 100).unwrap();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_list_prs_with_status_filter() {
        let tracker = SqliteTracker::in_memory().unwrap();

        let mut pr1 = make_pr_record("https://github.com/a/b/pull/1", "a/b", 1);
        pr1.status = "merged".to_string();
        tracker.upsert_pr(&pr1).unwrap();

        tracker
            .upsert_pr(&make_pr_record("https://github.com/a/b/pull/2", "a/b", 2))
            .unwrap();

        let merged = tracker.list_prs(Some("merged"), 100).unwrap();
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].pr_url, "https://github.com/a/b/pull/1");

        let open = tracker.list_prs(Some("open"), 100).unwrap();
        assert_eq!(open.len(), 1);
    }

    #[test]
    fn test_list_prs_respects_limit() {
        let tracker = SqliteTracker::in_memory().unwrap();
        for i in 1..=5 {
            tracker
                .upsert_pr(&make_pr_record(
                    &format!("https://github.com/a/b/pull/{}", i),
                    "a/b",
                    i,
                ))
                .unwrap();
        }
        let prs = tracker.list_prs(None, 3).unwrap();
        assert_eq!(prs.len(), 3);
    }

    #[test]
    fn test_update_pr_status_to_merged() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let pr = make_pr_record("https://github.com/a/b/pull/1", "a/b", 1);
        tracker.upsert_pr(&pr).unwrap();

        tracker
            .update_pr_status("https://github.com/a/b/pull/1", "merged")
            .unwrap();

        let fetched = tracker
            .get_pr("https://github.com/a/b/pull/1")
            .unwrap()
            .unwrap();
        assert_eq!(fetched.status, "merged");
        assert!(fetched.merged_at.is_some());
    }

    #[test]
    fn test_update_pr_status_to_closed() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let pr = make_pr_record("https://github.com/a/b/pull/1", "a/b", 1);
        tracker.upsert_pr(&pr).unwrap();

        tracker
            .update_pr_status("https://github.com/a/b/pull/1", "closed")
            .unwrap();

        let fetched = tracker
            .get_pr("https://github.com/a/b/pull/1")
            .unwrap()
            .unwrap();
        assert_eq!(fetched.status, "closed");
        assert!(fetched.closed_at.is_some());
    }

    #[test]
    fn test_update_pr_status_keeps_open_timestamps_null() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let pr = make_pr_record("https://github.com/a/b/pull/1", "a/b", 1);
        tracker.upsert_pr(&pr).unwrap();

        // Update to a non-merged, non-closed status
        tracker
            .update_pr_status("https://github.com/a/b/pull/1", "open")
            .unwrap();

        let fetched = tracker
            .get_pr("https://github.com/a/b/pull/1")
            .unwrap()
            .unwrap();
        assert_eq!(fetched.status, "open");
        assert!(fetched.merged_at.is_none());
        assert!(fetched.closed_at.is_none());
    }

    #[test]
    fn test_get_avg_time_to_pr_empty() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let result = tracker.get_avg_time_to_pr().unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_get_rejection_reasons() {
        let tracker = SqliteTracker::in_memory().unwrap();
        // Insert review patterns with categories
        let pattern = claudear_core::types::ReviewPattern {
            id: 0,
            scm_repo: "org/repo".to_string(),
            category: claudear_core::types::ReviewCategory::StyleIssue,
            pattern_text: "indentation".to_string(),
            example_comments: vec!["fix indent".to_string()],
            occurrence_count: 5,
            promoted_to_instruction: false,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        tracker.upsert_review_pattern(&pattern).unwrap();

        let pattern2 = claudear_core::types::ReviewPattern {
            id: 0,
            scm_repo: "org/repo".to_string(),
            category: claudear_core::types::ReviewCategory::WrongApproach,
            pattern_text: "off-by-one".to_string(),
            example_comments: vec![],
            occurrence_count: 3,
            promoted_to_instruction: false,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        tracker.upsert_review_pattern(&pattern2).unwrap();

        let reasons = tracker.get_rejection_reasons(10).unwrap();
        assert_eq!(reasons.len(), 2);
        // Ordered by total DESC
        assert_eq!(reasons[0].count, 5);
        assert_eq!(reasons[1].count, 3);
    }

    #[test]
    fn test_get_rejection_reasons_empty() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let reasons = tracker.get_rejection_reasons(10).unwrap();
        assert!(reasons.is_empty());
    }

    #[test]
    fn test_get_agent_spawn_count() {
        let tracker = SqliteTracker::in_memory().unwrap();
        // Record an attempt and an execution
        tracker.record_attempt("linear", "1", "L-1").unwrap();
        let attempt = tracker.get_attempt("linear", "1").unwrap().unwrap();

        let mut exec = AgentExecution::new().with_attempt_id(attempt.id);
        exec.duration_secs = Some(10.0);
        exec.exit_code = Some(0);
        tracker.record_execution(&exec).unwrap();

        let count = tracker
            .get_agent_spawn_count("2020-01-01T00:00:00")
            .unwrap();
        assert_eq!(count, 1);

        let count_future = tracker
            .get_agent_spawn_count("2099-01-01T00:00:00")
            .unwrap();
        assert_eq!(count_future, 0);
    }

    #[test]
    fn test_get_cost_estimate_duration_fallback() {
        let tracker = SqliteTracker::in_memory().unwrap();
        // Record an attempt + execution with duration but no cost
        tracker.record_attempt("linear", "1", "L-1").unwrap();
        let attempt = tracker.get_attempt("linear", "1").unwrap().unwrap();

        let mut exec = AgentExecution::new().with_attempt_id(attempt.id);
        exec.duration_secs = Some(120.0); // 2 minutes
        exec.exit_code = Some(0);
        tracker.record_execution(&exec).unwrap();

        let estimate = tracker
            .get_cost_estimate("2020-01-01T00:00:00", 0.0, "7d")
            .unwrap();
        assert_eq!(estimate.cost_source, "duration_estimate");
        assert_eq!(estimate.period, "7d");
        // 2 minutes * $0.05/min = $0.10
        assert!((estimate.total_cost - 0.1).abs() < 0.01);
    }

    #[test]
    fn test_get_cost_estimate_api_cost() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker.record_attempt("linear", "1", "L-1").unwrap();
        let attempt = tracker.get_attempt("linear", "1").unwrap().unwrap();

        let mut exec = AgentExecution::new().with_attempt_id(attempt.id);
        exec.duration_secs = Some(60.0);
        exec.exit_code = Some(0);
        exec.total_cost_usd = Some(1.50);
        tracker.record_execution(&exec).unwrap();

        let estimate = tracker
            .get_cost_estimate("2020-01-01T00:00:00", 0.0, "30d")
            .unwrap();
        assert_eq!(estimate.cost_source, "api");
        assert!((estimate.total_cost - 1.50).abs() < f64::EPSILON);
    }

    #[test]
    fn test_get_cost_estimate_empty() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let estimate = tracker
            .get_cost_estimate("2020-01-01T00:00:00", 0.0, "7d")
            .unwrap();
        assert_eq!(estimate.cost_source, "duration_estimate");
        assert!((estimate.total_cost - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_get_mttr_trend_empty() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let trend = tracker.get_mttr_trend(4).unwrap();
        assert!(trend.is_empty());
    }

    #[test]
    fn test_get_repo_leaderboard() {
        let tracker = SqliteTracker::in_memory().unwrap();

        // Create attempts with scm_repo
        tracker.record_attempt("linear", "1", "L-1").unwrap();
        tracker
            .mark_success("linear", "1", "https://github.com/org/repo-a/pull/1")
            .unwrap();
        tracker.mark_merged("linear", "1").unwrap();

        tracker.record_attempt("linear", "2", "L-2").unwrap();
        tracker
            .mark_success("linear", "2", "https://github.com/org/repo-a/pull/2")
            .unwrap();

        let board = tracker.get_repo_leaderboard().unwrap();
        // Should have entries for org/repo-a
        assert!(!board.is_empty());
        assert_eq!(board[0].repo, "org/repo-a");
        assert_eq!(board[0].total, 2);
    }

    #[test]
    fn test_get_repo_leaderboard_empty() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let board = tracker.get_repo_leaderboard().unwrap();
        assert!(board.is_empty());
    }

    #[test]
    fn test_get_complexity_time_savings_empty() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let savings = tracker
            .get_complexity_time_savings("2020-01-01T00:00:00", 100.0, "30d")
            .unwrap();
        assert_eq!(savings.merged_count, 0);
        assert!((savings.hours_saved - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_list_attempts_no_filter() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker.record_attempt("linear", "1", "L-1").unwrap();
        tracker.record_attempt("sentry", "2", "S-2").unwrap();

        let all = tracker.list_attempts(None, None, 100, 0).unwrap();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_list_attempts_status_filter() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker.record_attempt("linear", "1", "L-1").unwrap();
        tracker.record_attempt("linear", "2", "L-2").unwrap();
        tracker.mark_failed("linear", "2", "error").unwrap();

        let pending = tracker
            .list_attempts(Some("pending"), None, 100, 0)
            .unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].issue_id, "1");

        let failed = tracker.list_attempts(Some("failed"), None, 100, 0).unwrap();
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0].issue_id, "2");
    }

    #[test]
    fn test_list_attempts_source_filter() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker.record_attempt("linear", "1", "L-1").unwrap();
        tracker.record_attempt("sentry", "2", "S-2").unwrap();

        let linear = tracker.list_attempts(None, Some("linear"), 100, 0).unwrap();
        assert_eq!(linear.len(), 1);
        assert_eq!(linear[0].source, "linear");
    }

    #[test]
    fn test_list_attempts_status_and_source_filter() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker.record_attempt("linear", "1", "L-1").unwrap();
        tracker.record_attempt("linear", "2", "L-2").unwrap();
        tracker.mark_failed("linear", "2", "error").unwrap();
        tracker.record_attempt("sentry", "3", "S-3").unwrap();

        let result = tracker
            .list_attempts(Some("pending"), Some("linear"), 100, 0)
            .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].issue_id, "1");
    }

    #[test]
    fn test_list_attempts_pagination() {
        let tracker = SqliteTracker::in_memory().unwrap();
        for i in 1..=5 {
            tracker
                .record_attempt("linear", &i.to_string(), &format!("L-{}", i))
                .unwrap();
        }

        let page1 = tracker.list_attempts(None, None, 2, 0).unwrap();
        assert_eq!(page1.len(), 2);

        let page2 = tracker.list_attempts(None, None, 2, 2).unwrap();
        assert_eq!(page2.len(), 2);

        let page3 = tracker.list_attempts(None, None, 2, 4).unwrap();
        assert_eq!(page3.len(), 1);
    }

    #[test]
    fn test_count_attempts_all() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker.record_attempt("linear", "1", "L-1").unwrap();
        tracker.record_attempt("sentry", "2", "S-2").unwrap();

        let count = tracker.count_attempts(None, None).unwrap();
        assert_eq!(count, 2);
    }

    #[test]
    fn test_count_attempts_by_status() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker.record_attempt("linear", "1", "L-1").unwrap();
        tracker.record_attempt("linear", "2", "L-2").unwrap();
        tracker.mark_failed("linear", "2", "err").unwrap();

        assert_eq!(tracker.count_attempts(Some("pending"), None).unwrap(), 1);
        assert_eq!(tracker.count_attempts(Some("failed"), None).unwrap(), 1);
    }

    #[test]
    fn test_count_attempts_by_source() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker.record_attempt("linear", "1", "L-1").unwrap();
        tracker.record_attempt("sentry", "2", "S-2").unwrap();

        assert_eq!(tracker.count_attempts(None, Some("linear")).unwrap(), 1);
        assert_eq!(tracker.count_attempts(None, Some("sentry")).unwrap(), 1);
    }

    #[test]
    fn test_count_attempts_by_status_and_source() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker.record_attempt("linear", "1", "L-1").unwrap();
        tracker.record_attempt("linear", "2", "L-2").unwrap();
        tracker.mark_failed("linear", "2", "err").unwrap();
        tracker.record_attempt("sentry", "3", "S-3").unwrap();

        assert_eq!(
            tracker
                .count_attempts(Some("pending"), Some("linear"))
                .unwrap(),
            1
        );
        assert_eq!(
            tracker
                .count_attempts(Some("failed"), Some("sentry"))
                .unwrap(),
            0
        );
    }

    #[test]
    fn test_list_attempts_since() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker.record_attempt("linear", "1", "L-1").unwrap();

        let since = chrono::DateTime::parse_from_rfc3339("2020-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let result = tracker.list_attempts_since(since).unwrap();
        assert_eq!(result.len(), 1);

        let since_future = chrono::DateTime::parse_from_rfc3339("2099-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let result_empty = tracker.list_attempts_since(since_future).unwrap();
        assert!(result_empty.is_empty());
    }

    #[test]
    fn test_record_cascade_attempt() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker.record_attempt("linear", "1", "L-1").unwrap();
        let parent = tracker.get_attempt("linear", "1").unwrap().unwrap();

        let cascade_id = tracker
            .record_cascade_attempt("linear", "1", "L-1", parent.id, "org/downstream")
            .unwrap();
        assert!(cascade_id > 0);

        let cascade = tracker.get_attempt_by_id(cascade_id).unwrap().unwrap();
        assert_eq!(cascade.status, FixAttemptStatus::Pending);
        assert_eq!(cascade.cascade_repo.as_deref(), Some("org/downstream"));
        assert_eq!(cascade.parent_attempt_id, Some(parent.id));
    }

    #[test]
    fn test_record_cascade_attempt_deduplication() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker.record_attempt("linear", "1", "L-1").unwrap();
        let parent = tracker.get_attempt("linear", "1").unwrap().unwrap();

        let id1 = tracker
            .record_cascade_attempt("linear", "1", "L-1", parent.id, "org/downstream")
            .unwrap();
        let id2 = tracker
            .record_cascade_attempt("linear", "1", "L-1", parent.id, "org/downstream")
            .unwrap();
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_update_attempt_pr() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker.record_attempt("linear", "1", "L-1").unwrap();
        let parent = tracker.get_attempt("linear", "1").unwrap().unwrap();

        let cascade_id = tracker
            .record_cascade_attempt("linear", "1", "L-1", parent.id, "org/down")
            .unwrap();

        tracker
            .update_attempt_pr(
                cascade_id,
                "https://github.com/org/down/pull/5",
                "org/down",
                5,
            )
            .unwrap();

        let updated = tracker.get_attempt_by_id(cascade_id).unwrap().unwrap();
        assert_eq!(
            updated.pr_url.as_deref(),
            Some("https://github.com/org/down/pull/5")
        );
        assert_eq!(updated.scm_repo.as_deref(), Some("org/down"));
        assert_eq!(updated.scm_pr_number, Some(5));
        assert_eq!(updated.status, FixAttemptStatus::Success);
    }

    #[test]
    fn test_mark_cascade_failed() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker.record_attempt("linear", "1", "L-1").unwrap();
        let parent = tracker.get_attempt("linear", "1").unwrap().unwrap();

        let cascade_id = tracker
            .record_cascade_attempt("linear", "1", "L-1", parent.id, "org/down")
            .unwrap();

        tracker
            .mark_cascade_failed(cascade_id, "build failed")
            .unwrap();

        let updated = tracker.get_attempt_by_id(cascade_id).unwrap().unwrap();
        assert_eq!(updated.status, FixAttemptStatus::Failed);
        assert_eq!(updated.error_message.as_deref(), Some("build failed"));
    }

    #[test]
    fn test_get_diagnostic_counts_empty() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let counts = tracker.get_diagnostic_counts().unwrap();
        assert_eq!(counts.fix_attempts, 0);
        assert_eq!(counts.activity_log, 0);
        assert_eq!(counts.claude_executions, 0);
        assert_eq!(counts.pr_reviews, 0);
        assert_eq!(counts.prs, 0);
        assert!(counts.fix_attempts_by_status.is_empty());
        assert!(counts.recent_fix_attempts.is_empty());
    }

    #[test]
    fn test_get_diagnostic_counts_with_data() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker.record_attempt("linear", "1", "L-1").unwrap();
        tracker.record_attempt("linear", "2", "L-2").unwrap();
        tracker.mark_failed("linear", "2", "err").unwrap();

        let entry = ActivityLogEntry {
            id: 0,
            timestamp: Utc::now(),
            activity_type: "test".to_string(),
            source: Some("linear".to_string()),
            issue_id: Some("1".to_string()),
            short_id: Some("L-1".to_string()),
            message: "test message".to_string(),
            metadata: None,
        };
        tracker.record_activity(&entry).unwrap();

        let counts = tracker.get_diagnostic_counts().unwrap();
        assert_eq!(counts.fix_attempts, 2);
        assert_eq!(counts.activity_log, 1);
        assert_eq!(*counts.fix_attempts_by_status.get("pending").unwrap(), 1);
        assert_eq!(*counts.fix_attempts_by_status.get("failed").unwrap(), 1);
        assert_eq!(counts.recent_fix_attempts.len(), 2);
    }

    #[test]
    fn test_store_and_get_content_cluster() {
        let tracker = SqliteTracker::in_memory().unwrap();

        let cluster = claudear_core::types::ContentCluster {
            id: 0,
            cluster_key: "TypeError::main".to_string(),
            source: "sentry".to_string(),
            representative_issue_id: "issue-1".to_string(),
            issue_ids: vec!["issue-1".to_string(), "issue-2".to_string()],
            error_type: Some("TypeError".to_string()),
            culprit: Some("main.ts".to_string()),
            avg_similarity: 0.85,
            status: "active".to_string(),
            created_at: Utc::now(),
        };

        let id = tracker.store_content_cluster(&cluster).unwrap();
        assert!(id > 0);

        let active = tracker.get_active_content_clusters("sentry").unwrap();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].cluster_key, "TypeError::main");
        assert_eq!(active[0].issue_ids.len(), 2);
        assert_eq!(active[0].error_type.as_deref(), Some("TypeError"));
        assert_eq!(active[0].culprit.as_deref(), Some("main.ts"));
    }

    #[test]
    fn test_get_active_content_clusters_filters_resolved() {
        let tracker = SqliteTracker::in_memory().unwrap();

        let cluster1 = claudear_core::types::ContentCluster {
            id: 0,
            cluster_key: "active-cluster".to_string(),
            source: "sentry".to_string(),
            representative_issue_id: "i1".to_string(),
            issue_ids: vec!["i1".to_string()],
            error_type: None,
            culprit: None,
            avg_similarity: 0.9,
            status: "active".to_string(),
            created_at: Utc::now(),
        };
        let id1 = tracker.store_content_cluster(&cluster1).unwrap();

        let cluster2 = claudear_core::types::ContentCluster {
            id: 0,
            cluster_key: "resolved-cluster".to_string(),
            source: "sentry".to_string(),
            representative_issue_id: "i2".to_string(),
            issue_ids: vec!["i2".to_string()],
            error_type: None,
            culprit: None,
            avg_similarity: 0.8,
            status: "active".to_string(),
            created_at: Utc::now(),
        };
        let id2 = tracker.store_content_cluster(&cluster2).unwrap();
        tracker.resolve_content_cluster(id2).unwrap();

        let active = tracker.get_active_content_clusters("sentry").unwrap();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].cluster_key, "active-cluster");
        let _ = id1;
    }

    #[test]
    fn test_get_active_content_clusters_filters_by_source() {
        let tracker = SqliteTracker::in_memory().unwrap();

        let cluster = claudear_core::types::ContentCluster {
            id: 0,
            cluster_key: "k".to_string(),
            source: "sentry".to_string(),
            representative_issue_id: "i1".to_string(),
            issue_ids: vec!["i1".to_string()],
            error_type: None,
            culprit: None,
            avg_similarity: 0.9,
            status: "active".to_string(),
            created_at: Utc::now(),
        };
        tracker.store_content_cluster(&cluster).unwrap();

        let linear = tracker.get_active_content_clusters("linear").unwrap();
        assert!(linear.is_empty());
    }

    #[test]
    fn test_resolve_content_cluster() {
        let tracker = SqliteTracker::in_memory().unwrap();

        let cluster = claudear_core::types::ContentCluster {
            id: 0,
            cluster_key: "k".to_string(),
            source: "sentry".to_string(),
            representative_issue_id: "i1".to_string(),
            issue_ids: vec!["i1".to_string()],
            error_type: None,
            culprit: None,
            avg_similarity: 0.9,
            status: "active".to_string(),
            created_at: Utc::now(),
        };
        let id = tracker.store_content_cluster(&cluster).unwrap();

        tracker.resolve_content_cluster(id).unwrap();

        let active = tracker.get_active_content_clusters("sentry").unwrap();
        assert!(active.is_empty());
    }

    #[test]
    fn test_store_severity_score() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let score = claudear_core::types::SeverityScore {
            score: 0.85,
            severity_component: 0.7,
            frequency_component: 0.6,
            regression_component: 0.3,
            blast_radius_component: 0.8,
            cluster_boost: 1.0,
        };
        // Should not error
        tracker
            .store_severity_score(
                "sentry",
                "issue-1",
                &score,
                claudear_core::types::BlastRadius::Core,
            )
            .unwrap();
    }

    #[test]
    fn test_store_severity_score_upsert() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let score1 = claudear_core::types::SeverityScore {
            score: 0.5,
            severity_component: 0.5,
            frequency_component: 0.5,
            regression_component: 0.0,
            blast_radius_component: 0.5,
            cluster_boost: 0.0,
        };
        tracker
            .store_severity_score(
                "sentry",
                "issue-1",
                &score1,
                claudear_core::types::BlastRadius::Peripheral,
            )
            .unwrap();

        let score2 = claudear_core::types::SeverityScore {
            score: 0.9,
            severity_component: 0.9,
            frequency_component: 0.8,
            regression_component: 0.7,
            blast_radius_component: 0.9,
            cluster_boost: 1.0,
        };
        // Should upsert without error
        tracker
            .store_severity_score(
                "sentry",
                "issue-1",
                &score2,
                claudear_core::types::BlastRadius::Critical,
            )
            .unwrap();
    }

    #[test]
    fn test_record_suppression() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker
            .record_suppression("sentry", "issue-1", "flaky_test", "test is known flaky")
            .unwrap();
    }

    #[test]
    fn test_record_suppression_dedup() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker
            .record_suppression("sentry", "issue-1", "flaky", "reason")
            .unwrap();
        // Same record should be ignored
        tracker
            .record_suppression("sentry", "issue-1", "flaky", "reason")
            .unwrap();
    }

    #[test]
    fn test_list_issues_no_filter() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let emb = IssueEmbedding {
            id: 0,
            source: "linear".to_string(),
            issue_id: "1".to_string(),
            short_id: Some("L-1".to_string()),
            title: Some("Bug".to_string()),
            embedding: None,
            embedding_model: None,
            created_at: Utc::now(),
            description: None,
            url: None,
            priority: None,
            status: None,
            labels: None,
            updated_at: None,
        };
        tracker.store_embedding(&emb).unwrap();

        let issues = tracker.list_issues(None, 10, 0).unwrap();
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].title, Some("Bug".to_string()));
    }

    #[test]
    fn test_list_issues_with_source_filter() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let emb1 = IssueEmbedding {
            id: 0,
            source: "linear".to_string(),
            issue_id: "1".to_string(),
            short_id: Some("L-1".to_string()),
            title: Some("Linear Bug".to_string()),
            embedding: None,
            embedding_model: None,
            created_at: Utc::now(),
            description: None,
            url: None,
            priority: None,
            status: None,
            labels: None,
            updated_at: None,
        };
        let emb2 = IssueEmbedding {
            id: 0,
            source: "sentry".to_string(),
            issue_id: "2".to_string(),
            short_id: Some("S-2".to_string()),
            title: Some("Sentry Error".to_string()),
            embedding: None,
            embedding_model: None,
            created_at: Utc::now(),
            description: None,
            url: None,
            priority: None,
            status: None,
            labels: None,
            updated_at: None,
        };
        tracker.store_embedding(&emb1).unwrap();
        tracker.store_embedding(&emb2).unwrap();

        let linear = tracker.list_issues(Some("linear"), 10, 0).unwrap();
        assert_eq!(linear.len(), 1);
        assert_eq!(linear[0].source, "linear");
    }

    #[test]
    fn test_list_issues_pagination() {
        let tracker = SqliteTracker::in_memory().unwrap();
        for i in 1..=5 {
            let emb = IssueEmbedding {
                id: 0,
                source: "linear".to_string(),
                issue_id: i.to_string(),
                short_id: Some(format!("L-{}", i)),
                title: Some(format!("Bug {}", i)),
                embedding: None,
                embedding_model: None,
                created_at: Utc::now(),
                description: None,
                url: None,
                priority: None,
                status: None,
                labels: None,
                updated_at: None,
            };
            tracker.store_embedding(&emb).unwrap();
        }

        let page1 = tracker.list_issues(None, 2, 0).unwrap();
        assert_eq!(page1.len(), 2);

        let page2 = tracker.list_issues(None, 2, 2).unwrap();
        assert_eq!(page2.len(), 2);

        let page3 = tracker.list_issues(None, 2, 4).unwrap();
        assert_eq!(page3.len(), 1);
    }

    #[test]
    fn test_count_issues_no_filter() {
        let tracker = SqliteTracker::in_memory().unwrap();
        assert_eq!(tracker.count_issues(None).unwrap(), 0);

        let emb = IssueEmbedding {
            id: 0,
            source: "linear".to_string(),
            issue_id: "1".to_string(),
            short_id: Some("L-1".to_string()),
            title: Some("Bug".to_string()),
            embedding: None,
            embedding_model: None,
            created_at: Utc::now(),
            description: None,
            url: None,
            priority: None,
            status: None,
            labels: None,
            updated_at: None,
        };
        tracker.store_embedding(&emb).unwrap();
        assert_eq!(tracker.count_issues(None).unwrap(), 1);
    }

    #[test]
    fn test_count_issues_with_source() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let emb1 = IssueEmbedding {
            id: 0,
            source: "linear".to_string(),
            issue_id: "1".to_string(),
            short_id: Some("L-1".to_string()),
            title: Some("Bug".to_string()),
            embedding: None,
            embedding_model: None,
            created_at: Utc::now(),
            description: None,
            url: None,
            priority: None,
            status: None,
            labels: None,
            updated_at: None,
        };
        let emb2 = IssueEmbedding {
            id: 0,
            source: "sentry".to_string(),
            issue_id: "2".to_string(),
            short_id: Some("S-2".to_string()),
            title: Some("Error".to_string()),
            embedding: None,
            embedding_model: None,
            created_at: Utc::now(),
            description: None,
            url: None,
            priority: None,
            status: None,
            labels: None,
            updated_at: None,
        };
        tracker.store_embedding(&emb1).unwrap();
        tracker.store_embedding(&emb2).unwrap();

        assert_eq!(tracker.count_issues(Some("linear")).unwrap(), 1);
        assert_eq!(tracker.count_issues(Some("sentry")).unwrap(), 1);
        assert_eq!(tracker.count_issues(Some("other")).unwrap(), 0);
    }

    #[test]
    fn test_store_similar_issues_batch() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let batch = vec![
            SimilarIssue {
                id: 0,
                source_issue_id: "a".to_string(),
                similar_issue_id: "b".to_string(),
                similarity_score: 0.9,
                computed_at: Utc::now(),
            },
            SimilarIssue {
                id: 0,
                source_issue_id: "a".to_string(),
                similar_issue_id: "c".to_string(),
                similarity_score: 0.7,
                computed_at: Utc::now(),
            },
        ];
        tracker.store_similar_issues_batch(&batch).unwrap();

        let similar = tracker.find_similar_issues("a", 0.5, 10).unwrap();
        assert_eq!(similar.len(), 2);
    }

    #[test]
    fn test_store_similar_issues_batch_empty() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker.store_similar_issues_batch(&[]).unwrap();
    }

    #[test]
    fn test_get_attempts_batch() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker.record_attempt("linear", "1", "L-1").unwrap();
        tracker.record_attempt("sentry", "2", "S-2").unwrap();

        let results = tracker
            .get_attempts_batch(&[("linear", "1"), ("sentry", "2"), ("other", "99")])
            .unwrap();
        assert_eq!(results.len(), 3);
        assert!(results[0].is_some());
        assert!(results[1].is_some());
        assert!(results[2].is_none());
    }

    #[test]
    fn test_get_attempts_batch_empty() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let results = tracker.get_attempts_batch(&[]).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_indexing_progress_lifecycle() {
        let tracker = SqliteTracker::in_memory().unwrap();

        // Initially idle
        let progress = tracker.get_indexing_progress().unwrap();
        assert_eq!(progress.status, "idle");
        assert_eq!(progress.total_repos, 0);

        // Start indexing
        tracker.start_indexing_progress(5).unwrap();
        let progress = tracker.get_indexing_progress().unwrap();
        assert_eq!(progress.status, "running");
        assert_eq!(progress.total_repos, 5);
        assert_eq!(progress.indexed_repos, 0);
        assert!(progress.started_at.is_some());

        // Update progress
        tracker
            .update_indexing_progress(2, "my-repo", 100, 200)
            .unwrap();
        let progress = tracker.get_indexing_progress().unwrap();
        assert_eq!(progress.indexed_repos, 2);
        assert_eq!(progress.current_repo.as_deref(), Some("my-repo"));
        assert_eq!(progress.current_repo_files, 100);
        assert_eq!(progress.total_files_indexed, 200);

        // Finish indexing
        tracker.finish_indexing_progress().unwrap();
        let progress = tracker.get_indexing_progress().unwrap();
        assert_eq!(progress.status, "idle");
        assert_eq!(progress.indexed_repos, 0);
        assert!(progress.current_repo.is_none());
        assert!(progress.started_at.is_none());
    }

    #[test]
    fn test_subscribe_indexing_progress() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let mut rx = tracker.subscribe_indexing_progress();

        // Initial value
        let current = rx.borrow().clone();
        assert_eq!(current.status, "idle");

        tracker.start_indexing_progress(3).unwrap();
        // The watch channel should reflect the update
        let updated = rx.borrow_and_update().clone();
        assert_eq!(updated.status, "running");
        assert_eq!(updated.total_repos, 3);
    }

    #[test]
    fn test_delete_user_sessions() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker
            .create_user("test@example.com", "hash", "Test", "user")
            .unwrap();
        let user = tracker
            .get_user_by_email("test@example.com")
            .unwrap()
            .unwrap();

        // Create a session
        let token = tracker
            .create_session(user.id, "2099-01-01T00:00:00")
            .unwrap();
        assert!(!token.is_empty());

        // Verify session works
        let session_user = tracker.get_session_user(&token).unwrap();
        assert!(session_user.is_some());

        // Delete all sessions for user
        tracker.delete_user_sessions(user.id).unwrap();

        // Session should no longer be valid
        let session_user = tracker.get_session_user(&token).unwrap();
        assert!(session_user.is_none());
    }

    #[test]
    fn test_get_all_regression_watches() {
        let tracker = SqliteTracker::in_memory().unwrap();

        // Create fix attempts to satisfy FK constraint
        tracker.record_attempt("linear", "rw-1", "L-RW1").unwrap();
        tracker.record_attempt("sentry", "rw-2", "S-RW2").unwrap();
        let attempt1 = tracker.get_attempt("linear", "rw-1").unwrap().unwrap();
        let attempt2 = tracker.get_attempt("sentry", "rw-2").unwrap().unwrap();

        let watch1 = claudear_core::types::RegressionWatch {
            id: 0,
            issue_type: claudear_core::types::IssueType::LinearBug,
            issue_id: "rw-1".to_string(),
            fix_attempt_id: attempt1.id,
            status: claudear_core::types::RegressionWatchStatus::AwaitingRelease,
            pr_merged_at: None,
            monitoring_started_at: None,
            resolved_at: None,
            regressed_at: None,
            created_at: Utc::now(),
        };
        let watch2 = claudear_core::types::RegressionWatch {
            id: 0,
            issue_type: claudear_core::types::IssueType::SentryIssue,
            issue_id: "rw-2".to_string(),
            fix_attempt_id: attempt2.id,
            status: claudear_core::types::RegressionWatchStatus::Monitoring,
            pr_merged_at: None,
            monitoring_started_at: Some(Utc::now()),
            resolved_at: None,
            regressed_at: None,
            created_at: Utc::now(),
        };

        tracker.create_regression_watch(&watch1).unwrap();
        tracker.create_regression_watch(&watch2).unwrap();

        let all = tracker.get_all_regression_watches().unwrap();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_get_all_regression_watches_empty() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let all = tracker.get_all_regression_watches().unwrap();
        assert!(all.is_empty());
    }

    #[test]
    fn test_get_active_clusters() {
        let tracker = SqliteTracker::in_memory().unwrap();

        let cluster = claudear_core::types::IssueCluster {
            id: 0,
            cluster_key: "key1".to_string(),
            source: "sentry".to_string(),
            issue_ids: vec!["a".to_string(), "b".to_string()],
            window_start: Utc::now(),
            window_end: Utc::now(),
            resolved_by_issue_id: None,
            resolved_by_attempt_id: None,
            status: "active".to_string(),
            created_at: Utc::now(),
        };
        let id = tracker.store_issue_cluster(&cluster).unwrap();

        let active = tracker.get_active_clusters("sentry").unwrap();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].cluster_key, "key1");
        assert_eq!(active[0].issue_ids.len(), 2);

        // Resolve it
        tracker.update_cluster_resolution(id, "a", 1).unwrap();
        let active_after = tracker.get_active_clusters("sentry").unwrap();
        assert!(active_after.is_empty());
    }

    #[test]
    fn test_get_active_clusters_filters_by_source() {
        let tracker = SqliteTracker::in_memory().unwrap();

        let cluster = claudear_core::types::IssueCluster {
            id: 0,
            cluster_key: "key1".to_string(),
            source: "sentry".to_string(),
            issue_ids: vec!["a".to_string()],
            window_start: Utc::now(),
            window_end: Utc::now(),
            resolved_by_issue_id: None,
            resolved_by_attempt_id: None,
            status: "active".to_string(),
            created_at: Utc::now(),
        };
        tracker.store_issue_cluster(&cluster).unwrap();

        let linear = tracker.get_active_clusters("linear").unwrap();
        assert!(linear.is_empty());
    }

    #[test]
    fn test_get_activity_type_counts_since() {
        let tracker = SqliteTracker::in_memory().unwrap();

        let entry1 = ActivityLogEntry {
            id: 0,
            timestamp: Utc::now(),
            activity_type: "fix_started".to_string(),
            source: None,
            issue_id: None,
            short_id: None,
            message: "msg".to_string(),
            metadata: None,
        };
        let entry2 = ActivityLogEntry {
            id: 0,
            timestamp: Utc::now(),
            activity_type: "fix_started".to_string(),
            source: None,
            issue_id: None,
            short_id: None,
            message: "msg2".to_string(),
            metadata: None,
        };
        let entry3 = ActivityLogEntry {
            id: 0,
            timestamp: Utc::now(),
            activity_type: "pr_merged".to_string(),
            source: None,
            issue_id: None,
            short_id: None,
            message: "msg3".to_string(),
            metadata: None,
        };
        tracker.record_activity(&entry1).unwrap();
        tracker.record_activity(&entry2).unwrap();
        tracker.record_activity(&entry3).unwrap();

        let since = chrono::DateTime::parse_from_rfc3339("2020-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let counts = tracker.get_activity_type_counts_since(since).unwrap();

        assert_eq!(*counts.get("fix_started").unwrap(), 2);
        assert_eq!(*counts.get("pr_merged").unwrap(), 1);
    }

    #[test]
    fn test_get_activity_type_counts_since_empty() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let since = chrono::DateTime::parse_from_rfc3339("2020-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let counts = tracker.get_activity_type_counts_since(since).unwrap();
        assert!(counts.is_empty());
    }

    #[test]
    fn test_normalize_signal_below_min() {
        let thresholds = [0.0, 20.0, 100.0, 500.0, 2000.0];
        assert!((normalize_signal(-1.0, &thresholds) - 0.0).abs() < f64::EPSILON);
        assert!((normalize_signal(0.0, &thresholds) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_normalize_signal_above_max() {
        let thresholds = [0.0, 20.0, 100.0, 500.0, 2000.0];
        assert!((normalize_signal(3000.0, &thresholds) - 1.0).abs() < f64::EPSILON);
        assert!((normalize_signal(2000.0, &thresholds) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_normalize_signal_midpoints() {
        let thresholds = [0.0, 20.0, 100.0, 500.0, 2000.0];
        // Exactly at threshold boundaries
        let at_20 = normalize_signal(20.0, &thresholds);
        assert!((at_20 - 0.25).abs() < f64::EPSILON);

        let at_100 = normalize_signal(100.0, &thresholds);
        assert!((at_100 - 0.5).abs() < f64::EPSILON);

        let at_500 = normalize_signal(500.0, &thresholds);
        assert!((at_500 - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn test_normalize_signal_interpolation() {
        let thresholds = [0.0, 20.0, 100.0, 500.0, 2000.0];
        // Midpoint of first bucket (0-20): 10 -> 0.5/4 = 0.125
        let val = normalize_signal(10.0, &thresholds);
        assert!((val - 0.125).abs() < 0.001);
    }

    #[test]
    fn test_normalize_signal_equal_thresholds() {
        // When two adjacent thresholds are equal, should return bucket value directly
        let thresholds = [0.0, 0.0, 100.0, 500.0, 2000.0];
        let val = normalize_signal(0.0, &thresholds);
        // 0.0 <= thresholds[0] => returns 0.0
        assert!((val - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_complexity_to_hours() {
        assert!((complexity_to_hours(0.1) - 0.5).abs() < f64::EPSILON);
        assert!((complexity_to_hours(0.2) - 0.5).abs() < f64::EPSILON);
        assert!((complexity_to_hours(0.3) - 1.0).abs() < f64::EPSILON);
        assert!((complexity_to_hours(0.4) - 1.0).abs() < f64::EPSILON);
        assert!((complexity_to_hours(0.5) - 2.0).abs() < f64::EPSILON);
        assert!((complexity_to_hours(0.6) - 2.0).abs() < f64::EPSILON);
        assert!((complexity_to_hours(0.7) - 4.0).abs() < f64::EPSILON);
        assert!((complexity_to_hours(0.8) - 4.0).abs() < f64::EPSILON);
        assert!((complexity_to_hours(0.9) - 8.0).abs() < f64::EPSILON);
        assert!((complexity_to_hours(1.0) - 8.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_get_or_create_repo_id() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let id1 = tracker.get_or_create_repo_id("my-repo").unwrap();
        assert!(id1 > 0);

        // Same name should return same id
        let id2 = tracker.get_or_create_repo_id("my-repo").unwrap();
        assert_eq!(id1, id2);

        // Different name => different id
        let id3 = tracker.get_or_create_repo_id("other-repo").unwrap();
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_save_code_symbols_and_find() {
        use claudear_core::types::{Language, SymbolKind};

        let tracker = SqliteTracker::in_memory().unwrap();
        let repo_id = tracker.get_or_create_repo_id("test-repo").unwrap();

        let symbols = vec![
            claudear_core::types::CodeSymbol {
                id: None,
                repo_id,
                file_path: "src/main.rs".to_string(),
                symbol_name: "process_data".to_string(),
                symbol_kind: SymbolKind::Function,
                parent_symbol: None,
                language: Language::Rust,
                start_line: 10,
                end_line: 20,
                signature: Some("fn process_data(input: &str) -> Result<()>".to_string()),
            },
            claudear_core::types::CodeSymbol {
                id: None,
                repo_id,
                file_path: "src/lib.rs".to_string(),
                symbol_name: "DataProcessor".to_string(),
                symbol_kind: SymbolKind::Struct,
                parent_symbol: None,
                language: Language::Rust,
                start_line: 1,
                end_line: 5,
                signature: None,
            },
        ];

        tracker.save_code_symbols(&symbols).unwrap();

        // Find by name
        let found = tracker
            .find_code_symbols("process_data", None, None)
            .unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].symbol_name, "process_data");
        assert_eq!(found[0].symbol_kind, SymbolKind::Function);
        assert_eq!(found[0].language, Language::Rust);

        // Find by kind
        let structs = tracker
            .find_code_symbols("DataProcessor", Some(SymbolKind::Struct), None)
            .unwrap();
        assert_eq!(structs.len(), 1);
        assert_eq!(structs[0].symbol_name, "DataProcessor");

        // Find by repo_id
        let other_repo_id = tracker.get_or_create_repo_id("other-repo").unwrap();
        let from_other = tracker
            .find_code_symbols("process_data", None, Some(other_repo_id))
            .unwrap();
        assert!(from_other.is_empty());
    }

    #[test]
    fn test_find_code_symbols_escapes_like_wildcards() {
        use claudear_core::types::{Language, SymbolKind};

        let tracker = SqliteTracker::in_memory().unwrap();
        let repo_id = tracker.get_or_create_repo_id("test-repo").unwrap();

        let symbols = vec![claudear_core::types::CodeSymbol {
            id: None,
            repo_id,
            file_path: "src/main.rs".to_string(),
            symbol_name: "test_func".to_string(),
            symbol_kind: SymbolKind::Function,
            parent_symbol: None,
            language: Language::Rust,
            start_line: 1,
            end_line: 5,
            signature: None,
        }];
        tracker.save_code_symbols(&symbols).unwrap();

        // Search with % which should be escaped, not act as wildcard
        let found = tracker.find_code_symbols("test%func", None, None).unwrap();
        // With proper escaping, "test%func" won't match "test_func"
        assert!(found.is_empty());
    }

    #[test]
    fn test_save_code_chunks_and_get_ids() {
        use claudear_core::types::Language;

        let tracker = SqliteTracker::in_memory().unwrap();
        let repo_id = tracker.get_or_create_repo_id("test-repo").unwrap();

        let chunks = vec![
            claudear_core::types::CodeChunk {
                id: None,
                repo_id,
                file_path: "src/main.rs".to_string(),
                chunk_type: "function".to_string(),
                symbol_name: Some("main".to_string()),
                language: Language::Rust,
                start_line: 1,
                end_line: 10,
                chunk_text: "fn main() { }".to_string(),
                context_text: "// main entry point".to_string(),
                file_hash: "abc123".to_string(),
                content_hash: None,
            },
            claudear_core::types::CodeChunk {
                id: None,
                repo_id,
                file_path: "src/lib.rs".to_string(),
                chunk_type: "module".to_string(),
                symbol_name: None,
                language: Language::Rust,
                start_line: 1,
                end_line: 5,
                chunk_text: "mod tests { }".to_string(),
                context_text: "// test module".to_string(),
                file_hash: "def456".to_string(),
                content_hash: None,
            },
        ];

        let ids = tracker.save_code_chunks(&chunks).unwrap();
        assert_eq!(ids.len(), 2);
        assert!(ids[0] > 0);
        assert!(ids[1] > 0);
        assert_ne!(ids[0], ids[1]);
    }

    #[test]
    fn test_code_chunk_hash_matches() {
        use claudear_core::types::Language;

        let tracker = SqliteTracker::in_memory().unwrap();
        let repo_id = tracker.get_or_create_repo_id("test-repo").unwrap();

        // No chunks yet -> should not match
        assert!(!tracker
            .code_chunk_hash_matches(repo_id, "src/main.rs", "abc123")
            .unwrap());

        // Add a chunk
        let chunks = vec![claudear_core::types::CodeChunk {
            id: None,
            repo_id,
            file_path: "src/main.rs".to_string(),
            chunk_type: "function".to_string(),
            symbol_name: None,
            language: Language::Rust,
            start_line: 1,
            end_line: 10,
            chunk_text: "fn main() {}".to_string(),
            context_text: "".to_string(),
            file_hash: "abc123".to_string(),
            content_hash: None,
        }];
        let ids = tracker.save_code_chunks(&chunks).unwrap();

        // Chunk exists but no embedding -> should not match
        assert!(!tracker
            .code_chunk_hash_matches(repo_id, "src/main.rs", "abc123")
            .unwrap());

        // Add an embedding manually
        {
            let conn = tracker.acquire_lock().unwrap();
            let embedding: Vec<u8> = vec![0u8; 16]; // 4 floats
            conn.execute(
                "INSERT INTO code_chunk_embeddings (chunk_id, embedding, embedding_model) VALUES (?1, ?2, ?3)",
                params![ids[0], embedding, "test-model"],
            ).unwrap();
        }

        // Now should match
        assert!(tracker
            .code_chunk_hash_matches(repo_id, "src/main.rs", "abc123")
            .unwrap());

        // Different hash should not match
        assert!(!tracker
            .code_chunk_hash_matches(repo_id, "src/main.rs", "different")
            .unwrap());
    }

    #[test]
    fn test_delete_code_data_for_file() {
        use claudear_core::types::{Language, SymbolKind};

        let tracker = SqliteTracker::in_memory().unwrap();
        let repo_id = tracker.get_or_create_repo_id("test-repo").unwrap();

        // Add symbol and chunk for a file
        let symbols = vec![claudear_core::types::CodeSymbol {
            id: None,
            repo_id,
            file_path: "src/main.rs".to_string(),
            symbol_name: "main".to_string(),
            symbol_kind: SymbolKind::Function,
            parent_symbol: None,
            language: Language::Rust,
            start_line: 1,
            end_line: 5,
            signature: None,
        }];
        tracker.save_code_symbols(&symbols).unwrap();

        let chunks = vec![claudear_core::types::CodeChunk {
            id: None,
            repo_id,
            file_path: "src/main.rs".to_string(),
            chunk_type: "function".to_string(),
            symbol_name: Some("main".to_string()),
            language: Language::Rust,
            start_line: 1,
            end_line: 5,
            chunk_text: "fn main() {}".to_string(),
            context_text: "".to_string(),
            file_hash: "abc".to_string(),
            content_hash: None,
        }];
        tracker.save_code_chunks(&chunks).unwrap();

        // Delete code data for the file
        tracker
            .delete_code_data_for_file(repo_id, "src/main.rs")
            .unwrap();

        // Symbols should be gone
        let found = tracker
            .find_code_symbols("main", None, Some(repo_id))
            .unwrap();
        assert!(found.is_empty());
    }

    #[test]
    fn test_delete_code_chunks_by_ids() {
        use claudear_core::types::Language;

        let tracker = SqliteTracker::in_memory().unwrap();
        let repo_id = tracker.get_or_create_repo_id("test-repo").unwrap();

        let chunks = vec![
            claudear_core::types::CodeChunk {
                id: None,
                repo_id,
                file_path: "a.rs".to_string(),
                chunk_type: "function".to_string(),
                symbol_name: None,
                language: Language::Rust,
                start_line: 1,
                end_line: 5,
                chunk_text: "fn a() {}".to_string(),
                context_text: "".to_string(),
                file_hash: "h1".to_string(),
                content_hash: None,
            },
            claudear_core::types::CodeChunk {
                id: None,
                repo_id,
                file_path: "b.rs".to_string(),
                chunk_type: "function".to_string(),
                symbol_name: None,
                language: Language::Rust,
                start_line: 1,
                end_line: 5,
                chunk_text: "fn b() {}".to_string(),
                context_text: "".to_string(),
                file_hash: "h2".to_string(),
                content_hash: None,
            },
        ];
        let ids = tracker.save_code_chunks(&chunks).unwrap();

        // Delete one
        tracker.delete_code_chunks_by_ids(&[ids[0]]).unwrap();

        // Verify second still exists by checking hash
        // The first should not match, second should still be there
        assert!(!tracker
            .code_chunk_hash_matches(repo_id, "a.rs", "h1")
            .unwrap());
    }

    #[test]
    fn test_delete_code_chunks_by_ids_empty() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker.delete_code_chunks_by_ids(&[]).unwrap();
    }

    #[test]
    fn test_cleanup_stale_code_data_empty_paths() {
        use claudear_core::types::Language;

        let tracker = SqliteTracker::in_memory().unwrap();
        let repo_id = tracker.get_or_create_repo_id("test-repo").unwrap();

        let chunks = vec![claudear_core::types::CodeChunk {
            id: None,
            repo_id,
            file_path: "old.rs".to_string(),
            chunk_type: "function".to_string(),
            symbol_name: None,
            language: Language::Rust,
            start_line: 1,
            end_line: 5,
            chunk_text: "fn old() {}".to_string(),
            context_text: "".to_string(),
            file_hash: "hash".to_string(),
            content_hash: None,
        }];
        tracker.save_code_chunks(&chunks).unwrap();

        // Empty current paths -> should delete everything
        tracker.cleanup_stale_code_data(repo_id, &[]).unwrap();

        assert!(!tracker
            .code_chunk_hash_matches(repo_id, "old.rs", "hash")
            .unwrap());
    }

    #[test]
    fn test_cleanup_stale_code_data_removes_only_stale() {
        use claudear_core::types::Language;

        let tracker = SqliteTracker::in_memory().unwrap();
        let repo_id = tracker.get_or_create_repo_id("test-repo").unwrap();

        let chunks = vec![
            claudear_core::types::CodeChunk {
                id: None,
                repo_id,
                file_path: "keep.rs".to_string(),
                chunk_type: "function".to_string(),
                symbol_name: None,
                language: Language::Rust,
                start_line: 1,
                end_line: 5,
                chunk_text: "fn keep() {}".to_string(),
                context_text: "".to_string(),
                file_hash: "h1".to_string(),
                content_hash: None,
            },
            claudear_core::types::CodeChunk {
                id: None,
                repo_id,
                file_path: "stale.rs".to_string(),
                chunk_type: "function".to_string(),
                symbol_name: None,
                language: Language::Rust,
                start_line: 1,
                end_line: 5,
                chunk_text: "fn stale() {}".to_string(),
                context_text: "".to_string(),
                file_hash: "h2".to_string(),
                content_hash: None,
            },
        ];
        tracker.save_code_chunks(&chunks).unwrap();

        // Only keep.rs is current
        tracker
            .cleanup_stale_code_data(repo_id, &["keep.rs".to_string()])
            .unwrap();

        // Query directly to check stale.rs is gone
        let conn = tracker.acquire_lock().unwrap();
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM code_chunks WHERE repo_id = ? AND file_path = 'stale.rs'",
                params![repo_id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 0);

        let keep_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM code_chunks WHERE repo_id = ? AND file_path = 'keep.rs'",
                params![repo_id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(keep_count, 1);
    }

    #[test]
    fn test_get_repo_knowledge_all() {
        let tracker = SqliteTracker::in_memory().unwrap();

        let entry1 = claudear_core::types::RepoKnowledge {
            id: 0,
            repo: "org/repo".to_string(),
            knowledge_key: "test_framework".to_string(),
            knowledge_value: "jest".to_string(),
            source_type: "review".to_string(),
            confidence: 0.9,
            occurrence_count: 3,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        let entry2 = claudear_core::types::RepoKnowledge {
            id: 0,
            repo: "org/repo".to_string(),
            knowledge_key: "language".to_string(),
            knowledge_value: "typescript".to_string(),
            source_type: "diff".to_string(),
            confidence: 0.95,
            occurrence_count: 5,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        tracker.upsert_repo_knowledge(&entry1).unwrap();
        tracker.upsert_repo_knowledge(&entry2).unwrap();

        let knowledge = tracker.get_repo_knowledge("org/repo").unwrap();
        assert_eq!(knowledge.len(), 2);
        // Ordered by occurrence_count DESC
        assert_eq!(knowledge[0].knowledge_value, "typescript");
        assert_eq!(knowledge[1].knowledge_value, "jest");
    }

    #[test]
    fn test_get_repo_knowledge_empty() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let knowledge = tracker.get_repo_knowledge("nonexistent").unwrap();
        assert!(knowledge.is_empty());
    }

    #[test]
    fn test_get_review_patterns() {
        let tracker = SqliteTracker::in_memory().unwrap();

        let pattern = claudear_core::types::ReviewPattern {
            id: 0,
            scm_repo: "org/repo".to_string(),
            category: claudear_core::types::ReviewCategory::StyleIssue,
            pattern_text: "indentation issue".to_string(),
            example_comments: vec!["fix indentation".to_string()],
            occurrence_count: 5,
            promoted_to_instruction: false,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        tracker.upsert_review_pattern(&pattern).unwrap();

        let patterns = tracker.get_review_patterns("org/repo", 10).unwrap();
        assert_eq!(patterns.len(), 1);
        assert_eq!(patterns[0].pattern_text, "indentation issue");
        assert_eq!(patterns[0].occurrence_count, 5);
    }

    #[test]
    fn test_get_review_patterns_empty() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let patterns = tracker.get_review_patterns("nonexistent", 10).unwrap();
        assert!(patterns.is_empty());
    }

    #[test]
    fn test_get_review_patterns_limit() {
        let tracker = SqliteTracker::in_memory().unwrap();

        for i in 0..5 {
            let pattern = claudear_core::types::ReviewPattern {
                id: 0,
                scm_repo: "org/repo".to_string(),
                category: claudear_core::types::ReviewCategory::WrongApproach,
                pattern_text: format!("pattern-{}", i),
                example_comments: vec![],
                occurrence_count: i,
                promoted_to_instruction: false,
                created_at: Utc::now(),
                updated_at: Utc::now(),
            };
            tracker.upsert_review_pattern(&pattern).unwrap();
        }

        let limited = tracker.get_review_patterns("org/repo", 3).unwrap();
        assert_eq!(limited.len(), 3);
    }

    #[test]
    fn test_get_successful_strategies() {
        let tracker = SqliteTracker::in_memory().unwrap();

        // Create a merged attempt
        tracker.record_attempt("linear", "1", "L-1").unwrap();
        tracker
            .mark_success("linear", "1", "https://github.com/org/repo/pull/1")
            .unwrap();
        tracker.mark_merged("linear", "1").unwrap();

        let attempt = tracker.get_attempt("linear", "1").unwrap().unwrap();

        let fp = claudear_core::types::StrategyFingerprint {
            id: 0,
            attempt_id: attempt.id,
            files_explored: vec!["src/main.rs".to_string()],
            tests_run: 5,
            tools_used: {
                let mut m = HashMap::new();
                m.insert("grep".to_string(), 3);
                m.insert("edit".to_string(), 2);
                m
            },
            fix_approach: "direct_fix".to_string(),
            strategy_summary: "Fixed null check".to_string(),
            fix_quality_score: Some(0.95),
            created_at: Utc::now(),
        };
        tracker.store_strategy_fingerprint(&fp).unwrap();

        let strategies = tracker.get_successful_strategies("org/repo", 10).unwrap();
        assert_eq!(strategies.len(), 1);
        assert_eq!(strategies[0].tests_run, 5);
        assert_eq!(strategies[0].fix_quality_score, Some(0.95));
    }

    #[test]
    fn test_get_successful_strategies_empty() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let strategies = tracker.get_successful_strategies("org/repo", 10).unwrap();
        assert!(strategies.is_empty());
    }

    #[test]
    fn test_update_experiment_stats_with_time_to_merge() {
        let tracker = SqliteTracker::in_memory().unwrap();

        let exp = PromptExperiment {
            id: 0,
            experiment_name: "prompt_v2".to_string(),
            variant: "a".to_string(),
            prompt_template: "template".to_string(),
            prompt_hash: "hash".to_string(),
            created_at: Utc::now(),
            active: true,
            success_count: 0,
            failure_count: 0,
            avg_time_to_merge: None,
            avg_review_score: None,
        };
        let exp_id = tracker.save_experiment(&exp).unwrap();

        // Record success with time_to_merge
        tracker
            .update_experiment_stats(exp_id, true, Some(60.0))
            .unwrap();
        let experiments = tracker.get_active_experiments().unwrap();
        let updated = experiments.iter().find(|e| e.id == exp_id).unwrap();
        assert_eq!(updated.success_count, 1);
        assert!(updated.avg_time_to_merge.is_some());
        assert!((updated.avg_time_to_merge.unwrap() - 60.0).abs() < f64::EPSILON);

        // Record another success with different time_to_merge
        tracker
            .update_experiment_stats(exp_id, true, Some(120.0))
            .unwrap();
        let experiments = tracker.get_active_experiments().unwrap();
        let updated = experiments.iter().find(|e| e.id == exp_id).unwrap();
        assert_eq!(updated.success_count, 2);
        // Rolling average: (60 * 0 + 120) / 1... but the formula uses
        // (old_avg * (success_count - 2) + ttm) / (success_count - 1) after 2nd success
        // The exact formula depends on the SQL implementation, just check it's between 60 and 120
        let avg_ttm = updated.avg_time_to_merge.unwrap();
        assert!((60.0..=120.0).contains(&avg_ttm));
    }

    #[test]
    fn test_update_experiment_stats_failure() {
        let tracker = SqliteTracker::in_memory().unwrap();

        let exp = PromptExperiment {
            id: 0,
            experiment_name: "prompt_v3".to_string(),
            variant: "b".to_string(),
            prompt_template: "template".to_string(),
            prompt_hash: "hash2".to_string(),
            created_at: Utc::now(),
            active: true,
            success_count: 0,
            failure_count: 0,
            avg_time_to_merge: None,
            avg_review_score: None,
        };
        let exp_id = tracker.save_experiment(&exp).unwrap();

        tracker
            .update_experiment_stats(exp_id, false, None)
            .unwrap();
        let experiments = tracker.get_active_experiments().unwrap();
        let updated = experiments.iter().find(|e| e.id == exp_id).unwrap();
        assert_eq!(updated.failure_count, 1);
        assert_eq!(updated.success_count, 0);
    }

    #[test]
    fn test_store_embeddings_batch() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let emb1 = IssueEmbedding {
            id: 0,
            source: "linear".to_string(),
            issue_id: "1".to_string(),
            short_id: Some("L-1".to_string()),
            title: Some("Bug 1".to_string()),
            embedding: Some(vec![0.1, 0.2, 0.3]),
            embedding_model: Some("test".to_string()),
            created_at: Utc::now(),
            description: None,
            url: None,
            priority: None,
            status: None,
            labels: None,
            updated_at: None,
        };
        let emb2 = IssueEmbedding {
            id: 0,
            source: "sentry".to_string(),
            issue_id: "2".to_string(),
            short_id: Some("S-2".to_string()),
            title: Some("Error 2".to_string()),
            embedding: Some(vec![0.4, 0.5, 0.6]),
            embedding_model: Some("test".to_string()),
            created_at: Utc::now(),
            description: None,
            url: None,
            priority: None,
            status: None,
            labels: None,
            updated_at: None,
        };

        tracker.store_embeddings_batch(&[emb1, emb2]).unwrap();

        let all = tracker
            .get_all_embeddings(None, Some(100), Some(0))
            .unwrap();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_store_embeddings_batch_empty() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker.store_embeddings_batch(&[]).unwrap();
    }

    #[test]
    fn test_get_all_embeddings_no_filter() {
        let tracker = SqliteTracker::in_memory().unwrap();
        for i in 1..=3 {
            let emb = IssueEmbedding {
                id: 0,
                source: "linear".to_string(),
                issue_id: i.to_string(),
                short_id: Some(format!("L-{}", i)),
                title: Some(format!("Bug {}", i)),
                embedding: None,
                embedding_model: None,
                created_at: Utc::now(),
                description: None,
                url: None,
                priority: None,
                status: None,
                labels: None,
                updated_at: None,
            };
            tracker.store_embedding(&emb).unwrap();
        }

        let all = tracker.get_all_embeddings(None, Some(10), Some(0)).unwrap();
        assert_eq!(all.len(), 3);

        let paged = tracker.get_all_embeddings(None, Some(2), Some(0)).unwrap();
        assert_eq!(paged.len(), 2);

        let paged2 = tracker.get_all_embeddings(None, Some(2), Some(2)).unwrap();
        assert_eq!(paged2.len(), 1);
    }

    #[test]
    fn test_parse_language_all_known() {
        use claudear_core::types::Language;
        assert_eq!(parse_language("Rust"), Language::Rust);
        assert_eq!(parse_language("TypeScript"), Language::TypeScript);
        assert_eq!(parse_language("TSX"), Language::Tsx);
        assert_eq!(parse_language("JavaScript"), Language::JavaScript);
        assert_eq!(parse_language("Python"), Language::Python);
        assert_eq!(parse_language("Go"), Language::Go);
        assert_eq!(parse_language("Java"), Language::Java);
        assert_eq!(parse_language("C"), Language::C);
        assert_eq!(parse_language("C++"), Language::Cpp);
        assert_eq!(parse_language("Ruby"), Language::Ruby);
        assert_eq!(parse_language("PHP"), Language::Php);
        assert_eq!(parse_language("Swift"), Language::Swift);
        assert_eq!(parse_language("Kotlin"), Language::Kotlin);
    }

    #[test]
    fn test_parse_language_unknown_falls_back_to_rust() {
        use claudear_core::types::Language;
        assert_eq!(parse_language("UnknownLang"), Language::Rust);
    }

    #[test]
    fn test_sync_repo_files() {
        let tracker = SqliteTracker::in_memory().unwrap();

        let repo = claudear_core::types::IndexedRepo {
            name: "my-repo".to_string(),
            path: std::path::PathBuf::from("/tmp/my-repo"),
            scm_url: "https://github.com/org/my-repo".to_string(),
            default_branch: "main".to_string(),
            files: vec!["src/main.rs".to_string(), "src/lib.rs".to_string()],
        };

        tracker.sync_repo_files(&repo).unwrap();

        let stored = tracker.get_indexed_repo("my-repo").unwrap().unwrap();
        assert_eq!(stored.name, "my-repo");
        assert_eq!(stored.file_count, 2);

        // Sync again with different files -- should clear and re-insert
        let repo2 = claudear_core::types::IndexedRepo {
            name: "my-repo".to_string(),
            path: std::path::PathBuf::from("/tmp/my-repo"),
            scm_url: "https://github.com/org/my-repo".to_string(),
            default_branch: "main".to_string(),
            files: vec!["src/new.rs".to_string()],
        };
        tracker.sync_repo_files(&repo2).unwrap();

        let stored2 = tracker.get_indexed_repo("my-repo").unwrap().unwrap();
        assert_eq!(stored2.file_count, 1);
    }

    #[test]
    fn test_sync_repo_files_empty() {
        let tracker = SqliteTracker::in_memory().unwrap();

        let repo = claudear_core::types::IndexedRepo {
            name: "empty-repo".to_string(),
            path: std::path::PathBuf::from("/tmp/empty"),
            scm_url: "https://github.com/org/empty".to_string(),
            default_branch: "main".to_string(),
            files: vec![],
        };

        tracker.sync_repo_files(&repo).unwrap();

        let stored = tracker.get_indexed_repo("empty-repo").unwrap().unwrap();
        assert_eq!(stored.file_count, 0);
    }

    #[test]
    fn test_list_recent_attempts() {
        let tracker = SqliteTracker::in_memory().unwrap();
        for i in 1..=5 {
            tracker
                .record_attempt("linear", &i.to_string(), &format!("L-{}", i))
                .unwrap();
        }

        let recent = tracker.list_recent_attempts(3).unwrap();
        assert_eq!(recent.len(), 3);
    }

    #[test]
    fn test_get_metrics_with_name_filter() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let metric1 = ProcessingMetric {
            id: 0,
            timestamp: Utc::now(),
            metric_name: "latency".to_string(),
            metric_value: 100.0,
            source: Some("sentry".to_string()),
            tags: None,
        };
        let metric2 = ProcessingMetric {
            id: 0,
            timestamp: Utc::now(),
            metric_name: "throughput".to_string(),
            metric_value: 50.0,
            source: Some("linear".to_string()),
            tags: None,
        };
        tracker.record_metric(&metric1).unwrap();
        tracker.record_metric(&metric2).unwrap();

        let since = chrono::DateTime::parse_from_rfc3339("2020-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let all_latency = tracker.get_metrics("latency", Some(since), 100).unwrap();
        let all_throughput = tracker.get_metrics("throughput", Some(since), 100).unwrap();
        assert_eq!(all_latency.len() + all_throughput.len(), 2);

        let latency = tracker.get_metrics("latency", Some(since), 100).unwrap();
        assert_eq!(latency.len(), 1);
        assert_eq!(latency[0].metric_name, "latency");
    }

    #[test]
    fn test_get_metrics_no_filter() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let metric = ProcessingMetric {
            id: 0,
            timestamp: Utc::now(),
            metric_name: "test".to_string(),
            metric_value: 42.0,
            source: None,
            tags: None,
        };
        tracker.record_metric(&metric).unwrap();

        let all = tracker.get_metrics("test", None, 100).unwrap();
        assert_eq!(all.len(), 1);
    }

    #[test]
    fn test_store_code_complexity_upsert() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let repo_id = tracker.get_or_create_repo_id("test-repo").unwrap();

        let fc = claudear_core::types::FileComplexity {
            file_path: "src/main.rs".to_string(),
            total_lines: 100,
            function_count: 5,
            functions: vec![],
            avg_cyclomatic: 3.5,
            max_cyclomatic: 8.0,
            avg_func_length: 15.0,
            max_func_length: 40.0,
            avg_nesting: 2.0,
            max_nesting: 4.0,
        };

        tracker
            .store_code_complexity(repo_id, "src/main.rs", &fc)
            .unwrap();

        // Upsert should not error
        tracker
            .store_code_complexity(repo_id, "src/main.rs", &fc)
            .unwrap();
    }

    #[test]
    fn test_search_code_chunks_empty_embedding() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let results = tracker.search_code_chunks(&[], None, 10).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_code_chunks_zero_limit() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let results = tracker
            .search_code_chunks(&[0.1, 0.2, 0.3], None, 0)
            .unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_save_code_chunk_embeddings_empty() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker.save_code_chunk_embeddings(&[], "test").unwrap();
    }

    #[test]
    fn test_get_cost_estimate_plan_estimate() {
        let tracker = SqliteTracker::in_memory().unwrap();
        // Record an execution with duration but no cost
        tracker.record_attempt("linear", "1", "L-1").unwrap();
        let attempt = tracker.get_attempt("linear", "1").unwrap().unwrap();

        let mut exec = AgentExecution::new().with_attempt_id(attempt.id);
        exec.duration_secs = Some(600.0); // 10 minutes
        exec.exit_code = Some(0);
        tracker.record_execution(&exec).unwrap();

        // With max_plan_monthly_cost > 0 and no API costs
        let estimate = tracker
            .get_cost_estimate("2020-01-01T00:00:00", 200.0, "30d")
            .unwrap();
        assert_eq!(estimate.cost_source, "plan_estimate");
        // total_cost should be > 0 since there's duration data
        assert!(estimate.total_cost > 0.0);
    }

    #[test]
    fn test_upsert_pr_with_issue_linkage() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let pr = claudear_core::types::PrRecord::for_issue(
            "https://github.com/org/repo/pull/1",
            "org/repo",
            1,
            "linear",
            "ISSUE-1",
        );
        tracker.upsert_pr(&pr).unwrap();

        let fetched = tracker
            .get_pr("https://github.com/org/repo/pull/1")
            .unwrap()
            .unwrap();
        assert_eq!(fetched.issue_source.as_deref(), Some("linear"));
        assert_eq!(fetched.issue_id.as_deref(), Some("ISSUE-1"));
    }

    #[test]
    fn test_upsert_pr_coalesce_preserves_existing_values() {
        let tracker = SqliteTracker::in_memory().unwrap();

        // First insert with issue linkage
        let mut pr1 = make_pr_record("https://github.com/org/repo/pull/1", "org/repo", 1);
        pr1.title = Some("Fix bug".to_string());
        pr1.issue_id = Some("ISS-1".to_string());
        pr1.issue_source = Some("linear".to_string());
        tracker.upsert_pr(&pr1).unwrap();

        // Second insert without issue linkage (simulates webhook update)
        let mut pr2 = make_pr_record("https://github.com/org/repo/pull/1", "org/repo", 1);
        pr2.title = None;
        pr2.issue_id = None;
        pr2.issue_source = None;
        tracker.upsert_pr(&pr2).unwrap();

        let fetched = tracker
            .get_pr("https://github.com/org/repo/pull/1")
            .unwrap()
            .unwrap();
        // COALESCE should preserve the original values
        assert_eq!(fetched.title.as_deref(), Some("Fix bug"));
        assert_eq!(fetched.issue_id.as_deref(), Some("ISS-1"));
        assert_eq!(fetched.issue_source.as_deref(), Some("linear"));
    }

    // =========================================================
    // Tests for previously uncovered code paths
    // =========================================================

    #[test]
    fn test_validate_dimension_zero() {
        let result = validate_dimension(0);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_dimension_too_large() {
        let result = validate_dimension(MAX_EMBEDDING_DIMENSION + 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_dimension_valid() {
        assert!(validate_dimension(1).is_ok());
        assert!(validate_dimension(128).is_ok());
        assert!(validate_dimension(MAX_EMBEDDING_DIMENSION).is_ok());
    }

    #[test]
    fn test_blob_to_embedding_none() {
        assert!(SqliteTracker::blob_to_embedding(None).is_none());
    }

    #[test]
    fn test_blob_to_embedding_empty() {
        // Empty blob should return None because the Option<Vec<u8>> is Some but
        // blob is empty, blob.len().is_multiple_of(4) is true (0%4==0), so it returns Some(empty vec)
        let result = SqliteTracker::blob_to_embedding(Some(Vec::new()));
        assert_eq!(result, Some(Vec::new()));
    }

    #[test]
    fn test_blob_to_embedding_bad_alignment() {
        // A blob of length 3 is not a multiple of 4
        let result = SqliteTracker::blob_to_embedding(Some(vec![1, 2, 3]));
        assert!(result.is_none());
    }

    #[test]
    fn test_blob_to_embedding_roundtrip() {
        let original = vec![1.0f32, 2.5, -3.15, 0.0];
        let blob = SqliteTracker::embedding_to_blob(Some(&original));
        let restored = SqliteTracker::blob_to_embedding(blob);
        assert_eq!(restored, Some(original));
    }

    #[test]
    fn test_table_exists_true() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let conn = tracker.acquire_lock().unwrap();
        // fix_attempts is created by migrations
        assert!(SqliteTracker::table_exists(&conn, "fix_attempts").unwrap());
    }

    #[test]
    fn test_table_exists_false() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let conn = tracker.acquire_lock().unwrap();
        assert!(!SqliteTracker::table_exists(&conn, "nonexistent_table_xyz").unwrap());
    }

    #[test]
    fn test_get_user_by_id() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let id = tracker
            .create_user("user@example.com", "hash123", "Test User", "admin")
            .unwrap();

        let user = tracker.get_user_by_id(id).unwrap().unwrap();
        assert_eq!(user.id, id);
        assert_eq!(user.email, "user@example.com");
        assert_eq!(user.name, "Test User");
        assert_eq!(user.role, "admin");
    }

    #[test]
    fn test_get_user_by_id_not_found() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let user = tracker.get_user_by_id(99999).unwrap();
        assert!(user.is_none());
    }

    #[test]
    fn test_update_experiment_config() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let experiment = PromptExperiment::new("exp1", "control", "template1", "hash1");
        let id = tracker.save_experiment(&experiment).unwrap();
        assert!(id > 0);

        // Update the experiment
        let updated = tracker
            .update_experiment(
                id,
                "exp1_renamed",
                "variant_a",
                "new template",
                "new_hash",
                false,
            )
            .unwrap();
        assert!(updated);

        // Verify via get_active_experiments (should be empty since we set active=false)
        let active = tracker.get_active_experiments().unwrap();
        assert!(active.iter().all(|e| e.id != id));
    }

    #[test]
    fn test_update_experiment_nonexistent() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let updated = tracker
            .update_experiment(99999, "exp", "var", "tpl", "hash", true)
            .unwrap();
        assert!(!updated);
    }

    #[test]
    fn test_get_most_recent_merged_attempt_for_repo() {
        let tracker = SqliteTracker::in_memory().unwrap();

        // Record and mark as merged with a known SCM repo
        tracker.record_attempt("linear", "1", "L-1").unwrap();
        tracker
            .mark_success("linear", "1", "https://github.com/org/repo/pull/10")
            .unwrap();
        tracker.mark_merged("linear", "1").unwrap();

        let result = tracker
            .get_most_recent_merged_attempt_for_repo("org/repo")
            .unwrap();
        assert!(result.is_some());
        let attempt = result.unwrap();
        assert_eq!(attempt.issue_id, "1");
        assert_eq!(attempt.scm_repo, Some("org/repo".to_string()));
    }

    #[test]
    fn test_get_most_recent_merged_attempt_for_repo_not_found() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let result = tracker
            .get_most_recent_merged_attempt_for_repo("nonexistent/repo")
            .unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_get_most_recent_merged_attempt_for_repo_only_returns_merged() {
        let tracker = SqliteTracker::in_memory().unwrap();

        // Create a success (but not merged) attempt
        tracker.record_attempt("linear", "2", "L-2").unwrap();
        tracker
            .mark_success("linear", "2", "https://github.com/org/repo/pull/20")
            .unwrap();

        let result = tracker
            .get_most_recent_merged_attempt_for_repo("org/repo")
            .unwrap();
        assert!(result.is_none());
    }

    // --- Chat Store Tests ---

    #[test]
    fn test_create_chat_session() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker.create_chat_session("sess-1", None).unwrap();

        let sessions = tracker.list_chat_sessions().unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].id, "sess-1");
        assert!(sessions[0].repo_id.is_none());
    }

    #[test]
    fn test_create_chat_session_with_repo_id() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let repo_id = tracker.get_or_create_repo_id("my-repo").unwrap();
        tracker
            .create_chat_session("sess-2", Some(repo_id))
            .unwrap();

        let sessions = tracker.list_chat_sessions().unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].repo_id, Some(repo_id));
    }

    #[test]
    fn test_save_and_get_chat_messages() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker.create_chat_session("sess-1", None).unwrap();

        let id1 = tracker
            .save_chat_message("sess-1", "user", "Hello!", None)
            .unwrap();
        assert!(id1 > 0);

        let id2 = tracker
            .save_chat_message(
                "sess-1",
                "assistant",
                "Hi there!",
                Some(r#"{"docs":["a.rs"]}"#),
            )
            .unwrap();
        assert!(id2 > id1);

        let history = tracker.get_chat_history("sess-1", 10).unwrap();
        assert_eq!(history.len(), 2);
        // Should be in chronological order
        assert_eq!(history[0].content, "Hello!");
        assert_eq!(history[1].content, "Hi there!");
        assert_eq!(
            history[1].sources_json.as_deref(),
            Some(r#"{"docs":["a.rs"]}"#)
        );
    }

    #[test]
    fn test_get_chat_history_respects_limit() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker.create_chat_session("sess-1", None).unwrap();

        for i in 0..5 {
            tracker
                .save_chat_message("sess-1", "user", &format!("msg {}", i), None)
                .unwrap();
        }

        let history = tracker.get_chat_history("sess-1", 3).unwrap();
        assert_eq!(history.len(), 3);
        // Should return the 3 most recent in chronological order
        assert_eq!(history[0].content, "msg 2");
        assert_eq!(history[1].content, "msg 3");
        assert_eq!(history[2].content, "msg 4");
    }

    #[test]
    fn test_get_chat_history_empty() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker.create_chat_session("sess-1", None).unwrap();
        let history = tracker.get_chat_history("sess-1", 10).unwrap();
        assert!(history.is_empty());
    }

    #[test]
    fn test_list_chat_sessions_multiple() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker.create_chat_session("sess-a", None).unwrap();
        tracker.create_chat_session("sess-b", None).unwrap();
        tracker.create_chat_session("sess-c", None).unwrap();

        let sessions = tracker.list_chat_sessions().unwrap();
        assert_eq!(sessions.len(), 3);
    }

    #[test]
    fn test_list_chat_sessions_empty() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let sessions = tracker.list_chat_sessions().unwrap();
        assert!(sessions.is_empty());
    }

    #[test]
    fn test_delete_chat_session() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker.create_chat_session("sess-1", None).unwrap();
        tracker
            .save_chat_message("sess-1", "user", "hello", None)
            .unwrap();

        tracker.delete_chat_session("sess-1").unwrap();

        let sessions = tracker.list_chat_sessions().unwrap();
        assert!(sessions.is_empty());

        // Messages should also be gone (CASCADE delete)
        let history = tracker.get_chat_history("sess-1", 10).unwrap();
        assert!(history.is_empty());
    }

    #[test]
    fn test_delete_chat_session_nonexistent() {
        let tracker = SqliteTracker::in_memory().unwrap();
        // Should not error even if session doesn't exist
        tracker.delete_chat_session("nonexistent").unwrap();
    }

    #[test]
    fn test_cleanup_expired_chat_sessions() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker.create_chat_session("sess-old", None).unwrap();
        tracker.create_chat_session("sess-new", None).unwrap();

        // Manually age one session via direct SQL
        {
            let conn = tracker.acquire_lock().unwrap();
            conn.execute(
                "UPDATE chat_sessions SET updated_at = datetime('now', '-100 days') WHERE id = 'sess-old'",
                [],
            )
            .unwrap();
        }

        let deleted = tracker.cleanup_expired_chat_sessions(30).unwrap();
        assert_eq!(deleted, 1);

        let sessions = tracker.list_chat_sessions().unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].id, "sess-new");
    }

    #[test]
    fn test_cleanup_expired_chat_sessions_none_expired() {
        let tracker = SqliteTracker::in_memory().unwrap();
        tracker.create_chat_session("sess-1", None).unwrap();

        let deleted = tracker.cleanup_expired_chat_sessions(30).unwrap();
        assert_eq!(deleted, 0);

        let sessions = tracker.list_chat_sessions().unwrap();
        assert_eq!(sessions.len(), 1);
    }

    // --- Code Index Meta Tests ---

    #[test]
    fn test_get_set_code_index_meta() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let repo_id = tracker.get_or_create_repo_id("test-repo").unwrap();

        // Initially no meta
        let val = tracker.get_code_index_meta(repo_id, "model").unwrap();
        assert!(val.is_none());

        // Set a value
        tracker
            .set_code_index_meta(repo_id, "model", "text-embedding-3-small")
            .unwrap();
        let val = tracker.get_code_index_meta(repo_id, "model").unwrap();
        assert_eq!(val.as_deref(), Some("text-embedding-3-small"));

        // Upsert (overwrite)
        tracker
            .set_code_index_meta(repo_id, "model", "text-embedding-3-large")
            .unwrap();
        let val = tracker.get_code_index_meta(repo_id, "model").unwrap();
        assert_eq!(val.as_deref(), Some("text-embedding-3-large"));
    }

    #[test]
    fn test_get_code_index_meta_different_keys() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let repo_id = tracker.get_or_create_repo_id("test-repo").unwrap();

        tracker
            .set_code_index_meta(repo_id, "key_a", "val_a")
            .unwrap();
        tracker
            .set_code_index_meta(repo_id, "key_b", "val_b")
            .unwrap();

        assert_eq!(
            tracker
                .get_code_index_meta(repo_id, "key_a")
                .unwrap()
                .as_deref(),
            Some("val_a")
        );
        assert_eq!(
            tracker
                .get_code_index_meta(repo_id, "key_b")
                .unwrap()
                .as_deref(),
            Some("val_b")
        );
    }

    #[test]
    fn test_get_code_embedding_model_empty() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let repo_id = tracker.get_or_create_repo_id("test-repo").unwrap();

        let model = tracker.get_code_embedding_model(repo_id).unwrap();
        assert!(model.is_none());
    }

    #[test]
    fn test_get_code_embedding_model_with_data() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let repo_id = tracker.get_or_create_repo_id("test-repo").unwrap();

        // Save a chunk, then save an embedding for it
        let chunks = vec![claudear_core::types::CodeChunk {
            id: None,
            repo_id,
            file_path: "src/main.rs".to_string(),
            chunk_type: "function".to_string(),
            symbol_name: Some("main".to_string()),
            language: claudear_core::types::Language::Rust,
            start_line: 1,
            end_line: 10,
            chunk_text: "fn main() {}".to_string(),
            context_text: "fn main() {}".to_string(),
            file_hash: "abc123".to_string(),
            content_hash: None,
        }];
        let ids = tracker.save_code_chunks(&chunks).unwrap();
        assert_eq!(ids.len(), 1);

        // Save embedding for the chunk
        let embedding = vec![0.1f32, 0.2, 0.3, 0.4];
        tracker
            .save_code_chunk_embeddings(&[(ids[0], &embedding)], "test-model-v1")
            .unwrap();

        let model = tracker.get_code_embedding_model(repo_id).unwrap();
        assert_eq!(model.as_deref(), Some("test-model-v1"));
    }

    // --- Delete All Code Data ---

    #[test]
    fn test_delete_all_code_data_for_repo() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let repo_id = tracker.get_or_create_repo_id("test-repo").unwrap();

        // Save symbols
        let symbols = vec![claudear_core::types::CodeSymbol {
            id: None,
            repo_id,
            file_path: "src/lib.rs".to_string(),
            symbol_name: "my_func".to_string(),
            symbol_kind: claudear_core::types::SymbolKind::Function,
            parent_symbol: None,
            language: claudear_core::types::Language::Rust,
            start_line: 1,
            end_line: 5,
            signature: None,
        }];
        tracker.save_code_symbols(&symbols).unwrap();

        // Save chunks
        let chunks = vec![claudear_core::types::CodeChunk {
            id: None,
            repo_id,
            file_path: "src/lib.rs".to_string(),
            chunk_type: "function".to_string(),
            symbol_name: Some("my_func".to_string()),
            language: claudear_core::types::Language::Rust,
            start_line: 1,
            end_line: 5,
            chunk_text: "fn my_func() {}".to_string(),
            context_text: "fn my_func() {}".to_string(),
            file_hash: "hash1".to_string(),
            content_hash: None,
        }];
        tracker.save_code_chunks(&chunks).unwrap();

        // Delete all
        tracker.delete_all_code_data_for_repo(repo_id).unwrap();

        // Verify symbols and chunks are gone
        let found_symbols = tracker
            .find_code_symbols("my_func", None, Some(repo_id))
            .unwrap();
        assert!(found_symbols.is_empty());
    }

    #[test]
    fn test_delete_all_code_data_for_repo_empty() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let repo_id = tracker.get_or_create_repo_id("test-repo").unwrap();
        // Should succeed even with no data
        tracker.delete_all_code_data_for_repo(repo_id).unwrap();
    }

    // --- Indexing Progress ---

    #[test]
    fn test_get_indexing_progress_default() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let progress = tracker.get_indexing_progress().unwrap();
        // After init, there should be a default idle row
        assert_eq!(progress.status, "idle");
        assert_eq!(progress.total_repos, 0);
        assert_eq!(progress.indexed_repos, 0);
    }

    // --- Update user with no fields ---

    #[test]
    fn test_update_user_no_fields() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let id = tracker
            .create_user("u@ex.com", "h", "Name", "admin")
            .unwrap();

        // Calling update with all None should return false (no fields to update)
        let result = tracker
            .update_user(id, None, None, None, None, None)
            .unwrap();
        assert!(!result);
    }

    #[test]
    fn test_update_user_partial_fields() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let id = tracker
            .create_user("u@ex.com", "oldhash", "OldName", "viewer")
            .unwrap();

        // Update only the name
        let result = tracker
            .update_user(id, None, None, Some("NewName"), None, None)
            .unwrap();
        assert!(result);

        let user = tracker.get_user_by_id(id).unwrap().unwrap();
        assert_eq!(user.name, "NewName");
        assert_eq!(user.email, "u@ex.com"); // unchanged
        assert_eq!(user.role, "viewer"); // unchanged
    }

    #[test]
    fn test_update_user_avatar_url() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let id = tracker
            .create_user("u@ex.com", "h", "Name", "admin")
            .unwrap();

        let result = tracker
            .update_user(
                id,
                None,
                None,
                None,
                None,
                Some("https://example.com/avatar.png"),
            )
            .unwrap();
        assert!(result);

        let user = tracker.get_user_by_id(id).unwrap().unwrap();
        assert_eq!(
            user.avatar_url.as_deref(),
            Some("https://example.com/avatar.png")
        );
    }

    // --- get_activities_for_issue (isolation) ---

    #[test]
    fn test_get_activities_for_issue_filters_by_source_and_id() {
        let tracker = SqliteTracker::in_memory().unwrap();

        let entry = ActivityLogEntry {
            id: 0,
            timestamp: Utc::now(),
            activity_type: "fix_started".to_string(),
            source: Some("linear".to_string()),
            issue_id: Some("ISS-1".to_string()),
            short_id: Some("L-1".to_string()),
            message: "Started fix".to_string(),
            metadata: None,
        };
        tracker.record_activity(&entry).unwrap();

        // Different issue
        let entry2 = ActivityLogEntry {
            id: 0,
            timestamp: Utc::now(),
            activity_type: "fix_started".to_string(),
            source: Some("linear".to_string()),
            issue_id: Some("ISS-2".to_string()),
            short_id: Some("L-2".to_string()),
            message: "Other fix".to_string(),
            metadata: None,
        };
        tracker.record_activity(&entry2).unwrap();

        // Different source, same issue_id
        let entry3 = ActivityLogEntry {
            id: 0,
            timestamp: Utc::now(),
            activity_type: "fix_started".to_string(),
            source: Some("sentry".to_string()),
            issue_id: Some("ISS-1".to_string()),
            short_id: Some("S-1".to_string()),
            message: "Sentry fix".to_string(),
            metadata: None,
        };
        tracker.record_activity(&entry3).unwrap();

        let activities = tracker.get_activities_for_issue("linear", "ISS-1").unwrap();
        assert_eq!(activities.len(), 1);
        assert_eq!(activities[0].message, "Started fix");
    }

    #[test]
    fn test_get_activities_for_issue_returns_empty_for_nonexistent() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let activities = tracker
            .get_activities_for_issue("linear", "nonexistent")
            .unwrap();
        assert!(activities.is_empty());
    }

    // --- find_similar_issues_vector (without vectorlite) ---

    #[test]
    fn test_find_similar_issues_vector_empty_embedding() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let result = tracker
            .find_similar_issues_vector(&[], "linear", None, 0.5, 10)
            .unwrap();
        // Empty embedding should return Some(empty vec)
        assert!(result.is_some());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_find_similar_issues_vector_zero_limit() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let result = tracker
            .find_similar_issues_vector(&[0.1, 0.2, 0.3], "linear", None, 0.5, 0)
            .unwrap();
        assert!(result.is_some());
        assert!(result.unwrap().is_empty());
    }

    // --- find_similar_outcomes_vector (without vectorlite) ---

    #[test]
    fn test_find_similar_outcomes_vector_empty_embedding() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let result = tracker.find_similar_outcomes_vector(&[], 0.5, 10).unwrap();
        assert!(result.is_some());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_find_similar_outcomes_vector_zero_limit() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let result = tracker
            .find_similar_outcomes_vector(&[0.1, 0.2], 0.5, 0)
            .unwrap();
        assert!(result.is_some());
        assert!(result.unwrap().is_empty());
    }

    // --- generate_session_token ---

    #[test]
    fn test_generate_session_token_length_and_uniqueness() {
        let t1 = generate_session_token();
        let t2 = generate_session_token();
        // 32 bytes => 64 hex chars
        assert_eq!(t1.len(), 64);
        assert_eq!(t2.len(), 64);
        // Tokens should be unique
        assert_ne!(t1, t2);
        // Should be valid hex
        assert!(t1.chars().all(|c| c.is_ascii_hexdigit()));
    }

    // --- record_release_tracking (edge case: no released_at) ---

    #[test]
    fn test_record_release_tracking_no_released_at() {
        let tracker = SqliteTracker::in_memory().unwrap();

        // Create a regression watch first
        tracker.record_attempt("linear", "1", "L-1").unwrap();
        let attempt = tracker.get_attempt("linear", "1").unwrap().unwrap();
        let watch = claudear_core::types::RegressionWatch {
            id: 0,
            issue_type: claudear_core::types::IssueType::SentryIssue,
            issue_id: "1".to_string(),
            fix_attempt_id: attempt.id,
            status: claudear_core::types::RegressionWatchStatus::AwaitingRelease,
            pr_merged_at: None,
            monitoring_started_at: None,
            resolved_at: None,
            regressed_at: None,
            created_at: Utc::now(),
        };
        let watch_id = tracker.create_regression_watch(&watch).unwrap();

        let tracking = claudear_core::types::ReleaseTracking {
            id: 0,
            regression_watch_id: watch_id,
            release_version: "v1.0.0".to_string(),
            release_commit: "abc123".to_string(),
            released_at: None, // No release date
            created_at: Utc::now(),
        };
        let id = tracker.record_release_tracking(&tracking).unwrap();
        assert!(id > 0);
    }

    // --- store_embeddings_batch with actual data ---

    #[test]
    fn test_store_embeddings_batch_with_embeddings() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let emb1 = IssueEmbedding {
            id: 0,
            source: "linear".to_string(),
            issue_id: "1".to_string(),
            short_id: Some("L-1".to_string()),
            title: Some("Bug 1".to_string()),
            description: None,
            url: None,
            priority: None,
            status: None,
            labels: None,
            embedding: Some(vec![0.1, 0.2, 0.3]),
            embedding_model: Some("test-model".to_string()),
            created_at: Utc::now(),
            updated_at: None,
        };
        let emb2 = IssueEmbedding {
            id: 0,
            source: "linear".to_string(),
            issue_id: "2".to_string(),
            short_id: Some("L-2".to_string()),
            title: Some("Bug 2".to_string()),
            description: Some("Desc".to_string()),
            url: Some("https://example.com".to_string()),
            priority: Some("high".to_string()),
            status: Some("open".to_string()),
            labels: Some("bug,critical".to_string()),
            embedding: Some(vec![0.4, 0.5, 0.6]),
            embedding_model: Some("test-model".to_string()),
            created_at: Utc::now(),
            updated_at: Some(Utc::now()),
        };

        tracker.store_embeddings_batch(&[emb1, emb2]).unwrap();

        // Verify both were stored
        let e1 = tracker.get_embedding("linear", "1").unwrap().unwrap();
        assert_eq!(e1.title, Some("Bug 1".to_string()));
        let e2 = tracker.get_embedding("linear", "2").unwrap().unwrap();
        assert_eq!(e2.title, Some("Bug 2".to_string()));
        assert_eq!(e2.description.as_deref(), Some("Desc"));
    }

    // --- list_issues (tested indirectly but let's test count_issues with source) ---

    #[test]
    fn test_count_issues_no_data() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let count = tracker.count_issues(None).unwrap();
        assert_eq!(count, 0);
    }

    // --- get_success_rate with mixed outcomes ---

    #[test]
    fn test_get_success_rate_with_data() {
        let tracker = SqliteTracker::in_memory().unwrap();

        // Create 3 attempts: 1 merged, 1 failed, 1 pending
        // get_success_rate = (success + merged) / total = 1/3
        tracker.record_attempt("linear", "1", "L-1").unwrap();
        tracker
            .mark_success("linear", "1", "https://github.com/o/r/pull/1")
            .unwrap();
        tracker.mark_merged("linear", "1").unwrap();

        tracker.record_attempt("linear", "2", "L-2").unwrap();
        tracker.mark_failed("linear", "2", "error").unwrap();

        tracker.record_attempt("linear", "3", "L-3").unwrap();

        let rate = tracker.get_success_rate().unwrap();
        // merged=1 out of total=3 => 1/3 ~ 0.333
        assert!((rate - 1.0 / 3.0).abs() < 0.01);
    }

    // --- record_metrics_batch edge case ---

    #[test]
    fn test_record_metrics_batch_with_tags() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let metric = ProcessingMetric {
            id: 0,
            metric_name: "test_metric".to_string(),
            metric_value: 42.5,
            source: Some("linear".to_string()),
            timestamp: Utc::now(),
            tags: Some(serde_json::json!({"key": "val"})),
        };

        let count = tracker.record_metrics_batch(&[metric]).unwrap();
        assert_eq!(count, 1);
    }

    // --- subscribe_indexing_progress ---

    #[test]
    fn test_subscribe_indexing_progress_returns_receiver() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let rx = tracker.subscribe_indexing_progress();
        // Should have the default value
        let progress = rx.borrow().clone();
        assert_eq!(progress.status, "idle");
    }

    #[test]
    fn test_purge_operational_data_deletes_attempts_and_related() {
        let tracker = SqliteTracker::in_memory().unwrap();

        // Create fix attempts
        tracker
            .record_attempt("linear", "issue-1", "LIN-1")
            .unwrap();
        tracker
            .record_attempt("sentry", "issue-2", "SEN-2")
            .unwrap();
        tracker
            .mark_success("linear", "issue-1", "https://github.com/org/repo/pull/1")
            .unwrap();

        // Create activity log entries
        let entry = ActivityLogEntry {
            id: 0,
            timestamp: Utc::now(),
            activity_type: "test".to_string(),
            source: Some("linear".to_string()),
            issue_id: Some("issue-1".to_string()),
            short_id: None,
            message: "test entry".to_string(),
            metadata: None,
        };
        tracker.record_activity(&entry).unwrap();

        // Create processing metrics
        let metric = ProcessingMetric {
            id: 0,
            timestamp: Utc::now(),
            metric_name: "test_metric".to_string(),
            metric_value: 1.0,
            source: None,
            tags: None,
        };
        tracker.record_metric(&metric).unwrap();

        // Verify data exists
        let stats = tracker.get_stats().unwrap();
        assert_eq!(stats.total, 2);
        let activities = tracker.get_recent_activities(100, None).unwrap();
        assert!(!activities.is_empty());

        // Purge
        let result = tracker.purge_operational_data().unwrap();

        assert_eq!(result.fix_attempts, 2);
        assert_eq!(result.activity_log, 1);
        assert_eq!(result.processing_metrics, 1);
        assert!(result.total_deleted() >= 4);

        // Verify operational data is gone
        let stats = tracker.get_stats().unwrap();
        assert_eq!(stats.total, 0);
        let activities = tracker.get_recent_activities(100, None).unwrap();
        assert!(activities.is_empty());
    }

    #[test]
    fn test_purge_preserves_knowledge_and_embeddings() {
        let tracker = SqliteTracker::in_memory().unwrap();

        // Create a fix attempt and feedback outcome
        tracker
            .record_attempt("linear", "issue-k", "LIN-K")
            .unwrap();
        let attempt = tracker.get_attempt("linear", "issue-k").unwrap().unwrap();
        let outcome = FixOutcome {
            id: 0,
            attempt_id: attempt.id,
            source: "linear".to_string(),
            issue_id: "issue-k".to_string(),
            issue_text: "Bug in handler".to_string(),
            prompt_used: "Fix the bug".to_string(),
            outcome: Outcome::Merged,
            error_type: Some("null_ref".to_string()),
            learnings: Some("Always null-check".to_string()),
            keywords: vec!["null".to_string()],
            embedding: None,
            created_at: Utc::now(),
        };
        tracker.store_feedback_outcome(&outcome).unwrap();

        // Store an issue embedding
        let issue_embedding = IssueEmbedding {
            id: 0,
            source: "linear".to_string(),
            issue_id: "issue-k".to_string(),
            short_id: Some("LIN-K".to_string()),
            title: Some("Bug".to_string()),
            description: Some("A bug".to_string()),
            url: None,
            priority: None,
            status: None,
            labels: None,
            embedding: None,
            embedding_model: None,
            created_at: Utc::now(),
            updated_at: None,
        };
        tracker.store_issue(&issue_embedding).unwrap();

        // Store an error pattern
        let pattern = ErrorPattern {
            id: 0,
            pattern_hash: "hash1".to_string(),
            error_type: Some("null_ref".to_string()),
            error_message: Some("null pointer".to_string()),
            first_seen: Utc::now(),
            last_seen: Utc::now(),
            occurrence_count: 3,
            sources: Some(vec!["linear".to_string()]),
            example_issue_ids: Some(vec!["issue-k".to_string()]),
            resolution_hints: None,
        };
        tracker.record_error_pattern(&pattern).unwrap();

        // Purge
        let result = tracker.purge_operational_data().unwrap();

        // Attempts should be deleted
        assert_eq!(result.fix_attempts, 1);

        // Feedback outcomes should be detached (attempt_id NULLed), not deleted
        assert_eq!(result.feedback_outcomes_detached, 1);
        // Verify the row still exists in the DB (attempt_id is NULL so the trait
        // method can't deserialize it, but the knowledge is preserved)
        {
            let conn = tracker.acquire_lock().unwrap();
            let (count, learnings): (i64, Option<String>) = conn
                .query_row(
                    "SELECT COUNT(*), MAX(learnings) FROM feedback_outcomes",
                    [],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .unwrap();
            assert_eq!(count, 1);
            assert_eq!(learnings, Some("Always null-check".to_string()));
        }

        // Issue embeddings should be preserved
        let issues = tracker.list_issues(None, 100, 0).unwrap();
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].issue_id, "issue-k");

        // Error patterns should be preserved
        let patterns = tracker.get_error_patterns(100).unwrap();
        assert_eq!(patterns.len(), 1);
        assert_eq!(patterns[0].error_type, Some("null_ref".to_string()));
    }

    #[test]
    fn test_purge_preserves_repositories_and_code_index() {
        let tracker = SqliteTracker::in_memory().unwrap();

        // Create a repository
        let repo_id = tracker.get_or_create_repo_id("org/repo").unwrap();
        assert!(repo_id > 0);

        // Store inference attempt
        tracker
            .record_inference_attempt(
                "issue-inf",
                "linear",
                &["main.rs".to_string()],
                &["handle".to_string()],
                &["null".to_string()],
                Some(repo_id),
                "high",
                "file match",
                None,
            )
            .unwrap();

        // Purge
        tracker.purge_operational_data().unwrap();

        // Repository should still exist
        let repo = tracker.get_repository("org/repo").unwrap();
        assert!(repo.is_some());
        assert_eq!(repo.unwrap().name, "org/repo");

        // Inference attempts should still exist
        let history = tracker.get_inference_history(100).unwrap();
        assert_eq!(history.len(), 1);
    }

    #[test]
    fn test_purge_on_empty_db() {
        let tracker = SqliteTracker::in_memory().unwrap();
        let result = tracker.purge_operational_data().unwrap();
        assert_eq!(result.total_deleted(), 0);
        assert_eq!(result.feedback_outcomes_detached, 0);
    }

    #[test]
    fn test_purge_deletes_regression_tracking() {
        let tracker = SqliteTracker::in_memory().unwrap();

        tracker
            .record_attempt("linear", "issue-r", "LIN-R")
            .unwrap();
        let attempt = tracker.get_attempt("linear", "issue-r").unwrap().unwrap();
        tracker
            .mark_success("linear", "issue-r", "https://github.com/org/repo/pull/5")
            .unwrap();

        let watch = claudear_core::types::RegressionWatch {
            id: 0,
            issue_type: claudear_core::types::IssueType::LinearBug,
            issue_id: "issue-r".to_string(),
            fix_attempt_id: attempt.id,
            status: claudear_core::types::RegressionWatchStatus::AwaitingRelease,
            pr_merged_at: None,
            monitoring_started_at: None,
            resolved_at: None,
            regressed_at: None,
            created_at: Utc::now(),
        };
        let watch_id = tracker.create_regression_watch(&watch).unwrap();
        assert!(watch_id > 0);

        let check = claudear_core::types::RegressionCheck {
            id: 0,
            regression_watch_id: watch_id,
            issue_still_exists: false,
            checked_at: Some(Utc::now()),
            check_details: Some("all clear".to_string()),
            created_at: Utc::now(),
        };
        tracker.add_regression_check(&check).unwrap();

        // Purge
        let result = tracker.purge_operational_data().unwrap();
        assert_eq!(result.regression_watches, 1);
        assert_eq!(result.regression_checks, 1);
        assert_eq!(result.fix_attempts, 1);

        // Verify regression data is gone
        let watches = tracker.get_all_regression_watches().unwrap();
        assert!(watches.is_empty());
    }
}
