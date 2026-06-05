//! Minimal embedded migration runner for SQLite.
//!
//! SQL files are included at compile time from the `migrations/` directory.
//! Each file must be named `V<N>__<description>.sql` (e.g. `V1__initial_schema.sql`).
//! Migrations are applied in version order and tracked in a `schema_migrations` table.

use rusqlite::Connection;

/// A single compiled-in migration.
struct Migration {
    version: u32,
    name: &'static str,
    sql: &'static str,
}

/// All migrations, embedded at compile time.
/// Add new entries here when adding migration files.
const MIGRATIONS: &[Migration] = &[
    Migration {
        version: 1,
        name: "initial_schema",
        sql: include_str!("../../../migrations/V1__initial_schema.sql"),
    },
    Migration {
        version: 2,
        name: "add_session_last_active",
        sql: include_str!("../../../migrations/V2__add_session_last_active.sql"),
    },
    Migration {
        version: 3,
        name: "action_runs",
        sql: include_str!("../../../migrations/V3__action_runs.sql"),
    },
];

/// Run all pending migrations against the given connection.
///
/// Creates the `schema_migrations` tracking table if it doesn't exist,
/// then applies any migrations whose version exceeds the current maximum.
pub fn run(conn: &Connection) -> Result<(), String> {
    // Create tracking table
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            applied_at TEXT NOT NULL DEFAULT (datetime('now'))
        );",
    )
    .map_err(|e| format!("Failed to create schema_migrations table: {e}"))?;

    // Get current version
    let current: u32 = conn
        .query_row(
            "SELECT COALESCE(MAX(version), 0) FROM schema_migrations",
            [],
            |row| row.get(0),
        )
        .map_err(|e| format!("Failed to query schema version: {e}"))?;

    // Apply pending migrations in order
    for m in MIGRATIONS {
        if m.version <= current {
            continue;
        }

        conn.execute_batch(m.sql)
            .map_err(|e| format!("Migration V{}_{} failed: {e}", m.version, m.name))?;

        conn.execute(
            "INSERT INTO schema_migrations (version, name) VALUES (?1, ?2)",
            rusqlite::params![m.version, m.name],
        )
        .map_err(|e| format!("Failed to record migration V{}: {e}", m.version))?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_migrations_apply_to_fresh_db() {
        let conn = Connection::open_in_memory().unwrap();
        run(&conn).unwrap();

        // Verify tracking table
        let version: u32 = conn
            .query_row("SELECT MAX(version) FROM schema_migrations", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(version, 3);

        // Verify a table from V1 exists
        let count: u32 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='fix_attempts'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_migrations_are_idempotent() {
        let conn = Connection::open_in_memory().unwrap();
        run(&conn).unwrap();
        // Running again should be a no-op
        run(&conn).unwrap();

        let version: u32 = conn
            .query_row("SELECT MAX(version) FROM schema_migrations", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(version, 3);
    }

    #[test]
    fn test_all_tables_created() {
        let conn = Connection::open_in_memory().unwrap();
        run(&conn).unwrap();

        let expected_tables = [
            "fix_attempts",
            "feedback_outcomes",
            "pr_review_states",
            "repositories",
            "repository_dependencies",
            "activity_log",
            "claude_executions",
            "pr_reviews",
            "pr_review_comments",
            "issues",
            "error_patterns",
            "processing_metrics",
            "prompt_experiments",
            "similar_issues",
            "qa_knowledge",
            "qa_usage",
            "question_channel_cursor",
            "repo_files",
            "inference_attempts",
            "prs",
            "regression_watches",
            "release_tracking",
            "regression_checks",
            "users",
            "sessions",
            "webhook_deliveries",
            "diff_analyses",
            "promoted_instructions",
            "repo_knowledge",
            "review_patterns",
            "strategy_fingerprints",
            "issue_clusters",
            "issue_cluster_members",
            "content_clusters",
            "severity_scores",
            "suppression_log",
            "code_symbols",
            "code_chunks",
            "code_chunk_embeddings",
            "code_index_metadata",
            "indexing_progress",
            "cross_repo_correlations",
            "code_complexity",
            "eval_snapshots",
            "eval_deltas",
            "chat_sessions",
            "chat_messages",
            "action_runs",
        ];

        for table in expected_tables {
            let count: u32 = conn
                .query_row(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?1",
                    [table],
                    |row| row.get(0),
                )
                .unwrap();
            assert_eq!(count, 1, "Table '{}' should exist", table);
        }
    }
}
