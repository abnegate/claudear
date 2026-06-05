-- Records runs of the action pipeline (verify verdicts, reply posts) for
-- queryability and dashboard analytics. Resolve actions continue to use the
-- fix_attempts table; action transitions also land in activity_log.
CREATE TABLE IF NOT EXISTS action_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    issue_id TEXT NOT NULL,
    short_id TEXT NOT NULL,
    -- "reply" | "verify" | "resolve"
    action_kind TEXT NOT NULL,
    -- action-specific status: verify -> reproduced/not_reproduced; reply -> answer/need_repro/fix_shipped
    status TEXT NOT NULL,
    detail TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_action_runs_issue ON action_runs(source, issue_id);
CREATE INDEX IF NOT EXISTS idx_action_runs_kind ON action_runs(action_kind, created_at);
