-- Operator-authored agent instructions, injected into the agent's context on
-- every run so it knows a repo's role and any cross-repo relationships (e.g.
-- "this is the generated SDK; edit the sdk-generator repo instead").
--
-- Scoped either global (repo IS NULL) or per-repo (repo = 'org/name'). Global
-- and per-repo text are concatenated at resolve time, global first. This is
-- prompt-string-only; no file is written into the working tree, so a repo's own
-- AGENTS.md / CLAUDE.md is never overwritten.
CREATE TABLE IF NOT EXISTS agent_instructions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scope TEXT NOT NULL,            -- 'global' or 'repo'
    repo TEXT,                      -- NULL for global; 'org/name' for repo scope
    instruction_text TEXT NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1,
    updated_by TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- One row per scope. Global is a singleton; each repo has at most one row.
-- IFNULL keeps the single global row unique despite the NULL repo.
CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_instructions_scope_repo
    ON agent_instructions(scope, IFNULL(repo, ''));
