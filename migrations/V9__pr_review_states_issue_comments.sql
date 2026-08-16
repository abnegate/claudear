-- Track PR conversation (issue) comments separately from inline review comments.
-- GitHub delivers plain "@claudear ..." PR comments via the issues/{n}/comments
-- endpoint, which uses a distinct comment id space from inline review comments,
-- so the review watcher advances an independent cursor for them.
ALTER TABLE pr_review_states ADD COLUMN last_issue_comment_id INTEGER;
ALTER TABLE pr_review_states ADD COLUMN last_issue_comment_time TEXT;

-- Make PR review-comment processing at-least-once, and namespace the comment id.
--
-- The pr_review_comments ledger becomes the authority for "what still needs
-- action", so a crash or downstream failure between detecting a comment and
-- acting on it no longer drops it: unhandled rows are re-surfaced on the next
-- poll regardless of the polling cursor.
--   handled_at:   set once the comment's feedback has been durably acted upon.
--   attempts:     consecutive processing failures; used to give up on a poison
--                 comment instead of retrying the fix agent forever.
--   comment_kind: 'inline' (/pulls/{n}/comments) vs 'conversation'
--                 (/issues/{n}/comments). These are distinct GitHub id sequences,
--                 so a global UNIQUE(scm_comment_id) could let one silently
--                 overwrite the other. Uniqueness is keyed on
--                 (comment_kind, scm_comment_id) instead.
--
-- A column-level UNIQUE can't be dropped in place (SQLite forbids DROP INDEX on a
-- constraint's auto-index), so pr_review_comments is rebuilt: stash the rows in a
-- transient shadow table, recreate the real table with the namespaced schema,
-- copy the rows back, drop the shadow. Nothing references pr_review_comments, so
-- this is FK-safe; every pre-existing row is an inline review comment.
CREATE TABLE pr_review_comments_shadow AS SELECT * FROM pr_review_comments;

DROP TABLE pr_review_comments;

CREATE TABLE pr_review_comments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scm_comment_id INTEGER NOT NULL,
    comment_kind TEXT NOT NULL DEFAULT 'inline',
    pr_url TEXT NOT NULL,
    review_id INTEGER REFERENCES pr_reviews(id),
    path TEXT NOT NULL,
    position INTEGER,
    line INTEGER,
    body TEXT NOT NULL,
    author TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    html_url TEXT,
    handled_at TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    UNIQUE(comment_kind, scm_comment_id)
);

INSERT INTO pr_review_comments
    (id, scm_comment_id, comment_kind, pr_url, review_id, path, position, line,
     body, author, created_at, updated_at, html_url, handled_at, attempts)
SELECT id, scm_comment_id, 'inline', pr_url, review_id, path, position, line,
       body, author, created_at, updated_at, html_url, NULL, 0
FROM pr_review_comments_shadow;

DROP TABLE pr_review_comments_shadow;

CREATE INDEX IF NOT EXISTS idx_pr_review_comments_pr ON pr_review_comments(pr_url);
