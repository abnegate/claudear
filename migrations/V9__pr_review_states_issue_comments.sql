-- Track PR conversation (issue) comments separately from inline review comments.
-- GitHub delivers plain "@claudear ..." PR comments via the issues/{n}/comments
-- endpoint, which uses a distinct comment id space from inline review comments,
-- so the review watcher advances an independent cursor for them.
ALTER TABLE pr_review_states ADD COLUMN last_issue_comment_id INTEGER;
ALTER TABLE pr_review_states ADD COLUMN last_issue_comment_time TEXT;

-- Make PR review-comment processing at-least-once.
-- The pr_review_comments ledger (keyed by scm_comment_id UNIQUE) becomes the
-- authority for "what still needs action", so a crash or downstream failure
-- between detecting a comment and acting on it no longer drops it: unhandled
-- rows are re-surfaced on the next poll regardless of the polling cursor.
--   handled_at: set once the comment's feedback has been durably acted upon.
--   attempts:   consecutive processing failures; used to give up on a poison
--               comment instead of retrying the fix agent forever.
ALTER TABLE pr_review_comments ADD COLUMN handled_at TEXT;
ALTER TABLE pr_review_comments ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0;
