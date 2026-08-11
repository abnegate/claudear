-- Track PR conversation (issue) comments separately from inline review comments.
-- GitHub delivers plain "@claudear ..." PR comments via the issues/{n}/comments
-- endpoint, which uses a distinct comment id space from inline review comments,
-- so the review watcher advances an independent cursor for them.
ALTER TABLE pr_review_states ADD COLUMN last_issue_comment_id INTEGER;
ALTER TABLE pr_review_states ADD COLUMN last_issue_comment_time TEXT;
