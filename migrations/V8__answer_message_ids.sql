-- V8: Track the Discord message ids of the answers Claudear sends, so a user's
-- reply to any answer chunk can be mapped back to the issue it belongs to
-- (reply-chain context) without fetching the message from Discord.
--
-- Stored as a comma-delimited list with leading/trailing commas, e.g.
--   ",123,456,789,"
-- so reverse lookups can match a whole id with LIKE '%,<id>,%' and never match
-- a partial id.
ALTER TABLE fix_attempts ADD COLUMN answer_message_ids TEXT;
