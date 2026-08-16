-- Persist the routing intent (QA/fix/bug/security) classified for an attempt.
--
-- The Discord reply-chain transcript, when built for routing/classification
-- (TranscriptTrust::ClaudearOnly), must not re-inject Claudear's generated
-- answer body: a prior answer can echo untrusted user text, which would then
-- steer the QA-vs-fix decision. Instead it emits a trusted structural marker
-- derived from this label (classified upstream on trusted content). Older
-- attempts have no stored intent and fall back to a generic marker.
ALTER TABLE fix_attempts ADD COLUMN routing_intent TEXT;
