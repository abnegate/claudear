//! Intent classification contract.
//!
//! Defines the [`Intent`] type, the [`IntentClassifier`] trait, and the
//! prompt/response helpers shared by both backends. The concrete classifiers
//! live with their execution machinery (mirroring the `RepoClassifier` split):
//! [`crate::llm_classifier::LocalLlmIntentClassifier`] (offline GGUF) and
//! [`crate::agent_classifier::AgentIntentClassifier`] (agent, `--json-schema`).

use crate::llm_analyzer::truncate;
use async_trait::async_trait;
use claudear_core::types::Issue;

/// Maximum description length included in an intent-classification prompt.
pub(crate) const MAX_INTENT_DESC_CHARS: usize = 1500;

/// Maximum title length included in an intent-classification prompt.
pub(crate) const MAX_INTENT_TITLE_CHARS: usize = 400;

/// Category definitions shared by both backends' prompts, so they classify
/// against identical wording.
pub(crate) const CATEGORY_RULES: &str = "\
- \"bug\": reports something broken — an error, crash, incorrect behavior, or regression.\n\
- \"security\": reports a security vulnerability, exploit, data leak, or authentication problem.\n\
- \"question\": ONLY asks for information, an explanation, or how something works, and does NOT want any code changed.\n\
- \"fix\": requests a feature, change, or improvement that is NOT itself a bug or security issue.\n\
When in doubt about a code problem, answer \"bug\".";

/// Classification of an incoming payload, used to route it into the action
/// pipeline. `Bug`/`Security` start the verify -> resolve chain; `Question`/`Fix`
/// start a reply. `Bug` is the safe default on any doubt about a code problem.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Intent {
    /// Reports something broken: an error, crash, incorrect behavior, or regression.
    Bug,
    /// Reports a security vulnerability, exploit, data leak, or auth problem.
    Security,
    /// The user is only asking for information/explanation; no code change wanted.
    Question,
    /// Requests a feature/change/improvement that is not itself a bug or security issue.
    Fix,
}

impl Intent {
    /// Whether this category is a bug or security report (routes to Verify).
    pub fn is_bug_or_security(&self) -> bool {
        matches!(self, Intent::Bug | Intent::Security)
    }
}

/// Classifies an issue's [`Intent`]. Returns `None` when the backend is
/// unavailable or the response cannot be interpreted, so callers can fall back
/// to a heuristic.
///
/// `conversation` carries the prior messages of an ongoing thread (e.g. a
/// Discord reply chain), oldest-first, so a follow-up is classified in context
/// rather than in isolation. A confirmation like "yes create a pr now" is
/// ambiguous alone but clearly a `Fix` once the thread is visible. Pass `None`
/// when there is no prior conversation.
#[async_trait]
pub trait IntentClassifier: Send + Sync {
    async fn classify_intent(&self, issue: &Issue, conversation: Option<&str>) -> Option<Intent>;
}

/// The prompt section that carries the prior conversation and instructs the
/// model to classify the latest message in that context. Empty when there is no
/// conversation, so single-message classification is unchanged. Shared by both
/// backends so they frame the follow-up case identically.
pub(crate) fn intent_conversation_section(conversation: Option<&str>) -> String {
    match conversation.map(str::trim).filter(|c| !c.is_empty()) {
        Some(convo) => format!(
            "This message is the latest turn in an ongoing conversation. Prior turns \
             (oldest first) are context only:\n\
             {convo}\n\n\
             Classify the LATEST message below, not the prior turns. A follow-up that \
             asks to open a PR, apply a change, or proceed with a fix is \"fix\" (or \
             \"bug\"/\"security\" if it points at a defect), even when earlier turns \
             were questions.\n\n"
        ),
        None => String::new(),
    }
}

/// The message body shared by both prompts: the issue description, truncated and
/// omitted when it merely duplicates the title.
pub(crate) fn intent_body(issue: &Issue) -> String {
    if let Some(desc) = issue.description.as_deref() {
        let desc = desc.trim();
        if !desc.is_empty() && desc != issue.title {
            return truncate(desc, MAX_INTENT_DESC_CHARS);
        }
    }
    String::new()
}

/// The issue title, truncated for prompt inclusion.
pub(crate) fn intent_title(issue: &Issue) -> String {
    truncate(&issue.title, MAX_INTENT_TITLE_CHARS)
}

/// Map a classification response to an [`Intent`]. The first clean token among
/// `bug | security | question | fix` wins; otherwise a contains-fallback is
/// applied (biased toward `bug` for code problems). Returns `None` when no
/// category can be determined.
///
/// Handles both the local model's freeform one-word replies and the agent's
/// schema-constrained enum value.
pub(crate) fn parse_intent(s: &str) -> Option<Intent> {
    let lower = s.trim().to_lowercase();

    // First clean alphanumeric token decides when it is an exact category.
    let first = lower
        .split(|c: char| !c.is_alphanumeric())
        .find(|t| !t.is_empty())
        .unwrap_or("");
    match first {
        "bug" => return Some(Intent::Bug),
        "security" => return Some(Intent::Security),
        "question" => return Some(Intent::Question),
        "fix" => return Some(Intent::Fix),
        _ => {}
    }

    // Contains-fallback for noisy output. Order matters: a mentioned bug/security
    // problem dominates a generic "fix" verb that often co-occurs with it.
    if lower.contains("security") {
        Some(Intent::Security)
    } else if lower.contains("bug") {
        Some(Intent::Bug)
    } else if lower.contains("question") && !lower.contains("fix") {
        Some(Intent::Question)
    } else if lower.contains("fix") {
        Some(Intent::Fix)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use claudear_core::types::{IssuePriority, IssueStatus};
    use std::collections::HashMap;

    fn sample_issue(title: &str, description: Option<&str>) -> Issue {
        Issue {
            id: "ID-1".to_string(),
            short_id: "ID-1".to_string(),
            title: title.to_string(),
            description: description.map(|s| s.to_string()),
            url: String::new(),
            source: "discord".to_string(),
            priority: IssuePriority::Medium,
            status: IssueStatus::Open,
            metadata: HashMap::new(),
            created_at: None,
            updated_at: None,
        }
    }

    #[test]
    fn test_parse_intent_question() {
        assert_eq!(parse_intent("question"), Some(Intent::Question));
        assert_eq!(parse_intent("  Question\n"), Some(Intent::Question));
        assert_eq!(parse_intent("question."), Some(Intent::Question));
    }

    #[test]
    fn test_parse_intent_fix() {
        assert_eq!(parse_intent("fix"), Some(Intent::Fix));
        assert_eq!(parse_intent("FIX"), Some(Intent::Fix));
        assert_eq!(parse_intent("fix request"), Some(Intent::Fix));
    }

    #[test]
    fn test_parse_intent_bug_and_security() {
        assert_eq!(parse_intent("bug"), Some(Intent::Bug));
        assert_eq!(parse_intent("BUG\n"), Some(Intent::Bug));
        assert_eq!(parse_intent("security"), Some(Intent::Security));
        assert_eq!(parse_intent("Security."), Some(Intent::Security));
        assert!(Intent::Bug.is_bug_or_security());
        assert!(Intent::Security.is_bug_or_security());
        assert!(!Intent::Question.is_bug_or_security());
        assert!(!Intent::Fix.is_bug_or_security());
    }

    #[test]
    fn test_parse_intent_first_clean_token_decides() {
        assert_eq!(
            parse_intent("bug, though it could be a question"),
            Some(Intent::Bug)
        );
        assert_eq!(parse_intent("- security issue"), Some(Intent::Security));
        assert_eq!(parse_intent("Answer: question"), Some(Intent::Question));
    }

    #[test]
    fn test_parse_intent_contains_fallback_prefers_problem() {
        assert_eq!(
            parse_intent("this looks like a security vulnerability we should fix"),
            Some(Intent::Security)
        );
        assert_eq!(
            parse_intent("seems to be a bug that needs a fix"),
            Some(Intent::Bug)
        );
        assert_eq!(
            parse_intent("just a quick question about usage"),
            Some(Intent::Question)
        );
    }

    #[test]
    fn test_parse_intent_unparseable_is_none() {
        assert_eq!(parse_intent("i am not sure"), None);
        assert_eq!(parse_intent(""), None);
    }

    #[test]
    fn test_parse_intent_handles_noisy_output() {
        assert_eq!(parse_intent("question\n"), Some(Intent::Question));
        assert_eq!(parse_intent("\"question\""), Some(Intent::Question));
        assert_eq!(parse_intent("- fix"), Some(Intent::Fix));
        assert_eq!(parse_intent("bug\nbecause it crashes"), Some(Intent::Bug));
    }

    #[test]
    fn test_intent_body_omitted_when_same_as_title() {
        let issue = sample_issue("how do I paginate?", Some("how do I paginate?"));
        assert_eq!(intent_body(&issue), "");
    }

    #[test]
    fn test_intent_body_included_when_distinct() {
        let issue = sample_issue("Realtime onClose error", Some("triggerStats() null given"));
        assert_eq!(intent_body(&issue), "triggerStats() null given");
    }

    #[test]
    fn test_intent_body_truncated() {
        // 'Z' never appears in the ellipsis, so counting it reflects the truncated
        // body length (excluding the appended "...").
        let issue = sample_issue("t", Some(&"Z".repeat(MAX_INTENT_DESC_CHARS * 4)));
        let body = intent_body(&issue);
        assert!(body.matches('Z').count() <= MAX_INTENT_DESC_CHARS);
        assert!(body.ends_with("..."));
    }
}
