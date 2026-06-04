//! LLM-enhanced analysis implementation.
//!
//! Uses the local LLM engine (shared with `LlmRepoClassifier`) to provide
//! richer issue assessment, review classification, log extraction, and
//! cross-repo correlation enrichment.

use claudear_analysis::llm::{
    LlmAnalyzer, LlmCorrelationExplanation, LlmIssueAssessment, LlmLogAnalysis,
};
use claudear_core::types::{BlastRadius, ExtractedLearnings, Issue, MatchResult, ReviewCategory};
use claudear_integrations::chat::llm::{GenerationParams, LlmEngine};
use serde::de::DeserializeOwned;
use std::sync::Arc;
use std::time::Instant;

/// Maximum issues per batch (keeps prompt within 4096-token context).
const MAX_BATCH_SIZE: usize = 8;

/// Maximum metadata value length in prompt.
const MAX_METADATA_CHARS: usize = 100;

/// Maximum comment length for review classification.
const MAX_COMMENT_CHARS: usize = 800;

/// Maximum log text length (prioritize end of log).
const MAX_LOG_CHARS: usize = 8192;

/// Whether an incoming chat-source message is a pure question or a request
/// that should be routed to the issue-resolution (fix/feature) pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Intent {
    /// The user is only asking for information/explanation; no code change wanted.
    Question,
    /// The user wants something fixed/changed/built (the safe default on any doubt).
    FixRequest,
}

/// Maximum description length included in the intent-classification prompt.
const MAX_INTENT_DESC_CHARS: usize = 1500;

pub struct LlmAnalyzerImpl {
    engine: Arc<LlmEngine>,
}

impl LlmAnalyzerImpl {
    pub fn new(engine: Arc<LlmEngine>) -> Self {
        Self { engine }
    }

    /// Classify an incoming message as a pure question vs a fix/feature request.
    ///
    /// Returns `None` when the model is unavailable or the response cannot be
    /// parsed. Callers MUST treat `None` (and any ambiguity) as a fix request,
    /// preserving the default issue-resolution behaviour.
    pub fn classify_intent(&self, issue: &Issue) -> Option<Intent> {
        let prompt = build_intent_prompt(issue);
        let response = self.complete(&prompt, 8)?;
        parse_intent(&response)
    }

    /// Run a completion against the LLM engine.
    fn complete(&self, prompt: &str, max_tokens: u32) -> Option<String> {
        let params = GenerationParams {
            temperature: 0.1,
            max_tokens,
            top_p: 0.9,
            stop_sequences: vec![
                "\n\n".to_string(),
                "<|end|>".to_string(),
                "<|user|>".to_string(),
            ],
        };

        let start = Instant::now();
        let tokens = match self.engine.complete_streaming(prompt, &params) {
            Ok(t) => t,
            Err(e) => {
                tracing::warn!(error = %e, "LLM analyzer inference failed");
                return None;
            }
        };

        let response = tokens.join("");
        tracing::debug!(
            elapsed_ms = start.elapsed().as_millis(),
            response_len = response.len(),
            "LLM analyzer completion"
        );

        if response.trim().is_empty() {
            None
        } else {
            Some(response)
        }
    }
}

/// Try to parse a JSON value from a potentially noisy LLM response.
fn parse_json_response<T: DeserializeOwned>(response: &str) -> Option<T> {
    let trimmed = response.trim();

    // Direct parse
    if let Ok(val) = serde_json::from_str::<T>(trimmed) {
        return Some(val);
    }

    // Extract from markdown code block
    if let Some(start) = trimmed.find("```json") {
        let after = &trimmed[start + 7..];
        if let Some(end) = after.find("```") {
            if let Ok(val) = serde_json::from_str::<T>(after[..end].trim()) {
                return Some(val);
            }
        }
    }
    if let Some(start) = trimmed.find("```") {
        let after = &trimmed[start + 3..];
        // Skip optional language tag on same line
        let after = if let Some(nl) = after.find('\n') {
            &after[nl + 1..]
        } else {
            after
        };
        if let Some(end) = after.find("```") {
            if let Ok(val) = serde_json::from_str::<T>(after[..end].trim()) {
                return Some(val);
            }
        }
    }

    // Find outermost JSON structure
    let first_bracket = trimmed.find('[').map(|i| (i, ']'));
    let first_brace = trimmed.find('{').map(|i| (i, '}'));

    let (start, close_char) = match (first_bracket, first_brace) {
        (Some((bi, _)), Some((ci, _))) => {
            if bi < ci {
                (bi, ']')
            } else {
                (ci, '}')
            }
        }
        (Some((bi, _)), None) => (bi, ']'),
        (None, Some((ci, _))) => (ci, '}'),
        (None, None) => return None,
    };

    let end = trimmed.rfind(close_char)?;
    if end <= start {
        return None;
    }

    serde_json::from_str::<T>(&trimmed[start..=end]).ok()
}

/// Truncate a string to approximately `max_chars` bytes, safely respecting
/// UTF-8 char boundaries. Appends "..." when truncated.
fn truncate(s: &str, max_chars: usize) -> String {
    if s.len() <= max_chars {
        return s.to_string();
    }
    let target = max_chars.saturating_sub(3);
    let end = s
        .char_indices()
        .take_while(|(i, _)| *i <= target)
        .last()
        .map(|(i, c)| i + c.len_utf8())
        .unwrap_or(0);
    format!("{}...", &s[..end])
}

/// Build the batch assessment prompt for issues.
fn build_assessment_prompt(candidates: &[(Issue, MatchResult)]) -> String {
    let limit = candidates.len().min(MAX_BATCH_SIZE);
    let mut parts = Vec::new();

    parts.push(
        "<|system|>\n\
         You are a software issue triage expert. For each issue, assess:\n\
         - severity (0.0-1.0, where 1.0 is most critical)\n\
         - blast_radius (one of: critical, infrastructure, core, peripheral, test, cosmetic)\n\
         - fingerprint (short canonical root cause key for deduplication)\n\
         Respond ONLY with a JSON array like:\n\
         [{\"id\":\"ISSUE-123\",\"severity\":0.9,\"blast_radius\":\"critical\",\"fingerprint\":\"root_cause_key\"}]\n\
         <|end|>"
            .to_string(),
    );

    let mut user = String::from("<|user|>\n");
    for (i, (issue, mr)) in candidates.iter().take(limit).enumerate() {
        // Use issue.id (not short_id) so the LLM echoes the same key used for lookup
        user.push_str(&format!("#{} [{}]: {}\n", i + 1, issue.id, issue.title));

        // Include relevant metadata
        let meta_keys = [
            "error_type",
            "culprit",
            "level",
            "events",
            "users",
            "stacktrace",
        ];
        let mut meta_parts = Vec::new();
        for key in &meta_keys {
            if let Some(val) = issue.metadata.get(*key) {
                let val_str = match val {
                    serde_json::Value::String(s) => truncate(s, MAX_METADATA_CHARS),
                    other => {
                        let s = other.to_string();
                        truncate(&s, MAX_METADATA_CHARS)
                    }
                };
                meta_parts.push(format!("{}={}", key, val_str));
            }
        }
        if !meta_parts.is_empty() {
            user.push_str(&format!("  {}\n", meta_parts.join(" ")));
        }

        // Include match reason for context
        if !mr.reason.is_empty() {
            user.push_str(&format!(
                "  reason={}\n",
                truncate(&mr.reason, MAX_METADATA_CHARS)
            ));
        }
    }
    user.push_str("<|end|>\n<|assistant|>");
    parts.push(user);

    parts.join("\n")
}

/// Build the review classification prompt.
fn build_review_prompt(comment_body: &str) -> String {
    let body = truncate(comment_body, MAX_COMMENT_CHARS);
    format!(
        "<|system|>\n\
         Classify this code review comment into exactly one category: \
         security, missing_tests, wrong_approach, style_issue, incomplete, \
         performance, documentation, other.\n\
         Respond with ONLY the category name.\n\
         <|end|>\n\
         <|user|>\n\
         {}\n\
         <|end|>\n\
         <|assistant|>",
        body
    )
}

/// Build the log extraction prompt.
fn build_learnings_prompt(log_text: &str) -> String {
    // Prioritize end of log where results appear (find a safe UTF-8 boundary)
    let text = if log_text.len() > MAX_LOG_CHARS {
        let start = log_text.len() - MAX_LOG_CHARS;
        // Advance to the next char boundary
        let safe_start = log_text[start..]
            .char_indices()
            .next()
            .map(|(i, _)| start + i)
            .unwrap_or(start);
        &log_text[safe_start..]
    } else {
        log_text
    };

    format!(
        "<|system|>\n\
         Extract structured learnings from this execution log. \
         Respond ONLY with JSON: \
         {{\"root_cause\":\"...\",\"files_modified\":[...],\
         \"strategy\":\"tdd|direct_fix|investigation|exploration\",\
         \"tests_added\":true/false,\"summary\":\"...\"}}\n\
         <|end|>\n\
         <|user|>\n\
         {}\n\
         <|end|>\n\
         <|assistant|>",
        text
    )
}

/// Build the correlation explanation prompt.
fn build_correlation_prompt(
    correlations: &[(String, String, i64)],
    issues_context: &str,
) -> String {
    let mut pairs = String::new();
    for (a, b, count) in correlations {
        pairs.push_str(&format!(
            "- {} \u{2194} {}: {} co-occurring windows\n",
            a, b, count
        ));
    }

    format!(
        "<|system|>\n\
         Analyze cross-repo failure correlations. For each repo pair, explain \
         the likely causal relationship and rate confidence (0.0-1.0). \
         Respond ONLY with JSON array: \
         [{{\"repo_a\":\"...\",\"repo_b\":\"...\",\"explanation\":\"...\",\"confidence\":0.8}}]\n\
         <|end|>\n\
         <|user|>\n\
         {}\n\
         Context:\n\
         {}\n\
         <|end|>\n\
         <|assistant|>",
        pairs, issues_context
    )
}

/// Build the question-vs-fix intent classification prompt.
///
/// The prompt is biased toward `fix`: only an unambiguous, information-seeking
/// message that wants no code change should be classified as a question.
fn build_intent_prompt(issue: &Issue) -> String {
    let mut body = String::new();
    if let Some(desc) = issue.description.as_deref() {
        let desc = desc.trim();
        if !desc.is_empty() && desc != issue.title {
            body = truncate(desc, MAX_INTENT_DESC_CHARS);
        }
    }

    format!(
        "<|system|>\n\
         You classify an incoming developer-support message into exactly one of:\n\
         - \"question\": the user is ONLY asking for information, an explanation, or how \
         something works, and does NOT want any code changed.\n\
         - \"fix\": the user wants something fixed, changed, added, built, or implemented, \
         OR the message reports a bug, OR it is ambiguous / could require a code change.\n\
         When in doubt, answer \"fix\".\n\
         Respond with ONLY the single word: question or fix.\n\
         <|end|>\n\
         <|user|>\n\
         Title: {}\n\
         {}\n\
         <|end|>\n\
         <|assistant|>",
        truncate(&issue.title, MAX_METADATA_CHARS * 4),
        body
    )
}

/// Parse the intent classification response. Biased toward `FixRequest`:
/// only a clear "question" answer yields `Intent::Question`.
fn parse_intent(s: &str) -> Option<Intent> {
    let lower = s.trim().to_lowercase();
    let first = lower
        .split(|c: char| !c.is_alphanumeric())
        .find(|t| !t.is_empty())
        .unwrap_or("");

    match first {
        "question" => Some(Intent::Question),
        "fix" => Some(Intent::FixRequest),
        _ => {
            if lower.contains("question") && !lower.contains("fix") {
                Some(Intent::Question)
            } else if lower.contains("fix") {
                Some(Intent::FixRequest)
            } else {
                None
            }
        }
    }
}

/// Parse a blast radius string (case-insensitive) into the enum.
fn parse_blast_radius(s: &str) -> Option<BlastRadius> {
    match s.trim().to_lowercase().as_str() {
        "critical" => Some(BlastRadius::Critical),
        "infrastructure" => Some(BlastRadius::Infrastructure),
        "core" => Some(BlastRadius::Core),
        "peripheral" => Some(BlastRadius::Peripheral),
        "test" => Some(BlastRadius::Test),
        "cosmetic" => Some(BlastRadius::Cosmetic),
        _ => None,
    }
}

/// Parse a review category string (case-insensitive, with contains-fallback).
fn parse_review_category(s: &str) -> Option<ReviewCategory> {
    let lower = s.trim().to_lowercase();

    // Exact match first
    match lower.as_str() {
        "security" => return Some(ReviewCategory::Security),
        "missing_tests" => return Some(ReviewCategory::MissingTests),
        "wrong_approach" => return Some(ReviewCategory::WrongApproach),
        "style_issue" => return Some(ReviewCategory::StyleIssue),
        "incomplete" => return Some(ReviewCategory::Incomplete),
        "performance" => return Some(ReviewCategory::Performance),
        "documentation" => return Some(ReviewCategory::Documentation),
        "other" => return Some(ReviewCategory::Other),
        _ => {}
    }

    // Contains-fallback
    if lower.contains("security") {
        return Some(ReviewCategory::Security);
    }
    if lower.contains("missing_tests") || lower.contains("missing tests") {
        return Some(ReviewCategory::MissingTests);
    }
    if lower.contains("wrong_approach") || lower.contains("wrong approach") {
        return Some(ReviewCategory::WrongApproach);
    }
    if lower.contains("style") {
        return Some(ReviewCategory::StyleIssue);
    }
    if lower.contains("incomplete") {
        return Some(ReviewCategory::Incomplete);
    }
    if lower.contains("performance") {
        return Some(ReviewCategory::Performance);
    }
    if lower.contains("documentation") || lower.contains("docs") {
        return Some(ReviewCategory::Documentation);
    }
    if lower.contains("other") {
        return Some(ReviewCategory::Other);
    }

    None
}

impl LlmAnalyzer for LlmAnalyzerImpl {
    fn assess_issues(
        &self,
        candidates: &[(Issue, MatchResult)],
    ) -> Option<Vec<LlmIssueAssessment>> {
        if candidates.is_empty() {
            return Some(Vec::new());
        }

        let prompt = build_assessment_prompt(candidates);
        let response = self.complete(&prompt, 512)?;

        #[derive(serde::Deserialize)]
        #[allow(dead_code)]
        struct RawAssessment {
            id: String,
            severity: f64,
            blast_radius: String,
            fingerprint: String,
        }

        let raw: Vec<RawAssessment> = parse_json_response(&response)?;

        let assessments = raw
            .into_iter()
            .filter_map(|r| {
                let severity = r.severity.clamp(0.0, 1.0);
                let blast_radius = parse_blast_radius(&r.blast_radius)?;
                Some(LlmIssueAssessment {
                    issue_id: r.id,
                    severity,
                    blast_radius,
                    fingerprint: r.fingerprint,
                })
            })
            .collect::<Vec<_>>();

        if assessments.is_empty() {
            None
        } else {
            Some(assessments)
        }
    }

    fn classify_review(&self, comment_body: &str) -> Option<ReviewCategory> {
        let prompt = build_review_prompt(comment_body);
        let response = self.complete(&prompt, 32)?;
        parse_review_category(&response)
    }

    fn extract_learnings(&self, log_text: &str) -> Option<LlmLogAnalysis> {
        let prompt = build_learnings_prompt(log_text);
        let response = self.complete(&prompt, 256)?;

        #[derive(serde::Deserialize)]
        struct RawLearnings {
            root_cause: Option<String>,
            #[serde(default)]
            files_modified: Vec<String>,
            strategy: Option<String>,
            #[serde(default)]
            tests_added: bool,
            summary: Option<String>,
        }

        let raw: RawLearnings = parse_json_response(&response)?;

        let fix_approach = raw
            .strategy
            .clone()
            .unwrap_or_else(|| "unknown".to_string());
        let strategy_summary = raw
            .summary
            .clone()
            .unwrap_or_else(|| format!("LLM: {}", fix_approach));

        Some(LlmLogAnalysis {
            learnings: ExtractedLearnings {
                root_cause: raw.root_cause,
                files_modified: raw.files_modified,
                strategy_used: raw.strategy,
                tests_added: raw.tests_added,
                key_decisions: Vec::new(),
            },
            fix_approach,
            strategy_summary,
        })
    }

    fn explain_correlations(
        &self,
        correlations: &[(String, String, i64)],
        issues_context: &str,
    ) -> Vec<LlmCorrelationExplanation> {
        if correlations.is_empty() {
            return Vec::new();
        }

        let prompt = build_correlation_prompt(correlations, issues_context);
        let response = match self.complete(&prompt, 256) {
            Some(r) => r,
            None => return Vec::new(),
        };

        #[derive(serde::Deserialize)]
        struct RawExplanation {
            repo_a: String,
            repo_b: String,
            explanation: String,
            #[serde(default)]
            confidence: f64,
        }

        let raw: Vec<RawExplanation> = match parse_json_response(&response) {
            Some(r) => r,
            None => return Vec::new(),
        };

        raw.into_iter()
            .map(|r| LlmCorrelationExplanation {
                repo_a: r.repo_a,
                repo_b: r.repo_b,
                explanation: r.explanation,
                confidence: r.confidence.clamp(0.0, 1.0),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use claudear_core::types::{IssuePriority, IssueStatus, MatchPriority};
    use std::collections::HashMap;

    fn sample_issue(id: &str, title: &str) -> Issue {
        Issue {
            id: id.to_string(),
            short_id: id.to_string(),
            title: title.to_string(),
            description: None,
            url: String::new(),
            source: "sentry".to_string(),
            priority: IssuePriority::Medium,
            status: IssueStatus::Open,
            metadata: HashMap::new(),
            created_at: None,
            updated_at: None,
        }
    }

    fn sample_match_result() -> MatchResult {
        MatchResult {
            matches: true,
            reason: "test match".to_string(),
            priority: MatchPriority::High,
        }
    }

    // --- Prompt building tests ---

    #[test]
    fn test_build_assessment_prompt() {
        let mut issue = sample_issue("ISSUE-123", "NullPointerException in PaymentService.charge");
        issue.metadata.insert(
            "error_type".to_string(),
            serde_json::Value::String("NullPointerException".to_string()),
        );
        issue.metadata.insert(
            "culprit".to_string(),
            serde_json::Value::String("PaymentService.charge".to_string()),
        );
        issue.metadata.insert(
            "level".to_string(),
            serde_json::Value::String("fatal".to_string()),
        );

        let candidates = vec![(issue, sample_match_result())];
        let prompt = build_assessment_prompt(&candidates);

        assert!(prompt.contains("ISSUE-123"));
        assert!(prompt.contains("NullPointerException in PaymentService.charge"));
        assert!(prompt.contains("error_type=NullPointerException"));
        assert!(prompt.contains("culprit=PaymentService.charge"));
        assert!(prompt.contains("level=fatal"));
        assert!(prompt.contains("<|system|>"));
        assert!(prompt.contains("<|assistant|>"));
        assert!(prompt.len() < 12000);
    }

    #[test]
    fn test_build_assessment_prompt_caps_at_8() {
        let candidates: Vec<(Issue, MatchResult)> = (0..12)
            .map(|i| {
                (
                    sample_issue(&format!("ISSUE-{}", i), &format!("Error {}", i)),
                    sample_match_result(),
                )
            })
            .collect();

        let prompt = build_assessment_prompt(&candidates);

        // Should contain issues 1-8 but not 9-12
        assert!(prompt.contains("#1 [ISSUE-0]"));
        assert!(prompt.contains("#8 [ISSUE-7]"));
        assert!(!prompt.contains("#9 [ISSUE-8]"));
    }

    #[test]
    fn test_build_review_prompt() {
        let prompt = build_review_prompt("This function is missing unit tests for edge cases");
        assert!(prompt.contains("This function is missing unit tests"));
        assert!(prompt.contains("security"));
        assert!(prompt.contains("missing_tests"));
        assert!(prompt.contains("<|system|>"));
        assert!(prompt.contains("<|assistant|>"));
    }

    #[test]
    fn test_build_learnings_prompt() {
        let prompt = build_learnings_prompt("Read src/main.rs\nEdit src/handler.rs\ncargo test");
        assert!(prompt.contains("src/main.rs"));
        assert!(prompt.contains("root_cause"));
        assert!(prompt.contains("<|system|>"));
    }

    #[test]
    fn test_build_learnings_prompt_truncates_long_log() {
        let long_log = "x".repeat(20000);
        let prompt = build_learnings_prompt(&long_log);
        // Should use last MAX_LOG_CHARS of the log
        assert!(prompt.len() < 20000);
    }

    #[test]
    fn test_build_correlation_prompt() {
        let correlations = vec![
            ("org/core".to_string(), "org/web".to_string(), 5i64),
            ("org/auth".to_string(), "org/api".to_string(), 3i64),
        ];
        let prompt = build_correlation_prompt(&correlations, "Recent issues in org/core: crash");

        assert!(prompt.contains("org/core"));
        assert!(prompt.contains("org/web"));
        assert!(prompt.contains("5 co-occurring windows"));
        assert!(prompt.contains("Recent issues"));
    }

    // --- JSON parsing tests ---

    #[test]
    fn test_parse_json_response_direct() {
        let json = r#"[{"id":"ISSUE-1","severity":0.9,"blast_radius":"critical","fingerprint":"null_ref"}]"#;
        let result: Option<Vec<serde_json::Value>> = parse_json_response(json);
        assert!(result.is_some());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_parse_json_response_markdown_block() {
        let response = "Here is the result:\n```json\n{\"key\": \"value\"}\n```\nDone.";
        let result: Option<serde_json::Value> = parse_json_response(response);
        assert!(result.is_some());
        assert_eq!(result.unwrap()["key"], "value");
    }

    #[test]
    fn test_parse_json_response_garbage() {
        let result: Option<serde_json::Value> =
            parse_json_response("I don't know the answer to that question");
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_json_response_embedded_json() {
        let response = "The analysis shows: [{\"id\":\"A\",\"val\":1}] based on the data.";
        let result: Option<Vec<serde_json::Value>> = parse_json_response(response);
        assert!(result.is_some());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_parse_json_response_object_in_noise() {
        let response = "Result: {\"severity\": 0.8} end";
        let result: Option<serde_json::Value> = parse_json_response(response);
        assert!(result.is_some());
    }

    // --- Blast radius parsing ---

    #[test]
    fn test_parse_blast_radius_strings() {
        assert_eq!(parse_blast_radius("critical"), Some(BlastRadius::Critical));
        assert_eq!(
            parse_blast_radius("infrastructure"),
            Some(BlastRadius::Infrastructure)
        );
        assert_eq!(parse_blast_radius("core"), Some(BlastRadius::Core));
        assert_eq!(
            parse_blast_radius("peripheral"),
            Some(BlastRadius::Peripheral)
        );
        assert_eq!(parse_blast_radius("test"), Some(BlastRadius::Test));
        assert_eq!(parse_blast_radius("cosmetic"), Some(BlastRadius::Cosmetic));
        // Case-insensitive
        assert_eq!(parse_blast_radius("CRITICAL"), Some(BlastRadius::Critical));
        assert_eq!(parse_blast_radius("Core"), Some(BlastRadius::Core));
        // Invalid
        assert_eq!(parse_blast_radius("unknown"), None);
        assert_eq!(parse_blast_radius(""), None);
    }

    // --- Review category parsing ---

    #[test]
    fn test_parse_review_category() {
        assert_eq!(
            parse_review_category("security"),
            Some(ReviewCategory::Security)
        );
        assert_eq!(
            parse_review_category("missing_tests"),
            Some(ReviewCategory::MissingTests)
        );
        assert_eq!(
            parse_review_category("wrong_approach"),
            Some(ReviewCategory::WrongApproach)
        );
        assert_eq!(
            parse_review_category("style_issue"),
            Some(ReviewCategory::StyleIssue)
        );
        assert_eq!(
            parse_review_category("incomplete"),
            Some(ReviewCategory::Incomplete)
        );
        assert_eq!(
            parse_review_category("performance"),
            Some(ReviewCategory::Performance)
        );
        assert_eq!(
            parse_review_category("documentation"),
            Some(ReviewCategory::Documentation)
        );
        assert_eq!(parse_review_category("other"), Some(ReviewCategory::Other));
    }

    #[test]
    fn test_parse_review_category_case_insensitive() {
        assert_eq!(
            parse_review_category("SECURITY"),
            Some(ReviewCategory::Security)
        );
        assert_eq!(
            parse_review_category("Missing_Tests"),
            Some(ReviewCategory::MissingTests)
        );
    }

    #[test]
    fn test_parse_review_category_contains_fallback() {
        assert_eq!(
            parse_review_category("The category is security."),
            Some(ReviewCategory::Security)
        );
        assert_eq!(
            parse_review_category("This is about performance issues"),
            Some(ReviewCategory::Performance)
        );
    }

    #[test]
    fn test_parse_review_category_invalid() {
        assert_eq!(parse_review_category("hello world"), None);
    }

    // --- Intent classification parsing ---

    #[test]
    fn test_parse_intent_question() {
        assert_eq!(parse_intent("question"), Some(Intent::Question));
        assert_eq!(parse_intent("  Question\n"), Some(Intent::Question));
        assert_eq!(parse_intent("question."), Some(Intent::Question));
    }

    #[test]
    fn test_parse_intent_fix() {
        assert_eq!(parse_intent("fix"), Some(Intent::FixRequest));
        assert_eq!(parse_intent("FIX"), Some(Intent::FixRequest));
        assert_eq!(parse_intent("fix request"), Some(Intent::FixRequest));
    }

    #[test]
    fn test_parse_intent_biases_to_fix_when_both_present() {
        // If the model hedges and mentions both, prefer fix.
        assert_eq!(
            parse_intent("this is a question but might need a fix"),
            Some(Intent::FixRequest)
        );
    }

    #[test]
    fn test_parse_intent_unparseable_is_none() {
        // None must be treated as fix by the caller.
        assert_eq!(parse_intent("i am not sure"), None);
        assert_eq!(parse_intent(""), None);
    }

    #[test]
    fn test_parse_intent_handles_noisy_model_output() {
        // Real local-LLM outputs are rarely a bare word: leading labels,
        // punctuation, code fences, and trailing prose all appear in practice.
        assert_eq!(parse_intent("question\n"), Some(Intent::Question));
        assert_eq!(parse_intent("\"question\""), Some(Intent::Question));
        assert_eq!(parse_intent("- fix"), Some(Intent::FixRequest));
        assert_eq!(parse_intent("Answer: question"), Some(Intent::Question));
        assert_eq!(
            parse_intent("FIX\nbecause it reports a bug"),
            Some(Intent::FixRequest)
        );
    }

    #[test]
    fn test_parse_intent_first_clean_token_decides() {
        // A clean leading verdict short-circuits: the first exact "question"/"fix"
        // token wins immediately, before the both-present fix-bias fallback.
        assert_eq!(
            parse_intent("question, you could also fix the docs later"),
            Some(Intent::Question)
        );
        assert_eq!(
            parse_intent("fix the rest is noise"),
            Some(Intent::FixRequest)
        );
        // The fix-bias only applies when neither word leads cleanly.
        assert_eq!(
            parse_intent("this looks like a question but may need a fix"),
            Some(Intent::FixRequest)
        );
    }

    // --- Intent prompt construction (what the model actually sees) ---

    #[test]
    fn test_build_intent_prompt_embeds_title_and_instructions() {
        let issue = sample_issue("DISCORD-1", "what is query not equal syntax?");
        let prompt = build_intent_prompt(&issue);

        // The user's actual message must reach the model.
        assert!(prompt.contains("what is query not equal syntax?"));
        // The classification contract must be present and biased to fix.
        assert!(prompt.contains("\"question\""));
        assert!(prompt.contains("\"fix\""));
        assert!(prompt.contains("When in doubt, answer \"fix\"."));
        assert!(prompt.contains("ONLY the single word"));
    }

    #[test]
    fn test_build_intent_prompt_includes_distinct_description() {
        let mut issue = sample_issue("DISCORD-2", "Realtime onClose error");
        issue.description = Some(
            "triggerStats(): Argument #2 ($projectId) must be of type string, null given".into(),
        );
        let prompt = build_intent_prompt(&issue);

        assert!(prompt.contains("Realtime onClose error"));
        assert!(prompt.contains("triggerStats()"));
    }

    #[test]
    fn test_build_intent_prompt_omits_description_when_same_as_title() {
        // Avoid duplicating the title into the body (wastes context / confuses
        // the classifier).
        let mut issue = sample_issue("DISCORD-3", "how do I paginate results?");
        issue.description = Some("how do I paginate results?".into());
        let prompt = build_intent_prompt(&issue);

        // Title appears exactly once (in the `Title:` line), not duplicated.
        assert_eq!(prompt.matches("how do I paginate results?").count(), 1);
    }

    #[test]
    fn test_build_intent_prompt_truncates_long_description() {
        let mut issue = sample_issue("DISCORD-4", "long one");
        // Use 'Z', which never appears in the static prompt template, so the
        // count reflects only the (truncated) description body.
        issue.description = Some("Z".repeat(MAX_INTENT_DESC_CHARS * 4));
        let prompt = build_intent_prompt(&issue);

        let body_len = prompt.matches('Z').count();
        assert!(body_len > 0, "description missing from prompt");
        assert!(
            body_len <= MAX_INTENT_DESC_CHARS,
            "body not truncated: {body_len}"
        );
        assert!(!prompt.contains(&"Z".repeat(MAX_INTENT_DESC_CHARS * 4)));
    }

    // --- Assessment response parsing ---

    #[test]
    fn test_parse_assessment_response() {
        let json = r#"[
            {"id":"ISSUE-123","severity":0.95,"blast_radius":"critical","fingerprint":"null_ref_payment_charge"},
            {"id":"ISSUE-456","severity":0.2,"blast_radius":"cosmetic","fingerprint":"css_alignment_mobile"}
        ]"#;

        #[derive(serde::Deserialize)]
        #[allow(dead_code)]
        struct RawAssessment {
            id: String,
            severity: f64,
            blast_radius: String,
            fingerprint: String,
        }

        let raw: Option<Vec<RawAssessment>> = parse_json_response(json);
        assert!(raw.is_some());
        let raw = raw.unwrap();
        assert_eq!(raw.len(), 2);
        assert_eq!(raw[0].id, "ISSUE-123");
        assert!((raw[0].severity - 0.95).abs() < f64::EPSILON);
        assert_eq!(raw[0].blast_radius, "critical");
        assert_eq!(raw[1].id, "ISSUE-456");
    }

    #[test]
    fn test_parse_assessment_response_garbage() {
        #[derive(serde::Deserialize)]
        struct RawAssessment {
            #[allow(dead_code)]
            id: String,
        }

        let result: Option<Vec<RawAssessment>> =
            parse_json_response("I cannot assess these issues");
        assert!(result.is_none());
    }

    // --- Learnings response parsing ---

    #[test]
    fn test_parse_learnings_response() {
        let json = r#"{"root_cause":"missing null check","files_modified":["src/handler.rs"],"strategy":"tdd","tests_added":true,"summary":"Fixed null check with TDD"}"#;

        #[derive(serde::Deserialize)]
        struct RawLearnings {
            root_cause: Option<String>,
            files_modified: Vec<String>,
            strategy: Option<String>,
            tests_added: bool,
            summary: Option<String>,
        }

        let raw: Option<RawLearnings> = parse_json_response(json);
        assert!(raw.is_some());
        let raw = raw.unwrap();
        assert_eq!(raw.root_cause, Some("missing null check".to_string()));
        assert_eq!(raw.files_modified, vec!["src/handler.rs"]);
        assert_eq!(raw.strategy, Some("tdd".to_string()));
        assert!(raw.tests_added);
        assert!(raw.summary.unwrap().contains("TDD"));
    }

    // --- Truncation ---

    #[test]
    fn test_truncate_short_string() {
        assert_eq!(truncate("hello", 10), "hello");
    }

    #[test]
    fn test_truncate_long_string() {
        let result = truncate("a very long string here", 10);
        assert!(result.len() <= 13); // 10 chars + "..."
        assert!(result.ends_with("..."));
    }

    #[test]
    fn test_truncate_multibyte_utf8() {
        // Each emoji is 4 bytes. Truncating at byte boundary 5 would panic without safe handling.
        let emoji_str = "🚀🎉🔥💡✨";
        let result = truncate(emoji_str, 8);
        assert!(result.ends_with("..."));
        // Should not panic and should be valid UTF-8
        assert!(result.is_char_boundary(result.len()));

        // CJK characters (3 bytes each)
        let cjk = "你好世界测试";
        let result = truncate(cjk, 7);
        assert!(result.ends_with("..."));
    }

    // --- Live integration test ---

    #[test]
    fn test_live_llm_analyzer() {
        let model_path = match std::env::var("CLAUDEAR_LLM_MODEL_PATH") {
            Ok(p) => {
                let path = if let Some(rest) = p.strip_prefix("~/") {
                    std::env::var("HOME")
                        .map(|h| std::path::PathBuf::from(h).join(rest))
                        .unwrap_or_else(|_| std::path::PathBuf::from(&p))
                } else {
                    std::path::PathBuf::from(&p)
                };
                if !path.exists() {
                    eprintln!(
                        "CLAUDEAR_LLM_MODEL_PATH set but file not found: {}",
                        path.display()
                    );
                    return;
                }
                path
            }
            Err(_) => {
                eprintln!("Skipping live LLM analyzer test: CLAUDEAR_LLM_MODEL_PATH not set");
                return;
            }
        };

        let config = claudear_integrations::chat::llm::LlmConfig {
            model_path,
            context_length: 4096,
            gpu_layers: 99,
            threads: 0,
            timeout: Some(std::time::Duration::from_secs(120)),
        };
        let engine = Arc::new(
            claudear_integrations::chat::llm::LlmEngine::load(&config)
                .expect("Failed to load LLM model"),
        );
        let analyzer = LlmAnalyzerImpl::new(engine);

        // Test assess_issues
        let mut issue = sample_issue("ISSUE-123", "NullPointerException in PaymentService.charge");
        issue.metadata.insert(
            "error_type".to_string(),
            serde_json::Value::String("NullPointerException".to_string()),
        );
        issue.metadata.insert(
            "level".to_string(),
            serde_json::Value::String("fatal".to_string()),
        );

        let candidates = vec![(issue, sample_match_result())];
        let result = analyzer.assess_issues(&candidates);
        eprintln!("assess_issues result: {:?}", result);
        if let Some(assessments) = result {
            assert!(!assessments.is_empty());
            assert!(assessments[0].severity >= 0.0 && assessments[0].severity <= 1.0);
        }

        // Test classify_review
        let cat = analyzer.classify_review("This code has a SQL injection vulnerability");
        eprintln!("classify_review result: {:?}", cat);

        // Test extract_learnings
        let log = "Read src/main.rs\nThe bug was a missing null check\nEdit src/handler.rs\ncargo test\nAll tests passed";
        let analysis = analyzer.extract_learnings(log);
        eprintln!("extract_learnings result: {:?}", analysis);
    }
}
