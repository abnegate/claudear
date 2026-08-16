//! Agent-based classifiers.
//!
//! Uses the configured agent (claude/codex) for classification instead of the
//! local LLM. Much faster but costs API credits. Provides [`AgentRepoClassifier`]
//! (which repo an issue belongs to) and [`AgentIntentClassifier`] (bug/security
//! vs question/fix routing). Both use schema-constrained structured output
//! (`--json-schema`) so the reply is a guaranteed enum value — for the repo
//! classifier the enum is built dynamically from the candidate repo names.
//! Providers without structured-output support fall back to a freeform
//! completion parsed leniently.

use crate::intent::CATEGORY_RULES;
use crate::intent::{
    intent_body, intent_conversation_section, intent_title, parse_intent, Intent, IntentClassifier,
};
use async_trait::async_trait;
use claudear_analysis::inference::{ClassificationRequest, RepoClassifier};
use claudear_core::types::Issue;
use claudear_integrations::runner::AgentRunner;
use std::sync::Arc;
use std::time::Instant;

/// Agent-based repository classifier.
pub struct AgentRepoClassifier {
    agent: Arc<dyn AgentRunner>,
}

impl AgentRepoClassifier {
    /// Create a new classifier with the given agent runner.
    pub fn new(agent: Arc<dyn AgentRunner>) -> Self {
        Self { agent }
    }
}

impl RepoClassifier for AgentRepoClassifier {
    fn classify(&self, request: &ClassificationRequest) -> Option<(String, f32)> {
        let prompt = build_prompt(request);
        let temp_dir = std::env::temp_dir();
        let candidate_names: Vec<&str> =
            request.candidates.iter().map(|(n, _)| n.as_str()).collect();

        let start = Instant::now();

        // Preferred path: schema-constrained structured output. The reply is
        // guaranteed to be one of the candidate repo names (or "NONE"), so no
        // brittle text parsing is needed and we avoid spinning up a full
        // fix-agent session. `block_in_place` bridges the async call into this
        // sync trait method (avoids "runtime within a runtime" on the watcher).
        let schema = build_repo_schema(&candidate_names);
        let structured = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                self.agent
                    .structured_query(&prompt, &schema, &temp_dir)
                    .await
            })
        });

        match structured {
            Ok(value) => {
                tracing::debug!(
                    response = %value,
                    elapsed_ms = start.elapsed().as_millis(),
                    "Agent classifier structured response"
                );
                return parse_structured_response(&value, &candidate_names);
            }
            Err(e) => {
                // Providers without `--json-schema` support fall back to a
                // freeform completion parsed leniently below.
                tracing::debug!(
                    error = %e,
                    "structured_query unavailable, falling back to freeform classification"
                );
            }
        }

        let freeform = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                self.agent
                    .execute_with_attempt(&prompt, None, None, &temp_dir)
                    .await
            })
        });

        match freeform {
            Ok(agent_result) => {
                let response = agent_result.output.trim().to_string();
                tracing::debug!(
                    response = %response,
                    elapsed_ms = start.elapsed().as_millis(),
                    "Agent classifier freeform response"
                );
                parse_response(&response, &candidate_names)
            }
            Err(e) => {
                tracing::warn!(error = %e, "Agent classifier execution failed");
                None
            }
        }
    }
}

/// Build a JSON schema constraining the reply to exactly one of the candidate
/// repo names, or `"NONE"`. Built via `serde_json` so repo names are escaped
/// correctly.
fn build_repo_schema(candidates: &[&str]) -> String {
    let mut enum_values: Vec<String> = candidates.iter().map(|s| s.to_string()).collect();
    enum_values.push("NONE".to_string());
    serde_json::json!({
        "type": "object",
        "required": ["repo"],
        "additionalProperties": false,
        "properties": {
            "repo": {
                "type": "string",
                "enum": enum_values,
                "description": "The exact repository this issue belongs to, or \"NONE\" if no candidate matches."
            }
        }
    })
    .to_string()
}

/// Extract the chosen repo from a schema-constrained structured response.
///
/// Returns `None` only for a missing/empty field or an explicit `"NONE"` — the
/// cases where the model deliberately declined. For any other value we reuse the
/// freeform matcher [`parse_response`], which recovers a present-but-imperfect
/// value in place (exact → 1.0, case-insensitive → 0.9, contains → 0.7) without
/// needing a second, expensive agent round-trip. This means a provider that
/// doesn't perfectly honour the enum (e.g. a backtick-wrapped or verbose `repo`
/// field) still gets classified rather than silently dropped.
fn parse_structured_response(
    value: &serde_json::Value,
    candidates: &[&str],
) -> Option<(String, f32)> {
    let repo = value.get("repo").and_then(|v| v.as_str())?.trim();
    if repo.is_empty() || repo.eq_ignore_ascii_case("none") {
        return None;
    }
    parse_response(repo, candidates)
}

/// Build the classification prompt (plain text, no model-specific tokens).
fn build_prompt(request: &ClassificationRequest) -> String {
    let mut parts: Vec<String> = Vec::new();

    parts.push(
        "You are a code repository classifier. Given an issue with its full context and a list of \
         candidate repositories with their profiles, determine which repository the issue belongs to.\n\
         Respond with ONLY the exact repository name (e.g. \"org/repo\"). If none match, respond \"NONE\"."
            .to_string(),
    );

    // Issue section
    parts.push("\n## Issue".to_string());
    parts.push(format!("Title: {}", request.title));
    parts.push(format!("Source: {}", request.source));
    if let Some(ref desc) = request.description {
        let truncated = if desc.len() > 500 {
            format!("{}...", &desc[..500])
        } else {
            desc.clone()
        };
        parts.push(format!("Description: {}", truncated));
    }

    // Metadata
    let metadata_keys = [
        "stacktrace",
        "culprit",
        "filename",
        "function",
        "project",
        "message",
    ];
    for key in &metadata_keys {
        if let Some(val) = request.metadata.get(*key) {
            let display_val = if *key == "stacktrace" && val.len() > 500 {
                format!("{}...", &val[..500])
            } else {
                val.clone()
            };
            parts.push(format!("{}: {}", key, display_val));
        }
    }

    // Extracted signals
    parts.push("\n## Extracted Signals".to_string());
    if !request.extracted_filenames.is_empty() {
        parts.push(format!(
            "Files referenced: {}",
            request.extracted_filenames.join(", ")
        ));
    }
    if !request.extracted_functions.is_empty() {
        parts.push(format!(
            "Functions referenced: {}",
            request.extracted_functions.join(", ")
        ));
    }
    if !request.extracted_keywords.is_empty() {
        parts.push(format!(
            "Keywords: {}",
            request.extracted_keywords.join(", ")
        ));
    }
    if !request.extracted_repos.is_empty() {
        parts.push(format!(
            "Referenced repos: {}",
            request.extracted_repos.join(", ")
        ));
    }

    // Candidate repositories
    parts.push("\n## Candidate Repositories".to_string());
    for (i, (name, profile)) in request.candidates.iter().enumerate() {
        parts.push(format!("\n### {}. {}", i + 1, name));
        parts.push(profile.clone());
    }

    parts.push("\nWhich repository does this issue belong to?".to_string());

    parts.join("\n")
}

/// Parse the agent response to extract a repo name and confidence.
fn parse_response(response: &str, candidates: &[&str]) -> Option<(String, f32)> {
    let trimmed = response.trim();

    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("none") {
        return None;
    }

    // Exact match
    for &candidate in candidates {
        if trimmed == candidate {
            return Some((candidate.to_string(), 1.0));
        }
    }

    // Case-insensitive match
    let lower = trimmed.to_lowercase();
    for &candidate in candidates {
        if lower == candidate.to_lowercase() {
            return Some((candidate.to_string(), 0.9));
        }
    }

    // Contains match (response contains a candidate name)
    for &candidate in candidates {
        if lower.contains(&candidate.to_lowercase()) {
            return Some((candidate.to_string(), 0.7));
        }
    }

    None
}

/// Schema constraining the agent's reply to one of the four intent categories.
const INTENT_SCHEMA: &str = r#"{
    "type": "object",
    "required": ["intent"],
    "additionalProperties": false,
    "properties": {
        "intent": {
            "type": "string",
            "enum": ["bug", "security", "question", "fix"],
            "description": "The single category the message belongs to"
        }
    }
}"#;

/// Agent-based intent classifier (Claude Code / configured provider).
///
/// Uses `--json-schema` constrained decoding via [`AgentRunner::structured_query`]
/// so the reply is a guaranteed one of `bug | security | question | fix`. Returns
/// `None` if the provider doesn't support structured output or the call fails, so
/// the caller falls back to the heuristic.
pub struct AgentIntentClassifier {
    agent: Arc<dyn AgentRunner>,
}

impl AgentIntentClassifier {
    /// Create a new intent classifier with the given agent runner.
    pub fn new(agent: Arc<dyn AgentRunner>) -> Self {
        Self { agent }
    }
}

#[async_trait]
impl IntentClassifier for AgentIntentClassifier {
    async fn classify_intent(&self, issue: &Issue, conversation: Option<&str>) -> Option<Intent> {
        let prompt = build_intent_prompt(issue, conversation);
        let temp_dir = std::env::temp_dir();

        let start = Instant::now();
        let result = self
            .agent
            .structured_query(&prompt, INTENT_SCHEMA, &temp_dir)
            .await;

        match result {
            Ok(value) => {
                tracing::debug!(
                    response = %value,
                    elapsed_ms = start.elapsed().as_millis(),
                    "Agent intent classifier structured response"
                );
                value
                    .get("intent")
                    .and_then(|v| v.as_str())
                    .and_then(parse_intent)
            }
            Err(e) => {
                tracing::warn!(error = %e, "Agent intent classifier failed");
                None
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Retrieval-quality relevance judge (agent backend)
// ---------------------------------------------------------------------------

/// JSON schema constraining the agent's relevance judgment to a single number.
const RELEVANCE_SCHEMA: &str = r#"{
    "type": "object",
    "required": ["relevance"],
    "additionalProperties": false,
    "properties": {
        "relevance": {
            "type": "number",
            "description": "How relevant the snippet is for resolving the issue, from 0.0 (irrelevant) to 1.0 (directly relevant)"
        }
    }
}"#;

/// Build the shared ISSUE/SNIPPET context block for the retrieval-quality
/// relevance judge, clipped to keep prompts bounded. Shared by both the
/// agent backend ([`score_chunk_relevance_via_agent`]) and the local-LLM
/// backend ([`crate::llm_analyzer::LlmAnalyzerImpl::score_chunk_relevance`]).
pub fn build_relevance_context(issue_summary: &str, chunk_text: &str) -> String {
    fn clip(s: &str, n: usize) -> String {
        s.chars().take(n).collect()
    }
    format!(
        "Rate how relevant the SNIPPET is for resolving the ISSUE, from 0.0 \
         (irrelevant) to 1.0 (directly relevant).\nISSUE:\n{}\n\nSNIPPET:\n{}",
        clip(issue_summary, 1200),
        clip(chunk_text, 1500)
    )
}

/// Score chunk relevance (0.0-1.0) via the **coding agent** (external provider)
/// using schema-constrained structured output. Used by the opt-in
/// retrieval-quality judge when `agent.use_llm` is false. Returns `None` if the
/// provider doesn't support structured output or the call fails.
pub async fn score_chunk_relevance_via_agent(
    agent: &dyn AgentRunner,
    issue_summary: &str,
    chunk_text: &str,
) -> Option<f64> {
    let prompt = format!(
        "{}\n\nSet `relevance` to a number between 0.0 and 1.0.",
        build_relevance_context(issue_summary, chunk_text)
    );
    let temp_dir = std::env::temp_dir();
    match agent
        .structured_query(&prompt, RELEVANCE_SCHEMA, &temp_dir)
        .await
    {
        Ok(value) => value
            .get("relevance")
            .and_then(|v| v.as_f64())
            .map(|v| v.clamp(0.0, 1.0)),
        Err(e) => {
            tracing::warn!(error = %e, "Agent retrieval relevance judge failed");
            None
        }
    }
}

/// Build the intent-classification prompt for the coding agent (plain text; the
/// `--json-schema` result shape is enforced by constrained decoding, not prose).
fn build_intent_prompt(issue: &Issue, conversation: Option<&str>) -> String {
    format!(
        "You classify an incoming developer-support message into exactly one of:\n\
         {rules}\n\
         {conversation}\
         Set `intent` to the single matching category.\n\n\
         Title: {title}\n\
         {body}",
        rules = CATEGORY_RULES,
        conversation = intent_conversation_section(conversation),
        title = intent_title(issue),
        body = intent_body(issue),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use claudear_core::types::AgentResult;
    use claudear_integrations::runner::ProviderCapabilities;
    use std::collections::HashMap;
    use std::path::Path;

    struct MockAgent {
        response: Result<AgentResult, String>,
    }

    impl MockAgent {
        fn with_output(output: &str) -> Self {
            Self {
                response: Ok(AgentResult {
                    success: true,
                    output: output.to_string(),
                    pr_url: None,
                    changelog: None,
                    error: None,
                    blocking_question: None,
                    used_qa_ids: vec![],
                    confidence: 80,
                    confidence_reasoning: None,
                    wrong_repo: None,
                }),
            }
        }

        fn with_error() -> Self {
            Self {
                response: Err("mock error".to_string()),
            }
        }
    }

    #[async_trait]
    impl AgentRunner for MockAgent {
        fn name(&self) -> &str {
            "mock"
        }

        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::default()
        }

        fn build_prompt_for_issue(
            &self,
            _issue: &Issue,
            _context: &str,
            _project_dir: &Path,
        ) -> String {
            String::new()
        }

        async fn execute_with_attempt(
            &self,
            _prompt: &str,
            _issue: Option<&Issue>,
            _attempt_id: Option<i64>,
            _project_dir: &Path,
        ) -> claudear_core::error::Result<AgentResult> {
            match &self.response {
                Ok(r) => Ok(r.clone()),
                Err(e) => Err(claudear_core::error::Error::runner(e)),
            }
        }
    }

    fn sample_request() -> ClassificationRequest {
        ClassificationRequest {
            title: "MySQL server has gone away".to_string(),
            description: Some("Connection lost during query execution".to_string()),
            source: "sentry".to_string(),
            metadata: {
                let mut m = HashMap::new();
                m.insert("project".to_string(), "cloud-staging".to_string());
                m
            },
            extracted_filenames: vec!["src/Database/Adapter/SQL.php".to_string()],
            extracted_functions: vec!["query".to_string()],
            extracted_keywords: vec!["MySQL".to_string()],
            extracted_repos: vec![],
            candidates: vec![
                ("appwrite/cloud".to_string(), "Cloud backend".to_string()),
                (
                    "utopia-php/database".to_string(),
                    "Database abstraction".to_string(),
                ),
            ],
        }
    }

    #[test]
    fn test_agent_classifier_builds_prompt() {
        let request = sample_request();
        let prompt = build_prompt(&request);

        assert!(prompt.contains("MySQL server has gone away"));
        assert!(prompt.contains("appwrite/cloud"));
        assert!(prompt.contains("utopia-php/database"));
        assert!(prompt.contains("src/Database/Adapter/SQL.php"));
        assert!(prompt.contains("cloud-staging"));
    }

    #[test]
    fn test_agent_classifier_parses_exact_match() {
        let candidates = vec!["appwrite/cloud", "utopia-php/database"];
        let result = parse_response("appwrite/cloud", &candidates);
        assert_eq!(result, Some(("appwrite/cloud".to_string(), 1.0)));
    }

    #[test]
    fn test_agent_classifier_parses_case_insensitive() {
        let candidates = vec!["appwrite/cloud", "utopia-php/database"];
        let result = parse_response("Appwrite/Cloud", &candidates);
        assert_eq!(result, Some(("appwrite/cloud".to_string(), 0.9)));
    }

    #[test]
    fn test_agent_classifier_handles_none_response() {
        let candidates = vec!["appwrite/cloud", "utopia-php/database"];
        let result = parse_response("NONE", &candidates);
        assert_eq!(result, None);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_agent_classifier_handles_agent_error() {
        let mock = MockAgent::with_error();
        let classifier = AgentRepoClassifier::new(Arc::new(mock));
        let request = sample_request();
        let result = classifier.classify(&request);
        assert!(result.is_none());
    }

    #[test]
    fn test_agent_classifier_prompt_excludes_model_tokens() {
        let request = sample_request();
        let prompt = build_prompt(&request);

        assert!(!prompt.contains("<|system|>"));
        assert!(!prompt.contains("<|user|>"));
        assert!(!prompt.contains("<|assistant|>"));
        assert!(!prompt.contains("<|end|>"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_agent_classifier_parses_output() {
        let mock = MockAgent::with_output("appwrite/cloud");
        let classifier = AgentRepoClassifier::new(Arc::new(mock));
        let request = sample_request();
        let result = classifier.classify(&request);
        assert_eq!(result, Some(("appwrite/cloud".to_string(), 1.0)));
    }

    #[test]
    fn test_build_repo_schema_lists_candidates_and_none() {
        let schema = build_repo_schema(&["appwrite/cloud", "utopia-php/database"]);
        let parsed: serde_json::Value = serde_json::from_str(&schema).expect("valid JSON");
        let enum_vals = parsed["properties"]["repo"]["enum"]
            .as_array()
            .expect("enum array");
        let names: Vec<&str> = enum_vals.iter().filter_map(|v| v.as_str()).collect();
        assert!(names.contains(&"appwrite/cloud"));
        assert!(names.contains(&"utopia-php/database"));
        assert!(names.contains(&"NONE"));
    }

    #[test]
    fn test_parse_structured_response_matches_and_rejects() {
        let candidates = vec!["appwrite/cloud", "utopia-php/database"];
        // Exact enum value → canonical name, full confidence.
        let v = serde_json::json!({ "repo": "utopia-php/database" });
        assert_eq!(
            parse_structured_response(&v, &candidates),
            Some(("utopia-php/database".to_string(), 1.0))
        );
        // Case-insensitive maps back to the canonical candidate at reduced
        // confidence (the value didn't match the enum verbatim).
        let v = serde_json::json!({ "repo": "Appwrite/Cloud" });
        assert_eq!(
            parse_structured_response(&v, &candidates),
            Some(("appwrite/cloud".to_string(), 0.9))
        );
        // Present-but-imperfect value (enum not honoured verbatim) is recovered
        // in place via the contains-match tier rather than dropped.
        let v = serde_json::json!({ "repo": "`utopia-php/database`" });
        assert_eq!(
            parse_structured_response(&v, &candidates),
            Some(("utopia-php/database".to_string(), 0.7))
        );
        // Explicit NONE and missing field → no match.
        assert_eq!(
            parse_structured_response(&serde_json::json!({ "repo": "NONE" }), &candidates),
            None
        );
        assert_eq!(
            parse_structured_response(&serde_json::json!({}), &candidates),
            None
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_agent_classifier_uses_structured_output() {
        let mock = StructuredMock {
            response: Ok(serde_json::json!({ "repo": "utopia-php/database" })),
        };
        let classifier = AgentRepoClassifier::new(Arc::new(mock));
        let result = classifier.classify(&sample_request());
        assert_eq!(result, Some(("utopia-php/database".to_string(), 1.0)));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_agent_classifier_structured_none_returns_none() {
        let mock = StructuredMock {
            response: Ok(serde_json::json!({ "repo": "NONE" })),
        };
        let classifier = AgentRepoClassifier::new(Arc::new(mock));
        assert!(classifier.classify(&sample_request()).is_none());
    }

    // --- Intent classifier ---

    /// Mock agent whose `structured_query` returns a fixed JSON value (or errors).
    struct StructuredMock {
        response: Result<serde_json::Value, String>,
    }

    #[async_trait]
    impl AgentRunner for StructuredMock {
        fn name(&self) -> &str {
            "structured-mock"
        }
        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::default()
        }
        fn build_prompt_for_issue(&self, _: &Issue, _: &str, _: &Path) -> String {
            String::new()
        }
        async fn execute_with_attempt(
            &self,
            _prompt: &str,
            _issue: Option<&Issue>,
            _attempt_id: Option<i64>,
            _project_dir: &Path,
        ) -> claudear_core::error::Result<AgentResult> {
            Err(claudear_core::error::Error::runner("unused"))
        }
        async fn structured_query(
            &self,
            _prompt: &str,
            _json_schema: &str,
            _project_dir: &Path,
        ) -> claudear_core::error::Result<serde_json::Value> {
            match &self.response {
                Ok(v) => Ok(v.clone()),
                Err(e) => Err(claudear_core::error::Error::runner(e)),
            }
        }
    }

    fn intent_issue(title: &str, description: Option<&str>) -> Issue {
        use claudear_core::types::{IssuePriority, IssueStatus};
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
    fn test_intent_prompt_is_plain_text_with_contract() {
        let issue = intent_issue("Realtime onClose error", Some("triggerStats() null given"));
        let prompt = build_intent_prompt(&issue, None);

        assert!(prompt.contains("Realtime onClose error"));
        assert!(prompt.contains("triggerStats()"));
        assert!(prompt.contains("\"bug\""));
        assert!(prompt.contains("Set `intent`"));
        // No local-LLM control tokens leak into an external-agent prompt.
        assert!(!prompt.contains("<|system|>"));
        assert!(!prompt.contains("<|assistant|>"));
    }

    #[test]
    fn test_intent_prompt_includes_conversation_when_present() {
        let issue = intent_issue("yes create a pr now", None);
        let convo = "[User]: can you add dedicated scopes?\n[Claudear]: here is the plan ...";
        let prompt = build_intent_prompt(&issue, Some(convo));

        assert!(prompt.contains("can you add dedicated scopes?"));
        assert!(prompt.contains("Classify the LATEST message"));
        // The current message is still present and classified.
        assert!(prompt.contains("yes create a pr now"));

        // Absent conversation leaves the prompt unchanged (no dangling framing).
        let plain = build_intent_prompt(&issue, None);
        assert!(!plain.contains("Classify the LATEST message"));
        assert!(!plain.contains("ongoing conversation"));
    }

    #[test]
    fn test_intent_schema_is_valid_json_with_enum() {
        let schema: serde_json::Value = serde_json::from_str(INTENT_SCHEMA).unwrap();
        let variants = schema["properties"]["intent"]["enum"].as_array().unwrap();
        assert_eq!(variants.len(), 4);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_agent_intent_classifier_parses_structured_value() {
        let classifier = AgentIntentClassifier::new(Arc::new(StructuredMock {
            response: Ok(serde_json::json!({ "intent": "security" })),
        }));
        let issue = intent_issue("SQL injection in login", None);
        assert_eq!(
            classifier.classify_intent(&issue, None).await,
            Some(Intent::Security)
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_agent_intent_classifier_error_returns_none() {
        let classifier = AgentIntentClassifier::new(Arc::new(StructuredMock {
            response: Err("not supported".to_string()),
        }));
        let issue = intent_issue("anything", None);
        assert_eq!(classifier.classify_intent(&issue, None).await, None);
    }

    // --- Retrieval relevance judge (agent backend) ---

    #[test]
    fn test_relevance_schema_is_valid_json_with_number() {
        let schema: serde_json::Value = serde_json::from_str(RELEVANCE_SCHEMA).unwrap();
        assert_eq!(schema["properties"]["relevance"]["type"], "number");
    }

    #[test]
    fn test_build_relevance_context_labels_and_clips() {
        let ctx = build_relevance_context("the issue", "the snippet");
        assert!(ctx.contains("ISSUE:\nthe issue"));
        assert!(ctx.contains("SNIPPET:\nthe snippet"));
        let long = "x".repeat(5000);
        let ctx = build_relevance_context(&long, &long);
        // issue clipped to 1200, snippet to 1500 (+ labels/prose)
        assert!(ctx.len() < 3200);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_score_chunk_relevance_via_agent() {
        let agent = StructuredMock {
            response: Ok(serde_json::json!({ "relevance": 0.73 })),
        };
        assert_eq!(
            score_chunk_relevance_via_agent(&agent, "issue", "snippet").await,
            Some(0.73)
        );

        // Out-of-range values are clamped.
        let agent = StructuredMock {
            response: Ok(serde_json::json!({ "relevance": 2.0 })),
        };
        assert_eq!(
            score_chunk_relevance_via_agent(&agent, "i", "s").await,
            Some(1.0)
        );

        // Provider without structured output -> None (caller leaves score unset).
        let agent = StructuredMock {
            response: Err("not supported".to_string()),
        };
        assert_eq!(
            score_chunk_relevance_via_agent(&agent, "i", "s").await,
            None
        );
    }
}
