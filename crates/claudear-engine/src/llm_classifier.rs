//! Local-LLM-based classifiers.
//!
//! Uses a local LLM (via llama-cpp-2) for classification. Provides
//! [`LlmRepoClassifier`] (which repo an issue belongs to) and
//! [`LocalLlmIntentClassifier`] (bug/security vs question/fix routing). Both run
//! freeform completions against the model and parse the reply — no structured
//! output.

use crate::intent::CATEGORY_RULES;
use crate::intent::{
    intent_body, intent_conversation_section, intent_title, parse_intent, Intent, IntentClassifier,
};
use async_trait::async_trait;
use claudear_analysis::inference::{ClassificationRequest, RepoClassifier};
use claudear_core::types::Issue;
use claudear_integrations::chat::llm::{GenerationParams, LlmEngine};
use std::sync::Arc;
use std::time::Instant;

/// Maximum total prompt size in characters (~16000 tokens at ~3 chars/token).
const MAX_PROMPT_CHARS: usize = 48000;

/// LLM-based repository classifier.
pub struct LlmRepoClassifier {
    engine: Arc<LlmEngine>,
}

impl LlmRepoClassifier {
    /// Create a new classifier with the given LLM engine.
    pub fn new(engine: Arc<LlmEngine>) -> Self {
        Self { engine }
    }
}

impl RepoClassifier for LlmRepoClassifier {
    fn classify(&self, request: &ClassificationRequest) -> Option<(String, f32)> {
        let prompt = build_prompt(request);
        let params = GenerationParams {
            temperature: 0.1,
            max_tokens: 64,
            top_p: 0.9,
            stop_sequences: vec![
                "\n".to_string(),
                "<|end|>".to_string(),
                "<|user|>".to_string(),
            ],
        };

        let start = Instant::now();
        let tokens = match self.engine.complete_streaming(&prompt, &params) {
            Ok(t) => t,
            Err(e) => {
                tracing::warn!(error = %e, "LLM classifier inference failed");
                return None;
            }
        };

        let response = tokens.join("");
        let elapsed = start.elapsed();

        tracing::debug!(
            response = %response.trim(),
            elapsed_ms = elapsed.as_millis(),
            "LLM classifier raw response"
        );

        let candidate_names: Vec<&str> =
            request.candidates.iter().map(|(n, _)| n.as_str()).collect();
        parse_response(&response, &candidate_names)
    }
}

/// Build the classification prompt from the request.
fn build_prompt(request: &ClassificationRequest) -> String {
    let mut sections: Vec<String> = Vec::new();

    // System message
    sections.push(
        "<|system|>\n\
         You are a code repository classifier. Given an issue with its full context and a list of \
         candidate repositories with their profiles, determine which repository the issue belongs to.\n\
         Respond with ONLY the exact repository name (e.g. \"org/repo\"). If none match, respond \"NONE\".\n\
         <|end|>"
            .to_string(),
    );

    // User message
    let mut user_parts: Vec<String> = Vec::new();
    user_parts.push("<|user|>".to_string());

    // Issue section
    user_parts.push("## Issue".to_string());
    user_parts.push(format!("Title: {}", request.title));
    user_parts.push(format!("Source: {}", request.source));
    if let Some(ref desc) = request.description {
        let truncated = if desc.len() > 500 {
            format!("{}...", &desc[..500])
        } else {
            desc.clone()
        };
        user_parts.push(format!("Description: {}", truncated));
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
            user_parts.push(format!("{}: {}", key, display_val));
        }
    }

    // Extracted signals
    user_parts.push("\n## Extracted Signals".to_string());
    if !request.extracted_filenames.is_empty() {
        user_parts.push(format!(
            "Files referenced: {}",
            request.extracted_filenames.join(", ")
        ));
    }
    if !request.extracted_functions.is_empty() {
        user_parts.push(format!(
            "Functions referenced: {}",
            request.extracted_functions.join(", ")
        ));
    }
    if !request.extracted_keywords.is_empty() {
        user_parts.push(format!(
            "Keywords: {}",
            request.extracted_keywords.join(", ")
        ));
    }
    if !request.extracted_repos.is_empty() {
        user_parts.push(format!(
            "Referenced repos: {}",
            request.extracted_repos.join(", ")
        ));
    }

    // Candidate repositories
    user_parts.push("\n## Candidate Repositories".to_string());
    for (i, (name, profile)) in request.candidates.iter().enumerate() {
        user_parts.push(format!("\n### {}. {}", i + 1, name));
        user_parts.push(profile.clone());
    }

    user_parts.push("\nWhich repository does this issue belong to?".to_string());
    user_parts.push("<|end|>".to_string());

    // Assistant prefix
    user_parts.push("<|assistant|>".to_string());

    sections.push(user_parts.join("\n"));

    let prompt = sections.join("\n");
    truncate_prompt(&prompt, MAX_PROMPT_CHARS)
}

/// Parse the LLM response to extract a repo name and confidence.
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

/// Truncate an oversized prompt to fit within the context window.
///
/// Truncates sample files sections first, then README excerpts, then
/// stacktrace sections to reduce size while preserving the most important context.
fn truncate_prompt(prompt: &str, max_chars: usize) -> String {
    if prompt.len() <= max_chars {
        return prompt.to_string();
    }

    // Simple truncation: cut from the end of the candidate repos section,
    // preserving the issue context and assistant prompt suffix.
    // Find the assistant tag and keep it
    let assistant_tag = "<|assistant|>";

    if let Some(assistant_pos) = prompt.rfind(assistant_tag) {
        // Keep the last part (question + end + assistant)
        let suffix_start = prompt[..assistant_pos]
            .rfind("\nWhich repository")
            .unwrap_or(assistant_pos.saturating_sub(100));
        let suffix = &prompt[suffix_start..];
        let available = max_chars.saturating_sub(suffix.len());

        if available > 0 && available < prompt.len() {
            // Truncate at a newline boundary
            let truncated = &prompt[..available];
            let cut_at = truncated.rfind('\n').unwrap_or(available);
            return format!("{}{}", &prompt[..cut_at], suffix);
        }
    }

    // Fallback: simple truncation
    prompt[..max_chars].to_string()
}

/// Local-LLM-backed intent classifier (offline GGUF model).
///
/// Runs a freeform one-word completion and parses it — no structured output.
/// The inference is synchronous and CPU-bound, so it is offloaded to a blocking
/// thread to avoid stalling the async runtime.
pub struct LocalLlmIntentClassifier {
    engine: Arc<LlmEngine>,
}

impl LocalLlmIntentClassifier {
    pub fn new(engine: Arc<LlmEngine>) -> Self {
        Self { engine }
    }
}

#[async_trait]
impl IntentClassifier for LocalLlmIntentClassifier {
    async fn classify_intent(&self, issue: &Issue, conversation: Option<&str>) -> Option<Intent> {
        let engine = self.engine.clone();
        let issue = issue.clone();
        let conversation = conversation.map(str::to_owned);
        tokio::task::spawn_blocking(move || {
            classify_intent_blocking(&engine, &issue, conversation.as_deref())
        })
        .await
        .unwrap_or(None)
    }
}

/// Synchronous local-LLM intent classification.
fn classify_intent_blocking(
    engine: &LlmEngine,
    issue: &Issue,
    conversation: Option<&str>,
) -> Option<Intent> {
    let prompt = build_intent_prompt(issue, conversation);
    let params = GenerationParams {
        temperature: 0.1,
        max_tokens: 8,
        top_p: 0.9,
        stop_sequences: vec![
            "\n\n".to_string(),
            "<|end|>".to_string(),
            "<|user|>".to_string(),
        ],
    };

    let start = Instant::now();
    let tokens = match engine.complete_streaming(&prompt, &params) {
        Ok(t) => t,
        Err(e) => {
            tracing::warn!(error = %e, "Local LLM intent inference failed");
            return None;
        }
    };
    let response = tokens.join("");
    tracing::debug!(
        response = %response.trim(),
        elapsed_ms = start.elapsed().as_millis(),
        "Local LLM intent completion"
    );
    parse_intent(&response)
}

/// Build the intent-classification prompt using the model's chat control tokens.
fn build_intent_prompt(issue: &Issue, conversation: Option<&str>) -> String {
    format!(
        "<|system|>\n\
         You classify an incoming developer-support message into exactly one of:\n\
         {rules}\n\
         Respond with ONLY one word: bug, security, question, or fix.\n\
         <|end|>\n\
         <|user|>\n\
         {conversation}\
         Title: {title}\n\
         {body}\n\
         <|end|>\n\
         <|assistant|>",
        rules = CATEGORY_RULES,
        conversation = intent_conversation_section(conversation),
        title = intent_title(issue),
        body = intent_body(issue),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn sample_issue(title: &str, description: Option<&str>) -> Issue {
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
    fn test_build_intent_prompt_uses_control_tokens_and_contract() {
        let issue = sample_issue("what is query not equal syntax?", None);
        let prompt = build_intent_prompt(&issue, None);

        assert!(prompt.contains("what is query not equal syntax?"));
        assert!(prompt.contains("\"bug\""));
        assert!(prompt.contains("\"security\""));
        assert!(prompt.contains("\"question\""));
        assert!(prompt.contains("\"fix\""));
        assert!(prompt.contains("ONLY one word"));
        // Local prompt uses the model's chat control tokens.
        assert!(prompt.contains("<|system|>"));
        assert!(prompt.contains("<|assistant|>"));
    }

    #[test]
    fn test_build_intent_prompt_omits_duplicate_description() {
        let issue = sample_issue("how do I paginate?", Some("how do I paginate?"));
        let prompt = build_intent_prompt(&issue, None);
        assert_eq!(prompt.matches("how do I paginate?").count(), 1);
    }

    #[test]
    fn test_build_intent_prompt_includes_conversation() {
        let issue = sample_issue("yes create a pr now", None);
        let convo = "[User]: can you add dedicated scopes?\n[Claudear]: here is the plan";
        let prompt = build_intent_prompt(&issue, Some(convo));
        assert!(prompt.contains("can you add dedicated scopes?"));
        assert!(prompt.contains("Classify the LATEST message"));
        assert!(prompt.contains("yes create a pr now"));
    }

    fn sample_request() -> ClassificationRequest {
        ClassificationRequest {
            title: "MySQL server has gone away".to_string(),
            description: Some("Connection lost during query execution".to_string()),
            source: "sentry".to_string(),
            metadata: {
                let mut m = HashMap::new();
                m.insert("project".to_string(), "cloud-staging".to_string());
                m.insert("culprit".to_string(), "Database\\Adapter\\SQL::query".to_string());
                m
            },
            extracted_filenames: vec!["src/Database/Adapter/SQL.php".to_string()],
            extracted_functions: vec!["query".to_string()],
            extracted_keywords: vec!["MySQL".to_string(), "connection".to_string()],
            extracted_repos: vec![],
            candidates: vec![
                (
                    "appwrite/cloud".to_string(),
                    "Repository: appwrite/cloud\nDescription: Cloud backend\nLanguages: 45 php, 30 js\nDirectories: src, app, tests".to_string(),
                ),
                (
                    "utopia-php/database".to_string(),
                    "Repository: utopia-php/database\nDescription: Database abstraction layer\nLanguages: 100 php\nDirectories: src, tests\nSample files: src/Database/Adapter/SQL.php".to_string(),
                ),
            ],
        }
    }

    #[test]
    fn test_build_prompt() {
        let request = sample_request();
        let prompt = build_prompt(&request);

        assert!(prompt.contains("MySQL server has gone away"));
        assert!(prompt.contains("Connection lost"));
        assert!(prompt.contains("cloud-staging"));
        assert!(prompt.contains("src/Database/Adapter/SQL.php"));
        assert!(prompt.contains("appwrite/cloud"));
        assert!(prompt.contains("utopia-php/database"));
        assert!(prompt.contains("<|system|>"));
        assert!(prompt.contains("<|assistant|>"));
        assert!(prompt.contains("query"));
    }

    #[test]
    fn test_build_prompt_with_stacktrace() {
        let mut request = sample_request();
        let long_stacktrace = "a".repeat(1000);
        request
            .metadata
            .insert("stacktrace".to_string(), long_stacktrace);

        let prompt = build_prompt(&request);

        // Stacktrace should be truncated to ~500 chars + "..."
        assert!(prompt.contains("stacktrace:"));
        // The full 1000-char stacktrace should not appear
        assert!(!prompt.contains(&"a".repeat(1000)));
    }

    #[test]
    fn test_build_prompt_truncation() {
        let mut request = sample_request();
        // Add many candidates to make the prompt huge
        for i in 0..100 {
            request.candidates.push((
                format!("org/repo-{}", i),
                format!(
                    "Repository: org/repo-{}\nDescription: {}\nLanguages: rust\nDirectories: src\nSample files: {}",
                    i,
                    "x".repeat(200),
                    "y".repeat(200),
                ),
            ));
        }

        let prompt = build_prompt(&request);

        // Should be truncated to MAX_PROMPT_CHARS
        assert!(prompt.len() <= MAX_PROMPT_CHARS + 200); // allow some slack for suffix re-attach
                                                         // Should still contain the assistant tag
        assert!(prompt.contains("<|assistant|>"));
    }

    #[test]
    fn test_parse_response_exact_match() {
        let candidates = vec!["appwrite/cloud", "utopia-php/database"];
        let result = parse_response("utopia-php/database", &candidates);
        assert_eq!(result, Some(("utopia-php/database".to_string(), 1.0)));
    }

    #[test]
    fn test_parse_response_case_insensitive() {
        let candidates = vec!["appwrite/cloud", "utopia-php/database"];
        let result = parse_response("Appwrite/Cloud", &candidates);
        assert_eq!(result, Some(("appwrite/cloud".to_string(), 0.9)));
    }

    #[test]
    fn test_parse_response_contains() {
        let candidates = vec!["appwrite/cloud", "utopia-php/database"];
        let result = parse_response(
            "The issue belongs to utopia-php/database repository",
            &candidates,
        );
        assert_eq!(result, Some(("utopia-php/database".to_string(), 0.7)));
    }

    #[test]
    fn test_parse_response_none() {
        let candidates = vec!["appwrite/cloud", "utopia-php/database"];
        let result = parse_response("NONE", &candidates);
        assert_eq!(result, None);
    }

    #[test]
    fn test_parse_response_empty() {
        let candidates = vec!["appwrite/cloud"];
        let result = parse_response("", &candidates);
        assert_eq!(result, None);
    }

    #[test]
    fn test_parse_response_garbage() {
        let candidates = vec!["appwrite/cloud", "utopia-php/database"];
        let result = parse_response("I don't know which repo this belongs to", &candidates);
        assert_eq!(result, None);
    }

    #[test]
    fn test_parse_response_whitespace() {
        let candidates = vec!["appwrite/cloud"];
        let result = parse_response("  appwrite/cloud  \n", &candidates);
        assert_eq!(result, Some(("appwrite/cloud".to_string(), 1.0)));
    }

    #[test]
    fn test_parse_response_none_case_variants() {
        let candidates = vec!["appwrite/cloud"];
        assert_eq!(parse_response("none", &candidates), None);
        assert_eq!(parse_response("None", &candidates), None);
        assert_eq!(parse_response("NONE", &candidates), None);
    }

    #[test]
    fn test_build_prompt_no_description() {
        let mut request = sample_request();
        request.description = None;

        let prompt = build_prompt(&request);
        assert!(prompt.contains("MySQL server has gone away"));
        // The issue description line should not appear (candidates may still have "Description:" in their profiles)
        assert!(!prompt.contains("Description: Connection lost"));
    }

    #[test]
    fn test_build_prompt_empty_signals() {
        let request = ClassificationRequest {
            title: "Some error".to_string(),
            description: None,
            source: "linear".to_string(),
            metadata: HashMap::new(),
            extracted_filenames: vec![],
            extracted_functions: vec![],
            extracted_keywords: vec![],
            extracted_repos: vec![],
            candidates: vec![("org/repo".to_string(), "profile text".to_string())],
        };

        let prompt = build_prompt(&request);
        assert!(prompt.contains("Some error"));
        assert!(prompt.contains("org/repo"));
        assert!(!prompt.contains("Files referenced:"));
    }

    /// Integration test that runs against the real Qwen model.
    ///
    /// Skipped unless `CLAUDEAR_LLM_MODEL_PATH` is set to a valid GGUF file.
    /// Run with: CLAUDEAR_LLM_MODEL_PATH=~/.cache/claudear/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf cargo test -p claudear-engine --features sqlite -- llm_classifier::tests::test_live_classification --nocapture
    #[test]
    fn test_live_classification() {
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
                eprintln!("Skipping live LLM test: CLAUDEAR_LLM_MODEL_PATH not set");
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
        let classifier = LlmRepoClassifier::new(engine);

        // Test 1: Clear file-path signal should map to the right repo
        let request = ClassificationRequest {
            title: "TypeError in OAuth handler".to_string(),
            description: Some("Null reference in OAuth2 token refresh flow".to_string()),
            source: "sentry".to_string(),
            metadata: {
                let mut m = HashMap::new();
                m.insert("filename".to_string(), "src/Cloud/Auth/OAuth.php".to_string());
                m.insert("function".to_string(), "refreshToken".to_string());
                m
            },
            extracted_filenames: vec!["src/Cloud/Auth/OAuth.php".to_string()],
            extracted_functions: vec!["refreshToken".to_string()],
            extracted_keywords: vec!["OAuth".to_string(), "token".to_string()],
            extracted_repos: vec![],
            candidates: vec![
                (
                    "appwrite/cloud".to_string(),
                    "Repository: appwrite/cloud\nDescription: Cloud backend infrastructure for Appwrite\nLanguages: 45 php, 30 js, 15 ts\nDirectories: src, app, tests, config\nSample files: src/Cloud/Auth/OAuth.php, src/Cloud/Functions/Runtime.php, app/controllers/api/account.php".to_string(),
                ),
                (
                    "appwrite/console".to_string(),
                    "Repository: appwrite/console\nDescription: Appwrite web console UI\nLanguages: 80 ts, 15 svelte, 5 css\nDirectories: src, tests, static\nSample files: src/routes/auth.ts, src/components/Button.svelte, src/lib/api/client.ts".to_string(),
                ),
                (
                    "utopia-php/database".to_string(),
                    "Repository: utopia-php/database\nDescription: Database abstraction layer for PHP\nLanguages: 100 php\nDirectories: src, tests\nSample files: src/Database/Adapter/SQL.php, src/Database/Database.php".to_string(),
                ),
            ],
        };

        let result = classifier.classify(&request);
        eprintln!("Test 1 (OAuth in cloud): {:?}", result);
        assert!(result.is_some(), "Classifier should return a result");
        let (repo, confidence) = result.unwrap();
        assert_eq!(repo, "appwrite/cloud", "Should classify to appwrite/cloud");
        assert!(
            confidence >= 0.7,
            "Confidence should be >= 0.7, got {}",
            confidence
        );

        // Test 2: TypeScript UI issue should map to console
        let request2 = ClassificationRequest {
            title: "Button component renders incorrectly".to_string(),
            description: Some("The submit button in the auth flow has wrong styling".to_string()),
            source: "linear".to_string(),
            metadata: HashMap::new(),
            extracted_filenames: vec!["src/components/Button.svelte".to_string()],
            extracted_functions: vec![],
            extracted_keywords: vec!["button".to_string(), "styling".to_string(), "auth".to_string()],
            extracted_repos: vec![],
            candidates: vec![
                (
                    "appwrite/cloud".to_string(),
                    "Repository: appwrite/cloud\nDescription: Cloud backend infrastructure for Appwrite\nLanguages: 45 php, 30 js, 15 ts\nDirectories: src, app, tests, config\nSample files: src/Cloud/Auth/OAuth.php, src/Cloud/Functions/Runtime.php".to_string(),
                ),
                (
                    "appwrite/console".to_string(),
                    "Repository: appwrite/console\nDescription: Appwrite web console UI\nLanguages: 80 ts, 15 svelte, 5 css\nDirectories: src, tests, static\nSample files: src/routes/auth.ts, src/components/Button.svelte, src/lib/api/client.ts".to_string(),
                ),
            ],
        };

        let result2 = classifier.classify(&request2);
        eprintln!("Test 2 (Button in console): {:?}", result2);
        assert!(result2.is_some(), "Classifier should return a result");
        let (repo2, confidence2) = result2.unwrap();
        assert_eq!(
            repo2, "appwrite/console",
            "Should classify to appwrite/console"
        );
        assert!(
            confidence2 >= 0.7,
            "Confidence should be >= 0.7, got {}",
            confidence2
        );

        // Test 3: Database SQL issue should map to database lib
        let request3 = ClassificationRequest {
            title: "MySQL server has gone away".to_string(),
            description: Some("PDOException: SQLSTATE[HY000] [2006] MySQL server has gone away".to_string()),
            source: "sentry".to_string(),
            metadata: {
                let mut m = HashMap::new();
                m.insert("stacktrace".to_string(), "/usr/src/code/vendor/utopia-php/database/src/Database/Adapter/SQL.php in query at line 393".to_string());
                m
            },
            extracted_filenames: vec!["src/Database/Adapter/SQL.php".to_string()],
            extracted_functions: vec!["query".to_string()],
            extracted_keywords: vec!["MySQL".to_string(), "PDOException".to_string()],
            extracted_repos: vec!["utopia-php/database".to_string()],
            candidates: vec![
                (
                    "appwrite/cloud".to_string(),
                    "Repository: appwrite/cloud\nDescription: Cloud backend infrastructure\nLanguages: 45 php\nDirectories: src, app\nSample files: src/Cloud/Auth/OAuth.php".to_string(),
                ),
                (
                    "utopia-php/database".to_string(),
                    "Repository: utopia-php/database\nDescription: Database abstraction layer\nLanguages: 100 php\nDirectories: src, tests\nSample files: src/Database/Adapter/SQL.php, src/Database/Database.php".to_string(),
                ),
            ],
        };

        let result3 = classifier.classify(&request3);
        eprintln!("Test 3 (SQL in database): {:?}", result3);
        assert!(result3.is_some(), "Classifier should return a result");
        let (repo3, confidence3) = result3.unwrap();
        assert_eq!(
            repo3, "utopia-php/database",
            "Should classify to utopia-php/database"
        );
        assert!(
            confidence3 >= 0.7,
            "Confidence should be >= 0.7, got {}",
            confidence3
        );

        // Test 4: NONE — garbage issue with no matching signals
        let request4 = ClassificationRequest {
            title: "Update company holiday calendar".to_string(),
            description: Some("Need to add Q2 holidays to the internal calendar system".to_string()),
            source: "linear".to_string(),
            metadata: HashMap::new(),
            extracted_filenames: vec![],
            extracted_functions: vec![],
            extracted_keywords: vec!["calendar".to_string(), "holiday".to_string()],
            extracted_repos: vec![],
            candidates: vec![
                (
                    "appwrite/cloud".to_string(),
                    "Repository: appwrite/cloud\nDescription: Cloud backend infrastructure\nLanguages: php\nSample files: src/Cloud/Auth/OAuth.php".to_string(),
                ),
                (
                    "utopia-php/database".to_string(),
                    "Repository: utopia-php/database\nDescription: Database abstraction layer\nLanguages: php\nSample files: src/Database/Adapter/SQL.php".to_string(),
                ),
            ],
        };

        let result4 = classifier.classify(&request4);
        eprintln!("Test 4 (unrelated issue): {:?}", result4);
        // This might or might not return None — the LLM might still guess.
        // We just log the result for manual inspection.

        // --- Obscure / ambiguous cases that should still classify correctly ---

        // Test 5: Error message mentions "pool" and "connection" — both cloud and database
        // deal with connections, but the function name Pool::reclaim and the PHP type hint
        // to Adapter should tip it toward the database library.
        let request5 = ClassificationRequest {
            title: "Connection pool exhausted during peak traffic".to_string(),
            description: Some("All connections in the pool are busy. Workers are stalling.".to_string()),
            source: "sentry".to_string(),
            metadata: {
                let mut m = HashMap::new();
                m.insert("function".to_string(), "reclaim".to_string());
                m
            },
            extracted_filenames: vec![],
            extracted_functions: vec!["reclaim".to_string()],
            extracted_keywords: vec!["pool".to_string(), "connection".to_string()],
            extracted_repos: vec![],
            candidates: vec![
                (
                    "appwrite/cloud".to_string(),
                    "Repository: appwrite/cloud\nDescription: Cloud backend infrastructure for Appwrite, handles deployments, billing, and orchestration\nLanguages: 45 php, 30 js\nDirectories: src, app, tests, config, workers\nSample files: src/Cloud/Platform/Workers/ConnectionPool.php, src/Cloud/Auth/OAuth.php, app/controllers/api/health.php".to_string(),
                ),
                (
                    "utopia-php/database".to_string(),
                    "Repository: utopia-php/database\nDescription: Lightweight database abstraction layer with connection pooling and query builder\nLanguages: 100 php\nDirectories: src, tests\nSample files: src/Database/Adapter/Pool.php, src/Database/Adapter/SQL.php, src/Database/Database.php".to_string(),
                ),
                (
                    "utopia-php/pools".to_string(),
                    "Repository: utopia-php/pools\nDescription: Generic resource pool manager for PHP — manages connection lifecycle, reclaim, and eviction\nLanguages: 100 php\nDirectories: src, tests\nSample files: src/Pool.php, src/Connection.php, src/Group.php".to_string(),
                ),
            ],
        };

        let result5 = classifier.classify(&request5);
        eprintln!("Test 5 (pool exhaustion → pools lib): {:?}", result5);
        assert!(result5.is_some(), "Should classify pool exhaustion");
        let (repo5, _) = result5.unwrap();
        assert_eq!(
            repo5, "utopia-php/pools",
            "Pool reclaim belongs in the pools library, not cloud or database"
        );

        // Test 6: Generic "timeout" error — only clue is the extracted filename fragment
        // "Runtime.php" which appears in cloud's Functions/Runtime.php but NOT in the
        // framework repo's Router/Runtime equivalent. No stacktrace, no function name.
        let request6 = ClassificationRequest {
            title: "Function execution timeout".to_string(),
            description: Some("Execution exceeded 30s limit".to_string()),
            source: "sentry".to_string(),
            metadata: HashMap::new(),
            extracted_filenames: vec!["Runtime.php".to_string()],
            extracted_functions: vec![],
            extracted_keywords: vec!["timeout".to_string(), "execution".to_string(), "function".to_string()],
            extracted_repos: vec![],
            candidates: vec![
                (
                    "appwrite/cloud".to_string(),
                    "Repository: appwrite/cloud\nDescription: Cloud backend — serverless function execution, deployments, billing\nLanguages: 45 php, 30 js\nDirectories: src, app, workers, config\nSample files: src/Cloud/Functions/Runtime.php, src/Cloud/Functions/Executor.php, src/Cloud/Platform/Workers/FunctionWorker.php".to_string(),
                ),
                (
                    "utopia-php/framework".to_string(),
                    "Repository: utopia-php/framework\nDescription: Lightweight PHP MVC framework with routing and middleware\nLanguages: 100 php\nDirectories: src, tests\nSample files: src/Framework/App.php, src/Framework/Router.php, src/Framework/Middleware.php".to_string(),
                ),
                (
                    "appwrite/appwrite".to_string(),
                    "Repository: appwrite/appwrite\nDescription: Open source backend server for web, mobile, and Flutter devs\nLanguages: 60 php, 20 js, 10 dockerfile\nDirectories: src, app, tests, docker\nSample files: app/controllers/api/functions.php, src/Appwrite/Platform/Workers/Functions.php, app/realtime.php".to_string(),
                ),
            ],
        };

        let result6 = classifier.classify(&request6);
        eprintln!("Test 6 (function timeout → cloud): {:?}", result6);
        assert!(result6.is_some(), "Should classify function timeout");
        let (repo6, _) = result6.unwrap();
        // Cloud has Functions/Runtime.php — the direct match
        assert_eq!(
            repo6, "appwrite/cloud",
            "Function execution timeout with Runtime.php should map to cloud"
        );

        // Test 7: The error mentions the storage API controller rejecting uploads.
        // The validator lib and storage lib are dependencies, but the bug is in the
        // server's controller that orchestrates them — the file size limit is set
        // in the API route, not in the library.
        let request7 = ClassificationRequest {
            title: "POST /v1/storage/files returns 413 for files under limit".to_string(),
            description: Some("The storage upload API endpoint rejects files that are under the configured 50MB limit. The controller's max size check is comparing bytes vs kilobytes.".to_string()),
            source: "github".to_string(),
            metadata: HashMap::new(),
            extracted_filenames: vec!["app/controllers/api/storage.php".to_string()],
            extracted_functions: vec!["createFile".to_string()],
            extracted_keywords: vec!["storage".to_string(), "upload".to_string(), "413".to_string(), "file size".to_string()],
            extracted_repos: vec![],
            candidates: vec![
                (
                    "utopia-php/validator".to_string(),
                    "Repository: utopia-php/validator\nDescription: Data validation library for PHP — type checking, range, URL, email, IP\nLanguages: 100 php\nDirectories: src, tests\nSample files: src/Validator/URL.php, src/Validator/Email.php, src/Validator/Range.php".to_string(),
                ),
                (
                    "appwrite/appwrite".to_string(),
                    "Repository: appwrite/appwrite\nDescription: Open source backend server — auth, database, storage, functions, messaging\nLanguages: 60 php, 20 js\nDirectories: src, app, tests, docker\nSample files: app/controllers/api/storage.php, src/Appwrite/Utopia/Storage/Validator/Upload.php, app/controllers/api/account.php".to_string(),
                ),
                (
                    "utopia-php/storage".to_string(),
                    "Repository: utopia-php/storage\nDescription: Storage abstraction — local disk, S3, DO Spaces adapters\nLanguages: 100 php\nDirectories: src, tests\nSample files: src/Storage/Device/S3.php, src/Storage/Device/Local.php, src/Storage/Storage.php".to_string(),
                ),
            ],
        };

        let result7 = classifier.classify(&request7);
        eprintln!("Test 7 (storage controller bug → appwrite): {:?}", result7);
        assert!(
            result7.is_some(),
            "Should classify storage controller issue"
        );
        let (repo7, _) = result7.unwrap();
        assert_eq!(
            repo7, "appwrite/appwrite",
            "Controller-level storage bug belongs in main appwrite server"
        );

        // Test 8: Purely semantic — no file paths, no function names, just a vague
        // description about "rate limiting on API endpoints". The only differentiator
        // is that the abuse/rate-limit library handles rate limiting primitives, while
        // the main server uses them. The mention of "API endpoints" and "console requests"
        // should point to the main server.
        let request8 = ClassificationRequest {
            title: "Rate limiter blocking legitimate console requests".to_string(),
            description: Some("Users report 429 errors when navigating between dashboard pages quickly. The rate limit on the account endpoint is too aggressive.".to_string()),
            source: "linear".to_string(),
            metadata: HashMap::new(),
            extracted_filenames: vec![],
            extracted_functions: vec![],
            extracted_keywords: vec!["rate limit".to_string(), "429".to_string(), "account".to_string(), "dashboard".to_string()],
            extracted_repos: vec![],
            candidates: vec![
                (
                    "utopia-php/abuse".to_string(),
                    "Repository: utopia-php/abuse\nDescription: Rate limiting and abuse prevention library — token bucket, sliding window, IP-based limits\nLanguages: 100 php\nDirectories: src, tests\nSample files: src/Abuse/Abuse.php, src/Abuse/Adapters/Database.php, src/Abuse/TimeLimit.php".to_string(),
                ),
                (
                    "appwrite/appwrite".to_string(),
                    "Repository: appwrite/appwrite\nDescription: Open source backend server — REST API with auth, database, storage, functions. Serves console dashboard and client SDKs.\nLanguages: 60 php, 20 js\nDirectories: src, app, tests, docker\nSample files: app/controllers/api/account.php, app/controllers/api/teams.php, src/Appwrite/Utopia/Response.php".to_string(),
                ),
                (
                    "appwrite/console".to_string(),
                    "Repository: appwrite/console\nDescription: Appwrite web console — Svelte dashboard for managing projects\nLanguages: 80 svelte, 15 ts, 5 css\nDirectories: src, tests, static\nSample files: src/routes/console/project-[project]/overview/+page.svelte, src/lib/stores/sdk.ts".to_string(),
                ),
            ],
        };

        let result8 = classifier.classify(&request8);
        eprintln!(
            "Test 8 (rate limit on API endpoint → appwrite server): {:?}",
            result8
        );
        assert!(result8.is_some(), "Should classify rate limit issue");
        let (repo8, _) = result8.unwrap();
        assert_eq!(
            repo8, "appwrite/appwrite",
            "Rate limit on API endpoints is a server config issue, not the abuse library"
        );

        // Test 9: Cross-language red herring — error in a TypeScript SDK but the
        // description mentions "REST API returning wrong status code". The SDK just
        // surfaces the error; the bug is server-side. Only signal: extracted keyword
        // "createDocument" and the mention of "REST API".
        let request9 = ClassificationRequest {
            title: "createDocument returns 500 instead of 400 for invalid data".to_string(),
            description: Some("REST API returns internal server error when document schema validation fails. SDK surfaces the raw error.".to_string()),
            source: "github".to_string(),
            metadata: HashMap::new(),
            extracted_filenames: vec![],
            extracted_functions: vec!["createDocument".to_string()],
            extracted_keywords: vec!["REST API".to_string(), "500".to_string(), "schema".to_string(), "validation".to_string()],
            extracted_repos: vec![],
            candidates: vec![
                (
                    "appwrite/sdk-for-web".to_string(),
                    "Repository: appwrite/sdk-for-web\nDescription: Appwrite Web SDK — TypeScript client for browser apps\nLanguages: 95 ts, 5 js\nDirectories: src, tests\nSample files: src/services/databases.ts, src/services/account.ts, src/client.ts".to_string(),
                ),
                (
                    "appwrite/appwrite".to_string(),
                    "Repository: appwrite/appwrite\nDescription: Open source backend server — REST API, document validation, schema enforcement\nLanguages: 60 php, 20 js\nDirectories: src, app, tests\nSample files: app/controllers/api/databases.php, src/Appwrite/Utopia/Database/Validator/Structure.php, app/controllers/api/account.php".to_string(),
                ),
                (
                    "utopia-php/database".to_string(),
                    "Repository: utopia-php/database\nDescription: Database abstraction layer with document model and query builder\nLanguages: 100 php\nDirectories: src, tests\nSample files: src/Database/Database.php, src/Database/Validator/Structure.php, src/Database/Document.php".to_string(),
                ),
            ],
        };

        let result9 = classifier.classify(&request9);
        eprintln!(
            "Test 9 (REST API 500 error → appwrite server): {:?}",
            result9
        );
        assert!(result9.is_some(), "Should classify REST API error");
        let (repo9, _) = result9.unwrap();
        assert_eq!(
            repo9, "appwrite/appwrite",
            "REST API status code bug is server-side, not SDK"
        );

        // Test 10: Only signal is a Go package import path — no filenames, no functions.
        // The error is about a gRPC streaming timeout. Two Go repos: one is a proxy,
        // one is the runtime executor. The mention of "function build" and "container
        // image" should distinguish the executor from the proxy.
        let request10 = ClassificationRequest {
            title: "gRPC stream deadline exceeded during function build".to_string(),
            description: Some("Context deadline exceeded when streaming build logs from executor. Container image pull is slow on first deploy.".to_string()),
            source: "sentry".to_string(),
            metadata: HashMap::new(),
            extracted_filenames: vec![],
            extracted_functions: vec![],
            extracted_keywords: vec!["gRPC".to_string(), "deadline".to_string(), "build".to_string(), "container".to_string()],
            extracted_repos: vec![],
            candidates: vec![
                (
                    "appwrite/executor".to_string(),
                    "Repository: appwrite/executor\nDescription: Serverless function executor — builds container images, runs user code, streams logs via gRPC\nLanguages: 90 go, 10 dockerfile\nDirectories: cmd, pkg, internal\nSample files: cmd/executor/main.go, pkg/runtime/builder.go, pkg/runtime/docker.go, internal/grpc/server.go".to_string(),
                ),
                (
                    "appwrite/proxy".to_string(),
                    "Repository: appwrite/proxy\nDescription: HTTP/gRPC reverse proxy — routes traffic to function containers, handles TLS termination\nLanguages: 85 go, 15 yaml\nDirectories: cmd, pkg, deploy\nSample files: cmd/proxy/main.go, pkg/proxy/handler.go, pkg/proxy/grpc.go".to_string(),
                ),
                (
                    "appwrite/cloud".to_string(),
                    "Repository: appwrite/cloud\nDescription: Cloud backend — orchestrates function deployments, billing, team management\nLanguages: 45 php, 30 js\nDirectories: src, app, workers\nSample files: src/Cloud/Functions/Deployer.php, src/Cloud/Platform/Workers/Build.php".to_string(),
                ),
            ],
        };

        let result10 = classifier.classify(&request10);
        eprintln!("Test 10 (gRPC build timeout → executor): {:?}", result10);
        assert!(result10.is_some(), "Should classify gRPC build timeout");
        let (repo10, _) = result10.unwrap();
        assert_eq!(
            repo10, "appwrite/executor",
            "Build/container gRPC issue belongs in executor, not proxy"
        );

        // Test 11: Non-English error message from a Sentry event. The title is in
        // German but the metadata and extracted signals are in English code identifiers.
        // The model should ignore the German text and focus on the code signals.
        let request11 = ClassificationRequest {
            title: "Fehler beim Erstellen des Benutzers".to_string(), // "Error creating the user"
            description: Some("Unerwarteter Fehler in der Benutzerverwaltung. Doppelter Eintrag für E-Mail.".to_string()), // "Unexpected error in user management. Duplicate entry for email."
            source: "sentry".to_string(),
            metadata: {
                let mut m = HashMap::new();
                m.insert("culprit".to_string(), "Account::create".to_string());
                m
            },
            extracted_filenames: vec![],
            extracted_functions: vec!["create".to_string()],
            extracted_keywords: vec!["Account".to_string(), "duplicate".to_string(), "email".to_string()],
            extracted_repos: vec![],
            candidates: vec![
                (
                    "appwrite/appwrite".to_string(),
                    "Repository: appwrite/appwrite\nDescription: Backend server — user authentication, account management, team invites\nLanguages: 60 php, 20 js\nDirectories: src, app, tests\nSample files: app/controllers/api/account.php, src/Appwrite/Auth/Auth.php, app/controllers/api/teams.php".to_string(),
                ),
                (
                    "appwrite/console".to_string(),
                    "Repository: appwrite/console\nDescription: Web console dashboard — project settings, user list, team management UI\nLanguages: 80 svelte, 15 ts\nDirectories: src, tests\nSample files: src/routes/console/project-[project]/auth/+page.svelte, src/lib/stores/user.ts".to_string(),
                ),
                (
                    "appwrite/sdk-for-php".to_string(),
                    "Repository: appwrite/sdk-for-php\nDescription: PHP SDK for Appwrite API\nLanguages: 100 php\nDirectories: src, tests\nSample files: src/Appwrite/Services/Account.php, src/Appwrite/Client.php".to_string(),
                ),
            ],
        };

        let result11 = classifier.classify(&request11);
        eprintln!(
            "Test 11 (German error message → appwrite server): {:?}",
            result11
        );
        assert!(
            result11.is_some(),
            "Should classify despite non-English title"
        );
        let (repo11, _) = result11.unwrap();
        assert_eq!(
            repo11, "appwrite/appwrite",
            "Account::create duplicate email is a server-side auth bug"
        );

        // Test 12: Misleading repo name in extracted_repos that doesn't exist in
        // candidates. The real signal is a subtle one: the error mentions "realtime"
        // and "websocket", and only one candidate has a realtime file.
        let request12 = ClassificationRequest {
            title: "WebSocket connection drops after 60s idle".to_string(),
            description: Some("Realtime subscriptions disconnect silently. Clients don't receive reconnection event.".to_string()),
            source: "github".to_string(),
            metadata: HashMap::new(),
            extracted_filenames: vec![],
            extracted_functions: vec![],
            extracted_keywords: vec!["websocket".to_string(), "realtime".to_string(), "subscription".to_string()],
            extracted_repos: vec!["appwrite/realtime".to_string()], // red herring — not in candidates
            candidates: vec![
                (
                    "appwrite/appwrite".to_string(),
                    "Repository: appwrite/appwrite\nDescription: Backend server — REST API, realtime WebSocket server, event system\nLanguages: 60 php, 20 js\nDirectories: src, app, tests\nSample files: app/realtime.php, src/Appwrite/Messaging/Adapter/Realtime.php, app/controllers/api/databases.php".to_string(),
                ),
                (
                    "appwrite/sdk-for-web".to_string(),
                    "Repository: appwrite/sdk-for-web\nDescription: Web SDK — REST client, realtime subscription client\nLanguages: 95 ts\nDirectories: src, tests\nSample files: src/services/realtime.ts, src/services/databases.ts, src/client.ts".to_string(),
                ),
                (
                    "utopia-php/websocket".to_string(),
                    "Repository: utopia-php/websocket\nDescription: Lightweight PHP WebSocket server library\nLanguages: 100 php\nDirectories: src, tests\nSample files: src/WebSocket/Server.php, src/WebSocket/Connection.php".to_string(),
                ),
            ],
        };

        let result12 = classifier.classify(&request12);
        eprintln!(
            "Test 12 (WebSocket idle disconnect → appwrite server): {:?}",
            result12
        );
        assert!(result12.is_some(), "Should classify websocket disconnect");
        let (repo12, _) = result12.unwrap();
        // The server owns the realtime endpoint and event subscriptions.
        // The websocket lib is just the transport layer; the SDK is the client side.
        assert_eq!(
            repo12, "appwrite/appwrite",
            "Realtime subscription disconnect is a server-side issue"
        );
    }
}
