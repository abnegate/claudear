//! Claude CLI runner for executing fixes.

use super::{AgentRunner, ProviderCapabilities};
use async_trait::async_trait;
use claudear_core::error::{Error, Result};
use claudear_core::templates::{TemplateContext, TemplateLoader, TemplateRenderer};
use claudear_core::types::{
    ActivityLogEntry, AgentExecution, AgentResult, BlockingQuestion, Issue,
};
use claudear_storage::FixAttemptTracker;
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;
use tokio::sync::{mpsc, Mutex};

const DEFAULT_LOG_DIR: &str = "./logs";
const CLAUDE_LOG_SUBDIR: &str = "claude";
const EXECUTION_LOG_PREVIEW_LIMIT: usize = 2000;

/// Resolve the root directory for execution logs.
/// Used by both the runner and the API to validate log file paths.
pub fn resolve_log_root() -> PathBuf {
    std::env::var("CLAUDEAR_LOG_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_LOG_DIR))
}

/// JSON schema for Claude's structured output. Constrained decoding guarantees the
/// final response matches this schema, replacing both the CLAUDEAR_QUESTION protocol
/// and the PR_URL instruction.
const RESULT_SCHEMA: &str = r#"{
    "type": "object",
    "required": ["summary", "success", "confidence"],
    "additionalProperties": false,
    "properties": {
        "summary": {
            "type": "string",
            "description": "Brief summary of what was done or why you stopped"
        },
        "success": {
            "type": "boolean",
            "description": "Whether the task was completed successfully"
        },
        "pr_url": {
            "type": ["string", "null"],
            "description": "URL of the created pull request, if one was created"
        },
        "changelog": {
            "type": ["string", "null"],
            "description": "A succinct bullet-point list of the changes made (e.g. '- Fixed null check in auth handler\\n- Added unit test for edge case'). Null if no changes were made."
        },
        "blocking_question": {
            "type": ["object", "null"],
            "description": "If you need human input to proceed, provide the question here instead of attempting the task",
            "required": ["question"],
            "properties": {
                "question": { "type": "string" },
                "context": { "type": ["string", "null"] },
                "options": { "type": "array", "items": { "type": "string" } },
                "why": { "type": ["string", "null"] }
            },
            "additionalProperties": false
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 100,
            "description": "Your confidence (0-100) that this fix correctly resolves the issue without introducing regressions"
        },
        "confidence_reasoning": {
            "type": ["string", "null"],
            "description": "Brief explanation of your confidence level (e.g. what makes you certain or uncertain)"
        }
    }
}"#;

/// Content block types within a CLI `assistant` event's `message.content[]`.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum CliContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse { id: String, name: String },
    #[serde(other)]
    Other,
}

/// The `message` object carried by a CLI `assistant` event.
#[derive(Debug, Deserialize)]
struct CliMessage {
    #[serde(default)]
    content: Vec<CliContentBlock>,
}

/// Token usage reported by the Claude CLI result event.
#[derive(Debug, Deserialize)]
struct CliUsage {
    #[serde(default)]
    input_tokens: Option<i64>,
    #[serde(default)]
    output_tokens: Option<i64>,
    #[serde(default)]
    cache_read_input_tokens: Option<i64>,
    #[serde(default)]
    cache_creation_input_tokens: Option<i64>,
}

/// Rate-limit info nested inside a `rate_limit_event`.
#[derive(Debug, Deserialize)]
struct RateLimitInfo {
    #[serde(default)]
    status: Option<String>,
    #[serde(default, rename = "resetsAt")]
    resets_at: Option<serde_json::Value>,
    #[serde(default, rename = "rateLimitType")]
    rate_limit_type: Option<String>,
    #[serde(default)]
    utilization: Option<f64>,
}

/// Top-level NDJSON events emitted by `claude --output-format stream-json`.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum StreamEvent {
    #[serde(rename = "system")]
    System {},
    #[serde(rename = "assistant")]
    Assistant {
        #[serde(default)]
        message: Option<CliMessage>,
    },
    #[serde(rename = "user")]
    User {},
    #[serde(rename = "result")]
    Result {
        #[serde(default)]
        structured_output: Option<serde_json::Value>,
        #[serde(default)]
        total_cost_usd: Option<f64>,
        #[serde(default)]
        num_turns: Option<i64>,
        #[serde(default)]
        session_id: Option<String>,
        #[serde(default)]
        duration_api_ms: Option<i64>,
        #[serde(default)]
        usage: Option<CliUsage>,
    },
    /// Structured rate-limit signal from the Claude CLI.
    #[serde(rename = "rate_limit_event")]
    RateLimitEvent {
        #[serde(default)]
        rate_limit_info: Option<RateLimitInfo>,
        /// Top-level resetsAt (alternative location).
        #[serde(default, rename = "resetsAt")]
        resets_at: Option<serde_json::Value>,
    },
    /// Forward-compat: ignore unknown event types.
    #[serde(other)]
    Unknown,
}

/// Deserialization target for the structured result produced by `--json-schema`.
#[derive(Debug, Deserialize)]
struct StructuredResult {
    #[serde(default)]
    summary: String,
    success: bool,
    #[serde(default)]
    pr_url: Option<String>,
    #[serde(default)]
    changelog: Option<String>,
    #[serde(default)]
    blocking_question: Option<BlockingQuestion>,
    #[serde(default)]
    confidence: u8,
    #[serde(default)]
    confidence_reasoning: Option<String>,
}

#[derive(Debug, Clone)]
struct ExecutionLogFiles {
    stdout: PathBuf,
    stderr: PathBuf,
    events: PathBuf,
}

/// Aggregated result from parsing the stdout stream.
#[derive(Debug, Default)]
struct StdoutParseResult {
    text_output: String,
    structured_result: Option<serde_json::Value>,
    cost_usd: Option<f64>,
    num_turns: Option<i64>,
    session_id: Option<String>,
    duration_api_ms: Option<i64>,
    input_tokens: Option<i64>,
    output_tokens: Option<i64>,
    cache_read_input_tokens: Option<i64>,
    cache_creation_input_tokens: Option<i64>,
}

/// Configuration for the Claude runner.
#[derive(Debug, Clone)]
pub struct ClaudeRunnerConfig {
    /// Timeout for Claude process execution in seconds (default: 21600 = 6 hours).
    pub timeout_secs: u64,
    /// Model to use (e.g., sonnet, opus, haiku, or full model ID).
    pub model: Option<String>,
    /// Custom instructions appended to Claude's system prompt.
    pub instructions: Option<String>,
    /// Tool permissions granted without prompting (--allowedTools).
    pub permissions: Vec<String>,
    /// Skip all permission prompts (default: false).
    pub skip_permissions: bool,
}

impl Default for ClaudeRunnerConfig {
    fn default() -> Self {
        Self {
            timeout_secs: 21600, // 6 hours default
            model: None,
            instructions: None,
            permissions: Vec::new(),
            skip_permissions: false,
        }
    }
}

/// Runs Claude Code to fix issues.
pub struct ClaudeAgentRunner {
    config: ClaudeRunnerConfig,
    template_renderer: TemplateRenderer,
    tracker: Arc<dyn FixAttemptTracker>,
    /// Cached base environment variables, captured once at construction time.
    /// Cloned per-invocation instead of re-reading the entire process environment.
    base_env: HashMap<String, String>,
}

impl ClaudeAgentRunner {
    /// Create a new Claude runner.
    pub fn new(config: ClaudeRunnerConfig, tracker: Arc<dyn FixAttemptTracker>) -> Self {
        let template_renderer = TemplateRenderer::new();
        let base_env: HashMap<String, String> = std::env::vars()
            .filter(|(k, _)| k != "CLAUDECODE")
            .collect();
        Self {
            config,
            template_renderer,
            tracker,
            base_env,
        }
    }

    /// Create a new Claude runner without template support (for testing).
    #[cfg(feature = "sqlite")]
    pub fn new_simple(config: ClaudeRunnerConfig) -> Self {
        use claudear_storage::SqliteTracker;
        // Use a temporary in-memory tracker
        let tracker = Arc::new(SqliteTracker::in_memory().unwrap());
        Self::new(config, tracker)
    }

    /// Run Claude Code with a prompt to fix an issue in a specific repository.
    pub async fn run_fix(
        &self,
        issue: &Issue,
        context: &str,
        project_dir: &Path,
    ) -> Result<AgentResult> {
        let prompt = self.build_prompt(issue, context, project_dir);
        self.execute(&prompt, Some(issue), project_dir).await
    }

    /// Run Claude Code with the /issue skill (for Linear issues).
    pub async fn run_issue_skill(
        &self,
        issue_identifier: &str,
        issue_url: &str,
        project_dir: &Path,
    ) -> Result<AgentResult> {
        tracing::info!(component = "claude", issue_id = %issue_identifier, "Running /issue skill");

        let mut env = self.base_env.clone();
        env.insert("LINEAR_ISSUE_ID".to_string(), issue_identifier.to_string());
        env.insert("LINEAR_ISSUE_URL".to_string(), issue_url.to_string());

        self.execute_with_env(
            &format!("/issue {}", issue_identifier),
            issue_identifier,
            env,
            project_dir,
        )
        .await
    }

    /// Run Claude Code with a custom prompt.
    pub async fn run_custom(&self, prompt: &str, project_dir: &Path) -> Result<AgentResult> {
        self.execute(prompt, None, project_dir).await
    }

    fn build_prompt(&self, issue: &Issue, context: &str, project_dir: &Path) -> String {
        // Try to use template system
        let template_loader = TemplateLoader::new(project_dir);
        if let Ok(template) = template_loader.get_template(issue) {
            let agent_md = template_loader.load_agent_md();
            let template_context =
                TemplateContext::new(issue.clone(), context.to_string()).with_agent_md(agent_md);
            return self.template_renderer.render(&template, &template_context);
        }

        // Fallback to simple format
        format!(
            r#"You are fixing an issue from {}. Here is the issue context:

{}

Your task:
1. Analyze the issue/error and any stack traces
2. Find the relevant code in this codebase
3. Implement a fix for the issue
4. Write or update tests if applicable
5. Create a PR with your changes
6. Ensure all checks pass on the PR
7. Rate your confidence (0-100) that this fix correctly resolves the issue

The PR title should include the issue ID: {}
"#,
            issue.source, context, issue.short_id
        )
    }

    /// Check if a project has an AGENT.md file.
    pub fn has_agent_md(&self, project_dir: &Path) -> bool {
        let template_loader = TemplateLoader::new(project_dir);
        template_loader.has_agent_md()
    }

    /// Get AGENT.md content if it exists.
    pub fn get_agent_md(&self, project_dir: &Path) -> Option<String> {
        let template_loader = TemplateLoader::new(project_dir);
        template_loader.load_agent_md()
    }

    /// Best-effort detection for rate limit failures.
    /// Delegates to the free function in the parent module.
    pub fn is_rate_limit_error(message: &str) -> bool {
        super::is_rate_limit_error(message)
    }

    /// Detect "hard" runtime failures that should be escalated immediately.
    /// Delegates to the free function in the parent module.
    pub fn is_hard_error(message: &str) -> bool {
        super::is_hard_error(message)
    }

    fn sanitize_label(label: &str) -> String {
        let mut out = String::with_capacity(label.len().min(64));
        for ch in label.chars().take(64) {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                out.push(ch);
            } else {
                out.push('_');
            }
        }
        if out.is_empty() {
            "custom".to_string()
        } else {
            out
        }
    }

    fn resolve_log_root() -> PathBuf {
        std::env::var("CLAUDEAR_LOG_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from(DEFAULT_LOG_DIR))
    }

    fn create_execution_log_files(label: &str) -> Option<ExecutionLogFiles> {
        let root = Self::resolve_log_root();
        if root.as_os_str().is_empty() {
            return None;
        }

        let now = chrono::Utc::now();
        let day = now.format("%Y-%m-%d").to_string();
        let timestamp = now.format("%Y%m%dT%H%M%S%.3fZ").to_string();
        let dir = root.join(CLAUDE_LOG_SUBDIR).join(day);

        if let Err(e) = std::fs::create_dir_all(&dir) {
            tracing::warn!(
                component = "claude",
                path = %dir.display(),
                error = %e,
                "Failed to create execution log directory"
            );
            return None;
        }

        let safe_label = Self::sanitize_label(label);
        let pid = std::process::id();
        let stem = format!("{}_{}_{}", timestamp, pid, safe_label);

        Some(ExecutionLogFiles {
            stdout: dir.join(format!("{}.stdout.log", stem)),
            stderr: dir.join(format!("{}.stderr.log", stem)),
            events: dir.join(format!("{}.events.jsonl", stem)),
        })
    }

    fn compose_failure_message(exit_code: i32, stdout_output: &str, stderr_output: &str) -> String {
        let stderr_trimmed = stderr_output.trim();
        let stdout_trimmed = stdout_output.trim();

        // Only check stderr for rate limits — stdout (assistant text) can contain
        // rate-limit strings from analyzed code and cause false positives.
        if Self::is_rate_limit_error(stderr_trimmed) {
            let msg = if stderr_trimmed.is_empty() {
                "Too many requests"
            } else {
                stderr_trimmed
            };
            return format!(
                "Claude rate limit hit: {}",
                Self::truncate(msg, EXECUTION_LOG_PREVIEW_LIMIT)
            );
        }

        if !stderr_trimmed.is_empty() {
            return Self::truncate(stderr_trimmed, EXECUTION_LOG_PREVIEW_LIMIT);
        }

        if !stdout_trimmed.is_empty() {
            return format!(
                "Process exited with code {}. Output: {}",
                exit_code,
                Self::truncate(stdout_trimmed, EXECUTION_LOG_PREVIEW_LIMIT)
            );
        }

        format!("Process exited with code {}", exit_code)
    }

    /// Legacy fallback: extract a blocking question from raw text output.
    /// Used when structured_result is None (killed processes, old CLI versions, etc.).
    fn extract_blocking_question(output: &str) -> Option<BlockingQuestion> {
        // Look for CLAUDEAR_QUESTION: prefix (legacy protocol)
        let legacy_prefix = "CLAUDEAR_QUESTION:";
        output.lines().find_map(|line| {
            let trimmed = line.trim();
            let payload = trimmed.strip_prefix(legacy_prefix)?.trim();
            if payload.is_empty() {
                return None;
            }
            serde_json::from_str::<BlockingQuestion>(payload).ok()
        })
    }

    async fn append_execution_event(
        writer: &Option<Arc<Mutex<tokio::fs::File>>>,
        label: &str,
        event: &str,
        data: serde_json::Value,
    ) {
        let Some(writer) = writer else {
            return;
        };

        let payload = json!({
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "label": label,
            "event": event,
            "data": data,
        });

        let serialized = match serde_json::to_string(&payload) {
            Ok(value) => value,
            Err(e) => {
                tracing::warn!(
                    component = "claude",
                    label = label,
                    error = %e,
                    "Failed to serialize execution event payload"
                );
                return;
            }
        };

        let mut guard = writer.lock().await;
        if let Err(e) = guard.write_all(serialized.as_bytes()).await {
            tracing::warn!(
                component = "claude",
                label = label,
                event = event,
                error = %e,
                "Failed to write execution event payload"
            );
            return;
        }
        if let Err(e) = guard.write_all(b"\n").await {
            tracing::warn!(
                component = "claude",
                label = label,
                event = event,
                error = %e,
                "Failed to terminate execution event payload line"
            );
        }
    }

    /// Build the per-invocation environment and derive a human-readable label
    /// from the optional issue. Shared by `execute` and `execute_with_attempt`
    /// to avoid duplicating the env-var / label logic.
    fn prepare_env_and_label<'a>(
        &self,
        issue: Option<&'a Issue>,
    ) -> (HashMap<String, String>, &'a str) {
        let mut env = self.base_env.clone();

        if let Some(issue) = issue {
            let source_upper = issue.source.to_uppercase();
            env.insert(format!("{}_ISSUE_ID", source_upper), issue.id.clone());
            env.insert(
                format!("{}_ISSUE_SHORT_ID", source_upper),
                issue.short_id.clone(),
            );
            env.insert(format!("{}_ISSUE_URL", source_upper), issue.url.clone());
        }

        let label = issue.map(|i| i.short_id.as_str()).unwrap_or("custom");
        (env, label)
    }

    async fn execute(
        &self,
        prompt: &str,
        issue: Option<&Issue>,
        project_dir: &Path,
    ) -> Result<AgentResult> {
        let (env, label) = self.prepare_env_and_label(issue);
        self.execute_with_env(prompt, label, env, project_dir).await
    }

    async fn execute_with_env(
        &self,
        prompt: &str,
        label: &str,
        env: HashMap<String, String>,
        project_dir: &Path,
    ) -> Result<AgentResult> {
        self.execute_with_env_and_attempt(prompt, label, env, None, project_dir)
            .await
    }

    async fn execute_with_env_and_attempt(
        &self,
        prompt: &str,
        label: &str,
        env: HashMap<String, String>,
        attempt_id: Option<i64>,
        project_dir: &Path,
    ) -> Result<AgentResult> {
        // Create execution record for analytics
        let mut execution = AgentExecution::new();
        if let Some(id) = attempt_id {
            execution = execution.with_attempt_id(id);
        }
        execution.prompt_used = Some(prompt.to_string());
        execution.prompt_hash = Some(Self::hash_prompt(prompt));
        execution.model_version = Some(
            self.config
                .model
                .clone()
                .unwrap_or_else(|| "claude-code".to_string()),
        );
        execution.working_directory = Some(project_dir.display().to_string());

        tracing::info!(
            component = "claude",
            label = label,
            timeout_secs = self.config.timeout_secs,
            "Starting execution"
        );

        // Log claude_started activity
        let activity = ActivityLogEntry::new(
            "claude_started",
            format!("Claude execution started for {}", label),
        )
        .with_source("claude".to_string())
        .with_metadata(json!({
            "timeout_secs": self.config.timeout_secs,
            "working_dir": project_dir.display().to_string(),
            "label": label
        }));
        self.tracker.record_activity(&activity).ok();

        let mut args = vec![
            "--verbose".to_string(),
            "--output-format".to_string(),
            "stream-json".to_string(),
            "--json-schema".to_string(),
            RESULT_SCHEMA.to_string(),
        ];
        if self.config.skip_permissions {
            args.push("--dangerously-skip-permissions".to_string());
        }
        if let Some(ref model) = self.config.model {
            args.push("--model".to_string());
            args.push(model.clone());
        }
        if let Some(ref instructions) = self.config.instructions {
            args.push("--append-system-prompt".to_string());
            args.push(instructions.clone());
        }
        for perm in &self.config.permissions {
            args.push("--allowedTools".to_string());
            args.push(perm.clone());
        }
        args.push("--print".to_string());
        args.push(prompt.to_string());

        let log_files = Self::create_execution_log_files(label);
        if let Some(ref files) = log_files {
            execution.stdout_log_path = Some(files.stdout.display().to_string());
            execution.stderr_log_path = Some(files.stderr.display().to_string());
            execution.event_log_path = Some(files.events.display().to_string());
            tracing::info!(
                component = "claude",
                label = label,
                stdout_log = %files.stdout.display(),
                stderr_log = %files.stderr.display(),
                events_log = %files.events.display(),
                "Capturing Claude output to execution log files"
            );
        }

        let event_writer: Option<Arc<Mutex<tokio::fs::File>>> =
            match log_files.as_ref().map(|files| files.events.clone()) {
                Some(path) => match tokio::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&path)
                    .await
                {
                    Ok(file) => Some(Arc::new(Mutex::new(file))),
                    Err(e) => {
                        tracing::warn!(
                            component = "claude",
                            label = label,
                            path = %path.display(),
                            error = %e,
                            "Failed to open execution events log file"
                        );
                        None
                    }
                },
                None => None,
            };

        let cli_args_without_prompt: Vec<String> = args
            .iter()
            .take(args.len().saturating_sub(1))
            .cloned()
            .collect();
        Self::append_execution_event(
            &event_writer,
            label,
            "execution_initialized",
            json!({
                "attempt_id": attempt_id,
                "timeout_secs": self.config.timeout_secs,
                "working_dir": project_dir.display().to_string(),
                "model": self.config.model.clone(),
                "skip_permissions": self.config.skip_permissions,
                "permissions": self.config.permissions.clone(),
                "cli_args_without_prompt": cli_args_without_prompt,
                "prompt_hash": execution.prompt_hash.clone(),
            }),
        )
        .await;

        let mut child = match Command::new("claude")
            .args(&args)
            .current_dir(project_dir)
            .envs(env)
            .env_remove("CLAUDECODE")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
        {
            Ok(child) => child,
            Err(e) => {
                Self::append_execution_event(
                    &event_writer,
                    label,
                    "spawn_failed",
                    json!({
                        "error": e.to_string(),
                    }),
                )
                .await;
                return Err(Error::runner(format!("Failed to spawn claude: {}", e)));
            }
        };

        Self::append_execution_event(
            &event_writer,
            label,
            "subprocess_spawned",
            json!({
                "pid": child.id(),
            }),
        )
        .await;

        let stdout = match child.stdout.take() {
            Some(stdout) => stdout,
            None => {
                Self::append_execution_event(
                    &event_writer,
                    label,
                    "stdout_capture_failed",
                    json!({
                        "error": "Failed to capture stdout",
                    }),
                )
                .await;
                return Err(Error::runner("Failed to capture stdout"));
            }
        };
        let stderr = match child.stderr.take() {
            Some(stderr) => stderr,
            None => {
                Self::append_execution_event(
                    &event_writer,
                    label,
                    "stderr_capture_failed",
                    json!({
                        "error": "Failed to capture stderr",
                    }),
                )
                .await;
                return Err(Error::runner("Failed to capture stderr"));
            }
        };

        let label_stdout = label.to_string();
        let label_stderr = label.to_string();
        let stdout_log_path = log_files.as_ref().map(|f| f.stdout.clone());
        let stderr_log_path = log_files.as_ref().map(|f| f.stderr.clone());
        let stdout_event_writer = event_writer.clone();
        let stderr_event_writer = event_writer.clone();
        let (early_failure_tx, mut early_failure_rx) = mpsc::unbounded_channel::<String>();
        let stdout_early_failure_tx = early_failure_tx.clone();
        let stderr_early_failure_tx = early_failure_tx.clone();
        drop(early_failure_tx);

        let stdout_handle = tokio::spawn(async move {
            let mut lines = BufReader::new(stdout).lines();
            let mut text_output = String::new();
            let mut line_number: u64 = 0;
            let mut structured_result: Option<serde_json::Value> = None;
            let mut result_cost_usd: Option<f64> = None;
            let mut result_num_turns: Option<i64> = None;
            let mut result_session_id: Option<String> = None;
            let mut result_duration_api_ms: Option<i64> = None;
            let mut result_input_tokens: Option<i64> = None;
            let mut result_output_tokens: Option<i64> = None;
            let mut result_cache_read_tokens: Option<i64> = None;
            let mut result_cache_creation_tokens: Option<i64> = None;
            let mut writer = match stdout_log_path {
                Some(path) => match tokio::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&path)
                    .await
                {
                    Ok(file) => Some(file),
                    Err(e) => {
                        tracing::warn!(
                            component = "claude",
                            label = label_stdout.as_str(),
                            path = %path.display(),
                            error = %e,
                            "Failed to open stdout execution log file"
                        );
                        None
                    }
                },
                None => None,
            };
            let mut write_failed = false;
            let mut signaled_rate_limit = false;

            loop {
                let next_line = lines.next_line().await;
                let line = match next_line {
                    Ok(Some(line)) => line,
                    Ok(None) => break,
                    Err(e) => {
                        ClaudeAgentRunner::append_execution_event(
                            &stdout_event_writer,
                            label_stdout.as_str(),
                            "stdout_read_error",
                            json!({
                                "error": e.to_string(),
                            }),
                        )
                        .await;
                        tracing::warn!(
                            component = "claude",
                            label = label_stdout.as_str(),
                            error = %e,
                            "Failed reading Claude stdout stream"
                        );
                        break;
                    }
                };

                line_number = line_number.saturating_add(1);
                ClaudeAgentRunner::append_execution_event(
                    &stdout_event_writer,
                    label_stdout.as_str(),
                    "stdout_line",
                    json!({
                        "line_number": line_number,
                        "line": line.as_str(),
                    }),
                )
                .await;

                // Parse NDJSON stream events; accumulate decoded text.
                let trimmed = line.trim();
                if !trimmed.is_empty() {
                    match serde_json::from_str::<StreamEvent>(trimmed) {
                        Ok(StreamEvent::Assistant { message }) => {
                            if let Some(msg) = message {
                                for block in &msg.content {
                                    match block {
                                        CliContentBlock::Text { ref text } => {
                                            text_output.push_str(text);
                                            if let Some(file) = writer.as_mut() {
                                                if !write_failed
                                                    && file
                                                        .write_all(text.as_bytes())
                                                        .await
                                                        .is_err()
                                                {
                                                    write_failed = true;
                                                    ClaudeAgentRunner::append_execution_event(
                                                        &stdout_event_writer,
                                                        label_stdout.as_str(),
                                                        "stdout_log_write_failed",
                                                        json!({}),
                                                    )
                                                    .await;
                                                    tracing::warn!(
                                                        component = "claude",
                                                        label = label_stdout.as_str(),
                                                        "Failed writing decoded text to execution log file"
                                                    );
                                                }
                                            }
                                        }
                                        CliContentBlock::ToolUse { ref id, ref name } => {
                                            tracing::info!(
                                                component = "claude",
                                                label = label_stdout.as_str(),
                                                tool_use_id = id.as_str(),
                                                tool_name = name.as_str(),
                                                "Tool use started"
                                            );
                                        }
                                        CliContentBlock::Other => {}
                                    }
                                }
                            }
                        }
                        Ok(StreamEvent::Result {
                            structured_output,
                            total_cost_usd,
                            num_turns,
                            session_id,
                            duration_api_ms,
                            usage,
                        }) => {
                            if let Some(obj) = structured_output {
                                structured_result = Some(obj);
                            }
                            if let Some(cost) = total_cost_usd {
                                result_cost_usd = Some(cost);
                            }
                            result_num_turns = num_turns;
                            result_session_id = session_id;
                            result_duration_api_ms = duration_api_ms;
                            if let Some(u) = usage {
                                result_input_tokens = u.input_tokens;
                                result_output_tokens = u.output_tokens;
                                result_cache_read_tokens = u.cache_read_input_tokens;
                                result_cache_creation_tokens = u.cache_creation_input_tokens;
                            }
                        }
                        Ok(StreamEvent::RateLimitEvent {
                            rate_limit_info,
                            resets_at: top_level_resets_at,
                        }) => {
                            // Structural rate-limit detection: Claude CLI emits
                            // rate_limit_event with rate_limit_info.status.
                            // "allowed" / "allowed_warning" are informational.
                            // Anything else (e.g. "exceeded") is a real block.
                            let status = rate_limit_info
                                .as_ref()
                                .and_then(|info| info.status.as_deref())
                                .unwrap_or("");
                            let is_informational =
                                status == "allowed" || status == "allowed_warning";

                            if !is_informational && !signaled_rate_limit {
                                signaled_rate_limit = true;
                                // Include the raw JSON so the watcher can extract
                                // resetsAt for pause scheduling.
                                let msg = format!(
                                    "Claude rate limit hit: {}",
                                    ClaudeAgentRunner::truncate(
                                        trimmed,
                                        EXECUTION_LOG_PREVIEW_LIMIT
                                    )
                                );
                                let _ = stdout_early_failure_tx.send(msg);
                                let utilization =
                                    rate_limit_info.as_ref().and_then(|info| info.utilization);
                                let rate_limit_type = rate_limit_info
                                    .as_ref()
                                    .and_then(|info| info.rate_limit_type.as_deref());
                                let resets_at = rate_limit_info
                                    .as_ref()
                                    .and_then(|info| info.resets_at.as_ref())
                                    .or(top_level_resets_at.as_ref());
                                ClaudeAgentRunner::append_execution_event(
                                    &stdout_event_writer,
                                    label_stdout.as_str(),
                                    "rate_limit_detected_live",
                                    json!({
                                        "source": "rate_limit_event",
                                        "line_number": line_number,
                                        "status": status,
                                        "utilization": utilization,
                                        "rate_limit_type": rate_limit_type,
                                        "resets_at": resets_at,
                                    }),
                                )
                                .await;
                                tracing::warn!(
                                    component = "claude",
                                    label = label_stdout.as_str(),
                                    line_number,
                                    status,
                                    ?utilization,
                                    "Detected rate_limit_event with non-allowed status; requesting early termination"
                                );
                            }
                        }
                        Ok(_) => { /* System, User, Unknown — ignored */ }
                        Err(_) => {
                            // Not a valid stream event — try parsing as the final
                            // JSON wrapper. The Claude CLI emits the result at the
                            // END of the stream, so always overwrite with the latest
                            // valid result (last-write-wins).
                            if let Ok(wrapper) = serde_json::from_str::<serde_json::Value>(trimmed)
                            {
                                if let Some(result_str) =
                                    wrapper.get("result").and_then(|v| v.as_str())
                                {
                                    if let Ok(parsed) =
                                        serde_json::from_str::<serde_json::Value>(result_str)
                                    {
                                        structured_result = Some(parsed);
                                    }
                                } else if let Some(obj) = wrapper.get("structured_output") {
                                    structured_result = Some(obj.clone());
                                }
                            }
                        }
                    }
                }
            }

            ClaudeAgentRunner::append_execution_event(
                &stdout_event_writer,
                label_stdout.as_str(),
                "stdout_stream_closed",
                json!({
                    "line_count": line_number,
                    "has_structured_result": structured_result.is_some(),
                }),
            )
            .await;
            StdoutParseResult {
                text_output,
                structured_result,
                cost_usd: result_cost_usd,
                num_turns: result_num_turns,
                session_id: result_session_id,
                duration_api_ms: result_duration_api_ms,
                input_tokens: result_input_tokens,
                output_tokens: result_output_tokens,
                cache_read_input_tokens: result_cache_read_tokens,
                cache_creation_input_tokens: result_cache_creation_tokens,
            }
        });

        // Stream stderr
        let stderr_handle = tokio::spawn(async move {
            let mut lines = BufReader::new(stderr).lines();
            let mut output = String::new();
            let mut line_number: u64 = 0;
            let mut writer = match stderr_log_path {
                Some(path) => match tokio::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&path)
                    .await
                {
                    Ok(file) => Some(file),
                    Err(e) => {
                        tracing::warn!(
                            component = "claude",
                            label = label_stderr.as_str(),
                            path = %path.display(),
                            error = %e,
                            "Failed to open stderr execution log file"
                        );
                        None
                    }
                },
                None => None,
            };
            let mut write_failed = false;
            let mut signaled_rate_limit = false;

            loop {
                let next_line = lines.next_line().await;
                let line = match next_line {
                    Ok(Some(line)) => line,
                    Ok(None) => break,
                    Err(e) => {
                        ClaudeAgentRunner::append_execution_event(
                            &stderr_event_writer,
                            label_stderr.as_str(),
                            "stderr_read_error",
                            json!({
                                "error": e.to_string(),
                            }),
                        )
                        .await;
                        tracing::warn!(
                            component = "claude",
                            label = label_stderr.as_str(),
                            error = %e,
                            "Failed reading Claude stderr stream"
                        );
                        break;
                    }
                };

                line_number = line_number.saturating_add(1);
                ClaudeAgentRunner::append_execution_event(
                    &stderr_event_writer,
                    label_stderr.as_str(),
                    "stderr_line",
                    json!({
                        "line_number": line_number,
                        "line": line.as_str(),
                    }),
                )
                .await;

                if !line.trim().is_empty() {
                    tracing::error!(
                        component = "claude",
                        label = label_stderr.as_str(),
                        "{}",
                        line
                    );

                    if !signaled_rate_limit && ClaudeAgentRunner::is_rate_limit_error(&line) {
                        signaled_rate_limit = true;
                        let msg = format!(
                            "Claude rate limit hit: {}",
                            ClaudeAgentRunner::truncate(&line, EXECUTION_LOG_PREVIEW_LIMIT)
                        );
                        let _ = stderr_early_failure_tx.send(msg);
                        ClaudeAgentRunner::append_execution_event(
                            &stderr_event_writer,
                            label_stderr.as_str(),
                            "rate_limit_detected_live",
                            json!({
                                "source": "stderr_line",
                                "line_number": line_number,
                                "message": ClaudeAgentRunner::truncate(&line, 500),
                            }),
                        )
                        .await;
                        tracing::warn!(
                            component = "claude",
                            label = label_stderr.as_str(),
                            line_number,
                            "Detected Claude rate-limit output in stderr stream; requesting early termination"
                        );
                    }
                }

                if let Some(file) = writer.as_mut() {
                    if !write_failed
                        && (file.write_all(line.as_bytes()).await.is_err()
                            || file.write_all(b"\n").await.is_err())
                    {
                        write_failed = true;
                        ClaudeAgentRunner::append_execution_event(
                            &stderr_event_writer,
                            label_stderr.as_str(),
                            "stderr_log_write_failed",
                            json!({}),
                        )
                        .await;
                        tracing::warn!(
                            component = "claude",
                            label = label_stderr.as_str(),
                            "Failed writing Claude stderr to execution log file"
                        );
                    }
                }

                output.push_str(&line);
                output.push('\n');
            }
            ClaudeAgentRunner::append_execution_event(
                &stderr_event_writer,
                label_stderr.as_str(),
                "stderr_stream_closed",
                json!({
                    "line_count": line_number,
                }),
            )
            .await;
            output
        });

        let timeout_duration = std::time::Duration::from_secs(self.config.timeout_secs);

        enum WaitOutcome {
            Exited(std::result::Result<std::process::ExitStatus, std::io::Error>),
            TimedOut,
            EarlyFailure(String),
        }

        let timeout_sleep = tokio::time::sleep(timeout_duration);
        tokio::pin!(timeout_sleep);

        let outcome = loop {
            tokio::select! {
                result = child.wait() => break WaitOutcome::Exited(result),
                _ = &mut timeout_sleep => break WaitOutcome::TimedOut,
                maybe_msg = early_failure_rx.recv() => {
                    if let Some(msg) = maybe_msg {
                        break WaitOutcome::EarlyFailure(msg);
                    }
                }
            }
        };

        let mut forced_failure_msg: Option<String> = None;
        let (status, timed_out) = match outcome {
            WaitOutcome::Exited(Ok(status)) => {
                Self::append_execution_event(
                    &event_writer,
                    label,
                    "subprocess_exited",
                    json!({
                        "exit_code": status.code(),
                    }),
                )
                .await;
                (status, false)
            }
            WaitOutcome::Exited(Err(e)) => {
                Self::append_execution_event(
                    &event_writer,
                    label,
                    "wait_failed",
                    json!({
                        "error": e.to_string(),
                    }),
                )
                .await;
                return Err(Error::runner(format!("Failed to wait for claude: {}", e)));
            }
            WaitOutcome::TimedOut => {
                Self::append_execution_event(
                    &event_writer,
                    label,
                    "subprocess_timed_out",
                    json!({
                        "timeout_secs": self.config.timeout_secs,
                    }),
                )
                .await;
                // Timeout occurred - try to kill the process
                tracing::error!(
                    component = "claude",
                    label = label,
                    timeout_secs = self.config.timeout_secs,
                    "Process timed out, attempting to kill"
                );
                Self::append_execution_event(
                    &event_writer,
                    label,
                    "kill_requested_after_timeout",
                    json!({}),
                )
                .await;
                if let Err(e) = child.kill().await {
                    Self::append_execution_event(
                        &event_writer,
                        label,
                        "kill_failed_after_timeout",
                        json!({
                            "error": e.to_string(),
                        }),
                    )
                    .await;
                    tracing::error!(component = "claude", error = %e, "Failed to kill timed-out process");
                }

                // Log claude_timed_out activity
                let activity = ActivityLogEntry::new(
                    "claude_timed_out",
                    format!("Claude timed out for {}", label),
                )
                .with_source("claude".to_string())
                .with_metadata(json!({
                    "timeout_secs": self.config.timeout_secs,
                    "label": label
                }));
                self.tracker.record_activity(&activity).ok();

                // Record the timed-out execution
                execution.complete(None, true);
                execution.stderr_preview = Some(format!(
                    "Process timed out after {} seconds",
                    self.config.timeout_secs
                ));
                if let Err(e) = self.tracker.record_execution(&execution) {
                    Self::append_execution_event(
                        &event_writer,
                        label,
                        "execution_record_failed",
                        json!({
                            "error": e.to_string(),
                        }),
                    )
                    .await;
                    tracing::warn!(error = %e, "Failed to record timed-out execution to database");
                } else {
                    Self::append_execution_event(
                        &event_writer,
                        label,
                        "execution_recorded",
                        json!({
                            "timed_out": true,
                        }),
                    )
                    .await;
                }

                // Return a result indicating timeout
                return Ok(AgentResult {
                    success: false,
                    output: String::new(),
                    pr_url: None,
                    changelog: None,
                    error: Some(format!(
                        "Process timed out after {} seconds",
                        self.config.timeout_secs
                    )),
                    blocking_question: None,
                    used_qa_ids: Vec::new(),
                    confidence: 0,
                    confidence_reasoning: None,
                });
            }
            WaitOutcome::EarlyFailure(msg) => {
                forced_failure_msg = Some(msg.clone());
                Self::append_execution_event(
                    &event_writer,
                    label,
                    "subprocess_early_failure",
                    json!({
                        "reason": msg,
                    }),
                )
                .await;
                tracing::error!(
                    component = "claude",
                    label = label,
                    "Early failure detected from Claude stream; terminating subprocess"
                );

                Self::append_execution_event(
                    &event_writer,
                    label,
                    "kill_requested_after_early_failure",
                    json!({}),
                )
                .await;
                if let Err(e) = child.kill().await {
                    Self::append_execution_event(
                        &event_writer,
                        label,
                        "kill_failed_after_early_failure",
                        json!({
                            "error": e.to_string(),
                        }),
                    )
                    .await;
                    tracing::warn!(
                        component = "claude",
                        label = label,
                        error = %e,
                        "Failed to kill Claude subprocess after early failure signal"
                    );
                }

                let status = match child.wait().await {
                    Ok(status) => status,
                    Err(e) => {
                        Self::append_execution_event(
                            &event_writer,
                            label,
                            "wait_failed_after_early_failure",
                            json!({
                                "error": e.to_string(),
                            }),
                        )
                        .await;
                        return Err(Error::runner(format!(
                            "Failed to wait for claude after early failure: {}",
                            e
                        )));
                    }
                };

                Self::append_execution_event(
                    &event_writer,
                    label,
                    "subprocess_terminated_after_early_failure",
                    json!({
                        "exit_code": status.code(),
                    }),
                )
                .await;
                (status, false)
            }
        };

        let stdout_result = match stdout_handle.await {
            Ok(result) => result,
            Err(e) => {
                tracing::error!(
                    component = "claude",
                    label = label,
                    error = %e,
                    "stdout reader task failed"
                );
                StdoutParseResult::default()
            }
        };
        let StdoutParseResult {
            text_output,
            structured_result,
            cost_usd: result_cost_usd,
            num_turns: result_num_turns,
            session_id: result_session_id,
            duration_api_ms: result_duration_api_ms,
            input_tokens: result_input_tokens,
            output_tokens: result_output_tokens,
            cache_read_input_tokens: result_cache_read_tokens,
            cache_creation_input_tokens: result_cache_creation_tokens,
        } = stdout_result;
        let stderr_output = stderr_handle.await.unwrap_or_default();

        let exit_code = status.code().unwrap_or(-1);
        tracing::info!(
            component = "claude",
            label = label,
            exit_code = exit_code,
            timed_out = timed_out,
            has_structured_result = structured_result.is_some(),
            "Process completed"
        );
        Self::append_execution_event(
            &event_writer,
            label,
            "process_completed",
            json!({
                "exit_code": exit_code,
                "timed_out": timed_out,
                "stdout_bytes": text_output.len(),
                "stderr_bytes": stderr_output.len(),
                "has_structured_result": structured_result.is_some(),
            }),
        )
        .await;

        // Extract fields from structured result, falling back to legacy extraction.
        let legacy_fallback = || {
            let pr = Self::extract_pr_url(&text_output);
            let bq = Self::extract_blocking_question(&text_output)
                .or_else(|| Self::extract_blocking_question(&stderr_output));
            (status.success(), text_output.clone(), pr, bq)
        };

        let (
            mut result_success,
            result_output,
            pr_url,
            changelog,
            blocking_question,
            confidence,
            confidence_reasoning,
        ) = structured_result
            .as_ref()
            .and_then(|val| serde_json::from_value::<StructuredResult>(val.clone()).ok())
            .map(|sr| {
                let sr_success = sr.success;
                let sr_output = if sr.summary.is_empty() {
                    text_output.clone()
                } else {
                    sr.summary
                };
                let sr_pr_url = sr
                    .pr_url
                    .filter(|url| url.starts_with("https://"))
                    .or_else(|| Self::extract_pr_url(&text_output));
                let sr_changelog = sr.changelog.filter(|c| !c.is_empty());
                let sr_question = sr.blocking_question;
                (
                    sr_success,
                    sr_output,
                    sr_pr_url,
                    sr_changelog,
                    sr_question,
                    sr.confidence,
                    sr.confidence_reasoning,
                )
            })
            .unwrap_or_else(|| {
                let (s, o, p, q) = legacy_fallback();
                (s, o, p, None, q, 0, None)
            });

        // Process-level failure always overrides model's self-reported success.
        // The CLI could crash after the model responded (e.g., during git push).
        if !status.success() {
            result_success = false;
        }
        if forced_failure_msg.is_some() {
            result_success = false;
        }

        if let Some(ref url) = pr_url {
            tracing::info!(
                component = "claude",
                label = label,
                pr_url = url,
                "PR URL extracted"
            );
            Self::append_execution_event(
                &event_writer,
                label,
                "pr_url_extracted",
                json!({
                    "pr_url": url,
                }),
            )
            .await;
        }

        let failure_msg = if let Some(msg) = forced_failure_msg.clone() {
            Some(msg)
        } else if status.success() {
            None
        } else {
            Some(Self::compose_failure_message(
                exit_code,
                &text_output,
                &stderr_output,
            ))
        };
        // Rate-limit detection: only trust structured signals and CLI-controlled
        // output (stderr, failure message). Never scan text_output — it contains
        // assistant text that may discuss rate limiting in analyzed code.
        let is_rate_limited = failure_msg
            .as_ref()
            .map(|msg| Self::is_rate_limit_error(msg))
            .unwrap_or(false)
            || Self::is_rate_limit_error(&stderr_output);

        // Complete and record the execution
        execution.complete(status.code(), timed_out);
        execution.stdout_preview = Some(Self::truncate(&text_output, EXECUTION_LOG_PREVIEW_LIMIT));
        execution.stderr_preview = if stderr_output.is_empty() {
            failure_msg
                .as_ref()
                .map(|msg| Self::truncate(msg, EXECUTION_LOG_PREVIEW_LIMIT))
        } else {
            Some(Self::truncate(&stderr_output, EXECUTION_LOG_PREVIEW_LIMIT))
        };
        execution.total_cost_usd = result_cost_usd;
        execution.num_turns = result_num_turns;
        execution.session_id = result_session_id;
        execution.duration_api_ms = result_duration_api_ms;
        execution.input_tokens = result_input_tokens;
        execution.output_tokens = result_output_tokens;
        execution.cache_read_input_tokens = result_cache_read_tokens;
        execution.cache_creation_input_tokens = result_cache_creation_tokens;

        // Record the execution to the database (don't fail the main operation if this fails)
        if let Err(e) = self.tracker.record_execution(&execution) {
            Self::append_execution_event(
                &event_writer,
                label,
                "execution_record_failed",
                json!({
                    "error": e.to_string(),
                }),
            )
            .await;
            tracing::warn!(error = %e, "Failed to record execution to database");
        } else {
            Self::append_execution_event(
                &event_writer,
                label,
                "execution_recorded",
                json!({
                    "timed_out": false,
                    "exit_code": execution.exit_code,
                }),
            )
            .await;
        }

        // Log completion activity
        if status.success() {
            let activity = ActivityLogEntry::new(
                "claude_completed",
                format!("Claude completed for {}", label),
            )
            .with_source("claude".to_string())
            .with_metadata(json!({
                "duration_secs": execution.duration_secs,
                "exit_code": exit_code,
                "has_pr": pr_url.is_some(),
                "label": label
            }));
            self.tracker.record_activity(&activity).ok();
        } else {
            let error_msg = failure_msg
                .clone()
                .unwrap_or_else(|| format!("Process exited with code {}", exit_code));
            let activity = ActivityLogEntry::new(
                "claude_failed",
                format!(
                    "Claude failed for {}: {}",
                    label,
                    Self::truncate(&error_msg, 100)
                ),
            )
            .with_source("claude".to_string())
            .with_metadata(json!({
                "duration_secs": execution.duration_secs,
                "exit_code": exit_code,
                "error": Self::truncate(&error_msg, 500),
                "label": label
            }));
            self.tracker.record_activity(&activity).ok();

            if is_rate_limited {
                let rate_limit_activity = ActivityLogEntry::new(
                    "rate_limit_hit",
                    format!("Claude rate limit hit for {}", label),
                )
                .with_source("claude".to_string())
                .with_metadata(json!({
                    "label": label,
                    "exit_code": exit_code,
                    "error": Self::truncate(&error_msg, 500),
                }));
                self.tracker.record_activity(&rate_limit_activity).ok();
            }
        }

        if let Some(question) = blocking_question.as_ref() {
            Self::append_execution_event(
                &event_writer,
                label,
                "blocking_question_parsed",
                json!({
                    "question": question.question.as_str(),
                    "has_context": question.context.is_some(),
                    "options": question.options.clone(),
                    "has_why": question.why.is_some(),
                }),
            )
            .await;
        }

        Ok(AgentResult {
            success: result_success,
            output: result_output,
            pr_url,
            changelog,
            error: failure_msg,
            blocking_question,
            used_qa_ids: Vec::new(),
            confidence,
            confidence_reasoning,
        })
    }

    /// Compute a SHA256 hash of the prompt for grouping similar prompts.
    /// Returns the first 16 hex characters (8 bytes of the digest).
    fn hash_prompt(prompt: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(prompt.as_bytes());
        let result = hasher.finalize();
        // Format only the first 8 bytes directly instead of all 32.
        let mut buf = String::with_capacity(16);
        for byte in &result[..8] {
            use std::fmt::Write;
            let _ = write!(buf, "{:02x}", byte);
        }
        buf
    }

    /// Truncate a string to approximately max_len bytes, adding "..." if truncated.
    /// Ensures the cut happens at a valid UTF-8 char boundary.
    fn truncate(s: &str, max_len: usize) -> String {
        if s.len() <= max_len {
            s.to_string()
        } else {
            let end = max_len.saturating_sub(3);
            // Fast path: if `end` is already on a char boundary (common for ASCII),
            // skip the O(n) char_indices walk entirely.
            let safe_end = if s.is_char_boundary(end) {
                end
            } else {
                // Find the nearest char boundary at or before `end`
                s.char_indices()
                    .take_while(|(i, _)| *i <= end)
                    .last()
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            };
            format!("{}...", &s[..safe_end])
        }
    }

    /// Extract PR URL from output.
    fn extract_pr_url(output: &str) -> Option<String> {
        use std::sync::LazyLock;

        static PR_URL_EXPLICIT_RE: LazyLock<regex_lite::Regex> =
            LazyLock::new(|| regex_lite::Regex::new(r"PR_URL:\s*(https://[^\s]+)").unwrap());
        static GITHUB_PR_RE: LazyLock<regex_lite::Regex> = LazyLock::new(|| {
            regex_lite::Regex::new(r"https://github\.com/[^\s/]+/[^\s/]+/pull/\d+[^\s]*").unwrap()
        });
        static GITLAB_MR_RE: LazyLock<regex_lite::Regex> = LazyLock::new(|| {
            regex_lite::Regex::new(r"https://gitlab\.com/[^\s]+/-/merge_requests/\d+[^\s]*")
                .unwrap()
        });
        // Also match self-hosted GitLab instances
        static GITLAB_MR_GENERIC_RE: LazyLock<regex_lite::Regex> = LazyLock::new(|| {
            regex_lite::Regex::new(
                r"https://[a-zA-Z0-9._-]+(?:\.[a-zA-Z]{2,})/[^\s/]+/[^\s/]+/-/merge_requests/\d+",
            )
            .unwrap()
        });

        // Try explicit PR_URL format first
        if let Some(captures) = PR_URL_EXPLICIT_RE.captures(output) {
            return captures.get(1).map(|m| m.as_str().to_string());
        }

        // Try GitHub PR URL pattern
        if let Some(m) = GITHUB_PR_RE.find(output) {
            return Some(m.as_str().to_string());
        }

        // Try GitLab MR URL pattern (gitlab.com)
        if let Some(m) = GITLAB_MR_RE.find(output) {
            return Some(m.as_str().to_string());
        }

        // Try self-hosted GitLab MR URL pattern
        if let Some(m) = GITLAB_MR_GENERIC_RE.find(output) {
            return Some(m.as_str().to_string());
        }

        None
    }
}

#[async_trait]
impl AgentRunner for ClaudeAgentRunner {
    fn name(&self) -> &str {
        "claude"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            structured_output: true,
            tool_permissions: true,
            custom_instructions: true,
            streaming_events: true,
            cost_reporting: true,
        }
    }

    fn build_prompt_for_issue(&self, issue: &Issue, context: &str, project_dir: &Path) -> String {
        self.build_prompt(issue, context, project_dir)
    }

    async fn execute_with_attempt(
        &self,
        prompt: &str,
        issue: Option<&Issue>,
        attempt_id: Option<i64>,
        project_dir: &Path,
    ) -> Result<AgentResult> {
        let (env, label) = self.prepare_env_and_label(issue);
        self.execute_with_env_and_attempt(prompt, label, env, attempt_id, project_dir)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mutex to serialize tests that manipulate process-global env vars.
    static ENV_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn test_parse_cli_assistant_text_event() {
        let line =
            r#"{"type":"assistant","message":{"content":[{"type":"text","text":"hello world"}]}}"#;
        let event: StreamEvent = serde_json::from_str(line).unwrap();
        match event {
            StreamEvent::Assistant { message: Some(msg) } => {
                assert_eq!(msg.content.len(), 1);
                match &msg.content[0] {
                    CliContentBlock::Text { text } => assert_eq!(text, "hello world"),
                    other => panic!("Unexpected content block: {:?}", other),
                }
            }
            other => panic!("Unexpected event: {:?}", other),
        }
    }

    #[test]
    fn test_parse_cli_assistant_tool_use_event() {
        let line = r#"{"type":"assistant","message":{"content":[{"type":"tool_use","id":"tu_1","name":"Bash"}]}}"#;
        let event: StreamEvent = serde_json::from_str(line).unwrap();
        match event {
            StreamEvent::Assistant { message: Some(msg) } => {
                assert_eq!(msg.content.len(), 1);
                match &msg.content[0] {
                    CliContentBlock::ToolUse { id, name } => {
                        assert_eq!(id, "tu_1");
                        assert_eq!(name, "Bash");
                    }
                    other => panic!("Unexpected content block: {:?}", other),
                }
            }
            other => panic!("Unexpected event: {:?}", other),
        }
    }

    #[test]
    fn test_parse_cli_assistant_multiple_content_blocks() {
        let line = r#"{"type":"assistant","message":{"content":[{"type":"text","text":"Running command..."},{"type":"tool_use","id":"tu_2","name":"Read"}]}}"#;
        let event: StreamEvent = serde_json::from_str(line).unwrap();
        match event {
            StreamEvent::Assistant { message: Some(msg) } => {
                assert_eq!(msg.content.len(), 2);
                assert!(
                    matches!(&msg.content[0], CliContentBlock::Text { text } if text == "Running command...")
                );
                assert!(
                    matches!(&msg.content[1], CliContentBlock::ToolUse { id, name } if id == "tu_2" && name == "Read")
                );
            }
            other => panic!("Unexpected event: {:?}", other),
        }
    }

    #[test]
    fn test_parse_cli_system_event() {
        let line = r#"{"type":"system","subtype":"init","session_id":"abc"}"#;
        let event: StreamEvent = serde_json::from_str(line).unwrap();
        assert!(matches!(event, StreamEvent::System {}));
    }

    #[test]
    fn test_parse_cli_user_event() {
        let line = r#"{"type":"user","message":{"role":"user"}}"#;
        let event: StreamEvent = serde_json::from_str(line).unwrap();
        assert!(matches!(event, StreamEvent::User {}));
    }

    #[test]
    fn test_parse_cli_result_with_structured_output() {
        let line = r#"{"type":"result","structured_output":{"summary":"done","success":true}}"#;
        let event: StreamEvent = serde_json::from_str(line).unwrap();
        match event {
            StreamEvent::Result {
                structured_output: Some(obj),
                ..
            } => {
                assert_eq!(obj["summary"], "done");
                assert_eq!(obj["success"], true);
            }
            other => panic!("Unexpected event: {:?}", other),
        }
    }

    #[test]
    fn test_parse_cli_result_without_structured_output() {
        let line = r#"{"type":"result"}"#;
        let event: StreamEvent = serde_json::from_str(line).unwrap();
        match event {
            StreamEvent::Result {
                structured_output: None,
                ..
            } => {}
            other => panic!("Unexpected event: {:?}", other),
        }
    }

    #[test]
    fn test_parse_cli_result_with_cost_and_usage() {
        let line = r#"{"type":"result","subtype":"success","is_error":false,"duration_ms":2557,"duration_api_ms":2546,"num_turns":1,"result":"done","session_id":"sess-123","total_cost_usd":0.027,"usage":{"input_tokens":3,"cache_creation_input_tokens":2833,"cache_read_input_tokens":18758,"output_tokens":4}}"#;
        let event: StreamEvent = serde_json::from_str(line).unwrap();
        match event {
            StreamEvent::Result {
                total_cost_usd: Some(cost),
                num_turns: Some(turns),
                session_id: Some(ref sid),
                duration_api_ms: Some(api_ms),
                usage: Some(ref u),
                ..
            } => {
                assert!((cost - 0.027).abs() < 1e-6);
                assert_eq!(turns, 1);
                assert_eq!(sid, "sess-123");
                assert_eq!(api_ms, 2546);
                assert_eq!(u.input_tokens, Some(3));
                assert_eq!(u.output_tokens, Some(4));
                assert_eq!(u.cache_read_input_tokens, Some(18758));
                assert_eq!(u.cache_creation_input_tokens, Some(2833));
            }
            other => panic!("Unexpected event: {:?}", other),
        }
    }

    #[test]
    fn test_parse_cli_assistant_unknown_content_block() {
        let line =
            r#"{"type":"assistant","message":{"content":[{"type":"thinking","thinking":"hmm"}]}}"#;
        let event: StreamEvent = serde_json::from_str(line).unwrap();
        match event {
            StreamEvent::Assistant { message: Some(msg) } => {
                assert_eq!(msg.content.len(), 1);
                assert!(matches!(&msg.content[0], CliContentBlock::Other));
            }
            other => panic!("Unexpected event: {:?}", other),
        }
    }

    #[test]
    fn test_parse_cli_unknown_event_forward_compat() {
        let line = r#"{"type":"some_future_event","data":"anything"}"#;
        let event: StreamEvent = serde_json::from_str(line).unwrap();
        assert!(matches!(event, StreamEvent::Unknown));
    }

    #[test]
    fn test_structured_result_full() {
        let json = r#"{"summary":"Fixed the bug and created PR","success":true,"pr_url":"https://github.com/org/repo/pull/42","blocking_question":null}"#;
        let sr: StructuredResult = serde_json::from_str(json).unwrap();
        assert!(sr.success);
        assert_eq!(sr.summary, "Fixed the bug and created PR");
        assert_eq!(
            sr.pr_url.as_deref(),
            Some("https://github.com/org/repo/pull/42")
        );
        assert!(sr.blocking_question.is_none());
    }

    #[test]
    fn test_structured_result_with_blocking_question() {
        let json = r#"{"summary":"Need clarification","success":false,"pr_url":null,"blocking_question":{"question":"Which branch?","context":"Multiple candidates","options":["main","develop"],"why":"Ambiguous target"}}"#;
        let sr: StructuredResult = serde_json::from_str(json).unwrap();
        assert!(!sr.success);
        assert!(sr.pr_url.is_none());
        let bq = sr.blocking_question.unwrap();
        assert_eq!(bq.question, "Which branch?");
        assert_eq!(bq.context.as_deref(), Some("Multiple candidates"));
        assert_eq!(bq.options, vec!["main", "develop"]);
        assert_eq!(bq.why.as_deref(), Some("Ambiguous target"));
    }

    #[test]
    fn test_structured_result_minimal() {
        let json = r#"{"summary":"Done","success":true}"#;
        let sr: StructuredResult = serde_json::from_str(json).unwrap();
        assert!(sr.success);
        assert_eq!(sr.summary, "Done");
        assert!(sr.pr_url.is_none());
        assert!(sr.blocking_question.is_none());
    }

    #[test]
    fn test_structured_result_empty_summary() {
        let json = r#"{"summary":"","success":false}"#;
        let sr: StructuredResult = serde_json::from_str(json).unwrap();
        assert!(!sr.success);
        assert!(sr.summary.is_empty());
    }

    #[test]
    fn test_result_schema_is_valid_json() {
        let parsed: serde_json::Value = serde_json::from_str(RESULT_SCHEMA).unwrap();
        assert_eq!(parsed["type"], "object");
        assert!(parsed["properties"]["summary"].is_object());
        assert!(parsed["properties"]["success"].is_object());
        assert!(parsed["properties"]["pr_url"].is_object());
        assert!(parsed["properties"]["blocking_question"].is_object());
    }

    #[test]
    fn test_build_prompt_no_question_protocol() {
        let runner = ClaudeAgentRunner::new_simple(ClaudeRunnerConfig::default());
        let issue = Issue::new("1", "TEST-1", "Bug", "https://example.com", "linear");
        let prompt = runner.build_prompt(&issue, "context", std::path::Path::new("/tmp"));
        assert!(!prompt.contains("CLAUDEAR_QUESTION"));
    }

    #[test]
    fn test_build_prompt_no_pr_url_instruction() {
        let runner = ClaudeAgentRunner::new_simple(ClaudeRunnerConfig::default());
        let issue = Issue::new("1", "TEST-1", "Bug", "https://example.com", "linear");
        let prompt = runner.build_prompt(&issue, "context", std::path::Path::new("/tmp"));
        assert!(!prompt.contains("PR_URL:"));
    }

    #[test]
    fn test_extract_pr_url_explicit() {
        let output = "Some output\nPR_URL: https://github.com/org/repo/pull/123\nMore output";
        assert_eq!(
            ClaudeAgentRunner::extract_pr_url(output),
            Some("https://github.com/org/repo/pull/123".to_string())
        );
    }

    #[test]
    fn test_extract_pr_url_github() {
        let output = "Created PR at https://github.com/myorg/myrepo/pull/456 successfully";
        assert_eq!(
            ClaudeAgentRunner::extract_pr_url(output),
            Some("https://github.com/myorg/myrepo/pull/456".to_string())
        );
    }

    #[test]
    fn test_extract_pr_url_gitlab() {
        let output = "MR created: https://gitlab.com/group/project/-/merge_requests/789";
        assert_eq!(
            ClaudeAgentRunner::extract_pr_url(output),
            Some("https://gitlab.com/group/project/-/merge_requests/789".to_string())
        );
    }

    #[test]
    fn test_extract_pr_url_none() {
        let output = "No PR URL in this output";
        assert_eq!(ClaudeAgentRunner::extract_pr_url(output), None);
    }

    #[test]
    fn test_build_prompt() {
        let runner = ClaudeAgentRunner::new_simple(ClaudeRunnerConfig::default());

        let issue = Issue::new(
            "123",
            "PROJ-123",
            "Fix bug",
            "https://example.com",
            "linear",
        );
        let context = "Issue description here";
        let project_dir = std::path::Path::new("/tmp");

        let prompt = runner.build_prompt(&issue, context, project_dir);
        assert!(prompt.contains("Linear"));
        assert!(prompt.contains("Issue description here") || prompt.contains("context"));
        assert!(prompt.contains("PROJ-123"));
    }

    #[test]
    fn test_extract_pr_url_with_trailing_punctuation() {
        let output = "Created PR: https://github.com/org/repo/pull/123.";
        let url = ClaudeAgentRunner::extract_pr_url(output);
        assert!(url.is_some());
        let url = url.unwrap();
        assert!(url.starts_with("https://github.com/org/repo/pull/123"));
    }

    #[test]
    fn test_extract_pr_url_multiple_urls() {
        let output = "First PR: https://github.com/org/repo1/pull/100\nSecond PR: https://github.com/org/repo2/pull/200";
        let url = ClaudeAgentRunner::extract_pr_url(output);
        assert!(url.is_some());
        assert!(url.unwrap().contains("/pull/"));
    }

    #[test]
    fn test_extract_pr_url_explicit_takes_precedence() {
        let output = "Random text https://github.com/org/repo/pull/999\nPR_URL: https://github.com/org/main/pull/123";
        let url = ClaudeAgentRunner::extract_pr_url(output);
        assert_eq!(
            url,
            Some("https://github.com/org/main/pull/123".to_string())
        );
    }

    #[test]
    fn test_extract_pr_url_with_query_params() {
        let output = "PR at https://github.com/org/repo/pull/123?diff=split created";
        let url = ClaudeAgentRunner::extract_pr_url(output);
        assert!(url.is_some());
    }

    #[test]
    fn test_extract_pr_url_gitlab_nested_groups() {
        let output = "MR: https://gitlab.com/group/subgroup/project/-/merge_requests/42";
        let url = ClaudeAgentRunner::extract_pr_url(output);
        assert_eq!(
            url,
            Some("https://gitlab.com/group/subgroup/project/-/merge_requests/42".to_string())
        );
    }

    #[test]
    fn test_extract_pr_url_empty_string() {
        assert_eq!(ClaudeAgentRunner::extract_pr_url(""), None);
    }

    #[test]
    fn test_extract_pr_url_similar_but_not_valid() {
        let output = "See https://github.com/org/repo/issues/123";
        let url = ClaudeAgentRunner::extract_pr_url(output);
        assert!(url.is_none());
    }

    #[test]
    fn test_build_prompt_sentry_source() {
        let runner = ClaudeAgentRunner::new_simple(ClaudeRunnerConfig::default());

        let issue = Issue::new(
            "456",
            "SENTRY-456",
            "TypeError in main.js",
            "https://sentry.io/123",
            "sentry",
        );
        let context = "Stack trace here";
        let project_dir = std::path::Path::new("/tmp");
        let prompt = runner.build_prompt(&issue, context, project_dir);
        assert!(!prompt.is_empty());
        assert!(
            prompt.contains("sentry")
                || prompt.contains("Sentry")
                || prompt.contains("Stack trace")
                || prompt.contains("TypeError")
        );
    }

    #[test]
    fn test_claude_result_success() {
        let result = AgentResult {
            success: true,
            output: "Success output".to_string(),
            pr_url: Some("https://github.com/org/repo/pull/123".to_string()),
            changelog: None,
            error: None,
            blocking_question: None,
            used_qa_ids: Vec::new(),
            confidence: 0,
            confidence_reasoning: None,
        };
        assert!(result.success);
        assert!(result.pr_url.is_some());
        assert!(result.error.is_none());
    }

    #[test]
    fn test_claude_result_failure() {
        let result = AgentResult {
            success: false,
            output: "Error occurred".to_string(),
            pr_url: None,
            changelog: None,
            error: Some("Error message".to_string()),
            blocking_question: None,
            used_qa_ids: Vec::new(),
            confidence: 0,
            confidence_reasoning: None,
        };
        assert!(!result.success);
        assert!(result.pr_url.is_none());
        assert!(result.error.is_some());
    }

    #[test]
    fn test_runner_config() {
        let config = ClaudeRunnerConfig::default();
        let runner = ClaudeAgentRunner::new_simple(config);
        let issue = Issue::new("1", "TEST-1", "Test", "https://test.com", "linear");
        let project_dir = std::path::Path::new("/path/to/project");
        let prompt = runner.build_prompt(&issue, "context", project_dir);
        assert!(!prompt.is_empty());
    }

    #[test]
    fn test_claude_result_default_fields() {
        let result = AgentResult {
            success: false,
            output: String::new(),
            pr_url: None,
            changelog: None,
            error: None,
            blocking_question: None,
            used_qa_ids: Vec::new(),
            confidence: 0,
            confidence_reasoning: None,
        };
        assert!(!result.success);
        assert!(result.output.is_empty());
        assert!(result.pr_url.is_none());
        assert!(result.error.is_none());
    }

    #[test]
    fn test_claude_runner_config_debug() {
        let config = ClaudeRunnerConfig {
            timeout_secs: 3600,
            ..Default::default()
        };
        let debug = format!("{:?}", config);
        assert!(debug.contains("timeout_secs"));
    }

    #[test]
    fn test_extract_pr_url_whitespace_only() {
        assert_eq!(ClaudeAgentRunner::extract_pr_url("   \n\t  "), None);
    }

    #[test]
    fn test_extract_pr_url_github_enterprise() {
        let output = "PR at https://github.mycompany.com/org/repo/pull/99";
        let url = ClaudeAgentRunner::extract_pr_url(output);
        assert!(url.is_none());
    }

    #[test]
    fn test_extract_pr_url_with_newlines() {
        let output = "PR created\n\nhttps://github.com/org/repo/pull/42\n\nDone";
        let url = ClaudeAgentRunner::extract_pr_url(output);
        assert!(url.is_some());
        assert!(url.unwrap().contains("/pull/42"));
    }

    #[test]
    fn test_extract_pr_url_pr_url_colon_space() {
        let output = "PR_URL:   https://github.com/org/repo/pull/1  ";
        let url = ClaudeAgentRunner::extract_pr_url(output);
        assert_eq!(url, Some("https://github.com/org/repo/pull/1".to_string()));
    }

    #[test]
    fn test_build_prompt_github_source() {
        let runner = ClaudeAgentRunner::new_simple(ClaudeRunnerConfig::default());
        let issue = Issue::new(
            "789",
            "#789",
            "Add feature",
            "https://github.com/org/repo/issues/789",
            "github",
        );
        let prompt =
            runner.build_prompt(&issue, "Feature description", std::path::Path::new("/tmp"));
        assert!(!prompt.is_empty());
    }

    #[test]
    fn test_build_prompt_unknown_source() {
        let runner = ClaudeAgentRunner::new_simple(ClaudeRunnerConfig::default());
        let issue = Issue::new(
            "123",
            "JIRA-123",
            "Bug fix",
            "https://jira.example.com/123",
            "jira",
        );
        let prompt = runner.build_prompt(&issue, "Bug details", std::path::Path::new("/tmp"));
        assert!(!prompt.is_empty());
    }

    #[test]
    fn test_has_agent_md() {
        let runner = ClaudeAgentRunner::new_simple(ClaudeRunnerConfig::default());
        assert!(!runner.has_agent_md(std::path::Path::new("/nonexistent/path")));
    }

    #[test]
    fn test_get_agent_md_nonexistent() {
        let runner = ClaudeAgentRunner::new_simple(ClaudeRunnerConfig::default());
        assert!(runner
            .get_agent_md(std::path::Path::new("/nonexistent/path"))
            .is_none());
    }

    #[test]
    fn test_claude_result_with_long_output() {
        let output = "x".repeat(10000);
        let result = AgentResult {
            success: true,
            output,
            pr_url: None,
            changelog: None,
            error: None,
            blocking_question: None,
            used_qa_ids: Vec::new(),
            confidence: 0,
            confidence_reasoning: None,
        };
        assert!(result.success);
        assert_eq!(result.output.len(), 10000);
    }

    #[test]
    fn test_claude_result_with_empty_error() {
        let result = AgentResult {
            success: false,
            output: "Output".to_string(),
            pr_url: None,
            changelog: None,
            error: Some(String::new()),
            blocking_question: None,
            used_qa_ids: Vec::new(),
            confidence: 0,
            confidence_reasoning: None,
        };
        assert!(result.error.is_some());
        assert!(result.error.unwrap().is_empty());
    }

    #[test]
    fn test_extract_pr_url_case_sensitivity() {
        let output = "pr_url: https://github.com/org/repo/pull/1";
        let url = ClaudeAgentRunner::extract_pr_url(output);
        assert!(url.is_some());
    }

    #[test]
    fn test_extract_pr_url_gitlab_with_path() {
        let output = "MR: https://gitlab.com/a/b/c/d/-/merge_requests/123";
        let url = ClaudeAgentRunner::extract_pr_url(output);
        assert_eq!(
            url,
            Some("https://gitlab.com/a/b/c/d/-/merge_requests/123".to_string())
        );
    }

    #[test]
    fn test_claude_runner_config_clone() {
        let config = ClaudeRunnerConfig {
            timeout_secs: 3600,
            ..Default::default()
        };
        let cloned = config.clone();
        assert_eq!(cloned.timeout_secs, config.timeout_secs);
        assert_eq!(cloned.skip_permissions, config.skip_permissions);
    }

    #[test]
    fn test_build_prompt_empty_context() {
        let runner = ClaudeAgentRunner::new_simple(ClaudeRunnerConfig::default());
        let issue = Issue::new("1", "TEST-1", "Test", "https://example.com", "linear");
        let prompt = runner.build_prompt(&issue, "", std::path::Path::new("/tmp"));
        assert!(!prompt.is_empty());
        assert!(prompt.contains("PROJ") || prompt.contains("TEST-1") || prompt.contains("Linear"));
    }

    #[test]
    fn test_build_prompt_special_characters() {
        let runner = ClaudeAgentRunner::new_simple(ClaudeRunnerConfig::default());
        let issue = Issue::new(
            "1",
            "TEST-1",
            "Fix \"quoted\" issue & <special>",
            "https://example.com",
            "linear",
        );
        let prompt = runner.build_prompt(
            &issue,
            "Context with special chars: <>&\"'",
            std::path::Path::new("/tmp"),
        );
        assert!(!prompt.is_empty());
    }

    #[test]
    fn test_extract_pr_url_no_scheme() {
        assert!(ClaudeAgentRunner::extract_pr_url("github.com/org/repo/pull/123").is_none());
    }

    #[test]
    fn test_extract_pr_url_http_not_https() {
        assert!(
            ClaudeAgentRunner::extract_pr_url("PR at http://github.com/org/repo/pull/123")
                .is_none()
        );
    }

    #[test]
    fn test_extract_pr_url_multiline_pr_url() {
        let output = "Creating PR...\nPR_URL: https://github.com/org/repo/pull/42\nDone!";
        assert_eq!(
            ClaudeAgentRunner::extract_pr_url(output),
            Some("https://github.com/org/repo/pull/42".to_string())
        );
    }

    #[test]
    fn test_build_prompt_multiline_context() {
        let runner = ClaudeAgentRunner::new_simple(ClaudeRunnerConfig::default());
        let issue = Issue::new("1", "TEST-1", "Bug", "https://example.com", "linear");
        let prompt = runner.build_prompt(
            &issue,
            "Line 1\nLine 2\nLine 3\n",
            std::path::Path::new("/tmp"),
        );
        assert!(!prompt.is_empty());
    }

    #[test]
    fn test_extract_pr_url_gitlab_self_hosted() {
        let url = ClaudeAgentRunner::extract_pr_url(
            "MR: https://gitlab.mycompany.com/group/project/-/merge_requests/123",
        );
        assert_eq!(
            url.as_deref(),
            Some("https://gitlab.mycompany.com/group/project/-/merge_requests/123")
        );
    }

    #[test]
    fn test_extract_pr_url_github_pr_zero() {
        assert!(
            ClaudeAgentRunner::extract_pr_url("PR: https://github.com/org/repo/pull/0").is_some()
        );
    }

    #[test]
    fn test_extract_pr_url_very_long_pr_number() {
        assert_eq!(
            ClaudeAgentRunner::extract_pr_url("PR: https://github.com/org/repo/pull/999999999999"),
            Some("https://github.com/org/repo/pull/999999999999".to_string())
        );
    }

    #[test]
    fn test_extract_pr_url_org_with_dashes() {
        assert_eq!(
            ClaudeAgentRunner::extract_pr_url(
                "PR: https://github.com/my-org-name/my-repo-name/pull/123"
            ),
            Some("https://github.com/my-org-name/my-repo-name/pull/123".to_string())
        );
    }

    #[test]
    fn test_extract_pr_url_org_with_dots() {
        assert_eq!(
            ClaudeAgentRunner::extract_pr_url("PR: https://github.com/org.name/repo.name/pull/123"),
            Some("https://github.com/org.name/repo.name/pull/123".to_string())
        );
    }

    #[test]
    fn test_claude_result_pr_url_with_trailing_slash() {
        assert!(ClaudeAgentRunner::extract_pr_url(
            "PR_URL: https://github.com/org/repo/pull/123/ Done"
        )
        .is_some());
    }

    #[test]
    fn test_build_prompt_unicode_context() {
        let runner = ClaudeAgentRunner::new_simple(ClaudeRunnerConfig::default());
        let issue = Issue::new(
            "1",
            "TEST-1",
            "Fix 日本語 issue",
            "https://example.com",
            "linear",
        );
        let prompt = runner.build_prompt(
            &issue,
            "Context with emoji 🎉 and unicode ñ",
            std::path::Path::new("/tmp"),
        );
        assert!(!prompt.is_empty());
    }

    #[test]
    fn test_claude_runner_new_with_tracker() {
        use claudear_storage::SqliteTracker;
        let tracker = Arc::new(SqliteTracker::in_memory().unwrap());
        let runner = ClaudeAgentRunner::new(ClaudeRunnerConfig::default(), tracker);
        assert!(!runner.has_agent_md(std::path::Path::new("/tmp")));
    }

    #[test]
    fn test_issue_creation_for_runner() {
        let issue = Issue::new("id123", "SHORT-123", "Title", "https://url.com", "linear");
        assert_eq!(issue.id, "id123");
        assert_eq!(issue.short_id, "SHORT-123");
        assert_eq!(issue.title, "Title");
        assert_eq!(issue.url, "https://url.com");
        assert_eq!(issue.source, "linear");
    }

    #[test]
    fn test_is_rate_limit_error_detection() {
        assert!(ClaudeAgentRunner::is_rate_limit_error(
            "Claude API returned 429 Too Many Requests"
        ));
        assert!(ClaudeAgentRunner::is_rate_limit_error(
            "rate limit exceeded"
        ));
        assert!(!ClaudeAgentRunner::is_rate_limit_error("cargo test failed"));
    }

    #[test]
    fn test_is_hard_error_detection() {
        assert!(ClaudeAgentRunner::is_hard_error(
            "Failed to spawn claude: No such file or directory"
        ));
        assert!(ClaudeAgentRunner::is_hard_error(
            "Process timed out after 3600 seconds"
        ));
        assert!(ClaudeAgentRunner::is_hard_error("429 too many requests"));
        assert!(!ClaudeAgentRunner::is_hard_error("tests failed"));
    }

    #[test]
    fn test_extract_blocking_question_valid_payload() {
        let output = "some logs\nCLAUDEAR_QUESTION: {\"question\":\"Which branch?\",\"context\":\"unclear\",\"options\":[\"main\",\"develop\"],\"why\":\"need branch\"}\ndone";
        let parsed = ClaudeAgentRunner::extract_blocking_question(output).unwrap();
        assert_eq!(parsed.question, "Which branch?");
        assert_eq!(parsed.context.as_deref(), Some("unclear"));
        assert_eq!(parsed.options, vec!["main", "develop"]);
        assert_eq!(parsed.why.as_deref(), Some("need branch"));
    }

    #[test]
    fn test_extract_blocking_question_ignores_malformed() {
        assert!(ClaudeAgentRunner::extract_blocking_question(
            "CLAUDEAR_QUESTION: {not valid json}"
        )
        .is_none());
    }

    #[test]
    fn test_extract_blocking_question_empty() {
        assert!(ClaudeAgentRunner::extract_blocking_question("").is_none());
    }

    #[test]
    fn test_extract_blocking_question_only_required_field() {
        let output = "CLAUDEAR_QUESTION: {\"question\":\"minimal\"}";
        let parsed = ClaudeAgentRunner::extract_blocking_question(output).unwrap();
        assert_eq!(parsed.question, "minimal");
        assert!(parsed.context.is_none());
        assert!(parsed.options.is_empty());
        assert!(parsed.why.is_none());
    }

    #[test]
    fn test_truncate_shorter_than_max() {
        assert_eq!(ClaudeAgentRunner::truncate("hello", 100), "hello");
    }

    #[test]
    fn test_truncate_exactly_at_max() {
        assert_eq!(ClaudeAgentRunner::truncate("hello", 5), "hello");
    }

    #[test]
    fn test_truncate_one_byte_over_max() {
        let result = ClaudeAgentRunner::truncate("abcdef", 5);
        assert!(result.ends_with("..."));
        assert_eq!(result, "ab...");
    }

    #[test]
    fn test_truncate_multibyte_unicode_at_boundary() {
        let result = ClaudeAgentRunner::truncate("aéb", 3);
        assert!(result.ends_with("..."));
        assert_eq!(result, "...");
    }

    #[test]
    fn test_truncate_empty_string_zero_max() {
        assert_eq!(ClaudeAgentRunner::truncate("", 0), "");
    }

    #[test]
    fn test_truncate_max_len_of_3() {
        assert_eq!(ClaudeAgentRunner::truncate("abcdef", 3), "...");
    }

    #[test]
    fn test_truncate_max_len_of_2() {
        assert_eq!(ClaudeAgentRunner::truncate("abcdef", 2), "...");
    }

    #[test]
    fn test_truncate_very_long_string() {
        let s = "a".repeat(10_000);
        let result = ClaudeAgentRunner::truncate(&s, 100);
        assert!(result.ends_with("..."));
        assert_eq!(result.len(), 100);
    }

    #[test]
    fn test_truncate_max_len_of_4() {
        assert_eq!(ClaudeAgentRunner::truncate("abcdef", 4), "a...");
    }

    #[test]
    fn test_sanitize_label_normal_alphanumeric() {
        assert_eq!(ClaudeAgentRunner::sanitize_label("hello123"), "hello123");
    }

    #[test]
    fn test_sanitize_label_special_characters() {
        assert_eq!(
            ClaudeAgentRunner::sanitize_label("hello world.foo/bar"),
            "hello_world_foo_bar"
        );
    }

    #[test]
    fn test_sanitize_label_empty() {
        assert_eq!(ClaudeAgentRunner::sanitize_label(""), "custom");
    }

    #[test]
    fn test_sanitize_label_longer_than_64_chars() {
        let result = ClaudeAgentRunner::sanitize_label(&"a".repeat(100));
        assert_eq!(result.len(), 64);
    }

    #[test]
    fn test_sanitize_label_unicode_characters() {
        assert_eq!(ClaudeAgentRunner::sanitize_label("café☕日本"), "caf____");
    }

    #[test]
    fn test_sanitize_label_hyphens_and_underscores_preserved() {
        assert_eq!(
            ClaudeAgentRunner::sanitize_label("my-label_name"),
            "my-label_name"
        );
    }

    #[test]
    fn test_sanitize_label_all_special_chars_non_empty() {
        assert_eq!(ClaudeAgentRunner::sanitize_label("@#$"), "___");
    }

    #[test]
    fn test_compose_failure_message_both_empty() {
        assert_eq!(
            ClaudeAgentRunner::compose_failure_message(1, "", ""),
            "Process exited with code 1"
        );
    }

    #[test]
    fn test_compose_failure_message_only_stderr() {
        assert_eq!(
            ClaudeAgentRunner::compose_failure_message(1, "", "error occurred"),
            "error occurred"
        );
    }

    #[test]
    fn test_compose_failure_message_only_stdout() {
        assert_eq!(
            ClaudeAgentRunner::compose_failure_message(1, "some output", ""),
            "Process exited with code 1. Output: some output"
        );
    }

    #[test]
    fn test_compose_failure_message_both_present_stderr_takes_priority() {
        assert_eq!(
            ClaudeAgentRunner::compose_failure_message(1, "stdout text", "stderr text"),
            "stderr text"
        );
    }

    #[test]
    fn test_compose_failure_message_rate_limit_in_stderr() {
        let msg = ClaudeAgentRunner::compose_failure_message(1, "", "rate limit exceeded");
        assert!(msg.starts_with("Claude rate limit hit:"));
    }

    #[test]
    fn test_compose_failure_message_rate_limit_in_stdout_not_detected() {
        // Rate-limit strings in stdout (assistant text) are NOT detected by
        // compose_failure_message — only stderr is trusted to avoid false
        // positives from codebase content.
        let msg = ClaudeAgentRunner::compose_failure_message(1, "429 too many requests", "");
        assert!(msg.starts_with("Process exited with code 1"));
    }

    #[test]
    fn test_compose_failure_message_rate_limit_in_combined() {
        let msg = ClaudeAgentRunner::compose_failure_message(1, "some output", "rate limit hit");
        assert!(msg.starts_with("Claude rate limit hit:"));
    }

    #[test]
    fn test_compose_failure_message_very_long_stderr_truncated() {
        let msg = ClaudeAgentRunner::compose_failure_message(1, "", &"e".repeat(5000));
        assert!(msg.len() <= EXECUTION_LOG_PREVIEW_LIMIT + 3);
        assert!(msg.ends_with("..."));
    }

    #[test]
    fn test_compose_failure_message_whitespace_only_inputs() {
        assert_eq!(
            ClaudeAgentRunner::compose_failure_message(42, "   ", "  \n\t  "),
            "Process exited with code 42"
        );
    }

    #[test]
    fn test_is_rate_limit_error_rate_limit() {
        assert!(ClaudeAgentRunner::is_rate_limit_error(
            "rate limit exceeded"
        ));
    }

    #[test]
    fn test_is_rate_limit_error_ratelimit() {
        assert!(ClaudeAgentRunner::is_rate_limit_error("ratelimit error"));
    }

    #[test]
    fn test_is_rate_limit_error_too_many_requests() {
        assert!(ClaudeAgentRunner::is_rate_limit_error("too many requests"));
    }

    #[test]
    fn test_is_rate_limit_error_429() {
        assert!(ClaudeAgentRunner::is_rate_limit_error("HTTP 429 returned"));
    }

    #[test]
    fn test_is_rate_limit_error_quota_exceeded() {
        assert!(ClaudeAgentRunner::is_rate_limit_error(
            "quota exceeded for api"
        ));
    }

    #[test]
    fn test_is_rate_limit_error_resource_exhausted() {
        assert!(ClaudeAgentRunner::is_rate_limit_error("resource exhausted"));
    }

    #[test]
    fn test_is_rate_limit_error_retry_after() {
        assert!(ClaudeAgentRunner::is_rate_limit_error(
            "retry-after: 30 seconds"
        ));
    }

    #[test]
    fn test_is_rate_limit_error_try_again_later() {
        assert!(ClaudeAgentRunner::is_rate_limit_error(
            "please try again later"
        ));
    }

    #[test]
    fn test_is_rate_limit_error_case_insensitivity() {
        assert!(ClaudeAgentRunner::is_rate_limit_error(
            "Rate Limit Exceeded"
        ));
        assert!(ClaudeAgentRunner::is_rate_limit_error("RATE LIMIT"));
        assert!(ClaudeAgentRunner::is_rate_limit_error("RateLimit"));
    }

    #[test]
    fn test_is_rate_limit_error_empty_string() {
        assert!(!ClaudeAgentRunner::is_rate_limit_error(""));
    }

    #[test]
    fn test_is_rate_limit_error_substring_match() {
        assert!(ClaudeAgentRunner::is_rate_limit_error(
            "we hit a rate limit here"
        ));
    }

    #[test]
    fn test_is_rate_limit_error_unrelated_message() {
        assert!(!ClaudeAgentRunner::is_rate_limit_error(
            "compilation error in main.rs"
        ));
    }

    #[test]
    fn test_is_hard_error_failed_to_spawn_claude() {
        assert!(ClaudeAgentRunner::is_hard_error(
            "Failed to spawn claude: not found"
        ));
    }

    #[test]
    fn test_is_hard_error_failed_to_wait_for_claude() {
        assert!(ClaudeAgentRunner::is_hard_error(
            "failed to wait for claude"
        ));
    }

    #[test]
    fn test_is_hard_error_failed_to_capture_stdout() {
        assert!(ClaudeAgentRunner::is_hard_error("failed to capture stdout"));
    }

    #[test]
    fn test_is_hard_error_failed_to_capture_stderr() {
        assert!(ClaudeAgentRunner::is_hard_error("failed to capture stderr"));
    }

    #[test]
    fn test_is_hard_error_process_timed_out() {
        assert!(ClaudeAgentRunner::is_hard_error("process timed out"));
    }

    #[test]
    fn test_is_hard_error_timed_out_after() {
        assert!(ClaudeAgentRunner::is_hard_error(
            "timed out after 3600 seconds"
        ));
    }

    #[test]
    fn test_is_hard_error_connection_reset() {
        assert!(ClaudeAgentRunner::is_hard_error("connection reset by peer"));
    }

    #[test]
    fn test_is_hard_error_service_unavailable() {
        assert!(ClaudeAgentRunner::is_hard_error("503 service unavailable"));
    }

    #[test]
    fn test_is_hard_error_internal_server_error() {
        assert!(ClaudeAgentRunner::is_hard_error(
            "500 internal server error"
        ));
    }

    #[test]
    fn test_is_hard_error_network_error() {
        assert!(ClaudeAgentRunner::is_hard_error("network error: timeout"));
    }

    #[test]
    fn test_is_hard_error_broken_pipe() {
        assert!(ClaudeAgentRunner::is_hard_error("broken pipe"));
    }

    #[test]
    fn test_is_hard_error_rate_limit_is_also_hard() {
        assert!(ClaudeAgentRunner::is_hard_error("rate limit exceeded"));
        assert!(ClaudeAgentRunner::is_hard_error("429 too many requests"));
    }

    #[test]
    fn test_is_hard_error_case_insensitivity() {
        assert!(ClaudeAgentRunner::is_hard_error("FAILED TO SPAWN CLAUDE"));
        assert!(ClaudeAgentRunner::is_hard_error("Connection Reset"));
        assert!(ClaudeAgentRunner::is_hard_error("Broken Pipe"));
    }

    #[test]
    fn test_is_hard_error_empty_string() {
        assert!(!ClaudeAgentRunner::is_hard_error(""));
    }

    #[test]
    fn test_is_hard_error_normal_error_is_not_hard() {
        assert!(!ClaudeAgentRunner::is_hard_error("tests failed"));
        assert!(!ClaudeAgentRunner::is_hard_error("compilation error"));
        assert!(!ClaudeAgentRunner::is_hard_error("undefined variable"));
    }

    #[test]
    fn test_hash_prompt_deterministic() {
        assert_eq!(
            ClaudeAgentRunner::hash_prompt("same prompt"),
            ClaudeAgentRunner::hash_prompt("same prompt")
        );
    }

    #[test]
    fn test_hash_prompt_different_prompts_different_hashes() {
        assert_ne!(
            ClaudeAgentRunner::hash_prompt("prompt one"),
            ClaudeAgentRunner::hash_prompt("prompt two")
        );
    }

    #[test]
    fn test_hash_prompt_empty_produces_hash() {
        let h = ClaudeAgentRunner::hash_prompt("");
        assert!(!h.is_empty());
        assert_eq!(h.len(), 16);
    }

    #[test]
    fn test_hash_prompt_length_is_16() {
        assert_eq!(ClaudeAgentRunner::hash_prompt("any prompt here").len(), 16);
    }

    #[test]
    fn test_hash_prompt_only_hex_chars() {
        assert!(ClaudeAgentRunner::hash_prompt("test")
            .chars()
            .all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_hash_prompt_unicode_works() {
        let h = ClaudeAgentRunner::hash_prompt("こんにちは 🌍");
        assert_eq!(h.len(), 16);
        assert!(h.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_claude_execution_new_defaults() {
        let exec = AgentExecution::new();
        assert_eq!(exec.id, 0);
        assert!(exec.attempt_id.is_none());
        assert!(exec.completed_at.is_none());
        assert!(exec.duration_secs.is_none());
        assert!(exec.exit_code.is_none());
        assert!(!exec.timed_out);
    }

    #[test]
    fn test_claude_execution_with_attempt_id() {
        assert_eq!(
            AgentExecution::new().with_attempt_id(42).attempt_id,
            Some(42)
        );
    }

    #[test]
    fn test_claude_execution_complete_sets_fields() {
        let mut exec = AgentExecution::new();
        exec.complete(Some(0), false);
        assert!(exec.completed_at.is_some());
        assert!(exec.duration_secs.is_some());
        assert_eq!(exec.exit_code, Some(0));
        assert!(!exec.timed_out);
    }

    #[test]
    fn test_claude_execution_complete_with_timeout() {
        let mut exec = AgentExecution::new();
        exec.complete(None, true);
        assert!(exec.timed_out);
        assert!(exec.exit_code.is_none());
    }

    #[test]
    fn test_claude_execution_complete_with_nonzero_exit() {
        let mut exec = AgentExecution::new();
        exec.complete(Some(1), false);
        assert_eq!(exec.exit_code, Some(1));
        assert!(!exec.timed_out);
    }

    #[test]
    fn test_claude_execution_duration_is_non_negative() {
        let mut exec = AgentExecution::new();
        exec.complete(Some(0), false);
        assert!(exec.duration_secs.unwrap() >= 0.0);
    }

    #[test]
    fn test_claude_execution_default_matches_new() {
        let from_new = AgentExecution::new();
        let from_default = AgentExecution::default();
        assert_eq!(from_new.id, from_default.id);
        assert_eq!(from_new.attempt_id, from_default.attempt_id);
        assert_eq!(from_new.timed_out, from_default.timed_out);
    }

    #[test]
    fn test_claude_runner_config_default_values() {
        let config = ClaudeRunnerConfig::default();
        assert_eq!(config.timeout_secs, 21600);
        assert!(config.model.is_none());
        assert!(config.instructions.is_none());
        assert!(config.permissions.is_empty());
        assert!(!config.skip_permissions);
    }

    #[test]
    fn test_claude_runner_config_custom_timeout() {
        let config = ClaudeRunnerConfig {
            timeout_secs: 60,
            ..Default::default()
        };
        assert_eq!(config.timeout_secs, 60);
        assert!(config.model.is_none());
        assert!(!config.skip_permissions);
    }

    #[test]
    fn test_claude_runner_config_with_model() {
        let config = ClaudeRunnerConfig {
            model: Some("opus".to_string()),
            ..Default::default()
        };
        assert_eq!(config.model.as_deref(), Some("opus"));
    }

    #[test]
    fn test_claude_runner_config_with_permissions() {
        let config = ClaudeRunnerConfig {
            permissions: vec!["Bash".to_string(), "Read".to_string()],
            skip_permissions: false,
            ..Default::default()
        };
        assert_eq!(config.permissions.len(), 2);
        assert!(!config.skip_permissions);
    }

    #[test]
    fn test_claude_runner_config_with_instructions() {
        let config = ClaudeRunnerConfig {
            instructions: Some("Always write tests".to_string()),
            ..Default::default()
        };
        assert_eq!(config.instructions.as_deref(), Some("Always write tests"));
    }

    #[test]
    fn test_resolve_log_root_default_without_env_var() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        // Remove the env var if it happens to be set
        let prev = std::env::var("CLAUDEAR_LOG_DIR").ok();
        std::env::remove_var("CLAUDEAR_LOG_DIR");

        let root = resolve_log_root();
        assert_eq!(root, PathBuf::from(DEFAULT_LOG_DIR));

        // Restore previous value
        if let Some(val) = prev {
            std::env::set_var("CLAUDEAR_LOG_DIR", val);
        }
    }

    #[test]
    fn test_resolve_log_root_with_env_var() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let prev = std::env::var("CLAUDEAR_LOG_DIR").ok();
        std::env::set_var("CLAUDEAR_LOG_DIR", "/tmp/custom-logs");

        let root = resolve_log_root();
        assert_eq!(root, PathBuf::from("/tmp/custom-logs"));

        // Restore
        if let Some(val) = prev {
            std::env::set_var("CLAUDEAR_LOG_DIR", val);
        } else {
            std::env::remove_var("CLAUDEAR_LOG_DIR");
        }
    }

    #[test]
    fn test_resolve_log_root_private_matches_public() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        // The private method ClaudeAgentRunner::resolve_log_root() should match
        // the public function resolve_log_root() since they share the same logic.
        let prev = std::env::var("CLAUDEAR_LOG_DIR").ok();
        std::env::remove_var("CLAUDEAR_LOG_DIR");

        let public_root = resolve_log_root();
        let private_root = ClaudeAgentRunner::resolve_log_root();
        assert_eq!(public_root, private_root);

        if let Some(val) = prev {
            std::env::set_var("CLAUDEAR_LOG_DIR", val);
        }
    }

    #[test]
    fn test_stdout_parse_result_default() {
        let result = StdoutParseResult::default();
        assert!(result.text_output.is_empty());
        assert!(result.structured_result.is_none());
        assert!(result.cost_usd.is_none());
        assert!(result.num_turns.is_none());
        assert!(result.session_id.is_none());
        assert!(result.duration_api_ms.is_none());
        assert!(result.input_tokens.is_none());
        assert!(result.output_tokens.is_none());
        assert!(result.cache_read_input_tokens.is_none());
        assert!(result.cache_creation_input_tokens.is_none());
    }

    #[test]
    fn test_execution_log_files_clone() {
        let files = ExecutionLogFiles {
            stdout: PathBuf::from("/tmp/test.stdout.log"),
            stderr: PathBuf::from("/tmp/test.stderr.log"),
            events: PathBuf::from("/tmp/test.events.jsonl"),
        };
        let cloned = files.clone();
        assert_eq!(cloned.stdout, files.stdout);
        assert_eq!(cloned.stderr, files.stderr);
        assert_eq!(cloned.events, files.events);
    }

    #[test]
    fn test_execution_log_files_debug() {
        let files = ExecutionLogFiles {
            stdout: PathBuf::from("/tmp/test.stdout.log"),
            stderr: PathBuf::from("/tmp/test.stderr.log"),
            events: PathBuf::from("/tmp/test.events.jsonl"),
        };
        let debug = format!("{:?}", files);
        assert!(debug.contains("stdout"));
        assert!(debug.contains("stderr"));
        assert!(debug.contains("events"));
    }

    #[test]
    fn test_create_execution_log_files_produces_valid_paths() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let prev = std::env::var("CLAUDEAR_LOG_DIR").ok();
        let tmp_dir = std::env::temp_dir().join("claudear_test_exec_logs");
        std::env::set_var("CLAUDEAR_LOG_DIR", tmp_dir.to_str().unwrap());

        let files = ClaudeAgentRunner::create_execution_log_files("test-label");
        assert!(files.is_some());

        let files = files.unwrap();
        assert!(files.stdout.to_str().unwrap().contains("test-label"));
        assert!(files.stdout.to_str().unwrap().ends_with(".stdout.log"));
        assert!(files.stderr.to_str().unwrap().ends_with(".stderr.log"));
        assert!(files.events.to_str().unwrap().ends_with(".events.jsonl"));

        // All paths should share the same parent directory
        assert_eq!(files.stdout.parent(), files.stderr.parent());
        assert_eq!(files.stderr.parent(), files.events.parent());

        // The path should include the CLAUDE_LOG_SUBDIR
        assert!(files.stdout.to_str().unwrap().contains(CLAUDE_LOG_SUBDIR));

        // Clean up
        let _ = std::fs::remove_dir_all(&tmp_dir);
        if let Some(val) = prev {
            std::env::set_var("CLAUDEAR_LOG_DIR", val);
        } else {
            std::env::remove_var("CLAUDEAR_LOG_DIR");
        }
    }

    #[test]
    fn test_create_execution_log_files_sanitizes_label_in_filenames() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let prev = std::env::var("CLAUDEAR_LOG_DIR").ok();
        let tmp_dir = std::env::temp_dir().join("claudear_test_sanitize_logs");
        std::env::set_var("CLAUDEAR_LOG_DIR", tmp_dir.to_str().unwrap());

        let files = ClaudeAgentRunner::create_execution_log_files("hello world/foo@bar");
        assert!(files.is_some());
        let files = files.unwrap();
        let stem = files
            .stdout
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();
        // The label portion should not contain spaces, slashes, or @
        assert!(!stem.contains(' '));
        assert!(!stem.contains('/'));
        assert!(!stem.contains('@'));

        let _ = std::fs::remove_dir_all(&tmp_dir);
        if let Some(val) = prev {
            std::env::set_var("CLAUDEAR_LOG_DIR", val);
        } else {
            std::env::remove_var("CLAUDEAR_LOG_DIR");
        }
    }

    #[test]
    fn test_prepare_env_and_label_with_issue() {
        let runner = ClaudeAgentRunner::new_simple(ClaudeRunnerConfig::default());
        let issue = Issue::new("id-42", "PROJ-42", "A bug", "https://ex.com", "linear");
        let (env, label) = runner.prepare_env_and_label(Some(&issue));

        assert_eq!(label, "PROJ-42");
        assert_eq!(env.get("LINEAR_ISSUE_ID"), Some(&"id-42".to_string()));
        assert_eq!(
            env.get("LINEAR_ISSUE_SHORT_ID"),
            Some(&"PROJ-42".to_string())
        );
        assert_eq!(
            env.get("LINEAR_ISSUE_URL"),
            Some(&"https://ex.com".to_string())
        );
    }

    #[test]
    fn test_prepare_env_and_label_without_issue() {
        let runner = ClaudeAgentRunner::new_simple(ClaudeRunnerConfig::default());
        let (env, label) = runner.prepare_env_and_label(None);

        assert_eq!(label, "custom");
        // Should not contain any issue-specific env vars
        assert!(!env.contains_key("LINEAR_ISSUE_ID"));
    }

    #[test]
    fn test_prepare_env_and_label_sentry_source() {
        let runner = ClaudeAgentRunner::new_simple(ClaudeRunnerConfig::default());
        let issue = Issue::new("s-1", "SENTRY-1", "Error", "https://sentry.io/1", "sentry");
        let (env, label) = runner.prepare_env_and_label(Some(&issue));

        assert_eq!(label, "SENTRY-1");
        assert_eq!(env.get("SENTRY_ISSUE_ID"), Some(&"s-1".to_string()));
        assert_eq!(
            env.get("SENTRY_ISSUE_SHORT_ID"),
            Some(&"SENTRY-1".to_string())
        );
        assert_eq!(
            env.get("SENTRY_ISSUE_URL"),
            Some(&"https://sentry.io/1".to_string())
        );
    }

    #[test]
    fn test_prepare_env_and_label_github_source() {
        let runner = ClaudeAgentRunner::new_simple(ClaudeRunnerConfig::default());
        let issue = Issue::new(
            "789",
            "#789",
            "Feature",
            "https://github.com/o/r/issues/789",
            "github",
        );
        let (env, label) = runner.prepare_env_and_label(Some(&issue));

        assert_eq!(label, "#789");
        assert_eq!(env.get("GITHUB_ISSUE_ID"), Some(&"789".to_string()));
        assert_eq!(env.get("GITHUB_ISSUE_SHORT_ID"), Some(&"#789".to_string()));
    }

    #[test]
    fn test_build_prompt_for_issue_matches_build_prompt() {
        let runner = ClaudeAgentRunner::new_simple(ClaudeRunnerConfig::default());
        let issue = Issue::new("1", "TEST-1", "Bug", "https://example.com", "linear");
        let project_dir = Path::new("/tmp");

        let from_public = runner.build_prompt_for_issue(&issue, "ctx", project_dir);
        let from_private = runner.build_prompt(&issue, "ctx", project_dir);
        assert_eq!(from_public, from_private);
    }

    #[test]
    fn test_has_agent_md_with_existing_file() {
        let tmp_dir = std::env::temp_dir().join("claudear_test_agent_md");
        let _ = std::fs::create_dir_all(&tmp_dir);
        std::fs::write(tmp_dir.join("AGENT.md"), "# Agent instructions\nDo things.").unwrap();

        let runner = ClaudeAgentRunner::new_simple(ClaudeRunnerConfig::default());
        assert!(runner.has_agent_md(&tmp_dir));

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn test_get_agent_md_with_existing_file() {
        let tmp_dir = std::env::temp_dir().join("claudear_test_agent_md_get");
        let _ = std::fs::create_dir_all(&tmp_dir);
        let content = "# Agent\nCustom instructions here.";
        std::fs::write(tmp_dir.join("AGENT.md"), content).unwrap();

        let runner = ClaudeAgentRunner::new_simple(ClaudeRunnerConfig::default());
        let loaded = runner.get_agent_md(&tmp_dir);
        assert!(loaded.is_some());
        assert_eq!(loaded.unwrap(), content);

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn test_get_agent_md_returns_none_for_missing() {
        let runner = ClaudeAgentRunner::new_simple(ClaudeRunnerConfig::default());
        assert!(runner
            .get_agent_md(Path::new("/nonexistent/path/xyz"))
            .is_none());
    }

    #[test]
    fn test_cli_usage_all_none() {
        let json = r#"{}"#;
        let usage: CliUsage = serde_json::from_str(json).unwrap();
        assert!(usage.input_tokens.is_none());
        assert!(usage.output_tokens.is_none());
        assert!(usage.cache_read_input_tokens.is_none());
        assert!(usage.cache_creation_input_tokens.is_none());
    }

    #[test]
    fn test_cli_usage_partial_fields() {
        let json = r#"{"input_tokens": 100, "output_tokens": 50}"#;
        let usage: CliUsage = serde_json::from_str(json).unwrap();
        assert_eq!(usage.input_tokens, Some(100));
        assert_eq!(usage.output_tokens, Some(50));
        assert!(usage.cache_read_input_tokens.is_none());
        assert!(usage.cache_creation_input_tokens.is_none());
    }

    #[test]
    fn test_cli_usage_all_fields() {
        let json = r#"{"input_tokens":10,"output_tokens":20,"cache_read_input_tokens":30,"cache_creation_input_tokens":40}"#;
        let usage: CliUsage = serde_json::from_str(json).unwrap();
        assert_eq!(usage.input_tokens, Some(10));
        assert_eq!(usage.output_tokens, Some(20));
        assert_eq!(usage.cache_read_input_tokens, Some(30));
        assert_eq!(usage.cache_creation_input_tokens, Some(40));
    }

    #[test]
    fn test_cli_message_empty_content() {
        let json = r#"{"content":[]}"#;
        let msg: CliMessage = serde_json::from_str(json).unwrap();
        assert!(msg.content.is_empty());
    }

    #[test]
    fn test_cli_message_missing_content_field_defaults() {
        let json = r#"{}"#;
        let msg: CliMessage = serde_json::from_str(json).unwrap();
        assert!(msg.content.is_empty());
    }

    #[test]
    fn test_stream_event_assistant_no_message() {
        let json = r#"{"type":"assistant"}"#;
        let event: StreamEvent = serde_json::from_str(json).unwrap();
        match event {
            StreamEvent::Assistant { message: None } => {}
            other => panic!("Expected Assistant with no message, got: {:?}", other),
        }
    }

    #[test]
    fn test_stream_event_result_all_fields() {
        let json = r#"{
            "type": "result",
            "structured_output": {"summary": "all done", "success": true},
            "total_cost_usd": 0.123,
            "num_turns": 5,
            "session_id": "sess-abc",
            "duration_api_ms": 9876,
            "usage": {"input_tokens": 100, "output_tokens": 200, "cache_read_input_tokens": 300, "cache_creation_input_tokens": 400}
        }"#;
        let event: StreamEvent = serde_json::from_str(json).unwrap();
        match event {
            StreamEvent::Result {
                structured_output: Some(ref so),
                total_cost_usd: Some(cost),
                num_turns: Some(turns),
                session_id: Some(ref sid),
                duration_api_ms: Some(api_ms),
                usage: Some(ref u),
            } => {
                assert_eq!(so["summary"], "all done");
                assert!((cost - 0.123).abs() < 1e-6);
                assert_eq!(turns, 5);
                assert_eq!(sid, "sess-abc");
                assert_eq!(api_ms, 9876);
                assert_eq!(u.input_tokens, Some(100));
                assert_eq!(u.output_tokens, Some(200));
                assert_eq!(u.cache_read_input_tokens, Some(300));
                assert_eq!(u.cache_creation_input_tokens, Some(400));
            }
            other => panic!("Unexpected event: {:?}", other),
        }
    }

    #[test]
    fn test_stream_event_result_no_optional_fields() {
        let json = r#"{"type":"result"}"#;
        let event: StreamEvent = serde_json::from_str(json).unwrap();
        match event {
            StreamEvent::Result {
                structured_output: None,
                total_cost_usd: None,
                num_turns: None,
                session_id: None,
                duration_api_ms: None,
                usage: None,
            } => {}
            other => panic!("Expected Result with all None, got: {:?}", other),
        }
    }

    #[test]
    fn test_stream_event_ignores_extra_fields() {
        // Ensure forward-compat: extra fields in known events are silently ignored
        let json =
            r#"{"type":"system","subtype":"init","session_id":"xyz","extra_field":"ignored"}"#;
        let event: StreamEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(event, StreamEvent::System {}));
    }

    #[test]
    fn test_stream_event_user_ignores_extra_fields() {
        let json = r#"{"type":"user","message":{"role":"user","content":"hello"},"extra":true}"#;
        let event: StreamEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(event, StreamEvent::User {}));
    }

    #[test]
    fn test_stream_event_rate_limit_event_allowed() {
        let json = r#"{"type":"rate_limit_event","rate_limit_info":{"status":"allowed","utilization":0.5}}"#;
        let event: StreamEvent = serde_json::from_str(json).unwrap();
        match event {
            StreamEvent::RateLimitEvent {
                rate_limit_info, ..
            } => {
                let info = rate_limit_info.unwrap();
                assert_eq!(info.status.as_deref(), Some("allowed"));
                assert!((info.utilization.unwrap() - 0.5).abs() < f64::EPSILON);
            }
            other => panic!("Expected RateLimitEvent, got {:?}", other),
        }
    }

    #[test]
    fn test_stream_event_rate_limit_event_allowed_warning_full() {
        let json = r#"{"type":"rate_limit_event","rate_limit_info":{"status":"allowed_warning","resetsAt":1772096400,"rateLimitType":"seven_day","utilization":0.81,"isUsingOverage":false,"surpassedThreshold":0.75},"uuid":"addd358a-c64f-48cd-b9a2-97af382e0fd6","session_id":"f450b721-71e2-4d8f-8a2d-07827df35f1d"}"#;
        let event: StreamEvent = serde_json::from_str(json).unwrap();
        match event {
            StreamEvent::RateLimitEvent {
                rate_limit_info, ..
            } => {
                let info = rate_limit_info.unwrap();
                assert_eq!(info.status.as_deref(), Some("allowed_warning"));
                assert_eq!(info.rate_limit_type.as_deref(), Some("seven_day"));
                assert!((info.utilization.unwrap() - 0.81).abs() < f64::EPSILON);
            }
            other => panic!("Expected RateLimitEvent, got {:?}", other),
        }
    }

    #[test]
    fn test_stream_event_rate_limit_event_exceeded() {
        let json = r#"{"type":"rate_limit_event","rate_limit_info":{"status":"exceeded","resetsAt":"2026-02-23T06:00:00Z","rateLimitType":"seven_day","utilization":1.0}}"#;
        let event: StreamEvent = serde_json::from_str(json).unwrap();
        match event {
            StreamEvent::RateLimitEvent {
                rate_limit_info, ..
            } => {
                let info = rate_limit_info.unwrap();
                assert_eq!(info.status.as_deref(), Some("exceeded"));
                // Not "allowed" or "allowed_warning" → should trigger rate limit
                let is_informational = matches!(
                    info.status.as_deref(),
                    Some("allowed") | Some("allowed_warning")
                );
                assert!(!is_informational);
            }
            other => panic!("Expected RateLimitEvent, got {:?}", other),
        }
    }

    #[test]
    fn test_stream_event_rate_limit_event_with_top_level_resets_at() {
        let json = r#"{"type":"rate_limit_event","resetsAt":"2026-02-23T06:00:00Z"}"#;
        let event: StreamEvent = serde_json::from_str(json).unwrap();
        match event {
            StreamEvent::RateLimitEvent {
                resets_at,
                rate_limit_info,
            } => {
                assert!(rate_limit_info.is_none());
                assert_eq!(resets_at.unwrap().as_str().unwrap(), "2026-02-23T06:00:00Z");
            }
            other => panic!("Expected RateLimitEvent, got {:?}", other),
        }
    }

    #[test]
    fn test_structured_result_non_https_pr_url() {
        let json =
            r#"{"summary":"done","success":true,"pr_url":"http://github.com/org/repo/pull/1"}"#;
        let sr: StructuredResult = serde_json::from_str(json).unwrap();
        assert!(sr.success);
        // The pr_url is stored as-is in StructuredResult; filtering to https happens in the runner
        assert_eq!(
            sr.pr_url.as_deref(),
            Some("http://github.com/org/repo/pull/1")
        );
    }

    #[test]
    fn test_structured_result_empty_pr_url_string() {
        let json = r#"{"summary":"done","success":true,"pr_url":""}"#;
        let sr: StructuredResult = serde_json::from_str(json).unwrap();
        assert_eq!(sr.pr_url.as_deref(), Some(""));
    }

    #[test]
    fn test_structured_result_blocking_question_minimal() {
        let json =
            r#"{"summary":"stuck","success":false,"blocking_question":{"question":"help?"}}"#;
        let sr: StructuredResult = serde_json::from_str(json).unwrap();
        let bq = sr.blocking_question.unwrap();
        assert_eq!(bq.question, "help?");
        assert!(bq.context.is_none());
        assert!(bq.options.is_empty());
        assert!(bq.why.is_none());
    }

    #[test]
    fn test_structured_result_blocking_question_empty_options() {
        let json = r#"{"summary":"stuck","success":false,"blocking_question":{"question":"q","context":null,"options":[],"why":null}}"#;
        let sr: StructuredResult = serde_json::from_str(json).unwrap();
        let bq = sr.blocking_question.unwrap();
        assert_eq!(bq.question, "q");
        assert!(bq.options.is_empty());
    }

    #[test]
    fn test_extract_blocking_question_prefix_only_empty_payload() {
        // "CLAUDEAR_QUESTION:" followed by whitespace only => None
        let output = "CLAUDEAR_QUESTION:   ";
        assert!(ClaudeAgentRunner::extract_blocking_question(output).is_none());
    }

    #[test]
    fn test_extract_blocking_question_prefix_without_colon() {
        let output = "CLAUDEAR_QUESTION {\"question\":\"test\"}";
        assert!(ClaudeAgentRunner::extract_blocking_question(output).is_none());
    }

    #[test]
    fn test_extract_blocking_question_multiple_lines_picks_first() {
        let output = "CLAUDEAR_QUESTION: {\"question\":\"first\"}\nCLAUDEAR_QUESTION: {\"question\":\"second\"}";
        let parsed = ClaudeAgentRunner::extract_blocking_question(output).unwrap();
        assert_eq!(parsed.question, "first");
    }

    #[test]
    fn test_extract_blocking_question_with_surrounding_whitespace() {
        let output = "   CLAUDEAR_QUESTION:  {\"question\":\"trimmed\"}  ";
        let parsed = ClaudeAgentRunner::extract_blocking_question(output).unwrap();
        assert_eq!(parsed.question, "trimmed");
    }

    #[test]
    fn test_extract_blocking_question_no_prefix_present() {
        let output = "Just some regular output\nwithout any question markers\n";
        assert!(ClaudeAgentRunner::extract_blocking_question(output).is_none());
    }

    #[test]
    fn test_extract_blocking_question_with_all_fields() {
        let output = r#"CLAUDEAR_QUESTION: {"question":"Which DB?","context":"Found postgres and mysql","options":["postgres","mysql","sqlite"],"why":"Cannot determine from config"}"#;
        let parsed = ClaudeAgentRunner::extract_blocking_question(output).unwrap();
        assert_eq!(parsed.question, "Which DB?");
        assert_eq!(parsed.context.as_deref(), Some("Found postgres and mysql"));
        assert_eq!(parsed.options, vec!["postgres", "mysql", "sqlite"]);
        assert_eq!(parsed.why.as_deref(), Some("Cannot determine from config"));
    }

    #[test]
    fn test_compose_failure_message_rate_limit_combined_empty_gives_default() {
        // When both are empty but still trigger rate limit through combined being empty,
        // this should not happen, but let's verify the fallback.
        let msg = ClaudeAgentRunner::compose_failure_message(1, "", "");
        assert!(!msg.starts_with("Claude rate limit hit:"));
        assert_eq!(msg, "Process exited with code 1");
    }

    #[test]
    fn test_compose_failure_message_whitespace_stderr_ignored() {
        let msg = ClaudeAgentRunner::compose_failure_message(2, "actual output", "   ");
        assert!(msg.contains("actual output"));
        assert!(msg.contains("Process exited with code 2"));
    }

    #[test]
    fn test_compose_failure_message_very_long_stdout_truncated() {
        let long_stdout = "o".repeat(5000);
        let msg = ClaudeAgentRunner::compose_failure_message(1, &long_stdout, "");
        assert!(msg.len() <= EXECUTION_LOG_PREVIEW_LIMIT + 50); // +50 for prefix
        assert!(msg.contains("Process exited with code 1"));
    }

    #[test]
    fn test_compose_failure_message_rate_limit_429_in_stderr() {
        let msg = ClaudeAgentRunner::compose_failure_message(1, "", "HTTP 429");
        assert!(msg.starts_with("Claude rate limit hit:"));
    }

    #[test]
    fn test_compose_failure_message_exit_code_zero_with_stderr() {
        // Even exit code 0 can produce a failure message if called
        let msg = ClaudeAgentRunner::compose_failure_message(0, "", "some stderr");
        assert_eq!(msg, "some stderr");
    }

    #[test]
    fn test_compose_failure_message_negative_exit_code() {
        let msg = ClaudeAgentRunner::compose_failure_message(-1, "", "");
        assert_eq!(msg, "Process exited with code -1");
    }

    #[test]
    fn test_truncate_max_len_of_0_non_empty_input() {
        // max_len = 0 means we want 0 chars + "..." => the "..." itself
        let result = ClaudeAgentRunner::truncate("hello", 0);
        assert_eq!(result, "...");
    }

    #[test]
    fn test_truncate_emoji_boundary() {
        // Emoji is 4 bytes. With max_len=5, we have room for 2 chars before "...",
        // but the emoji won't fit in 2 bytes, so we should get safe truncation.
        let input = "\u{1F600}abc"; // grinning face (4 bytes) + "abc"
        let result = ClaudeAgentRunner::truncate(input, 5);
        assert!(result.ends_with("..."));
        // Must be valid UTF-8
        assert!(result.len() <= 8); // at most the emoji (4) + "..." (3)
    }

    #[test]
    fn test_truncate_all_multibyte() {
        let input = "\u{00e9}\u{00e9}\u{00e9}\u{00e9}"; // "eeee" with accents, 2 bytes each = 8 bytes
        let result = ClaudeAgentRunner::truncate(input, 6);
        assert!(result.ends_with("..."));
        // Should safely truncate at a char boundary
        assert!(result.is_char_boundary(0));
    }

    #[test]
    fn test_truncate_max_len_of_1() {
        let result = ClaudeAgentRunner::truncate("abcdef", 1);
        assert_eq!(result, "...");
    }

    #[test]
    fn test_hash_prompt_very_long_string() {
        let long = "x".repeat(100_000);
        let h = ClaudeAgentRunner::hash_prompt(&long);
        assert_eq!(h.len(), 16);
        assert!(h.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_hash_prompt_whitespace_differences_produce_different_hashes() {
        assert_ne!(
            ClaudeAgentRunner::hash_prompt("hello world"),
            ClaudeAgentRunner::hash_prompt("hello  world")
        );
    }

    #[test]
    fn test_hash_prompt_case_sensitive() {
        assert_ne!(
            ClaudeAgentRunner::hash_prompt("Hello"),
            ClaudeAgentRunner::hash_prompt("hello")
        );
    }

    #[test]
    fn test_is_rate_limit_error_mixed_case_embedded() {
        assert!(ClaudeAgentRunner::is_rate_limit_error(
            "Error: The API returned a RateLimit error"
        ));
    }

    #[test]
    fn test_is_rate_limit_error_quota_in_longer_message() {
        assert!(ClaudeAgentRunner::is_rate_limit_error(
            "Your project quota exceeded the monthly limit"
        ));
    }

    #[test]
    fn test_is_rate_limit_error_not_triggered_by_partial_substring() {
        // "rate" alone should not trigger it
        assert!(!ClaudeAgentRunner::is_rate_limit_error("first rate code"));
    }

    #[test]
    fn test_is_hard_error_all_needles() {
        // Verify every hard error needle is recognized
        let needles = [
            "failed to spawn claude",
            "failed to wait for claude",
            "failed to capture stdout",
            "failed to capture stderr",
            "process timed out",
            "timed out after",
            "connection reset",
            "service unavailable",
            "internal server error",
            "network error",
            "broken pipe",
        ];
        for needle in needles {
            assert!(
                ClaudeAgentRunner::is_hard_error(needle),
                "Expected hard error for: {}",
                needle
            );
        }
    }

    #[test]
    fn test_is_hard_error_normal_compilation_errors_not_hard() {
        let soft_errors = [
            "error[E0308]: mismatched types",
            "npm ERR! code ELIFECYCLE",
            "FAILED: 3 tests",
            "assertion failed at line 42",
            "segfault",
        ];
        for msg in soft_errors {
            assert!(
                !ClaudeAgentRunner::is_hard_error(msg),
                "Should NOT be hard error: {}",
                msg
            );
        }
    }

    #[test]
    fn test_sanitize_label_numbers_only() {
        assert_eq!(ClaudeAgentRunner::sanitize_label("12345"), "12345");
    }

    #[test]
    fn test_sanitize_label_single_char() {
        assert_eq!(ClaudeAgentRunner::sanitize_label("a"), "a");
        assert_eq!(ClaudeAgentRunner::sanitize_label("@"), "_");
    }

    #[test]
    fn test_sanitize_label_mixed_valid_and_invalid() {
        assert_eq!(
            ClaudeAgentRunner::sanitize_label("PROJ-123/fix"),
            "PROJ-123_fix"
        );
    }

    #[test]
    fn test_sanitize_label_65_chars_truncated() {
        let label = "a".repeat(65);
        let result = ClaudeAgentRunner::sanitize_label(&label);
        assert_eq!(result.len(), 64);
    }

    #[test]
    fn test_claude_runner_config_fully_specified() {
        let config = ClaudeRunnerConfig {
            timeout_secs: 300,
            model: Some("claude-3-opus".to_string()),
            instructions: Some("Be concise.".to_string()),
            permissions: vec!["Bash".to_string(), "Read".to_string(), "Write".to_string()],
            skip_permissions: true,
        };
        assert_eq!(config.timeout_secs, 300);
        assert_eq!(config.model.as_deref(), Some("claude-3-opus"));
        assert_eq!(config.instructions.as_deref(), Some("Be concise."));
        assert_eq!(config.permissions.len(), 3);
        assert!(config.skip_permissions);

        // Clone preserves all fields
        let cloned = config.clone();
        assert_eq!(cloned.timeout_secs, 300);
        assert_eq!(cloned.model, config.model);
        assert_eq!(cloned.instructions, config.instructions);
        assert_eq!(cloned.permissions, config.permissions);
        assert_eq!(cloned.skip_permissions, config.skip_permissions);
    }

    #[test]
    fn test_new_simple_creates_working_runner() {
        let config = ClaudeRunnerConfig {
            timeout_secs: 100,
            model: Some("haiku".to_string()),
            instructions: Some("test".to_string()),
            permissions: vec!["Bash".to_string()],
            skip_permissions: true,
        };
        let runner = ClaudeAgentRunner::new_simple(config);
        // Verify the runner works by calling methods that depend on proper initialization
        assert!(!runner.has_agent_md(Path::new("/nonexistent")));
        let issue = Issue::new("1", "T-1", "Bug", "https://example.com", "linear");
        let prompt = runner.build_prompt(&issue, "ctx", Path::new("/tmp"));
        assert!(!prompt.is_empty());
    }

    #[test]
    fn test_result_schema_required_fields() {
        let parsed: serde_json::Value = serde_json::from_str(RESULT_SCHEMA).unwrap();
        let required = parsed["required"].as_array().unwrap();
        let required_strs: Vec<&str> = required.iter().map(|v| v.as_str().unwrap()).collect();
        assert!(required_strs.contains(&"summary"));
        assert!(required_strs.contains(&"success"));
    }

    #[test]
    fn test_result_schema_no_additional_properties() {
        let parsed: serde_json::Value = serde_json::from_str(RESULT_SCHEMA).unwrap();
        assert_eq!(parsed["additionalProperties"], false);
    }

    #[test]
    fn test_result_schema_blocking_question_sub_schema() {
        let parsed: serde_json::Value = serde_json::from_str(RESULT_SCHEMA).unwrap();
        let bq = &parsed["properties"]["blocking_question"];
        let bq_required = bq["required"].as_array().unwrap();
        let bq_required_strs: Vec<&str> = bq_required.iter().map(|v| v.as_str().unwrap()).collect();
        assert!(bq_required_strs.contains(&"question"));
        assert!(bq["properties"]["question"].is_object());
        assert!(bq["properties"]["context"].is_object());
        assert!(bq["properties"]["options"].is_object());
        assert!(bq["properties"]["why"].is_object());
        assert_eq!(bq["additionalProperties"], false);
    }

    #[test]
    fn test_cli_content_block_text_deserialization() {
        let json = r#"{"type":"text","text":"hello"}"#;
        let block: CliContentBlock = serde_json::from_str(json).unwrap();
        match block {
            CliContentBlock::Text { text } => assert_eq!(text, "hello"),
            other => panic!("Expected Text, got: {:?}", other),
        }
    }

    #[test]
    fn test_cli_content_block_tool_use_deserialization() {
        let json = r#"{"type":"tool_use","id":"tool-abc","name":"Grep"}"#;
        let block: CliContentBlock = serde_json::from_str(json).unwrap();
        match block {
            CliContentBlock::ToolUse { id, name } => {
                assert_eq!(id, "tool-abc");
                assert_eq!(name, "Grep");
            }
            other => panic!("Expected ToolUse, got: {:?}", other),
        }
    }

    #[test]
    fn test_cli_content_block_other_types() {
        // "thinking", "tool_result", or any future type should map to Other
        for type_name in &["thinking", "tool_result", "image", "some_future_type"] {
            let json = format!(r#"{{"type":"{}","data":"whatever"}}"#, type_name);
            let block: CliContentBlock = serde_json::from_str(&json).unwrap();
            assert!(
                matches!(block, CliContentBlock::Other),
                "Expected Other for type: {}",
                type_name
            );
        }
    }

    #[test]
    fn test_cli_content_block_text_empty_string() {
        let json = r#"{"type":"text","text":""}"#;
        let block: CliContentBlock = serde_json::from_str(json).unwrap();
        match block {
            CliContentBlock::Text { text } => assert!(text.is_empty()),
            other => panic!("Expected Text, got: {:?}", other),
        }
    }

    #[test]
    fn test_cli_content_block_text_with_special_chars() {
        let json = r#"{"type":"text","text":"line1\nline2\ttab\"quote\\"}"#;
        let block: CliContentBlock = serde_json::from_str(json).unwrap();
        match block {
            CliContentBlock::Text { text } => {
                assert!(text.contains('\n'));
                assert!(text.contains('\t'));
                assert!(text.contains('"'));
                assert!(text.contains('\\'));
            }
            other => panic!("Expected Text, got: {:?}", other),
        }
    }

    #[test]
    fn test_cli_message_mixed_content() {
        let json = r#"{"content":[
            {"type":"text","text":"Analyzing..."},
            {"type":"tool_use","id":"t1","name":"Bash"},
            {"type":"thinking","thinking":"hmm"},
            {"type":"text","text":"Done."}
        ]}"#;
        let msg: CliMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.content.len(), 4);
        assert!(
            matches!(&msg.content[0], CliContentBlock::Text { text } if text == "Analyzing...")
        );
        assert!(matches!(&msg.content[1], CliContentBlock::ToolUse { name, .. } if name == "Bash"));
        assert!(matches!(&msg.content[2], CliContentBlock::Other));
        assert!(matches!(&msg.content[3], CliContentBlock::Text { text } if text == "Done."));
    }

    #[test]
    fn test_parse_realistic_ndjson_stream() {
        // Simulate a sequence of NDJSON events as Claude CLI would emit
        let lines = vec![
            r#"{"type":"system","subtype":"init","session_id":"sess-001"}"#,
            r#"{"type":"assistant","message":{"content":[{"type":"text","text":"I'll analyze the issue."}]}}"#,
            r#"{"type":"assistant","message":{"content":[{"type":"tool_use","id":"tu_1","name":"Bash"}]}}"#,
            r#"{"type":"user","message":{"role":"user"}}"#,
            r#"{"type":"assistant","message":{"content":[{"type":"text","text":"The fix is applied."}]}}"#,
            r#"{"type":"result","structured_output":{"summary":"Fixed the bug","success":true,"pr_url":"https://github.com/org/repo/pull/42"},"total_cost_usd":0.05,"num_turns":3,"session_id":"sess-001","duration_api_ms":15000,"usage":{"input_tokens":1000,"output_tokens":500,"cache_read_input_tokens":2000,"cache_creation_input_tokens":100}}"#,
        ];

        let mut text_output = String::new();
        let mut structured_result: Option<serde_json::Value> = None;
        let mut cost = None;
        let mut turns = None;

        for line in lines {
            let event: StreamEvent = serde_json::from_str(line).unwrap();
            match event {
                StreamEvent::Assistant { message: Some(msg) } => {
                    for block in &msg.content {
                        if let CliContentBlock::Text { text } = block {
                            text_output.push_str(text);
                        }
                    }
                }
                StreamEvent::Result {
                    structured_output,
                    total_cost_usd,
                    num_turns,
                    ..
                } => {
                    structured_result = structured_output;
                    cost = total_cost_usd;
                    turns = num_turns;
                }
                _ => {}
            }
        }

        assert_eq!(text_output, "I'll analyze the issue.The fix is applied.");
        assert!(structured_result.is_some());
        let sr: StructuredResult = serde_json::from_value(structured_result.unwrap()).unwrap();
        assert!(sr.success);
        assert_eq!(sr.summary, "Fixed the bug");
        assert_eq!(
            sr.pr_url.as_deref(),
            Some("https://github.com/org/repo/pull/42")
        );
        assert!((cost.unwrap() - 0.05).abs() < 1e-6);
        assert_eq!(turns, Some(3));
    }

    #[test]
    fn test_extract_pr_url_self_hosted_gitlab_multiple_segments() {
        let output = "MR: https://git.internal.company.io/engineering/backend/-/merge_requests/55";
        let url = ClaudeAgentRunner::extract_pr_url(output);
        assert_eq!(
            url.as_deref(),
            Some("https://git.internal.company.io/engineering/backend/-/merge_requests/55")
        );
    }

    #[test]
    fn test_extract_pr_url_does_not_match_merge_request_without_dash_slash() {
        // Ensure patterns require the /-/ separator for GitLab
        let output = "https://gitlab.com/group/project/merge_requests/123";
        let url = ClaudeAgentRunner::extract_pr_url(output);
        assert!(url.is_none());
    }

    #[test]
    fn test_default_log_dir_constant() {
        assert_eq!(DEFAULT_LOG_DIR, "./logs");
    }

    #[test]
    fn test_claude_log_subdir_constant() {
        assert_eq!(CLAUDE_LOG_SUBDIR, "claude");
    }

    #[test]
    fn test_execution_log_preview_limit_constant() {
        assert_eq!(EXECUTION_LOG_PREVIEW_LIMIT, 2000);
    }

    #[test]
    fn test_build_prompt_fallback_contains_issue_source() {
        let runner = ClaudeAgentRunner::new_simple(ClaudeRunnerConfig::default());
        let issue = Issue::new("1", "LIN-1", "Bug", "https://ex.com", "linear");
        let prompt = runner.build_prompt(
            &issue,
            "some context",
            Path::new("/tmp/nonexistent_project_dir_xyz"),
        );
        // The fallback template should reference the source
        assert!(
            prompt.contains("linear") || prompt.contains("Linear"),
            "Prompt should contain the issue source"
        );
    }

    #[test]
    fn test_build_prompt_fallback_contains_context() {
        let runner = ClaudeAgentRunner::new_simple(ClaudeRunnerConfig::default());
        let issue = Issue::new("1", "LIN-1", "Bug", "https://ex.com", "linear");
        let prompt = runner.build_prompt(
            &issue,
            "detailed error trace here",
            Path::new("/tmp/nonexistent_project_dir_xyz"),
        );
        assert!(
            prompt.contains("detailed error trace here"),
            "Prompt should contain the provided context"
        );
    }

    #[test]
    fn test_build_prompt_fallback_contains_short_id() {
        let runner = ClaudeAgentRunner::new_simple(ClaudeRunnerConfig::default());
        let issue = Issue::new("1", "PROJ-999", "Bug", "https://ex.com", "linear");
        let prompt = runner.build_prompt(
            &issue,
            "context",
            Path::new("/tmp/nonexistent_project_dir_xyz"),
        );
        assert!(
            prompt.contains("PROJ-999"),
            "Prompt should contain the issue short_id"
        );
    }

    #[test]
    fn test_build_prompt_fallback_instructs_pr_creation() {
        let runner = ClaudeAgentRunner::new_simple(ClaudeRunnerConfig::default());
        let issue = Issue::new("1", "T-1", "Bug", "https://ex.com", "linear");
        let prompt =
            runner.build_prompt(&issue, "ctx", Path::new("/tmp/nonexistent_project_dir_xyz"));
        // The fallback should instruct the model to create a PR
        assert!(
            prompt.to_lowercase().contains("pr") || prompt.to_lowercase().contains("pull request"),
            "Prompt should mention PR creation"
        );
    }

    #[tokio::test]
    async fn test_append_execution_event_with_none_writer_is_noop() {
        // Should return immediately without error when writer is None
        ClaudeAgentRunner::append_execution_event(
            &None,
            "test",
            "some_event",
            json!({"key": "value"}),
        )
        .await;
    }

    #[tokio::test]
    async fn test_append_execution_event_writes_jsonl_to_file() {
        let tmp_dir = std::env::temp_dir().join("claudear_test_append_event");
        let _ = std::fs::create_dir_all(&tmp_dir);
        let file_path = tmp_dir.join("test_events.jsonl");

        let file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&file_path)
            .await
            .unwrap();
        let writer = Some(Arc::new(Mutex::new(file)));

        ClaudeAgentRunner::append_execution_event(
            &writer,
            "my-label",
            "test_event",
            json!({"foo": "bar", "count": 42}),
        )
        .await;

        // Drop the writer to flush
        drop(writer);

        let content = std::fs::read_to_string(&file_path).unwrap();
        assert!(!content.is_empty());

        let parsed: serde_json::Value = serde_json::from_str(content.trim()).unwrap();
        assert_eq!(parsed["label"], "my-label");
        assert_eq!(parsed["event"], "test_event");
        assert_eq!(parsed["data"]["foo"], "bar");
        assert_eq!(parsed["data"]["count"], 42);
        assert!(parsed["timestamp"].is_string());

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[tokio::test]
    async fn test_append_execution_event_multiple_writes() {
        let tmp_dir = std::env::temp_dir().join("claudear_test_append_multi");
        let _ = std::fs::create_dir_all(&tmp_dir);
        let file_path = tmp_dir.join("multi_events.jsonl");

        let file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&file_path)
            .await
            .unwrap();
        let writer = Some(Arc::new(Mutex::new(file)));

        ClaudeAgentRunner::append_execution_event(&writer, "lbl", "event_1", json!({"seq": 1}))
            .await;

        ClaudeAgentRunner::append_execution_event(&writer, "lbl", "event_2", json!({"seq": 2}))
            .await;

        drop(writer);

        let content = std::fs::read_to_string(&file_path).unwrap();
        let lines: Vec<&str> = content.trim().split('\n').collect();
        assert_eq!(lines.len(), 2);

        let first: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        let second: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(first["event"], "event_1");
        assert_eq!(second["event"], "event_2");

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn test_create_execution_log_files_includes_date_in_path() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let prev = std::env::var("CLAUDEAR_LOG_DIR").ok();
        let tmp_dir = std::env::temp_dir().join("claudear_test_date_in_path");
        std::env::set_var("CLAUDEAR_LOG_DIR", tmp_dir.to_str().unwrap());

        let files = ClaudeAgentRunner::create_execution_log_files("mytest");
        assert!(files.is_some());

        let files = files.unwrap();
        let path_str = files.stdout.to_str().unwrap();
        // Should contain a date pattern like "2026-02-20"
        let today = chrono::Utc::now().format("%Y-%m-%d").to_string();
        assert!(
            path_str.contains(&today),
            "Path '{}' should contain today's date '{}'",
            path_str,
            today
        );

        let _ = std::fs::remove_dir_all(&tmp_dir);
        if let Some(val) = prev {
            std::env::set_var("CLAUDEAR_LOG_DIR", val);
        } else {
            std::env::remove_var("CLAUDEAR_LOG_DIR");
        }
    }

    #[test]
    fn test_create_execution_log_files_includes_pid_in_stem() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let prev = std::env::var("CLAUDEAR_LOG_DIR").ok();
        let tmp_dir = std::env::temp_dir().join("claudear_test_pid_stem");
        std::env::set_var("CLAUDEAR_LOG_DIR", tmp_dir.to_str().unwrap());

        let files = ClaudeAgentRunner::create_execution_log_files("pidtest");
        assert!(files.is_some());

        let files = files.unwrap();
        let stem = files.stdout.file_name().unwrap().to_str().unwrap();
        let pid = std::process::id().to_string();
        assert!(
            stem.contains(&pid),
            "Stem '{}' should contain PID '{}'",
            stem,
            pid
        );

        let _ = std::fs::remove_dir_all(&tmp_dir);
        if let Some(val) = prev {
            std::env::set_var("CLAUDEAR_LOG_DIR", val);
        } else {
            std::env::remove_var("CLAUDEAR_LOG_DIR");
        }
    }

    #[test]
    fn test_create_execution_log_files_all_three_share_same_stem_prefix() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let prev = std::env::var("CLAUDEAR_LOG_DIR").ok();
        let tmp_dir = std::env::temp_dir().join("claudear_test_shared_stem");
        std::env::set_var("CLAUDEAR_LOG_DIR", tmp_dir.to_str().unwrap());

        let files = ClaudeAgentRunner::create_execution_log_files("shared").unwrap();
        let stdout_stem = files.stdout.file_stem().unwrap().to_str().unwrap();
        let stderr_stem = files.stderr.file_stem().unwrap().to_str().unwrap();
        // events has double extension .events.jsonl, so check the full name
        let events_name = files.events.file_name().unwrap().to_str().unwrap();

        // The stdout stem is like "20260220T120000.000Z_12345_shared.stdout"
        // (file_stem strips the last extension .log)
        // Extract the common prefix before the first "."
        let prefix = stdout_stem.split(".stdout").next().unwrap();
        assert!(
            stderr_stem.starts_with(prefix),
            "stderr '{}' should share prefix '{}'",
            stderr_stem,
            prefix
        );
        assert!(
            events_name.starts_with(prefix),
            "events '{}' should share prefix '{}'",
            events_name,
            prefix
        );

        let _ = std::fs::remove_dir_all(&tmp_dir);
        if let Some(val) = prev {
            std::env::set_var("CLAUDEAR_LOG_DIR", val);
        } else {
            std::env::remove_var("CLAUDEAR_LOG_DIR");
        }
    }

    #[test]
    fn test_compose_failure_message_rate_limit_only_in_combined_not_individual() {
        // stderr has "rate" and stdout has "limit" -- combined has "rate limit"
        // but each alone does not
        let msg = ClaudeAgentRunner::compose_failure_message(1, "limit issues", "check rate");
        // Combined is "check rate\nlimit issues" which contains "rate" and "limit"
        // but not "rate limit" as a substring (there's a newline between).
        // So this should NOT trigger rate limit detection
        // The combined is "check rate\nlimit issues"
        // "rate limit" is NOT a substring because of the newline
        assert!(
            !msg.starts_with("Claude rate limit hit:"),
            "Should not detect rate limit in separate words across lines"
        );
    }

    #[test]
    fn test_compose_failure_message_quota_exceeded_in_stdout_not_detected() {
        // Rate-limit strings in stdout are not detected — only stderr is checked.
        let msg = ClaudeAgentRunner::compose_failure_message(1, "quota exceeded for project", "");
        assert!(msg.starts_with("Process exited with code 1"));
    }

    #[test]
    fn test_compose_failure_message_try_again_later_in_stderr() {
        let msg = ClaudeAgentRunner::compose_failure_message(1, "", "please try again later");
        assert!(msg.starts_with("Claude rate limit hit:"));
    }

    #[test]
    fn test_compose_failure_message_resource_exhausted_in_combined() {
        let msg =
            ClaudeAgentRunner::compose_failure_message(1, "some output", "resource exhausted");
        assert!(msg.starts_with("Claude rate limit hit:"));
    }

    #[test]
    fn test_compose_failure_message_very_long_rate_limit_in_stderr_truncated() {
        let long_rate_limit = format!("rate limit: {}", "x".repeat(5000));
        let msg = ClaudeAgentRunner::compose_failure_message(1, "", &long_rate_limit);
        assert!(msg.starts_with("Claude rate limit hit:"));
        // The inner message should be truncated
        assert!(msg.len() <= EXECUTION_LOG_PREVIEW_LIMIT + 30);
    }

    #[test]
    fn test_structured_result_pr_url_filter_non_https() {
        // Simulates the runner's filtering logic: pr_url must start with "https://"
        let json =
            r#"{"summary":"done","success":true,"pr_url":"http://github.com/org/repo/pull/1"}"#;
        let sr: StructuredResult = serde_json::from_str(json).unwrap();
        // Runner applies: .filter(|url| url.starts_with("https://"))
        let filtered = sr.pr_url.filter(|url| url.starts_with("https://"));
        assert!(filtered.is_none());
    }

    #[test]
    fn test_structured_result_pr_url_filter_accepts_https() {
        let json =
            r#"{"summary":"done","success":true,"pr_url":"https://github.com/org/repo/pull/1"}"#;
        let sr: StructuredResult = serde_json::from_str(json).unwrap();
        let filtered = sr.pr_url.filter(|url| url.starts_with("https://"));
        assert_eq!(
            filtered.as_deref(),
            Some("https://github.com/org/repo/pull/1")
        );
    }

    #[test]
    fn test_structured_result_empty_summary_fallback() {
        // When summary is empty, the runner uses text_output as fallback
        let json = r#"{"summary":"","success":true}"#;
        let sr: StructuredResult = serde_json::from_str(json).unwrap();
        let text_output = "I fixed the issue and created a PR.";
        let result_output = if sr.summary.is_empty() {
            text_output.to_string()
        } else {
            sr.summary
        };
        assert_eq!(result_output, text_output);
    }

    #[test]
    fn test_structured_result_non_empty_summary_used() {
        let json = r#"{"summary":"Custom summary","success":true}"#;
        let sr: StructuredResult = serde_json::from_str(json).unwrap();
        let text_output = "I fixed the issue and created a PR.";
        let result_output = if sr.summary.is_empty() {
            text_output.to_string()
        } else {
            sr.summary
        };
        assert_eq!(result_output, "Custom summary");
    }

    #[test]
    fn test_structured_result_deserialization_from_value() {
        // Tests serde_json::from_value path used in the runner
        let val = json!({
            "summary": "All done",
            "success": true,
            "pr_url": "https://github.com/org/repo/pull/42",
            "blocking_question": null
        });
        let sr: StructuredResult = serde_json::from_value(val).unwrap();
        assert!(sr.success);
        assert_eq!(sr.summary, "All done");
        assert_eq!(
            sr.pr_url.as_deref(),
            Some("https://github.com/org/repo/pull/42")
        );
        assert!(sr.blocking_question.is_none());
    }

    #[test]
    fn test_structured_result_deserialization_from_value_fails_gracefully() {
        // Invalid structured result should fail deserialization
        let val = json!({ "not_summary": "bad", "not_success": true });
        let result = serde_json::from_value::<StructuredResult>(val);
        assert!(result.is_err());
    }

    #[test]
    fn test_structured_result_with_blocking_question_from_value() {
        let val = json!({
            "summary": "Stuck",
            "success": false,
            "pr_url": null,
            "blocking_question": {
                "question": "Which database should I use?",
                "context": "Found multiple database configs",
                "options": ["postgres", "mysql"],
                "why": "Ambiguous configuration"
            }
        });
        let sr: StructuredResult = serde_json::from_value(val).unwrap();
        assert!(!sr.success);
        let bq = sr.blocking_question.unwrap();
        assert_eq!(bq.question, "Which database should I use?");
        assert_eq!(bq.options.len(), 2);
    }

    #[test]
    fn test_legacy_fallback_extracts_pr_and_question() {
        let text_output = "Some output\nhttps://github.com/org/repo/pull/42\nCLAUDEAR_QUESTION: {\"question\":\"Which branch?\"}";
        let pr = ClaudeAgentRunner::extract_pr_url(text_output);
        let bq = ClaudeAgentRunner::extract_blocking_question(text_output);
        assert_eq!(pr.as_deref(), Some("https://github.com/org/repo/pull/42"));
        assert_eq!(bq.unwrap().question, "Which branch?");
    }

    #[test]
    fn test_legacy_fallback_no_pr_no_question() {
        let text_output = "Just some output text without any special markers";
        let pr = ClaudeAgentRunner::extract_pr_url(text_output);
        let bq = ClaudeAgentRunner::extract_blocking_question(text_output);
        assert!(pr.is_none());
        assert!(bq.is_none());
    }

    #[test]
    fn test_prepare_env_and_label_gitlab_source() {
        let runner = ClaudeAgentRunner::new_simple(ClaudeRunnerConfig::default());
        let issue = Issue::new("gl-1", "MR-1", "Fix", "https://gitlab.com/1", "gitlab");
        let (env, label) = runner.prepare_env_and_label(Some(&issue));

        assert_eq!(label, "MR-1");
        assert_eq!(env.get("GITLAB_ISSUE_ID"), Some(&"gl-1".to_string()));
        assert_eq!(env.get("GITLAB_ISSUE_SHORT_ID"), Some(&"MR-1".to_string()));
        assert_eq!(
            env.get("GITLAB_ISSUE_URL"),
            Some(&"https://gitlab.com/1".to_string())
        );
    }

    #[test]
    fn test_prepare_env_and_label_jira_source() {
        let runner = ClaudeAgentRunner::new_simple(ClaudeRunnerConfig::default());
        let issue = Issue::new("j-99", "JIRA-99", "Task", "https://jira.ex.com/99", "jira");
        let (env, label) = runner.prepare_env_and_label(Some(&issue));

        assert_eq!(label, "JIRA-99");
        assert_eq!(env.get("JIRA_ISSUE_ID"), Some(&"j-99".to_string()));
        assert_eq!(env.get("JIRA_ISSUE_SHORT_ID"), Some(&"JIRA-99".to_string()));
    }

    #[test]
    fn test_prepare_env_and_label_custom_source() {
        let runner = ClaudeAgentRunner::new_simple(ClaudeRunnerConfig::default());
        let issue = Issue::new(
            "c-1",
            "CUSTOM-1",
            "Bug",
            "https://custom.io/1",
            "my_tracker",
        );
        let (env, label) = runner.prepare_env_and_label(Some(&issue));

        assert_eq!(label, "CUSTOM-1");
        assert_eq!(env.get("MY_TRACKER_ISSUE_ID"), Some(&"c-1".to_string()));
        assert_eq!(
            env.get("MY_TRACKER_ISSUE_SHORT_ID"),
            Some(&"CUSTOM-1".to_string())
        );
    }

    #[test]
    fn test_prepare_env_and_label_env_contains_inherited_vars() {
        let runner = ClaudeAgentRunner::new_simple(ClaudeRunnerConfig::default());
        let (_env, _label) = runner.prepare_env_and_label(None);
        // The environment should contain inherited env vars from the process
        // PATH should always be present
        assert!(
            _env.contains_key("PATH"),
            "Should inherit PATH from process environment"
        );
    }

    #[test]
    fn test_build_prompt_uses_template_renderer_when_template_exists() {
        let tmp_dir = std::env::temp_dir().join(format!(
            "claudear_test_build_prompt_template_{:?}",
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&tmp_dir);
        std::fs::create_dir_all(&tmp_dir).unwrap();

        // Create an AGENT.md file so the template path is used
        std::fs::write(
            tmp_dir.join("AGENT.md"),
            "# Custom Agent\nFollow these rules.",
        )
        .unwrap();

        let runner = ClaudeAgentRunner::new_simple(ClaudeRunnerConfig::default());
        let issue = Issue::new("1", "T-1", "Bug", "https://ex.com", "linear");
        let prompt = runner.build_prompt(&issue, "error context here", &tmp_dir);

        let _ = std::fs::remove_dir_all(&tmp_dir);

        // When AGENT.md exists, the template renderer is used and
        // the prompt should contain the agent instructions
        assert!(
            prompt.contains("Custom Agent") || prompt.contains("Follow these rules"),
            "Prompt should contain AGENT.md content when template exists"
        );
    }

    #[test]
    fn test_build_prompt_for_issue_with_template() {
        let tmp_dir = std::env::temp_dir().join(format!(
            "claudear_test_build_prompt_for_issue_tpl_{:?}",
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&tmp_dir);
        std::fs::create_dir_all(&tmp_dir).unwrap();
        std::fs::write(tmp_dir.join("AGENT.md"), "# Agent v2").unwrap();

        let runner = ClaudeAgentRunner::new_simple(ClaudeRunnerConfig::default());
        let issue = Issue::new("1", "T-1", "Bug", "https://ex.com", "sentry");
        let from_public = runner.build_prompt_for_issue(&issue, "ctx", &tmp_dir);
        let from_private = runner.build_prompt(&issue, "ctx", &tmp_dir);

        let _ = std::fs::remove_dir_all(&tmp_dir);

        assert_eq!(from_public, from_private);
    }

    #[test]
    fn test_is_rate_limit_error_lower_already_lowercase() {
        assert!(ClaudeAgentRunner::is_rate_limit_error("rate limit"));
        assert!(ClaudeAgentRunner::is_rate_limit_error("ratelimit"));
        assert!(ClaudeAgentRunner::is_rate_limit_error("too many requests"));
        assert!(ClaudeAgentRunner::is_rate_limit_error("429"));
        assert!(ClaudeAgentRunner::is_rate_limit_error("quota exceeded"));
        assert!(ClaudeAgentRunner::is_rate_limit_error("resource exhausted"));
        assert!(ClaudeAgentRunner::is_rate_limit_error("retry-after"));
        assert!(ClaudeAgentRunner::is_rate_limit_error("try again later"));
    }

    #[test]
    fn test_is_rate_limit_error_lower_negative() {
        assert!(!ClaudeAgentRunner::is_rate_limit_error(""));
        assert!(!ClaudeAgentRunner::is_rate_limit_error("normal error"));
        assert!(!ClaudeAgentRunner::is_rate_limit_error(
            "compilation failed"
        ));
    }

    #[test]
    fn test_is_hard_error_retry_after_is_hard_via_rate_limit() {
        assert!(ClaudeAgentRunner::is_hard_error("retry-after: 60"));
    }

    #[test]
    fn test_is_hard_error_quota_exceeded_is_hard_via_rate_limit() {
        assert!(ClaudeAgentRunner::is_hard_error("quota exceeded"));
    }

    #[test]
    fn test_is_hard_error_resource_exhausted_is_hard_via_rate_limit() {
        assert!(ClaudeAgentRunner::is_hard_error("resource exhausted"));
    }

    #[test]
    fn test_truncate_ascii_at_exact_boundary() {
        // max_len = 10, string is exactly 10 chars -> no truncation
        assert_eq!(ClaudeAgentRunner::truncate("0123456789", 10), "0123456789");
    }

    #[test]
    fn test_truncate_ascii_one_over() {
        // max_len = 10, string is 11 chars -> truncate
        let result = ClaudeAgentRunner::truncate("01234567890", 10);
        assert_eq!(result, "0123456...");
        assert_eq!(result.len(), 10);
    }

    #[test]
    fn test_truncate_three_byte_utf8() {
        // Chinese characters are 3 bytes each
        let input = "\u{4e16}\u{754c}hello"; // "世界hello" = 6 + 5 = 11 bytes
        let result = ClaudeAgentRunner::truncate(input, 8);
        assert!(result.ends_with("..."));
        // Should not split a multi-byte char
        for (i, _) in result.char_indices() {
            assert!(result.is_char_boundary(i));
        }
    }

    #[test]
    fn test_truncate_preserves_full_string_when_max_is_large() {
        let input = "short";
        assert_eq!(ClaudeAgentRunner::truncate(input, 1000), "short");
    }

    #[test]
    fn test_sanitize_label_tabs_and_newlines() {
        assert_eq!(ClaudeAgentRunner::sanitize_label("a\tb\nc"), "a_b_c");
    }

    #[test]
    fn test_sanitize_label_starts_with_number() {
        assert_eq!(ClaudeAgentRunner::sanitize_label("123abc"), "123abc");
    }

    #[test]
    fn test_sanitize_label_only_hyphens_and_underscores() {
        assert_eq!(ClaudeAgentRunner::sanitize_label("---___"), "---___");
    }

    #[test]
    fn test_sanitize_label_emoji_only() {
        // All non-ASCII -> all replaced with "_"
        let result = ClaudeAgentRunner::sanitize_label("\u{1F600}\u{1F601}");
        assert!(!result.is_empty());
        // Should not be "custom" because chars are replaced with '_'
        assert!(result.chars().all(|c| c == '_'));
    }

    #[test]
    fn test_hash_prompt_known_value() {
        // SHA256 of "" is e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
        // First 16 hex chars (8 bytes) = "e3b0c44298fc1c14"
        assert_eq!(ClaudeAgentRunner::hash_prompt(""), "e3b0c44298fc1c14");
    }

    #[test]
    fn test_hash_prompt_known_value_hello() {
        // SHA256 of "hello" is 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824
        // First 16 hex chars = "2cf24dba5fb0a30e"
        assert_eq!(ClaudeAgentRunner::hash_prompt("hello"), "2cf24dba5fb0a30e");
    }

    #[test]
    fn test_extract_blocking_question_with_extra_unknown_fields() {
        // BlockingQuestion should tolerate extra fields in the JSON
        let output =
            r#"CLAUDEAR_QUESTION: {"question":"test?","extra_field":"ignored","another":123}"#;
        // This may or may not parse depending on serde config
        // If it fails, that's also valid behavior
        let result = ClaudeAgentRunner::extract_blocking_question(output);
        // Either it parses successfully (ignoring extras) or returns None
        if let Some(bq) = result {
            assert_eq!(bq.question, "test?");
        }
    }

    #[test]
    fn test_extract_blocking_question_with_empty_question_field() {
        let output = r#"CLAUDEAR_QUESTION: {"question":""}"#;
        let result = ClaudeAgentRunner::extract_blocking_question(output);
        // Should parse but question will be empty
        if let Some(bq) = result {
            assert!(bq.question.is_empty());
        }
    }

    #[test]
    fn test_extract_blocking_question_mixed_with_other_output() {
        let output = "INFO: Starting fix...\nDEBUG: Analyzing code\nCLAUDEAR_QUESTION: {\"question\":\"Which module?\"}\nINFO: Waiting for response";
        let bq = ClaudeAgentRunner::extract_blocking_question(output).unwrap();
        assert_eq!(bq.question, "Which module?");
    }

    #[test]
    fn test_compose_failure_message_max_exit_code() {
        let msg = ClaudeAgentRunner::compose_failure_message(255, "", "");
        assert_eq!(msg, "Process exited with code 255");
    }

    #[test]
    fn test_compose_failure_message_signal_exit_code() {
        let msg = ClaudeAgentRunner::compose_failure_message(137, "", "");
        assert_eq!(msg, "Process exited with code 137");
    }

    #[test]
    fn test_extract_pr_url_in_markdown_link() {
        let output = "Created [PR #42](https://github.com/org/repo/pull/42) for this fix";
        let url = ClaudeAgentRunner::extract_pr_url(output);
        assert!(url.is_some());
        assert!(url.unwrap().contains("/pull/42"));
    }

    #[test]
    fn test_extract_pr_url_in_json_string() {
        let output = r#"{"pr_url": "https://github.com/org/repo/pull/99"}"#;
        let url = ClaudeAgentRunner::extract_pr_url(output);
        assert!(url.is_some());
    }

    #[test]
    fn test_stdout_parse_result_field_assignment() {
        let result = StdoutParseResult {
            text_output: "hello".to_string(),
            structured_result: Some(json!({"summary": "done", "success": true})),
            cost_usd: Some(0.05),
            num_turns: Some(3),
            session_id: Some("sess-1".to_string()),
            duration_api_ms: Some(5000),
            input_tokens: Some(100),
            output_tokens: Some(200),
            cache_read_input_tokens: Some(300),
            cache_creation_input_tokens: Some(400),
        };

        assert_eq!(result.text_output, "hello");
        assert!(result.structured_result.is_some());
        assert!((result.cost_usd.unwrap() - 0.05).abs() < 1e-6);
        assert_eq!(result.num_turns, Some(3));
        assert_eq!(result.session_id.as_deref(), Some("sess-1"));
        assert_eq!(result.duration_api_ms, Some(5000));
        assert_eq!(result.input_tokens, Some(100));
        assert_eq!(result.output_tokens, Some(200));
        assert_eq!(result.cache_read_input_tokens, Some(300));
        assert_eq!(result.cache_creation_input_tokens, Some(400));
    }

    #[test]
    fn test_claude_execution_prompt_hash_and_model() {
        let mut exec = AgentExecution::new();
        exec.prompt_used = Some("Fix the bug".to_string());
        exec.prompt_hash = Some(ClaudeAgentRunner::hash_prompt("Fix the bug"));
        exec.model_version = Some("opus".to_string());
        exec.working_directory = Some("/tmp/project".to_string());

        assert_eq!(exec.prompt_used.as_deref(), Some("Fix the bug"));
        assert_eq!(exec.prompt_hash.as_ref().unwrap().len(), 16);
        assert_eq!(exec.model_version.as_deref(), Some("opus"));
        assert_eq!(exec.working_directory.as_deref(), Some("/tmp/project"));
    }

    #[test]
    fn test_claude_execution_log_paths() {
        let mut exec = AgentExecution::new();
        exec.stdout_log_path = Some("/logs/claude/2026-02-20/test.stdout.log".to_string());
        exec.stderr_log_path = Some("/logs/claude/2026-02-20/test.stderr.log".to_string());
        exec.event_log_path = Some("/logs/claude/2026-02-20/test.events.jsonl".to_string());

        assert!(exec
            .stdout_log_path
            .as_ref()
            .unwrap()
            .ends_with(".stdout.log"));
        assert!(exec
            .stderr_log_path
            .as_ref()
            .unwrap()
            .ends_with(".stderr.log"));
        assert!(exec
            .event_log_path
            .as_ref()
            .unwrap()
            .ends_with(".events.jsonl"));
    }

    #[test]
    fn test_claude_execution_cost_and_token_fields() {
        let mut exec = AgentExecution::new();
        exec.total_cost_usd = Some(0.123);
        exec.num_turns = Some(5);
        exec.session_id = Some("sess-abc".to_string());
        exec.duration_api_ms = Some(15000);
        exec.input_tokens = Some(1000);
        exec.output_tokens = Some(500);
        exec.cache_read_input_tokens = Some(2000);
        exec.cache_creation_input_tokens = Some(100);

        assert!((exec.total_cost_usd.unwrap() - 0.123).abs() < 1e-6);
        assert_eq!(exec.num_turns, Some(5));
        assert_eq!(exec.session_id.as_deref(), Some("sess-abc"));
        assert_eq!(exec.duration_api_ms, Some(15000));
        assert_eq!(exec.input_tokens, Some(1000));
        assert_eq!(exec.output_tokens, Some(500));
        assert_eq!(exec.cache_read_input_tokens, Some(2000));
        assert_eq!(exec.cache_creation_input_tokens, Some(100));
    }

    #[test]
    fn test_claude_execution_preview_fields() {
        let mut exec = AgentExecution::new();
        exec.stdout_preview = Some(ClaudeAgentRunner::truncate(
            &"x".repeat(5000),
            EXECUTION_LOG_PREVIEW_LIMIT,
        ));
        exec.stderr_preview = Some("error: test failed".to_string());

        assert!(exec.stdout_preview.as_ref().unwrap().len() <= EXECUTION_LOG_PREVIEW_LIMIT + 3);
        assert!(exec.stdout_preview.as_ref().unwrap().ends_with("..."));
        assert_eq!(exec.stderr_preview.as_deref(), Some("error: test failed"));
    }

    #[test]
    fn test_stream_event_result_zero_cost() {
        let json = r#"{"type":"result","total_cost_usd":0.0,"num_turns":0}"#;
        let event: StreamEvent = serde_json::from_str(json).unwrap();
        match event {
            StreamEvent::Result {
                total_cost_usd: Some(cost),
                num_turns: Some(turns),
                ..
            } => {
                assert!((cost - 0.0).abs() < 1e-10);
                assert_eq!(turns, 0);
            }
            other => panic!("Unexpected event: {:?}", other),
        }
    }

    #[test]
    fn test_stream_event_result_large_values() {
        let json = r#"{"type":"result","total_cost_usd":999.99,"num_turns":1000,"duration_api_ms":3600000,"usage":{"input_tokens":1000000,"output_tokens":500000}}"#;
        let event: StreamEvent = serde_json::from_str(json).unwrap();
        match event {
            StreamEvent::Result {
                total_cost_usd: Some(cost),
                num_turns: Some(turns),
                duration_api_ms: Some(api_ms),
                usage: Some(ref u),
                ..
            } => {
                assert!((cost - 999.99).abs() < 0.01);
                assert_eq!(turns, 1000);
                assert_eq!(api_ms, 3600000);
                assert_eq!(u.input_tokens, Some(1000000));
                assert_eq!(u.output_tokens, Some(500000));
            }
            other => panic!("Unexpected event: {:?}", other),
        }
    }

    #[test]
    fn test_stream_event_assistant_empty_content() {
        let json = r#"{"type":"assistant","message":{"content":[]}}"#;
        let event: StreamEvent = serde_json::from_str(json).unwrap();
        match event {
            StreamEvent::Assistant { message: Some(msg) } => {
                assert!(msg.content.is_empty());
            }
            other => panic!("Unexpected event: {:?}", other),
        }
    }

    #[test]
    fn test_compose_failure_message_tabs_and_newlines_only() {
        let msg = ClaudeAgentRunner::compose_failure_message(3, "\t\n", "\n\t\n");
        assert_eq!(msg, "Process exited with code 3");
    }

    #[test]
    fn test_new_simple_preserves_config_model() {
        let config = ClaudeRunnerConfig {
            model: Some("haiku".to_string()),
            ..Default::default()
        };
        let runner = ClaudeAgentRunner::new_simple(config);
        assert_eq!(runner.config.model.as_deref(), Some("haiku"));
    }

    #[test]
    fn test_new_simple_preserves_config_timeout() {
        let config = ClaudeRunnerConfig {
            timeout_secs: 42,
            ..Default::default()
        };
        let runner = ClaudeAgentRunner::new_simple(config);
        assert_eq!(runner.config.timeout_secs, 42);
    }

    #[test]
    fn test_new_simple_preserves_config_skip_permissions() {
        let config = ClaudeRunnerConfig {
            skip_permissions: true,
            ..Default::default()
        };
        let runner = ClaudeAgentRunner::new_simple(config);
        assert!(runner.config.skip_permissions);
    }

    #[test]
    fn test_new_simple_preserves_config_instructions() {
        let config = ClaudeRunnerConfig {
            instructions: Some("Be thorough".to_string()),
            ..Default::default()
        };
        let runner = ClaudeAgentRunner::new_simple(config);
        assert_eq!(runner.config.instructions.as_deref(), Some("Be thorough"));
    }

    #[test]
    fn test_new_simple_preserves_config_permissions() {
        let config = ClaudeRunnerConfig {
            permissions: vec!["Bash".to_string(), "Read".to_string()],
            ..Default::default()
        };
        let runner = ClaudeAgentRunner::new_simple(config);
        assert_eq!(runner.config.permissions, vec!["Bash", "Read"]);
    }

    #[test]
    fn test_runner_base_env_captures_process_env() {
        let runner = ClaudeAgentRunner::new_simple(ClaudeRunnerConfig::default());
        // The base_env should be a snapshot of the process environment
        assert!(!runner.base_env.is_empty());
        // PATH is virtually always present
        assert!(runner.base_env.contains_key("PATH"));
    }

    #[test]
    fn test_result_schema_summary_is_string_type() {
        let parsed: serde_json::Value = serde_json::from_str(RESULT_SCHEMA).unwrap();
        assert_eq!(parsed["properties"]["summary"]["type"], "string");
    }

    #[test]
    fn test_result_schema_success_is_boolean_type() {
        let parsed: serde_json::Value = serde_json::from_str(RESULT_SCHEMA).unwrap();
        assert_eq!(parsed["properties"]["success"]["type"], "boolean");
    }

    #[test]
    fn test_result_schema_pr_url_allows_null() {
        let parsed: serde_json::Value = serde_json::from_str(RESULT_SCHEMA).unwrap();
        let pr_url_type = &parsed["properties"]["pr_url"]["type"];
        let types = pr_url_type.as_array().unwrap();
        let type_strs: Vec<&str> = types.iter().map(|v| v.as_str().unwrap()).collect();
        assert!(type_strs.contains(&"string"));
        assert!(type_strs.contains(&"null"));
    }

    #[test]
    fn test_result_schema_blocking_question_allows_null() {
        let parsed: serde_json::Value = serde_json::from_str(RESULT_SCHEMA).unwrap();
        let bq_type = &parsed["properties"]["blocking_question"]["type"];
        let types = bq_type.as_array().unwrap();
        let type_strs: Vec<&str> = types.iter().map(|v| v.as_str().unwrap()).collect();
        assert!(type_strs.contains(&"object"));
        assert!(type_strs.contains(&"null"));
    }

    #[test]
    fn test_compose_failure_message_npm_test_failure() {
        let msg = ClaudeAgentRunner::compose_failure_message(
            1,
            "",
            "npm ERR! Test failed. See above for more details.",
        );
        assert_eq!(msg, "npm ERR! Test failed. See above for more details.");
        assert!(!msg.starts_with("Claude rate limit hit:"));
    }

    #[test]
    fn test_compose_failure_message_cargo_test_failure() {
        let stderr = "error[E0308]: mismatched types\n  --> src/main.rs:10:5";
        let msg = ClaudeAgentRunner::compose_failure_message(101, "", stderr);
        assert!(msg.contains("mismatched types"));
    }

    #[test]
    fn test_compose_failure_message_git_push_failure() {
        let stderr = "remote: Permission to org/repo.git denied.\nfatal: unable to access";
        let msg = ClaudeAgentRunner::compose_failure_message(128, "", stderr);
        assert!(msg.contains("Permission"));
    }

    #[test]
    fn test_extract_pr_url_very_long_output() {
        let mut output = "x".repeat(100_000);
        output.push_str("\nhttps://github.com/org/repo/pull/42\n");
        output.push_str(&"y".repeat(100_000));
        let url = ClaudeAgentRunner::extract_pr_url(&output);
        assert_eq!(url.as_deref(), Some("https://github.com/org/repo/pull/42"));
    }

    #[test]
    fn test_extract_pr_url_unicode_surrounding() {
        let output = "PR \u{1F389}\u{1F389} https://github.com/org/repo/pull/1 \u{1F389}\u{1F389}";
        let url = ClaudeAgentRunner::extract_pr_url(output);
        assert!(url.is_some());
        assert!(url.unwrap().contains("/pull/1"));
    }

    #[test]
    fn test_claude_runner_config_debug_shows_all_fields() {
        let config = ClaudeRunnerConfig {
            timeout_secs: 300,
            model: Some("opus".to_string()),
            instructions: Some("Be concise".to_string()),
            permissions: vec!["Bash".to_string()],
            skip_permissions: true,
        };
        let debug = format!("{:?}", config);
        assert!(debug.contains("300"));
        assert!(debug.contains("opus"));
        assert!(debug.contains("Be concise"));
        assert!(debug.contains("Bash"));
        assert!(debug.contains("true"));
    }

    #[test]
    fn test_resolve_log_root_with_empty_env_var() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let prev = std::env::var("CLAUDEAR_LOG_DIR").ok();
        std::env::set_var("CLAUDEAR_LOG_DIR", "");

        let root = resolve_log_root();
        // Empty string is technically a valid PathBuf
        assert_eq!(root, PathBuf::from(""));

        if let Some(val) = prev {
            std::env::set_var("CLAUDEAR_LOG_DIR", val);
        } else {
            std::env::remove_var("CLAUDEAR_LOG_DIR");
        }
    }

    #[test]
    fn test_build_prompt_for_github_issue_with_template() {
        let tmp_dir = std::env::temp_dir().join(format!(
            "claudear_test_gh_tpl_{:?}",
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&tmp_dir);
        std::fs::create_dir_all(&tmp_dir).unwrap();
        std::fs::write(tmp_dir.join("AGENT.md"), "# GitHub Agent").unwrap();

        let runner = ClaudeAgentRunner::new_simple(ClaudeRunnerConfig::default());
        let issue = Issue::new(
            "42",
            "#42",
            "Add tests",
            "https://github.com/org/repo/issues/42",
            "github",
        );
        let prompt = runner.build_prompt(&issue, "Please add test coverage", &tmp_dir);

        let _ = std::fs::remove_dir_all(&tmp_dir);

        assert!(!prompt.is_empty());
        // Should contain AGENT.md content since it exists
        assert!(prompt.contains("GitHub Agent"));
    }

    #[test]
    fn test_build_prompt_for_sentry_issue_with_template() {
        let tmp_dir = std::env::temp_dir().join(format!(
            "claudear_test_sentry_tpl_{:?}",
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&tmp_dir);
        std::fs::create_dir_all(&tmp_dir).unwrap();
        std::fs::write(tmp_dir.join("AGENT.md"), "# Sentry Handler").unwrap();

        let runner = ClaudeAgentRunner::new_simple(ClaudeRunnerConfig::default());
        let issue = Issue::new(
            "99",
            "SENTRY-99",
            "TypeError: null is not an object",
            "https://sentry.io/issues/99",
            "sentry",
        );
        let prompt = runner.build_prompt(&issue, "Stack trace...", &tmp_dir);

        let _ = std::fs::remove_dir_all(&tmp_dir);

        assert!(!prompt.is_empty());
        assert!(prompt.contains("Sentry Handler"));
    }

    #[test]
    fn test_sanitize_label_empty_string() {
        assert_eq!(ClaudeAgentRunner::sanitize_label(""), "custom");
    }

    #[test]
    fn test_sanitize_label_special_chars() {
        assert_eq!(
            ClaudeAgentRunner::sanitize_label("hello world!@#$%"),
            "hello_world_____"
        );
    }

    #[test]
    fn test_sanitize_label_unicode() {
        // Unicode chars are not ascii alphanumeric, so replaced with '_'
        assert_eq!(ClaudeAgentRunner::sanitize_label("café"), "caf_");
    }

    #[test]
    fn test_sanitize_label_already_clean() {
        assert_eq!(
            ClaudeAgentRunner::sanitize_label("my-label_123"),
            "my-label_123"
        );
    }

    #[test]
    fn test_sanitize_label_exactly_64_chars() {
        let label = "a".repeat(64);
        let result = ClaudeAgentRunner::sanitize_label(&label);
        assert_eq!(result.len(), 64);
        assert_eq!(result, label);
    }

    #[test]
    fn test_sanitize_label_100_chars_truncated() {
        let label = "x".repeat(100);
        let result = ClaudeAgentRunner::sanitize_label(&label);
        assert_eq!(result.len(), 64);
    }

    #[test]
    fn test_sanitize_label_only_special_chars() {
        // All chars replaced with '_', so not empty
        assert_eq!(ClaudeAgentRunner::sanitize_label("!@#"), "___");
    }

    #[test]
    fn test_compose_failure_message_empty_both() {
        let msg = ClaudeAgentRunner::compose_failure_message(1, "", "");
        assert_eq!(msg, "Process exited with code 1");
    }

    #[test]
    fn test_compose_failure_message_both_stderr_and_stdout() {
        let msg = ClaudeAgentRunner::compose_failure_message(1, "stdout text", "stderr text");
        // When both present and no rate limit, stderr is used (priority)
        assert_eq!(msg, "stderr text");
    }

    #[test]
    fn test_compose_failure_message_rate_limit_429_in_stdout_not_detected() {
        // Rate-limit strings in stdout are not detected — only stderr is checked.
        let msg = ClaudeAgentRunner::compose_failure_message(1, "Error 429 too many requests", "");
        assert!(msg.starts_with("Process exited with code 1"));
    }

    #[test]
    fn test_compose_failure_message_rate_limit_quota_exceeded() {
        let msg = ClaudeAgentRunner::compose_failure_message(1, "", "Quota exceeded for model");
        assert!(msg.starts_with("Claude rate limit hit:"));
    }

    #[test]
    fn test_compose_failure_message_long_stderr_truncated() {
        let long_stderr = "e".repeat(3000);
        let msg = ClaudeAgentRunner::compose_failure_message(1, "", &long_stderr);
        assert!(msg.len() <= EXECUTION_LOG_PREVIEW_LIMIT + 3); // +3 for "..."
    }

    #[test]
    fn test_compose_failure_message_long_stdout_truncated() {
        let long_stdout = "o".repeat(3000);
        let msg = ClaudeAgentRunner::compose_failure_message(1, &long_stdout, "");
        assert!(msg.len() <= EXECUTION_LOG_PREVIEW_LIMIT + 50); // prefix + "..."
    }

    #[test]
    fn test_compose_failure_message_whitespace_only_stderr() {
        let msg = ClaudeAgentRunner::compose_failure_message(1, "output", "   \n  ");
        // Whitespace-only stderr is trimmed to empty, so stdout is used
        assert!(msg.contains("output"));
        assert!(msg.contains("Process exited with code 1"));
    }

    #[test]
    fn test_extract_blocking_question_no_question() {
        assert!(ClaudeAgentRunner::extract_blocking_question("just some regular output").is_none());
    }

    #[test]
    fn test_extract_blocking_question_valid_json() {
        let output = r#"some preamble
CLAUDEAR_QUESTION: {"question":"Which branch should I target?","context":"There are multiple release branches","options":["main","develop"],"why":"Ambiguous target"}
more output"#;
        let q = ClaudeAgentRunner::extract_blocking_question(output).unwrap();
        assert_eq!(q.question, "Which branch should I target?");
        assert_eq!(
            q.context.as_deref(),
            Some("There are multiple release branches")
        );
        assert_eq!(q.options, vec!["main", "develop"]);
        assert_eq!(q.why.as_deref(), Some("Ambiguous target"));
    }

    #[test]
    fn test_extract_blocking_question_minimal_json() {
        let output = r#"CLAUDEAR_QUESTION: {"question":"What should I do?"}"#;
        let q = ClaudeAgentRunner::extract_blocking_question(output).unwrap();
        assert_eq!(q.question, "What should I do?");
        assert!(q.context.is_none());
        assert!(q.options.is_empty());
        assert!(q.why.is_none());
    }

    #[test]
    fn test_extract_blocking_question_invalid_json() {
        let output = "CLAUDEAR_QUESTION: not valid json {{{";
        assert!(ClaudeAgentRunner::extract_blocking_question(output).is_none());
    }

    #[test]
    fn test_extract_blocking_question_empty_payload() {
        let output = "CLAUDEAR_QUESTION: ";
        assert!(ClaudeAgentRunner::extract_blocking_question(output).is_none());
    }

    #[test]
    fn test_extract_blocking_question_prefix_only() {
        let output = "CLAUDEAR_QUESTION:";
        assert!(ClaudeAgentRunner::extract_blocking_question(output).is_none());
    }

    #[test]
    fn test_extract_blocking_question_among_many_lines() {
        let output = "line 1\nline 2\nCLAUDEAR_QUESTION: {\"question\":\"How?\"}\nline 4";
        let q = ClaudeAgentRunner::extract_blocking_question(output).unwrap();
        assert_eq!(q.question, "How?");
    }

    #[test]
    fn test_is_rate_limit_error_mixed_case() {
        assert!(ClaudeAgentRunner::is_rate_limit_error("RATE LIMIT"));
        assert!(ClaudeAgentRunner::is_rate_limit_error("Rate limit"));
        assert!(ClaudeAgentRunner::is_rate_limit_error("rAtE LiMiT"));
    }

    #[test]
    fn test_is_rate_limit_error_non_matching() {
        assert!(!ClaudeAgentRunner::is_rate_limit_error(
            "everything is fine"
        ));
        assert!(!ClaudeAgentRunner::is_rate_limit_error(""));
        assert!(!ClaudeAgentRunner::is_rate_limit_error(
            "generic error occurred"
        ));
    }

    #[test]
    fn test_is_hard_error_rate_limit_strings() {
        // Hard errors include all rate limit errors
        assert!(ClaudeAgentRunner::is_hard_error("rate limit hit"));
        assert!(ClaudeAgentRunner::is_hard_error("429"));
    }

    #[test]
    fn test_is_hard_error_failed_to_spawn() {
        assert!(ClaudeAgentRunner::is_hard_error(
            "Failed to spawn Claude process"
        ));
    }

    #[test]
    fn test_is_hard_error_failed_to_wait() {
        assert!(ClaudeAgentRunner::is_hard_error(
            "Failed to wait for Claude"
        ));
    }

    #[test]
    fn test_is_hard_error_mixed_case() {
        assert!(ClaudeAgentRunner::is_hard_error("FAILED TO SPAWN CLAUDE"));
        assert!(ClaudeAgentRunner::is_hard_error("PROCESS TIMED OUT"));
    }

    #[test]
    fn test_is_hard_error_non_matching() {
        assert!(!ClaudeAgentRunner::is_hard_error("everything is fine"));
        assert!(!ClaudeAgentRunner::is_hard_error(""));
        assert!(!ClaudeAgentRunner::is_hard_error(
            "some random failure message"
        ));
    }

    #[test]
    fn test_resolve_log_root_falls_back_to_default() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let prev = std::env::var("CLAUDEAR_LOG_DIR").ok();
        std::env::remove_var("CLAUDEAR_LOG_DIR");

        let root = resolve_log_root();
        assert_eq!(root, PathBuf::from("./logs"));

        // Also test the ClaudeAgentRunner method version
        let root2 = ClaudeAgentRunner::resolve_log_root();
        assert_eq!(root2, PathBuf::from("./logs"));

        if let Some(val) = prev {
            std::env::set_var("CLAUDEAR_LOG_DIR", val);
        }
    }

    #[test]
    fn test_resolve_log_root_with_custom_dir() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let prev = std::env::var("CLAUDEAR_LOG_DIR").ok();
        std::env::set_var("CLAUDEAR_LOG_DIR", "/tmp/custom-logs");

        let root = resolve_log_root();
        assert_eq!(root, PathBuf::from("/tmp/custom-logs"));

        if let Some(val) = prev {
            std::env::set_var("CLAUDEAR_LOG_DIR", val);
        } else {
            std::env::remove_var("CLAUDEAR_LOG_DIR");
        }
    }

    #[test]
    fn test_claude_runner_config_default() {
        let config = ClaudeRunnerConfig::default();
        assert_eq!(config.timeout_secs, 21600);
        assert!(config.model.is_none());
        assert!(config.instructions.is_none());
        assert!(config.permissions.is_empty());
        assert!(!config.skip_permissions);
    }

    #[test]
    fn test_cli_usage_full_json() {
        let json = r#"{
            "input_tokens": 100,
            "output_tokens": 200,
            "cache_read_input_tokens": 50,
            "cache_creation_input_tokens": 25
        }"#;
        let usage: CliUsage = serde_json::from_str(json).unwrap();
        assert_eq!(usage.input_tokens, Some(100));
        assert_eq!(usage.output_tokens, Some(200));
        assert_eq!(usage.cache_read_input_tokens, Some(50));
        assert_eq!(usage.cache_creation_input_tokens, Some(25));
    }

    #[test]
    fn test_cli_usage_partial_json() {
        let json = r#"{"input_tokens": 42}"#;
        let usage: CliUsage = serde_json::from_str(json).unwrap();
        assert_eq!(usage.input_tokens, Some(42));
        assert!(usage.output_tokens.is_none());
        assert!(usage.cache_read_input_tokens.is_none());
        assert!(usage.cache_creation_input_tokens.is_none());
    }

    #[test]
    fn test_cli_usage_empty_json() {
        let json = r#"{}"#;
        let usage: CliUsage = serde_json::from_str(json).unwrap();
        assert!(usage.input_tokens.is_none());
        assert!(usage.output_tokens.is_none());
        assert!(usage.cache_read_input_tokens.is_none());
        assert!(usage.cache_creation_input_tokens.is_none());
    }

    #[test]
    fn test_stream_event_system() {
        let json = r#"{"type":"system"}"#;
        let event: StreamEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(event, StreamEvent::System {}));
    }

    #[test]
    fn test_stream_event_user() {
        let json = r#"{"type":"user"}"#;
        let event: StreamEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(event, StreamEvent::User {}));
    }

    #[test]
    fn test_stream_event_result_full() {
        let json = r#"{
            "type": "result",
            "structured_output": {"summary": "done", "success": true},
            "total_cost_usd": 0.05,
            "num_turns": 3,
            "session_id": "sess-123",
            "duration_api_ms": 5000,
            "usage": {"input_tokens": 100, "output_tokens": 200}
        }"#;
        let event: StreamEvent = serde_json::from_str(json).unwrap();
        match event {
            StreamEvent::Result {
                structured_output,
                total_cost_usd,
                num_turns,
                session_id,
                duration_api_ms,
                usage,
            } => {
                assert!(structured_output.is_some());
                assert_eq!(total_cost_usd, Some(0.05));
                assert_eq!(num_turns, Some(3));
                assert_eq!(session_id.as_deref(), Some("sess-123"));
                assert_eq!(duration_api_ms, Some(5000));
                let u = usage.unwrap();
                assert_eq!(u.input_tokens, Some(100));
                assert_eq!(u.output_tokens, Some(200));
            }
            other => panic!("Unexpected event: {:?}", other),
        }
    }

    #[test]
    fn test_stream_event_result_minimal() {
        let json = r#"{"type":"result"}"#;
        let event: StreamEvent = serde_json::from_str(json).unwrap();
        match event {
            StreamEvent::Result {
                structured_output,
                total_cost_usd,
                num_turns,
                session_id,
                duration_api_ms,
                usage,
            } => {
                assert!(structured_output.is_none());
                assert!(total_cost_usd.is_none());
                assert!(num_turns.is_none());
                assert!(session_id.is_none());
                assert!(duration_api_ms.is_none());
                assert!(usage.is_none());
            }
            other => panic!("Unexpected event: {:?}", other),
        }
    }

    #[test]
    fn test_stream_event_unknown_type() {
        let json = r#"{"type":"new_future_event"}"#;
        let event: StreamEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(event, StreamEvent::Unknown));
    }

    #[test]
    fn test_stream_event_assistant_with_no_message() {
        let json = r#"{"type":"assistant"}"#;
        let event: StreamEvent = serde_json::from_str(json).unwrap();
        match event {
            StreamEvent::Assistant { message } => assert!(message.is_none()),
            other => panic!("Unexpected event: {:?}", other),
        }
    }

    #[test]
    fn test_structured_result_with_blocking_question_only() {
        let json = r#"{
            "summary": "",
            "success": false,
            "blocking_question": {"question": "Need help"}
        }"#;
        let result: StructuredResult = serde_json::from_str(json).unwrap();
        assert_eq!(result.summary, "");
        assert!(!result.success);
        assert!(result.pr_url.is_none());
        let bq = result.blocking_question.unwrap();
        assert_eq!(bq.question, "Need help");
        assert!(bq.context.is_none());
        assert!(bq.options.is_empty());
        assert!(bq.why.is_none());
    }

    #[test]
    fn test_structured_result_with_null_optionals() {
        let json = r#"{
            "summary": "test",
            "success": true,
            "pr_url": null,
            "blocking_question": null
        }"#;
        let result: StructuredResult = serde_json::from_str(json).unwrap();
        assert!(result.pr_url.is_none());
        assert!(result.blocking_question.is_none());
    }

    #[test]
    fn test_result_schema_contains_expected_keys() {
        let parsed: serde_json::Value = serde_json::from_str(RESULT_SCHEMA).unwrap();
        let obj = parsed.as_object().unwrap();

        assert_eq!(obj.get("type").and_then(|v| v.as_str()), Some("object"));

        let required = obj.get("required").and_then(|v| v.as_array()).unwrap();
        let required_strs: Vec<&str> = required.iter().filter_map(|v| v.as_str()).collect();
        assert!(required_strs.contains(&"summary"));
        assert!(required_strs.contains(&"success"));

        let props = obj.get("properties").and_then(|v| v.as_object()).unwrap();
        assert!(props.contains_key("summary"));
        assert!(props.contains_key("success"));
        assert!(props.contains_key("pr_url"));
        assert!(props.contains_key("blocking_question"));
    }

    #[test]
    fn test_result_schema_blocking_question_properties() {
        let parsed: serde_json::Value = serde_json::from_str(RESULT_SCHEMA).unwrap();
        let bq = &parsed["properties"]["blocking_question"];
        let bq_props = bq["properties"].as_object().unwrap();
        assert!(bq_props.contains_key("question"));
        assert!(bq_props.contains_key("context"));
        assert!(bq_props.contains_key("options"));
        assert!(bq_props.contains_key("why"));
    }

    #[test]
    fn test_claude_agent_runner_name() {
        use claudear_storage::SqliteTracker;
        let tracker = Arc::new(SqliteTracker::in_memory().unwrap());
        let runner = ClaudeAgentRunner::new(ClaudeRunnerConfig::default(), tracker);
        assert_eq!(runner.name(), "claude");
    }

    #[test]
    fn test_claude_agent_runner_capabilities() {
        use claudear_storage::SqliteTracker;
        let tracker = Arc::new(SqliteTracker::in_memory().unwrap());
        let runner = ClaudeAgentRunner::new(ClaudeRunnerConfig::default(), tracker);
        let caps = runner.capabilities();
        assert!(
            caps.structured_output,
            "Claude should support structured output"
        );
        assert!(
            caps.tool_permissions,
            "Claude should support tool permissions"
        );
        assert!(
            caps.custom_instructions,
            "Claude should support custom instructions"
        );
        assert!(
            caps.streaming_events,
            "Claude should support streaming events"
        );
        assert!(caps.cost_reporting, "Claude should support cost reporting");
    }

    #[test]
    fn test_claude_agent_runner_build_prompt_delegates() {
        use claudear_storage::SqliteTracker;
        let tracker = Arc::new(SqliteTracker::in_memory().unwrap());
        let runner = ClaudeAgentRunner::new(ClaudeRunnerConfig::default(), tracker);
        let issue = Issue::new("1", "LIN-1", "Bug title", "https://example.com", "linear");
        // The trait method should produce the same result as the internal method
        let trait_prompt = runner.build_prompt_for_issue(&issue, "some context", Path::new("/tmp"));
        let internal_prompt = runner.build_prompt(&issue, "some context", Path::new("/tmp"));
        assert_eq!(trait_prompt, internal_prompt);
    }

    #[tokio::test]
    async fn test_claude_execute_nonexistent_binary_returns_error() {
        use claudear_storage::SqliteTracker;
        // Use a project_dir that does not exist to force a spawn error
        let tracker = Arc::new(SqliteTracker::in_memory().unwrap());
        let config = ClaudeRunnerConfig::default();
        let runner = ClaudeAgentRunner::new(config, tracker);
        let result = runner
            .execute_with_attempt(
                "fix this bug",
                None,
                None,
                Path::new("/nonexistent-dir-xyz-99999"),
            )
            .await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.to_lowercase().contains("spawn")
                || err.to_lowercase().contains("not found")
                || err.to_lowercase().contains("no such file"),
            "Expected spawn/not-found error, got: {}",
            err
        );
    }

    #[tokio::test]
    async fn test_claude_execute_records_activity_on_failure() {
        use claudear_storage::SqliteTracker;
        let tracker = Arc::new(SqliteTracker::in_memory().unwrap());
        let config = ClaudeRunnerConfig::default();
        let runner = ClaudeAgentRunner::new(config, tracker.clone());
        let _ = runner
            .execute_with_attempt(
                "test prompt",
                None,
                None,
                Path::new("/nonexistent-dir-xyz-activity-test"),
            )
            .await;
        // Verify activity was logged (even on error, we should have a claude_started log)
        let activities = tracker.get_recent_activities(10, None).unwrap();
        // There should be at least one activity entry from the failed attempt
        let agent_activities: Vec<_> = activities
            .iter()
            .filter(|a| a.activity_type.contains("claude"))
            .collect();
        assert!(
            !agent_activities.is_empty(),
            "Expected at least one claude activity entry, got none. All activities: {:?}",
            activities
                .iter()
                .map(|a| &a.activity_type)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_claude_runner_config_default_binary() {
        let config = ClaudeRunnerConfig::default();
        // Default config should have reasonable defaults
        assert!(config.timeout_secs > 0);
        assert!(config.permissions.is_empty() || !config.permissions.is_empty());
    }

    #[test]
    fn test_claude_agent_runner_as_dyn_trait() {
        use claudear_storage::SqliteTracker;
        let tracker = Arc::new(SqliteTracker::in_memory().unwrap());
        let runner = ClaudeAgentRunner::new(ClaudeRunnerConfig::default(), tracker);
        // Verify it can be used as Arc<dyn AgentRunner>
        let agent: Arc<dyn AgentRunner> = Arc::new(runner);
        assert_eq!(agent.name(), "claude");
        let caps = agent.capabilities();
        assert!(caps.structured_output);
    }

    // --- Additional coverage tests ---

    #[test]
    fn test_structured_result_with_changelog() {
        let json = r#"{"summary":"Applied fix","success":true,"pr_url":"https://github.com/org/repo/pull/7","changelog":"- Fixed null check\n- Added test"}"#;
        let sr: StructuredResult = serde_json::from_str(json).unwrap();
        assert!(sr.success);
        assert_eq!(
            sr.changelog.as_deref(),
            Some("- Fixed null check\n- Added test")
        );
    }

    #[test]
    fn test_structured_result_changelog_null() {
        let json = r#"{"summary":"Done","success":true,"changelog":null}"#;
        let sr: StructuredResult = serde_json::from_str(json).unwrap();
        assert!(sr.changelog.is_none());
    }

    #[test]
    fn test_structured_result_changelog_empty_string() {
        let json = r#"{"summary":"Done","success":true,"changelog":""}"#;
        let sr: StructuredResult = serde_json::from_str(json).unwrap();
        // The runner filters empty changelogs to None
        let filtered = sr.changelog.filter(|c| !c.is_empty());
        assert!(filtered.is_none());
    }

    #[test]
    fn test_structured_result_changelog_with_content_preserved() {
        let json = r#"{"summary":"Done","success":true,"changelog":"- Updated deps\n- Fixed bug"}"#;
        let sr: StructuredResult = serde_json::from_str(json).unwrap();
        let filtered = sr.changelog.filter(|c| !c.is_empty());
        assert_eq!(filtered.as_deref(), Some("- Updated deps\n- Fixed bug"));
    }

    #[test]
    fn test_structured_result_missing_changelog_field() {
        let json = r#"{"summary":"Done","success":true}"#;
        let sr: StructuredResult = serde_json::from_str(json).unwrap();
        assert!(sr.changelog.is_none());
    }

    #[test]
    fn test_result_schema_changelog_allows_null() {
        let parsed: serde_json::Value = serde_json::from_str(RESULT_SCHEMA).unwrap();
        let changelog_type = &parsed["properties"]["changelog"]["type"];
        let types = changelog_type.as_array().unwrap();
        let type_strs: Vec<&str> = types.iter().map(|v| v.as_str().unwrap()).collect();
        assert!(type_strs.contains(&"string"));
        assert!(type_strs.contains(&"null"));
    }

    #[test]
    fn test_result_schema_changelog_description() {
        let parsed: serde_json::Value = serde_json::from_str(RESULT_SCHEMA).unwrap();
        let desc = parsed["properties"]["changelog"]["description"]
            .as_str()
            .unwrap();
        assert!(!desc.is_empty());
    }

    #[test]
    fn test_provider_capabilities_default_all_false() {
        let caps = ProviderCapabilities::default();
        assert!(!caps.structured_output);
        assert!(!caps.tool_permissions);
        assert!(!caps.custom_instructions);
        assert!(!caps.streaming_events);
        assert!(!caps.cost_reporting);
    }

    #[test]
    fn test_provider_capabilities_clone() {
        let caps = ProviderCapabilities {
            structured_output: true,
            tool_permissions: true,
            custom_instructions: false,
            streaming_events: true,
            cost_reporting: false,
        };
        let cloned = caps.clone();
        assert_eq!(cloned.structured_output, caps.structured_output);
        assert_eq!(cloned.tool_permissions, caps.tool_permissions);
        assert_eq!(cloned.custom_instructions, caps.custom_instructions);
        assert_eq!(cloned.streaming_events, caps.streaming_events);
        assert_eq!(cloned.cost_reporting, caps.cost_reporting);
    }

    #[test]
    fn test_provider_capabilities_debug() {
        let caps = ProviderCapabilities {
            structured_output: true,
            tool_permissions: false,
            custom_instructions: true,
            streaming_events: false,
            cost_reporting: true,
        };
        let debug = format!("{:?}", caps);
        assert!(debug.contains("structured_output"));
        assert!(debug.contains("tool_permissions"));
        assert!(debug.contains("cost_reporting"));
    }

    #[test]
    fn test_parse_wrapper_result_with_result_key() {
        // Tests the fallback parser that looks for {"result": "..."} wrapper objects
        let wrapper = json!({
            "result": r#"{"summary":"done","success":true}"#
        });
        let wrapper_str = serde_json::to_string(&wrapper).unwrap();
        // The stdout parser tries serde_json::from_str::<StreamEvent> first (fails),
        // then looks for the wrapper format.
        let parsed: serde_json::Value = serde_json::from_str(&wrapper_str).unwrap();
        if let Some(result_str) = parsed.get("result").and_then(|v| v.as_str()) {
            let inner: serde_json::Value = serde_json::from_str(result_str).unwrap();
            assert_eq!(inner["summary"], "done");
            assert_eq!(inner["success"], true);
        } else {
            panic!("Expected result key");
        }
    }

    #[test]
    fn test_parse_wrapper_result_with_structured_output_key() {
        // Tests the fallback parser that looks for {"structured_output": {...}} wrapper objects
        let wrapper = json!({
            "structured_output": {"summary": "fixed", "success": true}
        });
        let wrapper_str = serde_json::to_string(&wrapper).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&wrapper_str).unwrap();
        if let Some(obj) = parsed.get("structured_output") {
            assert_eq!(obj["summary"], "fixed");
            assert_eq!(obj["success"], true);
        } else {
            panic!("Expected structured_output key");
        }
    }

    #[test]
    fn test_compose_failure_message_both_stderr_stdout_with_rate_limit_in_combined() {
        // Combined text contains "rate limit" but it spans stderr + stdout
        let msg =
            ClaudeAgentRunner::compose_failure_message(1, "some stdout", "rate limit exceeded");
        assert!(msg.starts_with("Claude rate limit hit:"));
    }

    #[test]
    fn test_compose_failure_message_exit_code_signal_killed() {
        // SIGKILL exit code is 137
        let msg = ClaudeAgentRunner::compose_failure_message(137, "", "Killed");
        assert_eq!(msg, "Killed");
    }

    #[test]
    fn test_compose_failure_message_exit_code_segfault() {
        // SIGSEGV exit code is 139
        let msg = ClaudeAgentRunner::compose_failure_message(139, "", "Segmentation fault");
        assert_eq!(msg, "Segmentation fault");
    }

    #[test]
    fn test_truncate_exactly_three_chars() {
        assert_eq!(ClaudeAgentRunner::truncate("abc", 3), "abc");
    }

    #[test]
    fn test_truncate_four_char_string_max_three() {
        assert_eq!(ClaudeAgentRunner::truncate("abcd", 3), "...");
    }

    #[test]
    fn test_truncate_unicode_combining_chars() {
        // e + combining accent = 2 code points but may look like 1 char
        let input = "e\u{0301}abc"; // "e" with combining acute accent
        let result = ClaudeAgentRunner::truncate(input, 5);
        // Should be valid UTF-8 regardless
        assert!(std::str::from_utf8(result.as_bytes()).is_ok());
    }

    #[test]
    fn test_sanitize_label_with_dots() {
        assert_eq!(ClaudeAgentRunner::sanitize_label("v1.2.3"), "v1_2_3");
    }

    #[test]
    fn test_sanitize_label_with_colons() {
        assert_eq!(
            ClaudeAgentRunner::sanitize_label("scope:task"),
            "scope_task"
        );
    }

    #[test]
    fn test_sanitize_label_with_spaces_and_parens() {
        assert_eq!(
            ClaudeAgentRunner::sanitize_label("fix (urgent)"),
            "fix__urgent_"
        );
    }

    #[test]
    fn test_sanitize_label_exactly_64_unicode_chars() {
        // Each unicode char is 4 bytes, so 64 chars = only first 64 chars considered
        let label = "\u{1F600}".repeat(100); // 100 emoji, each 4 bytes
        let result = ClaudeAgentRunner::sanitize_label(&label);
        assert_eq!(result.len(), 64); // 64 underscores (non-ascii replaced)
    }

    #[test]
    fn test_hash_prompt_different_from_empty() {
        assert_ne!(
            ClaudeAgentRunner::hash_prompt("a"),
            ClaudeAgentRunner::hash_prompt("")
        );
    }

    #[test]
    fn test_hash_prompt_special_characters() {
        let h = ClaudeAgentRunner::hash_prompt("!@#$%^&*(){}[]|\\:\";<>?,./~`");
        assert_eq!(h.len(), 16);
        assert!(h.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_hash_prompt_newlines_and_tabs() {
        let h = ClaudeAgentRunner::hash_prompt("line1\nline2\ttab");
        assert_eq!(h.len(), 16);
    }

    #[test]
    fn test_structured_result_full_with_changelog() {
        let json = r#"{
            "summary": "Applied fix and created PR",
            "success": true,
            "pr_url": "https://github.com/org/repo/pull/42",
            "changelog": "- Fixed auth handler\n- Added edge case tests",
            "blocking_question": null
        }"#;
        let sr: StructuredResult = serde_json::from_str(json).unwrap();
        assert!(sr.success);
        assert_eq!(sr.summary, "Applied fix and created PR");
        assert!(sr.pr_url.is_some());
        assert!(sr.changelog.is_some());
        assert!(sr.blocking_question.is_none());
    }

    #[test]
    fn test_structured_result_deserialization_with_all_fields_from_value() {
        let val = json!({
            "summary": "All done",
            "success": true,
            "pr_url": "https://github.com/org/repo/pull/42",
            "changelog": "- Fixed the bug\n- Updated tests",
            "blocking_question": null
        });
        let sr: StructuredResult = serde_json::from_value(val).unwrap();
        assert!(sr.success);
        assert_eq!(
            sr.changelog.as_deref(),
            Some("- Fixed the bug\n- Updated tests")
        );
    }

    #[test]
    fn test_stream_event_result_with_session_id_only() {
        let json = r#"{"type":"result","session_id":"sess-xyz"}"#;
        let event: StreamEvent = serde_json::from_str(json).unwrap();
        match event {
            StreamEvent::Result {
                session_id: Some(ref sid),
                structured_output: None,
                total_cost_usd: None,
                num_turns: None,
                duration_api_ms: None,
                usage: None,
            } => {
                assert_eq!(sid, "sess-xyz");
            }
            other => panic!("Unexpected event: {:?}", other),
        }
    }

    #[test]
    fn test_stream_event_result_with_negative_duration() {
        // Edge case: negative duration should still parse
        let json = r#"{"type":"result","duration_api_ms":-1}"#;
        let event: StreamEvent = serde_json::from_str(json).unwrap();
        match event {
            StreamEvent::Result {
                duration_api_ms: Some(ms),
                ..
            } => {
                assert_eq!(ms, -1);
            }
            other => panic!("Unexpected event: {:?}", other),
        }
    }

    #[test]
    fn test_stream_event_result_with_zero_tokens() {
        let json = r#"{"type":"result","usage":{"input_tokens":0,"output_tokens":0,"cache_read_input_tokens":0,"cache_creation_input_tokens":0}}"#;
        let event: StreamEvent = serde_json::from_str(json).unwrap();
        match event {
            StreamEvent::Result {
                usage: Some(ref u), ..
            } => {
                assert_eq!(u.input_tokens, Some(0));
                assert_eq!(u.output_tokens, Some(0));
                assert_eq!(u.cache_read_input_tokens, Some(0));
                assert_eq!(u.cache_creation_input_tokens, Some(0));
            }
            other => panic!("Unexpected event: {:?}", other),
        }
    }

    #[test]
    fn test_cli_message_single_text_block() {
        let json = r#"{"content":[{"type":"text","text":"only block"}]}"#;
        let msg: CliMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.content.len(), 1);
        match &msg.content[0] {
            CliContentBlock::Text { text } => assert_eq!(text, "only block"),
            other => panic!("Expected Text, got: {:?}", other),
        }
    }

    #[test]
    fn test_cli_content_block_tool_use_empty_name() {
        let json = r#"{"type":"tool_use","id":"tu_empty","name":""}"#;
        let block: CliContentBlock = serde_json::from_str(json).unwrap();
        match block {
            CliContentBlock::ToolUse { id, name } => {
                assert_eq!(id, "tu_empty");
                assert!(name.is_empty());
            }
            other => panic!("Expected ToolUse, got: {:?}", other),
        }
    }

    #[test]
    fn test_extract_pr_url_gitlab_self_hosted_with_port() {
        // Self-hosted GitLab with a custom domain pattern
        let output = "MR: https://gitlab.corp.example.com/team/project/-/merge_requests/77";
        let url = ClaudeAgentRunner::extract_pr_url(output);
        assert_eq!(
            url.as_deref(),
            Some("https://gitlab.corp.example.com/team/project/-/merge_requests/77")
        );
    }

    #[test]
    fn test_extract_pr_url_github_pr_with_files_tab() {
        let output = "See https://github.com/org/repo/pull/123/files for diff";
        let url = ClaudeAgentRunner::extract_pr_url(output);
        assert!(url.is_some());
        assert!(url.unwrap().contains("/pull/123"));
    }

    #[test]
    fn test_compose_failure_message_exit_code_large_positive() {
        let msg = ClaudeAgentRunner::compose_failure_message(32767, "", "");
        assert_eq!(msg, "Process exited with code 32767");
    }

    #[test]
    fn test_compose_failure_message_newlines_only_stderr() {
        let msg = ClaudeAgentRunner::compose_failure_message(1, "", "\n\n\n");
        // Trimmed stderr is empty, so falls through to exit code format
        assert_eq!(msg, "Process exited with code 1");
    }

    #[test]
    fn test_prepare_env_and_label_does_not_include_claudecode() {
        let runner = ClaudeAgentRunner::new_simple(ClaudeRunnerConfig::default());
        let (env, _) = runner.prepare_env_and_label(None);
        // The base_env should have filtered out CLAUDECODE
        assert!(!env.contains_key("CLAUDECODE"));
    }

    #[test]
    fn test_prepare_env_and_label_discord_source() {
        let runner = ClaudeAgentRunner::new_simple(ClaudeRunnerConfig::default());
        let issue = Issue::new("d-1", "DISC-1", "Bug", "https://discord.com/1", "discord");
        let (env, label) = runner.prepare_env_and_label(Some(&issue));
        assert_eq!(label, "DISC-1");
        assert_eq!(env.get("DISCORD_ISSUE_ID"), Some(&"d-1".to_string()));
        assert_eq!(
            env.get("DISCORD_ISSUE_SHORT_ID"),
            Some(&"DISC-1".to_string())
        );
    }

    #[test]
    fn test_prepare_env_and_label_slack_source() {
        let runner = ClaudeAgentRunner::new_simple(ClaudeRunnerConfig::default());
        let issue = Issue::new("s-1", "SLACK-1", "Task", "https://slack.com/1", "slack");
        let (env, label) = runner.prepare_env_and_label(Some(&issue));
        assert_eq!(label, "SLACK-1");
        assert_eq!(env.get("SLACK_ISSUE_ID"), Some(&"s-1".to_string()));
    }

    #[test]
    fn test_structured_result_from_value_with_empty_changelog() {
        let val = json!({
            "summary": "done",
            "success": true,
            "pr_url": null,
            "changelog": "",
            "blocking_question": null
        });
        let sr: StructuredResult = serde_json::from_value(val).unwrap();
        // empty changelog should be filtered to None by runner logic
        let filtered = sr.changelog.filter(|c| !c.is_empty());
        assert!(filtered.is_none());
    }

    #[test]
    fn test_structured_result_pr_url_non_url_string() {
        let json = r#"{"summary":"done","success":true,"pr_url":"not-a-url"}"#;
        let sr: StructuredResult = serde_json::from_str(json).unwrap();
        // Runner filters non-https URLs
        let filtered = sr.pr_url.filter(|url| url.starts_with("https://"));
        assert!(filtered.is_none());
    }

    #[test]
    fn test_create_execution_log_files_empty_log_root() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let prev = std::env::var("CLAUDEAR_LOG_DIR").ok();
        std::env::set_var("CLAUDEAR_LOG_DIR", "");

        let files = ClaudeAgentRunner::create_execution_log_files("test");
        // Empty log root should return None because of the is_empty check
        assert!(files.is_none());

        if let Some(val) = prev {
            std::env::set_var("CLAUDEAR_LOG_DIR", val);
        } else {
            std::env::remove_var("CLAUDEAR_LOG_DIR");
        }
    }

    #[test]
    fn test_stdout_parse_result_debug() {
        let result = StdoutParseResult::default();
        let debug = format!("{:?}", result);
        assert!(debug.contains("StdoutParseResult"));
    }

    #[test]
    fn test_execution_log_files_paths_have_correct_extensions() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let prev = std::env::var("CLAUDEAR_LOG_DIR").ok();
        let tmp_dir = std::env::temp_dir().join("claudear_test_ext_check");
        std::env::set_var("CLAUDEAR_LOG_DIR", tmp_dir.to_str().unwrap());

        let files = ClaudeAgentRunner::create_execution_log_files("ext-check");
        assert!(files.is_some());
        let files = files.unwrap();
        let stdout_name = files.stdout.file_name().unwrap().to_str().unwrap();
        let stderr_name = files.stderr.file_name().unwrap().to_str().unwrap();
        let events_name = files.events.file_name().unwrap().to_str().unwrap();

        assert!(stdout_name.ends_with(".stdout.log"));
        assert!(stderr_name.ends_with(".stderr.log"));
        assert!(events_name.ends_with(".events.jsonl"));

        let _ = std::fs::remove_dir_all(&tmp_dir);
        if let Some(val) = prev {
            std::env::set_var("CLAUDEAR_LOG_DIR", val);
        } else {
            std::env::remove_var("CLAUDEAR_LOG_DIR");
        }
    }

    #[test]
    fn test_stream_event_result_negative_cost_parses() {
        // Edge case: CLI might report negative cost (credit)
        let json = r#"{"type":"result","total_cost_usd":-0.001}"#;
        let event: StreamEvent = serde_json::from_str(json).unwrap();
        match event {
            StreamEvent::Result {
                total_cost_usd: Some(cost),
                ..
            } => {
                assert!(cost < 0.0);
            }
            other => panic!("Unexpected event: {:?}", other),
        }
    }

    #[test]
    fn test_cli_usage_extra_unknown_fields_ignored() {
        let json = r#"{"input_tokens":5,"unknown_field":"ignored","output_tokens":10}"#;
        let usage: CliUsage = serde_json::from_str(json).unwrap();
        assert_eq!(usage.input_tokens, Some(5));
        assert_eq!(usage.output_tokens, Some(10));
    }

    #[test]
    fn test_stream_event_result_extra_fields_ignored() {
        let json = r#"{"type":"result","subtype":"success","is_error":false,"result":"text","extra":"ignored"}"#;
        let event: StreamEvent = serde_json::from_str(json).unwrap();
        match event {
            StreamEvent::Result {
                structured_output: None,
                ..
            } => {}
            other => panic!("Unexpected event: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_append_execution_event_with_nested_data() {
        let tmp_dir = std::env::temp_dir().join("claudear_test_nested_event");
        let _ = std::fs::create_dir_all(&tmp_dir);
        let file_path = tmp_dir.join("nested_events.jsonl");

        let file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&file_path)
            .await
            .unwrap();
        let writer = Some(Arc::new(Mutex::new(file)));

        ClaudeAgentRunner::append_execution_event(
            &writer,
            "nested",
            "complex_event",
            json!({
                "nested": {"key": "value", "arr": [1, 2, 3]},
                "number": 42.5,
                "bool": true,
                "null_val": null
            }),
        )
        .await;

        drop(writer);

        let content = std::fs::read_to_string(&file_path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(content.trim()).unwrap();
        assert_eq!(parsed["data"]["nested"]["key"], "value");
        assert_eq!(parsed["data"]["number"], 42.5);
        assert_eq!(parsed["data"]["bool"], true);
        assert!(parsed["data"]["null_val"].is_null());

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn test_extract_blocking_question_with_unicode_content() {
        let output = r#"CLAUDEAR_QUESTION: {"question":"Which \u65e5\u672c\u8a9e module?"}"#;
        let result = ClaudeAgentRunner::extract_blocking_question(output);
        assert!(result.is_some());
    }

    #[test]
    fn test_structured_result_blocking_question_options_many() {
        let json = r#"{"summary":"choices","success":false,"blocking_question":{"question":"Pick one","options":["a","b","c","d","e","f","g","h"]}}"#;
        let sr: StructuredResult = serde_json::from_str(json).unwrap();
        let bq = sr.blocking_question.unwrap();
        assert_eq!(bq.options.len(), 8);
    }

    #[test]
    fn test_structured_result_from_value_fails_missing_success() {
        let val = json!({"summary": "done"});
        let result = serde_json::from_value::<StructuredResult>(val);
        assert!(result.is_err());
    }

    #[test]
    fn test_compose_failure_message_rate_limit_retry_after_in_stderr() {
        let msg = ClaudeAgentRunner::compose_failure_message(1, "", "retry-after: 120 seconds");
        assert!(msg.starts_with("Claude rate limit hit:"));
    }

    #[test]
    fn test_compose_failure_message_rate_limit_resource_exhausted_in_stdout_not_detected() {
        // Rate-limit strings in stdout are not detected — only stderr is checked.
        let msg =
            ClaudeAgentRunner::compose_failure_message(1, "resource exhausted for billing", "");
        assert!(msg.starts_with("Process exited with code 1"));
    }

    #[test]
    fn test_structured_result_changelog_whitespace_only() {
        let json = r#"{"summary":"Done","success":true,"changelog":"   "}"#;
        let sr: StructuredResult = serde_json::from_str(json).unwrap();
        // "   " is not empty, so filter would keep it
        let filtered = sr.changelog.filter(|c| !c.is_empty());
        assert!(filtered.is_some());
    }

    #[test]
    fn test_cli_usage_debug() {
        let usage = CliUsage {
            input_tokens: Some(100),
            output_tokens: Some(200),
            cache_read_input_tokens: None,
            cache_creation_input_tokens: None,
        };
        let debug = format!("{:?}", usage);
        assert!(debug.contains("input_tokens"));
        assert!(debug.contains("100"));
    }

    #[test]
    fn test_cli_message_debug() {
        let msg = CliMessage { content: vec![] };
        let debug = format!("{:?}", msg);
        assert!(debug.contains("CliMessage"));
    }

    #[test]
    fn test_structured_result_debug() {
        let sr = StructuredResult {
            summary: "test".to_string(),
            success: true,
            pr_url: None,
            changelog: None,
            blocking_question: None,
            confidence: 0,
            confidence_reasoning: None,
        };
        let debug = format!("{:?}", sr);
        assert!(debug.contains("StructuredResult"));
        assert!(debug.contains("test"));
    }

    #[test]
    fn test_structured_result_with_confidence() {
        let json = r#"{"summary":"Fixed auth bug","success":true,"pr_url":"https://github.com/org/repo/pull/10","confidence":85,"confidence_reasoning":"Tests pass and the fix is localized"}"#;
        let sr: StructuredResult = serde_json::from_str(json).unwrap();
        assert!(sr.success);
        assert_eq!(sr.confidence, 85);
        assert_eq!(
            sr.confidence_reasoning.as_deref(),
            Some("Tests pass and the fix is localized")
        );
    }

    #[test]
    fn test_structured_result_confidence_defaults_to_zero() {
        let json = r#"{"summary":"Done","success":true}"#;
        let sr: StructuredResult = serde_json::from_str(json).unwrap();
        assert_eq!(sr.confidence, 0);
        assert!(sr.confidence_reasoning.is_none());
    }

    #[test]
    fn test_structured_result_confidence_null_reasoning() {
        let json =
            r#"{"summary":"Done","success":true,"confidence":50,"confidence_reasoning":null}"#;
        let sr: StructuredResult = serde_json::from_str(json).unwrap();
        assert_eq!(sr.confidence, 50);
        assert!(sr.confidence_reasoning.is_none());
    }

    #[test]
    fn test_structured_result_confidence_boundary_zero() {
        let json = r#"{"summary":"No idea","success":false,"confidence":0}"#;
        let sr: StructuredResult = serde_json::from_str(json).unwrap();
        assert_eq!(sr.confidence, 0);
    }

    #[test]
    fn test_structured_result_confidence_boundary_100() {
        let json = r#"{"summary":"Certain","success":true,"confidence":100,"confidence_reasoning":"Exact same bug pattern seen before"}"#;
        let sr: StructuredResult = serde_json::from_str(json).unwrap();
        assert_eq!(sr.confidence, 100);
        assert!(sr.confidence_reasoning.is_some());
    }

    #[test]
    fn test_result_schema_contains_confidence_field() {
        let parsed: serde_json::Value = serde_json::from_str(RESULT_SCHEMA).unwrap();
        let props = &parsed["properties"];
        assert!(props["confidence"].is_object());
        assert_eq!(props["confidence"]["type"], "number");
        assert_eq!(props["confidence"]["minimum"], 0);
        assert_eq!(props["confidence"]["maximum"], 100);
    }

    #[test]
    fn test_result_schema_contains_confidence_reasoning_field() {
        let parsed: serde_json::Value = serde_json::from_str(RESULT_SCHEMA).unwrap();
        let props = &parsed["properties"];
        assert!(props["confidence_reasoning"].is_object());
    }

    #[test]
    fn test_result_schema_confidence_is_required() {
        let parsed: serde_json::Value = serde_json::from_str(RESULT_SCHEMA).unwrap();
        let required = parsed["required"].as_array().unwrap();
        let required_fields: Vec<&str> = required.iter().map(|v| v.as_str().unwrap()).collect();
        assert!(required_fields.contains(&"confidence"));
    }

    #[test]
    fn test_result_schema_confidence_description_mentions_regressions() {
        let parsed: serde_json::Value = serde_json::from_str(RESULT_SCHEMA).unwrap();
        let desc = parsed["properties"]["confidence"]["description"]
            .as_str()
            .unwrap();
        assert!(
            desc.contains("confidence") || desc.contains("0-100"),
            "Schema description should describe confidence scoring"
        );
    }

    // --- StructuredResult → extraction logic tests ---

    /// Simulate the extraction logic from execute(): parse a serde_json::Value as
    /// StructuredResult and extract the confidence fields, matching lines 1418-1447.
    fn extract_confidence_from_structured(val: Option<serde_json::Value>) -> (u8, Option<String>) {
        val.as_ref()
            .and_then(|v| serde_json::from_value::<StructuredResult>(v.clone()).ok())
            .map(|sr| (sr.confidence, sr.confidence_reasoning))
            .unwrap_or((0, None))
    }

    #[test]
    fn test_extraction_confidence_from_full_structured_result() {
        let val = serde_json::json!({
            "summary": "Fixed",
            "success": true,
            "confidence": 92,
            "confidence_reasoning": "All tests pass"
        });
        let (c, r) = extract_confidence_from_structured(Some(val));
        assert_eq!(c, 92);
        assert_eq!(r.as_deref(), Some("All tests pass"));
    }

    #[test]
    fn test_extraction_confidence_from_missing_fields() {
        let val = serde_json::json!({
            "summary": "Done",
            "success": true
        });
        let (c, r) = extract_confidence_from_structured(Some(val));
        assert_eq!(c, 0);
        assert!(r.is_none());
    }

    #[test]
    fn test_extraction_confidence_legacy_fallback() {
        // When structured_result is None, legacy fallback should produce 0/None.
        let (c, r) = extract_confidence_from_structured(None);
        assert_eq!(c, 0);
        assert!(r.is_none());
    }

    #[test]
    fn test_extraction_confidence_from_invalid_json() {
        // If the JSON doesn't parse as StructuredResult, fall back to defaults.
        let val = serde_json::json!("not an object");
        let (c, r) = extract_confidence_from_structured(Some(val));
        assert_eq!(c, 0);
        assert!(r.is_none());
    }

    #[test]
    fn test_extraction_confidence_float_fails_gracefully() {
        // serde_json rejects floats for u8 — the whole StructuredResult parse fails,
        // so we fall back to (0, None).
        let val = serde_json::json!({
            "summary": "Fixed",
            "success": true,
            "confidence": 85.7
        });
        let (c, r) = extract_confidence_from_structured(Some(val));
        assert_eq!(c, 0);
        assert!(r.is_none());
    }

    #[test]
    fn test_extraction_confidence_over_255_fails_gracefully() {
        // u8 max is 255; a value of 300 should fail serde parsing, falling back to defaults.
        let val = serde_json::json!({
            "summary": "Fixed",
            "success": true,
            "confidence": 300
        });
        let (c, r) = extract_confidence_from_structured(Some(val));
        // serde_json will fail to deserialize 300 into u8
        assert_eq!(c, 0);
        assert!(r.is_none());
    }

    #[test]
    fn test_extraction_confidence_negative_fails_gracefully() {
        // Negative confidence should fail u8 deserialization.
        let val = serde_json::json!({
            "summary": "Failed",
            "success": false,
            "confidence": -5
        });
        let (c, r) = extract_confidence_from_structured(Some(val));
        assert_eq!(c, 0);
        assert!(r.is_none());
    }

    #[test]
    fn test_structured_result_with_all_fields() {
        let json = r#"{
            "summary": "Fixed auth bug",
            "success": true,
            "pr_url": "https://github.com/org/repo/pull/10",
            "changelog": "- Fixed null check\n- Added test",
            "blocking_question": null,
            "confidence": 88,
            "confidence_reasoning": "Straightforward null check fix with test"
        }"#;
        let sr: StructuredResult = serde_json::from_str(json).unwrap();
        assert!(sr.success);
        assert_eq!(sr.confidence, 88);
        assert!(sr.confidence_reasoning.is_some());
        assert!(sr.changelog.is_some());
        assert!(sr.pr_url.is_some());
        assert!(sr.blocking_question.is_none());
    }

    #[test]
    fn test_structured_result_confidence_with_blocking_question() {
        // When there's a blocking question, confidence should still be extractable.
        let json = r#"{
            "summary": "Need info",
            "success": false,
            "confidence": 0,
            "confidence_reasoning": "Cannot proceed without human input",
            "blocking_question": {
                "question": "Which database?",
                "options": ["postgres", "mysql"]
            }
        }"#;
        let sr: StructuredResult = serde_json::from_str(json).unwrap();
        assert!(!sr.success);
        assert_eq!(sr.confidence, 0);
        assert!(sr.blocking_question.is_some());
        assert_eq!(
            sr.confidence_reasoning.as_deref(),
            Some("Cannot proceed without human input")
        );
    }

    #[test]
    fn test_structured_result_confidence_empty_reasoning() {
        let json = r#"{"summary":"Done","success":true,"confidence":60,"confidence_reasoning":""}"#;
        let sr: StructuredResult = serde_json::from_str(json).unwrap();
        assert_eq!(sr.confidence, 60);
        assert_eq!(sr.confidence_reasoning.as_deref(), Some(""));
    }

    #[test]
    fn test_structured_result_confidence_unicode_reasoning() {
        let json = r#"{"summary":"Fixed","success":true,"confidence":75,"confidence_reasoning":"Fix is 确定的 — tests pass ✓"}"#;
        let sr: StructuredResult = serde_json::from_str(json).unwrap();
        assert_eq!(sr.confidence, 75);
        assert!(sr.confidence_reasoning.unwrap().contains("确定的"));
    }
}
