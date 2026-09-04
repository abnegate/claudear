#![cfg(unix)]

use async_trait::async_trait;
use claudear::config::{AskConfig, Config, RetryConfig};
use claudear::notifier::Notifier;
use claudear::repo::{IndexedRepo, RepoIndex};
use claudear::source::IssueSource;
use claudear::storage::{AttemptTracker, FixAttemptTracker, SqliteTracker};
use claudear::types::{FixAttemptStatus, Issue, MatchPriority, MatchResult};
use claudear::watcher::{Watcher, WatcherOptions};
use claudear::{Error, RepoInferrer, Result, UserRegistry};
use std::collections::HashMap;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};
use tempfile::TempDir;

static ENV_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

struct EnvVarGuard {
    key: &'static str,
    original: Option<String>,
}

impl EnvVarGuard {
    fn set(key: &'static str, value: impl Into<String>) -> Self {
        let original = std::env::var(key).ok();
        std::env::set_var(key, value.into());
        Self { key, original }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        match self.original.as_ref() {
            Some(value) => std::env::set_var(self.key, value),
            None => std::env::remove_var(self.key),
        }
    }
}

#[derive(Default)]
struct RecordingNotifier {
    events: Mutex<Vec<String>>,
}

impl RecordingNotifier {
    fn record(&self, event: impl Into<String>) {
        self.events.lock().unwrap().push(event.into());
    }

    fn events(&self) -> Vec<String> {
        self.events.lock().unwrap().clone()
    }
}

#[async_trait]
impl Notifier for RecordingNotifier {
    fn name(&self) -> &str {
        "recording"
    }

    fn is_enabled(&self) -> bool {
        true
    }

    async fn notify_start(&self, issue: &Issue) -> Result<()> {
        self.record(format!("start:{}", issue.short_id));
        Ok(())
    }

    async fn notify_success(&self, issue: &Issue, _pr_url: &str) -> Result<()> {
        self.record(format!("success:{}", issue.short_id));
        Ok(())
    }

    async fn notify_completed(&self, issue: &Issue) -> Result<()> {
        self.record(format!("completed:{}", issue.short_id));
        Ok(())
    }

    async fn notify_failed(&self, issue: &Issue, _error: &str) -> Result<()> {
        self.record(format!("failed:{}", issue.short_id));
        Ok(())
    }

    async fn notify_status(&self, _message: &str) -> Result<()> {
        Ok(())
    }

    async fn notify_urgent_issues(&self, _issues: &[Issue]) -> Result<()> {
        Ok(())
    }
}

#[derive(Clone)]
struct TaskSource {
    name: String,
    tasks: Arc<Mutex<HashMap<String, Issue>>>,
}

impl TaskSource {
    fn new(name: &str, tasks: Vec<Issue>) -> Self {
        let tasks_map = tasks.into_iter().map(|i| (i.id.clone(), i)).collect();
        Self {
            name: name.to_string(),
            tasks: Arc::new(Mutex::new(tasks_map)),
        }
    }

    fn reset_tasks(&self) {
        self.tasks.lock().unwrap().clear();
    }

    fn task_count(&self) -> usize {
        self.tasks.lock().unwrap().len()
    }
}

#[async_trait]
impl IssueSource for TaskSource {
    fn name(&self) -> &str {
        &self.name
    }

    fn display_name(&self) -> &str {
        &self.name
    }

    async fn fetch_issues(&self) -> Result<Vec<Issue>> {
        Ok(self.tasks.lock().unwrap().values().cloned().collect())
    }

    fn matches_criteria(&self, _issue: &Issue) -> MatchResult {
        MatchResult::matched("e2e task", MatchPriority::High)
    }

    async fn build_issue_context(&self, issue: &Issue) -> Result<String> {
        let description = issue.description.as_deref().unwrap_or("");
        Ok(format!("{}\n{}", issue.title, description))
    }

    async fn get_issue(&self, issue_id: &str) -> Result<Issue> {
        self.tasks
            .lock()
            .unwrap()
            .get(issue_id)
            .cloned()
            .ok_or_else(|| Error::issue_not_found(&self.name, issue_id))
    }
}

struct E2eHarness {
    _temp_dir: TempDir,
    _path_guard: EnvVarGuard,
    _log_dir_guard: EnvVarGuard,
    repo_path: PathBuf,
    remote_path: PathBuf,
    baseline_file_contents: String,
    source: Arc<TaskSource>,
    notifier: Arc<RecordingNotifier>,
    tracker: Arc<SqliteTracker>,
    watcher: Watcher,
}

fn run_git(cwd: &Path, args: &[&str]) {
    let output = Command::new("git")
        .args(args)
        .current_dir(cwd)
        .output()
        .expect("failed to run git command");

    assert!(
        output.status.success(),
        "git {:?} failed in {}.\nstdout:\n{}\nstderr:\n{}",
        args,
        cwd.display(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Read a file from a specific branch in a bare repo via `git show`.
fn read_remote_file(bare_repo: &Path, branch: &str, file_path: &str) -> Option<String> {
    let output = Command::new("git")
        .args(["show", &format!("{}:{}", branch, file_path)])
        .current_dir(bare_repo)
        .output()
        .ok()?;
    if output.status.success() {
        Some(String::from_utf8_lossy(&output.stdout).to_string())
    } else {
        None
    }
}

fn create_real_repo(temp_dir: &TempDir) -> (PathBuf, PathBuf) {
    let remote_path = temp_dir.path().join("remote.git");
    let repo_path = temp_dir.path().join("repo");

    let remote_path_str = remote_path.to_string_lossy().to_string();
    let repo_path_str = repo_path.to_string_lossy().to_string();

    run_git(
        temp_dir.path(),
        &["init", "--bare", remote_path_str.as_str()],
    );
    run_git(
        temp_dir.path(),
        &["clone", remote_path_str.as_str(), repo_path_str.as_str()],
    );

    run_git(&repo_path, &["config", "user.email", "e2e@example.com"]);
    run_git(&repo_path, &["config", "user.name", "E2E Bot"]);

    fs::create_dir_all(repo_path.join("src")).expect("failed to create src directory");
    fs::write(
        repo_path.join("src/buggy.rs"),
        "pub fn buggy() -> &'static str {\n    \"baseline\"\n}\n",
    )
    .expect("failed to write baseline file");

    run_git(&repo_path, &["add", "."]);
    run_git(&repo_path, &["commit", "-m", "chore: seed e2e repo"]);
    run_git(&repo_path, &["branch", "-M", "main"]);
    run_git(&repo_path, &["push", "-u", "origin", "main"]);

    (repo_path, remote_path)
}

fn write_claude_stub(temp_dir: &TempDir) -> PathBuf {
    let bin_dir = temp_dir.path().join("bin");
    fs::create_dir_all(&bin_dir).expect("failed to create bin directory");

    let stub_path = bin_dir.join("claude");
    let script = r#"#!/usr/bin/env bash
set -euo pipefail

issue_id="${LINEAR_ISSUE_ID:-unknown}"

# Helper: emit a stream-json assistant event (new CLI format)
emit_text() {
  local text="$1"
  printf '{"type":"assistant","message":{"content":[{"type":"text","text":"%s"}]}}\n' "$text"
}

# Helper: emit the final structured result wrapper
emit_result() {
  local json="$1"
  printf '{"type":"result","structured_output":%s}\n' "$json"
}

case "${issue_id}" in
  task-success)
    git config user.email "e2e@example.com"
    git config user.name "E2E Bot"
    git checkout -b fix/${issue_id} 2>/dev/null || true
    echo "// e2e-success-marker" >> src/buggy.rs
    git add src/buggy.rs
    git commit -m "test: apply fix for ${issue_id}" >/dev/null 2>&1 || true
    git push origin HEAD 2>/dev/null || true
    emit_text "PR_URL: https://github.com/test-org/my-repo/pull/42"
    emit_result '{"summary":"Fixed the bug and created PR","success":true,"pr_url":"https://github.com/test-org/my-repo/pull/42"}'
    ;;
  task-no-pr)
    emit_text "Applied investigation for ${issue_id}, no PR raised"
    emit_result '{"summary":"Investigation complete, no PR needed","success":true,"pr_url":null}'
    ;;
  *)
    echo "Unhandled issue id: ${issue_id}" >&2
    exit 7
    ;;
esac
"#;

    fs::write(&stub_path, script).expect("failed to write stub claude script");
    let mut perms = fs::metadata(&stub_path)
        .expect("missing stub script")
        .permissions();
    perms.set_mode(0o755);
    fs::set_permissions(&stub_path, perms).expect("failed to make stub executable");

    bin_dir
}

fn make_linear_task(id: &str, short_id: &str) -> Issue {
    let mut issue = Issue::new(
        id,
        short_id,
        format!("Fix regression in test-org/my-repo at src/buggy.rs ({short_id})"),
        format!("https://example.test/issues/{id}"),
        "linear",
    );
    issue.description = Some(
        "Task details:\n- repo: test-org/my-repo\n- file: src/buggy.rs\n- reproduce and fix"
            .to_string(),
    );
    issue
}

fn build_config(temp_dir: &TempDir) -> Config {
    Config {
        workspace: temp_dir.path().join("work"),
        db_path: temp_dir.path().join("e2e-tracker.db"),
        known_orgs: vec!["test-org".to_string()],
        ask: AskConfig {
            enabled: false,
            ..Default::default()
        },
        processing_delay_ms: 0,
        max_concurrent: 1,
        max_issues_per_cycle: 10,
        agent: claudear::config::AgentConfig {
            timeout_secs: 30,
            ..Default::default()
        },
        retry: RetryConfig {
            base_delay_ms: 0,
            max_delay_ms: 0,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn create_harness(tasks: Vec<Issue>) -> E2eHarness {
    let temp_dir = TempDir::new().expect("failed to create temp dir");
    let (repo_path, remote_path) = create_real_repo(&temp_dir);
    let baseline_file_contents =
        fs::read_to_string(repo_path.join("src/buggy.rs")).expect("missing baseline file");

    let bin_dir = write_claude_stub(&temp_dir);
    let old_path = std::env::var("PATH").unwrap_or_default();
    let _path_guard = EnvVarGuard::set("PATH", format!("{}:{}", bin_dir.display(), old_path));
    let _log_dir_guard = EnvVarGuard::set(
        "CLAUDEAR_LOG_DIR",
        temp_dir.path().join("logs").to_string_lossy().to_string(),
    );

    let mut index = RepoIndex::new();
    let mut repo = IndexedRepo::new("test-org/my-repo", &repo_path);
    repo.scm_url = remote_path.to_string_lossy().to_string();
    repo.default_branch = "main".to_string();
    repo.files = vec!["src/buggy.rs".to_string()];
    index.add_repo(repo);

    let inferrer = RepoInferrer::new(index);
    let tracker = Arc::new(SqliteTracker::in_memory().expect("failed to initialize tracker"));
    let tracker_trait: Arc<dyn FixAttemptTracker> = tracker.clone();

    let source = Arc::new(TaskSource::new("linear", tasks));
    let notifier = Arc::new(RecordingNotifier::default());

    let agent: Arc<dyn claudear::runner::AgentRunner> =
        Arc::new(claudear::runner::ClaudeAgentRunner::new(
            claudear::runner::ClaudeRunnerConfig::default(),
            tracker.clone(),
        ));

    let watcher = Watcher::new(WatcherOptions {
        config: build_config(&temp_dir),
        sources: vec![source.clone() as Arc<dyn IssueSource>],
        notifier: notifier.clone() as Arc<dyn Notifier>,
        tracker: tracker_trait,
        inferrer: Some(inferrer),
        embedding_client: None,
        review_watcher: None,
        issue_embedding_service: None,
        code_search_service: None,
        discord_search_service: None,
        discord_index_orchestrator: None,
        relationships: None,
        github_client: None,
        scm_provider: None,
        user_registry: UserRegistry::new(HashMap::new()),
        agent,
        classification_agent: None,
        repo_classification_agent: None,
        qa_agent: None,
        dry_run: false,
        llm_engine: None,
    });

    E2eHarness {
        _temp_dir: temp_dir,
        _path_guard,
        _log_dir_guard,
        repo_path,
        remote_path,
        baseline_file_contents,
        source,
        notifier,
        tracker,
        watcher,
    }
}

#[tokio::test]
async fn e2e_successful_task_updates_real_repo_and_tracker() {
    let _env_guard = ENV_LOCK.lock().await;
    let harness = create_harness(vec![make_linear_task("task-success", "TASK-1001")]);

    harness
        .watcher
        .trigger_issue("linear", "task-success")
        .await
        .expect("trigger_issue should succeed");

    let attempt = harness
        .tracker
        .get_attempt("linear", "task-success")
        .unwrap()
        .expect("missing attempt");
    assert_eq!(attempt.status, FixAttemptStatus::Success);
    assert_eq!(
        attempt.pr_url.as_deref(),
        Some("https://github.com/test-org/my-repo/pull/42")
    );

    // Worktrees are cleaned up after processing, so check the remote bare repo
    // for the pushed branch instead of the local working tree.
    let remote_file = read_remote_file(&harness.remote_path, "fix/task-success", "src/buggy.rs")
        .expect("fix branch not found in remote repo");
    assert!(remote_file.contains("// e2e-success-marker"));

    let executions = harness
        .tracker
        .get_executions_for_attempt(attempt.id)
        .expect("execution lookup failed");
    assert!(!executions.is_empty());
    assert!(executions
        .first()
        .and_then(|e| e.stdout_preview.as_deref())
        .unwrap_or_default()
        .contains("PR_URL"));

    let pr_created_metrics = harness
        .tracker
        .get_metrics("pr_created", None, 10)
        .expect("metric lookup failed");
    assert_eq!(pr_created_metrics.len(), 1);

    let events = harness.notifier.events();
    assert!(events.iter().any(|e| e == "start:TASK-1001"));
    assert!(events.iter().any(|e| e == "success:TASK-1001"));
}

#[tokio::test]
async fn e2e_multiple_mock_tasks_reset_repo_and_clear_task_state() {
    let _env_guard = ENV_LOCK.lock().await;
    let harness = create_harness(vec![
        make_linear_task("task-success", "TASK-2001"),
        make_linear_task("task-no-pr", "TASK-2002"),
    ]);

    harness
        .watcher
        .trigger_issue("linear", "task-success")
        .await
        .expect("first task should succeed");
    // Worktrees are cleaned up after processing — check the remote bare repo
    let changed_file = read_remote_file(&harness.remote_path, "fix/task-success", "src/buggy.rs")
        .expect("fix branch not found in remote repo");
    assert!(changed_file.contains("// e2e-success-marker"));

    harness
        .watcher
        .trigger_issue("linear", "task-no-pr")
        .await
        .expect("second task should complete pipeline");

    let second_attempt = harness
        .tracker
        .get_attempt("linear", "task-no-pr")
        .unwrap()
        .expect("missing second attempt");
    assert_eq!(second_attempt.status, FixAttemptStatus::Failed);
    assert!(second_attempt
        .error_message
        .as_deref()
        .unwrap_or_default()
        .contains("Claude completed without creating a PR"));

    let file_after_second_task =
        fs::read_to_string(harness.repo_path.join("src/buggy.rs")).unwrap();
    assert_eq!(file_after_second_task, harness.baseline_file_contents);

    harness.source.reset_tasks();
    assert_eq!(harness.source.task_count(), 0);
}
