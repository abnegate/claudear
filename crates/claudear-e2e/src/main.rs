//! claudear-e2e: Production E2E smoke test binary.
//!
//! Runs the same 3 scenarios as `scripts/prod-e2e-smoke.sh` but in Rust,
//! with swappable backends (GitHub/GitLab SCM, Linear/Jira source, Discord/Slack ask).

mod ask;
mod cleanup;
mod config;
mod daemon;
mod db;
mod scenarios;
mod wait;

use anyhow::{bail, Context, Result};
use clap::Parser;
use claudear::scm::ScmProvider;
use claudear::source::IssueSource;
use scenarios::ScenarioContext;
use std::sync::Arc;

/// Production E2E smoke tests for claudear.
#[derive(Parser)]
#[command(
    name = "claudear-e2e",
    about = "E2E smoke tests with swappable backends"
)]
struct Cli {
    /// Comma-separated scenario numbers to run (e.g., "1,2,3").
    #[arg(long, env = "CLAUDEAR_E2E_SCENARIOS", default_value = "1,2,3")]
    scenarios: String,

    /// SCM provider: "github" or "gitlab".
    #[arg(long, env = "CLAUDEAR_E2E_SCM", default_value = "github")]
    scm: String,

    /// Issue source: "linear", "jira", "discord", or "slack".
    #[arg(long, env = "CLAUDEAR_E2E_SOURCE", default_value = "linear")]
    source: String,

    /// Ask/question backend: "discord" or "slack".
    #[arg(long, env = "CLAUDEAR_E2E_ASK_BACKEND", default_value = "discord")]
    ask_backend: String,

    /// Run the claudear daemon inside Docker.
    #[arg(long, env = "CLAUDEAR_E2E_USE_DOCKER")]
    use_docker: bool,

    /// Docker image name (only used with --use-docker).
    #[arg(
        long,
        env = "CLAUDEAR_E2E_DOCKER_IMAGE",
        default_value = "claudear-app:latest"
    )]
    docker_image: String,

    /// Path to a prebuilt claudear binary (skips cargo build).
    #[arg(long, env = "CLAUDEAR_E2E_BINARY")]
    binary: Option<String>,

    /// Wait timeout in seconds for polling operations.
    #[arg(long, env = "CLAUDEAR_E2E_WAIT_TIMEOUT", default_value = "600")]
    wait_timeout: u64,

    /// Claude process execution timeout in seconds.
    #[arg(long, env = "CLAUDEAR_E2E_CLAUDE_TIMEOUT_SECS", default_value = "600")]
    claude_timeout: u64,
}

fn parse_scenario_list(s: &str) -> Vec<u32> {
    s.split(',')
        .filter_map(|part| part.trim().parse::<u32>().ok())
        .collect()
}

fn build_scm(cli: &Cli) -> Result<Arc<dyn ScmProvider>> {
    match cli.scm.as_str() {
        "github" => {
            let token = std::env::var("CLAUDEAR_E2E_GITHUB_TOKEN")
                .context("CLAUDEAR_E2E_GITHUB_TOKEN required")?;
            let config = claudear::config::GitHubConfig {
                token: Some(token.into()),
                auto_resolve_on_merge: true,
                review_trigger: "@claudear".to_string(),
                ..Default::default()
            };
            Ok(Arc::new(claudear::GitHubClient::new(config)))
        }
        "gitlab" => {
            let token = std::env::var("CLAUDEAR_E2E_GITLAB_TOKEN")
                .context("CLAUDEAR_E2E_GITLAB_TOKEN required")?;
            let base_url = std::env::var("CLAUDEAR_E2E_GITLAB_URL")
                .unwrap_or_else(|_| "https://gitlab.com".to_string());
            let config = claudear::config::GitLabConfig {
                enabled: true,
                token: Some(token.into()),
                base_url,
                review_trigger: "@claudear".to_string(),
                ..Default::default()
            };
            Ok(Arc::new(claudear::GitLabClient::new(config)))
        }
        other => bail!("Unknown SCM provider: {}", other),
    }
}

fn build_source(cli: &Cli) -> Result<Arc<dyn IssueSource>> {
    match cli.source.as_str() {
        "linear" => {
            let api_key = std::env::var("CLAUDEAR_E2E_LINEAR_API_KEY")
                .context("CLAUDEAR_E2E_LINEAR_API_KEY required")?;
            let team_id = std::env::var("CLAUDEAR_E2E_LINEAR_TEAM_ID")
                .context("CLAUDEAR_E2E_LINEAR_TEAM_ID required")?;
            let config = claudear::config::LinearConfig {
                enabled: true,
                api_key: api_key.into(),
                team_id: Some(team_id),
                trigger_labels: vec!["claudear-e2e".to_string()],
                trigger_states: vec!["backlog".to_string(), "todo".to_string()],
                ..Default::default()
            };
            Ok(Arc::new(claudear::source::LinearSource::new(config)))
        }
        "jira" => {
            let base_url =
                std::env::var("CLAUDEAR_E2E_JIRA_URL").context("CLAUDEAR_E2E_JIRA_URL required")?;
            let email = std::env::var("CLAUDEAR_E2E_JIRA_EMAIL")
                .context("CLAUDEAR_E2E_JIRA_EMAIL required")?;
            let api_token = std::env::var("CLAUDEAR_E2E_JIRA_API_TOKEN")
                .context("CLAUDEAR_E2E_JIRA_API_TOKEN required")?;
            let project_key = std::env::var("CLAUDEAR_E2E_JIRA_PROJECT_KEY")
                .context("CLAUDEAR_E2E_JIRA_PROJECT_KEY required")?;
            let config = claudear::config::JiraConfig {
                enabled: true,
                base_url,
                email,
                api_token: api_token.into(),
                auth_mode: "basic".to_string(),
                project_keys: vec![project_key],
                trigger_labels: vec!["claudear-e2e".to_string()],
                trigger_statuses: vec!["To Do".to_string(), "Backlog".to_string()],
                ..Default::default()
            };
            Ok(Arc::new(claudear::source::JiraSource::new(config)))
        }
        "discord" => {
            let bot_token = std::env::var("CLAUDEAR_E2E_DISCORD_BOT_TOKEN")
                .context("CLAUDEAR_E2E_DISCORD_BOT_TOKEN required")?;
            let channel_id = std::env::var("CLAUDEAR_E2E_DISCORD_CHANNEL_ID")
                .context("CLAUDEAR_E2E_DISCORD_CHANNEL_ID required")?;
            // Webhook URL lets create_issue post as a non-bot (bypasses is_bot_message filter)
            let webhook_url = std::env::var("CLAUDEAR_E2E_DISCORD_WEBHOOK_URL")
                .ok()
                .filter(|s| !s.is_empty())
                .map(claudear::secret::SecretValue::new);
            let config = claudear::config::DiscordConfig {
                bot_token: Some(bot_token.into()),
                channel_id: Some(channel_id.clone()),
                source_enabled: true,
                listen_channel_id: Some(channel_id),
                webhook_url,
                ..Default::default()
            };
            Ok(Arc::new(claudear::source::DiscordSource::new(config)))
        }
        "slack" => {
            let bot_token = std::env::var("CLAUDEAR_E2E_SLACK_BOT_TOKEN")
                .context("CLAUDEAR_E2E_SLACK_BOT_TOKEN required")?;
            let channel_id = std::env::var("CLAUDEAR_E2E_SLACK_CHANNEL_ID")
                .context("CLAUDEAR_E2E_SLACK_CHANNEL_ID required")?;
            // Webhook URL lets create_issue post as a non-bot (bypasses is_bot_message filter)
            let webhook_url = std::env::var("CLAUDEAR_E2E_SLACK_WEBHOOK_URL")
                .ok()
                .filter(|s| !s.is_empty())
                .map(claudear::secret::SecretValue::new);
            let config = claudear::config::SlackConfig {
                bot_token: Some(bot_token.into()),
                channel_id: Some(channel_id.clone()),
                source_enabled: true,
                listen_channel_id: Some(channel_id),
                webhook_url,
                ..Default::default()
            };
            Ok(Arc::new(claudear::source::SlackSource::new(config)))
        }
        other => bail!("Unknown issue source: {}", other),
    }
}

fn build_ask(cli: &Cli) -> Result<Option<Box<dyn ask::E2eAsk>>> {
    match cli.ask_backend.as_str() {
        "discord" => {
            let bot_token = match std::env::var("CLAUDEAR_E2E_DISCORD_BOT_TOKEN") {
                Ok(t) => t,
                Err(_) => return Ok(None),
            };
            let channel_id = match std::env::var("CLAUDEAR_E2E_DISCORD_CHANNEL_ID") {
                Ok(c) => c,
                Err(_) => return Ok(None),
            };
            Ok(Some(Box::new(ask::DiscordAsk::new(bot_token, channel_id)?)))
        }
        "slack" => {
            let bot_token = match std::env::var("CLAUDEAR_E2E_SLACK_BOT_TOKEN") {
                Ok(t) => t,
                Err(_) => return Ok(None),
            };
            let channel_id = match std::env::var("CLAUDEAR_E2E_SLACK_CHANNEL_ID") {
                Ok(c) => c,
                Err(_) => return Ok(None),
            };
            Ok(Some(Box::new(ask::SlackAsk::new(bot_token, channel_id))))
        }
        other => bail!("Unknown ask backend: {}", other),
    }
}

/// Primary repo (all scenarios). Respects the selected SCM backend.
fn repo_name(scm: &str) -> Result<String> {
    match scm {
        "gitlab" => std::env::var("CLAUDEAR_E2E_GITLAB_REPO")
            .or_else(|_| std::env::var("CLAUDEAR_E2E_GITHUB_REPO"))
            .context("CLAUDEAR_E2E_GITLAB_REPO or CLAUDEAR_E2E_GITHUB_REPO required"),
        _ => std::env::var("CLAUDEAR_E2E_GITHUB_REPO")
            .or_else(|_| std::env::var("CLAUDEAR_E2E_GITLAB_REPO"))
            .context("CLAUDEAR_E2E_GITHUB_REPO or CLAUDEAR_E2E_GITLAB_REPO required"),
    }
}

/// Secondary repo (cascade scenario 3). Respects the selected SCM backend.
fn repo_name_2(scm: &str) -> Option<String> {
    match scm {
        "gitlab" => std::env::var("CLAUDEAR_E2E_GITLAB_REPO_2")
            .or_else(|_| std::env::var("CLAUDEAR_E2E_GITHUB_REPO_2"))
            .ok(),
        _ => std::env::var("CLAUDEAR_E2E_GITHUB_REPO_2")
            .or_else(|_| std::env::var("CLAUDEAR_E2E_GITLAB_REPO_2"))
            .ok(),
    }
}

/// Reviewer token for posting reviews on PRs. Respects the selected SCM backend.
fn reviewer_token(scm: &str) -> Option<String> {
    match scm {
        "gitlab" => std::env::var("CLAUDEAR_E2E_GITLAB_REVIEWER_TOKEN")
            .or_else(|_| std::env::var("CLAUDEAR_E2E_GITHUB_REVIEWER_TOKEN"))
            .ok(),
        _ => std::env::var("CLAUDEAR_E2E_GITHUB_REVIEWER_TOKEN")
            .or_else(|_| std::env::var("CLAUDEAR_E2E_GITLAB_REVIEWER_TOKEN"))
            .ok(),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();
    let scenario_nums = parse_scenario_list(&cli.scenarios);

    if scenario_nums.is_empty() {
        bail!("No scenarios specified");
    }

    log("=== claudear E2E smoke test (Rust) ===");
    log(&format!(
        "Backends: scm={}, source={}, ask={}",
        cli.scm, cli.source, cli.ask_backend
    ));
    log(&format!("Scenarios: {:?}", scenario_nums));

    let scm = build_scm(&cli)?;
    let source = build_source(&cli)?;
    let ask = build_ask(&cli)?;
    let repo = repo_name(&cli.scm)?;
    let repo2 = repo_name_2(&cli.scm);
    let reviewer = reviewer_token(&cli.scm);

    let mut results: Vec<(&str, Result<()>)> = Vec::new();

    for num in &scenario_nums {
        let ctx = ScenarioContext {
            scm: scm.clone(),
            source: source.clone(),
            ask_backend: &ask,
            repo: &repo,
            repo2: repo2.as_deref(),
            reviewer_token: reviewer.as_deref(),
            use_docker: cli.use_docker,
            docker_image: &cli.docker_image,
            binary: cli.binary.as_deref(),
            wait_timeout: cli.wait_timeout,
            claude_timeout: cli.claude_timeout,
            scm_name: &cli.scm,
            source_name: &cli.source,
            ask_name: &cli.ask_backend,
        };

        match num {
            1 => {
                log(">>> Starting Scenario 1: Full Lifecycle");
                let result = scenarios::s1_lifecycle::run(&ctx).await;
                log_result("S1", &result);
                results.push(("S1", result));
            }
            2 => {
                if ask.is_none() {
                    log(">>> Skipping Scenario 2: ask backend not configured");
                    continue;
                }
                log(">>> Starting Scenario 2: Ask + Regression");
                let result = scenarios::s2_ask::run(&ctx).await;
                log_result("S2", &result);
                results.push(("S2", result));
            }
            3 => {
                if repo2.is_none() {
                    log(">>> Skipping Scenario 3: second repo not configured");
                    continue;
                }
                log(">>> Starting Scenario 3: Cascade");
                let result = scenarios::s3_cascade::run(&ctx).await;
                log_result("S3", &result);
                results.push(("S3", result));
            }
            4 => {
                // S4 seeds issues via create_issue, so it needs a label/poll
                // source (Linear/Jira), not a cursor-based chat source.
                if matches!(cli.source.as_str(), "discord" | "slack") {
                    log(">>> Skipping Scenario 4: requires a seedable label source (linear/jira)");
                    continue;
                }
                log(">>> Starting Scenario 4: Action pipeline (classify → verify → resolve → reply)");
                let result = scenarios::s4_action::run(&ctx).await;
                log_result("S4", &result);
                results.push(("S4", result));
            }
            other => {
                log(&format!(">>> Unknown scenario {}, skipping", other));
            }
        }
    }

    // Summary
    log("\n=== E2E Results ===");
    let mut all_pass = true;
    for (name, result) in &results {
        let status = if result.is_ok() { "PASS" } else { "FAIL" };
        log(&format!("  {}: {}", name, status));
        if result.is_err() {
            all_pass = false;
        }
    }

    if all_pass {
        log("All scenarios passed!");
        Ok(())
    } else {
        bail!("Some scenarios failed")
    }
}

fn log(msg: &str) {
    eprintln!("[claudear-e2e] {}", msg);
}

fn log_result(name: &str, result: &Result<()>) {
    match result {
        Ok(()) => log(&format!("<<< {} PASSED", name)),
        Err(e) => log(&format!("<<< {} FAILED: {:?}", name, e)),
    }
}
