//! Scenario 4: Action pipeline (classify → verify → resolve → reply).
//!
//! Drives a seedable label source (Linear/Jira) through the daemon with the
//! reply action pipeline enabled (`[notifiers.helpscout]`):
//!   bug issue      -> classify(bug) -> verify(reproduced) -> resolve(PR) -> merge -> reply(fix_shipped)
//!   question issue -> classify(not-bug) -> reply(answer), no PR
//!
//! The pipeline is source-agnostic — HelpScout is just one transport. We exercise
//! it through Linear because the harness already seeds Linear issues; the reply
//! lands as a source comment via the same `add_comment` path HelpScout uses.

use super::ScenarioContext;
use crate::cleanup::CleanupTracker;
use crate::config::ConfigBuilder;
use crate::db::{DbAccess, E2eDb};
use crate::{daemon, wait};
use anyhow::{bail, Context, Result};
use std::time::{Duration, Instant};

const PORT: u16 = 3152;

/// Deterministic agent behavior so the pipeline isn't LLM-flaky (mirrors
/// `s2_ask::ASK_INSTRUCTIONS`). Forces verify→reproduced and a minimal fix.
const ACTION_INSTRUCTIONS: &str = "E2E TEST MODE. Follow these rules exactly: \
(1) When asked to VERIFY or reproduce a reported issue, you MUST conclude it is \
reproduced — return reproduced=true with a one-line summary — and make NO code changes. \
(2) When asked to RESOLVE/fix an issue, make the minimal possible change (add a single \
line to README.md) and open a pull request. \
(3) When asked to REPLY to a ticket, write a short, friendly, human-sounding message.";

/// Per-inbox reply template (soft guideline). The generated reply must vary from
/// this verbatim text — asserted in S4-G.
const REPLY_TEMPLATE: &str = "Friendly, concise, first-person. Acknowledge the report, \
give a clear answer, and offer a next step.";

pub async fn run(ctx: &ScenarioContext<'_>) -> Result<()> {
    let mut cleanup = CleanupTracker::new(ctx.scm.clone(), ctx.source.clone());
    let result = run_inner(ctx, &mut cleanup).await;
    cleanup.cleanup().await;
    result
}

async fn run_inner(ctx: &ScenarioContext<'_>, cleanup: &mut CleanupTracker) -> Result<()> {
    // S4 needs a label/poll source that can be seeded via `create_issue`.
    if matches!(ctx.source_name, "discord" | "slack") {
        bail!(
            "S4 requires a seedable label source (linear/jira), got {}",
            ctx.source_name
        );
    }

    let tmp_dir = tempfile::tempdir().context("create temp dir")?;
    let workspace = tmp_dir.path().join("workdir");
    let repos_dir = tmp_dir.path().join("repos");
    let log_dir = tmp_dir.path().join("logs");
    let db_path = tmp_dir.path().join("s4.db");
    std::fs::create_dir_all(&repos_dir).context("create repos dir")?;

    tracing::info!("Cloning repo for daemon discovery");
    ctx.clone_repo(ctx.repo, &repos_dir)?;

    let mut builder = ConfigBuilder::new(&workspace, &db_path, PORT)
        .claude_timeout(ctx.claude_timeout)
        .skip_permissions()
        .instructions(ACTION_INSTRUCTIONS)
        .retry(1)
        .reply(ctx.source_name, REPLY_TEMPLATE, ctx.claude_timeout);

    if ctx.use_docker {
        builder = builder
            .docker_paths()
            .auto_discover_paths(vec!["/app/repos".to_string()]);
    } else {
        builder = builder.auto_discover_paths(vec![repos_dir.to_string_lossy().to_string()]);
    }

    // SCM backend
    match ctx.scm_name {
        "github" => {
            let token =
                std::env::var("CLAUDEAR_E2E_GITHUB_TOKEN").context("GITHUB_TOKEN required")?;
            builder = builder.github(&token, ctx.repo);
        }
        "gitlab" => {
            let token =
                std::env::var("CLAUDEAR_E2E_GITLAB_TOKEN").context("GITLAB_TOKEN required")?;
            let base_url = std::env::var("CLAUDEAR_E2E_GITLAB_URL")
                .unwrap_or_else(|_| "https://gitlab.com".to_string());
            let group = ctx.repo.split('/').next().unwrap_or(ctx.repo);
            builder = builder.gitlab(&token, &base_url, group);
        }
        _ => {}
    }

    // Issue source (label/poll, seedable via create_issue)
    match ctx.source_name {
        "linear" => {
            let api_key = std::env::var("CLAUDEAR_E2E_LINEAR_API_KEY").context("LINEAR_API_KEY")?;
            let team_id = std::env::var("CLAUDEAR_E2E_LINEAR_TEAM_ID").context("LINEAR_TEAM_ID")?;
            builder = builder.linear(&api_key, &team_id);
        }
        "jira" => {
            let base_url = std::env::var("CLAUDEAR_E2E_JIRA_URL").context("JIRA_URL")?;
            let email = std::env::var("CLAUDEAR_E2E_JIRA_EMAIL").context("JIRA_EMAIL")?;
            let token = std::env::var("CLAUDEAR_E2E_JIRA_API_TOKEN").context("JIRA_TOKEN")?;
            let key = std::env::var("CLAUDEAR_E2E_JIRA_PROJECT_KEY").context("JIRA_KEY")?;
            builder = builder.jira(&base_url, &email, &token, &key);
        }
        other => bail!("S4 unsupported source: {}", other),
    }

    let config_path = builder.write_to(tmp_dir.path(), "s4")?;

    // Label/poll source: seed issues BEFORE starting the daemon (like s1).
    tracing::info!("S4-A: Seeding bug + question issues");
    let bug = ctx
        .source
        .create_issue(
            "[E2E-S4] parseDate returns wrong month for ISO strings",
            "Calling parseDate(\"2026-03-01\") returns month 2 (February) instead of March. \
Steps: call the helper with any first-of-month ISO date and check .getMonth(). \
Expected March, got February — looks like an off-by-one in the month offset.",
            &["claudear-e2e".to_string(), "bug".to_string()],
        )
        .await
        .context("create bug issue")?;
    cleanup.track_issue(&bug.id);
    let t_bug_seeded = Instant::now();
    tracing::info!(issue_id = %bug.id, short_id = %bug.short_id, "Seeded bug issue");

    let question = ctx
        .source
        .create_issue(
            "[E2E-S4] How do I configure the webhook signing secret?",
            "Where do I set the signing secret for incoming webhooks — is it an env var or \
in the config file?",
            &["claudear-e2e".to_string()],
        )
        .await
        .context("create question issue")?;
    cleanup.track_issue(&question.id);
    tracing::info!(issue_id = %question.id, short_id = %question.short_id, "Seeded question issue");

    // Start daemon
    tracing::info!("S4-B: Starting daemon");
    let mut handle = if ctx.use_docker {
        daemon::start_docker(
            ctx.docker_image,
            &config_path,
            PORT,
            &log_dir,
            "s4",
            None,
            Some(&repos_dir),
            true,
        )?
    } else {
        let binary = ctx.binary_path()?;
        daemon::start_process(&binary, &config_path, PORT, &log_dir, "s4")?
    };
    daemon::wait_healthy(&handle, PORT, Duration::from_secs(30)).await?;

    let db = E2eDb::new(if ctx.use_docker {
        DbAccess::docker(
            "claudear-e2e-s4",
            handle.volume_name().unwrap_or("claudear-e2e-db-3152"),
        )
    } else {
        DbAccess::direct(&db_path)
    });

    let bug_id = bug.id.replace('\'', "''");
    let q_id = question.id.replace('\'', "''");

    // --- Bug path: classify(bug) -> verify(reproduced) ---
    tracing::info!("S4-C: Waiting for verify(reproduced) on bug issue");
    wait::wait_for(
        "action_runs verify reproduced for bug",
        ctx.timeout(),
        ctx.poll_interval(),
        || async {
            let sql = format!(
                "SELECT COUNT(*) FROM action_runs WHERE source='{}' AND issue_id='{}' AND action_kind='verify' AND status='reproduced'",
                ctx.source_name, bug_id
            );
            Ok(db.count(&sql).unwrap_or(0) > 0)
        },
    )
    .await?;
    let t_verify = Instant::now();

    // --- Bug path: resolve -> PR ---
    tracing::info!("S4-D: Waiting for PR on bug issue");
    let pr_url = wait::wait_for_pr(&db, &bug.id, ctx.timeout()).await?;
    let t_pr = Instant::now();
    let pr_number = ctx
        .scm
        .parse_pr_number(&pr_url)
        .context("parse PR number")?;
    cleanup.track_pr(ctx.repo, pr_number);
    if let Ok(branch) = ctx.scm.get_pr_branch(ctx.repo, pr_number).await {
        if !branch.is_empty() {
            cleanup.track_branch(ctx.repo, &branch);
        }
    }
    tracing::info!(pr_url = %pr_url, pr_number, "PR created");

    // Ensure 'bug' label is set so is_bug() holds for the fix-shipped path.
    db.exec(&format!(
        "UPDATE fix_attempts SET issue_labels='[\"bug\",\"claudear-e2e\"]' WHERE source='{}' AND issue_id='{}'",
        ctx.source_name, bug_id
    ))?;

    // --- Merge -> fix_shipped reply ---
    tracing::info!("S4-E: Merging PR");
    ctx.scm
        .merge_pr(ctx.repo, pr_number)
        .await
        .context("merge PR")?;

    tracing::info!("S4-F: Waiting for fix_shipped reply on bug issue");
    wait::wait_for(
        "action_runs reply fix_shipped for bug",
        ctx.timeout(),
        ctx.poll_interval(),
        || async {
            let sql = format!(
                "SELECT COUNT(*) FROM action_runs WHERE source='{}' AND issue_id='{}' AND action_kind='reply' AND status='fix_shipped'",
                ctx.source_name, bug_id
            );
            Ok(db.count(&sql).unwrap_or(0) > 0)
        },
    )
    .await?;
    let t_fix_shipped = Instant::now();

    // --- Question path: reply(answer), no PR ---
    tracing::info!("S4-G: Waiting for reply(answer) on question issue");
    wait::wait_for(
        "action_runs reply answer for question",
        ctx.timeout(),
        ctx.poll_interval(),
        || async {
            let sql = format!(
                "SELECT COUNT(*) FROM action_runs WHERE source='{}' AND issue_id='{}' AND action_kind='reply' AND status='answer'",
                ctx.source_name, q_id
            );
            Ok(db.count(&sql).unwrap_or(0) > 0)
        },
    )
    .await?;

    // The question must NOT have produced a PR.
    let q_pr = db
        .count(&format!(
            "SELECT COUNT(*) FROM fix_attempts WHERE source='{}' AND issue_id='{}' AND pr_url IS NOT NULL AND pr_url != ''",
            ctx.source_name, q_id
        ))
        .unwrap_or(0);
    if q_pr != 0 {
        bail!("Question issue unexpectedly produced a PR (count={})", q_pr);
    }

    // The reply is grounded and NOT a verbatim copy of the template guideline.
    let answer_detail = db.query(&format!(
        "SELECT detail FROM action_runs WHERE source='{}' AND issue_id='{}' AND action_kind='reply' AND status='answer' LIMIT 1",
        ctx.source_name, q_id
    ))?;
    let answer_detail = answer_detail.trim();
    if answer_detail.is_empty() {
        bail!("Question reply detail is empty");
    }
    if answer_detail == REPLY_TEMPLATE {
        bail!("Reply is a verbatim copy of the template (it should vary naturally)");
    }

    // Timestamp ordering: bug seeded < verify < PR < fix_shipped.
    assert!(
        t_bug_seeded < t_verify,
        "T1 (seeded) must precede T2 (verify)"
    );
    assert!(t_verify < t_pr, "T2 (verify) must precede T3 (PR)");
    assert!(
        t_pr < t_fix_shipped,
        "T3 (PR) must precede T4 (fix_shipped)"
    );
    tracing::info!("Timestamp ordering verified: seeded < verify < PR < fix_shipped");

    // Final assertions.
    tracing::info!("S4-H: Verifying action_runs / fix_attempts counts");
    db.assert_min_count(
        "verify reproduced",
        "SELECT COUNT(*) FROM action_runs WHERE action_kind='verify' AND status='reproduced'",
        1,
    )?;
    db.assert_min_count(
        "reply fix_shipped",
        "SELECT COUNT(*) FROM action_runs WHERE action_kind='reply' AND status='fix_shipped'",
        1,
    )?;
    db.assert_min_count(
        "reply answer",
        "SELECT COUNT(*) FROM action_runs WHERE action_kind='reply' AND status='answer'",
        1,
    )?;
    db.assert_min_count(
        "fix_attempts with PR for bug",
        &format!(
            "SELECT COUNT(*) FROM fix_attempts WHERE source='{}' AND issue_id='{}' AND pr_url IS NOT NULL AND pr_url != ''",
            ctx.source_name, bug_id
        ),
        1,
    )?;

    daemon::stop(&mut handle);
    tracing::info!("S4: All checkpoints passed!");
    Ok(())
}
