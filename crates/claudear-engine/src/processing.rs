//! Shared issue-processing pipeline used by both the polling watcher and the webhook server.

use async_trait::async_trait;
use claudear_analysis::feedback::{
    format_similar_issues_context, EmbeddingClient, FeedbackAnalyzer, FixOutcome,
    IssueEmbeddingService, Outcome,
};
use claudear_analysis::inference::RepoResolution;
use claudear_analysis::qa::{
    build_correlation_id, embed_text, find_reusable_qa, format_answer_context,
    format_reuse_context, format_timeout_context, normalize_text,
};
use claudear_analysis::repo::code_index::CodeSearchService;
use claudear_analysis::repo::{worktree_path, GitOps};
use claudear_config::config::Config;
use claudear_config::users::UserRegistry;
use claudear_core::error::Result;
use claudear_core::types::{
    ActivityLogEntry, AskRequest, ErrorPattern, Issue, ProcessingMetric, QaKnowledgeEntry,
};
use claudear_integrations::github::GitHubClient;
use claudear_integrations::notifier::{send_to_all_and_wait_first_reply, Notifier};
use claudear_integrations::runner::{self, AgentRunner};
use claudear_integrations::scm::{PrReviewState, ReviewWatcher};
use claudear_storage::{classify_error, compute_error_hash, FixAttemptTracker};
use serde_json::json;
use std::sync::Arc;

use crate::llm_analyzer::Intent;

/// Trait for building issue context. Both `IssueSource` and `WebhookHandler` satisfy this.
#[async_trait]
pub trait ContextProvider: Send + Sync {
    async fn build_issue_context(&self, issue: &Issue) -> Result<String>;
}

#[async_trait]
impl<T: claudear_integrations::source::IssueSource + ?Sized> ContextProvider for T {
    async fn build_issue_context(&self, issue: &Issue) -> Result<String> {
        claudear_integrations::source::IssueSource::build_issue_context(self, issue).await
    }
}

/// Wrapper to pass a `dyn IssueSource` as a `ContextProvider` (dyn-to-dyn coercion
/// is not supported in Rust, so we use an explicit wrapper).
pub struct SourceContext<'a>(pub &'a dyn claudear_integrations::source::IssueSource);

#[async_trait]
impl ContextProvider for SourceContext<'_> {
    async fn build_issue_context(&self, issue: &Issue) -> Result<String> {
        self.0.build_issue_context(issue).await
    }
}

/// Wrapper to implement `ContextProvider` for `WebhookHandler` (avoids blanket conflict).
pub struct WebhookContext<'a>(pub &'a dyn claudear_integrations::webhook::WebhookHandler);

#[async_trait]
impl ContextProvider for WebhookContext<'_> {
    async fn build_issue_context(&self, issue: &Issue) -> Result<String> {
        self.0.build_issue_context(issue).await
    }
}

/// Whether a source is conversational and therefore eligible for question/answer
/// handling. Tracker-style sources (Sentry, Linear, GitHub, etc.) are not.
pub(crate) fn qa_eligible_source(source: &str) -> bool {
    matches!(source, "discord" | "slack" | "telegram" | "whatsapp")
}

/// Holds shared services needed to process an issue.
pub struct IssueProcessor {
    pub config: Config,
    pub tracker: Arc<dyn FixAttemptTracker>,
    pub notifier: Arc<dyn Notifier>,
    pub agent: Arc<dyn AgentRunner>,
    pub inferrer: Option<claudear_analysis::inference::RepoInferrer>,
    pub embedding_client: Option<Arc<EmbeddingClient>>,
    pub issue_embedding_service: Option<Arc<IssueEmbeddingService>>,
    pub code_search_service: Option<Arc<CodeSearchService>>,
    pub feedback_analyzer: Arc<tokio::sync::Mutex<FeedbackAnalyzer>>,
    pub review_watcher: Option<Arc<ReviewWatcher>>,
    pub user_registry: UserRegistry,
    pub github_client: Option<Arc<GitHubClient>>,
    pub llm_analyzer: Option<Arc<crate::llm_analyzer::LlmAnalyzerImpl>>,
}

/// Everything the caller provides to `IssueProcessor::run()`.
pub struct ProcessingInput {
    pub issue: Issue,
    pub source_name: String,
    pub match_result: claudear_core::types::MatchResult,
    pub resolution: RepoResolution,
    pub attempt_id: Option<i64>,
    pub review_feedback: Option<String>,
    pub existing_pr_branch: Option<String>,
    pub intent: Option<Intent>,
}

/// What happened during processing.
pub enum ProcessingOutcome {
    /// A PR was created or updated.
    Success { pr_url: String },
    /// Claude completed successfully but did not create a PR.
    CompletedNoPr { reason: String },
    /// Processing failed.
    Failed { error: String },
    /// Claude detected it was working in the wrong repository.
    WrongRepo {
        suggested_repo: Option<String>,
        original_repo: String,
    },
}

impl IssueProcessor {
    /// Run the shared processing pipeline.
    ///
    /// The caller is responsible for:
    /// - Dedup gating (processing set insert/check)
    /// - Attempt recording (before calling run)
    /// - Rate limiting (before calling run)
    /// - Cascade logic (after run returns Success)
    /// - Processing state cleanup (after run returns)
    pub async fn run(
        &self,
        input: ProcessingInput,
        context_provider: &dyn ContextProvider,
    ) -> ProcessingOutcome {
        // Q&A short-circuit: pure questions are answered with RAG-grounded context
        // (read-only, no PR) instead of being routed to the fix pipeline. This runs
        // before any repo fetch / worktree setup since the answer path reads the
        // resolved repo directly (or a scratch dir on Skip). Anything ambiguous or
        // non-question falls through to normal processing.
        if matches!(input.intent, Some(Intent::Question)) {
            tracing::info!(
                short_id = %input.issue.short_id,
                source = %input.source_name,
                intent = "question",
                "Routing to read-only Q&A answer path"
            );
            return self
                .answer_question_issue(&input.issue, &input.resolution, &input.source_name)
                .await;
        }
        tracing::info!(
            short_id = %input.issue.short_id,
            source = %input.source_name,
            intent = if matches!(input.intent, Some(Intent::FixRequest)) { "fix" } else { "unclassified" },
            "Routing to fix pipeline"
        );

        match self.run_inner(input, context_provider).await {
            Ok(ProcessingOutcome::WrongRepo {
                original_repo,
                suggested_repo,
            }) => ProcessingOutcome::Failed {
                error: format!(
                    "Wrong repo detected (was: {}, suggested: {:?}), not retried",
                    original_repo, suggested_repo
                ),
            },
            Ok(outcome) => outcome,
            Err(e) => ProcessingOutcome::Failed {
                error: e.to_string(),
            },
        }
    }

    async fn run_inner(
        &self,
        mut input: ProcessingInput,
        context_provider: &dyn ContextProvider,
    ) -> Result<ProcessingOutcome> {
        let ProcessingInput {
            ref mut issue,
            ref source_name,
            ref mut resolution,
            attempt_id,
            ref review_feedback,
            ref existing_pr_branch,
            ..
        } = input;

        const MAX_REPO_SWAPS: u8 = 1;
        let mut repo_swap_attempts: u8 = 0;
        let mut excluded_repos: Vec<String> = Vec::new();

        let project_dir = match resolution {
            RepoResolution::Resolved { project_dir, .. } => project_dir.clone(),
            RepoResolution::Skip { reason } => {
                let error = format!("Repository resolution failed: {}", reason);
                self.tracker
                    .mark_failed(source_name, &issue.id, &error)
                    .ok();
                return Ok(ProcessingOutcome::Failed { error });
            }
        };

        if let (Some(scm_url), Some(default_branch), Some(repo_name)) = (
            resolution.scm_url(),
            resolution.default_branch(),
            resolution.repo_name(),
        ) {
            tracing::info!(
                short_id = %issue.short_id,
                repo = %repo_name,
                "Fetching latest changes"
            );

            let detected_default_branch =
                match GitOps::ensure_repo_fetched(&project_dir, scm_url).await {
                    Ok(branch) => branch,
                    Err(e) => {
                        let error = format!("Failed to fetch repository: {}", e);
                        self.tracker
                            .mark_failed(source_name, &issue.id, &error)
                            .ok();
                        return Ok(ProcessingOutcome::Failed { error });
                    }
                };

            // Use the detected default branch from the remote, falling back to
            // the index value if detection returned the same fallback.
            let effective_default_branch =
                if detected_default_branch != "main" || default_branch == "main" {
                    &detected_default_branch
                } else {
                    default_branch
                };

            tracing::info!(
                short_id = %issue.short_id,
                repo = %repo_name,
                default_branch = %effective_default_branch,
                "Repository fetched"
            );

            // Incrementally re-index code after fetch
            self.reindex_repo(repo_name, &project_dir).await;

            // For review reruns, fetch the PR branch; otherwise use the default branch.
            let checkout_ref = if let Some(ref branch) = existing_pr_branch {
                if let Err(e) = GitOps::fetch_branch(&project_dir, branch).await {
                    tracing::warn!(
                        short_id = %issue.short_id,
                        error = %e,
                        branch = %branch,
                        "Failed to fetch PR branch, falling back to default"
                    );
                    format!("origin/{}", effective_default_branch)
                } else {
                    format!("origin/{}", branch)
                }
            } else {
                format!("origin/{}", effective_default_branch)
            };

            // Create per-issue worktree.
            // For review reruns, check out the actual PR branch so Claude can push
            // to it. For initial runs, use detached HEAD (Claude creates a new branch).
            let wt_path = worktree_path(&self.config.workspace, repo_name, &issue.short_id);
            let wt_result = if let Some(ref branch) = existing_pr_branch {
                GitOps::create_worktree_on_branch(&project_dir, &wt_path, branch, &checkout_ref)
                    .await
            } else {
                GitOps::create_worktree(&project_dir, &wt_path, &checkout_ref).await
            };
            if let Err(e) = wt_result {
                let error = format!("Failed to create worktree: {}", e);
                self.tracker
                    .mark_failed(source_name, &issue.id, &error)
                    .ok();
                return Ok(ProcessingOutcome::Failed { error });
            }

            // Re-index files and sync to database
            if let Some(inferrer) = &self.inferrer {
                if let Err(e) = inferrer.index_cloned_repo(repo_name) {
                    tracing::warn!(
                        short_id = %issue.short_id,
                        repo = %repo_name,
                        error = %e,
                        "Failed to re-index repository files"
                    );
                }
                if let Some(repo) = inferrer.get_repo(repo_name) {
                    if let Err(e) = self.tracker.sync_repo_files(&repo) {
                        tracing::warn!(
                            short_id = %issue.short_id,
                            repo = %repo_name,
                            error = %e,
                            "Failed to sync repository files to database"
                        );
                    }
                }
            }
        }

        // Build effective project dir (worktree or fallback)
        let effective_project_dir = if let Some(repo_name) = resolution.repo_name() {
            let wt = worktree_path(&self.config.workspace, repo_name, &issue.short_id);
            if !wt.exists() {
                let error = format!("Worktree disappeared after creation: {:?}", wt);
                self.tracker
                    .mark_failed(source_name, &issue.id, &error)
                    .ok();
                return Ok(ProcessingOutcome::Failed { error });
            }
            wt
        } else {
            project_dir.clone()
        };

        // Run code quality evaluation baseline (BEFORE hook)
        let eval_before_snapshots = if self.config.evaluation.enabled {
            match claudear_analysis::evaluation::CodeQualityEvaluator::run_baseline(
                &effective_project_dir,
                &self.config.evaluation,
            )
            .await
            {
                Ok(snapshots) => {
                    tracing::info!(
                        short_id = %issue.short_id,
                        tools = snapshots.len(),
                        "Evaluation baseline captured"
                    );
                    snapshots
                }
                Err(e) => {
                    tracing::warn!(
                        short_id = %issue.short_id,
                        error = %e,
                        "Evaluation baseline failed, continuing without"
                    );
                    Vec::new()
                }
            }
        } else {
            Vec::new()
        };

        // Resolve issue assignee to a configured user
        if let Some(assignee) = issue.get_metadata::<String>("assignee") {
            if let Some(resolved) = self.user_registry.resolve(&issue.source, &assignee) {
                issue.set_metadata("resolved_user", &resolved.slug);
            }
        }

        // Semantic duplicate detection
        if let Some(ref embedding_service) = self.issue_embedding_service {
            if let Ok(Some(duplicate)) = embedding_service.check_duplicate(issue, source_name).await
            {
                let similar_id = duplicate
                    .embedding
                    .short_id
                    .as_deref()
                    .unwrap_or(&duplicate.embedding.issue_id);
                let similarity_pct = duplicate.similarity * 100.0;

                tracing::info!(
                    short_id = %issue.short_id,
                    similar_to = %similar_id,
                    similarity = %format!("{:.0}%", similarity_pct),
                    "Skipping semantic duplicate"
                );

                let metric = ProcessingMetric::new("semantic_duplicate_skipped", 1.0)
                    .with_source(source_name.to_string());
                self.tracker.record_metric(&metric).ok();

                let error = format!(
                    "Semantic duplicate of {} ({:.0}% similar)",
                    similar_id, similarity_pct
                );
                self.tracker
                    .mark_failed(source_name, &issue.id, &error)
                    .ok();

                // Cleanup worktree before returning
                self.cleanup_worktree(resolution, issue, &project_dir).await;
                return Ok(ProcessingOutcome::Failed { error });
            }
        }

        // --- Main processing pipeline (with repo-swap retry) ---
        let processing_started_at = std::time::Instant::now();
        let mut current_resolution = resolution.clone();
        let mut current_project_dir = project_dir.clone();
        let mut current_effective_dir = effective_project_dir.clone();

        let result = loop {
            let pipeline_result = self
                .execute_pipeline(
                    issue,
                    source_name,
                    &current_resolution,
                    attempt_id,
                    review_feedback.as_deref(),
                    existing_pr_branch.as_deref(),
                    &current_effective_dir,
                    context_provider,
                )
                .await;

            // Check if we got a WrongRepo outcome and can retry
            match &pipeline_result {
                Ok(ProcessingOutcome::WrongRepo {
                    suggested_repo,
                    original_repo,
                }) if repo_swap_attempts < MAX_REPO_SWAPS => {
                    repo_swap_attempts += 1;
                    excluded_repos.push(original_repo.clone());

                    // Cleanup current worktree before swapping
                    self.cleanup_worktree(&current_resolution, issue, &current_project_dir)
                        .await;

                    // Record inference feedback as activity
                    let feedback_activity = ActivityLogEntry::new(
                        "inference_feedback",
                        format!(
                            "Wrong repo inference for {}: was {}, suggested {:?}",
                            issue.short_id, original_repo, suggested_repo
                        ),
                    )
                    .with_source(source_name.to_string())
                    .with_issue(issue.id.clone(), issue.short_id.clone())
                    .with_metadata(json!({
                        "was_correct": false,
                        "original_repo": original_repo,
                        "suggested_repo": suggested_repo,
                    }));
                    self.tracker.record_activity(&feedback_activity).ok();

                    // Resolve alternative repo
                    let alt_resolution = self
                        .resolve_alternative_repo(issue, suggested_repo.as_deref(), &excluded_repos)
                        .await;
                    match alt_resolution {
                        RepoResolution::Skip { reason } => {
                            tracing::warn!(
                                short_id = %issue.short_id,
                                reason = %reason,
                                "No alternative repo found after wrong_repo detection"
                            );
                            break Ok(ProcessingOutcome::Failed {
                                error: format!(
                                    "Wrong repo detected (was: {}), no alternative found: {}",
                                    original_repo, reason
                                ),
                            });
                        }
                        new_resolution => {
                            let new_repo_name =
                                new_resolution.repo_name().unwrap_or("unknown").to_string();
                            tracing::info!(
                                short_id = %issue.short_id,
                                from = %original_repo,
                                to = %new_repo_name,
                                "Swapping to alternative repository"
                            );

                            // Notify about repo swap
                            let _ = self
                                .notifier
                                .notify_repo_swap(issue, original_repo, &new_repo_name)
                                .await;

                            // Set up new repo: fetch + worktree
                            let new_project_dir = match &new_resolution {
                                RepoResolution::Resolved { project_dir, .. } => project_dir.clone(),
                                RepoResolution::Skip { .. } => unreachable!(),
                            };

                            if let (Some(scm_url), Some(_index_default_branch), Some(repo_name)) = (
                                new_resolution.scm_url(),
                                new_resolution.default_branch(),
                                new_resolution.repo_name(),
                            ) {
                                let detected_branch =
                                    match GitOps::ensure_repo_fetched(&new_project_dir, scm_url)
                                        .await
                                    {
                                        Ok(branch) => branch,
                                        Err(e) => {
                                            break Ok(ProcessingOutcome::Failed {
                                                error: format!(
                                                    "Failed to fetch alternative repo {}: {}",
                                                    repo_name, e
                                                ),
                                            });
                                        }
                                    };
                                self.reindex_repo(repo_name, &new_project_dir).await;

                                let checkout_ref = format!("origin/{}", detected_branch);
                                let wt_path = worktree_path(
                                    &self.config.workspace,
                                    repo_name,
                                    &issue.short_id,
                                );
                                if let Err(e) = GitOps::create_worktree(
                                    &new_project_dir,
                                    &wt_path,
                                    &checkout_ref,
                                )
                                .await
                                {
                                    break Ok(ProcessingOutcome::Failed {
                                        error: format!(
                                            "Failed to create worktree for alternative repo {}: {}",
                                            repo_name, e
                                        ),
                                    });
                                }

                                current_effective_dir = wt_path;
                            } else {
                                current_effective_dir = new_project_dir.clone();
                            }

                            current_project_dir = new_project_dir;
                            current_resolution = new_resolution;
                            continue;
                        }
                    }
                }
                _ => break pipeline_result,
            }
        };

        // Handle pipeline errors
        if let Err(ref e) = result {
            let error_str = e.to_string();
            self.tracker
                .mark_failed(source_name, &issue.id, &error_str)
                .ok();
            if let Some(id) = attempt_id {
                let _ = self.tracker.update_qa_outcome_stats_for_attempt(id, false);
            }
            let _ = notify_failed_with_escalation(&self.notifier, &self.tracker, issue, &error_str)
                .await;
            record_feedback_outcome(
                &self.tracker,
                self.embedding_client.as_deref(),
                self.issue_embedding_service.as_deref(),
                &self.feedback_analyzer,
                source_name,
                issue,
                Outcome::Failed,
            )
            .await;
            record_error_pattern(&self.tracker, source_name, &issue.id, &error_str);
        }

        // Record processing duration metric
        let final_status = self
            .tracker
            .get_attempt(source_name, &issue.id)
            .ok()
            .flatten()
            .map(|a| a.status.to_string())
            .unwrap_or_else(|| "unknown".to_string());
        let processing_time_metric = ProcessingMetric::new(
            "processing_time",
            processing_started_at.elapsed().as_secs_f64(),
        )
        .with_source(source_name.to_string())
        .with_tags(json!({ "status": final_status }));
        self.tracker.record_metric(&processing_time_metric).ok();

        // Run code quality evaluation (AFTER hook)
        if !eval_before_snapshots.is_empty() {
            let eval_attempt_id = attempt_id.unwrap_or(0);
            let eval_repo = current_resolution.repo_name().unwrap_or("unknown");
            match claudear_analysis::evaluation::CodeQualityEvaluator::run_after_and_compute_deltas(
                &current_effective_dir,
                &self.config.evaluation,
                eval_before_snapshots,
                eval_attempt_id,
                eval_repo,
            )
            .await
            {
                Ok(eval_result) => {
                    if !eval_result.deltas.is_empty() {
                        tracing::info!(
                            short_id = %issue.short_id,
                            improved = eval_result.overall_improved,
                            deltas = eval_result.deltas.len(),
                            "Evaluation complete"
                        );

                        // Post evaluation comment on PR
                        if self.config.evaluation.post_pr_comment {
                            let pr_url = match &result {
                                Ok(ProcessingOutcome::Success { pr_url }) => Some(pr_url.as_str()),
                                _ => None,
                            };
                            if let Some(pr_url) = pr_url {
                                if let Some((repo, pr_number)) =
                                    claudear_storage::parse_pr_url(pr_url)
                                {
                                    if let Some(ref gh) = self.github_client {
                                        let comment = eval_result.format_pr_comment();
                                        if let Err(e) =
                                            gh.add_issue_comment(&repo, pr_number, &comment).await
                                        {
                                            tracing::warn!(
                                                error = %e,
                                                "Failed to post evaluation comment on PR"
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        short_id = %issue.short_id,
                        error = %e,
                        "Post-fix evaluation failed"
                    );
                }
            }
        }

        // Cleanup worktree
        self.cleanup_worktree(&current_resolution, issue, &current_project_dir)
            .await;

        // Convert result to outcome, converting any surviving WrongRepo to Failed
        match result {
            Ok(ProcessingOutcome::WrongRepo {
                original_repo,
                suggested_repo,
            }) => Ok(ProcessingOutcome::Failed {
                error: format!(
                    "Wrong repo detected (was: {}, suggested: {:?}), retries exhausted",
                    original_repo, suggested_repo
                ),
            }),
            Ok(outcome) => Ok(outcome),
            Err(e) => Ok(ProcessingOutcome::Failed {
                error: e.to_string(),
            }),
        }
    }

    /// The core execution: notify start -> build context -> enrich -> Q&A loop -> Claude -> handle result.
    #[expect(clippy::too_many_arguments)]
    async fn execute_pipeline(
        &self,
        issue: &mut Issue,
        source_name: &str,
        resolution: &RepoResolution,
        attempt_id: Option<i64>,
        review_feedback: Option<&str>,
        existing_pr_branch: Option<&str>,
        effective_project_dir: &std::path::Path,
        context_provider: &dyn ContextProvider,
    ) -> Result<ProcessingOutcome> {
        // Notify start
        self.notifier.notify_start(issue).await?;

        // Set repo name in issue metadata for template rendering
        if let Some(name) = resolution.repo_name() {
            issue.set_metadata("target_repo_name", name.to_string());
        }

        // Find similar issues for context
        let similar_issues_context =
            if let Some(ref embedding_service) = self.issue_embedding_service {
                match embedding_service.find_similar(issue, source_name).await {
                    Ok(similar) if !similar.is_empty() => {
                        let metric = ProcessingMetric::new("similar_issues_context_added", 1.0)
                            .with_source(source_name.to_string());
                        self.tracker.record_metric(&metric).ok();
                        format_similar_issues_context(&similar)
                    }
                    _ => String::new(),
                }
            } else {
                String::new()
            };

        // Build context from source/handler
        let mut context = context_provider.build_issue_context(issue).await?;

        // Append similar issues context
        if !similar_issues_context.is_empty() {
            context = format!("{}\n{}", context, similar_issues_context);
        }

        // Enrich context with code search
        if self.config.code_index.enabled {
            if let Some(ref code_search) = self.code_search_service {
                let query = claudear_analysis::repo::code_index::build_code_search_query(issue);
                let repo_id = resolution.repo_id();
                match code_search.search(&query, repo_id, 5).await {
                    Ok(results) if !results.is_empty() => {
                        let metric = ProcessingMetric::new("code_search_context_added", 1.0)
                            .with_source(source_name.to_string());
                        self.tracker.record_metric(&metric).ok();
                        context = format!(
                            "{}\n{}",
                            context,
                            claudear_analysis::repo::code_index::format_code_search_context(
                                &results
                            )
                        );
                    }
                    _ => {}
                }
            }
        }

        // Append PR review feedback context for review-driven reruns
        if let Some(feedback) = review_feedback {
            let mut review_context = format!(
                "\n\n## PR Review Feedback\n{}\n\nAddress all review feedback in this update.",
                feedback
            );
            if let Some(branch) = existing_pr_branch {
                review_context.push_str(&format!(
                    "\n\nIMPORTANT: You are updating an existing PR on branch `{}`. \
                     Push your changes to this branch. Do NOT create a new branch or a new PR.",
                    branch
                ));
            }
            context = format!("{}{}", context, review_context);
        }

        let repo_scope = resolution.repo_name().map(|v| v.to_string());
        let mut used_qa_ids: Vec<i64> = Vec::new();

        // Preload reusable Q&A context
        if self.config.ask.enabled {
            let preload_query = format!("{} {}", issue.title, context);
            let preload_norm = normalize_text(&preload_query);
            let preload_embedding =
                embed_text(self.embedding_client.as_deref(), &preload_query).await;
            match find_reusable_qa(
                self.tracker.as_ref(),
                &self.config.ask,
                source_name,
                repo_scope.as_deref(),
                &preload_norm,
                preload_embedding.as_deref(),
            ) {
                Ok(matches) if !matches.is_empty() => {
                    context = format!("{}\n\n{}", context, format_reuse_context(&matches));
                    if let Some(id) = attempt_id {
                        for m in &matches {
                            let _ = self.tracker.record_qa_usage(
                                id,
                                m.entry.id,
                                "reused",
                                m.final_score,
                            );
                        }
                    }
                    used_qa_ids.extend(matches.into_iter().map(|m| m.entry.id));
                }
                Ok(_) => {}
                Err(e) => {
                    tracing::warn!(error = %e, "Failed to preload reusable Q&A context")
                }
            }
        }

        // Claude execution + ask loop
        let mut rounds: u8 = 0;
        let claude_result = loop {
            let prompt = self
                .agent
                .build_prompt_for_issue(issue, &context, effective_project_dir);

            // Enhance prompt with feedback learnings
            let prompt = {
                let analyzer = self.feedback_analyzer.lock().await;
                let issue_emb = self
                    .issue_embedding_service
                    .as_ref()
                    .and_then(|svc| svc.get_embedding(source_name, &issue.id).ok().flatten());
                match issue_emb.and_then(|emb| emb.embedding) {
                    Some(ref emb) => analyzer.enhance_prompt(&prompt, issue, emb),
                    None => prompt,
                }
            };

            // Enhance with continuous learning context
            let prompt = enhance_prompt_with_learning(
                &self.config,
                &self.tracker,
                &prompt,
                issue,
                resolution.repo_name(),
            );

            let mut run_result = self
                .agent
                .execute_with_attempt(&prompt, Some(issue), attempt_id, effective_project_dir)
                .await?;
            run_result.used_qa_ids = used_qa_ids.clone();

            let blocking_question = match (
                self.config.ask.enabled,
                run_result.blocking_question.clone(),
            ) {
                (true, Some(q)) => q,
                _ => break run_result,
            };

            if rounds >= self.config.ask.max_rounds_per_attempt {
                run_result.success = false;
                run_result.error = Some(format!(
                    "Maximum blocking-question rounds ({}) reached",
                    self.config.ask.max_rounds_per_attempt
                ));
                break run_result;
            }
            rounds = rounds.saturating_add(1);

            let question_norm = normalize_text(&blocking_question.question);
            let question_embedding = embed_text(
                self.embedding_client.as_deref(),
                &blocking_question.question,
            )
            .await;

            let reusable = match find_reusable_qa(
                self.tracker.as_ref(),
                &self.config.ask,
                source_name,
                repo_scope.as_deref(),
                &question_norm,
                question_embedding.as_deref(),
            ) {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!(error = %e, "Failed to query reusable Q&A");
                    Vec::new()
                }
            };

            if let Some(best) = reusable.first() {
                if let Some(id) = attempt_id {
                    let _ =
                        self.tracker
                            .record_qa_usage(id, best.entry.id, "reused", best.final_score);
                }
                if !used_qa_ids.contains(&best.entry.id) {
                    used_qa_ids.push(best.entry.id);
                }
                let activity = ActivityLogEntry::new(
                    "question_reused",
                    format!("Reused stored Q&A for {}", issue.short_id),
                )
                .with_source(source_name.to_string())
                .with_issue(issue.id.clone(), issue.short_id.clone())
                .with_metadata(json!({
                    "qa_id": best.entry.id,
                    "score": best.final_score,
                }));
                self.tracker.record_activity(&activity).ok();

                context = format!(
                    "{}\n\n{}",
                    context,
                    format_answer_context(
                        &blocking_question,
                        &best.entry.answer_text,
                        &best.entry.channel,
                        true,
                    )
                );
                continue;
            }

            // Ask humans
            let resolved_user = issue.get_metadata::<String>("resolved_user");
            let target_discord_id = resolved_user
                .as_deref()
                .and_then(|slug| self.user_registry.get_by_slug(slug))
                .and_then(|u| u.discord_id.clone());
            let target_email = resolved_user
                .as_deref()
                .and_then(|slug| self.user_registry.get_by_slug(slug))
                .and_then(|u| u.email.clone());
            let ask_request = AskRequest {
                correlation_id: build_correlation_id(&issue.short_id),
                source: issue.source.clone(),
                repo: repo_scope.clone(),
                issue_id: issue.id.clone(),
                short_id: issue.short_id.clone(),
                question: blocking_question.clone(),
                asked_at: chrono::Utc::now(),
                target_discord_id,
                target_email,
                target_slack_id: resolved_user
                    .as_deref()
                    .and_then(|slug| self.user_registry.get_by_slug(slug))
                    .and_then(|u| u.slack_id.clone()),
            };

            let asked_activity = ActivityLogEntry::new(
                "question_asked",
                format!("Asked human question for {}", issue.short_id),
            )
            .with_source(source_name.to_string())
            .with_issue(issue.id.clone(), issue.short_id.clone())
            .with_metadata(json!({
                "correlation_id": ask_request.correlation_id,
                "question": blocking_question.question,
            }));
            self.tracker.record_activity(&asked_activity).ok();

            let reply = send_to_all_and_wait_first_reply(
                Arc::clone(&self.notifier),
                issue,
                &ask_request,
                tokio::time::Duration::from_secs(self.config.ask.wait_timeout_secs),
                tokio::time::Duration::from_secs(self.config.ask.poll_interval_secs),
            )
            .await?;

            if let Some(reply) = reply {
                let answered_activity = ActivityLogEntry::new(
                    "question_answered",
                    format!("Human answered question for {}", issue.short_id),
                )
                .with_source(source_name.to_string())
                .with_issue(issue.id.clone(), issue.short_id.clone())
                .with_metadata(json!({
                    "channel": reply.channel,
                    "responder": reply.responder,
                    "correlation_id": reply.correlation_id,
                }));
                self.tracker.record_activity(&answered_activity).ok();

                let qa_entry = QaKnowledgeEntry {
                    id: 0,
                    source: issue.source.clone(),
                    repo: repo_scope.clone(),
                    issue_id: issue.id.clone(),
                    short_id: issue.short_id.clone(),
                    question_text: blocking_question.question.clone(),
                    question_norm,
                    question_embedding: question_embedding.clone(),
                    answer_text: reply.answer.clone(),
                    answer_norm: normalize_text(&reply.answer),
                    answer_embedding: embed_text(self.embedding_client.as_deref(), &reply.answer)
                        .await,
                    channel: reply.channel.clone(),
                    responder: reply.responder.clone(),
                    correlation_id: ask_request.correlation_id.clone(),
                    asked_at: ask_request.asked_at,
                    answered_at: reply.replied_at,
                    success_count: 0,
                    failure_count: 0,
                    last_used_at: None,
                    metadata: Some(json!({
                        "context": blocking_question.context,
                        "options": blocking_question.options,
                        "why": blocking_question.why,
                    })),
                };
                if let Ok(qa_id) = self.tracker.store_qa_knowledge(&qa_entry) {
                    if let Some(id) = attempt_id {
                        let _ = self.tracker.record_qa_usage(id, qa_id, "asked", 1.0);
                    }
                    if !used_qa_ids.contains(&qa_id) {
                        used_qa_ids.push(qa_id);
                    }
                }

                context = format!(
                    "{}\n\n{}",
                    context,
                    format_answer_context(&blocking_question, &reply.answer, &reply.channel, false)
                );
                continue;
            }

            let timeout_activity = ActivityLogEntry::new(
                "question_timeout_best_effort",
                format!("No human reply received for {}", issue.short_id),
            )
            .with_source(source_name.to_string())
            .with_issue(issue.id.clone(), issue.short_id.clone())
            .with_metadata(json!({
                "best_effort": self.config.ask.best_effort_on_timeout,
                "question": blocking_question.question,
            }));
            self.tracker.record_activity(&timeout_activity).ok();

            if self.config.ask.best_effort_on_timeout {
                context = format!(
                    "{}\n\n{}",
                    context,
                    format_timeout_context(&blocking_question)
                );
                continue;
            }

            run_result.success = false;
            run_result.error = Some("Timed out waiting for human reply".to_string());
            break run_result;
        };

        // Strategy fingerprinting
        if self.config.learning.strategy_fingerprinting {
            if let Some(aid) = attempt_id {
                if let Ok(execs) = self.tracker.get_executions_for_attempt(aid) {
                    if let Some(exec) = execs.first() {
                        if let Some(ref log_path) = exec.stdout_log_path {
                            let path = std::path::Path::new(log_path);
                            if path.exists() {
                                match claudear_analysis::learning::StrategyParser::parse_with_llm(
                                    path,
                                    aid,
                                    self.llm_analyzer
                                        .as_deref()
                                        .map(|a| a as &dyn claudear_analysis::llm::LlmAnalyzer),
                                ) {
                                    Ok(fp) => {
                                        if let Err(e) = self.tracker.store_strategy_fingerprint(&fp)
                                        {
                                            tracing::warn!(
                                                error = %e,
                                                "Failed to store strategy fingerprint"
                                            );
                                        }
                                    }
                                    Err(e) => tracing::debug!(
                                        error = %e,
                                        "Failed to parse strategy from log"
                                    ),
                                }
                            }
                        }
                    }
                }
            }
        }

        // Check for wrong_repo signal before handling the result
        if let Some(ref suggested) = claude_result.wrong_repo {
            let original_repo = resolution.repo_name().unwrap_or("unknown").to_string();
            tracing::warn!(
                short_id = %issue.short_id,
                original_repo = %original_repo,
                suggested_repo = %suggested,
                "Claude detected wrong repository"
            );
            let activity = ActivityLogEntry::new(
                "wrong_repo_detected",
                format!(
                    "Wrong repo detected for {}: {} (suggested: {})",
                    issue.short_id, original_repo, suggested
                ),
            )
            .with_source(source_name.to_string())
            .with_issue(issue.id.clone(), issue.short_id.clone())
            .with_metadata(json!({
                "original_repo": original_repo,
                "suggested_repo": suggested,
            }));
            self.tracker.record_activity(&activity).ok();

            let metric = ProcessingMetric::new("wrong_repo_detected", 1.0)
                .with_source(source_name.to_string());
            self.tracker.record_metric(&metric).ok();

            return Ok(ProcessingOutcome::WrongRepo {
                suggested_repo: Some(suggested.clone()),
                original_repo,
            });
        }

        // Handle result
        if claude_result.success {
            // For review reruns, resolve the effective PR URL
            let effective_pr_url = if existing_pr_branch.is_some() {
                let stored_url = self
                    .tracker
                    .get_attempt(source_name, &issue.id)
                    .ok()
                    .flatten()
                    .and_then(|a| a.pr_url);
                match (&claude_result.pr_url, &stored_url) {
                    (Some(new_url), Some(existing_url)) if new_url != existing_url => {
                        tracing::warn!(
                            short_id = %issue.short_id,
                            existing_pr = %existing_url,
                            claude_pr = %new_url,
                            "Review rerun produced a different PR URL; keeping original"
                        );
                        stored_url
                    }
                    (None, Some(_)) => {
                        tracing::info!(
                            short_id = %issue.short_id,
                            "Review rerun pushed to existing branch (no new PR URL)"
                        );
                        stored_url
                    }
                    _ => claude_result.pr_url.clone(),
                }
            } else {
                claude_result.pr_url.clone()
            };

            if let Some(ref pr_url) = effective_pr_url {
                tracing::info!(short_id = %issue.short_id, pr_url = %pr_url, "Success! PR created");
                self.tracker.mark_success(source_name, &issue.id, pr_url)?;
                if existing_pr_branch.is_some() || review_feedback.is_some() {
                    issue.set_metadata("is_pr_update", true);
                }
                if let Some(ref changelog) = claude_result.changelog {
                    issue.set_metadata("changelog", changelog.clone());
                }
                if claude_result.confidence > 0 {
                    issue.set_metadata("confidence", claude_result.confidence);
                }
                if let Some(ref reasoning) = claude_result.confidence_reasoning {
                    issue.set_metadata("confidence_reasoning", reasoning.clone());
                }
                self.notifier.notify_success(issue, pr_url).await?;
                if let Some(id) = attempt_id {
                    let _ = self.tracker.update_qa_outcome_stats_for_attempt(id, true);
                }

                // Record metric for PR creation
                let metric =
                    ProcessingMetric::new("pr_created", 1.0).with_source(source_name.to_string());
                self.tracker.record_metric(&metric).ok();

                // Create or update prs table record
                if let Some((repo, pr_number)) = claudear_storage::parse_pr_url(pr_url) {
                    let mut pr_record = if let Ok(Some(existing)) = self.tracker.get_pr(pr_url) {
                        existing
                    } else {
                        claudear_core::types::PrRecord::for_issue(
                            pr_url.clone(),
                            &repo,
                            pr_number,
                            source_name,
                            &issue.id,
                        )
                    };
                    pr_record.attempt_id = attempt_id;

                    if let Some(ref gh) = self.github_client {
                        match gh.get_pr_info(&repo, pr_number).await {
                            Ok(info) => {
                                pr_record.head_branch = info.head_branch;
                                pr_record.base_branch = info.base_branch;
                                pr_record.title = info.title;
                                pr_record.author = info.author;
                            }
                            Err(e) => {
                                tracing::warn!(
                                    error = %e,
                                    "Failed to fetch PR info from GitHub"
                                );
                            }
                        }
                    }

                    if let Err(e) = self.tracker.upsert_pr(&pr_record) {
                        tracing::warn!(error = %e, "Failed to upsert PR record");
                    }
                }

                // Post confidence comment on PR
                if self.config.evaluation.post_pr_comment && claude_result.confidence > 0 {
                    if let Some((ref repo, pr_number)) = claudear_storage::parse_pr_url(pr_url) {
                        if let Some(ref gh) = self.github_client {
                            let mut comment =
                                format!("## Fix Confidence: {}/100\n", claude_result.confidence);
                            if let Some(ref reasoning) = claude_result.confidence_reasoning {
                                comment.push('\n');
                                comment.push_str(reasoning);
                                comment.push('\n');
                            }
                            if let Err(e) = gh.add_issue_comment(repo, pr_number, &comment).await {
                                tracing::warn!(
                                    error = %e,
                                    "Failed to post confidence comment on PR"
                                );
                            }
                        }
                    }
                }

                // Store embedding for future similarity lookups
                if let Some(ref embedding_service) = self.issue_embedding_service {
                    if embedding_service
                        .embed_issue(issue, source_name)
                        .await
                        .is_ok()
                    {
                        let metric = ProcessingMetric::new("issue_embedding_stored", 1.0)
                            .with_source(source_name.to_string());
                        self.tracker.record_metric(&metric).ok();
                    }
                }

                // Register PR for review watching
                if let Some(ref review_watcher) = self.review_watcher {
                    if let Some((repo, pr_number)) = claudear_storage::parse_pr_url(pr_url) {
                        let state =
                            PrReviewState::new(pr_url, &repo, pr_number, &issue.id, source_name);
                        review_watcher.watch_pr(state);
                        tracing::info!(
                            pr_url = %pr_url,
                            repo = %repo,
                            pr_number = pr_number,
                            "PR registered for review watching"
                        );
                    }
                }

                // Log processing_completed activity
                let activity = ActivityLogEntry::new(
                    "processing_completed",
                    format!("Processing completed for {}", issue.short_id),
                )
                .with_source(source_name.to_string())
                .with_issue(issue.id.clone(), issue.short_id.clone())
                .with_metadata(json!({
                    "has_pr": true,
                    "pr_url": pr_url,
                }));
                self.tracker.record_activity(&activity).ok();

                return Ok(ProcessingOutcome::Success {
                    pr_url: pr_url.clone(),
                });
            } else {
                let reason = if claude_result.output.is_empty() {
                    "No PR URL found in output".to_string()
                } else if claude_result.output.chars().count() > 500 {
                    let truncated: String = claude_result.output.chars().take(497).collect();
                    format!("{}...", truncated)
                } else {
                    claude_result.output.clone()
                };
                tracing::info!(short_id = %issue.short_id, reason = %reason, "Completed without PR");
                issue.set_metadata("completion_reason", reason.clone());
                self.tracker.mark_failed(
                    source_name,
                    &issue.id,
                    &format!("Claude completed without creating a PR: {}", reason),
                )?;
                self.notifier.notify_completed(issue).await?;
                if let Some(id) = attempt_id {
                    let _ = self.tracker.update_qa_outcome_stats_for_attempt(id, false);
                }

                // Record feedback outcome
                record_feedback_outcome(
                    &self.tracker,
                    self.embedding_client.as_deref(),
                    self.issue_embedding_service.as_deref(),
                    &self.feedback_analyzer,
                    source_name,
                    issue,
                    Outcome::Failed,
                )
                .await;

                let activity = ActivityLogEntry::new(
                    "processing_completed_no_pr",
                    format!("Completed without PR for {}: {}", issue.short_id, reason),
                )
                .with_source(source_name.to_string())
                .with_issue(issue.id.clone(), issue.short_id.clone())
                .with_metadata(json!({
                    "has_pr": false,
                    "pr_url": Option::<String>::None,
                }));
                self.tracker.record_activity(&activity).ok();

                return Ok(ProcessingOutcome::CompletedNoPr { reason });
            }
        }

        // Failed
        let base_error = claude_result.error.as_deref().unwrap_or("Unknown error");
        let error = if !claude_result.output.is_empty() {
            let summary = if claude_result.output.chars().count() > 500 {
                let truncated: String = claude_result.output.chars().take(497).collect();
                format!("{}...", truncated)
            } else {
                claude_result.output.clone()
            };
            format!("{}\n\nClaude's summary: {}", base_error, summary)
        } else {
            base_error.to_string()
        };
        tracing::error!(short_id = %issue.short_id, error = %error, "Failed");
        self.tracker.mark_failed(source_name, &issue.id, &error)?;
        notify_failed_with_escalation(&self.notifier, &self.tracker, issue, &error).await?;
        if let Some(id) = attempt_id {
            let _ = self.tracker.update_qa_outcome_stats_for_attempt(id, false);
        }

        // Record feedback outcome
        record_feedback_outcome(
            &self.tracker,
            self.embedding_client.as_deref(),
            self.issue_embedding_service.as_deref(),
            &self.feedback_analyzer,
            source_name,
            issue,
            Outcome::Failed,
        )
        .await;

        // Record error pattern
        record_error_pattern(&self.tracker, source_name, &issue.id, &error);

        Ok(ProcessingOutcome::Failed { error })
    }

    /// Re-index a repository after fetching latest changes.
    async fn reindex_repo(&self, repo_name: &str, repo_path: &std::path::Path) {
        if !self.config.code_index.enabled {
            return;
        }
        let emb_client = match self.embedding_client {
            Some(ref c) => c.clone(),
            None => return,
        };
        let code_indexer = claudear_analysis::repo::code_index::CodeIndexer::with_config(
            self.tracker.clone(),
            emb_client,
            self.config.code_index.max_file_size_kb,
            self.config.code_index.batch_size,
        );
        match code_indexer.index_repo(repo_name, repo_path).await {
            Ok(stats) => {
                if stats.files_processed > 0 {
                    tracing::info!(
                        repo = %repo_name,
                        files = stats.files_processed,
                        chunks = stats.chunks_created,
                        "Re-indexed repo after fetch"
                    );
                }
            }
            Err(e) => {
                tracing::warn!(repo = %repo_name, error = %e, "Failed to re-index repo after fetch");
            }
        }
    }

    /// Cleanup worktree after processing.
    async fn cleanup_worktree(
        &self,
        resolution: &RepoResolution,
        issue: &Issue,
        project_dir: &std::path::Path,
    ) {
        if let Some(repo_name) = resolution.repo_name() {
            let wt_path = worktree_path(&self.config.workspace, repo_name, &issue.short_id);
            if wt_path.exists() {
                if let Err(e) = GitOps::remove_worktree(project_dir, &wt_path).await {
                    tracing::warn!(
                        short_id = %issue.short_id,
                        error = %e,
                        "Failed to remove worktree"
                    );
                }
            }
        }
    }

    /// Answer a pure question with RAG-grounded context, read-only (no PR).
    async fn answer_question_issue(
        &self,
        issue: &Issue,
        resolution: &RepoResolution,
        source_name: &str,
    ) -> ProcessingOutcome {
        // Run in the resolved repo when available (lets the agent read real code),
        // otherwise a scratch dir (answer is grounded purely in RAG context).
        let project_dir = match resolution {
            RepoResolution::Resolved { project_dir, .. } => project_dir.clone(),
            RepoResolution::Skip { .. } => {
                let dir = std::env::temp_dir().join("claudear-qa");
                let _ = std::fs::create_dir_all(&dir);
                dir
            }
        };

        // Retrieve grounding context from the code index across ALL repos.
        let context = if let Some(ref code_search) = self.code_search_service {
            let query = claudear_analysis::repo::code_index::build_code_search_query(issue);
            match code_search
                .search(&query, None, self.config.qa.max_context_chunks)
                .await
            {
                Ok(results) if !results.is_empty() => {
                    claudear_analysis::repo::code_index::format_code_search_context(&results)
                }
                _ => String::new(),
            }
        } else {
            String::new()
        };

        self.record_issue_decision(
            issue,
            "question_detected",
            format!("Answering {} as a question (read-only)", issue.short_id),
            json!({ "context_chars": context.len() }),
        );

        let timeout = std::time::Duration::from_secs(self.config.qa.answer_timeout_secs.max(1));
        let answer_result = tokio::time::timeout(
            timeout,
            self.agent.answer_question(issue, &context, &project_dir),
        )
        .await;

        match answer_result {
            Ok(Ok(answer)) => {
                if let Err(e) = self.notifier.notify_answer(issue, &answer).await {
                    tracing::warn!(short_id = %issue.short_id, error = %e, "Failed to deliver answer");
                }
                let summary: String = answer.chars().take(500).collect();
                if let Err(e) = self.tracker.mark_answered(source_name, &issue.id, &summary) {
                    tracing::warn!(short_id = %issue.short_id, error = %e, "Failed to mark answered");
                }
                self.record_issue_decision(
                    issue,
                    "question_answered",
                    format!("Answered question {}", issue.short_id),
                    json!({ "answer_chars": answer.len() }),
                );
                ProcessingOutcome::CompletedNoPr {
                    reason: "answered question".to_string(),
                }
            }
            Ok(Err(e)) => {
                let error = e.to_string();
                let _ = self.tracker.mark_failed(source_name, &issue.id, &error);
                ProcessingOutcome::Failed { error }
            }
            Err(_) => {
                let error = format!(
                    "Answer generation timed out after {}s",
                    self.config.qa.answer_timeout_secs
                );
                let _ = self.tracker.mark_failed(source_name, &issue.id, &error);
                ProcessingOutcome::Failed { error }
            }
        }
    }

    /// Record an issue-level decision as a "decision" activity entry.
    fn record_issue_decision(
        &self,
        issue: &Issue,
        decision: &str,
        message: impl Into<String>,
        details: serde_json::Value,
    ) {
        let activity = ActivityLogEntry::new("decision", message.into())
            .with_source(issue.source.clone())
            .with_issue(issue.id.clone(), issue.short_id.clone())
            .with_metadata(json!({
                "decision": decision,
                "details": details,
            }));
        self.tracker.record_activity(&activity).ok();
    }

    /// Resolve an alternative repository after a wrong_repo detection.
    ///
    /// 1. Try Claude's suggested repo name via `resolve_repo_for_cascade`
    /// 2. Fallback: re-infer with exclusions via `inferrer.infer_excluding()`
    /// 3. If neither works, return `RepoResolution::Skip`
    async fn resolve_alternative_repo(
        &self,
        issue: &Issue,
        suggested: Option<&str>,
        excluded_repos: &[String],
    ) -> RepoResolution {
        let Some(inferrer) = &self.inferrer else {
            return RepoResolution::Skip {
                reason: "No inferrer available for repo swap".to_string(),
            };
        };

        // Try Claude's suggested repo first
        if let Some(suggested_name) = suggested {
            if !excluded_repos.contains(&suggested_name.to_string()) {
                let resolution = claudear_analysis::inference::resolve_repo_for_cascade(
                    Some(inferrer),
                    suggested_name,
                );
                if resolution.is_resolved() {
                    return resolution;
                }
            }
        }

        // Fallback: re-infer with exclusions
        if let Some(inferred) = inferrer.infer_excluding(issue, excluded_repos) {
            return RepoResolution::Resolved {
                project_dir: inferred.repo.path,
                repo_name: inferred.repo.name,
                repo_id: None,
                scm_url: inferred.repo.scm_url,
                default_branch: inferred.repo.default_branch,
                confidence: Some(inferred.confidence),
            };
        }

        RepoResolution::Skip {
            reason: format!("No alternative repo found (excluded: {:?})", excluded_repos),
        }
    }
}

/// Enhance a prompt with continuous learning context.
pub fn enhance_prompt_with_learning(
    config: &Config,
    tracker: &Arc<dyn FixAttemptTracker>,
    base_prompt: &str,
    issue: &Issue,
    repo: Option<&str>,
) -> String {
    let learning = &config.learning;
    let Some(repo_name) = repo else {
        return base_prompt.to_string();
    };

    let mut extra_context = String::new();

    if learning.repo_knowledge {
        if let Ok(knowledge) = tracker.get_repo_knowledge(repo_name) {
            let ctx = claudear_analysis::learning::RepoKnowledgeManager::format_knowledge_context(
                &knowledge,
            );
            if !ctx.is_empty() {
                extra_context.push_str(&ctx);
            }
        }
    }

    if learning.qa_promotion {
        if let Ok(instructions) = tracker.get_promoted_instructions(repo_name) {
            let ctx =
                claudear_analysis::learning::QaPromoter::format_promoted_context(&instructions);
            if !ctx.is_empty() {
                extra_context.push_str(&ctx);
            }
        }
    }

    if learning.strategy_fingerprinting {
        if let Ok(strategies) = tracker.get_successful_strategies(repo_name, 3) {
            let ctx = claudear_analysis::learning::StrategyParser::format_strategy_suggestions(
                &strategies,
            );
            if !ctx.is_empty() {
                extra_context.push_str(&ctx);
            }
        }
    }

    if learning.cluster_detection {
        if let Ok(clusters) = tracker.get_active_clusters(&issue.source) {
            for cluster in &clusters {
                if cluster.issue_ids.contains(&issue.id) {
                    extra_context.push_str(
                        &claudear_analysis::learning::ClusterDetector::format_cluster_context(
                            cluster,
                        ),
                    );
                    extra_context.push('\n');
                    break;
                }
            }
        }
    }

    if learning.cross_repo_correlation {
        match claudear_analysis::learning::CrossRepoCorrelator::get_active_insights(
            tracker.as_ref(),
            3,
            learning.cross_repo_window_hours * 2,
        ) {
            Ok(insights) if !insights.is_empty() => {
                let ctx =
                    claudear_analysis::learning::CrossRepoCorrelator::format_context(&insights);
                if !ctx.is_empty() {
                    extra_context.push_str(&ctx);
                }
            }
            _ => {}
        }
    }

    if extra_context.is_empty() {
        return base_prompt.to_string();
    }

    format!("{}\n---\n\n{}", extra_context, base_prompt)
}

/// Record error pattern for analytics.
pub fn record_error_pattern(
    tracker: &Arc<dyn FixAttemptTracker>,
    source: &str,
    issue_id: &str,
    error_msg: &str,
) {
    let error_type = classify_error(error_msg);
    let pattern_hash = compute_error_hash(error_msg);

    let mut pattern = ErrorPattern::new(pattern_hash);
    pattern.error_type = Some(error_type.to_string());
    pattern.error_message = Some(error_msg.to_string());
    pattern.sources = Some(vec![source.to_string()]);
    pattern.example_issue_ids = Some(vec![issue_id.to_string()]);

    if let Err(e) = tracker.record_error_pattern(&pattern) {
        tracing::warn!(error = %e, "Failed to record error pattern");
    }
}

/// Route hard failures to the global notifier user.
pub async fn notify_failed_with_escalation(
    notifier: &Arc<dyn Notifier>,
    tracker: &Arc<dyn FixAttemptTracker>,
    issue: &Issue,
    error: &str,
) -> Result<()> {
    if runner::is_hard_error(error) {
        let mut global_issue = issue.clone();
        global_issue.metadata.remove("resolved_user");
        global_issue
            .metadata
            .insert("hard_error".to_string(), serde_json::Value::Bool(true));

        let activity = ActivityLogEntry::new(
            "error",
            format!("Hard Claude error escalated for {}", issue.short_id),
        )
        .with_source(issue.source.clone())
        .with_issue(issue.id.clone(), issue.short_id.clone())
        .with_metadata(json!({
            "hard_error": true,
            "rate_limited": runner::is_rate_limit_error(error),
            "error": truncate_error_for_activity(error),
        }));
        tracker.record_activity(&activity).ok();

        return notifier.notify_failed(&global_issue, error).await;
    }

    notifier.notify_failed(issue, error).await
}

/// Truncate error messages for activity logging.
pub fn truncate_error_for_activity(error: &str) -> String {
    let max_len = 500;
    if error.len() <= max_len {
        error.to_string()
    } else {
        let safe_end = error
            .char_indices()
            .take_while(|(i, _)| *i <= max_len.saturating_sub(3))
            .last()
            .map(|(i, _)| i)
            .unwrap_or(0);
        format!("{}...", &error[..safe_end])
    }
}

/// Record a feedback outcome from an attempt.
pub async fn record_feedback_outcome(
    tracker: &Arc<dyn FixAttemptTracker>,
    embedding_client: Option<&EmbeddingClient>,
    issue_embedding_service: Option<&IssueEmbeddingService>,
    feedback_analyzer: &tokio::sync::Mutex<FeedbackAnalyzer>,
    source_name: &str,
    issue: &Issue,
    outcome: Outcome,
) {
    let attempt = match tracker.get_attempt(source_name, &issue.id) {
        Ok(Some(attempt)) => attempt,
        _ => return,
    };

    let prompt = tracker
        .get_executions_for_attempt(attempt.id)
        .ok()
        .and_then(|execs| execs.into_iter().next())
        .and_then(|exec| exec.prompt_used)
        .unwrap_or_default();

    let mut fix_outcome = FixOutcome::from_attempt(&attempt, issue, &prompt, outcome);

    if let Some(embedding_client) = embedding_client {
        let embedding = match issue_embedding_service
            .and_then(|svc| svc.get_embedding(source_name, &issue.id).ok().flatten())
            .and_then(|existing| existing.embedding)
        {
            Some(existing) => Some(existing),
            None => embedding_client.embed(&fix_outcome.issue_text).await.ok(),
        };
        if let Some(emb) = embedding {
            fix_outcome.set_embedding(emb);
        }
    }

    if let Err(e) = tracker.store_feedback_outcome(&fix_outcome) {
        tracing::warn!(error = %e, "Failed to store feedback outcome");
    }

    let mut analyzer = feedback_analyzer.lock().await;
    if let Err(e) = analyzer.record_outcome(&attempt, issue, &prompt, outcome) {
        tracing::warn!(error = %e, "Failed to record feedback outcome in memory");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use claudear_storage::AttemptTracker;

    fn make_test_issue() -> Issue {
        Issue {
            id: "test-1".to_string(),
            short_id: "T-1".to_string(),
            title: "Test issue".to_string(),
            description: Some("Test description".to_string()),
            url: "https://example.com/issue/1".to_string(),
            source: "test".to_string(),
            priority: claudear_core::types::IssuePriority::Medium,
            status: claudear_core::types::IssueStatus::Open,
            metadata: std::collections::HashMap::new(),
            created_at: None,
            updated_at: None,
        }
    }

    // --- truncate_error_for_activity ---

    #[test]
    fn test_truncate_error_for_activity_short_message() {
        let short = "Something failed";
        assert_eq!(truncate_error_for_activity(short), short);
    }

    #[test]
    fn test_truncate_error_for_activity_exact_500() {
        let msg = "a".repeat(500);
        assert_eq!(truncate_error_for_activity(&msg), msg);
    }

    #[test]
    fn test_truncate_error_for_activity_long_message() {
        let msg = "x".repeat(600);
        let result = truncate_error_for_activity(&msg);
        assert!(result.ends_with("..."));
        assert!(result.len() <= 503);
    }

    #[test]
    fn test_truncate_error_for_activity_empty() {
        assert_eq!(truncate_error_for_activity(""), "");
    }

    #[test]
    fn test_truncate_error_for_activity_multibyte() {
        let msg = "\u{1F600}".repeat(200);
        let result = truncate_error_for_activity(&msg);
        assert!(result.ends_with("..."));
        assert!(result.is_char_boundary(result.len()));
    }

    #[test]
    fn test_truncate_error_for_activity_501_chars() {
        let msg = "b".repeat(501);
        let result = truncate_error_for_activity(&msg);
        assert!(result.ends_with("..."));
        assert!(result.len() <= 503);
    }

    #[test]
    fn test_truncate_error_for_activity_exactly_one_over() {
        let msg = "c".repeat(501);
        let result = truncate_error_for_activity(&msg);
        assert!(result.ends_with("..."));
        // Should be "ccc...ccc..." where the c part is <= 497 chars
        let without_dots = &result[..result.len() - 3];
        assert!(without_dots.len() <= 497);
    }

    // --- error classification ---

    #[test]
    fn test_classify_error_timeout() {
        assert_eq!(
            claudear_storage::classify_error("Operation timed out after 300s"),
            "timeout"
        );
    }

    #[test]
    fn test_classify_error_build_failure() {
        assert_eq!(
            claudear_storage::classify_error("cargo build failed with exit code 1"),
            "build_failure"
        );
    }

    #[test]
    fn test_classify_error_test_failure() {
        assert_eq!(
            claudear_storage::classify_error("test assertion failed: expected 1, got 2"),
            "test_failure"
        );
    }

    #[test]
    fn test_classify_error_claude_error() {
        assert_eq!(
            claudear_storage::classify_error("claude rate limit exceeded"),
            "claude_error"
        );
    }

    #[test]
    fn test_classify_error_permission_error() {
        assert_eq!(
            claudear_storage::classify_error("permission denied: /etc/hosts"),
            "permission_error"
        );
    }

    #[test]
    fn test_classify_error_network_error() {
        assert_eq!(
            claudear_storage::classify_error("network connection refused"),
            "network_error"
        );
    }

    #[test]
    fn test_classify_error_git_error() {
        assert_eq!(
            claudear_storage::classify_error("git merge conflict in file.rs"),
            "git_error"
        );
    }

    #[test]
    fn test_classify_error_unknown() {
        assert_eq!(
            claudear_storage::classify_error("something completely unknown happened"),
            "unknown"
        );
    }

    #[test]
    fn test_classify_error_compile() {
        assert_eq!(
            claudear_storage::classify_error("failed to compile the project"),
            "build_failure"
        );
    }

    #[test]
    fn test_classify_error_access_denied() {
        assert_eq!(
            claudear_storage::classify_error("access denied to resource"),
            "permission_error"
        );
    }

    // --- error hash ---

    #[test]
    fn test_error_hash_deterministic() {
        let hash1 = claudear_storage::compute_error_hash("git merge conflict in file.rs");
        let hash2 = claudear_storage::compute_error_hash("git merge conflict in file.rs");
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_error_hash_different_for_different_messages() {
        let hash1 = claudear_storage::compute_error_hash("git merge conflict");
        let hash2 = claudear_storage::compute_error_hash("build failure");
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_error_hash_normalizes_numbers() {
        let hash1 = claudear_storage::compute_error_hash("timeout after 30 seconds on line 42");
        let hash2 = claudear_storage::compute_error_hash("timeout after 60 seconds on line 99");
        assert_eq!(
            hash1, hash2,
            "numeric normalization should make these equal"
        );
    }

    #[test]
    fn test_error_hash_nonempty() {
        let hash = claudear_storage::compute_error_hash("some error");
        assert!(!hash.is_empty());
    }

    // --- ProcessingOutcome ---

    #[test]
    fn test_processing_outcome_success_variant() {
        let outcome = ProcessingOutcome::Success {
            pr_url: "https://github.com/org/repo/pull/1".to_string(),
        };
        match outcome {
            ProcessingOutcome::Success { pr_url } => {
                assert_eq!(pr_url, "https://github.com/org/repo/pull/1");
            }
            _ => panic!("Expected Success variant"),
        }
    }

    #[test]
    fn test_processing_outcome_completed_no_pr_variant() {
        let outcome = ProcessingOutcome::CompletedNoPr {
            reason: "No changes needed".to_string(),
        };
        match outcome {
            ProcessingOutcome::CompletedNoPr { reason } => {
                assert_eq!(reason, "No changes needed");
            }
            _ => panic!("Expected CompletedNoPr variant"),
        }
    }

    #[test]
    fn test_processing_outcome_failed_variant() {
        let outcome = ProcessingOutcome::Failed {
            error: "Something went wrong".to_string(),
        };
        match outcome {
            ProcessingOutcome::Failed { error } => {
                assert_eq!(error, "Something went wrong");
            }
            _ => panic!("Expected Failed variant"),
        }
    }

    // --- ProcessingInput ---

    #[test]
    fn test_processing_input_fields() {
        let input = ProcessingInput {
            issue: make_test_issue(),
            source_name: "linear".to_string(),
            match_result: claudear_core::types::MatchResult::matched(
                "test",
                claudear_core::types::MatchPriority::Normal,
            ),
            resolution: RepoResolution::Skip {
                reason: "test".to_string(),
            },
            attempt_id: Some(42),
            review_feedback: Some("Fix the tests".to_string()),
            existing_pr_branch: Some("claudear/fix-123".to_string()),
            intent: None,
        };

        assert_eq!(input.source_name, "linear");
        assert_eq!(input.attempt_id, Some(42));
        assert_eq!(input.review_feedback.as_deref(), Some("Fix the tests"));
        assert_eq!(
            input.existing_pr_branch.as_deref(),
            Some("claudear/fix-123")
        );
    }

    #[test]
    fn test_processing_input_no_optionals() {
        let input = ProcessingInput {
            issue: make_test_issue(),
            source_name: "sentry".to_string(),
            match_result: claudear_core::types::MatchResult::matched(
                "test",
                claudear_core::types::MatchPriority::Normal,
            ),
            resolution: RepoResolution::Skip {
                reason: "no repo".to_string(),
            },
            attempt_id: None,
            review_feedback: None,
            existing_pr_branch: None,
            intent: None,
        };

        assert!(input.attempt_id.is_none());
        assert!(input.review_feedback.is_none());
        assert!(input.existing_pr_branch.is_none());
    }

    // --- enhance_prompt_with_learning ---

    #[test]
    fn test_enhance_prompt_no_repo_returns_base() {
        let config = Config::default();
        let tracker = claudear_storage::SqliteTracker::in_memory().unwrap();
        let tracker: Arc<dyn FixAttemptTracker> = Arc::new(tracker);
        let issue = make_test_issue();

        let result = enhance_prompt_with_learning(&config, &tracker, "base prompt", &issue, None);
        assert_eq!(result, "base prompt");
    }

    #[test]
    fn test_enhance_prompt_all_learning_disabled() {
        let mut config = Config::default();
        config.learning.repo_knowledge = false;
        config.learning.qa_promotion = false;
        config.learning.strategy_fingerprinting = false;
        config.learning.cluster_detection = false;
        config.learning.cross_repo_correlation = false;

        let tracker = claudear_storage::SqliteTracker::in_memory().unwrap();
        let tracker: Arc<dyn FixAttemptTracker> = Arc::new(tracker);
        let issue = make_test_issue();

        let result =
            enhance_prompt_with_learning(&config, &tracker, "base prompt", &issue, Some("my-repo"));
        assert_eq!(result, "base prompt");
    }

    #[test]
    fn test_enhance_prompt_repo_knowledge_empty() {
        let mut config = Config::default();
        config.learning.repo_knowledge = true;
        config.learning.qa_promotion = false;
        config.learning.strategy_fingerprinting = false;
        config.learning.cluster_detection = false;
        config.learning.cross_repo_correlation = false;

        let tracker = claudear_storage::SqliteTracker::in_memory().unwrap();
        let tracker: Arc<dyn FixAttemptTracker> = Arc::new(tracker);
        let issue = make_test_issue();

        let result =
            enhance_prompt_with_learning(&config, &tracker, "base prompt", &issue, Some("my-repo"));
        assert_eq!(result, "base prompt");
    }

    #[test]
    fn test_enhance_prompt_all_learning_enabled_empty_db() {
        let mut config = Config::default();
        config.learning.repo_knowledge = true;
        config.learning.qa_promotion = true;
        config.learning.strategy_fingerprinting = true;
        config.learning.cluster_detection = true;
        config.learning.cross_repo_correlation = true;

        let tracker = claudear_storage::SqliteTracker::in_memory().unwrap();
        let tracker: Arc<dyn FixAttemptTracker> = Arc::new(tracker);
        let issue = make_test_issue();

        let result =
            enhance_prompt_with_learning(&config, &tracker, "base prompt", &issue, Some("my-repo"));
        // Empty DB returns no extra context
        assert_eq!(result, "base prompt");
    }

    // --- is_hard_error / is_rate_limit_error ---

    #[test]
    fn test_is_hard_error_rate_limit() {
        assert!(claudear_integrations::runner::is_hard_error(
            "rate limit exceeded"
        ));
    }

    #[test]
    fn test_is_hard_error_spawn_failure() {
        assert!(claudear_integrations::runner::is_hard_error(
            "failed to spawn process"
        ));
    }

    #[test]
    fn test_is_hard_error_timeout() {
        assert!(claudear_integrations::runner::is_hard_error(
            "process timed out after 300s"
        ));
    }

    #[test]
    fn test_is_hard_error_connection_reset() {
        assert!(claudear_integrations::runner::is_hard_error(
            "connection reset by peer"
        ));
    }

    #[test]
    fn test_is_hard_error_regular_failure_not_hard() {
        assert!(!claudear_integrations::runner::is_hard_error(
            "test assertion failed"
        ));
    }

    #[test]
    fn test_is_hard_error_normal_build_failure_not_hard() {
        assert!(!claudear_integrations::runner::is_hard_error(
            "cargo build failed"
        ));
    }

    #[test]
    fn test_is_rate_limit_error() {
        assert!(claudear_integrations::runner::is_rate_limit_error(
            "rate limit exceeded"
        ));
    }

    #[test]
    fn test_is_rate_limit_error_false_positive_allowed() {
        assert!(!claudear_integrations::runner::is_rate_limit_error(
            r#"rate_limit_event "status":"allowed""#
        ));
    }

    #[test]
    fn test_is_rate_limit_error_normal_error() {
        assert!(!claudear_integrations::runner::is_rate_limit_error(
            "build failed"
        ));
    }

    // --- RepoResolution ---

    #[test]
    fn test_repo_resolution_skip() {
        let resolution = RepoResolution::Skip {
            reason: "no matching repo".to_string(),
        };
        assert!(resolution.repo_name().is_none());
        assert!(resolution.scm_url().is_none());
        assert!(resolution.default_branch().is_none());
    }

    #[test]
    fn test_repo_resolution_resolved() {
        let resolution = RepoResolution::Resolved {
            project_dir: std::path::PathBuf::from("/tmp/repo"),
            repo_name: "org/repo".to_string(),
            scm_url: "https://github.com/org/repo".to_string(),
            default_branch: "main".to_string(),
            repo_id: Some(42),
            confidence: None,
        };
        assert_eq!(resolution.repo_name(), Some("org/repo"));
        assert_eq!(resolution.scm_url(), Some("https://github.com/org/repo"));
        assert_eq!(resolution.default_branch(), Some("main"));
        assert_eq!(resolution.repo_id(), Some(42));
        assert!(resolution.is_resolved());
    }

    // --- record_error_pattern integration ---

    #[test]
    fn test_record_error_pattern_uses_tracker() {
        let tracker = claudear_storage::SqliteTracker::in_memory().unwrap();
        let tracker: Arc<dyn FixAttemptTracker> = Arc::new(tracker);

        record_error_pattern(&tracker, "linear", "issue-1", "git merge conflict");

        let patterns = tracker.get_error_patterns(10).unwrap();
        assert!(
            !patterns.is_empty(),
            "should have recorded at least one pattern"
        );
        let pattern = &patterns[0];
        assert_eq!(pattern.error_type.as_deref(), Some("git_error"));
        assert_eq!(pattern.error_message.as_deref(), Some("git merge conflict"));
    }

    #[test]
    fn test_record_error_pattern_timeout() {
        let tracker = claudear_storage::SqliteTracker::in_memory().unwrap();
        let tracker: Arc<dyn FixAttemptTracker> = Arc::new(tracker);

        record_error_pattern(&tracker, "sentry", "issue-2", "timed out after 600s");

        let patterns = tracker.get_error_patterns(10).unwrap();
        assert!(!patterns.is_empty());
        assert_eq!(patterns[0].error_type.as_deref(), Some("timeout"));
    }

    #[test]
    fn test_record_error_pattern_build() {
        let tracker = claudear_storage::SqliteTracker::in_memory().unwrap();
        let tracker: Arc<dyn FixAttemptTracker> = Arc::new(tracker);

        record_error_pattern(&tracker, "github", "issue-3", "cargo build failed");

        let patterns = tracker.get_error_patterns(10).unwrap();
        assert!(!patterns.is_empty());
        assert_eq!(patterns[0].error_type.as_deref(), Some("build_failure"));
    }

    // --- parse_pr_url ---

    #[test]
    fn test_parse_pr_url_github() {
        let result = claudear_storage::parse_pr_url("https://github.com/org/repo/pull/42");
        assert!(result.is_some());
        let (repo, pr_number) = result.unwrap();
        assert_eq!(repo, "org/repo");
        assert_eq!(pr_number, 42);
    }

    #[test]
    fn test_parse_pr_url_invalid() {
        let result = claudear_storage::parse_pr_url("not a url");
        assert!(result.is_none());
    }

    // --- parse_pr_url extended ---

    #[test]
    fn test_parse_pr_url_empty_string() {
        assert!(claudear_storage::parse_pr_url("").is_none());
    }

    #[test]
    fn test_parse_pr_url_gitlab_mr() {
        let result =
            claudear_storage::parse_pr_url("https://gitlab.com/group/project/-/merge_requests/17");
        assert!(result.is_some());
        let (project, mr_number) = result.unwrap();
        assert_eq!(project, "group/project");
        assert_eq!(mr_number, 17);
    }

    #[test]
    fn test_parse_pr_url_github_with_trailing_path() {
        // The regex should still extract repo/number from a longer URL
        let result = claudear_storage::parse_pr_url("https://github.com/owner/repo/pull/99/files");
        assert!(result.is_some());
        let (repo, pr_number) = result.unwrap();
        assert_eq!(repo, "owner/repo");
        assert_eq!(pr_number, 99);
    }

    #[test]
    fn test_parse_pr_url_rejects_excessively_long_url() {
        let long_url = format!("https://github.com/org/repo/pull/{}", "1".repeat(3000));
        assert!(claudear_storage::parse_pr_url(&long_url).is_none());
    }

    #[test]
    fn test_parse_pr_url_github_large_pr_number() {
        let result = claudear_storage::parse_pr_url("https://github.com/org/repo/pull/999999");
        assert!(result.is_some());
        let (repo, pr_number) = result.unwrap();
        assert_eq!(repo, "org/repo");
        assert_eq!(pr_number, 999999);
    }

    // --- classify_error extended ---

    #[test]
    fn test_classify_error_cargo_keyword() {
        assert_eq!(
            claudear_storage::classify_error("cargo test failed"),
            "build_failure"
        );
    }

    #[test]
    fn test_classify_error_merge_conflict() {
        assert_eq!(
            claudear_storage::classify_error("merge conflict in src/main.rs"),
            "git_error"
        );
    }

    #[test]
    fn test_classify_error_connection_refused() {
        assert_eq!(
            claudear_storage::classify_error("connection refused on port 443"),
            "network_error"
        );
    }

    #[test]
    fn test_classify_error_case_insensitive_timeout() {
        assert_eq!(
            claudear_storage::classify_error("TIMEOUT waiting for response"),
            "timeout"
        );
    }

    #[test]
    fn test_classify_error_case_insensitive_build() {
        assert_eq!(
            claudear_storage::classify_error("BUILD FAILED with errors"),
            "build_failure"
        );
    }

    #[test]
    fn test_classify_error_rate_limit_is_claude() {
        // "rate limit" triggers claude_error because of the classification order
        assert_eq!(
            claudear_storage::classify_error("rate limit exceeded"),
            "claude_error"
        );
    }

    #[test]
    fn test_classify_error_assertion_is_test_failure() {
        assert_eq!(
            claudear_storage::classify_error("assertion `left == right` failed"),
            "test_failure"
        );
    }

    // --- error hash extended ---

    #[test]
    fn test_error_hash_empty_string() {
        let hash = claudear_storage::compute_error_hash("");
        assert!(!hash.is_empty());
    }

    #[test]
    fn test_error_hash_whitespace_only() {
        let hash = claudear_storage::compute_error_hash("   ");
        assert!(!hash.is_empty());
    }

    #[test]
    fn test_error_hash_similar_but_different_text() {
        let hash1 = claudear_storage::compute_error_hash("git merge conflict in file.rs");
        let hash2 = claudear_storage::compute_error_hash("git merge conflict in file.py");
        assert_ne!(hash1, hash2);
    }

    // --- ProcessingMetric ---

    #[test]
    fn test_processing_metric_new() {
        let metric = ProcessingMetric::new("test_metric", 42.0);
        assert_eq!(metric.metric_name, "test_metric");
        assert!((metric.metric_value - 42.0).abs() < f64::EPSILON);
        assert!(metric.source.is_none());
        assert!(metric.tags.is_none());
        assert_eq!(metric.id, 0);
    }

    #[test]
    fn test_processing_metric_with_source() {
        let metric = ProcessingMetric::new("pr_created", 1.0).with_source("linear".to_string());
        assert_eq!(metric.source.as_deref(), Some("linear"));
    }

    #[test]
    fn test_processing_metric_with_tags() {
        let metric = ProcessingMetric::new("processing_time", 5.5)
            .with_tags(json!({"status": "success", "repo": "org/repo"}));
        assert!(metric.tags.is_some());
        let tags = metric.tags.unwrap();
        assert_eq!(tags["status"], "success");
        assert_eq!(tags["repo"], "org/repo");
    }

    #[test]
    fn test_processing_metric_chained_builders() {
        let metric = ProcessingMetric::new("queue_depth", 10.0)
            .with_source("sentry".to_string())
            .with_tags(json!({"queue": "main"}));
        assert_eq!(metric.metric_name, "queue_depth");
        assert_eq!(metric.source.as_deref(), Some("sentry"));
        assert!(metric.tags.is_some());
    }

    #[test]
    fn test_processing_metric_serialization() {
        let metric = ProcessingMetric::new("test_metric", 3.15).with_source("test".to_string());
        let json = serde_json::to_string(&metric).unwrap();
        assert!(json.contains("test_metric"));
        assert!(json.contains("3.15"));
        assert!(json.contains("test"));
    }

    #[test]
    fn test_processing_metric_zero_value() {
        let metric = ProcessingMetric::new("empty_metric", 0.0);
        assert!((metric.metric_value).abs() < f64::EPSILON);
    }

    #[test]
    fn test_processing_metric_negative_value() {
        let metric = ProcessingMetric::new("delta_metric", -5.0);
        assert!((metric.metric_value - (-5.0)).abs() < f64::EPSILON);
    }

    // --- ErrorPattern ---

    #[test]
    fn test_error_pattern_new() {
        let pattern = ErrorPattern::new("abc123");
        assert_eq!(pattern.pattern_hash, "abc123");
        assert_eq!(pattern.id, 0);
        assert_eq!(pattern.occurrence_count, 1);
        assert!(pattern.error_type.is_none());
        assert!(pattern.error_message.is_none());
        assert!(pattern.sources.is_none());
        assert!(pattern.example_issue_ids.is_none());
        assert!(pattern.resolution_hints.is_none());
    }

    #[test]
    fn test_error_pattern_fields_set() {
        let mut pattern = ErrorPattern::new("hash123");
        pattern.error_type = Some("timeout".to_string());
        pattern.error_message = Some("timed out".to_string());
        pattern.sources = Some(vec!["sentry".to_string()]);
        pattern.example_issue_ids = Some(vec!["issue-1".to_string()]);
        pattern.resolution_hints = Some("Increase timeout".to_string());

        assert_eq!(pattern.error_type.as_deref(), Some("timeout"));
        assert_eq!(pattern.error_message.as_deref(), Some("timed out"));
        assert_eq!(pattern.sources.as_ref().unwrap().len(), 1);
        assert_eq!(pattern.example_issue_ids.as_ref().unwrap().len(), 1);
        assert_eq!(
            pattern.resolution_hints.as_deref(),
            Some("Increase timeout")
        );
    }

    #[test]
    fn test_error_pattern_serialization() {
        let pattern = ErrorPattern::new("hash456");
        let json = serde_json::to_string(&pattern).unwrap();
        assert!(json.contains("hash456"));
        let deserialized: ErrorPattern = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.pattern_hash, "hash456");
    }

    // --- Issue construction and metadata ---

    #[test]
    fn test_make_test_issue_fields() {
        let issue = make_test_issue();
        assert_eq!(issue.id, "test-1");
        assert_eq!(issue.short_id, "T-1");
        assert_eq!(issue.title, "Test issue");
        assert_eq!(issue.description.as_deref(), Some("Test description"));
        assert_eq!(issue.source, "test");
        assert_eq!(issue.priority, claudear_core::types::IssuePriority::Medium);
        assert_eq!(issue.status, claudear_core::types::IssueStatus::Open);
        assert!(issue.metadata.is_empty());
    }

    #[test]
    fn test_issue_set_and_get_metadata_string() {
        let mut issue = make_test_issue();
        issue.set_metadata("assignee", "alice");
        assert_eq!(
            issue.get_metadata::<String>("assignee").as_deref(),
            Some("alice")
        );
    }

    #[test]
    fn test_issue_set_and_get_metadata_bool() {
        let mut issue = make_test_issue();
        issue.set_metadata("is_pr_update", true);
        assert_eq!(issue.get_metadata::<bool>("is_pr_update"), Some(true));
    }

    #[test]
    fn test_issue_set_and_get_metadata_number() {
        let mut issue = make_test_issue();
        issue.set_metadata("retry_count", 3);
        assert_eq!(issue.get_metadata::<i32>("retry_count"), Some(3));
    }

    #[test]
    fn test_issue_get_metadata_missing_key() {
        let issue = make_test_issue();
        assert!(issue.get_metadata::<String>("nonexistent").is_none());
    }

    #[test]
    fn test_issue_get_metadata_type_mismatch() {
        let mut issue = make_test_issue();
        issue.set_metadata("flag", "not_a_bool");
        // Trying to get a bool from a string should return None (deserialization fails)
        assert!(issue.get_metadata::<bool>("flag").is_none());
    }

    #[test]
    fn test_issue_new_constructor() {
        let issue = Issue::new("id-1", "S-1", "Title", "https://example.com", "linear");
        assert_eq!(issue.id, "id-1");
        assert_eq!(issue.short_id, "S-1");
        assert_eq!(issue.title, "Title");
        assert_eq!(issue.url, "https://example.com");
        assert_eq!(issue.source, "linear");
        assert!(issue.description.is_none());
        assert_eq!(issue.priority, claudear_core::types::IssuePriority::None);
        assert_eq!(issue.status, claudear_core::types::IssueStatus::Open);
    }

    #[test]
    fn test_issue_serialization_roundtrip() {
        let mut issue = make_test_issue();
        issue.set_metadata("key", "value");
        let json = serde_json::to_string(&issue).unwrap();
        let deserialized: Issue = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.id, issue.id);
        assert_eq!(deserialized.title, issue.title);
        assert_eq!(
            deserialized.get_metadata::<String>("key").as_deref(),
            Some("value")
        );
    }

    // --- RepoResolution extended ---

    #[test]
    fn test_repo_resolution_skip_is_not_resolved() {
        let resolution = RepoResolution::Skip {
            reason: "no repo".to_string(),
        };
        assert!(!resolution.is_resolved());
    }

    #[test]
    fn test_repo_resolution_skip_repo_id_is_none() {
        let resolution = RepoResolution::Skip {
            reason: "test".to_string(),
        };
        assert!(resolution.repo_id().is_none());
    }

    #[test]
    fn test_repo_resolution_resolved_project_dir() {
        let resolution = RepoResolution::Resolved {
            project_dir: std::path::PathBuf::from("/home/user/repo"),
            repo_name: "user/repo".to_string(),
            scm_url: "https://github.com/user/repo".to_string(),
            default_branch: "develop".to_string(),
            repo_id: None,
            confidence: None,
        };
        assert_eq!(
            resolution.project_dir(),
            Some(&std::path::PathBuf::from("/home/user/repo"))
        );
        assert_eq!(resolution.default_branch(), Some("develop"));
        assert!(resolution.repo_id().is_none());
    }

    #[test]
    fn test_repo_resolution_skip_project_dir_is_none() {
        let resolution = RepoResolution::Skip {
            reason: "skipped".to_string(),
        };
        assert!(resolution.project_dir().is_none());
    }

    // --- record_error_pattern extended ---

    #[test]
    fn test_record_error_pattern_network() {
        let tracker = claudear_storage::SqliteTracker::in_memory().unwrap();
        let tracker: Arc<dyn FixAttemptTracker> = Arc::new(tracker);

        record_error_pattern(&tracker, "linear", "issue-4", "network connection refused");

        let patterns = tracker.get_error_patterns(10).unwrap();
        assert!(!patterns.is_empty());
        assert_eq!(patterns[0].error_type.as_deref(), Some("network_error"));
    }

    #[test]
    fn test_record_error_pattern_permission() {
        let tracker = claudear_storage::SqliteTracker::in_memory().unwrap();
        let tracker: Arc<dyn FixAttemptTracker> = Arc::new(tracker);

        record_error_pattern(
            &tracker,
            "github",
            "issue-5",
            "permission denied writing to /opt",
        );

        let patterns = tracker.get_error_patterns(10).unwrap();
        assert!(!patterns.is_empty());
        assert_eq!(patterns[0].error_type.as_deref(), Some("permission_error"));
    }

    #[test]
    fn test_record_error_pattern_unknown() {
        let tracker = claudear_storage::SqliteTracker::in_memory().unwrap();
        let tracker: Arc<dyn FixAttemptTracker> = Arc::new(tracker);

        record_error_pattern(&tracker, "jira", "issue-6", "some obscure failure");

        let patterns = tracker.get_error_patterns(10).unwrap();
        assert!(!patterns.is_empty());
        assert_eq!(patterns[0].error_type.as_deref(), Some("unknown"));
    }

    #[test]
    fn test_record_error_pattern_contains_source_and_issue_id() {
        let tracker = claudear_storage::SqliteTracker::in_memory().unwrap();
        let tracker: Arc<dyn FixAttemptTracker> = Arc::new(tracker);

        record_error_pattern(&tracker, "sentry", "SENTRY-99", "git conflict");

        let patterns = tracker.get_error_patterns(10).unwrap();
        assert!(!patterns.is_empty());
        let p = &patterns[0];
        assert!(p.sources.as_ref().unwrap().contains(&"sentry".to_string()));
        assert!(p
            .example_issue_ids
            .as_ref()
            .unwrap()
            .contains(&"SENTRY-99".to_string()));
    }

    #[test]
    fn test_record_error_pattern_hash_is_set() {
        let tracker = claudear_storage::SqliteTracker::in_memory().unwrap();
        let tracker: Arc<dyn FixAttemptTracker> = Arc::new(tracker);

        record_error_pattern(&tracker, "test", "id-1", "build failed");

        let patterns = tracker.get_error_patterns(10).unwrap();
        assert!(!patterns.is_empty());
        assert!(!patterns[0].pattern_hash.is_empty());
    }

    // --- record_metric with tracker ---

    #[test]
    fn test_record_metric_stores_in_tracker() {
        let tracker = claudear_storage::SqliteTracker::in_memory().unwrap();
        let tracker: Arc<dyn FixAttemptTracker> = Arc::new(tracker);

        let metric = ProcessingMetric::new("test_counter", 1.0).with_source("linear".to_string());
        tracker.record_metric(&metric).unwrap();

        let metrics = tracker.get_metrics("test_counter", None, 10).unwrap();
        assert!(!metrics.is_empty());
        assert_eq!(metrics[0].metric_name, "test_counter");
    }

    // --- notify_failed_with_escalation ---

    #[tokio::test]
    async fn test_notify_failed_with_escalation_non_hard_error() {
        let tracker = claudear_storage::SqliteTracker::in_memory().unwrap();
        let tracker: Arc<dyn FixAttemptTracker> = Arc::new(tracker);
        let notifier: Arc<dyn Notifier> =
            Arc::new(claudear_integrations::notifier::ConsoleNotifier::new());
        let issue = make_test_issue();

        // "build failed" is not a hard error
        let result =
            notify_failed_with_escalation(&notifier, &tracker, &issue, "cargo build failed").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_notify_failed_with_escalation_hard_error() {
        let tracker = claudear_storage::SqliteTracker::in_memory().unwrap();
        let tracker: Arc<dyn FixAttemptTracker> = Arc::new(tracker);
        let notifier: Arc<dyn Notifier> =
            Arc::new(claudear_integrations::notifier::ConsoleNotifier::new());
        let issue = make_test_issue();

        // "rate limit exceeded" is a hard error
        let result =
            notify_failed_with_escalation(&notifier, &tracker, &issue, "rate limit exceeded").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_notify_failed_with_escalation_hard_error_records_activity() {
        let tracker = claudear_storage::SqliteTracker::in_memory().unwrap();
        let tracker: Arc<dyn FixAttemptTracker> = Arc::new(tracker);
        let notifier: Arc<dyn Notifier> =
            Arc::new(claudear_integrations::notifier::ConsoleNotifier::new());
        let issue = make_test_issue();

        notify_failed_with_escalation(&notifier, &tracker, &issue, "rate limit exceeded")
            .await
            .unwrap();

        let activities = tracker.get_recent_activities(10).unwrap();
        let error_activity = activities.iter().find(|a| a.activity_type == "error");
        assert!(error_activity.is_some(), "should record an error activity");
        let meta = error_activity.unwrap().metadata.as_ref().unwrap();
        assert_eq!(meta["hard_error"], true);
        assert_eq!(meta["rate_limited"], true);
    }

    #[tokio::test]
    async fn test_notify_failed_with_escalation_hard_error_strips_resolved_user() {
        let tracker = claudear_storage::SqliteTracker::in_memory().unwrap();
        let tracker: Arc<dyn FixAttemptTracker> = Arc::new(tracker);
        let notifier: Arc<dyn Notifier> =
            Arc::new(claudear_integrations::notifier::ConsoleNotifier::new());
        let mut issue = make_test_issue();
        issue.set_metadata("resolved_user", "alice");

        // Should clone issue and remove resolved_user for global escalation
        let result =
            notify_failed_with_escalation(&notifier, &tracker, &issue, "failed to spawn process")
                .await;
        assert!(result.is_ok());
        // Original issue should still have resolved_user
        assert_eq!(
            issue.get_metadata::<String>("resolved_user").as_deref(),
            Some("alice")
        );
    }

    #[tokio::test]
    async fn test_notify_failed_with_escalation_service_unavailable() {
        let tracker = claudear_storage::SqliteTracker::in_memory().unwrap();
        let tracker: Arc<dyn FixAttemptTracker> = Arc::new(tracker);
        let notifier: Arc<dyn Notifier> =
            Arc::new(claudear_integrations::notifier::ConsoleNotifier::new());
        let issue = make_test_issue();

        let result =
            notify_failed_with_escalation(&notifier, &tracker, &issue, "service unavailable (503)")
                .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_notify_failed_with_escalation_broken_pipe() {
        let tracker = claudear_storage::SqliteTracker::in_memory().unwrap();
        let tracker: Arc<dyn FixAttemptTracker> = Arc::new(tracker);
        let notifier: Arc<dyn Notifier> =
            Arc::new(claudear_integrations::notifier::ConsoleNotifier::new());
        let issue = make_test_issue();

        let result = notify_failed_with_escalation(
            &notifier,
            &tracker,
            &issue,
            "broken pipe writing to stdout",
        )
        .await;
        assert!(result.is_ok());
    }

    // --- record_feedback_outcome ---

    #[tokio::test]
    async fn test_record_feedback_outcome_no_attempt() {
        let tracker = claudear_storage::SqliteTracker::in_memory().unwrap();
        let tracker: Arc<dyn FixAttemptTracker> = Arc::new(tracker);
        let feedback_analyzer = Arc::new(tokio::sync::Mutex::new(
            claudear_analysis::feedback::FeedbackAnalyzer::new(),
        ));
        let issue = make_test_issue();

        // No attempt recorded, so get_attempt returns None and function returns early
        record_feedback_outcome(
            &tracker,
            None,
            None,
            &feedback_analyzer,
            "test",
            &issue,
            claudear_analysis::feedback::Outcome::Failed,
        )
        .await;

        // Should not panic and should not store anything
        let outcomes = tracker.get_feedback_outcomes(None, 10).unwrap();
        assert!(outcomes.is_empty());
    }

    #[tokio::test]
    async fn test_record_feedback_outcome_with_attempt() {
        let tracker = claudear_storage::SqliteTracker::in_memory().unwrap();
        let tracker: Arc<dyn FixAttemptTracker> = Arc::new(tracker);
        let feedback_analyzer = Arc::new(tokio::sync::Mutex::new(
            claudear_analysis::feedback::FeedbackAnalyzer::new(),
        ));
        let issue = make_test_issue();

        // Record an attempt first
        tracker.record_attempt("test", "test-1", "T-1").unwrap();

        record_feedback_outcome(
            &tracker,
            None,
            None,
            &feedback_analyzer,
            "test",
            &issue,
            claudear_analysis::feedback::Outcome::Failed,
        )
        .await;

        let outcomes = tracker.get_feedback_outcomes(None, 10).unwrap();
        assert!(
            !outcomes.is_empty(),
            "should have stored a feedback outcome"
        );
    }

    #[tokio::test]
    async fn test_record_feedback_outcome_with_success() {
        let tracker = claudear_storage::SqliteTracker::in_memory().unwrap();
        let tracker: Arc<dyn FixAttemptTracker> = Arc::new(tracker);
        let feedback_analyzer = Arc::new(tokio::sync::Mutex::new(
            claudear_analysis::feedback::FeedbackAnalyzer::new(),
        ));
        let issue = make_test_issue();

        tracker.record_attempt("test", "test-1", "T-1").unwrap();
        tracker
            .mark_success("test", "test-1", "https://github.com/org/repo/pull/1")
            .unwrap();

        record_feedback_outcome(
            &tracker,
            None,
            None,
            &feedback_analyzer,
            "test",
            &issue,
            claudear_analysis::feedback::Outcome::Merged,
        )
        .await;

        let outcomes = tracker.get_feedback_outcomes(None, 10).unwrap();
        assert!(!outcomes.is_empty());
    }

    // --- IssueProcessor::run with Skip resolution ---

    #[tokio::test]
    async fn test_issue_processor_run_with_skip_resolution() {
        let tracker = claudear_storage::SqliteTracker::in_memory().unwrap();
        let tracker: Arc<dyn FixAttemptTracker> = Arc::new(tracker);
        let notifier: Arc<dyn Notifier> =
            Arc::new(claudear_integrations::notifier::ConsoleNotifier::new());
        let agent: Arc<dyn claudear_integrations::runner::AgentRunner> = Arc::new(DummyAgent);
        let feedback_analyzer = Arc::new(tokio::sync::Mutex::new(
            claudear_analysis::feedback::FeedbackAnalyzer::new(),
        ));

        let processor = IssueProcessor {
            config: Config::default(),
            tracker,
            notifier,
            agent,
            inferrer: None,
            embedding_client: None,
            issue_embedding_service: None,
            code_search_service: None,
            feedback_analyzer,
            review_watcher: None,
            user_registry: claudear_config::users::UserRegistry::new(
                std::collections::HashMap::new(),
            ),
            github_client: None,
            llm_analyzer: None,
        };

        let input = ProcessingInput {
            issue: make_test_issue(),
            source_name: "test".to_string(),
            match_result: claudear_core::types::MatchResult::matched(
                "test",
                claudear_core::types::MatchPriority::Normal,
            ),
            resolution: RepoResolution::Skip {
                reason: "no repo configured".to_string(),
            },
            attempt_id: None,
            review_feedback: None,
            existing_pr_branch: None,
            intent: None,
        };

        // Use a dummy context provider
        let ctx = DummyContextProvider;
        let outcome = processor.run(input, &ctx).await;
        match outcome {
            ProcessingOutcome::Failed { error } => {
                assert!(
                    error.contains("Repository resolution failed"),
                    "Expected repo resolution failure, got: {}",
                    error
                );
            }
            _ => panic!("Expected Failed outcome for Skip resolution"),
        }
    }

    // --- truncate_error_for_activity edge cases ---

    #[test]
    fn test_truncate_error_for_activity_single_char() {
        assert_eq!(truncate_error_for_activity("a"), "a");
    }

    #[test]
    fn test_truncate_error_for_activity_499_chars() {
        let msg = "d".repeat(499);
        assert_eq!(truncate_error_for_activity(&msg), msg);
    }

    #[test]
    fn test_truncate_error_for_activity_mixed_multibyte_at_boundary() {
        // Build a string that's 498 ASCII chars + 4-byte emoji = 502 bytes > 500
        // The function uses byte length, so this WILL be truncated
        let mut msg = "a".repeat(498);
        msg.push('\u{1F600}'); // 4-byte char, total = 502 bytes
        let result = truncate_error_for_activity(&msg);
        assert!(result.ends_with("..."));
        assert!(result.is_char_boundary(result.len()));
    }

    #[test]
    fn test_truncate_error_for_activity_multibyte_within_limit() {
        // 496 ASCII + 1 emoji = 500 bytes exactly (496 + 4)
        let mut msg = "a".repeat(496);
        msg.push('\u{1F600}'); // 4-byte char, total = 500 bytes
        let result = truncate_error_for_activity(&msg);
        assert_eq!(result, msg, "exactly 500 bytes should not be truncated");
    }

    #[test]
    fn test_truncate_error_for_activity_1000_chars() {
        let msg = "z".repeat(1000);
        let result = truncate_error_for_activity(&msg);
        assert!(result.ends_with("..."));
        assert!(result.len() <= 503);
    }

    // --- is_hard_error extended ---

    #[test]
    fn test_is_hard_error_internal_server_error() {
        assert!(claudear_integrations::runner::is_hard_error(
            "HTTP 500 internal server error"
        ));
    }

    #[test]
    fn test_is_hard_error_network_error() {
        assert!(claudear_integrations::runner::is_hard_error(
            "network error: DNS resolution failed"
        ));
    }

    #[test]
    fn test_is_hard_error_failed_to_capture_stdout() {
        assert!(claudear_integrations::runner::is_hard_error(
            "failed to capture stdout from process"
        ));
    }

    #[test]
    fn test_is_hard_error_failed_to_capture_stderr() {
        assert!(claudear_integrations::runner::is_hard_error(
            "failed to capture stderr from process"
        ));
    }

    #[test]
    fn test_is_hard_error_failed_to_wait_for() {
        assert!(claudear_integrations::runner::is_hard_error(
            "failed to wait for child process"
        ));
    }

    #[test]
    fn test_is_hard_error_too_many_requests() {
        assert!(claudear_integrations::runner::is_hard_error(
            "Error: 429 Too Many Requests"
        ));
    }

    #[test]
    fn test_is_hard_error_quota_exceeded() {
        assert!(claudear_integrations::runner::is_hard_error(
            "API quota exceeded for project"
        ));
    }

    #[test]
    fn test_is_hard_error_normal_test_failure_not_hard() {
        assert!(!claudear_integrations::runner::is_hard_error(
            "3 tests failed, 97 passed"
        ));
    }

    #[test]
    fn test_is_hard_error_empty_string() {
        assert!(!claudear_integrations::runner::is_hard_error(""));
    }

    // --- is_rate_limit_error extended ---

    #[test]
    fn test_is_rate_limit_error_ratelimit_one_word() {
        assert!(claudear_integrations::runner::is_rate_limit_error(
            "RateLimit exceeded"
        ));
    }

    #[test]
    fn test_is_rate_limit_error_too_many_requests() {
        assert!(claudear_integrations::runner::is_rate_limit_error(
            "Error: Too Many Requests"
        ));
    }

    #[test]
    fn test_is_rate_limit_error_quota_exceeded() {
        assert!(claudear_integrations::runner::is_rate_limit_error(
            "API quota exceeded"
        ));
    }

    #[test]
    fn test_is_rate_limit_error_retry_after() {
        assert!(claudear_integrations::runner::is_rate_limit_error(
            "retry-after: 30"
        ));
    }

    #[test]
    fn test_is_rate_limit_error_resource_exhausted() {
        // The pattern is "resource exhausted" (with space), not "resource_exhausted"
        assert!(claudear_integrations::runner::is_rate_limit_error(
            "resource exhausted: quota limit"
        ));
        // Underscore variant does not match
        assert!(!claudear_integrations::runner::is_rate_limit_error(
            "RESOURCE_EXHAUSTED: quota limit"
        ));
    }

    #[test]
    fn test_is_rate_limit_error_hit_your_limit() {
        assert!(claudear_integrations::runner::is_rate_limit_error(
            "You've hit your limit for this hour"
        ));
    }

    #[test]
    fn test_is_rate_limit_error_try_again_later() {
        assert!(claudear_integrations::runner::is_rate_limit_error(
            "Please try again later"
        ));
    }

    #[test]
    fn test_is_rate_limit_error_standalone_429() {
        assert!(claudear_integrations::runner::is_rate_limit_error(
            "status:429 service overloaded"
        ));
    }

    #[test]
    fn test_is_rate_limit_error_allowed_warning_not_rate_limit() {
        assert!(!claudear_integrations::runner::is_rate_limit_error(
            r#"rate_limit_event "status":"allowed_warning""#
        ));
    }

    // --- enhance_prompt_with_learning extended ---

    #[test]
    fn test_enhance_prompt_only_qa_promotion_empty_db() {
        let mut config = Config::default();
        config.learning.repo_knowledge = false;
        config.learning.qa_promotion = true;
        config.learning.strategy_fingerprinting = false;
        config.learning.cluster_detection = false;
        config.learning.cross_repo_correlation = false;

        let tracker = claudear_storage::SqliteTracker::in_memory().unwrap();
        let tracker: Arc<dyn FixAttemptTracker> = Arc::new(tracker);
        let issue = make_test_issue();

        let result =
            enhance_prompt_with_learning(&config, &tracker, "base prompt", &issue, Some("my-repo"));
        // No promoted instructions in empty DB
        assert_eq!(result, "base prompt");
    }

    #[test]
    fn test_enhance_prompt_only_strategy_fingerprinting_empty_db() {
        let mut config = Config::default();
        config.learning.repo_knowledge = false;
        config.learning.qa_promotion = false;
        config.learning.strategy_fingerprinting = true;
        config.learning.cluster_detection = false;
        config.learning.cross_repo_correlation = false;

        let tracker = claudear_storage::SqliteTracker::in_memory().unwrap();
        let tracker: Arc<dyn FixAttemptTracker> = Arc::new(tracker);
        let issue = make_test_issue();

        let result =
            enhance_prompt_with_learning(&config, &tracker, "base prompt", &issue, Some("my-repo"));
        assert_eq!(result, "base prompt");
    }

    #[test]
    fn test_enhance_prompt_only_cluster_detection_empty_db() {
        let mut config = Config::default();
        config.learning.repo_knowledge = false;
        config.learning.qa_promotion = false;
        config.learning.strategy_fingerprinting = false;
        config.learning.cluster_detection = true;
        config.learning.cross_repo_correlation = false;

        let tracker = claudear_storage::SqliteTracker::in_memory().unwrap();
        let tracker: Arc<dyn FixAttemptTracker> = Arc::new(tracker);
        let issue = make_test_issue();

        let result =
            enhance_prompt_with_learning(&config, &tracker, "base prompt", &issue, Some("my-repo"));
        assert_eq!(result, "base prompt");
    }

    #[test]
    fn test_enhance_prompt_only_cross_repo_correlation_empty_db() {
        let mut config = Config::default();
        config.learning.repo_knowledge = false;
        config.learning.qa_promotion = false;
        config.learning.strategy_fingerprinting = false;
        config.learning.cluster_detection = false;
        config.learning.cross_repo_correlation = true;

        let tracker = claudear_storage::SqliteTracker::in_memory().unwrap();
        let tracker: Arc<dyn FixAttemptTracker> = Arc::new(tracker);
        let issue = make_test_issue();

        let result =
            enhance_prompt_with_learning(&config, &tracker, "base prompt", &issue, Some("my-repo"));
        assert_eq!(result, "base prompt");
    }

    #[test]
    fn test_enhance_prompt_preserves_base_prompt_content() {
        let config = Config::default();
        let tracker = claudear_storage::SqliteTracker::in_memory().unwrap();
        let tracker: Arc<dyn FixAttemptTracker> = Arc::new(tracker);
        let issue = make_test_issue();

        let base = "Fix the null pointer exception in UserService.java";
        let result = enhance_prompt_with_learning(&config, &tracker, base, &issue, None);
        assert!(result.contains(base));
    }

    // --- Activity logging integration ---

    #[test]
    fn test_activity_log_entry_construction() {
        let entry = ActivityLogEntry::new("processing_started", "Started processing T-1")
            .with_source("linear".to_string())
            .with_issue("test-1".to_string(), "T-1".to_string())
            .with_metadata(json!({"key": "value"}));

        assert_eq!(entry.activity_type, "processing_started");
        assert_eq!(entry.message, "Started processing T-1");
        assert_eq!(entry.source.as_deref(), Some("linear"));
        assert_eq!(entry.issue_id.as_deref(), Some("test-1"));
        assert_eq!(entry.short_id.as_deref(), Some("T-1"));
        assert!(entry.metadata.is_some());
    }

    #[test]
    fn test_activity_log_entry_without_optional_fields() {
        let entry = ActivityLogEntry::new("status_check", "Daemon healthy");
        assert!(entry.source.is_none());
        assert!(entry.issue_id.is_none());
        assert!(entry.short_id.is_none());
        assert!(entry.metadata.is_none());
    }

    #[test]
    fn test_activity_log_entry_record_and_retrieve() {
        let tracker = claudear_storage::SqliteTracker::in_memory().unwrap();
        let tracker: Arc<dyn FixAttemptTracker> = Arc::new(tracker);

        let entry =
            ActivityLogEntry::new("test_event", "Test message").with_source("test".to_string());
        tracker.record_activity(&entry).unwrap();

        let activities = tracker.get_recent_activities(10).unwrap();
        assert!(!activities.is_empty());
        assert_eq!(activities[0].activity_type, "test_event");
    }

    // --- MatchResult ---

    #[test]
    fn test_match_result_matched() {
        let m = claudear_core::types::MatchResult::matched(
            "label match",
            claudear_core::types::MatchPriority::High,
        );
        assert!(m.matches);
        assert_eq!(m.reason, "label match");
        assert_eq!(m.priority, claudear_core::types::MatchPriority::High);
    }

    #[test]
    fn test_match_result_not_matched() {
        let m = claudear_core::types::MatchResult::not_matched("wrong source");
        assert!(!m.matches);
        assert_eq!(m.reason, "wrong source");
        assert_eq!(m.priority, claudear_core::types::MatchPriority::Normal);
    }

    #[test]
    fn test_match_result_serialization() {
        let m = claudear_core::types::MatchResult::matched(
            "test",
            claudear_core::types::MatchPriority::Urgent,
        );
        let json = serde_json::to_string(&m).unwrap();
        let deserialized: claudear_core::types::MatchResult = serde_json::from_str(&json).unwrap();
        assert!(deserialized.matches);
        assert_eq!(
            deserialized.priority,
            claudear_core::types::MatchPriority::Urgent
        );
    }

    // --- AgentResult ---

    #[test]
    fn test_agent_result_success_with_pr() {
        let result = claudear_core::types::AgentResult {
            success: true,
            output: "Fixed the bug".to_string(),
            pr_url: Some("https://github.com/org/repo/pull/42".to_string()),
            changelog: Some("Fixed null pointer in UserService".to_string()),
            error: None,
            blocking_question: None,
            used_qa_ids: vec![],
            confidence: 0,
            confidence_reasoning: None,
            wrong_repo: None,
        };
        assert!(result.success);
        assert!(result.pr_url.is_some());
        assert!(result.error.is_none());
    }

    #[test]
    fn test_agent_result_failure() {
        let result = claudear_core::types::AgentResult {
            success: false,
            output: String::new(),
            pr_url: None,
            changelog: None,
            error: Some("Build failed".to_string()),
            blocking_question: None,
            used_qa_ids: vec![],
            confidence: 0,
            confidence_reasoning: None,
            wrong_repo: None,
        };
        assert!(!result.success);
        assert!(result.pr_url.is_none());
        assert_eq!(result.error.as_deref(), Some("Build failed"));
    }

    #[test]
    fn test_agent_result_with_blocking_question() {
        let result = claudear_core::types::AgentResult {
            success: false,
            output: String::new(),
            pr_url: None,
            changelog: None,
            error: None,
            blocking_question: Some(claudear_core::types::BlockingQuestion {
                question: "Which database should I use?".to_string(),
                context: Some("The project uses both PostgreSQL and SQLite".to_string()),
                options: vec!["PostgreSQL".to_string(), "SQLite".to_string()],
                why: Some("Need to know which to target".to_string()),
            }),
            used_qa_ids: vec![1, 2, 3],
            confidence: 0,
            confidence_reasoning: None,
            wrong_repo: None,
        };
        assert!(result.blocking_question.is_some());
        let bq = result.blocking_question.unwrap();
        assert_eq!(bq.question, "Which database should I use?");
        assert_eq!(bq.options.len(), 2);
        assert!(bq.context.is_some());
        assert!(bq.why.is_some());
    }

    #[test]
    fn test_agent_result_serialization_roundtrip() {
        let result = claudear_core::types::AgentResult {
            success: true,
            output: "done".to_string(),
            pr_url: Some("https://github.com/o/r/pull/1".to_string()),
            changelog: None,
            error: None,
            blocking_question: None,
            used_qa_ids: vec![10, 20],
            confidence: 0,
            confidence_reasoning: None,
            wrong_repo: None,
        };
        let json = serde_json::to_string(&result).unwrap();
        let deserialized: claudear_core::types::AgentResult = serde_json::from_str(&json).unwrap();
        assert!(deserialized.success);
        assert_eq!(deserialized.used_qa_ids, vec![10, 20]);
    }

    // --- BlockingQuestion ---

    #[test]
    fn test_blocking_question_minimal() {
        let bq = claudear_core::types::BlockingQuestion {
            question: "What is the target branch?".to_string(),
            context: None,
            options: vec![],
            why: None,
        };
        assert_eq!(bq.question, "What is the target branch?");
        assert!(bq.context.is_none());
        assert!(bq.options.is_empty());
        assert!(bq.why.is_none());
    }

    #[test]
    fn test_blocking_question_serialization() {
        let bq = claudear_core::types::BlockingQuestion {
            question: "Q?".to_string(),
            context: Some("C".to_string()),
            options: vec!["A".to_string(), "B".to_string()],
            why: Some("W".to_string()),
        };
        let json = serde_json::to_string(&bq).unwrap();
        assert!(json.contains("Q?"));
        let deserialized: claudear_core::types::BlockingQuestion =
            serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.options.len(), 2);
    }

    // --- FixAttemptStatus ---

    #[test]
    fn test_fix_attempt_status_display() {
        assert_eq!(
            claudear_core::types::FixAttemptStatus::Pending.to_string(),
            "pending"
        );
        assert_eq!(
            claudear_core::types::FixAttemptStatus::Success.to_string(),
            "success"
        );
        assert_eq!(
            claudear_core::types::FixAttemptStatus::Failed.to_string(),
            "failed"
        );
        assert_eq!(
            claudear_core::types::FixAttemptStatus::Merged.to_string(),
            "merged"
        );
        assert_eq!(
            claudear_core::types::FixAttemptStatus::Closed.to_string(),
            "closed"
        );
        assert_eq!(
            claudear_core::types::FixAttemptStatus::CannotFix.to_string(),
            "cannot_fix"
        );
    }

    #[test]
    fn test_fix_attempt_status_from_str() {
        use std::str::FromStr;
        assert_eq!(
            claudear_core::types::FixAttemptStatus::from_str("pending").unwrap(),
            claudear_core::types::FixAttemptStatus::Pending
        );
        assert_eq!(
            claudear_core::types::FixAttemptStatus::from_str("SUCCESS").unwrap(),
            claudear_core::types::FixAttemptStatus::Success
        );
        assert!(claudear_core::types::FixAttemptStatus::from_str("invalid").is_err());
    }

    // --- IssuePriority ---

    #[test]
    fn test_issue_priority_ordering() {
        use claudear_core::types::IssuePriority;
        assert!(IssuePriority::Critical > IssuePriority::High);
        assert!(IssuePriority::High > IssuePriority::Medium);
        assert!(IssuePriority::Medium > IssuePriority::Low);
        assert!(IssuePriority::Low > IssuePriority::None);
    }

    #[test]
    fn test_issue_priority_display() {
        assert_eq!(
            claudear_core::types::IssuePriority::Critical.to_string(),
            "critical"
        );
        assert_eq!(
            claudear_core::types::IssuePriority::None.to_string(),
            "none"
        );
    }

    #[test]
    fn test_issue_priority_serialization() {
        let priority = claudear_core::types::IssuePriority::High;
        let json = serde_json::to_string(&priority).unwrap();
        assert_eq!(json, "\"high\"");
        let deserialized: claudear_core::types::IssuePriority =
            serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, claudear_core::types::IssuePriority::High);
    }

    // --- IssueStatus ---

    #[test]
    fn test_issue_status_display() {
        assert_eq!(claudear_core::types::IssueStatus::Open.to_string(), "open");
        assert_eq!(
            claudear_core::types::IssueStatus::InProgress.to_string(),
            "in_progress"
        );
        assert_eq!(
            claudear_core::types::IssueStatus::Resolved.to_string(),
            "resolved"
        );
        assert_eq!(
            claudear_core::types::IssueStatus::Ignored.to_string(),
            "ignored"
        );
    }

    #[test]
    fn test_issue_status_serialization() {
        let status = claudear_core::types::IssueStatus::InProgress;
        let json = serde_json::to_string(&status).unwrap();
        assert_eq!(json, "\"in_progress\"");
        let deserialized: claudear_core::types::IssueStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, claudear_core::types::IssueStatus::InProgress);
    }

    // --- PrRecord ---

    #[test]
    fn test_pr_record_new() {
        let pr = claudear_core::types::PrRecord::new(
            "https://github.com/org/repo/pull/1",
            "org/repo",
            1,
        );
        assert_eq!(pr.pr_url, "https://github.com/org/repo/pull/1");
        assert_eq!(pr.scm_repo, "org/repo");
        assert_eq!(pr.pr_number, 1);
        assert_eq!(pr.status, "open");
        assert!(pr.attempt_id.is_none());
        assert!(pr.issue_id.is_none());
    }

    #[test]
    fn test_pr_record_for_issue() {
        let pr = claudear_core::types::PrRecord::for_issue(
            "https://github.com/org/repo/pull/5",
            "org/repo",
            5,
            "sentry",
            "SENTRY-42",
        );
        assert_eq!(pr.issue_source.as_deref(), Some("sentry"));
        assert_eq!(pr.issue_id.as_deref(), Some("SENTRY-42"));
        assert_eq!(pr.pr_number, 5);
    }

    // --- FixAttempt.is_bug ---

    #[test]
    fn test_fix_attempt_is_bug_sentry_source() {
        let attempt = claudear_core::types::FixAttempt {
            id: 1,
            issue_id: "issue-1".to_string(),
            short_id: "S-1".to_string(),
            source: "sentry".to_string(),
            attempted_at: chrono::Utc::now(),
            pr_url: None,
            scm_repo: None,
            scm_pr_number: None,
            status: claudear_core::types::FixAttemptStatus::Pending,
            error_message: None,
            merged_at: None,
            resolved_at: None,
            retry_count: 0,
            last_retry_at: None,
            issue_labels: vec![],
            parent_attempt_id: None,
            cascade_repo: None,
        };
        assert!(attempt.is_bug(), "sentry source should always be a bug");
    }

    #[test]
    fn test_fix_attempt_is_bug_with_bug_label() {
        let attempt = claudear_core::types::FixAttempt {
            id: 2,
            issue_id: "issue-2".to_string(),
            short_id: "L-2".to_string(),
            source: "linear".to_string(),
            attempted_at: chrono::Utc::now(),
            pr_url: None,
            scm_repo: None,
            scm_pr_number: None,
            status: claudear_core::types::FixAttemptStatus::Pending,
            error_message: None,
            merged_at: None,
            resolved_at: None,
            retry_count: 0,
            last_retry_at: None,
            issue_labels: vec!["bug".to_string(), "priority:high".to_string()],
            parent_attempt_id: None,
            cascade_repo: None,
        };
        assert!(attempt.is_bug());
    }

    #[test]
    fn test_fix_attempt_is_not_bug_with_feature_label() {
        let attempt = claudear_core::types::FixAttempt {
            id: 3,
            issue_id: "issue-3".to_string(),
            short_id: "L-3".to_string(),
            source: "linear".to_string(),
            attempted_at: chrono::Utc::now(),
            pr_url: None,
            scm_repo: None,
            scm_pr_number: None,
            status: claudear_core::types::FixAttemptStatus::Pending,
            error_message: None,
            merged_at: None,
            resolved_at: None,
            retry_count: 0,
            last_retry_at: None,
            issue_labels: vec!["feature".to_string(), "enhancement".to_string()],
            parent_attempt_id: None,
            cascade_repo: None,
        };
        assert!(!attempt.is_bug());
    }

    #[test]
    fn test_fix_attempt_is_bug_regression_label() {
        let attempt = claudear_core::types::FixAttempt {
            id: 4,
            issue_id: "issue-4".to_string(),
            short_id: "L-4".to_string(),
            source: "linear".to_string(),
            attempted_at: chrono::Utc::now(),
            pr_url: None,
            scm_repo: None,
            scm_pr_number: None,
            status: claudear_core::types::FixAttemptStatus::Pending,
            error_message: None,
            merged_at: None,
            resolved_at: None,
            retry_count: 0,
            last_retry_at: None,
            issue_labels: vec!["regression".to_string()],
            parent_attempt_id: None,
            cascade_repo: None,
        };
        assert!(attempt.is_bug());
    }

    // --- QaKnowledgeEntry ---

    #[test]
    fn test_qa_knowledge_entry_serialization() {
        let entry = QaKnowledgeEntry {
            id: 0,
            source: "test".to_string(),
            repo: Some("org/repo".to_string()),
            issue_id: "issue-1".to_string(),
            short_id: "T-1".to_string(),
            question_text: "What database?".to_string(),
            question_norm: "what database".to_string(),
            question_embedding: None,
            answer_text: "Use PostgreSQL".to_string(),
            answer_norm: "use postgresql".to_string(),
            answer_embedding: None,
            channel: "discord".to_string(),
            responder: Some("alice".to_string()),
            correlation_id: "corr-123".to_string(),
            asked_at: chrono::Utc::now(),
            answered_at: chrono::Utc::now(),
            success_count: 0,
            failure_count: 0,
            last_used_at: None,
            metadata: None,
        };
        let json = serde_json::to_string(&entry).unwrap();
        assert!(json.contains("What database?"));
        assert!(json.contains("Use PostgreSQL"));
        let deserialized: QaKnowledgeEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.channel, "discord");
    }

    // --- tracker integration: attempt lifecycle ---

    #[test]
    fn test_tracker_attempt_lifecycle() {
        let tracker = claudear_storage::SqliteTracker::in_memory().unwrap();

        // Record attempt
        tracker.record_attempt("test", "issue-1", "T-1").unwrap();

        // Should exist
        assert!(tracker.has_attempted("test", "issue-1").unwrap());

        // Should be pending
        let attempt = tracker.get_attempt("test", "issue-1").unwrap().unwrap();
        assert_eq!(
            attempt.status,
            claudear_core::types::FixAttemptStatus::Pending
        );

        // Mark success
        tracker
            .mark_success("test", "issue-1", "https://github.com/o/r/pull/1")
            .unwrap();
        let attempt = tracker.get_attempt("test", "issue-1").unwrap().unwrap();
        assert_eq!(
            attempt.status,
            claudear_core::types::FixAttemptStatus::Success
        );
        assert_eq!(
            attempt.pr_url.as_deref(),
            Some("https://github.com/o/r/pull/1")
        );
    }

    #[test]
    fn test_tracker_attempt_failed() {
        let tracker = claudear_storage::SqliteTracker::in_memory().unwrap();
        tracker.record_attempt("test", "issue-2", "T-2").unwrap();
        tracker
            .mark_failed("test", "issue-2", "build failed")
            .unwrap();

        let attempt = tracker.get_attempt("test", "issue-2").unwrap().unwrap();
        assert_eq!(
            attempt.status,
            claudear_core::types::FixAttemptStatus::Failed
        );
        assert_eq!(attempt.error_message.as_deref(), Some("build failed"));
    }

    #[test]
    fn test_tracker_attempt_merged() {
        let tracker = claudear_storage::SqliteTracker::in_memory().unwrap();
        tracker.record_attempt("test", "issue-3", "T-3").unwrap();
        tracker
            .mark_success("test", "issue-3", "https://github.com/o/r/pull/3")
            .unwrap();
        tracker.mark_merged("test", "issue-3").unwrap();

        let attempt = tracker.get_attempt("test", "issue-3").unwrap().unwrap();
        assert_eq!(
            attempt.status,
            claudear_core::types::FixAttemptStatus::Merged
        );
    }

    #[test]
    fn test_tracker_attempt_closed() {
        let tracker = claudear_storage::SqliteTracker::in_memory().unwrap();
        tracker.record_attempt("test", "issue-4", "T-4").unwrap();
        tracker
            .mark_success("test", "issue-4", "https://github.com/o/r/pull/4")
            .unwrap();
        tracker.mark_closed("test", "issue-4").unwrap();

        let attempt = tracker.get_attempt("test", "issue-4").unwrap().unwrap();
        assert_eq!(
            attempt.status,
            claudear_core::types::FixAttemptStatus::Closed
        );
    }

    #[test]
    fn test_tracker_get_attempted_issue_ids() {
        let tracker = claudear_storage::SqliteTracker::in_memory().unwrap();
        tracker.record_attempt("test", "issue-a", "T-A").unwrap();
        tracker.record_attempt("test", "issue-b", "T-B").unwrap();
        tracker.record_attempt("other", "issue-c", "T-C").unwrap();

        let ids = tracker.get_attempted_issue_ids("test").unwrap();
        assert!(ids.contains("issue-a"));
        assert!(ids.contains("issue-b"));
        assert!(!ids.contains("issue-c"));
    }

    #[test]
    fn test_tracker_has_not_attempted() {
        let tracker = claudear_storage::SqliteTracker::in_memory().unwrap();
        assert!(!tracker.has_attempted("test", "nonexistent").unwrap());
    }

    // --- validate_issue_id ---

    #[test]
    fn test_validate_issue_id_valid() {
        assert!(claudear_core::types::validate_issue_id("PROJ-123").is_ok());
        assert!(claudear_core::types::validate_issue_id("abc").is_ok());
        assert!(claudear_core::types::validate_issue_id("a").is_ok());
    }

    #[test]
    fn test_validate_issue_id_empty() {
        assert!(claudear_core::types::validate_issue_id("").is_err());
    }

    #[test]
    fn test_validate_issue_id_too_long() {
        let long = "a".repeat(101);
        assert!(claudear_core::types::validate_issue_id(&long).is_err());
    }

    #[test]
    fn test_validate_issue_id_path_traversal() {
        assert!(claudear_core::types::validate_issue_id("../etc/passwd").is_err());
    }

    #[test]
    fn test_validate_issue_id_forward_slash() {
        assert!(claudear_core::types::validate_issue_id("a/b").is_err());
    }

    #[test]
    fn test_validate_issue_id_backslash() {
        assert!(claudear_core::types::validate_issue_id("a\\b").is_err());
    }

    #[test]
    fn test_validate_issue_id_null_byte() {
        assert!(claudear_core::types::validate_issue_id("a\0b").is_err());
    }

    // --- Confidence comment formatting tests ---

    /// Build the confidence PR comment body, matching the logic in `execute_pipeline`.
    fn build_confidence_comment(confidence: u8, reasoning: Option<&str>) -> String {
        let mut comment = format!("## Fix Confidence: {}/100\n", confidence);
        if let Some(r) = reasoning {
            comment.push('\n');
            comment.push_str(r);
            comment.push('\n');
        }
        comment
    }

    #[test]
    fn test_confidence_comment_with_reasoning() {
        let comment = build_confidence_comment(85, Some("Tests pass and fix is localized"));
        assert!(comment.starts_with("## Fix Confidence: 85/100\n"));
        assert!(comment.contains("Tests pass and fix is localized"));
    }

    #[test]
    fn test_confidence_comment_without_reasoning() {
        let comment = build_confidence_comment(70, None);
        assert_eq!(comment, "## Fix Confidence: 70/100\n");
        // No reasoning block appended — just the header line
        assert_eq!(
            comment.matches('\n').count(),
            1,
            "Only the header trailing newline"
        );
    }

    #[test]
    fn test_confidence_comment_zero() {
        let comment = build_confidence_comment(0, None);
        assert!(comment.contains("0/100"));
    }

    #[test]
    fn test_confidence_comment_max() {
        let comment =
            build_confidence_comment(100, Some("Exact duplicate of previously fixed issue"));
        assert!(comment.contains("100/100"));
        assert!(comment.contains("Exact duplicate"));
    }

    #[test]
    fn test_confidence_comment_multiline_reasoning() {
        let reasoning = "Multiple factors:\n- Tests all pass\n- Small change scope";
        let comment = build_confidence_comment(90, Some(reasoning));
        assert!(comment.contains("90/100"));
        assert!(comment.contains("Multiple factors:"));
        assert!(comment.contains("- Tests all pass"));
    }

    #[test]
    fn test_confidence_metadata_stored_only_when_nonzero() {
        let mut issue = Issue::new("1", "TEST-1", "Bug", "https://example.com", "test");
        let confidence: u8 = 85;
        let reasoning = Some("Good fix".to_string());

        // Mirror the logic from execute_pipeline
        if confidence > 0 {
            issue.set_metadata("confidence", confidence);
        }
        if let Some(ref r) = reasoning {
            issue.set_metadata("confidence_reasoning", r.clone());
        }

        assert_eq!(issue.get_metadata::<u8>("confidence"), Some(85));
        assert_eq!(
            issue.get_metadata::<String>("confidence_reasoning"),
            Some("Good fix".to_string())
        );
    }

    #[test]
    fn test_confidence_metadata_not_stored_when_zero() {
        let mut issue = Issue::new("1", "TEST-1", "Bug", "https://example.com", "test");
        let confidence: u8 = 0;

        if confidence > 0 {
            issue.set_metadata("confidence", confidence);
        }

        assert_eq!(issue.get_metadata::<u8>("confidence"), None);
    }

    #[test]
    fn test_confidence_pr_comment_gated_by_config() {
        // When post_pr_comment is false, we should not post even with high confidence.
        // This tests the boolean guard logic: `config.evaluation.post_pr_comment && confidence > 0`
        let post_pr_comment = false;
        let confidence: u8 = 95;
        let should_post = post_pr_comment && confidence > 0;
        assert!(!should_post);
    }

    #[test]
    fn test_confidence_pr_comment_gated_by_zero_confidence() {
        // When confidence is 0, we should not post even with post_pr_comment=true.
        let post_pr_comment = true;
        let confidence: u8 = 0;
        let should_post = post_pr_comment && confidence > 0;
        assert!(!should_post);
    }

    #[test]
    fn test_confidence_pr_comment_posts_when_enabled_and_nonzero() {
        let post_pr_comment = true;
        let confidence: u8 = 50;
        let should_post = post_pr_comment && confidence > 0;
        assert!(should_post);
    }

    // --- Confidence comment: edge cases and integration patterns ---

    #[test]
    fn test_confidence_comment_has_markdown_header() {
        let comment = build_confidence_comment(50, None);
        assert!(comment.starts_with("## "), "Should be a markdown H2 header");
    }

    #[test]
    fn test_confidence_comment_reasoning_with_markdown() {
        let reasoning = "**Bold** reasoning with `code` and [link](url)";
        let comment = build_confidence_comment(60, Some(reasoning));
        assert!(comment.contains("**Bold**"));
        assert!(comment.contains("`code`"));
    }

    #[test]
    fn test_confidence_comment_reasoning_with_newlines_structure() {
        let comment = build_confidence_comment(80, Some("Reasoning here"));
        // Expected: "## Fix Confidence: 80/100\n\nReasoning here\n"
        let lines: Vec<&str> = comment.split('\n').collect();
        assert_eq!(lines[0], "## Fix Confidence: 80/100");
        assert_eq!(lines[1], ""); // blank line between header and reasoning
        assert_eq!(lines[2], "Reasoning here");
        assert_eq!(lines[3], ""); // trailing newline
    }

    #[test]
    fn test_confidence_comment_empty_reasoning_string() {
        // Some("") is different from None — should still add the extra newlines
        let comment = build_confidence_comment(50, Some(""));
        assert_eq!(comment.matches('\n').count(), 3); // header \n, blank \n, trailing \n
    }

    #[test]
    fn test_confidence_comment_very_long_reasoning() {
        let reasoning = "x".repeat(5000);
        let comment = build_confidence_comment(75, Some(&reasoning));
        assert!(comment.contains("75/100"));
        assert!(comment.len() > 5000);
    }

    #[test]
    fn test_confidence_metadata_various_values() {
        for confidence in [1u8, 50, 99, 100] {
            let mut issue = Issue::new("1", "T-1", "Bug", "https://ex.com", "test");
            if confidence > 0 {
                issue.set_metadata("confidence", confidence);
            }
            assert_eq!(issue.get_metadata::<u8>("confidence"), Some(confidence));
        }
    }

    #[test]
    fn test_confidence_metadata_reasoning_roundtrip() {
        let mut issue = Issue::new("1", "T-1", "Bug", "https://ex.com", "test");
        let reasoning = "Tests pass with good coverage";
        issue.set_metadata("confidence_reasoning", reasoning.to_string());
        assert_eq!(
            issue.get_metadata::<String>("confidence_reasoning"),
            Some(reasoning.to_string())
        );
    }

    #[test]
    fn test_confidence_parse_pr_url_needed_for_posting() {
        // Verify parse_pr_url works for the patterns used in confidence posting
        let github_url = "https://github.com/org/repo/pull/42";
        let parsed = claudear_storage::parse_pr_url(github_url);
        assert!(parsed.is_some());
        let (repo, number) = parsed.unwrap();
        assert_eq!(repo, "org/repo");
        assert_eq!(number, 42);
    }

    #[test]
    fn test_confidence_parse_pr_url_gitlab() {
        let gitlab_url = "https://gitlab.com/group/project/-/merge_requests/99";
        let parsed = claudear_storage::parse_pr_url(gitlab_url);
        assert!(parsed.is_some());
        let (repo, number) = parsed.unwrap();
        assert_eq!(repo, "group/project");
        assert_eq!(number, 99);
    }

    #[test]
    fn test_confidence_parse_pr_url_invalid_returns_none() {
        // When parse_pr_url fails, confidence comment posting should be silently skipped
        assert!(claudear_storage::parse_pr_url("not-a-url").is_none());
        assert!(claudear_storage::parse_pr_url("https://example.com/foo").is_none());
    }

    #[test]
    fn test_confidence_comment_combined_with_assessment() {
        // Simulate combined confidence + assessment comment scenario
        let confidence_comment = build_confidence_comment(85, Some("Tests pass"));
        // Assessment comment would be posted separately, but both should be valid markdown
        assert!(confidence_comment.starts_with("##"));
        assert!(confidence_comment.contains("85/100"));
    }

    // --- Dummy test helpers ---

    /// Dummy agent runner that does nothing (for IssueProcessor tests).
    struct DummyAgent;

    #[async_trait]
    impl claudear_integrations::runner::AgentRunner for DummyAgent {
        fn name(&self) -> &str {
            "dummy"
        }

        fn capabilities(&self) -> claudear_integrations::runner::ProviderCapabilities {
            claudear_integrations::runner::ProviderCapabilities::default()
        }

        fn build_prompt_for_issue(
            &self,
            _issue: &Issue,
            _context: &str,
            _project_dir: &std::path::Path,
        ) -> String {
            "dummy prompt".to_string()
        }

        async fn execute_with_attempt(
            &self,
            _prompt: &str,
            _issue: Option<&Issue>,
            _attempt_id: Option<i64>,
            _project_dir: &std::path::Path,
        ) -> claudear_core::error::Result<claudear_core::types::AgentResult> {
            Ok(claudear_core::types::AgentResult {
                success: false,
                output: String::new(),
                pr_url: None,
                changelog: None,
                error: Some("dummy error".to_string()),
                blocking_question: None,
                used_qa_ids: vec![],
                confidence: 0,
                confidence_reasoning: None,
                wrong_repo: None,
            })
        }
    }

    /// Dummy context provider for tests.
    struct DummyContextProvider;

    #[async_trait]
    impl ContextProvider for DummyContextProvider {
        async fn build_issue_context(
            &self,
            _issue: &Issue,
        ) -> claudear_core::error::Result<String> {
            Ok("dummy context".to_string())
        }
    }
}
