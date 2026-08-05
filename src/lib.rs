//! # Claudear
//!
//! A unified watcher service that monitors issue trackers and error monitoring services,
//! automatically spawning Claude Code agents to fix issues and create pull requests.
//!
//! ## Features
//!
//! - **Multi-Source Support**: Monitor Linear issues and Sentry errors from a single service
//! - **Extensible Architecture**: Easy to add new sources (GitHub Issues, Jira, etc.)
//! - **Discord Notifications**: Get notified about fix attempts with PR links
//! - **SQLite Tracking**: Persistent tracking of fix attempts to avoid duplicates
//! - **Priority-Based Processing**: Urgent/escalating issues are processed first
//! - **Graceful Handling**: Proper error handling and retry support
//!
//! ## Usage
//!
//! ```bash
//! # First-time setup - mark existing issues as seen
//! claudear seed
//!
//! # Start polling for new issues
//! claudear poll
//!
//! # Start webhook server for real-time events
//! claudear webhook
//! ```

// Re-exported from claudear-core
pub use claudear_core::error;
pub use claudear_core::http;
pub use claudear_core::secret;
pub use claudear_core::templates;
pub use claudear_core::types;

// Re-exported from claudear-config
pub use claudear_config::config;
pub use claudear_config::env_writer;
pub use claudear_config::users;

// Re-exported from claudear-storage
pub use claudear_storage as storage;

// Re-exported from claudear-analysis
pub use claudear_analysis::evaluation;
pub use claudear_analysis::feedback;
pub use claudear_analysis::inference;
pub use claudear_analysis::knowledgebase;
pub use claudear_analysis::learning;
pub use claudear_analysis::prioritisation;
pub use claudear_analysis::qa;
pub use claudear_analysis::regression;
pub use claudear_analysis::release;

// Re-exported from claudear-integrations
pub use claudear_integrations::ask_reply_inbox;
pub use claudear_integrations::chat;
pub use claudear_integrations::discord;
pub use claudear_integrations::github;
pub use claudear_integrations::github_app;
pub use claudear_integrations::gitlab;
pub use claudear_integrations::notifier;
pub use claudear_integrations::reports;
pub use claudear_integrations::runner;
pub use claudear_integrations::scm;
pub use claudear_integrations::source;
pub use claudear_integrations::telemetry;
pub use claudear_integrations::tls;
// Re-exported from claudear-engine
pub use claudear_engine::api;
pub use claudear_engine::api_events;
pub use claudear_engine::discord_index;
pub use claudear_engine::housekeeping;
pub use claudear_engine::ipc;
pub use claudear_engine::processing;
pub use claudear_engine::repo_index;
pub use claudear_engine::retry;
pub use claudear_engine::watcher;

// Local modules (thin wrappers)
pub mod repo;
pub mod webhook;

pub use chat::{ChatService, ChatState};
pub use config::{
    CascadeConfig, ChatConfig, CodeIndexConfig, Config, EvaluationConfig, RetryConfig, TlsConfig,
};
pub use discord::{DiscordClient, ThreadManager, ThreadState};
pub use error::{Error, Result};
pub use evaluation::{
    CodeQualityEvaluator, Diagnostic, EvalCategory, EvalDelta, EvalSnapshot, EvaluationResult,
};
pub use feedback::{
    cosine_similarity, euclidean_distance, format_similar_issues_context, normalize,
    EmbeddingClient, EmbeddingConfig, EmbeddingResult, FeedbackAnalyzer, FixOutcome,
    IssueEmbeddingConfig, IssueEmbeddingService, Outcome, PromptSuggestion, SimilarIssue,
    SimilarIssueWithDetails,
};
pub use github::GitHubClient;
pub use gitlab::GitLabClient;
pub use housekeeping::HousekeepingWorker;
pub use scm::{
    CodeReview, InlineReviewComment, OrgRepo, PostReviewAction, PrInfo, PrMonitor, PrReview,
    PrReviewComment, PrReviewState, PrStatus, PrStatusUpdate, PrSummary, RemoteRepo, ReviewComment,
    ReviewEvent, ReviewUser, ReviewWatcher, ScmProvider, ScmRelease,
};
pub use secret::SecretValue;
// Backward-compat alias
pub use github_app::{
    AppManifest, AppPermissions, CachedToken, GitHubAppAuth, GitHubAppClient, HookAttributes,
    SetupState,
};
pub use inference::{
    resolve_repo_for_cascade, resolve_repo_for_issue, Confidence, InferredRepo, IssueContext,
    RepoInferrer, RepoResolution,
};
pub use ipc::{
    default_socket_path, is_daemon_running, print_response, IpcClient, IpcCommand, IpcData,
    IpcResponse, IpcServer, WatcherState,
};
pub use repo::{
    build_repo_index, build_repo_index_from_github, build_repo_index_from_gitlab,
    build_repo_index_with_fallback, index_repo_files, DependencyDiscovery, DependencyGraph,
    DependencyType, DiscoveredDependency, IndexedRepo, RepoIndex, RepoRelationships, Repository,
};
pub use reports::{
    RecurringIssue, RepetitiveDigest, RepetitiveEntry, Report, ReportFrequency, ReportGenerator,
    ReportSchedule, ReportScheduler,
};
pub use retry::{RetryDecision, RetryManager};
pub use scm::GitHubUser;
pub use storage::{
    classify_error, compute_error_hash, parse_pr_url, AnalyticsService, FixAttemptTracker,
    StoredDependency, StoredRepository, TimePeriod, TrendAnalysis, TrendDirection,
};
#[cfg(feature = "sqlite")]
pub use storage::{is_vectorlite_available, try_load_vectorlite, SqliteTracker};
pub use types::*;
pub use users::{ResolvedUser, UserRegistry};

// Composable startup: lets alternative binaries (e.g. SaaS) bring their own
// storage backend while reusing all watcher / webhook / API logic.
use std::sync::Arc;

/// Shared state assembled during startup, passed to `run_daemon` / `run_webhook_server`.
pub struct AppComponents {
    pub config: Config,
    pub tracker: Arc<dyn storage::FixAttemptTracker>,
    pub sources: Vec<Arc<dyn source::IssueSource>>,
    pub notifier: Arc<dyn notifier::Notifier>,
    pub user_registry: UserRegistry,
    pub inferrer: Option<RepoInferrer>,
    pub embedding_client: Option<Arc<EmbeddingClient>>,
    pub review_watcher: Option<Arc<ReviewWatcher>>,
    pub issue_embedding_service: Option<Arc<IssueEmbeddingService>>,
    pub agent: Arc<dyn runner::AgentRunner>,
    pub classification_agent: Option<Arc<dyn runner::AgentRunner>>,
    /// Optional runner for repository classification (uses `repo_model`).
    /// Falls back to `classification_agent`, then `agent`, when not set.
    pub repo_classification_agent: Option<Arc<dyn runner::AgentRunner>>,
    /// Optional runner for answering questions (uses `qa_model`).
    /// Falls back to `agent` when not set.
    pub qa_agent: Option<Arc<dyn runner::AgentRunner>>,
}

/// Build a Claude agent runner for the default provider, optionally overriding
/// the model. When `model_override` is `None`, the provider's configured `model`
/// is used. All other settings (timeout, instructions, permissions, binary,
/// env) come from the default provider config.
pub fn build_provider_runner(
    config: &Config,
    tracker: Arc<dyn storage::FixAttemptTracker>,
    model_override: Option<String>,
) -> Arc<dyn runner::AgentRunner> {
    let provider = config.agent.default_provider_config();
    let model = model_override.or_else(|| provider.and_then(|p| p.model.clone()));
    let runner = runner::ClaudeAgentRunner::new(
        runner::ClaudeRunnerConfig {
            timeout_secs: config.agent.timeout_secs,
            model,
            instructions: provider.and_then(|p| p.instructions.clone()),
            permissions: provider.map(|p| p.permissions.clone()).unwrap_or_default(),
            readonly_tools: provider
                .map(|p| p.readonly_tools.clone())
                .unwrap_or_default(),
            skip_permissions: provider.map(|p| p.skip_permissions).unwrap_or(false),
            binary: provider
                .and_then(|p| p.binary.clone())
                .unwrap_or_else(|| "claude".to_string()),
            env: provider.map(|p| p.env.clone()).unwrap_or_default(),
            mcp: provider.map(|p| p.mcp.clone()).unwrap_or_default(),
        },
        tracker,
    );
    telemetry::InstrumentedRunner::wrap(Arc::new(runner))
}

/// Build an optional purpose-specific runner. Returns `None` when
/// `purpose_model` is unset, signalling the caller to fall back to a broader
/// runner (e.g. the main agent).
pub fn build_purpose_runner(
    config: &Config,
    tracker: Arc<dyn storage::FixAttemptTracker>,
    purpose_model: Option<String>,
) -> Option<Arc<dyn runner::AgentRunner>> {
    purpose_model.map(|model| build_provider_runner(config, tracker, Some(model)))
}

/// Build all non-storage components.
///
/// The caller provides the config and a tracker (which may be backed by
/// SQLite or anything else that implements [`FixAttemptTracker`]).
/// Everything else is wired up from the config.
pub async fn build_app(
    config: Config,
    tracker: Arc<dyn storage::FixAttemptTracker>,
) -> anyhow::Result<AppComponents> {
    let user_registry = UserRegistry::new(config.users.clone());

    // Notifier
    let notifier = build_notifier(&config, user_registry.clone());

    // Sources
    let sources = build_sources(&config);

    // GitHub client for inferrer
    let github_client = github::GitHubClient::new(config.github().clone());
    let (inferrer, embedding_client) = watcher::Watcher::build_inferrer_with_embeddings(
        &config,
        Some(&github_client),
        Some(tracker.as_ref()),
    )
    .await?;

    // Review watcher
    let review_watcher = build_review_watcher(&config, tracker.clone());

    // Issue embedding service (reuse shared embedding client)
    let issue_embedding_service = build_embedding_service(&tracker, embedding_client.as_ref());

    // Agent runner (uses the provider's default model)
    let agent: Arc<dyn runner::AgentRunner> = build_provider_runner(&config, tracker.clone(), None);

    // Purpose-specific runners. Each falls back to a broader runner at the point
    // of use when its model is unset (see AppComponents field docs).
    let provider = config.agent.default_provider_config();
    let classification_model = provider.and_then(|p| p.classification_model.clone());
    let repo_model = provider.and_then(|p| p.repo_model.clone());
    let qa_model = provider.and_then(|p| p.qa_model.clone());

    if let Some(ref m) = classification_model {
        tracing::info!(model = %m, "Building agent runner for classification (intent + repo default)");
    }
    if let Some(ref m) = repo_model {
        tracing::info!(model = %m, "Building agent runner for repo classification");
    }
    if let Some(ref m) = qa_model {
        tracing::info!(model = %m, "Building agent runner for QA");
    }

    let classification_agent = build_purpose_runner(&config, tracker.clone(), classification_model);
    let repo_classification_agent = build_purpose_runner(&config, tracker.clone(), repo_model);
    let qa_agent = build_purpose_runner(&config, tracker.clone(), qa_model);

    Ok(AppComponents {
        config,
        tracker,
        sources,
        notifier,
        user_registry,
        inferrer,
        embedding_client,
        review_watcher,
        issue_embedding_service,
        agent,
        classification_agent,
        repo_classification_agent,
        qa_agent,
    })
}

// --- internal helpers used by both build_app and main.rs ---

fn build_notifier(config: &Config, user_registry: UserRegistry) -> Arc<dyn notifier::Notifier> {
    use notifier::*;
    use telemetry::InstrumentedNotifier;

    let mut composite = CompositeNotifier::new();
    composite.add(InstrumentedNotifier::wrap(Arc::new(ConsoleNotifier::new())));

    let discord_notifier = DiscordNotifier::new(config.discord_merged(), user_registry.clone());
    if discord_notifier.is_enabled() {
        composite.add(InstrumentedNotifier::wrap(Arc::new(discord_notifier)));
    }

    let slack_notifier = SlackNotifier::new(config.slack_merged(), user_registry.clone());
    if slack_notifier.is_enabled() {
        composite.add(InstrumentedNotifier::wrap(Arc::new(slack_notifier)));
    }

    if let Ok(email_notifier) = EmailNotifier::new(config.email().clone(), user_registry.clone()) {
        if email_notifier.is_enabled() {
            composite.add(InstrumentedNotifier::wrap(Arc::new(email_notifier)));
        }
    }

    let sms_notifier = SmsNotifier::new(config.sms().clone(), user_registry.clone());
    if sms_notifier.is_enabled() {
        composite.add(InstrumentedNotifier::wrap(Arc::new(sms_notifier)));
    }

    let push_notifier = PushNotifier::new(config.push_config().clone(), user_registry.clone());
    if push_notifier.is_enabled() {
        composite.add(InstrumentedNotifier::wrap(Arc::new(push_notifier)));
    }

    let whatsapp_notifier =
        WhatsAppNotifier::new(config.notifiers.whatsapp.clone(), user_registry.clone());
    if whatsapp_notifier.is_enabled() {
        composite.add(InstrumentedNotifier::wrap(Arc::new(whatsapp_notifier)));
    }

    let telegram_notifier = TelegramNotifier::new(config.notifiers.telegram.clone(), user_registry);
    if telegram_notifier.is_enabled() {
        composite.add(InstrumentedNotifier::wrap(Arc::new(telegram_notifier)));
    }

    Arc::new(composite)
}

fn build_sources(config: &Config) -> Vec<Arc<dyn source::IssueSource>> {
    use source::*;

    let mut sources: Vec<Arc<dyn IssueSource>> = Vec::new();

    if let Some(linear_config) = config.linear() {
        if linear_config.enabled {
            sources.push(Arc::new(LinearSource::new(linear_config.clone())));
        }
    }

    if let Some(sentry_config) = config.sentry_config() {
        if sentry_config.enabled {
            sources.push(Arc::new(SentrySource::new(sentry_config.clone())));
        }
    }

    if let Some(jira_config) = config.jira() {
        if jira_config.enabled {
            sources.push(Arc::new(JiraSource::new(jira_config.clone())));
        }
    }

    let discord = config.discord_merged();
    if discord.source_enabled
        && discord.bot_token.is_some()
        && (discord.listen_channel_id.is_some() || discord.channel_id.is_some())
    {
        sources.push(Arc::new(DiscordSource::new(discord)));
    }

    let slack = config.slack_merged();
    if slack.source_enabled
        && slack.bot_token.is_some()
        && (slack.listen_channel_id.is_some() || slack.channel_id.is_some())
    {
        sources.push(Arc::new(SlackSource::new(slack)));
    }

    if config.notifiers.whatsapp.source_enabled
        && config.notifiers.whatsapp.access_token.is_some()
        && config.notifiers.whatsapp.phone_number_id.is_some()
    {
        sources.push(Arc::new(WhatsAppSource::new(
            config.notifiers.whatsapp.clone(),
        )));
    }

    if config.notifiers.telegram.source_enabled && config.notifiers.telegram.bot_token.is_some() {
        sources.push(Arc::new(TelegramSource::new(
            config.notifiers.telegram.clone(),
        )));
    }

    sources
        .into_iter()
        .map(telemetry::InstrumentedSource::wrap)
        .collect()
}

fn build_review_watcher(
    config: &Config,
    tracker: Arc<dyn storage::FixAttemptTracker>,
) -> Option<Arc<ReviewWatcher>> {
    if !config.is_github_enabled() {
        return None;
    }

    let github_client = github::GitHubClient::new(config.github().clone());
    if !github_client.is_enabled() {
        return None;
    }

    let provider: Arc<dyn ScmProvider> = telemetry::InstrumentedScm::wrap(Arc::new(github_client));
    let review_watcher = ReviewWatcher::with_tracker(provider, tracker.clone());

    if let Ok(states) = tracker.get_active_pr_review_states() {
        if !states.is_empty() {
            review_watcher.load_states(states);
        }
    }

    Some(Arc::new(review_watcher))
}

fn build_embedding_service(
    tracker: &Arc<dyn storage::FixAttemptTracker>,
    embedding_client: Option<&Arc<EmbeddingClient>>,
) -> Option<Arc<IssueEmbeddingService>> {
    embedding_client.map(|client| {
        Arc::new(IssueEmbeddingService::with_defaults(
            client.clone(),
            tracker.clone(),
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- build_sources with default config ---

    #[test]
    fn build_sources_default_config_returns_empty() {
        let config = Config::default();
        let sources = build_sources(&config);
        assert!(
            sources.is_empty(),
            "Default config should produce no sources"
        );
    }

    // --- build_notifier with default config ---

    #[test]
    fn build_notifier_default_config_has_console() {
        let config = Config::default();
        let user_registry = UserRegistry::new(std::collections::HashMap::new());
        let _notifier = build_notifier(&config, user_registry);
        // Should not panic -- console notifier is always added
    }

    // --- build_embedding_service ---

    #[test]
    fn build_embedding_service_none_when_no_client() {
        let tracker: Arc<dyn storage::FixAttemptTracker> =
            Arc::new(storage::SqliteTracker::in_memory().unwrap());
        let result = build_embedding_service(&tracker, None);
        assert!(result.is_none());
    }

    // --- build_review_watcher ---

    #[test]
    fn build_review_watcher_returns_none_when_github_disabled() {
        let config = Config::default();
        let tracker: Arc<dyn storage::FixAttemptTracker> =
            Arc::new(storage::SqliteTracker::in_memory().unwrap());
        let result = build_review_watcher(&config, tracker);
        assert!(result.is_none());
    }

    // --- build_sources with linear enabled ---

    #[test]
    fn build_sources_with_linear_enabled() {
        let mut config = Config::default();
        config.issues.linear = Some(config::LinearConfig {
            enabled: true,
            api_key: secret::SecretValue::new("test-key"),
            trigger_labels: vec!["claudear".to_string()],
            trigger_states: vec!["Todo".to_string()],
            team_id: None,
            project_id: None,
            webhook_secret: None,
            ..Default::default()
        });
        let sources = build_sources(&config);
        assert_eq!(sources.len(), 1);
    }

    #[test]
    fn build_sources_with_linear_disabled() {
        let mut config = Config::default();
        config.issues.linear = Some(config::LinearConfig {
            enabled: false,
            ..Default::default()
        });
        let sources = build_sources(&config);
        assert!(sources.is_empty());
    }

    // --- build_sources with jira enabled ---

    #[test]
    fn build_sources_with_jira_enabled() {
        let mut config = Config::default();
        config.issues.jira = Some(config::JiraConfig {
            enabled: true,
            base_url: "https://jira.example.com".to_string(),
            email: "user@example.com".to_string(),
            api_token: secret::SecretValue::new("token"),
            project_keys: vec!["PROJ".to_string()],
            ..Default::default()
        });
        let sources = build_sources(&config);
        assert_eq!(sources.len(), 1);
    }

    // --- build_sources with sentry enabled ---

    #[test]
    fn build_sources_with_sentry_enabled() {
        let mut config = Config::default();
        config.issues.sentry = Some(config::SentryConfig {
            enabled: true,
            auth_token: secret::SecretValue::new("token"),
            org_slug: "org".to_string(),
            project_slugs: vec!["proj".to_string()],
            ..Default::default()
        });
        let sources = build_sources(&config);
        assert_eq!(sources.len(), 1);
    }

    #[test]
    fn build_sources_multiple_enabled() {
        let mut config = Config::default();
        config.issues.linear = Some(config::LinearConfig {
            enabled: true,
            api_key: secret::SecretValue::new("key"),
            trigger_labels: vec!["claudear".to_string()],
            trigger_states: vec!["Todo".to_string()],
            ..Default::default()
        });
        config.issues.sentry = Some(config::SentryConfig {
            enabled: true,
            auth_token: secret::SecretValue::new("token"),
            org_slug: "org".to_string(),
            project_slugs: vec!["proj".to_string()],
            ..Default::default()
        });
        let sources = build_sources(&config);
        assert_eq!(sources.len(), 2);
    }
}
