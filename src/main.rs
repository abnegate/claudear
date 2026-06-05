//! Claudear - Unified watcher for issue trackers and error services.

use clap::{Parser, Subcommand};
use claudear::{
    api::ApiServer,
    config::Config,
    feedback::{EmbeddingClient, IssueEmbeddingService},
    github::GitHubClient,
    housekeeping::HousekeepingWorker,
    ipc::{default_socket_path, is_daemon_running, print_response, IpcClient, IpcServer},
    notifier::{
        CompositeNotifier, ConsoleNotifier, DiscordNotifier, EmailNotifier, Notifier, PushNotifier,
        SlackNotifier, SmsNotifier, TelegramNotifier, WhatsAppNotifier,
    },
    regression::{
        CompositeChecker, LinearRegressionChecker, LinearRegressionConfig, NoOpChecker,
        RegressionScheduler, RegressionSchedulerConfig, SentryRegressionChecker,
        SentryRegressionConfig,
    },
    release::{ReleaseTracker, ReleaseTrackerConfig},
    repo::{build_repo_index, DependencyType, RepoRelationships},
    reports::{ReportFrequency, ReportGenerator, ReportSchedule, ReportScheduler},
    retry::RetryManager,
    runner::{AgentRunner, ClaudeAgentRunner, ClaudeRunnerConfig},
    scm::{PrMonitor, PrStatus, ReviewWatcher, ScmProvider},
    source::{
        DiscordSource, HelpScoutSource, IssueSource, JiraSource, LinearSource, SentrySource,
        SlackSource, TelegramSource, WhatsAppSource,
    },
    storage::{
        ActivityStore, EmbeddingStore, FixAttemptTracker, RepoStore, SqliteTracker, UserStore,
    },
    telemetry::{InstrumentedNotifier, InstrumentedRunner, InstrumentedScm, InstrumentedSource},
    types::{ActionKind, ActivityLogEntry, FixAttemptStatus, Issue},
    users::UserRegistry,
    watcher::{Watcher, WatcherOptions},
    webhook::{
        print_setup_result, GitHubWebhookHandler, GitLabIssueWebhookHandler,
        HelpScoutWebhookHandler, JiraWebhookHandler, LinearWebhookHandler, SentryWebhookHandler,
        SlackWebhookHandler, TelegramWebhookHandler, WebhookConfigurator, WebhookHandlerRegistry,
        WebhookServer, WhatsAppWebhookHandler,
    },
};
use serde_json::json;
use std::sync::Arc;
use tokio::time::{interval, Duration};
use tracing_subscriber::{
    filter::LevelFilter, fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Layer,
};

#[derive(Parser)]
#[command(name = "claudear")]
#[command(about = "Unified watcher for issue trackers and error services")]
#[command(version)]
struct Cli {
    /// Path to config file
    #[arg(short, long, default_value = "claudear.toml")]
    config: String,

    /// Enable verbose console logging (timestamps, info/debug logs)
    #[arg(short, long, global = true)]
    verbose: bool,

    /// Directory for log files (with daily rotation). Set to empty string to disable file logging.
    #[arg(long, env = "CLAUDEAR_LOG_DIR", default_value = "./logs")]
    log_dir: std::path::PathBuf,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the watcher daemon with all configured services
    Start {
        /// HTTP port for dashboard API and webhooks
        #[arg(long, default_value = "3100")]
        port: u16,

        /// Enable background polling (in addition to webhooks)
        #[arg(long)]
        poll: bool,

        /// Polling interval in milliseconds (requires --poll)
        #[arg(long, default_value = "300000")]
        poll_interval: u64,

        /// Disable webhook server
        #[arg(long)]
        no_webhooks: bool,

        /// Disable dashboard API
        #[arg(long)]
        no_dashboard: bool,

        /// Run in the foreground instead of daemonizing (useful for systemd/launchd)
        #[arg(long)]
        foreground: bool,
    },

    /// Stop the running watcher daemon
    Stop,

    /// Show the status of the running daemon
    Status,

    /// Pause the running watcher (stops processing new issues)
    Pause,

    /// Resume a paused watcher
    Resume,

    /// Show recent activity from the daemon
    Activity {
        /// Number of entries to show
        #[arg(default_value = "20")]
        limit: usize,
    },

    /// Mark all existing issues as seen (run this first!)
    Seed,

    /// Show what would be processed without running Claude
    DryRun,

    /// Start polling all enabled sources (foreground, no daemon)
    Poll {
        /// Polling interval in milliseconds
        #[arg(default_value = "300000")]
        interval: u64,

        /// HTTP port for dashboard API
        #[arg(long, default_value = "3100")]
        port: u16,

        /// Disable dashboard API
        #[arg(long)]
        no_dashboard: bool,
    },

    /// Start webhook server for real-time events
    Webhook {
        /// Port to listen on
        #[arg(default_value = "3100")]
        port: u16,

        /// Auto-configure webhooks with Linear/Sentry APIs before starting
        #[arg(long)]
        setup: bool,

        /// Public base URL where webhooks will be received (required with --setup)
        #[arg(long)]
        base_url: Option<String>,

        /// Path to .env file for saving webhook secrets
        #[arg(long, default_value = ".env")]
        env_file: String,

        /// Run self-test: start server, send test payloads, verify processing
        #[arg(long)]
        self_test: bool,
    },

    /// Manually trigger a fix for an issue
    Trigger {
        /// Source name (linear, sentry)
        source: String,
        /// Issue ID
        issue_id: String,
    },

    /// Run a single action (reply, verify, or resolve) against an issue
    Action {
        /// Action to run: reply | verify | resolve
        action: String,
        /// Source name (helpscout, linear, sentry, ...)
        source: String,
        /// Issue ID
        issue_id: String,
    },

    /// Reset a failed attempt to allow retry
    Reset {
        /// Source name (linear, sentry)
        source: String,
        /// Issue ID
        issue_id: String,
    },

    /// Purge operational data (attempts, PRs, executions) while preserving knowledge, inference, and embeddings
    Purge {
        /// Skip confirmation prompt
        #[arg(long)]
        confirm: bool,
    },

    /// Show statistics about fix attempts
    Stats,

    /// List configured sources
    Sources,

    /// Start the dashboard API server
    Dashboard {
        /// Port to listen on
        #[arg(default_value = "3100")]
        port: u16,

        /// Path to built dashboard files (optional, serves static files)
        #[arg(long)]
        dashboard_dir: Option<std::path::PathBuf>,
    },

    /// Repository management commands
    #[command(subcommand)]
    Repos(ReposCommands),

    /// Pull request management commands
    #[command(subcommand)]
    Prs(PrsCommands),

    /// Retry management commands
    #[command(subcommand)]
    Retries(RetriesCommands),

    /// Inference analytics and management
    #[command(subcommand)]
    Inference(InferenceCommands),

    /// Report generation and scheduling
    #[command(subcommand)]
    Report(ReportCommands),

    /// Diagnostic commands for debugging
    #[command(subcommand)]
    Diag(DiagCommands),

    /// User management commands
    #[command(subcommand)]
    Users(UsersCommands),

    /// Interactive chat about indexed code
    Chat {
        /// Question to ask (omit for interactive REPL mode)
        question: Option<String>,

        /// Repository to scope the search to
        #[arg(long)]
        repo: Option<String>,

        /// Model override (path to .gguf file)
        #[arg(long)]
        model: Option<std::path::PathBuf>,

        /// Download the configured model if not present on disk
        #[arg(long)]
        download_model: bool,
    },
}

/// Repository management subcommands
#[derive(Subcommand)]
enum ReposCommands {
    /// List all indexed repositories
    List,

    /// Build/refresh the repository file index
    Index {
        /// Force full re-indexing even if already indexed
        #[arg(long, default_value = "false")]
        force: bool,
    },

    /// Search for files across all indexed repositories
    Search {
        /// Search query (file name or partial path)
        query: String,
    },

    /// Show index statistics
    Stats,

    /// Link two repositories (declare a dependency)
    Link {
        /// Upstream repository (the dependency)
        upstream: String,

        /// Downstream repository (depends on upstream)
        downstream: String,

        /// Dependency type (npm, composer, git_submodule, manual)
        #[arg(long, default_value = "manual")]
        dep_type: String,
    },

    /// Show the dependency graph
    Graph {
        /// Start from a specific repository
        #[arg(long)]
        root: Option<String>,
    },

    /// Show what would cascade from a repository change
    Cascade {
        /// Repository that changed
        repo: String,
    },

    /// Auto-discover dependencies by scanning directories
    Discover {
        /// Paths to scan (defaults to config's auto_discover_paths)
        #[arg(long)]
        paths: Vec<String>,

        /// Save discovered dependencies to database
        #[arg(long, default_value = "false")]
        save: bool,

        /// Clear existing dependencies before saving
        #[arg(long, default_value = "false")]
        clear: bool,
    },

    /// [DEPRECATED] Add a repository (repos now auto-discovered via known_orgs)
    #[command(hide = true)]
    Add {
        /// Repository name
        name: String,

        /// Local filesystem path
        #[arg(long)]
        path: Option<String>,

        /// GitHub URL (owner/repo format)
        #[arg(long)]
        scm_url: Option<String>,
    },

    /// Sync repository index to database (paths and files)
    Sync {
        /// Skip syncing file lists (faster but limits inference accuracy)
        #[arg(long, default_value = "false")]
        skip_files: bool,
    },

    /// Force re-index code embeddings for all (or specific) repositories
    Reindex {
        /// Only re-index a specific repository (e.g., "org/repo")
        #[arg(long)]
        repo: Option<String>,
    },
}

/// Inference analytics subcommands
#[derive(Subcommand)]
enum InferenceCommands {
    /// Show inference statistics (success rates by confidence level)
    Stats,

    /// List recent inference attempts
    History {
        /// Maximum number of entries to show
        #[arg(long, default_value = "20")]
        limit: usize,
    },

    /// Provide feedback on an inference attempt
    Feedback {
        /// Inference attempt ID
        id: i64,

        /// Whether the inference was correct
        #[arg(long)]
        correct: bool,

        /// Actual repository name if inference was incorrect
        #[arg(long)]
        actual_repo: Option<String>,
    },
}

/// Pull request management subcommands
#[derive(Subcommand)]
enum PrsCommands {
    /// List all PRs that are being tracked
    List,

    /// Monitor PRs for merge status and auto-resolve issues
    Monitor {
        /// Run continuously with polling
        #[arg(long, default_value = "false")]
        continuous: bool,
    },
}

/// Retry management subcommands
#[derive(Subcommand)]
enum RetriesCommands {
    /// List issues that are eligible for retry
    List,

    /// Process ready retries now
    Process,
}

/// Report subcommands
#[derive(Subcommand)]
enum ReportCommands {
    /// Generate and show a report (preview without sending)
    Preview {
        /// Report frequency: daily, weekly, monthly
        #[arg(default_value = "daily")]
        frequency: String,
    },

    /// Generate and send a report immediately
    Send {
        /// Report frequency: daily, weekly, monthly
        #[arg(default_value = "daily")]
        frequency: String,
    },

    /// Start the report scheduler (runs in background)
    Schedule {
        /// Enable daily reports
        #[arg(long, default_value = "true")]
        daily: bool,

        /// Enable weekly reports (Monday)
        #[arg(long, default_value = "false")]
        weekly: bool,

        /// Hour to send reports (0-23 UTC)
        #[arg(long, default_value = "9")]
        hour: u32,
    },
}

/// Diagnostic subcommands
#[derive(Subcommand)]
enum DiagCommands {
    /// Show database table counts and recent operations
    Db,

    /// Show the dependency graph used for release tracking
    ReleaseGraph,

    /// Check if a PR's fix would be detected in a target release (dry-run)
    ReleaseCheck {
        /// Repository (owner/repo format)
        repo: String,
        /// PR number
        pr: i64,
        /// Target repository to check against (optional, uses config targets if not specified)
        #[arg(long)]
        target: Option<String>,
    },

    /// Show the dependency path from source to target repo
    ReleasePath {
        /// Source repository (where fix was made)
        source: String,
        /// Target repository (where release happens)
        target: String,
    },
}

/// User management subcommands
#[derive(Subcommand)]
enum UsersCommands {
    /// Seed an admin user (creates or updates password if email exists)
    Seed {
        /// User email
        #[arg(long)]
        email: String,

        /// User password
        #[arg(long)]
        password: String,

        /// User display name
        #[arg(long, default_value = "Admin")]
        name: String,
    },
}

/// Initialize logging with both console and file output.
/// Returns a guard that must be kept alive for the duration of the program.
fn init_logging(
    log_dir: Option<&std::path::Path>,
    verbose: bool,
    suppress_console_info: bool,
) -> Option<tracing_appender::non_blocking::WorkerGuard> {
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info,ort=warn"));

    // Console layer is always enabled, but concise by default.
    let console_layer = if verbose {
        fmt::layer()
            .with_target(false)
            .with_writer(std::io::stdout)
            .boxed()
    } else if suppress_console_info {
        fmt::layer()
            .with_target(false)
            .without_time()
            .with_writer(std::io::stdout)
            .with_filter(LevelFilter::WARN)
            .boxed()
    } else {
        fmt::layer()
            .with_target(false)
            .without_time()
            .with_writer(std::io::stdout)
            .boxed()
    };

    // File layer - optional, with daily rotation
    if let Some(dir) = log_dir {
        // Create log directory if it doesn't exist
        if let Err(e) = std::fs::create_dir_all(dir) {
            eprintln!("Warning: Failed to create log directory {:?}: {}", dir, e);
            // Fall back to console-only logging
            tracing_subscriber::registry()
                .with(filter)
                .with(console_layer)
                .with(sentry::integrations::tracing::layer())
                .init();
            return None;
        }

        // Create file appender with daily rotation
        let file_appender = tracing_appender::rolling::daily(dir, "claudear.log");
        let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);

        let file_layer = fmt::layer()
            .with_target(true)
            .with_ansi(false) // No ANSI colors in file
            .with_writer(non_blocking);

        tracing_subscriber::registry()
            .with(filter)
            .with(console_layer)
            .with(file_layer)
            .with(sentry::integrations::tracing::layer())
            .init();

        tracing::info!("Logging to file: {}/claudear.log", dir.display());
        Some(guard)
    } else {
        // Console-only logging
        tracing_subscriber::registry()
            .with(filter)
            .with(console_layer)
            .with(sentry::integrations::tracing::layer())
            .init();
        None
    }
}

fn format_interval_compact(ms: u64) -> String {
    if ms.is_multiple_of(3_600_000) {
        format!("{}h", ms / 3_600_000)
    } else if ms.is_multiple_of(60_000) {
        format!("{}m", ms / 60_000)
    } else if ms.is_multiple_of(1_000) {
        format!("{}s", ms / 1_000)
    } else {
        format!("{}ms", ms)
    }
}

fn dashboard_host_for_display(bind_address: &str) -> &str {
    match bind_address {
        "127.0.0.1" | "0.0.0.0" | "::1" => "localhost",
        _ => bind_address,
    }
}

fn format_status_with_detail(label: &str, detail: Option<String>) -> String {
    match detail.filter(|d| !d.is_empty()) {
        Some(detail) => format!("{label} ({detail})"),
        None => label.to_string(),
    }
}

fn print_startup_ok(message: impl AsRef<str>) {
    println!("  [ok] {}", message.as_ref());
}

#[expect(clippy::too_many_arguments)]
fn print_startup_banner_and_status(
    config: &Config,
    config_path: &str,
    port: u16,
    enable_dashboard: bool,
    enable_webhooks: bool,
    enable_polling: bool,
    poll_interval_ms: u64,
    inferrer_embedding_count: Option<usize>,
    user_registry: &UserRegistry,
) {
    println!();
    println!(" ██████╗██╗      █████╗ ██╗   ██╗██████╗ ███████╗ █████╗ ██████╗ ");
    println!("██╔════╝██║     ██╔══██╗██║   ██║██╔══██╗██╔════╝██╔══██╗██╔══██╗");
    println!("██║     ██║     ███████║██║   ██║██║  ██║█████╗  ███████║██████╔╝");
    println!("██║     ██║     ██╔══██║██║   ██║██║  ██║██╔══╝  ██╔══██║██╔══██╗");
    println!("╚██████╗███████╗██║  ██║╚██████╔╝██████╔╝███████╗██║  ██║██║  ██║");
    println!(" ╚═════╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝");
    println!();
    println!("    v{}", env!("CARGO_PKG_VERSION"));
    let host = dashboard_host_for_display(&config.bind_address);
    if enable_dashboard {
        println!("    Dashboard: http://{}:{}", host, port);
    }
    if enable_webhooks {
        println!("    Webhooks:  http://{}:{}/webhook/{{source}}", host, port);
    }
    println!();

    print_startup_ok(format!("Config loaded from {}", config_path));

    print_startup_ok("Database initialized (SQLite)");

    if let Some(linear) = config.linear().filter(|c| c.enabled) {
        let detail = linear
            .team_id
            .clone()
            .or_else(|| linear.project_id.clone())
            .or_else(|| linear.trigger_assignee.clone());
        print_startup_ok(format_status_with_detail("Connected: Linear", detail));
    }

    if let Some(sentry_cfg) = config.sentry_config().filter(|c| c.enabled) {
        let detail = sentry_cfg
            .project_slugs
            .first()
            .cloned()
            .or_else(|| (!sentry_cfg.org_slug.is_empty()).then(|| sentry_cfg.org_slug.clone()));
        print_startup_ok(format_status_with_detail("Connected: Sentry", detail));
    }

    if config.is_github_enabled() {
        let github_cfg = config.github();
        let detail = github_cfg.repos.first().cloned().or_else(|| {
            config
                .is_github_app_configured()
                .then(|| "GitHub App".to_string())
        });
        print_startup_ok(format_status_with_detail("Connected: GitHub", detail));
    }

    if DiscordNotifier::new(config.discord_merged(), user_registry.clone()).is_enabled() {
        print_startup_ok("Discord notifier ready");
    }

    if SlackNotifier::new(config.slack_merged(), user_registry.clone()).is_enabled() {
        print_startup_ok("Slack notifier ready");
    }

    if let Some(count) = inferrer_embedding_count {
        print_startup_ok(format!("Vector store loaded ({count} embeddings)"));
    }

    if enable_polling {
        print_startup_ok(format!(
            "Polling started (interval: {})",
            format_interval_compact(poll_interval_ms)
        ));
    }

    println!();
    println!("  Watching for issues...");
    println!();
}

/// Build an `IssueEmbeddingService` for semantic dedup and context enrichment.
///
/// Returns `None` if no embedding client is provided, allowing graceful degradation.
fn build_issue_embedding_service(
    tracker: &Arc<dyn FixAttemptTracker>,
    embedding_client: Option<&Arc<EmbeddingClient>>,
) -> Option<Arc<IssueEmbeddingService>> {
    embedding_client.map(|client| {
        Arc::new(IssueEmbeddingService::with_defaults(
            client.clone(),
            tracker.clone(),
        ))
    })
}

/// Common dependencies needed to construct a [`Watcher`].
///
/// Built once by [`build_watcher_deps`] and consumed by both `Commands::Start`
/// and `Commands::Webhook`.
struct WatcherDeps {
    sources: Vec<Arc<dyn IssueSource>>,
    scm_provider: Option<Arc<dyn ScmProvider>>,
    github_client: Option<Arc<GitHubClient>>,
    relationships: Option<RepoRelationships>,
    inferrer: Option<claudear::inference::RepoInferrer>,
    embedding_client: Option<Arc<EmbeddingClient>>,
    review_watcher: Option<Arc<ReviewWatcher>>,
    issue_embedding_service: Option<Arc<IssueEmbeddingService>>,
    code_search_service: Option<Arc<claudear::repo::code_index::CodeSearchService>>,
    agent: Arc<dyn AgentRunner>,
    classification_agent: Option<Arc<dyn AgentRunner>>,
    llm_engine: Option<Arc<claudear::chat::llm::LlmEngine>>,
}

/// Build the common watcher dependencies shared between `Commands::Start` and
/// `Commands::Webhook`.
async fn build_watcher_deps(
    config: &Config,
    tracker: &Arc<dyn FixAttemptTracker>,
) -> anyhow::Result<WatcherDeps> {
    let sources = create_sources(config);

    // GitHub client for API-based repo discovery
    let github_client = GitHubClient::new(config.github().clone());

    // Inferrer + embedding client
    let (inferrer, embedding_client) = Watcher::build_inferrer_with_embeddings(
        config,
        Some(&github_client),
        Some(tracker.as_ref()),
    )
    .await?;
    if inferrer.is_some() {
        tracing::info!("Repository inference enabled");
    }

    // ReviewWatcher for PR review tracking
    let review_watcher = create_review_watcher(config, tracker.clone());

    // Dependency graph for cascade support
    let relationships = if config.cascade.enabled {
        let mut rels = RepoRelationships::with_defaults();
        match tracker.list_all_dependencies() {
            Ok(db_deps) if !db_deps.is_empty() => {
                tracing::info!(count = db_deps.len(), "Loading dependencies from database");
                for dep in &db_deps {
                    if let Some(dep_type) = DependencyType::parse(&dep.dep_type) {
                        rels.add_dependency(&dep.upstream, &dep.downstream, dep_type, None)
                            .ok();
                    } else {
                        tracing::warn!(
                            dep_type = %dep.dep_type,
                            "Unknown dependency type, skipping"
                        );
                    }
                }
            }
            Ok(_) => {}
            Err(e) => {
                tracing::error!(error = %e, "Failed to load dependencies from database");
            }
        }
        Some(rels)
    } else {
        None
    };

    // GitHub client for PR merge checking
    let github_client_for_watcher = if config.is_github_enabled() {
        Some(Arc::new(GitHubClient::new(config.github().clone())))
    } else {
        None
    };

    // Issue embedding service for semantic dedup (reuse shared embedding client)
    let issue_embedding_service = build_issue_embedding_service(tracker, embedding_client.as_ref());

    // Code search service for enriching issues with relevant code context (reuse shared embedding client)
    let code_search_service = if config.code_index.enabled {
        embedding_client.as_ref().map(|emb| {
            Arc::new(claudear::repo::code_index::CodeSearchService::new(
                tracker.clone(),
                emb.clone(),
            ))
        })
    } else {
        None
    };

    // Generic SCM provider for PR merge detection (GitLab, etc.)
    let scm_provider: Option<Arc<dyn ScmProvider>> = if let Some(gitlab_config) = config.gitlab() {
        if gitlab_config.enabled && gitlab_config.token.is_some() {
            Some(InstrumentedScm::wrap(Arc::new(
                claudear::gitlab::GitLabClient::new(gitlab_config.clone()),
            )))
        } else {
            None
        }
    } else {
        None
    };

    // Agent runner
    let agent: Arc<dyn AgentRunner> = InstrumentedRunner::wrap(Arc::new(ClaudeAgentRunner::new(
        ClaudeRunnerConfig {
            timeout_secs: config.agent.timeout_secs,
            model: config
                .agent
                .default_provider_config()
                .and_then(|p| p.model.clone()),
            instructions: config
                .agent
                .default_provider_config()
                .and_then(|p| p.instructions.clone()),
            permissions: config
                .agent
                .default_provider_config()
                .map(|p| p.permissions.clone())
                .unwrap_or_default(),
            skip_permissions: config
                .agent
                .default_provider_config()
                .map(|p| p.skip_permissions)
                .unwrap_or(false),
            binary: config
                .agent
                .default_provider_config()
                .and_then(|p| p.binary.clone())
                .unwrap_or_else(|| "claude".to_string()),
            env: config
                .agent
                .default_provider_config()
                .map(|p| p.env.clone())
                .unwrap_or_default(),
        },
        tracker.clone(),
    )));

    // Eagerly load LLM engine — download model if not present on disk
    let llm_engine = if config.llm.enabled {
        let model_path = claudear::chat::service::expand_tilde(&config.llm.model_path);
        let model_ready = if model_path.exists() && model_path.is_file() {
            true
        } else if !config.llm.model_url.is_empty() {
            tracing::info!(
                url = %config.llm.model_url,
                target = %model_path.display(),
                "LLM model not found, downloading..."
            );
            let progress = Arc::new(claudear::chat::models::download::DownloadProgress::new());
            match claudear::chat::models::download::download_gguf(
                &config.llm.model_url,
                &model_path,
                progress.clone(),
            )
            .await
            {
                Ok(()) => {
                    tracing::info!(
                        size_mb = progress
                            .total_bytes
                            .load(std::sync::atomic::Ordering::Relaxed)
                            / 1_048_576,
                        "LLM model downloaded successfully"
                    );
                    true
                }
                Err(e) => {
                    tracing::warn!(error = %e, "Failed to download LLM model, classification disabled");
                    false
                }
            }
        } else {
            tracing::debug!("LLM model not found and no download URL configured");
            false
        };

        if model_ready {
            let llm_config = claudear::chat::llm::LlmConfig {
                model_path,
                context_length: config.llm.context_length,
                gpu_layers: config.llm.gpu_layers,
                threads: config.llm.threads,
                timeout: if config.llm.inference_timeout_secs > 0 {
                    Some(std::time::Duration::from_secs(
                        config.llm.inference_timeout_secs,
                    ))
                } else {
                    None
                },
            };
            match claudear::chat::llm::LlmEngine::load(&llm_config) {
                Ok(engine) => {
                    tracing::info!("LLM engine loaded for classification + chat");
                    Some(Arc::new(engine))
                }
                Err(e) => {
                    tracing::warn!(error = %e, "Failed to load LLM engine");
                    None
                }
            }
        } else {
            None
        }
    } else {
        None
    };

    // Optionally swap agent runner to use local LLM
    let agent: Arc<dyn AgentRunner> = if config.agent.use_llm {
        if let Some(ref engine) = llm_engine {
            tracing::info!("Using local LLM as agent runner (agent.use_llm = true)");
            Arc::new(claudear_engine::llm_agent_runner::LlmAgentRunner::new(
                engine.clone(),
            ))
        } else {
            tracing::warn!(
                "agent.use_llm = true but LLM engine not available, using default agent"
            );
            agent
        }
    } else {
        agent
    };

    // Build a separate agent runner for classification if a classification_model is configured
    let classification_agent: Option<Arc<dyn AgentRunner>> = config
        .agent
        .default_provider_config()
        .and_then(|p| p.classification_model.clone())
        .map(|model| {
            tracing::info!(model = %model, "Building separate agent runner for classification");
            let runner = ClaudeAgentRunner::new(
                ClaudeRunnerConfig {
                    timeout_secs: config.agent.timeout_secs,
                    model: Some(model),
                    instructions: config
                        .agent
                        .default_provider_config()
                        .and_then(|p| p.instructions.clone()),
                    permissions: config
                        .agent
                        .default_provider_config()
                        .map(|p| p.permissions.clone())
                        .unwrap_or_default(),
                    skip_permissions: config
                        .agent
                        .default_provider_config()
                        .map(|p| p.skip_permissions)
                        .unwrap_or(false),
                    binary: config
                        .agent
                        .default_provider_config()
                        .and_then(|p| p.binary.clone())
                        .unwrap_or_else(|| "claude".to_string()),
                    env: config
                        .agent
                        .default_provider_config()
                        .map(|p| p.env.clone())
                        .unwrap_or_default(),
                },
                tracker.clone(),
            );
            InstrumentedRunner::wrap(Arc::new(runner)) as Arc<dyn AgentRunner>
        });

    Ok(WatcherDeps {
        sources,
        scm_provider,
        github_client: github_client_for_watcher,
        relationships,
        inferrer,
        embedding_client,
        review_watcher,
        issue_embedding_service,
        code_search_service,
        agent,
        classification_agent,
        llm_engine,
    })
}

/// Create a ReviewWatcher if GitHub is configured.
fn create_review_watcher(
    config: &Config,
    tracker: Arc<dyn FixAttemptTracker>,
) -> Option<Arc<ReviewWatcher>> {
    if !config.is_github_enabled() {
        tracing::debug!("GitHub not configured, ReviewWatcher disabled");
        return None;
    }

    let github_client = GitHubClient::new(config.github().clone());
    if !github_client.is_enabled() {
        tracing::debug!("GitHub client not enabled, ReviewWatcher disabled");
        return None;
    }

    let provider: Arc<dyn ScmProvider> = InstrumentedScm::wrap(Arc::new(github_client));
    let review_watcher = ReviewWatcher::with_tracker(provider, tracker.clone());

    // Restore states from database
    match tracker.get_active_pr_review_states() {
        Ok(states) => {
            let count = states.len();
            if count > 0 {
                review_watcher.load_states(states);
                tracing::info!(
                    component = "review_watcher",
                    count = count,
                    "Restored PR review states from database"
                );
            }
        }
        Err(e) => {
            tracing::warn!(
                component = "review_watcher",
                error = %e,
                "Failed to restore PR review states from database"
            );
        }
    }

    tracing::info!(
        component = "review_watcher",
        "ReviewWatcher enabled for PR review tracking"
    );

    Some(Arc::new(review_watcher))
}

fn create_sources(config: &Config) -> Vec<Arc<dyn IssueSource>> {
    let mut sources: Vec<Arc<dyn IssueSource>> = Vec::new();

    if let Some(linear_config) = config.linear() {
        if linear_config.enabled {
            sources.push(Arc::new(LinearSource::new(linear_config.clone())));
            tracing::info!("Linear source initialized");
        }
    }

    if let Some(sentry_config) = config.sentry_config() {
        if sentry_config.enabled {
            sources.push(Arc::new(SentrySource::new(sentry_config.clone())));
            tracing::info!("Sentry source initialized");
        }
    }

    if let Some(jira_config) = config.jira() {
        if jira_config.enabled {
            sources.push(Arc::new(JiraSource::new(jira_config.clone())));
            tracing::info!("Jira source initialized");
        }
    }

    if let Some(helpscout_config) = config.helpscout() {
        if helpscout_config.enabled {
            if !helpscout_config.app_id.expose().is_empty()
                && !helpscout_config.app_secret.expose().is_empty()
                && !helpscout_config.mailbox_ids.is_empty()
            {
                sources.push(Arc::new(HelpScoutSource::new(helpscout_config.clone())));
                tracing::info!("HelpScout source initialized");
            } else {
                tracing::warn!(
                    "HelpScout enabled but missing app_id/app_secret/mailbox_ids; skipping"
                );
            }
        }
    }

    let discord = config.discord_merged();
    if discord.source_enabled {
        if discord.bot_token.is_some()
            && (discord.listen_channel_id.is_some() || discord.channel_id.is_some())
        {
            sources.push(Arc::new(DiscordSource::new(discord)));
            tracing::info!("Discord source initialized");
        } else {
            tracing::warn!("Discord source_enabled but missing bot_token or channel_id; skipping");
        }
    }

    let slack = config.slack_merged();
    if slack.source_enabled {
        if slack.bot_token.is_some()
            && (slack.listen_channel_id.is_some() || slack.channel_id.is_some())
        {
            sources.push(Arc::new(SlackSource::new(slack)));
            tracing::info!("Slack source initialized");
        } else {
            tracing::warn!("Slack source_enabled but missing bot_token or channel_id; skipping");
        }
    }

    if config.notifiers.whatsapp.source_enabled {
        if config.notifiers.whatsapp.access_token.is_some()
            && config.notifiers.whatsapp.phone_number_id.is_some()
        {
            sources.push(Arc::new(WhatsAppSource::new(
                config.notifiers.whatsapp.clone(),
            )));
            tracing::info!("WhatsApp source initialized");
        } else {
            tracing::warn!(
                "WhatsApp source_enabled but missing access_token or phone_number_id; skipping"
            );
        }
    }

    if config.notifiers.telegram.source_enabled {
        if config.notifiers.telegram.bot_token.is_some() {
            sources.push(Arc::new(TelegramSource::new(
                config.notifiers.telegram.clone(),
            )));
            tracing::info!("Telegram source initialized");
        } else {
            tracing::warn!("Telegram source_enabled but missing bot_token; skipping");
        }
    }

    sources.into_iter().map(InstrumentedSource::wrap).collect()
}

fn create_webhook_handlers(config: &Config) -> WebhookHandlerRegistry {
    let mut registry = WebhookHandlerRegistry::new();

    if let Some(linear_config) = config.linear() {
        if linear_config.enabled {
            registry.register(Arc::new(LinearWebhookHandler::new(linear_config.clone())));
            tracing::info!("Linear webhook handler registered");
        }
    }

    if let Some(sentry_config) = config.sentry_config() {
        if sentry_config.enabled {
            registry.register(Arc::new(SentryWebhookHandler::new(sentry_config.clone())));
            tracing::info!("Sentry webhook handler registered");
        }
    }

    if let Some(helpscout_config) = config.helpscout() {
        if helpscout_config.enabled && helpscout_config.webhook_secret.is_some() {
            registry.register(Arc::new(HelpScoutWebhookHandler::new(
                helpscout_config.clone(),
            )));
            tracing::info!("HelpScout webhook handler registered");
        }
    }

    if let Some(gitlab_config) = config.gitlab() {
        if gitlab_config.enabled {
            registry.register(Arc::new(GitLabIssueWebhookHandler::new(
                gitlab_config.clone(),
            )));
            tracing::info!("GitLab webhook handler registered");
        }
    }

    if let Some(jira_config) = config.jira() {
        if jira_config.enabled {
            registry.register(Arc::new(JiraWebhookHandler::new(jira_config.clone())));
            tracing::info!("Jira webhook handler registered");
        }
    }

    if let Some(slack_config) = config.issues.slack.as_ref() {
        registry.register(Arc::new(SlackWebhookHandler::new(slack_config.clone())));
        tracing::info!("Slack webhook handler registered");
    }

    if config.notifiers.telegram.source_enabled {
        registry.register(Arc::new(TelegramWebhookHandler::new(
            config.notifiers.telegram.clone(),
        )));
        tracing::info!("Telegram webhook handler registered");
    }

    if config.notifiers.whatsapp.source_enabled {
        registry.register(Arc::new(WhatsAppWebhookHandler::new(
            config.notifiers.whatsapp.clone(),
        )));
        tracing::info!("WhatsApp webhook handler registered");
    }

    registry
}

fn create_github_webhook_handler(
    config: &Config,
    review_watcher: Option<Arc<ReviewWatcher>>,
) -> Option<GitHubWebhookHandler> {
    let handler = GitHubWebhookHandler::new(config.github().clone(), review_watcher);
    if handler.is_enabled() {
        tracing::info!("GitHub webhook handler registered");
        Some(handler)
    } else {
        None
    }
}

fn create_notifier(config: &Config, user_registry: UserRegistry) -> Arc<dyn Notifier> {
    let mut composite = CompositeNotifier::new();

    // Always add console notifier
    composite.add(InstrumentedNotifier::wrap(Arc::new(ConsoleNotifier::new())));

    // Add Discord if configured
    let discord_notifier = DiscordNotifier::new(config.discord_merged(), user_registry.clone());
    if discord_notifier.is_enabled() {
        composite.add(InstrumentedNotifier::wrap(Arc::new(discord_notifier)));
        tracing::info!("Discord notifier enabled");
    }

    // Add Slack if configured
    let slack_notifier = SlackNotifier::new(config.slack_merged(), user_registry.clone());
    if slack_notifier.is_enabled() {
        composite.add(InstrumentedNotifier::wrap(Arc::new(slack_notifier)));
        tracing::info!("Slack notifier enabled");
    }

    // Add Email if configured
    if let Ok(email_notifier) = EmailNotifier::new(config.email().clone(), user_registry.clone()) {
        if email_notifier.is_enabled() {
            composite.add(InstrumentedNotifier::wrap(Arc::new(email_notifier)));
            tracing::info!("Email notifier enabled");
        }
    }

    // Add SMS if configured
    let sms_notifier = SmsNotifier::new(config.sms().clone(), user_registry.clone());
    if sms_notifier.is_enabled() {
        composite.add(InstrumentedNotifier::wrap(Arc::new(sms_notifier)));
        tracing::info!("SMS notifier enabled");
    }

    // Add Push if configured
    let push_notifier = PushNotifier::new(config.push_config().clone(), user_registry.clone());
    if push_notifier.is_enabled() {
        composite.add(InstrumentedNotifier::wrap(Arc::new(push_notifier)));
        tracing::info!("Push notifier enabled");
    }

    // Add WhatsApp if configured
    let whatsapp_notifier =
        WhatsAppNotifier::new(config.notifiers.whatsapp.clone(), user_registry.clone());
    if whatsapp_notifier.is_enabled() {
        composite.add(InstrumentedNotifier::wrap(Arc::new(whatsapp_notifier)));
        tracing::info!("WhatsApp notifier enabled");
    }

    // Add Telegram if configured
    let telegram_notifier = TelegramNotifier::new(config.notifiers.telegram.clone(), user_registry);
    if telegram_notifier.is_enabled() {
        composite.add(InstrumentedNotifier::wrap(Arc::new(telegram_notifier)));
        tracing::info!("Telegram notifier enabled");
    }

    Arc::new(composite)
}

fn create_tracker(config: &Config) -> Arc<dyn FixAttemptTracker> {
    Arc::new(SqliteTracker::new(&config.db_path).expect("Failed to initialize SQLite tracker"))
}

/// Start the regression monitoring background tasks.
///
/// This runs two background tasks:
/// 1. ReleaseTracker: Checks if fixes have been included in releases and transitions watches to Monitoring
/// 2. RegressionScheduler: Runs hourly checks on watches in Monitoring state
///
/// When a regression is detected, automatically triggers a retry of the fix attempt.
///
/// Returns a join handle that can be used to stop the monitoring.
fn start_regression_monitoring(
    config: &Config,
    tracker: Arc<dyn FixAttemptTracker>,
    sources: Vec<Arc<dyn IssueSource>>,
    notifier: Arc<dyn Notifier>,
) -> Option<tokio::task::JoinHandle<()>> {
    if !config.regression.enabled {
        tracing::info!("Regression monitoring disabled in configuration");
        return None;
    }

    // Get GitHub token (from regression config or fall back to github config)
    let github_token = config.regression.github_token.clone().or_else(|| {
        config
            .github()
            .token
            .as_ref()
            .map(|t| t.expose().to_string())
    });

    let github_token = match github_token {
        Some(token) if !token.is_empty() => token,
        _ => {
            tracing::warn!("No GitHub token configured, regression monitoring disabled");
            return None;
        }
    };

    // Create release tracker config (uses default dependency chains for transitive tracking)
    let effective_check_secs = config.regression.effective_check_interval_secs();
    let release_config = ReleaseTrackerConfig {
        target_repos: config.regression.target_repos.clone(),
        poll_interval_ms: effective_check_secs * 1000,
        package_names: config.regression.package_names.clone(),
    };

    // Create scheduler config
    let scheduler_config = RegressionSchedulerConfig {
        check_interval_secs: effective_check_secs,
        monitoring_duration_secs: config.regression.effective_monitoring_duration_secs(),
        sentry_event_threshold: config.regression.sentry_event_threshold,
        similarity_threshold: config.regression.similarity_threshold,
    };

    // Create the sentry regression checker if sentry is configured
    let sentry_checker: Box<dyn claudear::regression::RegressionChecker> =
        if let Some(sentry_config) = config.sentry_config() {
            if sentry_config.enabled && !sentry_config.auth_token.is_empty() {
                let sentry_regression_config = SentryRegressionConfig {
                    auth_token: sentry_config.auth_token.expose().to_string(),
                    org_slug: sentry_config.org_slug.clone(),
                    event_threshold: config.regression.sentry_event_threshold,
                };
                // Use the public HTTP client
                let http_client = claudear::source::sentry::ReqwestSentryClient::new();
                Box::new(SentryRegressionChecker::new(
                    sentry_regression_config,
                    http_client,
                ))
            } else {
                Box::new(NoOpChecker)
            }
        } else {
            Box::new(NoOpChecker)
        };

    // Create the linear regression checker
    // NOTE: LinearRegressionChecker requires per-issue keywords for effective similarity matching.
    // Currently we create it with empty keywords, which means it will only check appwrite.io/threads
    // and won't be able to search GitHub issues effectively. A future improvement would be to
    // store issue keywords in the regression_watches table and load them during each check.
    let linear_checker: Box<dyn claudear::regression::RegressionChecker> = {
        let linear_regression_config = LinearRegressionConfig {
            github_token: github_token.clone(),
            scm_repos: config.regression.github_search_repos.clone(),
            similarity_threshold: config.regression.similarity_threshold,
        };
        tracing::warn!(
            "Linear regression checker initialized without issue-specific keywords. \
             GitHub issue search will be limited. Only thread similarity checking will work."
        );
        Box::new(LinearRegressionChecker::new(
            linear_regression_config,
            vec![],
            String::new(),
        ))
    };

    // Create composite checker
    let composite_checker = CompositeChecker::new(sentry_checker, linear_checker);

    // Create release tracker and scheduler
    let release_tracker =
        ReleaseTracker::with_config(github_token, tracker.clone(), release_config);

    let scheduler =
        RegressionScheduler::new(composite_checker, tracker.clone(), scheduler_config.clone());

    let check_interval_secs = scheduler_config.check_interval_secs.max(1);
    if scheduler_config.check_interval_secs == 0 {
        tracing::warn!(
            component = "regression_monitor",
            "check_interval_secs evaluated to 0, clamping to 1 second to avoid timer panic"
        );
    }

    // Create retry manager for triggering retries on regression
    let retry_manager = RetryManager::new(config.retry.clone(), tracker.clone());

    // Start background task
    let handle = tokio::spawn(async move {
        let mut release_check_interval = interval(Duration::from_secs(300)); // Check for releases every 5 minutes
        let mut regression_check_interval = interval(Duration::from_secs(check_interval_secs));

        tracing::info!(
            component = "regression_monitor",
            check_interval_secs = check_interval_secs,
            "Regression monitoring started"
        );

        loop {
            tokio::select! {
                _ = release_check_interval.tick() => {
                    // Check for releases and transition watches to Monitoring
                    match release_tracker.check_pending_watches().await {
                        Ok(transitioned) => {
                            if !transitioned.is_empty() {
                                tracing::info!(
                                    component = "regression_monitor",
                                    count = transitioned.len(),
                                    "Transitioned watches to monitoring state"
                                );
                            }
                        }
                        Err(e) => {
                            tracing::error!(
                                component = "regression_monitor",
                                error = %e,
                                "Error checking for releases"
                            );
                        }
                    }
                }
                _ = regression_check_interval.tick() => {
                    // Run regression checks on watches in Monitoring state
                    match scheduler.check_monitoring_watches().await {
                        Ok(results) => {
                            for result in &results {
                                if result.regression_detected {
                                    tracing::warn!(
                                        component = "regression_monitor",
                                        watch_id = result.watch_id,
                                        check_number = result.check_number,
                                        issue_id = %result.issue_id,
                                        "Regression detected! Triggering retry."
                                    );

                                    // Trigger retry for the regressed issue
                                    let source = result.issue_type.source_name();
                                    match retry_manager.handle_regression(source, &result.issue_id) {
                                        Ok(decision) => {
                                            tracing::info!(
                                                component = "regression_monitor",
                                                issue_id = %result.issue_id,
                                                decision = ?decision,
                                                "Retry scheduled for regressed issue"
                                            );
                                        }
                                        Err(e) => {
                                            tracing::error!(
                                                component = "regression_monitor",
                                                issue_id = %result.issue_id,
                                                error = %e,
                                                "Failed to schedule retry for regressed issue"
                                            );
                                        }
                                    }

                                    // Notify regression detected
                                    let mut regression_issue = Issue::new(
                                        &result.issue_id,
                                        &result.issue_id,
                                        format!("Regression detected on check #{}", result.check_number),
                                        "",
                                        source,
                                    );
                                    regression_issue.set_metadata("regression_detected", true);
                                    let _ = notifier.notify_failed(
                                        &regression_issue,
                                        &format!("Regression detected on check #{}", result.check_number),
                                    ).await;
                                } else if result.is_final_check {
                                    tracing::info!(
                                        component = "regression_monitor",
                                        watch_id = result.watch_id,
                                        "Final check complete, no regression - issue resolved"
                                    );
                                    let source_name = result.issue_type.source_name();
                                    if let Some(source) =
                                        sources.iter().find(|s| s.name() == source_name)
                                    {
                                        match source.resolve_issue(&result.issue_id).await {
                                            Ok(()) => {
                                                if let Err(e) =
                                                    tracker.mark_resolved(source_name, &result.issue_id)
                                                {
                                                    tracing::warn!(
                                                        component = "regression_monitor",
                                                        source = source_name,
                                                        issue_id = %result.issue_id,
                                                        error = %e,
                                                        "Failed to mark attempt as resolved after final regression check"
                                                    );
                                                }
                                                // Notify regression resolved
                                                let mut resolved_issue = Issue::new(
                                                    &result.issue_id,
                                                    &result.issue_id,
                                                    format!("Regression monitoring complete: {} resolved", result.issue_id),
                                                    "",
                                                    source_name,
                                                );
                                                resolved_issue.set_metadata("regression_resolved", true);
                                                let _ = notifier.notify_completed(&resolved_issue).await;
                                            }
                                            Err(e) => {
                                                tracing::warn!(
                                                    component = "regression_monitor",
                                                    source = source_name,
                                                    issue_id = %result.issue_id,
                                                    error = %e,
                                                    "Failed to resolve source issue after final regression check"
                                                );
                                            }
                                        }
                                    } else {
                                        tracing::warn!(
                                            component = "regression_monitor",
                                            source = source_name,
                                            issue_id = %result.issue_id,
                                            "No source handler found to resolve issue after final regression check"
                                        );
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            tracing::error!(
                                component = "regression_monitor",
                                error = %e,
                                "Error running regression checks"
                            );
                        }
                    }
                }
            }
        }
    });

    Some(handle)
}

/// Enrich the process PATH with entries from the user's login shell.
///
/// When started via systemd or cron, the PATH is minimal and won't include
/// user-local directories like `~/.local/bin` or nvm/fnm node paths. This
/// runs `$SHELL -l -c 'echo $PATH'` to capture the full login PATH and
/// prepends any missing entries to the current PATH.
fn enrich_path_from_login_shell() {
    let shell = std::env::var("SHELL").unwrap_or_else(|_| "/bin/sh".to_string());
    let output = std::process::Command::new(&shell)
        .args(["-l", "-c", "echo $PATH"])
        .output();

    let login_path = match output {
        Ok(out) if out.status.success() => String::from_utf8_lossy(&out.stdout).trim().to_string(),
        _ => return,
    };

    if login_path.is_empty() {
        return;
    }

    let current_path = std::env::var("PATH").unwrap_or_default();
    let current_entries: std::collections::HashSet<&str> = current_path.split(':').collect();

    let mut new_entries: Vec<&str> = Vec::new();
    for entry in login_path.split(':') {
        if !entry.is_empty() && !current_entries.contains(entry) {
            new_entries.push(entry);
        }
    }

    if !new_entries.is_empty() {
        let merged = if current_path.is_empty() {
            new_entries.join(":")
        } else {
            format!("{}:{}", new_entries.join(":"), current_path)
        };
        std::env::set_var("PATH", &merged);
    }
}

/// Daemonize the current process using the classic double-fork pattern.
///
/// 1. First fork: parent prints the daemon PID and exits, returning control to the shell.
/// 2. Child calls `setsid()` to become a session leader (detach from terminal).
/// 3. Second fork: ensures the daemon cannot reacquire a controlling terminal.
/// 4. Grandchild redirects stdin/stdout/stderr to `/dev/null` and continues execution.
///
/// This must be called BEFORE the tokio runtime is created — forking after
/// spawning threads leads to undefined behaviour.
#[cfg(unix)]
fn daemonize() -> anyhow::Result<()> {
    use std::ffi::CString;

    // First fork
    let pid = unsafe { libc::fork() };
    if pid < 0 {
        anyhow::bail!("First fork failed: {}", std::io::Error::last_os_error());
    }
    if pid > 0 {
        // Parent: wait briefly for the child to setsid + second-fork so we can
        // report the final daemon PID. The child writes it to the PID file path
        // as a temp signal file, then we read it.
        //
        // Simpler approach: the first child will print the PID and the parent just exits.
        // We don't know the grandchild PID here, so the child will print it.
        std::process::exit(0);
    }

    // First child: create a new session
    if unsafe { libc::setsid() } < 0 {
        anyhow::bail!("setsid failed: {}", std::io::Error::last_os_error());
    }

    // Second fork
    let pid2 = unsafe { libc::fork() };
    if pid2 < 0 {
        anyhow::bail!("Second fork failed: {}", std::io::Error::last_os_error());
    }
    if pid2 > 0 {
        // First child exits — grandchild is now fully detached
        std::process::exit(0);
    }

    // Grandchild: this is the actual daemon process.
    // Print the PID so users know what to expect (goes to original stdout before redirect).
    eprintln!("Daemon started (PID: {})", std::process::id());

    // Change working directory to / to avoid holding mount points
    // (skip this — claudear needs its working dir for config and repos)

    // Set umask
    unsafe { libc::umask(0o027) };

    // Redirect stdin/stdout/stderr to /dev/null
    let devnull = CString::new("/dev/null").unwrap();
    unsafe {
        let fd = libc::open(devnull.as_ptr(), libc::O_RDWR);
        if fd >= 0 {
            libc::dup2(fd, libc::STDIN_FILENO);
            libc::dup2(fd, libc::STDOUT_FILENO);
            libc::dup2(fd, libc::STDERR_FILENO);
            if fd > libc::STDERR_FILENO {
                libc::close(fd);
            }
        }
    }

    Ok(())
}

fn main() -> anyhow::Result<()> {
    // Resolve the user's login shell PATH so that user-local binaries
    // (e.g. claude via nvm/node) are discoverable when started via systemd.
    enrich_path_from_login_shell();

    // Parse CLI args early so we can daemonize before creating the tokio runtime.
    // Forking after spawning threads is undefined behaviour.
    let cli = Cli::parse();

    #[cfg(unix)]
    if let Commands::Start { foreground, .. } = &cli.command {
        if !foreground {
            daemonize()?;
        }
    }

    // Initialize Sentry before the async runtime to ensure proper flushing on shutdown
    let _sentry_guard = sentry::init((
        std::env::var("CLAUDEAR_SENTRY_DSN").unwrap_or_default(),
        sentry::ClientOptions {
            release: std::env::var("CLAUDEAR_SENTRY_RELEASE")
                .ok()
                .filter(|s| !s.is_empty())
                .map(Into::into)
                .or_else(|| sentry::release_name!()),
            environment: std::env::var("CLAUDEAR_SENTRY_ENVIRONMENT")
                .ok()
                .map(Into::into),
            traces_sample_rate: 0.2,
            ..Default::default()
        },
    ));

    sentry::configure_scope(|scope| {
        scope.set_tag("app.component", "claudear-backend");
    });

    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(async_main(cli))
}

async fn async_main(cli: Cli) -> anyhow::Result<()> {
    let verbose = cli.verbose;

    // Initialize logging (must keep _guard alive for file logging to work)
    // Empty path disables file logging
    let log_dir = if cli.log_dir.as_os_str().is_empty() {
        None
    } else {
        Some(cli.log_dir.as_path())
    };
    let suppress_console_info = !verbose
        && matches!(
            &cli.command,
            Commands::Start { .. } | Commands::Webhook { .. } | Commands::Poll { .. }
        );
    let _log_guard = init_logging(log_dir, verbose, suppress_console_info);

    let config_path = cli.config.clone();

    // Load and validate config from YAML file
    let config = Config::load(&config_path)?;

    // Handle daemon control commands early (don't need full config validation)
    match &cli.command {
        Commands::Stop => {
            if !is_daemon_running() {
                println!("No daemon is running.");
                return Ok(());
            }

            let client = IpcClient::new();
            match client.shutdown().await {
                Ok(response) => {
                    print_response(&response);
                }
                Err(e) => {
                    eprintln!("Failed to stop daemon: {}", e);
                    std::process::exit(1);
                }
            }
            return Ok(());
        }

        Commands::Status => {
            if !is_daemon_running() {
                println!("No daemon is running.");
                println!("Socket path: {:?}", default_socket_path());
                return Ok(());
            }

            let client = IpcClient::new();
            match client.status().await {
                Ok(response) => {
                    print_response(&response);
                }
                Err(e) => {
                    eprintln!("Failed to get status: {}", e);
                    std::process::exit(1);
                }
            }
            return Ok(());
        }

        Commands::Pause => {
            if !is_daemon_running() {
                anyhow::bail!("No daemon is running. Start one with 'claudear start'");
            }

            let client = IpcClient::new();
            match client.pause().await {
                Ok(response) => {
                    print_response(&response);
                }
                Err(e) => {
                    eprintln!("Failed to pause: {}", e);
                    std::process::exit(1);
                }
            }
            return Ok(());
        }

        Commands::Resume => {
            if !is_daemon_running() {
                anyhow::bail!("No daemon is running. Start one with 'claudear start'");
            }

            let client = IpcClient::new();
            match client.resume().await {
                Ok(response) => {
                    print_response(&response);
                }
                Err(e) => {
                    eprintln!("Failed to resume: {}", e);
                    std::process::exit(1);
                }
            }
            return Ok(());
        }

        Commands::Activity { limit } => {
            if !is_daemon_running() {
                anyhow::bail!("No daemon is running. Start one with 'claudear start'");
            }

            let client = IpcClient::new();
            match client.activity(*limit).await {
                Ok(response) => {
                    print_response(&response);
                }
                Err(e) => {
                    eprintln!("Failed to get activity: {}", e);
                    std::process::exit(1);
                }
            }
            return Ok(());
        }

        _ => {}
    }

    // Handle sources command early (doesn't need full validation)
    if matches!(cli.command, Commands::Sources) {
        println!("\nConfigured Sources:");
        if config.is_linear_enabled() {
            if let Some(linear) = config.linear() {
                println!("  Linear (labels: {})", linear.trigger_labels.join(", "));
                if linear.webhook_secret.is_some() {
                    println!("    Webhook secret: configured");
                }
            }
        } else {
            println!("  Linear (not configured)");
        }

        if config.is_sentry_enabled() {
            if let Some(sentry) = config.sentry_config() {
                println!("  Sentry (org: {})", sentry.org_slug);
                if sentry.client_secret.is_some() {
                    println!("    Client secret: configured");
                }
            }
        } else {
            println!("  Sentry (not configured)");
        }

        println!("\nRate Limiting:");
        println!("  Max issues per cycle: {}", config.max_issues_per_cycle);
        println!("  Max concurrent: {}", config.max_concurrent);
        println!("  Processing delay: {}ms", config.processing_delay_ms);

        println!("\nNotifiers:");
        println!("  Console: enabled");
        if config.notifiers.discord.webhook_url.is_some() {
            println!("  Discord: enabled");
        }
        if config.email().smtp_host.is_some() {
            println!("  Email: enabled");
        }
        if config.sms().account_sid.is_some() {
            println!("  SMS (Twilio): enabled");
        }
        if config.push_config().api_token.is_some() {
            println!("  Push (Pushover): enabled");
        }

        return Ok(());
    }

    // Handle Repos commands early (don't need sources or repository validation)
    if let Commands::Repos(ref repos_cmd) = cli.command {
        let db_tracker = SqliteTracker::new(&config.db_path)?;

        /// Guard that calls `finish_indexing_progress` on drop so error paths
        /// (via `?`) never leave the DB stuck in "running" status.
        struct IndexingGuard<'a> {
            tracker: &'a SqliteTracker,
        }
        impl<'a> Drop for IndexingGuard<'a> {
            fn drop(&mut self) {
                let _ = self.tracker.finish_indexing_progress();
            }
        }

        match repos_cmd {
            ReposCommands::List => {
                // First show indexed repos from the database
                let indexed_repos = db_tracker.list_indexed_repos()?;

                if indexed_repos.is_empty() {
                    println!("\nNo indexed repositories.");
                    println!("Run 'claudear repos index' to build the repository index.");
                } else {
                    println!("\nIndexed Repositories ({}):", indexed_repos.len());
                    for repo in &indexed_repos {
                        println!(
                            "  {} - {} files [{}]",
                            repo.name, repo.file_count, repo.path
                        );
                    }
                }

                // Show known orgs from config
                if !config.known_orgs.is_empty() {
                    println!("\nKnown Organizations: {:?}", config.known_orgs);
                }

                // Show auto-discover paths
                if !config.auto_discover_paths.is_empty() {
                    println!("Auto-Discover Paths: {:?}", config.auto_discover_paths);
                }
            }

            ReposCommands::Index { force } => {
                use claudear::DependencyDiscovery;
                use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

                println!("\nBuilding repository index...");
                println!("  Known orgs: {:?}", config.known_orgs);
                println!("  Scanning: {:?}", config.auto_discover_paths);

                if config.known_orgs.is_empty() || config.auto_discover_paths.is_empty() {
                    anyhow::bail!(
                        "known_orgs and auto_discover_paths must be configured in claudear.toml"
                    );
                }

                let index = build_repo_index(&config.known_orgs, &config.auto_discover_paths)?;

                if index.is_empty() {
                    println!("\nNo repositories found from known orgs in the specified paths.");
                    return Ok(());
                }

                let repos = index.list();
                let total_repos = repos.len();

                // Track progress in DB for dashboard
                let _ = db_tracker.start_indexing_progress(total_repos);
                let _indexing_guard = IndexingGuard {
                    tracker: &db_tracker,
                };

                let multi = MultiProgress::new();

                // Overall progress bar
                let overall_style = ProgressStyle::with_template(
                    "{prefix:.bold} [{bar:40.cyan/dim}] {pos}/{len} repos ({msg})",
                )
                .unwrap()
                .progress_chars("=> ");
                let overall_pb = multi.add(ProgressBar::new(total_repos as u64));
                overall_pb.set_style(overall_style);
                overall_pb.set_prefix("Indexing");

                // Current repo progress bar
                let repo_style = ProgressStyle::with_template(
                    "  {prefix:.dim} [{bar:35.green/dim}] {pos}/{len} files",
                )
                .unwrap()
                .progress_chars("=> ");
                let repo_pb = multi.add(ProgressBar::new(0));
                repo_pb.set_style(repo_style);

                let mut saved_count = 0;
                let mut skipped_count = 0;
                let mut total_files_saved: usize = 0;

                for repo in &repos {
                    // Check if already indexed (skip unless force)
                    if !force {
                        if let Ok(Some(_)) = db_tracker.get_indexed_repo(&repo.name) {
                            skipped_count += 1;
                            overall_pb.inc(1);
                            overall_pb.set_message(format!(
                                "{} saved, {} skipped",
                                saved_count, skipped_count
                            ));
                            // Notify dashboard so progress bar doesn't jump
                            let _ = db_tracker.update_indexing_progress(
                                saved_count + skipped_count,
                                &repo.name,
                                0,
                                total_files_saved,
                            );
                            continue;
                        }
                    }

                    // Update current repo progress
                    repo_pb.set_length(repo.files.len() as u64);
                    repo_pb.set_position(0);
                    repo_pb.set_prefix(repo.name.clone());

                    // Update dashboard progress
                    let _ = db_tracker.update_indexing_progress(
                        saved_count + skipped_count,
                        &repo.name,
                        repo.files.len(),
                        total_files_saved,
                    );

                    // Save repo metadata
                    let repo_id = db_tracker.save_indexed_repo(
                        &repo.name,
                        &repo.path.to_string_lossy(),
                        Some(repo.scm_url.as_str()),
                        &repo.default_branch,
                        repo.files.len(),
                    )?;

                    // Convert files to (path, file_type) tuples for storage
                    let files_with_types: Vec<(String, Option<String>)> = repo
                        .files
                        .iter()
                        .map(|f: &String| {
                            let file_type = std::path::Path::new(f)
                                .extension()
                                .map(|e| e.to_string_lossy().to_string());
                            (f.clone(), file_type)
                        })
                        .collect();

                    // Save file index
                    db_tracker.save_repo_files(repo_id, &files_with_types)?;

                    total_files_saved += repo.files.len();
                    repo_pb.set_position(repo.files.len() as u64);

                    saved_count += 1;
                    overall_pb.inc(1);
                    overall_pb
                        .set_message(format!("{} saved, {} skipped", saved_count, skipped_count));
                }

                repo_pb.finish_and_clear();
                overall_pb.finish_with_message(format!(
                    "{} saved, {} skipped, {} total files",
                    saved_count, skipped_count, total_files_saved
                ));

                println!(
                    "\nIndexed {} repositories ({} files) to {:?}",
                    saved_count, total_files_saved, config.db_path
                );

                // Auto-discover dependencies between indexed repos
                let dep_pb = multi.add(ProgressBar::new_spinner());
                dep_pb.set_style(
                    ProgressStyle::with_template("{spinner:.blue} {msg}")
                        .unwrap()
                        .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"),
                );
                dep_pb.set_message("Discovering dependencies...");
                dep_pb.enable_steady_tick(std::time::Duration::from_millis(80));

                let discovery = DependencyDiscovery::new(config.known_orgs.clone());
                match discovery.scan_directories(&config.auto_discover_paths) {
                    Ok(discovered) if !discovered.is_empty() => {
                        let mut dep_count = 0;
                        for dep in &discovered {
                            if let Err(e) =
                                db_tracker.add_dependency(&dep.depends_on, &dep.repo, &dep.dep_type)
                            {
                                tracing::warn!(
                                    error = %e,
                                    upstream = %dep.depends_on,
                                    downstream = %dep.repo,
                                    "Failed to save dependency"
                                );
                            } else {
                                dep_count += 1;
                            }
                        }
                        dep_pb.finish_with_message(format!(
                            "Discovered and saved {} dependencies",
                            dep_count
                        ));
                    }
                    Ok(_) => {
                        dep_pb.finish_with_message("No dependencies found between indexed repos.")
                    }
                    Err(e) => dep_pb.finish_with_message(format!(
                        "Warning: dependency discovery failed: {}",
                        e
                    )),
                }

                // Guard will call finish_indexing_progress() on drop
            }

            ReposCommands::Search { query } => {
                // Build index from config for searching
                if config.known_orgs.is_empty() || config.auto_discover_paths.is_empty() {
                    anyhow::bail!("known_orgs and auto_discover_paths must be configured");
                }

                println!("\nSearching for '{}'...", query);

                let index = build_repo_index(&config.known_orgs, &config.auto_discover_paths)?;
                let matches = index.search_files(query);

                if matches.is_empty() {
                    println!("No matches found.");
                } else {
                    println!("\nMatches ({}):", matches.len());
                    for (repo, file) in matches.iter().take(50) {
                        println!("  [{}] {}", repo.name, file);
                    }
                    if matches.len() > 50 {
                        println!("  ... and {} more", matches.len() - 50);
                    }
                }
            }

            ReposCommands::Stats => {
                let stats = db_tracker.get_index_stats()?;

                println!("\nRepository Index Statistics:");
                println!("  Total repositories: {}", stats.repo_count);
                println!("  Total files indexed: {}", stats.file_count);

                if stats.repo_count > 0 {
                    println!(
                        "  Average files per repo: {:.1}",
                        stats.file_count as f64 / stats.repo_count as f64
                    );
                }

                if let Some(ref last_indexed) = stats.last_indexed_at {
                    println!("  Last indexed: {}", last_indexed);
                }

                // Also show inference stats if available
                if let Ok(inference_stats) = db_tracker.get_inference_stats() {
                    println!("\n  Inference Statistics:");
                    println!("    Total attempts: {}", inference_stats.total_attempts);
                    if inference_stats.total_attempts > 0 {
                        println!("    With feedback: {}", inference_stats.with_feedback);
                        if inference_stats.with_feedback > 0 {
                            println!("    Accuracy: {:.1}%", inference_stats.accuracy * 100.0);
                        }
                    }
                }
            }

            ReposCommands::Add {
                name,
                path,
                scm_url,
            } => {
                println!("\n=== Deprecated ===\n");
                println!("The 'repos add' command is deprecated.");
                println!("Repositories are now auto-discovered from known_orgs config.");
                println!("\nUse 'claudear repos index' instead.");
                println!(
                    "\nIf you still want to manually track '{}', add the org to known_orgs",
                    name
                );
                let _ = (path, scm_url); // Suppress unused warning
            }

            ReposCommands::Link {
                upstream,
                downstream,
                dep_type,
            } => {
                db_tracker.add_dependency(upstream, downstream, dep_type)?;
                println!(
                    "Linked: {} depends on {} ({})",
                    downstream, upstream, dep_type
                );
                println!("  Saved to database at: {:?}", config.db_path);
            }

            ReposCommands::Graph { root } => {
                // Load dependencies from DB
                let mut manager = RepoRelationships::with_defaults();

                // Add DB dependencies to the manager
                let db_deps = db_tracker.list_all_dependencies()?;
                for dep in db_deps {
                    let dtype =
                        DependencyType::parse(&dep.dep_type).unwrap_or(DependencyType::Manual);
                    manager
                        .add_dependency(&dep.upstream, &dep.downstream, dtype, None)
                        .ok();
                }

                println!("\n=== Repository Dependency Graph ===\n");
                let tree = manager.print_tree(root.as_deref());
                print!("{}", tree);

                if root.is_none() {
                    println!("\nLegend: upstream -> downstream (changes in upstream may require updates in downstream)");
                }
            }

            ReposCommands::Cascade { repo } => {
                // Query DB directly for dependants
                let direct = db_tracker.get_direct_dependants(repo)?;
                let all = db_tracker.get_all_dependants(repo)?;

                println!("\n=== Cascading Changes from {} ===\n", repo);

                if direct.is_empty() {
                    println!("  No downstream repositories depend on {}", repo);
                } else {
                    println!("  Changes in {} would trigger updates in:", repo);
                    for dep in &direct {
                        println!("    -> {} (VersionBump, {})", dep.downstream, dep.dep_type);
                    }

                    // Show transitive dependants (depth > 1)
                    let indirect: Vec<_> = all.iter().filter(|(_, depth)| *depth > 1).collect();
                    if !indirect.is_empty() {
                        println!("\n  Transitively affected:");
                        for (name, depth) in indirect {
                            println!("    -> {} (indirect, depth {})", name, depth);
                        }
                    }
                }
            }

            ReposCommands::Discover { paths, save, clear } => {
                use claudear::DependencyDiscovery;

                // Use provided paths or fall back to config
                let scan_paths = if paths.is_empty() {
                    config.auto_discover_paths.clone()
                } else {
                    paths.clone()
                };

                if scan_paths.is_empty() {
                    anyhow::bail!(
                        "No paths to scan. Provide --paths or set auto_discover_paths in config.\n\
                        Example: claudear repos discover --paths ~/Local"
                    );
                }

                println!("\n=== Auto-Discovering Dependencies ===\n");
                println!("Known orgs: {:?}", config.known_orgs);
                println!("Scanning: {:?}\n", scan_paths);

                let discovery = DependencyDiscovery::new(config.known_orgs.clone());
                let discovered = discovery.scan_directories(&scan_paths)?;

                if discovered.is_empty() {
                    println!("No dependencies found from known organizations.");
                    return Ok(());
                }

                // Group by repo for display
                let mut by_repo: std::collections::HashMap<String, Vec<_>> =
                    std::collections::HashMap::new();
                for dep in &discovered {
                    by_repo.entry(dep.repo.clone()).or_default().push(dep);
                }

                println!(
                    "Discovered {} dependencies across {} repositories:\n",
                    discovered.len(),
                    by_repo.len()
                );
                for (repo, deps) in &by_repo {
                    println!("  {}", repo);
                    for dep in deps {
                        println!("    -> {} ({})", dep.depends_on, dep.dep_type);
                    }
                }

                if *save {
                    if *clear {
                        println!("\nClearing existing dependencies...");
                        db_tracker.clear_repositories()?;
                    }

                    println!("\nSaving to database...");
                    for dep in &discovered {
                        // Add repo with its path
                        db_tracker.upsert_repository(&dep.repo, Some(&dep.repo_path), None)?;
                        // Add dependency
                        db_tracker.add_dependency(&dep.depends_on, &dep.repo, &dep.dep_type)?;
                    }
                    println!(
                        "Saved {} dependencies to {:?}",
                        discovered.len(),
                        config.db_path
                    );
                } else {
                    println!("\nRun with --save to persist to database.");
                }
            }

            ReposCommands::Reindex { repo } => {
                use claudear::feedback::{EmbeddingClient, EmbeddingConfig};
                use claudear::repo::build_repo_index;
                use claudear::repo::code_index::CodeIndexer;

                if !config.code_index.enabled {
                    anyhow::bail!(
                        "Code indexing is disabled in config (code_index.enabled = false)"
                    );
                }

                // Build execution providers from config
                #[allow(unused_mut)]
                let mut execution_providers = Vec::new();
                if config.embedding.gpu {
                    #[cfg(feature = "cuda")]
                    {
                        let cuda_ep = ort::execution_providers::CUDA::default()
                            .with_device_id(config.embedding.device_id)
                            .build();
                        execution_providers.push(cuda_ep);
                        tracing::info!(
                            device_id = config.embedding.device_id,
                            "CUDA execution provider configured for embeddings"
                        );
                    }
                    #[cfg(not(feature = "cuda"))]
                    {
                        tracing::warn!(
                            "embedding.gpu = true but binary was compiled without --features cuda; falling back to CPU"
                        );
                    }
                }

                let emb_pool_size = if config.embedding.pool_size > 0 {
                    config.embedding.pool_size as usize
                } else if config.embedding.gpu {
                    1
                } else {
                    EmbeddingConfig::default().pool_size
                };

                let emb_config = EmbeddingConfig {
                    pool_size: emb_pool_size,
                    execution_providers,
                    sub_batch_size: config.embedding.sub_batch_size as usize,
                    ..EmbeddingConfig::default()
                };

                let emb_client = Arc::new(EmbeddingClient::new(emb_config)?);
                let tracker: Arc<dyn claudear::storage::FixAttemptTracker> =
                    Arc::new(SqliteTracker::new(&config.db_path)?);

                let code_indexer = CodeIndexer::with_config(
                    tracker.clone(),
                    emb_client,
                    config.code_index.max_file_size_kb,
                    config.code_index.batch_size,
                )
                .with_force_reindex(true);

                // Collect repos to reindex
                let index = build_repo_index(&config.known_orgs, &config.auto_discover_paths)?;
                let repos: Vec<_> = index
                    .list()
                    .into_iter()
                    .filter(|r| r.path.exists())
                    .filter(|r| repo.as_ref().is_none_or(|name| &r.name == name))
                    .collect();

                if repos.is_empty() {
                    if let Some(name) = &repo {
                        anyhow::bail!("Repository '{}' not found in index", name);
                    }
                    println!("No repositories found to reindex.");
                    return Ok(());
                }

                println!(
                    "\nForce re-indexing {} repositories (all existing embeddings will be regenerated)...",
                    repos.len()
                );

                for r in &repos {
                    print!("  {} ... ", r.name);
                    match code_indexer.index_repo(&r.name, &r.path).await {
                        Ok(stats) => {
                            println!(
                                "{} files, {} chunks, {} embeddings",
                                stats.files_processed,
                                stats.chunks_created,
                                stats.embeddings_generated
                            );
                        }
                        Err(e) => {
                            println!("FAILED: {}", e);
                        }
                    }
                }

                println!("\nDone.");
            }

            ReposCommands::Sync { skip_files } => {
                use claudear::repo::build_repo_index;

                println!("\nSyncing repository index to database...");

                if config.known_orgs.is_empty() || config.auto_discover_paths.is_empty() {
                    anyhow::bail!("known_orgs and auto_discover_paths must be configured");
                }

                // Build in-memory index
                let index = build_repo_index(&config.known_orgs, &config.auto_discover_paths)?;
                if index.is_empty() {
                    println!("No repositories found.");
                    return Ok(());
                }

                println!(
                    "  Found {} repositories with {} files",
                    index.len(),
                    index.total_files()
                );

                // Sync to database (includes files by default)
                let sync_files = !*skip_files;
                let synced = db_tracker.sync_from_index(&index, sync_files)?;

                println!("\nSynced {} repository paths to database", synced);
                if sync_files {
                    // Show stats after file sync
                    let stats = db_tracker.get_index_stats()?;
                    println!("  Including {} files in database", stats.file_count);
                } else {
                    println!("  Skipped file lists (use without --skip-files for full indexing)");
                }

                // Auto-discover dependencies
                use claudear::DependencyDiscovery;
                let discovery = DependencyDiscovery::new(config.known_orgs.clone());
                match discovery.scan_directories(&config.auto_discover_paths) {
                    Ok(discovered) if !discovered.is_empty() => {
                        let mut dep_count = 0;
                        for dep in &discovered {
                            if let Err(e) =
                                db_tracker.add_dependency(&dep.depends_on, &dep.repo, &dep.dep_type)
                            {
                                tracing::warn!(
                                    error = %e,
                                    upstream = %dep.depends_on,
                                    downstream = %dep.repo,
                                    "Failed to save dependency"
                                );
                            } else {
                                dep_count += 1;
                            }
                        }
                        println!("  Discovered and saved {} dependencies", dep_count);
                    }
                    Ok(_) => {
                        println!("  No dependencies found between indexed repos.");
                    }
                    Err(e) => {
                        println!("  Warning: dependency discovery failed: {}", e);
                    }
                }
            }
        }

        return Ok(());
    }

    // Handle Inference commands early
    if let Commands::Inference(ref inference_cmd) = cli.command {
        let db_tracker = SqliteTracker::new(&config.db_path)?;

        match inference_cmd {
            InferenceCommands::Stats => {
                let stats = db_tracker.get_inference_stats()?;

                println!("\nInference Statistics:");
                println!("  Total inference attempts: {}", stats.total_attempts);
                println!("  Attempts with feedback: {}", stats.with_feedback);

                if stats.with_feedback > 0 {
                    println!(
                        "  Correct inferences: {} ({:.1}%)",
                        stats.correct,
                        stats.accuracy * 100.0
                    );
                }

                println!("\n  By confidence level:");
                println!("    High:   {} attempts", stats.by_confidence.high);
                println!("    Medium: {} attempts", stats.by_confidence.medium);
                println!("    Low:    {} attempts", stats.by_confidence.low);
                println!("    None:   {} attempts", stats.by_confidence.none);
            }

            InferenceCommands::History { limit } => {
                let history = db_tracker.get_inference_history(*limit)?;

                if history.is_empty() {
                    println!("\nNo inference attempts recorded yet.");
                    println!("  Run 'claudear process' to start processing issues.");
                } else {
                    println!("\nRecent Inference Attempts (last {}):\n", history.len());

                    for entry in &history {
                        let status = match entry.was_correct {
                            Some(true) => "✓ correct",
                            Some(false) => "✗ incorrect",
                            None => "? pending",
                        };

                        let repo_display =
                            entry.inferred_repo_name.as_deref().unwrap_or("(no match)");

                        let confidence_display = entry.confidence.as_deref().unwrap_or("none");

                        let duration_display = entry
                            .duration_ms
                            .map(|ms| format!("{}ms", ms))
                            .unwrap_or_else(|| "-".to_string());

                        println!(
                            "  #{} [{}] {} → {} ({}) [{}]",
                            entry.id,
                            entry.issue_source,
                            entry.issue_id,
                            repo_display,
                            confidence_display,
                            status
                        );

                        if let Some(ref reason) = entry.inference_reason {
                            // Truncate long reasons
                            let truncated = if reason.len() > 60 {
                                format!("{}...", &reason[..reason.floor_char_boundary(57)])
                            } else {
                                reason.clone()
                            };
                            println!("      Reason: {}", truncated);
                        }

                        if let Some(ref keywords) = entry.extracted_keywords {
                            // Truncate long keyword lists
                            let truncated = if keywords.len() > 50 {
                                format!("{}...", &keywords[..keywords.floor_char_boundary(47)])
                            } else {
                                keywords.clone()
                            };
                            println!("      Keywords: {}", truncated);
                        }

                        println!(
                            "      Time: {} | Duration: {}",
                            entry.created_at, duration_display
                        );
                        println!();
                    }

                    println!("  Use 'claudear inference feedback <id> --correct/--incorrect' to provide feedback.");
                }
            }

            InferenceCommands::Feedback {
                id,
                correct,
                actual_repo,
            } => {
                let actual_repo_id = if let Some(ref repo_name) = actual_repo {
                    // Look up repo ID by name
                    if let Some(repo) = db_tracker.get_indexed_repo(repo_name)? {
                        Some(repo.id)
                    } else {
                        anyhow::bail!("Repository '{}' not found in index", repo_name);
                    }
                } else {
                    None
                };

                db_tracker.record_inference_feedback(*id, *correct, actual_repo_id, "manual")?;

                println!("\nFeedback recorded for inference attempt {}:", id);
                println!("  Correct: {}", correct);
                if let Some(ref repo_name) = actual_repo {
                    println!("  Actual repo: {}", repo_name);
                }
            }
        }

        return Ok(());
    }

    if let Commands::Purge { confirm } = cli.command {
        if is_daemon_running() {
            anyhow::bail!("Daemon is running. Stop it first with 'claudear stop' before purging.");
        }

        let db_tracker = SqliteTracker::new(&config.db_path)?;

        if !confirm {
            let counts = db_tracker.get_diagnostic_counts()?;

            println!("\nThis will DELETE all operational data:");
            println!("  fix_attempts:     {}", counts.fix_attempts);
            println!("  prs:              {}", counts.prs);
            println!("  pr_reviews:       {}", counts.pr_reviews);
            println!("  pr_review_states: {}", counts.pr_review_states);
            println!("  claude_executions:{}", counts.claude_executions);
            println!("  activity_log:     {}", counts.activity_log);
            println!("  processing_metrics:{}", counts.processing_metrics);
            println!(
                "  feedback_outcomes: {} (detached, not deleted)",
                counts.feedback_outcomes
            );

            println!("\nThis will PRESERVE:");
            println!("  issues (with embeddings): {}", counts.issues);
            println!("  similar_issues:           {}", counts.similar_issues);
            println!("  repositories:             {}", counts.repositories);
            println!("  repo_files:               {}", counts.repo_files);
            println!("  inference_attempts:        {}", counts.inference_attempts);
            println!("  error_patterns:           {}", counts.error_patterns);
            println!("  qa_knowledge, promoted_instructions, repo_knowledge, review_patterns");
            println!("  code_symbols, code_chunks, code_chunk_embeddings");

            println!("\nRe-run with --confirm to proceed.");
            return Ok(());
        }

        let result = db_tracker.purge_operational_data()?;

        println!("\nPurged operational data:");
        println!("  fix_attempts:         {}", result.fix_attempts);
        println!("  prs:                  {}", result.prs);
        println!("  pr_reviews:           {}", result.pr_reviews);
        println!("  pr_review_comments:   {}", result.pr_review_comments);
        println!("  pr_review_states:     {}", result.pr_review_states);
        println!("  claude_executions:    {}", result.claude_executions);
        println!("  strategy_fingerprints:{}", result.strategy_fingerprints);
        println!("  diff_analyses:        {}", result.diff_analyses);
        println!("  regression_watches:   {}", result.regression_watches);
        println!("  release_tracking:     {}", result.release_tracking);
        println!("  regression_checks:    {}", result.regression_checks);
        println!("  qa_usage:             {}", result.qa_usage);
        println!("  activity_log:         {}", result.activity_log);
        println!("  processing_metrics:   {}", result.processing_metrics);
        println!("  webhook_deliveries:   {}", result.webhook_deliveries);
        println!("  issue_clusters:       {}", result.issue_clusters);
        println!("  issue_cluster_members:{}", result.issue_cluster_members);
        println!("  content_clusters:     {}", result.content_clusters);
        println!("  severity_scores:      {}", result.severity_scores);
        println!("  suppression_log:      {}", result.suppression_log);
        println!("  eval_snapshots:       {}", result.eval_snapshots);
        println!("  eval_deltas:          {}", result.eval_deltas);

        println!("\nPreserved:");
        println!(
            "  feedback_outcomes detached: {}",
            result.feedback_outcomes_detached
        );
        println!("  Knowledge, inference, embeddings, and code index retained.");

        println!("\nTotal rows deleted: {}", result.total_deleted());

        return Ok(());
    }

    // Handle Diag commands early (don't need sources or full validation)
    if let Commands::Diag(ref diag_cmd) = cli.command {
        let db_tracker = SqliteTracker::new(&config.db_path)?;

        match diag_cmd {
            DiagCommands::Db => {
                let counts = db_tracker.get_diagnostic_counts()?;

                println!("\n=== Database Diagnostics ===\n");
                println!("Database path: {:?}", config.db_path);
                println!();

                println!("Table Counts:");
                println!("  fix_attempts:       {}", counts.fix_attempts);
                if !counts.fix_attempts_by_status.is_empty() {
                    println!("    By status:");
                    for (status, count) in &counts.fix_attempts_by_status {
                        println!("      {}: {}", status, count);
                    }
                }
                println!("  activity_log:       {}", counts.activity_log);
                println!("  claude_executions:  {}", counts.claude_executions);
                println!("  pr_reviews:         {}", counts.pr_reviews);
                println!("  pr_review_states:   {}", counts.pr_review_states);
                println!("  issues:             {}", counts.issues);
                println!("  similar_issues:     {}", counts.similar_issues);
                println!("  repositories:       {}", counts.repositories);
                println!("  repo_files:         {}", counts.repo_files);
                println!("  inference_attempts: {}", counts.inference_attempts);
                println!("  error_patterns:     {}", counts.error_patterns);
                println!("  processing_metrics: {}", counts.processing_metrics);
                println!("  feedback_outcomes:  {}", counts.feedback_outcomes);
                println!("  prs:                {}", counts.prs);

                if !counts.recent_fix_attempts.is_empty() {
                    println!("\nRecent Fix Attempts (last 5):");
                    for (source, issue_id, short_id, status) in &counts.recent_fix_attempts {
                        println!("  [{}] {} ({}) - {}", source, short_id, issue_id, status);
                    }
                } else {
                    println!("\nNo fix attempts recorded yet.");
                    println!("\nTo debug why fix_attempts is empty:");
                    println!(
                        "  1. Run with verbose logging: RUST_LOG=claudear=debug claudear poll"
                    );
                    println!("  2. Check that issues have the trigger labels configured");
                    println!("  3. Verify the watcher is processing issues (check activity_log)");
                }
            }

            DiagCommands::ReleaseGraph => {
                use claudear::repo::RepoRelationships;

                let relationships = RepoRelationships::with_defaults();
                println!("\n=== Dependency Graph for Release Tracking ===\n");
                println!("{}", relationships.print_tree(None));

                println!("Target repos (from config):");
                if config.regression.target_repos.is_empty() {
                    println!("  (none configured - set regression.target_repos in config)");
                } else {
                    for repo in &config.regression.target_repos {
                        println!("  - {}", repo);
                    }
                }
            }

            DiagCommands::ReleasePath { source, target } => {
                use claudear::repo::RepoRelationships;

                let relationships = RepoRelationships::with_defaults();
                let graph = relationships.get_graph();

                println!("\n=== Dependency Path: {} → {} ===\n", source, target);

                if graph.depends_on(target, source) {
                    println!(
                        "✓ Path exists: {} depends on {} (directly or transitively)",
                        target, source
                    );

                    // Show direct dependants of source
                    let dependants = relationships.get_dependants(source);
                    if !dependants.is_empty() {
                        println!("\nDirect dependants of {}:", source);
                        for dep in &dependants {
                            let dep_type = graph
                                .get_first_hop_dependency_type(source)
                                .map(|t| format!("{:?}", t))
                                .unwrap_or_else(|| "Unknown".to_string());
                            println!("  → {} ({})", dep.name, dep_type);
                        }
                    }

                    // Show what lock file would be checked
                    if let Some(dep_type) = graph.get_first_hop_dependency_type(source) {
                        let lock_file = match dep_type {
                            claudear::repo::DependencyType::Composer => "composer.lock",
                            claudear::repo::DependencyType::Npm => "package-lock.json",
                            claudear::repo::DependencyType::GitSubmodule => {
                                "(commit ancestry check)"
                            }
                            claudear::repo::DependencyType::Manual => "(release_after check)",
                        };
                        println!("\nVerification method: {:?} → {}", dep_type, lock_file);
                    }
                } else {
                    println!("✗ No path found: {} does not depend on {}", target, source);
                    println!("\nHint: Check if the dependency is defined in RepoRelationships::seed_appwrite_defaults()");
                }
            }

            DiagCommands::ReleaseCheck { repo, pr, target } => {
                use claudear::release::ReleaseClient;

                let github_token = config
                    .github()
                    .token
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("GitHub token not configured"))?;

                let client = ReleaseClient::new(github_token.expose());

                println!("\n=== Release Check: {}/#{} ===\n", repo, pr);

                // Get PR details
                match client.get_pr_details(repo, *pr).await? {
                    Some(pr_details) => {
                        println!(
                            "PR #{}: {}",
                            pr_details.number,
                            if pr_details.merged {
                                "MERGED"
                            } else {
                                "NOT MERGED"
                            }
                        );
                        if let Some(ref merged_at) = pr_details.merged_at {
                            println!("Merged at: {}", merged_at);
                        }
                        if let Some(ref sha) = pr_details.merge_commit_sha {
                            println!("Merge commit: {}", sha);
                        }

                        if !pr_details.merged {
                            println!("\n⚠ PR is not merged yet - cannot check release inclusion");
                            return Ok(());
                        }

                        // Determine target repos to check
                        let targets: Vec<String> = if let Some(t) = target {
                            vec![t.clone()]
                        } else if !config.regression.target_repos.is_empty() {
                            config.regression.target_repos.clone()
                        } else {
                            println!("\n⚠ No target repos configured. Use --target or set regression.target_repos");
                            return Ok(());
                        };

                        // Check each target
                        for target_repo in &targets {
                            println!("\n--- Checking target: {} ---", target_repo);

                            // Get latest release
                            match client.get_latest_release(target_repo).await? {
                                Some(release) => {
                                    println!(
                                        "Latest release: {} ({})",
                                        release.tag_name,
                                        release.published_at.as_deref().unwrap_or("no date")
                                    );

                                    // Check if commit is in release
                                    if let Some(ref sha) = pr_details.merge_commit_sha {
                                        let in_release = client
                                            .is_commit_in_release(
                                                target_repo,
                                                sha,
                                                &release.tag_name,
                                            )
                                            .await?;

                                        if in_release {
                                            println!(
                                                "✓ Merge commit {} IS in release {}",
                                                &sha[..7],
                                                release.tag_name
                                            );
                                        } else {
                                            println!(
                                                "✗ Merge commit {} is NOT in release {}",
                                                &sha[..7],
                                                release.tag_name
                                            );
                                        }
                                    }
                                }
                                None => {
                                    println!("No releases found in {}", target_repo);
                                }
                            }
                        }
                    }
                    None => {
                        println!("PR #{} not found in {}", pr, repo);
                    }
                }
            }
        }

        return Ok(());
    }

    // Handle Users commands early (don't need sources or full validation)
    if let Commands::Users(ref users_cmd) = cli.command {
        let db_tracker = SqliteTracker::new(&config.db_path)?;

        match users_cmd {
            UsersCommands::Seed {
                email,
                password,
                name,
            } => {
                let hash = bcrypt::hash(password, bcrypt::DEFAULT_COST)
                    .map_err(|e| anyhow::anyhow!("Failed to hash password: {}", e))?;

                match db_tracker.get_user_by_email(email)? {
                    Some(existing) => {
                        db_tracker.update_user(
                            existing.id,
                            None,
                            Some(&hash),
                            Some(name.as_str()),
                            Some("admin"),
                            None,
                        )?;
                        println!(
                            "Updated existing user '{}' (id={}) with new password and admin role",
                            email, existing.id
                        );
                    }
                    None => {
                        let id = db_tracker.create_user(email, &hash, name, "admin")?;
                        println!("Created admin user '{}' (id={})", email, id);
                    }
                }
            }
        }

        return Ok(());
    }

    // Handle Chat command early
    if let Commands::Chat {
        ref question,
        ref repo,
        ref model,
        download_model,
    } = cli.command
    {
        let db_tracker = Arc::new(SqliteTracker::new(&config.db_path)?);

        // Apply model override if provided
        let chat_config = config.chat.clone();
        let mut llm_config = config.llm.clone();
        if let Some(model_path) = model {
            llm_config.model_path = model_path.clone();
        }

        // Download model if requested and not present
        if download_model {
            let target = claudear::chat::service::expand_tilde(&llm_config.model_path);
            if !target.exists() {
                use std::sync::Arc as StdArc;
                let progress = StdArc::new(claudear::chat::models::DownloadProgress::new());
                let progress_clone = progress.clone();
                let url = llm_config.model_url.clone();
                let target_clone = target.clone();

                println!("Downloading model to {}...", target.display());
                let pb = indicatif::ProgressBar::new(0);
                pb.set_style(
                    indicatif::ProgressStyle::default_bar()
                        .template(
                            "{spinner:.green} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})",
                        )
                        .unwrap()
                        .progress_chars("#>-"),
                );

                let download_handle = tokio::spawn(async move {
                    claudear::chat::models::download::download_gguf(
                        &url,
                        &target_clone,
                        progress_clone,
                    )
                    .await
                });

                // Poll progress
                loop {
                    let downloaded = progress
                        .downloaded_bytes
                        .load(std::sync::atomic::Ordering::Relaxed);
                    let total = progress
                        .total_bytes
                        .load(std::sync::atomic::Ordering::Relaxed);
                    if total > 0 {
                        pb.set_length(total);
                    }
                    pb.set_position(downloaded);

                    if progress
                        .completed
                        .load(std::sync::atomic::Ordering::Relaxed)
                    {
                        pb.finish_with_message("Download complete");
                        break;
                    }
                    if progress.failed.load(std::sync::atomic::Ordering::Relaxed) {
                        pb.abandon_with_message("Download failed");
                        let err = progress
                            .error_message
                            .lock()
                            .ok()
                            .and_then(|g| g.clone())
                            .unwrap_or_else(|| "Unknown error".to_string());
                        eprintln!("Error: {err}");
                        std::process::exit(1);
                    }

                    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                }

                download_handle.await?.map_err(|e| anyhow::anyhow!(e))?;
                println!();
            } else {
                println!("Model already exists at {}", target.display());
            }
        }

        // Build embedding client for code search
        let (_, embedding_client) =
            claudear::watcher::Watcher::build_inferrer_with_embeddings(&config, None, None).await?;

        let embedding_client = embedding_client.ok_or_else(|| {
            anyhow::anyhow!("Embedding client required for chat. Ensure code_index is enabled.")
        })?;

        let code_search = claudear::repo::code_index::CodeSearchService::new(
            db_tracker.clone(),
            embedding_client,
        );

        let chat_service = claudear::chat::ChatService::new(
            chat_config,
            llm_config,
            code_search,
            db_tracker.clone(),
        );

        // Resolve repo_id if repo name given
        let repo_id = if let Some(repo_name) = repo {
            db_tracker.get_indexed_repo(repo_name)?.map(|r| r.id)
        } else {
            None
        };

        if let Some(q) = question {
            // One-shot mode
            let session_id = uuid::Uuid::new_v4().to_string();
            match chat_service.chat(&session_id, q, repo_id, None).await {
                Ok((tokens, sources)) => {
                    // Print streaming tokens
                    for token in &tokens {
                        print!("{}", token);
                    }
                    println!();

                    // Print sources
                    if !sources.is_empty() {
                        println!();
                        println!("Sources:");
                        for src in &sources {
                            let symbol = src
                                .symbol_name
                                .as_deref()
                                .map(|s| format!(" ({s})"))
                                .unwrap_or_default();
                            println!(
                                "  {}:{}-{}{} ({:.0}%)",
                                src.file_path,
                                src.start_line,
                                src.end_line,
                                symbol,
                                src.similarity * 100.0
                            );
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Error: {e}");
                    std::process::exit(1);
                }
            }
        } else {
            // Interactive REPL mode
            println!("Claudear Code Chat (type /quit to exit, /clear to reset session)");
            if let Some(ref r) = repo {
                println!("Repository: {r}");
            }
            println!();

            let mut session_id = uuid::Uuid::new_v4().to_string();
            let stdin = std::io::stdin();
            let mut show_sources = true;

            loop {
                use std::io::Write;
                print!("> ");
                std::io::stdout().flush()?;

                let mut input = String::new();
                if stdin.read_line(&mut input)? == 0 {
                    break; // EOF
                }

                let input = input.trim();
                if input.is_empty() {
                    continue;
                }

                match input {
                    "/quit" | "/exit" => break,
                    "/clear" => {
                        session_id = uuid::Uuid::new_v4().to_string();
                        println!("Session cleared.");
                        continue;
                    }
                    "/sources" => {
                        show_sources = !show_sources;
                        println!(
                            "Source display: {}",
                            if show_sources { "on" } else { "off" }
                        );
                        continue;
                    }
                    _ => {}
                }

                match chat_service.chat(&session_id, input, repo_id, None).await {
                    Ok((tokens, sources)) => {
                        println!();
                        for token in &tokens {
                            print!("{}", token);
                        }
                        println!();

                        if show_sources && !sources.is_empty() {
                            println!();
                            for src in &sources {
                                println!(
                                    "  [{:.0}%] {}:{}-{}",
                                    src.similarity * 100.0,
                                    src.file_path,
                                    src.start_line,
                                    src.end_line,
                                );
                            }
                        }
                        println!();
                    }
                    Err(e) => {
                        eprintln!("Error: {e}");
                    }
                }
            }
        }

        return Ok(());
    }

    // Handle Dashboard command early (doesn't need source validation)
    if let Commands::Dashboard {
        port,
        dashboard_dir,
    } = cli.command
    {
        let tracker = create_tracker(&config);
        let server = if let Some(dir) = dashboard_dir {
            ApiServer::with_dashboard(
                config,
                tracker,
                port,
                dir,
                std::path::PathBuf::from(config_path.clone()),
            )
        } else {
            ApiServer::with_port(
                config,
                tracker,
                port,
                std::path::PathBuf::from(config_path.clone()),
            )
        };

        let shutdown = async {
            tokio::signal::ctrl_c()
                .await
                .expect("Failed to install signal handler");
            tracing::info!("\nReceived shutdown signal...");
        };

        tokio::select! {
            result = server.start() => result?,
            _ = shutdown => {}
        }

        return Ok(());
    }

    config.validate()?;

    // Initialize components
    tracing::info!("Initializing...");
    let user_registry = UserRegistry::new(config.users.clone());
    let notifier = create_notifier(&config, user_registry.clone());
    let tracker = create_tracker(&config);

    // Handle Start command (daemon mode with IPC - runs all services concurrently)
    if let Commands::Start {
        port,
        poll,
        poll_interval,
        no_webhooks,
        no_dashboard,
        foreground: _,
    } = &cli.command
    {
        if is_daemon_running() {
            anyhow::bail!("A daemon is already running. Stop it first with 'claudear stop'");
        }

        // Determine what services to run
        let enable_webhooks = !no_webhooks;
        let enable_dashboard = !no_dashboard;
        let enable_polling = *poll;

        if !enable_webhooks && !enable_dashboard && !enable_polling {
            anyhow::bail!("No services enabled. Remove --no-webhooks/--no-dashboard or add --poll");
        }

        // Build mode string for status
        let mut modes = Vec::new();
        if enable_dashboard {
            modes.push("dashboard");
        }
        if enable_webhooks {
            modes.push("webhooks");
        }
        if enable_polling {
            modes.push("polling");
        }
        let mode_str = modes.join("+");

        // Build shared watcher dependencies
        let deps = build_watcher_deps(&config, &tracker).await?;
        let vector_store_embeddings = deps.inferrer.as_ref().map(|i| i.embedding_count());
        let sources = deps.sources;
        if sources.is_empty() {
            anyhow::bail!("No sources were initialized");
        }

        let github_webhook_handler =
            create_github_webhook_handler(&config, deps.review_watcher.clone());

        // Always create watcher (used for both polling and housekeeping-only)
        let watcher = Arc::new(Watcher::new(WatcherOptions {
            config: config.clone(),
            sources: sources.clone(),
            notifier: notifier.clone(),
            tracker: tracker.clone(),
            inferrer: deps.inferrer.clone(),
            embedding_client: deps.embedding_client.clone(),
            review_watcher: deps.review_watcher.clone(),
            issue_embedding_service: deps.issue_embedding_service.clone(),
            code_search_service: deps.code_search_service.clone(),
            relationships: deps.relationships,
            github_client: deps.github_client,
            scm_provider: deps.scm_provider,
            user_registry: user_registry.clone(),
            agent: deps.agent.clone(),
            classification_agent: deps.classification_agent.clone(),
            dry_run: false,
            llm_engine: deps.llm_engine.clone(),
        }));

        // Create IPC server
        let ipc_server = Arc::new(
            IpcServer::builder(tracker.clone(), sources.clone(), notifier.clone())
                .max_retries(config.retry.max_retries)
                .build()
                .with_watcher(watcher.clone()),
        );

        ipc_server.set_mode(&mode_str).await;
        if enable_polling {
            ipc_server.set_poll_interval(*poll_interval);
        }

        // Log watcher_started activity
        let activity = ActivityLogEntry::new(
            "watcher_started",
            format!("Watcher daemon started in {} mode", mode_str),
        )
        .with_source("system".to_string())
        .with_metadata(json!({
            "mode": mode_str,
            "port": port,
            "sources": sources.iter().map(|s| s.name()).collect::<Vec<_>>()
        }));
        tracker.record_activity(&activity).ok();

        // Log startup info
        tracing::info!("Starting watcher daemon...");
        tracing::info!("  Mode: {}", mode_str);
        tracing::info!("  Port: {}", port);
        tracing::info!("  Socket: {:?}", default_socket_path());
        tracing::info!(
            "  Sources: {}",
            sources
                .iter()
                .map(|s| s.name())
                .collect::<Vec<_>>()
                .join(", ")
        );
        if enable_polling {
            tracing::info!("  Poll interval: {}ms", poll_interval);
        }

        // Start regression monitoring background task
        let regression_handle = start_regression_monitoring(
            &config,
            tracker.clone(),
            sources.clone(),
            notifier.clone(),
        );
        if regression_handle.is_some() {
            tracing::info!("  Regression monitoring: enabled");
        }

        print_startup_banner_and_status(
            &config,
            &config_path,
            *port,
            enable_dashboard,
            enable_webhooks,
            enable_polling,
            *poll_interval,
            vector_store_embeddings,
            &user_registry,
        );

        // Shutdown signal handler with graceful drain
        let watcher_for_shutdown = watcher.clone();
        let tracker_for_shutdown = tracker.clone();
        let mut shutdown_rx = ipc_server.shutdown_receiver();
        let shutdown = async move {
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {
                    tracing::info!("\nReceived shutdown signal, press Ctrl+C again to force quit...");
                }
                _ = shutdown_rx.recv() => {
                    tracing::info!("\nReceived IPC shutdown command, initiating graceful shutdown...");
                }
            }

            // Race graceful drain against a second Ctrl+C for force quit
            tokio::select! {
                _ = watcher_for_shutdown.stop_and_drain() => {}
                _ = tokio::signal::ctrl_c() => {
                    tracing::warn!("\nForce shutdown requested, exiting immediately");
                    std::process::exit(130);
                }
            }

            // Log watcher_stopped activity
            let activity = ActivityLogEntry::new("watcher_stopped", "Watcher daemon stopped")
                .with_source("system".to_string());
            tracker_for_shutdown.record_activity(&activity).ok();
        };

        // Abort regression monitoring on shutdown
        let regression_shutdown = {
            let handle = regression_handle;
            async move {
                if let Some(h) = handle {
                    h.abort();
                }
            }
        };

        // Build the unified HTTP server with dashboard API + webhooks
        let mut config = config.clone();
        config.webhook_port = *port;

        // Start all services concurrently
        let ipc_future = ipc_server.start();

        let inferrer_clone = deps.inferrer.clone();
        let embedding_client_clone = deps.embedding_client.clone();
        let github_webhook_handler_for_http = github_webhook_handler;
        let review_watcher_clone = deps.review_watcher.clone();
        let issue_embedding_service_clone = deps.issue_embedding_service.clone();
        let code_search_service_clone = deps.code_search_service.clone();
        let agent_clone = deps.agent.clone();
        let http_future = async move {
            if enable_webhooks {
                let handlers = create_webhook_handlers(&config);
                if handlers.get_all().is_empty()
                    && github_webhook_handler_for_http.is_none()
                    && !enable_dashboard
                {
                    return Err(anyhow::anyhow!("No webhook handlers configured"));
                }

                let mut server = WebhookServer::new_with_github(
                    config.clone(),
                    handlers,
                    notifier.clone(),
                    tracker.clone(),
                    Some(tracker.clone()),
                    inferrer_clone,
                    github_webhook_handler_for_http,
                    agent_clone,
                );
                server.set_embedding_client(embedding_client_clone);
                server.set_issue_embedding_service(issue_embedding_service_clone);
                server.set_code_search_service(code_search_service_clone);
                server.set_review_watcher(review_watcher_clone);
                if enable_dashboard {
                    server.set_dashboard(std::path::PathBuf::from(config_path.clone()));
                }
                server.start().await?;
            } else if enable_dashboard {
                // Dashboard only (no webhooks)
                let server = ApiServer::with_port(
                    config.clone(),
                    tracker.clone(),
                    *port,
                    std::path::PathBuf::from(config_path.clone()),
                );
                server.start().await?;
            }
            Ok::<(), anyhow::Error>(())
        };

        let watcher_for_poll = watcher.clone();
        let poll_future = async move {
            if enable_polling {
                watcher_for_poll.start(Some(*poll_interval)).await?;
            } else {
                // Run housekeeping without source polling
                let worker = HousekeepingWorker::new(watcher_for_poll, *poll_interval);
                worker.start().await?;
            }
            Ok::<(), anyhow::Error>(())
        };

        // Run everything concurrently
        tokio::select! {
            result = ipc_future => {
                if let Err(e) = result {
                    tracing::error!("IPC server error: {}", e);
                }
            }
            result = http_future => {
                if let Err(e) = result {
                    tracing::error!("HTTP server error: {}", e);
                }
            }
            result = poll_future => {
                if let Err(e) = result {
                    tracing::error!("Polling error: {}", e);
                }
            }
            _ = shutdown => {
                // Stop regression monitoring
                regression_shutdown.await;
            }
        }

        return Ok(());
    }

    // Handle PR commands early since they don't need sources
    if let Commands::Prs(PrsCommands::Monitor { continuous }) = cli.command {
        if !config.is_github_enabled() {
            anyhow::bail!("GitHub token not configured. Set GITHUB_TOKEN environment variable.");
        }

        let github_client = GitHubClient::new(config.github().clone());
        let provider: Arc<dyn ScmProvider> = Arc::new(github_client);
        let sources = create_sources(&config);
        let pr_monitor = if config.regression.enabled {
            PrMonitor::with_regression_tracking(
                provider,
                tracker.clone(),
                config.github().auto_resolve_on_merge,
                tracker.clone(),
            )
        } else {
            PrMonitor::new(
                provider,
                tracker.clone(),
                config.github().auto_resolve_on_merge,
            )
        };

        if continuous {
            tracing::info!("Starting PR monitor (continuous mode)...");
            let configured_poll_interval_ms = config.github().poll_interval_ms;
            let poll_interval_ms = configured_poll_interval_ms.max(1);
            if configured_poll_interval_ms == 0 {
                tracing::warn!(
                    component = "pr_monitor",
                    "poll_interval_ms evaluated to 0, clamping to 1ms to avoid timer panic"
                );
            }
            tracing::info!("  Poll interval: {}ms", poll_interval_ms);
            tracing::info!(
                "  Auto-resolve on merge: {}",
                config.github().auto_resolve_on_merge
            );

            let mut poll_timer = interval(Duration::from_millis(poll_interval_ms));

            let shutdown = async {
                tokio::signal::ctrl_c()
                    .await
                    .expect("Failed to install signal handler");
                tracing::info!("\nReceived shutdown signal...");
            };

            tokio::select! {
                _ = async {
                    loop {
                        poll_timer.tick().await;
                        match pr_monitor.check_pending_prs().await {
                            Ok(updates) => {
                                for update in updates {
                                    match update.new_status {
                                        PrStatus::Merged => {
                                            tracing::info!(component = "pr_monitor", short_id = %update.short_id, pr_url = %update.pr_url, "PR merged");

                                            // Auto-resolve on source if enabled
                                            if update.should_resolve {
                                                if let Some(source) = sources.iter().find(|s| s.name() == update.source) {
                                                    if let Err(e) = source.resolve_issue(&update.issue_id).await {
                                                        tracing::warn!(component = "pr_monitor", source = %update.source, error = %e, "Failed to resolve issue");
                                                    } else {
                                                        tracker.mark_resolved(&update.source, &update.issue_id).ok();
                                                        notifier.notify_merged(&claudear::types::Issue::new(
                                                            &update.issue_id,
                                                            &update.short_id,
                                                            "Issue resolved",
                                                            &update.pr_url,
                                                            &update.source,
                                                        ), &update.pr_url).await.ok();
                                                    }
                                                }
                                            }
                                        }
                                        PrStatus::Closed => {
                                            tracing::info!(component = "pr_monitor", short_id = %update.short_id, pr_url = %update.pr_url, "PR closed without merge");
                                        }
                                        PrStatus::Open => {}
                                    }
                                }
                            }
                            Err(e) => {
                                tracing::error!(component = "pr_monitor", error = %e, "Error checking PRs");
                            }
                        }
                    }
                } => {}
                _ = shutdown => {}
            }
        } else {
            // One-shot check
            tracing::info!("Checking pending PRs...");
            let updates = pr_monitor.check_pending_prs().await?;

            if updates.is_empty() {
                println!("\nNo PR status changes detected.");
            } else {
                println!("\nPR Status Updates:");
                for update in updates {
                    let status = match update.new_status {
                        PrStatus::Merged => "MERGED",
                        PrStatus::Closed => "CLOSED",
                        PrStatus::Open => "OPEN",
                    };
                    println!("  [{}] {} - {}", status, update.short_id, update.pr_url);

                    // Auto-resolve on source if merged
                    if update.should_resolve {
                        if let Some(source) = sources.iter().find(|s| s.name() == update.source) {
                            if let Err(e) = source.resolve_issue(&update.issue_id).await {
                                tracing::warn!(
                                    "Failed to resolve issue on {}: {}",
                                    update.source,
                                    e
                                );
                            } else {
                                tracker.mark_resolved(&update.source, &update.issue_id).ok();
                                println!("    -> Issue resolved on {}", update.source);
                            }
                        }
                    }
                }
            }
        }
        return Ok(());
    }

    if let Commands::Prs(PrsCommands::List) = cli.command {
        let pending_prs = tracker.get_pending_prs()?;
        let merged = tracker.get_attempts_by_status(FixAttemptStatus::Merged)?;
        let closed = tracker.get_attempts_by_status(FixAttemptStatus::Closed)?;

        println!("\n=== Pending PRs (awaiting merge) ===");
        if pending_prs.is_empty() {
            println!("  No pending PRs");
        } else {
            for attempt in &pending_prs {
                println!(
                    "  [{}] {} - {}",
                    attempt.source,
                    attempt.short_id,
                    attempt.pr_url.as_deref().unwrap_or("N/A")
                );
            }
        }

        println!("\n=== Merged PRs ===");
        if merged.is_empty() {
            println!("  No merged PRs");
        } else {
            for attempt in merged.iter().take(10) {
                let resolved = if attempt.resolved_at.is_some() {
                    " (resolved)"
                } else {
                    ""
                };
                println!(
                    "  [{}] {}{} - {}",
                    attempt.source,
                    attempt.short_id,
                    resolved,
                    attempt.pr_url.as_deref().unwrap_or("N/A")
                );
            }
            if merged.len() > 10 {
                println!("  ... and {} more", merged.len() - 10);
            }
        }

        println!("\n=== Closed PRs (not merged) ===");
        if closed.is_empty() {
            println!("  No closed PRs");
        } else {
            for attempt in closed.iter().take(10) {
                println!(
                    "  [{}] {} - {}",
                    attempt.source,
                    attempt.short_id,
                    attempt.pr_url.as_deref().unwrap_or("N/A")
                );
            }
            if closed.len() > 10 {
                println!("  ... and {} more", closed.len() - 10);
            }
        }

        return Ok(());
    }

    if let Commands::Retries(RetriesCommands::List) = cli.command {
        let retry_manager = RetryManager::new(config.retry.clone(), tracker.clone());
        let retryable = tracker.get_retryable_issues(config.retry.max_retries)?;
        let ready = retry_manager.get_ready_retries()?;

        println!(
            "\n=== Retryable Issues (max_retries: {}) ===",
            config.retry.max_retries
        );
        if retryable.is_empty() {
            println!("  No issues eligible for retry");
        } else {
            for attempt in &retryable {
                let next_retry = retry_manager.get_next_retry_time(attempt);
                let ready_str = if ready.iter().any(|r| r.id == attempt.id) {
                    " [READY]"
                } else {
                    ""
                };
                let next_time = next_retry
                    .map(|t| t.format("%Y-%m-%d %H:%M:%S").to_string())
                    .unwrap_or_else(|| "N/A".to_string());

                println!(
                    "  [{}] {} - retry {}/{} - next: {}{}",
                    attempt.source,
                    attempt.short_id,
                    attempt.retry_count,
                    config.retry.max_retries,
                    next_time,
                    ready_str
                );
                if let Some(ref error) = attempt.error_message {
                    let truncated = if error.len() > 60 {
                        format!("{}...", &error[..error.floor_char_boundary(57)])
                    } else {
                        error.clone()
                    };
                    println!("       Error: {}", truncated);
                }
            }
        }

        println!("\n=== Ready for Retry Now ===");
        if ready.is_empty() {
            println!("  No issues ready for retry");
        } else {
            println!("  {} issues ready", ready.len());
            for attempt in &ready {
                println!("  - [{}] {}", attempt.source, attempt.short_id);
            }
        }

        return Ok(());
    }

    if let Commands::Retries(RetriesCommands::Process) = cli.command {
        let retry_manager = RetryManager::new(config.retry.clone(), tracker.clone());
        let ready = retry_manager.get_ready_retries()?;

        if ready.is_empty() {
            println!("\nNo issues ready for retry.");
            return Ok(());
        }

        println!("\nProcessing {} retries...", ready.len());

        // Need sources and watcher for processing
        let sources = create_sources(&config);
        if sources.is_empty() {
            anyhow::bail!("No sources were initialized");
        }

        // Create GitHub client for API-based repo discovery
        let github_client = GitHubClient::new(config.github().clone());

        // Build inferrer for retry processing (with embeddings for semantic matching)
        let (inferrer, embedding_client) = Watcher::build_inferrer_with_embeddings(
            &config,
            Some(&github_client),
            Some(tracker.as_ref()),
        )
        .await?;

        // Create ReviewWatcher for PR review tracking
        let review_watcher = create_review_watcher(&config, tracker.clone());

        let issue_embedding_service =
            build_issue_embedding_service(&tracker, embedding_client.as_ref());

        let code_search_service = if config.code_index.enabled {
            embedding_client.as_ref().map(|emb| {
                Arc::new(claudear::repo::code_index::CodeSearchService::new(
                    tracker.clone(),
                    emb.clone(),
                ))
            })
        } else {
            None
        };

        let agent: Arc<dyn AgentRunner> =
            InstrumentedRunner::wrap(Arc::new(ClaudeAgentRunner::new(
                ClaudeRunnerConfig {
                    timeout_secs: config.agent.timeout_secs,
                    model: config
                        .agent
                        .default_provider_config()
                        .and_then(|p| p.model.clone()),
                    instructions: config
                        .agent
                        .default_provider_config()
                        .and_then(|p| p.instructions.clone()),
                    permissions: config
                        .agent
                        .default_provider_config()
                        .map(|p| p.permissions.clone())
                        .unwrap_or_default(),
                    skip_permissions: config
                        .agent
                        .default_provider_config()
                        .map(|p| p.skip_permissions)
                        .unwrap_or(false),
                    binary: config
                        .agent
                        .default_provider_config()
                        .and_then(|p| p.binary.clone())
                        .unwrap_or_else(|| "claude".to_string()),
                    env: config
                        .agent
                        .default_provider_config()
                        .map(|p| p.env.clone())
                        .unwrap_or_default(),
                },
                tracker.clone(),
            )));

        let watcher = Watcher::new(WatcherOptions {
            config: config.clone(),
            sources,
            notifier,
            tracker: tracker.clone(),
            inferrer,
            embedding_client,
            review_watcher,
            issue_embedding_service,
            code_search_service,
            relationships: None,
            github_client: None,
            scm_provider: None,
            user_registry: user_registry.clone(),
            agent,
            classification_agent: None,
            dry_run: false,
            llm_engine: None,
        });

        for attempt in ready {
            println!("\n  Retrying [{}] {}...", attempt.source, attempt.short_id);

            // Prepare for retry
            retry_manager.prepare_retry(&attempt.source, &attempt.issue_id)?;

            // Trigger the fix
            if let Err(e) = watcher
                .trigger_issue(&attempt.source, &attempt.issue_id)
                .await
            {
                tracing::error!("Failed to retry {}: {}", attempt.short_id, e);
            }
        }

        println!("\nRetry processing complete.");
        return Ok(());
    }

    if let Commands::Report(ReportCommands::Preview { frequency }) = cli.command {
        let freq = ReportFrequency::parse(&frequency).ok_or_else(|| {
            anyhow::anyhow!(
                "Invalid frequency: {}. Use daily, weekly, or monthly",
                frequency
            )
        })?;

        let generator = ReportGenerator::new(tracker);
        let report = match freq {
            ReportFrequency::Daily => generator.generate_daily()?,
            ReportFrequency::Weekly(_) => generator.generate_weekly()?,
            ReportFrequency::Monthly => generator.generate_monthly()?,
        };

        println!("{}", report.format_text());
        return Ok(());
    }

    if let Commands::Report(ReportCommands::Send { frequency }) = cli.command {
        let freq = ReportFrequency::parse(&frequency).ok_or_else(|| {
            anyhow::anyhow!(
                "Invalid frequency: {}. Use daily, weekly, or monthly",
                frequency
            )
        })?;

        let scheduler = ReportScheduler::new(tracker, notifier);
        let report = scheduler.send_now(freq).await?;

        println!("Report sent successfully!");
        println!("{}", report.format_text());
        return Ok(());
    }

    if let Commands::Report(ReportCommands::Schedule {
        daily,
        weekly,
        hour,
    }) = cli.command
    {
        let mut scheduler = ReportScheduler::new(tracker, notifier);

        if daily {
            scheduler.add_schedule(ReportSchedule::daily("daily-report", hour));
            tracing::info!("Daily reports scheduled at {:02}:00 UTC", hour);
        }

        if weekly {
            scheduler.add_schedule(ReportSchedule::weekly(
                "weekly-report",
                chrono::Weekday::Mon,
                hour,
            ));
            tracing::info!("Weekly reports scheduled for Monday at {:02}:00 UTC", hour);
        }

        if scheduler.schedules().is_empty() {
            anyhow::bail!("No report schedules enabled. Use --daily or --weekly");
        }

        tracing::info!("Report scheduler started...");

        // Run scheduler every hour to check for due reports
        let mut check_timer = interval(Duration::from_secs(3600));

        let shutdown = async {
            tokio::signal::ctrl_c()
                .await
                .expect("Failed to install signal handler");
            tracing::info!("\nReceived shutdown signal...");
        };

        tokio::select! {
            _ = async {
                loop {
                    check_timer.tick().await;
                    match scheduler.check_and_send().await {
                        Ok(sent) => {
                            for name in sent {
                                tracing::info!("Sent report: {}", name);
                            }
                        }
                        Err(e) => {
                            tracing::error!("Error checking schedules: {}", e);
                        }
                    }
                }
            } => {}
            _ = shutdown => {}
        }

        return Ok(());
    }

    match cli.command {
        Commands::Webhook {
            port,
            setup,
            base_url,
            env_file,
            self_test,
        } => {
            let mut config = config;
            config.webhook_port = port;

            // Auto-configure webhooks if requested
            if setup {
                let base_url = base_url.ok_or_else(|| {
                    anyhow::anyhow!(
                        "--base-url is required with --setup. \
                        Example: --base-url https://my-server.example.com:3100"
                    )
                })?;

                let env_path = std::path::PathBuf::from(&env_file);
                let configurator = WebhookConfigurator::new(config.clone(), &env_path);

                match configurator.configure(&base_url).await {
                    Ok(result) => {
                        print_setup_result(&result);

                        // Reload config to get the new secrets from env vars
                        // (webhook secrets are stored in env file, which overrides YAML config)
                        tracing::info!("Reloading configuration with new secrets...");
                        config = Config::load(&config_path)?;
                        config.webhook_port = port;
                    }
                    Err(e) => {
                        anyhow::bail!("Webhook auto-configuration failed: {}", e);
                    }
                }
            }

            // When self-testing, inject temporary webhook secrets into configured
            // sources that don't already have one. This lets the self-test exercise
            // the full webhook pipeline without requiring prior --setup registration.
            if self_test {
                claudear::webhook::self_test::inject_test_secrets(&mut config);
            }

            let handlers = create_webhook_handlers(&config);

            // Build shared watcher dependencies (needed for housekeeping)
            let deps = build_watcher_deps(&config, &tracker).await?;
            let vector_store_embeddings = deps.inferrer.as_ref().map(|i| i.embedding_count());
            let github_webhook_handler =
                create_github_webhook_handler(&config, deps.review_watcher.clone());

            if handlers.get_all().is_empty() && github_webhook_handler.is_none() {
                anyhow::bail!("No webhook handlers were registered");
            }

            print_startup_banner_and_status(
                &config,
                &config_path,
                port,
                false, // dashboard not available in standalone webhook mode
                true,  // webhooks
                false, // polling
                0,
                vector_store_embeddings,
                &user_registry,
            );

            // Create a Watcher for housekeeping (retries, cascades, auto-close, etc.)
            let watcher = Arc::new(Watcher::new(WatcherOptions {
                config: config.clone(),
                sources: deps.sources,
                notifier: notifier.clone(),
                tracker: tracker.clone(),
                inferrer: deps.inferrer.clone(),
                embedding_client: deps.embedding_client.clone(),
                review_watcher: deps.review_watcher.clone(),
                issue_embedding_service: deps.issue_embedding_service.clone(),
                code_search_service: deps.code_search_service.clone(),
                relationships: deps.relationships,
                github_client: deps.github_client,
                scm_provider: deps.scm_provider,
                user_registry: user_registry.clone(),
                agent: deps.agent.clone(),
                classification_agent: deps.classification_agent.clone(),
                dry_run: false,
                llm_engine: deps.llm_engine.clone(),
            }));

            let worker = HousekeepingWorker::new(watcher.clone(), config.poll_interval_ms);

            let mut server = WebhookServer::new_with_github(
                config.clone(),
                handlers,
                notifier.clone(),
                tracker.clone(),
                Some(tracker.clone()),
                deps.inferrer,
                github_webhook_handler,
                deps.agent,
            );
            server.set_embedding_client(deps.embedding_client.clone());
            server.set_issue_embedding_service(deps.issue_embedding_service);
            server.set_code_search_service(deps.code_search_service);
            server.set_review_watcher(deps.review_watcher);

            // Start regression monitoring background task
            let regression_handle = start_regression_monitoring(
                &config,
                tracker.clone(),
                create_sources(&config),
                notifier.clone(),
            );
            if regression_handle.is_some() {
                tracing::info!("Regression monitoring: enabled");
            }

            // Handle shutdown signals
            let watcher_for_shutdown = watcher.clone();
            let shutdown = async move {
                tokio::signal::ctrl_c()
                    .await
                    .expect("Failed to install signal handler");
                tracing::info!("\nReceived shutdown signal, press Ctrl+C again to force quit...");
                tokio::select! {
                    _ = watcher_for_shutdown.stop_and_drain() => {}
                    _ = tokio::signal::ctrl_c() => {
                        tracing::warn!("\nForce shutdown requested, exiting immediately");
                        std::process::exit(130);
                    }
                }
                if let Some(h) = regression_handle {
                    h.abort();
                }
            };

            if self_test {
                let port_for_test = port;
                let config_for_test = config.clone();
                tokio::spawn(async move {
                    tokio::time::sleep(Duration::from_secs(2)).await;
                    match claudear::webhook::self_test::run(port_for_test, &config_for_test).await {
                        Ok(()) => {
                            tracing::info!("Self-test passed!");
                            std::process::exit(0);
                        }
                        Err(e) => {
                            tracing::error!("Self-test failed: {}", e);
                            std::process::exit(1);
                        }
                    }
                });
            }

            tokio::select! {
                result = server.start() => result?,
                result = worker.start() => {
                    if let Err(e) = result {
                        tracing::error!("Housekeeping worker error: {}", e);
                    }
                }
                _ = shutdown => {}
            }
        }

        _ => {
            // Initialize sources for polling/seed/dry-run modes
            tracing::info!("Initializing sources...");
            let sources = create_sources(&config);

            if sources.is_empty() {
                anyhow::bail!("No sources were initialized");
            }

            let dry_run = matches!(cli.command, Commands::DryRun);
            let sources_for_regression = sources.clone();
            let notifier_for_regression = notifier.clone();

            // Create GitHub client for API-based repo discovery
            let github_client = GitHubClient::new(config.github().clone());

            // Build inferrer for repo inference (with embeddings for semantic matching)
            let (inferrer, embedding_client) = Watcher::build_inferrer_with_embeddings(
                &config,
                Some(&github_client),
                Some(tracker.as_ref()),
            )
            .await?;

            // Create ReviewWatcher for PR review tracking
            let review_watcher = create_review_watcher(&config, tracker.clone());

            let issue_embedding_service =
                build_issue_embedding_service(&tracker, embedding_client.as_ref());

            let code_search_service = if config.code_index.enabled {
                embedding_client.as_ref().map(|emb| {
                    Arc::new(claudear::repo::code_index::CodeSearchService::new(
                        tracker.clone(),
                        emb.clone(),
                    ))
                })
            } else {
                None
            };

            let agent: Arc<dyn AgentRunner> =
                InstrumentedRunner::wrap(Arc::new(ClaudeAgentRunner::new(
                    ClaudeRunnerConfig {
                        timeout_secs: config.agent.timeout_secs,
                        model: config
                            .agent
                            .default_provider_config()
                            .and_then(|p| p.model.clone()),
                        instructions: config
                            .agent
                            .default_provider_config()
                            .and_then(|p| p.instructions.clone()),
                        permissions: config
                            .agent
                            .default_provider_config()
                            .map(|p| p.permissions.clone())
                            .unwrap_or_default(),
                        skip_permissions: config
                            .agent
                            .default_provider_config()
                            .map(|p| p.skip_permissions)
                            .unwrap_or(false),
                        binary: config
                            .agent
                            .default_provider_config()
                            .and_then(|p| p.binary.clone())
                            .unwrap_or_else(|| "claude".to_string()),
                        env: config
                            .agent
                            .default_provider_config()
                            .map(|p| p.env.clone())
                            .unwrap_or_default(),
                    },
                    tracker.clone(),
                )));

            // Eagerly load LLM engine — download model if not present on disk
            let llm_engine = if config.llm.enabled {
                let model_path = claudear::chat::service::expand_tilde(&config.llm.model_path);
                let model_ready = if model_path.exists() && model_path.is_file() {
                    true
                } else if !config.llm.model_url.is_empty() {
                    tracing::info!(
                        url = %config.llm.model_url,
                        target = %model_path.display(),
                        "LLM model not found, downloading..."
                    );
                    let progress =
                        Arc::new(claudear::chat::models::download::DownloadProgress::new());
                    match claudear::chat::models::download::download_gguf(
                        &config.llm.model_url,
                        &model_path,
                        progress.clone(),
                    )
                    .await
                    {
                        Ok(()) => {
                            tracing::info!(
                                size_mb = progress
                                    .total_bytes
                                    .load(std::sync::atomic::Ordering::Relaxed)
                                    / 1_048_576,
                                "LLM model downloaded successfully"
                            );
                            true
                        }
                        Err(e) => {
                            tracing::warn!(error = %e, "Failed to download LLM model, classification disabled");
                            false
                        }
                    }
                } else {
                    tracing::debug!("LLM model not found and no download URL configured");
                    false
                };

                if model_ready {
                    let llm_config = claudear::chat::llm::LlmConfig {
                        model_path,
                        context_length: config.llm.context_length,
                        gpu_layers: config.llm.gpu_layers,
                        threads: config.llm.threads,
                        timeout: if config.llm.inference_timeout_secs > 0 {
                            Some(std::time::Duration::from_secs(
                                config.llm.inference_timeout_secs,
                            ))
                        } else {
                            None
                        },
                    };
                    match claudear::chat::llm::LlmEngine::load(&llm_config) {
                        Ok(engine) => {
                            tracing::info!("LLM engine loaded for classification + chat");
                            Some(Arc::new(engine))
                        }
                        Err(e) => {
                            tracing::warn!(error = %e, "Failed to load LLM engine");
                            None
                        }
                    }
                } else {
                    None
                }
            } else {
                None
            };

            // Build a separate agent runner for classification if configured
            let classification_agent: Option<Arc<dyn AgentRunner>> = config
                .agent
                .default_provider_config()
                .and_then(|p| p.classification_model.clone())
                .map(|model| {
                    tracing::info!(model = %model, "Building separate agent runner for classification");
                    let runner = ClaudeAgentRunner::new(
                        ClaudeRunnerConfig {
                            timeout_secs: config.agent.timeout_secs,
                            model: Some(model),
                            instructions: config
                                .agent
                                .default_provider_config()
                                .and_then(|p| p.instructions.clone()),
                            permissions: config
                                .agent
                                .default_provider_config()
                                .map(|p| p.permissions.clone())
                                .unwrap_or_default(),
                            skip_permissions: config
                                .agent
                                .default_provider_config()
                                .map(|p| p.skip_permissions)
                                .unwrap_or(false),
                            binary: config
                                .agent
                                .default_provider_config()
                                .and_then(|p| p.binary.clone())
                                .unwrap_or_else(|| "claude".to_string()),
                            env: config
                                .agent
                                .default_provider_config()
                                .map(|p| p.env.clone())
                                .unwrap_or_default(),
                        },
                        tracker.clone(),
                    );
                    InstrumentedRunner::wrap(Arc::new(runner)) as Arc<dyn AgentRunner>
                });

            let tracker_for_api = tracker.clone();
            let vector_store_embeddings = inferrer.as_ref().map(|i| i.embedding_count());
            let watcher = Arc::new(Watcher::new(WatcherOptions {
                config: config.clone(),
                sources,
                notifier,
                tracker,
                inferrer,
                embedding_client,
                review_watcher,
                issue_embedding_service,
                code_search_service,
                relationships: None,
                github_client: if config.is_github_enabled() {
                    Some(Arc::new(GitHubClient::new(config.github().clone())))
                } else {
                    None
                },
                scm_provider: None,
                user_registry: user_registry.clone(),
                agent,
                classification_agent,
                dry_run,
                llm_engine,
            }));

            // Handle shutdown signals
            let watcher_ref = &watcher;

            match cli.command {
                Commands::Seed => {
                    watcher.seed().await?;
                }

                Commands::DryRun => {
                    let shutdown = async {
                        tokio::signal::ctrl_c()
                            .await
                            .expect("Failed to install signal handler");
                        tracing::info!("\nReceived shutdown signal...");
                        watcher_ref.stop();
                    };

                    tokio::select! {
                        result = watcher.start(None) => result?,
                        _ = shutdown => {}
                    }
                }

                Commands::Poll {
                    interval,
                    port,
                    no_dashboard,
                } => {
                    print_startup_banner_and_status(
                        &config,
                        &config_path,
                        port,
                        !no_dashboard, // dashboard
                        false,         // webhooks
                        true,          // polling
                        interval,
                        vector_store_embeddings,
                        &user_registry,
                    );

                    // Keep regression monitoring active in foreground poll mode so merged bug
                    // fixes complete the regression-final-check -> resolve flow.
                    let _regression_handle = start_regression_monitoring(
                        &config,
                        tracker_for_api.clone(),
                        sources_for_regression.clone(),
                        notifier_for_regression.clone(),
                    );

                    let shutdown = async {
                        tokio::signal::ctrl_c()
                            .await
                            .expect("Failed to install signal handler");
                        tracing::info!("\nReceived shutdown signal...");
                        watcher_ref.stop();
                    };

                    if no_dashboard {
                        tokio::select! {
                            result = watcher.start(Some(interval)) => result?,
                            _ = shutdown => {}
                        }
                    } else {
                        let api_server = ApiServer::with_port(
                            config.clone(),
                            tracker_for_api.clone(),
                            port,
                            std::path::PathBuf::from(config_path.clone()),
                        );
                        tokio::select! {
                            result = watcher.start(Some(interval)) => result?,
                            result = api_server.start() => result?,
                            _ = shutdown => {}
                        }
                    }
                }

                Commands::Trigger { source, issue_id } => {
                    watcher.trigger_issue(&source, &issue_id).await?;
                }

                Commands::Action {
                    action,
                    source,
                    issue_id,
                } => {
                    let kind: ActionKind =
                        action.parse().map_err(|e: String| anyhow::anyhow!(e))?;
                    let outcome = watcher.run_action(kind, &source, &issue_id).await?;
                    use claudear::processing::ProcessingOutcome;
                    match outcome {
                        ProcessingOutcome::Success { pr_url } => {
                            println!("✓ {action}: PR {pr_url}");
                        }
                        ProcessingOutcome::CompletedNoPr { reason } => {
                            println!("✓ {action}: {reason}");
                        }
                        ProcessingOutcome::Failed { error } => {
                            println!("✗ {action} failed: {error}");
                        }
                        ProcessingOutcome::WrongRepo { suggested_repo, .. } => {
                            println!("✗ {action}: wrong repo (suggested: {suggested_repo:?})");
                        }
                    }
                }

                Commands::Reset { source, issue_id } => {
                    watcher.reset_attempt(&source, &issue_id)?;
                }

                Commands::Stats => {
                    let stats = watcher.get_stats()?;
                    println!("\nFix Attempt Statistics:");
                    println!("  Total:      {}", stats.total);
                    println!("  Pending:    {}", stats.pending);
                    println!("  Success:    {} (PRs created)", stats.success);
                    println!(
                        "  Merged:     {} (PRs merged, issues resolved)",
                        stats.merged
                    );
                    println!("  Closed:     {} (PRs closed without merge)", stats.closed);
                    println!("  Failed:     {}", stats.failed);
                    println!("  Cannot Fix: {} (max retries reached)", stats.cannot_fix);

                    // Calculate success rate
                    let completed = stats.merged + stats.closed + stats.failed + stats.cannot_fix;
                    if completed > 0 {
                        let merge_rate = (stats.merged as f64 / completed as f64) * 100.0;
                        println!("\n  Merge Rate: {:.1}%", merge_rate);
                    }

                    if !stats.by_source.is_empty() {
                        println!("\nBy Source:");
                        for (source, source_stats) in &stats.by_source {
                            println!("  {}:", source);
                            println!(
                                "    Total: {}, Success: {}, Merged: {}, Closed: {}, Failed: {}, Cannot Fix: {}",
                                source_stats.total, source_stats.success, source_stats.merged,
                                source_stats.closed, source_stats.failed, source_stats.cannot_fix
                            );
                        }
                    }
                }

                Commands::Start { .. }
                | Commands::Stop
                | Commands::Status
                | Commands::Pause
                | Commands::Resume
                | Commands::Activity { .. }
                | Commands::Sources
                | Commands::Webhook { .. }
                | Commands::Prs(_)
                | Commands::Retries(_)
                | Commands::Dashboard { .. }
                | Commands::Report(_)
                | Commands::Repos(_)
                | Commands::Inference(_)
                | Commands::Diag(_)
                | Commands::Users(_)
                | Commands::Chat { .. }
                | Commands::Purge { .. } => unreachable!(),
            }
        }
    }

    Ok(())
}
