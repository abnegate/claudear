//! API route handlers for the dashboard.

use super::auth::*;
use crate::retry::RetryManager;
use axum::{
    extract::{
        ws::{Message, WebSocket},
        Path, Query, State, WebSocketUpgrade,
    },
    http::StatusCode,
    response::IntoResponse,
    routing::get,
    Json, Router,
};
use chrono::{DateTime, Duration, TimeZone, Utc};
use claudear_config::config::Config;
use claudear_core::types::{FixAttempt, FixAttemptStats, FixAttemptStatus, RegressionWatchStatus};
use claudear_storage::FixAttemptTracker;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tower_http::services::{ServeDir, ServeFile};

/// Shared state for API handlers.
#[derive(Clone)]
pub struct ApiState {
    pub config: Config,
    pub tracker: Arc<dyn FixAttemptTracker>,
    /// Instant when the server started, for uptime calculation.
    pub start_time: Instant,
    /// Path to the config file on disk (for read/write from dashboard).
    pub config_path: PathBuf,
    /// Watch channel for real-time indexing progress pushed from SQLite hooks.
    pub indexing_rx: tokio::sync::watch::Receiver<claudear_storage::IndexingProgress>,
    /// General-purpose storage directory for user uploads (avatars, etc.).
    pub storage_dir: PathBuf,
    /// Chat service for model browsing/downloading (None if chat is disabled).
    pub chat_service: Option<Arc<claudear_integrations::chat::ChatService>>,
}

/// Create the API router.
pub fn create_api_router(
    config: Config,
    tracker: Arc<dyn FixAttemptTracker>,
    config_path: PathBuf,
    indexing_rx: tokio::sync::watch::Receiver<claudear_storage::IndexingProgress>,
) -> Router {
    create_api_router_with_dashboard(config, tracker, config_path, indexing_rx, None)
}

/// Create the API router with optional dashboard static file serving.
pub fn create_api_router_with_dashboard(
    config: Config,
    tracker: Arc<dyn FixAttemptTracker>,
    config_path: PathBuf,
    indexing_rx: tokio::sync::watch::Receiver<claudear_storage::IndexingProgress>,
    dashboard_dir: Option<PathBuf>,
) -> Router {
    create_api_router_full(
        config,
        tracker,
        config_path,
        indexing_rx,
        dashboard_dir,
        None,
    )
}

/// Create the API router with all optional features.
pub fn create_api_router_full(
    config: Config,
    tracker: Arc<dyn FixAttemptTracker>,
    config_path: PathBuf,
    indexing_rx: tokio::sync::watch::Receiver<claudear_storage::IndexingProgress>,
    dashboard_dir: Option<PathBuf>,
    chat_state: Option<claudear_integrations::chat::ChatState>,
) -> Router {
    let storage_dir = config.storage_dir.clone();

    // Ensure avatar upload directory exists
    let avatars_dir = storage_dir.join("avatars");
    if let Err(e) = std::fs::create_dir_all(&avatars_dir) {
        tracing::warn!(error = %e, path = %avatars_dir.display(), "Failed to create avatars directory");
    }

    let chat_service = chat_state.as_ref().map(|cs| cs.chat_service.clone());

    let state = ApiState {
        config,
        tracker,
        start_time: Instant::now(),
        config_path,
        indexing_rx,
        storage_dir: storage_dir.clone(),
        chat_service,
    };

    let api_routes = Router::new()
        .route("/api/health", get(health_handler))
        .route("/api/stats", get(stats_handler))
        .route("/api/stats/overview", get(overview_handler))
        .route("/api/attempts", get(attempts_handler))
        .route("/api/attempts/{id}", get(attempt_detail_handler))
        .route(
            "/api/attempts/{id}/detail",
            get(attempt_full_detail_handler),
        )
        .route(
            "/api/attempts/{id}/logs/{execution_id}/{stream}",
            get(attempt_execution_log_handler),
        )
        .route("/api/sources", get(sources_handler))
        .route("/api/retries", get(retries_handler))
        .route("/api/activity", get(activity_handler))
        .route("/api/analytics/summary", get(analytics_summary_handler))
        .route("/api/metrics", get(metrics_handler))
        .route("/api/errors", get(errors_handler))
        .route("/api/issues", get(issues_handler))
        .route("/api/prs", get(prs_handler))
        .route("/api/prs/analytics", get(pr_analytics_handler))
        .route("/api/feedback", get(feedback_handler))
        .route("/api/regressions", get(regressions_handler))
        .route(
            "/api/regressions/{id}/checks",
            get(regression_checks_handler),
        )
        .route(
            "/api/experiments",
            get(experiments_handler).post(create_experiment_handler),
        )
        .route(
            "/api/experiments/{id}",
            axum::routing::put(update_experiment_handler),
        )
        .route("/api/repos", get(repos_handler))
        .route("/api/repos/stats", get(repo_stats_handler))
        .route(
            "/api/repos/indexing-progress",
            get(indexing_progress_handler),
        )
        .route("/api/repos/dependencies", get(dependencies_handler))
        .route("/api/repos/{repo}/learning", get(repo_learning_handler))
        .route("/api/inference/stats", get(inference_stats_handler))
        .route("/api/inference/history", get(inference_history_handler))
        .route("/api/telemetry/overview", get(telemetry_overview_handler))
        .route(
            "/api/telemetry/timeseries",
            get(telemetry_timeseries_handler),
        )
        .route("/api/telemetry/pipeline", get(telemetry_pipeline_handler))
        .route("/api/telemetry/latency", get(telemetry_latency_handler))
        // Auth routes
        .route("/api/auth/login", axum::routing::post(login_handler))
        .route("/api/auth/logout", axum::routing::post(logout_handler))
        .route("/api/auth/me", axum::routing::get(me_handler))
        .route(
            "/api/auth/profile",
            axum::routing::put(update_profile_handler),
        )
        .route(
            "/api/auth/avatar",
            axum::routing::post(upload_avatar_handler)
                .layer(axum::extract::DefaultBodyLimit::max(6 * 1024 * 1024)),
        )
        // Config routes (admin only)
        .route(
            "/api/config",
            axum::routing::get(get_config_handler).put(put_config_handler),
        )
        // User CRUD routes
        .route(
            "/api/users",
            axum::routing::get(list_users_handler).post(create_user_handler),
        )
        .route(
            "/api/users/{id}",
            axum::routing::get(get_user_handler)
                .put(update_user_handler)
                .delete(delete_user_handler),
        )
        // Model browsing & download routes
        .route(
            "/api/chat/models/browse",
            axum::routing::get(browse_models_handler),
        )
        .route(
            "/api/chat/models/browse/{name}",
            axum::routing::get(model_info_handler),
        )
        .route(
            "/api/chat/models/download",
            axum::routing::post(download_model_handler),
        )
        .with_state(state)
        .nest_service("/avatars", ServeDir::new(avatars_dir));

    // Merge chat routes if enabled
    let api_routes = if let Some(chat_state) = chat_state {
        api_routes.merge(claudear_integrations::chat::create_chat_router(chat_state))
    } else {
        api_routes
    };

    // If dashboard directory is provided, serve from filesystem (development override)
    if let Some(dashboard_path) = dashboard_dir {
        let index_file = dashboard_path.join("index.html");
        let serve_dir =
            ServeDir::new(&dashboard_path).not_found_service(ServeFile::new(&index_file));

        api_routes.fallback_service(serve_dir)
    } else if super::embedded::has_dashboard() {
        // Serve the embedded dashboard compiled into the binary
        api_routes.fallback_service(tower::service_fn(super::embedded::embedded_fallback))
    } else {
        api_routes
    }
}

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    version: String,
    uptime_secs: u64,
    database: DatabaseStatus,
}

#[derive(Serialize)]
struct DatabaseStatus {
    status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Serialize)]
struct OverviewResponse {
    stats: FixAttemptStats,
    success_rate: f64,
    merge_rate: f64,
    recent_attempts: Vec<AttemptSummary>,
    sources: Vec<SourceSummary>,
    time_savings: Option<claudear_core::types::TimeSavings>,
    agent_spawns_today: i64,
}

#[derive(Serialize, Clone)]
struct AttemptSummary {
    id: i64,
    source: String,
    short_id: String,
    title: String,
    status: String,
    pr_url: Option<String>,
    attempted_at: String,
    retry_count: u32,
}

#[derive(Serialize)]
struct SourceSummary {
    name: String,
    total: usize,
    success: usize,
    failed: usize,
    merged: usize,
    success_rate: f64,
}

#[derive(Serialize)]
struct AttemptsResponse {
    attempts: Vec<AttemptSummary>,
    total: usize,
    page: usize,
    per_page: usize,
}

#[derive(Serialize)]
struct SourcesResponse {
    sources: Vec<SourceInfo>,
}

#[derive(Serialize)]
struct SourceInfo {
    name: String,
    enabled: bool,
    config: serde_json::Value,
}

#[derive(Serialize)]
struct RetriesResponse {
    retryable: Vec<AttemptSummary>,
    ready: Vec<AttemptSummary>,
    max_retries: u32,
}

#[derive(Deserialize)]
struct AttemptsQuery {
    status: Option<String>,
    source: Option<String>,
    page: Option<usize>,
    per_page: Option<usize>,
}

async fn health_handler(_user: AuthUser, State(state): State<ApiState>) -> Json<HealthResponse> {
    let uptime_secs = state.start_time.elapsed().as_secs();

    // Check database connectivity by attempting to get stats
    let database = match state.tracker.get_stats() {
        Ok(_) => DatabaseStatus {
            status: "ok".to_string(),
            error: None,
        },
        Err(e) => {
            tracing::error!("Database health check failed: {}", e);
            DatabaseStatus {
                status: "error".to_string(),
                error: Some("Database connection failed".to_string()),
            }
        }
    };

    let overall_status = if database.status == "ok" {
        "ok"
    } else {
        "degraded"
    };

    Json(HealthResponse {
        status: overall_status.to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime_secs,
        database,
    })
}

async fn stats_handler(
    _user: AuthUser,
    State(state): State<ApiState>,
) -> Result<Json<FixAttemptStats>, StatusCode> {
    state.tracker.get_stats().map(Json).map_err(|e| {
        tracing::error!(error = %e, "Internal server error");
        sentry::capture_error(&e);
        StatusCode::INTERNAL_SERVER_ERROR
    })
}

async fn overview_handler(
    _user: AuthUser,
    State(state): State<ApiState>,
) -> Result<Json<OverviewResponse>, StatusCode> {
    let stats = state.tracker.get_stats().map_err(|e| {
        tracing::error!(error = %e, "Internal server error");
        sentry::capture_error(&e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    // Calculate rates
    let completed = stats.merged + stats.closed + stats.failed + stats.cannot_fix;
    let success_rate = if stats.total > 0 {
        (stats.success + stats.merged) as f64 / stats.total as f64 * 100.0
    } else {
        0.0
    };
    let merge_rate = if completed > 0 {
        stats.merged as f64 / completed as f64 * 100.0
    } else {
        0.0
    };

    // Get recent attempts (last 10).
    let recent = state
        .tracker
        .list_recent_attempts(10)
        .map(|records| {
            records
                .into_iter()
                .map(|attempt| attempt_to_summary(&attempt))
                .collect()
        })
        .map_err(|e| {
            tracing::error!(error = %e, "Internal server error");
            sentry::capture_error(&e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    // Build source summaries
    let sources: Vec<SourceSummary> = stats
        .by_source
        .iter()
        .map(|(name, s)| {
            let rate = if s.total > 0 {
                (s.success + s.merged) as f64 / s.total as f64 * 100.0
            } else {
                0.0
            };
            SourceSummary {
                name: name.clone(),
                total: s.total,
                success: s.success,
                failed: s.failed,
                merged: s.merged,
                success_rate: rate,
            }
        })
        .collect();

    let now_utc = chrono::Utc::now();
    let seven_days_ago = (now_utc - chrono::Duration::days(7))
        .format("%Y-%m-%d %H:%M:%S")
        .to_string();
    let twenty_four_h_ago = (now_utc - chrono::Duration::hours(24))
        .format("%Y-%m-%d %H:%M:%S")
        .to_string();

    let time_savings = state
        .tracker
        .get_complexity_time_savings(
            &seven_days_ago,
            state.config.dashboard.hourly_engineer_rate,
            "7d",
        )
        .ok();

    let agent_spawns_today = state
        .tracker
        .get_agent_spawn_count(&twenty_four_h_ago)
        .unwrap_or(0);

    Ok(Json(OverviewResponse {
        stats,
        success_rate,
        merge_rate,
        recent_attempts: recent,
        sources,
        time_savings,
        agent_spawns_today,
    }))
}

async fn attempts_handler(
    _user: AuthUser,
    State(state): State<ApiState>,
    Query(query): Query<AttemptsQuery>,
) -> Result<Json<AttemptsResponse>, StatusCode> {
    let page = query.page.unwrap_or(1).max(1);
    let per_page = query.per_page.unwrap_or(20).min(100);
    let status_filter = query.status.as_ref().map(|s| s.to_lowercase());
    let source_filter = query.source.as_ref().map(|s| s.to_lowercase());
    let status_filter = status_filter.as_deref();
    let source_filter = source_filter.as_deref();

    let offset = (page - 1) * per_page;
    let rows = state
        .tracker
        .list_attempts(status_filter, source_filter, per_page, offset)
        .map_err(|e| {
            tracing::error!(error = %e, "Internal server error");
            sentry::capture_error(&e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
    let total = state
        .tracker
        .count_attempts(status_filter, source_filter)
        .map_err(|e| {
            tracing::error!(error = %e, "Internal server error");
            sentry::capture_error(&e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
    let attempts: Vec<AttemptSummary> = rows
        .into_iter()
        .map(|attempt| attempt_to_summary(&attempt))
        .collect();

    Ok(Json(AttemptsResponse {
        attempts,
        total,
        page,
        per_page,
    }))
}

async fn attempt_detail_handler(
    _user: AuthUser,
    State(state): State<ApiState>,
    Path(id): Path<i64>,
) -> Result<Json<FixAttempt>, StatusCode> {
    state
        .tracker
        .get_attempt_by_id(id)
        .map_err(|e| {
            tracing::error!(error = %e, "Internal server error");
            sentry::capture_error(&e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?
        .map(Json)
        .ok_or(StatusCode::NOT_FOUND)
}

async fn sources_handler(_user: AuthUser, State(state): State<ApiState>) -> Json<SourcesResponse> {
    let mut sources = Vec::new();

    if let Some(linear) = state.config.linear() {
        sources.push(SourceInfo {
            name: "linear".to_string(),
            enabled: linear.enabled,
            config: serde_json::json!({
                "trigger_labels": linear.trigger_labels,
                "trigger_states": linear.trigger_states,
                "has_webhook_secret": linear.webhook_secret.is_some(),
            }),
        });
    }

    if let Some(ref sentry) = state.config.issues.sentry {
        sources.push(SourceInfo {
            name: "sentry".to_string(),
            enabled: sentry.enabled,
            config: serde_json::json!({
                "org_slug": sentry.org_slug,
                "project_slugs": sentry.project_slugs,
                "min_event_count": sentry.min_event_count,
                "has_client_secret": sentry.client_secret.is_some(),
            }),
        });
    }

    if state.config.notifiers.whatsapp.source_enabled {
        sources.push(SourceInfo {
            name: "whatsapp".to_string(),
            enabled: true,
            config: serde_json::json!({
                "has_access_token": state.config.notifiers.whatsapp.access_token.is_some(),
                "has_phone_number_id": state.config.notifiers.whatsapp.phone_number_id.is_some(),
            }),
        });
    }

    if state.config.notifiers.telegram.source_enabled {
        sources.push(SourceInfo {
            name: "telegram".to_string(),
            enabled: true,
            config: serde_json::json!({
                "has_bot_token": state.config.notifiers.telegram.bot_token.is_some(),
                "chat_id": state.config.notifiers.telegram.chat_id.is_some(),
            }),
        });
    }

    Json(SourcesResponse { sources })
}

async fn retries_handler(
    _user: AuthUser,
    State(state): State<ApiState>,
) -> Result<Json<RetriesResponse>, StatusCode> {
    use crate::retry::RetryManager;

    let max_retries = state.config.retry.max_retries;

    let retryable = state
        .tracker
        .get_retryable_issues(max_retries)
        .map_err(|e| {
            tracing::error!(error = %e, "Internal server error");
            sentry::capture_error(&e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    // Create RetryManager to compute which attempts are ready for retry
    let retry_manager = RetryManager::new(state.config.retry.clone(), state.tracker.clone());

    let retryable_summaries: Vec<AttemptSummary> =
        retryable.iter().map(attempt_to_summary).collect();

    // Filter to find attempts that are ready for retry now
    let ready_summaries: Vec<AttemptSummary> = retryable
        .iter()
        .filter(|a| retry_manager.is_ready_for_retry(a))
        .map(attempt_to_summary)
        .collect();

    Ok(Json(RetriesResponse {
        retryable: retryable_summaries,
        ready: ready_summaries,
        max_retries,
    }))
}

fn attempt_to_summary(attempt: &FixAttempt) -> AttemptSummary {
    AttemptSummary {
        id: attempt.id,
        source: attempt.source.clone(),
        short_id: attempt.short_id.clone(),
        title: attempt.short_id.clone(), // We don't store title, use short_id
        status: attempt.status.to_string(),
        pr_url: attempt.pr_url.clone(),
        attempted_at: attempt.attempted_at.to_rfc3339(),
        retry_count: attempt.retry_count,
    }
}

/// Get attempts from tracker, optionally limited.
#[cfg(test)]
fn get_attempts(tracker: &Arc<dyn FixAttemptTracker>, limit: Option<usize>) -> Vec<AttemptSummary> {
    if let Some(max) = limit {
        if let Ok(attempts) = tracker.list_recent_attempts(max) {
            return attempts
                .into_iter()
                .map(|a| attempt_to_summary(&a))
                .collect();
        }
    }

    let mut all: Vec<FixAttempt> = Vec::new();

    for status in [
        FixAttemptStatus::Pending,
        FixAttemptStatus::Success,
        FixAttemptStatus::Failed,
        FixAttemptStatus::Merged,
        FixAttemptStatus::Closed,
        FixAttemptStatus::CannotFix,
    ] {
        if let Ok(attempts) = tracker.get_attempts_by_status(status) {
            all.extend(attempts);
        }
    }

    // Sort by attempted_at descending
    all.sort_by(|a, b| b.attempted_at.cmp(&a.attempted_at));

    let iter = all.into_iter().map(|a| attempt_to_summary(&a));

    match limit {
        Some(n) => iter.take(n).collect(),
        None => iter.collect(),
    }
}

/// Get raw attempt records from tracker for telemetry aggregation.
fn get_attempt_records(tracker: &Arc<dyn FixAttemptTracker>) -> Vec<FixAttempt> {
    let mut all: Vec<FixAttempt> = Vec::new();

    for status in [
        FixAttemptStatus::Pending,
        FixAttemptStatus::Success,
        FixAttemptStatus::Failed,
        FixAttemptStatus::Merged,
        FixAttemptStatus::Closed,
        FixAttemptStatus::CannotFix,
    ] {
        if let Ok(attempts) = tracker.get_attempts_by_status(status) {
            all.extend(attempts);
        }
    }

    all
}

/// Get raw attempt records since a timestamp for telemetry aggregation.
fn get_attempt_records_since(
    tracker: &Arc<dyn FixAttemptTracker>,
    since: DateTime<Utc>,
) -> Vec<FixAttempt> {
    if let Ok(attempts) = tracker.list_attempts_since(since) {
        return attempts;
    }

    get_attempt_records(tracker)
        .into_iter()
        .filter(|a| a.attempted_at >= since)
        .collect()
}

fn compute_processing_time_summary(
    metrics: Vec<claudear_core::types::ProcessingMetric>,
) -> ProcessingTimeSummary {
    let values: Vec<f64> = metrics.into_iter().map(|m| m.metric_value).collect();
    compute_processing_value_summary(values)
}

fn compute_processing_value_summary(mut values: Vec<f64>) -> ProcessingTimeSummary {
    if values.is_empty() {
        return ProcessingTimeSummary::default();
    }

    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let samples = values.len() as i64;
    let sum: f64 = values.iter().sum();
    let avg_secs = Some(sum / samples as f64);
    let max_secs = values.last().copied();

    let percentile = |p: f64| -> Option<f64> {
        if values.is_empty() {
            return None;
        }
        let rank = ((values.len() as f64 - 1.0) * p).round() as usize;
        values.get(rank).copied()
    };

    ProcessingTimeSummary {
        samples,
        avg_secs,
        p50_secs: percentile(0.50),
        p95_secs: percentile(0.95),
        p99_secs: percentile(0.99),
        max_secs,
    }
}

fn parse_telemetry_period(period: Option<&str>, default: &str) -> (String, Duration) {
    match period.unwrap_or(default).to_lowercase().as_str() {
        "hour" => ("hour".to_string(), Duration::hours(1)),
        "day" => ("day".to_string(), Duration::days(1)),
        "month" => ("month".to_string(), Duration::days(30)),
        "week" => ("week".to_string(), Duration::days(7)),
        _ => match default {
            "hour" => ("hour".to_string(), Duration::hours(1)),
            "day" => ("day".to_string(), Duration::days(1)),
            "month" => ("month".to_string(), Duration::days(30)),
            _ => ("week".to_string(), Duration::days(7)),
        },
    }
}

fn sum_metric_values(metrics: &[claudear_core::types::ProcessingMetric]) -> f64 {
    metrics.iter().map(|m| m.metric_value).sum()
}

fn average_metric_value(metrics: &[claudear_core::types::ProcessingMetric]) -> Option<f64> {
    if metrics.is_empty() {
        return None;
    }
    Some(sum_metric_values(metrics) / metrics.len() as f64)
}

fn max_metric_value(metrics: &[claudear_core::types::ProcessingMetric]) -> Option<f64> {
    metrics
        .iter()
        .map(|m| m.metric_value)
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
}

fn latest_metric_value(metrics: &[claudear_core::types::ProcessingMetric]) -> Option<f64> {
    metrics.first().map(|m| m.metric_value)
}

fn ratio(numerator: f64, denominator: f64) -> Option<f64> {
    if denominator > 0.0 {
        Some(numerator / denominator)
    } else {
        None
    }
}

fn summarize_window(
    attempts: &[FixAttempt],
    window: &str,
    start: DateTime<Utc>,
) -> TelemetryWindowMetric {
    let in_window: Vec<&FixAttempt> = attempts
        .iter()
        .filter(|a| a.attempted_at >= start)
        .collect();

    let processed = in_window
        .iter()
        .filter(|a| a.status != FixAttemptStatus::Pending)
        .count() as i64;
    let successful = in_window
        .iter()
        .filter(|a| {
            matches!(
                a.status,
                FixAttemptStatus::Success | FixAttemptStatus::Merged
            )
        })
        .count() as i64;
    let failed = in_window
        .iter()
        .filter(|a| {
            matches!(
                a.status,
                FixAttemptStatus::Failed | FixAttemptStatus::Closed | FixAttemptStatus::CannotFix
            )
        })
        .count() as i64;
    let merged = in_window
        .iter()
        .filter(|a| a.status == FixAttemptStatus::Merged)
        .count() as i64;

    let success_rate = if processed > 0 {
        (successful as f64 / processed as f64) * 100.0
    } else {
        0.0
    };
    let error_rate = if processed > 0 {
        (failed as f64 / processed as f64) * 100.0
    } else {
        0.0
    };

    let hours = match window {
        "1h" => 1.0,
        "24h" => 24.0,
        "7d" => 24.0 * 7.0,
        _ => 1.0,
    };
    let throughput_per_hour = if hours > 0.0 {
        processed as f64 / hours
    } else {
        0.0
    };

    TelemetryWindowMetric {
        window: window.to_string(),
        processed,
        successful,
        failed,
        merged,
        success_rate,
        error_rate,
        throughput_per_hour,
    }
}

fn floor_to_bucket(timestamp: DateTime<Utc>, bucket_minutes: i64) -> DateTime<Utc> {
    let bucket_seconds = (bucket_minutes.max(1) * 60).max(60);
    let ts = timestamp.timestamp();
    let floored = ts - ts.rem_euclid(bucket_seconds);
    Utc.timestamp_opt(floored, 0).single().unwrap_or(timestamp)
}

#[derive(Deserialize)]
struct ActivityQuery {
    limit: Option<usize>,
    source: Option<String>,
}

#[derive(Deserialize)]
struct MetricsQuery {
    name: Option<String>,
    period: Option<String>,
    limit: Option<usize>,
}

#[derive(Deserialize)]
struct ErrorsQuery {
    limit: Option<usize>,
}

#[derive(Deserialize)]
struct IssuesQuery {
    source: Option<String>,
    page: Option<usize>,
    per_page: Option<usize>,
}

#[derive(Serialize)]
struct IssueSummary {
    id: i64,
    source: String,
    issue_id: String,
    short_id: Option<String>,
    title: Option<String>,
    description: Option<String>,
    url: Option<String>,
    priority: Option<String>,
    status: Option<String>,
    labels: Option<Vec<String>>,
    has_embedding: bool,
    created_at: String,
    updated_at: Option<String>,
}

#[derive(Serialize)]
struct IssuesResponse {
    issues: Vec<IssueSummary>,
    total: usize,
    page: usize,
    per_page: usize,
}

#[derive(Deserialize)]
struct PrsQuery {
    status: Option<String>,
    limit: Option<usize>,
}

#[derive(Deserialize)]
struct FeedbackQuery {
    source: Option<String>,
    limit: Option<usize>,
}

#[derive(Deserialize)]
struct RegressionsQuery {
    status: Option<String>,
}

#[derive(Deserialize)]
struct InferenceHistoryQuery {
    limit: Option<usize>,
}

#[derive(Deserialize)]
struct TelemetryTimeseriesQuery {
    period: Option<String>,
    bucket_minutes: Option<i64>,
}

#[derive(Deserialize)]
struct TelemetryPeriodQuery {
    period: Option<String>,
}

#[derive(Serialize)]
struct AttemptDetailResponse {
    attempt: FixAttempt,
    executions: Vec<claudear_core::types::AgentExecution>,
    reviews: Vec<claudear_core::types::PrReviewRecord>,
    feedback: Option<claudear_analysis::feedback::FixOutcome>,
}

#[derive(Serialize)]
struct AttemptExecutionLogResponse {
    attempt_id: i64,
    execution_id: i64,
    stream: String,
    path: Option<String>,
    content: Option<String>,
    truncated: bool,
}

#[derive(Debug, Clone, Serialize)]
struct TelemetryWindowMetric {
    window: String,
    processed: i64,
    successful: i64,
    failed: i64,
    merged: i64,
    success_rate: f64,
    error_rate: f64,
    throughput_per_hour: f64,
}

#[derive(Debug, Clone, Serialize, Default)]
struct TelemetryQueueMetrics {
    pending_attempts: i64,
    retryable_attempts: i64,
    ready_retries: i64,
    open_prs: i64,
    watches_awaiting_release: i64,
    watches_monitoring: i64,
    watches_resolved: i64,
    watches_regressed: i64,
}

#[derive(Debug, Clone, Serialize, Default)]
struct ProcessingTimeSummary {
    samples: i64,
    avg_secs: Option<f64>,
    p50_secs: Option<f64>,
    p95_secs: Option<f64>,
    p99_secs: Option<f64>,
    max_secs: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Default)]
struct TelemetryProcessingTime {
    all_time: ProcessingTimeSummary,
    last_24h: ProcessingTimeSummary,
}

#[derive(Debug, Clone, Serialize, Default)]
struct SourceTelemetry {
    source: String,
    total: i64,
    pending: i64,
    success: i64,
    failed: i64,
    merged: i64,
    closed: i64,
    cannot_fix: i64,
    retryable: i64,
    success_rate: f64,
}

#[derive(Debug, Clone, Serialize)]
struct TelemetryOverviewResponse {
    generated_at: String,
    uptime_secs: u64,
    windows: Vec<TelemetryWindowMetric>,
    queue: TelemetryQueueMetrics,
    processing_time: TelemetryProcessingTime,
    source_breakdown: Vec<SourceTelemetry>,
    top_errors: Vec<claudear_core::types::ErrorPattern>,
    activity_last_hour: HashMap<String, i64>,
    metric_counts_last_24h: HashMap<String, i64>,
    diagnostics: Option<claudear_storage::DiagnosticCounts>,
    pr_analytics: claudear_core::types::PrAnalytics,
    agent_spawns_today: i64,
    agent_spawns_this_week: i64,
}

#[derive(Debug, Clone, Serialize, Default)]
struct TelemetryTimeseriesPoint {
    bucket_start: String,
    total: i64,
    pending: i64,
    success: i64,
    failed: i64,
    merged: i64,
    closed: i64,
    cannot_fix: i64,
}

#[derive(Debug, Clone, Serialize)]
struct TelemetryTimeseriesResponse {
    period: String,
    bucket_minutes: i64,
    generated_at: String,
    points: Vec<TelemetryTimeseriesPoint>,
}

#[derive(Debug, Clone, Serialize, Default)]
struct TelemetryPipelineTotals {
    fetched: f64,
    matched: f64,
    queued: f64,
    processed: f64,
    pr_created: f64,
    retries_found: f64,
    retries_executed: f64,
    retries_failed: f64,
    pr_status_checks: f64,
    pr_status_merged: f64,
    pr_status_closed: f64,
    pr_status_errors: f64,
    regression_watches_created: f64,
    auto_resolved_on_merge: f64,
    cascade_triggered: f64,
    cascade_failed: f64,
}

#[derive(Debug, Clone, Serialize, Default)]
struct TelemetryPipelineConversion {
    match_rate: Option<f64>,
    queue_rate: Option<f64>,
    processing_rate: Option<f64>,
    pr_yield_rate: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Default)]
struct TelemetryPollLoad {
    poll_cycles: i64,
    avg_cycle_secs: Option<f64>,
    p95_cycle_secs: Option<f64>,
    active_avg: Option<f64>,
    active_max: Option<f64>,
    pending_avg: Option<f64>,
    pending_max: Option<f64>,
    total_latest: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Default)]
struct TelemetryPipelineSource {
    source: String,
    fetched: f64,
    matched: f64,
    queued: f64,
    processed: f64,
    pr_created: f64,
    retries_executed: f64,
    retries_failed: f64,
    match_rate: Option<f64>,
    queue_rate: Option<f64>,
    processing_rate: Option<f64>,
    pr_yield_rate: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct TelemetryPipelineResponse {
    generated_at: String,
    period: String,
    totals: TelemetryPipelineTotals,
    conversion: TelemetryPipelineConversion,
    poll_load: TelemetryPollLoad,
    per_source: Vec<TelemetryPipelineSource>,
}

#[derive(Debug, Clone, Serialize)]
struct TelemetryLatencyByStatus {
    status: String,
    summary: ProcessingTimeSummary,
}

#[derive(Debug, Clone, Serialize)]
struct TelemetryLatencyHistogramBucket {
    label: String,
    upper_bound_secs: Option<f64>,
    count: i64,
}

#[derive(Debug, Clone, Serialize)]
struct TelemetryLatencyResponse {
    generated_at: String,
    period: String,
    overall: ProcessingTimeSummary,
    by_status: Vec<TelemetryLatencyByStatus>,
    histogram: Vec<TelemetryLatencyHistogramBucket>,
}

async fn attempt_full_detail_handler(
    _user: AuthUser,
    State(state): State<ApiState>,
    Path(id): Path<i64>,
) -> Result<Json<AttemptDetailResponse>, StatusCode> {
    let attempt = state
        .tracker
        .get_attempt_by_id(id)
        .map_err(|e| {
            tracing::error!(error = %e, "Internal server error");
            sentry::capture_error(&e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?
        .ok_or(StatusCode::NOT_FOUND)?;

    let executions = state
        .tracker
        .get_executions_for_attempt(id)
        .unwrap_or_default();

    let reviews = state
        .tracker
        .get_reviews_for_attempt(id)
        .unwrap_or_default();

    let feedback = state
        .tracker
        .get_feedback_outcome_by_attempt(id)
        .unwrap_or(None);

    Ok(Json(AttemptDetailResponse {
        attempt,
        executions,
        reviews,
        feedback,
    }))
}

async fn attempt_execution_log_handler(
    _user: AuthUser,
    State(state): State<ApiState>,
    Path((attempt_id, execution_id, stream)): Path<(i64, i64, String)>,
) -> Result<Json<AttemptExecutionLogResponse>, StatusCode> {
    if stream != "stdout" && stream != "stderr" && stream != "events" {
        return Err(StatusCode::BAD_REQUEST);
    }

    let attempt_exists = state
        .tracker
        .get_attempt_by_id(attempt_id)
        .map_err(|e| {
            tracing::error!(error = %e, "Internal server error");
            sentry::capture_error(&e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?
        .is_some();
    if !attempt_exists {
        return Err(StatusCode::NOT_FOUND);
    }

    let execution = state
        .tracker
        .get_execution_for_attempt(attempt_id, execution_id)
        .map_err(|e| {
            tracing::error!(error = %e, "Internal server error");
            sentry::capture_error(&e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?
        .ok_or(StatusCode::NOT_FOUND)?;

    let log_path = match stream.as_str() {
        "stdout" => execution.stdout_log_path.clone(),
        "stderr" => execution.stderr_log_path.clone(),
        "events" => execution.event_log_path.clone(),
        _ => None,
    };

    let fallback_preview = match stream.as_str() {
        "stdout" => execution.stdout_preview.clone(),
        "stderr" => execution.stderr_preview.clone(),
        "events" => None,
        _ => None,
    };

    // Validate that the log path is within the expected log root directory
    // to prevent path traversal attacks via crafted database records.
    if let Some(ref path) = log_path {
        let log_root = claudear_integrations::runner::resolve_log_root();
        let canonical_root = tokio::fs::canonicalize(&log_root)
            .await
            .unwrap_or(std::path::absolute(&log_root).unwrap_or_else(|_| log_root.clone()));
        match tokio::fs::canonicalize(path).await {
            Ok(canonical_path) => {
                if !canonical_path.starts_with(&canonical_root) {
                    tracing::warn!(
                        path = %path,
                        log_root = %canonical_root.display(),
                        "Execution log path traversal blocked"
                    );
                    return Err(StatusCode::FORBIDDEN);
                }
            }
            Err(_) => {
                // File doesn't exist or can't be resolved; fall through to normal read
                // which will produce a fallback_preview
            }
        }
    }

    let mut truncated = false;
    let content = if let Some(path) = &log_path {
        match tokio::fs::read_to_string(path).await {
            Ok(raw) => {
                let (value, is_truncated) = tail_utf8(&raw, 200_000);
                truncated = is_truncated;
                Some(value)
            }
            Err(e) => {
                tracing::warn!(
                    attempt_id = attempt_id,
                    execution_id = execution_id,
                    stream = %stream,
                    path = %path,
                    error = %e,
                    "Failed to read execution log file"
                );
                fallback_preview
            }
        }
    } else {
        fallback_preview
    };

    Ok(Json(AttemptExecutionLogResponse {
        attempt_id,
        execution_id,
        stream,
        path: log_path,
        content,
        truncated,
    }))
}

fn tail_utf8(value: &str, max_bytes: usize) -> (String, bool) {
    if value.len() <= max_bytes {
        return (value.to_string(), false);
    }

    let start = value.len().saturating_sub(max_bytes);
    let safe_start = value
        .char_indices()
        .find(|(idx, _)| *idx >= start)
        .map(|(idx, _)| idx)
        .unwrap_or(value.len());
    (format!("...[truncated]\n{}", &value[safe_start..]), true)
}

async fn activity_handler(
    _user: AuthUser,
    State(state): State<ApiState>,
    Query(query): Query<ActivityQuery>,
) -> Result<Json<Vec<claudear_core::types::ActivityLogEntry>>, StatusCode> {
    let limit = query.limit.unwrap_or(50).min(500);
    let source_filter = query.source.as_deref();

    state
        .tracker
        .get_recent_activities_filtered(limit, source_filter)
        .map(Json)
        .map_err(|e| {
            tracing::error!(error = %e, "Internal server error");
            sentry::capture_error(&e);
            StatusCode::INTERNAL_SERVER_ERROR
        })
}

async fn analytics_summary_handler(
    _user: AuthUser,
    State(state): State<ApiState>,
) -> Result<Json<claudear_core::types::AnalyticsSummary>, StatusCode> {
    let mut summary = state.tracker.get_analytics_summary().map_err(|e| {
        tracing::error!(error = %e, "Internal server error");
        sentry::capture_error(&e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    summary.avg_time_to_pr_mins = state.tracker.get_avg_time_to_pr().unwrap_or(None);

    let thirty_days_ago = (chrono::Utc::now() - chrono::Duration::days(30))
        .format("%Y-%m-%d %H:%M:%S")
        .to_string();
    summary.cost_estimate = state
        .tracker
        .get_cost_estimate(
            &thirty_days_ago,
            state.config.dashboard.max_plan_monthly_cost,
            "30d",
        )
        .ok();
    summary.mttr_trend = state.tracker.get_mttr_trend(8).unwrap_or_default();
    summary.repo_leaderboard = state.tracker.get_repo_leaderboard().unwrap_or_default();

    Ok(Json(summary))
}

async fn metrics_handler(
    _user: AuthUser,
    State(state): State<ApiState>,
    Query(query): Query<MetricsQuery>,
) -> Result<Json<Vec<claudear_core::types::ProcessingMetric>>, StatusCode> {
    let metric_name = query.name.as_deref().unwrap_or("processing_time");
    let limit = query.limit.unwrap_or(100).min(1000);

    let since = query.period.as_deref().and_then(|p| {
        let duration = match p {
            "hour" => chrono::Duration::hours(1),
            "day" => chrono::Duration::days(1),
            "week" => chrono::Duration::days(7),
            "month" => chrono::Duration::days(30),
            _ => return None,
        };
        Some(chrono::Utc::now() - duration)
    });

    state
        .tracker
        .get_metrics(metric_name, since, limit)
        .map(Json)
        .map_err(|e| {
            tracing::error!(error = %e, "Internal server error");
            sentry::capture_error(&e);
            StatusCode::INTERNAL_SERVER_ERROR
        })
}

async fn errors_handler(
    _user: AuthUser,
    State(state): State<ApiState>,
    Query(query): Query<ErrorsQuery>,
) -> Result<Json<Vec<claudear_core::types::ErrorPattern>>, StatusCode> {
    let limit = query.limit.unwrap_or(50).min(200);

    state
        .tracker
        .get_error_patterns(limit)
        .map(Json)
        .map_err(|e| {
            tracing::error!(error = %e, "Internal server error");
            sentry::capture_error(&e);
            StatusCode::INTERNAL_SERVER_ERROR
        })
}

async fn issues_handler(
    _user: AuthUser,
    State(state): State<ApiState>,
    Query(query): Query<IssuesQuery>,
) -> Result<Json<IssuesResponse>, StatusCode> {
    let page = query.page.unwrap_or(1).max(1);
    let per_page = query.per_page.unwrap_or(100).min(500);
    let offset = (page - 1) * per_page;

    let total = state
        .tracker
        .count_issues(query.source.as_deref())
        .map_err(|e| {
            tracing::error!(error = %e, "Internal server error");
            sentry::capture_error(&e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    let rows = state
        .tracker
        .list_issues(query.source.as_deref(), per_page, offset)
        .map_err(|e| {
            tracing::error!(error = %e, "Internal server error");
            sentry::capture_error(&e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    let issues: Vec<IssueSummary> = rows
        .into_iter()
        .map(|ie| {
            let labels: Option<Vec<String>> = ie
                .labels
                .as_deref()
                .and_then(|s| serde_json::from_str(s).ok());
            IssueSummary {
                id: ie.id,
                source: ie.source,
                issue_id: ie.issue_id,
                short_id: ie.short_id,
                title: ie.title,
                description: ie.description,
                url: ie.url,
                priority: ie.priority,
                status: ie.status,
                labels,
                has_embedding: ie.embedding.is_some(),
                created_at: ie.created_at.format("%Y-%m-%d %H:%M:%S").to_string(),
                updated_at: ie
                    .updated_at
                    .map(|dt| dt.format("%Y-%m-%d %H:%M:%S").to_string()),
            }
        })
        .collect();

    Ok(Json(IssuesResponse {
        issues,
        total,
        page,
        per_page,
    }))
}

async fn prs_handler(
    _user: AuthUser,
    State(state): State<ApiState>,
    Query(query): Query<PrsQuery>,
) -> Result<Json<Vec<claudear_core::types::PrRecord>>, StatusCode> {
    let limit = query.limit.unwrap_or(100).min(500);

    state
        .tracker
        .list_prs(query.status.as_deref(), limit)
        .map(Json)
        .map_err(|e| {
            tracing::error!(error = %e, "Internal server error");
            sentry::capture_error(&e);
            StatusCode::INTERNAL_SERVER_ERROR
        })
}

async fn pr_analytics_handler(
    _user: AuthUser,
    State(state): State<ApiState>,
) -> Result<Json<claudear_core::types::PrAnalytics>, StatusCode> {
    let mut analytics = state.tracker.get_pr_analytics().map_err(|e| {
        tracing::error!(error = %e, "Internal server error");
        sentry::capture_error(&e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    analytics.avg_time_to_pr_mins = state.tracker.get_avg_time_to_pr().unwrap_or(None);
    analytics.rejection_reasons = state.tracker.get_rejection_reasons(10).unwrap_or_default();

    Ok(Json(analytics))
}

async fn feedback_handler(
    _user: AuthUser,
    State(state): State<ApiState>,
    Query(query): Query<FeedbackQuery>,
) -> Result<Json<Vec<claudear_analysis::feedback::FixOutcome>>, StatusCode> {
    let limit = query.limit.unwrap_or(50).min(200);

    state
        .tracker
        .get_feedback_outcomes(query.source.as_deref(), limit)
        .map(Json)
        .map_err(|e| {
            tracing::error!(error = %e, "Internal server error");
            sentry::capture_error(&e);
            StatusCode::INTERNAL_SERVER_ERROR
        })
}

async fn regressions_handler(
    _user: AuthUser,
    State(state): State<ApiState>,
    Query(query): Query<RegressionsQuery>,
) -> Result<Json<Vec<claudear_core::types::RegressionWatch>>, StatusCode> {
    match query.status.as_deref() {
        Some(status_str) => {
            let status: RegressionWatchStatus =
                status_str.parse().map_err(|_| StatusCode::BAD_REQUEST)?;
            state
                .tracker
                .get_regression_watches_by_status(status)
                .map(Json)
                .map_err(|e| {
                    tracing::error!(error = %e, "Internal server error");
                    sentry::capture_error(&e);
                    StatusCode::INTERNAL_SERVER_ERROR
                })
        }
        None => state
            .tracker
            .get_all_regression_watches()
            .map(Json)
            .map_err(|e| {
                tracing::error!(error = %e, "Internal server error");
                sentry::capture_error(&e);
                StatusCode::INTERNAL_SERVER_ERROR
            }),
    }
}

async fn regression_checks_handler(
    _user: AuthUser,
    State(state): State<ApiState>,
    Path(id): Path<i64>,
) -> Result<Json<Vec<claudear_core::types::RegressionCheck>>, StatusCode> {
    state
        .tracker
        .get_regression_checks(id)
        .map(Json)
        .map_err(|e| {
            tracing::error!(error = %e, "Internal server error");
            sentry::capture_error(&e);
            StatusCode::INTERNAL_SERVER_ERROR
        })
}

async fn experiments_handler(
    _user: AuthUser,
    State(state): State<ApiState>,
) -> Result<Json<Vec<claudear_core::types::PromptExperiment>>, StatusCode> {
    state
        .tracker
        .get_active_experiments()
        .map(Json)
        .map_err(|e| {
            tracing::error!(error = %e, "Internal server error");
            sentry::capture_error(&e);
            StatusCode::INTERNAL_SERVER_ERROR
        })
}

#[derive(Deserialize)]
struct CreateExperimentRequest {
    experiment_name: String,
    variant: String,
    prompt_template: String,
    #[serde(default)]
    active: Option<bool>,
}

#[derive(Deserialize)]
struct UpdateExperimentRequest {
    experiment_name: String,
    variant: String,
    prompt_template: String,
    #[serde(default)]
    active: Option<bool>,
}

async fn create_experiment_handler(
    _user: AdminUser,
    State(state): State<ApiState>,
    Json(body): Json<CreateExperimentRequest>,
) -> Result<(StatusCode, Json<claudear_core::types::PromptExperiment>), StatusCode> {
    let experiment_name = body.experiment_name.trim().to_string();
    let variant = body.variant.trim().to_string();
    if experiment_name.is_empty() || variant.is_empty() || body.prompt_template.trim().is_empty() {
        return Err(StatusCode::BAD_REQUEST);
    }

    let mut hasher = Sha256::new();
    hasher.update(body.prompt_template.as_bytes());
    let prompt_hash = hex::encode(hasher.finalize());

    let mut experiment = claudear_core::types::PromptExperiment::new(
        experiment_name,
        variant,
        body.prompt_template,
        prompt_hash,
    );
    experiment.active = body.active.unwrap_or(true);

    let id = state.tracker.save_experiment(&experiment).map_err(|e| {
        tracing::error!(error = %e, "Internal server error");
        sentry::capture_error(&e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;
    experiment.id = id;

    Ok((StatusCode::CREATED, Json(experiment)))
}

async fn update_experiment_handler(
    _user: AdminUser,
    State(state): State<ApiState>,
    Path(id): Path<i64>,
    Json(body): Json<UpdateExperimentRequest>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let experiment_name = body.experiment_name.trim().to_string();
    let variant = body.variant.trim().to_string();
    if experiment_name.is_empty() || variant.is_empty() || body.prompt_template.trim().is_empty() {
        return Err(StatusCode::BAD_REQUEST);
    }

    let mut hasher = Sha256::new();
    hasher.update(body.prompt_template.as_bytes());
    let prompt_hash = hex::encode(hasher.finalize());

    let updated = state
        .tracker
        .update_experiment(
            id,
            &experiment_name,
            &variant,
            &body.prompt_template,
            &prompt_hash,
            body.active.unwrap_or(true),
        )
        .map_err(|e| {
            tracing::error!(error = %e, "Internal server error");
            sentry::capture_error(&e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    if !updated {
        return Err(StatusCode::NOT_FOUND);
    }

    Ok(Json(serde_json::json!({ "ok": true })))
}

async fn repos_handler(
    _user: AuthUser,
    State(state): State<ApiState>,
) -> Result<Json<Vec<claudear_storage::StoredIndexedRepo>>, StatusCode> {
    state.tracker.list_indexed_repos().map(Json).map_err(|e| {
        tracing::error!(error = %e, "Internal server error");
        sentry::capture_error(&e);
        StatusCode::INTERNAL_SERVER_ERROR
    })
}

async fn repo_stats_handler(
    _user: AuthUser,
    State(state): State<ApiState>,
) -> Result<Json<claudear_storage::IndexStats>, StatusCode> {
    state.tracker.get_index_stats().map(Json).map_err(|e| {
        tracing::error!(error = %e, "Internal server error");
        sentry::capture_error(&e);
        StatusCode::INTERNAL_SERVER_ERROR
    })
}

async fn dependencies_handler(
    _user: AuthUser,
    State(state): State<ApiState>,
) -> Result<Json<Vec<claudear_storage::StoredDependency>>, StatusCode> {
    state
        .tracker
        .list_all_dependencies()
        .map(Json)
        .map_err(|e| {
            tracing::error!(error = %e, "Internal server error");
            sentry::capture_error(&e);
            StatusCode::INTERNAL_SERVER_ERROR
        })
}

#[derive(Serialize)]
struct KnowledgeEntry {
    id: i64,
    value: String,
    source_type: String,
    confidence: f64,
    occurrence_count: i64,
    updated_at: DateTime<Utc>,
}

#[derive(Serialize)]
struct KnowledgeGroup {
    key: String,
    label: String,
    entries: Vec<KnowledgeEntry>,
}

#[derive(Serialize)]
struct ReviewPatternSummary {
    total_patterns: usize,
    by_category: HashMap<String, usize>,
    promoted_count: usize,
}

#[derive(Serialize)]
struct RepoLearningResponse {
    repo: String,
    knowledge: Vec<KnowledgeGroup>,
    knowledge_total: usize,
    instructions: Vec<claudear_core::types::PromotedInstruction>,
    review_patterns: Vec<claudear_core::types::ReviewPattern>,
    review_pattern_summary: ReviewPatternSummary,
    strategies: Vec<claudear_core::types::StrategyFingerprint>,
    diff_analyses: Vec<claudear_core::types::DiffAnalysis>,
    correlations: Vec<claudear_core::types::CrossRepoCorrelation>,
}

fn knowledge_key_label(key: &str) -> &str {
    match key {
        "common_fix_dirs" => "Common Fix Directories",
        "file_conventions" => "File Conventions",
        "test_pattern" => "Test Patterns",
        "review_preferences" => "Review Preferences",
        "common_root_causes" => "Common Root Causes",
        "promoted_qa" => "Promoted Q&A",
        _ => key,
    }
}

async fn repo_learning_handler(
    _user: AuthUser,
    State(state): State<ApiState>,
    Path(repo): Path<String>,
) -> Result<Json<RepoLearningResponse>, StatusCode> {
    let tracker = &state.tracker;

    let raw_knowledge = tracker.get_repo_knowledge(&repo).map_err(|e| {
        tracing::error!(error = %e, "Internal server error");
        sentry::capture_error(&e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    let mut groups_map: BTreeMap<String, Vec<KnowledgeEntry>> = BTreeMap::new();
    for rk in &raw_knowledge {
        groups_map
            .entry(rk.knowledge_key.clone())
            .or_default()
            .push(KnowledgeEntry {
                id: rk.id,
                value: rk.knowledge_value.clone(),
                source_type: rk.source_type.clone(),
                confidence: rk.confidence,
                occurrence_count: rk.occurrence_count,
                updated_at: rk.updated_at,
            });
    }
    let knowledge_total = raw_knowledge.len();
    let knowledge: Vec<KnowledgeGroup> = groups_map
        .into_iter()
        .map(|(key, entries)| KnowledgeGroup {
            label: knowledge_key_label(&key).to_string(),
            key,
            entries,
        })
        .collect();

    let instructions = tracker.get_promoted_instructions(&repo).map_err(|e| {
        tracing::error!(error = %e, "Internal server error");
        sentry::capture_error(&e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    let review_patterns = tracker.get_review_patterns(&repo, 200).map_err(|e| {
        tracing::error!(error = %e, "Internal server error");
        sentry::capture_error(&e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    let mut by_category: HashMap<String, usize> = HashMap::new();
    let mut promoted_count: usize = 0;
    for rp in &review_patterns {
        *by_category.entry(rp.category.to_string()).or_default() += 1;
        if rp.promoted_to_instruction {
            promoted_count += 1;
        }
    }
    let review_pattern_summary = ReviewPatternSummary {
        total_patterns: review_patterns.len(),
        by_category,
        promoted_count,
    };

    let strategies = tracker.get_successful_strategies(&repo, 100).map_err(|e| {
        tracing::error!(error = %e, "Internal server error");
        sentry::capture_error(&e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    let diff_analyses = tracker
        .get_diff_analyses_for_repo(&repo, 100)
        .map_err(|e| {
            tracing::error!(error = %e, "Internal server error");
            sentry::capture_error(&e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    let all_correlations = tracker.get_cross_repo_correlations(1, 168).map_err(|e| {
        tracing::error!(error = %e, "Internal server error");
        sentry::capture_error(&e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;
    let correlations: Vec<_> = all_correlations
        .into_iter()
        .filter(|c| c.repo_a == repo || c.repo_b == repo)
        .collect();

    Ok(Json(RepoLearningResponse {
        repo,
        knowledge,
        knowledge_total,
        instructions,
        review_patterns,
        review_pattern_summary,
        strategies,
        diff_analyses,
        correlations,
    }))
}

async fn indexing_progress_handler(
    _user: AuthUser,
    ws: WebSocketUpgrade,
    State(state): State<ApiState>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| indexing_progress_ws(socket, state.indexing_rx))
}

async fn indexing_progress_ws(
    mut socket: WebSocket,
    mut rx: tokio::sync::watch::Receiver<claudear_storage::IndexingProgress>,
) {
    // Send current state immediately on connect
    {
        let progress = rx.borrow_and_update().clone();
        if let Ok(json) = serde_json::to_string(&progress) {
            if socket.send(Message::Text(json.into())).await.is_err() {
                return;
            }
        }
    }

    let mut ping_interval = tokio::time::interval(std::time::Duration::from_secs(30));
    // The first tick fires immediately; consume it so we don't send a spurious ping.
    ping_interval.tick().await;

    // Wait for changes pushed from SQLite write hooks
    loop {
        tokio::select! {
            // Watch channel notified — a write method updated progress
            result = rx.changed() => {
                if result.is_err() {
                    // Sender dropped (server shutting down)
                    break;
                }
                let progress = rx.borrow_and_update().clone();
                let json = match serde_json::to_string(&progress) {
                    Ok(j) => j,
                    Err(_) => break,
                };
                if socket.send(Message::Text(json.into())).await.is_err() {
                    break;
                }
                // If idle, we've sent the final state — close gracefully
                if progress.status == "idle" {
                    let _ = socket.send(Message::Close(None)).await;
                    break;
                }
            }
            // Keepalive ping every 30 seconds
            _ = ping_interval.tick() => {
                if socket.send(Message::Ping(vec![].into())).await.is_err() {
                    break;
                }
            }
            // Client sent something (close frame or disconnect)
            msg = socket.recv() => {
                match msg {
                    Some(Ok(Message::Close(_))) | None => break,
                    _ => {} // ignore pings, text, etc.
                }
            }
        }
    }
}

async fn inference_stats_handler(
    _user: AuthUser,
    State(state): State<ApiState>,
) -> Result<Json<claudear_storage::InferenceStats>, StatusCode> {
    state.tracker.get_inference_stats().map(Json).map_err(|e| {
        tracing::error!(error = %e, "Internal server error");
        sentry::capture_error(&e);
        StatusCode::INTERNAL_SERVER_ERROR
    })
}

async fn inference_history_handler(
    _user: AuthUser,
    State(state): State<ApiState>,
    Query(query): Query<InferenceHistoryQuery>,
) -> Result<Json<Vec<claudear_storage::InferenceHistoryEntry>>, StatusCode> {
    let limit = query.limit.unwrap_or(50).min(500);

    state
        .tracker
        .get_inference_history(limit)
        .map(Json)
        .map_err(|e| {
            tracing::error!(error = %e, "Internal server error");
            sentry::capture_error(&e);
            StatusCode::INTERNAL_SERVER_ERROR
        })
}

async fn telemetry_overview_handler(
    _user: AuthUser,
    State(state): State<ApiState>,
) -> Result<Json<TelemetryOverviewResponse>, StatusCode> {
    let now = Utc::now();
    let stats = state.tracker.get_stats().map_err(|e| {
        tracing::error!(error = %e, "Internal server error");
        sentry::capture_error(&e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;
    let recent_attempts = get_attempt_records_since(&state.tracker, now - Duration::days(7));

    let windows = vec![
        summarize_window(&recent_attempts, "1h", now - Duration::hours(1)),
        summarize_window(&recent_attempts, "24h", now - Duration::hours(24)),
        summarize_window(&recent_attempts, "7d", now - Duration::days(7)),
    ];

    let retryable = state
        .tracker
        .get_retryable_issues(state.config.retry.max_retries)
        .map_err(|e| {
            tracing::error!(error = %e, "Internal server error");
            sentry::capture_error(&e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
    let retry_manager = RetryManager::new(state.config.retry.clone(), state.tracker.clone());
    let ready_retries = retryable
        .iter()
        .filter(|a| retry_manager.is_ready_for_retry(a))
        .count() as i64;

    let pr_analytics = state.tracker.get_pr_analytics().map_err(|e| {
        tracing::error!(error = %e, "Internal server error");
        sentry::capture_error(&e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    let watches = state.tracker.get_all_regression_watches().map_err(|e| {
        tracing::error!(error = %e, "Internal server error");
        sentry::capture_error(&e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;
    let mut queue = TelemetryQueueMetrics {
        pending_attempts: stats.pending as i64,
        retryable_attempts: retryable.len() as i64,
        ready_retries,
        open_prs: pr_analytics.open,
        ..TelemetryQueueMetrics::default()
    };
    for watch in &watches {
        match watch.status {
            RegressionWatchStatus::AwaitingRelease => queue.watches_awaiting_release += 1,
            RegressionWatchStatus::Monitoring => queue.watches_monitoring += 1,
            RegressionWatchStatus::Resolved => queue.watches_resolved += 1,
            RegressionWatchStatus::Regressed => queue.watches_regressed += 1,
        }
    }

    let processing_time_all = state
        .tracker
        .get_metrics("processing_time", None, 20_000)
        .unwrap_or_default();
    let processing_time_last_24h = state
        .tracker
        .get_metrics("processing_time", Some(now - Duration::hours(24)), 20_000)
        .unwrap_or_default();
    let processing_time = TelemetryProcessingTime {
        all_time: compute_processing_time_summary(processing_time_all),
        last_24h: compute_processing_time_summary(processing_time_last_24h),
    };

    let top_errors = state.tracker.get_error_patterns(10).unwrap_or_default();

    let activity_last_hour = state
        .tracker
        .get_activity_type_counts_since(now - Duration::hours(1))
        .unwrap_or_default();

    let metric_names = [
        "processing_time",
        "batch_processed",
        "pr_created",
        "issues_fetched",
        "issues_matched",
        "issues_queued",
        "poll_cycle_duration_secs",
        "poll_sources",
        "active_processing",
        "pending_attempts",
        "total_attempts",
        "ready_retries_found",
        "ready_retries_executed_total",
        "ready_retries_failed_total",
        "ready_retry_executed",
        "ready_retry_failed",
        "pr_status_checks",
        "pr_status_merged",
        "pr_status_closed",
        "pr_status_errors",
        "regression_watches_created",
        "auto_resolved_on_merge",
        "cascade_triggered",
        "cascade_failed",
    ];
    let counts = state
        .tracker
        .get_metric_counts_since(&metric_names, now - Duration::hours(24))
        .unwrap_or_default();
    let mut metric_counts_last_24h: HashMap<String, i64> = HashMap::new();
    for metric_name in metric_names {
        metric_counts_last_24h.insert(
            metric_name.to_string(),
            counts.get(metric_name).copied().unwrap_or(0),
        );
    }

    let mut by_source: HashMap<String, SourceTelemetry> = HashMap::new();
    for (source, source_stats) in &stats.by_source {
        let processed = source_stats.success
            + source_stats.failed
            + source_stats.merged
            + source_stats.closed
            + source_stats.cannot_fix;
        let pending = source_stats.total.saturating_sub(processed);
        by_source.insert(
            source.clone(),
            SourceTelemetry {
                source: source.clone(),
                total: source_stats.total as i64,
                pending: pending as i64,
                success: source_stats.success as i64,
                failed: source_stats.failed as i64,
                merged: source_stats.merged as i64,
                closed: source_stats.closed as i64,
                cannot_fix: source_stats.cannot_fix as i64,
                retryable: 0,
                success_rate: 0.0,
            },
        );
    }
    for attempt in &retryable {
        let entry = by_source
            .entry(attempt.source.clone())
            .or_insert_with(|| SourceTelemetry {
                source: attempt.source.clone(),
                ..SourceTelemetry::default()
            });
        entry.retryable += 1;
    }
    let mut source_breakdown: Vec<SourceTelemetry> = by_source
        .into_values()
        .map(|mut s| {
            let processed = s.success + s.merged + s.failed + s.closed + s.cannot_fix;
            s.success_rate = if processed > 0 {
                ((s.success + s.merged) as f64 / processed as f64) * 100.0
            } else {
                0.0
            };
            s
        })
        .collect();
    source_breakdown.sort_by(|a, b| a.source.cmp(&b.source));

    let diagnostics = state.tracker.get_diagnostic_counts().ok();

    let twenty_four_h_ago_iso = (now - Duration::hours(24))
        .format("%Y-%m-%d %H:%M:%S")
        .to_string();
    let seven_days_ago_iso = (now - Duration::days(7))
        .format("%Y-%m-%d %H:%M:%S")
        .to_string();
    let agent_spawns_today = state
        .tracker
        .get_agent_spawn_count(&twenty_four_h_ago_iso)
        .unwrap_or(0);
    let agent_spawns_this_week = state
        .tracker
        .get_agent_spawn_count(&seven_days_ago_iso)
        .unwrap_or(0);

    let response = TelemetryOverviewResponse {
        generated_at: now.to_rfc3339(),
        uptime_secs: state.start_time.elapsed().as_secs(),
        windows,
        queue,
        processing_time,
        source_breakdown,
        top_errors,
        activity_last_hour,
        metric_counts_last_24h,
        diagnostics,
        pr_analytics,
        agent_spawns_today,
        agent_spawns_this_week,
    };

    Ok(Json(response))
}

async fn telemetry_timeseries_handler(
    _user: AuthUser,
    State(state): State<ApiState>,
    Query(query): Query<TelemetryTimeseriesQuery>,
) -> Result<Json<TelemetryTimeseriesResponse>, StatusCode> {
    let now = Utc::now();
    let (period_label, period_duration) = parse_telemetry_period(query.period.as_deref(), "week");
    let default_bucket_minutes = match period_label.as_str() {
        "hour" => 5,
        "day" => 15,
        "month" => 360,
        _ => 60,
    };

    let bucket_minutes = query
        .bucket_minutes
        .unwrap_or(default_bucket_minutes)
        .clamp(1, 24 * 60);
    let bucket_seconds = bucket_minutes * 60;

    let start = now - period_duration;
    let mut buckets: BTreeMap<i64, TelemetryTimeseriesPoint> = BTreeMap::new();

    let mut cursor = floor_to_bucket(start, bucket_minutes);
    let end = floor_to_bucket(now, bucket_minutes);
    while cursor <= end {
        buckets.insert(
            cursor.timestamp(),
            TelemetryTimeseriesPoint {
                bucket_start: cursor.to_rfc3339(),
                ..TelemetryTimeseriesPoint::default()
            },
        );
        cursor += Duration::seconds(bucket_seconds);
    }

    for attempt in get_attempt_records_since(&state.tracker, start) {
        let bucket = floor_to_bucket(attempt.attempted_at, bucket_minutes).timestamp();
        if let Some(point) = buckets.get_mut(&bucket) {
            point.total += 1;
            match attempt.status {
                FixAttemptStatus::Pending => point.pending += 1,
                FixAttemptStatus::Success => point.success += 1,
                FixAttemptStatus::Failed => point.failed += 1,
                FixAttemptStatus::Merged => point.merged += 1,
                FixAttemptStatus::Closed => point.closed += 1,
                FixAttemptStatus::CannotFix => point.cannot_fix += 1,
                FixAttemptStatus::Answered => {}
            }
        }
    }

    Ok(Json(TelemetryTimeseriesResponse {
        period: period_label,
        bucket_minutes,
        generated_at: now.to_rfc3339(),
        points: buckets.into_values().collect(),
    }))
}

async fn telemetry_pipeline_handler(
    _user: AuthUser,
    State(state): State<ApiState>,
    Query(query): Query<TelemetryPeriodQuery>,
) -> Result<Json<TelemetryPipelineResponse>, StatusCode> {
    let now = Utc::now();
    let (period_label, period_duration) = parse_telemetry_period(query.period.as_deref(), "week");
    let since = now - period_duration;
    let total_metric_names = [
        "issues_fetched",
        "issues_matched",
        "issues_queued",
        "batch_processed",
        "pr_created",
        "ready_retries_found",
        "ready_retries_executed_total",
        "ready_retries_failed_total",
        "pr_status_checks",
        "pr_status_merged",
        "pr_status_closed",
        "pr_status_errors",
        "regression_watches_created",
        "auto_resolved_on_merge",
        "cascade_triggered",
        "cascade_failed",
    ];
    let per_source_metric_names = [
        "issues_fetched",
        "issues_matched",
        "issues_queued",
        "batch_processed",
        "pr_created",
        "ready_retry_executed",
        "ready_retry_failed",
    ];

    let mut totals = TelemetryPipelineTotals::default();
    let mut per_source: HashMap<String, TelemetryPipelineSource> = HashMap::new();

    let sums = state
        .tracker
        .get_metric_sums_since(&total_metric_names, since)
        .unwrap_or_default();

    totals.fetched = sums.get("issues_fetched").copied().unwrap_or(0.0);
    totals.matched = sums.get("issues_matched").copied().unwrap_or(0.0);
    totals.queued = sums.get("issues_queued").copied().unwrap_or(0.0);
    totals.processed = sums.get("batch_processed").copied().unwrap_or(0.0);
    totals.pr_created = sums.get("pr_created").copied().unwrap_or(0.0);
    totals.retries_found = sums.get("ready_retries_found").copied().unwrap_or(0.0);
    totals.retries_executed = sums
        .get("ready_retries_executed_total")
        .copied()
        .unwrap_or(0.0);
    totals.retries_failed = sums
        .get("ready_retries_failed_total")
        .copied()
        .unwrap_or(0.0);
    totals.pr_status_checks = sums.get("pr_status_checks").copied().unwrap_or(0.0);
    totals.pr_status_merged = sums.get("pr_status_merged").copied().unwrap_or(0.0);
    totals.pr_status_closed = sums.get("pr_status_closed").copied().unwrap_or(0.0);
    totals.pr_status_errors = sums.get("pr_status_errors").copied().unwrap_or(0.0);
    totals.regression_watches_created = sums
        .get("regression_watches_created")
        .copied()
        .unwrap_or(0.0);
    totals.auto_resolved_on_merge = sums.get("auto_resolved_on_merge").copied().unwrap_or(0.0);
    totals.cascade_triggered = sums.get("cascade_triggered").copied().unwrap_or(0.0);
    totals.cascade_failed = sums.get("cascade_failed").copied().unwrap_or(0.0);

    let per_source_sums = state
        .tracker
        .get_metric_sums_by_source_since(&per_source_metric_names, since)
        .unwrap_or_default();
    for ((metric_name, source), value) in per_source_sums {
        let entry = per_source
            .entry(source.clone())
            .or_insert_with(|| TelemetryPipelineSource {
                source,
                ..TelemetryPipelineSource::default()
            });
        match metric_name.as_str() {
            "issues_fetched" => entry.fetched += value,
            "issues_matched" => entry.matched += value,
            "issues_queued" => entry.queued += value,
            "batch_processed" => entry.processed += value,
            "pr_created" => entry.pr_created += value,
            "ready_retry_executed" => entry.retries_executed += value,
            "ready_retry_failed" => entry.retries_failed += value,
            _ => {}
        }
    }

    let poll_cycle_duration = state
        .tracker
        .get_metrics("poll_cycle_duration_secs", Some(since), 50_000)
        .unwrap_or_default();
    let active_processing = state
        .tracker
        .get_metrics("active_processing", Some(since), 50_000)
        .unwrap_or_default();
    let pending_attempts = state
        .tracker
        .get_metrics("pending_attempts", Some(since), 50_000)
        .unwrap_or_default();
    let total_attempts = state
        .tracker
        .get_metrics("total_attempts", Some(since), 50_000)
        .unwrap_or_default();

    let conversion = TelemetryPipelineConversion {
        match_rate: ratio(totals.matched, totals.fetched),
        queue_rate: ratio(totals.queued, totals.matched),
        processing_rate: ratio(totals.processed, totals.queued),
        pr_yield_rate: ratio(totals.pr_created, totals.processed),
    };

    let cycle_summary = compute_processing_time_summary(poll_cycle_duration);
    let poll_load = TelemetryPollLoad {
        poll_cycles: cycle_summary.samples,
        avg_cycle_secs: cycle_summary.avg_secs,
        p95_cycle_secs: cycle_summary.p95_secs,
        active_avg: average_metric_value(&active_processing),
        active_max: max_metric_value(&active_processing),
        pending_avg: average_metric_value(&pending_attempts),
        pending_max: max_metric_value(&pending_attempts),
        total_latest: latest_metric_value(&total_attempts),
    };

    let mut per_source: Vec<TelemetryPipelineSource> = per_source
        .into_values()
        .map(|mut src| {
            src.match_rate = ratio(src.matched, src.fetched);
            src.queue_rate = ratio(src.queued, src.matched);
            src.processing_rate = ratio(src.processed, src.queued);
            src.pr_yield_rate = ratio(src.pr_created, src.processed);
            src
        })
        .collect();
    per_source.sort_by(|a, b| a.source.cmp(&b.source));

    Ok(Json(TelemetryPipelineResponse {
        generated_at: now.to_rfc3339(),
        period: period_label,
        totals,
        conversion,
        poll_load,
        per_source,
    }))
}

async fn telemetry_latency_handler(
    _user: AuthUser,
    State(state): State<ApiState>,
    Query(query): Query<TelemetryPeriodQuery>,
) -> Result<Json<TelemetryLatencyResponse>, StatusCode> {
    let now = Utc::now();
    let (period_label, period_duration) = parse_telemetry_period(query.period.as_deref(), "week");
    let since = now - period_duration;

    let processing_metrics = state
        .tracker
        .get_metrics("processing_time", Some(since), 50_000)
        .unwrap_or_default();
    let overall = compute_processing_time_summary(processing_metrics.clone());

    let all_values: Vec<f64> = processing_metrics.iter().map(|m| m.metric_value).collect();

    let mut by_status_values: HashMap<String, Vec<f64>> = HashMap::new();
    for metric in processing_metrics {
        let status = metric
            .tags
            .as_ref()
            .and_then(|tags| tags.get("status"))
            .and_then(|value| value.as_str())
            .unwrap_or("unknown")
            .to_string();
        by_status_values
            .entry(status)
            .or_default()
            .push(metric.metric_value);
    }

    let mut by_status: Vec<TelemetryLatencyByStatus> = by_status_values
        .into_iter()
        .map(|(status, values)| TelemetryLatencyByStatus {
            status,
            summary: compute_processing_value_summary(values),
        })
        .collect();
    by_status.sort_by(|a, b| {
        b.summary
            .samples
            .cmp(&a.summary.samples)
            .then_with(|| a.status.cmp(&b.status))
    });

    let bucket_defs: [(f64, &str); 5] = [
        (15.0, "<=15s"),
        (30.0, "<=30s"),
        (60.0, "<=60s"),
        (120.0, "<=2m"),
        (300.0, "<=5m"),
    ];
    let mut counts = vec![0i64; bucket_defs.len() + 1];
    for value in all_values {
        let mut placed = false;
        for (idx, (upper, _label)) in bucket_defs.iter().enumerate() {
            if value <= *upper {
                counts[idx] += 1;
                placed = true;
                break;
            }
        }
        if !placed {
            counts[bucket_defs.len()] += 1;
        }
    }

    let mut histogram = Vec::with_capacity(bucket_defs.len() + 1);
    for (idx, (upper, label)) in bucket_defs.iter().enumerate() {
        histogram.push(TelemetryLatencyHistogramBucket {
            label: (*label).to_string(),
            upper_bound_secs: Some(*upper),
            count: counts[idx],
        });
    }
    histogram.push(TelemetryLatencyHistogramBucket {
        label: ">5m".to_string(),
        upper_bound_secs: None,
        count: counts[bucket_defs.len()],
    });

    Ok(Json(TelemetryLatencyResponse {
        generated_at: now.to_rfc3339(),
        period: period_label,
        overall,
        by_status,
        histogram,
    }))
}

#[derive(Serialize)]
struct ConfigResponse {
    content: String,
}

#[derive(Deserialize)]
struct ConfigUpdateRequest {
    content: String,
}

/// Redact values of keys that look like secrets in raw TOML content.
fn redact_secrets(content: &str) -> String {
    let re = regex_lite::Regex::new(
        r#"(?im)^(\s*(?:[a-z_]*(?:token|secret|password|api_key|auth)[a-z_]*)\s*=\s*)("[^"]*"|'[^']*'|[^\n]*)"#,
    )
    .expect("valid regex");
    re.replace_all(content, r#"${1}"[REDACTED]""#).into_owned()
}

/// GET /api/config — return the raw TOML config file content with secrets redacted.
async fn get_config_handler(
    _user: AdminUser,
    State(state): State<ApiState>,
) -> Result<Json<ConfigResponse>, (StatusCode, Json<serde_json::Value>)> {
    let content = tokio::fs::read_to_string(&state.config_path)
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": format!("Failed to read config: {}", e) })),
            )
        })?;

    Ok(Json(ConfigResponse {
        content: redact_secrets(&content),
    }))
}

/// PUT /api/config — validate and write the TOML config file.
async fn put_config_handler(
    _user: AdminUser,
    State(state): State<ApiState>,
    Json(body): Json<ConfigUpdateRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    if !check_api_rate_limit(_user.0.id) {
        return Err((
            StatusCode::TOO_MANY_REQUESTS,
            Json(serde_json::json!({ "error": "Rate limit exceeded" })),
        ));
    }

    let parsed = toml::from_str::<Config>(&body.content).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": format!("Invalid TOML config: {}", e) })),
        )
    })?;

    parsed.validate().map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": format!("Config validation failed: {}", e) })),
        )
    })?;

    // Re-serialize the validated config to ensure only parsed fields are written to disk.
    // This prevents injection of arbitrary content via raw user input.
    let re_serialized = toml::to_string_pretty(&parsed).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": format!("Failed to serialize config: {}", e) })),
        )
    })?;

    tokio::fs::write(&state.config_path, &re_serialized)
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": format!("Failed to write config: {}", e) })),
            )
        })?;

    Ok(Json(
        serde_json::json!({ "ok": true, "message": "Config saved. Restart to apply changes." }),
    ))
}

/// Browse GGUF models from HuggingFace.
async fn browse_models_handler(
    _user: AuthUser,
    State(state): State<ApiState>,
    Query(query): Query<claudear_integrations::chat::models::types::ListModelsQuery>,
) -> Result<Json<claudear_integrations::chat::models::types::BrowseResponse>, StatusCode> {
    // Chat service not required — this queries HuggingFace directly
    let _ = &state;

    let search = query.search.as_deref().unwrap_or("");
    let cursor = query.cursor.as_deref();
    let limit = query.limit;

    claudear_integrations::chat::models::HuggingFaceProvider::search(search, cursor, limit)
        .await
        .map(Json)
        .map_err(|e| {
            tracing::error!(error = %e, "Failed to browse models from HuggingFace");
            StatusCode::BAD_GATEWAY
        })
}

/// Get info about a specific HuggingFace model.
async fn model_info_handler(
    _user: AuthUser,
    State(_state): State<ApiState>,
    Path(name): Path<String>,
) -> Result<Json<claudear_integrations::chat::models::types::ModelInfoResponse>, StatusCode> {
    // Fetch model info from HuggingFace API
    let url = format!("https://huggingface.co/api/models/{}", name);
    let response = claudear_integrations::chat::models::HTTP_CLIENT
        .get(&url)
        .send()
        .await
        .map_err(|e| {
            tracing::error!(error = %e, "Failed to fetch model info");
            StatusCode::BAD_GATEWAY
        })?
        .error_for_status()
        .map_err(|e| {
            tracing::error!(error = %e, "HuggingFace API error");
            if e.status() == Some(reqwest::StatusCode::NOT_FOUND) {
                StatusCode::NOT_FOUND
            } else {
                StatusCode::BAD_GATEWAY
            }
        })?;

    #[derive(serde::Deserialize)]
    struct HfModel {
        #[serde(alias = "id")]
        model_id: String,
        #[serde(default)]
        siblings: Option<Vec<HfSibling>>,
    }

    #[derive(serde::Deserialize)]
    struct HfSibling {
        rfilename: String,
        #[serde(default)]
        size: Option<u64>,
    }

    let model: HfModel = response.json().await.map_err(|e| {
        tracing::error!(error = %e, "Failed to parse model info");
        StatusCode::BAD_GATEWAY
    })?;

    let gguf_size = model.siblings.as_ref().and_then(|s| {
        s.iter()
            .find(|f| f.rfilename.ends_with(".gguf"))
            .and_then(|f| f.size)
    });

    let details = Some(claudear_integrations::chat::models::types::ModelDetails {
        format: Some("gguf".to_string()),
        family: claudear_integrations::chat::models::providers::extract_model_family(
            &model.model_id,
        ),
        parameter_size: claudear_integrations::chat::models::providers::extract_param_size(
            &model.model_id,
        ),
        quantization_level: model
            .siblings
            .as_ref()
            .and_then(|s| {
                s.iter()
                    .find(|f| f.rfilename.ends_with(".gguf"))
                    .map(|f| &f.rfilename)
            })
            .and_then(|name| {
                claudear_integrations::chat::models::providers::extract_quantization(name)
            }),
    });

    Ok(Json(
        claudear_integrations::chat::models::types::ModelInfoResponse {
            name: model.model_id,
            gguf_size,
            details,
        },
    ))
}

/// Download a GGUF model (admin only).
async fn download_model_handler(
    _admin: AdminUser,
    State(state): State<ApiState>,
    Json(body): Json<claudear_integrations::chat::models::types::DownloadRequest>,
) -> Result<StatusCode, (StatusCode, Json<serde_json::Value>)> {
    let chat_service = state.chat_service.as_ref().ok_or_else(|| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({ "error": "Chat feature is not enabled" })),
        )
    })?;

    // Check if model already exists on disk
    if chat_service.is_model_available() && body.url.is_none() {
        return Err((
            StatusCode::CONFLICT,
            Json(serde_json::json!({ "error": "Model already exists on disk" })),
        ));
    }

    let url = body
        .url
        .as_deref()
        .unwrap_or_else(|| chat_service.model_url());

    chat_service.start_download(url).map_err(|e| {
        (
            StatusCode::CONFLICT,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
    })?;

    Ok(StatusCode::ACCEPTED)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use claudear_config::config::{
        AgentConfig, AskConfig, CascadeConfig, CodeIndexConfig, IssuesConfig, LearningConfig,
        NotifiersConfig, PrioritisationConfig, RegressionConfig, RetryConfig, ScmConfig,
    };
    use claudear_core::secret::SecretValue;
    use claudear_storage::{IndexingProgress, SqliteTracker};
    use http_body_util::BodyExt;
    use tower::ServiceExt;
    use tower_cookies::CookieManagerLayer;

    fn test_config() -> Config {
        Config {
            workspace: "/tmp/repos".into(),
            known_orgs: vec!["test-org".to_string()],
            auto_discover_paths: vec![],
            poll_interval_ms: 300_000,
            webhook_port: 3100,
            bind_address: "127.0.0.1".to_string(),
            db_path: ":memory:".into(),
            max_issues_per_cycle: 5,
            max_concurrent: 1,
            processing_delay_ms: 5000,
            max_activity_entries: 100,
            ipc_timeout_secs: 30,
            agent: AgentConfig::default(),
            scm: ScmConfig::default(),
            issues: IssuesConfig::default(),
            notifiers: NotifiersConfig::default(),
            ask: AskConfig::default(),
            retry: RetryConfig::default(),
            regression: RegressionConfig::default(),
            cascade: CascadeConfig::default(),
            users: std::collections::HashMap::new(),
            learning: LearningConfig::default(),
            prioritisation: PrioritisationConfig::default(),
            code_index: CodeIndexConfig::default(),
            evaluation: claudear_config::config::EvaluationConfig::default(),
            storage_dir: "/tmp/claudear-storage".into(),
            dashboard: claudear_config::config::DashboardConfig::default(),
            llm: claudear_config::config::LlmModelConfig::default(),
            chat: claudear_config::config::ChatConfig::default(),
            tls: claudear_config::config::TlsConfig::default(),
            embedding: claudear_config::config::EmbeddingModelConfig::default(),
            qa: claudear_config::config::QaConfig::default(),
        }
    }

    fn create_test_tracker() -> Arc<dyn FixAttemptTracker> {
        Arc::new(SqliteTracker::in_memory().unwrap())
    }

    fn test_indexing_rx(
        tracker: &Arc<dyn FixAttemptTracker>,
    ) -> tokio::sync::watch::Receiver<IndexingProgress> {
        tracker.subscribe_indexing_progress()
    }

    /// Create an authenticated test router with CookieManagerLayer and a session cookie.
    /// Returns (router, session_cookie_value).
    fn create_authenticated_router(tracker: &Arc<dyn FixAttemptTracker>) -> (Router, String) {
        let config = test_config();

        // Create a test user and session
        let password_hash = bcrypt::hash("testpass", 4).unwrap(); // cost=4 for speed
        tracker
            .create_user("test@example.com", &password_hash, "Test User", "admin")
            .unwrap();
        let token = tracker.create_session(1, "2099-12-31 23:59:59").unwrap();

        let indexing_rx = test_indexing_rx(tracker);
        let router = create_api_router(
            config,
            tracker.clone(),
            PathBuf::from("claudear.toml"),
            indexing_rx,
        )
        .layer(CookieManagerLayer::new());

        (router, token)
    }

    /// Build an authenticated GET request with the session cookie.
    fn auth_get(uri: &str, token: &str) -> Request<Body> {
        Request::builder()
            .uri(uri)
            .header("cookie", format!("claudear_session={}", token))
            .body(Body::empty())
            .unwrap()
    }

    fn auth_post_json(uri: &str, token: &str, body: serde_json::Value) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri(uri)
            .header("content-type", "application/json")
            .header("cookie", format!("claudear_session={}", token))
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    fn auth_put_json(uri: &str, token: &str, body: serde_json::Value) -> Request<Body> {
        Request::builder()
            .method("PUT")
            .uri(uri)
            .header("content-type", "application/json")
            .header("cookie", format!("claudear_session={}", token))
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    #[tokio::test]
    async fn test_health_endpoint() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/health", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_stats_endpoint() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/stats", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_overview_endpoint() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/stats/overview", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_attempts_endpoint() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/attempts", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_attempts_with_pagination() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/attempts?page=1&per_page=10", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_attempts_with_filter() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get(
                "/api/attempts?status=success&source=linear",
                &token,
            ))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_attempt_detail_not_found() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/attempts/99999", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_sources_endpoint() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/sources", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_retries_endpoint() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/retries", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_telemetry_overview_endpoint() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/telemetry/overview", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_telemetry_timeseries_endpoint() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get(
                "/api/telemetry/timeseries?period=day&bucket_minutes=30",
                &token,
            ))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_telemetry_pipeline_endpoint() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/telemetry/pipeline?period=day", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_telemetry_latency_endpoint() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/telemetry/latency?period=week", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[test]
    fn test_attempt_to_summary() {
        let attempt = FixAttempt {
            id: 1,
            source: "linear".to_string(),
            issue_id: "123".to_string(),
            short_id: "PROJ-123".to_string(),
            status: FixAttemptStatus::Success,
            pr_url: Some("https://github.com/org/repo/pull/1".to_string()),
            scm_repo: None,
            scm_pr_number: None,
            error_message: None,
            attempted_at: chrono::Utc::now(),
            resolved_at: None,
            merged_at: None,
            retry_count: 0,
            last_retry_at: None,
            issue_labels: vec![],
            parent_attempt_id: None,
            cascade_repo: None,
        };

        let summary = attempt_to_summary(&attempt);
        assert_eq!(summary.id, 1);
        assert_eq!(summary.source, "linear");
        assert_eq!(summary.short_id, "PROJ-123");
        assert_eq!(summary.status, "success");
        assert!(summary.pr_url.is_some());
    }

    #[test]
    fn test_health_response_serialization() {
        let response = HealthResponse {
            status: "ok".to_string(),
            version: "1.0.0".to_string(),
            uptime_secs: 3600,
            database: DatabaseStatus {
                status: "ok".to_string(),
                error: None,
            },
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("ok"));
        assert!(json.contains("1.0.0"));
        assert!(json.contains("3600"));
        assert!(json.contains("database"));
    }

    #[test]
    fn test_health_response_with_database_error() {
        let response = HealthResponse {
            status: "degraded".to_string(),
            version: "1.0.0".to_string(),
            uptime_secs: 100,
            database: DatabaseStatus {
                status: "error".to_string(),
                error: Some("Connection failed".to_string()),
            },
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("degraded"));
        assert!(json.contains("error"));
        assert!(json.contains("Connection failed"));
    }

    #[test]
    fn test_attempts_response_serialization() {
        let response = AttemptsResponse {
            attempts: vec![],
            total: 0,
            page: 1,
            per_page: 20,
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"total\":0"));
        assert!(json.contains("\"page\":1"));
    }

    #[test]
    fn test_source_summary_serialization() {
        let summary = SourceSummary {
            name: "linear".to_string(),
            total: 100,
            success: 80,
            failed: 10,
            merged: 70,
            success_rate: 80.0,
        };
        let json = serde_json::to_string(&summary).unwrap();
        assert!(json.contains("linear"));
        assert!(json.contains("100"));
        assert!(json.contains("80.0"));
    }

    #[test]
    fn test_source_info_serialization() {
        let info = SourceInfo {
            name: "sentry".to_string(),
            enabled: true,
            config: serde_json::json!({"key": "value"}),
        };
        let json = serde_json::to_string(&info).unwrap();
        assert!(json.contains("sentry"));
        assert!(json.contains("true"));
        assert!(json.contains("value"));
    }

    #[test]
    fn test_overview_response_serialization() {
        let overview = OverviewResponse {
            stats: FixAttemptStats::default(),
            success_rate: 85.5,
            merge_rate: 75.0,
            recent_attempts: vec![],
            sources: vec![],
            time_savings: None,
            agent_spawns_today: 0,
        };
        let json = serde_json::to_string(&overview).unwrap();
        assert!(json.contains("85.5"));
        assert!(json.contains("stats"));
    }

    #[test]
    fn test_retries_response_serialization() {
        let retries = RetriesResponse {
            retryable: vec![],
            ready: vec![],
            max_retries: 3,
        };
        let json = serde_json::to_string(&retries).unwrap();
        assert!(json.contains("retryable"));
        assert!(json.contains("3"));
    }

    #[test]
    fn test_sources_response_serialization() {
        let sources = SourcesResponse {
            sources: vec![SourceInfo {
                name: "linear".to_string(),
                enabled: true,
                config: serde_json::json!({}),
            }],
        };
        let json = serde_json::to_string(&sources).unwrap();
        assert!(json.contains("linear"));
    }

    #[test]
    fn test_attempt_summary_serialization() {
        let summary = AttemptSummary {
            id: 1,
            source: "linear".to_string(),
            short_id: "PROJ-1".to_string(),
            title: "Fix bug".to_string(),
            status: "success".to_string(),
            pr_url: Some("https://github.com/org/repo/pull/1".to_string()),
            attempted_at: "2024-01-01T00:00:00Z".to_string(),
            retry_count: 0,
        };
        let json = serde_json::to_string(&summary).unwrap();
        assert!(json.contains("linear"));
        assert!(json.contains("PROJ-1"));
        assert!(json.contains("success"));
    }

    #[test]
    fn test_attempts_query_deserialization() {
        let query: AttemptsQuery = serde_json::from_str(
            r#"{
            "page": 2,
            "per_page": 50,
            "status": "success",
            "source": "linear"
        }"#,
        )
        .unwrap();
        assert_eq!(query.page, Some(2));
        assert_eq!(query.per_page, Some(50));
        assert_eq!(query.status, Some("success".to_string()));
        assert_eq!(query.source, Some("linear".to_string()));
    }

    #[test]
    fn test_attempts_query_defaults() {
        let query: AttemptsQuery = serde_json::from_str("{}").unwrap();
        assert!(query.page.is_none());
        assert!(query.per_page.is_none());
        assert!(query.status.is_none());
        assert!(query.source.is_none());
    }

    #[tokio::test]
    async fn test_404_for_unknown_endpoint() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/unknown", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_health_response_content() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/health", &token))
            .await
            .unwrap();

        let body = response.into_body().collect().await.unwrap().to_bytes();
        let body_str = String::from_utf8(body.to_vec()).unwrap();
        assert!(body_str.contains("\"status\":\"ok\""));
        assert!(body_str.contains("version"));
        assert!(body_str.contains("uptime_secs"));
        assert!(body_str.contains("database"));
    }

    #[tokio::test]
    async fn test_stats_response_content() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/stats", &token))
            .await
            .unwrap();

        let body = response.into_body().collect().await.unwrap().to_bytes();
        let stats: FixAttemptStats = serde_json::from_slice(&body).unwrap();
        assert_eq!(stats.total, 0);
    }

    #[tokio::test]
    async fn test_attempts_response_content() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/attempts", &token))
            .await
            .unwrap();

        let body = response.into_body().collect().await.unwrap().to_bytes();
        let body_str = String::from_utf8(body.to_vec()).unwrap();
        assert!(body_str.contains("\"page\":1"));
        assert!(body_str.contains("\"per_page\":20"));
    }

    #[tokio::test]
    async fn test_attempts_pagination_limits() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        // Test that per_page is capped at 100
        let response = router
            .oneshot(auth_get("/api/attempts?per_page=200", &token))
            .await
            .unwrap();

        let body = response.into_body().collect().await.unwrap().to_bytes();
        let body_str = String::from_utf8(body.to_vec()).unwrap();
        assert!(body_str.contains("\"per_page\":100")); // Should be capped
    }

    #[test]
    fn test_attempt_to_summary_without_pr_url() {
        let attempt = FixAttempt {
            id: 2,
            source: "sentry".to_string(),
            issue_id: "456".to_string(),
            short_id: "SENTRY-456".to_string(),
            status: FixAttemptStatus::Failed,
            pr_url: None,
            scm_repo: None,
            scm_pr_number: None,
            error_message: Some("Error message".to_string()),
            attempted_at: chrono::Utc::now(),
            resolved_at: None,
            merged_at: None,
            retry_count: 2,
            last_retry_at: Some(chrono::Utc::now()),
            issue_labels: vec![],
            parent_attempt_id: None,
            cascade_repo: None,
        };

        let summary = attempt_to_summary(&attempt);
        assert_eq!(summary.id, 2);
        assert_eq!(summary.source, "sentry");
        assert!(summary.pr_url.is_none());
        assert_eq!(summary.retry_count, 2);
        assert_eq!(summary.status, "failed");
    }

    #[tokio::test]
    async fn test_unauthenticated_returns_401() {
        let tracker = create_test_tracker();
        let config = test_config();
        let indexing_rx = test_indexing_rx(&tracker);
        let router =
            create_api_router(config, tracker, PathBuf::from("claudear.toml"), indexing_rx)
                .layer(CookieManagerLayer::new());

        let response = router
            .oneshot(
                Request::builder()
                    .uri("/api/stats")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[test]
    fn test_source_summary_zero_values() {
        let summary = SourceSummary {
            name: "empty".to_string(),
            total: 0,
            success: 0,
            failed: 0,
            merged: 0,
            success_rate: 0.0,
        };
        let json = serde_json::to_string(&summary).unwrap();
        assert!(json.contains("empty"));
        assert!(json.contains("0"));
    }

    #[tokio::test]
    async fn test_activity_endpoint_empty() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/activity", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let entries: Vec<serde_json::Value> = serde_json::from_slice(&body).unwrap();
        assert!(entries.is_empty());
    }

    #[tokio::test]
    async fn test_activity_endpoint_with_limit() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/activity?limit=5", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_activity_endpoint_with_source_filter() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/activity?source=linear", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_analytics_summary_endpoint_empty() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/analytics/summary", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let summary: claudear_core::types::AnalyticsSummary =
            serde_json::from_slice(&body).unwrap();
        assert_eq!(summary.total_processed, 0);
        assert_eq!(summary.total_successful, 0);
        assert_eq!(summary.total_merged, 0);
    }

    #[tokio::test]
    async fn test_metrics_endpoint_empty() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/metrics", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let metrics: Vec<serde_json::Value> = serde_json::from_slice(&body).unwrap();
        assert!(metrics.is_empty());
    }

    #[tokio::test]
    async fn test_metrics_endpoint_with_name_filter() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get(
                "/api/metrics?name=processing_time&period=day&limit=10",
                &token,
            ))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_errors_endpoint_empty() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/errors", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let errors: Vec<serde_json::Value> = serde_json::from_slice(&body).unwrap();
        assert!(errors.is_empty());
    }

    #[tokio::test]
    async fn test_errors_endpoint_with_limit() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/errors?limit=10", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_prs_endpoint_empty() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router.oneshot(auth_get("/api/prs", &token)).await.unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let prs: Vec<serde_json::Value> = serde_json::from_slice(&body).unwrap();
        assert!(prs.is_empty());
    }

    #[tokio::test]
    async fn test_prs_endpoint_with_status_filter() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/prs?status=open&limit=5", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_pr_analytics_endpoint_empty() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/prs/analytics", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let analytics: claudear_core::types::PrAnalytics = serde_json::from_slice(&body).unwrap();
        assert_eq!(analytics.open, 0);
    }

    #[tokio::test]
    async fn test_feedback_endpoint_empty() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/feedback", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let feedback: Vec<serde_json::Value> = serde_json::from_slice(&body).unwrap();
        assert!(feedback.is_empty());
    }

    #[tokio::test]
    async fn test_feedback_endpoint_with_filters() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/feedback?source=linear&limit=10", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_regressions_endpoint_empty() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/regressions", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let regressions: Vec<serde_json::Value> = serde_json::from_slice(&body).unwrap();
        assert!(regressions.is_empty());
    }

    #[tokio::test]
    async fn test_regressions_endpoint_with_status_filter() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/regressions?status=awaiting_release", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_regressions_endpoint_invalid_status() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/regressions?status=invalid_status", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_regression_checks_endpoint_empty() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/regressions/1/checks", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let checks: Vec<serde_json::Value> = serde_json::from_slice(&body).unwrap();
        assert!(checks.is_empty());
    }

    #[tokio::test]
    async fn test_experiments_endpoint_empty() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/experiments", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let experiments: Vec<serde_json::Value> = serde_json::from_slice(&body).unwrap();
        assert!(experiments.is_empty());
    }

    #[tokio::test]
    async fn test_create_experiment_endpoint() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_post_json(
                "/api/experiments",
                &token,
                serde_json::json!({
                    "experiment_name": "prompt-ab",
                    "variant": "control",
                    "prompt_template": "Fix issue: {{issue}}",
                    "active": true
                }),
            ))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::CREATED);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let experiment: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(experiment["experiment_name"], "prompt-ab");
        assert_eq!(experiment["variant"], "control");
        assert_eq!(experiment["prompt_template"], "Fix issue: {{issue}}");
        assert_eq!(experiment["active"], true);
        assert!(experiment["id"].as_i64().unwrap() > 0);
        assert_eq!(experiment["success_count"], 0);
        assert_eq!(experiment["failure_count"], 0);
    }

    #[tokio::test]
    async fn test_create_experiment_endpoint_invalid_payload() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_post_json(
                "/api/experiments",
                &token,
                serde_json::json!({
                    "experiment_name": "  ",
                    "variant": "control",
                    "prompt_template": ""
                }),
            ))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_update_experiment_endpoint() {
        let tracker = create_test_tracker();
        let exp_id = tracker
            .save_experiment(&claudear_core::types::PromptExperiment::new(
                "prompt-ab",
                "control",
                "Fix issue: {{issue}}",
                "oldhash",
            ))
            .unwrap();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_put_json(
                &format!("/api/experiments/{exp_id}"),
                &token,
                serde_json::json!({
                    "experiment_name": "prompt-ab-v2",
                    "variant": "variant-a",
                    "prompt_template": "Fix issue carefully: {{issue}}",
                    "active": true
                }),
            ))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let experiments = tracker.get_active_experiments().unwrap();
        assert_eq!(experiments.len(), 1);
        assert_eq!(experiments[0].id, exp_id);
        assert_eq!(experiments[0].experiment_name, "prompt-ab-v2");
        assert_eq!(experiments[0].variant, "variant-a");
        assert_eq!(
            experiments[0].prompt_template,
            "Fix issue carefully: {{issue}}"
        );
        assert_ne!(experiments[0].prompt_hash, "oldhash");
    }

    #[tokio::test]
    async fn test_update_experiment_endpoint_can_deactivate() {
        let tracker = create_test_tracker();
        let exp_id = tracker
            .save_experiment(&claudear_core::types::PromptExperiment::new(
                "prompt-ab",
                "control",
                "Fix issue: {{issue}}",
                "oldhash",
            ))
            .unwrap();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_put_json(
                &format!("/api/experiments/{exp_id}"),
                &token,
                serde_json::json!({
                    "experiment_name": "prompt-ab",
                    "variant": "control",
                    "prompt_template": "Fix issue: {{issue}}",
                    "active": false
                }),
            ))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert!(tracker.get_active_experiments().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_repos_endpoint_empty() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/repos", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let repos: Vec<serde_json::Value> = serde_json::from_slice(&body).unwrap();
        assert!(repos.is_empty());
    }

    #[tokio::test]
    async fn test_repo_stats_endpoint() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/repos/stats", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_dependencies_endpoint_empty() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/repos/dependencies", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let deps: Vec<serde_json::Value> = serde_json::from_slice(&body).unwrap();
        assert!(deps.is_empty());
    }

    #[tokio::test]
    async fn test_inference_stats_endpoint() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/inference/stats", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_inference_history_endpoint_empty() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/inference/history", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let history: Vec<serde_json::Value> = serde_json::from_slice(&body).unwrap();
        assert!(history.is_empty());
    }

    #[tokio::test]
    async fn test_inference_history_endpoint_with_limit() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/inference/history?limit=10", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_get_config_endpoint_admin() {
        let tracker = create_test_tracker();
        let config = test_config();

        // Use a nonexistent config path to test the error case
        let password_hash = bcrypt::hash("testpass", 4).unwrap();
        tracker
            .create_user("admin@test.com", &password_hash, "Admin", "admin")
            .unwrap();
        let token = tracker.create_session(1, "2099-12-31 23:59:59").unwrap();

        let indexing_rx = test_indexing_rx(&tracker);
        let router = create_api_router(
            config,
            tracker.clone(),
            PathBuf::from("/tmp/nonexistent_claudear_test_config.toml"),
            indexing_rx,
        )
        .layer(CookieManagerLayer::new());

        let response = router
            .oneshot(auth_get("/api/config", &token))
            .await
            .unwrap();

        // Config file doesn't exist on disk, so expect 500
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[tokio::test]
    async fn test_get_config_endpoint_admin_success() {
        let tracker = create_test_tracker();
        let config = test_config();

        let password_hash = bcrypt::hash("testpass", 4).unwrap();
        tracker
            .create_user("admin@test.com", &password_hash, "Admin", "admin")
            .unwrap();
        let token = tracker.create_session(1, "2099-12-31 23:59:59").unwrap();

        // Write a temp config file
        let config_path = std::env::temp_dir().join("claudear_test_config.toml");
        std::fs::write(&config_path, "# test config\nworkspace = \"/tmp\"\n").unwrap();

        let indexing_rx = test_indexing_rx(&tracker);
        let router = create_api_router(config, tracker.clone(), config_path.clone(), indexing_rx)
            .layer(CookieManagerLayer::new());

        let response = router
            .oneshot(auth_get("/api/config", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let resp: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(resp["content"].as_str().unwrap().contains("workspace"));
        assert!(resp["path"].is_null(), "path field should not be exposed");

        // Cleanup
        let _ = std::fs::remove_file(&config_path);
    }

    #[tokio::test]
    async fn test_get_config_endpoint_viewer_forbidden() {
        let tracker = create_test_tracker();
        let config = test_config();
        let password_hash = bcrypt::hash("testpass", 4).unwrap();
        tracker
            .create_user("viewer@test.com", &password_hash, "Viewer", "viewer")
            .unwrap();
        let token = tracker.create_session(1, "2099-12-31 23:59:59").unwrap();

        let indexing_rx = test_indexing_rx(&tracker);
        let router = create_api_router(
            config,
            tracker.clone(),
            PathBuf::from("claudear.toml"),
            indexing_rx,
        )
        .layer(CookieManagerLayer::new());

        let response = router
            .oneshot(auth_get("/api/config", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn test_put_config_endpoint_invalid_toml() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let request = Request::builder()
            .method("PUT")
            .uri("/api/config")
            .header("cookie", format!("claudear_session={}", token))
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::to_string(&serde_json::json!({
                    "content": "this is not valid toml [[[["
                }))
                .unwrap(),
            ))
            .unwrap();

        let response = router.oneshot(request).await.unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_put_config_endpoint_viewer_forbidden() {
        let tracker = create_test_tracker();
        let config = test_config();
        let password_hash = bcrypt::hash("testpass", 4).unwrap();
        tracker
            .create_user("viewer@test.com", &password_hash, "Viewer", "viewer")
            .unwrap();
        let token = tracker.create_session(1, "2099-12-31 23:59:59").unwrap();

        let indexing_rx = test_indexing_rx(&tracker);
        let router = create_api_router(
            config,
            tracker.clone(),
            PathBuf::from("claudear.toml"),
            indexing_rx,
        )
        .layer(CookieManagerLayer::new());

        let request = Request::builder()
            .method("PUT")
            .uri("/api/config")
            .header("cookie", format!("claudear_session={}", token))
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::to_string(&serde_json::json!({
                    "content": "key = \"value\""
                }))
                .unwrap(),
            ))
            .unwrap();

        let response = router.oneshot(request).await.unwrap();

        assert_eq!(response.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn test_attempt_full_detail_not_found() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/attempts/99999/detail", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_attempt_execution_log_not_found() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        // Attempt doesn't exist
        let response = router
            .oneshot(auth_get("/api/attempts/99999/logs/1/stdout", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_attempt_execution_log_invalid_stream() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/attempts/1/logs/1/invalid_stream", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    /// Helper to seed a fix attempt and return the tracker.
    fn seed_attempt(
        tracker: &Arc<dyn FixAttemptTracker>,
        source: &str,
        issue_id: &str,
        short_id: &str,
    ) {
        tracker.record_attempt(source, issue_id, short_id).unwrap();
    }

    #[tokio::test]
    async fn test_stats_with_seeded_data() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        seed_attempt(&tracker, "linear", "issue-1", "PROJ-1");
        seed_attempt(&tracker, "linear", "issue-2", "PROJ-2");
        tracker
            .mark_success("linear", "issue-1", "https://github.com/org/repo/pull/1")
            .unwrap();

        let response = router
            .oneshot(auth_get("/api/stats", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let stats: FixAttemptStats = serde_json::from_slice(&body).unwrap();
        assert_eq!(stats.total, 2);
        assert_eq!(stats.success, 1);
        assert_eq!(stats.pending, 1);
    }

    #[tokio::test]
    async fn test_overview_with_seeded_data() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        seed_attempt(&tracker, "linear", "issue-1", "PROJ-1");
        tracker
            .mark_success("linear", "issue-1", "https://github.com/org/repo/pull/1")
            .unwrap();
        tracker.mark_merged("linear", "issue-1").unwrap();

        seed_attempt(&tracker, "sentry", "issue-2", "SENTRY-1");
        tracker
            .mark_failed("sentry", "issue-2", "Build failed")
            .unwrap();

        let response = router
            .oneshot(auth_get("/api/stats/overview", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let body_str = String::from_utf8(body.to_vec()).unwrap();
        assert!(body_str.contains("success_rate"));
        assert!(body_str.contains("merge_rate"));
        assert!(body_str.contains("recent_attempts"));
        assert!(body_str.contains("sources"));
    }

    #[tokio::test]
    async fn test_attempts_with_seeded_data() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        seed_attempt(&tracker, "linear", "issue-1", "PROJ-1");
        seed_attempt(&tracker, "sentry", "issue-2", "SENTRY-1");
        tracker
            .mark_success("linear", "issue-1", "https://github.com/org/repo/pull/1")
            .unwrap();

        let response = router
            .oneshot(auth_get("/api/attempts", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let body_str = String::from_utf8(body.to_vec()).unwrap();
        assert!(body_str.contains("\"total\":2"));
    }

    #[tokio::test]
    async fn test_attempts_status_filter_with_data() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        seed_attempt(&tracker, "linear", "issue-1", "PROJ-1");
        seed_attempt(&tracker, "linear", "issue-2", "PROJ-2");
        tracker
            .mark_success("linear", "issue-1", "https://github.com/org/repo/pull/1")
            .unwrap();

        let response = router
            .oneshot(auth_get("/api/attempts?status=success", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let body_str = String::from_utf8(body.to_vec()).unwrap();
        assert!(body_str.contains("\"total\":1"));
    }

    #[tokio::test]
    async fn test_attempts_source_filter_with_data() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        seed_attempt(&tracker, "linear", "issue-1", "PROJ-1");
        seed_attempt(&tracker, "sentry", "issue-2", "SENTRY-1");

        let response = router
            .oneshot(auth_get("/api/attempts?source=sentry", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let body_str = String::from_utf8(body.to_vec()).unwrap();
        assert!(body_str.contains("\"total\":1"));
    }

    #[tokio::test]
    async fn test_attempt_detail_with_data() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        seed_attempt(&tracker, "linear", "issue-1", "PROJ-1");

        let response = router
            .oneshot(auth_get("/api/attempts/1", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let attempt: FixAttempt = serde_json::from_slice(&body).unwrap();
        assert_eq!(attempt.source, "linear");
        assert_eq!(attempt.short_id, "PROJ-1");
        assert_eq!(attempt.status, FixAttemptStatus::Pending);
    }

    #[tokio::test]
    async fn test_attempt_full_detail_with_data() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        seed_attempt(&tracker, "linear", "issue-1", "PROJ-1");
        tracker
            .mark_success("linear", "issue-1", "https://github.com/org/repo/pull/1")
            .unwrap();

        // Use attempt ID 1 (assuming it's the first one recorded).
        // Note: the ID might be 2 if the user creation in create_authenticated_router
        // interferes, but record_attempt in the fix_attempts table is separate.
        let response = router
            .oneshot(auth_get("/api/attempts/1/detail", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let body_str = String::from_utf8(body.to_vec()).unwrap();
        assert!(body_str.contains("\"attempt\""));
        assert!(body_str.contains("\"executions\""));
        assert!(body_str.contains("\"reviews\""));
    }

    #[tokio::test]
    async fn test_retries_with_seeded_failed_attempt() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        seed_attempt(&tracker, "linear", "issue-1", "PROJ-1");
        tracker
            .mark_failed("linear", "issue-1", "Build failed")
            .unwrap();

        let response = router
            .oneshot(auth_get("/api/retries", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let body_str = String::from_utf8(body.to_vec()).unwrap();
        assert!(body_str.contains("retryable"));
        assert!(body_str.contains("ready"));
        assert!(body_str.contains("max_retries"));
    }

    #[tokio::test]
    async fn test_activity_with_seeded_data() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let entry =
            claudear_core::types::ActivityLogEntry::new("issue_received", "Received PROJ-1")
                .with_source("linear")
                .with_issue("issue-1", "PROJ-1");
        tracker.record_activity(&entry).unwrap();

        let response = router
            .oneshot(auth_get("/api/activity", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let entries: Vec<serde_json::Value> = serde_json::from_slice(&body).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0]["activity_type"], "issue_received");
        assert_eq!(entries[0]["message"], "Received PROJ-1");
    }

    #[tokio::test]
    async fn test_metrics_with_seeded_data() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let metric = claudear_core::types::ProcessingMetric {
            id: 0,
            timestamp: chrono::Utc::now(),
            metric_name: "processing_time".to_string(),
            metric_value: 42.5,
            source: Some("linear".to_string()),
            tags: None,
        };
        tracker.record_metric(&metric).unwrap();

        let response = router
            .oneshot(auth_get("/api/metrics?name=processing_time", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let metrics: Vec<serde_json::Value> = serde_json::from_slice(&body).unwrap();
        assert_eq!(metrics.len(), 1);
        assert_eq!(metrics[0]["metric_name"], "processing_time");
    }

    #[tokio::test]
    async fn test_errors_with_seeded_data() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let error_pattern = claudear_core::types::ErrorPattern {
            id: 0,
            pattern_hash: "abc123".to_string(),
            error_type: Some("build_failure".to_string()),
            error_message: Some("Failed to compile".to_string()),
            first_seen: chrono::Utc::now(),
            last_seen: chrono::Utc::now(),
            occurrence_count: 3,
            sources: Some(vec!["linear".to_string()]),
            example_issue_ids: None,
            resolution_hints: None,
        };
        tracker.record_error_pattern(&error_pattern).unwrap();

        let response = router
            .oneshot(auth_get("/api/errors", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let errors: Vec<serde_json::Value> = serde_json::from_slice(&body).unwrap();
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0]["pattern_hash"], "abc123");
    }

    #[tokio::test]
    async fn test_overview_rates_computation() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        // Seed 4 attempts, 2 success, 1 merged, 1 failed
        seed_attempt(&tracker, "linear", "issue-1", "PROJ-1");
        seed_attempt(&tracker, "linear", "issue-2", "PROJ-2");
        seed_attempt(&tracker, "linear", "issue-3", "PROJ-3");
        seed_attempt(&tracker, "linear", "issue-4", "PROJ-4");

        tracker
            .mark_success("linear", "issue-1", "https://github.com/org/repo/pull/1")
            .unwrap();
        tracker
            .mark_success("linear", "issue-2", "https://github.com/org/repo/pull/2")
            .unwrap();
        tracker.mark_merged("linear", "issue-2").unwrap();
        tracker
            .mark_failed("linear", "issue-3", "Build failed")
            .unwrap();

        let response = router
            .oneshot(auth_get("/api/stats/overview", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let overview: serde_json::Value = serde_json::from_slice(&body).unwrap();

        // success_rate = (success + merged) / total * 100 = (1 + 1) / 4 * 100 = 50.0
        let success_rate = overview["success_rate"].as_f64().unwrap();
        assert!(success_rate > 0.0);

        // stats.total should be 4
        assert_eq!(overview["stats"]["total"], 4);
    }

    #[tokio::test]
    async fn test_sources_response_structure() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/sources", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let body_str = String::from_utf8(body.to_vec()).unwrap();
        // Default config has no sources configured, so sources array should be empty
        assert!(body_str.contains("\"sources\":[]"));
    }

    #[tokio::test]
    async fn test_health_reports_database_ok() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/health", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let health: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(health["status"], "ok");
        assert_eq!(health["database"]["status"], "ok");
        assert!(health["uptime_secs"].as_u64().is_some());
        assert!(health["version"].as_str().is_some());
    }

    #[tokio::test]
    async fn test_attempts_pagination_page_2() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        // Seed 3 attempts
        seed_attempt(&tracker, "linear", "issue-1", "PROJ-1");
        seed_attempt(&tracker, "linear", "issue-2", "PROJ-2");
        seed_attempt(&tracker, "linear", "issue-3", "PROJ-3");

        let response = router
            .oneshot(auth_get("/api/attempts?page=2&per_page=2", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let resp: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(resp["page"], 2);
        assert_eq!(resp["per_page"], 2);
        assert_eq!(resp["total"], 3);
        // Page 2 with per_page=2 should have 1 result
        let attempts = resp["attempts"].as_array().unwrap();
        assert_eq!(attempts.len(), 1);
    }

    #[tokio::test]
    async fn test_unauthenticated_activity_returns_401() {
        let tracker = create_test_tracker();
        let config = test_config();
        let indexing_rx = test_indexing_rx(&tracker);
        let router =
            create_api_router(config, tracker, PathBuf::from("claudear.toml"), indexing_rx)
                .layer(CookieManagerLayer::new());

        let response = router
            .oneshot(
                Request::builder()
                    .uri("/api/activity")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_unauthenticated_config_returns_401() {
        let tracker = create_test_tracker();
        let config = test_config();
        let indexing_rx = test_indexing_rx(&tracker);
        let router =
            create_api_router(config, tracker, PathBuf::from("claudear.toml"), indexing_rx)
                .layer(CookieManagerLayer::new());

        let response = router
            .oneshot(
                Request::builder()
                    .uri("/api/config")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_unauthenticated_experiments_returns_401() {
        let tracker = create_test_tracker();
        let config = test_config();
        let indexing_rx = test_indexing_rx(&tracker);
        let router =
            create_api_router(config, tracker, PathBuf::from("claudear.toml"), indexing_rx)
                .layer(CookieManagerLayer::new());

        let response = router
            .oneshot(
                Request::builder()
                    .uri("/api/experiments")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_analytics_summary_with_seeded_data() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        seed_attempt(&tracker, "linear", "issue-1", "PROJ-1");
        tracker
            .mark_success("linear", "issue-1", "https://github.com/org/repo/pull/1")
            .unwrap();
        tracker.mark_merged("linear", "issue-1").unwrap();

        let response = router
            .oneshot(auth_get("/api/analytics/summary", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let summary: claudear_core::types::AnalyticsSummary =
            serde_json::from_slice(&body).unwrap();
        assert_eq!(summary.total_merged, 1);
    }

    #[tokio::test]
    async fn test_telemetry_timeseries_with_different_periods() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        // Test hour period
        let response = router
            .oneshot(auth_get(
                "/api/telemetry/timeseries?period=hour&bucket_minutes=5",
                &token,
            ))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let resp: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(resp["period"], "hour");
        assert_eq!(resp["bucket_minutes"], 5);
        assert!(!resp["points"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_telemetry_pipeline_with_different_periods() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/telemetry/pipeline?period=month", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let resp: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(resp["period"], "month");
        assert!(resp["totals"].is_object());
        assert!(resp["conversion"].is_object());
        assert!(resp["poll_load"].is_object());
    }

    #[tokio::test]
    async fn test_telemetry_latency_default_period() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        // No period parameter - should default to "week"
        let response = router
            .oneshot(auth_get("/api/telemetry/latency", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let resp: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(resp["period"], "week");
        assert!(resp["overall"].is_object());
        assert!(!resp["histogram"].as_array().unwrap().is_empty());
    }

    #[test]
    fn test_tail_utf8_short_string() {
        let (result, truncated) = tail_utf8("hello", 100);
        assert_eq!(result, "hello");
        assert!(!truncated);
    }

    #[test]
    fn test_tail_utf8_long_string() {
        let long = "a".repeat(1000);
        let (result, truncated) = tail_utf8(&long, 100);
        assert!(truncated);
        assert!(result.contains("...[truncated]"));
        // The tail should contain at most ~100 bytes of the original plus the prefix
        assert!(result.len() <= 200);
    }

    #[test]
    fn test_parse_telemetry_period_known() {
        let (label, _duration) = parse_telemetry_period(Some("hour"), "week");
        assert_eq!(label, "hour");

        let (label, _duration) = parse_telemetry_period(Some("day"), "week");
        assert_eq!(label, "day");

        let (label, _duration) = parse_telemetry_period(Some("month"), "week");
        assert_eq!(label, "month");

        let (label, _duration) = parse_telemetry_period(Some("week"), "day");
        assert_eq!(label, "week");
    }

    #[test]
    fn test_parse_telemetry_period_unknown_falls_back() {
        let (label, _duration) = parse_telemetry_period(Some("year"), "day");
        assert_eq!(label, "day");

        let (label, _duration) = parse_telemetry_period(None, "hour");
        assert_eq!(label, "hour");
    }

    #[test]
    fn test_floor_to_bucket() {
        use chrono::{TimeZone, Timelike};
        let ts = chrono::Utc
            .with_ymd_and_hms(2024, 6, 15, 10, 37, 42)
            .unwrap();

        // Floor to 15-minute buckets
        let floored = floor_to_bucket(ts, 15);
        assert_eq!(floored.minute(), 30);
        assert_eq!(floored.second(), 0);

        // Floor to 60-minute buckets
        let floored = floor_to_bucket(ts, 60);
        assert_eq!(floored.minute(), 0);
        assert_eq!(floored.second(), 0);
    }

    #[test]
    fn test_compute_processing_value_summary_empty() {
        let summary = compute_processing_value_summary(vec![]);
        assert_eq!(summary.samples, 0);
        assert!(summary.avg_secs.is_none());
        assert!(summary.max_secs.is_none());
    }

    #[test]
    fn test_compute_processing_value_summary_with_data() {
        let summary = compute_processing_value_summary(vec![10.0, 20.0, 30.0, 40.0, 50.0]);
        assert_eq!(summary.samples, 5);
        assert!((summary.avg_secs.unwrap() - 30.0).abs() < 0.01);
        assert!((summary.max_secs.unwrap() - 50.0).abs() < 0.01);
        assert!(summary.p50_secs.is_some());
        assert!(summary.p95_secs.is_some());
        assert!(summary.p99_secs.is_some());
    }

    #[test]
    fn test_ratio_function() {
        assert!((ratio(50.0, 100.0).unwrap() - 0.5).abs() < 0.001);
        assert!(ratio(10.0, 0.0).is_none());
    }

    #[test]
    fn test_summarize_window_empty() {
        let attempts: Vec<FixAttempt> = vec![];
        let metric = summarize_window(&attempts, "1h", chrono::Utc::now() - Duration::hours(1));
        assert_eq!(metric.window, "1h");
        assert_eq!(metric.processed, 0);
        assert_eq!(metric.successful, 0);
        assert_eq!(metric.failed, 0);
    }

    #[test]
    fn test_summarize_window_with_attempts() {
        let now = chrono::Utc::now();
        let attempts = vec![
            FixAttempt {
                id: 1,
                source: "linear".to_string(),
                issue_id: "1".to_string(),
                short_id: "P-1".to_string(),
                status: FixAttemptStatus::Success,
                pr_url: None,
                scm_repo: None,
                scm_pr_number: None,
                error_message: None,
                attempted_at: now,
                resolved_at: None,
                merged_at: None,
                retry_count: 0,
                last_retry_at: None,
                issue_labels: vec![],
                parent_attempt_id: None,
                cascade_repo: None,
            },
            FixAttempt {
                id: 2,
                source: "linear".to_string(),
                issue_id: "2".to_string(),
                short_id: "P-2".to_string(),
                status: FixAttemptStatus::Failed,
                pr_url: None,
                scm_repo: None,
                scm_pr_number: None,
                error_message: Some("error".to_string()),
                attempted_at: now,
                resolved_at: None,
                merged_at: None,
                retry_count: 0,
                last_retry_at: None,
                issue_labels: vec![],
                parent_attempt_id: None,
                cascade_repo: None,
            },
        ];
        let metric = summarize_window(&attempts, "24h", now - Duration::hours(24));
        assert_eq!(metric.window, "24h");
        assert_eq!(metric.processed, 2);
        assert_eq!(metric.successful, 1);
        assert_eq!(metric.failed, 1);
        assert!((metric.success_rate - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_sum_metric_values_empty() {
        let metrics: Vec<claudear_core::types::ProcessingMetric> = vec![];
        assert!((sum_metric_values(&metrics) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_sum_metric_values_multiple() {
        let metrics = vec![
            claudear_core::types::ProcessingMetric {
                id: 0,
                timestamp: chrono::Utc::now(),
                metric_name: "test".to_string(),
                metric_value: 10.0,
                source: None,
                tags: None,
            },
            claudear_core::types::ProcessingMetric {
                id: 1,
                timestamp: chrono::Utc::now(),
                metric_name: "test".to_string(),
                metric_value: 20.5,
                source: None,
                tags: None,
            },
            claudear_core::types::ProcessingMetric {
                id: 2,
                timestamp: chrono::Utc::now(),
                metric_name: "test".to_string(),
                metric_value: 5.5,
                source: None,
                tags: None,
            },
        ];
        assert!((sum_metric_values(&metrics) - 36.0).abs() < 0.001);
    }

    #[test]
    fn test_average_metric_value_empty() {
        let metrics: Vec<claudear_core::types::ProcessingMetric> = vec![];
        assert!(average_metric_value(&metrics).is_none());
    }

    #[test]
    fn test_average_metric_value_single() {
        let metrics = vec![claudear_core::types::ProcessingMetric {
            id: 0,
            timestamp: chrono::Utc::now(),
            metric_name: "test".to_string(),
            metric_value: 42.0,
            source: None,
            tags: None,
        }];
        assert!((average_metric_value(&metrics).unwrap() - 42.0).abs() < 0.001);
    }

    #[test]
    fn test_average_metric_value_multiple() {
        let metrics = vec![
            claudear_core::types::ProcessingMetric {
                id: 0,
                timestamp: chrono::Utc::now(),
                metric_name: "test".to_string(),
                metric_value: 10.0,
                source: None,
                tags: None,
            },
            claudear_core::types::ProcessingMetric {
                id: 1,
                timestamp: chrono::Utc::now(),
                metric_name: "test".to_string(),
                metric_value: 30.0,
                source: None,
                tags: None,
            },
        ];
        assert!((average_metric_value(&metrics).unwrap() - 20.0).abs() < 0.001);
    }

    #[test]
    fn test_max_metric_value_empty() {
        let metrics: Vec<claudear_core::types::ProcessingMetric> = vec![];
        assert!(max_metric_value(&metrics).is_none());
    }

    #[test]
    fn test_max_metric_value_multiple() {
        let metrics = vec![
            claudear_core::types::ProcessingMetric {
                id: 0,
                timestamp: chrono::Utc::now(),
                metric_name: "test".to_string(),
                metric_value: 5.0,
                source: None,
                tags: None,
            },
            claudear_core::types::ProcessingMetric {
                id: 1,
                timestamp: chrono::Utc::now(),
                metric_name: "test".to_string(),
                metric_value: 99.9,
                source: None,
                tags: None,
            },
            claudear_core::types::ProcessingMetric {
                id: 2,
                timestamp: chrono::Utc::now(),
                metric_name: "test".to_string(),
                metric_value: 50.0,
                source: None,
                tags: None,
            },
        ];
        assert!((max_metric_value(&metrics).unwrap() - 99.9).abs() < 0.001);
    }

    #[test]
    fn test_latest_metric_value_empty() {
        let metrics: Vec<claudear_core::types::ProcessingMetric> = vec![];
        assert!(latest_metric_value(&metrics).is_none());
    }

    #[test]
    fn test_latest_metric_value_returns_first() {
        let metrics = vec![
            claudear_core::types::ProcessingMetric {
                id: 0,
                timestamp: chrono::Utc::now(),
                metric_name: "test".to_string(),
                metric_value: 77.7,
                source: None,
                tags: None,
            },
            claudear_core::types::ProcessingMetric {
                id: 1,
                timestamp: chrono::Utc::now(),
                metric_name: "test".to_string(),
                metric_value: 88.8,
                source: None,
                tags: None,
            },
        ];
        assert!((latest_metric_value(&metrics).unwrap() - 77.7).abs() < 0.001);
    }

    #[test]
    fn test_ratio_positive_denominator() {
        let r = ratio(25.0, 50.0);
        assert!(r.is_some());
        assert!((r.unwrap() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_ratio_zero_denominator() {
        assert!(ratio(25.0, 0.0).is_none());
    }

    #[test]
    fn test_ratio_negative_denominator() {
        assert!(ratio(25.0, -1.0).is_none());
    }

    #[test]
    fn test_ratio_zero_numerator() {
        let r = ratio(0.0, 100.0);
        assert!(r.is_some());
        assert!((r.unwrap() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_compute_processing_value_summary_single_value() {
        let summary = compute_processing_value_summary(vec![42.0]);
        assert_eq!(summary.samples, 1);
        assert!((summary.avg_secs.unwrap() - 42.0).abs() < 0.01);
        assert!((summary.max_secs.unwrap() - 42.0).abs() < 0.01);
        assert!((summary.p50_secs.unwrap() - 42.0).abs() < 0.01);
        assert!((summary.p95_secs.unwrap() - 42.0).abs() < 0.01);
        assert!((summary.p99_secs.unwrap() - 42.0).abs() < 0.01);
    }

    #[test]
    fn test_compute_processing_value_summary_two_values() {
        let summary = compute_processing_value_summary(vec![10.0, 90.0]);
        assert_eq!(summary.samples, 2);
        assert!((summary.avg_secs.unwrap() - 50.0).abs() < 0.01);
        assert!((summary.max_secs.unwrap() - 90.0).abs() < 0.01);
    }

    #[test]
    fn test_compute_processing_time_summary_with_metrics() {
        let metrics = vec![
            claudear_core::types::ProcessingMetric {
                id: 0,
                timestamp: chrono::Utc::now(),
                metric_name: "processing_time".to_string(),
                metric_value: 5.0,
                source: None,
                tags: None,
            },
            claudear_core::types::ProcessingMetric {
                id: 1,
                timestamp: chrono::Utc::now(),
                metric_name: "processing_time".to_string(),
                metric_value: 15.0,
                source: None,
                tags: None,
            },
            claudear_core::types::ProcessingMetric {
                id: 2,
                timestamp: chrono::Utc::now(),
                metric_name: "processing_time".to_string(),
                metric_value: 25.0,
                source: None,
                tags: None,
            },
        ];
        let summary = compute_processing_time_summary(metrics);
        assert_eq!(summary.samples, 3);
        assert!((summary.avg_secs.unwrap() - 15.0).abs() < 0.01);
        assert!((summary.max_secs.unwrap() - 25.0).abs() < 0.01);
    }

    #[test]
    fn test_parse_telemetry_period_none_with_various_defaults() {
        let (label, _) = parse_telemetry_period(None, "hour");
        assert_eq!(label, "hour");

        let (label, _) = parse_telemetry_period(None, "day");
        assert_eq!(label, "day");

        let (label, _) = parse_telemetry_period(None, "month");
        assert_eq!(label, "month");

        let (label, _) = parse_telemetry_period(None, "year");
        assert_eq!(label, "week");
    }

    #[test]
    fn test_parse_telemetry_period_case_insensitive() {
        let (label, _) = parse_telemetry_period(Some("HOUR"), "week");
        assert_eq!(label, "hour");

        let (label, _) = parse_telemetry_period(Some("Day"), "week");
        assert_eq!(label, "day");

        let (label, _) = parse_telemetry_period(Some("MONTH"), "week");
        assert_eq!(label, "month");
    }

    #[test]
    fn test_floor_to_bucket_5_minute() {
        use chrono::{TimeZone, Timelike};
        let ts = chrono::Utc
            .with_ymd_and_hms(2024, 6, 15, 10, 13, 45)
            .unwrap();
        let floored = floor_to_bucket(ts, 5);
        assert_eq!(floored.minute(), 10);
        assert_eq!(floored.second(), 0);
    }

    #[test]
    fn test_floor_to_bucket_already_on_boundary() {
        use chrono::TimeZone;
        let ts = chrono::Utc
            .with_ymd_and_hms(2024, 6, 15, 10, 30, 0)
            .unwrap();
        let floored = floor_to_bucket(ts, 15);
        assert_eq!(floored, ts);
    }

    #[test]
    fn test_floor_to_bucket_minimum_clamp() {
        use chrono::{TimeZone, Timelike};
        let ts = chrono::Utc
            .with_ymd_and_hms(2024, 6, 15, 10, 37, 42)
            .unwrap();
        let floored = floor_to_bucket(ts, 0);
        assert_eq!(floored.second(), 0);
    }

    #[test]
    fn test_tail_utf8_exact_boundary() {
        let input = "abcde";
        let (result, truncated) = tail_utf8(input, 5);
        assert_eq!(result, "abcde");
        assert!(!truncated);
    }

    #[test]
    fn test_tail_utf8_one_under_boundary() {
        let input = "abcdef";
        let (result, truncated) = tail_utf8(input, 5);
        assert!(truncated);
        assert!(result.contains("...[truncated]"));
    }

    #[test]
    fn test_tail_utf8_multibyte_chars() {
        let input = "\u{1F600}\u{1F601}\u{1F602}\u{1F603}";
        let (result, truncated) = tail_utf8(input, 8);
        assert!(truncated);
        assert!(result.contains("...[truncated]"));
    }

    #[test]
    fn test_tail_utf8_empty_string() {
        let (result, truncated) = tail_utf8("", 100);
        assert_eq!(result, "");
        assert!(!truncated);
    }

    #[test]
    fn test_redact_secrets_basic() {
        let content = "api_token = \"sk-12345\"\nwebhook_secret = \"whsec_abc\"\nnormal_key = \"not_a_secret\"\n";
        let redacted = redact_secrets(content);
        assert!(redacted.contains("[REDACTED]"));
        assert!(redacted.contains("normal_key"));
        assert!(!redacted.contains("sk-12345"));
        assert!(!redacted.contains("whsec_abc"));
    }

    #[test]
    fn test_redact_secrets_no_secrets() {
        let content = "workspace = \"/tmp/repos\"\npoll_interval_ms = 300000\n";
        let redacted = redact_secrets(content);
        assert!(!redacted.contains("[REDACTED]"));
        assert!(redacted.contains("/tmp/repos"));
        assert!(redacted.contains("300000"));
    }

    #[test]
    fn test_redact_secrets_password_key() {
        let content = "password = \"my_secret_pass\"";
        let redacted = redact_secrets(content);
        assert!(redacted.contains("[REDACTED]"));
        assert!(!redacted.contains("my_secret_pass"));
    }

    #[test]
    fn test_redact_secrets_api_key() {
        let content = "api_key = \"key-abc123\"";
        let redacted = redact_secrets(content);
        assert!(redacted.contains("[REDACTED]"));
        assert!(!redacted.contains("key-abc123"));
    }

    #[test]
    fn test_summarize_window_with_merged_and_cannotfix() {
        let now = chrono::Utc::now();
        let attempts = vec![
            FixAttempt {
                id: 1,
                source: "linear".to_string(),
                issue_id: "1".to_string(),
                short_id: "P-1".to_string(),
                status: FixAttemptStatus::Merged,
                pr_url: None,
                scm_repo: None,
                scm_pr_number: None,
                error_message: None,
                attempted_at: now,
                resolved_at: None,
                merged_at: None,
                retry_count: 0,
                last_retry_at: None,
                issue_labels: vec![],
                parent_attempt_id: None,
                cascade_repo: None,
            },
            FixAttempt {
                id: 2,
                source: "linear".to_string(),
                issue_id: "2".to_string(),
                short_id: "P-2".to_string(),
                status: FixAttemptStatus::CannotFix,
                pr_url: None,
                scm_repo: None,
                scm_pr_number: None,
                error_message: None,
                attempted_at: now,
                resolved_at: None,
                merged_at: None,
                retry_count: 0,
                last_retry_at: None,
                issue_labels: vec![],
                parent_attempt_id: None,
                cascade_repo: None,
            },
            FixAttempt {
                id: 3,
                source: "linear".to_string(),
                issue_id: "3".to_string(),
                short_id: "P-3".to_string(),
                status: FixAttemptStatus::Closed,
                pr_url: None,
                scm_repo: None,
                scm_pr_number: None,
                error_message: None,
                attempted_at: now,
                resolved_at: None,
                merged_at: None,
                retry_count: 0,
                last_retry_at: None,
                issue_labels: vec![],
                parent_attempt_id: None,
                cascade_repo: None,
            },
            FixAttempt {
                id: 4,
                source: "linear".to_string(),
                issue_id: "4".to_string(),
                short_id: "P-4".to_string(),
                status: FixAttemptStatus::Pending,
                pr_url: None,
                scm_repo: None,
                scm_pr_number: None,
                error_message: None,
                attempted_at: now,
                resolved_at: None,
                merged_at: None,
                retry_count: 0,
                last_retry_at: None,
                issue_labels: vec![],
                parent_attempt_id: None,
                cascade_repo: None,
            },
        ];
        let metric = summarize_window(&attempts, "7d", now - Duration::days(7));
        assert_eq!(metric.processed, 3);
        assert_eq!(metric.successful, 1);
        assert_eq!(metric.failed, 2);
        assert_eq!(metric.merged, 1);
    }

    #[test]
    fn test_summarize_window_excludes_old_attempts() {
        let now = chrono::Utc::now();
        let old_time = now - Duration::hours(48);
        let attempts = vec![FixAttempt {
            id: 1,
            source: "linear".to_string(),
            issue_id: "1".to_string(),
            short_id: "P-1".to_string(),
            status: FixAttemptStatus::Success,
            pr_url: None,
            scm_repo: None,
            scm_pr_number: None,
            error_message: None,
            attempted_at: old_time,
            resolved_at: None,
            merged_at: None,
            retry_count: 0,
            last_retry_at: None,
            issue_labels: vec![],
            parent_attempt_id: None,
            cascade_repo: None,
        }];
        let metric = summarize_window(&attempts, "1h", now - Duration::hours(1));
        assert_eq!(metric.processed, 0);
        assert_eq!(metric.successful, 0);
    }

    #[test]
    fn test_summarize_window_throughput_calculation() {
        let now = chrono::Utc::now();
        let attempts = vec![FixAttempt {
            id: 1,
            source: "linear".to_string(),
            issue_id: "1".to_string(),
            short_id: "P-1".to_string(),
            status: FixAttemptStatus::Success,
            pr_url: None,
            scm_repo: None,
            scm_pr_number: None,
            error_message: None,
            attempted_at: now,
            resolved_at: None,
            merged_at: None,
            retry_count: 0,
            last_retry_at: None,
            issue_labels: vec![],
            parent_attempt_id: None,
            cascade_repo: None,
        }];
        let metric = summarize_window(&attempts, "1h", now - Duration::hours(1));
        assert!((metric.throughput_per_hour - 1.0).abs() < 0.01);

        let metric = summarize_window(&attempts, "24h", now - Duration::hours(24));
        assert!((metric.throughput_per_hour - 1.0 / 24.0).abs() < 0.01);
    }

    #[test]
    fn test_activity_query_deserialization() {
        let query: ActivityQuery =
            serde_json::from_str(r#"{"limit": 10, "source": "sentry"}"#).unwrap();
        assert_eq!(query.limit, Some(10));
        assert_eq!(query.source, Some("sentry".to_string()));
    }

    #[test]
    fn test_activity_query_defaults() {
        let query: ActivityQuery = serde_json::from_str("{}").unwrap();
        assert!(query.limit.is_none());
        assert!(query.source.is_none());
    }

    #[test]
    fn test_metrics_query_deserialization() {
        let query: MetricsQuery =
            serde_json::from_str(r#"{"name": "processing_time", "period": "day", "limit": 100}"#)
                .unwrap();
        assert_eq!(query.name, Some("processing_time".to_string()));
        assert_eq!(query.period, Some("day".to_string()));
        assert_eq!(query.limit, Some(100));
    }

    #[test]
    fn test_metrics_query_defaults() {
        let query: MetricsQuery = serde_json::from_str("{}").unwrap();
        assert!(query.name.is_none());
        assert!(query.period.is_none());
        assert!(query.limit.is_none());
    }

    #[test]
    fn test_errors_query_deserialization() {
        let query: ErrorsQuery = serde_json::from_str(r#"{"limit": 25}"#).unwrap();
        assert_eq!(query.limit, Some(25));
    }

    #[test]
    fn test_errors_query_defaults() {
        let query: ErrorsQuery = serde_json::from_str("{}").unwrap();
        assert!(query.limit.is_none());
    }

    #[test]
    fn test_issues_query_deserialization() {
        let query: IssuesQuery =
            serde_json::from_str(r#"{"source": "linear", "page": 2, "per_page": 50}"#).unwrap();
        assert_eq!(query.source, Some("linear".to_string()));
        assert_eq!(query.page, Some(2));
        assert_eq!(query.per_page, Some(50));
    }

    #[test]
    fn test_issues_query_defaults() {
        let query: IssuesQuery = serde_json::from_str("{}").unwrap();
        assert!(query.source.is_none());
        assert!(query.page.is_none());
        assert!(query.per_page.is_none());
    }

    #[test]
    fn test_prs_query_deserialization() {
        let query: PrsQuery = serde_json::from_str(r#"{"status": "open", "limit": 10}"#).unwrap();
        assert_eq!(query.status, Some("open".to_string()));
        assert_eq!(query.limit, Some(10));
    }

    #[test]
    fn test_prs_query_defaults() {
        let query: PrsQuery = serde_json::from_str("{}").unwrap();
        assert!(query.status.is_none());
        assert!(query.limit.is_none());
    }

    #[test]
    fn test_feedback_query_deserialization() {
        let query: FeedbackQuery =
            serde_json::from_str(r#"{"source": "sentry", "limit": 30}"#).unwrap();
        assert_eq!(query.source, Some("sentry".to_string()));
        assert_eq!(query.limit, Some(30));
    }

    #[test]
    fn test_feedback_query_defaults() {
        let query: FeedbackQuery = serde_json::from_str("{}").unwrap();
        assert!(query.source.is_none());
        assert!(query.limit.is_none());
    }

    #[test]
    fn test_regressions_query_deserialization() {
        let query: RegressionsQuery = serde_json::from_str(r#"{"status": "monitoring"}"#).unwrap();
        assert_eq!(query.status, Some("monitoring".to_string()));
    }

    #[test]
    fn test_regressions_query_defaults() {
        let query: RegressionsQuery = serde_json::from_str("{}").unwrap();
        assert!(query.status.is_none());
    }

    #[test]
    fn test_inference_history_query_deserialization() {
        let query: InferenceHistoryQuery = serde_json::from_str(r#"{"limit": 20}"#).unwrap();
        assert_eq!(query.limit, Some(20));
    }

    #[test]
    fn test_inference_history_query_defaults() {
        let query: InferenceHistoryQuery = serde_json::from_str("{}").unwrap();
        assert!(query.limit.is_none());
    }

    #[test]
    fn test_telemetry_timeseries_query_deserialization() {
        let query: TelemetryTimeseriesQuery =
            serde_json::from_str(r#"{"period": "day", "bucket_minutes": 30}"#).unwrap();
        assert_eq!(query.period, Some("day".to_string()));
        assert_eq!(query.bucket_minutes, Some(30));
    }

    #[test]
    fn test_telemetry_timeseries_query_defaults() {
        let query: TelemetryTimeseriesQuery = serde_json::from_str("{}").unwrap();
        assert!(query.period.is_none());
        assert!(query.bucket_minutes.is_none());
    }

    #[test]
    fn test_telemetry_period_query_deserialization() {
        let query: TelemetryPeriodQuery = serde_json::from_str(r#"{"period": "month"}"#).unwrap();
        assert_eq!(query.period, Some("month".to_string()));
    }

    #[test]
    fn test_telemetry_period_query_defaults() {
        let query: TelemetryPeriodQuery = serde_json::from_str("{}").unwrap();
        assert!(query.period.is_none());
    }

    #[test]
    fn test_config_update_request_deserialization() {
        let req: ConfigUpdateRequest =
            serde_json::from_str(r#"{"content": "key = \"value\""}"#).unwrap();
        assert_eq!(req.content, r#"key = "value""#);
    }

    #[test]
    fn test_database_status_skip_error_when_none() {
        let status = DatabaseStatus {
            status: "ok".to_string(),
            error: None,
        };
        let json = serde_json::to_string(&status).unwrap();
        assert!(!json.contains("error"));
    }

    #[test]
    fn test_database_status_include_error_when_some() {
        let status = DatabaseStatus {
            status: "error".to_string(),
            error: Some("Connection refused".to_string()),
        };
        let json = serde_json::to_string(&status).unwrap();
        assert!(json.contains("error"));
        assert!(json.contains("Connection refused"));
    }

    #[test]
    fn test_telemetry_window_metric_serialization() {
        let metric = TelemetryWindowMetric {
            window: "1h".to_string(),
            processed: 100,
            successful: 80,
            failed: 20,
            merged: 50,
            success_rate: 80.0,
            error_rate: 20.0,
            throughput_per_hour: 100.0,
        };
        let json = serde_json::to_string(&metric).unwrap();
        assert!(json.contains("\"window\":\"1h\""));
        assert!(json.contains("\"processed\":100"));
        assert!(json.contains("\"throughput_per_hour\":100.0"));
    }

    #[test]
    fn test_telemetry_queue_metrics_default() {
        let queue = TelemetryQueueMetrics::default();
        assert_eq!(queue.pending_attempts, 0);
        assert_eq!(queue.retryable_attempts, 0);
        assert_eq!(queue.ready_retries, 0);
        assert_eq!(queue.open_prs, 0);
        assert_eq!(queue.watches_awaiting_release, 0);
    }

    #[test]
    fn test_processing_time_summary_default() {
        let summary = ProcessingTimeSummary::default();
        assert_eq!(summary.samples, 0);
        assert!(summary.avg_secs.is_none());
        assert!(summary.p50_secs.is_none());
        assert!(summary.p95_secs.is_none());
        assert!(summary.p99_secs.is_none());
        assert!(summary.max_secs.is_none());
    }

    #[test]
    fn test_telemetry_timeseries_point_default() {
        let point = TelemetryTimeseriesPoint::default();
        assert_eq!(point.total, 0);
        assert_eq!(point.pending, 0);
        assert_eq!(point.success, 0);
        assert_eq!(point.failed, 0);
        assert_eq!(point.merged, 0);
        assert_eq!(point.closed, 0);
        assert_eq!(point.cannot_fix, 0);
    }

    #[test]
    fn test_telemetry_pipeline_totals_default() {
        let totals = TelemetryPipelineTotals::default();
        assert!((totals.fetched - 0.0).abs() < f64::EPSILON);
        assert!((totals.matched - 0.0).abs() < f64::EPSILON);
        assert!((totals.queued - 0.0).abs() < f64::EPSILON);
        assert!((totals.processed - 0.0).abs() < f64::EPSILON);
        assert!((totals.pr_created - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_telemetry_pipeline_conversion_default() {
        let conv = TelemetryPipelineConversion::default();
        assert!(conv.match_rate.is_none());
        assert!(conv.queue_rate.is_none());
        assert!(conv.processing_rate.is_none());
        assert!(conv.pr_yield_rate.is_none());
    }

    #[test]
    fn test_config_response_serialization() {
        let resp = ConfigResponse {
            content: "workspace = \"/tmp\"".to_string(),
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("workspace"));
        assert!(json.contains("content"));
    }

    #[test]
    fn test_attempt_execution_log_response_serialization() {
        let resp = AttemptExecutionLogResponse {
            attempt_id: 1,
            execution_id: 2,
            stream: "stdout".to_string(),
            path: Some("/var/log/test.log".to_string()),
            content: Some("hello world".to_string()),
            truncated: false,
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"attempt_id\":1"));
        assert!(json.contains("\"execution_id\":2"));
        assert!(json.contains("stdout"));
        assert!(json.contains("hello world"));
        assert!(json.contains("\"truncated\":false"));
    }

    #[test]
    fn test_attempt_execution_log_response_no_content() {
        let resp = AttemptExecutionLogResponse {
            attempt_id: 1,
            execution_id: 2,
            stream: "stderr".to_string(),
            path: None,
            content: None,
            truncated: false,
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("stderr"));
        assert!(json.contains("null"));
    }

    #[test]
    fn test_source_telemetry_default() {
        let st = SourceTelemetry::default();
        assert_eq!(st.source, "");
        assert_eq!(st.total, 0);
        assert_eq!(st.pending, 0);
        assert!((st.success_rate - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_telemetry_poll_load_default() {
        let load = TelemetryPollLoad::default();
        assert_eq!(load.poll_cycles, 0);
        assert!(load.avg_cycle_secs.is_none());
        assert!(load.p95_cycle_secs.is_none());
        assert!(load.active_avg.is_none());
    }

    #[test]
    fn test_issue_summary_serialization() {
        let summary = IssueSummary {
            id: 1,
            source: "linear".to_string(),
            issue_id: "LIN-123".to_string(),
            short_id: Some("LIN-123".to_string()),
            title: Some("Fix bug".to_string()),
            description: None,
            url: Some("https://linear.app/issue/LIN-123".to_string()),
            priority: Some("high".to_string()),
            status: Some("open".to_string()),
            labels: Some(vec!["bug".to_string(), "urgent".to_string()]),
            has_embedding: true,
            created_at: "2024-01-01 00:00:00".to_string(),
            updated_at: None,
        };
        let json = serde_json::to_string(&summary).unwrap();
        assert!(json.contains("LIN-123"));
        assert!(json.contains("linear"));
        assert!(json.contains("has_embedding"));
        assert!(json.contains("true"));
        assert!(json.contains("bug"));
    }

    #[test]
    fn test_issues_response_serialization() {
        let resp = IssuesResponse {
            issues: vec![],
            total: 0,
            page: 1,
            per_page: 100,
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"total\":0"));
        assert!(json.contains("\"page\":1"));
        assert!(json.contains("\"per_page\":100"));
    }

    #[test]
    fn test_telemetry_latency_histogram_bucket_serialization() {
        let bucket = TelemetryLatencyHistogramBucket {
            label: "<=15s".to_string(),
            upper_bound_secs: Some(15.0),
            count: 42,
        };
        let json = serde_json::to_string(&bucket).unwrap();
        assert!(json.contains("<=15s"));
        assert!(json.contains("15.0"));
        assert!(json.contains("42"));
    }

    #[test]
    fn test_telemetry_latency_histogram_bucket_no_upper_bound() {
        let bucket = TelemetryLatencyHistogramBucket {
            label: ">5m".to_string(),
            upper_bound_secs: None,
            count: 3,
        };
        let json = serde_json::to_string(&bucket).unwrap();
        assert!(json.contains(">5m"));
        assert!(json.contains("null"));
    }

    #[tokio::test]
    async fn test_issues_endpoint_empty() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/issues", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let resp: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(resp["total"], 0);
        assert_eq!(resp["page"], 1);
        assert_eq!(resp["per_page"], 100);
    }

    #[tokio::test]
    async fn test_issues_endpoint_with_source_filter() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get(
                "/api/issues?source=linear&page=1&per_page=10",
                &token,
            ))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    fn make_attempt(id: i64, source: &str, status: FixAttemptStatus) -> FixAttempt {
        FixAttempt {
            id,
            source: source.to_string(),
            issue_id: format!("issue-{}", id),
            short_id: format!("P-{}", id),
            status,
            pr_url: None,
            scm_repo: None,
            scm_pr_number: None,
            error_message: None,
            attempted_at: chrono::Utc::now(),
            resolved_at: None,
            merged_at: None,
            retry_count: 0,
            last_retry_at: None,
            issue_labels: vec![],
            parent_attempt_id: None,
            cascade_repo: None,
        }
    }

    fn make_metric(
        name: &str,
        value: f64,
        source: Option<&str>,
    ) -> claudear_core::types::ProcessingMetric {
        claudear_core::types::ProcessingMetric {
            id: 0,
            timestamp: chrono::Utc::now(),
            metric_name: name.to_string(),
            metric_value: value,
            source: source.map(|s| s.to_string()),
            tags: None,
        }
    }

    fn make_metric_with_tags(
        name: &str,
        value: f64,
        tags: serde_json::Value,
    ) -> claudear_core::types::ProcessingMetric {
        claudear_core::types::ProcessingMetric {
            id: 0,
            timestamp: chrono::Utc::now(),
            metric_name: name.to_string(),
            metric_value: value,
            source: None,
            tags: Some(tags),
        }
    }

    #[tokio::test]
    async fn test_sources_with_linear_configured() {
        let tracker = create_test_tracker();
        let mut config = test_config();
        config.issues.linear = Some(claudear_config::config::LinearConfig {
            enabled: true,
            trigger_labels: vec!["autofix".to_string()],
            trigger_states: vec!["Triage".to_string()],
            webhook_secret: Some("whsec_123".into()),
            ..claudear_config::config::LinearConfig::default()
        });

        let password_hash = bcrypt::hash("testpass", 4).unwrap();
        tracker
            .create_user("test@example.com", &password_hash, "Test", "admin")
            .unwrap();
        let token = tracker.create_session(1, "2099-12-31 23:59:59").unwrap();

        let indexing_rx = test_indexing_rx(&tracker);
        let router = create_api_router(
            config,
            tracker.clone(),
            PathBuf::from("claudear.toml"),
            indexing_rx,
        )
        .layer(CookieManagerLayer::new());

        let response = router
            .oneshot(auth_get("/api/sources", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let resp: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let sources = resp["sources"].as_array().unwrap();
        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0]["name"], "linear");
        assert_eq!(sources[0]["enabled"], true);
        assert_eq!(sources[0]["config"]["has_webhook_secret"], true);
    }

    #[tokio::test]
    async fn test_sources_with_sentry_configured() {
        let tracker = create_test_tracker();
        let mut config = test_config();
        config.issues.sentry = Some(claudear_config::config::SentryConfig {
            enabled: true,
            org_slug: "my-org".to_string(),
            project_slugs: vec!["proj-1".to_string()],
            min_event_count: 5,
            client_secret: None,
            ..claudear_config::config::SentryConfig::default()
        });

        let password_hash = bcrypt::hash("testpass", 4).unwrap();
        tracker
            .create_user("test@example.com", &password_hash, "Test", "admin")
            .unwrap();
        let token = tracker.create_session(1, "2099-12-31 23:59:59").unwrap();

        let indexing_rx = test_indexing_rx(&tracker);
        let router = create_api_router(
            config,
            tracker.clone(),
            PathBuf::from("claudear.toml"),
            indexing_rx,
        )
        .layer(CookieManagerLayer::new());

        let response = router
            .oneshot(auth_get("/api/sources", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let resp: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let sources = resp["sources"].as_array().unwrap();
        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0]["name"], "sentry");
        assert_eq!(sources[0]["config"]["org_slug"], "my-org");
        assert_eq!(sources[0]["config"]["has_client_secret"], false);
    }

    #[tokio::test]
    async fn test_sources_with_both_linear_and_sentry() {
        let tracker = create_test_tracker();
        let mut config = test_config();
        config.issues.linear = Some(claudear_config::config::LinearConfig::default());
        config.issues.sentry = Some(claudear_config::config::SentryConfig::default());

        let password_hash = bcrypt::hash("testpass", 4).unwrap();
        tracker
            .create_user("test@example.com", &password_hash, "Test", "admin")
            .unwrap();
        let token = tracker.create_session(1, "2099-12-31 23:59:59").unwrap();

        let indexing_rx = test_indexing_rx(&tracker);
        let router = create_api_router(
            config,
            tracker.clone(),
            PathBuf::from("claudear.toml"),
            indexing_rx,
        )
        .layer(CookieManagerLayer::new());

        let response = router
            .oneshot(auth_get("/api/sources", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let resp: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let sources = resp["sources"].as_array().unwrap();
        assert_eq!(sources.len(), 2);
    }

    #[test]
    fn test_get_attempts_with_limit() {
        let tracker = create_test_tracker();
        seed_attempt(&tracker, "linear", "issue-1", "PROJ-1");
        seed_attempt(&tracker, "linear", "issue-2", "PROJ-2");
        seed_attempt(&tracker, "linear", "issue-3", "PROJ-3");

        let attempts = get_attempts(&tracker, Some(2));
        assert_eq!(attempts.len(), 2);
    }

    #[test]
    fn test_get_attempts_without_limit() {
        let tracker = create_test_tracker();
        seed_attempt(&tracker, "linear", "issue-1", "PROJ-1");
        seed_attempt(&tracker, "linear", "issue-2", "PROJ-2");
        seed_attempt(&tracker, "linear", "issue-3", "PROJ-3");

        let attempts = get_attempts(&tracker, None);
        assert_eq!(attempts.len(), 3);
    }

    #[test]
    fn test_get_attempts_empty() {
        let tracker = create_test_tracker();
        let attempts = get_attempts(&tracker, None);
        assert!(attempts.is_empty());
    }

    #[test]
    fn test_get_attempt_records_empty() {
        let tracker = create_test_tracker();
        let records = get_attempt_records(&tracker);
        assert!(records.is_empty());
    }

    #[test]
    fn test_get_attempt_records_with_data() {
        let tracker = create_test_tracker();
        seed_attempt(&tracker, "linear", "issue-1", "PROJ-1");
        tracker
            .mark_success("linear", "issue-1", "https://github.com/org/repo/pull/1")
            .unwrap();
        seed_attempt(&tracker, "linear", "issue-2", "PROJ-2");
        tracker
            .mark_failed("linear", "issue-2", "Build failed")
            .unwrap();

        let records = get_attempt_records(&tracker);
        assert_eq!(records.len(), 2);
    }

    #[test]
    fn test_get_attempt_records_since_filters_by_time() {
        let tracker = create_test_tracker();
        seed_attempt(&tracker, "linear", "issue-1", "PROJ-1");

        let since = chrono::Utc::now() - Duration::seconds(1);
        let records = get_attempt_records_since(&tracker, since);
        assert_eq!(records.len(), 1);

        // Future timestamp should yield no records
        let future = chrono::Utc::now() + Duration::hours(1);
        let records = get_attempt_records_since(&tracker, future);
        assert!(records.is_empty());
    }

    #[test]
    fn test_summarize_window_unknown_window_label() {
        let now = chrono::Utc::now();
        let attempts = vec![make_attempt(1, "linear", FixAttemptStatus::Success)];
        let metric = summarize_window(&attempts, "unknown", now - Duration::hours(1));
        // Unknown windows default to hours=1.0
        assert!((metric.throughput_per_hour - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_floor_to_bucket_1_minute() {
        use chrono::{TimeZone, Timelike};
        let ts = chrono::Utc
            .with_ymd_and_hms(2024, 6, 15, 10, 37, 42)
            .unwrap();
        let floored = floor_to_bucket(ts, 1);
        assert_eq!(floored.minute(), 37);
        assert_eq!(floored.second(), 0);
    }

    #[test]
    fn test_floor_to_bucket_large_bucket() {
        use chrono::{TimeZone, Timelike};
        let ts = chrono::Utc
            .with_ymd_and_hms(2024, 6, 15, 10, 37, 42)
            .unwrap();
        // 360-minute bucket = 6 hours
        let floored = floor_to_bucket(ts, 360);
        assert_eq!(floored.minute(), 0);
        assert_eq!(floored.second(), 0);
        // 10:37 should floor to 6:00 (0, 6, 12, 18)
        assert_eq!(floored.hour(), 6);
    }

    #[test]
    fn test_compute_processing_value_summary_large_dataset() {
        let values: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let summary = compute_processing_value_summary(values);
        assert_eq!(summary.samples, 100);
        assert!((summary.avg_secs.unwrap() - 50.5).abs() < 0.01);
        assert!((summary.max_secs.unwrap() - 100.0).abs() < 0.01);
        assert!((summary.p50_secs.unwrap() - 50.5).abs() < 2.0);
        assert!(summary.p95_secs.unwrap() >= 94.0);
        assert!(summary.p99_secs.unwrap() >= 98.0);
    }

    #[test]
    fn test_redact_secrets_single_quoted_values() {
        let content = "api_token = 'sk-12345'\n";
        let redacted = redact_secrets(content);
        assert!(redacted.contains("[REDACTED]"));
        assert!(!redacted.contains("sk-12345"));
    }

    #[test]
    fn test_redact_secrets_unquoted_values() {
        let content = "auth_token = mytoken123\n";
        let redacted = redact_secrets(content);
        assert!(redacted.contains("[REDACTED]"));
        assert!(!redacted.contains("mytoken123"));
    }

    #[test]
    fn test_redact_secrets_with_leading_whitespace() {
        let content = "  client_secret = \"supersecret\"\n";
        let redacted = redact_secrets(content);
        assert!(redacted.contains("[REDACTED]"));
        assert!(!redacted.contains("supersecret"));
    }

    #[test]
    fn test_redact_secrets_preserves_structure() {
        let content = "[linear]\napi_key = \"lin_key\"\ntrigger_labels = [\"autofix\"]\n";
        let redacted = redact_secrets(content);
        assert!(!redacted.contains("lin_key"));
        assert!(redacted.contains("trigger_labels"));
        assert!(redacted.contains("[\"autofix\"]"));
    }

    #[test]
    fn test_redact_secrets_empty_string() {
        let redacted = redact_secrets("");
        assert_eq!(redacted, "");
    }

    #[test]
    fn test_tail_utf8_max_bytes_1() {
        let (result, truncated) = tail_utf8("hello", 1);
        assert!(truncated);
        assert!(result.contains("...[truncated]"));
        assert!(result.contains("o"));
    }

    #[tokio::test]
    async fn test_telemetry_overview_with_seeded_attempts() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        // Seed some attempts and metrics
        seed_attempt(&tracker, "linear", "issue-1", "PROJ-1");
        tracker
            .mark_success("linear", "issue-1", "https://github.com/org/repo/pull/1")
            .unwrap();
        tracker.mark_merged("linear", "issue-1").unwrap();

        seed_attempt(&tracker, "sentry", "issue-2", "SENTRY-1");
        tracker
            .mark_failed("sentry", "issue-2", "Build failed")
            .unwrap();

        // Record some processing time metrics
        tracker
            .record_metric(&make_metric("processing_time", 15.0, Some("linear")))
            .unwrap();
        tracker
            .record_metric(&make_metric("processing_time", 30.0, Some("sentry")))
            .unwrap();

        let response = router
            .oneshot(auth_get("/api/telemetry/overview", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let resp: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert!(resp["generated_at"].as_str().is_some());
        assert!(resp["uptime_secs"].as_u64().is_some());
        assert!(resp["windows"].as_array().unwrap().len() == 3);
        assert!(resp["queue"].is_object());
        assert!(resp["processing_time"].is_object());
        assert!(!resp["source_breakdown"].as_array().unwrap().is_empty());
        assert!(resp["pr_analytics"].is_object());
    }

    #[tokio::test]
    async fn test_telemetry_timeseries_with_seeded_data() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        seed_attempt(&tracker, "linear", "issue-1", "PROJ-1");
        tracker
            .mark_success("linear", "issue-1", "https://github.com/org/repo/pull/1")
            .unwrap();

        let response = router
            .oneshot(auth_get(
                "/api/telemetry/timeseries?period=hour&bucket_minutes=5",
                &token,
            ))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let resp: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(resp["period"], "hour");
        assert_eq!(resp["bucket_minutes"], 5);
        let points = resp["points"].as_array().unwrap();
        assert!(!points.is_empty());

        // At least one point should have non-zero total
        let total: i64 = points
            .iter()
            .map(|p| p["total"].as_i64().unwrap_or(0))
            .sum();
        assert!(total > 0);
    }

    #[tokio::test]
    async fn test_telemetry_timeseries_month_period() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/telemetry/timeseries?period=month", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let resp: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(resp["period"], "month");
        // Default bucket for month is 360 minutes
        assert_eq!(resp["bucket_minutes"], 360);
    }

    #[tokio::test]
    async fn test_telemetry_timeseries_default_period() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/telemetry/timeseries", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let resp: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(resp["period"], "week");
        assert_eq!(resp["bucket_minutes"], 60);
    }

    #[tokio::test]
    async fn test_telemetry_latency_with_seeded_metrics() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        // Seed processing_time metrics with status tags
        tracker
            .record_metric(&make_metric_with_tags(
                "processing_time",
                10.0,
                serde_json::json!({"status": "success"}),
            ))
            .unwrap();
        tracker
            .record_metric(&make_metric_with_tags(
                "processing_time",
                25.0,
                serde_json::json!({"status": "success"}),
            ))
            .unwrap();
        tracker
            .record_metric(&make_metric_with_tags(
                "processing_time",
                45.0,
                serde_json::json!({"status": "failed"}),
            ))
            .unwrap();
        tracker
            .record_metric(&make_metric_with_tags(
                "processing_time",
                120.0,
                serde_json::json!({"status": "failed"}),
            ))
            .unwrap();
        tracker
            .record_metric(&make_metric("processing_time", 350.0, None))
            .unwrap();

        let response = router
            .oneshot(auth_get("/api/telemetry/latency?period=week", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let resp: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(resp["period"], "week");
        assert!(resp["overall"]["samples"].as_i64().unwrap() > 0);
        assert!(!resp["by_status"].as_array().unwrap().is_empty());

        // Verify histogram has 6 buckets (5 defined + 1 overflow)
        let histogram = resp["histogram"].as_array().unwrap();
        assert_eq!(histogram.len(), 6);
        assert_eq!(histogram[0]["label"], "<=15s");
        assert_eq!(histogram[5]["label"], ">5m");

        // Verify histogram counts sum to total samples
        let hist_total: i64 = histogram
            .iter()
            .map(|b| b["count"].as_i64().unwrap_or(0))
            .sum();
        assert_eq!(hist_total, resp["overall"]["samples"].as_i64().unwrap());
    }

    #[tokio::test]
    async fn test_telemetry_latency_hour_period() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/telemetry/latency?period=hour", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let resp: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(resp["period"], "hour");
    }

    #[tokio::test]
    async fn test_telemetry_pipeline_with_seeded_metrics() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        // Seed pipeline metrics
        tracker
            .record_metric(&make_metric("issues_fetched", 100.0, Some("linear")))
            .unwrap();
        tracker
            .record_metric(&make_metric("issues_matched", 50.0, Some("linear")))
            .unwrap();
        tracker
            .record_metric(&make_metric("issues_queued", 40.0, Some("linear")))
            .unwrap();
        tracker
            .record_metric(&make_metric("batch_processed", 35.0, Some("linear")))
            .unwrap();
        tracker
            .record_metric(&make_metric("pr_created", 20.0, Some("linear")))
            .unwrap();
        tracker
            .record_metric(&make_metric("poll_cycle_duration_secs", 2.5, None))
            .unwrap();
        tracker
            .record_metric(&make_metric("active_processing", 3.0, None))
            .unwrap();
        tracker
            .record_metric(&make_metric("pending_attempts", 5.0, None))
            .unwrap();
        tracker
            .record_metric(&make_metric("total_attempts", 50.0, None))
            .unwrap();

        let response = router
            .oneshot(auth_get("/api/telemetry/pipeline?period=week", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let resp: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(resp["period"], "week");
        assert!(resp["totals"]["fetched"].as_f64().unwrap() > 0.0);
        assert!(resp["totals"]["matched"].as_f64().unwrap() > 0.0);
        assert!(resp["conversion"]["match_rate"].as_f64().is_some());
        assert!(resp["poll_load"]["poll_cycles"].as_i64().unwrap() > 0);
        assert!(!resp["per_source"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_telemetry_pipeline_hour_period() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/telemetry/pipeline?period=hour", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let resp: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(resp["period"], "hour");
    }

    #[tokio::test]
    async fn test_issues_with_seeded_data() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let issue = claudear_core::types::IssueEmbedding {
            id: 0,
            source: "linear".to_string(),
            issue_id: "LIN-100".to_string(),
            short_id: Some("LIN-100".to_string()),
            title: Some("Fix authentication bug".to_string()),
            description: Some("Users unable to log in".to_string()),
            url: Some("https://linear.app/issue/LIN-100".to_string()),
            priority: Some("high".to_string()),
            status: Some("open".to_string()),
            labels: Some(r#"["bug","auth"]"#.to_string()),
            embedding: None,
            embedding_model: None,
            created_at: chrono::Utc::now(),
            updated_at: None,
        };
        tracker.store_issue(&issue).unwrap();

        let response = router
            .oneshot(auth_get("/api/issues", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let resp: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(resp["total"], 1);
        assert_eq!(resp["page"], 1);
        let issues = resp["issues"].as_array().unwrap();
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0]["source"], "linear");
        assert_eq!(issues[0]["issue_id"], "LIN-100");
        assert_eq!(issues[0]["title"], "Fix authentication bug");
        assert_eq!(issues[0]["priority"], "high");
        assert_eq!(issues[0]["has_embedding"], false);
        let labels = issues[0]["labels"].as_array().unwrap();
        assert_eq!(labels.len(), 2);
    }

    #[tokio::test]
    async fn test_issues_pagination_with_data() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        for i in 1..=5 {
            let issue = claudear_core::types::IssueEmbedding {
                id: 0,
                source: "linear".to_string(),
                issue_id: format!("LIN-{}", i),
                short_id: Some(format!("LIN-{}", i)),
                title: Some(format!("Issue {}", i)),
                description: None,
                url: None,
                priority: None,
                status: None,
                labels: None,
                embedding: None,
                embedding_model: None,
                created_at: chrono::Utc::now(),
                updated_at: None,
            };
            tracker.store_issue(&issue).unwrap();
        }

        let response = router
            .oneshot(auth_get("/api/issues?page=2&per_page=2", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let resp: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(resp["total"], 5);
        assert_eq!(resp["page"], 2);
        assert_eq!(resp["per_page"], 2);
        let issues = resp["issues"].as_array().unwrap();
        assert_eq!(issues.len(), 2);
    }

    #[tokio::test]
    async fn test_prs_with_seeded_data() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let pr = claudear_core::types::PrRecord {
            id: 0,
            pr_url: "https://github.com/org/repo/pull/1".to_string(),
            scm_repo: "org/repo".to_string(),
            pr_number: 1,
            attempt_id: None,
            issue_id: Some("LIN-1".to_string()),
            issue_source: Some("linear".to_string()),
            title: Some("Fix auth bug".to_string()),
            description: None,
            author: Some("claudear-bot".to_string()),
            head_branch: Some("fix/lin-1".to_string()),
            base_branch: Some("main".to_string()),
            status: "open".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: None,
            merged_at: None,
            closed_at: None,
            approvals_count: 0,
            changes_requested_count: 0,
            comments_count: 0,
            last_review_at: None,
            time_to_first_review_mins: None,
            time_to_merge_mins: None,
            review_cycles: 0,
            files_changed: Some(3),
            lines_added: Some(50),
            lines_removed: Some(10),
        };
        tracker.upsert_pr(&pr).unwrap();

        let response = router.oneshot(auth_get("/api/prs", &token)).await.unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let prs: Vec<serde_json::Value> = serde_json::from_slice(&body).unwrap();
        assert_eq!(prs.len(), 1);
        assert_eq!(prs[0]["status"], "open");
        assert_eq!(prs[0]["pr_url"], "https://github.com/org/repo/pull/1");
    }

    #[tokio::test]
    async fn test_pr_analytics_with_seeded_prs() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let pr_open = claudear_core::types::PrRecord {
            id: 0,
            pr_url: "https://github.com/org/repo/pull/1".to_string(),
            scm_repo: "org/repo".to_string(),
            pr_number: 1,
            attempt_id: None,
            issue_id: None,
            issue_source: None,
            title: Some("Fix bug 1".to_string()),
            description: None,
            author: None,
            head_branch: None,
            base_branch: None,
            status: "open".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: None,
            merged_at: None,
            closed_at: None,
            approvals_count: 0,
            changes_requested_count: 0,
            comments_count: 0,
            last_review_at: None,
            time_to_first_review_mins: None,
            time_to_merge_mins: None,
            review_cycles: 0,
            files_changed: None,
            lines_added: None,
            lines_removed: None,
        };
        let mut pr_merged = pr_open.clone();
        pr_merged.pr_url = "https://github.com/org/repo/pull/2".to_string();
        pr_merged.pr_number = 2;
        pr_merged.status = "merged".to_string();
        pr_merged.merged_at = Some(chrono::Utc::now());

        tracker.upsert_pr(&pr_open).unwrap();
        tracker.upsert_pr(&pr_merged).unwrap();

        let response = router
            .oneshot(auth_get("/api/prs/analytics", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let analytics: claudear_core::types::PrAnalytics = serde_json::from_slice(&body).unwrap();
        assert_eq!(analytics.open, 1);
        assert_eq!(analytics.merged, 1);
    }

    #[tokio::test]
    async fn test_feedback_with_seeded_data() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        seed_attempt(&tracker, "linear", "issue-1", "PROJ-1");
        tracker
            .mark_success("linear", "issue-1", "https://github.com/org/repo/pull/1")
            .unwrap();

        let outcome = claudear_analysis::feedback::FixOutcome {
            id: 0,
            attempt_id: 1,
            source: "linear".to_string(),
            issue_id: "issue-1".to_string(),
            issue_text: "Fix login bug".to_string(),
            prompt_used: "fix this issue".to_string(),
            outcome: claudear_analysis::feedback::Outcome::Merged,
            error_type: None,
            learnings: Some("Always check auth middleware".to_string()),
            keywords: vec!["auth".to_string(), "login".to_string()],
            embedding: None,
            created_at: chrono::Utc::now(),
        };
        tracker.store_feedback_outcome(&outcome).unwrap();

        let response = router
            .oneshot(auth_get("/api/feedback", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let feedback: Vec<serde_json::Value> = serde_json::from_slice(&body).unwrap();
        assert_eq!(feedback.len(), 1);
        assert_eq!(feedback[0]["source"], "linear");
        assert_eq!(feedback[0]["outcome"], "merged");
    }

    #[tokio::test]
    async fn test_attempt_full_detail_with_executions() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        seed_attempt(&tracker, "linear", "issue-1", "PROJ-1");

        let mut execution = claudear_core::types::AgentExecution::new();
        execution.attempt_id = Some(1);
        execution.completed_at = Some(chrono::Utc::now());
        execution.duration_secs = Some(30.0);
        execution.exit_code = Some(0);
        execution.stdout_preview = Some("All tests passed".to_string());
        execution.prompt_used = Some("Fix the bug".to_string());
        tracker.record_execution(&execution).unwrap();

        let response = router
            .oneshot(auth_get("/api/attempts/1/detail", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let resp: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(resp["attempt"]["source"], "linear");
        let executions = resp["executions"].as_array().unwrap();
        assert_eq!(executions.len(), 1);
        assert_eq!(executions[0]["duration_secs"], 30.0);
    }

    #[tokio::test]
    async fn test_attempt_execution_log_valid_streams() {
        let tracker = create_test_tracker();

        // Test that "events" is also a valid stream
        let password_hash = bcrypt::hash("testpass", 4).unwrap();
        tracker
            .create_user("test@example.com", &password_hash, "Test", "admin")
            .unwrap();
        let token = tracker.create_session(1, "2099-12-31 23:59:59").unwrap();

        let indexing_rx = test_indexing_rx(&tracker);
        let router = create_api_router(
            test_config(),
            tracker.clone(),
            PathBuf::from("claudear.toml"),
            indexing_rx,
        )
        .layer(CookieManagerLayer::new());

        seed_attempt(&tracker, "linear", "issue-1", "PROJ-1");

        let mut execution = claudear_core::types::AgentExecution::new();
        execution.attempt_id = Some(1);
        execution.completed_at = Some(chrono::Utc::now());
        execution.duration_secs = Some(10.0);
        execution.exit_code = Some(0);
        execution.stdout_preview = Some("stdout content".to_string());
        execution.stderr_preview = Some("stderr content".to_string());
        let exec_id = tracker.record_execution(&execution).unwrap();

        // Test stdout stream returns fallback preview
        let response = router
            .oneshot(auth_get(
                &format!("/api/attempts/1/logs/{}/stdout", exec_id),
                &token,
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let resp: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(resp["stream"], "stdout");
        assert_eq!(resp["content"], "stdout content");
        assert_eq!(resp["truncated"], false);
    }

    #[tokio::test]
    async fn test_attempt_execution_log_stderr_stream() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        seed_attempt(&tracker, "linear", "issue-1", "PROJ-1");

        let mut execution = claudear_core::types::AgentExecution::new();
        execution.attempt_id = Some(1);
        execution.completed_at = Some(chrono::Utc::now());
        execution.duration_secs = Some(10.0);
        execution.exit_code = Some(1);
        execution.stderr_preview = Some("error output".to_string());
        let exec_id = tracker.record_execution(&execution).unwrap();

        let response = router
            .oneshot(auth_get(
                &format!("/api/attempts/1/logs/{}/stderr", exec_id),
                &token,
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let resp: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(resp["stream"], "stderr");
        assert_eq!(resp["content"], "error output");
    }

    #[tokio::test]
    async fn test_attempt_execution_log_events_stream() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        seed_attempt(&tracker, "linear", "issue-1", "PROJ-1");

        let mut execution = claudear_core::types::AgentExecution::new();
        execution.attempt_id = Some(1);
        execution.completed_at = Some(chrono::Utc::now());
        execution.duration_secs = Some(10.0);
        execution.exit_code = Some(0);
        let exec_id = tracker.record_execution(&execution).unwrap();

        let response = router
            .oneshot(auth_get(
                &format!("/api/attempts/1/logs/{}/events", exec_id),
                &token,
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let resp: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(resp["stream"], "events");
        // No event_log_path and no fallback_preview for events
        assert!(resp["content"].is_null());
    }

    #[tokio::test]
    async fn test_attempt_execution_log_missing_execution() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        seed_attempt(&tracker, "linear", "issue-1", "PROJ-1");

        let response = router
            .oneshot(auth_get("/api/attempts/1/logs/999/stdout", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_put_config_endpoint_success() {
        let tracker = create_test_tracker();

        // Write a temp file to overwrite
        let config_path = std::env::temp_dir().join("claudear_put_test_config.toml");
        std::fs::write(&config_path, "").unwrap();

        // Build a valid config TOML (must pass validate(), which requires a source
        // with a non-empty API key)
        let mut valid_config = test_config();
        valid_config.issues.linear = Some(claudear_config::config::LinearConfig {
            api_key: SecretValue::new("lin_api_test_key_123"),
            ..claudear_config::config::LinearConfig::default()
        });
        let valid_toml = toml::to_string_pretty(&valid_config).unwrap();

        let password_hash = bcrypt::hash("testpass", 4).unwrap();
        tracker
            .create_user("admin@test.com", &password_hash, "Admin", "admin")
            .unwrap();
        let token = tracker.create_session(1, "2099-12-31 23:59:59").unwrap();

        let indexing_rx = test_indexing_rx(&tracker);
        let router = create_api_router(
            test_config(),
            tracker.clone(),
            config_path.clone(),
            indexing_rx,
        )
        .layer(CookieManagerLayer::new());

        let request = Request::builder()
            .method("PUT")
            .uri("/api/config")
            .header("cookie", format!("claudear_session={}", token))
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::to_string(&serde_json::json!({
                    "content": valid_toml
                }))
                .unwrap(),
            ))
            .unwrap();

        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = response.into_body().collect().await.unwrap().to_bytes();
        let resp: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(resp["ok"], true);
        assert!(resp["message"].as_str().unwrap().contains("Config saved"));

        // Verify file was written
        let saved = std::fs::read_to_string(&config_path).unwrap();
        assert!(saved.contains("workspace"));

        // Cleanup
        let _ = std::fs::remove_file(&config_path);
    }

    #[tokio::test]
    async fn test_retries_with_multiple_failed_attempts() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        seed_attempt(&tracker, "linear", "issue-1", "PROJ-1");
        tracker
            .mark_failed("linear", "issue-1", "Compile error")
            .unwrap();

        seed_attempt(&tracker, "sentry", "issue-2", "SENTRY-1");
        tracker
            .mark_failed("sentry", "issue-2", "Test failure")
            .unwrap();

        seed_attempt(&tracker, "linear", "issue-3", "PROJ-3");
        tracker
            .mark_cannot_fix("linear", "issue-3", "Too complex")
            .unwrap();

        let response = router
            .oneshot(auth_get("/api/retries", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let resp: serde_json::Value = serde_json::from_slice(&body).unwrap();

        // Failed attempts should show in retryable list
        let retryable = resp["retryable"].as_array().unwrap();
        assert!(retryable.len() >= 2); // At least the 2 failed ones
    }

    #[tokio::test]
    async fn test_overview_zero_totals() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/stats/overview", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let resp: serde_json::Value = serde_json::from_slice(&body).unwrap();

        // With no data, rates should be 0
        assert!((resp["success_rate"].as_f64().unwrap() - 0.0).abs() < f64::EPSILON);
        assert!((resp["merge_rate"].as_f64().unwrap() - 0.0).abs() < f64::EPSILON);
        assert_eq!(resp["stats"]["total"], 0);
    }

    #[tokio::test]
    async fn test_metrics_with_week_period() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        tracker
            .record_metric(&make_metric("processing_time", 5.0, None))
            .unwrap();

        let response = router
            .oneshot(auth_get("/api/metrics?period=week", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let metrics: Vec<serde_json::Value> = serde_json::from_slice(&body).unwrap();
        assert_eq!(metrics.len(), 1);
    }

    #[tokio::test]
    async fn test_metrics_with_hour_period() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        tracker
            .record_metric(&make_metric("processing_time", 5.0, None))
            .unwrap();

        let response = router
            .oneshot(auth_get("/api/metrics?period=hour", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_metrics_with_month_period() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        tracker
            .record_metric(&make_metric("processing_time", 5.0, None))
            .unwrap();

        let response = router
            .oneshot(auth_get("/api/metrics?period=month", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_metrics_with_invalid_period_returns_all() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        tracker
            .record_metric(&make_metric("processing_time", 5.0, None))
            .unwrap();

        let response = router
            .oneshot(auth_get("/api/metrics?period=bogus", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let metrics: Vec<serde_json::Value> = serde_json::from_slice(&body).unwrap();
        // Invalid period gives None for since, so returns all metrics
        assert_eq!(metrics.len(), 1);
    }

    #[tokio::test]
    async fn test_metrics_limit_capping() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        // Insert 5 metrics
        for i in 0..5 {
            tracker
                .record_metric(&make_metric("processing_time", i as f64, None))
                .unwrap();
        }

        // Request with limit=3
        let response = router
            .oneshot(auth_get("/api/metrics?limit=3", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let metrics: Vec<serde_json::Value> = serde_json::from_slice(&body).unwrap();
        assert_eq!(metrics.len(), 3);
    }

    #[tokio::test]
    async fn test_activity_with_limit_and_source() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        for i in 0..5 {
            let entry = claudear_core::types::ActivityLogEntry::new(
                "issue_received",
                format!("Received PROJ-{}", i),
            )
            .with_source("linear")
            .with_issue(format!("issue-{}", i), format!("PROJ-{}", i));
            tracker.record_activity(&entry).unwrap();
        }

        let response = router
            .oneshot(auth_get("/api/activity?limit=2&source=linear", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let entries: Vec<serde_json::Value> = serde_json::from_slice(&body).unwrap();
        assert_eq!(entries.len(), 2);
    }

    #[tokio::test]
    async fn test_activity_limit_capped_at_500() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/activity?limit=9999", &token))
            .await
            .unwrap();

        // Should succeed even with a huge limit (capped to 500 internally)
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_errors_limit_capped_at_200() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/errors?limit=9999", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_regressions_with_monitoring_status() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/regressions?status=monitoring", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_regressions_with_resolved_status() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/regressions?status=resolved", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_regressions_with_regressed_status() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/regressions?status=regressed", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_telemetry_overview_queue_with_regression_watches() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        seed_attempt(&tracker, "linear", "issue-1", "PROJ-1");
        tracker
            .mark_success("linear", "issue-1", "https://github.com/org/repo/pull/1")
            .unwrap();

        let watch = claudear_core::types::RegressionWatch {
            id: 0,
            issue_type: claudear_core::types::IssueType::LinearBug,
            issue_id: "issue-1".to_string(),
            fix_attempt_id: 1,
            status: RegressionWatchStatus::AwaitingRelease,
            pr_merged_at: Some(chrono::Utc::now()),
            monitoring_started_at: None,
            resolved_at: None,
            regressed_at: None,
            created_at: chrono::Utc::now(),
        };
        tracker.create_regression_watch(&watch).unwrap();

        let response = router
            .oneshot(auth_get("/api/telemetry/overview", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let resp: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(resp["queue"]["watches_awaiting_release"], 1);
    }

    #[test]
    fn test_parse_telemetry_period_unknown_with_month_default() {
        let (label, _) = parse_telemetry_period(Some("invalid"), "month");
        assert_eq!(label, "month");
    }

    #[test]
    fn test_parse_telemetry_period_unknown_with_hour_default() {
        let (label, _) = parse_telemetry_period(Some("invalid"), "hour");
        assert_eq!(label, "hour");
    }

    #[test]
    fn test_parse_telemetry_period_unknown_with_day_default() {
        let (label, _) = parse_telemetry_period(Some("invalid"), "day");
        assert_eq!(label, "day");
    }

    #[test]
    fn test_parse_telemetry_period_unknown_with_unknown_default() {
        let (label, _) = parse_telemetry_period(Some("invalid"), "invalid");
        assert_eq!(label, "week");
    }

    #[test]
    fn test_telemetry_pipeline_source_default() {
        let src = TelemetryPipelineSource::default();
        assert_eq!(src.source, "");
        assert!((src.fetched - 0.0).abs() < f64::EPSILON);
        assert!(src.match_rate.is_none());
    }

    #[test]
    fn test_telemetry_processing_time_default() {
        let pt = TelemetryProcessingTime::default();
        assert_eq!(pt.all_time.samples, 0);
        assert_eq!(pt.last_24h.samples, 0);
    }

    #[test]
    fn test_telemetry_latency_by_status_serialization() {
        let lbs = TelemetryLatencyByStatus {
            status: "success".to_string(),
            summary: ProcessingTimeSummary {
                samples: 10,
                avg_secs: Some(5.0),
                p50_secs: Some(4.0),
                p95_secs: Some(8.0),
                p99_secs: Some(9.5),
                max_secs: Some(10.0),
            },
        };
        let json = serde_json::to_string(&lbs).unwrap();
        assert!(json.contains("\"status\":\"success\""));
        assert!(json.contains("\"samples\":10"));
        assert!(json.contains("5.0"));
    }

    #[test]
    fn test_attempt_detail_response_serialization() {
        let resp = AttemptDetailResponse {
            attempt: FixAttempt {
                id: 1,
                source: "linear".to_string(),
                issue_id: "issue-1".to_string(),
                short_id: "P-1".to_string(),
                status: FixAttemptStatus::Success,
                pr_url: None,
                scm_repo: None,
                scm_pr_number: None,
                error_message: None,
                attempted_at: chrono::Utc::now(),
                resolved_at: None,
                merged_at: None,
                retry_count: 0,
                last_retry_at: None,
                issue_labels: vec![],
                parent_attempt_id: None,
                cascade_repo: None,
            },
            executions: vec![],
            reviews: vec![],
            feedback: None,
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"attempt\""));
        assert!(json.contains("\"executions\""));
        assert!(json.contains("\"reviews\""));
        assert!(json.contains("\"feedback\":null"));
    }

    #[tokio::test]
    async fn test_unauthenticated_health_returns_401() {
        let tracker = create_test_tracker();
        let config = test_config();
        let indexing_rx = test_indexing_rx(&tracker);
        let router =
            create_api_router(config, tracker, PathBuf::from("claudear.toml"), indexing_rx)
                .layer(CookieManagerLayer::new());

        let response = router
            .oneshot(
                Request::builder()
                    .uri("/api/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_unauthenticated_telemetry_returns_401() {
        let tracker = create_test_tracker();
        let config = test_config();
        let indexing_rx = test_indexing_rx(&tracker);
        let router =
            create_api_router(config, tracker, PathBuf::from("claudear.toml"), indexing_rx)
                .layer(CookieManagerLayer::new());

        let response = router
            .oneshot(
                Request::builder()
                    .uri("/api/telemetry/overview")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_expired_session_returns_401() {
        let tracker = create_test_tracker();
        let config = test_config();
        let password_hash = bcrypt::hash("testpass", 4).unwrap();
        tracker
            .create_user("test@test.com", &password_hash, "Test", "admin")
            .unwrap();
        // Create a session that already expired
        let token = tracker.create_session(1, "2020-01-01 00:00:00").unwrap();

        let indexing_rx = test_indexing_rx(&tracker);
        let router = create_api_router(
            config,
            tracker.clone(),
            PathBuf::from("claudear.toml"),
            indexing_rx,
        )
        .layer(CookieManagerLayer::new());

        let response = router
            .oneshot(auth_get("/api/health", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_invalid_session_cookie_returns_401() {
        let tracker = create_test_tracker();
        let config = test_config();
        let indexing_rx = test_indexing_rx(&tracker);
        let router =
            create_api_router(config, tracker, PathBuf::from("claudear.toml"), indexing_rx)
                .layer(CookieManagerLayer::new());

        let response = router
            .oneshot(auth_get("/api/health", "invalid-token-value"))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_overview_merge_rate_with_completed() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        // Create 3 attempts: 1 merged, 1 closed, 1 failed
        seed_attempt(&tracker, "linear", "issue-1", "PROJ-1");
        tracker
            .mark_success("linear", "issue-1", "https://github.com/org/repo/pull/1")
            .unwrap();
        tracker.mark_merged("linear", "issue-1").unwrap();

        seed_attempt(&tracker, "linear", "issue-2", "PROJ-2");
        tracker
            .mark_success("linear", "issue-2", "https://github.com/org/repo/pull/2")
            .unwrap();
        tracker.mark_closed("linear", "issue-2").unwrap();

        seed_attempt(&tracker, "linear", "issue-3", "PROJ-3");
        tracker.mark_failed("linear", "issue-3", "Error").unwrap();

        let response = router
            .oneshot(auth_get("/api/stats/overview", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let resp: serde_json::Value = serde_json::from_slice(&body).unwrap();

        // merge_rate = merged / (merged + closed + failed + cannot_fix) * 100
        // = 1 / 3 * 100 = 33.33...
        let merge_rate = resp["merge_rate"].as_f64().unwrap();
        assert!(merge_rate > 30.0 && merge_rate < 35.0);
    }

    #[tokio::test]
    async fn test_telemetry_timeseries_bucket_minutes_clamped() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        // Test that bucket_minutes is clamped to [1, 1440]
        let response = router
            .oneshot(auth_get(
                "/api/telemetry/timeseries?period=day&bucket_minutes=0",
                &token,
            ))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let resp: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(resp["bucket_minutes"], 1);
    }

    #[tokio::test]
    async fn test_telemetry_timeseries_bucket_minutes_max_clamped() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get(
                "/api/telemetry/timeseries?period=day&bucket_minutes=99999",
                &token,
            ))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let resp: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(resp["bucket_minutes"], 1440);
    }

    #[tokio::test]
    async fn test_attempts_page_0_clamped_to_1() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        seed_attempt(&tracker, "linear", "issue-1", "PROJ-1");

        let response = router
            .oneshot(auth_get("/api/attempts?page=0", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let resp: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(resp["page"], 1);
    }

    #[test]
    fn test_telemetry_pipeline_conversion_with_data() {
        let conv = TelemetryPipelineConversion {
            match_rate: ratio(50.0, 100.0),
            queue_rate: ratio(40.0, 50.0),
            processing_rate: ratio(35.0, 40.0),
            pr_yield_rate: ratio(20.0, 35.0),
        };
        assert!((conv.match_rate.unwrap() - 0.5).abs() < 0.001);
        assert!((conv.queue_rate.unwrap() - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_summarize_window_7d_throughput() {
        let now = chrono::Utc::now();
        let attempts: Vec<FixAttempt> = (0..10)
            .map(|i| make_attempt(i, "linear", FixAttemptStatus::Success))
            .collect();
        let metric = summarize_window(&attempts, "7d", now - Duration::days(7));
        // 10 processed / (24 * 7) hours
        let expected_throughput = 10.0 / (24.0 * 7.0);
        assert!((metric.throughput_per_hour - expected_throughput).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_feedback_with_source_filter_returns_matching() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        seed_attempt(&tracker, "linear", "issue-1", "PROJ-1");
        seed_attempt(&tracker, "sentry", "issue-2", "SENTRY-1");

        let outcome1 = claudear_analysis::feedback::FixOutcome {
            id: 0,
            attempt_id: 1,
            source: "linear".to_string(),
            issue_id: "issue-1".to_string(),
            issue_text: "Fix linear bug".to_string(),
            prompt_used: "fix".to_string(),
            outcome: claudear_analysis::feedback::Outcome::Merged,
            error_type: None,
            learnings: None,
            keywords: vec![],
            embedding: None,
            created_at: chrono::Utc::now(),
        };
        let outcome2 = claudear_analysis::feedback::FixOutcome {
            id: 0,
            attempt_id: 2,
            source: "sentry".to_string(),
            issue_id: "issue-2".to_string(),
            issue_text: "Fix sentry error".to_string(),
            prompt_used: "fix".to_string(),
            outcome: claudear_analysis::feedback::Outcome::Failed,
            error_type: Some("build_error".to_string()),
            learnings: None,
            keywords: vec![],
            embedding: None,
            created_at: chrono::Utc::now(),
        };
        tracker.store_feedback_outcome(&outcome1).unwrap();
        tracker.store_feedback_outcome(&outcome2).unwrap();

        let response = router
            .oneshot(auth_get("/api/feedback?source=sentry", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let feedback: Vec<serde_json::Value> = serde_json::from_slice(&body).unwrap();
        assert_eq!(feedback.len(), 1);
        assert_eq!(feedback[0]["source"], "sentry");
    }

    #[tokio::test]
    async fn test_analytics_summary_structure() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/analytics/summary", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let resp: serde_json::Value = serde_json::from_slice(&body).unwrap();

        // Verify key fields are present
        assert!(resp.get("total_processed").is_some());
        assert!(resp.get("total_successful").is_some());
        assert!(resp.get("total_merged").is_some());
        assert!(resp.get("success_rate").is_some());
        assert!(resp.get("mttr_trend").is_some());
        assert!(resp.get("repo_leaderboard").is_some());
    }

    #[tokio::test]
    async fn test_inference_history_limit_capped() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/inference/history?limit=9999", &token))
            .await
            .unwrap();

        // Should succeed (limit capped to 500 internally)
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_health_response_contains_package_version() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/health", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let resp: serde_json::Value = serde_json::from_slice(&body).unwrap();

        // Version should match CARGO_PKG_VERSION
        let version = resp["version"].as_str().unwrap();
        assert_eq!(version, env!("CARGO_PKG_VERSION"));
    }

    #[tokio::test]
    async fn test_issues_endpoint_returns_empty_array() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/issues", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let resp: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(resp.is_object());
        assert!(resp["issues"].is_array());
        assert_eq!(resp["total"], 0);
    }

    #[tokio::test]
    async fn test_telemetry_overview_endpoint_response() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/telemetry/overview", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let resp: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(resp.get("generated_at").is_some());
        assert!(resp.get("windows").is_some());
        assert!(resp.get("queue").is_some());
        assert!(resp.get("processing_time").is_some());
    }

    #[tokio::test]
    async fn test_telemetry_overview_returns_windows_array() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/telemetry/overview", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let resp: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(resp["windows"].is_array());
    }

    #[tokio::test]
    async fn test_telemetry_overview_returns_queue_metrics() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/telemetry/overview", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let resp: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(resp["queue"].is_object());
    }

    #[tokio::test]
    async fn test_telemetry_overview_returns_processing_time() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/telemetry/overview", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let resp: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(resp["processing_time"].is_object());
    }

    #[tokio::test]
    async fn test_telemetry_pipeline_endpoint_response() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/telemetry/pipeline", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_telemetry_latency_endpoint_response() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/telemetry/latency", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_get_attempt_records_since_with_recent_data() {
        let tracker = create_test_tracker();

        // Seed an attempt
        tracker
            .record_attempt("linear", "test-issue-since", "TEST-SINCE")
            .unwrap();

        // Query since 1 hour ago -- should include the just-created attempt
        let since = chrono::Utc::now() - chrono::Duration::hours(1);
        let records = get_attempt_records_since(&tracker, since);
        assert_eq!(records.len(), 1);
    }

    #[tokio::test]
    async fn test_get_attempt_records_since_far_future_returns_empty() {
        let tracker = create_test_tracker();

        tracker
            .record_attempt("linear", "test-issue-future", "TEST-FUTURE")
            .unwrap();

        // Query since far in the future -- should exclude everything
        let since = chrono::Utc::now() + chrono::Duration::hours(24);
        let records = get_attempt_records_since(&tracker, since);
        assert!(records.is_empty());
    }

    #[tokio::test]
    async fn test_config_endpoint_response() {
        let tracker = create_test_tracker();
        let config = test_config();

        let password_hash = bcrypt::hash("testpass", 4).unwrap();
        tracker
            .create_user("admin@test.com", &password_hash, "Admin", "admin")
            .unwrap();
        let token = tracker.create_session(1, "2099-12-31 23:59:59").unwrap();

        // Write a temp config file so the handler can read it
        let config_path = std::env::temp_dir().join("claudear_test_config_endpoint_response.toml");
        std::fs::write(&config_path, "# test config\nworkspace = \"/tmp\"\n").unwrap();

        let indexing_rx = test_indexing_rx(&tracker);
        let router = create_api_router(config, tracker.clone(), config_path.clone(), indexing_rx)
            .layer(CookieManagerLayer::new());

        let response = router
            .oneshot(auth_get("/api/config", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let resp: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(resp.is_object());

        let _ = std::fs::remove_file(&config_path);
    }

    #[tokio::test]
    async fn test_repos_endpoint_no_indexed_repos() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/repos", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[test]
    fn test_compute_processing_time_summary_hundred_items() {
        let now = chrono::Utc::now();
        let metrics: Vec<claudear_core::types::ProcessingMetric> = (1..=100)
            .map(|i| claudear_core::types::ProcessingMetric {
                id: i,
                timestamp: now,
                metric_name: "processing_time".to_string(),
                metric_value: i as f64,
                source: None,
                tags: None,
            })
            .collect();

        let result = compute_processing_time_summary(metrics);
        assert_eq!(result.samples, 100);
        assert_eq!(result.avg_secs, Some(50.5));
        assert_eq!(result.max_secs, Some(100.0));
        // p50 should be around 50
        assert!(result.p50_secs.unwrap() >= 49.0 && result.p50_secs.unwrap() <= 51.0);
        // p95 should be around 95
        assert!(result.p95_secs.unwrap() >= 94.0 && result.p95_secs.unwrap() <= 96.0);
        // p99 should be around 99
        assert!(result.p99_secs.unwrap() >= 98.0 && result.p99_secs.unwrap() <= 100.0);
    }

    #[test]
    fn test_knowledge_key_label_known_keys() {
        assert_eq!(
            knowledge_key_label("common_fix_dirs"),
            "Common Fix Directories"
        );
        assert_eq!(knowledge_key_label("file_conventions"), "File Conventions");
        assert_eq!(knowledge_key_label("test_pattern"), "Test Patterns");
        assert_eq!(
            knowledge_key_label("review_preferences"),
            "Review Preferences"
        );
        assert_eq!(
            knowledge_key_label("common_root_causes"),
            "Common Root Causes"
        );
        assert_eq!(knowledge_key_label("promoted_qa"), "Promoted Q&A");
    }

    #[test]
    fn test_knowledge_key_label_unknown_key() {
        assert_eq!(
            knowledge_key_label("something_unknown"),
            "something_unknown"
        );
        assert_eq!(knowledge_key_label(""), "");
    }

    #[tokio::test]
    async fn test_sources_with_whatsapp_configured() {
        let tracker = create_test_tracker();
        let mut config = test_config();
        config.notifiers.whatsapp.source_enabled = true;
        config.notifiers.whatsapp.access_token =
            Some(claudear_core::secret::SecretValue::new("wa_token"));
        config.notifiers.whatsapp.phone_number_id = Some("12345".to_string());

        let password_hash = bcrypt::hash("testpass", 4).unwrap();
        tracker
            .create_user("test@example.com", &password_hash, "Test", "admin")
            .unwrap();
        let token = tracker.create_session(1, "2099-12-31 23:59:59").unwrap();

        let indexing_rx = test_indexing_rx(&tracker);
        let router = create_api_router(
            config,
            tracker.clone(),
            PathBuf::from("claudear.toml"),
            indexing_rx,
        )
        .layer(CookieManagerLayer::new());

        let response = router
            .oneshot(auth_get("/api/sources", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let resp: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let sources = resp["sources"].as_array().unwrap();
        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0]["name"], "whatsapp");
        assert_eq!(sources[0]["config"]["has_access_token"], true);
        assert_eq!(sources[0]["config"]["has_phone_number_id"], true);
    }

    #[tokio::test]
    async fn test_sources_with_telegram_configured() {
        let tracker = create_test_tracker();
        let mut config = test_config();
        config.notifiers.telegram.source_enabled = true;
        config.notifiers.telegram.bot_token =
            Some(claudear_core::secret::SecretValue::new("tg_bot_token"));
        config.notifiers.telegram.chat_id = Some("123456789".to_string());

        let password_hash = bcrypt::hash("testpass", 4).unwrap();
        tracker
            .create_user("test@example.com", &password_hash, "Test", "admin")
            .unwrap();
        let token = tracker.create_session(1, "2099-12-31 23:59:59").unwrap();

        let indexing_rx = test_indexing_rx(&tracker);
        let router = create_api_router(
            config,
            tracker.clone(),
            PathBuf::from("claudear.toml"),
            indexing_rx,
        )
        .layer(CookieManagerLayer::new());

        let response = router
            .oneshot(auth_get("/api/sources", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let resp: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let sources = resp["sources"].as_array().unwrap();
        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0]["name"], "telegram");
        assert_eq!(sources[0]["config"]["has_bot_token"], true);
        assert_eq!(sources[0]["config"]["chat_id"], true);
    }

    #[tokio::test]
    async fn test_repo_learning_endpoint_empty() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/repos/test-org%2Ftest-repo/learning", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let resp: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(resp["repo"], "test-org/test-repo");
        assert_eq!(resp["knowledge_total"], 0);
        assert!(resp["knowledge"].as_array().unwrap().is_empty());
        assert!(resp["instructions"].as_array().unwrap().is_empty());
        assert!(resp["review_patterns"].as_array().unwrap().is_empty());
        assert_eq!(resp["review_pattern_summary"]["total_patterns"], 0);
        assert_eq!(resp["review_pattern_summary"]["promoted_count"], 0);
        assert!(resp["strategies"].as_array().unwrap().is_empty());
        assert!(resp["diff_analyses"].as_array().unwrap().is_empty());
        assert!(resp["correlations"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_users_list_endpoint() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/users", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let users: Vec<serde_json::Value> = serde_json::from_slice(&body).unwrap();
        // Should have at least the test user created by create_authenticated_router
        assert!(!users.is_empty());
        assert_eq!(users[0]["email"], "test@example.com");
    }

    #[tokio::test]
    async fn test_create_user_endpoint() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_post_json(
                "/api/users",
                &token,
                serde_json::json!({
                    "email": "newuser@example.com",
                    "password": "securepass123",
                    "name": "New User",
                    "role": "viewer"
                }),
            ))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::CREATED);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let user: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(user["email"], "newuser@example.com");
        assert_eq!(user["name"], "New User");
        assert_eq!(user["role"], "viewer");
    }

    #[tokio::test]
    async fn test_create_user_duplicate_email() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        // test@example.com already exists from create_authenticated_router
        let response = router
            .oneshot(auth_post_json(
                "/api/users",
                &token,
                serde_json::json!({
                    "email": "test@example.com",
                    "password": "password123",
                    "name": "Duplicate",
                    "role": "viewer"
                }),
            ))
            .await
            .unwrap();

        // Should fail with conflict
        assert_eq!(response.status(), StatusCode::CONFLICT);
    }

    #[tokio::test]
    async fn test_get_user_by_id() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/users/1", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let user: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(user["email"], "test@example.com");
    }

    #[tokio::test]
    async fn test_get_user_not_found() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_get("/api/users/99999", &token))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_update_user_endpoint() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        let response = router
            .oneshot(auth_put_json(
                "/api/users/1",
                &token,
                serde_json::json!({
                    "name": "Updated Name",
                    "role": "admin"
                }),
            ))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_delete_user_endpoint() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        // Create a second user to delete
        tracker
            .create_user(
                "todelete@example.com",
                &bcrypt::hash("pass", 4).unwrap(),
                "Delete Me",
                "viewer",
            )
            .unwrap();

        let request = Request::builder()
            .method("DELETE")
            .uri("/api/users/2")
            .header("cookie", format!("claudear_session={}", token))
            .body(Body::empty())
            .unwrap();

        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_users_endpoint_viewer_forbidden() {
        let tracker = create_test_tracker();
        let config = test_config();
        let password_hash = bcrypt::hash("testpass", 4).unwrap();
        tracker
            .create_user("viewer@test.com", &password_hash, "Viewer", "viewer")
            .unwrap();
        let token = tracker.create_session(1, "2099-12-31 23:59:59").unwrap();

        let indexing_rx = test_indexing_rx(&tracker);
        let router = create_api_router(
            config,
            tracker.clone(),
            PathBuf::from("claudear.toml"),
            indexing_rx,
        )
        .layer(CookieManagerLayer::new());

        // Viewer can list users (GET) but not create (POST)
        let response = router
            .oneshot(auth_post_json(
                "/api/users",
                &token,
                serde_json::json!({
                    "email": "new@example.com",
                    "password": "password123",
                    "name": "New User",
                    "role": "viewer"
                }),
            ))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn test_put_config_endpoint_valid_toml_fails_validation() {
        let tracker = create_test_tracker();
        let (router, token) = create_authenticated_router(&tracker);

        // Valid TOML that fails Config::validate() (no sources configured)
        let request = Request::builder()
            .method("PUT")
            .uri("/api/config")
            .header("cookie", format!("claudear_session={}", token))
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::to_string(&serde_json::json!({
                    "content": "workspace = \"/tmp/repos\"\n"
                }))
                .unwrap(),
            ))
            .unwrap();

        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn test_knowledge_group_serialization() {
        let group = KnowledgeGroup {
            key: "test_pattern".to_string(),
            label: "Test Patterns".to_string(),
            entries: vec![KnowledgeEntry {
                id: 1,
                value: "always run unit tests".to_string(),
                source_type: "feedback".to_string(),
                confidence: 0.95,
                occurrence_count: 3,
                updated_at: chrono::Utc::now(),
            }],
        };
        let json = serde_json::to_string(&group).unwrap();
        assert!(json.contains("test_pattern"));
        assert!(json.contains("Test Patterns"));
        assert!(json.contains("always run unit tests"));
        assert!(json.contains("0.95"));
    }

    #[test]
    fn test_review_pattern_summary_serialization() {
        let mut by_category = HashMap::new();
        by_category.insert("security".to_string(), 5);
        by_category.insert("missing_tests".to_string(), 3);
        let summary = ReviewPatternSummary {
            total_patterns: 8,
            by_category,
            promoted_count: 2,
        };
        let json = serde_json::to_string(&summary).unwrap();
        assert!(json.contains("\"total_patterns\":8"));
        assert!(json.contains("\"promoted_count\":2"));
        assert!(json.contains("security"));
    }

    #[test]
    fn test_compute_processing_time_summary_empty() {
        let metrics: Vec<claudear_core::types::ProcessingMetric> = vec![];
        let summary = compute_processing_time_summary(metrics);
        assert_eq!(summary.samples, 0);
        assert!(summary.avg_secs.is_none());
        assert!(summary.max_secs.is_none());
    }
}
