//! Webhook auto-configuration orchestrator.

use crate::github_app::GitHubAppAuth;
use crate::webhook::linear_api::LinearApiClient;
use crate::webhook::sentry_api::SentryApiClient;
use claudear_config::config::{
    Config, DiscordNotifierConfig, GitHubConfig, GitLabConfig, JiraConfig, LinearConfig,
    SentryConfig, SlackSourceConfig, TelegramConfig, WhatsAppConfig,
};
use claudear_config::env_writer::update_env_file;
use claudear_core::error::{Error, Result};
use serde::Deserialize;
use std::collections::HashMap;
use uuid::Uuid;

/// Result of webhook auto-configuration.
#[derive(Debug, Default)]
pub struct WebhookSetupResult {
    /// Linear webhook was configured.
    pub linear_configured: bool,
    /// Linear webhook ID.
    pub linear_webhook_id: Option<String>,
    /// Linear webhook secret (for display only - also written to .env).
    pub linear_secret: Option<String>,
    /// Sentry webhooks were configured (one per project).
    pub sentry_configured: bool,
    /// Number of Sentry projects configured.
    pub sentry_project_count: usize,
    /// Sentry webhook secret (for display only - also written to .env).
    /// Note: If multiple projects, they may have different secrets. We use the first one.
    pub sentry_secret: Option<String>,
    /// Errors encountered during setup (non-fatal).
    pub warnings: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct GitHubHookListItem {
    #[serde(default)]
    config: GitHubHookConfig,
}

#[derive(Debug, Default, Deserialize)]
struct GitHubHookConfig {
    #[serde(default)]
    url: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GitLabHookListItem {
    #[serde(default)]
    url: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TelegramApiResponse {
    ok: bool,
    #[serde(default)]
    description: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SlackApiOkResponse {
    ok: bool,
    #[serde(default)]
    error: Option<String>,
}

#[derive(Debug, Deserialize)]
struct DiscordWebhookCreateResponse {
    id: String,
    #[serde(default)]
    token: Option<String>,
}

/// Orchestrates webhook auto-configuration for supported webhook services.
pub struct WebhookConfigurator {
    config: Config,
    env_path: std::path::PathBuf,
}

impl WebhookConfigurator {
    /// Create a new webhook configurator.
    ///
    /// # Arguments
    /// * `config` - The application configuration
    /// * `env_path` - Path to the .env file for writing secrets
    pub fn new(config: Config, env_path: impl Into<std::path::PathBuf>) -> Self {
        Self {
            config,
            env_path: env_path.into(),
        }
    }

    /// Run the webhook auto-configuration.
    ///
    /// This will:
    /// 1. Create webhooks for enabled auto-configurable services
    ///    (Linear, Sentry, GitHub PAT/App, GitLab, Jira, Telegram, Slack, WhatsApp,
    ///    and Discord notifier webhook URLs)
    /// 2. Emit notes for enabled services that currently require manual setup
    /// 2. Write the returned secrets to the .env file
    ///
    /// # Arguments
    /// * `base_url` - The public URL where webhooks will be received
    pub async fn configure(&self, base_url: &str) -> Result<WebhookSetupResult> {
        tracing::info!("Starting webhook auto-configuration...");
        tracing::info!("Base URL: {}", base_url);

        let mut result = WebhookSetupResult::default();
        let mut env_updates: HashMap<String, String> = HashMap::new();
        let mut configured_any = false;
        let mut attempted_auto_config = false;
        let mut failure_warnings: Vec<String> = Vec::new();
        let mut notes: Vec<String> = Vec::new();

        // Configure Linear webhook
        if let Some(linear_config) = self.config.linear() {
            if linear_config.enabled {
                attempted_auto_config = true;
                match self.configure_linear(linear_config, base_url).await {
                    Ok((webhook_id, secret)) => {
                        result.linear_configured = true;
                        result.linear_webhook_id = Some(webhook_id);
                        result.linear_secret = Some(secret.clone());
                        env_updates.insert("LINEAR_WEBHOOK_SECRET".to_string(), secret);
                        configured_any = true;
                        tracing::info!("Linear webhook configured successfully");
                    }
                    Err(e) => {
                        let warning = format!("Failed to configure Linear webhook: {}", e);
                        tracing::warn!("{}", warning);
                        failure_warnings.push(warning);
                    }
                }
            }
        }

        // Configure Sentry webhooks
        if let Some(ref sentry_config) = self.config.issues.sentry {
            if sentry_config.enabled {
                attempted_auto_config = true;
                match self.configure_sentry(sentry_config, base_url).await {
                    Ok((count, secret)) => {
                        result.sentry_configured = true;
                        result.sentry_project_count = count;
                        if let Some(s) = secret {
                            result.sentry_secret = Some(s.clone());
                            env_updates.insert("SENTRY_CLIENT_SECRET".to_string(), s);
                        }
                        configured_any = true;
                        tracing::info!("Sentry webhooks configured for {} project(s)", count);
                    }
                    Err(e) => {
                        let warning = format!("Failed to configure Sentry webhooks: {}", e);
                        tracing::warn!("{}", warning);
                        failure_warnings.push(warning);
                    }
                }
            }
        }

        // Configure Jira issue webhooks (/webhook/jira)
        if let Some(jira_config) = self.config.jira() {
            if jira_config.enabled {
                if !jira_config.base_url.trim().is_empty() && !jira_config.api_token.is_empty() {
                    attempted_auto_config = true;
                    match self.configure_jira(jira_config, base_url).await {
                        Ok(created) => {
                            configured_any = true;
                            let note = if created {
                                "Jira issue webhook configured".to_string()
                            } else {
                                "Jira issue webhook already configured".to_string()
                            };
                            tracing::info!("{}", note);
                            notes.push(note);
                        }
                        Err(e) => {
                            let warning = format!("Failed to configure Jira webhook: {}", e);
                            tracing::warn!("{}", warning);
                            failure_warnings.push(warning);
                        }
                    }
                } else if jira_config.base_url.trim().is_empty() {
                    notes.push(
                        "Jira is enabled but jira.base_url is empty; skipping Jira webhook auto-setup."
                            .to_string(),
                    );
                } else {
                    notes.push(
                        "Jira is enabled but jira.api_token is empty; skipping Jira webhook auto-setup."
                            .to_string(),
                    );
                }
            }
        }

        // Discord source is polling-based (no inbound webhook registration API).
        if self.config.issues.discord.is_some() {
            notes.push(
                "Discord source uses channel polling (bot API) and does not support inbound webhook auto-setup."
                    .to_string(),
            );
        }

        // Configure Telegram webhook (/webhook/telegram)
        if self.config.notifiers.telegram.source_enabled {
            let telegram_config = &self.config.notifiers.telegram;
            let has_bot_token = telegram_config
                .bot_token
                .as_ref()
                .is_some_and(|s| !s.expose().trim().is_empty());
            if has_bot_token {
                attempted_auto_config = true;
                match self.configure_telegram(telegram_config, base_url).await {
                    Ok(generated_secret) => {
                        configured_any = true;
                        if let Some(secret) = generated_secret {
                            env_updates.insert("TELEGRAM_WEBHOOK_SECRET".to_string(), secret);
                        }
                        notes.push("Telegram webhook configured (Bot API setWebhook)".to_string());
                    }
                    Err(e) => {
                        let warning = format!("Failed to configure Telegram webhook: {}", e);
                        tracing::warn!("{}", warning);
                        failure_warnings.push(warning);
                    }
                }
            } else {
                notes.push(
                    "Telegram source is enabled but telegram.bot_token is missing; skipping Telegram webhook auto-setup."
                        .to_string(),
                );
            }
        }

        // Configure Discord notifier webhook URL (channel webhook -> DISCORD_WEBHOOK_URL)
        let discord_notifier = &self.config.notifiers.discord;
        let discord_webhook_present = discord_notifier
            .webhook_url
            .as_ref()
            .is_some_and(|s| !s.expose().trim().is_empty());
        let discord_bot_present = discord_notifier
            .bot_token
            .as_ref()
            .is_some_and(|s| !s.expose().trim().is_empty());
        let discord_channel_present = discord_notifier
            .channel_id
            .as_ref()
            .is_some_and(|s| !s.trim().is_empty());

        if !discord_webhook_present {
            if discord_bot_present && discord_channel_present {
                attempted_auto_config = true;
                match self.configure_discord_notifier(discord_notifier).await {
                    Ok(webhook_url) => {
                        configured_any = true;
                        env_updates.insert("DISCORD_WEBHOOK_URL".to_string(), webhook_url);
                        notes.push(
                            "Discord notifier webhook URL auto-created for configured channel"
                                .to_string(),
                        );
                    }
                    Err(e) => {
                        let warning =
                            format!("Failed to configure Discord notifier webhook URL: {}", e);
                        tracing::warn!("{}", warning);
                        failure_warnings.push(warning);
                    }
                }
            } else if discord_bot_present || discord_channel_present {
                let mut missing = Vec::new();
                if !discord_bot_present {
                    missing.push("discord.bot_token");
                }
                if !discord_channel_present {
                    missing.push("discord.channel_id");
                }
                notes.push(format!(
                    "Discord notifier webhook auto-setup requires {} (or set discord.webhook_url manually).",
                    missing.join(", ")
                ));
            }
        }

        // Configure Slack Events API webhook (manifest API -> /webhook/slack)
        if let Some(slack_config) = self.config.issues.slack.as_ref() {
            let has_app_id = slack_config
                .app_id
                .as_ref()
                .is_some_and(|s| !s.trim().is_empty());
            let has_manifest_token = slack_config
                .app_config_token
                .as_ref()
                .is_some_and(|s| !s.expose().trim().is_empty());
            let has_signing_secret = slack_config
                .signing_secret
                .as_ref()
                .is_some_and(|s| !s.expose().trim().is_empty());

            if has_app_id && has_manifest_token && has_signing_secret {
                attempted_auto_config = true;
                match self.configure_slack(slack_config, base_url).await {
                    Ok(updated) => {
                        configured_any = true;
                        let note = if updated {
                            "Slack Events API callback configured via apps.manifest.update"
                                .to_string()
                        } else {
                            "Slack Events API callback already configured".to_string()
                        };
                        tracing::info!("{}", note);
                        notes.push(note);
                    }
                    Err(e) => {
                        let warning = format!("Failed to configure Slack webhook: {}", e);
                        tracing::warn!("{}", warning);
                        failure_warnings.push(warning);
                    }
                }
            } else {
                let mut missing = Vec::new();
                if !has_app_id {
                    missing.push("slack.app_id");
                }
                if !has_manifest_token {
                    missing.push("slack.app_config_token");
                }
                if !has_signing_secret {
                    missing.push("slack.signing_secret");
                }
                notes.push(format!(
                    "Slack source is configured, but webhook auto-setup requires {}.",
                    missing.join(", ")
                ));
            }
        }

        let slack_notifier = &self.config.notifiers.slack;
        let slack_notifier_has_webhook = slack_notifier
            .webhook_url
            .as_ref()
            .is_some_and(|s| !s.expose().trim().is_empty());
        let slack_notifier_has_bot_channel = slack_notifier
            .bot_token
            .as_ref()
            .is_some_and(|s| !s.expose().trim().is_empty())
            && slack_notifier
                .channel_id
                .as_ref()
                .is_some_and(|s| !s.trim().is_empty());
        if !slack_notifier_has_webhook && slack_notifier_has_bot_channel {
            notes.push(
                "Slack notifier incoming webhook URL is not auto-provisioned here; notifier can use slack.bot_token + slack.channel_id (chat.postMessage)."
                    .to_string(),
            );
        }

        // Configure WhatsApp webhook subscription (WABA subscribed_apps -> /webhook/whatsapp)
        if self.config.notifiers.whatsapp.source_enabled {
            let wa = &self.config.notifiers.whatsapp;
            let has_access_token = wa
                .access_token
                .as_ref()
                .is_some_and(|s| !s.expose().trim().is_empty());
            let has_business_account_id = wa
                .business_account_id
                .as_ref()
                .is_some_and(|s| !s.trim().is_empty());
            let has_app_secret = wa
                .app_secret
                .as_ref()
                .is_some_and(|s| !s.expose().trim().is_empty());

            if has_access_token && has_business_account_id && has_app_secret {
                attempted_auto_config = true;
                match self.configure_whatsapp(wa, base_url).await {
                    Ok(generated_verify_token) => {
                        configured_any = true;
                        if let Some(token) = generated_verify_token {
                            env_updates.insert("WHATSAPP_WEBHOOK_VERIFY_TOKEN".to_string(), token);
                        }
                        notes.push(
                            "WhatsApp webhook subscription configured (WABA subscribed_apps)"
                                .to_string(),
                        );
                    }
                    Err(e) => {
                        let warning = format!("Failed to configure WhatsApp webhook: {}", e);
                        tracing::warn!("{}", warning);
                        failure_warnings.push(warning);
                    }
                }
            } else {
                let mut missing = Vec::new();
                if !has_access_token {
                    missing.push("whatsapp.access_token");
                }
                if !has_business_account_id {
                    missing.push("whatsapp.business_account_id");
                }
                if !has_app_secret {
                    missing.push("whatsapp.app_secret");
                }
                notes.push(format!(
                    "WhatsApp source is enabled, but webhook auto-setup requires {}.",
                    missing.join(", ")
                ));
            }
        }

        // Configure GitLab issue webhooks (group hooks -> /webhook/gitlab)
        if let Some(gitlab_config) = self.config.gitlab() {
            if gitlab_config.enabled {
                if !gitlab_config.groups.is_empty() && gitlab_config.token.is_some() {
                    attempted_auto_config = true;
                    match self.configure_gitlab(gitlab_config, base_url).await {
                        Ok((group_count, generated_secret)) => {
                            configured_any = true;
                            if let Some(secret) = generated_secret {
                                env_updates.insert("GITLAB_WEBHOOK_SECRET".to_string(), secret);
                            }
                            let note = format!(
                                "GitLab issue webhooks configured for {} group(s)",
                                group_count
                            );
                            tracing::info!("{}", note);
                            notes.push(note);
                        }
                        Err(e) => {
                            let warning = format!("Failed to configure GitLab webhooks: {}", e);
                            tracing::warn!("{}", warning);
                            failure_warnings.push(warning);
                        }
                    }
                } else if gitlab_config.groups.is_empty() {
                    notes.push(
                        "GitLab is enabled but no groups are configured; skipping GitLab webhook auto-setup."
                            .to_string(),
                    );
                } else if gitlab_config.token.is_none() {
                    notes.push(
                        "GitLab is enabled but no token is configured; skipping GitLab webhook auto-setup."
                            .to_string(),
                    );
                }
            }
        }

        // Configure GitHub review webhooks (repo hooks -> /webhook/github)
        let github_config = self.config.github();
        if !github_config.repos.is_empty() && github_config.token.is_some() {
            attempted_auto_config = true;
            match self.configure_github(github_config, base_url).await {
                Ok((repo_count, generated_secret)) => {
                    configured_any = true;
                    if let Some(secret) = generated_secret {
                        env_updates.insert("GITHUB_WEBHOOK_SECRET".to_string(), secret);
                    }
                    let note = format!(
                        "GitHub review webhooks configured for {} repo(s)",
                        repo_count
                    );
                    tracing::info!("{}", note);
                    notes.push(note);
                }
                Err(e) => {
                    let warning = format!("Failed to configure GitHub webhooks: {}", e);
                    tracing::warn!("{}", warning);
                    failure_warnings.push(warning);
                }
            }
        } else if !github_config.repos.is_empty() && github_config.token.is_none() {
            if self.config.github_app().is_configured() {
                attempted_auto_config = true;
                match self.configure_github_app(github_config, base_url).await {
                    Ok((secret, persist_app_secret, persist_github_secret)) => {
                        configured_any = true;
                        if persist_app_secret {
                            env_updates
                                .insert("GITHUB_APP_WEBHOOK_SECRET".to_string(), secret.clone());
                        }
                        if persist_github_secret {
                            env_updates.insert("GITHUB_WEBHOOK_SECRET".to_string(), secret);
                        }
                        notes
                            .push("GitHub App webhook configured via /app/hook/config".to_string());
                    }
                    Err(e) => {
                        let warning = format!("Failed to configure GitHub App webhook: {}", e);
                        tracing::warn!("{}", warning);
                        failure_warnings.push(warning);
                    }
                }
            } else {
                notes.push(
                    "GitHub repos are configured but no GitHub token is available; skipping GitHub webhook auto-setup.".to_string(),
                );
            }
        }

        // Write secrets to .env file
        if !env_updates.is_empty() {
            tracing::info!("Writing secrets to {}", self.env_path.display());
            update_env_file(&self.env_path, &env_updates)?;
        }

        result.warnings.extend(failure_warnings.clone());
        result.warnings.extend(notes);

        if !configured_any {
            if attempted_auto_config && !failure_warnings.is_empty() {
                return Err(Error::config(format!(
                    "Failed to configure any webhooks: {}",
                    failure_warnings.join("; ")
                )));
            }
            if !attempted_auto_config && result.warnings.is_empty() {
                return Err(Error::config(
                    "No webhook sources are enabled. Enable Linear, Sentry, GitHub (with repos), GitLab (with groups), Jira, Telegram source, Slack source, WhatsApp source, or provide Discord notifier bot_token+channel_id for webhook URL auto-setup in your configuration."
                ));
            }
            if !attempted_auto_config {
                return Err(Error::config(format!(
                    "No auto-configurable webhook services are enabled. {}",
                    result.warnings.join("; ")
                )));
            }
        }

        Ok(result)
    }

    async fn configure_github(
        &self,
        config: &GitHubConfig,
        base_url: &str,
    ) -> Result<(usize, Option<String>)> {
        let token = config
            .token
            .as_ref()
            .ok_or_else(|| Error::config("GitHub token is required for webhook auto-setup"))?
            .expose()
            .to_string();
        if token.is_empty() {
            return Err(Error::config(
                "GitHub token is required for webhook auto-setup",
            ));
        }
        if config.repos.is_empty() {
            return Err(Error::config(
                "GitHub repos must be configured for webhook auto-setup",
            ));
        }

        let callback_url = format!("{}/webhook/github", base_url.trim_end_matches('/'));
        let secret = config
            .webhook_secret
            .as_ref()
            .map(|s| s.expose().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| {
                format!(
                    "ghwh_{}{}",
                    Uuid::new_v4().simple(),
                    Uuid::new_v4().simple()
                )
            });
        let generated_secret = config
            .webhook_secret
            .as_ref()
            .map(|s| s.expose().is_empty())
            .unwrap_or(true)
            .then(|| secret.clone());

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .connect_timeout(std::time::Duration::from_secs(10))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());

        let mut configured_count = 0usize;
        let mut repo_failures: Vec<String> = Vec::new();

        for repo in &config.repos {
            match self
                .ensure_github_repo_webhook(&client, &token, repo, &callback_url, &secret)
                .await
            {
                Ok(_) => configured_count += 1,
                Err(e) => repo_failures.push(format!("{} ({})", repo, e)),
            }
        }

        if configured_count == 0 {
            return Err(Error::api(format!(
                "GitHub webhook setup failed for all repos: {}",
                repo_failures.join("; ")
            )));
        }

        if !repo_failures.is_empty() {
            tracing::warn!(
                "GitHub webhook setup partially succeeded: configured {} repo(s), failed for {}",
                configured_count,
                repo_failures.join("; ")
            );
        }

        Ok((configured_count, generated_secret))
    }

    async fn configure_github_app(
        &self,
        github_config: &GitHubConfig,
        base_url: &str,
    ) -> Result<(String, bool, bool)> {
        let app_config = github_config.app.clone();
        if !app_config.is_configured() {
            return Err(Error::config(
                "GitHub App must be configured (app_id + private key) for webhook auto-setup",
            ));
        }

        let app_secret = app_config
            .webhook_secret
            .as_ref()
            .map(|s| s.expose().trim().to_string())
            .filter(|s| !s.is_empty());
        let github_secret = github_config
            .webhook_secret
            .as_ref()
            .map(|s| s.expose().trim().to_string())
            .filter(|s| !s.is_empty());

        let secret = app_secret
            .clone()
            .or(github_secret.clone())
            .unwrap_or_else(|| {
                format!(
                    "ghappwh_{}{}",
                    Uuid::new_v4().simple(),
                    Uuid::new_v4().simple()
                )
            });
        let persist_app_secret = app_secret.is_none();
        let persist_github_secret = github_secret.is_none();

        let auth = GitHubAppAuth::new(app_config);
        let headers = auth.jwt_headers()?;
        let callback_url = format!("{}/webhook/github", base_url.trim_end_matches('/'));

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .connect_timeout(std::time::Duration::from_secs(10))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());

        let mut request = client.patch("https://api.github.com/app/hook/config");
        for (name, value) in headers {
            request = request.header(name, value);
        }
        request = request.header("User-Agent", "claudear");

        let resp = request
            .json(&serde_json::json!({
                "url": callback_url,
                "content_type": "json",
                "secret": secret,
            }))
            .send()
            .await
            .map_err(|e| Error::api(format!("GitHub App webhook config request failed: {}", e)))?;

        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(Error::api(format!(
                "GitHub App webhook config returned {}: {}",
                status, body
            )));
        }

        Ok((secret, persist_app_secret, persist_github_secret))
    }

    async fn configure_discord_notifier(&self, config: &DiscordNotifierConfig) -> Result<String> {
        let bot_token = config
            .bot_token
            .as_ref()
            .map(|s| s.expose().trim())
            .filter(|s| !s.is_empty())
            .ok_or_else(|| {
                Error::config("Discord bot_token is required for notifier webhook auto-setup")
            })?;
        let channel_id = config
            .channel_id
            .as_ref()
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .ok_or_else(|| {
                Error::config("Discord channel_id is required for notifier webhook auto-setup")
            })?;

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .connect_timeout(std::time::Duration::from_secs(10))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());

        let url = format!(
            "https://discord.com/api/v10/channels/{}/webhooks",
            channel_id
        );
        let resp = client
            .post(&url)
            .header("Authorization", format!("Bot {}", bot_token))
            .header("Content-Type", "application/json")
            .json(&serde_json::json!({
                "name": "claudear",
            }))
            .send()
            .await
            .map_err(|e| Error::api(format!("Discord create webhook request failed: {}", e)))?;

        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(Error::api(format!(
                "Discord create webhook returned {}: {}",
                status, body
            )));
        }

        let parsed: DiscordWebhookCreateResponse = serde_json::from_str(&body)
            .map_err(|e| Error::api(format!("Failed to parse Discord webhook response: {}", e)))?;
        let token = parsed
            .token
            .filter(|t| !t.trim().is_empty())
            .ok_or_else(|| Error::api("Discord webhook response missing token".to_string()))?;

        Ok(format!(
            "https://discord.com/api/webhooks/{}/{}",
            parsed.id, token
        ))
    }

    async fn configure_jira(&self, config: &JiraConfig, base_url: &str) -> Result<bool> {
        let callback_url = format!("{}/webhook/jira", base_url.trim_end_matches('/'));
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .connect_timeout(std::time::Duration::from_secs(10))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        let auth_header = Self::jira_auth_header(config);

        if self
            .ensure_jira_webhook(&client, &auth_header, &config.base_url, &callback_url)
            .await?
        {
            return Ok(true);
        }

        Ok(false)
    }

    async fn configure_telegram(
        &self,
        config: &TelegramConfig,
        base_url: &str,
    ) -> Result<Option<String>> {
        let token = config
            .bot_token
            .as_ref()
            .ok_or_else(|| Error::config("Telegram bot_token is required for webhook auto-setup"))?
            .expose()
            .trim()
            .to_string();
        if token.is_empty() {
            return Err(Error::config(
                "Telegram bot_token is required for webhook auto-setup",
            ));
        }

        let callback_url = format!("{}/webhook/telegram", base_url.trim_end_matches('/'));
        let secret = std::env::var("CLAUDEAR_TELEGRAM_WEBHOOK_SECRET")
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| {
                format!(
                    "tgwh_{}{}",
                    Uuid::new_v4().simple(),
                    Uuid::new_v4().simple()
                )
            });
        let generated_secret = std::env::var("CLAUDEAR_TELEGRAM_WEBHOOK_SECRET")
            .ok()
            .map(|s| s.trim().is_empty())
            .unwrap_or(true)
            .then(|| secret.clone());

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .connect_timeout(std::time::Duration::from_secs(10))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        let url = format!("https://api.telegram.org/bot{}/setWebhook", token);
        let payload = serde_json::json!({
            "url": callback_url,
            "allowed_updates": ["message"],
            "secret_token": secret,
        });
        let resp = client
            .post(&url)
            .json(&payload)
            .send()
            .await
            .map_err(|e| Error::api(format!("Telegram setWebhook request failed: {}", e)))?;

        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(Error::api(format!(
                "Telegram setWebhook returned {}: {}",
                status, body
            )));
        }

        let parsed: TelegramApiResponse = serde_json::from_str(&body).map_err(|e| {
            Error::api(format!(
                "Failed to parse Telegram setWebhook response: {}",
                e
            ))
        })?;
        if !parsed.ok {
            return Err(Error::api(format!(
                "Telegram setWebhook failed: {}",
                parsed
                    .description
                    .unwrap_or_else(|| "unknown error".to_string())
            )));
        }

        Ok(generated_secret)
    }

    async fn configure_slack(&self, config: &SlackSourceConfig, base_url: &str) -> Result<bool> {
        let app_id = config
            .app_id
            .as_ref()
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .ok_or_else(|| Error::config("Slack app_id is required for webhook auto-setup"))?;
        let app_config_token = config
            .app_config_token
            .as_ref()
            .map(|s| s.expose().trim())
            .filter(|s| !s.is_empty())
            .ok_or_else(|| {
                Error::config("Slack app_config_token is required for webhook auto-setup")
            })?;

        let callback_url = format!("{}/webhook/slack", base_url.trim_end_matches('/'));
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .connect_timeout(std::time::Duration::from_secs(10))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());

        let export_resp = client
            .post("https://slack.com/api/apps.manifest.export")
            .header("Authorization", format!("Bearer {}", app_config_token))
            .header("Content-Type", "application/json")
            .json(&serde_json::json!({ "app_id": app_id }))
            .send()
            .await
            .map_err(|e| Error::api(format!("Slack apps.manifest.export request failed: {}", e)))?;
        let export_status = export_resp.status();
        let export_body = export_resp.text().await.unwrap_or_default();
        if !export_status.is_success() {
            return Err(Error::api(format!(
                "Slack apps.manifest.export returned {}: {}",
                export_status, export_body
            )));
        }

        let export_json: serde_json::Value = serde_json::from_str(&export_body)
            .map_err(|e| Error::api(format!("Failed to parse Slack export response: {}", e)))?;
        let ok_resp: SlackApiOkResponse = serde_json::from_value(export_json.clone())
            .map_err(|e| Error::api(format!("Failed to parse Slack ok/error fields: {}", e)))?;
        if !ok_resp.ok {
            return Err(Error::api(format!(
                "Slack apps.manifest.export failed: {}",
                ok_resp.error.unwrap_or_else(|| "unknown error".to_string())
            )));
        }

        let mut manifest = match export_json.get("manifest") {
            Some(serde_json::Value::Object(obj)) => serde_json::Value::Object(obj.clone()),
            Some(serde_json::Value::String(s)) => serde_json::from_str::<serde_json::Value>(s)
                .map_err(|e| Error::api(format!("Failed to parse Slack manifest JSON: {}", e)))?,
            _ => {
                return Err(Error::api(
                    "Slack apps.manifest.export response missing manifest".to_string(),
                ))
            }
        };

        let changed = Self::ensure_slack_events_subscription(&mut manifest, &callback_url)?;
        if !changed {
            return Ok(false);
        }

        let update_resp = client
            .post("https://slack.com/api/apps.manifest.update")
            .header("Authorization", format!("Bearer {}", app_config_token))
            .header("Content-Type", "application/json")
            .json(&serde_json::json!({
                "app_id": app_id,
                "manifest": serde_json::to_string(&manifest)
                    .map_err(|e| Error::api(format!("Failed to serialize Slack manifest: {}", e)))?
            }))
            .send()
            .await
            .map_err(|e| Error::api(format!("Slack apps.manifest.update request failed: {}", e)))?;
        let update_status = update_resp.status();
        let update_body = update_resp.text().await.unwrap_or_default();
        if !update_status.is_success() {
            return Err(Error::api(format!(
                "Slack apps.manifest.update returned {}: {}",
                update_status, update_body
            )));
        }

        let update_json: SlackApiOkResponse = serde_json::from_str(&update_body)
            .map_err(|e| Error::api(format!("Failed to parse Slack update response: {}", e)))?;
        if !update_json.ok {
            return Err(Error::api(format!(
                "Slack apps.manifest.update failed: {}",
                update_json
                    .error
                    .unwrap_or_else(|| "unknown error".to_string())
            )));
        }

        Ok(true)
    }

    async fn configure_whatsapp(
        &self,
        config: &WhatsAppConfig,
        base_url: &str,
    ) -> Result<Option<String>> {
        let access_token = config
            .access_token
            .as_ref()
            .map(|s| s.expose().trim())
            .filter(|s| !s.is_empty())
            .ok_or_else(|| {
                Error::config("WhatsApp access_token is required for webhook auto-setup")
            })?;
        let business_account_id = config
            .business_account_id
            .as_ref()
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .ok_or_else(|| {
                Error::config("WhatsApp business_account_id is required for webhook auto-setup")
            })?;

        let callback_url = format!("{}/webhook/whatsapp", base_url.trim_end_matches('/'));
        let verify_token = config
            .webhook_verify_token
            .as_ref()
            .map(|s| s.expose().trim().to_string())
            .filter(|s| !s.is_empty())
            .or_else(|| {
                std::env::var("CLAUDEAR_WHATSAPP_WEBHOOK_VERIFY_TOKEN")
                    .ok()
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
            })
            .unwrap_or_else(|| {
                format!(
                    "wawv_{}{}",
                    Uuid::new_v4().simple(),
                    Uuid::new_v4().simple()
                )
            });
        let generated_verify_token = config
            .webhook_verify_token
            .as_ref()
            .map(|s| s.expose().trim().is_empty())
            .unwrap_or(true)
            .then(|| verify_token.clone());

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .connect_timeout(std::time::Duration::from_secs(10))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        let url = format!(
            "https://graph.facebook.com/v21.0/{}/subscribed_apps",
            business_account_id
        );

        let resp = client
            .post(&url)
            .bearer_auth(access_token)
            .json(&serde_json::json!({
                "override_callback_uri": callback_url,
                "verify_token": verify_token,
            }))
            .send()
            .await
            .map_err(|e| Error::api(format!("WhatsApp subscribed_apps request failed: {}", e)))?;

        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(Error::api(format!(
                "WhatsApp subscribed_apps returned {}: {}",
                status, body
            )));
        }

        let parsed: serde_json::Value = serde_json::from_str(&body)
            .map_err(|e| Error::api(format!("Failed to parse WhatsApp response: {}", e)))?;
        if parsed
            .get("success")
            .and_then(|v| v.as_bool())
            .unwrap_or(true)
        {
            return Ok(generated_verify_token);
        }

        Err(Error::api(format!(
            "WhatsApp subscribed_apps failed: {}",
            body
        )))
    }

    async fn configure_gitlab(
        &self,
        config: &GitLabConfig,
        base_url: &str,
    ) -> Result<(usize, Option<String>)> {
        let token = config
            .token
            .as_ref()
            .ok_or_else(|| Error::config("GitLab token is required for webhook auto-setup"))?
            .expose()
            .to_string();
        if token.is_empty() {
            return Err(Error::config(
                "GitLab token is required for webhook auto-setup",
            ));
        }

        let groups: Vec<&str> = config
            .groups
            .iter()
            .map(|g| g.trim())
            .filter(|g| !g.is_empty())
            .collect();
        if groups.is_empty() {
            return Err(Error::config(
                "GitLab groups must be configured for webhook auto-setup",
            ));
        }

        let callback_url = format!("{}/webhook/gitlab", base_url.trim_end_matches('/'));
        let secret = config
            .webhook_secret
            .as_ref()
            .map(|s| s.expose().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| {
                format!(
                    "glwh_{}{}",
                    Uuid::new_v4().simple(),
                    Uuid::new_v4().simple()
                )
            });
        let generated_secret = config
            .webhook_secret
            .as_ref()
            .map(|s| s.expose().is_empty())
            .unwrap_or(true)
            .then(|| secret.clone());

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .connect_timeout(std::time::Duration::from_secs(10))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());

        let mut configured_count = 0usize;
        let mut group_failures: Vec<String> = Vec::new();

        for group in groups {
            match self
                .ensure_gitlab_group_webhook(
                    &client,
                    &token,
                    &config.base_url,
                    group,
                    &callback_url,
                    &secret,
                )
                .await
            {
                Ok(_) => configured_count += 1,
                Err(e) => group_failures.push(format!("{} ({})", group, e)),
            }
        }

        if configured_count == 0 {
            return Err(Error::api(format!(
                "GitLab webhook setup failed for all groups: {}",
                group_failures.join("; ")
            )));
        }

        if !group_failures.is_empty() {
            tracing::warn!(
                "GitLab webhook setup partially succeeded: configured {} group(s), failed for {}",
                configured_count,
                group_failures.join("; ")
            );
        }

        Ok((configured_count, generated_secret))
    }

    fn ensure_slack_events_subscription(
        manifest: &mut serde_json::Value,
        callback_url: &str,
    ) -> Result<bool> {
        let root = manifest
            .as_object_mut()
            .ok_or_else(|| Error::api("Slack manifest is not a JSON object"))?;

        let settings = root
            .entry("settings")
            .or_insert_with(|| serde_json::json!({}));
        if !settings.is_object() {
            *settings = serde_json::json!({});
        }
        let settings_obj = settings
            .as_object_mut()
            .ok_or_else(|| Error::api("Slack manifest.settings is not an object"))?;

        let event_subs = settings_obj
            .entry("event_subscriptions")
            .or_insert_with(|| serde_json::json!({}));
        if !event_subs.is_object() {
            *event_subs = serde_json::json!({});
        }
        let event_subs_obj = event_subs
            .as_object_mut()
            .ok_or_else(|| Error::api("Slack manifest.settings.event_subscriptions invalid"))?;

        let mut changed = false;
        if event_subs_obj.get("request_url").and_then(|v| v.as_str()) != Some(callback_url) {
            event_subs_obj.insert(
                "request_url".to_string(),
                serde_json::Value::String(callback_url.to_string()),
            );
            changed = true;
        }

        let bot_events = event_subs_obj
            .entry("bot_events")
            .or_insert_with(|| serde_json::json!([]));
        if !bot_events.is_array() {
            *bot_events = serde_json::json!([]);
            changed = true;
        }
        let arr = bot_events.as_array_mut().ok_or_else(|| {
            Error::api("Slack manifest.settings.event_subscriptions.bot_events invalid")
        })?;

        for event in [
            "message.channels",
            "message.groups",
            "message.im",
            "message.mpim",
        ] {
            if !arr.iter().any(|v| v.as_str() == Some(event)) {
                arr.push(serde_json::Value::String(event.to_string()));
                changed = true;
            }
        }

        Ok(changed)
    }

    async fn ensure_jira_webhook(
        &self,
        client: &reqwest::Client,
        auth_header: &str,
        jira_base_url: &str,
        callback_url: &str,
    ) -> Result<bool> {
        let hooks_url = format!(
            "{}/rest/webhooks/1.0/webhook",
            jira_base_url.trim_end_matches('/')
        );

        let list_resp = client
            .get(&hooks_url)
            .header("Authorization", auth_header)
            .header("Accept", "application/json")
            .send()
            .await
            .map_err(|e| Error::api(format!("Jira list webhooks request failed: {}", e)))?;

        if list_resp.status().is_success() {
            let body = list_resp.text().await.unwrap_or_default();
            let parsed: serde_json::Value = serde_json::from_str(&body).map_err(|e| {
                Error::api(format!("Failed to parse Jira webhook list response: {}", e))
            })?;
            let exists = parsed.as_array().is_some_and(|items| {
                items.iter().any(|item| {
                    item.get("url")
                        .and_then(|v| v.as_str())
                        .is_some_and(|u| u == callback_url)
                })
            });
            if exists {
                return Ok(false);
            }
        } else {
            let status = list_resp.status();
            let body = list_resp.text().await.unwrap_or_default();
            return Err(Error::api(format!(
                "Jira list webhooks returned {}: {}",
                status, body
            )));
        }

        let payload = serde_json::json!({
            "name": "claudear",
            "url": callback_url,
            "events": ["jira:issue_created", "jira:issue_updated"],
            "excludeBody": false
        });

        let create_resp = client
            .post(&hooks_url)
            .header("Authorization", auth_header)
            .header("Accept", "application/json")
            .json(&payload)
            .send()
            .await
            .map_err(|e| Error::api(format!("Jira create webhook request failed: {}", e)))?;

        if create_resp.status().is_success() {
            return Ok(true);
        }

        let status = create_resp.status();
        let body = create_resp.text().await.unwrap_or_default();
        if status.as_u16() == 400
            && (body.contains("already exists")
                || body.contains("already registered")
                || body.contains(callback_url))
        {
            return Ok(false);
        }

        Err(Error::api(format!(
            "Jira create webhook returned {}: {}",
            status, body
        )))
    }

    async fn ensure_github_repo_webhook(
        &self,
        client: &reqwest::Client,
        token: &str,
        repo: &str,
        callback_url: &str,
        secret: &str,
    ) -> Result<()> {
        let hooks_url = format!("https://api.github.com/repos/{}/hooks", repo);
        let list_resp = client
            .get(&hooks_url)
            .header("Authorization", format!("Bearer {}", token))
            .header("Accept", "application/vnd.github+json")
            .header("User-Agent", "claudear")
            .header("X-GitHub-Api-Version", "2022-11-28")
            .send()
            .await
            .map_err(|e| Error::api(format!("GitHub list hooks request failed: {}", e)))?;

        if !list_resp.status().is_success() {
            let status = list_resp.status();
            let body = list_resp.text().await.unwrap_or_default();
            return Err(Error::api(format!(
                "GitHub list hooks returned {}: {}",
                status, body
            )));
        }

        let hooks: Vec<GitHubHookListItem> = list_resp
            .json()
            .await
            .map_err(|e| Error::api(format!("Failed to parse GitHub hooks response: {}", e)))?;

        if hooks
            .iter()
            .any(|h| h.config.url.as_deref().is_some_and(|u| u == callback_url))
        {
            return Ok(());
        }

        let payload = serde_json::json!({
            "name": "web",
            "active": true,
            "events": ["pull_request_review", "pull_request_review_comment", "pull_request"],
            "config": {
                "url": callback_url,
                "content_type": "json",
                "secret": secret,
                "insecure_ssl": "0"
            }
        });

        let create_resp = client
            .post(&hooks_url)
            .header("Authorization", format!("Bearer {}", token))
            .header("Accept", "application/vnd.github+json")
            .header("User-Agent", "claudear")
            .header("X-GitHub-Api-Version", "2022-11-28")
            .json(&payload)
            .send()
            .await
            .map_err(|e| Error::api(format!("GitHub create hook request failed: {}", e)))?;

        if create_resp.status().is_success() {
            return Ok(());
        }

        let status = create_resp.status();
        let body = create_resp.text().await.unwrap_or_default();
        if status.as_u16() == 422
            && (body.contains("Hook already exists") || body.contains("already exists"))
        {
            return Ok(());
        }

        Err(Error::api(format!(
            "GitHub create hook returned {}: {}",
            status, body
        )))
    }

    async fn ensure_gitlab_group_webhook(
        &self,
        client: &reqwest::Client,
        token: &str,
        gitlab_base_url: &str,
        group: &str,
        callback_url: &str,
        secret: &str,
    ) -> Result<()> {
        let encoded_group: String =
            url::form_urlencoded::byte_serialize(group.as_bytes()).collect();
        let hooks_url = format!(
            "{}/api/v4/groups/{}/hooks",
            gitlab_base_url.trim_end_matches('/'),
            encoded_group
        );

        let list_resp = client
            .get(&hooks_url)
            .header("PRIVATE-TOKEN", token)
            .header("Accept", "application/json")
            .send()
            .await
            .map_err(|e| Error::api(format!("GitLab list hooks request failed: {}", e)))?;

        if !list_resp.status().is_success() {
            let status = list_resp.status();
            let body = list_resp.text().await.unwrap_or_default();
            return Err(Error::api(format!(
                "GitLab list hooks returned {}: {}",
                status, body
            )));
        }

        let hooks: Vec<GitLabHookListItem> = list_resp
            .json()
            .await
            .map_err(|e| Error::api(format!("Failed to parse GitLab hooks response: {}", e)))?;

        if hooks
            .iter()
            .any(|h| h.url.as_deref().is_some_and(|u| u == callback_url))
        {
            return Ok(());
        }

        let payload = serde_json::json!({
            "url": callback_url,
            "token": secret,
            "issues_events": true,
            "enable_ssl_verification": true
        });

        let create_resp = client
            .post(&hooks_url)
            .header("PRIVATE-TOKEN", token)
            .header("Accept", "application/json")
            .json(&payload)
            .send()
            .await
            .map_err(|e| Error::api(format!("GitLab create hook request failed: {}", e)))?;

        if create_resp.status().is_success() {
            return Ok(());
        }

        let status = create_resp.status();
        let body = create_resp.text().await.unwrap_or_default();
        if (status.as_u16() == 400 || status.as_u16() == 409)
            && (body.contains("already been taken")
                || body.contains("Hook already exists")
                || body.contains("has already been taken"))
        {
            return Ok(());
        }

        Err(Error::api(format!(
            "GitLab create hook returned {}: {}",
            status, body
        )))
    }

    fn jira_auth_header(config: &JiraConfig) -> String {
        if config.auth_mode.eq_ignore_ascii_case("bearer") {
            format!("Bearer {}", config.api_token.expose())
        } else {
            let credentials = format!("{}:{}", config.email, config.api_token.expose());
            let encoded = {
                use base64::Engine;
                base64::engine::general_purpose::STANDARD.encode(credentials.as_bytes())
            };
            format!("Basic {}", encoded)
        }
    }

    async fn configure_linear(
        &self,
        config: &LinearConfig,
        base_url: &str,
    ) -> Result<(String, String)> {
        let client = LinearApiClient::from_config(config);
        let webhook_url = format!("{}/webhook/linear", base_url.trim_end_matches('/'));

        // Check if webhook already exists
        if client.webhook_exists(&webhook_url).await? {
            return Err(Error::api(format!(
                "A webhook with URL {} already exists in Linear. \
                Delete it manually if you want to reconfigure.",
                webhook_url
            )));
        }

        tracing::info!("Creating Linear webhook: {}", webhook_url);

        let registration = client
            .create_webhook(&webhook_url, config.team_id.as_deref(), &["Issue"])
            .await?;

        Ok((registration.id, registration.secret))
    }

    async fn configure_sentry(
        &self,
        config: &SentryConfig,
        base_url: &str,
    ) -> Result<(usize, Option<String>)> {
        let client = SentryApiClient::from_config(config);
        let webhook_url = format!("{}/webhook/sentry", base_url.trim_end_matches('/'));

        tracing::info!("Creating Sentry webhooks: {}", webhook_url);

        let registrations = client
            .create_webhooks_for_projects(
                &config.project_slugs,
                &webhook_url,
                &["event.created", "event.alert"],
            )
            .await?;

        let count = registrations.len();
        let secret = registrations.first().map(|r| r.secret.clone());

        // Log warning if multiple projects have different secrets
        if registrations.len() > 1 {
            let first_secret = &registrations[0].secret;
            let different = registrations
                .iter()
                .skip(1)
                .any(|r| &r.secret != first_secret);
            if different {
                tracing::warn!(
                    "Sentry webhooks have different secrets for different projects. \
                    Only the first secret will be saved to .env. \
                    You may need to handle this manually."
                );
            }
        }

        Ok((count, secret))
    }

    /// Check if webhooks need to be configured.
    pub fn needs_configuration(&self) -> bool {
        let linear_needs = self
            .config
            .linear()
            .is_some_and(|c| c.enabled && c.webhook_secret.is_none());

        let sentry_needs = self
            .config
            .issues
            .sentry
            .as_ref()
            .is_some_and(|c| c.enabled && c.client_secret.is_none());

        let github = self.config.github();
        let github_needs =
            !github.repos.is_empty() && github.token.is_some() && github.webhook_secret.is_none();
        let github_app_secret_present = github
            .app
            .webhook_secret
            .as_ref()
            .is_some_and(|s| !s.expose().trim().is_empty())
            || github
                .webhook_secret
                .as_ref()
                .is_some_and(|s| !s.expose().trim().is_empty());
        let github_app_needs = !github.repos.is_empty()
            && github.token.is_none()
            && github.app.is_configured()
            && !github_app_secret_present;

        let gitlab_needs = self.config.gitlab().is_some_and(|c| {
            c.enabled && !c.groups.is_empty() && c.token.is_some() && c.webhook_secret.is_none()
        });

        let jira_needs = self
            .config
            .jira()
            .is_some_and(|c| c.enabled && !c.base_url.trim().is_empty() && !c.api_token.is_empty());

        let telegram_needs = self.config.notifiers.telegram.source_enabled
            && self
                .config
                .notifiers
                .telegram
                .bot_token
                .as_ref()
                .is_some_and(|s| !s.expose().trim().is_empty());

        let slack_needs = self.config.issues.slack.as_ref().is_some_and(|c| {
            c.app_id.as_ref().is_some_and(|v| !v.trim().is_empty())
                && c.app_config_token
                    .as_ref()
                    .is_some_and(|s| !s.expose().trim().is_empty())
                && c.signing_secret
                    .as_ref()
                    .is_some_and(|s| !s.expose().trim().is_empty())
        });

        let whatsapp = &self.config.notifiers.whatsapp;
        let whatsapp_needs = whatsapp.source_enabled
            && whatsapp
                .access_token
                .as_ref()
                .is_some_and(|s| !s.expose().trim().is_empty())
            && whatsapp
                .business_account_id
                .as_ref()
                .is_some_and(|s| !s.trim().is_empty())
            && whatsapp
                .app_secret
                .as_ref()
                .is_some_and(|s| !s.expose().trim().is_empty());

        let discord_notifier = &self.config.notifiers.discord;
        let discord_notifier_needs = discord_notifier
            .webhook_url
            .as_ref()
            .map(|s| s.expose().trim().is_empty())
            .unwrap_or(true)
            && discord_notifier
                .bot_token
                .as_ref()
                .is_some_and(|s| !s.expose().trim().is_empty())
            && discord_notifier
                .channel_id
                .as_ref()
                .is_some_and(|s| !s.trim().is_empty());

        linear_needs
            || sentry_needs
            || github_needs
            || github_app_needs
            || gitlab_needs
            || jira_needs
            || telegram_needs
            || slack_needs
            || whatsapp_needs
            || discord_notifier_needs
    }
}

/// Print the result of webhook configuration to the console.
pub fn print_setup_result(result: &WebhookSetupResult) {
    println!("\n=== Webhook Configuration Complete ===\n");

    if result.linear_configured {
        println!("Linear:");
        println!("  Status: Configured");
        if let Some(ref id) = result.linear_webhook_id {
            println!("  Webhook ID: {}", id);
        }
        if let Some(ref secret) = result.linear_secret {
            println!("  Secret: {} (saved to .env)", mask_secret(secret));
        }
        println!();
    }

    if result.sentry_configured {
        println!("Sentry:");
        println!("  Status: Configured");
        println!("  Projects: {}", result.sentry_project_count);
        if let Some(ref secret) = result.sentry_secret {
            println!("  Secret: {} (saved to .env)", mask_secret(secret));
        }
        println!();
    }

    if !result.warnings.is_empty() {
        println!("Warnings / Notes:");
        for warning in &result.warnings {
            println!("  - {}", warning);
        }
        println!();
    }

    println!("Webhook secrets have been saved to your .env file.");
    println!("The webhook server will now start and begin receiving events.");
}

fn mask_secret(secret: &str) -> String {
    let chars: Vec<char> = secret.chars().collect();
    if chars.is_empty() {
        return "****".to_string();
    }
    if chars.len() <= 12 {
        "*".repeat(chars.len())
    } else {
        let prefix: String = chars[..3].iter().collect();
        let suffix: String = chars[chars.len() - 3..].iter().collect();
        format!("{}...{}", prefix, suffix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> Config {
        Config {
            workspace: "/tmp/repos".into(),
            known_orgs: vec!["test-org".to_string()],
            auto_discover_paths: vec![],
            poll_interval_ms: 300_000,
            webhook_port: 3100,
            bind_address: "127.0.0.1".to_string(),
            db_path: "/tmp/test.db".into(),
            max_issues_per_cycle: 5,
            max_concurrent: 1,
            processing_delay_ms: 5000,
            max_activity_entries: 100,
            ipc_timeout_secs: 30,
            agent: claudear_config::config::AgentConfig::default(),
            scm: claudear_config::config::ScmConfig::default(),
            issues: claudear_config::config::IssuesConfig::default(),
            notifiers: claudear_config::config::NotifiersConfig::default(),
            ask: claudear_config::config::AskConfig::default(),
            retry: claudear_config::config::RetryConfig::default(),
            regression: claudear_config::config::RegressionConfig::default(),
            cascade: claudear_config::config::CascadeConfig::default(),
            users: std::collections::HashMap::new(),
            learning: claudear_config::config::LearningConfig::default(),
            prioritisation: claudear_config::config::PrioritisationConfig::default(),
            code_index: claudear_config::config::CodeIndexConfig::default(),
            evaluation: claudear_config::config::EvaluationConfig::default(),
            storage_dir: "/tmp/claudear-storage".into(),
            dashboard: claudear_config::config::DashboardConfig::default(),
            chat: claudear_config::config::ChatConfig::default(),
            tls: claudear_config::config::TlsConfig::default(),
            llm: claudear_config::config::LlmModelConfig::default(),
            embedding: claudear_config::config::EmbeddingModelConfig::default(),
            qa: claudear_config::config::QaConfig::default(),
        }
    }

    #[test]
    fn test_mask_secret_long() {
        assert_eq!(mask_secret("1234567890abcdef"), "123...def");
    }

    #[test]
    fn test_mask_secret_short() {
        assert_eq!(mask_secret("1234"), "****");
    }

    #[test]
    fn test_mask_secret_exactly_12() {
        assert_eq!(mask_secret("123456789012"), "************");
    }

    #[test]
    fn test_mask_secret_13_chars() {
        assert_eq!(mask_secret("1234567890abc"), "123...abc");
    }

    #[test]
    fn test_webhook_setup_result_default() {
        let result = WebhookSetupResult::default();
        assert!(!result.linear_configured);
        assert!(!result.sentry_configured);
        assert!(result.warnings.is_empty());
    }

    #[test]
    fn test_needs_configuration_no_sources() {
        let config = test_config();
        let configurator = WebhookConfigurator::new(config, "/tmp/.env");
        assert!(!configurator.needs_configuration());
    }

    #[test]
    fn test_needs_configuration_linear_no_secret() {
        let mut config = test_config();
        config.issues.linear = Some(claudear_config::config::LinearConfig {
            enabled: true,
            api_key: "test".into(),
            trigger_labels: vec![],
            trigger_states: vec![],
            team_id: None,
            project_id: None,
            webhook_secret: None, // No secret
            ..Default::default()
        });
        let configurator = WebhookConfigurator::new(config, "/tmp/.env");
        assert!(configurator.needs_configuration());
    }

    #[test]
    fn test_needs_configuration_linear_with_secret() {
        let mut config = test_config();
        config.issues.linear = Some(claudear_config::config::LinearConfig {
            enabled: true,
            api_key: "test".into(),
            trigger_labels: vec![],
            trigger_states: vec![],
            team_id: None,
            project_id: None,
            webhook_secret: Some("secret".into()), // Has secret
            ..Default::default()
        });
        let configurator = WebhookConfigurator::new(config, "/tmp/.env");
        assert!(!configurator.needs_configuration());
    }

    #[test]
    fn test_mask_secret_empty_string() {
        // Empty input now returns "****" instead of an empty string.
        let result = mask_secret("");
        assert_eq!(
            result, "****",
            "empty secret should produce a masked placeholder"
        );
    }

    #[test]
    fn test_mask_secret_single_char() {
        assert_eq!(mask_secret("a"), "*");
    }

    #[test]
    fn test_mask_secret_two_chars() {
        assert_eq!(mask_secret("ab"), "**");
    }

    #[test]
    fn test_mask_secret_three_chars() {
        assert_eq!(mask_secret("abc"), "***");
    }

    #[test]
    fn test_mask_secret_unicode_multibyte() {
        // Uses chars() not bytes(), so multi-byte characters should be
        // handled correctly — each CJK character counts as 1.
        let secret = "\u{65E5}\u{672C}\u{8A9E}\u{30C6}\u{30B9}\u{30C8}\u{6587}\u{5B57}\u{5217}\u{306E}\u{4F8B}\u{3067}\u{3059}\u{FF01}\u{5168}\u{90E8}\u{30DE}\u{30B9}\u{30AF}";
        let chars: Vec<char> = secret.chars().collect();
        assert_eq!(chars.len(), 19, "should be 19 unicode chars");
        // 19 > 12, so should use prefix...suffix format
        let masked = mask_secret(secret);
        let prefix: String = chars[..3].iter().collect();
        let suffix: String = chars[chars.len() - 3..].iter().collect();
        assert_eq!(masked, format!("{}...{}", prefix, suffix));
    }

    #[test]
    fn test_mask_secret_exactly_13_chars() {
        // 13 > 12, so should show prefix...suffix
        assert_eq!(mask_secret("abcdefghijklm"), "abc...klm");
    }

    #[test]
    fn test_mask_secret_exactly_12_chars_boundary() {
        // 12 <= 12, so should be all asterisks
        assert_eq!(mask_secret("abcdefghijkl"), "************");
    }

    #[test]
    fn test_needs_configuration_sentry_enabled_no_secret() {
        let mut config = test_config();
        config.issues.sentry = Some(claudear_config::config::SentryConfig {
            enabled: true,
            client_secret: None,
            ..Default::default()
        });
        let configurator = WebhookConfigurator::new(config, "/tmp/.env");
        assert!(
            configurator.needs_configuration(),
            "Sentry enabled without client_secret should need config"
        );
    }

    #[test]
    fn test_needs_configuration_sentry_enabled_with_secret() {
        let mut config = test_config();
        config.issues.sentry = Some(claudear_config::config::SentryConfig {
            enabled: true,
            client_secret: Some("sentry-secret".into()),
            ..Default::default()
        });
        let configurator = WebhookConfigurator::new(config, "/tmp/.env");
        assert!(
            !configurator.needs_configuration(),
            "Sentry enabled with client_secret should not need config"
        );
    }

    #[test]
    fn test_needs_configuration_sentry_disabled() {
        let mut config = test_config();
        config.issues.sentry = Some(claudear_config::config::SentryConfig {
            enabled: false,
            client_secret: None,
            ..Default::default()
        });
        let configurator = WebhookConfigurator::new(config, "/tmp/.env");
        assert!(
            !configurator.needs_configuration(),
            "Sentry disabled should not need config even without secret"
        );
    }

    #[test]
    fn test_needs_configuration_both_linear_and_sentry_need_config() {
        let mut config = test_config();
        config.issues.linear = Some(claudear_config::config::LinearConfig {
            enabled: true,
            webhook_secret: None,
            ..Default::default()
        });
        config.issues.sentry = Some(claudear_config::config::SentryConfig {
            enabled: true,
            client_secret: None,
            ..Default::default()
        });
        let configurator = WebhookConfigurator::new(config, "/tmp/.env");
        assert!(
            configurator.needs_configuration(),
            "Both sources needing config should return true"
        );
    }

    #[test]
    fn test_needs_configuration_linear_has_secret_sentry_does_not() {
        let mut config = test_config();
        config.issues.linear = Some(claudear_config::config::LinearConfig {
            enabled: true,
            webhook_secret: Some("linear-secret".into()),
            ..Default::default()
        });
        config.issues.sentry = Some(claudear_config::config::SentryConfig {
            enabled: true,
            client_secret: None,
            ..Default::default()
        });
        let configurator = WebhookConfigurator::new(config, "/tmp/.env");
        assert!(
            configurator.needs_configuration(),
            "Should still need config if Sentry lacks secret"
        );
    }

    #[test]
    fn test_needs_configuration_neither_enabled() {
        let mut config = test_config();
        config.issues.linear = Some(claudear_config::config::LinearConfig {
            enabled: false,
            webhook_secret: None,
            ..Default::default()
        });
        config.issues.sentry = Some(claudear_config::config::SentryConfig {
            enabled: false,
            client_secret: None,
            ..Default::default()
        });
        let configurator = WebhookConfigurator::new(config, "/tmp/.env");
        assert!(
            !configurator.needs_configuration(),
            "Neither enabled should not need config"
        );
    }

    #[test]
    fn test_webhook_setup_result_default_all_fields() {
        let result = WebhookSetupResult::default();
        assert!(!result.linear_configured);
        assert!(result.linear_webhook_id.is_none());
        assert!(result.linear_secret.is_none());
        assert!(!result.sentry_configured);
        assert_eq!(result.sentry_project_count, 0);
        assert!(result.sentry_secret.is_none());
        assert!(result.warnings.is_empty());
    }

    #[test]
    fn test_webhook_setup_result_set_all_fields() {
        let result = WebhookSetupResult {
            linear_configured: true,
            linear_webhook_id: Some("wh-123".to_string()),
            linear_secret: Some("lin-secret".to_string()),
            sentry_configured: true,
            sentry_project_count: 3,
            sentry_secret: Some("sen-secret".to_string()),
            warnings: vec!["warn1".to_string(), "warn2".to_string()],
        };
        assert!(result.linear_configured);
        assert_eq!(result.linear_webhook_id.as_deref(), Some("wh-123"));
        assert_eq!(result.linear_secret.as_deref(), Some("lin-secret"));
        assert!(result.sentry_configured);
        assert_eq!(result.sentry_project_count, 3);
        assert_eq!(result.sentry_secret.as_deref(), Some("sen-secret"));
        assert_eq!(result.warnings.len(), 2);
        assert_eq!(result.warnings[0], "warn1");
        assert_eq!(result.warnings[1], "warn2");
    }

    #[test]
    fn test_configurator_new_stores_config_and_path() {
        let config = test_config();
        let configurator = WebhookConfigurator::new(config.clone(), "/tmp/test.env");

        assert_eq!(
            configurator.env_path,
            std::path::PathBuf::from("/tmp/test.env")
        );
        assert_eq!(configurator.config.webhook_port, config.webhook_port);
    }

    #[test]
    fn test_configurator_new_with_pathbuf() {
        let config = test_config();
        let path = std::path::PathBuf::from("/home/user/.env");
        let configurator = WebhookConfigurator::new(config, path.clone());

        assert_eq!(configurator.env_path, path);
    }

    #[test]
    fn test_configurator_new_with_string() {
        let config = test_config();
        let configurator = WebhookConfigurator::new(config, String::from("/var/app/.env"));

        assert_eq!(
            configurator.env_path,
            std::path::PathBuf::from("/var/app/.env")
        );
    }

    #[test]
    fn test_configurator_new_with_relative_path() {
        let config = test_config();
        let configurator = WebhookConfigurator::new(config, ".env");

        assert_eq!(configurator.env_path, std::path::PathBuf::from(".env"));
    }

    #[test]
    fn test_configurator_new_with_empty_path() {
        let config = test_config();
        let configurator = WebhookConfigurator::new(config, "");

        assert_eq!(configurator.env_path, std::path::PathBuf::from(""));
    }

    #[test]
    fn test_configurator_new_preserves_all_config_fields() {
        let mut config = test_config();
        config.webhook_port = 5555;
        config.workspace = "/custom/dir".into();
        config.issues.linear = Some(claudear_config::config::LinearConfig {
            enabled: true,
            api_key: "key-123".into(),
            webhook_secret: Some("secret".into()),
            ..Default::default()
        });

        let configurator = WebhookConfigurator::new(config, "/tmp/.env");

        assert_eq!(configurator.config.webhook_port, 5555);
        assert_eq!(
            configurator.config.workspace,
            std::path::PathBuf::from("/custom/dir")
        );
        assert!(configurator.config.issues.linear.is_some());
        assert_eq!(
            configurator
                .config
                .issues
                .linear
                .as_ref()
                .unwrap()
                .api_key
                .expose(),
            "key-123"
        );
    }

    #[test]
    fn test_needs_configuration_linear_disabled_no_secret() {
        let mut config = test_config();
        config.issues.linear = Some(claudear_config::config::LinearConfig {
            enabled: false,
            webhook_secret: None,
            ..Default::default()
        });
        let configurator = WebhookConfigurator::new(config, "/tmp/.env");
        assert!(
            !configurator.needs_configuration(),
            "Linear disabled should not need config even without secret"
        );
    }

    #[test]
    fn test_needs_configuration_linear_disabled_with_secret() {
        let mut config = test_config();
        config.issues.linear = Some(claudear_config::config::LinearConfig {
            enabled: false,
            webhook_secret: Some("has-secret".into()),
            ..Default::default()
        });
        let configurator = WebhookConfigurator::new(config, "/tmp/.env");
        assert!(
            !configurator.needs_configuration(),
            "Linear disabled with secret should not need config"
        );
    }

    #[test]
    fn test_needs_configuration_sentry_has_secret_linear_does_not() {
        let mut config = test_config();
        config.issues.linear = Some(claudear_config::config::LinearConfig {
            enabled: true,
            webhook_secret: None,
            ..Default::default()
        });
        config.issues.sentry = Some(claudear_config::config::SentryConfig {
            enabled: true,
            client_secret: Some("sentry-secret".into()),
            ..Default::default()
        });
        let configurator = WebhookConfigurator::new(config, "/tmp/.env");
        assert!(
            configurator.needs_configuration(),
            "Should need config if Linear lacks secret"
        );
    }

    #[test]
    fn test_needs_configuration_both_have_secrets() {
        let mut config = test_config();
        config.issues.linear = Some(claudear_config::config::LinearConfig {
            enabled: true,
            webhook_secret: Some("lin-secret".into()),
            ..Default::default()
        });
        config.issues.sentry = Some(claudear_config::config::SentryConfig {
            enabled: true,
            client_secret: Some("sen-secret".into()),
            ..Default::default()
        });
        let configurator = WebhookConfigurator::new(config, "/tmp/.env");
        assert!(
            !configurator.needs_configuration(),
            "Both with secrets should not need config"
        );
    }

    #[test]
    fn test_needs_configuration_only_linear_enabled_and_configured() {
        let mut config = test_config();
        config.issues.linear = Some(claudear_config::config::LinearConfig {
            enabled: true,
            webhook_secret: Some("secret".into()),
            ..Default::default()
        });
        // No sentry at all
        config.issues.sentry = None;
        let configurator = WebhookConfigurator::new(config, "/tmp/.env");
        assert!(
            !configurator.needs_configuration(),
            "Only Linear enabled with secret should not need config"
        );
    }

    #[test]
    fn test_needs_configuration_only_sentry_enabled_and_configured() {
        let mut config = test_config();
        config.issues.linear = None;
        config.issues.sentry = Some(claudear_config::config::SentryConfig {
            enabled: true,
            client_secret: Some("secret".into()),
            ..Default::default()
        });
        let configurator = WebhookConfigurator::new(config, "/tmp/.env");
        assert!(
            !configurator.needs_configuration(),
            "Only Sentry enabled with secret should not need config"
        );
    }

    #[test]
    fn test_needs_configuration_only_sentry_enabled_no_secret() {
        let mut config = test_config();
        config.issues.linear = None;
        config.issues.sentry = Some(claudear_config::config::SentryConfig {
            enabled: true,
            client_secret: None,
            ..Default::default()
        });
        let configurator = WebhookConfigurator::new(config, "/tmp/.env");
        assert!(
            configurator.needs_configuration(),
            "Only Sentry enabled without secret should need config"
        );
    }

    #[test]
    fn test_needs_configuration_github_repos_no_webhook_secret() {
        let mut config = test_config();
        config.scm.github.token = Some("ghp_test".into());
        config.scm.github.repos = vec!["owner/repo".to_string()];
        config.scm.github.webhook_secret = None;

        let configurator = WebhookConfigurator::new(config, "/tmp/.env");
        assert!(configurator.needs_configuration());
    }

    #[test]
    fn test_needs_configuration_github_repos_with_webhook_secret() {
        let mut config = test_config();
        config.scm.github.token = Some("ghp_test".into());
        config.scm.github.repos = vec!["owner/repo".to_string()];
        config.scm.github.webhook_secret = Some("gh-secret".into());

        let configurator = WebhookConfigurator::new(config, "/tmp/.env");
        assert!(!configurator.needs_configuration());
    }

    #[test]
    fn test_needs_configuration_github_app_repos_no_pat_no_secret() {
        let mut config = test_config();
        config.scm.github.token = None;
        config.scm.github.repos = vec!["owner/repo".to_string()];
        config.scm.github.app.app_id = Some(12345);
        config.scm.github.app.private_key = Some(claudear_core::secret::SecretValue::new(
            "-----BEGIN RSA PRIVATE KEY-----\nfake\n-----END RSA PRIVATE KEY-----",
        ));
        config.scm.github.app.webhook_secret = None;
        config.scm.github.webhook_secret = None;

        let configurator = WebhookConfigurator::new(config, "/tmp/.env");
        assert!(configurator.needs_configuration());
    }

    #[test]
    fn test_needs_configuration_github_app_uses_existing_legacy_secret() {
        let mut config = test_config();
        config.scm.github.token = None;
        config.scm.github.repos = vec!["owner/repo".to_string()];
        config.scm.github.app.app_id = Some(12345);
        config.scm.github.app.private_key = Some(claudear_core::secret::SecretValue::new(
            "-----BEGIN RSA PRIVATE KEY-----\nfake\n-----END RSA PRIVATE KEY-----",
        ));
        config.scm.github.webhook_secret = Some("shared-secret".into());
        config.scm.github.app.webhook_secret = None;

        let configurator = WebhookConfigurator::new(config, "/tmp/.env");
        assert!(!configurator.needs_configuration());
    }

    #[test]
    fn test_needs_configuration_discord_notifier_webhook_missing_but_bot_channel_present() {
        let mut config = test_config();
        config.notifiers.discord.bot_token = Some("discord-bot-token".into());
        config.notifiers.discord.channel_id = Some("123456789".to_string());
        config.notifiers.discord.webhook_url = None;

        let configurator = WebhookConfigurator::new(config, "/tmp/.env");
        assert!(configurator.needs_configuration());
    }

    #[test]
    fn test_needs_configuration_discord_notifier_webhook_present() {
        let mut config = test_config();
        config.notifiers.discord.bot_token = Some("discord-bot-token".into());
        config.notifiers.discord.channel_id = Some("123456789".to_string());
        config.notifiers.discord.webhook_url = Some(claudear_core::secret::SecretValue::new(
            "https://discord.com/api/webhooks/1/token",
        ));

        let configurator = WebhookConfigurator::new(config, "/tmp/.env");
        assert!(!configurator.needs_configuration());
    }

    #[test]
    fn test_needs_configuration_gitlab_groups_no_webhook_secret() {
        let mut config = test_config();
        config.scm.gitlab = Some(claudear_config::config::GitLabConfig {
            enabled: true,
            token: Some("glpat_test".into()),
            groups: vec!["group/subgroup".to_string()],
            webhook_secret: None,
            ..Default::default()
        });

        let configurator = WebhookConfigurator::new(config, "/tmp/.env");
        assert!(configurator.needs_configuration());
    }

    #[test]
    fn test_needs_configuration_gitlab_groups_with_webhook_secret() {
        let mut config = test_config();
        config.scm.gitlab = Some(claudear_config::config::GitLabConfig {
            enabled: true,
            token: Some("glpat_test".into()),
            groups: vec!["group/subgroup".to_string()],
            webhook_secret: Some("gl-secret".into()),
            ..Default::default()
        });

        let configurator = WebhookConfigurator::new(config, "/tmp/.env");
        assert!(!configurator.needs_configuration());
    }

    #[test]
    fn test_mask_secret_six_chars() {
        // 6 <= 12, so should be all asterisks
        assert_eq!(mask_secret("abcdef"), "******");
    }

    #[test]
    fn test_mask_secret_eleven_chars() {
        assert_eq!(mask_secret("abcdefghijk"), "***********");
    }

    #[test]
    fn test_mask_secret_fourteen_chars() {
        // 14 > 12, so should show prefix...suffix
        assert_eq!(mask_secret("abcdefghijklmn"), "abc...lmn");
    }

    #[test]
    fn test_mask_secret_very_long_string() {
        let secret = "a".repeat(1000);
        let result = mask_secret(&secret);
        assert_eq!(result, "aaa...aaa");
    }

    #[test]
    fn test_mask_secret_with_special_chars() {
        let secret = "!@#$%^&*()_+-=[]{}";
        // 18 chars > 12, so should show prefix...suffix
        let chars: Vec<char> = secret.chars().collect();
        let prefix: String = chars[..3].iter().collect();
        let suffix: String = chars[chars.len() - 3..].iter().collect();
        let result = mask_secret(secret);
        assert_eq!(result, format!("{}...{}", prefix, suffix));
    }

    #[test]
    fn test_mask_secret_with_spaces() {
        let secret = "secret with spaces here";
        // 23 chars > 12, so should show prefix...suffix
        let result = mask_secret(secret);
        assert_eq!(result, "sec...ere");
    }

    #[test]
    fn test_mask_secret_all_asterisks() {
        let secret = "****";
        // 4 <= 12, so should be all asterisks
        assert_eq!(mask_secret(secret), "****");
    }

    #[test]
    fn test_mask_secret_numeric_string() {
        let secret = "1234567890123456";
        // 16 > 12, so prefix...suffix
        assert_eq!(mask_secret(secret), "123...456");
    }

    #[test]
    fn test_webhook_setup_result_debug_format() {
        let result = WebhookSetupResult::default();
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("WebhookSetupResult"));
        assert!(debug_str.contains("linear_configured: false"));
        assert!(debug_str.contains("sentry_configured: false"));
    }

    #[test]
    fn test_webhook_setup_result_debug_with_values() {
        let result = WebhookSetupResult {
            linear_configured: true,
            linear_webhook_id: Some("wh-999".to_string()),
            linear_secret: Some("secret-val".to_string()),
            sentry_configured: false,
            sentry_project_count: 0,
            sentry_secret: None,
            warnings: vec!["warning-1".to_string()],
        };
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("linear_configured: true"));
        assert!(debug_str.contains("wh-999"));
        assert!(debug_str.contains("warning-1"));
    }

    #[test]
    fn test_webhook_setup_result_warnings_can_be_added() {
        let mut result = WebhookSetupResult::default();
        assert!(result.warnings.is_empty());

        result.warnings.push("warning 1".to_string());
        result.warnings.push("warning 2".to_string());

        assert_eq!(result.warnings.len(), 2);
    }

    #[test]
    fn test_webhook_setup_result_partial_configuration() {
        let result = WebhookSetupResult {
            linear_configured: true,
            linear_webhook_id: Some("wh-abc".to_string()),
            linear_secret: Some("lin-secret-val".to_string()),
            sentry_configured: false,
            sentry_project_count: 0,
            sentry_secret: None,
            warnings: vec![],
        };

        assert!(result.linear_configured);
        assert!(!result.sentry_configured);
        assert!(result.linear_webhook_id.is_some());
        assert!(result.sentry_secret.is_none());
    }

    #[test]
    fn test_webhook_setup_result_sentry_only() {
        let result = WebhookSetupResult {
            linear_configured: false,
            linear_webhook_id: None,
            linear_secret: None,
            sentry_configured: true,
            sentry_project_count: 5,
            sentry_secret: Some("sen-secret-val".to_string()),
            warnings: vec![],
        };

        assert!(!result.linear_configured);
        assert!(result.sentry_configured);
        assert_eq!(result.sentry_project_count, 5);
    }

    #[test]
    fn test_print_setup_result_linear_only() {
        // Capture that this does not panic with linear-only config
        let result = WebhookSetupResult {
            linear_configured: true,
            linear_webhook_id: Some("wh-123".to_string()),
            linear_secret: Some("secret1234567890".to_string()),
            sentry_configured: false,
            sentry_project_count: 0,
            sentry_secret: None,
            warnings: vec![],
        };

        // Should not panic
        print_setup_result(&result);
    }

    #[test]
    fn test_print_setup_result_sentry_only() {
        let result = WebhookSetupResult {
            linear_configured: false,
            linear_webhook_id: None,
            linear_secret: None,
            sentry_configured: true,
            sentry_project_count: 3,
            sentry_secret: Some("sentry-secret-1234567890".to_string()),
            warnings: vec![],
        };

        // Should not panic
        print_setup_result(&result);
    }

    #[test]
    fn test_print_setup_result_both_configured() {
        let result = WebhookSetupResult {
            linear_configured: true,
            linear_webhook_id: Some("wh-456".to_string()),
            linear_secret: Some("linear-secret-1234567890".to_string()),
            sentry_configured: true,
            sentry_project_count: 2,
            sentry_secret: Some("sentry-secret-1234567890".to_string()),
            warnings: vec![],
        };

        // Should not panic
        print_setup_result(&result);
    }

    #[test]
    fn test_print_setup_result_with_warnings() {
        let result = WebhookSetupResult {
            linear_configured: true,
            linear_webhook_id: Some("wh-789".to_string()),
            linear_secret: Some("secret1234567890".to_string()),
            sentry_configured: false,
            sentry_project_count: 0,
            sentry_secret: None,
            warnings: vec![
                "Failed to configure Sentry: API error".to_string(),
                "Rate limit exceeded".to_string(),
            ],
        };

        // Should not panic
        print_setup_result(&result);
    }

    #[test]
    fn test_print_setup_result_nothing_configured_no_warnings() {
        let result = WebhookSetupResult::default();

        // Should not panic, even with no configuration
        print_setup_result(&result);
    }

    #[test]
    fn test_print_setup_result_with_no_webhook_id() {
        let result = WebhookSetupResult {
            linear_configured: true,
            linear_webhook_id: None, // No webhook ID
            linear_secret: Some("secret1234567890".to_string()),
            sentry_configured: false,
            sentry_project_count: 0,
            sentry_secret: None,
            warnings: vec![],
        };

        // Should not panic even without webhook ID
        print_setup_result(&result);
    }

    #[test]
    fn test_print_setup_result_with_no_secrets() {
        let result = WebhookSetupResult {
            linear_configured: true,
            linear_webhook_id: Some("wh-abc".to_string()),
            linear_secret: None, // No secret
            sentry_configured: true,
            sentry_project_count: 1,
            sentry_secret: None, // No secret
            warnings: vec![],
        };

        // Should not panic even without secrets
        print_setup_result(&result);
    }

    #[test]
    fn test_print_setup_result_with_short_secrets() {
        // Secrets that are <= 12 chars should be fully masked
        let result = WebhookSetupResult {
            linear_configured: true,
            linear_webhook_id: Some("wh-abc".to_string()),
            linear_secret: Some("short".to_string()), // 5 chars, masked as *****
            sentry_configured: true,
            sentry_project_count: 1,
            sentry_secret: Some("ab".to_string()), // 2 chars, masked as **
            warnings: vec![],
        };

        // Should not panic
        print_setup_result(&result);
    }

    #[test]
    fn test_print_setup_result_with_empty_secret_strings() {
        let result = WebhookSetupResult {
            linear_configured: true,
            linear_webhook_id: Some("wh-abc".to_string()),
            linear_secret: Some("".to_string()), // Empty secret
            sentry_configured: true,
            sentry_project_count: 1,
            sentry_secret: Some("".to_string()), // Empty secret
            warnings: vec![],
        };

        // Should not panic (mask_secret("") returns "****")
        print_setup_result(&result);
    }

    #[test]
    fn test_print_setup_result_many_warnings() {
        let result = WebhookSetupResult {
            linear_configured: false,
            linear_webhook_id: None,
            linear_secret: None,
            sentry_configured: false,
            sentry_project_count: 0,
            sentry_secret: None,
            warnings: (0..10).map(|i| format!("Warning number {}", i)).collect(),
        };

        // Should not panic with many warnings
        print_setup_result(&result);
    }

    #[test]
    fn test_print_setup_result() {
        // Comprehensive test: verify print_setup_result does not panic for various configurations
        // and exercises all branches (linear, sentry, warnings, secrets, missing IDs)

        // Both configured with secrets
        let full_result = WebhookSetupResult {
            linear_configured: true,
            linear_webhook_id: Some("wh-full".to_string()),
            linear_secret: Some("abcdef1234567890".to_string()),
            sentry_configured: true,
            sentry_project_count: 2,
            sentry_secret: Some("1234567890abcdef".to_string()),
            warnings: vec!["Some warning".to_string()],
        };
        print_setup_result(&full_result); // should not panic

        // No configured sources, just warnings
        let empty_result = WebhookSetupResult {
            linear_configured: false,
            linear_webhook_id: None,
            linear_secret: None,
            sentry_configured: false,
            sentry_project_count: 0,
            sentry_secret: None,
            warnings: vec!["warning A".to_string(), "warning B".to_string()],
        };
        print_setup_result(&empty_result); // should not panic

        // Linear without webhook ID or secret
        let partial_result = WebhookSetupResult {
            linear_configured: true,
            linear_webhook_id: None,
            linear_secret: None,
            sentry_configured: false,
            sentry_project_count: 0,
            sentry_secret: None,
            warnings: vec![],
        };
        print_setup_result(&partial_result); // should not panic
    }

    #[test]
    fn test_needs_configuration_edge_cases() {
        // Edge case 1: Both sources enabled, both missing secrets
        let mut config1 = test_config();
        config1.issues.linear = Some(claudear_config::config::LinearConfig {
            enabled: true,
            webhook_secret: None,
            ..Default::default()
        });
        config1.issues.sentry = Some(claudear_config::config::SentryConfig {
            enabled: true,
            client_secret: None,
            ..Default::default()
        });
        let c1 = WebhookConfigurator::new(config1, "/tmp/.env");
        assert!(
            c1.needs_configuration(),
            "both enabled, both missing secrets"
        );

        // Edge case 2: Both sources present but disabled
        let mut config2 = test_config();
        config2.issues.linear = Some(claudear_config::config::LinearConfig {
            enabled: false,
            webhook_secret: None,
            ..Default::default()
        });
        config2.issues.sentry = Some(claudear_config::config::SentryConfig {
            enabled: false,
            client_secret: None,
            ..Default::default()
        });
        let c2 = WebhookConfigurator::new(config2, "/tmp/.env");
        assert!(
            !c2.needs_configuration(),
            "both disabled should not need config"
        );

        // Edge case 3: One enabled with secret, one enabled without
        let mut config3 = test_config();
        config3.issues.linear = Some(claudear_config::config::LinearConfig {
            enabled: true,
            webhook_secret: Some("secret".into()),
            ..Default::default()
        });
        config3.issues.sentry = Some(claudear_config::config::SentryConfig {
            enabled: true,
            client_secret: None,
            ..Default::default()
        });
        let c3 = WebhookConfigurator::new(config3, "/tmp/.env");
        assert!(
            c3.needs_configuration(),
            "one missing secret should need config"
        );

        // Edge case 4: Both enabled, both have secrets
        let mut config4 = test_config();
        config4.issues.linear = Some(claudear_config::config::LinearConfig {
            enabled: true,
            webhook_secret: Some("lin-secret".into()),
            ..Default::default()
        });
        config4.issues.sentry = Some(claudear_config::config::SentryConfig {
            enabled: true,
            client_secret: Some("sen-secret".into()),
            ..Default::default()
        });
        let c4 = WebhookConfigurator::new(config4, "/tmp/.env");
        assert!(
            !c4.needs_configuration(),
            "both have secrets should not need config"
        );

        // Edge case 5: No sources configured at all
        let config5 = test_config(); // linear=None, sentry=None
        let c5 = WebhookConfigurator::new(config5, "/tmp/.env");
        assert!(
            !c5.needs_configuration(),
            "no sources means no config needed"
        );
    }

    #[test]
    fn test_print_setup_result_sentry_zero_projects() {
        let result = WebhookSetupResult {
            linear_configured: false,
            linear_webhook_id: None,
            linear_secret: None,
            sentry_configured: true,
            sentry_project_count: 0,
            sentry_secret: Some("secret1234567890".to_string()),
            warnings: vec![],
        };

        // Should not panic even with 0 projects
        print_setup_result(&result);
    }

    #[tokio::test]
    async fn test_configure_no_sources_enabled_returns_error() {
        // No linear or sentry configured at all => error about no sources enabled
        let config = test_config(); // linear=None, sentry=None
        let configurator = WebhookConfigurator::new(config, "/tmp/test-no-sources.env");

        let result = configurator.configure("https://example.com").await;
        assert!(result.is_err(), "configure with no sources should error");

        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("No webhook sources are enabled"),
            "Error should mention no sources are enabled, got: {}",
            err_msg
        );
    }

    #[tokio::test]
    async fn test_configure_slack_source_only_returns_manual_setup_note() {
        let mut config = test_config();
        config.issues.slack = Some(claudear_config::config::SlackSourceConfig {
            bot_token: Some("xoxb-test".into()),
            channel_id: Some("C123".to_string()),
            ..Default::default()
        });
        let configurator = WebhookConfigurator::new(config, "/tmp/test-slack-manual.env");

        let result = configurator.configure("https://example.com").await;
        assert!(
            result.is_err(),
            "configure should error without auto-configurable sources"
        );

        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("No auto-configurable webhook services are enabled"));
        assert!(err_msg.contains("Slack source is configured"));
    }

    #[tokio::test]
    async fn test_configure_discord_source_only_returns_polling_note() {
        let mut config = test_config();
        config.issues.discord = Some(claudear_config::config::DiscordSourceConfig {
            bot_token: Some("discord-bot".into()),
            channel_id: Some("123".to_string()),
            ..Default::default()
        });
        let configurator = WebhookConfigurator::new(config, "/tmp/test-discord-source-note.env");

        let result = configurator.configure("https://example.com").await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("No auto-configurable webhook services are enabled"));
        assert!(err_msg.contains("Discord source uses channel polling"));
    }

    #[tokio::test]
    async fn test_configure_slack_notifier_bot_channel_only_returns_chat_postmessage_note() {
        let mut config = test_config();
        config.notifiers.slack.bot_token = Some("xoxb-test".into());
        config.notifiers.slack.channel_id = Some("C123".to_string());
        config.notifiers.slack.webhook_url = None;
        let configurator = WebhookConfigurator::new(config, "/tmp/test-slack-notifier-note.env");

        let result = configurator.configure("https://example.com").await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("No auto-configurable webhook services are enabled"));
        assert!(err_msg.contains("Slack notifier incoming webhook URL is not auto-provisioned"));
    }

    #[tokio::test]
    async fn test_configure_jira_enabled_attempts_auto_setup() {
        let mut config = test_config();
        config.issues.jira = Some(claudear_config::config::JiraConfig {
            enabled: true,
            base_url: "https://example.atlassian.net".to_string(),
            api_token: "jira-token".into(),
            ..Default::default()
        });
        let configurator = WebhookConfigurator::new(config, "/tmp/test-jira-manual.env");

        let result = configurator.configure("https://example.com").await;
        assert!(
            result.is_err(),
            "configure should error when Jira API setup fails"
        );

        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Failed to configure any webhooks"));
        assert!(err_msg.contains("Jira"));
    }

    #[tokio::test]
    async fn test_configure_linear_disabled_sentry_disabled_returns_error() {
        // Both sources present but disabled => error about no sources enabled
        let mut config = test_config();
        config.issues.linear = Some(claudear_config::config::LinearConfig {
            enabled: false,
            ..Default::default()
        });
        config.issues.sentry = Some(claudear_config::config::SentryConfig {
            enabled: false,
            ..Default::default()
        });
        let configurator = WebhookConfigurator::new(config, "/tmp/test-disabled.env");

        let result = configurator.configure("https://example.com").await;
        assert!(
            result.is_err(),
            "configure with disabled sources should error"
        );

        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("No webhook sources are enabled"),
            "Error should mention no sources are enabled, got: {}",
            err_msg
        );
    }

    #[tokio::test]
    async fn test_configure_linear_disabled_sentry_none_returns_error() {
        // Linear present but disabled, sentry absent => error
        let mut config = test_config();
        config.issues.linear = Some(claudear_config::config::LinearConfig {
            enabled: false,
            ..Default::default()
        });
        config.issues.sentry = None;
        let configurator = WebhookConfigurator::new(config, "/tmp/test-lin-disabled.env");

        let result = configurator.configure("https://example.com").await;
        assert!(result.is_err());

        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("No webhook sources are enabled"),
            "Expected no-sources error, got: {}",
            err_msg
        );
    }

    #[tokio::test]
    async fn test_configure_sentry_disabled_linear_none_returns_error() {
        // Sentry present but disabled, linear absent => error
        let mut config = test_config();
        config.issues.linear = None;
        config.issues.sentry = Some(claudear_config::config::SentryConfig {
            enabled: false,
            ..Default::default()
        });
        let configurator = WebhookConfigurator::new(config, "/tmp/test-sen-disabled.env");

        let result = configurator.configure("https://example.com").await;
        assert!(result.is_err());

        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("No webhook sources are enabled"),
            "Expected no-sources error, got: {}",
            err_msg
        );
    }

    #[tokio::test]
    async fn test_configure_linear_enabled_but_fails_api_returns_error_with_warnings() {
        // Linear enabled with a dummy API key that will fail when trying to connect.
        // This exercises the warning-accumulation path (lines 75-80) and then the
        // "Failed to configure any webhooks" error path (lines 117-122).
        let mut config = test_config();
        config.issues.linear = Some(claudear_config::config::LinearConfig {
            enabled: true,
            api_key: "invalid-api-key".into(),
            trigger_labels: vec![],
            trigger_states: vec![],
            team_id: None,
            project_id: None,
            webhook_secret: None,
            ..Default::default()
        });
        config.issues.sentry = None;
        let configurator = WebhookConfigurator::new(config, "/tmp/test-lin-fail.env");

        let result = configurator.configure("https://example.com").await;
        assert!(
            result.is_err(),
            "configure should error when Linear API fails"
        );

        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Failed to configure any webhooks"),
            "Error should mention failure to configure, got: {}",
            err_msg
        );
    }

    #[tokio::test]
    async fn test_configure_sentry_enabled_but_fails_api_returns_error_with_warnings() {
        // Sentry enabled with dummy credentials that will fail.
        // Exercises the sentry warning-accumulation path (lines 97-101) and then the
        // "Failed to configure any webhooks" error path (lines 117-122).
        let mut config = test_config();
        config.issues.linear = None;
        config.issues.sentry = Some(claudear_config::config::SentryConfig {
            enabled: true,
            auth_token: "invalid-token".into(),
            org_slug: "nonexistent-org".to_string(),
            project_slugs: vec!["fake-project".to_string()],
            client_secret: None,
            ..Default::default()
        });
        let configurator = WebhookConfigurator::new(config, "/tmp/test-sen-fail.env");

        let result = configurator.configure("https://example.com").await;
        assert!(
            result.is_err(),
            "configure should error when Sentry API fails"
        );

        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Failed to configure any webhooks"),
            "Error should mention failure to configure, got: {}",
            err_msg
        );
    }

    #[tokio::test]
    async fn test_configure_both_enabled_both_fail_returns_combined_warnings() {
        // Both sources enabled with invalid credentials -- both will fail,
        // and the error message should include warnings from both.
        let mut config = test_config();
        config.issues.linear = Some(claudear_config::config::LinearConfig {
            enabled: true,
            api_key: "invalid-linear-key".into(),
            webhook_secret: None,
            ..Default::default()
        });
        config.issues.sentry = Some(claudear_config::config::SentryConfig {
            enabled: true,
            auth_token: "invalid-sentry-token".into(),
            org_slug: "nonexistent-org".to_string(),
            project_slugs: vec!["fake-project".to_string()],
            client_secret: None,
            ..Default::default()
        });
        let configurator = WebhookConfigurator::new(config, "/tmp/test-both-fail.env");

        let result = configurator.configure("https://example.com").await;
        assert!(
            result.is_err(),
            "configure should error when both APIs fail"
        );

        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Failed to configure any webhooks"),
            "Error should mention failure, got: {}",
            err_msg
        );
        // The error should contain both warnings joined by ";"
        assert!(
            err_msg.contains("Linear") || err_msg.contains("Sentry") || err_msg.contains(";"),
            "Error should contain warning info from at least one source, got: {}",
            err_msg
        );
    }

    #[tokio::test]
    async fn test_configure_base_url_trailing_slash_is_handled() {
        // Even though the API call will fail, this verifies configure()
        // does not panic with a trailing-slash base_url.
        let mut config = test_config();
        config.issues.linear = Some(claudear_config::config::LinearConfig {
            enabled: true,
            api_key: "test-key".into(),
            webhook_secret: None,
            ..Default::default()
        });
        let configurator = WebhookConfigurator::new(config, "/tmp/test-trailing.env");

        let result = configurator.configure("https://example.com/").await;
        // Will fail at the API level, but should not panic
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_configure_base_url_no_trailing_slash() {
        // Verifies configure() works without trailing slash
        let mut config = test_config();
        config.issues.linear = Some(claudear_config::config::LinearConfig {
            enabled: true,
            api_key: "test-key".into(),
            webhook_secret: None,
            ..Default::default()
        });
        let configurator = WebhookConfigurator::new(config, "/tmp/test-no-trailing.env");

        let result = configurator.configure("https://example.com").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_configure_linear_enabled_sentry_disabled_linear_fails() {
        // Linear enabled but fails, Sentry present but disabled.
        // Only the Linear warning should appear.
        let mut config = test_config();
        config.issues.linear = Some(claudear_config::config::LinearConfig {
            enabled: true,
            api_key: "bad-key".into(),
            webhook_secret: None,
            ..Default::default()
        });
        config.issues.sentry = Some(claudear_config::config::SentryConfig {
            enabled: false,
            ..Default::default()
        });
        let configurator = WebhookConfigurator::new(config, "/tmp/test-mixed.env");

        let result = configurator.configure("https://example.com").await;
        assert!(result.is_err());

        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Failed to configure any webhooks"),
            "Should get failure error, got: {}",
            err_msg
        );
    }

    #[tokio::test]
    async fn test_configure_sentry_enabled_linear_disabled_sentry_fails() {
        // Sentry enabled but fails, Linear present but disabled.
        let mut config = test_config();
        config.issues.linear = Some(claudear_config::config::LinearConfig {
            enabled: false,
            ..Default::default()
        });
        config.issues.sentry = Some(claudear_config::config::SentryConfig {
            enabled: true,
            auth_token: "bad-token".into(),
            org_slug: "fake-org".to_string(),
            project_slugs: vec!["fake".to_string()],
            client_secret: None,
            ..Default::default()
        });
        let configurator = WebhookConfigurator::new(config, "/tmp/test-mixed2.env");

        let result = configurator.configure("https://example.com").await;
        assert!(result.is_err());

        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Failed to configure any webhooks"),
            "Should get failure error, got: {}",
            err_msg
        );
    }

    // ---- jira_auth_header tests ----

    #[test]
    fn test_jira_auth_header_basic_mode() {
        let config = claudear_config::config::JiraConfig {
            auth_mode: "basic".to_string(),
            email: "user@example.com".to_string(),
            api_token: "my-api-token".into(),
            ..Default::default()
        };
        let header = WebhookConfigurator::jira_auth_header(&config);
        assert!(
            header.starts_with("Basic "),
            "Should start with 'Basic ', got: {}",
            header
        );
        // Decode and verify
        use base64::Engine;
        let encoded_part = header.strip_prefix("Basic ").unwrap();
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(encoded_part)
            .unwrap();
        let decoded_str = String::from_utf8(decoded).unwrap();
        assert_eq!(decoded_str, "user@example.com:my-api-token");
    }

    #[test]
    fn test_jira_auth_header_bearer_mode() {
        let config = claudear_config::config::JiraConfig {
            auth_mode: "bearer".to_string(),
            api_token: "my-pat-token".into(),
            ..Default::default()
        };
        let header = WebhookConfigurator::jira_auth_header(&config);
        assert_eq!(header, "Bearer my-pat-token");
    }

    #[test]
    fn test_jira_auth_header_bearer_case_insensitive() {
        let config = claudear_config::config::JiraConfig {
            auth_mode: "Bearer".to_string(),
            api_token: "token-123".into(),
            ..Default::default()
        };
        let header = WebhookConfigurator::jira_auth_header(&config);
        assert_eq!(header, "Bearer token-123");
    }

    #[test]
    fn test_jira_auth_header_bearer_uppercase() {
        let config = claudear_config::config::JiraConfig {
            auth_mode: "BEARER".to_string(),
            api_token: "token-upper".into(),
            ..Default::default()
        };
        let header = WebhookConfigurator::jira_auth_header(&config);
        assert_eq!(header, "Bearer token-upper");
    }

    #[test]
    fn test_jira_auth_header_default_is_basic() {
        // Default auth_mode is "basic"
        let config = claudear_config::config::JiraConfig {
            email: "dev@test.com".to_string(),
            api_token: "api-tok".into(),
            ..Default::default()
        };
        assert_eq!(config.auth_mode, "basic");
        let header = WebhookConfigurator::jira_auth_header(&config);
        assert!(header.starts_with("Basic "));
    }

    #[test]
    fn test_jira_auth_header_unknown_mode_falls_to_basic() {
        // Any non-bearer mode should fall through to basic
        let config = claudear_config::config::JiraConfig {
            auth_mode: "something-else".to_string(),
            email: "me@co.com".to_string(),
            api_token: "tok".into(),
            ..Default::default()
        };
        let header = WebhookConfigurator::jira_auth_header(&config);
        assert!(
            header.starts_with("Basic "),
            "Non-bearer should use Basic auth"
        );
    }

    // ---- ensure_slack_events_subscription tests ----

    #[test]
    fn test_ensure_slack_events_subscription_empty_manifest() {
        let mut manifest = serde_json::json!({});
        let changed = WebhookConfigurator::ensure_slack_events_subscription(
            &mut manifest,
            "https://example.com/webhook/slack",
        )
        .unwrap();
        assert!(changed, "Should report changes on empty manifest");

        // Verify structure was created
        let url = manifest["settings"]["event_subscriptions"]["request_url"]
            .as_str()
            .unwrap();
        assert_eq!(url, "https://example.com/webhook/slack");

        let events = manifest["settings"]["event_subscriptions"]["bot_events"]
            .as_array()
            .unwrap();
        assert_eq!(events.len(), 4);
        let event_strs: Vec<&str> = events.iter().map(|v| v.as_str().unwrap()).collect();
        assert!(event_strs.contains(&"message.channels"));
        assert!(event_strs.contains(&"message.groups"));
        assert!(event_strs.contains(&"message.im"));
        assert!(event_strs.contains(&"message.mpim"));
    }

    #[test]
    fn test_ensure_slack_events_subscription_already_correct() {
        let mut manifest = serde_json::json!({
            "settings": {
                "event_subscriptions": {
                    "request_url": "https://example.com/webhook/slack",
                    "bot_events": ["message.channels", "message.groups", "message.im", "message.mpim"]
                }
            }
        });
        let changed = WebhookConfigurator::ensure_slack_events_subscription(
            &mut manifest,
            "https://example.com/webhook/slack",
        )
        .unwrap();
        assert!(!changed, "Should report no changes when already correct");
    }

    #[test]
    fn test_ensure_slack_events_subscription_url_different() {
        let mut manifest = serde_json::json!({
            "settings": {
                "event_subscriptions": {
                    "request_url": "https://old.example.com/webhook/slack",
                    "bot_events": ["message.channels", "message.groups", "message.im", "message.mpim"]
                }
            }
        });
        let changed = WebhookConfigurator::ensure_slack_events_subscription(
            &mut manifest,
            "https://new.example.com/webhook/slack",
        )
        .unwrap();
        assert!(changed, "Should report changes when URL differs");
        assert_eq!(
            manifest["settings"]["event_subscriptions"]["request_url"]
                .as_str()
                .unwrap(),
            "https://new.example.com/webhook/slack"
        );
    }

    #[test]
    fn test_ensure_slack_events_subscription_missing_events() {
        let mut manifest = serde_json::json!({
            "settings": {
                "event_subscriptions": {
                    "request_url": "https://example.com/webhook/slack",
                    "bot_events": ["message.channels"]
                }
            }
        });
        let changed = WebhookConfigurator::ensure_slack_events_subscription(
            &mut manifest,
            "https://example.com/webhook/slack",
        )
        .unwrap();
        assert!(changed, "Should report changes when events are missing");
        let events = manifest["settings"]["event_subscriptions"]["bot_events"]
            .as_array()
            .unwrap();
        assert_eq!(events.len(), 4);
    }

    #[test]
    fn test_ensure_slack_events_subscription_settings_not_object() {
        let mut manifest = serde_json::json!({
            "settings": "not-an-object"
        });
        let changed = WebhookConfigurator::ensure_slack_events_subscription(
            &mut manifest,
            "https://example.com/webhook/slack",
        )
        .unwrap();
        assert!(changed);
        // settings should have been reset to an object
        assert!(manifest["settings"].is_object());
    }

    #[test]
    fn test_ensure_slack_events_subscription_event_subs_not_object() {
        let mut manifest = serde_json::json!({
            "settings": {
                "event_subscriptions": "bad"
            }
        });
        let changed = WebhookConfigurator::ensure_slack_events_subscription(
            &mut manifest,
            "https://example.com/webhook/slack",
        )
        .unwrap();
        assert!(changed);
        assert!(manifest["settings"]["event_subscriptions"].is_object());
    }

    #[test]
    fn test_ensure_slack_events_subscription_bot_events_not_array() {
        let mut manifest = serde_json::json!({
            "settings": {
                "event_subscriptions": {
                    "request_url": "https://example.com/webhook/slack",
                    "bot_events": "not-array"
                }
            }
        });
        let changed = WebhookConfigurator::ensure_slack_events_subscription(
            &mut manifest,
            "https://example.com/webhook/slack",
        )
        .unwrap();
        assert!(changed);
        let events = manifest["settings"]["event_subscriptions"]["bot_events"]
            .as_array()
            .unwrap();
        assert_eq!(events.len(), 4);
    }

    #[test]
    fn test_ensure_slack_events_subscription_not_json_object() {
        let mut manifest = serde_json::json!("not an object");
        let result = WebhookConfigurator::ensure_slack_events_subscription(
            &mut manifest,
            "https://example.com/webhook/slack",
        );
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("not a JSON object"));
    }

    #[test]
    fn test_ensure_slack_events_subscription_no_url_key() {
        let mut manifest = serde_json::json!({
            "settings": {
                "event_subscriptions": {
                    "bot_events": ["message.channels", "message.groups", "message.im", "message.mpim"]
                }
            }
        });
        let changed = WebhookConfigurator::ensure_slack_events_subscription(
            &mut manifest,
            "https://example.com/webhook/slack",
        )
        .unwrap();
        assert!(changed, "Should add request_url when missing");
    }

    #[test]
    fn test_ensure_slack_events_subscription_preserves_extra_events() {
        let mut manifest = serde_json::json!({
            "settings": {
                "event_subscriptions": {
                    "request_url": "https://example.com/webhook/slack",
                    "bot_events": ["message.channels", "message.groups", "message.im", "message.mpim", "app_mention"]
                }
            }
        });
        let changed = WebhookConfigurator::ensure_slack_events_subscription(
            &mut manifest,
            "https://example.com/webhook/slack",
        )
        .unwrap();
        assert!(
            !changed,
            "Nothing to change when all required events present"
        );
        let events = manifest["settings"]["event_subscriptions"]["bot_events"]
            .as_array()
            .unwrap();
        // Original 5 events should be preserved
        assert_eq!(events.len(), 5);
    }

    #[test]
    fn test_ensure_slack_events_subscription_empty_bot_events_array() {
        let mut manifest = serde_json::json!({
            "settings": {
                "event_subscriptions": {
                    "request_url": "https://example.com/webhook/slack",
                    "bot_events": []
                }
            }
        });
        let changed = WebhookConfigurator::ensure_slack_events_subscription(
            &mut manifest,
            "https://example.com/webhook/slack",
        )
        .unwrap();
        assert!(changed, "Should add all 4 events to empty array");
        let events = manifest["settings"]["event_subscriptions"]["bot_events"]
            .as_array()
            .unwrap();
        assert_eq!(events.len(), 4);
    }

    // ---- needs_configuration: Telegram ----

    #[test]
    fn test_needs_configuration_telegram_source_enabled_with_bot_token() {
        let mut config = test_config();
        config.notifiers.telegram.source_enabled = true;
        config.notifiers.telegram.bot_token =
            Some(claudear_core::secret::SecretValue::new("123:ABC"));
        let configurator = WebhookConfigurator::new(config, "/tmp/.env");
        assert!(configurator.needs_configuration());
    }

    #[test]
    fn test_needs_configuration_telegram_source_disabled() {
        let mut config = test_config();
        config.notifiers.telegram.source_enabled = false;
        config.notifiers.telegram.bot_token =
            Some(claudear_core::secret::SecretValue::new("123:ABC"));
        let configurator = WebhookConfigurator::new(config, "/tmp/.env");
        assert!(!configurator.needs_configuration());
    }

    #[test]
    fn test_needs_configuration_telegram_no_bot_token() {
        let mut config = test_config();
        config.notifiers.telegram.source_enabled = true;
        config.notifiers.telegram.bot_token = None;
        let configurator = WebhookConfigurator::new(config, "/tmp/.env");
        assert!(!configurator.needs_configuration());
    }

    #[test]
    fn test_needs_configuration_telegram_empty_bot_token() {
        let mut config = test_config();
        config.notifiers.telegram.source_enabled = true;
        config.notifiers.telegram.bot_token = Some(claudear_core::secret::SecretValue::new("  "));
        let configurator = WebhookConfigurator::new(config, "/tmp/.env");
        assert!(!configurator.needs_configuration());
    }

    // ---- needs_configuration: Slack source ----

    #[test]
    fn test_needs_configuration_slack_source_all_present() {
        let mut config = test_config();
        config.issues.slack = Some(claudear_config::config::SlackSourceConfig {
            app_id: Some("A123".to_string()),
            app_config_token: Some(claudear_core::secret::SecretValue::new("xoxe-config")),
            signing_secret: Some(claudear_core::secret::SecretValue::new("signing123")),
            ..Default::default()
        });
        let configurator = WebhookConfigurator::new(config, "/tmp/.env");
        assert!(configurator.needs_configuration());
    }

    #[test]
    fn test_needs_configuration_slack_source_missing_app_id() {
        let mut config = test_config();
        config.issues.slack = Some(claudear_config::config::SlackSourceConfig {
            app_id: None,
            app_config_token: Some(claudear_core::secret::SecretValue::new("xoxe-config")),
            signing_secret: Some(claudear_core::secret::SecretValue::new("signing123")),
            ..Default::default()
        });
        let configurator = WebhookConfigurator::new(config, "/tmp/.env");
        assert!(!configurator.needs_configuration());
    }

    #[test]
    fn test_needs_configuration_slack_source_missing_config_token() {
        let mut config = test_config();
        config.issues.slack = Some(claudear_config::config::SlackSourceConfig {
            app_id: Some("A123".to_string()),
            app_config_token: None,
            signing_secret: Some(claudear_core::secret::SecretValue::new("signing123")),
            ..Default::default()
        });
        let configurator = WebhookConfigurator::new(config, "/tmp/.env");
        assert!(!configurator.needs_configuration());
    }

    #[test]
    fn test_needs_configuration_slack_source_missing_signing_secret() {
        let mut config = test_config();
        config.issues.slack = Some(claudear_config::config::SlackSourceConfig {
            app_id: Some("A123".to_string()),
            app_config_token: Some(claudear_core::secret::SecretValue::new("xoxe-config")),
            signing_secret: None,
            ..Default::default()
        });
        let configurator = WebhookConfigurator::new(config, "/tmp/.env");
        assert!(!configurator.needs_configuration());
    }

    #[test]
    fn test_needs_configuration_slack_source_empty_app_id() {
        let mut config = test_config();
        config.issues.slack = Some(claudear_config::config::SlackSourceConfig {
            app_id: Some("  ".to_string()),
            app_config_token: Some(claudear_core::secret::SecretValue::new("xoxe-config")),
            signing_secret: Some(claudear_core::secret::SecretValue::new("signing123")),
            ..Default::default()
        });
        let configurator = WebhookConfigurator::new(config, "/tmp/.env");
        assert!(!configurator.needs_configuration());
    }

    // ---- needs_configuration: WhatsApp ----

    #[test]
    fn test_needs_configuration_whatsapp_all_present() {
        let mut config = test_config();
        config.notifiers.whatsapp.source_enabled = true;
        config.notifiers.whatsapp.access_token =
            Some(claudear_core::secret::SecretValue::new("wa-token"));
        config.notifiers.whatsapp.business_account_id = Some("12345".to_string());
        config.notifiers.whatsapp.app_secret =
            Some(claudear_core::secret::SecretValue::new("app-secret"));
        let configurator = WebhookConfigurator::new(config, "/tmp/.env");
        assert!(configurator.needs_configuration());
    }

    #[test]
    fn test_needs_configuration_whatsapp_source_disabled() {
        let mut config = test_config();
        config.notifiers.whatsapp.source_enabled = false;
        config.notifiers.whatsapp.access_token =
            Some(claudear_core::secret::SecretValue::new("wa-token"));
        config.notifiers.whatsapp.business_account_id = Some("12345".to_string());
        config.notifiers.whatsapp.app_secret =
            Some(claudear_core::secret::SecretValue::new("app-secret"));
        let configurator = WebhookConfigurator::new(config, "/tmp/.env");
        assert!(!configurator.needs_configuration());
    }

    #[test]
    fn test_needs_configuration_whatsapp_missing_access_token() {
        let mut config = test_config();
        config.notifiers.whatsapp.source_enabled = true;
        config.notifiers.whatsapp.access_token = None;
        config.notifiers.whatsapp.business_account_id = Some("12345".to_string());
        config.notifiers.whatsapp.app_secret =
            Some(claudear_core::secret::SecretValue::new("app-secret"));
        let configurator = WebhookConfigurator::new(config, "/tmp/.env");
        assert!(!configurator.needs_configuration());
    }

    #[test]
    fn test_needs_configuration_whatsapp_missing_business_account_id() {
        let mut config = test_config();
        config.notifiers.whatsapp.source_enabled = true;
        config.notifiers.whatsapp.access_token =
            Some(claudear_core::secret::SecretValue::new("wa-token"));
        config.notifiers.whatsapp.business_account_id = None;
        config.notifiers.whatsapp.app_secret =
            Some(claudear_core::secret::SecretValue::new("app-secret"));
        let configurator = WebhookConfigurator::new(config, "/tmp/.env");
        assert!(!configurator.needs_configuration());
    }

    #[test]
    fn test_needs_configuration_whatsapp_missing_app_secret() {
        let mut config = test_config();
        config.notifiers.whatsapp.source_enabled = true;
        config.notifiers.whatsapp.access_token =
            Some(claudear_core::secret::SecretValue::new("wa-token"));
        config.notifiers.whatsapp.business_account_id = Some("12345".to_string());
        config.notifiers.whatsapp.app_secret = None;
        let configurator = WebhookConfigurator::new(config, "/tmp/.env");
        assert!(!configurator.needs_configuration());
    }

    #[test]
    fn test_needs_configuration_whatsapp_empty_access_token() {
        let mut config = test_config();
        config.notifiers.whatsapp.source_enabled = true;
        config.notifiers.whatsapp.access_token =
            Some(claudear_core::secret::SecretValue::new("  "));
        config.notifiers.whatsapp.business_account_id = Some("12345".to_string());
        config.notifiers.whatsapp.app_secret =
            Some(claudear_core::secret::SecretValue::new("app-secret"));
        let configurator = WebhookConfigurator::new(config, "/tmp/.env");
        assert!(!configurator.needs_configuration());
    }

    // ---- needs_configuration: Discord notifier missing one field ----

    #[test]
    fn test_needs_configuration_discord_notifier_only_bot_token() {
        let mut config = test_config();
        config.notifiers.discord.bot_token =
            Some(claudear_core::secret::SecretValue::new("bot-tok"));
        config.notifiers.discord.channel_id = None;
        config.notifiers.discord.webhook_url = None;
        let configurator = WebhookConfigurator::new(config, "/tmp/.env");
        // Needs both bot_token AND channel_id
        assert!(!configurator.needs_configuration());
    }

    #[test]
    fn test_needs_configuration_discord_notifier_only_channel_id() {
        let mut config = test_config();
        config.notifiers.discord.bot_token = None;
        config.notifiers.discord.channel_id = Some("12345".to_string());
        config.notifiers.discord.webhook_url = None;
        let configurator = WebhookConfigurator::new(config, "/tmp/.env");
        assert!(!configurator.needs_configuration());
    }

    #[test]
    fn test_needs_configuration_discord_notifier_empty_webhook_url() {
        let mut config = test_config();
        config.notifiers.discord.bot_token =
            Some(claudear_core::secret::SecretValue::new("bot-tok"));
        config.notifiers.discord.channel_id = Some("12345".to_string());
        config.notifiers.discord.webhook_url = Some(claudear_core::secret::SecretValue::new("  "));
        let configurator = WebhookConfigurator::new(config, "/tmp/.env");
        // Empty webhook_url counts as not present
        assert!(configurator.needs_configuration());
    }

    // ---- needs_configuration: GitLab edge cases ----

    #[test]
    fn test_needs_configuration_gitlab_no_token() {
        let mut config = test_config();
        config.scm.gitlab = Some(claudear_config::config::GitLabConfig {
            enabled: true,
            token: None,
            groups: vec!["grp".to_string()],
            webhook_secret: None,
            ..Default::default()
        });
        let configurator = WebhookConfigurator::new(config, "/tmp/.env");
        assert!(!configurator.needs_configuration());
    }

    #[test]
    fn test_needs_configuration_gitlab_empty_groups() {
        let mut config = test_config();
        config.scm.gitlab = Some(claudear_config::config::GitLabConfig {
            enabled: true,
            token: Some(claudear_core::secret::SecretValue::new("glpat")),
            groups: vec![],
            webhook_secret: None,
            ..Default::default()
        });
        let configurator = WebhookConfigurator::new(config, "/tmp/.env");
        assert!(!configurator.needs_configuration());
    }

    #[test]
    fn test_needs_configuration_gitlab_disabled() {
        let mut config = test_config();
        config.scm.gitlab = Some(claudear_config::config::GitLabConfig {
            enabled: false,
            token: Some(claudear_core::secret::SecretValue::new("glpat")),
            groups: vec!["grp".to_string()],
            webhook_secret: None,
            ..Default::default()
        });
        let configurator = WebhookConfigurator::new(config, "/tmp/.env");
        assert!(!configurator.needs_configuration());
    }

    // ---- needs_configuration: GitHub edge cases ----

    #[test]
    fn test_needs_configuration_github_no_repos() {
        let mut config = test_config();
        config.scm.github.token = Some(claudear_core::secret::SecretValue::new("ghp_test"));
        config.scm.github.repos = vec![];
        config.scm.github.webhook_secret = None;
        let configurator = WebhookConfigurator::new(config, "/tmp/.env");
        assert!(!configurator.needs_configuration());
    }

    #[test]
    fn test_needs_configuration_github_no_token_no_app() {
        let mut config = test_config();
        config.scm.github.token = None;
        config.scm.github.repos = vec!["owner/repo".to_string()];
        config.scm.github.webhook_secret = None;
        config.scm.github.app = claudear_config::config::GitHubAppConfig::default();
        let configurator = WebhookConfigurator::new(config, "/tmp/.env");
        // No token and no app configured => needs_configuration returns false for github
        assert!(!configurator.needs_configuration());
    }

    #[test]
    fn test_needs_configuration_github_app_with_app_webhook_secret() {
        let mut config = test_config();
        config.scm.github.token = None;
        config.scm.github.repos = vec!["owner/repo".to_string()];
        config.scm.github.app.app_id = Some(12345);
        config.scm.github.app.private_key = Some(claudear_core::secret::SecretValue::new(
            "-----BEGIN RSA PRIVATE KEY-----\nfake\n-----END RSA PRIVATE KEY-----",
        ));
        config.scm.github.app.webhook_secret =
            Some(claudear_core::secret::SecretValue::new("app-wh-secret"));
        config.scm.github.webhook_secret = None;
        let configurator = WebhookConfigurator::new(config, "/tmp/.env");
        // App webhook_secret is present -> no need for config
        assert!(!configurator.needs_configuration());
    }

    // ---- needs_configuration: combined edge cases ----

    #[test]
    fn test_needs_configuration_multiple_sources_any_true_returns_true() {
        let mut config = test_config();
        // Only telegram needs config
        config.notifiers.telegram.source_enabled = true;
        config.notifiers.telegram.bot_token =
            Some(claudear_core::secret::SecretValue::new("123:ABC"));
        let configurator = WebhookConfigurator::new(config, "/tmp/.env");
        assert!(configurator.needs_configuration());
    }

    // ---- Deserialization tests for internal structs ----

    #[test]
    fn test_github_hook_list_item_deser() {
        let json = r#"{"config": {"url": "https://example.com/hook"}}"#;
        let item: GitHubHookListItem = serde_json::from_str(json).unwrap();
        assert_eq!(item.config.url.as_deref(), Some("https://example.com/hook"));
    }

    #[test]
    fn test_github_hook_list_item_deser_empty_config() {
        let json = r#"{}"#;
        let item: GitHubHookListItem = serde_json::from_str(json).unwrap();
        assert!(item.config.url.is_none());
    }

    #[test]
    fn test_github_hook_config_deser_no_url() {
        let json = r#"{"content_type": "json"}"#;
        let cfg: GitHubHookConfig = serde_json::from_str(json).unwrap();
        assert!(cfg.url.is_none());
    }

    #[test]
    fn test_github_hook_config_default() {
        let cfg = GitHubHookConfig::default();
        assert!(cfg.url.is_none());
    }

    #[test]
    fn test_gitlab_hook_list_item_deser() {
        let json = r#"{"url": "https://gitlab.example.com/hook", "id": 42}"#;
        let item: GitLabHookListItem = serde_json::from_str(json).unwrap();
        assert_eq!(item.url.as_deref(), Some("https://gitlab.example.com/hook"));
    }

    #[test]
    fn test_gitlab_hook_list_item_deser_no_url() {
        let json = r#"{"id": 42}"#;
        let item: GitLabHookListItem = serde_json::from_str(json).unwrap();
        assert!(item.url.is_none());
    }

    #[test]
    fn test_telegram_api_response_deser_ok() {
        let json = r#"{"ok": true}"#;
        let resp: TelegramApiResponse = serde_json::from_str(json).unwrap();
        assert!(resp.ok);
        assert!(resp.description.is_none());
    }

    #[test]
    fn test_telegram_api_response_deser_error() {
        let json = r#"{"ok": false, "description": "Unauthorized"}"#;
        let resp: TelegramApiResponse = serde_json::from_str(json).unwrap();
        assert!(!resp.ok);
        assert_eq!(resp.description.as_deref(), Some("Unauthorized"));
    }

    #[test]
    fn test_slack_api_ok_response_deser_ok() {
        let json = r#"{"ok": true}"#;
        let resp: SlackApiOkResponse = serde_json::from_str(json).unwrap();
        assert!(resp.ok);
        assert!(resp.error.is_none());
    }

    #[test]
    fn test_slack_api_ok_response_deser_error() {
        let json = r#"{"ok": false, "error": "invalid_auth"}"#;
        let resp: SlackApiOkResponse = serde_json::from_str(json).unwrap();
        assert!(!resp.ok);
        assert_eq!(resp.error.as_deref(), Some("invalid_auth"));
    }

    #[test]
    fn test_discord_webhook_create_response_deser() {
        let json = r#"{"id": "1234567890", "token": "webhook-token-abc"}"#;
        let resp: DiscordWebhookCreateResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.id, "1234567890");
        assert_eq!(resp.token.as_deref(), Some("webhook-token-abc"));
    }

    #[test]
    fn test_discord_webhook_create_response_deser_no_token() {
        let json = r#"{"id": "1234567890"}"#;
        let resp: DiscordWebhookCreateResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.id, "1234567890");
        assert!(resp.token.is_none());
    }

    // ---- configure() async: Jira path variations ----

    #[tokio::test]
    async fn test_configure_jira_enabled_empty_base_url() {
        let mut config = test_config();
        config.issues.jira = Some(claudear_config::config::JiraConfig {
            enabled: true,
            base_url: "  ".to_string(),
            api_token: "jira-token".into(),
            ..Default::default()
        });
        let configurator = WebhookConfigurator::new(config, "/tmp/test-jira-empty-url.env");

        let result = configurator.configure("https://example.com").await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        // Should have a note about empty base_url
        assert!(
            err_msg.contains("jira.base_url is empty") || err_msg.contains("No auto-configurable"),
            "Expected jira base_url empty note, got: {}",
            err_msg
        );
    }

    #[tokio::test]
    async fn test_configure_jira_enabled_empty_api_token() {
        let mut config = test_config();
        config.issues.jira = Some(claudear_config::config::JiraConfig {
            enabled: true,
            base_url: "https://example.atlassian.net".to_string(),
            api_token: "".into(),
            ..Default::default()
        });
        let configurator = WebhookConfigurator::new(config, "/tmp/test-jira-empty-token.env");

        let result = configurator.configure("https://example.com").await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("jira.api_token is empty") || err_msg.contains("No auto-configurable"),
            "Expected jira api_token empty note, got: {}",
            err_msg
        );
    }

    // ---- configure() async: Telegram path variations ----

    #[tokio::test]
    async fn test_configure_telegram_source_enabled_no_bot_token() {
        let mut config = test_config();
        config.notifiers.telegram.source_enabled = true;
        config.notifiers.telegram.bot_token = None;
        let configurator = WebhookConfigurator::new(config, "/tmp/test-tg-no-token.env");

        let result = configurator.configure("https://example.com").await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("telegram.bot_token is missing")
                || err_msg.contains("No auto-configurable"),
            "Expected telegram missing token note, got: {}",
            err_msg
        );
    }

    #[tokio::test]
    async fn test_configure_telegram_source_enabled_empty_bot_token() {
        let mut config = test_config();
        config.notifiers.telegram.source_enabled = true;
        config.notifiers.telegram.bot_token = Some(claudear_core::secret::SecretValue::new("  "));
        let configurator = WebhookConfigurator::new(config, "/tmp/test-tg-empty-token.env");

        let result = configurator.configure("https://example.com").await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("telegram.bot_token is missing")
                || err_msg.contains("No auto-configurable"),
            "Expected telegram missing token note, got: {}",
            err_msg
        );
    }

    // ---- configure() async: Discord notifier path variations ----

    #[tokio::test]
    async fn test_configure_discord_notifier_only_bot_token_note() {
        let mut config = test_config();
        config.notifiers.discord.bot_token =
            Some(claudear_core::secret::SecretValue::new("bot-tok"));
        config.notifiers.discord.channel_id = None;
        config.notifiers.discord.webhook_url = None;
        let configurator = WebhookConfigurator::new(config, "/tmp/test-discord-bot-only.env");

        let result = configurator.configure("https://example.com").await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("discord.channel_id") || err_msg.contains("No auto-configurable"),
            "Expected discord channel_id missing note, got: {}",
            err_msg
        );
    }

    #[tokio::test]
    async fn test_configure_discord_notifier_only_channel_id_note() {
        let mut config = test_config();
        config.notifiers.discord.bot_token = None;
        config.notifiers.discord.channel_id = Some("12345".to_string());
        config.notifiers.discord.webhook_url = None;
        let configurator = WebhookConfigurator::new(config, "/tmp/test-discord-chan-only.env");

        let result = configurator.configure("https://example.com").await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("discord.bot_token") || err_msg.contains("No auto-configurable"),
            "Expected discord bot_token missing note, got: {}",
            err_msg
        );
    }

    #[tokio::test]
    async fn test_configure_discord_notifier_both_missing_no_note() {
        // When neither bot_token nor channel_id is present, no note is emitted for discord notifier
        let mut config = test_config();
        config.notifiers.discord.bot_token = None;
        config.notifiers.discord.channel_id = None;
        config.notifiers.discord.webhook_url = None;
        let configurator = WebhookConfigurator::new(config, "/tmp/test-discord-both-missing.env");

        let result = configurator.configure("https://example.com").await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        // Should NOT contain discord notes since neither field is present
        assert!(
            !err_msg.contains("discord.bot_token") && !err_msg.contains("discord.channel_id"),
            "Should not mention discord fields when both missing, got: {}",
            err_msg
        );
    }

    // ---- configure() async: Slack source path variations ----

    #[tokio::test]
    async fn test_configure_slack_source_missing_all_fields() {
        let mut config = test_config();
        config.issues.slack = Some(claudear_config::config::SlackSourceConfig {
            bot_token: Some(claudear_core::secret::SecretValue::new("xoxb-test")),
            channel_id: Some("C123".to_string()),
            app_id: None,
            app_config_token: None,
            signing_secret: None,
            ..Default::default()
        });
        let configurator = WebhookConfigurator::new(config, "/tmp/test-slack-missing-all.env");

        let result = configurator.configure("https://example.com").await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("slack.app_id")
                && err_msg.contains("slack.app_config_token")
                && err_msg.contains("slack.signing_secret"),
            "Should list all missing Slack fields, got: {}",
            err_msg
        );
    }

    #[tokio::test]
    async fn test_configure_slack_source_missing_one_field() {
        let mut config = test_config();
        config.issues.slack = Some(claudear_config::config::SlackSourceConfig {
            app_id: Some("A123".to_string()),
            app_config_token: Some(claudear_core::secret::SecretValue::new("xoxe-config")),
            signing_secret: None, // Only this missing
            ..Default::default()
        });
        let configurator = WebhookConfigurator::new(config, "/tmp/test-slack-missing-one.env");

        let result = configurator.configure("https://example.com").await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("slack.signing_secret"),
            "Should mention missing signing_secret, got: {}",
            err_msg
        );
    }

    // ---- configure() async: WhatsApp path variations ----

    #[tokio::test]
    async fn test_configure_whatsapp_source_missing_fields() {
        let mut config = test_config();
        config.notifiers.whatsapp.source_enabled = true;
        config.notifiers.whatsapp.access_token = None;
        config.notifiers.whatsapp.business_account_id = None;
        config.notifiers.whatsapp.app_secret = None;
        let configurator = WebhookConfigurator::new(config, "/tmp/test-wa-missing.env");

        let result = configurator.configure("https://example.com").await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("whatsapp.access_token"),
            "Should mention missing whatsapp fields, got: {}",
            err_msg
        );
    }

    #[tokio::test]
    async fn test_configure_whatsapp_source_missing_only_business_account() {
        let mut config = test_config();
        config.notifiers.whatsapp.source_enabled = true;
        config.notifiers.whatsapp.access_token =
            Some(claudear_core::secret::SecretValue::new("wa-tok"));
        config.notifiers.whatsapp.business_account_id = None;
        config.notifiers.whatsapp.app_secret =
            Some(claudear_core::secret::SecretValue::new("app-sec"));
        let configurator = WebhookConfigurator::new(config, "/tmp/test-wa-no-biz.env");

        let result = configurator.configure("https://example.com").await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("whatsapp.business_account_id"),
            "Should mention missing business_account_id, got: {}",
            err_msg
        );
    }

    // ---- configure() async: GitLab path variations ----

    #[tokio::test]
    async fn test_configure_gitlab_enabled_no_groups() {
        let mut config = test_config();
        config.scm.gitlab = Some(claudear_config::config::GitLabConfig {
            enabled: true,
            token: Some(claudear_core::secret::SecretValue::new("glpat_test")),
            groups: vec![],
            ..Default::default()
        });
        let configurator = WebhookConfigurator::new(config, "/tmp/test-gl-no-groups.env");

        let result = configurator.configure("https://example.com").await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("no groups are configured")
                || err_msg.contains("No auto-configurable"),
            "Expected gitlab no groups note, got: {}",
            err_msg
        );
    }

    #[tokio::test]
    async fn test_configure_gitlab_enabled_no_token() {
        let mut config = test_config();
        config.scm.gitlab = Some(claudear_config::config::GitLabConfig {
            enabled: true,
            token: None,
            groups: vec!["mygroup".to_string()],
            ..Default::default()
        });
        let configurator = WebhookConfigurator::new(config, "/tmp/test-gl-no-token.env");

        let result = configurator.configure("https://example.com").await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("no token is configured") || err_msg.contains("No auto-configurable"),
            "Expected gitlab no token note, got: {}",
            err_msg
        );
    }

    // ---- configure() async: GitHub path variations ----

    #[tokio::test]
    async fn test_configure_github_repos_no_token_no_app() {
        let mut config = test_config();
        config.scm.github.token = None;
        config.scm.github.repos = vec!["owner/repo".to_string()];
        config.scm.github.app = claudear_config::config::GitHubAppConfig::default();
        let configurator = WebhookConfigurator::new(config, "/tmp/test-gh-no-token-no-app.env");

        let result = configurator.configure("https://example.com").await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("no GitHub token") || err_msg.contains("No auto-configurable"),
            "Expected github no token note, got: {}",
            err_msg
        );
    }

    // ---- configure() async: Slack notifier with webhook present does NOT emit note ----

    #[tokio::test]
    async fn test_configure_slack_notifier_has_webhook_no_note() {
        let mut config = test_config();
        config.notifiers.slack.webhook_url = Some(claudear_core::secret::SecretValue::new(
            "https://hooks.slack.com/xxx",
        ));
        config.notifiers.slack.bot_token =
            Some(claudear_core::secret::SecretValue::new("xoxb-test"));
        config.notifiers.slack.channel_id = Some("C123".to_string());
        let configurator = WebhookConfigurator::new(config, "/tmp/test-slack-notifier-webhook.env");

        // Will fail because no auto-configurable sources
        let result = configurator.configure("https://example.com").await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        // Should NOT contain the Slack notifier note since webhook_url is present
        assert!(
            !err_msg.contains("Slack notifier incoming webhook URL"),
            "Should not emit Slack notifier note when webhook_url present, got: {}",
            err_msg
        );
    }

    // ---- configure() async: Slack notifier no webhook, no bot_channel ----

    #[tokio::test]
    async fn test_configure_slack_notifier_no_webhook_no_bot() {
        let mut config = test_config();
        config.notifiers.slack.webhook_url = None;
        config.notifiers.slack.bot_token = None;
        config.notifiers.slack.channel_id = None;
        let configurator = WebhookConfigurator::new(config, "/tmp/test-slack-notifier-nothing.env");

        let result = configurator.configure("https://example.com").await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        // No note should be emitted when neither webhook nor bot+channel present
        assert!(
            !err_msg.contains("Slack notifier"),
            "Should not emit Slack notifier note when nothing configured, got: {}",
            err_msg
        );
    }

    // ---- configure() async: Discord source present adds polling note ----

    #[tokio::test]
    async fn test_configure_discord_source_with_other_source_adds_note() {
        // Discord source present along with an auto-configurable source that fails
        let mut config = test_config();
        config.issues.discord = Some(claudear_config::config::DiscordSourceConfig {
            bot_token: Some(claudear_core::secret::SecretValue::new("discord-bot")),
            channel_id: Some("123".to_string()),
            ..Default::default()
        });
        config.issues.linear = Some(claudear_config::config::LinearConfig {
            enabled: true,
            api_key: "bad-key".into(),
            webhook_secret: None,
            ..Default::default()
        });
        let configurator =
            WebhookConfigurator::new(config, "/tmp/test-discord-source-with-linear.env");

        let result = configurator.configure("https://example.com").await;
        // Will fail because linear API fails
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Failed to configure any webhooks"),
            "Should fail with warning, got: {}",
            err_msg
        );
    }

    // ---- configure() async: WhatsApp source enabled, API will fail ----

    #[tokio::test]
    async fn test_configure_whatsapp_source_all_present_api_fails() {
        let mut config = test_config();
        config.notifiers.whatsapp.source_enabled = true;
        config.notifiers.whatsapp.access_token =
            Some(claudear_core::secret::SecretValue::new("wa-tok"));
        config.notifiers.whatsapp.business_account_id = Some("12345".to_string());
        config.notifiers.whatsapp.app_secret =
            Some(claudear_core::secret::SecretValue::new("app-sec"));
        let configurator = WebhookConfigurator::new(config, "/tmp/test-wa-fail.env");

        let result = configurator.configure("https://example.com").await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("WhatsApp") || err_msg.contains("Failed to configure"),
            "Expected WhatsApp failure, got: {}",
            err_msg
        );
    }

    // ---- configure() async: Telegram source enabled, API will fail ----

    #[tokio::test]
    async fn test_configure_telegram_source_with_valid_token_api_fails() {
        let mut config = test_config();
        config.notifiers.telegram.source_enabled = true;
        config.notifiers.telegram.bot_token =
            Some(claudear_core::secret::SecretValue::new("123:ABCdef"));
        let configurator = WebhookConfigurator::new(config, "/tmp/test-tg-fail.env");

        let result = configurator.configure("https://example.com").await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Telegram") || err_msg.contains("Failed to configure"),
            "Expected Telegram failure, got: {}",
            err_msg
        );
    }

    // ---- configure() async: GitLab with groups and token, API fails ----

    #[tokio::test]
    async fn test_configure_gitlab_with_groups_and_token_api_fails() {
        let mut config = test_config();
        config.scm.gitlab = Some(claudear_config::config::GitLabConfig {
            enabled: true,
            token: Some(claudear_core::secret::SecretValue::new("glpat_test")),
            groups: vec!["mygroup".to_string()],
            webhook_secret: None,
            ..Default::default()
        });
        let configurator = WebhookConfigurator::new(config, "/tmp/test-gl-api-fail.env");

        let result = configurator.configure("https://example.com").await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("GitLab") || err_msg.contains("Failed to configure"),
            "Expected GitLab failure, got: {}",
            err_msg
        );
    }

    // ---- configure() async: GitHub PAT with repos, API fails ----

    #[tokio::test]
    async fn test_configure_github_with_token_and_repos_api_fails() {
        let mut config = test_config();
        config.scm.github.token = Some(claudear_core::secret::SecretValue::new("ghp_test"));
        config.scm.github.repos = vec!["owner/repo".to_string()];
        config.scm.github.webhook_secret = None;
        let configurator = WebhookConfigurator::new(config, "/tmp/test-gh-api-fail.env");

        let result = configurator.configure("https://example.com").await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("GitHub") || err_msg.contains("Failed to configure"),
            "Expected GitHub failure, got: {}",
            err_msg
        );
    }

    // ---- configure() async: Discord notifier bot+channel present, API fails ----

    #[tokio::test]
    async fn test_configure_discord_notifier_bot_channel_api_fails() {
        let mut config = test_config();
        config.notifiers.discord.bot_token =
            Some(claudear_core::secret::SecretValue::new("bot-tok"));
        config.notifiers.discord.channel_id = Some("12345".to_string());
        config.notifiers.discord.webhook_url = None;
        let configurator = WebhookConfigurator::new(config, "/tmp/test-discord-api-fail.env");

        let result = configurator.configure("https://example.com").await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Discord") || err_msg.contains("Failed to configure"),
            "Expected Discord failure, got: {}",
            err_msg
        );
    }

    // ---- configure() async: Multiple auto-configurable sources all fail ----

    #[tokio::test]
    async fn test_configure_multiple_sources_all_fail() {
        let mut config = test_config();
        config.issues.linear = Some(claudear_config::config::LinearConfig {
            enabled: true,
            api_key: "bad-linear".into(),
            webhook_secret: None,
            ..Default::default()
        });
        config.issues.sentry = Some(claudear_config::config::SentryConfig {
            enabled: true,
            auth_token: "bad-sentry".into(),
            org_slug: "fake".to_string(),
            project_slugs: vec!["proj".to_string()],
            client_secret: None,
            ..Default::default()
        });
        config.scm.github.token = Some(claudear_core::secret::SecretValue::new("ghp_bad"));
        config.scm.github.repos = vec!["owner/repo".to_string()];
        config.scm.github.webhook_secret = None;

        let configurator = WebhookConfigurator::new(config, "/tmp/test-multi-fail.env");
        let result = configurator.configure("https://example.com").await;
        assert!(result.is_err());
    }

    // ---- configure() async: Jira disabled ----

    #[tokio::test]
    async fn test_configure_jira_disabled_no_effect() {
        let mut config = test_config();
        config.issues.jira = Some(claudear_config::config::JiraConfig {
            enabled: false,
            base_url: "https://example.atlassian.net".to_string(),
            api_token: "jira-token".into(),
            ..Default::default()
        });
        let configurator = WebhookConfigurator::new(config, "/tmp/test-jira-disabled.env");

        let result = configurator.configure("https://example.com").await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        // Jira disabled should not contribute any warnings about Jira
        assert!(
            !err_msg.contains("Jira webhook"),
            "Disabled Jira should not show warnings, got: {}",
            err_msg
        );
    }

    // ---- configure() async: Telegram source disabled ----

    #[tokio::test]
    async fn test_configure_telegram_source_disabled_no_effect() {
        let mut config = test_config();
        config.notifiers.telegram.source_enabled = false;
        config.notifiers.telegram.bot_token =
            Some(claudear_core::secret::SecretValue::new("123:ABC"));
        let configurator = WebhookConfigurator::new(config, "/tmp/test-tg-disabled.env");

        let result = configurator.configure("https://example.com").await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        // Should not contain telegram notes
        assert!(
            !err_msg.contains("Telegram webhook"),
            "Disabled Telegram should not show webhook warnings, got: {}",
            err_msg
        );
    }

    // ---- configure() async: WhatsApp source disabled ----

    #[tokio::test]
    async fn test_configure_whatsapp_source_disabled_no_effect() {
        let mut config = test_config();
        config.notifiers.whatsapp.source_enabled = false;
        config.notifiers.whatsapp.access_token =
            Some(claudear_core::secret::SecretValue::new("wa-tok"));
        config.notifiers.whatsapp.business_account_id = Some("12345".to_string());
        config.notifiers.whatsapp.app_secret =
            Some(claudear_core::secret::SecretValue::new("app-sec"));
        let configurator = WebhookConfigurator::new(config, "/tmp/test-wa-disabled.env");

        let result = configurator.configure("https://example.com").await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            !err_msg.contains("WhatsApp webhook"),
            "Disabled WhatsApp should not show webhook warnings, got: {}",
            err_msg
        );
    }

    // ---- configure() async: GitLab disabled ----

    #[tokio::test]
    async fn test_configure_gitlab_disabled_no_effect() {
        let mut config = test_config();
        config.scm.gitlab = Some(claudear_config::config::GitLabConfig {
            enabled: false,
            token: Some(claudear_core::secret::SecretValue::new("glpat_test")),
            groups: vec!["mygroup".to_string()],
            ..Default::default()
        });
        let configurator = WebhookConfigurator::new(config, "/tmp/test-gl-disabled.env");

        let result = configurator.configure("https://example.com").await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            !err_msg.contains("GitLab issue webhooks"),
            "Disabled GitLab should not show webhook configured, got: {}",
            err_msg
        );
    }

    // ---- configure() async: Discord notifier webhook already present ----

    #[tokio::test]
    async fn test_configure_discord_notifier_webhook_already_present_no_auto_setup() {
        let mut config = test_config();
        config.notifiers.discord.webhook_url = Some(claudear_core::secret::SecretValue::new(
            "https://discord.com/api/webhooks/1/token",
        ));
        config.notifiers.discord.bot_token =
            Some(claudear_core::secret::SecretValue::new("bot-tok"));
        config.notifiers.discord.channel_id = Some("12345".to_string());
        let configurator = WebhookConfigurator::new(config, "/tmp/test-discord-wh-present.env");

        let result = configurator.configure("https://example.com").await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        // Should NOT attempt Discord notifier auto-setup when webhook_url is present
        assert!(
            !err_msg.contains("Discord notifier webhook"),
            "Should not attempt auto-setup when webhook_url present, got: {}",
            err_msg
        );
    }

    // ---- configure() async: Slack source with all tokens, API fails ----

    #[tokio::test]
    async fn test_configure_slack_source_all_tokens_api_fails() {
        let mut config = test_config();
        config.issues.slack = Some(claudear_config::config::SlackSourceConfig {
            app_id: Some("A123".to_string()),
            app_config_token: Some(claudear_core::secret::SecretValue::new("xoxe-config")),
            signing_secret: Some(claudear_core::secret::SecretValue::new("signing123")),
            ..Default::default()
        });
        let configurator = WebhookConfigurator::new(config, "/tmp/test-slack-source-api-fail.env");

        let result = configurator.configure("https://example.com").await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Slack") || err_msg.contains("Failed to configure"),
            "Expected Slack failure, got: {}",
            err_msg
        );
    }

    // ---- GitHub App path variations in configure() ----

    #[tokio::test]
    async fn test_configure_github_app_configured_api_fails() {
        let mut config = test_config();
        config.scm.github.token = None;
        config.scm.github.repos = vec!["owner/repo".to_string()];
        config.scm.github.app.app_id = Some(12345);
        config.scm.github.app.private_key = Some(claudear_core::secret::SecretValue::new(
            "-----BEGIN RSA PRIVATE KEY-----\nfake\n-----END RSA PRIVATE KEY-----",
        ));
        config.scm.github.webhook_secret = None;
        let configurator = WebhookConfigurator::new(config, "/tmp/test-gh-app-fail.env");

        let result = configurator.configure("https://example.com").await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("GitHub") || err_msg.contains("Failed to configure"),
            "Expected GitHub App failure, got: {}",
            err_msg
        );
    }

    // ---- GitHub no repos does not attempt setup ----

    #[tokio::test]
    async fn test_configure_github_no_repos_no_attempt() {
        let mut config = test_config();
        config.scm.github.token = Some(claudear_core::secret::SecretValue::new("ghp_test"));
        config.scm.github.repos = vec![];
        let configurator = WebhookConfigurator::new(config, "/tmp/test-gh-no-repos.env");

        let result = configurator.configure("https://example.com").await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        // Should not mention GitHub setup failure since no repos
        assert!(
            !err_msg.contains("GitHub webhook setup failed"),
            "No repos means no GitHub attempt, got: {}",
            err_msg
        );
    }

    // ---- Webhook setup result: print with sentry zero projects and secret ----

    #[test]
    fn test_print_setup_result_sentry_zero_projects_with_secret() {
        let result = WebhookSetupResult {
            linear_configured: false,
            linear_webhook_id: None,
            linear_secret: None,
            sentry_configured: true,
            sentry_project_count: 0,
            sentry_secret: Some("secret1234567890".to_string()),
            warnings: vec![],
        };
        // Should not panic
        print_setup_result(&result);
    }

    // ---- Combined path: multiple notes accumulation ----

    #[tokio::test]
    async fn test_configure_many_notes_accumulation() {
        let mut config = test_config();
        // Discord source -> polling note
        config.issues.discord = Some(claudear_config::config::DiscordSourceConfig {
            bot_token: Some(claudear_core::secret::SecretValue::new("discord-bot")),
            channel_id: Some("123".to_string()),
            ..Default::default()
        });
        // Telegram no token -> missing note
        config.notifiers.telegram.source_enabled = true;
        config.notifiers.telegram.bot_token = None;
        // WhatsApp missing fields -> missing note
        config.notifiers.whatsapp.source_enabled = true;
        config.notifiers.whatsapp.access_token = None;
        // Slack notifier with bot+channel, no webhook -> chat.postMessage note
        config.notifiers.slack.bot_token =
            Some(claudear_core::secret::SecretValue::new("xoxb-test"));
        config.notifiers.slack.channel_id = Some("C123".to_string());
        config.notifiers.slack.webhook_url = None;

        let configurator = WebhookConfigurator::new(config, "/tmp/test-many-notes.env");
        let result = configurator.configure("https://example.com").await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        // Should contain multiple notes
        assert!(
            err_msg.contains("Discord source uses channel polling"),
            "Expected discord polling note, got: {}",
            err_msg
        );
        assert!(
            err_msg.contains("telegram.bot_token is missing"),
            "Expected telegram token note, got: {}",
            err_msg
        );
    }

    // ---- Jira base_url with just whitespace ----

    #[tokio::test]
    async fn test_configure_jira_enabled_whitespace_base_url() {
        let mut config = test_config();
        config.issues.jira = Some(claudear_config::config::JiraConfig {
            enabled: true,
            base_url: "   ".to_string(),
            api_token: "jira-token".into(),
            ..Default::default()
        });
        let configurator = WebhookConfigurator::new(config, "/tmp/test-jira-ws-url.env");

        let result = configurator.configure("https://example.com").await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("jira.base_url is empty") || err_msg.contains("No auto-configurable"),
            "Expected jira base_url empty note for whitespace, got: {}",
            err_msg
        );
    }

    // ---- Slack source with only app_id missing ----

    #[tokio::test]
    async fn test_configure_slack_source_only_app_id_missing() {
        let mut config = test_config();
        config.issues.slack = Some(claudear_config::config::SlackSourceConfig {
            app_id: None,
            app_config_token: Some(claudear_core::secret::SecretValue::new("xoxe-config")),
            signing_secret: Some(claudear_core::secret::SecretValue::new("signing")),
            ..Default::default()
        });
        let configurator = WebhookConfigurator::new(config, "/tmp/test-slack-no-appid.env");

        let result = configurator.configure("https://example.com").await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("slack.app_id"),
            "Should mention missing app_id, got: {}",
            err_msg
        );
        // Should NOT mention the fields that ARE present
        assert!(
            !err_msg.contains("slack.app_config_token")
                && !err_msg.contains("slack.signing_secret"),
            "Should not mention present fields, got: {}",
            err_msg
        );
    }

    // ---- WhatsApp missing only app_secret ----

    #[tokio::test]
    async fn test_configure_whatsapp_source_missing_only_app_secret() {
        let mut config = test_config();
        config.notifiers.whatsapp.source_enabled = true;
        config.notifiers.whatsapp.access_token =
            Some(claudear_core::secret::SecretValue::new("wa-tok"));
        config.notifiers.whatsapp.business_account_id = Some("12345".to_string());
        config.notifiers.whatsapp.app_secret = None;
        let configurator = WebhookConfigurator::new(config, "/tmp/test-wa-no-secret.env");

        let result = configurator.configure("https://example.com").await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("whatsapp.app_secret"),
            "Should mention missing app_secret, got: {}",
            err_msg
        );
    }

    // ---- ensure_slack_events_subscription: manifest with only settings key ----

    #[test]
    fn test_ensure_slack_events_subscription_settings_exists_no_event_subs() {
        let mut manifest = serde_json::json!({
            "settings": {
                "other_key": "value"
            }
        });
        let changed = WebhookConfigurator::ensure_slack_events_subscription(
            &mut manifest,
            "https://example.com/webhook/slack",
        )
        .unwrap();
        assert!(changed);
        assert!(manifest["settings"]["event_subscriptions"].is_object());
    }

    // ---- GitHub repos with no webhooks secret => generates secret ----

    #[test]
    fn test_needs_configuration_github_token_repos_no_secret() {
        let mut config = test_config();
        config.scm.github.token = Some(claudear_core::secret::SecretValue::new("ghp_test"));
        config.scm.github.repos = vec!["a/b".to_string(), "c/d".to_string()];
        config.scm.github.webhook_secret = None;
        let configurator = WebhookConfigurator::new(config, "/tmp/.env");
        assert!(configurator.needs_configuration());
    }

    // ---- Jira needs configuration ----

    #[test]
    fn test_needs_configuration_jira_enabled_base_url_token() {
        let mut config = test_config();
        config.issues.jira = Some(claudear_config::config::JiraConfig {
            enabled: true,
            base_url: "https://jira.example.com".to_string(),
            api_token: "tok".into(),
            ..Default::default()
        });
        let configurator = WebhookConfigurator::new(config, "/tmp/.env");
        assert!(configurator.needs_configuration());
    }

    #[test]
    fn test_needs_configuration_jira_disabled() {
        let mut config = test_config();
        config.issues.jira = Some(claudear_config::config::JiraConfig {
            enabled: false,
            base_url: "https://jira.example.com".to_string(),
            api_token: "tok".into(),
            ..Default::default()
        });
        let configurator = WebhookConfigurator::new(config, "/tmp/.env");
        assert!(!configurator.needs_configuration());
    }

    #[test]
    fn test_needs_configuration_jira_empty_base_url() {
        let mut config = test_config();
        config.issues.jira = Some(claudear_config::config::JiraConfig {
            enabled: true,
            base_url: "".to_string(),
            api_token: "tok".into(),
            ..Default::default()
        });
        let configurator = WebhookConfigurator::new(config, "/tmp/.env");
        // Empty base_url means !c.base_url.trim().is_empty() is false
        assert!(!configurator.needs_configuration());
    }

    #[test]
    fn test_needs_configuration_jira_empty_api_token() {
        let mut config = test_config();
        config.issues.jira = Some(claudear_config::config::JiraConfig {
            enabled: true,
            base_url: "https://jira.example.com".to_string(),
            api_token: "".into(),
            ..Default::default()
        });
        let configurator = WebhookConfigurator::new(config, "/tmp/.env");
        // Empty api_token => is_empty() returns true => needs_config false
        assert!(!configurator.needs_configuration());
    }

    // ---- Comprehensive needs_configuration: all sources enabled ----

    #[test]
    fn test_needs_configuration_all_sources_enabled_various_secrets() {
        let mut config = test_config();
        // Linear: needs config (no secret)
        config.issues.linear = Some(claudear_config::config::LinearConfig {
            enabled: true,
            webhook_secret: None,
            ..Default::default()
        });
        // Sentry: configured (has secret)
        config.issues.sentry = Some(claudear_config::config::SentryConfig {
            enabled: true,
            client_secret: Some(claudear_core::secret::SecretValue::new("sen-sec")),
            ..Default::default()
        });
        // GitHub: configured (has secret)
        config.scm.github.token = Some(claudear_core::secret::SecretValue::new("ghp"));
        config.scm.github.repos = vec!["o/r".to_string()];
        config.scm.github.webhook_secret = Some(claudear_core::secret::SecretValue::new("gh-sec"));
        // GitLab: needs config (no secret)
        config.scm.gitlab = Some(claudear_config::config::GitLabConfig {
            enabled: true,
            token: Some(claudear_core::secret::SecretValue::new("glpat")),
            groups: vec!["grp".to_string()],
            webhook_secret: None,
            ..Default::default()
        });
        // Overall: should need config because Linear and GitLab need it
        let configurator = WebhookConfigurator::new(config, "/tmp/.env");
        assert!(configurator.needs_configuration());
    }
}
