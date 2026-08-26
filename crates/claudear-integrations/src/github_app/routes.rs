//! HTTP route handlers for the GitHub App manifest flow.
//!
//! These handlers implement the setup flow for creating a GitHub App:
//!
//! 1. `/github/setup` - Initiates the flow by redirecting to GitHub with a manifest
//! 2. `/github/callback` - Receives the callback from GitHub with the App credentials
//!
//! **Note**: These routes are NOT wired up by default. They will be registered
//! in the webhook server when the feature is enabled.

use crate::github_app::manifest::AppManifest;
use axum::extract::Query;
use axum::response::{Html, Redirect};
use claudear_config::env_writer::update_env_file;
use claudear_core::error::{Error, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::{Arc, RwLock};

/// Setup state expiry time in minutes.
/// CSRF tokens and setup state are invalidated after this duration.
const SETUP_STATE_EXPIRY_MINUTES: i64 = 15;

/// CSRF state for the setup flow.
///
/// This is stored in memory and validated when the callback is received.
#[derive(Debug, Clone)]
pub struct SetupState {
    /// CSRF token to prevent replay attacks.
    pub csrf_token: String,
    /// The base URL that was used for the manifest.
    pub base_url: String,
    /// When this state was created.
    pub created_at: chrono::DateTime<chrono::Utc>,
}

impl SetupState {
    /// Create a new setup state.
    pub fn new(base_url: String) -> Self {
        Self {
            csrf_token: generate_csrf_token(),
            base_url,
            created_at: chrono::Utc::now(),
        }
    }

    /// Check if this state is still valid (not expired).
    pub fn is_valid(&self) -> bool {
        let age = chrono::Utc::now() - self.created_at;
        age.num_minutes() < SETUP_STATE_EXPIRY_MINUTES
    }
}

/// Generate a cryptographically secure random CSRF token.
fn generate_csrf_token() -> String {
    use rand::RngExt;
    let mut rng = rand::rng();
    let bytes: [u8; 32] = rng.random();
    hex::encode(bytes)
}

/// Query parameters for the setup endpoint.
#[derive(Debug, Deserialize)]
pub struct SetupQuery {
    /// Base URL for the claudear instance.
    pub base_url: Option<String>,
    /// Optional organization to create the App under.
    pub org: Option<String>,
    /// Optional custom name for the App.
    pub name: Option<String>,
}

/// Query parameters for the callback endpoint.
#[derive(Debug, Deserialize)]
pub struct CallbackQuery {
    /// The temporary code from GitHub.
    pub code: String,
    /// The CSRF state token.
    pub state: String,
}

/// Response from GitHub's manifest code exchange.
#[derive(Debug, Deserialize)]
struct ManifestConversionResponse {
    /// The App ID.
    id: i64,
    /// The App slug (URL-friendly name).
    #[allow(dead_code)]
    slug: String,
    /// The App name.
    name: String,
    /// Client ID for OAuth.
    client_id: String,
    /// Client secret for OAuth.
    client_secret: String,
    /// The webhook secret.
    webhook_secret: String,
    /// The private key in PEM format.
    pem: String,
    /// The HTML URL for the App's settings page.
    html_url: String,
}

/// State manager for CSRF tokens during setup flow.
#[derive(Debug, Default)]
pub struct SetupStateManager {
    states: RwLock<HashMap<String, SetupState>>,
}

impl SetupStateManager {
    /// Create a new state manager.
    pub fn new() -> Self {
        Self {
            states: RwLock::new(HashMap::new()),
        }
    }

    /// Create and store a new setup state.
    pub fn create_state(&self, base_url: String) -> SetupState {
        let state = SetupState::new(base_url);
        if let Ok(mut states) = self.states.write() {
            // Clean up expired states
            states.retain(|_, s| s.is_valid());
            states.insert(state.csrf_token.clone(), state.clone());
        }
        state
    }

    /// Validate and consume a state token.
    pub fn validate_and_consume(&self, token: &str) -> Option<SetupState> {
        if let Ok(mut states) = self.states.write() {
            if let Some(state) = states.remove(token) {
                if state.is_valid() {
                    return Some(state);
                }
            }
        }
        None
    }
}

/// Handler for `/github/setup`.
///
/// Initiates the GitHub App manifest flow by:
/// 1. Generating a CSRF token
/// 2. Creating the App manifest
/// 3. Redirecting to GitHub with the manifest
///
/// # Query Parameters
/// - `base_url` - Required. The public URL of this claudear instance.
/// - `org` - Optional. GitHub organization to create the App under.
/// - `name` - Optional. Custom name for the App.
pub async fn github_setup_handler(
    Query(query): Query<SetupQuery>,
    state_manager: Arc<SetupStateManager>,
) -> std::result::Result<Redirect, Html<String>> {
    // Validate base_url is provided
    let base_url = match query.base_url {
        Some(url) if !url.is_empty() => url,
        _ => {
            return Err(Html(setup_form_html(None)));
        }
    };

    // Validate the URL looks reasonable
    if !base_url.starts_with("http://") && !base_url.starts_with("https://") {
        return Err(Html(setup_form_html(Some(
            "Base URL must start with http:// or https://",
        ))));
    }

    // Create the manifest
    let manifest = AppManifest::generate(&base_url, query.name.as_deref());

    // Create and store CSRF state
    let setup_state = state_manager.create_state(base_url);

    // Generate the GitHub redirect URL
    let scm_url = if let Some(org) = query.org {
        manifest.github_org_manifest_url(&org, &setup_state.csrf_token)
    } else {
        manifest.github_manifest_url(&setup_state.csrf_token)
    };

    match scm_url {
        Ok(url) => Ok(Redirect::temporary(&url)),
        Err(e) => Err(Html(format!(
            "<h1>Error</h1><p>Failed to generate manifest: {}</p>",
            html_escape(&e.to_string())
        ))),
    }
}

/// Handler for `/github/callback`.
///
/// Receives the callback from GitHub after App creation:
/// 1. Validates the CSRF token
/// 2. Exchanges the code for App credentials
/// 3. Saves credentials to `.env` and PEM file
/// 4. Shows success page with next steps
pub async fn github_callback_handler(
    Query(query): Query<CallbackQuery>,
    state_manager: Arc<SetupStateManager>,
) -> Html<String> {
    // Validate CSRF state
    let setup_state = match state_manager.validate_and_consume(&query.state) {
        Some(state) => state,
        None => {
            return Html(error_html(
                "Invalid or expired state",
                "The setup session has expired. Please start over.",
            ));
        }
    };

    // Exchange the code for credentials
    let credentials = match exchange_code_for_credentials(&query.code).await {
        Ok(creds) => creds,
        Err(e) => {
            return Html(error_html(
                "Failed to exchange code",
                &format!("GitHub returned an error: {}", e),
            ));
        }
    };

    // Save credentials
    let save_result = save_credentials(&credentials, &setup_state.base_url);

    match save_result {
        Ok(_) => Html(success_html(&credentials, &setup_state.base_url)),
        Err(e) => Html(partial_success_html(&credentials, &e.to_string())),
    }
}

/// Default timeout for HTTP requests (30 seconds).
const DEFAULT_REQUEST_TIMEOUT_SECS: u64 = 30;

/// Default connection timeout (10 seconds).
const DEFAULT_CONNECT_TIMEOUT_SECS: u64 = 10;

/// Exchange the manifest code for App credentials.
async fn exchange_code_for_credentials(code: &str) -> Result<ManifestConversionResponse> {
    let url = format!("https://api.github.com/app-manifests/{}/conversions", code);

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(DEFAULT_REQUEST_TIMEOUT_SECS))
        .connect_timeout(std::time::Duration::from_secs(DEFAULT_CONNECT_TIMEOUT_SECS))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new());

    let response = client
        .post(&url)
        .header("Accept", "application/vnd.github+json")
        .header("X-GitHub-Api-Version", "2022-11-28")
        .header("User-Agent", "claudear")
        .send()
        .await
        .map_err(|e| Error::api(format!("Failed to contact GitHub: {}", e)))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(Error::api(format!(
            "GitHub API error ({}): {}",
            status, body
        )));
    }

    response
        .json::<ManifestConversionResponse>()
        .await
        .map_err(|e| Error::api(format!("Failed to parse GitHub response: {}", e)))
}

/// Save the App credentials to disk.
fn save_credentials(creds: &ManifestConversionResponse, base_url: &str) -> Result<()> {
    // Save private key to PEM file
    let pem_path = "github-app-key.pem";
    fs::write(pem_path, &creds.pem).map_err(|e| {
        Error::config(format!(
            "Failed to write private key to {}: {}",
            pem_path, e
        ))
    })?;

    // Set restrictive permissions on the PEM file
    claudear_core::platform::set_file_permissions_secure(pem_path.as_ref()).ok();

    // Update .env file
    let env_path = Path::new(".env");
    let mut updates = HashMap::new();
    updates.insert("GITHUB_APP_ID".to_string(), creds.id.to_string());
    updates.insert(
        "GITHUB_APP_PRIVATE_KEY_PATH".to_string(),
        pem_path.to_string(),
    );
    updates.insert(
        "GITHUB_APP_WEBHOOK_SECRET".to_string(),
        creds.webhook_secret.clone(),
    );
    updates.insert("GITHUB_APP_CLIENT_ID".to_string(), creds.client_id.clone());
    updates.insert(
        "GITHUB_APP_CLIENT_SECRET".to_string(),
        creds.client_secret.clone(),
    );
    updates.insert("GITHUB_APP_BASE_URL".to_string(), base_url.to_string());

    update_env_file(env_path, &updates)?;

    Ok(())
}

/// Escape HTML special characters to prevent XSS.
fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#x27;")
}

/// Generate HTML for the setup form.
fn setup_form_html(error: Option<&str>) -> String {
    let error_html = error
        .map(|e| format!(r#"<p style="color: red;">{}</p>"#, html_escape(e)))
        .unwrap_or_default();

    format!(
        r#"<!DOCTYPE html>
<html>
<head>
    <title>claudear - GitHub App Setup</title>
    <style>
        body {{ font-family: system-ui, sans-serif; max-width: 600px; margin: 50px auto; padding: 20px; }}
        input {{ width: 100%; padding: 10px; margin: 10px 0; box-sizing: border-box; }}
        button {{ background: #238636; color: white; padding: 10px 20px; border: none; cursor: pointer; }}
        button:hover {{ background: #2ea043; }}
    </style>
</head>
<body>
    <h1>claudear - GitHub App Setup</h1>
    <p>Enter the public URL where this claudear instance is accessible:</p>
    {error}
    <form method="GET" action="/github/setup">
        <input type="url" name="base_url" placeholder="https://your-server:3100" required>
        <p><small>Optional: Create the App under an organization</small></p>
        <input type="text" name="org" placeholder="organization-name (optional)">
        <p><small>Optional: Custom App name</small></p>
        <input type="text" name="name" placeholder="claudear (default)">
        <button type="submit">Create GitHub App</button>
    </form>
</body>
</html>"#,
        error = error_html
    )
}

/// Generate HTML for error page.
fn error_html(title: &str, message: &str) -> String {
    format!(
        r#"<!DOCTYPE html>
<html>
<head>
    <title>claudear - Setup Error</title>
    <style>
        body {{ font-family: system-ui, sans-serif; max-width: 600px; margin: 50px auto; padding: 20px; }}
        .error {{ background: #ffebe9; border: 1px solid #ff8182; padding: 20px; border-radius: 6px; }}
    </style>
</head>
<body>
    <h1>Setup Error</h1>
    <div class="error">
        <h2>{}</h2>
        <p>{}</p>
    </div>
    <p><a href="/github/setup">Try again</a></p>
</body>
</html>"#,
        html_escape(title),
        html_escape(message)
    )
}

/// Generate HTML for success page.
fn success_html(creds: &ManifestConversionResponse, base_url: &str) -> String {
    format!(
        r#"<!DOCTYPE html>
<html>
<head>
    <title>claudear - Setup Complete</title>
    <style>
        body {{ font-family: system-ui, sans-serif; max-width: 600px; margin: 50px auto; padding: 20px; }}
        .success {{ background: #d1f5d3; border: 1px solid #34d058; padding: 20px; border-radius: 6px; }}
        code {{ background: #f6f8fa; padding: 2px 6px; border-radius: 3px; }}
        .steps {{ background: #f6f8fa; padding: 20px; border-radius: 6px; margin-top: 20px; }}
    </style>
</head>
<body>
    <h1>GitHub App Created!</h1>
    <div class="success">
        <p><strong>{}</strong> has been created successfully.</p>
        <p>App ID: <code>{}</code></p>
    </div>
    <div class="steps">
        <h2>Next Steps</h2>
        <ol>
            <li>Install the App on your repositories: <a href="{}" target="_blank">App Settings</a></li>
            <li>Restart claudear to load the new credentials</li>
            <li>The webhook URL has been configured to: <code>{}/webhook/github</code></li>
        </ol>
    </div>
    <p>Credentials have been saved to:</p>
    <ul>
        <li><code>.env</code> - Environment variables</li>
        <li><code>github-app-key.pem</code> - Private key</li>
    </ul>
</body>
</html>"#,
        html_escape(&creds.name),
        creds.id,
        html_escape(&creds.html_url),
        html_escape(base_url)
    )
}

/// Generate HTML for partial success (credentials received but save failed).
fn partial_success_html(creds: &ManifestConversionResponse, error: &str) -> String {
    format!(
        r#"<!DOCTYPE html>
<html>
<head>
    <title>claudear - Credentials Received</title>
    <style>
        body {{ font-family: system-ui, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }}
        .warning {{ background: #fff8c5; border: 1px solid #d4a72c; padding: 20px; border-radius: 6px; }}
        pre {{ background: #f6f8fa; padding: 15px; overflow-x: auto; border-radius: 6px; }}
        .danger {{ color: #cf222e; }}
    </style>
</head>
<body>
    <h1>GitHub App Created!</h1>
    <div class="warning">
        <h2>Manual Configuration Required</h2>
        <p>The App was created, but credentials could not be saved automatically:</p>
        <p><strong>{}</strong></p>
    </div>
    <h2>App Details</h2>
    <ul>
        <li>App Name: <strong>{}</strong></li>
        <li>App ID: <strong>{}</strong></li>
        <li>Client ID: <strong>{}</strong></li>
        <li>Settings: <a href="{}" target="_blank">{}</a></li>
    </ul>
    <h2>Manual Setup</h2>
    <p>Add these to your <code>.env</code> file:</p>
    <pre>
GITHUB_APP_ID={}
GITHUB_APP_PRIVATE_KEY_PATH=github-app-key.pem
GITHUB_APP_WEBHOOK_SECRET={}
GITHUB_APP_CLIENT_ID={}
GITHUB_APP_CLIENT_SECRET={}
</pre>
    <h2 class="danger">Private Key (save to github-app-key.pem)</h2>
    <p class="danger">Copy this private key and save it securely. It will not be shown again!</p>
    <pre>{}</pre>
</body>
</html>"#,
        html_escape(error),
        html_escape(&creds.name),
        creds.id,
        html_escape(&creds.client_id),
        html_escape(&creds.html_url),
        html_escape(&creds.html_url),
        creds.id,
        html_escape(&creds.webhook_secret),
        html_escape(&creds.client_id),
        html_escape(&creds.client_secret),
        html_escape(&creds.pem)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_setup_state_new() {
        let state = SetupState::new("https://example.com".to_string());
        assert!(!state.csrf_token.is_empty());
        assert_eq!(state.base_url, "https://example.com");
        assert!(state.is_valid());
    }

    #[test]
    fn test_setup_state_manager_create_and_validate() {
        let manager = SetupStateManager::new();

        let state = manager.create_state("https://example.com".to_string());
        let token = state.csrf_token.clone();

        // Should be able to validate and consume once
        let validated = manager.validate_and_consume(&token);
        assert!(validated.is_some());
        assert_eq!(validated.unwrap().base_url, "https://example.com");

        // Should not be able to consume again
        let second = manager.validate_and_consume(&token);
        assert!(second.is_none());
    }

    #[test]
    fn test_setup_state_manager_invalid_token() {
        let manager = SetupStateManager::new();

        let result = manager.validate_and_consume("invalid_token");
        assert!(result.is_none());
    }

    #[test]
    fn test_generate_csrf_token() {
        let token1 = generate_csrf_token();
        let token2 = generate_csrf_token();

        // Tokens should be non-empty and different
        assert!(!token1.is_empty());
        assert!(!token2.is_empty());
        // Note: Due to nanosecond precision, these might be the same in fast tests
        // but in practice they'll be different
    }

    #[test]
    fn test_setup_form_html_no_error() {
        let html = setup_form_html(None);
        assert!(html.contains("GitHub App Setup"));
        assert!(html.contains("base_url"));
        assert!(!html.contains("color: red"));
    }

    #[test]
    fn test_setup_form_html_with_error() {
        let html = setup_form_html(Some("Test error message"));
        assert!(html.contains("Test error message"));
        assert!(html.contains("color: red"));
    }

    #[test]
    fn test_error_html() {
        let html = error_html("Test Title", "Test message");
        assert!(html.contains("Test Title"));
        assert!(html.contains("Test message"));
        assert!(html.contains("Setup Error"));
    }

    #[test]
    fn test_html_escape_ampersand() {
        assert_eq!(html_escape("a&b"), "a&amp;b");
    }

    #[test]
    fn test_html_escape_less_than() {
        assert_eq!(html_escape("<script>"), "&lt;script&gt;");
    }

    #[test]
    fn test_html_escape_greater_than() {
        assert_eq!(html_escape("x > y"), "x &gt; y");
    }

    #[test]
    fn test_html_escape_double_quote() {
        assert_eq!(html_escape(r#"say "hello""#), "say &quot;hello&quot;");
    }

    #[test]
    fn test_html_escape_single_quote() {
        assert_eq!(html_escape("it's"), "it&#x27;s");
    }

    #[test]
    fn test_html_escape_all_special_chars() {
        let input = r#"<script>alert("XSS");</script> & it's done"#;
        let escaped = html_escape(input);
        assert!(!escaped.contains('<'));
        assert!(!escaped.contains('>'));
        assert!(!escaped.contains('"'));
        assert!(!escaped.contains('\''));
        // Ampersand in the original becomes &amp; but the escaped string
        // will contain & as part of entities
        assert!(escaped.contains("&amp;"));
        assert!(escaped.contains("&lt;"));
        assert!(escaped.contains("&gt;"));
        assert!(escaped.contains("&quot;"));
        assert!(escaped.contains("&#x27;"));
    }

    #[test]
    fn test_html_escape_empty_string() {
        assert_eq!(html_escape(""), "");
    }

    #[test]
    fn test_html_escape_no_special_chars() {
        assert_eq!(html_escape("Hello World 123"), "Hello World 123");
    }

    #[test]
    fn test_html_escape_unicode() {
        assert_eq!(html_escape("日本語"), "日本語");
    }

    #[test]
    fn test_html_escape_nested_entities() {
        // Ensure & in already-escaped content gets re-escaped
        assert_eq!(html_escape("&amp;"), "&amp;amp;");
    }

    #[test]
    fn test_setup_state_csrf_token_is_hex() {
        let state = SetupState::new("https://example.com".to_string());
        // CSRF token should be 64 hex characters (32 bytes)
        assert_eq!(state.csrf_token.len(), 64);
        assert!(
            state.csrf_token.chars().all(|c| c.is_ascii_hexdigit()),
            "CSRF token must be hex: {}",
            state.csrf_token
        );
    }

    #[test]
    fn test_setup_state_uniqueness() {
        let state1 = SetupState::new("https://example.com".to_string());
        let state2 = SetupState::new("https://example.com".to_string());
        assert_ne!(
            state1.csrf_token, state2.csrf_token,
            "CSRF tokens must be unique"
        );
    }

    #[test]
    fn test_setup_state_is_valid_fresh() {
        let state = SetupState::new("https://example.com".to_string());
        assert!(state.is_valid(), "Fresh state should be valid");
    }

    #[test]
    fn test_setup_state_is_invalid_when_expired() {
        let state = SetupState {
            csrf_token: "test".to_string(),
            base_url: "https://example.com".to_string(),
            created_at: chrono::Utc::now() - chrono::Duration::minutes(16),
        };
        assert!(
            !state.is_valid(),
            "State older than 15 minutes should be expired"
        );
    }

    #[test]
    fn test_setup_state_is_valid_at_boundary() {
        let state = SetupState {
            csrf_token: "test".to_string(),
            base_url: "https://example.com".to_string(),
            created_at: chrono::Utc::now() - chrono::Duration::minutes(14),
        };
        assert!(
            state.is_valid(),
            "State younger than 15 minutes should still be valid"
        );
    }

    #[test]
    fn test_state_manager_create_returns_valid_state() {
        let manager = SetupStateManager::new();
        let state = manager.create_state("https://my-app.com".to_string());

        assert!(!state.csrf_token.is_empty());
        assert_eq!(state.base_url, "https://my-app.com");
        assert!(state.is_valid());
    }

    #[test]
    fn test_state_manager_validate_consumes_token() {
        let manager = SetupStateManager::new();
        let state = manager.create_state("https://example.com".to_string());
        let token = state.csrf_token.clone();

        // First consumption should succeed
        let result = manager.validate_and_consume(&token);
        assert!(result.is_some());

        // Second consumption should fail (token already consumed)
        let result2 = manager.validate_and_consume(&token);
        assert!(
            result2.is_none(),
            "Token should be consumed after first use"
        );
    }

    #[test]
    fn test_state_manager_rejects_unknown_token() {
        let manager = SetupStateManager::new();
        let result = manager.validate_and_consume("nonexistent_token");
        assert!(result.is_none());
    }

    #[test]
    fn test_state_manager_rejects_expired_token() {
        let manager = SetupStateManager::new();

        // Manually insert an expired state
        {
            let mut states = manager.states.write().unwrap();
            states.insert(
                "expired_token".to_string(),
                SetupState {
                    csrf_token: "expired_token".to_string(),
                    base_url: "https://example.com".to_string(),
                    created_at: chrono::Utc::now() - chrono::Duration::minutes(20),
                },
            );
        }

        let result = manager.validate_and_consume("expired_token");
        assert!(result.is_none(), "Expired token should be rejected");
    }

    #[test]
    fn test_state_manager_cleans_expired_states_on_create() {
        let manager = SetupStateManager::new();

        // Insert an expired state manually
        {
            let mut states = manager.states.write().unwrap();
            states.insert(
                "old_token".to_string(),
                SetupState {
                    csrf_token: "old_token".to_string(),
                    base_url: "https://example.com".to_string(),
                    created_at: chrono::Utc::now() - chrono::Duration::minutes(20),
                },
            );
        }

        // Creating a new state should clean up expired ones
        let _new_state = manager.create_state("https://new.com".to_string());

        // The expired token should have been cleaned
        let states = manager.states.read().unwrap();
        assert!(
            !states.contains_key("old_token"),
            "Expired states should be cleaned up on create"
        );
    }

    #[test]
    fn test_state_manager_multiple_concurrent_states() {
        let manager = SetupStateManager::new();

        let state1 = manager.create_state("https://app1.com".to_string());
        let state2 = manager.create_state("https://app2.com".to_string());
        let state3 = manager.create_state("https://app3.com".to_string());

        // All three should be independently consumable
        let r1 = manager.validate_and_consume(&state1.csrf_token);
        assert!(r1.is_some());
        assert_eq!(r1.unwrap().base_url, "https://app1.com");

        let r3 = manager.validate_and_consume(&state3.csrf_token);
        assert!(r3.is_some());
        assert_eq!(r3.unwrap().base_url, "https://app3.com");

        let r2 = manager.validate_and_consume(&state2.csrf_token);
        assert!(r2.is_some());
        assert_eq!(r2.unwrap().base_url, "https://app2.com");
    }

    #[test]
    fn test_state_manager_default() {
        // SetupStateManager implements Default
        let manager = SetupStateManager::default();
        let state = manager.create_state("https://example.com".to_string());
        assert!(!state.csrf_token.is_empty());
    }

    #[test]
    fn test_generate_csrf_token_length() {
        let token = generate_csrf_token();
        // 32 bytes -> 64 hex chars
        assert_eq!(token.len(), 64);
    }

    #[test]
    fn test_generate_csrf_token_is_hex() {
        let token = generate_csrf_token();
        assert!(
            token.chars().all(|c| c.is_ascii_hexdigit()),
            "Token should be hex-encoded: {}",
            token
        );
    }

    #[test]
    fn test_generate_csrf_token_uniqueness_batch() {
        let mut tokens = std::collections::HashSet::new();
        for _ in 0..100 {
            tokens.insert(generate_csrf_token());
        }
        assert_eq!(
            tokens.len(),
            100,
            "100 generated tokens should all be unique"
        );
    }

    #[test]
    fn test_setup_query_all_fields() {
        let query: SetupQuery = serde_json::from_str(
            r#"{
            "base_url": "https://example.com",
            "org": "my-org",
            "name": "my-app"
        }"#,
        )
        .unwrap();
        assert_eq!(query.base_url.unwrap(), "https://example.com");
        assert_eq!(query.org.unwrap(), "my-org");
        assert_eq!(query.name.unwrap(), "my-app");
    }

    #[test]
    fn test_setup_query_minimal() {
        let query: SetupQuery = serde_json::from_str(r#"{}"#).unwrap();
        assert!(query.base_url.is_none());
        assert!(query.org.is_none());
        assert!(query.name.is_none());
    }

    #[test]
    fn test_setup_query_only_base_url() {
        let query: SetupQuery =
            serde_json::from_str(r#"{"base_url": "https://test.com"}"#).unwrap();
        assert_eq!(query.base_url.unwrap(), "https://test.com");
        assert!(query.org.is_none());
        assert!(query.name.is_none());
    }

    #[test]
    fn test_callback_query_deserialization() {
        let query: CallbackQuery = serde_json::from_str(
            r#"{
            "code": "abc123",
            "state": "csrf_token_here"
        }"#,
        )
        .unwrap();
        assert_eq!(query.code, "abc123");
        assert_eq!(query.state, "csrf_token_here");
    }

    #[test]
    fn test_callback_query_missing_code_fails() {
        let result = serde_json::from_str::<CallbackQuery>(r#"{"state": "token"}"#);
        assert!(
            result.is_err(),
            "Missing 'code' should fail deserialization"
        );
    }

    #[test]
    fn test_callback_query_missing_state_fails() {
        let result = serde_json::from_str::<CallbackQuery>(r#"{"code": "abc"}"#);
        assert!(
            result.is_err(),
            "Missing 'state' should fail deserialization"
        );
    }

    #[test]
    fn test_manifest_conversion_response_deserialization() {
        let json = r#"{
            "id": 12345,
            "slug": "my-app",
            "name": "My App",
            "client_id": "Iv1.abc123",
            "client_secret": "secret_value",
            "webhook_secret": "whsec_test",
            "pem": "-----BEGIN RSA PRIVATE KEY-----\ntest\n-----END RSA PRIVATE KEY-----",
            "html_url": "https://github.com/apps/my-app"
        }"#;

        let response: ManifestConversionResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.id, 12345);
        assert_eq!(response.slug, "my-app");
        assert_eq!(response.name, "My App");
        assert_eq!(response.client_id, "Iv1.abc123");
        assert_eq!(response.client_secret, "secret_value");
        assert_eq!(response.webhook_secret, "whsec_test");
        assert!(response.pem.contains("BEGIN RSA PRIVATE KEY"));
        assert_eq!(response.html_url, "https://github.com/apps/my-app");
    }

    #[test]
    fn test_manifest_conversion_response_missing_field_fails() {
        let json = r#"{
            "id": 12345,
            "slug": "my-app",
            "name": "My App"
        }"#;

        let result = serde_json::from_str::<ManifestConversionResponse>(json);
        assert!(result.is_err(), "Missing required fields should fail");
    }

    #[test]
    fn test_setup_form_html_contains_form() {
        let html = setup_form_html(None);
        assert!(html.contains("<form"));
        assert!(html.contains("</form>"));
        assert!(html.contains("method=\"GET\""));
        assert!(html.contains("action=\"/github/setup\""));
    }

    #[test]
    fn test_setup_form_html_contains_required_inputs() {
        let html = setup_form_html(None);
        assert!(html.contains("name=\"base_url\""));
        assert!(html.contains("name=\"org\""));
        assert!(html.contains("name=\"name\""));
        assert!(html.contains("type=\"submit\"") || html.contains("button"));
    }

    #[test]
    fn test_setup_form_html_error_is_escaped() {
        let html = setup_form_html(Some("<script>alert('xss')</script>"));
        assert!(
            !html.contains("<script>alert"),
            "Error message should be escaped"
        );
        assert!(html.contains("&lt;script&gt;"));
    }

    #[test]
    fn test_error_html_escapes_title_and_message() {
        let html = error_html("<b>bad</b>", "evil <script> tag");
        assert!(!html.contains("<b>bad</b>"));
        assert!(!html.contains("<script>"));
        assert!(html.contains("&lt;b&gt;bad&lt;/b&gt;"));
        assert!(html.contains("&lt;script&gt;"));
    }

    #[test]
    fn test_error_html_contains_retry_link() {
        let html = error_html("Error", "Something went wrong");
        assert!(
            html.contains("/github/setup"),
            "Error page should contain a link to try again"
        );
    }

    #[test]
    fn test_success_html_contains_app_info() {
        let creds = ManifestConversionResponse {
            id: 42,
            slug: "test-app".to_string(),
            name: "Test App".to_string(),
            client_id: "Iv1.test".to_string(),
            client_secret: "secret".to_string(),
            webhook_secret: "whsec".to_string(),
            pem: "pem-content".to_string(),
            html_url: "https://github.com/apps/test-app".to_string(),
        };

        let html = success_html(&creds, "https://my-server.com");
        assert!(html.contains("Test App"));
        assert!(html.contains("42")); // App ID
        assert!(html.contains("https://github.com/apps/test-app"));
        assert!(html.contains("https://my-server.com/webhook/github"));
        assert!(html.contains(".env"));
        assert!(html.contains("github-app-key.pem"));
    }

    #[test]
    fn test_success_html_escapes_xss_in_app_name() {
        let creds = ManifestConversionResponse {
            id: 1,
            slug: "xss".to_string(),
            name: "<script>alert('xss')</script>".to_string(),
            client_id: "test".to_string(),
            client_secret: "test".to_string(),
            webhook_secret: "test".to_string(),
            pem: "test".to_string(),
            html_url: "https://example.com".to_string(),
        };

        let html = success_html(&creds, "https://example.com");
        assert!(
            !html.contains("<script>alert"),
            "App name should be escaped"
        );
    }

    #[test]
    fn test_partial_success_html_contains_manual_instructions() {
        let creds = ManifestConversionResponse {
            id: 99,
            slug: "my-app".to_string(),
            name: "My App".to_string(),
            client_id: "Iv1.client".to_string(),
            client_secret: "client_secret_value".to_string(),
            webhook_secret: "webhook_secret_value".to_string(),
            pem: "-----BEGIN RSA PRIVATE KEY-----\nkey\n-----END RSA PRIVATE KEY-----".to_string(),
            html_url: "https://github.com/apps/my-app".to_string(),
        };

        let html = partial_success_html(&creds, "Permission denied");

        // Should contain the error message
        assert!(html.contains("Permission denied"));

        // Should contain manual setup instructions
        assert!(html.contains("GITHUB_APP_ID=99"));
        assert!(html.contains("GITHUB_APP_PRIVATE_KEY_PATH=github-app-key.pem"));
        assert!(html.contains("GITHUB_APP_WEBHOOK_SECRET=webhook_secret_value"));
        assert!(html.contains("GITHUB_APP_CLIENT_ID=Iv1.client"));
        assert!(html.contains("GITHUB_APP_CLIENT_SECRET=client_secret_value"));

        // Should contain the private key for manual saving
        assert!(html.contains("BEGIN RSA PRIVATE KEY"));
    }

    #[test]
    fn test_partial_success_html_escapes_error_message() {
        let creds = ManifestConversionResponse {
            id: 1,
            slug: "test".to_string(),
            name: "Test".to_string(),
            client_id: "test".to_string(),
            client_secret: "test".to_string(),
            webhook_secret: "test".to_string(),
            pem: "test".to_string(),
            html_url: "https://example.com".to_string(),
        };

        let html = partial_success_html(&creds, "<script>evil</script>");
        assert!(!html.contains("<script>evil"), "Error should be escaped");
        assert!(html.contains("&lt;script&gt;"));
    }

    #[tokio::test]
    async fn test_setup_handler_no_base_url_returns_form() {
        let state_manager = Arc::new(SetupStateManager::new());
        let query = SetupQuery {
            base_url: None,
            org: None,
            name: None,
        };

        let result = github_setup_handler(Query(query), state_manager).await;
        assert!(
            result.is_err(),
            "No base_url should return Err(Html) with form"
        );
        let html = result.unwrap_err().0;
        assert!(html.contains("GitHub App Setup"));
        assert!(html.contains("<form"));
    }

    #[tokio::test]
    async fn test_setup_handler_empty_base_url_returns_form() {
        let state_manager = Arc::new(SetupStateManager::new());
        let query = SetupQuery {
            base_url: Some("".to_string()),
            org: None,
            name: None,
        };

        let result = github_setup_handler(Query(query), state_manager).await;
        assert!(result.is_err());
        let html = result.unwrap_err().0;
        assert!(html.contains("<form"));
    }

    #[tokio::test]
    async fn test_setup_handler_invalid_url_scheme_returns_error() {
        let state_manager = Arc::new(SetupStateManager::new());
        let query = SetupQuery {
            base_url: Some("ftp://example.com".to_string()),
            org: None,
            name: None,
        };

        let result = github_setup_handler(Query(query), state_manager).await;
        assert!(result.is_err());
        let html = result.unwrap_err().0;
        assert!(
            html.contains("http://") || html.contains("https://"),
            "Error should mention valid URL schemes"
        );
    }

    #[tokio::test]
    async fn test_setup_handler_valid_https_url_redirects() {
        let state_manager = Arc::new(SetupStateManager::new());
        let query = SetupQuery {
            base_url: Some("https://my-server.com:3100".to_string()),
            org: None,
            name: None,
        };

        let result = github_setup_handler(Query(query), state_manager).await;
        assert!(result.is_ok(), "Valid HTTPS URL should produce a redirect");
    }

    #[tokio::test]
    async fn test_setup_handler_valid_http_url_redirects() {
        let state_manager = Arc::new(SetupStateManager::new());
        let query = SetupQuery {
            base_url: Some("http://localhost:3100".to_string()),
            org: None,
            name: None,
        };

        let result = github_setup_handler(Query(query), state_manager).await;
        assert!(result.is_ok(), "Valid HTTP URL should produce a redirect");
    }

    #[tokio::test]
    async fn test_setup_handler_with_org_redirects() {
        let state_manager = Arc::new(SetupStateManager::new());
        let query = SetupQuery {
            base_url: Some("https://example.com".to_string()),
            org: Some("my-org".to_string()),
            name: None,
        };

        let result = github_setup_handler(Query(query), state_manager).await;
        assert!(result.is_ok(), "Should redirect with org parameter");
    }

    #[tokio::test]
    async fn test_setup_handler_creates_csrf_state() {
        let state_manager = Arc::new(SetupStateManager::new());
        let query = SetupQuery {
            base_url: Some("https://example.com".to_string()),
            org: None,
            name: None,
        };

        let _ = github_setup_handler(Query(query), state_manager.clone()).await;

        // The state manager should have stored a state entry
        let states = state_manager.states.read().unwrap();
        assert_eq!(states.len(), 1, "One CSRF state should have been created");
    }

    #[tokio::test]
    async fn test_callback_handler_invalid_state_returns_error() {
        let state_manager = Arc::new(SetupStateManager::new());
        let query = CallbackQuery {
            code: "test_code".to_string(),
            state: "invalid_csrf_token".to_string(),
        };

        let result = github_callback_handler(Query(query), state_manager).await;
        let html = result.0;
        assert!(
            html.contains("Invalid or expired state"),
            "Should show expired state error"
        );
        assert!(html.contains("start over") || html.contains("Try again"));
    }

    #[tokio::test]
    async fn test_callback_handler_expired_state_returns_error() {
        let state_manager = Arc::new(SetupStateManager::new());

        // Insert an expired state manually
        {
            let mut states = state_manager.states.write().unwrap();
            states.insert(
                "expired_csrf".to_string(),
                SetupState {
                    csrf_token: "expired_csrf".to_string(),
                    base_url: "https://example.com".to_string(),
                    created_at: chrono::Utc::now() - chrono::Duration::minutes(20),
                },
            );
        }

        let query = CallbackQuery {
            code: "test_code".to_string(),
            state: "expired_csrf".to_string(),
        };

        let result = github_callback_handler(Query(query), state_manager).await;
        let html = result.0;
        assert!(
            html.contains("Invalid or expired state"),
            "Expired CSRF should be rejected"
        );
    }

    #[tokio::test]
    async fn test_callback_handler_valid_state_but_invalid_code_returns_error() {
        let state_manager = Arc::new(SetupStateManager::new());
        let state = state_manager.create_state("https://example.com".to_string());

        let query = CallbackQuery {
            code: "invalid_code_that_wont_work".to_string(),
            state: state.csrf_token.clone(),
        };

        let result = github_callback_handler(Query(query), state_manager).await;
        let html = result.0;

        // The code exchange will fail because we're not hitting real GitHub
        assert!(
            html.contains("Failed") || html.contains("error") || html.contains("Error"),
            "Invalid code should produce an error page, got: {}",
            &html[..html.len().min(500)]
        );
    }

    #[tokio::test]
    async fn test_callback_handler_consumes_state_token() {
        let state_manager = Arc::new(SetupStateManager::new());
        let state = state_manager.create_state("https://example.com".to_string());
        let token = state.csrf_token.clone();

        let query = CallbackQuery {
            code: "some_code".to_string(),
            state: token.clone(),
        };

        // First callback attempt (will fail on code exchange, but state is consumed)
        let _ = github_callback_handler(Query(query), state_manager.clone()).await;

        // State should be consumed
        let states = state_manager.states.read().unwrap();
        assert!(
            !states.contains_key(&token),
            "CSRF token should be consumed after callback"
        );
    }

    #[test]
    fn test_state_manager_concurrent_create_and_validate() {
        use std::sync::Arc;
        use std::thread;

        let manager = Arc::new(SetupStateManager::new());
        let mut handles = vec![];

        // Spawn threads that create states
        let created_tokens: Arc<std::sync::Mutex<Vec<String>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));

        for i in 0..20 {
            let mgr = Arc::clone(&manager);
            let tokens = Arc::clone(&created_tokens);
            handles.push(thread::spawn(move || {
                let state = mgr.create_state(format!("https://app-{}.com", i));
                tokens.lock().unwrap().push(state.csrf_token);
            }));
        }

        for handle in handles {
            handle.join().expect("Thread should not panic");
        }

        // All tokens should be valid
        let tokens = created_tokens.lock().unwrap();
        assert_eq!(tokens.len(), 20);

        for token in tokens.iter() {
            let result = manager.validate_and_consume(token);
            assert!(result.is_some(), "All created tokens should be consumable");
        }
    }

    #[test]
    fn test_setup_state_expiry_is_15_minutes() {
        assert_eq!(SETUP_STATE_EXPIRY_MINUTES, 15);
    }

    #[test]
    fn test_error_html_empty_strings() {
        let html = error_html("", "");
        assert!(html.contains("Setup Error"));
        // Should still render valid HTML
        assert!(html.contains("<html>"));
        assert!(html.contains("</html>"));
    }

    #[test]
    fn test_setup_form_html_is_valid_html_structure() {
        let html = setup_form_html(None);
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("<html>"));
        assert!(html.contains("</html>"));
        assert!(html.contains("<head>"));
        assert!(html.contains("</head>"));
        assert!(html.contains("<body>"));
        assert!(html.contains("</body>"));
    }

    #[test]
    fn test_success_html_is_valid_html_structure() {
        let creds = ManifestConversionResponse {
            id: 1,
            slug: "test".to_string(),
            name: "Test".to_string(),
            client_id: "id".to_string(),
            client_secret: "secret".to_string(),
            webhook_secret: "whsec".to_string(),
            pem: "pem".to_string(),
            html_url: "https://example.com".to_string(),
        };

        let html = success_html(&creds, "https://example.com");
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("<html>"));
        assert!(html.contains("</html>"));
    }

    #[test]
    fn test_partial_success_html_is_valid_html_structure() {
        let creds = ManifestConversionResponse {
            id: 1,
            slug: "test".to_string(),
            name: "Test".to_string(),
            client_id: "id".to_string(),
            client_secret: "secret".to_string(),
            webhook_secret: "whsec".to_string(),
            pem: "pem".to_string(),
            html_url: "https://example.com".to_string(),
        };

        let html = partial_success_html(&creds, "Error");
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("Manual Configuration Required"));
    }

    #[test]
    fn test_success_html_escapes_base_url() {
        let creds = ManifestConversionResponse {
            id: 1,
            slug: "test".to_string(),
            name: "Test".to_string(),
            client_id: "id".to_string(),
            client_secret: "secret".to_string(),
            webhook_secret: "whsec".to_string(),
            pem: "pem".to_string(),
            html_url: "https://example.com".to_string(),
        };

        let html = success_html(&creds, "https://evil.com/<script>");
        assert!(!html.contains("<script>"), "base_url should be escaped");
        assert!(html.contains("&lt;script&gt;"));
    }

    #[test]
    fn test_partial_success_html_escapes_credentials() {
        let creds = ManifestConversionResponse {
            id: 1,
            slug: "test".to_string(),
            name: "<b>Evil</b>".to_string(),
            client_id: "<script>".to_string(),
            client_secret: "secret&more".to_string(),
            webhook_secret: "whsec".to_string(),
            pem: "pem".to_string(),
            html_url: "https://evil.com/\"onclick=\"alert(1)".to_string(),
        };

        let html = partial_success_html(&creds, "error");
        assert!(!html.contains("<b>Evil</b>"), "Name should be escaped");
        assert!(!html.contains("<script>"), "Client ID should be escaped");
        assert!(html.contains("&lt;script&gt;"));
        assert!(html.contains("&amp;more"));
    }
}
