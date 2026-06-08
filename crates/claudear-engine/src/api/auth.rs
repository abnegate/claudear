//! Authentication and user management handlers.

use axum::{
    extract::{FromRequestParts, Path, State},
    http::{request::Parts, HeaderMap, StatusCode},
    Json,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Instant;
use tower_cookies::{Cookie, Cookies};

use super::routes::ApiState;

const SESSION_COOKIE: &str = "claudear_session";
const SESSION_MAX_AGE_DAYS: i64 = 7;

/// Maximum number of login attempts per email within the rate limit window.
const LOGIN_RATE_LIMIT_MAX_ATTEMPTS: usize = 10;

/// Duration of the rate limit window in seconds.
const LOGIN_RATE_LIMIT_WINDOW_SECS: u64 = 300; // 5 minutes

/// Maximum number of unique keys (email addresses) tracked in the rate limiter.
/// When exceeded, the oldest entries are evicted to prevent memory exhaustion.
const LOGIN_RATE_LIMIT_MAX_KEYS: usize = 10_000;

/// Maximum number of login attempts per IP address within the rate limit window.
/// More generous than per-email to avoid blocking legitimate users behind shared IPs.
const IP_RATE_LIMIT_MAX_ATTEMPTS: usize = 100;

/// Duration of the IP rate limit window in seconds.
const IP_RATE_LIMIT_WINDOW_SECS: u64 = 300; // 5 minutes

/// In-memory rate limiter for login attempts, keyed by email address.
/// This protects against brute force attacks on specific accounts and
/// mitigates CPU exhaustion from repeated bcrypt verification.
static LOGIN_RATE_LIMITER: std::sync::LazyLock<Mutex<HashMap<String, Vec<Instant>>>> =
    std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));

/// In-memory rate limiter for login attempts, keyed by client IP address.
/// This prevents a single IP from brute-forcing across many email addresses
/// and limits distributed credential-stuffing attacks.
static IP_RATE_LIMITER: std::sync::LazyLock<Mutex<HashMap<String, Vec<Instant>>>> =
    std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));

/// Maximum number of API requests per user within the rate limit window.
/// Applies to sensitive endpoints (profile, avatar, config, user management).
const API_RATE_LIMIT_MAX_REQUESTS: usize = 30;

/// Duration of the API rate limit window in seconds.
const API_RATE_LIMIT_WINDOW_SECS: u64 = 60; // 1 minute

/// In-memory rate limiter for sensitive API endpoints, keyed by user ID.
static API_RATE_LIMITER: std::sync::LazyLock<Mutex<HashMap<String, Vec<Instant>>>> =
    std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));

/// Check if a login attempt is allowed for the given key (email).
/// Returns true if the attempt is within rate limits, false if it should be rejected.
fn check_login_rate_limit(key: &str) -> bool {
    let mut limiter = match LOGIN_RATE_LIMITER.lock() {
        Ok(l) => l,
        Err(poisoned) => {
            tracing::warn!("Login rate limiter mutex was poisoned, recovering");
            poisoned.into_inner()
        }
    };

    let now = Instant::now();
    let window = std::time::Duration::from_secs(LOGIN_RATE_LIMIT_WINDOW_SECS);

    let attempts = limiter.entry(key.to_string()).or_default();

    // Remove attempts outside the window
    attempts.retain(|t| now.duration_since(*t) < window);

    if attempts.len() >= LOGIN_RATE_LIMIT_MAX_ATTEMPTS {
        return false;
    }

    attempts.push(now);

    // Sweep expired entries from other keys to prevent unbounded memory growth
    limiter.retain(|_, v| !v.is_empty() && v.iter().any(|t| now.duration_since(*t) < window));

    // Cap total entries to prevent memory exhaustion from distributed attacks
    if limiter.len() > LOGIN_RATE_LIMIT_MAX_KEYS {
        // Find and remove entries with the oldest most-recent attempt
        let mut entries: Vec<(String, Instant)> = limiter
            .iter()
            .filter_map(|(k, v)| v.last().map(|t| (k.clone(), *t)))
            .collect();
        entries.sort_by_key(|(_, t)| *t);
        let to_remove = limiter.len() - LOGIN_RATE_LIMIT_MAX_KEYS;
        for (k, _) in entries.into_iter().take(to_remove) {
            limiter.remove(&k);
        }
    }

    true
}

/// Check if a login attempt is allowed for the given IP address.
/// Returns true if the attempt is within rate limits, false if it should be rejected.
fn check_ip_rate_limit(ip: &str) -> bool {
    let mut limiter = match IP_RATE_LIMITER.lock() {
        Ok(l) => l,
        Err(poisoned) => {
            tracing::warn!("IP rate limiter mutex was poisoned, recovering");
            poisoned.into_inner()
        }
    };

    let now = Instant::now();
    let window = std::time::Duration::from_secs(IP_RATE_LIMIT_WINDOW_SECS);

    let attempts = limiter.entry(ip.to_string()).or_default();

    // Remove attempts outside the window
    attempts.retain(|t| now.duration_since(*t) < window);

    if attempts.len() >= IP_RATE_LIMIT_MAX_ATTEMPTS {
        return false;
    }

    attempts.push(now);

    // Sweep expired entries from other keys to prevent unbounded memory growth
    limiter.retain(|_, v| !v.is_empty() && v.iter().any(|t| now.duration_since(*t) < window));

    // Cap total entries to prevent memory exhaustion from distributed attacks
    if limiter.len() > LOGIN_RATE_LIMIT_MAX_KEYS {
        let mut entries: Vec<(String, Instant)> = limiter
            .iter()
            .filter_map(|(k, v)| v.last().map(|t| (k.clone(), *t)))
            .collect();
        entries.sort_by_key(|(_, t)| *t);
        let to_remove = limiter.len() - LOGIN_RATE_LIMIT_MAX_KEYS;
        for (k, _) in entries.into_iter().take(to_remove) {
            limiter.remove(&k);
        }
    }

    true
}

/// Reset the API rate limiter. Only for use in tests.
#[cfg(test)]
pub(crate) fn reset_api_rate_limiter() {
    let mut limiter = API_RATE_LIMITER.lock().unwrap();
    limiter.clear();
}

/// Check if an API request is allowed for the given user ID.
/// Applied to sensitive endpoints like profile, avatar, config, and user management.
pub(crate) fn check_api_rate_limit(user_id: i64) -> bool {
    let key = user_id.to_string();
    let mut limiter = match API_RATE_LIMITER.lock() {
        Ok(l) => l,
        Err(poisoned) => {
            tracing::warn!("API rate limiter mutex was poisoned, recovering");
            poisoned.into_inner()
        }
    };

    let now = Instant::now();
    let window = std::time::Duration::from_secs(API_RATE_LIMIT_WINDOW_SECS);

    let attempts = limiter.entry(key).or_default();
    attempts.retain(|t| now.duration_since(*t) < window);

    if attempts.len() >= API_RATE_LIMIT_MAX_REQUESTS {
        return false;
    }

    attempts.push(now);

    // Sweep expired entries
    limiter.retain(|_, v| !v.is_empty() && v.iter().any(|t| now.duration_since(*t) < window));

    true
}

/// Extract the client IP address from request headers.
/// Checks `x-forwarded-for` first, then `x-real-ip`, then falls back to "unknown".
fn extract_client_ip(headers: &HeaderMap) -> String {
    if let Some(forwarded) = headers.get("x-forwarded-for") {
        if let Ok(value) = forwarded.to_str() {
            // x-forwarded-for can contain multiple IPs; the first is the client
            if let Some(first_ip) = value.split(',').next() {
                let ip = first_ip.trim();
                if !ip.is_empty() {
                    return ip.to_string();
                }
            }
        }
    }

    if let Some(real_ip) = headers.get("x-real-ip") {
        if let Ok(value) = real_ip.to_str() {
            let ip = value.trim();
            if !ip.is_empty() {
                return ip.to_string();
            }
        }
    }

    "unknown".to_string()
}

/// Authenticated user extracted from session cookie.
#[derive(Debug, Clone, Serialize)]
pub struct AuthUser {
    pub id: i64,
    pub email: String,
    pub name: String,
    pub role: String,
    pub avatar_url: Option<String>,
}

impl FromRequestParts<ApiState> for AuthUser {
    type Rejection = StatusCode;

    async fn from_request_parts(
        parts: &mut Parts,
        state: &ApiState,
    ) -> Result<Self, Self::Rejection> {
        // Extract cookies from the request
        let cookies = Cookies::from_request_parts(parts, state)
            .await
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

        let token = cookies
            .get(SESSION_COOKIE)
            .map(|c| c.value().to_string())
            .ok_or(StatusCode::UNAUTHORIZED)?;

        let user = state
            .tracker
            .get_session_user(&token)
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
            .ok_or(StatusCode::UNAUTHORIZED)?;

        // Refresh the idle session timeout on every authenticated request
        let _ = state.tracker.touch_session(&token);

        let auth_user = AuthUser {
            id: user.id,
            email: user.email,
            name: user.name,
            role: user.role,
            avatar_url: user.avatar_url,
        };

        sentry::configure_scope(|scope| {
            scope.set_user(Some(sentry::User {
                id: Some(auth_user.id.to_string()),
                email: Some(auth_user.email.clone()),
                username: Some(auth_user.name.clone()),
                ..Default::default()
            }));
            scope.set_tag("user.role", &auth_user.role);
        });

        Ok(auth_user)
    }
}

/// Admin user extractor — wraps AuthUser and checks role == "admin".
#[derive(Debug, Clone)]
pub struct AdminUser(pub AuthUser);

impl FromRequestParts<ApiState> for AdminUser {
    type Rejection = StatusCode;

    async fn from_request_parts(
        parts: &mut Parts,
        state: &ApiState,
    ) -> Result<Self, Self::Rejection> {
        let user = AuthUser::from_request_parts(parts, state).await?;
        if user.role != "admin" {
            return Err(StatusCode::FORBIDDEN);
        }
        Ok(AdminUser(user))
    }
}

#[derive(Deserialize)]
pub struct LoginRequest {
    pub email: String,
    pub password: String,
}

#[derive(Serialize)]
pub struct LoginResponse {
    pub user: AuthUserResponse,
}

#[derive(Serialize)]
pub struct AuthUserResponse {
    pub id: i64,
    pub email: String,
    pub name: String,
    pub role: String,
    pub avatar_url: Option<String>,
}

impl From<&AuthUser> for AuthUserResponse {
    fn from(u: &AuthUser) -> Self {
        AuthUserResponse {
            id: u.id,
            email: u.email.clone(),
            name: u.name.clone(),
            role: u.role.clone(),
            avatar_url: u.avatar_url.clone(),
        }
    }
}

#[derive(Deserialize)]
pub struct CreateUserRequest {
    pub email: String,
    pub password: String,
    pub name: String,
    pub role: String,
}

#[derive(Deserialize)]
pub struct UpdateUserRequest {
    pub email: Option<String>,
    pub password: Option<String>,
    pub name: Option<String>,
    pub role: Option<String>,
}

#[derive(Serialize)]
pub struct UserResponse {
    pub id: i64,
    pub email: String,
    pub name: String,
    pub role: String,
    pub avatar_url: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Serialize)]
pub struct MessageResponse {
    pub message: String,
}

/// POST /api/auth/login
pub async fn login_handler(
    State(state): State<ApiState>,
    cookies: Cookies,
    headers: HeaderMap,
    Json(body): Json<LoginRequest>,
) -> Result<Json<LoginResponse>, StatusCode> {
    let client_ip = extract_client_ip(&headers);

    // Rate limit login attempts by IP to prevent distributed brute force across many emails
    if !check_ip_rate_limit(&client_ip) {
        tracing::warn!(ip = %client_ip, "Login IP rate limit exceeded");
        return Err(StatusCode::TOO_MANY_REQUESTS);
    }

    // Rate limit login attempts by email to prevent brute force and bcrypt CPU exhaustion
    if !check_login_rate_limit(&body.email) {
        tracing::warn!(email = %body.email, ip = %client_ip, "Login rate limit exceeded");
        return Err(StatusCode::TOO_MANY_REQUESTS);
    }

    // Look up user by email
    let user = state
        .tracker
        .get_user_by_email(&body.email)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    // Always perform bcrypt verification to prevent timing oracle that reveals user existence.
    // When no user is found, verify against a dummy hash to equalize response time.
    let (hash_to_verify, user_found) = match user {
        Some(ref u) => (u.password_hash.clone(), true),
        None => {
            // Pre-computed bcrypt hash of "dummy" with cost 12. The actual value doesn't matter;
            // the goal is to spend the same CPU time as a real verification to prevent
            // attackers from enumerating valid email addresses via response timing.
            (
                "$2b$12$K4v3LB7TzMIXvbQZDz1F9eZ8cU2smBGz.iAU5h1DhGGCk5mPIFY3K".to_string(),
                false,
            )
        }
    };

    let pw = body.password.clone();
    let valid = tokio::task::spawn_blocking(move || bcrypt::verify(&pw, &hash_to_verify))
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .unwrap_or(false);

    if !user_found || !valid {
        return Err(StatusCode::UNAUTHORIZED);
    }

    // SAFETY: user_found is true so user is Some
    let user = user.unwrap();

    // Create session
    let expires_at = chrono::Utc::now() + chrono::Duration::days(SESSION_MAX_AGE_DAYS);
    let expires_str = expires_at.format("%Y-%m-%d %H:%M:%S").to_string();

    let token = state
        .tracker
        .create_session(user.id, &expires_str)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    // Set cookie
    // Only set Secure flag when not on localhost — browsers (notably Firefox)
    // may refuse to send Secure cookies over ws:// WebSocket connections even
    // to localhost, breaking the indexing-progress WebSocket endpoint.
    let is_localhost = matches!(
        state.config.bind_address.as_str(),
        "127.0.0.1" | "localhost" | "::1"
    );
    let mut cookie = Cookie::new(SESSION_COOKIE, token);
    cookie.set_path("/");
    cookie.set_http_only(true);
    cookie.set_secure(!is_localhost);
    cookie.set_same_site(tower_cookies::cookie::SameSite::Lax);
    cookie.set_max_age(tower_cookies::cookie::time::Duration::days(
        SESSION_MAX_AGE_DAYS,
    ));
    cookies.add(cookie);

    sentry::configure_scope(|scope| {
        scope.set_user(Some(sentry::User {
            id: Some(user.id.to_string()),
            email: Some(user.email.clone()),
            username: Some(user.name.clone()),
            ..Default::default()
        }));
        scope.set_tag("user.role", &user.role);
    });

    Ok(Json(LoginResponse {
        user: AuthUserResponse {
            id: user.id,
            email: user.email,
            name: user.name,
            role: user.role,
            avatar_url: user.avatar_url,
        },
    }))
}

/// POST /api/auth/logout
pub async fn logout_handler(
    State(state): State<ApiState>,
    cookies: Cookies,
) -> Result<Json<MessageResponse>, StatusCode> {
    // Get session token from cookie
    if let Some(cookie) = cookies.get(SESSION_COOKIE) {
        let token = cookie.value().to_string();
        let _ = state.tracker.delete_session(&token);
    }

    // Clear cookie
    let mut cookie = Cookie::new(SESSION_COOKIE, "");
    cookie.set_path("/");
    cookie.set_max_age(tower_cookies::cookie::time::Duration::ZERO);
    cookies.remove(cookie);

    Ok(Json(MessageResponse {
        message: "Logged out".to_string(),
    }))
}

/// GET /api/auth/me
pub async fn me_handler(user: AuthUser) -> Json<AuthUserResponse> {
    Json(AuthUserResponse::from(&user))
}

#[derive(Deserialize)]
pub struct UpdateProfileRequest {
    pub name: Option<String>,
    pub password: Option<String>,
    pub current_password: Option<String>,
}

/// PUT /api/auth/profile
pub async fn update_profile_handler(
    user: AuthUser,
    State(state): State<ApiState>,
    Json(body): Json<UpdateProfileRequest>,
) -> Result<Json<UserResponse>, StatusCode> {
    if !check_api_rate_limit(user.id) {
        return Err(StatusCode::TOO_MANY_REQUESTS);
    }

    // Validate name if provided
    if let Some(ref name) = body.name {
        if name.trim().is_empty() {
            return Err(StatusCode::BAD_REQUEST);
        }
    }

    // If changing password, require current_password verification
    let password_hash = match &body.password {
        Some(new_pw) => {
            if new_pw.len() < 8 || new_pw.len() > 72 {
                return Err(StatusCode::BAD_REQUEST);
            }
            // Rate limit password change to prevent bcrypt CPU exhaustion
            if !check_login_rate_limit(&user.email) {
                return Err(StatusCode::TOO_MANY_REQUESTS);
            }
            let current_pw = body
                .current_password
                .as_deref()
                .ok_or(StatusCode::BAD_REQUEST)?;

            // Fetch current user to verify password
            let current_user = state
                .tracker
                .get_user_by_id(user.id)
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
                .ok_or(StatusCode::NOT_FOUND)?;

            let current_pw_owned = current_pw.to_string();
            let stored_hash = current_user.password_hash.clone();
            let valid = tokio::task::spawn_blocking(move || {
                bcrypt::verify(&current_pw_owned, &stored_hash)
            })
            .await
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
            if !valid {
                return Err(StatusCode::FORBIDDEN);
            }

            let new_pw_owned = new_pw.clone();
            let hashed = tokio::task::spawn_blocking(move || {
                bcrypt::hash(&new_pw_owned, bcrypt::DEFAULT_COST)
            })
            .await
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
            Some(hashed)
        }
        None => None,
    };

    let trimmed_name = body.name.as_deref().map(str::trim);

    state
        .tracker
        .update_user(
            user.id,
            None,
            password_hash.as_deref(),
            trimmed_name,
            None,
            None,
        )
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let updated = state
        .tracker
        .get_user_by_id(user.id)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .ok_or(StatusCode::NOT_FOUND)?;

    Ok(Json(UserResponse {
        id: updated.id,
        email: updated.email,
        name: updated.name,
        role: updated.role,
        avatar_url: updated.avatar_url,
        created_at: updated.created_at,
        updated_at: updated.updated_at,
    }))
}

/// POST /api/auth/avatar
pub async fn upload_avatar_handler(
    user: AuthUser,
    State(state): State<ApiState>,
    mut multipart: axum::extract::Multipart,
) -> Result<Json<serde_json::Value>, StatusCode> {
    if !check_api_rate_limit(user.id) {
        return Err(StatusCode::TOO_MANY_REQUESTS);
    }
    const MAX_SIZE: usize = 5 * 1024 * 1024; // 5MB
    const ALLOWED_TYPES: &[&str] = &["image/png", "image/jpeg", "image/gif", "image/webp"];

    let avatars_dir = state.storage_dir.join("avatars");

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|_| StatusCode::BAD_REQUEST)?
    {
        // Only process the "avatar" field
        if field.name() != Some("avatar") {
            continue;
        }

        let content_type = field
            .content_type()
            .unwrap_or("application/octet-stream")
            .to_string();

        if !ALLOWED_TYPES.contains(&content_type.as_str()) {
            return Err(StatusCode::BAD_REQUEST);
        }

        let ext = match content_type.as_str() {
            "image/png" => "png",
            "image/jpeg" => "jpg",
            "image/gif" => "gif",
            "image/webp" => "webp",
            _ => return Err(StatusCode::BAD_REQUEST),
        };

        let data = field.bytes().await.map_err(|_| StatusCode::BAD_REQUEST)?;

        if data.len() > MAX_SIZE {
            return Err(StatusCode::PAYLOAD_TOO_LARGE);
        }

        // Validate magic bytes match the claimed content type
        let valid_magic = match ext {
            "png" => data.starts_with(&[0x89, b'P', b'N', b'G']),
            "jpg" => data.starts_with(&[0xFF, 0xD8, 0xFF]),
            "gif" => data.starts_with(b"GIF87a") || data.starts_with(b"GIF89a"),
            "webp" => data.len() >= 12 && &data[0..4] == b"RIFF" && &data[8..12] == b"WEBP",
            _ => false,
        };
        if !valid_magic {
            return Err(StatusCode::BAD_REQUEST);
        }

        // Delete old avatar file using the path stored in DB (avoids prefix-matching bugs)
        if let Some(ref old_url) = user.avatar_url {
            if let Some(old_filename) = old_url.rsplit('/').next() {
                let old_path = avatars_dir.join(old_filename);
                let _ = tokio::fs::remove_file(&old_path).await;
            }
        }

        // Use random token in filename to prevent enumeration
        let random_token = hex::encode(rand::random::<[u8; 8]>());
        let filename = format!("{}_{}.{}", user.id, random_token, ext);
        let file_path = avatars_dir.join(&filename);
        tokio::fs::write(&file_path, &data)
            .await
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

        let avatar_url = format!("/avatars/{}", filename);

        // Update user's avatar_url in DB
        state
            .tracker
            .update_user(user.id, None, None, None, None, Some(&avatar_url))
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

        return Ok(Json(serde_json::json!({ "avatar_url": avatar_url })));
    }

    Err(StatusCode::BAD_REQUEST)
}

/// GET /api/users
pub async fn list_users_handler(
    _admin: AdminUser,
    State(state): State<ApiState>,
) -> Result<Json<Vec<UserResponse>>, StatusCode> {
    let users = state
        .tracker
        .list_users()
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let response: Vec<UserResponse> = users
        .into_iter()
        .map(|u| UserResponse {
            id: u.id,
            email: u.email,
            name: u.name,
            role: u.role,
            avatar_url: u.avatar_url,
            created_at: u.created_at,
            updated_at: u.updated_at,
        })
        .collect();

    Ok(Json(response))
}

/// GET /api/users/{id}
pub async fn get_user_handler(
    _admin: AdminUser,
    State(state): State<ApiState>,
    Path(id): Path<i64>,
) -> Result<Json<UserResponse>, StatusCode> {
    let user = state
        .tracker
        .get_user_by_id(id)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .ok_or(StatusCode::NOT_FOUND)?;

    Ok(Json(UserResponse {
        id: user.id,
        email: user.email,
        name: user.name,
        role: user.role,
        avatar_url: user.avatar_url,
        created_at: user.created_at,
        updated_at: user.updated_at,
    }))
}

/// POST /api/users
pub async fn create_user_handler(
    _admin: AdminUser,
    State(state): State<ApiState>,
    Json(body): Json<CreateUserRequest>,
) -> Result<(StatusCode, Json<UserResponse>), StatusCode> {
    if !check_api_rate_limit(_admin.0.id) {
        return Err(StatusCode::TOO_MANY_REQUESTS);
    }

    // Validate role
    if body.role != "admin" && body.role != "viewer" {
        return Err(StatusCode::BAD_REQUEST);
    }

    // Validate email
    if body.email.trim().is_empty() || !body.email.contains('@') {
        return Err(StatusCode::BAD_REQUEST);
    }

    // Validate password (minimum 8, maximum 72 characters — bcrypt limit)
    if body.password.len() < 8 || body.password.len() > 72 {
        return Err(StatusCode::BAD_REQUEST);
    }

    // Validate name
    if body.name.trim().is_empty() {
        return Err(StatusCode::BAD_REQUEST);
    }

    // Check for duplicate email
    if state
        .tracker
        .get_user_by_email(&body.email)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .is_some()
    {
        return Err(StatusCode::CONFLICT);
    }

    // Hash password (spawn_blocking to avoid blocking the async runtime)
    let pw = body.password.clone();
    let password_hash =
        tokio::task::spawn_blocking(move || bcrypt::hash(&pw, bcrypt::DEFAULT_COST))
            .await
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let id = state
        .tracker
        .create_user(&body.email, &password_hash, &body.name, &body.role)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let user = state
        .tracker
        .get_user_by_id(id)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .ok_or(StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok((
        StatusCode::CREATED,
        Json(UserResponse {
            id: user.id,
            email: user.email,
            name: user.name,
            role: user.role,
            avatar_url: user.avatar_url,
            created_at: user.created_at,
            updated_at: user.updated_at,
        }),
    ))
}

/// PUT /api/users/{id}
pub async fn update_user_handler(
    _admin: AdminUser,
    State(state): State<ApiState>,
    Path(id): Path<i64>,
    Json(body): Json<UpdateUserRequest>,
) -> Result<Json<UserResponse>, StatusCode> {
    if !check_api_rate_limit(_admin.0.id) {
        return Err(StatusCode::TOO_MANY_REQUESTS);
    }

    // Validate role if provided
    if let Some(ref role) = body.role {
        if role != "admin" && role != "viewer" {
            return Err(StatusCode::BAD_REQUEST);
        }
    }

    // Validate password length if provided (minimum 8, maximum 72 — bcrypt limit)
    if let Some(ref pw) = body.password {
        if pw.len() < 8 || pw.len() > 72 {
            return Err(StatusCode::BAD_REQUEST);
        }
    }

    // Hash password if provided (use spawn_blocking to avoid blocking the async runtime)
    let password_hash = match &body.password {
        Some(pw) => {
            let pw = pw.clone();
            Some(
                tokio::task::spawn_blocking(move || bcrypt::hash(pw, bcrypt::DEFAULT_COST))
                    .await
                    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
                    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?,
            )
        }
        None => None,
    };

    let updated = state
        .tracker
        .update_user(
            id,
            body.email.as_deref(),
            password_hash.as_deref(),
            body.name.as_deref(),
            body.role.as_deref(),
            None,
        )
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    if !updated {
        return Err(StatusCode::NOT_FOUND);
    }

    let user = state
        .tracker
        .get_user_by_id(id)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .ok_or(StatusCode::NOT_FOUND)?;

    Ok(Json(UserResponse {
        id: user.id,
        email: user.email,
        name: user.name,
        role: user.role,
        avatar_url: user.avatar_url,
        created_at: user.created_at,
        updated_at: user.updated_at,
    }))
}

/// DELETE /api/users/{id}
pub async fn delete_user_handler(
    admin: AdminUser,
    State(state): State<ApiState>,
    Path(id): Path<i64>,
) -> Result<Json<MessageResponse>, StatusCode> {
    if !check_api_rate_limit(admin.0.id) {
        return Err(StatusCode::TOO_MANY_REQUESTS);
    }

    // Can't delete yourself
    if admin.0.id == id {
        return Err(StatusCode::BAD_REQUEST);
    }

    // Delete user sessions first
    let _ = state.tracker.delete_user_sessions(id);

    let deleted = state
        .tracker
        .delete_user(id)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    if !deleted {
        return Err(StatusCode::NOT_FOUND);
    }

    Ok(Json(MessageResponse {
        message: "User deleted".to_string(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::routes::create_api_router;
    use axum::body::Body;
    use axum::http::Request;
    use claudear_config::config::{
        AgentConfig, AskConfig, CascadeConfig, CodeIndexConfig, Config, IssuesConfig,
        LearningConfig, NotifiersConfig, PrioritisationConfig, RegressionConfig, RetryConfig,
        ScmConfig,
    };
    use claudear_storage::SqliteTracker;
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

    /// Create router + tracker Arc for testing.
    fn create_test_app() -> (
        axum::Router,
        std::sync::Arc<dyn claudear_storage::FixAttemptTracker>,
    ) {
        // Reset API rate limiter to prevent cross-test interference
        reset_api_rate_limiter();

        let tracker: std::sync::Arc<dyn claudear_storage::FixAttemptTracker> =
            std::sync::Arc::new(SqliteTracker::in_memory().unwrap());
        let indexing_rx = tracker.subscribe_indexing_progress();
        let router = create_api_router(
            test_config(),
            tracker.clone(),
            std::path::PathBuf::from("claudear.toml"),
            indexing_rx,
        )
        .layer(CookieManagerLayer::new());
        (router, tracker)
    }

    /// Seed an admin user and return (user_id, session_token).
    fn seed_admin(
        tracker: &std::sync::Arc<dyn claudear_storage::FixAttemptTracker>,
    ) -> (i64, String) {
        let password_hash = bcrypt::hash("password", 4).unwrap();
        let user_id = tracker
            .create_user("admin@test.com", &password_hash, "Admin User", "admin")
            .unwrap();
        let token = tracker
            .create_session(user_id, "2099-12-31 23:59:59")
            .unwrap();
        (user_id, token)
    }

    /// Seed a viewer user and return (user_id, session_token).
    fn seed_viewer(
        tracker: &std::sync::Arc<dyn claudear_storage::FixAttemptTracker>,
    ) -> (i64, String) {
        let password_hash = bcrypt::hash("password", 4).unwrap();
        let user_id = tracker
            .create_user("viewer@test.com", &password_hash, "Viewer User", "viewer")
            .unwrap();
        let token = tracker
            .create_session(user_id, "2099-12-31 23:59:59")
            .unwrap();
        (user_id, token)
    }

    #[tokio::test]
    async fn test_login_success() {
        let (router, tracker) = create_test_app();

        // Seed a user (not via seed_admin, to test login flow directly)
        let password_hash = bcrypt::hash("secret123", 4).unwrap();
        tracker
            .create_user("user@test.com", &password_hash, "Test User", "admin")
            .unwrap();

        let response = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/auth/login")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"email":"user@test.com","password":"secret123"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        // Check that a set-cookie header is present
        let set_cookie = response.headers().get("set-cookie");
        assert!(set_cookie.is_some(), "Expected set-cookie header");
        let cookie_val = set_cookie.unwrap().to_str().unwrap();
        assert!(
            cookie_val.contains("claudear_session"),
            "Cookie should contain claudear_session"
        );

        // Check response body contains user data
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let body_str = String::from_utf8(body.to_vec()).unwrap();
        assert!(body_str.contains("user@test.com"));
        assert!(body_str.contains("Test User"));
    }

    #[tokio::test]
    async fn test_login_wrong_password() {
        let (router, tracker) = create_test_app();

        let password_hash = bcrypt::hash("correct_password", 4).unwrap();
        tracker
            .create_user("user@test.com", &password_hash, "Test User", "admin")
            .unwrap();

        let response = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/auth/login")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"email":"user@test.com","password":"wrong_password"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_login_unknown_email() {
        let (router, _tracker) = create_test_app();

        let response = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/auth/login")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"email":"nobody@test.com","password":"whatever"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_me_authenticated() {
        let (router, tracker) = create_test_app();
        let (_user_id, token) = seed_admin(&tracker);

        let response = router
            .oneshot(
                Request::builder()
                    .uri("/api/auth/me")
                    .header("cookie", format!("claudear_session={}", token))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = response.into_body().collect().await.unwrap().to_bytes();
        let body_str = String::from_utf8(body.to_vec()).unwrap();
        assert!(body_str.contains("admin@test.com"));
        assert!(body_str.contains("Admin User"));
        assert!(body_str.contains("admin"));
    }

    #[tokio::test]
    async fn test_me_unauthenticated() {
        let (router, _tracker) = create_test_app();

        let response = router
            .oneshot(
                Request::builder()
                    .uri("/api/auth/me")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_create_user_admin_only() {
        let (router, tracker) = create_test_app();
        let (_admin_id, token) = seed_admin(&tracker);

        let response = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/users")
                    .header("content-type", "application/json")
                    .header("cookie", format!("claudear_session={}", token))
                    .body(Body::from(
                        r#"{"email":"new@test.com","password":"newpass1!","name":"New User","role":"viewer"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::CREATED);

        let body = response.into_body().collect().await.unwrap().to_bytes();
        let body_str = String::from_utf8(body.to_vec()).unwrap();
        assert!(body_str.contains("new@test.com"));
        assert!(body_str.contains("New User"));
        assert!(body_str.contains("viewer"));
    }

    #[tokio::test]
    async fn test_viewer_cannot_create_user() {
        let (router, tracker) = create_test_app();
        let (_viewer_id, token) = seed_viewer(&tracker);

        let response = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/users")
                    .header("content-type", "application/json")
                    .header("cookie", format!("claudear_session={}", token))
                    .body(Body::from(
                        r#"{"email":"new@test.com","password":"newpass1!","name":"New User","role":"viewer"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn test_cannot_delete_self() {
        let (router, tracker) = create_test_app();
        let (admin_id, token) = seed_admin(&tracker);

        let response = router
            .oneshot(
                Request::builder()
                    .method("DELETE")
                    .uri(format!("/api/users/{}", admin_id))
                    .header("cookie", format!("claudear_session={}", token))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    /// Seed an admin user with a custom email and return (user_id, session_token).
    fn seed_admin_with_email(
        tracker: &std::sync::Arc<dyn claudear_storage::FixAttemptTracker>,
        email: &str,
    ) -> (i64, String) {
        let password_hash = bcrypt::hash("password", 4).unwrap();
        let user_id = tracker
            .create_user(email, &password_hash, "Admin User", "admin")
            .unwrap();
        let token = tracker
            .create_session(user_id, "2099-12-31 23:59:59")
            .unwrap();
        (user_id, token)
    }

    #[test]
    fn test_rate_limit_rejects_11th_attempt() {
        // Use a unique email to avoid interference with other tests
        let key = "ratelimit-11th@unique-test.com";
        // First 10 attempts should succeed
        for i in 0..LOGIN_RATE_LIMIT_MAX_ATTEMPTS {
            assert!(
                check_login_rate_limit(key),
                "Attempt {} should be allowed",
                i + 1
            );
        }
        // 11th attempt should be rejected
        assert!(
            !check_login_rate_limit(key),
            "11th attempt should be rejected"
        );
    }

    #[test]
    fn test_rate_limit_independent_keys() {
        let key_a = "ratelimit-indep-a@unique-test.com";
        let key_b = "ratelimit-indep-b@unique-test.com";

        // Exhaust rate limit for key_a
        for _ in 0..LOGIN_RATE_LIMIT_MAX_ATTEMPTS {
            check_login_rate_limit(key_a);
        }
        assert!(
            !check_login_rate_limit(key_a),
            "key_a should be rate-limited"
        );

        // key_b should still be allowed
        assert!(
            check_login_rate_limit(key_b),
            "key_b should NOT be rate-limited"
        );
    }

    #[tokio::test]
    async fn test_create_user_invalid_role() {
        let (router, tracker) = create_test_app();
        let (_admin_id, token) = seed_admin_with_email(&tracker, "admin-create-role@test.com");

        let response = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/users")
                    .header("content-type", "application/json")
                    .header("cookie", format!("claudear_session={}", token))
                    .body(Body::from(
                        r#"{"email":"x@test.com","password":"longpassword","name":"X","role":"superuser"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_create_user_empty_email() {
        let (router, tracker) = create_test_app();
        let (_admin_id, token) = seed_admin_with_email(&tracker, "admin-create-email@test.com");

        let response = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/users")
                    .header("content-type", "application/json")
                    .header("cookie", format!("claudear_session={}", token))
                    .body(Body::from(
                        r#"{"email":"","password":"longpassword","name":"X","role":"viewer"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_create_user_email_without_at() {
        let (router, tracker) = create_test_app();
        let (_admin_id, token) = seed_admin_with_email(&tracker, "admin-create-noat@test.com");

        let response = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/users")
                    .header("content-type", "application/json")
                    .header("cookie", format!("claudear_session={}", token))
                    .body(Body::from(
                        r#"{"email":"bademail","password":"longpassword","name":"X","role":"viewer"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_create_user_password_too_short() {
        let (router, tracker) = create_test_app();
        let (_admin_id, token) = seed_admin_with_email(&tracker, "admin-create-pw@test.com");

        let response = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/users")
                    .header("content-type", "application/json")
                    .header("cookie", format!("claudear_session={}", token))
                    .body(Body::from(
                        r#"{"email":"x@test.com","password":"short","name":"X","role":"viewer"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_create_user_empty_name() {
        let (router, tracker) = create_test_app();
        let (_admin_id, token) = seed_admin_with_email(&tracker, "admin-create-name@test.com");

        let response = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/users")
                    .header("content-type", "application/json")
                    .header("cookie", format!("claudear_session={}", token))
                    .body(Body::from(
                        r#"{"email":"x@test.com","password":"longpassword","name":"","role":"viewer"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_create_user_duplicate_email() {
        let (router, tracker) = create_test_app();
        let (_admin_id, token) = seed_admin_with_email(&tracker, "admin-create-dup@test.com");

        // Create a user first
        let hash = bcrypt::hash("password", 4).unwrap();
        tracker
            .create_user("dup@test.com", &hash, "Dup User", "viewer")
            .unwrap();

        // Try to create another user with the same email
        let response = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/users")
                    .header("content-type", "application/json")
                    .header("cookie", format!("claudear_session={}", token))
                    .body(Body::from(
                        r#"{"email":"dup@test.com","password":"longpassword","name":"Another","role":"viewer"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::CONFLICT);
    }

    #[tokio::test]
    async fn test_update_user_invalid_role() {
        let (router, tracker) = create_test_app();
        let (_admin_id, token) = seed_admin_with_email(&tracker, "admin-update-role@test.com");

        // Create a target user to update
        let hash = bcrypt::hash("password", 4).unwrap();
        let target_id = tracker
            .create_user("target-update@test.com", &hash, "Target", "viewer")
            .unwrap();

        let response = router
            .oneshot(
                Request::builder()
                    .method("PUT")
                    .uri(format!("/api/users/{}", target_id))
                    .header("content-type", "application/json")
                    .header("cookie", format!("claudear_session={}", token))
                    .body(Body::from(r#"{"role":"superuser"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_update_user_not_found() {
        let (router, tracker) = create_test_app();
        let (_admin_id, token) = seed_admin_with_email(&tracker, "admin-update-nf@test.com");

        let response = router
            .oneshot(
                Request::builder()
                    .method("PUT")
                    .uri("/api/users/99999")
                    .header("content-type", "application/json")
                    .header("cookie", format!("claudear_session={}", token))
                    .body(Body::from(r#"{"name":"Updated"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_delete_user_success() {
        let (router, tracker) = create_test_app();
        let (_admin_id, token) = seed_admin_with_email(&tracker, "admin-del-ok@test.com");

        // Create a target user to delete
        let hash = bcrypt::hash("password", 4).unwrap();
        let target_id = tracker
            .create_user("target-del@test.com", &hash, "Target", "viewer")
            .unwrap();

        let response = router
            .oneshot(
                Request::builder()
                    .method("DELETE")
                    .uri(format!("/api/users/{}", target_id))
                    .header("cookie", format!("claudear_session={}", token))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = response.into_body().collect().await.unwrap().to_bytes();
        let body_str = String::from_utf8(body.to_vec()).unwrap();
        assert!(body_str.contains("User deleted"));
    }

    #[tokio::test]
    async fn test_delete_user_not_found() {
        let (router, tracker) = create_test_app();
        let (_admin_id, token) = seed_admin_with_email(&tracker, "admin-del-nf@test.com");

        let response = router
            .oneshot(
                Request::builder()
                    .method("DELETE")
                    .uri("/api/users/99999")
                    .header("cookie", format!("claudear_session={}", token))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_list_users_as_admin() {
        let (router, tracker) = create_test_app();
        let (_admin_id, token) = seed_admin_with_email(&tracker, "admin-list@test.com");

        // Create an additional user
        let hash = bcrypt::hash("password", 4).unwrap();
        tracker
            .create_user("extra-list@test.com", &hash, "Extra User", "viewer")
            .unwrap();

        let response = router
            .oneshot(
                Request::builder()
                    .uri("/api/users")
                    .header("cookie", format!("claudear_session={}", token))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = response.into_body().collect().await.unwrap().to_bytes();
        let body_str = String::from_utf8(body.to_vec()).unwrap();
        // Should contain both users
        assert!(body_str.contains("admin-list@test.com"));
        assert!(body_str.contains("extra-list@test.com"));

        // Verify the response parses as a JSON array
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&body_str).unwrap();
        assert_eq!(parsed.len(), 2);
    }

    #[tokio::test]
    async fn test_get_user_by_id() {
        let (router, tracker) = create_test_app();
        let (_admin_id, token) = seed_admin_with_email(&tracker, "admin-getuser@test.com");

        // Create a target user
        let hash = bcrypt::hash("password", 4).unwrap();
        let target_id = tracker
            .create_user("target-get@test.com", &hash, "Target Get", "viewer")
            .unwrap();

        let response = router
            .oneshot(
                Request::builder()
                    .uri(format!("/api/users/{}", target_id))
                    .header("cookie", format!("claudear_session={}", token))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = response.into_body().collect().await.unwrap().to_bytes();
        let body_str = String::from_utf8(body.to_vec()).unwrap();
        assert!(body_str.contains("target-get@test.com"));
        assert!(body_str.contains("Target Get"));
        assert!(body_str.contains("viewer"));
    }

    #[tokio::test]
    async fn test_get_user_not_found() {
        let (router, tracker) = create_test_app();
        let (_admin_id, token) = seed_admin_with_email(&tracker, "admin-getuser-nf@test.com");

        let response = router
            .oneshot(
                Request::builder()
                    .uri("/api/users/99999")
                    .header("cookie", format!("claudear_session={}", token))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_logout_clears_session_cookie() {
        let (router, tracker) = create_test_app();
        let (_admin_id, token) = seed_admin_with_email(&tracker, "admin-logout@test.com");

        let response = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/auth/logout")
                    .header("cookie", format!("claudear_session={}", token))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        // Verify response body
        let body_bytes = response.into_body().collect().await.unwrap().to_bytes();
        let body_str = String::from_utf8(body_bytes.to_vec()).unwrap();
        assert!(body_str.contains("Logged out"));
    }

    #[test]
    fn test_auth_user_response_from_auth_user() {
        let user = AuthUser {
            id: 42,
            email: "convert@test.com".to_string(),
            name: "Convert User".to_string(),
            role: "admin".to_string(),
            avatar_url: None,
        };

        let response = AuthUserResponse::from(&user);

        assert_eq!(response.id, 42);
        assert_eq!(response.email, "convert@test.com");
        assert_eq!(response.name, "Convert User");
        assert_eq!(response.role, "admin");
    }

    #[test]
    fn test_extract_client_ip_from_x_forwarded_for() {
        let mut headers = HeaderMap::new();
        headers.insert("x-forwarded-for", "1.2.3.4, 5.6.7.8".parse().unwrap());
        assert_eq!(extract_client_ip(&headers), "1.2.3.4");
    }

    #[test]
    fn test_extract_client_ip_from_x_forwarded_for_single() {
        let mut headers = HeaderMap::new();
        headers.insert("x-forwarded-for", "10.0.0.1".parse().unwrap());
        assert_eq!(extract_client_ip(&headers), "10.0.0.1");
    }

    #[test]
    fn test_extract_client_ip_from_x_forwarded_for_trims_whitespace() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-forwarded-for",
            "  192.168.1.1  , 10.0.0.1".parse().unwrap(),
        );
        assert_eq!(extract_client_ip(&headers), "192.168.1.1");
    }

    #[test]
    fn test_extract_client_ip_from_x_real_ip() {
        let mut headers = HeaderMap::new();
        headers.insert("x-real-ip", "9.8.7.6".parse().unwrap());
        assert_eq!(extract_client_ip(&headers), "9.8.7.6");
    }

    #[test]
    fn test_extract_client_ip_from_x_real_ip_trims() {
        let mut headers = HeaderMap::new();
        headers.insert("x-real-ip", "  10.10.10.10  ".parse().unwrap());
        assert_eq!(extract_client_ip(&headers), "10.10.10.10");
    }

    #[test]
    fn test_extract_client_ip_prefers_x_forwarded_for() {
        let mut headers = HeaderMap::new();
        headers.insert("x-forwarded-for", "1.1.1.1".parse().unwrap());
        headers.insert("x-real-ip", "2.2.2.2".parse().unwrap());
        assert_eq!(extract_client_ip(&headers), "1.1.1.1");
    }

    #[test]
    fn test_extract_client_ip_falls_back_to_unknown() {
        let headers = HeaderMap::new();
        assert_eq!(extract_client_ip(&headers), "unknown");
    }

    #[test]
    fn test_extract_client_ip_empty_x_forwarded_for_falls_through() {
        let mut headers = HeaderMap::new();
        headers.insert("x-forwarded-for", "".parse().unwrap());
        headers.insert("x-real-ip", "3.3.3.3".parse().unwrap());
        assert_eq!(extract_client_ip(&headers), "3.3.3.3");
    }

    #[test]
    fn test_extract_client_ip_empty_both_returns_unknown() {
        let mut headers = HeaderMap::new();
        headers.insert("x-forwarded-for", "".parse().unwrap());
        headers.insert("x-real-ip", "".parse().unwrap());
        assert_eq!(extract_client_ip(&headers), "unknown");
    }

    #[test]
    fn test_extract_client_ip_whitespace_only_x_real_ip_returns_unknown() {
        let mut headers = HeaderMap::new();
        headers.insert("x-real-ip", "   ".parse().unwrap());
        assert_eq!(extract_client_ip(&headers), "unknown");
    }

    #[test]
    fn test_ip_rate_limit_allows_within_limit() {
        let ip = "ip-rate-test-allow-10.0.0.1";
        for i in 0..10 {
            assert!(
                check_ip_rate_limit(ip),
                "Attempt {} should be allowed",
                i + 1
            );
        }
    }

    #[test]
    fn test_ip_rate_limit_rejects_after_max() {
        let ip = "ip-rate-test-reject-10.0.0.2";
        for _ in 0..IP_RATE_LIMIT_MAX_ATTEMPTS {
            check_ip_rate_limit(ip);
        }
        assert!(
            !check_ip_rate_limit(ip),
            "Should be rejected after max attempts"
        );
    }

    #[test]
    fn test_ip_rate_limit_independent_keys() {
        let ip_a = "ip-rate-indep-a-10.0.0.3";
        let ip_b = "ip-rate-indep-b-10.0.0.4";
        for _ in 0..IP_RATE_LIMIT_MAX_ATTEMPTS {
            check_ip_rate_limit(ip_a);
        }
        assert!(!check_ip_rate_limit(ip_a), "ip_a should be rate-limited");
        assert!(check_ip_rate_limit(ip_b), "ip_b should NOT be rate-limited");
    }

    #[test]
    fn test_auth_user_response_from_auth_user_with_avatar() {
        let user = AuthUser {
            id: 99,
            email: "avatar@test.com".to_string(),
            name: "Avatar User".to_string(),
            role: "viewer".to_string(),
            avatar_url: Some("/avatars/99_abc.png".to_string()),
        };

        let response = AuthUserResponse::from(&user);

        assert_eq!(response.id, 99);
        assert_eq!(response.email, "avatar@test.com");
        assert_eq!(response.name, "Avatar User");
        assert_eq!(response.role, "viewer");
        assert_eq!(response.avatar_url.as_deref(), Some("/avatars/99_abc.png"));
    }

    #[tokio::test]
    async fn test_logout_without_session_cookie() {
        let (router, _tracker) = create_test_app();

        let response = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/auth/logout")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        // Logout should still succeed even without a cookie
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_update_user_short_password() {
        let (router, tracker) = create_test_app();
        let (_admin_id, token) = seed_admin_with_email(&tracker, "admin-update-shortpw@test.com");

        let hash = bcrypt::hash("password", 4).unwrap();
        let target_id = tracker
            .create_user("target-update-pw@test.com", &hash, "Target", "viewer")
            .unwrap();

        let response = router
            .oneshot(
                Request::builder()
                    .method("PUT")
                    .uri(format!("/api/users/{}", target_id))
                    .header("content-type", "application/json")
                    .header("cookie", format!("claudear_session={}", token))
                    .body(Body::from(r#"{"password":"short"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_update_user_password_too_long() {
        let (router, tracker) = create_test_app();
        let (_admin_id, token) = seed_admin_with_email(&tracker, "admin-update-longpw@test.com");

        let hash = bcrypt::hash("password", 4).unwrap();
        let target_id = tracker
            .create_user("target-update-longpw@test.com", &hash, "Target", "viewer")
            .unwrap();

        let long_pw = "a".repeat(73);
        let body = serde_json::json!({"password": long_pw});

        let response = router
            .oneshot(
                Request::builder()
                    .method("PUT")
                    .uri(format!("/api/users/{}", target_id))
                    .header("content-type", "application/json")
                    .header("cookie", format!("claudear_session={}", token))
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_update_user_with_valid_password() {
        let (router, tracker) = create_test_app();
        let (_admin_id, token) = seed_admin_with_email(&tracker, "admin-update-validpw@test.com");

        let hash = bcrypt::hash("password", 4).unwrap();
        let target_id = tracker
            .create_user("target-update-validpw@test.com", &hash, "Target", "viewer")
            .unwrap();

        let response = router
            .oneshot(
                Request::builder()
                    .method("PUT")
                    .uri(format!("/api/users/{}", target_id))
                    .header("content-type", "application/json")
                    .header("cookie", format!("claudear_session={}", token))
                    .body(Body::from(r#"{"password":"newlongpassword123"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_update_user_name() {
        let (router, tracker) = create_test_app();
        let (_admin_id, token) = seed_admin_with_email(&tracker, "admin-update-name@test.com");

        let hash = bcrypt::hash("password", 4).unwrap();
        let target_id = tracker
            .create_user("target-name-upd@test.com", &hash, "Original Name", "viewer")
            .unwrap();

        let response = router
            .oneshot(
                Request::builder()
                    .method("PUT")
                    .uri(format!("/api/users/{}", target_id))
                    .header("content-type", "application/json")
                    .header("cookie", format!("claudear_session={}", token))
                    .body(Body::from(r#"{"name":"Updated Name"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let body_str = String::from_utf8(body.to_vec()).unwrap();
        assert!(body_str.contains("Updated Name"));
    }

    #[tokio::test]
    async fn test_update_user_role_to_admin() {
        let (router, tracker) = create_test_app();
        let (_admin_id, token) = seed_admin_with_email(&tracker, "admin-role-to-admin@test.com");

        let hash = bcrypt::hash("password", 4).unwrap();
        let target_id = tracker
            .create_user("target-role-upd@test.com", &hash, "Target", "viewer")
            .unwrap();

        let response = router
            .oneshot(
                Request::builder()
                    .method("PUT")
                    .uri(format!("/api/users/{}", target_id))
                    .header("content-type", "application/json")
                    .header("cookie", format!("claudear_session={}", token))
                    .body(Body::from(r#"{"role":"admin"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let body_str = String::from_utf8(body.to_vec()).unwrap();
        assert!(body_str.contains("\"role\":\"admin\""));
    }

    #[tokio::test]
    async fn test_create_user_password_too_long() {
        let (router, tracker) = create_test_app();
        let (_admin_id, token) = seed_admin_with_email(&tracker, "admin-create-longpw@test.com");

        let long_pw = "b".repeat(73);
        let body = serde_json::json!({
            "email": "longpw@test.com",
            "password": long_pw,
            "name": "Long PW User",
            "role": "viewer"
        });

        let response = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/users")
                    .header("content-type", "application/json")
                    .header("cookie", format!("claudear_session={}", token))
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_create_user_whitespace_name() {
        let (router, tracker) = create_test_app();
        let (_admin_id, token) = seed_admin_with_email(&tracker, "admin-create-wsname@test.com");

        let response = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/users")
                    .header("content-type", "application/json")
                    .header("cookie", format!("claudear_session={}", token))
                    .body(Body::from(
                        r#"{"email":"x@test.com","password":"longpassword","name":"   ","role":"viewer"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_login_sends_x_forwarded_for_ip() {
        let (router, tracker) = create_test_app();
        let password_hash = bcrypt::hash("secret123", 4).unwrap();
        tracker
            .create_user("iptest@test.com", &password_hash, "IP User", "admin")
            .unwrap();

        let response = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/auth/login")
                    .header("content-type", "application/json")
                    .header("x-forwarded-for", "1.2.3.4")
                    .body(Body::from(
                        r#"{"email":"iptest@test.com","password":"secret123"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_update_profile_name_only() {
        let (router, tracker) = create_test_app();
        let (_admin_id, token) = seed_admin_with_email(&tracker, "admin-profile-name@test.com");

        let response = router
            .oneshot(
                Request::builder()
                    .method("PUT")
                    .uri("/api/auth/profile")
                    .header("content-type", "application/json")
                    .header("cookie", format!("claudear_session={}", token))
                    .body(Body::from(r#"{"name":"New Profile Name"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let body_str = String::from_utf8(body.to_vec()).unwrap();
        assert!(body_str.contains("New Profile Name"));
    }

    #[tokio::test]
    async fn test_update_profile_empty_name_rejected() {
        let (router, tracker) = create_test_app();
        let (_admin_id, token) =
            seed_admin_with_email(&tracker, "admin-profile-emptyname@test.com");

        let response = router
            .oneshot(
                Request::builder()
                    .method("PUT")
                    .uri("/api/auth/profile")
                    .header("content-type", "application/json")
                    .header("cookie", format!("claudear_session={}", token))
                    .body(Body::from(r#"{"name":"  "}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_update_profile_password_too_short() {
        let (router, tracker) = create_test_app();
        let (_admin_id, token) = seed_admin_with_email(&tracker, "admin-profile-shortpw@test.com");

        let response = router
            .oneshot(
                Request::builder()
                    .method("PUT")
                    .uri("/api/auth/profile")
                    .header("content-type", "application/json")
                    .header("cookie", format!("claudear_session={}", token))
                    .body(Body::from(
                        r#"{"password":"short","current_password":"password"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_update_profile_password_too_long() {
        let (router, tracker) = create_test_app();
        let (_admin_id, token) = seed_admin_with_email(&tracker, "admin-profile-longpw@test.com");

        let long_pw = "c".repeat(73);
        let body = serde_json::json!({
            "password": long_pw,
            "current_password": "password"
        });

        let response = router
            .oneshot(
                Request::builder()
                    .method("PUT")
                    .uri("/api/auth/profile")
                    .header("content-type", "application/json")
                    .header("cookie", format!("claudear_session={}", token))
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_update_profile_password_no_current_password() {
        let (router, tracker) = create_test_app();
        let (_admin_id, token) = seed_admin_with_email(&tracker, "admin-profile-nocurpw@test.com");

        let response = router
            .oneshot(
                Request::builder()
                    .method("PUT")
                    .uri("/api/auth/profile")
                    .header("content-type", "application/json")
                    .header("cookie", format!("claudear_session={}", token))
                    .body(Body::from(r#"{"password":"newpassword123"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_update_profile_password_wrong_current_password() {
        let (router, tracker) = create_test_app();
        let (_admin_id, token) =
            seed_admin_with_email(&tracker, "admin-profile-wrongcurpw@test.com");

        let response = router
            .oneshot(
                Request::builder()
                    .method("PUT")
                    .uri("/api/auth/profile")
                    .header("content-type", "application/json")
                    .header("cookie", format!("claudear_session={}", token))
                    .body(Body::from(
                        r#"{"password":"newpassword123","current_password":"wrongpassword"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn test_update_profile_password_success() {
        let (router, tracker) = create_test_app();
        let (_admin_id, token) =
            seed_admin_with_email(&tracker, "admin-profile-pwsuccess@test.com");

        let response = router
            .oneshot(
                Request::builder()
                    .method("PUT")
                    .uri("/api/auth/profile")
                    .header("content-type", "application/json")
                    .header("cookie", format!("claudear_session={}", token))
                    .body(Body::from(
                        r#"{"password":"newpassword123","current_password":"password"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_update_profile_unauthenticated() {
        let (router, _tracker) = create_test_app();

        let response = router
            .oneshot(
                Request::builder()
                    .method("PUT")
                    .uri("/api/auth/profile")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"name":"Nope"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_viewer_cannot_list_users() {
        let (router, tracker) = create_test_app();
        let (_viewer_id, token) = seed_viewer(&tracker);

        let response = router
            .oneshot(
                Request::builder()
                    .uri("/api/users")
                    .header("cookie", format!("claudear_session={}", token))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn test_viewer_cannot_get_user() {
        let (router, tracker) = create_test_app();
        let (_viewer_id, token) = seed_viewer(&tracker);

        let response = router
            .oneshot(
                Request::builder()
                    .uri("/api/users/1")
                    .header("cookie", format!("claudear_session={}", token))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn test_viewer_cannot_update_user() {
        let (router, tracker) = create_test_app();
        let (_viewer_id, token) = seed_viewer(&tracker);

        let response = router
            .oneshot(
                Request::builder()
                    .method("PUT")
                    .uri("/api/users/1")
                    .header("content-type", "application/json")
                    .header("cookie", format!("claudear_session={}", token))
                    .body(Body::from(r#"{"name":"Hacked"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn test_viewer_cannot_delete_user() {
        let (router, tracker) = create_test_app();
        let (_viewer_id, token) = seed_viewer(&tracker);

        let response = router
            .oneshot(
                Request::builder()
                    .method("DELETE")
                    .uri("/api/users/1")
                    .header("cookie", format!("claudear_session={}", token))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn test_unauthenticated_list_users() {
        let (router, _tracker) = create_test_app();

        let response = router
            .oneshot(
                Request::builder()
                    .uri("/api/users")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_unauthenticated_create_user() {
        let (router, _tracker) = create_test_app();

        let response = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/users")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"email":"x@test.com","password":"longpassword","name":"X","role":"viewer"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_invalid_session_token() {
        let (router, _tracker) = create_test_app();

        let response = router
            .oneshot(
                Request::builder()
                    .uri("/api/auth/me")
                    .header("cookie", "claudear_session=invalid-token-12345")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_create_user_admin_role() {
        let (router, tracker) = create_test_app();
        let (_admin_id, token) = seed_admin_with_email(&tracker, "admin-create-adminrole@test.com");

        let response = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/users")
                    .header("content-type", "application/json")
                    .header("cookie", format!("claudear_session={}", token))
                    .body(Body::from(
                        r#"{"email":"newadmin@test.com","password":"longpassword","name":"New Admin","role":"admin"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::CREATED);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let body_str = String::from_utf8(body.to_vec()).unwrap();
        assert!(body_str.contains("\"role\":\"admin\""));
    }

    #[tokio::test]
    async fn test_login_response_body_structure() {
        let (router, tracker) = create_test_app();
        let password_hash = bcrypt::hash("testpass1", 4).unwrap();
        tracker
            .create_user("struct@test.com", &password_hash, "Struct User", "viewer")
            .unwrap();

        let response = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/auth/login")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"email":"struct@test.com","password":"testpass1"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = response.into_body().collect().await.unwrap().to_bytes();
        let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(parsed["user"].is_object());
        assert_eq!(parsed["user"]["email"], "struct@test.com");
        assert_eq!(parsed["user"]["name"], "Struct User");
        assert_eq!(parsed["user"]["role"], "viewer");
    }

    #[tokio::test]
    async fn test_update_user_email() {
        let (router, tracker) = create_test_app();
        let (_admin_id, token) = seed_admin_with_email(&tracker, "admin-update-email@test.com");

        let hash = bcrypt::hash("password", 4).unwrap();
        let target_id = tracker
            .create_user("original-email@test.com", &hash, "Email User", "viewer")
            .unwrap();

        let response = router
            .oneshot(
                Request::builder()
                    .method("PUT")
                    .uri(format!("/api/users/{}", target_id))
                    .header("content-type", "application/json")
                    .header("cookie", format!("claudear_session={}", token))
                    .body(Body::from(r#"{"email":"new-email@test.com"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let body_str = String::from_utf8(body.to_vec()).unwrap();
        assert!(body_str.contains("new-email@test.com"));
    }

    #[tokio::test]
    async fn test_upload_avatar_no_auth() {
        let (router, _tracker) = create_test_app();

        // No auth cookie -> should be rejected
        let boundary = "---TestBoundary123";
        let body = format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"avatar\"; filename=\"test.png\"\r\nContent-Type: image/png\r\n\r\nfakedata\r\n--{boundary}--\r\n"
        );

        let response = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/auth/avatar")
                    .header(
                        "content-type",
                        format!("multipart/form-data; boundary={}", boundary),
                    )
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        // Should be 401 (no auth)
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_upload_avatar_with_valid_png() {
        reset_api_rate_limiter();
        let temp_dir = tempfile::TempDir::new().unwrap();
        let storage_dir = temp_dir.path().to_path_buf();
        let avatars_dir = storage_dir.join("avatars");
        std::fs::create_dir_all(&avatars_dir).unwrap();

        let tracker: std::sync::Arc<dyn claudear_storage::FixAttemptTracker> =
            std::sync::Arc::new(SqliteTracker::in_memory().unwrap());

        let mut config = test_config();
        config.storage_dir = storage_dir;

        let password_hash = bcrypt::hash("password", 4).unwrap();
        tracker
            .create_user("avatar@test.com", &password_hash, "Avatar User", "admin")
            .unwrap();
        let token = tracker.create_session(1, "2099-12-31 23:59:59").unwrap();

        let indexing_rx = tracker.subscribe_indexing_progress();
        let router = create_api_router(
            config,
            tracker.clone(),
            std::path::PathBuf::from("claudear.toml"),
            indexing_rx,
        )
        .layer(CookieManagerLayer::new());

        // Build a valid PNG (minimal PNG header)
        let mut png_data: Vec<u8> = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        png_data.extend_from_slice(&[0u8; 100]); // Some padding

        let boundary = "----TestBoundary456";
        let mut body_bytes: Vec<u8> = Vec::new();
        body_bytes.extend_from_slice(
            format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"avatar\"; filename=\"test.png\"\r\nContent-Type: image/png\r\n\r\n"
            ).as_bytes()
        );
        body_bytes.extend_from_slice(&png_data);
        body_bytes.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());

        let response = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/auth/avatar")
                    .header("cookie", format!("claudear_session={}", token))
                    .header(
                        "content-type",
                        format!("multipart/form-data; boundary={}", boundary),
                    )
                    .body(Body::from(body_bytes))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let resp: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(resp["avatar_url"]
            .as_str()
            .unwrap()
            .starts_with("/avatars/"));
        assert!(resp["avatar_url"].as_str().unwrap().ends_with(".png"));
    }

    #[tokio::test]
    async fn test_upload_avatar_invalid_content_type() {
        reset_api_rate_limiter();
        let temp_dir = tempfile::TempDir::new().unwrap();
        let storage_dir = temp_dir.path().to_path_buf();
        let avatars_dir = storage_dir.join("avatars");
        std::fs::create_dir_all(&avatars_dir).unwrap();

        let tracker: std::sync::Arc<dyn claudear_storage::FixAttemptTracker> =
            std::sync::Arc::new(SqliteTracker::in_memory().unwrap());

        let mut config = test_config();
        config.storage_dir = storage_dir;

        let password_hash = bcrypt::hash("password", 4).unwrap();
        tracker
            .create_user("avatar2@test.com", &password_hash, "Avatar User", "admin")
            .unwrap();
        let token = tracker.create_session(1, "2099-12-31 23:59:59").unwrap();

        let indexing_rx = tracker.subscribe_indexing_progress();
        let router = create_api_router(
            config,
            tracker.clone(),
            std::path::PathBuf::from("claudear.toml"),
            indexing_rx,
        )
        .layer(CookieManagerLayer::new());

        // Upload a text/plain file (not allowed)
        let boundary = "----TestBoundary789";
        let body = format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"avatar\"; filename=\"test.txt\"\r\nContent-Type: text/plain\r\n\r\nHello World\r\n--{boundary}--\r\n"
        );

        let response = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/auth/avatar")
                    .header("cookie", format!("claudear_session={}", token))
                    .header(
                        "content-type",
                        format!("multipart/form-data; boundary={}", boundary),
                    )
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_upload_avatar_wrong_magic_bytes() {
        reset_api_rate_limiter();
        let temp_dir = tempfile::TempDir::new().unwrap();
        let storage_dir = temp_dir.path().to_path_buf();
        let avatars_dir = storage_dir.join("avatars");
        std::fs::create_dir_all(&avatars_dir).unwrap();

        let tracker: std::sync::Arc<dyn claudear_storage::FixAttemptTracker> =
            std::sync::Arc::new(SqliteTracker::in_memory().unwrap());

        let mut config = test_config();
        config.storage_dir = storage_dir;

        let password_hash = bcrypt::hash("password", 4).unwrap();
        tracker
            .create_user("avatar3@test.com", &password_hash, "Avatar User", "admin")
            .unwrap();
        let token = tracker.create_session(1, "2099-12-31 23:59:59").unwrap();

        let indexing_rx = tracker.subscribe_indexing_progress();
        let router = create_api_router(
            config,
            tracker.clone(),
            std::path::PathBuf::from("claudear.toml"),
            indexing_rx,
        )
        .layer(CookieManagerLayer::new());

        // Claim image/png but provide wrong magic bytes
        let fake_data = b"not a real png file at all";
        let boundary = "----TestBoundaryMagic";
        let mut body_bytes: Vec<u8> = Vec::new();
        body_bytes.extend_from_slice(
            format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"avatar\"; filename=\"fake.png\"\r\nContent-Type: image/png\r\n\r\n"
            ).as_bytes()
        );
        body_bytes.extend_from_slice(fake_data);
        body_bytes.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());

        let response = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/auth/avatar")
                    .header("cookie", format!("claudear_session={}", token))
                    .header(
                        "content-type",
                        format!("multipart/form-data; boundary={}", boundary),
                    )
                    .body(Body::from(body_bytes))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_upload_avatar_no_avatar_field() {
        reset_api_rate_limiter();
        let temp_dir = tempfile::TempDir::new().unwrap();
        let storage_dir = temp_dir.path().to_path_buf();
        let avatars_dir = storage_dir.join("avatars");
        std::fs::create_dir_all(&avatars_dir).unwrap();

        let tracker: std::sync::Arc<dyn claudear_storage::FixAttemptTracker> =
            std::sync::Arc::new(SqliteTracker::in_memory().unwrap());

        let mut config = test_config();
        config.storage_dir = storage_dir;

        let password_hash = bcrypt::hash("password", 4).unwrap();
        tracker
            .create_user("avatar4@test.com", &password_hash, "Avatar User", "admin")
            .unwrap();
        let token = tracker.create_session(1, "2099-12-31 23:59:59").unwrap();

        let indexing_rx = tracker.subscribe_indexing_progress();
        let router = create_api_router(
            config,
            tracker.clone(),
            std::path::PathBuf::from("claudear.toml"),
            indexing_rx,
        )
        .layer(CookieManagerLayer::new());

        // Upload with wrong field name (not "avatar")
        let boundary = "----TestBoundaryNoField";
        let body = format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"photo\"; filename=\"test.png\"\r\nContent-Type: image/png\r\n\r\nfakedata\r\n--{boundary}--\r\n"
        );

        let response = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/auth/avatar")
                    .header("cookie", format!("claudear_session={}", token))
                    .header(
                        "content-type",
                        format!("multipart/form-data; boundary={}", boundary),
                    )
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        // No "avatar" field found => BAD_REQUEST
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_upload_avatar_valid_jpeg() {
        reset_api_rate_limiter();
        let temp_dir = tempfile::TempDir::new().unwrap();
        let storage_dir = temp_dir.path().to_path_buf();
        let avatars_dir = storage_dir.join("avatars");
        std::fs::create_dir_all(&avatars_dir).unwrap();

        let tracker: std::sync::Arc<dyn claudear_storage::FixAttemptTracker> =
            std::sync::Arc::new(SqliteTracker::in_memory().unwrap());

        let mut config = test_config();
        config.storage_dir = storage_dir;

        let password_hash = bcrypt::hash("password", 4).unwrap();
        tracker
            .create_user("avatar5@test.com", &password_hash, "Avatar User", "admin")
            .unwrap();
        let token = tracker.create_session(1, "2099-12-31 23:59:59").unwrap();

        let indexing_rx = tracker.subscribe_indexing_progress();
        let router = create_api_router(
            config,
            tracker.clone(),
            std::path::PathBuf::from("claudear.toml"),
            indexing_rx,
        )
        .layer(CookieManagerLayer::new());

        // Build minimal JPEG header
        let mut jpeg_data: Vec<u8> = vec![0xFF, 0xD8, 0xFF, 0xE0];
        jpeg_data.extend_from_slice(&[0u8; 100]);

        let boundary = "----TestBoundaryJpeg";
        let mut body_bytes: Vec<u8> = Vec::new();
        body_bytes.extend_from_slice(
            format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"avatar\"; filename=\"photo.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n"
            ).as_bytes()
        );
        body_bytes.extend_from_slice(&jpeg_data);
        body_bytes.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());

        let response = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/auth/avatar")
                    .header("cookie", format!("claudear_session={}", token))
                    .header(
                        "content-type",
                        format!("multipart/form-data; boundary={}", boundary),
                    )
                    .body(Body::from(body_bytes))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let resp: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(resp["avatar_url"].as_str().unwrap().ends_with(".jpg"));
    }
}
