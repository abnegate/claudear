//! GitHub PR monitoring and issue resolution.

use crate::scm::{
    CodeReview, InlineReviewComment, PostReviewAction, PrSummary, RemoteRepo, ReviewComment,
    ReviewUser, ScmProvider, ScmRelease,
};
use async_trait::async_trait;
use claudear_config::config::GitHubConfig;
use claudear_core::error::{Error, Result};
use claudear_core::secret::OptionalSecretExt;
use serde::Deserialize;

// Backward-compatibility re-exports (types moved to scm module)
pub use crate::scm::{
    CodeReview as PrReview, PrInfo, PrMonitor, PrReviewState, PrStatus, PrStatusUpdate,
    RemoteRepo as OrgRepo, ReviewComment as PrReviewComment, ReviewEvent, ReviewUser as GitHubUser,
    ReviewWatcher,
};

// Backward-compatibility re-exports (types moved to http module)
pub use claudear_core::http::{HttpClient, ReqwestHttpClient};

/// GitHub API client for PR monitoring.
pub struct GitHubClient<H: HttpClient = ReqwestHttpClient> {
    config: GitHubConfig,
    http: H,
}

#[derive(Debug, Deserialize)]
struct PullRequest {
    state: String,
    merged: bool,
    head: Option<PullRequestRef>,
    base: Option<PullRequestRef>,
    title: Option<String>,
    user: Option<ReviewUser>,
}

#[derive(Debug, Deserialize)]
struct PullRequestRef {
    #[serde(rename = "ref")]
    ref_name: String,
}

/// Minimal shape of the response from creating a PR / reading repo metadata.
#[derive(Debug, Deserialize)]
struct CreatedPr {
    html_url: String,
}

#[derive(Debug, Deserialize)]
struct RepoMeta {
    default_branch: String,
}

/// A GitHub issue from the Issues API.
#[derive(Debug, Deserialize)]
pub struct GitHubIssue {
    pub number: i64,
    pub title: String,
    pub body: Option<String>,
    pub state: String,
    pub html_url: String,
    #[serde(default)]
    pub labels: Vec<GitHubLabel>,
    pub user: Option<ReviewUser>,
    pub assignee: Option<ReviewUser>,
    /// Present when the "issue" is actually a pull request.
    pub pull_request: Option<serde_json::Value>,
}

/// A label on a GitHub issue.
#[derive(Debug, Deserialize)]
pub struct GitHubLabel {
    pub name: String,
}

impl GitHubClient<ReqwestHttpClient> {
    /// Create a new GitHub client with the default HTTP client.
    pub fn new(config: GitHubConfig) -> Self {
        Self {
            config,
            http: ReqwestHttpClient::new(),
        }
    }
}

impl<H: HttpClient> GitHubClient<H> {
    /// Create a new GitHub client with a custom HTTP client.
    pub fn with_http_client(config: GitHubConfig, http: H) -> Self {
        Self { config, http }
    }

    /// Check if configured (has token).
    pub fn is_enabled(&self) -> bool {
        self.config.token.is_some()
    }

    /// Get the review trigger tag (e.g., "@claudear").
    pub fn review_trigger(&self) -> &str {
        &self.config.review_trigger
    }

    /// Get the list of bot handles whose review comments should be processed.
    pub fn allowed_bots(&self) -> &[String] {
        &self.config.allowed_bots
    }

    /// Build standard GitHub API headers.
    fn build_headers(&self, token: &str) -> Vec<(&'static str, String)> {
        vec![
            ("Authorization", format!("Bearer {}", token)),
            ("Accept", "application/vnd.github+json".to_string()),
            ("User-Agent", "claudear".to_string()),
            ("X-GitHub-Api-Version", "2022-11-28".to_string()),
        ]
    }

    /// Get the status of a PR.
    pub async fn get_pr_status(&self, repo: &str, pr_number: i64) -> Result<PrStatus> {
        let token = self
            .config
            .token
            .as_ref()
            .ok_or_else(|| Error::config("GitHub token not configured"))?
            .expose();

        let url = format!("https://api.github.com/repos/{}/pulls/{}", repo, pr_number);
        let headers = self.build_headers(token);

        let response = self.http.get(&url, headers).await?;

        if response.is_not_found() {
            return Err(Error::Other(format!(
                "PR not found: {}/pull/{}",
                repo, pr_number
            )));
        }

        if !response.is_success() {
            return Err(Error::Other(format!("GitHub API error: {}", response.body)));
        }

        let pr: PullRequest = response.json()?;

        Ok(if pr.merged {
            PrStatus::Merged
        } else if pr.state == "closed" {
            PrStatus::Closed
        } else {
            PrStatus::Open
        })
    }

    /// Get lightweight PR info (branches, title, author).
    pub async fn get_pr_info(&self, repo: &str, pr_number: i64) -> Result<PrInfo> {
        let token = self
            .config
            .token
            .as_ref()
            .ok_or_else(|| Error::config("GitHub token not configured"))?
            .expose();

        let url = format!("https://api.github.com/repos/{}/pulls/{}", repo, pr_number);
        let headers = self.build_headers(token);

        let response = self.http.get(&url, headers).await?;

        if !response.is_success() {
            return Err(Error::Other(format!("GitHub API error: {}", response.body)));
        }

        let pr: PullRequest = response.json()?;

        Ok(PrInfo {
            head_branch: pr.head.map(|h| h.ref_name),
            base_branch: pr.base.map(|b| b.ref_name),
            title: pr.title,
            author: pr.user.map(|u| u.login),
        })
    }

    /// Get reviews for a PR.
    pub async fn get_pr_reviews(&self, repo: &str, pr_number: i64) -> Result<Vec<CodeReview>> {
        let token = self
            .config
            .token
            .as_ref()
            .ok_or_else(|| Error::config("GitHub token not configured"))?
            .expose();

        let base_url = format!(
            "https://api.github.com/repos/{}/pulls/{}/reviews",
            repo, pr_number
        );
        let headers = self.build_headers(token);

        let mut all_reviews = Vec::new();
        let mut page = 1usize;
        const DEFAULT_PAGE_SIZE: usize = 30;
        const MAX_PAGES: usize = 100;

        loop {
            let url = if page == 1 {
                base_url.clone()
            } else {
                format!("{}?page={}", base_url, page)
            };
            let response = self.http.get(&url, headers.clone()).await?;

            if !response.is_success() {
                return Err(Error::Other(format!(
                    "GitHub API error ({}): {}",
                    response.status, response.body
                )));
            }

            let reviews: Vec<CodeReview> = response.json()?;
            let count = reviews.len();
            all_reviews.extend(reviews);

            if count < DEFAULT_PAGE_SIZE {
                break;
            }

            page += 1;
            if page > MAX_PAGES {
                tracing::warn!(repo = %repo, pr_number, "Hit pagination limit for PR reviews");
                break;
            }
        }

        Ok(all_reviews)
    }

    /// Get review comments for a PR.
    pub async fn get_pr_review_comments(
        &self,
        repo: &str,
        pr_number: i64,
    ) -> Result<Vec<ReviewComment>> {
        let token = self
            .config
            .token
            .as_ref()
            .ok_or_else(|| Error::config("GitHub token not configured"))?
            .expose();

        let base_url = format!(
            "https://api.github.com/repos/{}/pulls/{}/comments",
            repo, pr_number
        );
        let headers = self.build_headers(token);

        let mut all_comments = Vec::new();
        let mut page = 1usize;
        const DEFAULT_PAGE_SIZE: usize = 30;
        const MAX_PAGES: usize = 100;

        loop {
            let url = if page == 1 {
                base_url.clone()
            } else {
                format!("{}?page={}", base_url, page)
            };
            let response = self.http.get(&url, headers.clone()).await?;

            if !response.is_success() {
                return Err(Error::Other(format!(
                    "GitHub API error ({}): {}",
                    response.status, response.body
                )));
            }

            let comments: Vec<ReviewComment> = response.json()?;
            let count = comments.len();
            all_comments.extend(comments);

            if count < DEFAULT_PAGE_SIZE {
                break;
            }

            page += 1;
            if page > MAX_PAGES {
                tracing::warn!(repo = %repo, pr_number, "Hit pagination limit for PR review comments");
                break;
            }
        }

        Ok(all_comments)
    }

    /// Get a PR's *conversation* comments (the issue-comment timeline), as opposed
    /// to inline review-thread comments. GitHub models a PR as an issue, so these
    /// come from `/repos/{repo}/issues/{n}/comments` — a plain "@claudear fix this"
    /// left on the PR conversation lands here, not in `/pulls/{n}/comments`.
    ///
    /// They are mapped into [`ReviewComment`] with an empty `path`/absent `line`
    /// and no `pull_request_review_id`, so they flow through the same standalone
    /// review-feedback pipeline as inline comments.
    pub async fn get_pr_issue_comments(
        &self,
        repo: &str,
        pr_number: i64,
    ) -> Result<Vec<ReviewComment>> {
        // A PR conversation comment as returned by the issues-comments endpoint.
        // Only the fields we carry forward are deserialized.
        #[derive(serde::Deserialize)]
        struct GitHubIssueComment {
            id: i64,
            #[serde(default)]
            body: Option<String>,
            user: ReviewUser,
            created_at: String,
            updated_at: String,
            html_url: String,
        }

        let token = self
            .config
            .token
            .as_ref()
            .ok_or_else(|| Error::config("GitHub token not configured"))?
            .expose();

        let base_url = format!(
            "https://api.github.com/repos/{}/issues/{}/comments",
            repo, pr_number
        );
        let headers = self.build_headers(token);

        let mut all_comments = Vec::new();
        let mut page = 1usize;
        const DEFAULT_PAGE_SIZE: usize = 30;
        const MAX_PAGES: usize = 100;

        loop {
            let url = if page == 1 {
                base_url.clone()
            } else {
                format!("{}?page={}", base_url, page)
            };
            let response = self.http.get(&url, headers.clone()).await?;

            if !response.is_success() {
                return Err(Error::Other(format!(
                    "GitHub API error ({}): {}",
                    response.status, response.body
                )));
            }

            let comments: Vec<GitHubIssueComment> = response.json()?;
            let count = comments.len();
            all_comments.extend(comments.into_iter().map(|c| ReviewComment {
                id: c.id,
                path: String::new(),
                position: None,
                original_position: None,
                body: c.body.unwrap_or_default(),
                user: c.user,
                created_at: c.created_at,
                updated_at: c.updated_at,
                html_url: c.html_url,
                pull_request_review_id: None,
                line: None,
                start_line: None,
                side: None,
            }));

            if count < DEFAULT_PAGE_SIZE {
                break;
            }

            page += 1;
            if page > MAX_PAGES {
                tracing::warn!(repo = %repo, pr_number, "Hit pagination limit for PR issue comments");
                break;
            }
        }

        Ok(all_comments)
    }

    /// Get the GitHub token (if configured).
    pub fn token(&self) -> Option<&str> {
        self.config.token.expose_as_deref()
    }

    /// Fetch the raw unified diff for a PR.
    pub async fn get_pr_diff(&self, repo: &str, pr_number: i64) -> Result<String> {
        let token = self
            .config
            .token
            .as_ref()
            .ok_or_else(|| Error::config("GitHub token not configured"))?
            .expose();

        let url = format!("https://api.github.com/repos/{}/pulls/{}", repo, pr_number);
        let headers = vec![
            ("Authorization", format!("Bearer {}", token)),
            ("Accept", "application/vnd.github.v3.diff".to_string()),
            ("User-Agent", "claudear".to_string()),
            ("X-GitHub-Api-Version", "2022-11-28".to_string()),
        ];

        let response = self.http.get(&url, headers).await?;

        if !response.is_success() {
            return Err(Error::Other(format!(
                "GitHub API error fetching diff for {}/pull/{}: {}",
                repo, pr_number, response.body
            )));
        }

        Ok(response.body)
    }

    /// List all repositories for an organization.
    ///
    /// Paginates through all results (100 per page) and returns all repos.
    /// Excludes archived repositories.
    pub async fn list_org_repos(&self, org: &str) -> Result<Vec<RemoteRepo>> {
        let token = self
            .config
            .token
            .as_ref()
            .ok_or_else(|| Error::config("GitHub token not configured"))?
            .expose();

        let mut all_repos = Vec::new();
        let mut page = 1;
        const PER_PAGE: usize = 100;

        loop {
            let url = format!(
                "https://api.github.com/orgs/{}/repos?per_page={}&page={}",
                org, PER_PAGE, page
            );
            let headers = self.build_headers(token);

            let response = self.http.get(&url, headers).await?;

            if response.is_not_found() {
                return Err(Error::Other(format!("Organization not found: {}", org)));
            }

            if !response.is_success() {
                return Err(Error::Other(format!(
                    "GitHub API error ({}): {}",
                    response.status, response.body
                )));
            }

            let repos: Vec<RemoteRepo> = response.json()?;
            let count = repos.len();

            // Filter out archived repos
            let active_repos: Vec<RemoteRepo> = repos.into_iter().filter(|r| !r.archived).collect();
            all_repos.extend(active_repos);

            // If we got fewer than per_page, we've reached the end
            if count < PER_PAGE {
                break;
            }

            page += 1;

            // Safety limit to prevent infinite loops
            if page > 100 {
                tracing::warn!(org = %org, "Hit pagination limit (100 pages) for org repos");
                break;
            }
        }

        tracing::info!(org = %org, count = all_repos.len(), "Fetched organization repositories");
        Ok(all_repos)
    }

    /// Merge a PR using squash merge.
    pub async fn merge_pr(&self, repo: &str, pr_number: i64) -> Result<()> {
        let token = self
            .config
            .token
            .as_ref()
            .ok_or_else(|| Error::config("GitHub token not configured"))?
            .expose();

        let url = format!(
            "https://api.github.com/repos/{}/pulls/{}/merge",
            repo, pr_number
        );
        let headers = self.build_headers(token);
        let body = serde_json::json!({"merge_method": "squash"}).to_string();

        let response = self.http.put(&url, headers, &body).await?;
        if !response.is_success() {
            return Err(Error::Other(format!(
                "Failed to merge PR {}/pull/{}: {}",
                repo, pr_number, response.body
            )));
        }
        Ok(())
    }

    /// Close a PR without merging.
    pub async fn close_pr(&self, repo: &str, pr_number: i64) -> Result<()> {
        let token = self
            .config
            .token
            .as_ref()
            .ok_or_else(|| Error::config("GitHub token not configured"))?
            .expose();

        let url = format!("https://api.github.com/repos/{}/pulls/{}", repo, pr_number);
        let headers = self.build_headers(token);
        let body = serde_json::json!({"state": "closed"}).to_string();

        let response = self.http.patch(&url, headers, &body).await?;
        if !response.is_success() {
            return Err(Error::Other(format!(
                "Failed to close PR {}/pull/{}: {}",
                repo, pr_number, response.body
            )));
        }
        Ok(())
    }

    /// Delete a remote branch.
    pub async fn delete_branch(&self, repo: &str, branch: &str) -> Result<()> {
        let token = self
            .config
            .token
            .as_ref()
            .ok_or_else(|| Error::config("GitHub token not configured"))?
            .expose();

        let url = format!(
            "https://api.github.com/repos/{}/git/refs/heads/{}",
            repo, branch
        );
        let headers = self.build_headers(token);

        let response = self.http.delete(&url, headers).await?;
        if !response.is_success() && !response.is_not_found() {
            return Err(Error::Other(format!(
                "Failed to delete branch {} in {}: {}",
                branch, repo, response.body
            )));
        }
        Ok(())
    }

    /// Post a review on a PR.
    pub async fn post_review(
        &self,
        repo: &str,
        pr_number: i64,
        action: PostReviewAction,
        body_text: &str,
    ) -> Result<()> {
        let token = self
            .config
            .token
            .as_ref()
            .ok_or_else(|| Error::config("GitHub token not configured"))?
            .expose();

        let url = format!(
            "https://api.github.com/repos/{}/pulls/{}/reviews",
            repo, pr_number
        );
        let headers = self.build_headers(token);

        let event = match action {
            PostReviewAction::Comment => "COMMENT",
            PostReviewAction::RequestChanges => "REQUEST_CHANGES",
            PostReviewAction::Approve => "APPROVE",
        };

        let body = serde_json::json!({
            "event": event,
            "body": body_text,
        })
        .to_string();

        let response = self.http.post(&url, headers, &body).await?;
        if !response.is_success() {
            return Err(Error::Other(format!(
                "Failed to post review on {}/pull/{}: {}",
                repo, pr_number, response.body
            )));
        }
        Ok(())
    }

    /// Post a review with file-level inline comments.
    pub async fn post_review_with_comments(
        &self,
        repo: &str,
        pr_number: i64,
        action: PostReviewAction,
        body_text: &str,
        comments: &[InlineReviewComment],
    ) -> Result<()> {
        let token = self
            .config
            .token
            .as_ref()
            .ok_or_else(|| Error::config("GitHub token not configured"))?
            .expose();

        let url = format!(
            "https://api.github.com/repos/{}/pulls/{}/reviews",
            repo, pr_number
        );
        let headers = self.build_headers(token);

        let event = match action {
            PostReviewAction::Comment => "COMMENT",
            PostReviewAction::RequestChanges => "REQUEST_CHANGES",
            PostReviewAction::Approve => "APPROVE",
        };

        let comments_json: Vec<_> = comments
            .iter()
            .map(|c| {
                serde_json::json!({
                    "path": c.path,
                    "body": c.body,
                    "position": c.position.unwrap_or(1)
                })
            })
            .collect();

        let body = serde_json::json!({
            "event": event,
            "body": body_text,
            "comments": comments_json,
        })
        .to_string();

        let response = self.http.post(&url, headers, &body).await?;
        if !response.is_success() {
            return Err(Error::Other(format!(
                "Failed to post review with comments on {}/pull/{}: {}",
                repo, pr_number, response.body
            )));
        }
        Ok(())
    }

    /// List open PRs for a repository.
    pub async fn list_open_prs(&self, repo: &str) -> Result<Vec<PrSummary>> {
        let token = self
            .config
            .token
            .as_ref()
            .ok_or_else(|| Error::config("GitHub token not configured"))?
            .expose();

        let url = format!(
            "https://api.github.com/repos/{}/pulls?state=open&per_page=100",
            repo
        );
        let headers = self.build_headers(token);
        let response = self.http.get(&url, headers).await?;

        if !response.is_success() {
            return Err(Error::Other(format!(
                "Failed to list open PRs for {}: {}",
                repo, response.body
            )));
        }

        let prs: Vec<serde_json::Value> = response.json()?;
        Ok(prs
            .into_iter()
            .filter_map(|pr| {
                Some(PrSummary {
                    number: pr.get("number")?.as_i64()?,
                    title: pr.get("title")?.as_str()?.to_string(),
                    branch: pr.get("head")?.get("ref")?.as_str()?.to_string(),
                    url: pr.get("html_url")?.as_str()?.to_string(),
                })
            })
            .collect())
    }

    /// Get the latest release for a repository.
    pub async fn get_latest_release(&self, repo: &str) -> Result<Option<ScmRelease>> {
        let token = self
            .config
            .token
            .as_ref()
            .ok_or_else(|| Error::config("GitHub token not configured"))?
            .expose();

        let url = format!("https://api.github.com/repos/{}/releases/latest", repo);
        let headers = self.build_headers(token);
        let response = self.http.get(&url, headers).await?;

        if response.status == 404 {
            return Ok(None);
        }

        if !response.is_success() {
            return Err(Error::Other(format!(
                "Failed to get latest release for {}: {}",
                repo, response.body
            )));
        }

        let val: serde_json::Value = response.json()?;
        Ok(Some(ScmRelease {
            tag: val
                .get("tag_name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            name: val.get("name").and_then(|v| v.as_str()).map(String::from),
            url: val
                .get("html_url")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            published_at: val
                .get("published_at")
                .and_then(|v| v.as_str())
                .map(String::from),
        }))
    }

    /// Create a release for a repository.
    pub async fn create_release(
        &self,
        repo: &str,
        tag: &str,
        name: &str,
        body: &str,
    ) -> Result<ScmRelease> {
        let token = self
            .config
            .token
            .as_ref()
            .ok_or_else(|| Error::config("GitHub token not configured"))?
            .expose();

        let url = format!("https://api.github.com/repos/{}/releases", repo);
        let headers = self.build_headers(token);
        let payload = serde_json::json!({
            "tag_name": tag,
            "name": name,
            "body": body,
        });

        let response = self.http.post(&url, headers, &payload.to_string()).await?;

        if !response.is_success() {
            return Err(Error::Other(format!(
                "Failed to create release for {}: {}",
                repo, response.body
            )));
        }

        let val: serde_json::Value = response.json()?;
        Ok(ScmRelease {
            tag: val
                .get("tag_name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            name: val.get("name").and_then(|v| v.as_str()).map(String::from),
            url: val
                .get("html_url")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            published_at: val
                .get("published_at")
                .and_then(|v| v.as_str())
                .map(String::from),
        })
    }

    /// Parse a PR number from a GitHub PR URL.
    pub fn parse_pr_number(url: &str) -> Option<i64> {
        // Match /pull/123 or /pulls/123
        let re = regex_lite::Regex::new(r"/pulls?/(\d+)").ok()?;
        let caps = re.captures(url)?;
        caps.get(1)?.as_str().parse().ok()
    }

    /// List issues for a repository, filtering by state and labels.
    ///
    /// Filters out pull requests (GitHub Issues API returns PRs as issues).
    pub async fn list_repo_issues(
        &self,
        repo: &str,
        state: Option<&str>,
        labels: &[String],
    ) -> Result<Vec<GitHubIssue>> {
        let token = self
            .config
            .token
            .as_ref()
            .ok_or_else(|| Error::config("GitHub token not configured"))?
            .expose();

        let mut all_issues = Vec::new();
        let mut page = 1usize;
        const PER_PAGE: usize = 100;
        const MAX_PAGES: usize = 10;

        loop {
            let mut url = format!(
                "https://api.github.com/repos/{}/issues?per_page={}&page={}",
                repo, PER_PAGE, page
            );
            if let Some(state) = state {
                url.push_str(&format!("&state={}", state));
            }
            if !labels.is_empty() {
                url.push_str(&format!("&labels={}", labels.join(",")));
            }

            let headers = self.build_headers(token);
            let response = self.http.get(&url, headers).await?;

            if !response.is_success() {
                return Err(Error::Other(format!(
                    "GitHub API error listing issues for {}: {}",
                    repo, response.body
                )));
            }

            let issues: Vec<GitHubIssue> = response.json()?;
            let count = issues.len();

            // Filter out pull requests (GitHub Issues API includes PRs)
            let real_issues: Vec<GitHubIssue> = issues
                .into_iter()
                .filter(|i| i.pull_request.is_none())
                .collect();
            all_issues.extend(real_issues);

            if count < PER_PAGE {
                break;
            }

            page += 1;
            if page > MAX_PAGES {
                tracing::warn!(repo = %repo, "Hit pagination limit for repo issues");
                break;
            }
        }

        Ok(all_issues)
    }

    /// Fetch a single issue by number.
    pub async fn get_issue(&self, repo: &str, issue_number: i64) -> Result<GitHubIssue> {
        let token = self
            .config
            .token
            .as_ref()
            .ok_or_else(|| Error::config("GitHub token not configured"))?
            .expose();

        let url = format!(
            "https://api.github.com/repos/{}/issues/{}",
            repo, issue_number
        );
        let headers = self.build_headers(token);
        let response = self.http.get(&url, headers).await?;

        if response.is_not_found() {
            return Err(Error::Other(format!(
                "Issue not found: {}/issues/{}",
                repo, issue_number
            )));
        }

        if !response.is_success() {
            return Err(Error::Other(format!("GitHub API error: {}", response.body)));
        }

        response.json()
    }

    /// Close an issue.
    pub async fn close_issue(&self, repo: &str, issue_number: i64) -> Result<()> {
        let token = self
            .config
            .token
            .as_ref()
            .ok_or_else(|| Error::config("GitHub token not configured"))?
            .expose();

        let url = format!(
            "https://api.github.com/repos/{}/issues/{}",
            repo, issue_number
        );
        let headers = self.build_headers(token);
        let body = serde_json::json!({"state": "closed"}).to_string();

        let response = self.http.patch(&url, headers, &body).await?;
        if !response.is_success() {
            return Err(Error::Other(format!(
                "Failed to close issue {}/issues/{}: {}",
                repo, issue_number, response.body
            )));
        }
        Ok(())
    }

    /// Add a comment to an issue.
    pub async fn add_issue_comment(
        &self,
        repo: &str,
        issue_number: i64,
        comment: &str,
    ) -> Result<()> {
        let token = self
            .config
            .token
            .as_ref()
            .ok_or_else(|| Error::config("GitHub token not configured"))?
            .expose();

        let url = format!(
            "https://api.github.com/repos/{}/issues/{}/comments",
            repo, issue_number
        );
        let headers = self.build_headers(token);
        let body = serde_json::json!({"body": comment}).to_string();

        let response = self.http.post(&url, headers, &body).await?;
        if !response.is_success() {
            return Err(Error::Other(format!(
                "Failed to add comment to {}/issues/{}: {}",
                repo, issue_number, response.body
            )));
        }
        Ok(())
    }

    pub async fn get_default_branch(&self, repo: &str) -> Result<String> {
        let token = self
            .config
            .token
            .as_ref()
            .ok_or_else(|| Error::config("GitHub token not configured"))?
            .expose();

        let url = format!("https://api.github.com/repos/{}", repo);
        let headers = self.build_headers(token);

        let response = self.http.get(&url, headers).await?;
        if !response.is_success() {
            return Err(Error::Other(format!(
                "Failed to fetch repo {}: {}",
                repo, response.body
            )));
        }
        let meta: RepoMeta = response.json()?;
        Ok(meta.default_branch)
    }

    /// Open a new pull request from `head` into `base`. Returns the html_url of
    /// the created PR (a real `/pull/<number>` link).
    pub async fn create_pr(
        &self,
        repo: &str,
        head: &str,
        base: &str,
        title: &str,
        body: &str,
    ) -> Result<String> {
        let token = self
            .config
            .token
            .as_ref()
            .ok_or_else(|| Error::config("GitHub token not configured"))?
            .expose();

        let url = format!("https://api.github.com/repos/{}/pulls", repo);
        let headers = self.build_headers(token);
        let payload = serde_json::json!({
            "title": title,
            "head": head,
            "base": base,
            "body": body,
        })
        .to_string();

        let response = self.http.post(&url, headers, &payload).await?;
        if !response.is_success() {
            return Err(Error::Other(format!(
                "Failed to create PR in {} ({} -> {}): {}",
                repo, head, base, response.body
            )));
        }
        let created: CreatedPr = response.json()?;
        Ok(created.html_url)
    }
}

#[async_trait]
impl<H: HttpClient> ScmProvider for GitHubClient<H> {
    fn name(&self) -> &str {
        "github"
    }

    fn is_enabled(&self) -> bool {
        self.is_enabled()
    }

    fn review_trigger(&self) -> &str {
        self.review_trigger()
    }

    fn allowed_bots(&self) -> &[String] {
        self.allowed_bots()
    }

    async fn get_pr_status(&self, project: &str, number: i64) -> Result<PrStatus> {
        GitHubClient::get_pr_status(self, project, number).await
    }

    async fn get_pr_info(&self, project: &str, number: i64) -> Result<PrInfo> {
        GitHubClient::get_pr_info(self, project, number).await
    }

    async fn get_pr_diff(&self, project: &str, number: i64) -> Result<String> {
        GitHubClient::get_pr_diff(self, project, number).await
    }

    async fn get_reviews(&self, project: &str, number: i64) -> Result<Vec<CodeReview>> {
        self.get_pr_reviews(project, number).await
    }

    async fn get_review_comments(&self, project: &str, number: i64) -> Result<Vec<ReviewComment>> {
        self.get_pr_review_comments(project, number).await
    }

    async fn get_pr_conversation_comments(
        &self,
        project: &str,
        number: i64,
    ) -> Result<Vec<ReviewComment>> {
        self.get_pr_issue_comments(project, number).await
    }

    async fn list_repos(&self, org_or_group: &str) -> Result<Vec<RemoteRepo>> {
        self.list_org_repos(org_or_group).await
    }

    async fn merge_pr(&self, project: &str, number: i64) -> Result<()> {
        GitHubClient::merge_pr(self, project, number).await
    }

    async fn close_pr(&self, project: &str, number: i64) -> Result<()> {
        GitHubClient::close_pr(self, project, number).await
    }

    async fn delete_branch(&self, project: &str, branch: &str) -> Result<()> {
        GitHubClient::delete_branch(self, project, branch).await
    }

    async fn post_review(
        &self,
        project: &str,
        number: i64,
        action: PostReviewAction,
        body: &str,
    ) -> Result<()> {
        GitHubClient::post_review(self, project, number, action, body).await
    }

    async fn post_review_with_comments(
        &self,
        project: &str,
        number: i64,
        action: PostReviewAction,
        body: &str,
        comments: &[InlineReviewComment],
    ) -> Result<()> {
        GitHubClient::post_review_with_comments(self, project, number, action, body, comments).await
    }

    async fn list_open_prs(&self, project: &str) -> Result<Vec<PrSummary>> {
        GitHubClient::list_open_prs(self, project).await
    }

    async fn get_pr_branch(&self, project: &str, number: i64) -> Result<String> {
        let info = self.get_pr_info(project, number).await?;
        info.head_branch.ok_or_else(|| {
            Error::Other(format!(
                "No head branch found for PR {} in {}",
                number, project
            ))
        })
    }

    fn pr_url_pattern(&self) -> &str {
        "https://github.com/%"
    }

    fn parse_pr_number(&self, url: &str) -> Option<i64> {
        GitHubClient::<H>::parse_pr_number(url)
    }

    async fn get_latest_release(&self, project: &str) -> Result<Option<ScmRelease>> {
        GitHubClient::get_latest_release(self, project).await
    }

    async fn create_release(
        &self,
        project: &str,
        tag: &str,
        name: &str,
        body: &str,
    ) -> Result<ScmRelease> {
        GitHubClient::create_release(self, project, tag, name, body).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use claudear_core::http::HttpResponse;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    /// Mock HTTP client for testing.
    #[expect(clippy::type_complexity)]
    pub struct MockHttpClient {
        responses: Mutex<HashMap<String, HttpResponse>>,
        requests: Mutex<Vec<(String, Vec<(String, String)>)>>,
    }

    impl MockHttpClient {
        pub fn new() -> Self {
            Self {
                responses: Mutex::new(HashMap::new()),
                requests: Mutex::new(Vec::new()),
            }
        }

        /// Add a mock response for a URL.
        pub fn mock_response(&self, url: impl Into<String>, status: u16, body: impl Into<String>) {
            let mut responses = self.responses.lock().unwrap();
            responses.insert(
                url.into(),
                HttpResponse {
                    status,
                    body: body.into(),
                },
            );
        }

        /// Get recorded requests.
        pub fn get_requests(&self) -> Vec<(String, Vec<(String, String)>)> {
            self.requests.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl HttpClient for MockHttpClient {
        async fn get(&self, url: &str, headers: Vec<(&str, String)>) -> Result<HttpResponse> {
            // Record the request
            let owned_headers: Vec<(String, String)> = headers
                .iter()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect();
            self.requests
                .lock()
                .unwrap()
                .push((url.to_string(), owned_headers));

            // Return mock response
            let responses = self.responses.lock().unwrap();
            if let Some(response) = responses.get(url) {
                Ok(HttpResponse {
                    status: response.status,
                    body: response.body.clone(),
                })
            } else {
                Ok(HttpResponse {
                    status: 404,
                    body: "Not found".to_string(),
                })
            }
        }

        async fn post(
            &self,
            url: &str,
            headers: Vec<(&str, String)>,
            _body: &str,
        ) -> Result<HttpResponse> {
            let owned_headers: Vec<(String, String)> = headers
                .iter()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect();
            self.requests
                .lock()
                .unwrap()
                .push((url.to_string(), owned_headers));

            let responses = self.responses.lock().unwrap();
            if let Some(response) = responses.get(url) {
                Ok(HttpResponse {
                    status: response.status,
                    body: response.body.clone(),
                })
            } else {
                Ok(HttpResponse {
                    status: 404,
                    body: "Not found".to_string(),
                })
            }
        }

        async fn put(
            &self,
            url: &str,
            headers: Vec<(&str, String)>,
            _body: &str,
        ) -> Result<HttpResponse> {
            let owned_headers: Vec<(String, String)> = headers
                .iter()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect();
            self.requests
                .lock()
                .unwrap()
                .push((url.to_string(), owned_headers));

            let responses = self.responses.lock().unwrap();
            if let Some(response) = responses.get(url) {
                Ok(HttpResponse {
                    status: response.status,
                    body: response.body.clone(),
                })
            } else {
                Ok(HttpResponse {
                    status: 404,
                    body: "Not found".to_string(),
                })
            }
        }

        async fn patch(
            &self,
            url: &str,
            headers: Vec<(&str, String)>,
            _body: &str,
        ) -> Result<HttpResponse> {
            let owned_headers: Vec<(String, String)> = headers
                .iter()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect();
            self.requests
                .lock()
                .unwrap()
                .push((url.to_string(), owned_headers));

            let responses = self.responses.lock().unwrap();
            if let Some(response) = responses.get(url) {
                Ok(HttpResponse {
                    status: response.status,
                    body: response.body.clone(),
                })
            } else {
                Ok(HttpResponse {
                    status: 404,
                    body: "Not found".to_string(),
                })
            }
        }

        async fn delete(&self, url: &str, headers: Vec<(&str, String)>) -> Result<HttpResponse> {
            let owned_headers: Vec<(String, String)> = headers
                .iter()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect();
            self.requests
                .lock()
                .unwrap()
                .push((url.to_string(), owned_headers));

            let responses = self.responses.lock().unwrap();
            if let Some(response) = responses.get(url) {
                Ok(HttpResponse {
                    status: response.status,
                    body: response.body.clone(),
                })
            } else {
                Ok(HttpResponse {
                    status: 404,
                    body: "Not found".to_string(),
                })
            }
        }
    }

    #[test]
    fn test_http_response_is_success() {
        let response = HttpResponse {
            status: 200,
            body: "{}".to_string(),
        };
        assert!(response.is_success());

        let response = HttpResponse {
            status: 201,
            body: "{}".to_string(),
        };
        assert!(response.is_success());

        let response = HttpResponse {
            status: 404,
            body: "{}".to_string(),
        };
        assert!(!response.is_success());

        let response = HttpResponse {
            status: 500,
            body: "{}".to_string(),
        };
        assert!(!response.is_success());
    }

    #[test]
    fn test_http_response_is_not_found() {
        let response = HttpResponse {
            status: 404,
            body: "{}".to_string(),
        };
        assert!(response.is_not_found());

        let response = HttpResponse {
            status: 200,
            body: "{}".to_string(),
        };
        assert!(!response.is_not_found());
    }

    #[test]
    fn test_http_response_json() {
        let response = HttpResponse {
            status: 200,
            body: r#"{"name": "test"}"#.to_string(),
        };
        let parsed: serde_json::Value = response.json().unwrap();
        assert_eq!(parsed["name"], "test");
    }

    #[test]
    fn test_http_response_json_error() {
        let response = HttpResponse {
            status: 200,
            body: "invalid json".to_string(),
        };
        let result: Result<serde_json::Value> = response.json();
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_get_pr_status_open() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/1",
            200,
            r#"{"number": 1, "state": "open", "merged": false}"#,
        );

        let config = GitHubConfig {
            token: Some("test_token".into()),
            poll_interval_ms: 60000,
            auto_resolve_on_merge: true,
            webhook_secret: None,
            review_trigger: "@claudear".to_string(),
            use_ssh: false,
            ..Default::default()
        };
        let client = GitHubClient::with_http_client(config, mock);

        let status = client.get_pr_status("owner/repo", 1).await.unwrap();
        assert_eq!(status, PrStatus::Open);
    }

    #[tokio::test]
    async fn test_create_pr_returns_real_url() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls",
            201,
            r#"{"html_url": "https://github.com/owner/repo/pull/123"}"#,
        );

        let config = GitHubConfig {
            token: Some("test_token".into()),
            review_trigger: "@claudear".to_string(),
            ..Default::default()
        };
        let client = GitHubClient::with_http_client(config, mock);

        let url = client
            .create_pr("owner/repo", "fix/branch", "main", "Fix: thing", "body")
            .await
            .unwrap();
        assert_eq!(url, "https://github.com/owner/repo/pull/123");
    }

    #[tokio::test]
    async fn test_get_default_branch() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo",
            200,
            r#"{"default_branch": "develop"}"#,
        );

        let config = GitHubConfig {
            token: Some("test_token".into()),
            review_trigger: "@claudear".to_string(),
            ..Default::default()
        };
        let client = GitHubClient::with_http_client(config, mock);

        let branch = client.get_default_branch("owner/repo").await.unwrap();
        assert_eq!(branch, "develop");
    }

    #[tokio::test]
    async fn test_create_pr_errors_on_failure() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls",
            422,
            r#"{"message": "Validation Failed"}"#,
        );

        let config = GitHubConfig {
            token: Some("test_token".into()),
            review_trigger: "@claudear".to_string(),
            ..Default::default()
        };
        let client = GitHubClient::with_http_client(config, mock);

        assert!(client
            .create_pr("owner/repo", "fix/branch", "main", "t", "b")
            .await
            .is_err());
    }

    #[tokio::test]
    async fn test_get_pr_status_merged() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/1",
            200,
            r#"{"number": 1, "state": "closed", "merged": true}"#,
        );

        let config = GitHubConfig {
            token: Some("test_token".into()),
            poll_interval_ms: 60000,
            auto_resolve_on_merge: true,
            webhook_secret: None,
            review_trigger: "@claudear".to_string(),
            use_ssh: false,
            ..Default::default()
        };
        let client = GitHubClient::with_http_client(config, mock);

        let status = client.get_pr_status("owner/repo", 1).await.unwrap();
        assert_eq!(status, PrStatus::Merged);
    }

    #[tokio::test]
    async fn test_get_pr_status_closed() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/1",
            200,
            r#"{"number": 1, "state": "closed", "merged": false}"#,
        );

        let config = GitHubConfig {
            token: Some("test_token".into()),
            poll_interval_ms: 60000,
            auto_resolve_on_merge: true,
            webhook_secret: None,
            review_trigger: "@claudear".to_string(),
            use_ssh: false,
            ..Default::default()
        };
        let client = GitHubClient::with_http_client(config, mock);

        let status = client.get_pr_status("owner/repo", 1).await.unwrap();
        assert_eq!(status, PrStatus::Closed);
    }

    #[tokio::test]
    async fn test_get_pr_status_not_found() {
        let mock = MockHttpClient::new();
        // No mock response means 404

        let config = GitHubConfig {
            token: Some("test_token".into()),
            poll_interval_ms: 60000,
            auto_resolve_on_merge: true,
            webhook_secret: None,
            review_trigger: "@claudear".to_string(),
            use_ssh: false,
            ..Default::default()
        };
        let client = GitHubClient::with_http_client(config, mock);

        let result = client.get_pr_status("owner/repo", 1).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("PR not found"));
    }

    #[tokio::test]
    async fn test_get_pr_status_api_error() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/1",
            500,
            "Internal Server Error",
        );

        let config = GitHubConfig {
            token: Some("test_token".into()),
            poll_interval_ms: 60000,
            auto_resolve_on_merge: true,
            webhook_secret: None,
            review_trigger: "@claudear".to_string(),
            use_ssh: false,
            ..Default::default()
        };
        let client = GitHubClient::with_http_client(config, mock);

        let result = client.get_pr_status("owner/repo", 1).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("GitHub API error"));
    }

    #[tokio::test]
    async fn test_get_pr_status_no_token() {
        let mock = MockHttpClient::new();
        let config = GitHubConfig::default(); // No token
        let client = GitHubClient::with_http_client(config, mock);

        let result = client.get_pr_status("owner/repo", 1).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("token"));
    }

    #[tokio::test]
    async fn test_get_pr_reviews() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/1/reviews",
            200,
            r#"[
                {"id": 1, "state": "APPROVED", "body": "LGTM", "user": {"id": 123, "login": "reviewer"}, "submitted_at": "2024-01-01T00:00:00Z"}
            ]"#,
        );

        let config = GitHubConfig {
            token: Some("test_token".into()),
            poll_interval_ms: 60000,
            auto_resolve_on_merge: true,
            webhook_secret: None,
            review_trigger: "@claudear".to_string(),
            use_ssh: false,
            ..Default::default()
        };
        let client = GitHubClient::with_http_client(config, mock);

        let reviews = client.get_pr_reviews("owner/repo", 1).await.unwrap();
        assert_eq!(reviews.len(), 1);
        assert_eq!(reviews[0].state, "APPROVED");
    }

    #[tokio::test]
    async fn test_get_new_reviews_reads_additional_pages() {
        let mock = MockHttpClient::new();

        let first_page_reviews = (1..=30)
            .map(|id| {
                format!(
                    r#"{{"id": {id}, "state": "APPROVED", "body": null, "user": {{"id": {id}, "login": "user-{id}", "type": "User"}}, "submitted_at": "2024-01-01T00:{:02}:00Z"}}"#,
                    id - 1
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/1/reviews",
            200,
            format!("[{}]", first_page_reviews),
        );
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/1/reviews?page=2",
            200,
            r#"[
                {"id": 31, "state": "CHANGES_REQUESTED", "body": "latest", "user": {"id": 131, "login": "latest-user", "type": "User"}, "submitted_at": "2024-01-01T01:00:00Z"}
            ]"#,
        );

        let config = GitHubConfig {
            token: Some("test_token".into()),
            poll_interval_ms: 60000,
            auto_resolve_on_merge: true,
            webhook_secret: None,
            review_trigger: "@claudear".to_string(),
            use_ssh: false,
            ..Default::default()
        };
        let client = GitHubClient::with_http_client(config, mock);

        let reviews = client
            .get_new_reviews("owner/repo", 1, Some("2024-01-01T00:30:00Z"))
            .await
            .unwrap();
        assert_eq!(reviews.len(), 1);
        assert_eq!(reviews[0].id, 31);

        let requests = client.http.get_requests();
        assert_eq!(requests.len(), 2);
        assert_eq!(
            requests[1].0,
            "https://api.github.com/repos/owner/repo/pulls/1/reviews?page=2"
        );
    }

    #[tokio::test]
    async fn test_get_pr_reviews_api_error() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/1/reviews",
            403,
            "Forbidden",
        );

        let config = GitHubConfig {
            token: Some("test_token".into()),
            poll_interval_ms: 60000,
            auto_resolve_on_merge: true,
            webhook_secret: None,
            review_trigger: "@claudear".to_string(),
            use_ssh: false,
            ..Default::default()
        };
        let client = GitHubClient::with_http_client(config, mock);

        let result = client.get_pr_reviews("owner/repo", 1).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_get_pr_review_comments() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/1/comments",
            200,
            r#"[
                {
                    "id": 1,
                    "path": "src/main.rs",
                    "body": "Please fix this",
                    "user": {"id": 123, "login": "reviewer"},
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-01-01T00:00:00Z",
                    "html_url": "https://github.com/test"
                }
            ]"#,
        );

        let config = GitHubConfig {
            token: Some("test_token".into()),
            poll_interval_ms: 60000,
            auto_resolve_on_merge: true,
            webhook_secret: None,
            review_trigger: "@claudear".to_string(),
            use_ssh: false,
            ..Default::default()
        };
        let client = GitHubClient::with_http_client(config, mock);

        let comments = client
            .get_pr_review_comments("owner/repo", 1)
            .await
            .unwrap();
        assert_eq!(comments.len(), 1);
        assert_eq!(comments[0].path, "src/main.rs");
    }

    #[tokio::test]
    async fn test_get_new_review_comments_reads_additional_pages() {
        let mock = MockHttpClient::new();

        let first_page_comments = (1..=30)
            .map(|id| {
                format!(
                    r#"{{"id": {id}, "path": "file-{id}.rs", "body": "older", "user": {{"id": {id}, "login": "user-{id}", "type": "User"}}, "created_at": "2024-01-01T00:{:02}:00Z", "updated_at": "2024-01-01T00:{:02}:00Z", "html_url": "url-{id}"}}"#,
                    id - 1,
                    id - 1
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/1/comments",
            200,
            format!("[{}]", first_page_comments),
        );
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/1/comments?page=2",
            200,
            r#"[
                {"id": 31, "path": "new.rs", "body": "new comment", "user": {"id": 231, "login": "latest-user", "type": "User"}, "created_at": "2024-01-01T01:00:00Z", "updated_at": "2024-01-01T01:00:00Z", "html_url": "url-31"}
            ]"#,
        );

        let config = GitHubConfig {
            token: Some("test_token".into()),
            poll_interval_ms: 60000,
            auto_resolve_on_merge: true,
            webhook_secret: None,
            review_trigger: "@claudear".to_string(),
            use_ssh: false,
            ..Default::default()
        };
        let client = GitHubClient::with_http_client(config, mock);

        let comments = client
            .get_new_review_comments("owner/repo", 1, Some("2024-01-01T00:30:00Z"))
            .await
            .unwrap();
        assert_eq!(comments.len(), 1);
        assert_eq!(comments[0].id, 31);

        let requests = client.http.get_requests();
        assert_eq!(requests.len(), 2);
        assert_eq!(
            requests[1].0,
            "https://api.github.com/repos/owner/repo/pulls/1/comments?page=2"
        );
    }

    #[tokio::test]
    async fn test_get_new_reviews_filters_by_time() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/1/reviews",
            200,
            r#"[
                {"id": 1, "state": "APPROVED", "body": null, "user": {"id": 123, "login": "old"}, "submitted_at": "2024-01-01T00:00:00Z"},
                {"id": 2, "state": "CHANGES_REQUESTED", "body": null, "user": {"id": 124, "login": "new"}, "submitted_at": "2024-01-02T00:00:00Z"}
            ]"#,
        );

        let config = GitHubConfig {
            token: Some("test_token".into()),
            poll_interval_ms: 60000,
            auto_resolve_on_merge: true,
            webhook_secret: None,
            review_trigger: "@claudear".to_string(),
            use_ssh: false,
            ..Default::default()
        };
        let client = GitHubClient::with_http_client(config, mock);

        // Filter to only reviews after 2024-01-01T12:00:00Z
        let reviews = client
            .get_new_reviews("owner/repo", 1, Some("2024-01-01T12:00:00Z"))
            .await
            .unwrap();
        assert_eq!(reviews.len(), 1);
        assert_eq!(reviews[0].user.login, "new");
    }

    #[tokio::test]
    async fn test_get_new_reviews_includes_equal_timestamp_boundary() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/1/reviews",
            200,
            r#"[
                {"id": 1, "state": "APPROVED", "body": null, "user": {"id": 123, "login": "old"}, "submitted_at": "2024-01-01T11:59:59Z"},
                {"id": 2, "state": "CHANGES_REQUESTED", "body": null, "user": {"id": 124, "login": "equal"}, "submitted_at": "2024-01-01T12:00:00Z"}
            ]"#,
        );

        let config = GitHubConfig {
            token: Some("test_token".into()),
            poll_interval_ms: 60000,
            auto_resolve_on_merge: true,
            webhook_secret: None,
            review_trigger: "@claudear".to_string(),
            use_ssh: false,
            ..Default::default()
        };
        let client = GitHubClient::with_http_client(config, mock);

        let reviews = client
            .get_new_reviews("owner/repo", 1, Some("2024-01-01T12:00:00Z"))
            .await
            .unwrap();
        assert_eq!(reviews.len(), 1);
        assert_eq!(reviews[0].user.login, "equal");
    }

    #[tokio::test]
    async fn test_get_new_reviews_filters_timezone_correctly() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/1/reviews",
            200,
            r#"[
                {"id": 1, "state": "APPROVED", "body": null, "user": {"id": 123, "login": "offset_before"}, "submitted_at": "2024-01-01T10:00:00+02:00"},
                {"id": 2, "state": "CHANGES_REQUESTED", "body": null, "user": {"id": 124, "login": "after"}, "submitted_at": "2024-01-01T08:45:00Z"}
            ]"#,
        );

        let config = GitHubConfig {
            token: Some("test_token".into()),
            poll_interval_ms: 60000,
            auto_resolve_on_merge: true,
            webhook_secret: None,
            review_trigger: "@claudear".to_string(),
            use_ssh: false,
            ..Default::default()
        };
        let client = GitHubClient::with_http_client(config, mock);

        let reviews = client
            .get_new_reviews("owner/repo", 1, Some("2024-01-01T08:30:00Z"))
            .await
            .unwrap();
        assert_eq!(reviews.len(), 1);
        assert_eq!(reviews[0].user.login, "after");
    }

    #[tokio::test]
    async fn test_get_new_reviews_no_filter() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/1/reviews",
            200,
            r#"[
                {"id": 1, "state": "APPROVED", "body": null, "user": {"id": 123, "login": "r1"}},
                {"id": 2, "state": "CHANGES_REQUESTED", "body": null, "user": {"id": 124, "login": "r2"}}
            ]"#,
        );

        let config = GitHubConfig {
            token: Some("test_token".into()),
            poll_interval_ms: 60000,
            auto_resolve_on_merge: true,
            webhook_secret: None,
            review_trigger: "@claudear".to_string(),
            use_ssh: false,
            ..Default::default()
        };
        let client = GitHubClient::with_http_client(config, mock);

        let reviews = client.get_new_reviews("owner/repo", 1, None).await.unwrap();
        assert_eq!(reviews.len(), 2);
    }

    #[tokio::test]
    async fn test_get_new_review_comments_filters_by_time() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/1/comments",
            200,
            r#"[
                {"id": 1, "path": "a.rs", "body": "old", "user": {"id": 1, "login": "u"}, "created_at": "2024-01-01T00:00:00Z", "updated_at": "2024-01-01T00:00:00Z", "html_url": "url"},
                {"id": 2, "path": "b.rs", "body": "new", "user": {"id": 1, "login": "u"}, "created_at": "2024-01-02T00:00:00Z", "updated_at": "2024-01-02T00:00:00Z", "html_url": "url"}
            ]"#,
        );

        let config = GitHubConfig {
            token: Some("test_token".into()),
            poll_interval_ms: 60000,
            auto_resolve_on_merge: true,
            webhook_secret: None,
            review_trigger: "@claudear".to_string(),
            use_ssh: false,
            ..Default::default()
        };
        let client = GitHubClient::with_http_client(config, mock);

        let comments = client
            .get_new_review_comments("owner/repo", 1, Some("2024-01-01T12:00:00Z"))
            .await
            .unwrap();
        assert_eq!(comments.len(), 1);
        assert_eq!(comments[0].path, "b.rs");
    }

    #[tokio::test]
    async fn test_get_new_review_comments_include_equal_timestamp_boundary() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/1/comments",
            200,
            r#"[
                {"id": 1, "path": "a.rs", "body": "old", "user": {"id": 1, "login": "u"}, "created_at": "2024-01-01T11:59:59Z", "updated_at": "2024-01-01T11:59:59Z", "html_url": "url"},
                {"id": 2, "path": "b.rs", "body": "equal", "user": {"id": 1, "login": "u"}, "created_at": "2024-01-01T12:00:00Z", "updated_at": "2024-01-01T12:00:00Z", "html_url": "url"}
            ]"#,
        );

        let config = GitHubConfig {
            token: Some("test_token".into()),
            poll_interval_ms: 60000,
            auto_resolve_on_merge: true,
            webhook_secret: None,
            review_trigger: "@claudear".to_string(),
            use_ssh: false,
            ..Default::default()
        };
        let client = GitHubClient::with_http_client(config, mock);

        let comments = client
            .get_new_review_comments("owner/repo", 1, Some("2024-01-01T12:00:00Z"))
            .await
            .unwrap();
        assert_eq!(comments.len(), 1);
        assert_eq!(comments[0].path, "b.rs");
    }

    #[tokio::test]
    async fn test_get_new_review_comments_filter_timezone_correctly() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/1/comments",
            200,
            r#"[
                {"id": 1, "path": "offset_before.rs", "body": "old", "user": {"id": 1, "login": "u"}, "created_at": "2024-01-01T10:00:00+02:00", "updated_at": "2024-01-01T10:00:00+02:00", "html_url": "url"},
                {"id": 2, "path": "after.rs", "body": "new", "user": {"id": 1, "login": "u"}, "created_at": "2024-01-01T08:45:00Z", "updated_at": "2024-01-01T08:45:00Z", "html_url": "url"}
            ]"#,
        );

        let config = GitHubConfig {
            token: Some("test_token".into()),
            poll_interval_ms: 60000,
            auto_resolve_on_merge: true,
            webhook_secret: None,
            review_trigger: "@claudear".to_string(),
            use_ssh: false,
            ..Default::default()
        };
        let client = GitHubClient::with_http_client(config, mock);

        let comments = client
            .get_new_review_comments("owner/repo", 1, Some("2024-01-01T08:30:00Z"))
            .await
            .unwrap();
        assert_eq!(comments.len(), 1);
        assert_eq!(comments[0].path, "after.rs");
    }

    #[tokio::test]
    async fn test_request_includes_auth_headers() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/1",
            200,
            r#"{"number": 1, "state": "open", "merged": false}"#,
        );

        let config = GitHubConfig {
            token: Some("my_secret_token".into()),
            poll_interval_ms: 60000,
            auto_resolve_on_merge: true,
            webhook_secret: None,
            review_trigger: "@claudear".to_string(),
            use_ssh: false,
            ..Default::default()
        };
        let client = GitHubClient::with_http_client(config, mock);

        let _ = client.get_pr_status("owner/repo", 1).await;

        let http = &client.http;
        let requests = http.get_requests();
        assert_eq!(requests.len(), 1);

        let (_, headers) = &requests[0];
        let auth_header = headers.iter().find(|(k, _)| k == "Authorization");
        assert!(auth_header.is_some());
        assert!(auth_header.unwrap().1.contains("my_secret_token"));
    }

    #[tokio::test]
    async fn test_review_watcher_check_for_reviews_with_mock() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/1/reviews",
            200,
            r#"[
                {"id": 1, "state": "CHANGES_REQUESTED", "body": "Fix this", "user": {"id": 123, "login": "reviewer", "type": "User"}, "submitted_at": "2024-01-01T00:00:00Z"}
            ]"#,
        );
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/1/comments",
            200,
            "[]",
        );

        let config = GitHubConfig {
            token: Some("test_token".into()),
            poll_interval_ms: 60000,
            auto_resolve_on_merge: true,
            webhook_secret: None,
            review_trigger: "@claudear".to_string(),
            use_ssh: false,
            ..Default::default()
        };
        let client = GitHubClient::with_http_client(config, mock);
        let provider: Arc<dyn ScmProvider> = Arc::new(client);
        let watcher = crate::scm::ReviewWatcher::new(provider);

        let state = PrReviewState::new("url", "owner/repo", 1, "issue", "linear");
        watcher.watch_pr(state);

        let events = watcher.check_for_reviews().await.unwrap();
        assert_eq!(events.len(), 1);
        match &events[0] {
            ReviewEvent::ReviewSubmitted { review, .. } => {
                assert_eq!(review.state, "CHANGES_REQUESTED");
            }
            _ => panic!("Expected ReviewSubmitted event"),
        }
    }

    #[tokio::test]
    async fn test_review_watcher_keeps_review_events_when_comment_fetch_fails() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/1/reviews",
            200,
            r#"[
                {"id": 1, "state": "CHANGES_REQUESTED", "body": "Fix this", "user": {"id": 123, "login": "reviewer", "type": "User"}, "submitted_at": "2024-01-01T00:00:00Z"}
            ]"#,
        );
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/1/comments",
            500,
            "Internal Server Error",
        );

        let config = GitHubConfig {
            token: Some("test_token".into()),
            poll_interval_ms: 60000,
            auto_resolve_on_merge: true,
            webhook_secret: None,
            review_trigger: "@claudear".to_string(),
            use_ssh: false,
            ..Default::default()
        };
        let client = GitHubClient::with_http_client(config, mock);
        let provider: Arc<dyn ScmProvider> = Arc::new(client);
        let watcher = crate::scm::ReviewWatcher::new(provider);

        watcher.watch_pr(PrReviewState::new(
            "url",
            "owner/repo",
            1,
            "issue",
            "linear",
        ));

        let events = watcher.check_for_reviews().await.unwrap();
        assert_eq!(events.len(), 1);
        match &events[0] {
            ReviewEvent::ReviewSubmitted { review, .. } => {
                assert_eq!(review.state, "CHANGES_REQUESTED");
            }
            _ => panic!("Expected ReviewSubmitted event"),
        }

        let updated_state = watcher.get_state("url").unwrap();
        assert_eq!(updated_state.last_review_id, Some(1));
    }

    #[tokio::test]
    async fn test_review_watcher_skips_bot_reviews() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/1/reviews",
            200,
            r#"[
                {"id": 1, "state": "APPROVED", "body": null, "user": {"id": 123, "login": "bot", "type": "Bot"}, "submitted_at": "2024-01-01T00:00:00Z"}
            ]"#,
        );
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/1/comments",
            200,
            "[]",
        );

        let config = GitHubConfig {
            token: Some("test_token".into()),
            poll_interval_ms: 60000,
            auto_resolve_on_merge: true,
            webhook_secret: None,
            review_trigger: "@claudear".to_string(),
            use_ssh: false,
            ..Default::default()
        };
        let client = GitHubClient::with_http_client(config, mock);
        let provider: Arc<dyn ScmProvider> = Arc::new(client);
        let watcher = crate::scm::ReviewWatcher::new(provider);

        let state = PrReviewState::new("url", "owner/repo", 1, "issue", "linear");
        watcher.watch_pr(state);

        let events = watcher.check_for_reviews().await.unwrap();
        assert!(events.is_empty()); // Bot reviews are skipped
    }

    #[tokio::test]
    async fn test_review_watcher_skips_pending_reviews() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/1/reviews",
            200,
            r#"[
                {"id": 1, "state": "PENDING", "body": null, "user": {"id": 123, "login": "user", "type": "User"}, "submitted_at": "2024-01-01T00:00:00Z"}
            ]"#,
        );
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/1/comments",
            200,
            "[]",
        );

        let config = GitHubConfig {
            token: Some("test_token".into()),
            poll_interval_ms: 60000,
            auto_resolve_on_merge: true,
            webhook_secret: None,
            review_trigger: "@claudear".to_string(),
            use_ssh: false,
            ..Default::default()
        };
        let client = GitHubClient::with_http_client(config, mock);
        let provider: Arc<dyn ScmProvider> = Arc::new(client);
        let watcher = crate::scm::ReviewWatcher::new(provider);

        let state = PrReviewState::new("url", "owner/repo", 1, "issue", "linear");
        watcher.watch_pr(state);

        let events = watcher.check_for_reviews().await.unwrap();
        assert!(events.is_empty()); // Pending reviews are skipped
    }

    #[tokio::test]
    async fn test_review_watcher_tracks_review_state() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/1/reviews",
            200,
            r#"[
                {"id": 5, "state": "APPROVED", "body": null, "user": {"id": 123, "login": "user", "type": "User"}, "submitted_at": "2024-01-01T00:00:00Z"}
            ]"#,
        );
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/1/comments",
            200,
            "[]",
        );

        let config = GitHubConfig {
            token: Some("test_token".into()),
            poll_interval_ms: 60000,
            auto_resolve_on_merge: true,
            webhook_secret: None,
            review_trigger: "@claudear".to_string(),
            use_ssh: false,
            ..Default::default()
        };
        let client = GitHubClient::with_http_client(config, mock);
        let provider: Arc<dyn ScmProvider> = Arc::new(client);
        let watcher = crate::scm::ReviewWatcher::new(provider);

        let state = PrReviewState::new("url", "owner/repo", 1, "issue", "linear");
        watcher.watch_pr(state);

        let _ = watcher.check_for_reviews().await.unwrap();

        // Check that the state was updated with the latest review ID
        let updated_state = watcher.get_state("url").unwrap();
        assert_eq!(updated_state.last_review_id, Some(5));
    }

    #[tokio::test]
    async fn test_review_watcher_tracks_max_review_cursor_with_descending_reviews() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/1/reviews",
            200,
            r#"[
                {"id": 9, "state": "CHANGES_REQUESTED", "body": "latest", "user": {"id": 123, "login": "user-a", "type": "User"}, "submitted_at": "2024-01-02T00:00:00Z"},
                {"id": 5, "state": "COMMENTED", "body": "older", "user": {"id": 124, "login": "user-b", "type": "User"}, "submitted_at": "2024-01-01T00:00:00Z"}
            ]"#,
        );
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/1/comments",
            200,
            "[]",
        );

        let config = GitHubConfig {
            token: Some("test_token".into()),
            poll_interval_ms: 60000,
            auto_resolve_on_merge: true,
            webhook_secret: None,
            review_trigger: "@claudear".to_string(),
            use_ssh: false,
            ..Default::default()
        };
        let client = GitHubClient::with_http_client(config, mock);
        let provider: Arc<dyn ScmProvider> = Arc::new(client);
        let watcher = crate::scm::ReviewWatcher::new(provider);

        watcher.watch_pr(PrReviewState::new(
            "url",
            "owner/repo",
            1,
            "issue",
            "linear",
        ));

        let events = watcher.check_for_reviews().await.unwrap();
        assert_eq!(events.len(), 2);

        let updated_state = watcher.get_state("url").unwrap();
        assert_eq!(updated_state.last_review_id, Some(9));
        assert_eq!(
            updated_state.last_review_time.as_deref(),
            Some("2024-01-02T00:00:00Z")
        );
    }

    #[tokio::test]
    async fn test_review_watcher_no_duplicate_events_on_descending_reviews() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/1/reviews",
            200,
            r#"[
                {"id": 9, "state": "CHANGES_REQUESTED", "body": "latest", "user": {"id": 123, "login": "user-a", "type": "User"}, "submitted_at": "2024-01-02T00:00:00Z"},
                {"id": 5, "state": "COMMENTED", "body": "older", "user": {"id": 124, "login": "user-b", "type": "User"}, "submitted_at": "2024-01-01T00:00:00Z"}
            ]"#,
        );
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/1/comments",
            200,
            "[]",
        );

        let config = GitHubConfig {
            token: Some("test_token".into()),
            poll_interval_ms: 60000,
            auto_resolve_on_merge: true,
            webhook_secret: None,
            review_trigger: "@claudear".to_string(),
            use_ssh: false,
            ..Default::default()
        };
        let client = GitHubClient::with_http_client(config, mock);
        let provider: Arc<dyn ScmProvider> = Arc::new(client);
        let watcher = crate::scm::ReviewWatcher::new(provider);

        watcher.watch_pr(PrReviewState::new(
            "url",
            "owner/repo",
            1,
            "issue",
            "linear",
        ));

        let first = watcher.check_for_reviews().await.unwrap();
        assert_eq!(first.len(), 2);

        let second = watcher.check_for_reviews().await.unwrap();
        assert!(
            second.is_empty(),
            "second poll should not emit duplicate review events"
        );
    }

    #[tokio::test]
    async fn test_review_watcher_comments_added_event() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/1/reviews",
            200,
            "[]",
        );
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/1/comments",
            200,
            r#"[
                {"id": 1, "path": "file.rs", "body": "@claudear Fix this", "user": {"id": 123, "login": "user", "type": "User"}, "created_at": "2024-01-01T00:00:00Z", "updated_at": "2024-01-01T00:00:00Z", "html_url": "url"}
            ]"#,
        );

        let config = GitHubConfig {
            token: Some("test_token".into()),
            poll_interval_ms: 60000,
            auto_resolve_on_merge: true,
            webhook_secret: None,
            review_trigger: "@claudear".to_string(),
            use_ssh: false,
            ..Default::default()
        };
        let client = GitHubClient::with_http_client(config, mock);
        let provider: Arc<dyn ScmProvider> = Arc::new(client);
        let watcher = crate::scm::ReviewWatcher::new(provider);

        let state = PrReviewState::new("url", "owner/repo", 1, "issue", "linear");
        watcher.watch_pr(state);

        let events = watcher.check_for_reviews().await.unwrap();
        assert_eq!(events.len(), 1);
        match &events[0] {
            ReviewEvent::CommentsAdded { comments, .. } => {
                assert_eq!(comments.len(), 1);
                assert_eq!(comments[0].body, "@claudear Fix this");
            }
            _ => panic!("Expected CommentsAdded event"),
        }
    }

    #[tokio::test]
    async fn test_review_watcher_advances_comment_cursor_without_trigger() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/1/reviews",
            200,
            "[]",
        );
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/1/comments",
            200,
            r#"[
                {"id": 1, "path": "file.rs", "body": "plain comment", "user": {"id": 123, "login": "user", "type": "User"}, "created_at": "2024-01-01T00:00:00Z", "updated_at": "2024-01-01T00:00:00Z", "html_url": "url"}
            ]"#,
        );

        let config = GitHubConfig {
            token: Some("test_token".into()),
            poll_interval_ms: 60000,
            auto_resolve_on_merge: true,
            webhook_secret: None,
            review_trigger: "@claudear".to_string(),
            use_ssh: false,
            ..Default::default()
        };
        let client = GitHubClient::with_http_client(config, mock);
        let provider: Arc<dyn ScmProvider> = Arc::new(client);
        let watcher = crate::scm::ReviewWatcher::new(provider);

        watcher.watch_pr(PrReviewState::new(
            "url",
            "owner/repo",
            1,
            "issue",
            "linear",
        ));

        let events = watcher.check_for_reviews().await.unwrap();
        assert!(events.is_empty());

        let updated_state = watcher.get_state("url").unwrap();
        assert_eq!(updated_state.last_comment_id, Some(1));
        assert_eq!(
            updated_state.last_comment_time.as_deref(),
            Some("2024-01-01T00:00:00Z")
        );
    }

    #[tokio::test]
    async fn test_review_watcher_skips_already_processed_reviews() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/1/reviews",
            200,
            r#"[
                {"id": 1, "state": "APPROVED", "body": null, "user": {"id": 123, "login": "user", "type": "User"}, "submitted_at": "2024-01-01T00:00:00Z"}
            ]"#,
        );
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/1/comments",
            200,
            "[]",
        );

        let config = GitHubConfig {
            token: Some("test_token".into()),
            poll_interval_ms: 60000,
            auto_resolve_on_merge: true,
            webhook_secret: None,
            review_trigger: "@claudear".to_string(),
            use_ssh: false,
            ..Default::default()
        };
        let client = GitHubClient::with_http_client(config, mock);
        let provider: Arc<dyn ScmProvider> = Arc::new(client);
        let watcher = crate::scm::ReviewWatcher::new(provider);

        // Pre-set the last_review_id to 1, so the review should be skipped
        let mut state = PrReviewState::new("url", "owner/repo", 1, "issue", "linear");
        state.last_review_id = Some(1);
        watcher.watch_pr(state);

        let events = watcher.check_for_reviews().await.unwrap();
        assert!(events.is_empty()); // Review was already processed
    }

    #[test]
    fn test_client_not_enabled_without_token() {
        let config = GitHubConfig::default();
        let client = GitHubClient::new(config);
        assert!(!client.is_enabled());
    }

    #[test]
    fn test_client_enabled_with_token() {
        let config = GitHubConfig {
            token: Some("test_token".into()),
            poll_interval_ms: 60000,
            auto_resolve_on_merge: true,
            webhook_secret: None,
            review_trigger: "@claudear".to_string(),
            use_ssh: false,
            ..Default::default()
        };
        let client = GitHubClient::new(config);
        assert!(client.is_enabled());
    }

    #[test]
    fn test_pr_review_state_new() {
        let state = PrReviewState::new(
            "https://github.com/owner/repo/pull/1",
            "owner/repo",
            1,
            "issue-123",
            "linear",
        );

        assert_eq!(state.pr_url, "https://github.com/owner/repo/pull/1");
        assert_eq!(state.repo, "owner/repo");
        assert_eq!(state.pr_number, 1);
        assert!(state.is_active);
        assert!(state.last_review_id.is_none());
    }

    #[test]
    fn test_review_event_pr_url() {
        let review = PrReview {
            id: 1,
            state: "APPROVED".to_string(),
            body: Some("LGTM".to_string()),
            user: GitHubUser {
                id: 123,
                login: "reviewer".to_string(),
                user_type: Some("User".to_string()),
            },
            submitted_at: Some("2024-01-01T00:00:00Z".to_string()),
            html_url: Some("https://github.com/test".to_string()),
        };

        let event = ReviewEvent::ReviewSubmitted {
            pr_url: "https://github.com/test/pull/1".to_string(),
            repo: "test/test".to_string(),
            pr_number: 1,
            review,
            inline_comments: vec![],
        };

        assert_eq!(event.pr_url(), "https://github.com/test/pull/1");
    }

    #[test]
    fn test_review_event_requires_action() {
        let approved_review = PrReview {
            id: 1,
            state: "APPROVED".to_string(),
            body: None,
            user: GitHubUser {
                id: 123,
                login: "reviewer".to_string(),
                user_type: Some("User".to_string()),
            },
            submitted_at: None,
            html_url: None,
        };

        let changes_requested = PrReview {
            id: 2,
            state: "CHANGES_REQUESTED".to_string(),
            body: Some("Please fix this".to_string()),
            user: GitHubUser {
                id: 123,
                login: "reviewer".to_string(),
                user_type: Some("User".to_string()),
            },
            submitted_at: None,
            html_url: None,
        };

        let approved_event = ReviewEvent::ReviewSubmitted {
            pr_url: "url".to_string(),
            repo: "repo".to_string(),
            pr_number: 1,
            review: approved_review,
            inline_comments: vec![],
        };

        let changes_event = ReviewEvent::ReviewSubmitted {
            pr_url: "url".to_string(),
            repo: "repo".to_string(),
            pr_number: 1,
            review: changes_requested,
            inline_comments: vec![],
        };

        // Approved reviews don't require action
        assert!(!approved_event.requires_action());
        // Changes requested do require action
        assert!(changes_event.requires_action());
    }

    #[test]
    fn test_review_event_feedback_summary() {
        let review = PrReview {
            id: 1,
            state: "CHANGES_REQUESTED".to_string(),
            body: Some("Please add tests".to_string()),
            user: GitHubUser {
                id: 123,
                login: "reviewer".to_string(),
                user_type: Some("User".to_string()),
            },
            submitted_at: None,
            html_url: None,
        };

        let event = ReviewEvent::ReviewSubmitted {
            pr_url: "url".to_string(),
            repo: "repo".to_string(),
            pr_number: 1,
            review,
            inline_comments: vec![],
        };

        let summary = event.get_feedback_summary();
        assert!(summary.contains("@reviewer"));
        assert!(summary.contains("CHANGES_REQUESTED"));
        assert!(summary.contains("Please add tests"));
    }

    #[test]
    fn test_review_event_comments_summary() {
        let comments = vec![PrReviewComment {
            id: 1,
            path: "src/main.rs".to_string(),
            position: None,
            original_position: None,
            body: "This should use a match statement".to_string(),
            user: GitHubUser {
                id: 123,
                login: "reviewer".to_string(),
                user_type: Some("User".to_string()),
            },
            created_at: "2024-01-01T00:00:00Z".to_string(),
            updated_at: "2024-01-01T00:00:00Z".to_string(),
            html_url: "https://github.com/test".to_string(),
            pull_request_review_id: Some(1),
            start_line: None,
            line: Some(42),
            side: Some("RIGHT".to_string()),
        }];

        let event = ReviewEvent::CommentsAdded {
            pr_url: "url".to_string(),
            repo: "repo".to_string(),
            pr_number: 1,
            comments,
        };

        let summary = event.get_feedback_summary();
        assert!(summary.contains("@reviewer"));
        assert!(summary.contains("src/main.rs"));
        assert!(summary.contains("line 42"));
        assert!(summary.contains("match statement"));
    }

    #[test]
    fn test_review_watcher_watch_unwatch() {
        let config = GitHubConfig {
            token: Some("test".into()),
            poll_interval_ms: 60000,
            auto_resolve_on_merge: true,
            webhook_secret: None,
            review_trigger: "@claudear".to_string(),
            use_ssh: false,
            ..Default::default()
        };
        let client = GitHubClient::new(config);
        let provider: Arc<dyn ScmProvider> = Arc::new(client);
        let watcher = crate::scm::ReviewWatcher::new(provider);

        let state = PrReviewState::new("url", "repo", 1, "issue", "linear");
        watcher.watch_pr(state);

        let retrieved = watcher.get_state("url");
        assert!(retrieved.is_some());
        assert!(retrieved.unwrap().is_active);

        watcher.unwatch_pr("url");
        let after_unwatch = watcher.get_state("url");
        assert!(after_unwatch.is_none());
    }

    #[test]
    fn test_review_watcher_watch_pr_preserves_existing_cursors() {
        let config = GitHubConfig {
            token: Some("test".into()),
            poll_interval_ms: 60000,
            auto_resolve_on_merge: true,
            webhook_secret: None,
            review_trigger: "@claudear".to_string(),
            use_ssh: false,
            ..Default::default()
        };
        let client = GitHubClient::new(config);
        let provider: Arc<dyn ScmProvider> = Arc::new(client);
        let watcher = crate::scm::ReviewWatcher::new(provider);

        let mut existing = PrReviewState::new("url", "repo", 1, "issue", "linear");
        existing.last_review_id = Some(42);
        existing.last_review_time = Some("2024-01-01T00:00:00Z".to_string());
        existing.last_comment_id = Some(64);
        existing.last_comment_time = Some("2024-01-01T01:00:00Z".to_string());
        watcher.watch_pr(existing);

        // Re-registering the same PR should not reset cursor state.
        watcher.watch_pr(PrReviewState::new("url", "repo", 1, "issue", "linear"));

        let state = watcher.get_state("url").unwrap();
        assert_eq!(state.last_review_id, Some(42));
        assert_eq!(
            state.last_review_time.as_deref(),
            Some("2024-01-01T00:00:00Z")
        );
        assert_eq!(state.last_comment_id, Some(64));
        assert_eq!(
            state.last_comment_time.as_deref(),
            Some("2024-01-01T01:00:00Z")
        );
        assert!(state.is_active);
    }

    #[tokio::test]
    async fn test_review_watcher_rewatch_does_not_replay_processed_reviews() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/1/reviews",
            200,
            r#"[
                {"id": 1, "state": "CHANGES_REQUESTED", "body": "Fix this", "user": {"id": 123, "login": "reviewer", "type": "User"}, "submitted_at": "2024-01-01T00:00:00Z"}
            ]"#,
        );
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/1/comments",
            200,
            "[]",
        );

        let config = GitHubConfig {
            token: Some("test_token".into()),
            poll_interval_ms: 60000,
            auto_resolve_on_merge: true,
            webhook_secret: None,
            review_trigger: "@claudear".to_string(),
            use_ssh: false,
            ..Default::default()
        };
        let client = GitHubClient::with_http_client(config, mock);
        let provider: Arc<dyn ScmProvider> = Arc::new(client);
        let watcher = crate::scm::ReviewWatcher::new(provider);

        watcher.watch_pr(PrReviewState::new(
            "url",
            "owner/repo",
            1,
            "issue",
            "linear",
        ));

        let first = watcher.check_for_reviews().await.unwrap();
        assert_eq!(first.len(), 1);

        // Simulate re-registration from a later successful rerun on the same PR URL.
        watcher.watch_pr(PrReviewState::new(
            "url",
            "owner/repo",
            1,
            "issue",
            "linear",
        ));
        let second = watcher.check_for_reviews().await.unwrap();
        assert!(
            second.is_empty(),
            "re-watching should preserve cursor state and avoid replay"
        );
    }

    #[test]
    fn test_review_watcher_active_states() {
        let config = GitHubConfig {
            token: Some("test".into()),
            poll_interval_ms: 60000,
            auto_resolve_on_merge: true,
            webhook_secret: None,
            review_trigger: "@claudear".to_string(),
            use_ssh: false,
            ..Default::default()
        };
        let client = GitHubClient::new(config);
        let provider: Arc<dyn ScmProvider> = Arc::new(client);
        let watcher = crate::scm::ReviewWatcher::new(provider);

        let state1 = PrReviewState::new("url1", "repo", 1, "issue1", "linear");
        let state2 = PrReviewState::new("url2", "repo", 2, "issue2", "linear");

        watcher.watch_pr(state1);
        watcher.watch_pr(state2);
        watcher.unwatch_pr("url1");

        let active = watcher.get_active_states();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].pr_url, "url2");
    }

    #[test]
    fn test_review_watcher_load_states() {
        let config = GitHubConfig {
            token: Some("test".into()),
            poll_interval_ms: 60000,
            auto_resolve_on_merge: true,
            webhook_secret: None,
            review_trigger: "@claudear".to_string(),
            use_ssh: false,
            ..Default::default()
        };
        let client = GitHubClient::new(config);
        let provider: Arc<dyn ScmProvider> = Arc::new(client);
        let watcher = crate::scm::ReviewWatcher::new(provider);

        let mut inactive_state = PrReviewState::new("url1", "repo", 1, "issue1", "linear");
        inactive_state.is_active = false;

        let active_state = PrReviewState::new("url2", "repo", 2, "issue2", "linear");

        watcher.load_states(vec![inactive_state, active_state]);

        // Only active states should be loaded
        let all = watcher.get_all_states();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].pr_url, "url2");
    }

    #[test]
    fn test_pr_status_equality() {
        assert_eq!(PrStatus::Open, PrStatus::Open);
        assert_eq!(PrStatus::Merged, PrStatus::Merged);
        assert_eq!(PrStatus::Closed, PrStatus::Closed);
        assert_ne!(PrStatus::Open, PrStatus::Merged);
    }

    #[test]
    fn test_client_token_accessor() {
        let config = GitHubConfig {
            token: Some("my_token".into()),
            poll_interval_ms: 60000,
            auto_resolve_on_merge: true,
            webhook_secret: None,
            review_trigger: "@claudear".to_string(),
            use_ssh: false,
            ..Default::default()
        };
        let client = GitHubClient::new(config);
        assert_eq!(client.token(), Some("my_token"));
    }

    #[test]
    fn test_client_token_none() {
        let config = GitHubConfig::default();
        let client = GitHubClient::new(config);
        assert!(client.token().is_none());
    }

    #[test]
    fn test_review_event_empty_comments() {
        let event = ReviewEvent::CommentsAdded {
            pr_url: "url".to_string(),
            repo: "repo".to_string(),
            pr_number: 1,
            comments: vec![],
        };
        assert!(!event.requires_action());
    }

    #[test]
    fn test_review_event_commented_state() {
        let review = PrReview {
            id: 1,
            state: "COMMENTED".to_string(),
            body: Some("Some feedback".to_string()),
            user: GitHubUser {
                id: 123,
                login: "reviewer".to_string(),
                user_type: Some("User".to_string()),
            },
            submitted_at: None,
            html_url: None,
        };

        let event = ReviewEvent::ReviewSubmitted {
            pr_url: "url".to_string(),
            repo: "repo".to_string(),
            pr_number: 1,
            review,
            inline_comments: vec![],
        };

        assert!(event.requires_action());
    }

    #[test]
    fn test_review_event_dismissed() {
        let review = PrReview {
            id: 1,
            state: "DISMISSED".to_string(),
            body: None,
            user: GitHubUser {
                id: 123,
                login: "reviewer".to_string(),
                user_type: Some("User".to_string()),
            },
            submitted_at: None,
            html_url: None,
        };

        let event = ReviewEvent::ReviewSubmitted {
            pr_url: "url".to_string(),
            repo: "repo".to_string(),
            pr_number: 1,
            review,
            inline_comments: vec![],
        };

        assert!(!event.requires_action());
    }

    #[test]
    fn test_pr_review_state_update_review_tracking() {
        let mut state = PrReviewState::new("url", "repo", 1, "issue", "linear");
        assert!(state.last_review_id.is_none());
        assert!(state.last_review_time.is_none());

        state.last_review_id = Some(123);
        state.last_review_time = Some("2024-01-01T00:00:00Z".to_string());

        assert_eq!(state.last_review_id, Some(123));
        assert_eq!(
            state.last_review_time,
            Some("2024-01-01T00:00:00Z".to_string())
        );
    }

    #[test]
    fn test_pr_review_state_update_comment_tracking() {
        let mut state = PrReviewState::new("url", "repo", 1, "issue", "linear");
        assert!(state.last_comment_id.is_none());

        state.last_comment_id = Some(456);
        state.last_comment_time = Some("2024-01-01T01:00:00Z".to_string());

        assert_eq!(state.last_comment_id, Some(456));
    }

    #[test]
    fn test_pr_status_update_fields() {
        let update = PrStatusUpdate {
            source: "linear".to_string(),
            issue_id: "issue-1".to_string(),
            short_id: "LIN-1".to_string(),
            pr_url: "https://github.com/org/repo/pull/1".to_string(),
            new_status: PrStatus::Merged,
            should_resolve: true,
            regression_watch_id: None,
        };

        assert_eq!(update.source, "linear");
        assert_eq!(update.new_status, PrStatus::Merged);
        assert!(update.should_resolve);
        assert!(update.regression_watch_id.is_none());
    }

    #[test]
    fn test_github_user_serialization() {
        let user = GitHubUser {
            id: 123,
            login: "testuser".to_string(),
            user_type: Some("User".to_string()),
        };
        let json = serde_json::to_string(&user).unwrap();
        assert!(json.contains("testuser"));
        assert!(json.contains("123"));
    }

    #[test]
    fn test_pr_review_serialization() {
        let review = PrReview {
            id: 1,
            state: "APPROVED".to_string(),
            body: Some("LGTM".to_string()),
            user: GitHubUser {
                id: 123,
                login: "reviewer".to_string(),
                user_type: None,
            },
            submitted_at: Some("2024-01-01T00:00:00Z".to_string()),
            html_url: Some("https://github.com/review".to_string()),
        };
        let json = serde_json::to_string(&review).unwrap();
        assert!(json.contains("APPROVED"));
        assert!(json.contains("LGTM"));
    }

    #[test]
    fn test_pr_review_comment_serialization() {
        let comment = PrReviewComment {
            id: 1,
            path: "src/main.rs".to_string(),
            position: Some(10),
            original_position: None,
            body: "Please fix this".to_string(),
            user: GitHubUser {
                id: 123,
                login: "commenter".to_string(),
                user_type: Some("User".to_string()),
            },
            created_at: "2024-01-01T00:00:00Z".to_string(),
            updated_at: "2024-01-01T00:00:00Z".to_string(),
            html_url: "https://github.com/comment".to_string(),
            pull_request_review_id: Some(5),
            start_line: None,
            line: Some(42),
            side: Some("RIGHT".to_string()),
        };
        let json = serde_json::to_string(&comment).unwrap();
        assert!(json.contains("src/main.rs"));
        assert!(json.contains("Please fix this"));
    }

    #[test]
    fn test_review_watcher_is_enabled() {
        let config_enabled = GitHubConfig {
            token: Some("test".into()),
            poll_interval_ms: 60000,
            auto_resolve_on_merge: true,
            webhook_secret: None,
            review_trigger: "@claudear".to_string(),
            use_ssh: false,
            ..Default::default()
        };
        let client_enabled = GitHubClient::new(config_enabled);
        let provider_enabled: Arc<dyn ScmProvider> = Arc::new(client_enabled);
        let watcher_enabled = crate::scm::ReviewWatcher::new(provider_enabled);
        assert!(watcher_enabled.is_enabled());

        let config_disabled = GitHubConfig::default();
        let client_disabled = GitHubClient::new(config_disabled);
        let provider_disabled: Arc<dyn ScmProvider> = Arc::new(client_disabled);
        let watcher_disabled = crate::scm::ReviewWatcher::new(provider_disabled);
        assert!(!watcher_disabled.is_enabled());
    }

    #[test]
    fn test_review_event_feedback_summary_empty_body() {
        let review = PrReview {
            id: 1,
            state: "CHANGES_REQUESTED".to_string(),
            body: None,
            user: GitHubUser {
                id: 123,
                login: "reviewer".to_string(),
                user_type: Some("User".to_string()),
            },
            submitted_at: None,
            html_url: None,
        };

        let event = ReviewEvent::ReviewSubmitted {
            pr_url: "url".to_string(),
            repo: "repo".to_string(),
            pr_number: 1,
            review,
            inline_comments: vec![],
        };

        let summary = event.get_feedback_summary();
        assert!(summary.contains("@reviewer"));
        assert!(summary.contains("CHANGES_REQUESTED"));
        assert!(!summary.contains("Review comment:"));
    }

    #[test]
    fn test_review_event_comments_multiple() {
        let comments = vec![
            PrReviewComment {
                id: 1,
                path: "file1.rs".to_string(),
                position: None,
                original_position: None,
                body: "Comment 1".to_string(),
                user: GitHubUser {
                    id: 1,
                    login: "user1".to_string(),
                    user_type: None,
                },
                created_at: String::new(),
                updated_at: String::new(),
                html_url: String::new(),
                pull_request_review_id: None,
                start_line: None,
                line: Some(10),
                side: None,
            },
            PrReviewComment {
                id: 2,
                path: "file2.rs".to_string(),
                position: None,
                original_position: None,
                body: "Comment 2".to_string(),
                user: GitHubUser {
                    id: 2,
                    login: "user2".to_string(),
                    user_type: None,
                },
                created_at: String::new(),
                updated_at: String::new(),
                html_url: String::new(),
                pull_request_review_id: None,
                start_line: None,
                line: None,
                side: None,
            },
        ];

        let event = ReviewEvent::CommentsAdded {
            pr_url: "url".to_string(),
            repo: "repo".to_string(),
            pr_number: 1,
            comments,
        };

        let summary = event.get_feedback_summary();
        assert!(summary.contains("file1.rs"));
        assert!(summary.contains("file2.rs"));
        assert!(summary.contains("line 10"));
        assert!(summary.contains("Comment 1"));
        assert!(summary.contains("Comment 2"));
    }

    #[test]
    fn test_pr_review_state_serialization() {
        let state = PrReviewState::new("url", "repo", 1, "issue", "linear");
        let json = serde_json::to_string(&state).unwrap();
        assert!(json.contains("url"));
        assert!(json.contains("repo"));
        assert!(json.contains("issue"));
    }

    #[test]
    fn test_pr_status_debug() {
        let open = format!("{:?}", PrStatus::Open);
        let merged = format!("{:?}", PrStatus::Merged);
        let closed = format!("{:?}", PrStatus::Closed);

        assert_eq!(open, "Open");
        assert_eq!(merged, "Merged");
        assert_eq!(closed, "Closed");
    }

    #[test]
    fn test_pr_status_copy_clone() {
        let status = PrStatus::Open;
        let copied = status;
        let cloned = status;
        assert_eq!(copied, status);
        assert_eq!(cloned, status);
    }

    #[test]
    fn test_pr_status_update_clone() {
        let update = PrStatusUpdate {
            source: "linear".to_string(),
            issue_id: "issue-1".to_string(),
            short_id: "LIN-1".to_string(),
            pr_url: "https://github.com/org/repo/pull/1".to_string(),
            new_status: PrStatus::Merged,
            should_resolve: true,
            regression_watch_id: Some(42),
        };

        let cloned = update.clone();
        assert_eq!(cloned.source, "linear");
        assert_eq!(cloned.new_status, PrStatus::Merged);
        assert_eq!(cloned.regression_watch_id, Some(42));
    }

    #[test]
    fn test_pr_status_update_debug() {
        let update = PrStatusUpdate {
            source: "sentry".to_string(),
            issue_id: "id".to_string(),
            short_id: "short".to_string(),
            pr_url: "url".to_string(),
            new_status: PrStatus::Closed,
            should_resolve: false,
            regression_watch_id: None,
        };
        let debug = format!("{:?}", update);
        assert!(debug.contains("sentry"));
        assert!(debug.contains("Closed"));
    }

    #[test]
    fn test_pr_review_state_deserialization() {
        let json = r#"{
            "pr_url": "https://github.com/test/test/pull/1",
            "repo": "test/test",
            "pr_number": 1,
            "issue_id": "issue-123",
            "source": "linear",
            "last_review_id": null,
            "last_review_time": null,
            "last_comment_id": null,
            "last_comment_time": null,
            "is_active": true
        }"#;

        let state: PrReviewState = serde_json::from_str(json).unwrap();
        assert_eq!(state.pr_url, "https://github.com/test/test/pull/1");
        assert_eq!(state.repo, "test/test");
        assert_eq!(state.pr_number, 1);
        assert!(state.is_active);
    }

    #[test]
    fn test_review_watcher_get_all_states_empty() {
        let config = GitHubConfig::default();
        let client = GitHubClient::new(config);
        let provider: Arc<dyn ScmProvider> = Arc::new(client);
        let watcher = crate::scm::ReviewWatcher::new(provider);

        let all_states = watcher.get_all_states();
        assert!(all_states.is_empty());
    }

    #[test]
    fn test_review_watcher_overwrite_state() {
        let config = GitHubConfig {
            token: Some("test".into()),
            poll_interval_ms: 60000,
            auto_resolve_on_merge: true,
            webhook_secret: None,
            review_trigger: "@claudear".to_string(),
            use_ssh: false,
            ..Default::default()
        };
        let client = GitHubClient::new(config);
        let provider: Arc<dyn ScmProvider> = Arc::new(client);
        let watcher = crate::scm::ReviewWatcher::new(provider);

        let state1 = PrReviewState::new("url", "repo", 1, "issue1", "linear");
        let state2 = PrReviewState::new("url", "repo", 1, "issue2", "sentry");

        watcher.watch_pr(state1);
        watcher.watch_pr(state2);

        // Second state should overwrite first (same url)
        let retrieved = watcher.get_state("url").unwrap();
        assert_eq!(retrieved.source, "sentry");
    }

    #[test]
    fn test_github_user_clone() {
        let user = GitHubUser {
            id: 123,
            login: "test".to_string(),
            user_type: Some("User".to_string()),
        };
        let cloned = user.clone();
        assert_eq!(cloned.id, 123);
        assert_eq!(cloned.login, "test");
    }

    #[test]
    fn test_pr_review_clone() {
        let review = PrReview {
            id: 1,
            state: "APPROVED".to_string(),
            body: Some("LGTM".to_string()),
            user: GitHubUser {
                id: 123,
                login: "reviewer".to_string(),
                user_type: None,
            },
            submitted_at: Some("2024-01-01T00:00:00Z".to_string()),
            html_url: Some("url".to_string()),
        };
        let cloned = review.clone();
        assert_eq!(cloned.id, 1);
        assert_eq!(cloned.state, "APPROVED");
    }

    #[test]
    fn test_pr_review_comment_clone() {
        let comment = PrReviewComment {
            id: 1,
            path: "file.rs".to_string(),
            position: Some(10),
            original_position: Some(5),
            body: "Comment".to_string(),
            user: GitHubUser {
                id: 1,
                login: "user".to_string(),
                user_type: None,
            },
            created_at: "2024-01-01T00:00:00Z".to_string(),
            updated_at: "2024-01-01T00:00:00Z".to_string(),
            html_url: "url".to_string(),
            pull_request_review_id: Some(5),
            start_line: Some(1),
            line: Some(10),
            side: Some("RIGHT".to_string()),
        };
        let cloned = comment.clone();
        assert_eq!(cloned.id, 1);
        assert_eq!(cloned.path, "file.rs");
        assert_eq!(cloned.position, Some(10));
    }

    #[test]
    fn test_review_event_clone() {
        let review = PrReview {
            id: 1,
            state: "APPROVED".to_string(),
            body: None,
            user: GitHubUser {
                id: 1,
                login: "user".to_string(),
                user_type: None,
            },
            submitted_at: None,
            html_url: None,
        };

        let event = ReviewEvent::ReviewSubmitted {
            pr_url: "url".to_string(),
            repo: "repo".to_string(),
            pr_number: 1,
            review,
            inline_comments: vec![],
        };

        let cloned = event.clone();
        assert_eq!(cloned.pr_url(), "url");
    }

    #[tokio::test]
    async fn test_review_watcher_check_for_reviews_disabled() {
        let config = GitHubConfig::default(); // No token = disabled
        let client = GitHubClient::new(config);
        let provider: Arc<dyn ScmProvider> = Arc::new(client);
        let watcher = crate::scm::ReviewWatcher::new(provider);

        // Should return empty when disabled
        let events = watcher.check_for_reviews().await.unwrap();
        assert!(events.is_empty());
    }

    #[test]
    fn test_unwatch_nonexistent_pr() {
        let config = GitHubConfig {
            token: Some("test".into()),
            poll_interval_ms: 60000,
            auto_resolve_on_merge: true,
            webhook_secret: None,
            review_trigger: "@claudear".to_string(),
            use_ssh: false,
            ..Default::default()
        };
        let client = GitHubClient::new(config);
        let provider: Arc<dyn ScmProvider> = Arc::new(client);
        let watcher = crate::scm::ReviewWatcher::new(provider);

        // Should not panic when unwatching a PR that doesn't exist
        watcher.unwatch_pr("nonexistent");

        // Verify nothing was added
        assert!(watcher.get_state("nonexistent").is_none());
    }

    #[test]
    fn test_review_event_debug() {
        let review = PrReview {
            id: 1,
            state: "APPROVED".to_string(),
            body: None,
            user: GitHubUser {
                id: 1,
                login: "user".to_string(),
                user_type: None,
            },
            submitted_at: None,
            html_url: None,
        };

        let event = ReviewEvent::ReviewSubmitted {
            pr_url: "url".to_string(),
            repo: "repo".to_string(),
            pr_number: 1,
            review,
            inline_comments: vec![],
        };

        let debug = format!("{:?}", event);
        assert!(debug.contains("ReviewSubmitted"));
    }

    #[test]
    fn test_pr_review_state_clone() {
        let state = PrReviewState::new("url", "repo", 1, "issue", "linear");
        let cloned = state.clone();
        assert_eq!(cloned.pr_url, "url");
        assert_eq!(cloned.repo, "repo");
        assert_eq!(cloned.pr_number, 1);
    }

    #[test]
    fn test_github_client_new() {
        let config = GitHubConfig {
            token: Some("test_token".into()),
            poll_interval_ms: 30000,
            auto_resolve_on_merge: true,
            webhook_secret: None,
            review_trigger: "@claudear".to_string(),
            use_ssh: false,
            ..Default::default()
        };
        let client = GitHubClient::new(config);
        assert!(client.is_enabled());
        assert_eq!(client.token(), Some("test_token"));
    }

    #[test]
    fn test_github_client_disabled() {
        let config = GitHubConfig::default();
        let client = GitHubClient::new(config);
        assert!(!client.is_enabled());
        assert!(client.token().is_none());
    }

    #[test]
    fn test_pull_request_deserialization() {
        let json = r#"{
            "number": 123,
            "state": "open",
            "merged": false,
            "html_url": "https://github.com/test/repo/pull/123",
            "title": "Test PR"
        }"#;
        let pr: PullRequest = serde_json::from_str(json).unwrap();
        assert_eq!(pr.state, "open");
        assert!(!pr.merged);
    }

    #[test]
    fn test_pull_request_merged() {
        let json = r#"{
            "number": 456,
            "state": "closed",
            "merged": true,
            "html_url": "https://github.com/test/repo/pull/456",
            "title": "Merged PR"
        }"#;
        let pr: PullRequest = serde_json::from_str(json).unwrap();
        assert!(pr.merged);
        assert_eq!(pr.state, "closed");
    }

    #[test]
    fn test_pr_status_update_with_regression_watch() {
        let update = PrStatusUpdate {
            source: "sentry".to_string(),
            issue_id: "issue-1".to_string(),
            short_id: "SENTRY-1".to_string(),
            pr_url: "https://github.com/org/repo/pull/1".to_string(),
            new_status: PrStatus::Merged,
            should_resolve: false, // Should be false when regression watch is active
            regression_watch_id: Some(123),
        };

        assert!(!update.should_resolve);
        assert_eq!(update.regression_watch_id, Some(123));
    }

    #[test]
    fn test_org_repo_deserialization() {
        let json = r#"{
            "id": 123,
            "full_name": "test-org/test-repo",
            "name": "test-repo",
            "default_branch": "main",
            "clone_url": "https://github.com/test-org/test-repo.git",
            "html_url": "https://github.com/test-org/test-repo",
            "private": false,
            "archived": false
        }"#;

        let repo: OrgRepo = serde_json::from_str(json).unwrap();
        assert_eq!(repo.id, 123);
        assert_eq!(repo.full_name, "test-org/test-repo");
        assert_eq!(repo.name, "test-repo");
        assert_eq!(repo.default_branch, "main");
        assert_eq!(repo.clone_url, "https://github.com/test-org/test-repo.git");
        assert!(!repo.private);
        assert!(!repo.archived);
    }

    #[test]
    fn test_org_repo_deserialization_with_develop_branch() {
        let json = r#"{
            "id": 456,
            "full_name": "my-org/my-repo",
            "name": "my-repo",
            "default_branch": "develop",
            "clone_url": "https://github.com/my-org/my-repo.git",
            "html_url": "https://github.com/my-org/my-repo",
            "private": true,
            "archived": false
        }"#;

        let repo: OrgRepo = serde_json::from_str(json).unwrap();
        assert_eq!(repo.default_branch, "develop");
        assert!(repo.private);
    }

    #[tokio::test]
    async fn test_list_org_repos_success() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/orgs/test-org/repos?per_page=100&page=1",
            200,
            r#"[
                {
                    "id": 1,
                    "full_name": "test-org/repo1",
                    "name": "repo1",
                    "default_branch": "main",
                    "clone_url": "https://github.com/test-org/repo1.git",
                    "html_url": "https://github.com/test-org/repo1",
                    "private": false,
                    "archived": false
                },
                {
                    "id": 2,
                    "full_name": "test-org/repo2",
                    "name": "repo2",
                    "default_branch": "develop",
                    "clone_url": "https://github.com/test-org/repo2.git",
                    "html_url": "https://github.com/test-org/repo2",
                    "private": false,
                    "archived": false
                }
            ]"#,
        );

        let config = GitHubConfig {
            token: Some("test-token".into()),
            ..Default::default()
        };
        let client = GitHubClient::with_http_client(config, mock);

        let repos = client.list_org_repos("test-org").await.unwrap();
        assert_eq!(repos.len(), 2);
        assert_eq!(repos[0].full_name, "test-org/repo1");
        assert_eq!(repos[1].full_name, "test-org/repo2");
    }

    #[tokio::test]
    async fn test_list_org_repos_filters_archived() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/orgs/test-org/repos?per_page=100&page=1",
            200,
            r#"[
                {
                    "id": 1,
                    "full_name": "test-org/active-repo",
                    "name": "active-repo",
                    "default_branch": "main",
                    "clone_url": "https://github.com/test-org/active-repo.git",
                    "html_url": "https://github.com/test-org/active-repo",
                    "private": false,
                    "archived": false
                },
                {
                    "id": 2,
                    "full_name": "test-org/archived-repo",
                    "name": "archived-repo",
                    "default_branch": "main",
                    "clone_url": "https://github.com/test-org/archived-repo.git",
                    "html_url": "https://github.com/test-org/archived-repo",
                    "private": false,
                    "archived": true
                }
            ]"#,
        );

        let config = GitHubConfig {
            token: Some("test-token".into()),
            ..Default::default()
        };
        let client = GitHubClient::with_http_client(config, mock);

        let repos = client.list_org_repos("test-org").await.unwrap();
        // Archived repos should be filtered out
        assert_eq!(repos.len(), 1);
        assert_eq!(repos[0].full_name, "test-org/active-repo");
    }

    #[tokio::test]
    async fn test_list_org_repos_org_not_found() {
        let mock = MockHttpClient::new();
        // No mock response added, so it will return 404

        let config = GitHubConfig {
            token: Some("test-token".into()),
            ..Default::default()
        };
        let client = GitHubClient::with_http_client(config, mock);

        let result = client.list_org_repos("nonexistent-org").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[tokio::test]
    async fn test_list_org_repos_no_token() {
        let mock = MockHttpClient::new();
        let config = GitHubConfig::default(); // No token
        let client = GitHubClient::with_http_client(config, mock);

        let result = client.list_org_repos("test-org").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("token"));
    }

    #[tokio::test]
    async fn test_get_pr_info_success() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/1",
            200,
            r#"{
                "number": 1,
                "state": "open",
                "merged": false,
                "title": "Fix authentication bug",
                "head": {"ref": "fix/auth-bug"},
                "base": {"ref": "main"},
                "user": {"id": 42, "login": "testuser"}
            }"#,
        );

        let config = GitHubConfig {
            token: Some("test_token".into()),
            poll_interval_ms: 60000,
            auto_resolve_on_merge: true,
            webhook_secret: None,
            review_trigger: "@claudear".to_string(),
            use_ssh: false,
            ..Default::default()
        };
        let client = GitHubClient::with_http_client(config, mock);

        let info = client.get_pr_info("owner/repo", 1).await.unwrap();
        assert_eq!(info.head_branch, Some("fix/auth-bug".to_string()));
        assert_eq!(info.base_branch, Some("main".to_string()));
        assert_eq!(info.title, Some("Fix authentication bug".to_string()));
        assert_eq!(info.author, Some("testuser".to_string()));
    }

    #[tokio::test]
    async fn test_get_pr_info_missing_optional_fields() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/2",
            200,
            r#"{
                "number": 2,
                "state": "open",
                "merged": false
            }"#,
        );

        let config = GitHubConfig {
            token: Some("test_token".into()),
            poll_interval_ms: 60000,
            auto_resolve_on_merge: true,
            webhook_secret: None,
            review_trigger: "@claudear".to_string(),
            use_ssh: false,
            ..Default::default()
        };
        let client = GitHubClient::with_http_client(config, mock);

        let info = client.get_pr_info("owner/repo", 2).await.unwrap();
        assert!(info.head_branch.is_none());
        assert!(info.base_branch.is_none());
        assert!(info.title.is_none());
        assert!(info.author.is_none());
    }

    #[tokio::test]
    async fn test_get_pr_info_api_error() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/3",
            500,
            "Internal Server Error",
        );

        let config = GitHubConfig {
            token: Some("test_token".into()),
            poll_interval_ms: 60000,
            auto_resolve_on_merge: true,
            webhook_secret: None,
            review_trigger: "@claudear".to_string(),
            use_ssh: false,
            ..Default::default()
        };
        let client = GitHubClient::with_http_client(config, mock);

        let result = client.get_pr_info("owner/repo", 3).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("GitHub API error"));
    }

    #[tokio::test]
    async fn test_get_pr_info_no_token() {
        let mock = MockHttpClient::new();
        let config = GitHubConfig::default();
        let client = GitHubClient::with_http_client(config, mock);

        let result = client.get_pr_info("owner/repo", 1).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("token"));
    }

    #[tokio::test]
    async fn test_get_pr_diff_success() {
        let mock = MockHttpClient::new();
        let diff_content = "diff --git a/src/main.rs b/src/main.rs\n--- a/src/main.rs\n+++ b/src/main.rs\n@@ -1,3 +1,4 @@\n fn main() {\n+    println!(\"Hello\");\n }";
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/1",
            200,
            diff_content,
        );

        let config = GitHubConfig {
            token: Some("test_token".into()),
            poll_interval_ms: 60000,
            auto_resolve_on_merge: true,
            webhook_secret: None,
            review_trigger: "@claudear".to_string(),
            use_ssh: false,
            ..Default::default()
        };
        let client = GitHubClient::with_http_client(config, mock);

        let diff = client.get_pr_diff("owner/repo", 1).await.unwrap();
        assert!(diff.contains("diff --git"));
        assert!(diff.contains("println!"));
    }

    #[tokio::test]
    async fn test_get_pr_diff_uses_diff_accept_header() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/5",
            200,
            "diff content",
        );

        let config = GitHubConfig {
            token: Some("test_token".into()),
            poll_interval_ms: 60000,
            auto_resolve_on_merge: true,
            webhook_secret: None,
            review_trigger: "@claudear".to_string(),
            use_ssh: false,
            ..Default::default()
        };
        let client = GitHubClient::with_http_client(config, mock);

        let _ = client.get_pr_diff("owner/repo", 5).await;

        let requests = client.http.get_requests();
        assert_eq!(requests.len(), 1);
        let (_, headers) = &requests[0];
        let accept = headers.iter().find(|(k, _)| k == "Accept");
        assert!(accept.is_some());
        assert_eq!(accept.unwrap().1, "application/vnd.github.v3.diff");
    }

    #[tokio::test]
    async fn test_get_pr_diff_api_error() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/1",
            403,
            "Forbidden",
        );

        let config = GitHubConfig {
            token: Some("test_token".into()),
            poll_interval_ms: 60000,
            auto_resolve_on_merge: true,
            webhook_secret: None,
            review_trigger: "@claudear".to_string(),
            use_ssh: false,
            ..Default::default()
        };
        let client = GitHubClient::with_http_client(config, mock);

        let result = client.get_pr_diff("owner/repo", 1).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("GitHub API error"));
    }

    #[tokio::test]
    async fn test_get_pr_diff_no_token() {
        let mock = MockHttpClient::new();
        let config = GitHubConfig::default();
        let client = GitHubClient::with_http_client(config, mock);

        let result = client.get_pr_diff("owner/repo", 1).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("token"));
    }

    #[tokio::test]
    async fn test_list_org_repos_paginates() {
        let mock = MockHttpClient::new();

        // First page: 100 repos (triggers pagination)
        let first_page: Vec<String> = (1..=100)
            .map(|i| {
                format!(
                    r#"{{"id": {i}, "full_name": "org/repo{i}", "name": "repo{i}", "default_branch": "main", "clone_url": "https://github.com/org/repo{i}.git", "html_url": "https://github.com/org/repo{i}", "private": false, "archived": false}}"#
                )
            })
            .collect();
        mock.mock_response(
            "https://api.github.com/orgs/org/repos?per_page=100&page=1",
            200,
            format!("[{}]", first_page.join(",")),
        );

        // Second page: 2 repos (stops pagination)
        mock.mock_response(
            "https://api.github.com/orgs/org/repos?per_page=100&page=2",
            200,
            r#"[
                {"id": 101, "full_name": "org/repo101", "name": "repo101", "default_branch": "main", "clone_url": "https://github.com/org/repo101.git", "html_url": "https://github.com/org/repo101", "private": false, "archived": false},
                {"id": 102, "full_name": "org/repo102", "name": "repo102", "default_branch": "main", "clone_url": "https://github.com/org/repo102.git", "html_url": "https://github.com/org/repo102", "private": false, "archived": false}
            ]"#,
        );

        let config = GitHubConfig {
            token: Some("test_token".into()),
            ..Default::default()
        };
        let client = GitHubClient::with_http_client(config, mock);

        let repos = client.list_org_repos("org").await.unwrap();
        assert_eq!(repos.len(), 102);

        // Verify pagination happened - 2 requests total
        let requests = client.http.get_requests();
        assert_eq!(requests.len(), 2);
    }

    #[tokio::test]
    async fn test_list_org_repos_api_error() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/orgs/test-org/repos?per_page=100&page=1",
            500,
            "Internal Server Error",
        );

        let config = GitHubConfig {
            token: Some("test_token".into()),
            ..Default::default()
        };
        let client = GitHubClient::with_http_client(config, mock);

        let result = client.list_org_repos("test-org").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("GitHub API error"));
    }

    #[test]
    fn test_build_headers_structure() {
        let config = GitHubConfig {
            token: Some("test_token".into()),
            poll_interval_ms: 60000,
            auto_resolve_on_merge: true,
            webhook_secret: None,
            review_trigger: "@claudear".to_string(),
            use_ssh: false,
            ..Default::default()
        };
        let client = GitHubClient::with_http_client(config, MockHttpClient::new());

        let headers = client.build_headers("my_token");
        assert_eq!(headers.len(), 4);

        let auth = headers.iter().find(|(k, _)| *k == "Authorization").unwrap();
        assert_eq!(auth.1, "Bearer my_token");

        let accept = headers.iter().find(|(k, _)| *k == "Accept").unwrap();
        assert_eq!(accept.1, "application/vnd.github+json");

        let ua = headers.iter().find(|(k, _)| *k == "User-Agent").unwrap();
        assert_eq!(ua.1, "claudear");

        let api_ver = headers
            .iter()
            .find(|(k, _)| *k == "X-GitHub-Api-Version")
            .unwrap();
        assert_eq!(api_ver.1, "2022-11-28");
    }

    #[test]
    fn test_scm_provider_name() {
        let config = GitHubConfig::default();
        let client = GitHubClient::new(config);
        let provider: &dyn ScmProvider = &client;
        assert_eq!(provider.name(), "github");
    }

    #[test]
    fn test_scm_provider_review_trigger() {
        let config = GitHubConfig {
            token: Some("test".into()),
            poll_interval_ms: 60000,
            auto_resolve_on_merge: true,
            webhook_secret: None,
            review_trigger: "@custom-trigger".to_string(),
            use_ssh: false,
            ..Default::default()
        };
        let client = GitHubClient::new(config);
        let provider: &dyn ScmProvider = &client;
        assert_eq!(provider.review_trigger(), "@custom-trigger");
    }

    #[test]
    fn test_scm_provider_is_enabled_delegates() {
        let config_enabled = GitHubConfig {
            token: Some("test".into()),
            ..Default::default()
        };
        let client_enabled = GitHubClient::new(config_enabled);
        let provider_enabled: &dyn ScmProvider = &client_enabled;
        assert!(provider_enabled.is_enabled());

        let config_disabled = GitHubConfig::default();
        let client_disabled = GitHubClient::new(config_disabled);
        let provider_disabled: &dyn ScmProvider = &client_disabled;
        assert!(!provider_disabled.is_enabled());
    }

    #[tokio::test]
    async fn test_scm_provider_get_pr_status_delegates() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/1",
            200,
            r#"{"number": 1, "state": "open", "merged": false}"#,
        );

        let config = GitHubConfig {
            token: Some("test_token".into()),
            ..Default::default()
        };
        let client = GitHubClient::with_http_client(config, mock);
        let provider: &dyn ScmProvider = &client;

        let status = provider.get_pr_status("owner/repo", 1).await.unwrap();
        assert_eq!(status, PrStatus::Open);
    }

    #[tokio::test]
    async fn test_scm_provider_get_pr_info_delegates() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/1",
            200,
            r#"{
                "number": 1,
                "state": "open",
                "merged": false,
                "title": "Test PR",
                "head": {"ref": "feature"},
                "base": {"ref": "main"},
                "user": {"id": 1, "login": "author"}
            }"#,
        );

        let config = GitHubConfig {
            token: Some("test_token".into()),
            ..Default::default()
        };
        let client = GitHubClient::with_http_client(config, mock);
        let provider: &dyn ScmProvider = &client;

        let info = provider.get_pr_info("owner/repo", 1).await.unwrap();
        assert_eq!(info.title, Some("Test PR".to_string()));
        assert_eq!(info.head_branch, Some("feature".to_string()));
    }

    #[tokio::test]
    async fn test_scm_provider_get_pr_diff_delegates() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/1",
            200,
            "diff content here",
        );

        let config = GitHubConfig {
            token: Some("test_token".into()),
            ..Default::default()
        };
        let client = GitHubClient::with_http_client(config, mock);
        let provider: &dyn ScmProvider = &client;

        let diff = provider.get_pr_diff("owner/repo", 1).await.unwrap();
        assert_eq!(diff, "diff content here");
    }

    #[tokio::test]
    async fn test_scm_provider_list_repos_delegates() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/orgs/test-org/repos?per_page=100&page=1",
            200,
            r#"[{
                "id": 1,
                "full_name": "test-org/repo1",
                "name": "repo1",
                "default_branch": "main",
                "clone_url": "https://github.com/test-org/repo1.git",
                "html_url": "https://github.com/test-org/repo1",
                "private": false,
                "archived": false
            }]"#,
        );

        let config = GitHubConfig {
            token: Some("test_token".into()),
            ..Default::default()
        };
        let client = GitHubClient::with_http_client(config, mock);
        let provider: &dyn ScmProvider = &client;

        let repos = provider.list_repos("test-org").await.unwrap();
        assert_eq!(repos.len(), 1);
        assert_eq!(repos[0].full_name, "test-org/repo1");
    }

    fn test_config() -> GitHubConfig {
        GitHubConfig {
            token: Some("test_token".into()),
            poll_interval_ms: 60000,
            auto_resolve_on_merge: true,
            webhook_secret: None,
            review_trigger: "@claudear".to_string(),
            use_ssh: false,
            ..Default::default()
        }
    }

    // --- merge_pr tests ---

    #[tokio::test]
    async fn test_merge_pr_success() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/1/merge",
            200,
            r#"{"sha": "abc123", "merged": true}"#,
        );

        let client = GitHubClient::with_http_client(test_config(), mock);
        let result = client.merge_pr("owner/repo", 1).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_merge_pr_api_error() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/1/merge",
            405,
            "Pull Request is not mergeable",
        );

        let client = GitHubClient::with_http_client(test_config(), mock);
        let result = client.merge_pr("owner/repo", 1).await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Failed to merge"),
            "error should mention merge failure: {err_msg}"
        );
    }

    #[tokio::test]
    async fn test_merge_pr_no_token() {
        let mock = MockHttpClient::new();
        let client = GitHubClient::with_http_client(GitHubConfig::default(), mock);
        let result = client.merge_pr("owner/repo", 1).await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("token"),
            "error should mention token: {err_msg}"
        );
    }

    #[tokio::test]
    async fn test_merge_pr_conflict() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/5/merge",
            409,
            r#"{"message": "Head branch was modified"}"#,
        );

        let client = GitHubClient::with_http_client(test_config(), mock);
        let result = client.merge_pr("owner/repo", 5).await;
        assert!(result.is_err());
    }

    // --- close_pr tests ---

    #[tokio::test]
    async fn test_close_pr_success() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/1",
            200,
            r#"{"number": 1, "state": "closed"}"#,
        );

        let client = GitHubClient::with_http_client(test_config(), mock);
        let result = client.close_pr("owner/repo", 1).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_close_pr_api_error() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/1",
            500,
            "Internal Server Error",
        );

        let client = GitHubClient::with_http_client(test_config(), mock);
        let result = client.close_pr("owner/repo", 1).await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Failed to close PR"),
            "error should mention close failure: {err_msg}"
        );
    }

    #[tokio::test]
    async fn test_close_pr_no_token() {
        let mock = MockHttpClient::new();
        let client = GitHubClient::with_http_client(GitHubConfig::default(), mock);
        let result = client.close_pr("owner/repo", 1).await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("token"),
            "error should mention token: {err_msg}"
        );
    }

    // --- delete_branch tests ---

    #[tokio::test]
    async fn test_delete_branch_success() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/git/refs/heads/feature-branch",
            204,
            "",
        );

        let client = GitHubClient::with_http_client(test_config(), mock);
        let result = client.delete_branch("owner/repo", "feature-branch").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_delete_branch_not_found_is_ok() {
        let mock = MockHttpClient::new();
        // Branch already deleted -- 404 should be treated as success
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/git/refs/heads/gone",
            404,
            "Reference does not exist",
        );

        let client = GitHubClient::with_http_client(test_config(), mock);
        let result = client.delete_branch("owner/repo", "gone").await;
        assert!(
            result.is_ok(),
            "404 on delete_branch should be treated as success"
        );
    }

    #[tokio::test]
    async fn test_delete_branch_api_error() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/git/refs/heads/protected",
            422,
            "Cannot delete a protected branch",
        );

        let client = GitHubClient::with_http_client(test_config(), mock);
        let result = client.delete_branch("owner/repo", "protected").await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Failed to delete branch"),
            "error should mention delete failure: {err_msg}"
        );
    }

    #[tokio::test]
    async fn test_delete_branch_no_token() {
        let mock = MockHttpClient::new();
        let client = GitHubClient::with_http_client(GitHubConfig::default(), mock);
        let result = client.delete_branch("owner/repo", "branch").await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("token"),
            "error should mention token: {err_msg}"
        );
    }

    // --- post_review tests ---

    #[tokio::test]
    async fn test_post_review_comment_success() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/1/reviews",
            200,
            r#"{"id": 1}"#,
        );

        let client = GitHubClient::with_http_client(test_config(), mock);
        let result = client
            .post_review("owner/repo", 1, PostReviewAction::Comment, "Looks good")
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_post_review_approve_success() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/1/reviews",
            200,
            r#"{"id": 2}"#,
        );

        let client = GitHubClient::with_http_client(test_config(), mock);
        let result = client
            .post_review("owner/repo", 1, PostReviewAction::Approve, "LGTM")
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_post_review_request_changes_success() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/1/reviews",
            200,
            r#"{"id": 3}"#,
        );

        let client = GitHubClient::with_http_client(test_config(), mock);
        let result = client
            .post_review(
                "owner/repo",
                1,
                PostReviewAction::RequestChanges,
                "Please fix",
            )
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_post_review_api_error() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/1/reviews",
            422,
            "Validation failed",
        );

        let client = GitHubClient::with_http_client(test_config(), mock);
        let result = client
            .post_review("owner/repo", 1, PostReviewAction::Comment, "text")
            .await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Failed to post review"),
            "error should mention review failure: {err_msg}"
        );
    }

    #[tokio::test]
    async fn test_post_review_no_token() {
        let mock = MockHttpClient::new();
        let client = GitHubClient::with_http_client(GitHubConfig::default(), mock);
        let result = client
            .post_review("owner/repo", 1, PostReviewAction::Comment, "hi")
            .await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("token"),
            "error should mention token: {err_msg}"
        );
    }

    // --- post_review_with_comments tests ---

    #[tokio::test]
    async fn test_post_review_with_comments_success() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/1/reviews",
            200,
            r#"{"id": 10}"#,
        );

        let comments = vec![
            InlineReviewComment {
                path: "src/main.rs".to_string(),
                body: "Fix this line".to_string(),
                position: Some(42),
            },
            InlineReviewComment {
                path: "src/lib.rs".to_string(),
                body: "Add a test".to_string(),
                position: None,
            },
        ];

        let client = GitHubClient::with_http_client(test_config(), mock);
        let result = client
            .post_review_with_comments(
                "owner/repo",
                1,
                PostReviewAction::RequestChanges,
                "Please address these",
                &comments,
            )
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_post_review_with_comments_empty_comments() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/1/reviews",
            200,
            r#"{"id": 11}"#,
        );

        let client = GitHubClient::with_http_client(test_config(), mock);
        let result = client
            .post_review_with_comments("owner/repo", 1, PostReviewAction::Comment, "Body text", &[])
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_post_review_with_comments_api_error() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/1/reviews",
            422,
            "Validation failed",
        );

        let comments = vec![InlineReviewComment {
            path: "file.rs".to_string(),
            body: "Fix".to_string(),
            position: Some(1),
        }];

        let client = GitHubClient::with_http_client(test_config(), mock);
        let result = client
            .post_review_with_comments(
                "owner/repo",
                1,
                PostReviewAction::Comment,
                "body",
                &comments,
            )
            .await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Failed to post review with comments"),
            "error should mention comments failure: {err_msg}"
        );
    }

    #[tokio::test]
    async fn test_post_review_with_comments_no_token() {
        let mock = MockHttpClient::new();
        let client = GitHubClient::with_http_client(GitHubConfig::default(), mock);
        let result = client
            .post_review_with_comments("owner/repo", 1, PostReviewAction::Comment, "body", &[])
            .await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("token"),
            "error should mention token: {err_msg}"
        );
    }

    // --- list_open_prs tests ---

    #[tokio::test]
    async fn test_list_open_prs_success() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls?state=open&per_page=100",
            200,
            r#"[
                {"number": 1, "title": "Feature A", "head": {"ref": "feature-a"}, "html_url": "https://github.com/owner/repo/pull/1"},
                {"number": 2, "title": "Fix B", "head": {"ref": "fix-b"}, "html_url": "https://github.com/owner/repo/pull/2"}
            ]"#,
        );

        let client = GitHubClient::with_http_client(test_config(), mock);
        let prs = client.list_open_prs("owner/repo").await.unwrap();
        assert_eq!(prs.len(), 2);
        assert_eq!(prs[0].number, 1);
        assert_eq!(prs[0].title, "Feature A");
        assert_eq!(prs[0].branch, "feature-a");
        assert!(prs[0].url.contains("pull/1"));
        assert_eq!(prs[1].number, 2);
    }

    #[tokio::test]
    async fn test_list_open_prs_empty() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls?state=open&per_page=100",
            200,
            "[]",
        );

        let client = GitHubClient::with_http_client(test_config(), mock);
        let prs = client.list_open_prs("owner/repo").await.unwrap();
        assert!(prs.is_empty());
    }

    #[tokio::test]
    async fn test_list_open_prs_api_error() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls?state=open&per_page=100",
            500,
            "Internal Server Error",
        );

        let client = GitHubClient::with_http_client(test_config(), mock);
        let result = client.list_open_prs("owner/repo").await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Failed to list open PRs"),
            "error should mention list failure: {err_msg}"
        );
    }

    #[tokio::test]
    async fn test_list_open_prs_no_token() {
        let mock = MockHttpClient::new();
        let client = GitHubClient::with_http_client(GitHubConfig::default(), mock);
        let result = client.list_open_prs("owner/repo").await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("token"),
            "error should mention token: {err_msg}"
        );
    }

    // --- get_latest_release tests ---

    #[tokio::test]
    async fn test_get_latest_release_success() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/releases/latest",
            200,
            r#"{
                "tag_name": "v1.0.0",
                "name": "Release 1.0.0",
                "html_url": "https://github.com/owner/repo/releases/tag/v1.0.0",
                "published_at": "2025-01-01T00:00:00Z"
            }"#,
        );

        let client = GitHubClient::with_http_client(test_config(), mock);
        let release = client.get_latest_release("owner/repo").await.unwrap();
        assert!(release.is_some());
        let r = release.unwrap();
        assert_eq!(r.tag, "v1.0.0");
        assert_eq!(r.name.as_deref(), Some("Release 1.0.0"));
        assert_eq!(r.url, "https://github.com/owner/repo/releases/tag/v1.0.0");
        assert_eq!(r.published_at.as_deref(), Some("2025-01-01T00:00:00Z"));
    }

    #[tokio::test]
    async fn test_get_latest_release_no_releases() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/releases/latest",
            404,
            "Not Found",
        );

        let client = GitHubClient::with_http_client(test_config(), mock);
        let release = client.get_latest_release("owner/repo").await.unwrap();
        assert!(release.is_none());
    }

    #[tokio::test]
    async fn test_get_latest_release_api_error() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/releases/latest",
            500,
            "Internal Server Error",
        );

        let client = GitHubClient::with_http_client(test_config(), mock);
        let result = client.get_latest_release("owner/repo").await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Failed to get latest release"),
            "error should mention release failure: {err_msg}"
        );
    }

    #[tokio::test]
    async fn test_get_latest_release_no_token() {
        let mock = MockHttpClient::new();
        let client = GitHubClient::with_http_client(GitHubConfig::default(), mock);
        let result = client.get_latest_release("owner/repo").await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("token"),
            "error should mention token: {err_msg}"
        );
    }

    // --- create_release tests ---

    #[tokio::test]
    async fn test_create_release_success() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/releases",
            201,
            r#"{
                "tag_name": "v2.0.0",
                "name": "Version 2",
                "html_url": "https://github.com/owner/repo/releases/tag/v2.0.0",
                "published_at": "2025-06-01T00:00:00Z"
            }"#,
        );

        let client = GitHubClient::with_http_client(test_config(), mock);
        let release = client
            .create_release("owner/repo", "v2.0.0", "Version 2", "Release notes")
            .await
            .unwrap();
        assert_eq!(release.tag, "v2.0.0");
        assert_eq!(release.name.as_deref(), Some("Version 2"));
        assert_eq!(
            release.url,
            "https://github.com/owner/repo/releases/tag/v2.0.0"
        );
    }

    #[tokio::test]
    async fn test_create_release_api_error() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/releases",
            422,
            "Validation Failed",
        );

        let client = GitHubClient::with_http_client(test_config(), mock);
        let result = client
            .create_release("owner/repo", "v1.0.0", "Version 1", "notes")
            .await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Failed to create release"),
            "error should mention create failure: {err_msg}"
        );
    }

    #[tokio::test]
    async fn test_create_release_no_token() {
        let mock = MockHttpClient::new();
        let client = GitHubClient::with_http_client(GitHubConfig::default(), mock);
        let result = client
            .create_release("owner/repo", "v1.0.0", "Version 1", "notes")
            .await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("token"),
            "error should mention token: {err_msg}"
        );
    }

    // --- parse_pr_number tests ---

    #[test]
    fn test_parse_pr_number_pull_url() {
        assert_eq!(
            GitHubClient::<MockHttpClient>::parse_pr_number(
                "https://github.com/owner/repo/pull/42"
            ),
            Some(42)
        );
    }

    #[test]
    fn test_parse_pr_number_pulls_url() {
        assert_eq!(
            GitHubClient::<MockHttpClient>::parse_pr_number(
                "https://github.com/owner/repo/pulls/123"
            ),
            Some(123)
        );
    }

    #[test]
    fn test_parse_pr_number_no_match() {
        assert_eq!(
            GitHubClient::<MockHttpClient>::parse_pr_number(
                "https://github.com/owner/repo/issues/5"
            ),
            None
        );
    }

    #[test]
    fn test_parse_pr_number_empty_string() {
        assert_eq!(GitHubClient::<MockHttpClient>::parse_pr_number(""), None);
    }

    #[test]
    fn test_parse_pr_number_non_numeric() {
        assert_eq!(
            GitHubClient::<MockHttpClient>::parse_pr_number(
                "https://github.com/owner/repo/pull/abc"
            ),
            None
        );
    }

    // --- list_repo_issues tests ---

    #[tokio::test]
    async fn test_list_repo_issues_success() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/issues?per_page=100&page=1&state=open",
            200,
            r#"[
                {
                    "number": 1,
                    "title": "Bug report",
                    "body": "Something is broken",
                    "state": "open",
                    "html_url": "https://github.com/owner/repo/issues/1",
                    "labels": [{"name": "bug"}],
                    "user": {"id": 10, "login": "reporter"},
                    "assignee": null
                },
                {
                    "number": 2,
                    "title": "PR title",
                    "body": "This is a PR, not an issue",
                    "state": "open",
                    "html_url": "https://github.com/owner/repo/pull/2",
                    "labels": [],
                    "user": {"id": 20, "login": "dev"},
                    "assignee": null,
                    "pull_request": {"url": "https://api.github.com/repos/owner/repo/pulls/2"}
                }
            ]"#,
        );

        let client = GitHubClient::with_http_client(test_config(), mock);
        let issues = client
            .list_repo_issues("owner/repo", Some("open"), &[])
            .await
            .unwrap();
        // PR (number 2) should be filtered out
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].number, 1);
        assert_eq!(issues[0].title, "Bug report");
        assert_eq!(issues[0].labels[0].name, "bug");
    }

    #[tokio::test]
    async fn test_list_repo_issues_with_labels() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/issues?per_page=100&page=1&state=open&labels=bug,urgent",
            200,
            "[]",
        );

        let client = GitHubClient::with_http_client(test_config(), mock);
        let labels = vec!["bug".to_string(), "urgent".to_string()];
        let issues = client
            .list_repo_issues("owner/repo", Some("open"), &labels)
            .await
            .unwrap();
        assert!(issues.is_empty());
    }

    #[tokio::test]
    async fn test_list_repo_issues_no_state() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/issues?per_page=100&page=1",
            200,
            "[]",
        );

        let client = GitHubClient::with_http_client(test_config(), mock);
        let issues = client
            .list_repo_issues("owner/repo", None, &[])
            .await
            .unwrap();
        assert!(issues.is_empty());
    }

    #[tokio::test]
    async fn test_list_repo_issues_api_error() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/issues?per_page=100&page=1",
            403,
            "Forbidden",
        );

        let client = GitHubClient::with_http_client(test_config(), mock);
        let result = client.list_repo_issues("owner/repo", None, &[]).await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("GitHub API error listing issues"),
            "error should mention listing failure: {err_msg}"
        );
    }

    #[tokio::test]
    async fn test_list_repo_issues_no_token() {
        let mock = MockHttpClient::new();
        let client = GitHubClient::with_http_client(GitHubConfig::default(), mock);
        let result = client.list_repo_issues("owner/repo", None, &[]).await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("token"),
            "error should mention token: {err_msg}"
        );
    }

    // --- get_issue tests ---

    #[tokio::test]
    async fn test_get_issue_success() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/issues/42",
            200,
            r#"{
                "number": 42,
                "title": "Single issue",
                "body": "Details here",
                "state": "open",
                "html_url": "https://github.com/owner/repo/issues/42",
                "labels": [{"name": "enhancement"}],
                "user": {"id": 1, "login": "admin"},
                "assignee": {"id": 2, "login": "dev"}
            }"#,
        );

        let client = GitHubClient::with_http_client(test_config(), mock);
        let issue = client.get_issue("owner/repo", 42).await.unwrap();
        assert_eq!(issue.number, 42);
        assert_eq!(issue.title, "Single issue");
        assert_eq!(issue.body.as_deref(), Some("Details here"));
        assert_eq!(issue.state, "open");
        assert_eq!(issue.labels[0].name, "enhancement");
    }

    #[tokio::test]
    async fn test_get_issue_not_found() {
        let mock = MockHttpClient::new();
        // No mock response -- returns 404 by default

        let client = GitHubClient::with_http_client(test_config(), mock);
        let result = client.get_issue("owner/repo", 999).await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Issue not found"),
            "error should mention issue not found: {err_msg}"
        );
    }

    #[tokio::test]
    async fn test_get_issue_api_error() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/issues/42",
            500,
            "Internal Server Error",
        );

        let client = GitHubClient::with_http_client(test_config(), mock);
        let result = client.get_issue("owner/repo", 42).await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("GitHub API error"),
            "error should mention API error: {err_msg}"
        );
    }

    #[tokio::test]
    async fn test_get_issue_no_token() {
        let mock = MockHttpClient::new();
        let client = GitHubClient::with_http_client(GitHubConfig::default(), mock);
        let result = client.get_issue("owner/repo", 1).await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("token"),
            "error should mention token: {err_msg}"
        );
    }

    // --- close_issue tests ---

    #[tokio::test]
    async fn test_close_issue_success() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/issues/42",
            200,
            r#"{"number": 42, "state": "closed"}"#,
        );

        let client = GitHubClient::with_http_client(test_config(), mock);
        let result = client.close_issue("owner/repo", 42).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_close_issue_api_error() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/issues/42",
            500,
            "Internal Server Error",
        );

        let client = GitHubClient::with_http_client(test_config(), mock);
        let result = client.close_issue("owner/repo", 42).await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Failed to close issue"),
            "error should mention close failure: {err_msg}"
        );
    }

    #[tokio::test]
    async fn test_close_issue_no_token() {
        let mock = MockHttpClient::new();
        let client = GitHubClient::with_http_client(GitHubConfig::default(), mock);
        let result = client.close_issue("owner/repo", 42).await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("token"),
            "error should mention token: {err_msg}"
        );
    }

    // --- add_issue_comment tests ---

    #[tokio::test]
    async fn test_add_issue_comment_success() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/issues/42/comments",
            201,
            r#"{"id": 1, "body": "My comment"}"#,
        );

        let client = GitHubClient::with_http_client(test_config(), mock);
        let result = client
            .add_issue_comment("owner/repo", 42, "My comment")
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_add_issue_comment_api_error() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/issues/42/comments",
            403,
            "Forbidden",
        );

        let client = GitHubClient::with_http_client(test_config(), mock);
        let result = client.add_issue_comment("owner/repo", 42, "comment").await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Failed to add comment"),
            "error should mention comment failure: {err_msg}"
        );
    }

    #[tokio::test]
    async fn test_add_issue_comment_no_token() {
        let mock = MockHttpClient::new();
        let client = GitHubClient::with_http_client(GitHubConfig::default(), mock);
        let result = client.add_issue_comment("owner/repo", 42, "comment").await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("token"),
            "error should mention token: {err_msg}"
        );
    }

    #[tokio::test]
    async fn test_list_org_repos_not_found() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/orgs/nonexistent/repos?per_page=100&page=1",
            404,
            "Not Found",
        );

        let client = GitHubClient::with_http_client(test_config(), mock);
        let result = client.list_org_repos("nonexistent").await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Organization not found"),
            "error should mention org not found: {err_msg}"
        );
    }

    #[tokio::test]
    async fn test_list_org_repos_empty() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/orgs/myorg/repos?per_page=100&page=1",
            200,
            "[]",
        );

        let client = GitHubClient::with_http_client(test_config(), mock);
        let repos = client.list_org_repos("myorg").await.unwrap();
        assert!(repos.is_empty());
    }

    // (test_list_org_repos_success, test_list_org_repos_api_error, test_list_org_repos_no_token already exist above)

    #[tokio::test]
    async fn test_list_org_repos_filters_archived_with_ssh() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/orgs/myorg/repos?per_page=100&page=1",
            200,
            r#"[
                {
                    "id": 1,
                    "full_name": "myorg/active-repo",
                    "name": "active-repo",
                    "default_branch": "main",
                    "clone_url": "https://github.com/myorg/active-repo.git",
                    "ssh_url": "git@github.com:myorg/active-repo.git",
                    "html_url": "https://github.com/myorg/active-repo",
                    "private": true,
                    "archived": false
                },
                {
                    "id": 2,
                    "full_name": "myorg/archived-repo",
                    "name": "archived-repo",
                    "default_branch": "main",
                    "clone_url": "https://github.com/myorg/archived-repo.git",
                    "ssh_url": "git@github.com:myorg/archived-repo.git",
                    "html_url": "https://github.com/myorg/archived-repo",
                    "private": false,
                    "archived": true
                }
            ]"#,
        );

        let client = GitHubClient::with_http_client(test_config(), mock);
        let repos = client.list_org_repos("myorg").await.unwrap();
        assert_eq!(repos.len(), 1, "archived repos should be filtered out");
        assert_eq!(repos[0].full_name, "myorg/active-repo");
        assert!(!repos[0].archived);
    }

    // Placeholder comment to avoid adjacent duplicate test resolution issues
    #[tokio::test]
    async fn test_list_org_repos_no_token_via_direct_method() {
        let mock = MockHttpClient::new();
        let client = GitHubClient::with_http_client(GitHubConfig::default(), mock);
        let result = client.list_org_repos("myorg").await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("token"),
            "error should mention token: {err_msg}"
        );
    }

    // --- ScmProvider trait delegation tests ---

    #[tokio::test]
    async fn test_scm_provider_merge_pr() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/1/merge",
            200,
            "{}",
        );

        let client = GitHubClient::with_http_client(test_config(), mock);
        let provider: &dyn ScmProvider = &client;
        let result = provider.merge_pr("owner/repo", 1).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_scm_provider_close_pr() {
        let mock = MockHttpClient::new();
        mock.mock_response("https://api.github.com/repos/owner/repo/pulls/1", 200, "{}");

        let client = GitHubClient::with_http_client(test_config(), mock);
        let provider: &dyn ScmProvider = &client;
        let result = provider.close_pr("owner/repo", 1).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_scm_provider_delete_branch() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/git/refs/heads/branch",
            204,
            "",
        );

        let client = GitHubClient::with_http_client(test_config(), mock);
        let provider: &dyn ScmProvider = &client;
        let result = provider.delete_branch("owner/repo", "branch").await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_scm_provider_pr_url_pattern() {
        let client = GitHubClient::new(test_config());
        let provider: &dyn ScmProvider = &client;
        assert_eq!(provider.pr_url_pattern(), "https://github.com/%");
    }

    #[test]
    fn test_scm_provider_parse_pr_number() {
        let client = GitHubClient::new(test_config());
        let provider: &dyn ScmProvider = &client;
        assert_eq!(
            provider.parse_pr_number("https://github.com/owner/repo/pull/99"),
            Some(99)
        );
        assert_eq!(
            provider.parse_pr_number("https://github.com/owner/repo/issues/5"),
            None
        );
    }

    #[test]
    fn test_scm_provider_name_is_github() {
        let client = GitHubClient::new(test_config());
        let provider: &dyn ScmProvider = &client;
        assert_eq!(provider.name(), "github");
    }

    #[tokio::test]
    async fn test_scm_provider_get_pr_branch() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/1",
            200,
            r#"{"state": "open", "merged": false, "head": {"ref": "my-branch"}, "base": {"ref": "main"}}"#,
        );

        let client = GitHubClient::with_http_client(test_config(), mock);
        let provider: &dyn ScmProvider = &client;
        let branch = provider.get_pr_branch("owner/repo", 1).await.unwrap();
        assert_eq!(branch, "my-branch");
    }

    #[tokio::test]
    async fn test_scm_provider_get_pr_branch_no_head() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/1",
            200,
            r#"{"state": "open", "merged": false}"#,
        );

        let client = GitHubClient::with_http_client(test_config(), mock);
        let provider: &dyn ScmProvider = &client;
        let result = provider.get_pr_branch("owner/repo", 1).await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("No head branch"),
            "error should mention missing branch: {err_msg}"
        );
    }

    #[tokio::test]
    async fn test_get_pr_info_optional_fields_missing() {
        let mock = MockHttpClient::new();
        mock.mock_response(
            "https://api.github.com/repos/owner/repo/pulls/1",
            200,
            r#"{"state": "open", "merged": false}"#,
        );

        let client = GitHubClient::with_http_client(test_config(), mock);
        let info = client.get_pr_info("owner/repo", 1).await.unwrap();
        assert!(info.head_branch.is_none());
        assert!(info.base_branch.is_none());
        assert!(info.title.is_none());
        assert!(info.author.is_none());
    }
}
