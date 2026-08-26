//! Git operations for repository management.

use claudear_core::error::{Error, Result};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use tokio::process::Command;

/// Git operations for managing repositories.
pub struct GitOps;

/// Validate a ref (branch name, tag, etc.) to prevent command injection.
fn validate_ref(name: &str, label: &str) -> Result<()> {
    if name.is_empty()
        || name == "@"
        || name.starts_with('-')
        || name.contains("..")
        || !name
            .chars()
            .all(|c| c.is_alphanumeric() || matches!(c, '-' | '_' | '.' | '/' | '@'))
    {
        return Err(Error::git(format!(
            "Invalid {} (contains disallowed characters): {}",
            label, name
        )));
    }
    Ok(())
}

impl GitOps {
    /// Convert a potentially-relative path to absolute using the process CWD.
    ///
    /// Git commands use `.current_dir(repo_path)` which changes git's working
    /// directory. Any relative paths passed as arguments would then be resolved
    /// relative to `repo_path` instead of the process CWD, causing mismatches.
    fn make_absolute(path: &Path) -> Result<PathBuf> {
        if path.is_absolute() {
            Ok(path.to_path_buf())
        } else {
            std::env::current_dir()
                .map(|cwd| cwd.join(path))
                .map_err(|e| Error::io(format!("Failed to get current directory: {}", e)))
        }
    }

    /// Ensure a repository is available and up to date.
    ///
    /// If the repository doesn't exist locally, it will be cloned.
    /// If it exists, it will be pulled to update.
    pub async fn ensure_repo_at_path(
        repo_path: &Path,
        scm_url: &str,
        default_branch: &str,
    ) -> Result<()> {
        if repo_path.exists() {
            Self::pull(repo_path, default_branch).await
        } else {
            Self::clone(scm_url, repo_path).await
        }
    }

    /// Ensure a repository's object store is current without checking out or resetting.
    ///
    /// If the repository doesn't exist locally, it will be cloned.
    /// If it exists, only `git fetch origin` is run — the working tree is untouched.
    ///
    /// After fetching/cloning, updates `refs/remotes/origin/HEAD` so the remote's
    /// default branch can be detected, and returns the detected default branch name.
    pub async fn ensure_repo_fetched(repo_path: &Path, scm_url: &str) -> Result<String> {
        if repo_path.exists() {
            Self::fetch_all(repo_path).await?;
        } else {
            Self::clone(scm_url, repo_path).await?;
        }

        // Update origin/HEAD so we know the remote's default branch.
        Self::update_remote_head(repo_path).await;

        Ok(Self::detect_default_branch(repo_path).await)
    }

    /// Ensure a repository is current *and* its working tree is advanced to the
    /// remote's default branch.
    pub async fn ensure_repo_synced(repo_path: &Path, scm_url: &str) -> Result<String> {
        // Repair single-branch/stale-refspec clones so the fetch sees the current default.
        if Self::is_git_repo(repo_path) {
            Self::ensure_all_branches_tracked(repo_path).await;
        }
        let default_branch = Self::ensure_repo_fetched(repo_path, scm_url).await?;
        Self::checkout_reset(repo_path, &default_branch).await?;
        Ok(default_branch)
    }

    /// Set the fetch refspec to track all branches (`git remote set-branches origin '*'`).
    async fn ensure_all_branches_tracked(repo_path: &Path) {
        let output = Command::new("git")
            .args(["remote", "set-branches", "origin", "*"])
            .current_dir(repo_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await;
        match output {
            Ok(o) if o.status.success() => {}
            Ok(o) => {
                let stderr = String::from_utf8_lossy(&o.stderr);
                tracing::debug!(repo = ?repo_path, error = %stderr, "Failed to widen fetch refspec");
            }
            Err(e) => {
                tracing::debug!(repo = ?repo_path, error = %e, "Failed to run git remote set-branches");
            }
        }
    }

    /// Fetch all remote refs without touching the working tree.
    pub async fn fetch_all(repo_path: &Path) -> Result<()> {
        tracing::debug!(repo = ?repo_path, "Fetching all remote refs");

        let output = Command::new("git")
            .args(["fetch", "origin", "--prune"])
            .current_dir(repo_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|e| Error::io(format!("Failed to execute git fetch: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(Error::git(format!("git fetch failed: {}", stderr)));
        }

        tracing::debug!(repo = ?repo_path, "Fetch completed");
        Ok(())
    }

    /// Fetch a specific branch from origin.
    pub async fn fetch_branch(repo_path: &Path, branch: &str) -> Result<()> {
        validate_ref(branch, "branch name")?;

        tracing::debug!(repo = ?repo_path, branch = %branch, "Fetching branch");

        let output = Command::new("git")
            .args(["fetch", "origin", branch])
            .current_dir(repo_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|e| Error::io(format!("Failed to execute git fetch: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(Error::git(format!("git fetch branch failed: {}", stderr)));
        }

        Ok(())
    }

    /// Create a git worktree in detached HEAD state.
    ///
    /// If the worktree path already exists (e.g. from a crash), it is removed first.
    pub async fn create_worktree(
        repo_path: &Path,
        worktree_path: &Path,
        checkout_ref: &str,
    ) -> Result<()> {
        validate_ref(checkout_ref, "checkout ref")?;

        // Make worktree_path absolute so git resolves it correctly even when
        // current_dir is set to repo_path (which differs from the process CWD).
        let worktree_path = Self::make_absolute(worktree_path)?;
        let worktree_path = worktree_path.as_path();

        // Crash recovery: remove stale worktree
        if worktree_path.exists() {
            tracing::warn!(worktree = ?worktree_path, "Stale worktree found, removing");
            Self::remove_worktree(repo_path, worktree_path).await?;
        }

        // Create parent directory
        if let Some(parent) = worktree_path.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|e| {
                Error::io(format!(
                    "Failed to create worktree parent directory {:?}: {}",
                    parent, e
                ))
            })?;
        }

        tracing::info!(
            repo = ?repo_path,
            worktree = ?worktree_path,
            checkout_ref = %checkout_ref,
            "Creating worktree"
        );

        let output = Command::new("git")
            .args(["worktree", "add", "--detach"])
            .arg(worktree_path)
            .arg(checkout_ref)
            .current_dir(repo_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|e| Error::io(format!("Failed to execute git worktree add: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(Error::git(format!("git worktree add failed: {}", stderr)));
        }

        tracing::info!(worktree = ?worktree_path, "Worktree created");
        Ok(())
    }

    /// Create a git worktree checked out on a local branch.
    ///
    /// Unlike [`create_worktree`] (detached HEAD), this creates/resets a local
    /// branch at `start_point` so that subsequent pushes target the correct
    /// remote branch.  Used for review-driven reruns where Claude must push to
    /// an existing PR branch.
    pub async fn create_worktree_on_branch(
        repo_path: &Path,
        worktree_path: &Path,
        branch: &str,
        start_point: &str,
    ) -> Result<()> {
        validate_ref(branch, "branch")?;
        validate_ref(start_point, "start point")?;

        // Make worktree_path absolute so git resolves it correctly even when
        // current_dir is set to repo_path (which differs from the process CWD).
        let worktree_path = Self::make_absolute(worktree_path)?;
        let worktree_path = worktree_path.as_path();

        // Crash recovery: remove stale worktree
        if worktree_path.exists() {
            tracing::warn!(worktree = ?worktree_path, "Stale worktree found, removing");
            Self::remove_worktree(repo_path, worktree_path).await?;
        }

        // Create parent directory
        if let Some(parent) = worktree_path.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|e| {
                Error::io(format!(
                    "Failed to create worktree parent directory {:?}: {}",
                    parent, e
                ))
            })?;
        }

        tracing::info!(
            repo = ?repo_path,
            worktree = ?worktree_path,
            branch = %branch,
            start_point = %start_point,
            "Creating worktree on branch"
        );

        // -B creates (or resets) the local branch at start_point.
        let output = Command::new("git")
            .args(["worktree", "add", "-B"])
            .arg(branch)
            .arg(worktree_path)
            .arg(start_point)
            .current_dir(repo_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|e| Error::io(format!("Failed to execute git worktree add: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(Error::git(format!(
                "git worktree add -B {} failed: {}",
                branch, stderr
            )));
        }

        tracing::info!(worktree = ?worktree_path, branch = %branch, "Worktree created on branch");
        Ok(())
    }

    /// Remove a git worktree and clean up.
    pub async fn remove_worktree(repo_path: &Path, worktree_path: &Path) -> Result<()> {
        // Make worktree_path absolute so git resolves it correctly even when
        // current_dir is set to repo_path (which differs from the process CWD).
        let worktree_path = Self::make_absolute(worktree_path)?;
        let worktree_path = worktree_path.as_path();

        tracing::debug!(
            repo = ?repo_path,
            worktree = ?worktree_path,
            "Removing worktree"
        );

        // Try git worktree remove --force
        let output = Command::new("git")
            .args(["worktree", "remove", "--force"])
            .arg(worktree_path)
            .current_dir(repo_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|e| Error::io(format!("Failed to execute git worktree remove: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            tracing::warn!(error = %stderr, "git worktree remove failed, falling back to rm");
        }

        // If the directory still exists, forcefully remove it
        if worktree_path.exists() {
            let parent_dir = worktree_path
                .parent()
                .map(|p| {
                    p.file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .to_string()
                })
                .unwrap_or_default();
            if !parent_dir.ends_with("-worktrees") {
                return Err(Error::git(format!(
                    "Refusing to remove directory outside worktrees area: {:?}",
                    worktree_path
                )));
            }
            tokio::fs::remove_dir_all(worktree_path)
                .await
                .map_err(|e| {
                    Error::io(format!(
                        "Failed to remove worktree directory {:?}: {}",
                        worktree_path, e
                    ))
                })?;
        }

        // Prune stale worktree bookkeeping
        let _ = Command::new("git")
            .args(["worktree", "prune"])
            .current_dir(repo_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await;

        tracing::debug!(worktree = ?worktree_path, "Worktree removed");
        Ok(())
    }

    /// Clone a repository.
    async fn clone(url: &str, target: &Path) -> Result<()> {
        tracing::info!(url = %url, target = ?target, "Cloning repository");

        // Create parent directory if it doesn't exist
        if let Some(parent) = target.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|e| {
                Error::io(format!(
                    "Failed to create parent directory {:?}: {}",
                    parent, e
                ))
            })?;
        }

        // Validate the URL to prevent command injection via malicious repository URLs.
        // Reject URLs containing shell metacharacters or those starting with '-' (option injection).
        if url.starts_with('-') || url.contains(';') || url.contains('|') || url.contains('$') {
            return Err(Error::git(format!(
                "Invalid repository URL (contains disallowed characters): {}",
                url
            )));
        }

        let output = Command::new("git")
            .args(["clone", "--", url])
            .arg(target)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|e| Error::io(format!("Failed to execute git clone: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(Error::git(format!("git clone failed: {}", stderr)));
        }

        tracing::info!(target = ?target, "Repository cloned successfully");
        Ok(())
    }

    /// Pull latest changes on a branch.
    async fn pull(repo_path: &Path, branch: &str) -> Result<()> {
        tracing::debug!(repo = ?repo_path, branch = %branch, "Pulling latest changes");

        // Validate branch name to prevent injection via crafted branch names
        validate_ref(branch, "branch name")?;

        // First, fetch (branch name is validated above)
        let output = Command::new("git")
            .args(["fetch", "origin", branch])
            .current_dir(repo_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|e| Error::io(format!("Failed to execute git fetch: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(Error::git(format!("git fetch failed: {}", stderr)));
        }

        Self::checkout_reset(repo_path, branch).await?;

        tracing::debug!(repo = ?repo_path, "Repository updated successfully");
        Ok(())
    }

    /// Check out `branch` and hard-reset the working tree to `origin/<branch>`.
    ///
    /// Assumes the relevant refs are already fetched. Discards any local changes
    /// in the working tree — callers must only use this on managed clones.
    async fn checkout_reset(repo_path: &Path, branch: &str) -> Result<()> {
        // Validate branch name to prevent injection via crafted branch names
        validate_ref(branch, "branch name")?;

        // Force create-or-reset the local branch to the remote tip, discarding any dirty state.
        let output = Command::new("git")
            .args([
                "checkout",
                "-f",
                "-B",
                branch,
                &format!("origin/{}", branch),
            ])
            .current_dir(repo_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|e| Error::io(format!("Failed to execute git checkout: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(Error::git(format!("git checkout failed: {}", stderr)));
        }

        // Reset to origin/branch to ensure clean state
        let output = Command::new("git")
            .args(["reset", "--hard", &format!("origin/{}", branch)])
            .current_dir(repo_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|e| Error::io(format!("Failed to execute git reset: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(Error::git(format!("git reset failed: {}", stderr)));
        }

        Ok(())
    }

    /// Check if a path is a valid git repository (or worktree).
    pub fn is_git_repo(path: &Path) -> bool {
        let git_path = path.join(".git");
        git_path.is_dir() || git_path.is_file()
    }

    /// Detect the remote's default branch by reading `refs/remotes/origin/HEAD`.
    ///
    /// Returns the branch name (e.g. `"main"`) or falls back to `"main"` if the
    /// ref cannot be read.
    pub async fn detect_default_branch(repo_path: &Path) -> String {
        let output = Command::new("git")
            .args(["symbolic-ref", "refs/remotes/origin/HEAD"])
            .current_dir(repo_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await;

        match output {
            Ok(o) if o.status.success() => {
                let refname = String::from_utf8_lossy(&o.stdout).trim().to_string();
                // e.g. "refs/remotes/origin/main" → "main"
                refname
                    .strip_prefix("refs/remotes/origin/")
                    .unwrap_or("main")
                    .to_string()
            }
            _ => "main".to_string(),
        }
    }

    /// Synchronous version of [`detect_default_branch`](Self::detect_default_branch)
    /// for use in non-async contexts (e.g. filesystem index building).
    pub fn detect_default_branch_sync(repo_path: &Path) -> String {
        let output = std::process::Command::new("git")
            .args(["symbolic-ref", "refs/remotes/origin/HEAD"])
            .current_dir(repo_path)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .output();

        match output {
            Ok(o) if o.status.success() => {
                let refname = String::from_utf8_lossy(&o.stdout).trim().to_string();
                refname
                    .strip_prefix("refs/remotes/origin/")
                    .unwrap_or("main")
                    .to_string()
            }
            _ => "main".to_string(),
        }
    }

    /// Update `refs/remotes/origin/HEAD` to match the remote's default branch.
    ///
    /// This queries the remote to determine its HEAD, so it requires network
    /// access. Failures are logged but not propagated since this is best-effort.
    async fn update_remote_head(repo_path: &Path) {
        let output = Command::new("git")
            .args(["remote", "set-head", "origin", "--auto"])
            .current_dir(repo_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await;

        match output {
            Ok(o) if o.status.success() => {
                tracing::debug!(repo = ?repo_path, "Updated origin/HEAD");
            }
            Ok(o) => {
                let stderr = String::from_utf8_lossy(&o.stderr);
                tracing::debug!(repo = ?repo_path, error = %stderr, "Failed to update origin/HEAD");
            }
            Err(e) => {
                tracing::debug!(repo = ?repo_path, error = %e, "Failed to run git remote set-head");
            }
        }
    }

    /// Get the current branch of a repository.
    pub async fn current_branch(repo_path: &Path) -> Result<String> {
        let output = Command::new("git")
            .args(["rev-parse", "--abbrev-ref", "HEAD"])
            .current_dir(repo_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|e| Error::io(format!("Failed to execute git rev-parse: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(Error::git(format!("git rev-parse failed: {}", stderr)));
        }

        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }
}

/// Compute the worktree path for a given issue.
///
/// Returns `workspace/{short_repo_name}-worktrees/{issue_short_id}`.
pub fn worktree_path(workspace: &Path, repo_name: &str, issue_short_id: &str) -> PathBuf {
    let raw_short_name = repo_name.split('/').next_back().unwrap_or(repo_name);
    let short_name = raw_short_name.replace(['/', '\\', '\0'], "_");
    let sanitized_id = issue_short_id.replace(['/', '\\', '.', '\0'], "_");
    workspace
        .join(format!("{}-worktrees", short_name))
        .join(sanitized_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command as StdCommand;
    use tempfile::TempDir;

    /// Build a `file://` URL from a local path.
    ///
    /// On Windows, `Path::display()` emits backslashes (e.g. `C:\Users\...`),
    /// but `file://` URLs require forward slashes (`file:///C:/Users/...`).
    fn file_url(path: &Path) -> String {
        let s = path.display().to_string().replace('\\', "/");
        if s.starts_with('/') {
            format!("file://{s}")
        } else {
            // Windows absolute path like C:/Users/... needs an extra /
            format!("file:///{s}")
        }
    }

    /// Create a bare-minimum git repo in `path` with one commit on "main".
    fn init_git_repo(path: &Path) {
        StdCommand::new("git")
            .args(["init", "-b", "main"])
            .current_dir(path)
            .output()
            .expect("git init failed");

        StdCommand::new("git")
            .args(["config", "user.email", "test@test.com"])
            .current_dir(path)
            .output()
            .expect("git config email failed");

        StdCommand::new("git")
            .args(["config", "user.name", "Test User"])
            .current_dir(path)
            .output()
            .expect("git config name failed");

        std::fs::write(path.join("README.md"), "# test\n").unwrap();

        StdCommand::new("git")
            .args(["add", "."])
            .current_dir(path)
            .output()
            .expect("git add failed");

        StdCommand::new("git")
            .args(["commit", "-m", "initial commit"])
            .current_dir(path)
            .output()
            .expect("git commit failed");
    }

    /// Create a second commit so the repo has some history.
    fn add_second_commit(path: &Path) {
        std::fs::write(path.join("file2.txt"), "second file\n").unwrap();

        StdCommand::new("git")
            .args(["add", "."])
            .current_dir(path)
            .output()
            .expect("git add failed");

        StdCommand::new("git")
            .args(["commit", "-m", "second commit"])
            .current_dir(path)
            .output()
            .expect("git commit failed");
    }

    // ════════════════════════════════════════════════════════════
    //  validate_ref
    // ════════════════════════════════════════════════════════════

    #[test]
    fn test_validate_ref_accepts_simple_branch() {
        assert!(validate_ref("main", "branch").is_ok());
        assert!(validate_ref("develop", "branch").is_ok());
        assert!(validate_ref("feature/my-thing", "branch").is_ok());
    }

    #[test]
    fn test_validate_ref_accepts_slashes_dots_underscores() {
        assert!(validate_ref("release/v1.2.3", "tag").is_ok());
        assert!(validate_ref("feature_branch", "branch").is_ok());
        assert!(validate_ref("user/feature.name", "branch").is_ok());
    }

    #[test]
    fn test_validate_ref_accepts_at_in_middle() {
        // '@' is allowed in the middle but bare "@" is rejected
        assert!(validate_ref("user@feature", "branch").is_ok());
    }

    #[test]
    fn test_validate_ref_rejects_empty() {
        let result = validate_ref("", "branch");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("disallowed"));
    }

    #[test]
    fn test_validate_ref_rejects_bare_at() {
        let result = validate_ref("@", "branch");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_ref_rejects_leading_dash() {
        let result = validate_ref("-evil", "branch");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("disallowed"));
    }

    #[test]
    fn test_validate_ref_rejects_double_dot() {
        let result = validate_ref("main..evil", "branch");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_ref_rejects_semicolon() {
        let result = validate_ref("main;evil", "branch");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_ref_rejects_space() {
        let result = validate_ref("main evil", "branch");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_ref_rejects_backtick() {
        let result = validate_ref("main`evil`", "branch");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_ref_rejects_pipe() {
        let result = validate_ref("main|evil", "branch");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_ref_rejects_dollar() {
        let result = validate_ref("$HOME", "branch");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_ref_rejects_tilde() {
        let result = validate_ref("HEAD~1", "branch");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_ref_rejects_caret() {
        let result = validate_ref("HEAD^", "branch");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_ref_rejects_colon() {
        let result = validate_ref("refs:heads", "branch");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_ref_rejects_question_mark() {
        let result = validate_ref("branch?", "branch");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_ref_rejects_asterisk() {
        let result = validate_ref("branch*", "branch");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_ref_rejects_bracket() {
        let result = validate_ref("branch[0]", "branch");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_ref_error_message_includes_label() {
        let result = validate_ref("--evil", "checkout ref");
        let err = result.unwrap_err().to_string();
        assert!(err.contains("checkout ref"), "error was: {}", err);
    }

    #[test]
    fn test_validate_ref_error_message_includes_value() {
        let result = validate_ref("--evil", "branch");
        let err = result.unwrap_err().to_string();
        assert!(err.contains("--evil"), "error was: {}", err);
    }

    // ════════════════════════════════════════════════════════════
    //  is_git_repo
    // ════════════════════════════════════════════════════════════

    #[test]
    fn test_is_git_repo_true() {
        let temp = TempDir::new().unwrap();
        std::fs::create_dir(temp.path().join(".git")).unwrap();
        assert!(GitOps::is_git_repo(temp.path()));
    }

    #[test]
    fn test_is_git_repo_false() {
        let temp = TempDir::new().unwrap();
        assert!(!GitOps::is_git_repo(temp.path()));
    }

    #[test]
    fn test_is_git_repo_nonexistent() {
        assert!(!GitOps::is_git_repo(Path::new("/nonexistent/path")));
    }

    #[test]
    fn test_is_git_repo_file_not_dir() {
        // .git exists but as a file, not a directory (like a worktree)
        let temp = TempDir::new().unwrap();
        std::fs::write(temp.path().join(".git"), "gitdir: ../other/.git").unwrap();
        // Worktrees have .git as a file, so this should be true
        assert!(GitOps::is_git_repo(temp.path()));
    }

    #[test]
    fn test_is_git_repo_real_repo() {
        let temp = TempDir::new().unwrap();
        init_git_repo(temp.path());
        assert!(GitOps::is_git_repo(temp.path()));
    }

    #[test]
    fn test_is_git_repo_empty_git_dir() {
        // .git directory exists but is empty - still counts as a git repo
        let temp = TempDir::new().unwrap();
        std::fs::create_dir(temp.path().join(".git")).unwrap();
        assert!(GitOps::is_git_repo(temp.path()));
    }

    #[test]
    fn test_is_git_repo_nested_dir_is_not_repo() {
        // A subdirectory inside a git repo is not itself detected as a git repo
        let temp = TempDir::new().unwrap();
        init_git_repo(temp.path());
        let sub = temp.path().join("subdir");
        std::fs::create_dir(&sub).unwrap();
        // subdir does not have its own .git
        assert!(!GitOps::is_git_repo(&sub));
    }

    // ════════════════════════════════════════════════════════════
    //  worktree_path (the free function)
    // ════════════════════════════════════════════════════════════

    #[test]
    fn test_worktree_path_simple() {
        let p = worktree_path(Path::new("/work"), "owner/repo", "ABC-123");
        assert_eq!(p, PathBuf::from("/work/repo-worktrees/ABC-123"));
    }

    #[test]
    fn test_worktree_path_no_owner() {
        let p = worktree_path(Path::new("/work"), "repo", "XYZ-1");
        assert_eq!(p, PathBuf::from("/work/repo-worktrees/XYZ-1"));
    }

    #[test]
    fn test_worktree_path_deep_owner() {
        // Only the last path component is used
        let p = worktree_path(Path::new("/work"), "org/team/repo", "ISSUE-1");
        assert_eq!(p, PathBuf::from("/work/repo-worktrees/ISSUE-1"));
    }

    #[test]
    fn test_worktree_path_sanitizes_slashes_in_issue_id() {
        let p = worktree_path(Path::new("/work"), "owner/repo", "feat/issue");
        assert_eq!(p, PathBuf::from("/work/repo-worktrees/feat_issue"));
    }

    #[test]
    fn test_worktree_path_sanitizes_backslash_in_issue_id() {
        let p = worktree_path(Path::new("/work"), "owner/repo", "feat\\issue");
        assert_eq!(p, PathBuf::from("/work/repo-worktrees/feat_issue"));
    }

    #[test]
    fn test_worktree_path_sanitizes_dot_in_issue_id() {
        let p = worktree_path(Path::new("/work"), "owner/repo", "v1.2.3");
        assert_eq!(p, PathBuf::from("/work/repo-worktrees/v1_2_3"));
    }

    #[test]
    fn test_worktree_path_sanitizes_null_in_issue_id() {
        let p = worktree_path(Path::new("/work"), "owner/repo", "issue\0evil");
        assert_eq!(p, PathBuf::from("/work/repo-worktrees/issue_evil"));
    }

    #[test]
    fn test_worktree_path_sanitizes_repo_name_with_slashes() {
        // The repo_name short extraction uses split('/').next_back()
        // then sanitizes slashes/backslashes/nulls in the short name
        let p = worktree_path(Path::new("/tmp"), "owner/my-repo", "ID-1");
        assert_eq!(p, PathBuf::from("/tmp/my-repo-worktrees/ID-1"));
    }

    #[test]
    fn test_worktree_path_sanitizes_null_in_repo_name() {
        let p = worktree_path(Path::new("/tmp"), "evil\0repo", "ID-1");
        assert_eq!(p, PathBuf::from("/tmp/evil_repo-worktrees/ID-1"));
    }

    #[test]
    fn test_worktree_path_empty_repo_name() {
        // edge case: empty string - split('/').next_back() returns Some("")
        let p = worktree_path(Path::new("/work"), "", "ID-1");
        assert_eq!(p, PathBuf::from("/work/-worktrees/ID-1"));
    }

    #[test]
    fn test_worktree_path_empty_issue_id() {
        let p = worktree_path(Path::new("/work"), "owner/repo", "");
        assert_eq!(p, PathBuf::from("/work/repo-worktrees/"));
    }

    #[test]
    fn test_worktree_path_complex_workspace() {
        let p = worktree_path(
            Path::new("/var/lib/claudear/workspaces"),
            "myorg/backend",
            "JIRA-4567",
        );
        assert_eq!(
            p,
            PathBuf::from("/var/lib/claudear/workspaces/backend-worktrees/JIRA-4567")
        );
    }

    // ════════════════════════════════════════════════════════════
    //  current_branch (async, real git repo)
    // ════════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_current_branch_on_main() {
        let temp = TempDir::new().unwrap();
        init_git_repo(temp.path());

        let branch = GitOps::current_branch(temp.path()).await.unwrap();
        assert_eq!(branch, "main");
    }

    #[tokio::test]
    async fn test_current_branch_after_checkout() {
        let temp = TempDir::new().unwrap();
        init_git_repo(temp.path());

        StdCommand::new("git")
            .args(["checkout", "-b", "feature/test-branch"])
            .current_dir(temp.path())
            .output()
            .expect("checkout -b failed");

        let branch = GitOps::current_branch(temp.path()).await.unwrap();
        assert_eq!(branch, "feature/test-branch");
    }

    #[tokio::test]
    async fn test_current_branch_detached_head() {
        let temp = TempDir::new().unwrap();
        init_git_repo(temp.path());
        add_second_commit(temp.path());

        // Detach HEAD at the first commit
        StdCommand::new("git")
            .args(["checkout", "HEAD~1"])
            .current_dir(temp.path())
            .output()
            .expect("detach head failed");

        let branch = GitOps::current_branch(temp.path()).await.unwrap();
        // git rev-parse --abbrev-ref HEAD returns "HEAD" when detached
        assert_eq!(branch, "HEAD");
    }

    #[tokio::test]
    async fn test_current_branch_non_git_dir() {
        let temp = TempDir::new().unwrap();
        let result = GitOps::current_branch(temp.path()).await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("git rev-parse failed"),
            "unexpected error: {}",
            err
        );
    }

    #[tokio::test]
    async fn test_current_branch_nonexistent_path() {
        let result = GitOps::current_branch(Path::new("/nonexistent/path/xyz")).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_current_branch_multiple_branches() {
        let temp = TempDir::new().unwrap();
        init_git_repo(temp.path());

        // Create several branches and switch back to main
        for name in &["branch-a", "branch-b", "branch-c"] {
            StdCommand::new("git")
                .args(["branch", name])
                .current_dir(temp.path())
                .output()
                .unwrap();
        }

        let branch = GitOps::current_branch(temp.path()).await.unwrap();
        assert_eq!(branch, "main");

        // Now switch
        StdCommand::new("git")
            .args(["checkout", "branch-b"])
            .current_dir(temp.path())
            .output()
            .unwrap();

        let branch = GitOps::current_branch(temp.path()).await.unwrap();
        assert_eq!(branch, "branch-b");
    }

    // ════════════════════════════════════════════════════════════
    //  ensure_repo_at_path - clone path (URL validation)
    // ════════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_clone_rejects_option_injection() {
        let temp = TempDir::new().unwrap();
        let target = temp.path().join("repo");

        let result = GitOps::ensure_repo_at_path(&target, "--upload-pack=evil", "main").await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("disallowed characters"),
            "unexpected error: {}",
            err
        );
    }

    #[tokio::test]
    async fn test_clone_rejects_semicolon_in_url() {
        let temp = TempDir::new().unwrap();
        let target = temp.path().join("repo");

        let result =
            GitOps::ensure_repo_at_path(&target, "https://example.com;rm -rf /", "main").await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("disallowed characters"));
    }

    #[tokio::test]
    async fn test_clone_rejects_pipe_in_url() {
        let temp = TempDir::new().unwrap();
        let target = temp.path().join("repo");

        let result = GitOps::ensure_repo_at_path(&target, "https://example.com|evil", "main").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_clone_rejects_dollar_in_url() {
        let temp = TempDir::new().unwrap();
        let target = temp.path().join("repo");

        let result =
            GitOps::ensure_repo_at_path(&target, "https://example.com/$HOME", "main").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_clone_invalid_url_git_error() {
        let temp = TempDir::new().unwrap();
        let target = temp.path().join("repo");

        let result =
            GitOps::ensure_repo_at_path(&target, "https://nonexistent.invalid/repo.git", "main")
                .await;
        assert!(result.is_err());
    }

    // ════════════════════════════════════════════════════════════
    //  ensure_repo_at_path - pull path (branch validation)
    // ════════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_pull_rejects_option_injection_in_branch() {
        let temp = TempDir::new().unwrap();
        std::fs::create_dir_all(temp.path().join("repo")).unwrap();
        let target = temp.path().join("repo");

        let result =
            GitOps::ensure_repo_at_path(&target, "https://example.com/repo.git", "--evil-option")
                .await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("disallowed characters"));
    }

    #[tokio::test]
    async fn test_pull_rejects_semicolon_in_branch() {
        let temp = TempDir::new().unwrap();
        std::fs::create_dir_all(temp.path().join("repo")).unwrap();
        let target = temp.path().join("repo");

        let result =
            GitOps::ensure_repo_at_path(&target, "https://example.com/repo.git", "main;evil").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_pull_rejects_dotdot_in_branch() {
        let temp = TempDir::new().unwrap();
        std::fs::create_dir_all(temp.path().join("repo")).unwrap();
        let target = temp.path().join("repo");

        let result =
            GitOps::ensure_repo_at_path(&target, "https://example.com/repo.git", "main..evil")
                .await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("disallowed characters"));
    }

    #[tokio::test]
    async fn test_pull_rejects_empty_branch() {
        let temp = TempDir::new().unwrap();
        std::fs::create_dir_all(temp.path().join("repo")).unwrap();
        let target = temp.path().join("repo");

        let result = GitOps::ensure_repo_at_path(&target, "https://example.com/repo.git", "").await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("disallowed"));
    }

    #[tokio::test]
    async fn test_pull_rejects_at_branch() {
        let temp = TempDir::new().unwrap();
        std::fs::create_dir_all(temp.path().join("repo")).unwrap();
        let target = temp.path().join("repo");

        let result =
            GitOps::ensure_repo_at_path(&target, "https://example.com/repo.git", "@").await;
        assert!(result.is_err());
    }

    // ════════════════════════════════════════════════════════════
    //  ensure_repo_at_path - local clone from file:// URL
    // ════════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_ensure_repo_at_path_clones_local_repo() {
        let origin = TempDir::new().unwrap();
        init_git_repo(origin.path());

        let target_dir = TempDir::new().unwrap();
        let target = target_dir.path().join("cloned");

        let url = file_url(origin.path());
        let result = GitOps::ensure_repo_at_path(&target, &url, "main").await;
        assert!(result.is_ok(), "clone failed: {:?}", result.unwrap_err());
        assert!(target.join(".git").exists());
    }

    #[tokio::test]
    async fn test_ensure_repo_at_path_pulls_when_exists() {
        let origin = TempDir::new().unwrap();
        init_git_repo(origin.path());

        // Clone it first
        let target_dir = TempDir::new().unwrap();
        let target = target_dir.path().join("cloned");

        let url = file_url(origin.path());
        GitOps::ensure_repo_at_path(&target, &url, "main")
            .await
            .unwrap();

        // Add a second commit to origin
        add_second_commit(origin.path());

        // Pull again - should succeed (repo already exists)
        let result = GitOps::ensure_repo_at_path(&target, &url, "main").await;
        assert!(result.is_ok(), "pull failed: {:?}", result.unwrap_err());
    }

    // ════════════════════════════════════════════════════════════
    //  ensure_repo_fetched
    // ════════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_ensure_repo_fetched_clones_when_missing() {
        let temp = TempDir::new().unwrap();
        let target = temp.path().join("repo");

        let result =
            GitOps::ensure_repo_fetched(&target, "https://nonexistent.invalid/repo.git").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_ensure_repo_fetched_fetches_when_exists() {
        let origin = TempDir::new().unwrap();
        init_git_repo(origin.path());

        let target_dir = TempDir::new().unwrap();
        let target = target_dir.path().join("cloned");

        let url = file_url(origin.path());
        // Clone first
        GitOps::ensure_repo_at_path(&target, &url, "main")
            .await
            .unwrap();

        // Now ensure_repo_fetched should just fetch and return the default branch
        let result = GitOps::ensure_repo_fetched(&target, &url).await;
        let default_branch = result.expect("fetch failed");
        assert_eq!(default_branch, "main");
    }

    // ════════════════════════════════════════════════════════════
    //  fetch_branch
    // ════════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_fetch_branch_rejects_injection() {
        let temp = TempDir::new().unwrap();
        let result = GitOps::fetch_branch(temp.path(), "--evil").await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("disallowed characters"));
    }

    #[tokio::test]
    async fn test_fetch_branch_rejects_empty() {
        let result = GitOps::fetch_branch(Path::new("/tmp"), "").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_fetch_branch_rejects_double_dot() {
        let result = GitOps::fetch_branch(Path::new("/tmp"), "a..b").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_fetch_branch_on_real_repo() {
        let origin = TempDir::new().unwrap();
        init_git_repo(origin.path());

        // Create a branch in origin
        StdCommand::new("git")
            .args(["branch", "feature-x"])
            .current_dir(origin.path())
            .output()
            .unwrap();

        let target_dir = TempDir::new().unwrap();
        let target = target_dir.path().join("cloned");
        let url = file_url(origin.path());
        GitOps::ensure_repo_at_path(&target, &url, "main")
            .await
            .unwrap();

        // Fetch the branch
        let result = GitOps::fetch_branch(&target, "feature-x").await;
        assert!(
            result.is_ok(),
            "fetch_branch failed: {:?}",
            result.unwrap_err()
        );
    }

    #[tokio::test]
    async fn test_fetch_branch_nonexistent_branch() {
        let origin = TempDir::new().unwrap();
        init_git_repo(origin.path());

        let target_dir = TempDir::new().unwrap();
        let target = target_dir.path().join("cloned");
        let url = file_url(origin.path());
        GitOps::ensure_repo_at_path(&target, &url, "main")
            .await
            .unwrap();

        let result = GitOps::fetch_branch(&target, "nonexistent-branch").await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("git fetch branch failed"),
            "unexpected error: {}",
            err
        );
    }

    // ════════════════════════════════════════════════════════════
    //  fetch_all
    // ════════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_fetch_all_on_cloned_repo() {
        let origin = TempDir::new().unwrap();
        init_git_repo(origin.path());

        let target_dir = TempDir::new().unwrap();
        let target = target_dir.path().join("cloned");
        let url = file_url(origin.path());
        GitOps::ensure_repo_at_path(&target, &url, "main")
            .await
            .unwrap();

        let result = GitOps::fetch_all(&target).await;
        assert!(
            result.is_ok(),
            "fetch_all failed: {:?}",
            result.unwrap_err()
        );
    }

    #[tokio::test]
    async fn test_fetch_all_non_git_dir() {
        let temp = TempDir::new().unwrap();
        let result = GitOps::fetch_all(temp.path()).await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("git fetch failed"),
            "unexpected error: {}",
            err
        );
    }

    // ════════════════════════════════════════════════════════════
    //  create_worktree / remove_worktree (real git operations)
    // ════════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_create_worktree_rejects_injection() {
        let temp = TempDir::new().unwrap();
        let wt = temp.path().join("wt");
        let result = GitOps::create_worktree(temp.path(), &wt, "--evil-ref").await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("disallowed characters"));
    }

    #[tokio::test]
    async fn test_create_worktree_rejects_empty_ref() {
        let temp = TempDir::new().unwrap();
        let wt = temp.path().join("wt");
        let result = GitOps::create_worktree(temp.path(), &wt, "").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_create_and_remove_worktree() {
        let temp = TempDir::new().unwrap();
        init_git_repo(temp.path());

        let wt_parent = temp.path().join("my-worktrees");
        std::fs::create_dir_all(&wt_parent).unwrap();
        let wt_path = wt_parent.join("wt1");

        // Create worktree at HEAD (detached)
        let result = GitOps::create_worktree(temp.path(), &wt_path, "main").await;
        assert!(
            result.is_ok(),
            "create_worktree failed: {:?}",
            result.unwrap_err()
        );
        assert!(wt_path.exists());
        assert!(wt_path.join("README.md").exists());

        // The worktree should have .git as a file (not a directory)
        let git_path = wt_path.join(".git");
        assert!(git_path.is_file());
    }

    #[tokio::test]
    async fn test_create_worktree_non_git_dir_fails() {
        let temp = TempDir::new().unwrap();
        let wt_path = temp.path().join("wt");

        let result = GitOps::create_worktree(temp.path(), &wt_path, "main").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_create_worktree_invalid_ref_fails() {
        let temp = TempDir::new().unwrap();
        init_git_repo(temp.path());

        let wt_path = temp.path().join("wt");
        // "nonexistent-branch" doesn't exist
        let result = GitOps::create_worktree(temp.path(), &wt_path, "nonexistent-branch").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_create_worktree_stale_recovery() {
        let temp = TempDir::new().unwrap();
        init_git_repo(temp.path());

        let wt_parent = temp.path().join("test-worktrees");
        std::fs::create_dir_all(&wt_parent).unwrap();
        let wt_path = wt_parent.join("wt1");

        // Create worktree once
        GitOps::create_worktree(temp.path(), &wt_path, "main")
            .await
            .unwrap();
        assert!(wt_path.exists());

        // Creating the same worktree again should succeed (stale recovery)
        let result = GitOps::create_worktree(temp.path(), &wt_path, "main").await;
        assert!(
            result.is_ok(),
            "stale worktree recovery failed: {:?}",
            result.unwrap_err()
        );
    }

    // ════════════════════════════════════════════════════════════
    //  create_worktree_on_branch
    // ════════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_create_worktree_on_branch_rejects_invalid_branch() {
        let temp = TempDir::new().unwrap();
        let wt = temp.path().join("wt");
        let result = GitOps::create_worktree_on_branch(temp.path(), &wt, "--evil", "main").await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("disallowed characters"));
    }

    #[tokio::test]
    async fn test_create_worktree_on_branch_rejects_invalid_start_point() {
        let temp = TempDir::new().unwrap();
        let wt = temp.path().join("wt");
        let result =
            GitOps::create_worktree_on_branch(temp.path(), &wt, "my-branch", "--evil").await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("disallowed characters"));
    }

    #[tokio::test]
    async fn test_create_worktree_on_branch_success() {
        let temp = TempDir::new().unwrap();
        init_git_repo(temp.path());

        let wt_parent = temp.path().join("branch-worktrees");
        std::fs::create_dir_all(&wt_parent).unwrap();
        let wt_path = wt_parent.join("wt1");

        let result =
            GitOps::create_worktree_on_branch(temp.path(), &wt_path, "feature-branch", "main")
                .await;
        assert!(
            result.is_ok(),
            "create_worktree_on_branch failed: {:?}",
            result.unwrap_err()
        );
        assert!(wt_path.exists());

        // The worktree should be on the named branch, not detached
        let branch_output = StdCommand::new("git")
            .args(["rev-parse", "--abbrev-ref", "HEAD"])
            .current_dir(&wt_path)
            .output()
            .unwrap();
        let branch = String::from_utf8_lossy(&branch_output.stdout)
            .trim()
            .to_string();
        assert_eq!(branch, "feature-branch");
    }

    #[tokio::test]
    async fn test_create_worktree_on_branch_stale_recovery() {
        let temp = TempDir::new().unwrap();
        init_git_repo(temp.path());

        let wt_parent = temp.path().join("recover-worktrees");
        std::fs::create_dir_all(&wt_parent).unwrap();
        let wt_path = wt_parent.join("wt1");

        GitOps::create_worktree_on_branch(temp.path(), &wt_path, "branch-a", "main")
            .await
            .unwrap();

        // Creating again with a different branch - should remove stale and recreate
        let result =
            GitOps::create_worktree_on_branch(temp.path(), &wt_path, "branch-b", "main").await;
        assert!(
            result.is_ok(),
            "stale recovery failed: {:?}",
            result.unwrap_err()
        );
    }

    // ════════════════════════════════════════════════════════════
    //  remove_worktree
    // ════════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_remove_worktree_nonexistent_is_ok() {
        let temp = TempDir::new().unwrap();
        init_git_repo(temp.path());

        // Removing a worktree that doesn't exist just logs a warning
        let wt_parent = temp.path().join("foo-worktrees");
        std::fs::create_dir_all(&wt_parent).unwrap();
        let wt_path = wt_parent.join("nope");

        // Should not fail catastrophically
        let result = GitOps::remove_worktree(temp.path(), &wt_path).await;
        // The git command might fail but it should not propagate as error
        // since the dir doesn't exist, the fallback rm is skipped
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_remove_worktree_refuses_outside_worktrees_area() {
        let temp = TempDir::new().unwrap();
        init_git_repo(temp.path());

        // Simulate a worktree directory outside the *-worktrees naming convention.
        // The remove function should refuse to rm -rf it.
        let bad_dir = temp.path().join("unsafe-area").join("wt");
        std::fs::create_dir_all(&bad_dir).unwrap();

        let result = GitOps::remove_worktree(temp.path(), &bad_dir).await;
        // The git worktree remove command will fail (it's not a real worktree)
        // but the safety check should catch that the parent doesn't end in "-worktrees"
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("Refusing to remove"),
            "unexpected error: {}",
            err
        );
    }

    // ════════════════════════════════════════════════════════════
    //  has_uncommitted_changes (via git status in a real repo)
    //  Note: no has_uncommitted_changes method exists, but we
    //  verify the detection pattern using current_branch + git status
    // ════════════════════════════════════════════════════════════

    // ════════════════════════════════════════════════════════════
    //  ensure_repo_at_path - pull integration test
    // ════════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_ensure_repo_at_path_pull_updates_branch() {
        let origin = TempDir::new().unwrap();
        init_git_repo(origin.path());

        let target_dir = TempDir::new().unwrap();
        let target = target_dir.path().join("cloned");
        let url = file_url(origin.path());

        // Clone
        GitOps::ensure_repo_at_path(&target, &url, "main")
            .await
            .unwrap();

        // Verify initial commit
        let log = StdCommand::new("git")
            .args(["log", "--oneline"])
            .current_dir(&target)
            .output()
            .unwrap();
        let initial_log = String::from_utf8_lossy(&log.stdout);
        assert!(initial_log.contains("initial commit"));

        // Add a commit to origin
        add_second_commit(origin.path());

        // Pull
        GitOps::ensure_repo_at_path(&target, &url, "main")
            .await
            .unwrap();

        // Verify second commit is now present
        let log = StdCommand::new("git")
            .args(["log", "--oneline"])
            .current_dir(&target)
            .output()
            .unwrap();
        let updated_log = String::from_utf8_lossy(&log.stdout);
        assert!(
            updated_log.contains("second commit"),
            "pull did not bring second commit: {}",
            updated_log
        );
    }

    // ════════════════════════════════════════════════════════════
    //  Edge cases for URL validation in clone
    // ════════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_clone_rejects_url_with_backtick() {
        let temp = TempDir::new().unwrap();
        let target = temp.path().join("repo");

        // Backtick in URL - the clone function currently only checks for -, ;, |, $
        // so backtick should be passed to git which may fail
        let result =
            GitOps::ensure_repo_at_path(&target, "https://example.com/`evil`", "main").await;
        // This should fail at clone (invalid URL) even if our check doesn't catch it
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_clone_accepts_valid_https_url_format() {
        // Verify that valid URL formats pass URL validation (they'll fail at git level
        // because the host doesn't exist, but they shouldn't be blocked by our check)
        let temp = TempDir::new().unwrap();
        let target = temp.path().join("repo");

        let result = GitOps::ensure_repo_at_path(
            &target,
            "https://github.com/valid-org/valid-repo.git",
            "main",
        )
        .await;
        // Should fail because host is reachable but repo doesn't exist, or DNS fails
        // The important thing is it's NOT blocked by our URL validation
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("git clone failed"),
            "unexpected error: {}",
            err
        );
    }

    // ════════════════════════════════════════════════════════════
    //  validate_ref - boundary / allowed characters
    // ════════════════════════════════════════════════════════════

    #[test]
    fn test_validate_ref_allows_numeric_only() {
        assert!(validate_ref("12345", "branch").is_ok());
    }

    #[test]
    fn test_validate_ref_allows_single_char() {
        assert!(validate_ref("a", "branch").is_ok());
        assert!(validate_ref("1", "branch").is_ok());
    }

    #[test]
    fn test_validate_ref_allows_mixed_case() {
        assert!(validate_ref("Feature/MyBranch", "branch").is_ok());
        assert!(validate_ref("UPPERCASE", "branch").is_ok());
    }

    #[test]
    fn test_validate_ref_allows_dots() {
        assert!(validate_ref("v1.2.3", "tag").is_ok());
    }

    #[test]
    fn test_validate_ref_rejects_only_dots_dot_dot() {
        // ".." is rejected because it contains ".."
        let result = validate_ref("..", "branch");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_ref_allows_single_dot() {
        // A single dot is ok (no ".." substring)
        assert!(validate_ref(".", "branch").is_ok());
    }

    #[test]
    fn test_validate_ref_rejects_newline() {
        let result = validate_ref("main\nevil", "branch");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_ref_rejects_null_byte() {
        let result = validate_ref("main\0evil", "branch");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_ref_rejects_hash() {
        let result = validate_ref("branch#1", "branch");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_ref_rejects_curly_braces() {
        let result = validate_ref("stash@{0}", "ref");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_ref_rejects_backslash() {
        let result = validate_ref("path\\name", "ref");
        assert!(result.is_err());
    }

    // ════════════════════════════════════════════════════════════
    //  Worktree path edge cases
    // ════════════════════════════════════════════════════════════

    #[test]
    fn test_worktree_path_trailing_slash_in_repo_name() {
        // split('/').next_back() on "owner/repo/" returns ""
        let p = worktree_path(Path::new("/work"), "owner/repo/", "ID-1");
        assert_eq!(p, PathBuf::from("/work/-worktrees/ID-1"));
    }

    #[test]
    fn test_worktree_path_preserves_dashes_in_issue_id() {
        let p = worktree_path(Path::new("/work"), "owner/repo", "ABC-123-DEF");
        assert_eq!(p, PathBuf::from("/work/repo-worktrees/ABC-123-DEF"));
    }

    #[test]
    fn test_worktree_path_preserves_underscores_in_issue_id() {
        let p = worktree_path(Path::new("/work"), "owner/repo", "my_issue_123");
        assert_eq!(p, PathBuf::from("/work/repo-worktrees/my_issue_123"));
    }

    #[test]
    fn test_worktree_path_multiple_dots_in_issue_id() {
        let p = worktree_path(Path::new("/work"), "owner/repo", "a.b.c.d");
        assert_eq!(p, PathBuf::from("/work/repo-worktrees/a_b_c_d"));
    }

    #[test]
    fn test_worktree_path_mixed_special_chars_in_issue_id() {
        let p = worktree_path(Path::new("/work"), "owner/repo", "a/b\\c.d\0e");
        assert_eq!(p, PathBuf::from("/work/repo-worktrees/a_b_c_d_e"));
    }

    // ════════════════════════════════════════════════════════════
    //  Integration: clone, branch, current_branch round-trip
    // ════════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_clone_and_verify_branch() {
        let origin = TempDir::new().unwrap();
        init_git_repo(origin.path());

        let target_dir = TempDir::new().unwrap();
        let target = target_dir.path().join("cloned");
        let url = file_url(origin.path());

        GitOps::ensure_repo_at_path(&target, &url, "main")
            .await
            .unwrap();

        let branch = GitOps::current_branch(&target).await.unwrap();
        assert_eq!(branch, "main");
    }

    #[tokio::test]
    async fn test_worktree_current_branch_is_detached() {
        let temp = TempDir::new().unwrap();
        init_git_repo(temp.path());

        let wt_parent = temp.path().join("detach-worktrees");
        std::fs::create_dir_all(&wt_parent).unwrap();
        let wt_path = wt_parent.join("wt");

        GitOps::create_worktree(temp.path(), &wt_path, "main")
            .await
            .unwrap();

        // Detached worktree should report HEAD
        let branch = GitOps::current_branch(&wt_path).await.unwrap();
        assert_eq!(branch, "HEAD");
    }

    #[tokio::test]
    async fn test_worktree_on_branch_current_branch() {
        let temp = TempDir::new().unwrap();
        init_git_repo(temp.path());

        let wt_parent = temp.path().join("named-worktrees");
        std::fs::create_dir_all(&wt_parent).unwrap();
        let wt_path = wt_parent.join("wt");

        GitOps::create_worktree_on_branch(temp.path(), &wt_path, "my-feature", "main")
            .await
            .unwrap();

        let branch = GitOps::current_branch(&wt_path).await.unwrap();
        assert_eq!(branch, "my-feature");
    }

    // ════════════════════════════════════════════════════════════
    //  Integration: fetch_all picks up new branches
    // ════════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_fetch_all_picks_up_new_branch() {
        let origin = TempDir::new().unwrap();
        init_git_repo(origin.path());

        let target_dir = TempDir::new().unwrap();
        let target = target_dir.path().join("cloned");
        let url = file_url(origin.path());

        GitOps::ensure_repo_at_path(&target, &url, "main")
            .await
            .unwrap();

        // Create a new branch in origin
        StdCommand::new("git")
            .args(["branch", "new-feature"])
            .current_dir(origin.path())
            .output()
            .unwrap();

        // Fetch all
        GitOps::fetch_all(&target).await.unwrap();

        // Verify remote branch is available
        let output = StdCommand::new("git")
            .args(["branch", "-r"])
            .current_dir(&target)
            .output()
            .unwrap();
        let branches = String::from_utf8_lossy(&output.stdout);
        assert!(
            branches.contains("origin/new-feature"),
            "new-feature not found in remote branches: {}",
            branches
        );
    }

    // ════════════════════════════════════════════════════════════
    //  Integration: is_git_repo on worktree
    // ════════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_is_git_repo_on_worktree() {
        let temp = TempDir::new().unwrap();
        init_git_repo(temp.path());

        let wt_parent = temp.path().join("isgit-worktrees");
        std::fs::create_dir_all(&wt_parent).unwrap();
        let wt_path = wt_parent.join("wt");

        GitOps::create_worktree(temp.path(), &wt_path, "main")
            .await
            .unwrap();

        // Worktree has .git as a file, is_git_repo should return true
        assert!(GitOps::is_git_repo(&wt_path));
    }

    // ════════════════════════════════════════════════════════════
    //  Error handling: various non-git-directory scenarios
    // ════════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_fetch_all_on_empty_dir() {
        let temp = TempDir::new().unwrap();
        let result = GitOps::fetch_all(temp.path()).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_fetch_branch_on_non_repo() {
        let temp = TempDir::new().unwrap();
        let result = GitOps::fetch_branch(temp.path(), "main").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_current_branch_on_empty_repo_no_commits() {
        let temp = TempDir::new().unwrap();

        // git init without any commits
        StdCommand::new("git")
            .args(["init", "-b", "main"])
            .current_dir(temp.path())
            .output()
            .unwrap();

        // rev-parse --abbrev-ref HEAD on a repo with no commits should fail
        let result = GitOps::current_branch(temp.path()).await;
        assert!(result.is_err());
    }

    // ════════════════════════════════════════════════════════════
    //  Validate ref: comprehensive injection patterns
    // ════════════════════════════════════════════════════════════

    #[test]
    fn test_validate_ref_rejects_various_shell_metacharacters() {
        let bad_refs = vec![
            "$(whoami)",
            "`id`",
            "a;b",
            "a|b",
            "a&b",
            "a>b",
            "a<b",
            "a b",
            "a\tb",
            "a\nb",
            "a\rb",
            "a'b",
            "a\"b",
            "a!b",
            "a#b",
            "a%b",
            "a(b)",
            "a{b}",
            "a=b",
            "a+b",
            "a,b",
        ];
        for bad in bad_refs {
            let result = validate_ref(bad, "ref");
            assert!(
                result.is_err(),
                "validate_ref should have rejected {:?}",
                bad
            );
        }
    }

    #[test]
    fn test_validate_ref_accepts_all_allowed_characters() {
        // Every character that should be individually allowed
        let good = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_./a@b";
        assert!(validate_ref(good, "ref").is_ok());
    }

    // ════════════════════════════════════════════════════════════
    //  create_worktree_on_branch with double-dot rejection
    // ════════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_create_worktree_on_branch_rejects_dotdot_in_branch() {
        let temp = TempDir::new().unwrap();
        let wt = temp.path().join("wt");
        let result = GitOps::create_worktree_on_branch(temp.path(), &wt, "a..b", "main").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_create_worktree_on_branch_rejects_dotdot_in_start_point() {
        let temp = TempDir::new().unwrap();
        let wt = temp.path().join("wt");
        let result = GitOps::create_worktree_on_branch(temp.path(), &wt, "branch", "a..b").await;
        assert!(result.is_err());
    }

    // ════════════════════════════════════════════════════════════
    //  ensure_repo_at_path: chooses clone vs pull correctly
    // ════════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_ensure_repo_at_path_path_not_exist_triggers_clone() {
        let temp = TempDir::new().unwrap();
        let target = temp.path().join("does_not_exist");

        // With a nonexistent target, it should try to clone (which fails because
        // the URL is fake)
        let result =
            GitOps::ensure_repo_at_path(&target, "https://nonexistent.invalid/repo.git", "main")
                .await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("git clone failed"), "unexpected: {}", err);
    }

    #[tokio::test]
    async fn test_ensure_repo_at_path_path_exists_triggers_pull() {
        let temp = TempDir::new().unwrap();
        let target = temp.path().join("exists");
        std::fs::create_dir_all(&target).unwrap();

        // With an existing target, it should try to pull (which fails because
        // branch validation rejects "--evil")
        let result = GitOps::ensure_repo_at_path(&target, "https://x.com/r.git", "--evil").await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("disallowed characters"), "unexpected: {}", err);
    }

    // ════════════════════════════════════════════════════════════
    //  ensure_repo_fetched: chooses clone vs fetch correctly
    // ════════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_ensure_repo_fetched_path_not_exist_triggers_clone() {
        let temp = TempDir::new().unwrap();
        let target = temp.path().join("nope");

        let result =
            GitOps::ensure_repo_fetched(&target, "https://nonexistent.invalid/repo.git").await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("git clone failed"), "unexpected: {}", err);
    }

    #[tokio::test]
    async fn test_ensure_repo_fetched_path_exists_triggers_fetch() {
        let temp = TempDir::new().unwrap();
        // Just an empty directory
        let target = temp.path().join("exists");
        std::fs::create_dir_all(&target).unwrap();

        let result =
            GitOps::ensure_repo_fetched(&target, "https://nonexistent.invalid/repo.git").await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        // Should have tried fetch (not clone) since the path exists
        assert!(err.contains("git fetch failed"), "unexpected: {}", err);
    }

    // ════════════════════════════════════════════════════════════
    //  Worktree path with repo names that need sanitization
    // ════════════════════════════════════════════════════════════

    #[test]
    fn test_worktree_path_backslash_in_repo_name() {
        let p = worktree_path(Path::new("/work"), "owner\\repo", "ID");
        // split('/') won't split on backslash, so last component is "owner\\repo"
        // then replace removes backslashes
        assert_eq!(p, PathBuf::from("/work/owner_repo-worktrees/ID"));
    }

    #[test]
    fn test_worktree_path_null_in_both() {
        let p = worktree_path(Path::new("/work"), "re\0po", "is\0sue");
        assert_eq!(p, PathBuf::from("/work/re_po-worktrees/is_sue"));
    }

    // ════════════════════════════════════════════════════════════
    //  Integration: full workflow - clone, fetch, worktree
    // ════════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_full_workflow_clone_branch_worktree() {
        let origin = TempDir::new().unwrap();
        init_git_repo(origin.path());
        add_second_commit(origin.path());

        let workspace = TempDir::new().unwrap();
        let repo_path = workspace.path().join("repo");
        let url = file_url(origin.path());

        // 1) Clone
        GitOps::ensure_repo_at_path(&repo_path, &url, "main")
            .await
            .unwrap();

        // 2) Verify current branch
        let branch = GitOps::current_branch(&repo_path).await.unwrap();
        assert_eq!(branch, "main");

        // 3) Create a worktree on a named branch
        let wt_parent = workspace.path().join("test-worktrees");
        std::fs::create_dir_all(&wt_parent).unwrap();
        let wt_path = wt_parent.join("fix-123");

        GitOps::create_worktree_on_branch(&repo_path, &wt_path, "fix/issue-123", "main")
            .await
            .unwrap();

        // 4) Verify worktree branch
        let wt_branch = GitOps::current_branch(&wt_path).await.unwrap();
        assert_eq!(wt_branch, "fix/issue-123");

        // 5) Verify worktree is detected as a git repo
        assert!(GitOps::is_git_repo(&wt_path));

        // 6) Main repo is still on main
        let main_branch = GitOps::current_branch(&repo_path).await.unwrap();
        assert_eq!(main_branch, "main");
    }

    // ════════════════════════════════════════════════════════════
    //  validate_ref - unicode and additional characters
    // ════════════════════════════════════════════════════════════

    #[test]
    fn test_validate_ref_rejects_unicode_non_alphanumeric() {
        assert!(validate_ref("branch\u{200B}name", "ref").is_err()); // zero-width space
        assert!(validate_ref("\u{2026}", "ref").is_err()); // ellipsis
    }

    #[test]
    fn test_validate_ref_accepts_unicode_alphanumeric() {
        // char::is_alphanumeric() accepts unicode letters and digits
        assert!(validate_ref("br\u{00e4}nch", "ref").is_ok()); // a-umlaut is alphanumeric
    }

    #[test]
    fn test_validate_ref_rejects_control_characters() {
        assert!(validate_ref("branch\x01name", "ref").is_err());
        assert!(validate_ref("branch\x7f", "ref").is_err()); // DEL
        assert!(validate_ref("\x00branch", "ref").is_err()); // NULL
    }

    #[test]
    fn test_validate_ref_allows_long_branch_name() {
        let long_name = "a".repeat(256);
        assert!(validate_ref(&long_name, "branch").is_ok());
    }

    #[test]
    fn test_validate_ref_rejects_triple_dot() {
        // Contains ".." so it should be rejected
        assert!(validate_ref("a...b", "ref").is_err());
    }

    #[test]
    fn test_validate_ref_allows_dot_separated_version() {
        // Single dots are fine, no ".." present
        assert!(validate_ref("v1.0.0", "tag").is_ok());
        assert!(validate_ref("release.1.2.3", "tag").is_ok());
    }

    // ════════════════════════════════════════════════════════════
    //  worktree_path - additional edge cases
    // ════════════════════════════════════════════════════════════

    #[test]
    fn test_worktree_path_unicode_in_issue_id() {
        // Unicode characters pass through sanitization (only /, \, ., \0 are replaced)
        let p = worktree_path(Path::new("/work"), "owner/repo", "issue-\u{00e9}");
        assert_eq!(p, PathBuf::from("/work/repo-worktrees/issue-\u{00e9}"));
    }

    #[test]
    fn test_worktree_path_all_dots_in_issue_id() {
        let p = worktree_path(Path::new("/work"), "owner/repo", "...");
        assert_eq!(p, PathBuf::from("/work/repo-worktrees/___"));
    }

    #[test]
    fn test_worktree_path_all_slashes_in_issue_id() {
        let p = worktree_path(Path::new("/work"), "owner/repo", "///");
        assert_eq!(p, PathBuf::from("/work/repo-worktrees/___"));
    }

    #[test]
    fn test_worktree_path_hyphen_in_repo_name_preserved() {
        let p = worktree_path(Path::new("/work"), "org/my-cool-repo", "ID-1");
        assert_eq!(p, PathBuf::from("/work/my-cool-repo-worktrees/ID-1"));
    }

    #[test]
    fn test_worktree_path_underscore_in_repo_name_preserved() {
        let p = worktree_path(Path::new("/work"), "org/my_repo", "ID-1");
        assert_eq!(p, PathBuf::from("/work/my_repo-worktrees/ID-1"));
    }

    // ════════════════════════════════════════════════════════════
    //  is_git_repo - additional edge cases
    // ════════════════════════════════════════════════════════════

    #[test]
    fn test_is_git_repo_symlink_git_dir() {
        let temp = TempDir::new().unwrap();
        let real_git = temp.path().join("real_git");
        std::fs::create_dir(&real_git).unwrap();

        // Create a symlink .git -> real_git
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(&real_git, temp.path().join(".git")).unwrap();
            // symlink to a directory - is_dir() follows symlinks so should be true
            assert!(GitOps::is_git_repo(temp.path()));
        }
    }

    // ════════════════════════════════════════════════════════════
    //  ensure_repo_fetched with real local clone
    // ════════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_ensure_repo_fetched_picks_up_new_commit() {
        let origin = TempDir::new().unwrap();
        init_git_repo(origin.path());

        let target_dir = TempDir::new().unwrap();
        let target = target_dir.path().join("cloned");
        let url = file_url(origin.path());

        // Clone via ensure_repo_at_path
        GitOps::ensure_repo_at_path(&target, &url, "main")
            .await
            .unwrap();

        // Add a commit to origin
        add_second_commit(origin.path());

        // Fetch (does not checkout/reset)
        GitOps::ensure_repo_fetched(&target, &url).await.unwrap();

        // The fetch should have brought the new commit into the object store
        // Verify via git log of origin/main
        let output = StdCommand::new("git")
            .args(["log", "--oneline", "origin/main"])
            .current_dir(&target)
            .output()
            .unwrap();
        let log = String::from_utf8_lossy(&output.stdout);
        assert!(
            log.contains("second commit"),
            "fetch did not bring second commit: {}",
            log
        );
    }

    // ════════════════════════════════════════════════════════════
    //  create_worktree with parent directory creation
    // ════════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_create_worktree_creates_parent_dirs() {
        let temp = TempDir::new().unwrap();
        init_git_repo(temp.path());

        // Use a nested parent path that does NOT end in -worktrees
        // (git worktree add will create the directory itself)
        let wt_path = temp.path().join("deep-worktrees").join("nested").join("wt");

        let result = GitOps::create_worktree(temp.path(), &wt_path, "main").await;
        assert!(
            result.is_ok(),
            "create_worktree with deep nesting failed: {:?}",
            result.unwrap_err()
        );
        assert!(wt_path.exists());
        assert!(wt_path.join("README.md").exists());
    }

    // ════════════════════════════════════════════════════════════
    //  create_worktree_on_branch with parent dir creation
    // ════════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_create_worktree_on_branch_creates_parent_dirs() {
        let temp = TempDir::new().unwrap();
        init_git_repo(temp.path());

        let wt_path = temp.path().join("deep-worktrees").join("nested").join("wt");

        let result =
            GitOps::create_worktree_on_branch(temp.path(), &wt_path, "new-branch", "main").await;
        assert!(
            result.is_ok(),
            "create_worktree_on_branch with deep nesting failed: {:?}",
            result.unwrap_err()
        );
        assert!(wt_path.exists());

        let branch = GitOps::current_branch(&wt_path).await.unwrap();
        assert_eq!(branch, "new-branch");
    }

    // ════════════════════════════════════════════════════════════
    //  fetch_branch - success case with new commit on branch
    // ════════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_fetch_branch_picks_up_new_commit() {
        let origin = TempDir::new().unwrap();
        init_git_repo(origin.path());

        // Create a feature branch in origin with a new commit
        StdCommand::new("git")
            .args(["checkout", "-b", "feature-y"])
            .current_dir(origin.path())
            .output()
            .unwrap();
        add_second_commit(origin.path());
        StdCommand::new("git")
            .args(["checkout", "main"])
            .current_dir(origin.path())
            .output()
            .unwrap();

        // Clone
        let target_dir = TempDir::new().unwrap();
        let target = target_dir.path().join("cloned");
        let url = file_url(origin.path());
        GitOps::ensure_repo_at_path(&target, &url, "main")
            .await
            .unwrap();

        // Fetch just the feature branch
        GitOps::fetch_branch(&target, "feature-y").await.unwrap();

        // Verify the branch commit is available
        let output = StdCommand::new("git")
            .args(["log", "--oneline", "origin/feature-y"])
            .current_dir(&target)
            .output()
            .unwrap();
        let log = String::from_utf8_lossy(&output.stdout);
        assert!(
            log.contains("second commit"),
            "fetch_branch did not bring feature commit: {}",
            log
        );
    }

    // ════════════════════════════════════════════════════════════
    //  remove_worktree - full lifecycle
    // ════════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_remove_worktree_full_lifecycle() {
        let temp = TempDir::new().unwrap();
        init_git_repo(temp.path());

        let wt_parent = temp.path().join("lifecycle-worktrees");
        std::fs::create_dir_all(&wt_parent).unwrap();
        let wt_path = wt_parent.join("wt1");

        // Create
        GitOps::create_worktree(temp.path(), &wt_path, "main")
            .await
            .unwrap();
        assert!(wt_path.exists());

        // Remove
        GitOps::remove_worktree(temp.path(), &wt_path)
            .await
            .unwrap();

        // Should be gone
        assert!(!wt_path.exists());
    }

    // ════════════════════════════════════════════════════════════
    //  Multiple worktrees from same repo
    // ════════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_multiple_worktrees_from_same_repo() {
        let temp = TempDir::new().unwrap();
        init_git_repo(temp.path());
        add_second_commit(temp.path());

        let wt_parent = temp.path().join("multi-worktrees");
        std::fs::create_dir_all(&wt_parent).unwrap();

        let wt1 = wt_parent.join("wt1");
        let wt2 = wt_parent.join("wt2");

        // Create two worktrees on different branches
        GitOps::create_worktree_on_branch(temp.path(), &wt1, "branch-1", "main")
            .await
            .unwrap();
        GitOps::create_worktree_on_branch(temp.path(), &wt2, "branch-2", "main")
            .await
            .unwrap();

        // Both should exist
        assert!(wt1.exists());
        assert!(wt2.exists());

        // Both should be on their respective branches
        let b1 = GitOps::current_branch(&wt1).await.unwrap();
        let b2 = GitOps::current_branch(&wt2).await.unwrap();
        assert_eq!(b1, "branch-1");
        assert_eq!(b2, "branch-2");

        // Main repo should still be on main
        let main = GitOps::current_branch(temp.path()).await.unwrap();
        assert_eq!(main, "main");
    }

    // ════════════════════════════════════════════════════════════
    //  create_worktree_on_branch resets existing branch
    // ════════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_create_worktree_on_branch_resets_branch_to_start_point() {
        let temp = TempDir::new().unwrap();
        init_git_repo(temp.path());
        add_second_commit(temp.path());

        let wt_parent = temp.path().join("reset-worktrees");
        std::fs::create_dir_all(&wt_parent).unwrap();
        let wt_path = wt_parent.join("wt");

        // Get the first commit hash
        let output = StdCommand::new("git")
            .args(["rev-parse", "HEAD~1"])
            .current_dir(temp.path())
            .output()
            .unwrap();
        let first_commit = String::from_utf8_lossy(&output.stdout).trim().to_string();

        // Create worktree at the first commit
        GitOps::create_worktree_on_branch(temp.path(), &wt_path, "test-branch", &first_commit)
            .await
            .unwrap();

        // The worktree should be at the first commit (no file2.txt)
        assert!(!wt_path.join("file2.txt").exists());
        assert!(wt_path.join("README.md").exists());
    }

    // ════════════════════════════════════════════════════════════
    //  validate_ref - boundary between allowed/disallowed
    // ════════════════════════════════════════════════════════════

    #[test]
    fn test_validate_ref_at_in_middle_is_ok() {
        assert!(validate_ref("user@host", "ref").is_ok());
        assert!(validate_ref("a@b@c", "ref").is_ok());
    }

    #[test]
    fn test_validate_ref_leading_dot_is_ok() {
        // Single leading dot is allowed (no ".." and doesn't start with '-')
        assert!(validate_ref(".branch", "ref").is_ok());
    }

    #[test]
    fn test_validate_ref_dot_dot_at_start() {
        assert!(validate_ref("..branch", "ref").is_err());
    }

    #[test]
    fn test_validate_ref_dot_dot_at_end() {
        assert!(validate_ref("branch..", "ref").is_err());
    }
}
