//! E2E scenario definitions.

pub mod s1_lifecycle;
pub mod s2_ask;
pub mod s3_cascade;
pub mod s4_action;

use claudear::scm::ScmProvider;
use claudear::source::IssueSource;
use std::sync::Arc;

/// Shared context for scenario execution.
pub struct ScenarioContext<'a> {
    pub scm: Arc<dyn ScmProvider>,
    pub source: Arc<dyn IssueSource>,
    pub ask_backend: &'a Option<Box<dyn crate::ask::E2eAsk>>,
    pub repo: &'a str,
    pub repo2: Option<&'a str>,
    pub reviewer_token: Option<&'a str>,
    pub use_docker: bool,
    pub docker_image: &'a str,
    pub binary: Option<&'a str>,
    pub wait_timeout: u64,
    pub claude_timeout: u64,
    pub scm_name: &'a str,
    pub source_name: &'a str,
    pub ask_name: &'a str,
}

impl<'a> ScenarioContext<'a> {
    /// Get the wait timeout as a Duration.
    pub fn timeout(&self) -> std::time::Duration {
        std::time::Duration::from_secs(self.wait_timeout)
    }

    /// Get the poll interval for wait loops.
    pub fn poll_interval(&self) -> std::time::Duration {
        std::time::Duration::from_secs(5)
    }

    /// Get the path to the claudear binary (builds if needed).
    pub fn binary_path(&self) -> anyhow::Result<String> {
        if let Some(bin) = self.binary {
            return Ok(bin.to_string());
        }

        // Build the binary
        let output = std::process::Command::new("cargo")
            .args(["build", "--release", "--bin", "claudear"])
            .output()?;

        if !output.status.success() {
            anyhow::bail!(
                "cargo build failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        Ok("target/release/claudear".to_string())
    }

    /// Clone a repo to a local directory for the daemon's repo discovery.
    ///
    /// Returns the path to the cloned repo.
    pub fn clone_repo(
        &self,
        repo: &str,
        dest: &std::path::Path,
    ) -> anyhow::Result<std::path::PathBuf> {
        let repo_name = repo.split('/').next_back().unwrap_or(repo);
        let clone_dest = dest.join(repo_name);

        if clone_dest.exists() {
            tracing::info!(repo, path = %clone_dest.display(), "Repo already cloned");
            return Ok(clone_dest);
        }

        let clone_url = match self.scm_name {
            "github" => {
                let token = std::env::var("CLAUDEAR_E2E_GITHUB_TOKEN").unwrap_or_default();
                format!("https://x-access-token:{}@github.com/{}.git", token, repo)
            }
            "gitlab" => {
                let token = std::env::var("CLAUDEAR_E2E_GITLAB_TOKEN").unwrap_or_default();
                let base = std::env::var("CLAUDEAR_E2E_GITLAB_URL")
                    .unwrap_or_else(|_| "https://gitlab.com".to_string());
                // Strip protocol for embedding token
                let host = base
                    .strip_prefix("https://")
                    .or_else(|| base.strip_prefix("http://"))
                    .unwrap_or(&base);
                format!("https://oauth2:{}@{}/{}.git", token, host, repo)
            }
            other => anyhow::bail!("Unknown SCM for clone: {}", other),
        };

        tracing::info!(repo, path = %clone_dest.display(), "Cloning repo");
        let output = std::process::Command::new("git")
            .args(["clone", &clone_url, clone_dest.to_str().unwrap_or("")])
            .output()
            .context("git clone")?;

        if !output.status.success() {
            anyhow::bail!(
                "git clone failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        // Configure a repo-local credential helper that returns the SCM bot
        // token. This ensures git push authenticates as the bot user even if
        // Claude Code (or macOS keychain) injects different credentials.
        let scm_token = match self.scm_name {
            "github" => std::env::var("CLAUDEAR_E2E_GITHUB_TOKEN").ok(),
            "gitlab" => std::env::var("CLAUDEAR_E2E_GITLAB_TOKEN").ok(),
            _ => None,
        };
        if let Some(token) = scm_token {
            let helper_script = clone_dest.join(".git-credential-helper.sh");
            let username = if self.scm_name == "gitlab" {
                "oauth2"
            } else {
                "x-access-token"
            };
            std::fs::write(
                &helper_script,
                format!(
                    "#!/bin/sh\nprintf 'username={}\\npassword={}\\n'\n",
                    username, token
                ),
            )
            .context("write credential helper")?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(&helper_script, std::fs::Permissions::from_mode(0o700))
                    .context("chmod credential helper")?;
            }
            let set_helper = std::process::Command::new("git")
                .args([
                    "-C",
                    clone_dest.to_str().unwrap_or(""),
                    "config",
                    "--local",
                    "credential.helper",
                    &format!("!{}", helper_script.display()),
                ])
                .output()
                .context("git config credential.helper")?;
            if !set_helper.status.success() {
                tracing::warn!(
                    "Failed to set repo-local credential helper: {}",
                    String::from_utf8_lossy(&set_helper.stderr)
                );
            }
        }

        Ok(clone_dest)
    }
}

use anyhow::Context;
