//! Guards the release workflow's package-repository targets.
//!
//! The APT repo and Homebrew tap are hosted under a different owner than the
//! `claudear` source repo, so `github.repository_owner` resolves to the wrong
//! org and the publish jobs fail with "repository not found". These tests pin
//! the workflow's push targets to the owner documented in the README install
//! instructions, so the two can never drift apart again.

use std::fs;

fn read(relative: &str) -> String {
    let path = format!("{}/{relative}", env!("CARGO_MANIFEST_DIR"));
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("failed to read {path}: {e}"))
}

/// Owner of the APT repository users are told to install from, e.g. `abnegate`
/// in `https://abnegate.github.io/apt-repo`.
fn documented_apt_owner(readme: &str) -> String {
    let before = readme
        .split_once(".github.io/apt-repo")
        .expect("README should document the APT repository URL")
        .0;

    before
        .rsplit_once("https://")
        .expect("the APT repository URL should be absolute")
        .1
        .to_string()
}

/// Owner of the Homebrew tap users are told to install from, e.g. `abnegate`
/// in `brew tap abnegate/tap`.
fn documented_tap_owner(readme: &str) -> String {
    let tap = readme
        .split_once("brew tap ")
        .expect("README should document the Homebrew tap")
        .1
        .lines()
        .next()
        .expect("the tap line should not be empty")
        .trim();

    tap.split_once('/')
        .expect("the tap should be written as owner/name")
        .0
        .to_string()
}

#[test]
fn apt_publish_job_clones_the_documented_apt_repository() {
    let workflow = read(".github/workflows/release.yml");
    let owner = documented_apt_owner(&read("README.md"));

    let expected = format!("github.com/{owner}/apt-repo.git");
    assert!(
        workflow.contains(&expected),
        "release.yml should clone the APT repo from {expected}, since that is \
         where the README tells users to install from"
    );
}

#[test]
fn apt_publish_job_does_not_use_the_source_repository_owner() {
    let workflow = read(".github/workflows/release.yml");

    let clone_line = workflow
        .lines()
        .find(|line| line.contains("apt-repo.git"))
        .expect("release.yml should clone the APT repository");

    assert!(
        !clone_line.contains("github.repository_owner"),
        "the APT repo is not owned by the source repository's owner, so \
         `github.repository_owner` resolves to a non-existent repo: {}",
        clone_line.trim()
    );
}

#[test]
fn homebrew_publish_job_clones_the_documented_tap() {
    let workflow = read(".github/workflows/release.yml");
    let owner = documented_tap_owner(&read("README.md"));

    let expected = format!("github.com/{owner}/homebrew-tap.git");
    assert!(
        workflow.contains(&expected),
        "release.yml should clone the Homebrew tap from {expected}, since that \
         is where the README tells users to tap from"
    );
}

#[test]
fn homebrew_publish_job_does_not_use_the_source_repository_owner() {
    let workflow = read(".github/workflows/release.yml");

    let clone_line = workflow
        .lines()
        .find(|line| line.contains("homebrew-tap.git"))
        .expect("release.yml should clone the Homebrew tap");

    assert!(
        !clone_line.contains("github.repository_owner"),
        "the Homebrew tap is not owned by the source repository's owner, so \
         `github.repository_owner` resolves to a non-existent repo: {}",
        clone_line.trim()
    );
}

/// The formula's download URLs point at the *source* repo's releases, which
/// really is `github.repository_owner` — make sure the fix above was not
/// over-applied to it.
#[test]
fn formula_template_still_uses_the_source_repository_owner() {
    let workflow = read(".github/workflows/release.yml");

    assert!(
        workflow.contains("s/{{REPO_OWNER}}/${{ github.repository_owner }}/g"),
        "the Homebrew formula downloads release assets from the source repo, \
         so the REPO_OWNER placeholder must stay bound to github.repository_owner"
    );
}
