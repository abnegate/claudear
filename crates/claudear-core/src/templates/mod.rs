//! Custom prompt templates with AGENT.md support.
//!
//! This module provides template loading and rendering for Claude prompts.
//! Templates can come from:
//! 1. The project's AGENT.md file (highest priority)
//! 2. Database-stored templates
//! 3. Built-in default templates

mod loader;
mod renderer;

pub use loader::TemplateLoader;
pub use renderer::{TemplateContext, TemplateRenderer};

/// Default template for issue fixing when no custom template is found.
pub const DEFAULT_FIX_TEMPLATE: &str = r#"You are fixing an issue from {{source}}. Here is the issue context:

{{context}}
{{#if repo_name}}
IMPORTANT - Repository Verification:
You are working in the repository: {{repo_name}}
Before starting any work, verify this is the correct repository for this issue by checking that the file paths, modules, or stack traces reference code in this codebase.
If this is NOT the correct repository, set "wrong_repo" in your response to the name of the repository you believe is correct (in "org/repo" format) and do NOT attempt any fixes.
{{/if}}
Your task:
1. Analyze the issue/error and any stack traces
2. Find the relevant code in this codebase
3. Write a failing test that reproduces the bug before changing any application code
4. Implement the minimal fix to make the failing test pass
5. Verify all existing tests still pass
6. Create a PR with your changes
7. Ensure all checks pass on the PR

IMPORTANT: Always use a test-driven development (TDD) approach for bug fixes. Start by adding a failing test that reproduces the issue, then fix the code to make it pass. Do not skip the failing test step.

The PR title should include the issue ID: {{short_id}}

"#;

/// Default template for Linear issues (uses /issue skill).
pub const DEFAULT_LINEAR_TEMPLATE: &str = r#"{{#if has_agent_md}}
{{agent_md}}

---
{{/if}}
{{#if repo_name}}
IMPORTANT - Repository Verification:
You are working in the repository: {{repo_name}}
Before starting any work, verify this is the correct repository for this issue by checking that the file paths, modules, or stack traces reference code in this codebase.
If this is NOT the correct repository, set "wrong_repo" in your response to the name of the repository you believe is correct (in "org/repo" format) and do NOT attempt any fixes.
{{/if}}
Address the following Linear issue:

Issue: {{short_id}} - {{title}}
URL: {{url}}
Priority: {{priority}}

{{#if description}}
Description:
{{description}}
{{/if}}

{{context}}

IMPORTANT: Use a test-driven development (TDD) approach for bug fixes. Before changing any application code, write a failing test that reproduces the issue. Then implement the minimal fix to make the test pass and verify all existing tests still pass.

Create a PR that addresses this issue. Include "{{short_id}}" in the PR title.
"#;

/// Default template for Sentry errors.
pub const DEFAULT_SENTRY_TEMPLATE: &str = r#"{{#if has_agent_md}}
{{agent_md}}

---
{{/if}}
{{#if repo_name}}
IMPORTANT - Repository Verification:
You are working in the repository: {{repo_name}}
Before starting any work, verify this is the correct repository for this issue by checking that the file paths, modules, or stack traces reference code in this codebase.
If this is NOT the correct repository, set "wrong_repo" in your response to the name of the repository you believe is correct (in "org/repo" format) and do NOT attempt any fixes.
{{/if}}
Fix the following error from Sentry:

Error: {{title}}
URL: {{url}}
Event count: {{event_count}}

{{context}}

Analyze the stack trace and error context to identify the root cause.

IMPORTANT: Use a test-driven development (TDD) approach. Before changing any application code, write a failing test that reproduces the error. Then implement the minimal fix to make the test pass and verify all existing tests still pass.

Ensure all checks pass on the PR.

Create a PR that fixes this error.
"#;

#[cfg(test)]
mod tests {
    use super::*;

    // ---- DEFAULT_FIX_TEMPLATE ----

    #[test]
    fn test_fix_template_is_not_empty() {
        assert!(
            !DEFAULT_FIX_TEMPLATE.is_empty(),
            "DEFAULT_FIX_TEMPLATE should not be empty"
        );
    }

    #[test]
    fn test_fix_template_contains_context_placeholder() {
        assert!(
            DEFAULT_FIX_TEMPLATE.contains("{{context}}"),
            "DEFAULT_FIX_TEMPLATE should contain '{{{{context}}}}'"
        );
    }

    #[test]
    fn test_fix_template_contains_source_placeholder() {
        assert!(
            DEFAULT_FIX_TEMPLATE.contains("{{source}}"),
            "DEFAULT_FIX_TEMPLATE should contain '{{{{source}}}}'"
        );
    }

    #[test]
    fn test_fix_template_contains_short_id_placeholder() {
        assert!(
            DEFAULT_FIX_TEMPLATE.contains("{{short_id}}"),
            "DEFAULT_FIX_TEMPLATE should contain '{{{{short_id}}}}'"
        );
    }

    // ---- DEFAULT_LINEAR_TEMPLATE ----

    #[test]
    fn test_linear_template_is_not_empty() {
        assert!(
            !DEFAULT_LINEAR_TEMPLATE.is_empty(),
            "DEFAULT_LINEAR_TEMPLATE should not be empty"
        );
    }

    #[test]
    fn test_linear_template_contains_short_id() {
        assert!(
            DEFAULT_LINEAR_TEMPLATE.contains("{{short_id}}"),
            "DEFAULT_LINEAR_TEMPLATE should contain '{{{{short_id}}}}'"
        );
    }

    #[test]
    fn test_linear_template_contains_title() {
        assert!(
            DEFAULT_LINEAR_TEMPLATE.contains("{{title}}"),
            "DEFAULT_LINEAR_TEMPLATE should contain '{{{{title}}}}'"
        );
    }

    #[test]
    fn test_linear_template_contains_url() {
        assert!(
            DEFAULT_LINEAR_TEMPLATE.contains("{{url}}"),
            "DEFAULT_LINEAR_TEMPLATE should contain '{{{{url}}}}'"
        );
    }

    #[test]
    fn test_linear_template_contains_priority() {
        assert!(
            DEFAULT_LINEAR_TEMPLATE.contains("{{priority}}"),
            "DEFAULT_LINEAR_TEMPLATE should contain '{{{{priority}}}}'"
        );
    }

    #[test]
    fn test_linear_template_contains_context() {
        assert!(
            DEFAULT_LINEAR_TEMPLATE.contains("{{context}}"),
            "DEFAULT_LINEAR_TEMPLATE should contain '{{{{context}}}}'"
        );
    }

    #[test]
    fn test_linear_template_contains_if_description() {
        assert!(
            DEFAULT_LINEAR_TEMPLATE.contains("{{#if description}}"),
            "DEFAULT_LINEAR_TEMPLATE should contain '{{{{#if description}}}}'"
        );
    }

    // ---- DEFAULT_SENTRY_TEMPLATE ----

    #[test]
    fn test_sentry_template_is_not_empty() {
        assert!(
            !DEFAULT_SENTRY_TEMPLATE.is_empty(),
            "DEFAULT_SENTRY_TEMPLATE should not be empty"
        );
    }

    #[test]
    fn test_sentry_template_contains_title() {
        assert!(
            DEFAULT_SENTRY_TEMPLATE.contains("{{title}}"),
            "DEFAULT_SENTRY_TEMPLATE should contain '{{{{title}}}}'"
        );
    }

    #[test]
    fn test_sentry_template_contains_url() {
        assert!(
            DEFAULT_SENTRY_TEMPLATE.contains("{{url}}"),
            "DEFAULT_SENTRY_TEMPLATE should contain '{{{{url}}}}'"
        );
    }

    #[test]
    fn test_sentry_template_contains_context() {
        assert!(
            DEFAULT_SENTRY_TEMPLATE.contains("{{context}}"),
            "DEFAULT_SENTRY_TEMPLATE should contain '{{{{context}}}}'"
        );
    }

    #[test]
    fn test_sentry_template_contains_event_count() {
        assert!(
            DEFAULT_SENTRY_TEMPLATE.contains("{{event_count}}"),
            "DEFAULT_SENTRY_TEMPLATE should contain '{{{{event_count}}}}'"
        );
    }

    #[test]
    fn test_sentry_template_contains_if_has_agent_md() {
        assert!(
            DEFAULT_SENTRY_TEMPLATE.contains("{{#if has_agent_md}}"),
            "DEFAULT_SENTRY_TEMPLATE should contain '{{{{#if has_agent_md}}}}'"
        );
    }

    // ---- Cross-template checks ----

    #[test]
    fn test_all_templates_mention_pr() {
        assert!(
            DEFAULT_FIX_TEMPLATE.contains("PR"),
            "DEFAULT_FIX_TEMPLATE should mention 'PR'"
        );
        assert!(
            DEFAULT_LINEAR_TEMPLATE.contains("PR"),
            "DEFAULT_LINEAR_TEMPLATE should mention 'PR'"
        );
        assert!(
            DEFAULT_SENTRY_TEMPLATE.contains("PR"),
            "DEFAULT_SENTRY_TEMPLATE should mention 'PR'"
        );
    }

    #[test]
    fn test_fix_template_has_matched_conditionals() {
        let if_count = DEFAULT_FIX_TEMPLATE.matches("{{#if").count();
        let endif_count = DEFAULT_FIX_TEMPLATE.matches("{{/if}}").count();
        assert_eq!(
            if_count, endif_count,
            "DEFAULT_FIX_TEMPLATE has {} '{{{{#if' but {} '{{{{/if}}}}' — unmatched conditionals",
            if_count, endif_count
        );
    }

    #[test]
    fn test_linear_template_has_matched_conditionals() {
        let if_count = DEFAULT_LINEAR_TEMPLATE.matches("{{#if").count();
        let endif_count = DEFAULT_LINEAR_TEMPLATE.matches("{{/if}}").count();
        assert_eq!(
            if_count, endif_count,
            "DEFAULT_LINEAR_TEMPLATE has {} '{{{{#if' but {} '{{{{/if}}}}' — unmatched conditionals",
            if_count, endif_count
        );
    }

    #[test]
    fn test_sentry_template_has_matched_conditionals() {
        let if_count = DEFAULT_SENTRY_TEMPLATE.matches("{{#if").count();
        let endif_count = DEFAULT_SENTRY_TEMPLATE.matches("{{/if}}").count();
        assert_eq!(
            if_count, endif_count,
            "DEFAULT_SENTRY_TEMPLATE has {} '{{{{#if' but {} '{{{{/if}}}}' — unmatched conditionals",
            if_count, endif_count
        );
    }
}
