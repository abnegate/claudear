//! Types for the evaluation system.

use serde::{Deserialize, Serialize};

pub use claudear_core::types::{
    Diagnostic, DiagnosticSeverity, EvalCategory, EvalDelta, EvalSnapshot,
};

/// Aggregated evaluation result for an attempt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationResult {
    pub attempt_id: i64,
    pub repo: String,
    pub deltas: Vec<EvalDelta>,
    pub overall_improved: bool,
    pub summary: String,
}

impl EvaluationResult {
    pub fn new(attempt_id: i64, repo: String, deltas: Vec<EvalDelta>) -> Self {
        let overall_improved = !deltas.is_empty() && deltas.iter().all(|d| d.is_improvement());
        let summary = Self::build_summary(&deltas);
        Self {
            attempt_id,
            repo,
            deltas,
            overall_improved,
            summary,
        }
    }

    /// Whether the fix introduced new failures or regressions in any tool.
    pub fn has_regressions(&self) -> bool {
        self.deltas
            .iter()
            .any(|d| d.new_failures > 0 || !d.regressions.is_empty())
    }

    fn build_summary(deltas: &[EvalDelta]) -> String {
        if deltas.is_empty() {
            return "No evaluation tools ran.".to_string();
        }

        let mut lines = Vec::new();
        for delta in deltas {
            let icon = if delta.is_improvement() {
                "\u{2705}"
            } else {
                "\u{26a0}\u{fe0f}"
            };
            let mut desc = format!(
                "{} **{}** ({})",
                icon, delta.after.tool_name, delta.after.category
            );
            if delta.new_failures != 0 {
                desc.push_str(&format!(": {} new failure(s)", delta.new_failures));
            }
            if !delta.regressions.is_empty() {
                desc.push_str(&format!(", {} regression(s)", delta.regressions.len()));
            }
            if !delta.fixed.is_empty() {
                desc.push_str(&format!(", {} fixed", delta.fixed.len()));
            }
            if let Some(cov) = delta.coverage_delta_pct {
                desc.push_str(&format!(", coverage {:+.1}%", cov));
            }
            lines.push(desc);
        }
        lines.join("\n")
    }

    /// Format as a PR comment body.
    pub fn format_pr_comment(&self) -> String {
        let mut comment = String::from("## \u{1f50d} Code Quality Evaluation\n\n");

        if self.overall_improved {
            comment.push_str("> \u{2705} Overall: **Improved**\n\n");
        } else {
            comment.push_str("> \u{26a0}\u{fe0f} Overall: **Regressions detected**\n\n");
        }

        for delta in &self.deltas {
            comment.push_str(&format!(
                "### {} ({})\n",
                delta.after.tool_name, delta.after.category
            ));
            comment.push_str("| Metric | Before | After | Delta |\n");
            comment.push_str("|--------|--------|-------|-------|\n");
            comment.push_str(&format!(
                "| Passed | {} | {} | {:+} |\n",
                delta.before.passed, delta.after.passed, delta.new_passes
            ));
            comment.push_str(&format!(
                "| Failed | {} | {} | {:+} |\n",
                delta.before.failed, delta.after.failed, delta.new_failures
            ));
            comment.push_str(&format!(
                "| Warnings | {} | {} | {:+} |\n",
                delta.before.warnings,
                delta.after.warnings,
                delta.after.warnings as i32 - delta.before.warnings as i32
            ));
            comment.push_str(&format!(
                "| Errors | {} | {} | {:+} |\n",
                delta.before.errors,
                delta.after.errors,
                delta.after.errors as i32 - delta.before.errors as i32
            ));
            if let Some(cov) = delta.coverage_delta_pct {
                comment.push_str(&format!(
                    "| Coverage | {:.1}% | {:.1}% | {:+.1}% |\n",
                    delta.before.line_coverage_pct.unwrap_or(0.0),
                    delta.after.line_coverage_pct.unwrap_or(0.0),
                    cov
                ));
            }

            if !delta.regressions.is_empty() {
                comment.push_str("\n**New issues:**\n");
                for d in delta.regressions.iter().take(10) {
                    comment.push_str(&format!(
                        "- `{}:{}` {}\n",
                        d.file,
                        d.line.unwrap_or(0),
                        d.message
                    ));
                }
                if delta.regressions.len() > 10 {
                    comment.push_str(&format!(
                        "- _...and {} more_\n",
                        delta.regressions.len() - 10
                    ));
                }
            }

            if !delta.fixed.is_empty() {
                comment.push_str(&format!(
                    "\n**Fixed:** {} issue(s) resolved\n",
                    delta.fixed.len()
                ));
            }
            comment.push('\n');
        }

        comment
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_snapshot(category: EvalCategory, tool: &str, passed: u32, failed: u32) -> EvalSnapshot {
        EvalSnapshot {
            category,
            tool_name: tool.to_string(),
            exit_code: if failed > 0 { 1 } else { 0 },
            passed,
            failed,
            skipped: 0,
            warnings: 0,
            errors: failed,
            diagnostics: Vec::new(),
            raw_output: String::new(),
            duration_secs: 1.0,
            line_coverage_pct: None,
            branch_coverage_pct: None,
        }
    }

    #[test]
    fn test_eval_delta_improvement() {
        let before = make_snapshot(EvalCategory::Test, "cargo test", 10, 2);
        let after = make_snapshot(EvalCategory::Test, "cargo test", 12, 0);
        let delta = EvalDelta::compute(before, after);
        assert_eq!(delta.new_passes, 2);
        assert_eq!(delta.new_failures, -2);
        assert!(delta.is_improvement());
    }

    #[test]
    fn test_eval_delta_regression() {
        let before = make_snapshot(EvalCategory::Test, "cargo test", 10, 0);
        let after = make_snapshot(EvalCategory::Test, "cargo test", 9, 1);
        let delta = EvalDelta::compute(before, after);
        assert_eq!(delta.new_failures, 1);
        assert!(!delta.is_improvement());
    }

    #[test]
    fn test_eval_delta_diagnostics() {
        let d1 = Diagnostic {
            file: "src/main.rs".into(),
            line: Some(10),
            column: None,
            severity: DiagnosticSeverity::Warning,
            code: Some("W001".into()),
            message: "unused variable".into(),
        };
        let d2 = Diagnostic {
            file: "src/lib.rs".into(),
            line: Some(20),
            column: None,
            severity: DiagnosticSeverity::Error,
            code: None,
            message: "type mismatch".into(),
        };

        let before = EvalSnapshot {
            diagnostics: vec![d1.clone()],
            ..make_snapshot(EvalCategory::StaticAnalysis, "clippy", 0, 1)
        };
        let after = EvalSnapshot {
            diagnostics: vec![d2.clone()],
            ..make_snapshot(EvalCategory::StaticAnalysis, "clippy", 0, 1)
        };

        let delta = EvalDelta::compute(before, after);
        assert_eq!(delta.regressions.len(), 1);
        assert_eq!(delta.regressions[0], d2);
        assert_eq!(delta.fixed.len(), 1);
        assert_eq!(delta.fixed[0], d1);
    }

    #[test]
    fn test_evaluation_result_summary() {
        let before = make_snapshot(EvalCategory::Test, "cargo test", 10, 2);
        let after = make_snapshot(EvalCategory::Test, "cargo test", 12, 0);
        let delta = EvalDelta::compute(before, after);
        let result = EvaluationResult::new(1, "org/repo".into(), vec![delta]);
        assert!(result.overall_improved);
        assert!(!result.summary.is_empty());
    }

    #[test]
    fn test_evaluation_result_pr_comment() {
        let before = make_snapshot(EvalCategory::Test, "cargo test", 10, 2);
        let after = make_snapshot(EvalCategory::Test, "cargo test", 12, 0);
        let delta = EvalDelta::compute(before, after);
        let result = EvaluationResult::new(1, "org/repo".into(), vec![delta]);
        let comment = result.format_pr_comment();
        assert!(comment.contains("Code Quality Evaluation"));
        assert!(comment.contains("cargo test"));
        assert!(comment.contains("Before"));
        assert!(comment.contains("After"));
    }

    #[test]
    fn test_evaluation_result_empty_deltas() {
        let result = EvaluationResult::new(1, "org/repo".into(), vec![]);
        assert!(!result.overall_improved);
        assert_eq!(result.summary, "No evaluation tools ran.");
    }

    #[test]
    fn test_coverage_delta() {
        let before = EvalSnapshot {
            line_coverage_pct: Some(80.0),
            ..make_snapshot(EvalCategory::Coverage, "llvm-cov", 10, 0)
        };
        let after = EvalSnapshot {
            line_coverage_pct: Some(85.5),
            ..make_snapshot(EvalCategory::Coverage, "llvm-cov", 10, 0)
        };
        let delta = EvalDelta::compute(before, after);
        assert!((delta.coverage_delta_pct.unwrap() - 5.5).abs() < 0.01);
        assert!(delta.is_improvement());
    }

    #[test]
    fn test_eval_category_display() {
        assert_eq!(EvalCategory::Test.to_string(), "test");
        assert_eq!(EvalCategory::Lint.to_string(), "lint");
        assert_eq!(EvalCategory::StaticAnalysis.to_string(), "static_analysis");
        assert_eq!(EvalCategory::Coverage.to_string(), "coverage");
    }

    #[test]
    fn test_eval_category_serde_roundtrip() {
        let categories = vec![
            EvalCategory::Test,
            EvalCategory::Lint,
            EvalCategory::StaticAnalysis,
            EvalCategory::Coverage,
        ];
        for cat in categories {
            let json = serde_json::to_string(&cat).unwrap();
            let parsed: EvalCategory = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, cat);
        }
    }

    #[test]
    fn test_eval_snapshot_default() {
        let snap = EvalSnapshot::default();
        assert_eq!(snap.category, EvalCategory::Test);
        assert_eq!(snap.tool_name, "");
        assert_eq!(snap.exit_code, -1);
        assert_eq!(snap.passed, 0);
        assert_eq!(snap.failed, 0);
        assert_eq!(snap.skipped, 0);
        assert_eq!(snap.warnings, 0);
        assert_eq!(snap.errors, 0);
        assert!(snap.diagnostics.is_empty());
        assert!(snap.raw_output.is_empty());
        assert_eq!(snap.duration_secs, 0.0);
        assert!(snap.line_coverage_pct.is_none());
        assert!(snap.branch_coverage_pct.is_none());
    }

    #[test]
    fn test_eval_snapshot_serde_roundtrip() {
        let snap = make_snapshot(EvalCategory::Test, "cargo test", 5, 2);
        let json = serde_json::to_string(&snap).unwrap();
        let parsed: EvalSnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.passed, 5);
        assert_eq!(parsed.failed, 2);
        assert_eq!(parsed.tool_name, "cargo test");
    }

    #[test]
    fn test_eval_delta_coverage_none() {
        let before = make_snapshot(EvalCategory::Coverage, "cov", 0, 0);
        let after = make_snapshot(EvalCategory::Coverage, "cov", 0, 0);
        let delta = EvalDelta::compute(before, after);
        assert!(delta.coverage_delta_pct.is_none());
        assert!(delta.is_improvement());
    }

    #[test]
    fn test_eval_delta_coverage_decrease_not_improvement() {
        let before = EvalSnapshot {
            line_coverage_pct: Some(90.0),
            ..make_snapshot(EvalCategory::Coverage, "cov", 10, 0)
        };
        let after = EvalSnapshot {
            line_coverage_pct: Some(85.0),
            ..make_snapshot(EvalCategory::Coverage, "cov", 10, 0)
        };
        let delta = EvalDelta::compute(before, after);
        assert!((delta.coverage_delta_pct.unwrap() - (-5.0)).abs() < 0.01);
        assert!(!delta.is_improvement());
    }

    #[test]
    fn test_eval_delta_new_regressions_not_improvement() {
        let d1 = Diagnostic {
            file: "src/lib.rs".into(),
            line: Some(10),
            column: None,
            severity: DiagnosticSeverity::Error,
            code: None,
            message: "new error".into(),
        };
        let before = make_snapshot(EvalCategory::StaticAnalysis, "clippy", 0, 0);
        let after = EvalSnapshot {
            diagnostics: vec![d1],
            ..make_snapshot(EvalCategory::StaticAnalysis, "clippy", 0, 0)
        };
        let delta = EvalDelta::compute(before, after);
        assert!(!delta.regressions.is_empty());
        assert!(!delta.is_improvement());
    }

    #[test]
    fn test_eval_delta_serde_roundtrip() {
        let before = make_snapshot(EvalCategory::Test, "test", 10, 2);
        let after = make_snapshot(EvalCategory::Test, "test", 12, 0);
        let delta = EvalDelta::compute(before, after);
        let json = serde_json::to_string(&delta).unwrap();
        let parsed: EvalDelta = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.new_passes, 2);
        assert_eq!(parsed.new_failures, -2);
    }

    #[test]
    fn test_evaluation_result_serde_roundtrip() {
        let result = EvaluationResult::new(1, "repo".into(), vec![]);
        let json = serde_json::to_string(&result).unwrap();
        let parsed: EvaluationResult = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.attempt_id, 1);
        assert_eq!(parsed.repo, "repo");
    }

    #[test]
    fn test_evaluation_result_not_overall_improved_with_regression() {
        let before = make_snapshot(EvalCategory::Test, "test", 10, 0);
        let after = make_snapshot(EvalCategory::Test, "test", 8, 2);
        let delta = EvalDelta::compute(before, after);
        let result = EvaluationResult::new(1, "repo".into(), vec![delta]);
        assert!(!result.overall_improved);
    }

    #[test]
    fn test_evaluation_result_summary_with_coverage() {
        let before = EvalSnapshot {
            line_coverage_pct: Some(80.0),
            ..make_snapshot(EvalCategory::Coverage, "cov", 10, 0)
        };
        let after = EvalSnapshot {
            line_coverage_pct: Some(85.5),
            ..make_snapshot(EvalCategory::Coverage, "cov", 10, 0)
        };
        let delta = EvalDelta::compute(before, after);
        let result = EvaluationResult::new(1, "repo".into(), vec![delta]);
        assert!(result.summary.contains("coverage"));
    }

    #[test]
    fn test_evaluation_result_summary_with_failures() {
        let before = make_snapshot(EvalCategory::Test, "cargo test", 10, 0);
        let after = make_snapshot(EvalCategory::Test, "cargo test", 8, 2);
        let delta = EvalDelta::compute(before, after);
        let result = EvaluationResult::new(1, "repo".into(), vec![delta]);
        assert!(result.summary.contains("new failure"));
    }

    #[test]
    fn test_evaluation_result_summary_with_fixed() {
        let d1 = Diagnostic {
            file: "src/lib.rs".into(),
            line: Some(10),
            column: None,
            severity: DiagnosticSeverity::Warning,
            code: None,
            message: "old warning".into(),
        };
        let before = EvalSnapshot {
            diagnostics: vec![d1],
            ..make_snapshot(EvalCategory::StaticAnalysis, "clippy", 0, 0)
        };
        let after = make_snapshot(EvalCategory::StaticAnalysis, "clippy", 0, 0);
        let delta = EvalDelta::compute(before, after);
        let result = EvaluationResult::new(1, "repo".into(), vec![delta]);
        assert!(result.summary.contains("fixed"));
    }

    #[test]
    fn test_format_pr_comment_with_regressions() {
        let d1 = Diagnostic {
            file: "src/lib.rs".into(),
            line: Some(42),
            column: None,
            severity: DiagnosticSeverity::Error,
            code: Some("E001".into()),
            message: "type mismatch".into(),
        };
        let before = make_snapshot(EvalCategory::StaticAnalysis, "clippy", 0, 0);
        let after = EvalSnapshot {
            diagnostics: vec![d1],
            ..make_snapshot(EvalCategory::StaticAnalysis, "clippy", 0, 1)
        };
        let delta = EvalDelta::compute(before, after);
        let result = EvaluationResult::new(1, "repo".into(), vec![delta]);
        let comment = result.format_pr_comment();
        assert!(comment.contains("Regressions detected"));
        assert!(comment.contains("New issues"));
        assert!(comment.contains("src/lib.rs:42"));
        assert!(comment.contains("type mismatch"));
    }

    #[test]
    fn test_format_pr_comment_with_many_regressions_truncated() {
        let diagnostics: Vec<Diagnostic> = (0..15)
            .map(|i| Diagnostic {
                file: format!("src/file{}.rs", i),
                line: Some(i),
                column: None,
                severity: DiagnosticSeverity::Warning,
                code: None,
                message: format!("warning {}", i),
            })
            .collect();
        let before = make_snapshot(EvalCategory::StaticAnalysis, "clippy", 0, 0);
        let after = EvalSnapshot {
            diagnostics,
            ..make_snapshot(EvalCategory::StaticAnalysis, "clippy", 0, 15)
        };
        let delta = EvalDelta::compute(before, after);
        let result = EvaluationResult::new(1, "repo".into(), vec![delta]);
        let comment = result.format_pr_comment();
        assert!(comment.contains("...and 5 more"));
    }

    #[test]
    fn test_format_pr_comment_with_fixed_issues() {
        let d1 = Diagnostic {
            file: "src/old.rs".into(),
            line: Some(1),
            column: None,
            severity: DiagnosticSeverity::Warning,
            code: None,
            message: "old issue".into(),
        };
        let before = EvalSnapshot {
            diagnostics: vec![d1],
            ..make_snapshot(EvalCategory::StaticAnalysis, "clippy", 0, 1)
        };
        let after = make_snapshot(EvalCategory::StaticAnalysis, "clippy", 0, 0);
        let delta = EvalDelta::compute(before, after);
        let result = EvaluationResult::new(1, "repo".into(), vec![delta]);
        let comment = result.format_pr_comment();
        assert!(comment.contains("Fixed"));
        assert!(comment.contains("1 issue(s) resolved"));
    }

    #[test]
    fn test_format_pr_comment_with_coverage() {
        let before = EvalSnapshot {
            line_coverage_pct: Some(70.0),
            ..make_snapshot(EvalCategory::Coverage, "cov", 10, 0)
        };
        let after = EvalSnapshot {
            line_coverage_pct: Some(85.0),
            ..make_snapshot(EvalCategory::Coverage, "cov", 10, 0)
        };
        let delta = EvalDelta::compute(before, after);
        let result = EvaluationResult::new(1, "repo".into(), vec![delta]);
        let comment = result.format_pr_comment();
        assert!(comment.contains("Coverage"));
        assert!(comment.contains("70.0%"));
        assert!(comment.contains("85.0%"));
    }

    #[test]
    fn test_format_pr_comment_multiple_deltas() {
        let d1 = EvalDelta::compute(
            make_snapshot(EvalCategory::Test, "cargo test", 10, 2),
            make_snapshot(EvalCategory::Test, "cargo test", 12, 0),
        );
        let d2 = EvalDelta::compute(
            make_snapshot(EvalCategory::Lint, "cargo fmt", 0, 0),
            make_snapshot(EvalCategory::Lint, "cargo fmt", 0, 0),
        );
        let result = EvaluationResult::new(1, "repo".into(), vec![d1, d2]);
        let comment = result.format_pr_comment();
        assert!(comment.contains("cargo test"));
        assert!(comment.contains("cargo fmt"));
        assert!(comment.contains("Improved"));
    }

    #[test]
    fn test_diagnostic_severity_serde() {
        let severities = vec![
            DiagnosticSeverity::Error,
            DiagnosticSeverity::Warning,
            DiagnosticSeverity::Info,
        ];
        for sev in severities {
            let json = serde_json::to_string(&sev).unwrap();
            let parsed: DiagnosticSeverity = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, sev);
        }
    }

    #[test]
    fn test_diagnostic_serde_roundtrip() {
        let d = Diagnostic {
            file: "src/main.rs".into(),
            line: Some(42),
            column: Some(10),
            severity: DiagnosticSeverity::Error,
            code: Some("E0001".into()),
            message: "type error".into(),
        };
        let json = serde_json::to_string(&d).unwrap();
        let parsed: Diagnostic = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, d);
    }

    #[test]
    fn test_diagnostic_hash_equal() {
        let d1 = Diagnostic {
            file: "src/main.rs".into(),
            line: Some(42),
            column: None,
            severity: DiagnosticSeverity::Warning,
            code: None,
            message: "warning".into(),
        };
        let d2 = d1.clone();
        let mut set = std::collections::HashSet::new();
        set.insert(d1);
        set.insert(d2);
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn test_eval_delta_no_changes() {
        let before = make_snapshot(EvalCategory::Test, "test", 10, 0);
        let after = make_snapshot(EvalCategory::Test, "test", 10, 0);
        let delta = EvalDelta::compute(before, after);
        assert_eq!(delta.new_passes, 0);
        assert_eq!(delta.new_failures, 0);
        assert!(delta.regressions.is_empty());
        assert!(delta.fixed.is_empty());
        assert!(delta.is_improvement());
    }

    #[test]
    fn test_evaluation_result_mixed_improvements_and_regressions() {
        let d1 = EvalDelta::compute(
            make_snapshot(EvalCategory::Test, "test", 10, 0),
            make_snapshot(EvalCategory::Test, "test", 12, 0),
        );
        let d2 = EvalDelta::compute(
            make_snapshot(EvalCategory::Lint, "lint", 0, 0),
            make_snapshot(EvalCategory::Lint, "lint", 0, 2),
        );
        let result = EvaluationResult::new(1, "repo".into(), vec![d1, d2]);
        assert!(!result.overall_improved); // one delta is not improvement
    }
}
