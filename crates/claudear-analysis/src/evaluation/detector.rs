//! Auto-detect available evaluation tools for a project.

use super::types::EvalCategory;
use std::path::Path;

/// A detected evaluation tool with its command.
#[derive(Debug, Clone)]
pub struct DetectedTool {
    pub category: EvalCategory,
    pub name: String,
    pub command: Vec<String>,
}

/// Configuration overrides for evaluation tools.
#[derive(Debug, Clone, Default)]
pub struct ToolOverrides {
    pub custom_test_cmd: Option<String>,
    pub custom_lint_cmd: Option<String>,
    pub custom_analysis_cmd: Option<String>,
    pub custom_coverage_cmd: Option<String>,
}

/// Detect available evaluation tools for a project directory.
pub fn detect_tools(project_dir: &Path, overrides: &ToolOverrides) -> Vec<DetectedTool> {
    let mut tools = Vec::new();

    // Check for custom overrides first
    if let Some(ref cmd) = overrides.custom_test_cmd {
        tools.push(DetectedTool {
            category: EvalCategory::Test,
            name: "custom".into(),
            command: shell_words(cmd),
        });
    }
    if let Some(ref cmd) = overrides.custom_lint_cmd {
        tools.push(DetectedTool {
            category: EvalCategory::Lint,
            name: "custom".into(),
            command: shell_words(cmd),
        });
    }
    if let Some(ref cmd) = overrides.custom_analysis_cmd {
        tools.push(DetectedTool {
            category: EvalCategory::StaticAnalysis,
            name: "custom".into(),
            command: shell_words(cmd),
        });
    }
    if let Some(ref cmd) = overrides.custom_coverage_cmd {
        tools.push(DetectedTool {
            category: EvalCategory::Coverage,
            name: "custom".into(),
            command: shell_words(cmd),
        });
    }

    // Only auto-detect categories not already covered by overrides
    let has_test = tools.iter().any(|t| t.category == EvalCategory::Test);
    let has_lint = tools.iter().any(|t| t.category == EvalCategory::Lint);
    let has_analysis = tools
        .iter()
        .any(|t| t.category == EvalCategory::StaticAnalysis);
    let has_coverage = tools.iter().any(|t| t.category == EvalCategory::Coverage);

    // Rust (Cargo.toml)
    if project_dir.join("Cargo.toml").exists() {
        if !has_test {
            tools.push(DetectedTool {
                category: EvalCategory::Test,
                name: "cargo test".into(),
                command: vec![
                    "cargo".into(),
                    "test".into(),
                    "--".into(),
                    "--format".into(),
                    "json".into(),
                ],
            });
        }
        if !has_analysis && which_exists("cargo") {
            tools.push(DetectedTool {
                category: EvalCategory::StaticAnalysis,
                name: "cargo clippy".into(),
                command: vec![
                    "cargo".into(),
                    "clippy".into(),
                    "--message-format=json".into(),
                    "--".into(),
                    "-D".into(),
                    "warnings".into(),
                ],
            });
        }
        if !has_lint && which_exists("cargo") {
            tools.push(DetectedTool {
                category: EvalCategory::Lint,
                name: "cargo fmt".into(),
                command: vec!["cargo".into(), "fmt".into(), "--check".into()],
            });
        }
        if !has_coverage && which_exists("cargo") && which_exists("cargo-llvm-cov") {
            tools.push(DetectedTool {
                category: EvalCategory::Coverage,
                name: "cargo llvm-cov".into(),
                command: vec!["cargo".into(), "llvm-cov".into(), "--json".into()],
            });
        }
    }

    // PHP (composer.json)
    if project_dir.join("composer.json").exists() {
        let vendor = project_dir.join("vendor/bin");
        if !has_test && vendor.join("phpunit").exists() {
            tools.push(DetectedTool {
                category: EvalCategory::Test,
                name: "phpunit".into(),
                command: vec![
                    "vendor/bin/phpunit".into(),
                    "--log-junit".into(),
                    "-".into(),
                ],
            });
        }
        if !has_analysis && vendor.join("phpstan").exists() {
            tools.push(DetectedTool {
                category: EvalCategory::StaticAnalysis,
                name: "phpstan".into(),
                command: vec![
                    "vendor/bin/phpstan".into(),
                    "analyse".into(),
                    "--error-format=json".into(),
                ],
            });
        }
        if !has_lint && vendor.join("php-cs-fixer").exists() {
            tools.push(DetectedTool {
                category: EvalCategory::Lint,
                name: "php-cs-fixer".into(),
                command: vec![
                    "vendor/bin/php-cs-fixer".into(),
                    "fix".into(),
                    "--dry-run".into(),
                    "--format=json".into(),
                ],
            });
        }
        if !has_coverage && vendor.join("phpunit").exists() {
            tools.push(DetectedTool {
                category: EvalCategory::Coverage,
                name: "phpunit coverage".into(),
                command: vec![
                    "vendor/bin/phpunit".into(),
                    "--coverage-clover".into(),
                    "php://stdout".into(),
                ],
            });
        }
    }

    // JS/TS (package.json)
    if project_dir.join("package.json").exists() {
        if !has_test && (which_exists("npx") || which_exists("jest")) {
            tools.push(DetectedTool {
                category: EvalCategory::Test,
                name: "jest".into(),
                command: vec!["npx".into(), "jest".into(), "--json".into()],
            });
        }
        if !has_analysis && which_exists("npx") {
            tools.push(DetectedTool {
                category: EvalCategory::StaticAnalysis,
                name: "eslint".into(),
                command: vec![
                    "npx".into(),
                    "eslint".into(),
                    "--format".into(),
                    "json".into(),
                    ".".into(),
                ],
            });
        }
        if !has_lint && which_exists("npx") {
            tools.push(DetectedTool {
                category: EvalCategory::Lint,
                name: "prettier".into(),
                command: vec![
                    "npx".into(),
                    "prettier".into(),
                    "--check".into(),
                    ".".into(),
                ],
            });
        }
        if !has_coverage && which_exists("npx") {
            tools.push(DetectedTool {
                category: EvalCategory::Coverage,
                name: "jest coverage".into(),
                command: vec![
                    "npx".into(),
                    "jest".into(),
                    "--coverage".into(),
                    "--coverageReporters=json-summary".into(),
                ],
            });
        }
    }

    // Kotlin (build.gradle or build.gradle.kts)
    if project_dir.join("build.gradle.kts").exists() || project_dir.join("build.gradle").exists() {
        let gradlew = if project_dir.join("gradlew").exists() {
            "./gradlew"
        } else {
            "gradle"
        };
        if !has_test {
            tools.push(DetectedTool {
                category: EvalCategory::Test,
                name: "gradle test".into(),
                command: vec![gradlew.into(), "test".into()],
            });
        }
        if !has_analysis && which_exists(gradlew) {
            tools.push(DetectedTool {
                category: EvalCategory::StaticAnalysis,
                name: "detekt".into(),
                command: vec![gradlew.into(), "detekt".into()],
            });
        }
        if !has_lint && which_exists(gradlew) {
            tools.push(DetectedTool {
                category: EvalCategory::Lint,
                name: "ktlint".into(),
                command: vec![gradlew.into(), "ktlintCheck".into()],
            });
        }
        if !has_coverage && which_exists(gradlew) {
            tools.push(DetectedTool {
                category: EvalCategory::Coverage,
                name: "jacoco".into(),
                command: vec![gradlew.into(), "jacocoTestReport".into()],
            });
        }
    }

    // Swift (Package.swift)
    if project_dir.join("Package.swift").exists() {
        if !has_test && which_exists("swift") {
            tools.push(DetectedTool {
                category: EvalCategory::Test,
                name: "swift test".into(),
                command: vec!["swift".into(), "test".into()],
            });
        }
        if !has_analysis && which_exists("swiftlint") {
            tools.push(DetectedTool {
                category: EvalCategory::StaticAnalysis,
                name: "swiftlint".into(),
                command: vec![
                    "swiftlint".into(),
                    "lint".into(),
                    "--reporter".into(),
                    "json".into(),
                ],
            });
        }
        if !has_lint && which_exists("swift-format") {
            tools.push(DetectedTool {
                category: EvalCategory::Lint,
                name: "swift-format".into(),
                command: vec![
                    "swift-format".into(),
                    "lint".into(),
                    "--recursive".into(),
                    ".".into(),
                ],
            });
        }
        if !has_coverage && which_exists("swift") {
            tools.push(DetectedTool {
                category: EvalCategory::Coverage,
                name: "swift coverage".into(),
                command: vec![
                    "swift".into(),
                    "test".into(),
                    "--enable-code-coverage".into(),
                ],
            });
        }
    }

    // Python (pyproject.toml / requirements.txt / setup.py)
    if project_dir.join("pyproject.toml").exists()
        || project_dir.join("requirements.txt").exists()
        || project_dir.join("setup.py").exists()
    {
        if !has_test && which_exists("pytest") {
            tools.push(DetectedTool {
                category: EvalCategory::Test,
                name: "pytest".into(),
                command: vec!["pytest".into(), "--tb=short".into(), "-q".into()],
            });
        }
        if !has_analysis {
            if which_exists("mypy") {
                tools.push(DetectedTool {
                    category: EvalCategory::StaticAnalysis,
                    name: "mypy".into(),
                    command: vec!["mypy".into(), ".".into()],
                });
            } else if which_exists("ruff") {
                tools.push(DetectedTool {
                    category: EvalCategory::StaticAnalysis,
                    name: "ruff check".into(),
                    command: vec!["ruff".into(), "check".into(), ".".into()],
                });
            }
        }
        if !has_lint {
            if which_exists("ruff") {
                tools.push(DetectedTool {
                    category: EvalCategory::Lint,
                    name: "ruff format".into(),
                    command: vec!["ruff".into(), "format".into(), "--check".into(), ".".into()],
                });
            } else if which_exists("black") {
                tools.push(DetectedTool {
                    category: EvalCategory::Lint,
                    name: "black".into(),
                    command: vec!["black".into(), "--check".into(), ".".into()],
                });
            }
        }
        if !has_coverage && which_exists("pytest") {
            tools.push(DetectedTool {
                category: EvalCategory::Coverage,
                name: "pytest coverage".into(),
                command: vec![
                    "pytest".into(),
                    "--cov=.".into(),
                    "--cov-report=term".into(),
                    "-q".into(),
                ],
            });
        }
    }

    // Go (go.mod)
    if project_dir.join("go.mod").exists() {
        if !has_test && which_exists("go") {
            tools.push(DetectedTool {
                category: EvalCategory::Test,
                name: "go test".into(),
                command: vec!["go".into(), "test".into(), "-json".into(), "./...".into()],
            });
        }
        if !has_analysis && which_exists("go") {
            tools.push(DetectedTool {
                category: EvalCategory::StaticAnalysis,
                name: "go vet".into(),
                command: vec!["go".into(), "vet".into(), "./...".into()],
            });
        }
        if !has_lint && which_exists("gofmt") {
            tools.push(DetectedTool {
                category: EvalCategory::Lint,
                name: "gofmt".into(),
                command: vec!["gofmt".into(), "-l".into(), ".".into()],
            });
        }
        if !has_coverage && which_exists("go") {
            tools.push(DetectedTool {
                category: EvalCategory::Coverage,
                name: "go test coverage".into(),
                command: vec!["go".into(), "test".into(), "-cover".into(), "./...".into()],
            });
        }
    }

    // Java (pom.xml — Gradle covered by Kotlin section)
    if project_dir.join("pom.xml").exists() {
        if !has_test && which_exists("mvn") {
            tools.push(DetectedTool {
                category: EvalCategory::Test,
                name: "mvn test".into(),
                command: vec!["mvn".into(), "test".into(), "-B".into()],
            });
        }
        if !has_analysis && which_exists("mvn") {
            tools.push(DetectedTool {
                category: EvalCategory::StaticAnalysis,
                name: "mvn verify".into(),
                command: vec![
                    "mvn".into(),
                    "verify".into(),
                    "-B".into(),
                    "-DskipTests".into(),
                ],
            });
        }
        if !has_lint && which_exists("google-java-format") {
            tools.push(DetectedTool {
                category: EvalCategory::Lint,
                name: "google-java-format".into(),
                command: vec!["google-java-format".into(), "--dry-run".into()],
            });
        }
        if !has_coverage && which_exists("mvn") {
            tools.push(DetectedTool {
                category: EvalCategory::Coverage,
                name: "mvn jacoco".into(),
                command: vec![
                    "mvn".into(),
                    "test".into(),
                    "jacoco:report".into(),
                    "-B".into(),
                ],
            });
        }
    }

    // C/C++ (CMakeLists.txt / Makefile)
    if project_dir.join("CMakeLists.txt").exists() || project_dir.join("Makefile").exists() {
        let has_cmake = project_dir.join("CMakeLists.txt").exists();
        if !has_test {
            if has_cmake && which_exists("ctest") {
                tools.push(DetectedTool {
                    category: EvalCategory::Test,
                    name: "ctest".into(),
                    command: vec!["ctest".into()],
                });
            } else if which_exists("make") {
                tools.push(DetectedTool {
                    category: EvalCategory::Test,
                    name: "make test".into(),
                    command: vec!["make".into(), "test".into()],
                });
            }
        }
        if !has_analysis && which_exists("cppcheck") {
            tools.push(DetectedTool {
                category: EvalCategory::StaticAnalysis,
                name: "cppcheck".into(),
                command: vec!["cppcheck".into(), "--enable=all".into(), ".".into()],
            });
        }
        if !has_lint && which_exists("clang-format") {
            tools.push(DetectedTool {
                category: EvalCategory::Lint,
                name: "clang-format".into(),
                command: vec!["clang-format".into(), "--dry-run".into(), "-Werror".into()],
            });
        }
        if !has_coverage && which_exists("gcov") {
            tools.push(DetectedTool {
                category: EvalCategory::Coverage,
                name: "gcov".into(),
                command: vec!["gcov".into()],
            });
        }
    }

    // Ruby (Gemfile)
    if project_dir.join("Gemfile").exists() {
        if !has_test {
            if which_exists("rspec") {
                tools.push(DetectedTool {
                    category: EvalCategory::Test,
                    name: "rspec".into(),
                    command: vec!["rspec".into()],
                });
            } else if which_exists("rake") {
                tools.push(DetectedTool {
                    category: EvalCategory::Test,
                    name: "rake test".into(),
                    command: vec!["rake".into(), "test".into()],
                });
            }
        }
        if !has_analysis && which_exists("rubocop") {
            tools.push(DetectedTool {
                category: EvalCategory::StaticAnalysis,
                name: "rubocop".into(),
                command: vec!["rubocop".into(), "--format".into(), "json".into()],
            });
        }
        if !has_lint && which_exists("rubocop") {
            tools.push(DetectedTool {
                category: EvalCategory::Lint,
                name: "rubocop lint".into(),
                command: vec![
                    "rubocop".into(),
                    "--auto-correct-all".into(),
                    "--dry-run".into(),
                ],
            });
        }
        if !has_coverage && which_exists("rspec") {
            tools.push(DetectedTool {
                category: EvalCategory::Coverage,
                name: "rspec coverage".into(),
                command: vec!["rspec".into()],
            });
        }
    }

    // Dart (pubspec.yaml)
    if project_dir.join("pubspec.yaml").exists() {
        if !has_test && which_exists("dart") {
            tools.push(DetectedTool {
                category: EvalCategory::Test,
                name: "dart test".into(),
                command: vec!["dart".into(), "test".into()],
            });
        }
        if !has_analysis && which_exists("dart") {
            tools.push(DetectedTool {
                category: EvalCategory::StaticAnalysis,
                name: "dart analyze".into(),
                command: vec!["dart".into(), "analyze".into()],
            });
        }
        if !has_lint && which_exists("dart") {
            tools.push(DetectedTool {
                category: EvalCategory::Lint,
                name: "dart format".into(),
                command: vec![
                    "dart".into(),
                    "format".into(),
                    "--output=none".into(),
                    "--set-exit-if-changed".into(),
                    ".".into(),
                ],
            });
        }
        if !has_coverage && which_exists("dart") {
            tools.push(DetectedTool {
                category: EvalCategory::Coverage,
                name: "dart test coverage".into(),
                command: vec!["dart".into(), "test".into(), "--coverage=coverage".into()],
            });
        }
    }

    // C# (.csproj or .sln)
    let has_csproj = project_dir.read_dir().is_ok_and(|mut entries| {
        entries.any(|e| {
            e.ok().is_some_and(|e| {
                let name = e.file_name().to_string_lossy().to_string();
                name.ends_with(".csproj") || name.ends_with(".sln")
            })
        })
    });
    if has_csproj && which_exists("dotnet") {
        if !has_test {
            tools.push(DetectedTool {
                category: EvalCategory::Test,
                name: "dotnet test".into(),
                command: vec![
                    "dotnet".into(),
                    "test".into(),
                    "--logger".into(),
                    "trx".into(),
                ],
            });
        }
        if !has_analysis {
            tools.push(DetectedTool {
                category: EvalCategory::StaticAnalysis,
                name: "dotnet build".into(),
                command: vec!["dotnet".into(), "build".into(), "/warnaserror".into()],
            });
        }
        if !has_lint {
            tools.push(DetectedTool {
                category: EvalCategory::Lint,
                name: "dotnet format".into(),
                command: vec![
                    "dotnet".into(),
                    "format".into(),
                    "--verify-no-changes".into(),
                ],
            });
        }
        if !has_coverage {
            tools.push(DetectedTool {
                category: EvalCategory::Coverage,
                name: "dotnet coverage".into(),
                command: vec![
                    "dotnet".into(),
                    "test".into(),
                    "--collect:XPlat Code Coverage".into(),
                ],
            });
        }
    }

    tools
}

fn which_exists(binary: &str) -> bool {
    claudear_core::platform::command_exists(binary)
}

fn shell_words(cmd: &str) -> Vec<String> {
    cmd.split_whitespace().map(String::from).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_detect_rust_project() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("Cargo.toml"), "[package]\nname = \"test\"").unwrap();
        let tools = detect_tools(dir.path(), &ToolOverrides::default());
        // Should detect at least cargo test
        assert!(tools.iter().any(|t| t.name.contains("cargo")));
    }

    #[test]
    fn test_custom_overrides() {
        let dir = TempDir::new().unwrap();
        let overrides = ToolOverrides {
            custom_test_cmd: Some("make test".into()),
            ..Default::default()
        };
        let tools = detect_tools(dir.path(), &overrides);
        assert!(tools
            .iter()
            .any(|t| t.name == "custom" && t.category == EvalCategory::Test));
    }

    #[test]
    fn test_empty_dir_no_tools() {
        let dir = TempDir::new().unwrap();
        let tools = detect_tools(dir.path(), &ToolOverrides::default());
        assert!(tools.is_empty());
    }

    #[test]
    fn test_shell_words() {
        assert_eq!(
            shell_words("cargo test --json"),
            vec!["cargo", "test", "--json"]
        );
    }

    #[test]
    fn test_shell_words_single_word() {
        assert_eq!(shell_words("cargo"), vec!["cargo"]);
    }

    #[test]
    fn test_shell_words_empty_string() {
        let result: Vec<String> = shell_words("");
        assert!(result.is_empty());
    }

    #[test]
    fn test_shell_words_extra_whitespace() {
        assert_eq!(
            shell_words("  cargo   test   --json  "),
            vec!["cargo", "test", "--json"]
        );
    }

    #[test]
    fn test_shell_words_tabs_and_spaces() {
        assert_eq!(
            shell_words("cargo\ttest\t--json"),
            vec!["cargo", "test", "--json"]
        );
    }

    #[test]
    fn test_custom_lint_override() {
        let dir = TempDir::new().unwrap();
        let overrides = ToolOverrides {
            custom_lint_cmd: Some("eslint --fix .".into()),
            ..Default::default()
        };
        let tools = detect_tools(dir.path(), &overrides);
        assert!(tools
            .iter()
            .any(|t| t.name == "custom" && t.category == EvalCategory::Lint));
        let lint_tool = tools
            .iter()
            .find(|t| t.category == EvalCategory::Lint)
            .unwrap();
        assert_eq!(lint_tool.command, vec!["eslint", "--fix", "."]);
    }

    #[test]
    fn test_custom_analysis_override() {
        let dir = TempDir::new().unwrap();
        let overrides = ToolOverrides {
            custom_analysis_cmd: Some("mypy --strict .".into()),
            ..Default::default()
        };
        let tools = detect_tools(dir.path(), &overrides);
        assert!(tools
            .iter()
            .any(|t| t.name == "custom" && t.category == EvalCategory::StaticAnalysis));
    }

    #[test]
    fn test_custom_coverage_override() {
        let dir = TempDir::new().unwrap();
        let overrides = ToolOverrides {
            custom_coverage_cmd: Some("coverage run".into()),
            ..Default::default()
        };
        let tools = detect_tools(dir.path(), &overrides);
        assert!(tools
            .iter()
            .any(|t| t.name == "custom" && t.category == EvalCategory::Coverage));
    }

    #[test]
    fn test_all_custom_overrides() {
        let dir = TempDir::new().unwrap();
        let overrides = ToolOverrides {
            custom_test_cmd: Some("make test".into()),
            custom_lint_cmd: Some("make lint".into()),
            custom_analysis_cmd: Some("make analyze".into()),
            custom_coverage_cmd: Some("make coverage".into()),
        };
        let tools = detect_tools(dir.path(), &overrides);
        assert_eq!(tools.len(), 4);
        assert!(tools.iter().all(|t| t.name == "custom"));
        assert!(tools.iter().any(|t| t.category == EvalCategory::Test));
        assert!(tools.iter().any(|t| t.category == EvalCategory::Lint));
        assert!(tools
            .iter()
            .any(|t| t.category == EvalCategory::StaticAnalysis));
        assert!(tools.iter().any(|t| t.category == EvalCategory::Coverage));
    }

    #[test]
    fn test_custom_test_override_prevents_rust_autodetect() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("Cargo.toml"), "[package]\nname = \"test\"").unwrap();
        let overrides = ToolOverrides {
            custom_test_cmd: Some("my-test-runner".into()),
            ..Default::default()
        };
        let tools = detect_tools(dir.path(), &overrides);
        // Should have the custom test tool, not cargo test
        let test_tools: Vec<_> = tools
            .iter()
            .filter(|t| t.category == EvalCategory::Test)
            .collect();
        assert_eq!(test_tools.len(), 1);
        assert_eq!(test_tools[0].name, "custom");
    }

    #[test]
    fn test_detected_tool_clone() {
        let tool = DetectedTool {
            category: EvalCategory::Test,
            name: "cargo test".into(),
            command: vec!["cargo".into(), "test".into()],
        };
        let cloned = tool.clone();
        assert_eq!(cloned.name, "cargo test");
        assert_eq!(cloned.category, EvalCategory::Test);
        assert_eq!(cloned.command, vec!["cargo", "test"]);
    }

    #[test]
    fn test_detected_tool_debug() {
        let tool = DetectedTool {
            category: EvalCategory::Lint,
            name: "eslint".into(),
            command: vec!["npx".into(), "eslint".into()],
        };
        let dbg = format!("{:?}", tool);
        assert!(dbg.contains("Lint"));
        assert!(dbg.contains("eslint"));
    }

    #[test]
    fn test_tool_overrides_default() {
        let overrides = ToolOverrides::default();
        assert!(overrides.custom_test_cmd.is_none());
        assert!(overrides.custom_lint_cmd.is_none());
        assert!(overrides.custom_analysis_cmd.is_none());
        assert!(overrides.custom_coverage_cmd.is_none());
    }

    #[test]
    fn test_tool_overrides_clone() {
        let overrides = ToolOverrides {
            custom_test_cmd: Some("make test".into()),
            custom_lint_cmd: None,
            custom_analysis_cmd: Some("mypy .".into()),
            custom_coverage_cmd: None,
        };
        let cloned = overrides.clone();
        assert_eq!(cloned.custom_test_cmd, Some("make test".into()));
        assert!(cloned.custom_lint_cmd.is_none());
        assert_eq!(cloned.custom_analysis_cmd, Some("mypy .".into()));
    }

    #[test]
    fn test_tool_overrides_debug() {
        let overrides = ToolOverrides::default();
        let dbg = format!("{:?}", overrides);
        assert!(dbg.contains("ToolOverrides"));
    }

    #[test]
    fn test_detect_js_project() {
        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("package.json"),
            r#"{"name": "test", "version": "1.0.0"}"#,
        )
        .unwrap();
        let tools = detect_tools(dir.path(), &ToolOverrides::default());
        // Should detect at least some JS tools (jest, eslint, prettier)
        // depending on whether npx is available
        // At minimum, the detection logic should not panic
        for tool in &tools {
            assert!(
                tool.category == EvalCategory::Test
                    || tool.category == EvalCategory::Lint
                    || tool.category == EvalCategory::StaticAnalysis
                    || tool.category == EvalCategory::Coverage
            );
        }
    }

    #[test]
    fn test_detect_rust_project_includes_clippy_and_fmt() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("Cargo.toml"), "[package]\nname = \"test\"").unwrap();
        let tools = detect_tools(dir.path(), &ToolOverrides::default());

        // Should have cargo test at minimum
        assert!(tools.iter().any(|t| t.name == "cargo test"));

        // Check that cargo test command includes json format flag
        let test_tool = tools.iter().find(|t| t.name == "cargo test").unwrap();
        assert!(test_tool.command.contains(&"--format".to_string()));
        assert!(test_tool.command.contains(&"json".to_string()));
    }

    #[test]
    fn test_detect_mixed_rust_and_js_project() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("Cargo.toml"), "[package]\nname = \"test\"").unwrap();
        fs::write(dir.path().join("package.json"), r#"{"name": "test"}"#).unwrap();
        let tools = detect_tools(dir.path(), &ToolOverrides::default());
        // Should detect tools for both Rust and JS
        let has_cargo = tools.iter().any(|t| t.name.contains("cargo"));
        assert!(has_cargo, "Should detect Rust tools");
    }

    #[test]
    fn test_detect_nonexistent_directory() {
        let tools = detect_tools(
            Path::new("/tmp/nonexistent_dir_for_claudear_test_12345"),
            &ToolOverrides::default(),
        );
        // Should not panic and return empty
        assert!(tools.is_empty());
    }

    #[test]
    fn test_which_exists_for_known_binary() {
        // Use a binary that exists on all platforms
        #[cfg(not(windows))]
        assert!(which_exists("ls"));
        #[cfg(windows)]
        assert!(which_exists("cmd"));
    }

    #[test]
    fn test_which_exists_for_unknown_binary() {
        assert!(!which_exists(
            "nonexistent_binary_that_does_not_exist_12345"
        ));
    }

    #[test]
    fn test_detect_php_project_with_phpunit() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("composer.json"), r#"{"name": "test/pkg"}"#).unwrap();
        fs::create_dir_all(dir.path().join("vendor/bin")).unwrap();
        fs::write(dir.path().join("vendor/bin/phpunit"), "#!/bin/sh").unwrap();
        let tools = detect_tools(dir.path(), &ToolOverrides::default());
        assert!(tools.iter().any(|t| t.name == "phpunit"));
        let phpunit = tools.iter().find(|t| t.name == "phpunit").unwrap();
        assert_eq!(phpunit.category, EvalCategory::Test);
        assert!(phpunit.command.contains(&"vendor/bin/phpunit".to_string()));
    }

    #[test]
    fn test_detect_php_project_with_phpstan() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("composer.json"), r#"{"name": "test/pkg"}"#).unwrap();
        fs::create_dir_all(dir.path().join("vendor/bin")).unwrap();
        fs::write(dir.path().join("vendor/bin/phpstan"), "#!/bin/sh").unwrap();
        let tools = detect_tools(dir.path(), &ToolOverrides::default());
        assert!(tools.iter().any(|t| t.name == "phpstan"));
        let phpstan = tools.iter().find(|t| t.name == "phpstan").unwrap();
        assert_eq!(phpstan.category, EvalCategory::StaticAnalysis);
    }

    #[test]
    fn test_detect_php_project_with_cs_fixer() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("composer.json"), r#"{"name": "test/pkg"}"#).unwrap();
        fs::create_dir_all(dir.path().join("vendor/bin")).unwrap();
        fs::write(dir.path().join("vendor/bin/php-cs-fixer"), "#!/bin/sh").unwrap();
        let tools = detect_tools(dir.path(), &ToolOverrides::default());
        assert!(tools.iter().any(|t| t.name == "php-cs-fixer"));
        let fixer = tools.iter().find(|t| t.name == "php-cs-fixer").unwrap();
        assert_eq!(fixer.category, EvalCategory::Lint);
    }

    #[test]
    fn test_detect_php_project_with_coverage() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("composer.json"), r#"{"name": "test/pkg"}"#).unwrap();
        fs::create_dir_all(dir.path().join("vendor/bin")).unwrap();
        fs::write(dir.path().join("vendor/bin/phpunit"), "#!/bin/sh").unwrap();
        let tools = detect_tools(dir.path(), &ToolOverrides::default());
        assert!(tools.iter().any(|t| t.name == "phpunit coverage"));
        let cov = tools.iter().find(|t| t.name == "phpunit coverage").unwrap();
        assert_eq!(cov.category, EvalCategory::Coverage);
        assert!(cov.command.contains(&"--coverage-clover".to_string()));
    }

    #[test]
    fn test_detect_php_project_no_vendor_bin() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("composer.json"), r#"{"name": "test/pkg"}"#).unwrap();
        // No vendor/bin directory
        let tools = detect_tools(dir.path(), &ToolOverrides::default());
        assert!(!tools.iter().any(|t| t.name == "phpunit"));
        assert!(!tools.iter().any(|t| t.name == "phpstan"));
    }

    #[test]
    fn test_detect_php_overrides_prevent_autodetect() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("composer.json"), r#"{"name": "test/pkg"}"#).unwrap();
        fs::create_dir_all(dir.path().join("vendor/bin")).unwrap();
        fs::write(dir.path().join("vendor/bin/phpunit"), "#!/bin/sh").unwrap();
        let overrides = ToolOverrides {
            custom_test_cmd: Some("custom-test".into()),
            custom_coverage_cmd: Some("custom-cov".into()),
            ..Default::default()
        };
        let tools = detect_tools(dir.path(), &overrides);
        // Custom overrides should prevent phpunit auto-detect for test and coverage
        let test_tools: Vec<_> = tools
            .iter()
            .filter(|t| t.category == EvalCategory::Test)
            .collect();
        assert_eq!(test_tools.len(), 1);
        assert_eq!(test_tools[0].name, "custom");
        let cov_tools: Vec<_> = tools
            .iter()
            .filter(|t| t.category == EvalCategory::Coverage)
            .collect();
        assert_eq!(cov_tools.len(), 1);
        assert_eq!(cov_tools[0].name, "custom");
    }

    #[test]
    fn test_detect_kotlin_project_gradle_kts() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("build.gradle.kts"), "plugins { }").unwrap();
        let tools = detect_tools(dir.path(), &ToolOverrides::default());
        assert!(tools.iter().any(|t| t.name == "gradle test"));
        let test_tool = tools.iter().find(|t| t.name == "gradle test").unwrap();
        assert_eq!(test_tool.category, EvalCategory::Test);
        // Without gradlew, should use "gradle"
        assert_eq!(test_tool.command[0], "gradle");
    }

    #[test]
    fn test_detect_kotlin_project_gradle_groovy() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("build.gradle"), "plugins { }").unwrap();
        let tools = detect_tools(dir.path(), &ToolOverrides::default());
        assert!(tools.iter().any(|t| t.name == "gradle test"));
    }

    #[test]
    fn test_detect_kotlin_project_with_gradlew() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("build.gradle.kts"), "plugins { }").unwrap();
        fs::write(dir.path().join("gradlew"), "#!/bin/sh").unwrap();
        let tools = detect_tools(dir.path(), &ToolOverrides::default());
        let test_tool = tools.iter().find(|t| t.name == "gradle test").unwrap();
        assert_eq!(test_tool.command[0], "./gradlew");
    }

    #[test]
    fn test_detect_kotlin_all_tools() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("build.gradle.kts"), "plugins { }").unwrap();
        let tools = detect_tools(dir.path(), &ToolOverrides::default());
        // gradle test always detected; detekt/ktlint/jacoco depend on which_exists(gradle)
        assert!(tools.iter().any(|t| t.name == "gradle test"));
    }

    #[test]
    fn test_detect_swift_project() {
        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("Package.swift"),
            "// swift-tools-version:5.5",
        )
        .unwrap();
        let tools = detect_tools(dir.path(), &ToolOverrides::default());
        // swift test depends on which_exists("swift")
        // At minimum, detection logic should not panic
        for tool in &tools {
            assert!(
                tool.category == EvalCategory::Test
                    || tool.category == EvalCategory::Lint
                    || tool.category == EvalCategory::StaticAnalysis
                    || tool.category == EvalCategory::Coverage
            );
        }
    }

    #[test]
    fn test_detect_csharp_project_csproj() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("MyProject.csproj"), "<Project></Project>").unwrap();
        let tools = detect_tools(dir.path(), &ToolOverrides::default());
        // dotnet tools depend on which_exists("dotnet")
        // At minimum, detection should not panic
        for tool in &tools {
            assert!(
                tool.category == EvalCategory::Test
                    || tool.category == EvalCategory::Lint
                    || tool.category == EvalCategory::StaticAnalysis
                    || tool.category == EvalCategory::Coverage
            );
        }
    }

    #[test]
    fn test_detect_csharp_project_sln() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("MySolution.sln"), "").unwrap();
        let tools = detect_tools(dir.path(), &ToolOverrides::default());
        for tool in &tools {
            assert!(
                tool.category == EvalCategory::Test
                    || tool.category == EvalCategory::Lint
                    || tool.category == EvalCategory::StaticAnalysis
                    || tool.category == EvalCategory::Coverage
            );
        }
    }

    #[test]
    fn test_detect_csharp_not_detected_without_csproj_or_sln() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("Program.cs"), "class Program {}").unwrap();
        let tools = detect_tools(dir.path(), &ToolOverrides::default());
        // Should not detect dotnet tools with just a .cs file
        assert!(!tools.iter().any(|t| t.name.contains("dotnet")));
    }

    #[test]
    fn test_detect_python_pyproject() {
        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("pyproject.toml"),
            "[project]\nname = \"test\"",
        )
        .unwrap();
        let tools = detect_tools(dir.path(), &ToolOverrides::default());
        // Detection depends on which_exists for pytest/mypy/ruff/black
        for tool in &tools {
            assert!(
                tool.category == EvalCategory::Test
                    || tool.category == EvalCategory::Lint
                    || tool.category == EvalCategory::StaticAnalysis
                    || tool.category == EvalCategory::Coverage
            );
        }
    }

    #[test]
    fn test_detect_python_requirements_txt() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("requirements.txt"), "flask==2.0").unwrap();
        let tools = detect_tools(dir.path(), &ToolOverrides::default());
        for tool in &tools {
            assert!(
                tool.category == EvalCategory::Test
                    || tool.category == EvalCategory::Lint
                    || tool.category == EvalCategory::StaticAnalysis
                    || tool.category == EvalCategory::Coverage
            );
        }
    }

    #[test]
    fn test_detect_go_project() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("go.mod"), "module example.com/test").unwrap();
        let tools = detect_tools(dir.path(), &ToolOverrides::default());
        for tool in &tools {
            assert!(
                tool.category == EvalCategory::Test
                    || tool.category == EvalCategory::Lint
                    || tool.category == EvalCategory::StaticAnalysis
                    || tool.category == EvalCategory::Coverage
            );
        }
    }

    #[test]
    fn test_detect_java_maven_project() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("pom.xml"), "<project></project>").unwrap();
        let tools = detect_tools(dir.path(), &ToolOverrides::default());
        for tool in &tools {
            assert!(
                tool.category == EvalCategory::Test
                    || tool.category == EvalCategory::Lint
                    || tool.category == EvalCategory::StaticAnalysis
                    || tool.category == EvalCategory::Coverage
            );
        }
    }

    #[test]
    fn test_detect_c_cmake_project() {
        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("CMakeLists.txt"),
            "cmake_minimum_required(VERSION 3.10)",
        )
        .unwrap();
        let tools = detect_tools(dir.path(), &ToolOverrides::default());
        for tool in &tools {
            assert!(
                tool.category == EvalCategory::Test
                    || tool.category == EvalCategory::Lint
                    || tool.category == EvalCategory::StaticAnalysis
                    || tool.category == EvalCategory::Coverage
            );
        }
    }

    #[test]
    fn test_detect_c_makefile_project() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("Makefile"), "all:\n\tgcc main.c").unwrap();
        let tools = detect_tools(dir.path(), &ToolOverrides::default());
        for tool in &tools {
            assert!(
                tool.category == EvalCategory::Test
                    || tool.category == EvalCategory::Lint
                    || tool.category == EvalCategory::StaticAnalysis
                    || tool.category == EvalCategory::Coverage
            );
        }
    }

    #[test]
    fn test_detect_ruby_project() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("Gemfile"), "source 'https://rubygems.org'").unwrap();
        let tools = detect_tools(dir.path(), &ToolOverrides::default());
        for tool in &tools {
            assert!(
                tool.category == EvalCategory::Test
                    || tool.category == EvalCategory::Lint
                    || tool.category == EvalCategory::StaticAnalysis
                    || tool.category == EvalCategory::Coverage
            );
        }
    }

    #[test]
    fn test_detect_dart_project() {
        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("pubspec.yaml"),
            "name: test\nversion: 1.0.0",
        )
        .unwrap();
        let tools = detect_tools(dir.path(), &ToolOverrides::default());
        for tool in &tools {
            assert!(
                tool.category == EvalCategory::Test
                    || tool.category == EvalCategory::Lint
                    || tool.category == EvalCategory::StaticAnalysis
                    || tool.category == EvalCategory::Coverage
            );
        }
    }

    #[test]
    fn test_custom_lint_override_prevents_autodetect_for_all_languages() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("Cargo.toml"), "[package]\nname = \"test\"").unwrap();
        fs::write(dir.path().join("package.json"), r#"{"name": "test"}"#).unwrap();
        let overrides = ToolOverrides {
            custom_lint_cmd: Some("my-linter".into()),
            ..Default::default()
        };
        let tools = detect_tools(dir.path(), &overrides);
        let lint_tools: Vec<_> = tools
            .iter()
            .filter(|t| t.category == EvalCategory::Lint)
            .collect();
        assert_eq!(lint_tools.len(), 1);
        assert_eq!(lint_tools[0].name, "custom");
    }

    #[test]
    fn test_custom_analysis_override_prevents_autodetect() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("Cargo.toml"), "[package]\nname = \"test\"").unwrap();
        let overrides = ToolOverrides {
            custom_analysis_cmd: Some("my-analyzer".into()),
            ..Default::default()
        };
        let tools = detect_tools(dir.path(), &overrides);
        let analysis_tools: Vec<_> = tools
            .iter()
            .filter(|t| t.category == EvalCategory::StaticAnalysis)
            .collect();
        assert_eq!(analysis_tools.len(), 1);
        assert_eq!(analysis_tools[0].name, "custom");
    }

    #[test]
    fn test_custom_coverage_override_prevents_autodetect() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("Cargo.toml"), "[package]\nname = \"test\"").unwrap();
        let overrides = ToolOverrides {
            custom_coverage_cmd: Some("my-coverage".into()),
            ..Default::default()
        };
        let tools = detect_tools(dir.path(), &overrides);
        let cov_tools: Vec<_> = tools
            .iter()
            .filter(|t| t.category == EvalCategory::Coverage)
            .collect();
        assert_eq!(cov_tools.len(), 1);
        assert_eq!(cov_tools[0].name, "custom");
    }

    #[test]
    fn test_detect_php_project_all_vendor_tools() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("composer.json"), r#"{"name": "test/pkg"}"#).unwrap();
        fs::create_dir_all(dir.path().join("vendor/bin")).unwrap();
        fs::write(dir.path().join("vendor/bin/phpunit"), "#!/bin/sh").unwrap();
        fs::write(dir.path().join("vendor/bin/phpstan"), "#!/bin/sh").unwrap();
        fs::write(dir.path().join("vendor/bin/php-cs-fixer"), "#!/bin/sh").unwrap();
        let tools = detect_tools(dir.path(), &ToolOverrides::default());
        assert!(tools.iter().any(|t| t.name == "phpunit"), "phpunit");
        assert!(tools.iter().any(|t| t.name == "phpstan"), "phpstan");
        assert!(
            tools.iter().any(|t| t.name == "php-cs-fixer"),
            "php-cs-fixer"
        );
        assert!(
            tools.iter().any(|t| t.name == "phpunit coverage"),
            "phpunit coverage"
        );
    }

    #[test]
    fn test_detect_rust_kotlin_mixed_detects_both() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("Cargo.toml"), "[package]\nname = \"test\"").unwrap();
        fs::write(dir.path().join("build.gradle.kts"), "plugins { }").unwrap();
        let tools = detect_tools(dir.path(), &ToolOverrides::default());
        // Both cargo test and gradle test are detected (has_test only blocks custom overrides)
        assert!(tools.iter().any(|t| t.name == "cargo test"));
        assert!(tools.iter().any(|t| t.name == "gradle test"));
    }

    #[test]
    fn test_shell_words_multiple_spaces_between_words() {
        assert_eq!(shell_words("a      b      c"), vec!["a", "b", "c"]);
    }

    #[test]
    fn test_shell_words_newlines_treated_as_whitespace() {
        assert_eq!(
            shell_words("cargo\ntest\n--json"),
            vec!["cargo", "test", "--json"]
        );
    }

    #[test]
    fn test_custom_test_and_coverage_overrides_together() {
        let dir = TempDir::new().unwrap();
        let overrides = ToolOverrides {
            custom_test_cmd: Some("make test".into()),
            custom_coverage_cmd: Some("make cov".into()),
            ..Default::default()
        };
        let tools = detect_tools(dir.path(), &overrides);
        assert_eq!(tools.len(), 2);
        let test_tool = tools
            .iter()
            .find(|t| t.category == EvalCategory::Test)
            .unwrap();
        assert_eq!(test_tool.command, vec!["make", "test"]);
        let cov_tool = tools
            .iter()
            .find(|t| t.category == EvalCategory::Coverage)
            .unwrap();
        assert_eq!(cov_tool.command, vec!["make", "cov"]);
    }

    #[test]
    fn test_detect_python_setup_py() {
        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("setup.py"),
            "from setuptools import setup\nsetup()",
        )
        .unwrap();
        let tools = detect_tools(dir.path(), &ToolOverrides::default());
        // Detection depends on which_exists for pytest/mypy/ruff, but should not panic
        for tool in &tools {
            assert!(
                tool.category == EvalCategory::Test
                    || tool.category == EvalCategory::Lint
                    || tool.category == EvalCategory::StaticAnalysis
                    || tool.category == EvalCategory::Coverage
            );
        }
    }

    #[test]
    fn test_detect_multiple_project_files_all_detected() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("Cargo.toml"), "[package]\nname = \"test\"").unwrap();
        fs::write(dir.path().join("package.json"), r#"{"name": "test"}"#).unwrap();
        fs::write(
            dir.path().join("pyproject.toml"),
            "[project]\nname = \"test\"",
        )
        .unwrap();
        fs::write(dir.path().join("go.mod"), "module example.com/test").unwrap();
        let tools = detect_tools(dir.path(), &ToolOverrides::default());
        // At minimum Rust tools should be detected
        assert!(tools.iter().any(|t| t.name == "cargo test"));
    }

    #[test]
    fn test_custom_overrides_block_all_autodetect_across_all_languages() {
        let dir = TempDir::new().unwrap();
        // Create files for multiple languages
        fs::write(dir.path().join("Cargo.toml"), "[package]\nname = \"test\"").unwrap();
        fs::write(dir.path().join("package.json"), r#"{"name": "test"}"#).unwrap();
        fs::write(dir.path().join("composer.json"), r#"{"name": "test/pkg"}"#).unwrap();
        fs::create_dir_all(dir.path().join("vendor/bin")).unwrap();
        fs::write(dir.path().join("vendor/bin/phpunit"), "#!/bin/sh").unwrap();
        fs::write(dir.path().join("vendor/bin/phpstan"), "#!/bin/sh").unwrap();
        fs::write(dir.path().join("vendor/bin/php-cs-fixer"), "#!/bin/sh").unwrap();

        let overrides = ToolOverrides {
            custom_test_cmd: Some("make test".into()),
            custom_lint_cmd: Some("make lint".into()),
            custom_analysis_cmd: Some("make analyze".into()),
            custom_coverage_cmd: Some("make coverage".into()),
        };
        let tools = detect_tools(dir.path(), &overrides);
        // Only 4 custom tools, no auto-detected ones
        assert_eq!(tools.len(), 4);
        assert!(tools.iter().all(|t| t.name == "custom"));
    }

    #[test]
    fn test_detect_dart_pubspec_no_dart_binary() {
        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("pubspec.yaml"),
            "name: test\nversion: 1.0.0",
        )
        .unwrap();
        let tools = detect_tools(dir.path(), &ToolOverrides::default());
        // Tools depend on which_exists("dart"), should not panic regardless
        for tool in &tools {
            assert!(
                tool.category == EvalCategory::Test
                    || tool.category == EvalCategory::Lint
                    || tool.category == EvalCategory::StaticAnalysis
                    || tool.category == EvalCategory::Coverage
            );
        }
    }

    #[test]
    fn test_detect_php_only_phpstan_no_phpunit() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("composer.json"), r#"{"name": "test/pkg"}"#).unwrap();
        fs::create_dir_all(dir.path().join("vendor/bin")).unwrap();
        fs::write(dir.path().join("vendor/bin/phpstan"), "#!/bin/sh").unwrap();
        let tools = detect_tools(dir.path(), &ToolOverrides::default());
        assert!(tools.iter().any(|t| t.name == "phpstan"));
        assert!(!tools.iter().any(|t| t.name == "phpunit"));
        assert!(!tools.iter().any(|t| t.name == "phpunit coverage"));
    }

    #[test]
    fn test_detected_tool_command_not_empty() {
        let tool = DetectedTool {
            category: EvalCategory::Test,
            name: "cargo test".into(),
            command: vec!["cargo".into(), "test".into()],
        };
        assert!(!tool.command.is_empty());
        assert_eq!(tool.command[0], "cargo");
    }

    #[test]
    fn test_detect_cmake_vs_makefile_priority() {
        // With both CMakeLists.txt and Makefile, has_cmake is true
        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("CMakeLists.txt"),
            "cmake_minimum_required(VERSION 3.10)",
        )
        .unwrap();
        fs::write(dir.path().join("Makefile"), "all:\n\tgcc main.c").unwrap();
        let tools = detect_tools(dir.path(), &ToolOverrides::default());
        // CMake+ctest takes priority if ctest is available
        // At minimum detection should not panic
        for tool in &tools {
            assert!(
                tool.category == EvalCategory::Test
                    || tool.category == EvalCategory::Lint
                    || tool.category == EvalCategory::StaticAnalysis
                    || tool.category == EvalCategory::Coverage
            );
        }
    }

    #[test]
    fn test_which_exists_empty_string() {
        // Empty string should not exist
        assert!(!which_exists(""));
    }

    #[test]
    fn test_detect_kotlin_custom_test_blocks_gradle_test() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("build.gradle.kts"), "plugins { }").unwrap();
        let overrides = ToolOverrides {
            custom_test_cmd: Some("my-test".into()),
            ..Default::default()
        };
        let tools = detect_tools(dir.path(), &overrides);
        let test_tools: Vec<_> = tools
            .iter()
            .filter(|t| t.category == EvalCategory::Test)
            .collect();
        assert_eq!(test_tools.len(), 1);
        assert_eq!(test_tools[0].name, "custom");
    }

    #[test]
    fn test_detect_rust_all_tools_when_binaries_present() {
        // cargo is always available in this environment
        if !which_exists("cargo") {
            return;
        }
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("Cargo.toml"), "[package]\nname = \"test\"").unwrap();
        let tools = detect_tools(dir.path(), &ToolOverrides::default());

        // cargo clippy (static analysis)
        let clippy = tools.iter().find(|t| t.name == "cargo clippy");
        assert!(clippy.is_some(), "Should detect cargo clippy");
        let clippy = clippy.unwrap();
        assert_eq!(clippy.category, EvalCategory::StaticAnalysis);
        assert!(clippy.command.contains(&"clippy".to_string()));
        assert!(clippy
            .command
            .contains(&"--message-format=json".to_string()));
        assert!(clippy.command.contains(&"-D".to_string()));
        assert!(clippy.command.contains(&"warnings".to_string()));

        // cargo fmt (lint)
        let fmt = tools.iter().find(|t| t.name == "cargo fmt");
        assert!(fmt.is_some(), "Should detect cargo fmt");
        let fmt = fmt.unwrap();
        assert_eq!(fmt.category, EvalCategory::Lint);
        assert!(fmt.command.contains(&"fmt".to_string()));
        assert!(fmt.command.contains(&"--check".to_string()));
    }

    #[test]
    fn test_detect_rust_coverage_with_llvm_cov() {
        if !which_exists("cargo") || !which_exists("cargo-llvm-cov") {
            return;
        }
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("Cargo.toml"), "[package]\nname = \"test\"").unwrap();
        let tools = detect_tools(dir.path(), &ToolOverrides::default());

        let cov = tools.iter().find(|t| t.name == "cargo llvm-cov");
        assert!(cov.is_some(), "Should detect cargo llvm-cov");
        let cov = cov.unwrap();
        assert_eq!(cov.category, EvalCategory::Coverage);
        assert_eq!(cov.command, vec!["cargo", "llvm-cov", "--json"]);
    }

    #[test]
    fn test_detect_swift_test_and_coverage() {
        if !which_exists("swift") {
            return;
        }
        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("Package.swift"),
            "// swift-tools-version:5.5",
        )
        .unwrap();
        let tools = detect_tools(dir.path(), &ToolOverrides::default());

        // swift test
        let swift_test = tools.iter().find(|t| t.name == "swift test");
        assert!(swift_test.is_some(), "Should detect swift test");
        let swift_test = swift_test.unwrap();
        assert_eq!(swift_test.category, EvalCategory::Test);
        assert_eq!(swift_test.command, vec!["swift", "test"]);

        // swift coverage
        let swift_cov = tools.iter().find(|t| t.name == "swift coverage");
        assert!(swift_cov.is_some(), "Should detect swift coverage");
        let swift_cov = swift_cov.unwrap();
        assert_eq!(swift_cov.category, EvalCategory::Coverage);
        assert_eq!(
            swift_cov.command,
            vec!["swift", "test", "--enable-code-coverage"]
        );
    }

    #[test]
    fn test_detect_swift_custom_overrides_block_autodetect() {
        if !which_exists("swift") {
            return;
        }
        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("Package.swift"),
            "// swift-tools-version:5.5",
        )
        .unwrap();
        let overrides = ToolOverrides {
            custom_test_cmd: Some("my-swift-test".into()),
            custom_coverage_cmd: Some("my-swift-cov".into()),
            ..Default::default()
        };
        let tools = detect_tools(dir.path(), &overrides);
        let test_tools: Vec<_> = tools
            .iter()
            .filter(|t| t.category == EvalCategory::Test)
            .collect();
        assert_eq!(test_tools.len(), 1);
        assert_eq!(test_tools[0].name, "custom");
        let cov_tools: Vec<_> = tools
            .iter()
            .filter(|t| t.category == EvalCategory::Coverage)
            .collect();
        assert_eq!(cov_tools.len(), 1);
        assert_eq!(cov_tools[0].name, "custom");
    }

    #[test]
    fn test_detect_c_cmake_with_ctest_and_tools() {
        if !which_exists("ctest") {
            return;
        }
        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("CMakeLists.txt"),
            "cmake_minimum_required(VERSION 3.10)",
        )
        .unwrap();
        let tools = detect_tools(dir.path(), &ToolOverrides::default());

        // ctest (test)
        let ctest = tools.iter().find(|t| t.name == "ctest");
        assert!(ctest.is_some(), "Should detect ctest");
        let ctest = ctest.unwrap();
        assert_eq!(ctest.category, EvalCategory::Test);
        assert_eq!(ctest.command, vec!["ctest"]);
    }

    #[test]
    fn test_detect_c_cmake_clang_format_lint() {
        if !which_exists("clang-format") {
            return;
        }
        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("CMakeLists.txt"),
            "cmake_minimum_required(VERSION 3.10)",
        )
        .unwrap();
        let tools = detect_tools(dir.path(), &ToolOverrides::default());

        let cf = tools.iter().find(|t| t.name == "clang-format");
        assert!(cf.is_some(), "Should detect clang-format");
        let cf = cf.unwrap();
        assert_eq!(cf.category, EvalCategory::Lint);
        assert_eq!(cf.command, vec!["clang-format", "--dry-run", "-Werror"]);
    }

    #[test]
    fn test_detect_c_cmake_gcov_coverage() {
        if !which_exists("gcov") {
            return;
        }
        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("CMakeLists.txt"),
            "cmake_minimum_required(VERSION 3.10)",
        )
        .unwrap();
        let tools = detect_tools(dir.path(), &ToolOverrides::default());

        let gcov = tools.iter().find(|t| t.name == "gcov");
        assert!(gcov.is_some(), "Should detect gcov");
        let gcov = gcov.unwrap();
        assert_eq!(gcov.category, EvalCategory::Coverage);
        assert_eq!(gcov.command, vec!["gcov"]);
    }

    #[test]
    fn test_detect_makefile_only_uses_make_test() {
        if !which_exists("make") {
            return;
        }
        let dir = TempDir::new().unwrap();
        // Only Makefile, no CMakeLists.txt => has_cmake=false => fallback to make test
        fs::write(dir.path().join("Makefile"), "all:\n\tgcc main.c").unwrap();
        let tools = detect_tools(dir.path(), &ToolOverrides::default());

        // Should use make test (not ctest) since there's no CMakeLists.txt
        let make_test = tools.iter().find(|t| t.name == "make test");
        assert!(
            make_test.is_some(),
            "Should detect make test for Makefile-only project"
        );
        let make_test = make_test.unwrap();
        assert_eq!(make_test.category, EvalCategory::Test);
        assert_eq!(make_test.command, vec!["make", "test"]);
    }

    #[test]
    fn test_detect_js_all_tools_with_npx() {
        if !which_exists("npx") {
            return;
        }
        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("package.json"),
            r#"{"name": "test", "version": "1.0.0"}"#,
        )
        .unwrap();
        let tools = detect_tools(dir.path(), &ToolOverrides::default());

        // jest (test)
        let jest = tools.iter().find(|t| t.name == "jest");
        assert!(jest.is_some(), "Should detect jest");
        let jest = jest.unwrap();
        assert_eq!(jest.category, EvalCategory::Test);
        assert_eq!(jest.command, vec!["npx", "jest", "--json"]);

        // eslint (static analysis)
        let eslint = tools.iter().find(|t| t.name == "eslint");
        assert!(eslint.is_some(), "Should detect eslint");
        let eslint = eslint.unwrap();
        assert_eq!(eslint.category, EvalCategory::StaticAnalysis);
        assert_eq!(
            eslint.command,
            vec!["npx", "eslint", "--format", "json", "."]
        );

        // prettier (lint)
        let prettier = tools.iter().find(|t| t.name == "prettier");
        assert!(prettier.is_some(), "Should detect prettier");
        let prettier = prettier.unwrap();
        assert_eq!(prettier.category, EvalCategory::Lint);
        assert_eq!(prettier.command, vec!["npx", "prettier", "--check", "."]);

        // jest coverage
        let jest_cov = tools.iter().find(|t| t.name == "jest coverage");
        assert!(jest_cov.is_some(), "Should detect jest coverage");
        let jest_cov = jest_cov.unwrap();
        assert_eq!(jest_cov.category, EvalCategory::Coverage);
        assert_eq!(
            jest_cov.command,
            vec![
                "npx",
                "jest",
                "--coverage",
                "--coverageReporters=json-summary"
            ]
        );
    }

    #[test]
    fn test_detect_js_custom_overrides_block_all_autodetect() {
        if !which_exists("npx") {
            return;
        }
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("package.json"), r#"{"name": "test"}"#).unwrap();
        let overrides = ToolOverrides {
            custom_test_cmd: Some("vitest".into()),
            custom_lint_cmd: Some("biome lint".into()),
            custom_analysis_cmd: Some("tsc --noEmit".into()),
            custom_coverage_cmd: Some("vitest coverage".into()),
        };
        let tools = detect_tools(dir.path(), &overrides);
        assert_eq!(tools.len(), 4);
        assert!(tools.iter().all(|t| t.name == "custom"));
    }

    #[test]
    fn test_detect_dart_all_tools() {
        if !which_exists("dart") {
            return;
        }
        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("pubspec.yaml"),
            "name: test\nversion: 1.0.0",
        )
        .unwrap();
        let tools = detect_tools(dir.path(), &ToolOverrides::default());

        // dart test
        let dt = tools.iter().find(|t| t.name == "dart test");
        assert!(dt.is_some(), "Should detect dart test");
        let dt = dt.unwrap();
        assert_eq!(dt.category, EvalCategory::Test);
        assert_eq!(dt.command, vec!["dart", "test"]);

        // dart analyze
        let da = tools.iter().find(|t| t.name == "dart analyze");
        assert!(da.is_some(), "Should detect dart analyze");
        let da = da.unwrap();
        assert_eq!(da.category, EvalCategory::StaticAnalysis);
        assert_eq!(da.command, vec!["dart", "analyze"]);

        // dart format
        let df = tools.iter().find(|t| t.name == "dart format");
        assert!(df.is_some(), "Should detect dart format");
        let df = df.unwrap();
        assert_eq!(df.category, EvalCategory::Lint);
        assert_eq!(
            df.command,
            vec![
                "dart",
                "format",
                "--output=none",
                "--set-exit-if-changed",
                "."
            ]
        );

        // dart test coverage
        let dc = tools.iter().find(|t| t.name == "dart test coverage");
        assert!(dc.is_some(), "Should detect dart test coverage");
        let dc = dc.unwrap();
        assert_eq!(dc.category, EvalCategory::Coverage);
        assert_eq!(dc.command, vec!["dart", "test", "--coverage=coverage"]);
    }

    #[test]
    fn test_detect_dart_custom_overrides_block_all() {
        if !which_exists("dart") {
            return;
        }
        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("pubspec.yaml"),
            "name: test\nversion: 1.0.0",
        )
        .unwrap();
        let overrides = ToolOverrides {
            custom_test_cmd: Some("my-dart-test".into()),
            custom_lint_cmd: Some("my-dart-lint".into()),
            custom_analysis_cmd: Some("my-dart-analysis".into()),
            custom_coverage_cmd: Some("my-dart-cov".into()),
        };
        let tools = detect_tools(dir.path(), &overrides);
        assert_eq!(tools.len(), 4);
        assert!(tools.iter().all(|t| t.name == "custom"));
        // None of the dart auto-detected tools should be present
        assert!(!tools.iter().any(|t| t.name.contains("dart")));
    }

    #[test]
    fn test_detect_csharp_all_dotnet_tools() {
        if !which_exists("dotnet") {
            return;
        }
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("MyProject.csproj"), "<Project></Project>").unwrap();
        let tools = detect_tools(dir.path(), &ToolOverrides::default());

        // dotnet test
        let dt = tools.iter().find(|t| t.name == "dotnet test");
        assert!(dt.is_some(), "Should detect dotnet test");
        let dt = dt.unwrap();
        assert_eq!(dt.category, EvalCategory::Test);
        assert!(dt.command.contains(&"--logger".to_string()));
        assert!(dt.command.contains(&"trx".to_string()));

        // dotnet build (static analysis)
        let db = tools.iter().find(|t| t.name == "dotnet build");
        assert!(db.is_some(), "Should detect dotnet build");
        let db = db.unwrap();
        assert_eq!(db.category, EvalCategory::StaticAnalysis);
        assert!(db.command.contains(&"/warnaserror".to_string()));

        // dotnet format (lint)
        let df = tools.iter().find(|t| t.name == "dotnet format");
        assert!(df.is_some(), "Should detect dotnet format");
        let df = df.unwrap();
        assert_eq!(df.category, EvalCategory::Lint);
        assert!(df.command.contains(&"--verify-no-changes".to_string()));

        // dotnet coverage
        let dc = tools.iter().find(|t| t.name == "dotnet coverage");
        assert!(dc.is_some(), "Should detect dotnet coverage");
        let dc = dc.unwrap();
        assert_eq!(dc.category, EvalCategory::Coverage);
        assert!(dc
            .command
            .contains(&"--collect:XPlat Code Coverage".to_string()));
    }

    #[test]
    fn test_detect_csharp_sln_all_dotnet_tools() {
        if !which_exists("dotnet") {
            return;
        }
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("MySolution.sln"), "").unwrap();
        let tools = detect_tools(dir.path(), &ToolOverrides::default());
        assert!(tools.iter().any(|t| t.name == "dotnet test"));
        assert!(tools.iter().any(|t| t.name == "dotnet build"));
        assert!(tools.iter().any(|t| t.name == "dotnet format"));
        assert!(tools.iter().any(|t| t.name == "dotnet coverage"));
    }

    #[test]
    fn test_detect_csharp_custom_overrides_block_dotnet() {
        if !which_exists("dotnet") {
            return;
        }
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("MyProject.csproj"), "<Project></Project>").unwrap();
        let overrides = ToolOverrides {
            custom_test_cmd: Some("my-test".into()),
            custom_lint_cmd: Some("my-lint".into()),
            custom_analysis_cmd: Some("my-analysis".into()),
            custom_coverage_cmd: Some("my-cov".into()),
        };
        let tools = detect_tools(dir.path(), &overrides);
        assert_eq!(tools.len(), 4);
        assert!(tools.iter().all(|t| t.name == "custom"));
    }

    #[test]
    fn test_detect_ruby_with_rake_no_rspec() {
        if !which_exists("rake") {
            return;
        }
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("Gemfile"), "source 'https://rubygems.org'").unwrap();
        let tools = detect_tools(dir.path(), &ToolOverrides::default());

        // rspec not available, so should fall back to rake test
        if !which_exists("rspec") {
            let rake_test = tools.iter().find(|t| t.name == "rake test");
            assert!(
                rake_test.is_some(),
                "Should detect rake test when rspec unavailable"
            );
            let rake_test = rake_test.unwrap();
            assert_eq!(rake_test.category, EvalCategory::Test);
            assert_eq!(rake_test.command, vec!["rake", "test"]);
        }
    }

    #[test]
    fn test_detect_c_cmake_and_makefile_prefers_ctest() {
        // With both CMakeLists.txt and Makefile + ctest available, ctest takes priority
        if !which_exists("ctest") {
            return;
        }
        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("CMakeLists.txt"),
            "cmake_minimum_required(VERSION 3.10)",
        )
        .unwrap();
        fs::write(dir.path().join("Makefile"), "all:\n\tgcc main.c").unwrap();
        let tools = detect_tools(dir.path(), &ToolOverrides::default());

        let test_tools: Vec<_> = tools
            .iter()
            .filter(|t| t.category == EvalCategory::Test)
            .collect();
        assert_eq!(test_tools.len(), 1, "Should have exactly one test tool");
        assert_eq!(test_tools[0].name, "ctest");
    }

    #[test]
    fn test_detect_c_cmake_custom_test_blocks_ctest() {
        if !which_exists("ctest") {
            return;
        }
        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("CMakeLists.txt"),
            "cmake_minimum_required(VERSION 3.10)",
        )
        .unwrap();
        let overrides = ToolOverrides {
            custom_test_cmd: Some("my-ctest".into()),
            ..Default::default()
        };
        let tools = detect_tools(dir.path(), &overrides);
        let test_tools: Vec<_> = tools
            .iter()
            .filter(|t| t.category == EvalCategory::Test)
            .collect();
        assert_eq!(test_tools.len(), 1);
        assert_eq!(test_tools[0].name, "custom");
        assert!(!tools.iter().any(|t| t.name == "ctest"));
    }

    #[test]
    fn test_detect_c_cmake_custom_lint_blocks_clang_format() {
        if !which_exists("clang-format") {
            return;
        }
        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("CMakeLists.txt"),
            "cmake_minimum_required(VERSION 3.10)",
        )
        .unwrap();
        let overrides = ToolOverrides {
            custom_lint_cmd: Some("my-lint".into()),
            ..Default::default()
        };
        let tools = detect_tools(dir.path(), &overrides);
        let lint_tools: Vec<_> = tools
            .iter()
            .filter(|t| t.category == EvalCategory::Lint)
            .collect();
        assert_eq!(lint_tools.len(), 1);
        assert_eq!(lint_tools[0].name, "custom");
    }

    #[test]
    fn test_detect_c_cmake_custom_coverage_blocks_gcov() {
        if !which_exists("gcov") {
            return;
        }
        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("CMakeLists.txt"),
            "cmake_minimum_required(VERSION 3.10)",
        )
        .unwrap();
        let overrides = ToolOverrides {
            custom_coverage_cmd: Some("my-cov".into()),
            ..Default::default()
        };
        let tools = detect_tools(dir.path(), &overrides);
        let cov_tools: Vec<_> = tools
            .iter()
            .filter(|t| t.category == EvalCategory::Coverage)
            .collect();
        assert_eq!(cov_tools.len(), 1);
        assert_eq!(cov_tools[0].name, "custom");
    }

    #[test]
    fn test_detect_all_languages_with_all_overrides() {
        let dir = TempDir::new().unwrap();
        // Create files for every supported language
        fs::write(dir.path().join("Cargo.toml"), "[package]\nname = \"test\"").unwrap();
        fs::write(dir.path().join("package.json"), r#"{"name": "test"}"#).unwrap();
        fs::write(dir.path().join("composer.json"), r#"{"name": "test/pkg"}"#).unwrap();
        fs::create_dir_all(dir.path().join("vendor/bin")).unwrap();
        fs::write(dir.path().join("vendor/bin/phpunit"), "#!/bin/sh").unwrap();
        fs::write(dir.path().join("build.gradle.kts"), "plugins { }").unwrap();
        fs::write(
            dir.path().join("Package.swift"),
            "// swift-tools-version:5.5",
        )
        .unwrap();
        fs::write(
            dir.path().join("pyproject.toml"),
            "[project]\nname = \"test\"",
        )
        .unwrap();
        fs::write(dir.path().join("go.mod"), "module example.com/test").unwrap();
        fs::write(dir.path().join("pom.xml"), "<project></project>").unwrap();
        fs::write(
            dir.path().join("CMakeLists.txt"),
            "cmake_minimum_required(VERSION 3.10)",
        )
        .unwrap();
        fs::write(dir.path().join("Gemfile"), "source 'https://rubygems.org'").unwrap();
        fs::write(
            dir.path().join("pubspec.yaml"),
            "name: test\nversion: 1.0.0",
        )
        .unwrap();
        fs::write(dir.path().join("MyProject.csproj"), "<Project></Project>").unwrap();

        let overrides = ToolOverrides {
            custom_test_cmd: Some("universal-test".into()),
            custom_lint_cmd: Some("universal-lint".into()),
            custom_analysis_cmd: Some("universal-analyze".into()),
            custom_coverage_cmd: Some("universal-cov".into()),
        };
        let tools = detect_tools(dir.path(), &overrides);
        // Only 4 custom tools despite having every language file
        assert_eq!(tools.len(), 4);
        assert!(tools.iter().all(|t| t.name == "custom"));
    }

    #[test]
    fn test_detect_kotlin_gradle_kts_detekt_ktlint_jacoco() {
        // When gradle binary is available, should detect detekt/ktlint/jacoco
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("build.gradle.kts"), "plugins { }").unwrap();
        let tools = detect_tools(dir.path(), &ToolOverrides::default());
        // gradle test is always detected (no which_exists check for gradle test)
        assert!(tools.iter().any(|t| t.name == "gradle test"));
        // detekt/ktlint/jacoco depend on which_exists("gradle")
        if which_exists("gradle") {
            assert!(tools.iter().any(|t| t.name == "detekt"));
            assert!(tools.iter().any(|t| t.name == "ktlint"));
            assert!(tools.iter().any(|t| t.name == "jacoco"));
        }
    }

    #[test]
    fn test_detect_kotlin_gradlew_detekt_command() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("build.gradle.kts"), "plugins { }").unwrap();
        fs::write(dir.path().join("gradlew"), "#!/bin/sh").unwrap();
        let tools = detect_tools(dir.path(), &ToolOverrides::default());
        // With gradlew, the command prefix should be ./gradlew
        let test_tool = tools.iter().find(|t| t.name == "gradle test").unwrap();
        assert_eq!(test_tool.command[0], "./gradlew");
        // If which_exists("./gradlew") is true, check analysis/lint/coverage tools
        if which_exists("./gradlew") {
            if let Some(detekt) = tools.iter().find(|t| t.name == "detekt") {
                assert_eq!(detekt.command[0], "./gradlew");
                assert_eq!(detekt.command[1], "detekt");
            }
        }
    }

    #[test]
    fn test_detect_php_custom_analysis_blocks_phpstan() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("composer.json"), r#"{"name": "test/pkg"}"#).unwrap();
        fs::create_dir_all(dir.path().join("vendor/bin")).unwrap();
        fs::write(dir.path().join("vendor/bin/phpstan"), "#!/bin/sh").unwrap();
        let overrides = ToolOverrides {
            custom_analysis_cmd: Some("my-analysis".into()),
            ..Default::default()
        };
        let tools = detect_tools(dir.path(), &overrides);
        assert!(!tools.iter().any(|t| t.name == "phpstan"));
        let analysis_tools: Vec<_> = tools
            .iter()
            .filter(|t| t.category == EvalCategory::StaticAnalysis)
            .collect();
        assert_eq!(analysis_tools.len(), 1);
        assert_eq!(analysis_tools[0].name, "custom");
    }

    #[test]
    fn test_detect_php_custom_lint_blocks_cs_fixer() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("composer.json"), r#"{"name": "test/pkg"}"#).unwrap();
        fs::create_dir_all(dir.path().join("vendor/bin")).unwrap();
        fs::write(dir.path().join("vendor/bin/php-cs-fixer"), "#!/bin/sh").unwrap();
        let overrides = ToolOverrides {
            custom_lint_cmd: Some("my-lint".into()),
            ..Default::default()
        };
        let tools = detect_tools(dir.path(), &overrides);
        assert!(!tools.iter().any(|t| t.name == "php-cs-fixer"));
        let lint_tools: Vec<_> = tools
            .iter()
            .filter(|t| t.category == EvalCategory::Lint)
            .collect();
        assert_eq!(lint_tools.len(), 1);
        assert_eq!(lint_tools[0].name, "custom");
    }

    #[test]
    fn test_detect_mixed_project_partial_overrides() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("Cargo.toml"), "[package]\nname = \"test\"").unwrap();
        fs::write(dir.path().join("package.json"), r#"{"name": "test"}"#).unwrap();
        // Only override test and lint, let analysis and coverage auto-detect
        let overrides = ToolOverrides {
            custom_test_cmd: Some("my-test".into()),
            custom_lint_cmd: Some("my-lint".into()),
            ..Default::default()
        };
        let tools = detect_tools(dir.path(), &overrides);
        // Test and lint should be custom
        let test_tools: Vec<_> = tools
            .iter()
            .filter(|t| t.category == EvalCategory::Test)
            .collect();
        assert_eq!(test_tools.len(), 1);
        assert_eq!(test_tools[0].name, "custom");
        let lint_tools: Vec<_> = tools
            .iter()
            .filter(|t| t.category == EvalCategory::Lint)
            .collect();
        assert_eq!(lint_tools.len(), 1);
        assert_eq!(lint_tools[0].name, "custom");
        // Analysis and coverage should be auto-detected (at least for Rust)
        if which_exists("cargo") {
            assert!(tools.iter().any(|t| t.name == "cargo clippy"));
        }
    }

    #[test]
    fn test_custom_override_command_parsing_complex() {
        let dir = TempDir::new().unwrap();
        let overrides = ToolOverrides {
            custom_test_cmd: Some("docker compose exec app php artisan test --parallel".into()),
            ..Default::default()
        };
        let tools = detect_tools(dir.path(), &overrides);
        let test_tool = tools
            .iter()
            .find(|t| t.category == EvalCategory::Test)
            .unwrap();
        assert_eq!(
            test_tool.command,
            vec![
                "docker",
                "compose",
                "exec",
                "app",
                "php",
                "artisan",
                "test",
                "--parallel"
            ]
        );
    }

    #[test]
    fn test_detect_go_all_tools_when_available() {
        if !which_exists("go") {
            return;
        }
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("go.mod"), "module example.com/test").unwrap();
        let tools = detect_tools(dir.path(), &ToolOverrides::default());

        let go_test = tools.iter().find(|t| t.name == "go test");
        assert!(go_test.is_some(), "Should detect go test");
        let go_test = go_test.unwrap();
        assert_eq!(go_test.category, EvalCategory::Test);
        assert_eq!(go_test.command, vec!["go", "test", "-json", "./..."]);

        let go_vet = tools.iter().find(|t| t.name == "go vet");
        assert!(go_vet.is_some(), "Should detect go vet");
        let go_vet = go_vet.unwrap();
        assert_eq!(go_vet.category, EvalCategory::StaticAnalysis);
        assert_eq!(go_vet.command, vec!["go", "vet", "./..."]);

        let go_cov = tools.iter().find(|t| t.name == "go test coverage");
        assert!(go_cov.is_some(), "Should detect go test coverage");
        let go_cov = go_cov.unwrap();
        assert_eq!(go_cov.category, EvalCategory::Coverage);
        assert_eq!(go_cov.command, vec!["go", "test", "-cover", "./..."]);
    }

    #[test]
    fn test_detect_go_gofmt_lint() {
        if !which_exists("gofmt") {
            return;
        }
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("go.mod"), "module example.com/test").unwrap();
        let tools = detect_tools(dir.path(), &ToolOverrides::default());

        let gofmt = tools.iter().find(|t| t.name == "gofmt");
        assert!(gofmt.is_some(), "Should detect gofmt");
        let gofmt = gofmt.unwrap();
        assert_eq!(gofmt.category, EvalCategory::Lint);
        assert_eq!(gofmt.command, vec!["gofmt", "-l", "."]);
    }

    #[test]
    fn test_detect_java_maven_all_tools() {
        if !which_exists("mvn") {
            return;
        }
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("pom.xml"), "<project></project>").unwrap();
        let tools = detect_tools(dir.path(), &ToolOverrides::default());

        let mvn_test = tools.iter().find(|t| t.name == "mvn test");
        assert!(mvn_test.is_some(), "Should detect mvn test");
        assert_eq!(mvn_test.unwrap().command, vec!["mvn", "test", "-B"]);

        let mvn_verify = tools.iter().find(|t| t.name == "mvn verify");
        assert!(mvn_verify.is_some(), "Should detect mvn verify");
        assert_eq!(
            mvn_verify.unwrap().command,
            vec!["mvn", "verify", "-B", "-DskipTests"]
        );

        let mvn_jacoco = tools.iter().find(|t| t.name == "mvn jacoco");
        assert!(mvn_jacoco.is_some(), "Should detect mvn jacoco");
        assert_eq!(
            mvn_jacoco.unwrap().command,
            vec!["mvn", "test", "jacoco:report", "-B"]
        );
    }
}
