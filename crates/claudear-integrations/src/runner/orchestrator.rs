//! Agent orchestrator for provider selection, experiments, and fallback chains.
//!
//! Implements `AgentRunner` itself (composite pattern), so consumers hold
//! `Arc<dyn AgentRunner>` and are completely unaware of orchestration logic.

use super::{AgentRunner, ProviderCapabilities};
use async_trait::async_trait;
use claudear_core::error::{Error, Result};
use claudear_core::types::{AgentResult, Issue};
use rand::RngExt;
use std::path::Path;
use std::sync::Arc;

/// A provider with an associated weight for experiment selection.
pub struct WeightedProvider {
    pub provider: Arc<dyn AgentRunner>,
    pub weight: f64,
}

/// Strategy for selecting which provider to use.
#[derive(Debug, Clone)]
pub enum SelectionStrategy {
    /// Always use the first provider.
    Primary,
    /// A/B split by weight.
    WeightedRandom,
    /// Try providers in order until one succeeds.
    Fallback,
}

/// Orchestrator that manages multiple providers with experiment support.
pub struct AgentOrchestrator {
    providers: Vec<WeightedProvider>,
    strategy: SelectionStrategy,
    experiment_name: Option<String>,
}

impl AgentOrchestrator {
    /// Create a new orchestrator with the given providers and strategy.
    pub fn new(
        providers: Vec<WeightedProvider>,
        strategy: SelectionStrategy,
        experiment_name: Option<String>,
    ) -> Self {
        Self {
            providers,
            strategy,
            experiment_name,
        }
    }

    /// Create an orchestrator from experiment config.
    pub fn from_experiment(
        providers: Vec<WeightedProvider>,
        experiment_name: &str,
        strategy_str: &str,
    ) -> Self {
        let strategy = match strategy_str {
            "weighted_random" => SelectionStrategy::WeightedRandom,
            "fallback" => SelectionStrategy::Fallback,
            _ => SelectionStrategy::Primary,
        };
        Self {
            providers,
            strategy,
            experiment_name: Some(experiment_name.to_string()),
        }
    }

    /// Select a provider index using weighted random selection.
    fn select_weighted_random(&self) -> usize {
        let total_weight: f64 = self.providers.iter().map(|p| p.weight).sum();
        if total_weight <= 0.0 || self.providers.is_empty() {
            return 0;
        }

        let mut rng = rand::rng();
        let roll: f64 = rng.random::<f64>() * total_weight;
        let mut cumulative = 0.0;

        for (i, provider) in self.providers.iter().enumerate() {
            cumulative += provider.weight;
            if roll < cumulative {
                return i;
            }
        }

        self.providers.len() - 1
    }
}

#[async_trait]
impl AgentRunner for AgentOrchestrator {
    fn name(&self) -> &str {
        "orchestrator"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        // Return capabilities of the primary (first) provider.
        self.providers
            .first()
            .map(|p| p.provider.capabilities())
            .unwrap_or_default()
    }

    fn build_prompt_for_issue(&self, issue: &Issue, context: &str, project_dir: &Path) -> String {
        // Use the primary provider's prompt building.
        self.providers
            .first()
            .map(|p| {
                p.provider
                    .build_prompt_for_issue(issue, context, project_dir)
            })
            .unwrap_or_default()
    }

    async fn execute_with_attempt(
        &self,
        prompt: &str,
        issue: Option<&Issue>,
        attempt_id: Option<i64>,
        project_dir: &Path,
    ) -> Result<AgentResult> {
        if self.providers.is_empty() {
            return Err(Error::runner("No providers configured in orchestrator"));
        }

        match self.strategy {
            SelectionStrategy::Primary => {
                let provider = &self.providers[0].provider;
                tracing::info!(
                    component = "orchestrator",
                    provider = provider.name(),
                    "Using primary provider"
                );
                provider
                    .execute_with_attempt(prompt, issue, attempt_id, project_dir)
                    .await
            }

            SelectionStrategy::WeightedRandom => {
                let idx = self.select_weighted_random();
                let provider = &self.providers[idx].provider;
                tracing::info!(
                    component = "orchestrator",
                    provider = provider.name(),
                    experiment = self.experiment_name.as_deref().unwrap_or("none"),
                    "Selected provider via weighted random"
                );
                provider
                    .execute_with_attempt(prompt, issue, attempt_id, project_dir)
                    .await
            }

            SelectionStrategy::Fallback => {
                let mut last_error = None;
                for (i, wp) in self.providers.iter().enumerate() {
                    let provider = &wp.provider;
                    tracing::info!(
                        component = "orchestrator",
                        provider = provider.name(),
                        attempt = i + 1,
                        total = self.providers.len(),
                        "Trying provider in fallback chain"
                    );

                    match provider
                        .execute_with_attempt(prompt, issue, attempt_id, project_dir)
                        .await
                    {
                        Ok(result) => {
                            // Only fall back on Err, NOT on AgentResult { success: false }.
                            // The agent tried but couldn't fix — that's a valid outcome.
                            return Ok(result);
                        }
                        Err(e) => {
                            tracing::warn!(
                                component = "orchestrator",
                                provider = provider.name(),
                                error = %e,
                                "Provider failed, trying next in fallback chain"
                            );
                            last_error = Some(e);
                        }
                    }
                }

                Err(last_error.unwrap_or_else(|| Error::runner("All providers failed")))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mock provider for testing.
    struct MockProvider {
        name: String,
        should_fail: bool,
    }

    #[async_trait]
    impl AgentRunner for MockProvider {
        fn name(&self) -> &str {
            &self.name
        }

        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::default()
        }

        fn build_prompt_for_issue(
            &self,
            _issue: &Issue,
            _context: &str,
            _project_dir: &Path,
        ) -> String {
            format!("prompt from {}", self.name)
        }

        async fn execute_with_attempt(
            &self,
            _prompt: &str,
            _issue: Option<&Issue>,
            _attempt_id: Option<i64>,
            _project_dir: &Path,
        ) -> Result<AgentResult> {
            if self.should_fail {
                Err(Error::runner(format!("{} failed", self.name)))
            } else {
                Ok(AgentResult {
                    success: true,
                    output: format!("result from {}", self.name),
                    pr_url: None,
                    changelog: None,
                    error: None,
                    blocking_question: None,
                    used_qa_ids: Vec::new(),
                    confidence: 0,
                    confidence_reasoning: None,
                    wrong_repo: None,
                })
            }
        }
    }

    #[test]
    fn test_primary_strategy_uses_first_provider() {
        let orchestrator = AgentOrchestrator::new(
            vec![
                WeightedProvider {
                    provider: Arc::new(MockProvider {
                        name: "alpha".into(),
                        should_fail: false,
                    }),
                    weight: 1.0,
                },
                WeightedProvider {
                    provider: Arc::new(MockProvider {
                        name: "beta".into(),
                        should_fail: false,
                    }),
                    weight: 1.0,
                },
            ],
            SelectionStrategy::Primary,
            None,
        );

        let prompt = orchestrator.build_prompt_for_issue(
            &Issue::new("1", "T-1", "Bug", "url", "test"),
            "ctx",
            Path::new("/tmp"),
        );
        assert_eq!(prompt, "prompt from alpha");
    }

    #[tokio::test]
    async fn test_fallback_skips_failing_provider() {
        let orchestrator = AgentOrchestrator::new(
            vec![
                WeightedProvider {
                    provider: Arc::new(MockProvider {
                        name: "failing".into(),
                        should_fail: true,
                    }),
                    weight: 1.0,
                },
                WeightedProvider {
                    provider: Arc::new(MockProvider {
                        name: "working".into(),
                        should_fail: false,
                    }),
                    weight: 1.0,
                },
            ],
            SelectionStrategy::Fallback,
            None,
        );

        let result = orchestrator
            .execute_with_attempt("test", None, None, Path::new("/tmp"))
            .await
            .unwrap();
        assert!(result.success);
        assert_eq!(result.output, "result from working");
    }

    #[tokio::test]
    async fn test_fallback_all_fail_returns_error() {
        let orchestrator = AgentOrchestrator::new(
            vec![
                WeightedProvider {
                    provider: Arc::new(MockProvider {
                        name: "fail1".into(),
                        should_fail: true,
                    }),
                    weight: 1.0,
                },
                WeightedProvider {
                    provider: Arc::new(MockProvider {
                        name: "fail2".into(),
                        should_fail: true,
                    }),
                    weight: 1.0,
                },
            ],
            SelectionStrategy::Fallback,
            None,
        );

        let result = orchestrator
            .execute_with_attempt("test", None, None, Path::new("/tmp"))
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_fallback_returns_unsuccessful_result_without_fallback() {
        // A provider that returns success=false should NOT trigger fallback.
        struct UnsuccessfulProvider;

        #[async_trait]
        impl AgentRunner for UnsuccessfulProvider {
            fn name(&self) -> &str {
                "unsuccessful"
            }
            fn capabilities(&self) -> ProviderCapabilities {
                ProviderCapabilities::default()
            }
            fn build_prompt_for_issue(
                &self,
                _issue: &Issue,
                _context: &str,
                _project_dir: &Path,
            ) -> String {
                String::new()
            }
            async fn execute_with_attempt(
                &self,
                _prompt: &str,
                _issue: Option<&Issue>,
                _attempt_id: Option<i64>,
                _project_dir: &Path,
            ) -> Result<AgentResult> {
                Ok(AgentResult {
                    success: false,
                    output: "could not fix".to_string(),
                    pr_url: None,
                    changelog: None,
                    error: Some("failed to fix".to_string()),
                    blocking_question: None,
                    used_qa_ids: Vec::new(),
                    confidence: 0,
                    confidence_reasoning: None,
                    wrong_repo: None,
                })
            }
        }

        let orchestrator = AgentOrchestrator::new(
            vec![
                WeightedProvider {
                    provider: Arc::new(UnsuccessfulProvider),
                    weight: 1.0,
                },
                WeightedProvider {
                    provider: Arc::new(MockProvider {
                        name: "backup".into(),
                        should_fail: false,
                    }),
                    weight: 1.0,
                },
            ],
            SelectionStrategy::Fallback,
            None,
        );

        let result = orchestrator
            .execute_with_attempt("test", None, None, Path::new("/tmp"))
            .await
            .unwrap();
        // Should return the unsuccessful result, NOT fall back to backup.
        assert!(!result.success);
        assert_eq!(result.output, "could not fix");
    }

    #[tokio::test]
    async fn test_empty_orchestrator_returns_error() {
        let orchestrator = AgentOrchestrator::new(vec![], SelectionStrategy::Primary, None);

        let result = orchestrator
            .execute_with_attempt("test", None, None, Path::new("/tmp"))
            .await;
        assert!(result.is_err());
    }

    #[test]
    fn test_weighted_selection_single_provider() {
        let orchestrator = AgentOrchestrator::new(
            vec![WeightedProvider {
                provider: Arc::new(MockProvider {
                    name: "only".into(),
                    should_fail: false,
                }),
                weight: 1.0,
            }],
            SelectionStrategy::WeightedRandom,
            None,
        );

        // With a single provider, it should always be selected.
        for _ in 0..10 {
            assert_eq!(orchestrator.select_weighted_random(), 0);
        }
    }

    #[test]
    fn test_weighted_selection_respects_weights() {
        let orchestrator = AgentOrchestrator::new(
            vec![
                WeightedProvider {
                    provider: Arc::new(MockProvider {
                        name: "heavy".into(),
                        should_fail: false,
                    }),
                    weight: 100.0,
                },
                WeightedProvider {
                    provider: Arc::new(MockProvider {
                        name: "light".into(),
                        should_fail: false,
                    }),
                    weight: 0.001,
                },
            ],
            SelectionStrategy::WeightedRandom,
            None,
        );

        // With 100 vs 0.001 weights, the heavy provider should be selected
        // almost exclusively. Run 100 trials.
        let mut heavy_count = 0;
        for _ in 0..100 {
            if orchestrator.select_weighted_random() == 0 {
                heavy_count += 1;
            }
        }
        assert!(
            heavy_count >= 95,
            "Heavy provider selected only {} times out of 100",
            heavy_count
        );
    }

    #[test]
    fn test_from_experiment() {
        let orchestrator = AgentOrchestrator::from_experiment(
            vec![WeightedProvider {
                provider: Arc::new(MockProvider {
                    name: "test".into(),
                    should_fail: false,
                }),
                weight: 1.0,
            }],
            "my-experiment",
            "fallback",
        );
        assert_eq!(
            orchestrator.experiment_name.as_deref(),
            Some("my-experiment")
        );
        assert!(matches!(orchestrator.strategy, SelectionStrategy::Fallback));
    }

    #[tokio::test]
    async fn test_weighted_random_executes_selected_provider() {
        // With a single provider and WeightedRandom strategy, it should always
        // execute that provider.
        let orchestrator = AgentOrchestrator::new(
            vec![WeightedProvider {
                provider: Arc::new(MockProvider {
                    name: "only-provider".into(),
                    should_fail: false,
                }),
                weight: 1.0,
            }],
            SelectionStrategy::WeightedRandom,
            Some("test-experiment".to_string()),
        );

        let result = orchestrator
            .execute_with_attempt("test prompt", None, None, Path::new("/tmp"))
            .await
            .unwrap();
        assert!(result.success);
        assert_eq!(result.output, "result from only-provider");
    }

    #[tokio::test]
    async fn test_weighted_random_with_failing_provider_returns_error() {
        let orchestrator = AgentOrchestrator::new(
            vec![WeightedProvider {
                provider: Arc::new(MockProvider {
                    name: "failing".into(),
                    should_fail: true,
                }),
                weight: 1.0,
            }],
            SelectionStrategy::WeightedRandom,
            None,
        );

        let result = orchestrator
            .execute_with_attempt("test", None, None, Path::new("/tmp"))
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_primary_executes_first_provider() {
        let orchestrator = AgentOrchestrator::new(
            vec![
                WeightedProvider {
                    provider: Arc::new(MockProvider {
                        name: "first".into(),
                        should_fail: false,
                    }),
                    weight: 1.0,
                },
                WeightedProvider {
                    provider: Arc::new(MockProvider {
                        name: "second".into(),
                        should_fail: false,
                    }),
                    weight: 1.0,
                },
            ],
            SelectionStrategy::Primary,
            None,
        );

        let result = orchestrator
            .execute_with_attempt("test", None, None, Path::new("/tmp"))
            .await
            .unwrap();
        assert_eq!(result.output, "result from first");
    }

    #[test]
    fn test_orchestrator_name() {
        let orchestrator = AgentOrchestrator::new(vec![], SelectionStrategy::Primary, None);
        assert_eq!(orchestrator.name(), "orchestrator");
    }

    #[test]
    fn test_orchestrator_capabilities_delegates_to_first_provider() {
        struct CapProvider;
        #[async_trait]
        impl AgentRunner for CapProvider {
            fn name(&self) -> &str {
                "cap-test"
            }
            fn capabilities(&self) -> ProviderCapabilities {
                ProviderCapabilities {
                    structured_output: true,
                    tool_permissions: true,
                    custom_instructions: true,
                    streaming_events: true,
                    cost_reporting: true,
                }
            }
            fn build_prompt_for_issue(&self, _: &Issue, _: &str, _: &Path) -> String {
                String::new()
            }
            async fn execute_with_attempt(
                &self,
                _: &str,
                _: Option<&Issue>,
                _: Option<i64>,
                _: &Path,
            ) -> Result<AgentResult> {
                Ok(AgentResult {
                    success: true,
                    output: String::new(),
                    pr_url: None,
                    changelog: None,
                    error: None,
                    blocking_question: None,
                    used_qa_ids: Vec::new(),
                    confidence: 0,
                    confidence_reasoning: None,
                    wrong_repo: None,
                })
            }
        }

        let orchestrator = AgentOrchestrator::new(
            vec![
                WeightedProvider {
                    provider: Arc::new(CapProvider),
                    weight: 1.0,
                },
                WeightedProvider {
                    provider: Arc::new(MockProvider {
                        name: "other".into(),
                        should_fail: false,
                    }),
                    weight: 1.0,
                },
            ],
            SelectionStrategy::Primary,
            None,
        );

        let caps = orchestrator.capabilities();
        assert!(caps.structured_output);
        assert!(caps.tool_permissions);
        assert!(caps.custom_instructions);
        assert!(caps.streaming_events);
        assert!(caps.cost_reporting);
    }

    #[test]
    fn test_orchestrator_capabilities_empty_returns_default() {
        let orchestrator = AgentOrchestrator::new(vec![], SelectionStrategy::Primary, None);
        let caps = orchestrator.capabilities();
        assert!(!caps.structured_output);
        assert!(!caps.cost_reporting);
    }

    #[test]
    fn test_orchestrator_build_prompt_empty_returns_empty() {
        let orchestrator = AgentOrchestrator::new(vec![], SelectionStrategy::Primary, None);
        let prompt = orchestrator.build_prompt_for_issue(
            &Issue::new("1", "T-1", "Bug", "url", "test"),
            "ctx",
            Path::new("/tmp"),
        );
        assert!(prompt.is_empty());
    }

    #[test]
    fn test_from_experiment_weighted_random() {
        let orchestrator = AgentOrchestrator::from_experiment(
            vec![WeightedProvider {
                provider: Arc::new(MockProvider {
                    name: "test".into(),
                    should_fail: false,
                }),
                weight: 1.0,
            }],
            "exp-wr",
            "weighted_random",
        );
        assert!(matches!(
            orchestrator.strategy,
            SelectionStrategy::WeightedRandom
        ));
        assert_eq!(orchestrator.experiment_name.as_deref(), Some("exp-wr"));
    }

    #[test]
    fn test_from_experiment_unknown_strategy_defaults_to_primary() {
        let orchestrator = AgentOrchestrator::from_experiment(
            vec![WeightedProvider {
                provider: Arc::new(MockProvider {
                    name: "test".into(),
                    should_fail: false,
                }),
                weight: 1.0,
            }],
            "exp-unknown",
            "round_robin",
        );
        assert!(matches!(orchestrator.strategy, SelectionStrategy::Primary));
    }

    #[test]
    fn test_weighted_selection_zero_weights_returns_zero() {
        let orchestrator = AgentOrchestrator::new(
            vec![
                WeightedProvider {
                    provider: Arc::new(MockProvider {
                        name: "a".into(),
                        should_fail: false,
                    }),
                    weight: 0.0,
                },
                WeightedProvider {
                    provider: Arc::new(MockProvider {
                        name: "b".into(),
                        should_fail: false,
                    }),
                    weight: 0.0,
                },
            ],
            SelectionStrategy::WeightedRandom,
            None,
        );
        assert_eq!(orchestrator.select_weighted_random(), 0);
    }

    #[tokio::test]
    async fn test_fallback_single_working_provider() {
        let orchestrator = AgentOrchestrator::new(
            vec![WeightedProvider {
                provider: Arc::new(MockProvider {
                    name: "sole".into(),
                    should_fail: false,
                }),
                weight: 1.0,
            }],
            SelectionStrategy::Fallback,
            None,
        );

        let result = orchestrator
            .execute_with_attempt("test", None, None, Path::new("/tmp"))
            .await
            .unwrap();
        assert!(result.success);
        assert_eq!(result.output, "result from sole");
    }

    #[tokio::test]
    async fn test_fallback_three_providers_first_two_fail() {
        let orchestrator = AgentOrchestrator::new(
            vec![
                WeightedProvider {
                    provider: Arc::new(MockProvider {
                        name: "fail-a".into(),
                        should_fail: true,
                    }),
                    weight: 1.0,
                },
                WeightedProvider {
                    provider: Arc::new(MockProvider {
                        name: "fail-b".into(),
                        should_fail: true,
                    }),
                    weight: 1.0,
                },
                WeightedProvider {
                    provider: Arc::new(MockProvider {
                        name: "success-c".into(),
                        should_fail: false,
                    }),
                    weight: 1.0,
                },
            ],
            SelectionStrategy::Fallback,
            None,
        );

        let result = orchestrator
            .execute_with_attempt("test", None, None, Path::new("/tmp"))
            .await
            .unwrap();
        assert!(result.success);
        assert_eq!(result.output, "result from success-c");
    }

    #[tokio::test]
    async fn test_fallback_all_fail_returns_last_error() {
        let orchestrator = AgentOrchestrator::new(
            vec![
                WeightedProvider {
                    provider: Arc::new(MockProvider {
                        name: "first".into(),
                        should_fail: true,
                    }),
                    weight: 1.0,
                },
                WeightedProvider {
                    provider: Arc::new(MockProvider {
                        name: "last".into(),
                        should_fail: true,
                    }),
                    weight: 1.0,
                },
            ],
            SelectionStrategy::Fallback,
            None,
        );

        let err = orchestrator
            .execute_with_attempt("test", None, None, Path::new("/tmp"))
            .await
            .unwrap_err();
        // Should contain the last provider's error
        assert!(
            err.to_string().contains("last failed"),
            "Expected error from last provider, got: {}",
            err
        );
    }

    #[tokio::test]
    async fn test_empty_orchestrator_weighted_random_returns_error() {
        let orchestrator = AgentOrchestrator::new(vec![], SelectionStrategy::WeightedRandom, None);
        let result = orchestrator
            .execute_with_attempt("test", None, None, Path::new("/tmp"))
            .await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("No providers configured"));
    }

    #[tokio::test]
    async fn test_empty_orchestrator_fallback_returns_error() {
        let orchestrator = AgentOrchestrator::new(vec![], SelectionStrategy::Fallback, None);
        let result = orchestrator
            .execute_with_attempt("test", None, None, Path::new("/tmp"))
            .await;
        assert!(result.is_err());
    }

    #[test]
    fn test_weighted_selection_equal_weights_distributed() {
        let orchestrator = AgentOrchestrator::new(
            vec![
                WeightedProvider {
                    provider: Arc::new(MockProvider {
                        name: "a".into(),
                        should_fail: false,
                    }),
                    weight: 1.0,
                },
                WeightedProvider {
                    provider: Arc::new(MockProvider {
                        name: "b".into(),
                        should_fail: false,
                    }),
                    weight: 1.0,
                },
            ],
            SelectionStrategy::WeightedRandom,
            None,
        );

        let mut count_a = 0;
        let trials = 1000;
        for _ in 0..trials {
            if orchestrator.select_weighted_random() == 0 {
                count_a += 1;
            }
        }
        // With equal weights, expect ~50% for each. Allow ±15% tolerance.
        assert!(
            count_a > 350 && count_a < 650,
            "Expected ~500 selections for provider A, got {} out of {}",
            count_a,
            trials
        );
    }

    #[tokio::test]
    async fn test_primary_with_attempt_id_passes_through() {
        use std::sync::atomic::{AtomicI64, Ordering};

        struct AttemptCapture {
            captured_attempt: AtomicI64,
        }

        #[async_trait]
        impl AgentRunner for AttemptCapture {
            fn name(&self) -> &str {
                "capture"
            }
            fn capabilities(&self) -> ProviderCapabilities {
                ProviderCapabilities::default()
            }
            fn build_prompt_for_issue(&self, _: &Issue, _: &str, _: &Path) -> String {
                String::new()
            }
            async fn execute_with_attempt(
                &self,
                _prompt: &str,
                _issue: Option<&Issue>,
                attempt_id: Option<i64>,
                _project_dir: &Path,
            ) -> Result<AgentResult> {
                if let Some(id) = attempt_id {
                    self.captured_attempt.store(id, Ordering::SeqCst);
                }
                Ok(AgentResult {
                    success: true,
                    output: String::new(),
                    pr_url: None,
                    changelog: None,
                    error: None,
                    blocking_question: None,
                    used_qa_ids: Vec::new(),
                    confidence: 0,
                    confidence_reasoning: None,
                    wrong_repo: None,
                })
            }
        }

        let capture = Arc::new(AttemptCapture {
            captured_attempt: AtomicI64::new(0),
        });
        let orchestrator = AgentOrchestrator::new(
            vec![WeightedProvider {
                provider: capture.clone(),
                weight: 1.0,
            }],
            SelectionStrategy::Primary,
            None,
        );

        orchestrator
            .execute_with_attempt("test", None, Some(42), Path::new("/tmp"))
            .await
            .unwrap();
        assert_eq!(capture.captured_attempt.load(Ordering::SeqCst), 42);
    }

    #[tokio::test]
    async fn test_orchestrator_passes_issue_through() {
        use std::sync::Mutex;

        struct IssueCapture {
            captured_source: Mutex<Option<String>>,
        }

        #[async_trait]
        impl AgentRunner for IssueCapture {
            fn name(&self) -> &str {
                "issue-capture"
            }
            fn capabilities(&self) -> ProviderCapabilities {
                ProviderCapabilities::default()
            }
            fn build_prompt_for_issue(&self, _: &Issue, _: &str, _: &Path) -> String {
                String::new()
            }
            async fn execute_with_attempt(
                &self,
                _prompt: &str,
                issue: Option<&Issue>,
                _attempt_id: Option<i64>,
                _project_dir: &Path,
            ) -> Result<AgentResult> {
                if let Some(i) = issue {
                    *self.captured_source.lock().unwrap() = Some(i.source.clone());
                }
                Ok(AgentResult {
                    success: true,
                    output: String::new(),
                    pr_url: None,
                    changelog: None,
                    error: None,
                    blocking_question: None,
                    used_qa_ids: Vec::new(),
                    confidence: 0,
                    confidence_reasoning: None,
                    wrong_repo: None,
                })
            }
        }

        let capture = Arc::new(IssueCapture {
            captured_source: Mutex::new(None),
        });
        let orchestrator = AgentOrchestrator::new(
            vec![WeightedProvider {
                provider: capture.clone(),
                weight: 1.0,
            }],
            SelectionStrategy::Primary,
            None,
        );

        let issue = Issue::new("42", "LIN-42", "Bug", "url", "linear");
        orchestrator
            .execute_with_attempt("test", Some(&issue), None, Path::new("/tmp"))
            .await
            .unwrap();
        assert_eq!(
            capture.captured_source.lock().unwrap().as_deref(),
            Some("linear")
        );
    }

    // --- Additional orchestrator tests ---

    #[tokio::test]
    async fn test_orchestrator_passes_prompt_through() {
        use std::sync::Mutex;

        struct PromptCapture {
            captured_prompt: Mutex<Option<String>>,
        }

        #[async_trait]
        impl AgentRunner for PromptCapture {
            fn name(&self) -> &str {
                "prompt-capture"
            }
            fn capabilities(&self) -> ProviderCapabilities {
                ProviderCapabilities::default()
            }
            fn build_prompt_for_issue(&self, _: &Issue, _: &str, _: &Path) -> String {
                String::new()
            }
            async fn execute_with_attempt(
                &self,
                prompt: &str,
                _issue: Option<&Issue>,
                _attempt_id: Option<i64>,
                _project_dir: &Path,
            ) -> Result<AgentResult> {
                *self.captured_prompt.lock().unwrap() = Some(prompt.to_string());
                Ok(AgentResult {
                    success: true,
                    output: String::new(),
                    pr_url: None,
                    changelog: None,
                    error: None,
                    blocking_question: None,
                    used_qa_ids: Vec::new(),
                    confidence: 0,
                    confidence_reasoning: None,
                    wrong_repo: None,
                })
            }
        }

        let capture = Arc::new(PromptCapture {
            captured_prompt: Mutex::new(None),
        });
        let orchestrator = AgentOrchestrator::new(
            vec![WeightedProvider {
                provider: capture.clone(),
                weight: 1.0,
            }],
            SelectionStrategy::Primary,
            None,
        );

        orchestrator
            .execute_with_attempt("fix the auth bug", None, None, Path::new("/tmp"))
            .await
            .unwrap();
        assert_eq!(
            capture.captured_prompt.lock().unwrap().as_deref(),
            Some("fix the auth bug")
        );
    }

    #[tokio::test]
    async fn test_orchestrator_passes_project_dir_through() {
        use std::sync::Mutex;

        struct DirCapture {
            captured_dir: Mutex<Option<String>>,
        }

        #[async_trait]
        impl AgentRunner for DirCapture {
            fn name(&self) -> &str {
                "dir-capture"
            }
            fn capabilities(&self) -> ProviderCapabilities {
                ProviderCapabilities::default()
            }
            fn build_prompt_for_issue(&self, _: &Issue, _: &str, _: &Path) -> String {
                String::new()
            }
            async fn execute_with_attempt(
                &self,
                _prompt: &str,
                _issue: Option<&Issue>,
                _attempt_id: Option<i64>,
                project_dir: &Path,
            ) -> Result<AgentResult> {
                *self.captured_dir.lock().unwrap() = Some(project_dir.display().to_string());
                Ok(AgentResult {
                    success: true,
                    output: String::new(),
                    pr_url: None,
                    changelog: None,
                    error: None,
                    blocking_question: None,
                    used_qa_ids: Vec::new(),
                    confidence: 0,
                    confidence_reasoning: None,
                    wrong_repo: None,
                })
            }
        }

        let capture = Arc::new(DirCapture {
            captured_dir: Mutex::new(None),
        });
        let orchestrator = AgentOrchestrator::new(
            vec![WeightedProvider {
                provider: capture.clone(),
                weight: 1.0,
            }],
            SelectionStrategy::Primary,
            None,
        );

        orchestrator
            .execute_with_attempt("test", None, None, Path::new("/my/project"))
            .await
            .unwrap();
        assert_eq!(
            capture.captured_dir.lock().unwrap().as_deref(),
            Some("/my/project")
        );
    }

    #[test]
    fn test_build_prompt_delegates_to_named_provider() {
        struct NamedPromptProvider {
            provider_name: String,
        }

        #[async_trait]
        impl AgentRunner for NamedPromptProvider {
            fn name(&self) -> &str {
                &self.provider_name
            }
            fn capabilities(&self) -> ProviderCapabilities {
                ProviderCapabilities::default()
            }
            fn build_prompt_for_issue(&self, issue: &Issue, context: &str, _: &Path) -> String {
                format!("[{}] {} - {}", self.provider_name, issue.short_id, context)
            }
            async fn execute_with_attempt(
                &self,
                _: &str,
                _: Option<&Issue>,
                _: Option<i64>,
                _: &Path,
            ) -> Result<AgentResult> {
                Ok(AgentResult {
                    success: true,
                    output: String::new(),
                    pr_url: None,
                    changelog: None,
                    error: None,
                    blocking_question: None,
                    used_qa_ids: Vec::new(),
                    confidence: 0,
                    confidence_reasoning: None,
                    wrong_repo: None,
                })
            }
        }

        let orchestrator = AgentOrchestrator::new(
            vec![
                WeightedProvider {
                    provider: Arc::new(NamedPromptProvider {
                        provider_name: "claude".into(),
                    }),
                    weight: 1.0,
                },
                WeightedProvider {
                    provider: Arc::new(NamedPromptProvider {
                        provider_name: "codex".into(),
                    }),
                    weight: 1.0,
                },
            ],
            SelectionStrategy::Primary,
            None,
        );

        let issue = Issue::new("1", "LIN-1", "Bug", "url", "linear");
        let prompt = orchestrator.build_prompt_for_issue(&issue, "context", Path::new("/tmp"));
        // Should delegate to the FIRST provider (claude), not codex
        assert!(
            prompt.starts_with("[claude]"),
            "Expected prompt from claude, got: {}",
            prompt
        );
        assert!(prompt.contains("LIN-1"));
    }

    #[test]
    fn test_from_experiment_all_strategies() {
        let make = |strategy: &str| {
            AgentOrchestrator::from_experiment(
                vec![WeightedProvider {
                    provider: Arc::new(MockProvider {
                        name: "t".into(),
                        should_fail: false,
                    }),
                    weight: 1.0,
                }],
                "exp",
                strategy,
            )
        };

        assert!(matches!(
            make("weighted_random").strategy,
            SelectionStrategy::WeightedRandom
        ));
        assert!(matches!(
            make("fallback").strategy,
            SelectionStrategy::Fallback
        ));
        assert!(matches!(
            make("primary").strategy,
            SelectionStrategy::Primary
        ));
        assert!(matches!(
            make("anything_else").strategy,
            SelectionStrategy::Primary
        ));
        assert!(matches!(make("").strategy, SelectionStrategy::Primary));
    }

    #[test]
    fn test_weighted_selection_negative_weights_returns_zero() {
        let orchestrator = AgentOrchestrator::new(
            vec![
                WeightedProvider {
                    provider: Arc::new(MockProvider {
                        name: "a".into(),
                        should_fail: false,
                    }),
                    weight: -1.0,
                },
                WeightedProvider {
                    provider: Arc::new(MockProvider {
                        name: "b".into(),
                        should_fail: false,
                    }),
                    weight: -2.0,
                },
            ],
            SelectionStrategy::WeightedRandom,
            None,
        );
        // Negative weights sum to < 0, should return index 0
        assert_eq!(orchestrator.select_weighted_random(), 0);
    }

    #[test]
    fn test_weighted_selection_empty_providers_returns_zero() {
        let orchestrator = AgentOrchestrator::new(vec![], SelectionStrategy::WeightedRandom, None);
        assert_eq!(orchestrator.select_weighted_random(), 0);
    }

    #[tokio::test]
    async fn test_fallback_with_pr_url_preserved() {
        struct PrProvider;

        #[async_trait]
        impl AgentRunner for PrProvider {
            fn name(&self) -> &str {
                "pr-provider"
            }
            fn capabilities(&self) -> ProviderCapabilities {
                ProviderCapabilities::default()
            }
            fn build_prompt_for_issue(&self, _: &Issue, _: &str, _: &Path) -> String {
                String::new()
            }
            async fn execute_with_attempt(
                &self,
                _: &str,
                _: Option<&Issue>,
                _: Option<i64>,
                _: &Path,
            ) -> Result<AgentResult> {
                Ok(AgentResult {
                    success: true,
                    output: "created PR".to_string(),
                    pr_url: Some("https://github.com/org/repo/pull/42".to_string()),
                    changelog: Some("- Fixed auth".to_string()),
                    error: None,
                    blocking_question: None,
                    used_qa_ids: vec![1],
                    confidence: 0,
                    confidence_reasoning: None,
                    wrong_repo: None,
                })
            }
        }

        let orchestrator = AgentOrchestrator::new(
            vec![
                WeightedProvider {
                    provider: Arc::new(MockProvider {
                        name: "fail".into(),
                        should_fail: true,
                    }),
                    weight: 1.0,
                },
                WeightedProvider {
                    provider: Arc::new(PrProvider),
                    weight: 1.0,
                },
            ],
            SelectionStrategy::Fallback,
            Some("test-exp".to_string()),
        );

        let result = orchestrator
            .execute_with_attempt("test", None, None, Path::new("/tmp"))
            .await
            .unwrap();
        assert!(result.success);
        assert_eq!(
            result.pr_url.as_deref(),
            Some("https://github.com/org/repo/pull/42")
        );
        assert_eq!(result.changelog.as_deref(), Some("- Fixed auth"));
        assert_eq!(result.used_qa_ids, vec![1]);
    }

    #[tokio::test]
    async fn test_primary_failing_provider_returns_error() {
        let orchestrator = AgentOrchestrator::new(
            vec![
                WeightedProvider {
                    provider: Arc::new(MockProvider {
                        name: "failing-primary".into(),
                        should_fail: true,
                    }),
                    weight: 1.0,
                },
                WeightedProvider {
                    provider: Arc::new(MockProvider {
                        name: "backup".into(),
                        should_fail: false,
                    }),
                    weight: 1.0,
                },
            ],
            SelectionStrategy::Primary,
            None,
        );

        // Primary strategy should NOT fall back to the backup, it should return the error
        let result = orchestrator
            .execute_with_attempt("test", None, None, Path::new("/tmp"))
            .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("failing-primary"));
    }

    #[test]
    fn test_selection_strategy_debug_format() {
        // SelectionStrategy derives Debug
        let s = format!("{:?}", SelectionStrategy::Primary);
        assert_eq!(s, "Primary");
        let s = format!("{:?}", SelectionStrategy::WeightedRandom);
        assert_eq!(s, "WeightedRandom");
        let s = format!("{:?}", SelectionStrategy::Fallback);
        assert_eq!(s, "Fallback");
    }

    #[test]
    fn test_selection_strategy_clone() {
        let original = SelectionStrategy::Fallback;
        let cloned = original.clone();
        assert!(matches!(cloned, SelectionStrategy::Fallback));
    }
}
