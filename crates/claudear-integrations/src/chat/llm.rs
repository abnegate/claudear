//! Local LLM inference engine using llama-cpp-2.

use claudear_core::error::{Error, Result};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing;

/// Configuration for the LLM engine.
#[derive(Debug, Clone)]
pub struct LlmConfig {
    /// Path to the GGUF model file.
    pub model_path: PathBuf,
    /// Context window length (tokens).
    pub context_length: u32,
    /// Number of layers to offload to GPU (0 = CPU only, 99 = all).
    pub gpu_layers: u32,
    /// Number of threads for inference (0 = auto-detect).
    pub threads: u32,
    /// Maximum time for a single inference call (None = no limit).
    pub timeout: Option<Duration>,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            context_length: 16384,
            gpu_layers: 99,
            threads: 0,
            timeout: Some(Duration::from_secs(120)),
        }
    }
}

/// Parameters controlling text generation.
#[derive(Debug, Clone)]
pub struct GenerationParams {
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub stop_sequences: Vec<String>,
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            max_tokens: 2048,
            temperature: 0.7,
            top_p: 0.9,
            stop_sequences: Vec::new(),
        }
    }
}

/// Local LLM inference engine wrapping llama-cpp-2.
pub struct LlmEngine {
    model: llama_cpp_2::model::LlamaModel,
    _backend: llama_cpp_2::llama_backend::LlamaBackend,
    context_length: u32,
    timeout: Option<Duration>,
}

/// Extra tokens added on top of (prompt + max_tokens) when sizing a per-call
/// context window, to leave a small margin for sampler/decoder bookkeeping.
const CONTEXT_SLACK_TOKENS: u32 = 16;

/// Context window to allocate for a *single* completion call.
///
/// Every completion builds a fresh llama context whose KV cache and compute
/// buffers are sized to `n_ctx`. Allocating the full configured window
/// (e.g. 16384) on every call is very expensive and, for short prompts with
/// small generations (e.g. an 8-token intent classification), dominates
/// latency. Size the context to what the call actually needs, clamped to
/// `[prompt + 1, context_length]`.
///
/// Callers verify `prompt_tokens < context_length` before invoking, so the
/// clamp range is always valid (min <= max).
fn effective_context_len(prompt_tokens: usize, max_tokens: u32, context_length: u32) -> u32 {
    let prompt = prompt_tokens as u32;
    let needed = prompt
        .saturating_add(max_tokens)
        .saturating_add(CONTEXT_SLACK_TOKENS);
    // `min` is capped at `context_length` so the clamp range stays valid even
    // for an over-long prompt (callers reject those earlier; this just avoids a
    // panic on degenerate input).
    let min = prompt.saturating_add(1).min(context_length);
    needed.clamp(min, context_length)
}

impl LlmEngine {
    /// Per-call context window (see [`effective_context_len`]).
    fn effective_context_len(&self, prompt_tokens: usize, max_tokens: u32) -> u32 {
        effective_context_len(prompt_tokens, max_tokens, self.context_length)
    }

    /// Load a GGUF model from disk.
    pub fn load(config: &LlmConfig) -> Result<Self> {
        tracing::info!(
            model_path = %config.model_path.display(),
            context_length = config.context_length,
            gpu_layers = config.gpu_layers,
            threads = config.threads,
            "Loading LLM model"
        );

        let mut backend = llama_cpp_2::llama_backend::LlamaBackend::init()
            .map_err(|e| Error::config(format!("Failed to init llama backend: {e}")))?;

        // Suppress verbose llama.cpp internal logging (KV cache layers, scheduler, etc.)
        backend.void_logs();

        let model_params = {
            let params = llama_cpp_2::model::params::LlamaModelParams::default();
            params.with_n_gpu_layers(config.gpu_layers)
        };

        let model = llama_cpp_2::model::LlamaModel::load_from_file(
            &backend,
            &config.model_path,
            &model_params,
        )
        .map_err(|e| Error::config(format!("Failed to load model: {e}")))?;

        tracing::info!("LLM model loaded successfully");

        Ok(Self {
            model,
            _backend: backend,
            context_length: config.context_length,
            timeout: config.timeout,
        })
    }

    /// Generate a completion, returning collected token strings.
    ///
    /// Returns a vec of decoded token strings. The vec is empty if generation produced nothing.
    pub fn complete_streaming(
        &self,
        prompt: &str,
        params: &GenerationParams,
    ) -> Result<Vec<String>> {
        use llama_cpp_2::context::params::LlamaContextParams;
        use llama_cpp_2::llama_batch::LlamaBatch;
        use llama_cpp_2::sampling::LlamaSampler;

        // Tokenize and check length BEFORE creating the expensive context
        let tokens = self
            .model
            .str_to_token(prompt, llama_cpp_2::model::AddBos::Always)
            .map_err(|e| Error::Other(format!("Tokenization failed: {e}")))?;

        if tokens.len() as u32 >= self.context_length {
            return Err(Error::Other(format!(
                "Prompt ({} tokens) exceeds context length ({})",
                tokens.len(),
                self.context_length
            )));
        }

        let threads = std::thread::available_parallelism()
            .map(|n| (n.get() as u32).max(1) / 2)
            .unwrap_or(4)
            .max(1);

        let n_ctx = self.effective_context_len(tokens.len(), params.max_tokens);

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(n_ctx))
            .with_n_batch(n_ctx)
            .with_n_threads(threads as i32)
            .with_n_threads_batch(threads as i32);

        let mut ctx = self
            .model
            .new_context(&self._backend, ctx_params)
            .map_err(|e| Error::Other(format!("Failed to create context: {e}")))?;

        // Create batch and fill with prompt tokens
        let mut batch = LlamaBatch::new(n_ctx as usize, 1);

        for (i, &token) in tokens.iter().enumerate() {
            let is_last = i == tokens.len() - 1;
            batch
                .add(token, i as i32, &[0], is_last)
                .map_err(|_| Error::Other("Failed to add token to batch".into()))?;
        }

        // Process prompt
        ctx.decode(&mut batch)
            .map_err(|e| Error::Other(format!("Decode failed: {e}")))?;

        // Build sampler chain: temperature -> top-p -> dist (random sampling)
        let seed = rand::random::<u32>();
        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::temp(params.temperature),
            LlamaSampler::top_p(params.top_p, 1),
            LlamaSampler::dist(seed),
        ]);

        // Create a UTF-8 decoder for token-to-string conversion
        let mut decoder = encoding_rs::UTF_8.new_decoder();

        // Generate tokens
        let mut output_tokens = Vec::new();
        let prompt_len = tokens.len() as i32;
        let max_gen = params
            .max_tokens
            .min(n_ctx.saturating_sub(tokens.len() as u32));
        let mut accumulated = String::new();
        let start = Instant::now();

        for i in 0..max_gen {
            // Check inference timeout
            if let Some(timeout) = self.timeout {
                if start.elapsed() > timeout {
                    tracing::warn!(
                        elapsed_secs = start.elapsed().as_secs(),
                        timeout_secs = timeout.as_secs(),
                        tokens_generated = output_tokens.len(),
                        "LLM inference timed out"
                    );
                    return Err(Error::Other(format!(
                        "LLM inference timed out after {}s",
                        timeout.as_secs()
                    )));
                }
            }

            // Sample next token using the sampler chain
            let new_token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(new_token);

            // Check for end of generation
            if self.model.is_eog_token(new_token) {
                break;
            }

            // Decode token to string
            let token_str = self
                .model
                .token_to_piece(new_token, &mut decoder, false, None)
                .unwrap_or_default();

            // Check stop sequences against accumulated output
            accumulated.push_str(&token_str);
            let should_stop = params
                .stop_sequences
                .iter()
                .any(|seq| accumulated.contains(seq.as_str()));

            if should_stop {
                // Remove the stop sequence from the last token if partially included
                break;
            }

            output_tokens.push(token_str);

            // Prepare next batch
            batch.clear();
            batch
                .add(new_token, prompt_len + i as i32, &[0], true)
                .map_err(|_| Error::Other("Failed to add token to batch".into()))?;

            ctx.decode(&mut batch)
                .map_err(|e| Error::Other(format!("Decode step failed: {e}")))?;
        }

        Ok(output_tokens)
    }

    /// Generate a completion, sending tokens through an mpsc channel as they are produced.
    ///
    /// Checks `cancel` at the top of each iteration and breaks if set.
    /// Breaks if the receiver is dropped (tx.blocking_send fails).
    pub fn complete_streaming_channel(
        &self,
        prompt: &str,
        params: &GenerationParams,
        tx: tokio::sync::mpsc::Sender<String>,
        cancel: Arc<AtomicBool>,
    ) -> Result<()> {
        use llama_cpp_2::context::params::LlamaContextParams;
        use llama_cpp_2::llama_batch::LlamaBatch;
        use llama_cpp_2::sampling::LlamaSampler;

        // Tokenize and check length BEFORE creating the expensive context
        let tokens = self
            .model
            .str_to_token(prompt, llama_cpp_2::model::AddBos::Always)
            .map_err(|e| Error::Other(format!("Tokenization failed: {e}")))?;

        if tokens.len() as u32 >= self.context_length {
            return Err(Error::Other(format!(
                "Prompt ({} tokens) exceeds context length ({})",
                tokens.len(),
                self.context_length
            )));
        }

        let threads = std::thread::available_parallelism()
            .map(|n| (n.get() as u32).max(1) / 2)
            .unwrap_or(4)
            .max(1);

        let n_ctx = self.effective_context_len(tokens.len(), params.max_tokens);

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(n_ctx))
            .with_n_batch(n_ctx)
            .with_n_threads(threads as i32)
            .with_n_threads_batch(threads as i32);

        let mut ctx = self
            .model
            .new_context(&self._backend, ctx_params)
            .map_err(|e| Error::Other(format!("Failed to create context: {e}")))?;

        let mut batch = LlamaBatch::new(n_ctx as usize, 1);

        for (i, &token) in tokens.iter().enumerate() {
            let is_last = i == tokens.len() - 1;
            batch
                .add(token, i as i32, &[0], is_last)
                .map_err(|_| Error::Other("Failed to add token to batch".into()))?;
        }

        ctx.decode(&mut batch)
            .map_err(|e| Error::Other(format!("Decode failed: {e}")))?;

        let seed = rand::random::<u32>();
        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::temp(params.temperature),
            LlamaSampler::top_p(params.top_p, 1),
            LlamaSampler::dist(seed),
        ]);

        let mut decoder = encoding_rs::UTF_8.new_decoder();

        let prompt_len = tokens.len() as i32;
        let max_gen = params
            .max_tokens
            .min(n_ctx.saturating_sub(tokens.len() as u32));
        let mut accumulated = String::new();
        let start = Instant::now();

        for i in 0..max_gen {
            if cancel.load(Ordering::Relaxed) {
                break;
            }

            // Check inference timeout
            if let Some(timeout) = self.timeout {
                if start.elapsed() > timeout {
                    tracing::warn!(
                        elapsed_secs = start.elapsed().as_secs(),
                        timeout_secs = timeout.as_secs(),
                        "LLM inference timed out (streaming channel)"
                    );
                    return Err(Error::Other(format!(
                        "LLM inference timed out after {}s",
                        timeout.as_secs()
                    )));
                }
            }

            let new_token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(new_token);

            if self.model.is_eog_token(new_token) {
                break;
            }

            let token_str = self
                .model
                .token_to_piece(new_token, &mut decoder, false, None)
                .unwrap_or_default();

            accumulated.push_str(&token_str);
            let should_stop = params
                .stop_sequences
                .iter()
                .any(|seq| accumulated.contains(seq.as_str()));

            if should_stop {
                break;
            }

            // Send token through channel; break if receiver dropped
            if tx.blocking_send(token_str).is_err() {
                break;
            }

            batch.clear();
            batch
                .add(new_token, prompt_len + i as i32, &[0], true)
                .map_err(|_| Error::Other("Failed to add token to batch".into()))?;

            ctx.decode(&mut batch)
                .map_err(|e| Error::Other(format!("Decode step failed: {e}")))?;
        }

        Ok(())
    }

    /// Get the model's context length.
    pub fn context_length(&self) -> u32 {
        self.context_length
    }
}

/// Validate that a model path exists and is readable.
pub fn validate_model_path(path: &Path) -> Result<()> {
    if !path.exists() {
        return Err(Error::config(format!(
            "Model file not found: {}",
            path.display()
        )));
    }
    if !path.is_file() {
        return Err(Error::config(format!(
            "Model path is not a file: {}",
            path.display()
        )));
    }
    // Check extension
    if path.extension().and_then(|e| e.to_str()) != Some("gguf") {
        tracing::warn!(
            path = %path.display(),
            "Model file does not have .gguf extension"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_llm_config_default() {
        let config = LlmConfig::default();
        assert_eq!(config.context_length, 16384);
        assert_eq!(config.gpu_layers, 99);
        assert_eq!(config.threads, 0);
        assert_eq!(config.timeout, Some(Duration::from_secs(120)));
    }

    #[test]
    fn test_generation_params_default() {
        let params = GenerationParams::default();
        assert_eq!(params.max_tokens, 2048);
        assert!((params.temperature - 0.7).abs() < f32::EPSILON);
    }

    // --- Per-call context sizing ---

    #[test]
    fn test_effective_context_len_short_prompt_is_tiny() {
        // An intent classification: ~40-token prompt, 8 output tokens. We should
        // allocate a few dozen tokens, NOT the full 16384 window.
        let n = effective_context_len(40, 8, 16384);
        assert_eq!(n, 40 + 8 + CONTEXT_SLACK_TOKENS);
        assert!(n < 100, "expected a tiny context, got {n}");
    }

    #[test]
    fn test_effective_context_len_clamped_to_max() {
        // Large generation request must never exceed the configured window.
        let n = effective_context_len(1000, 100_000, 16384);
        assert_eq!(n, 16384);
    }

    #[test]
    fn test_effective_context_len_at_least_prompt_plus_one() {
        // With zero generation tokens we still need room for the prompt itself.
        let n = effective_context_len(500, 0, 16384);
        assert!(n >= 501, "context must hold the prompt, got {n}");
    }

    #[test]
    fn test_effective_context_len_no_overflow_on_huge_inputs() {
        // Saturating arithmetic: pathological inputs clamp instead of wrapping.
        let n = effective_context_len(usize::MAX, u32::MAX, 16384);
        assert_eq!(n, 16384);
    }

    #[test]
    fn test_validate_model_path_missing() {
        let result = validate_model_path(&PathBuf::from("/nonexistent/model.gguf"));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[test]
    fn test_validate_model_path_directory() {
        let result = validate_model_path(&PathBuf::from("/tmp"));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not a file"));
    }

    #[test]
    fn test_validate_model_path_valid_gguf() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("model.gguf");
        std::fs::write(&model_path, b"fake gguf data").unwrap();
        let result = validate_model_path(&model_path);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_model_path_wrong_extension() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("model.bin");
        std::fs::write(&model_path, b"fake data").unwrap();
        // Should still succeed but emit a warning
        let result = validate_model_path(&model_path);
        assert!(result.is_ok());
    }

    #[test]
    fn test_llm_config_custom() {
        let config = LlmConfig {
            model_path: PathBuf::from("/some/path.gguf"),
            context_length: 4096,
            gpu_layers: 0,
            threads: 4,
            timeout: None,
        };
        assert_eq!(config.context_length, 4096);
        assert_eq!(config.gpu_layers, 0);
        assert_eq!(config.threads, 4);
        assert_eq!(config.timeout, None);
    }

    #[test]
    fn test_generation_params_custom() {
        let params = GenerationParams {
            max_tokens: 512,
            temperature: 0.5,
            top_p: 0.8,
            stop_sequences: vec!["<stop>".to_string()],
        };
        assert_eq!(params.max_tokens, 512);
        assert_eq!(params.stop_sequences.len(), 1);
    }
}
