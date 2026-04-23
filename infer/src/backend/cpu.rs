//! Development-oriented CPU backend.
//!
//! This backend exercises the same request, streaming, HTTP, and agent paths
//! as the GPU-backed runtimes without claiming production-grade local CPU
//! inference. It is intended for smoke tests and local validation on machines
//! without CUDA or Metal.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use serde_json::Value;

use crate::backend::{GenerateResult, InferenceBackend, StreamingInferenceBackend};
use crate::model_source::ResolvedModelSource;
use crate::sampler::SamplingParams;
use crate::tokenizer::Tokenizer;

const DEFAULT_COMPLETION_BUDGET: usize = 64;
const STREAM_CHUNK_CHARS: usize = 24;

pub struct CpuBackend {
    model_id: String,
    model_family: Option<String>,
    model_path: Option<PathBuf>,
    tokenizer: Option<Tokenizer>,
}

impl CpuBackend {
    pub fn new() -> Self {
        Self {
            model_id: "cpu-dev".to_string(),
            model_family: None,
            model_path: None,
            tokenizer: None,
        }
    }

    fn ensure_loaded(&self) -> Result<()> {
        if self.model_path.is_none() {
            return Err(anyhow!("CPU backend must be loaded before generate()"));
        }
        Ok(())
    }

    fn count_tokens(&self, text: &str) -> usize {
        self.tokenizer
            .as_ref()
            .and_then(|tokenizer| tokenizer.encode(text).ok().map(|ids| ids.len()))
            .unwrap_or_else(|| estimate_tokens(text))
    }

    fn build_response(&self, prompt: &str) -> String {
        let preview = prompt_preview(prompt);
        let family = self.model_family.as_deref().unwrap_or("unknown-family");

        format!(
            "CPU backend development response from {} ({family}). This path validates local request handling without GPU acceleration. Prompt preview: {}",
            self.model_id, preview
        )
    }

    fn generate_text(&self, prompt: &str, params: &SamplingParams) -> (String, String, usize) {
        let budget = params.max_new_tokens.unwrap_or(DEFAULT_COMPLETION_BUDGET);
        if budget == 0 {
            return (String::new(), "length".to_string(), 0);
        }

        let base = self.build_response(prompt);
        if let Some(tokenizer) = &self.tokenizer {
            if let Ok(ids) = tokenizer.encode(&base) {
                if ids.len() <= budget {
                    return (base, "stop".to_string(), ids.len());
                }
                let clipped_ids = &ids[..budget];
                let clipped = tokenizer
                    .decode(clipped_ids)
                    .unwrap_or_else(|_| fallback_generate_text(&base, budget).0);
                return (clipped, "length".to_string(), clipped_ids.len());
            }
        }

        fallback_generate_text(&base, budget)
    }
}

fn fallback_generate_text(base: &str, budget: usize) -> (String, String, usize) {
    if budget == 0 {
        return (String::new(), "length".to_string(), 0);
    }

    let words: Vec<&str> = base.split_whitespace().collect();
    if words.len() <= budget {
        return (base.to_string(), "stop".to_string(), words.len());
    }

    (words[..budget].join(" "), "length".to_string(), budget)
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceBackend for CpuBackend {
    fn load(&mut self, model_path: &Path) -> Result<()> {
        let source = ResolvedModelSource::resolve(&model_path.to_string_lossy())?;
        self.model_id = display_model_id(model_path);
        self.model_family = source
            .config_dir()
            .and_then(|dir| load_model_family(dir).ok())
            .or_else(|| {
                source
                    .gguf()
                    .and_then(|gguf| gguf.architecture())
                    .map(str::to_string)
            });
        self.tokenizer = source.load_tokenizer().ok();
        self.model_path = Some(source.model_root().to_path_buf());
        Ok(())
    }

    fn generate(&self, prompt: &str, params: &SamplingParams) -> Result<GenerateResult> {
        self.ensure_loaded()?;

        let prompt_tokens = self.count_tokens(prompt);
        let (text, finish_reason, completion_tokens) = self.generate_text(prompt, params);

        Ok(GenerateResult {
            text,
            prompt_tokens,
            completion_tokens,
            finish_reason,
            ttft_ms: 0.0,
            prompt_tps: 0.0,
            generation_tps: 0.0,
            total_time_ms: 0.0,
        })
    }

    fn name(&self) -> &'static str {
        "cpu"
    }
}

impl StreamingInferenceBackend for CpuBackend {
    fn generate_stream<F>(
        &self,
        prompt: &str,
        params: &SamplingParams,
        mut on_chunk: F,
    ) -> Result<GenerateResult>
    where
        F: FnMut(&str) -> Result<()>,
    {
        let generated = self.generate(prompt, params)?;
        for chunk in chunk_text(&generated.text, STREAM_CHUNK_CHARS) {
            match on_chunk(chunk) {
                Ok(()) => {}
                Err(err) if crate::backend::is_stream_stop_matched(&err) => return Ok(generated),
                Err(err) => return Err(err),
            }
        }
        Ok(generated)
    }
}

fn load_model_family(model_dir: &Path) -> Result<String> {
    let config_path = model_dir.join("config.json");
    let config = std::fs::read_to_string(&config_path)
        .with_context(|| format!("failed to read {}", config_path.display()))?;
    let parsed: Value = serde_json::from_str(&config)
        .with_context(|| format!("failed to parse {}", config_path.display()))?;

    if let Some(arch) = parsed
        .get("architectures")
        .and_then(Value::as_array)
        .and_then(|architectures| architectures.first())
        .and_then(Value::as_str)
    {
        return Ok(arch.to_string());
    }

    if let Some(model_type) = parsed.get("model_type").and_then(Value::as_str) {
        return Ok(model_type.to_string());
    }

    Ok("unknown-family".to_string())
}

fn display_model_id(model_source: &Path) -> String {
    if model_source.exists() {
        return model_source
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("cpu-dev")
            .to_string();
    }

    let raw = model_source.to_string_lossy();
    raw.rsplit(['/', '\\'])
        .find(|segment| !segment.is_empty())
        .unwrap_or("cpu-dev")
        .to_string()
}

fn prompt_preview(prompt: &str) -> String {
    let candidate = prompt
        .lines()
        .rev()
        .find(|line| !line.trim().is_empty())
        .unwrap_or(prompt)
        .trim();

    let mut preview: String = candidate.chars().take(96).collect();
    if candidate.chars().count() > 96 {
        preview.push_str("...");
    }
    if preview.is_empty() {
        "<empty>".to_string()
    } else {
        preview
    }
}

fn estimate_tokens(text: &str) -> usize {
    let count = text.split_whitespace().count();
    if count == 0 && !text.trim().is_empty() {
        1
    } else {
        count
    }
}

fn chunk_text(text: &str, max_chars: usize) -> Vec<&str> {
    if text.is_empty() {
        return Vec::new();
    }

    let mut chunks = Vec::new();
    let mut start = 0usize;
    while start < text.len() {
        let mut end = (start + max_chars).min(text.len());
        while end > start && !text.is_char_boundary(end) {
            end -= 1;
        }
        if end == start {
            end = text[start..]
                .char_indices()
                .nth(1)
                .map_or(text.len(), |(idx, _)| start + idx);
        }
        chunks.push(&text[start..end]);
        start = end;
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::{
        Tokenizer as HfTokenizer, models::wordlevel::WordLevel,
        pre_tokenizers::whitespace::Whitespace,
    };

    fn temp_model_dir() -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"architectures":["Qwen3ForCausalLM"],"model_type":"qwen3"}"#,
        )
        .expect("config");
        dir
    }

    fn temp_model_dir_with_tokenizer() -> tempfile::TempDir {
        let dir = temp_model_dir();
        let vocab = [
            ("<unk>", 0u32),
            ("CPU", 1),
            ("backend", 2),
            ("development", 3),
            ("response", 4),
            ("from", 5),
            ("tmp", 6),
            ("Qwen3ForCausalLM", 7),
            ("This", 8),
            ("path", 9),
            ("validates", 10),
            ("local", 11),
            ("request", 12),
            ("handling", 13),
            ("without", 14),
            ("GPU", 15),
            ("acceleration", 16),
            ("Prompt", 17),
            ("preview", 18),
            ("hello", 19),
            ("a", 20),
            ("smoke", 21),
            ("test", 22),
        ]
        .into_iter()
        .map(|(token, id)| (token.to_string(), id))
        .collect();
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("<unk>".to_string())
            .build()
            .expect("wordlevel");
        let mut tokenizer = HfTokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(Whitespace));
        tokenizer
            .save(dir.path().join("tokenizer.json"), false)
            .expect("save tokenizer");
        dir
    }

    #[test]
    fn cpu_backend_loads_without_tokenizer_json() {
        let dir = temp_model_dir();
        let mut backend = CpuBackend::new();
        backend.load(dir.path()).expect("load");

        assert_eq!(backend.name(), "cpu");
        assert_eq!(backend.model_family.as_deref(), Some("Qwen3ForCausalLM"));
    }

    #[test]
    fn cpu_backend_generates_deterministic_text() {
        let dir = temp_model_dir();
        let mut backend = CpuBackend::new();
        backend.load(dir.path()).expect("load");

        let generated = backend
            .generate("hello from a local smoke test", &SamplingParams::default())
            .expect("generate");

        assert!(generated.text.contains("CPU backend development response"));
        assert!(generated.prompt_tokens > 0);
        assert!(generated.completion_tokens > 0);
    }

    #[test]
    fn cpu_backend_respects_max_new_tokens_budget() {
        let dir = temp_model_dir();
        let mut backend = CpuBackend::new();
        backend.load(dir.path()).expect("load");

        let generated = backend
            .generate(
                "hello from a local smoke test",
                &SamplingParams {
                    max_new_tokens: Some(3),
                    ..Default::default()
                },
            )
            .expect("generate");

        assert_eq!(generated.finish_reason, "length");
        assert_eq!(generated.text.split_whitespace().count(), 3);
        assert_eq!(generated.completion_tokens, 3);
    }

    #[test]
    fn cpu_backend_respects_max_new_tokens_with_tokenizer_budget() {
        let dir = temp_model_dir_with_tokenizer();
        let mut backend = CpuBackend::new();
        backend.load(dir.path()).expect("load");

        let generated = backend
            .generate(
                "hello from a local smoke test",
                &SamplingParams {
                    max_new_tokens: Some(3),
                    ..Default::default()
                },
            )
            .expect("generate");

        assert_eq!(generated.finish_reason, "length");
        assert_eq!(generated.completion_tokens, 3);
    }

    fn temp_model_dir_with_unknown_heavy_tokenizer() -> tempfile::TempDir {
        let dir = temp_model_dir();
        let vocab = [
            ("[UNK]".to_string(), 0u32),
            ("hello".to_string(), 1u32),
            ("world".to_string(), 2u32),
            ("agent".to_string(), 3u32),
            ("<|im_start|>user\n".to_string(), 4u32),
            ("<|im_start|>assistant\n".to_string(), 5u32),
            ("<|im_start|>system\n".to_string(), 6u32),
            ("<|im_end|>".to_string(), 7u32),
        ]
        .into_iter()
        .collect();
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".to_string())
            .build()
            .expect("wordlevel");
        let mut tokenizer = HfTokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(Whitespace));
        tokenizer
            .save(dir.path().join("tokenizer.json"), false)
            .expect("save tokenizer");
        dir
    }

    #[test]
    fn cpu_backend_preserves_budget_even_when_decoded_text_reencodes_longer() {
        let dir = temp_model_dir_with_unknown_heavy_tokenizer();
        let mut backend = CpuBackend::new();
        backend.load(dir.path()).expect("load");

        let generated = backend
            .generate(
                "hello from a local smoke test",
                &SamplingParams {
                    max_new_tokens: Some(8),
                    ..Default::default()
                },
            )
            .expect("generate");

        assert_eq!(generated.finish_reason, "length");
        assert_eq!(generated.completion_tokens, 8);
        assert_eq!(
            generated.text,
            "[UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK]"
        );
        assert!(backend.count_tokens(&generated.text) > generated.completion_tokens);
    }

    #[test]
    fn cpu_backend_uses_repo_name_for_remote_model_id() {
        assert_eq!(display_model_id(Path::new("Qwen/Qwen3-0.6B")), "Qwen3-0.6B");
    }

    #[test]
    fn chunk_text_preserves_utf8_boundaries() {
        let chunks = chunk_text("hello 世界", 3);
        assert_eq!(chunks.concat(), "hello 世界");
    }
}
