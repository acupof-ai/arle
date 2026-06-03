//! Tokenizer and chat-template adapter for the OpenAI v1 facade (COLD).
//!
//! Loads `tokenizer.json` directly and reads `tokenizer_config.json` to detect a
//! model-provided chat template. R5 keeps rendering minimal and Qwen-compatible
//! rather than carrying the legacy prompt stack.

use std::path::Path;

use anyhow::{Context, Result, anyhow, ensure};
use tokenizers::Tokenizer;

use crate::schema::ChatMessage;

/// Tokenizer and chat-template adapter for the OpenAI v1 facade.
///
/// The tokenizer is loaded directly from `tokenizer.json`. `tokenizer_config`
/// is read to detect a model-provided chat template; R5 keeps rendering minimal
/// and Qwen-compatible rather than carrying the legacy prompt stack.
#[derive(Clone)]
pub struct OpenAiTokenizer {
    inner: Tokenizer,
    chat_template: Option<String>,
}

impl OpenAiTokenizer {
    /// Load `tokenizer.json` and optional `tokenizer_config.json` from a model dir.
    pub fn from_model_dir(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let tokenizer_path = model_dir.join("tokenizer.json");
        let inner = Tokenizer::from_file(&tokenizer_path)
            .map_err(|err| anyhow!("load tokenizer {} failed: {err}", tokenizer_path.display()))?;
        let chat_template = read_chat_template(model_dir)?;
        Ok(Self {
            inner,
            chat_template,
        })
    }

    /// Encode text into token ids without adding special tokens.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|err| anyhow!("tokenize prompt failed: {err}"))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode token ids into text, skipping special tokens.
    pub fn decode(&self, token_ids: &[u32]) -> Result<String> {
        self.inner
            .decode(token_ids, true)
            .map_err(|err| anyhow!("decode generated tokens failed: {err}"))
    }

    /// Render OpenAI chat messages into the Qwen ChatML prompt form.
    pub fn render_chat(&self, messages: &[ChatMessage]) -> Result<String> {
        ensure!(
            !messages.is_empty(),
            "messages must contain at least one message"
        );
        if let Some(template) = &self.chat_template
            && !template.contains("<|im_start|>")
        {
            log::warn!(
                "tokenizer_config.json has an unknown chat_template shape; using Qwen ChatML fallback"
            );
        }

        let mut out = String::new();
        for message in messages {
            let role = message.role.trim();
            ensure!(!role.is_empty(), "message role must not be empty");
            out.push_str("<|im_start|>");
            out.push_str(role);
            out.push('\n');
            if let Some(content) = message.content.as_deref() {
                out.push_str(content);
            }
            out.push_str("<|im_end|>\n");
        }
        out.push_str("<|im_start|>assistant\n");
        Ok(out)
    }
}

fn read_chat_template(model_dir: &Path) -> Result<Option<String>> {
    let path = model_dir.join("tokenizer_config.json");
    if !path.exists() {
        return Ok(None);
    }
    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("read tokenizer config {}", path.display()))?;
    let value: serde_json::Value = serde_json::from_str(&raw)
        .with_context(|| format!("parse tokenizer config {}", path.display()))?;
    Ok(value
        .get("chat_template")
        .and_then(|template| template.as_str().map(str::to_owned)))
}
