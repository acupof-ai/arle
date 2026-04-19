//! Thin wrapper around HF `tokenizers` for the training side.
//!
//! Mirrors the split already in `infer/` (which only ever decodes today).
//! Training needs both encode + decode + chat-template assembly against
//! pretrained vocabs (Qwen3 family, GLM4 share the `<|im_start|>` template).

use std::path::Path;

use tokenizers::Tokenizer;

use autograd::{AutogradError, Result};

pub struct TrainTokenizer {
    inner: Tokenizer,
}

#[derive(Debug, Clone)]
pub struct ChatMsg<'a> {
    pub role: &'a str,
    pub content: &'a str,
}

impl TrainTokenizer {
    pub fn from_file(path: &Path) -> Result<Self> {
        let inner = Tokenizer::from_file(path).map_err(|e| {
            AutogradError::TapeInvariant(Box::leak(
                format!("tokenizer load failed: {e}").into_boxed_str(),
            ))
        })?;
        Ok(Self { inner })
    }

    pub fn encode(&self, text: &str, add_special: bool) -> Result<Vec<u32>> {
        let enc = self.inner.encode(text, add_special).map_err(|e| {
            AutogradError::TapeInvariant(Box::leak(
                format!("tokenizer encode failed: {e}").into_boxed_str(),
            ))
        })?;
        Ok(enc.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32], skip_special: bool) -> Result<String> {
        self.inner.decode(ids, skip_special).map_err(|e| {
            AutogradError::TapeInvariant(Box::leak(
                format!("tokenizer decode failed: {e}").into_boxed_str(),
            ))
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Qwen3 / Qwen3.5 / GLM4 chat template:
    ///
    /// ```text
    /// <|im_start|>role\ncontent<|im_end|>\n
    /// ```
    ///
    /// Appended with `<|im_start|>assistant\n` when `add_generation_prompt`
    /// so the model knows it's the assistant's turn.
    pub fn render_chat(msgs: &[ChatMsg<'_>], add_generation_prompt: bool) -> String {
        let mut out = String::new();
        for m in msgs {
            out.push_str("<|im_start|>");
            out.push_str(m.role);
            out.push('\n');
            out.push_str(m.content);
            out.push_str("<|im_end|>\n");
        }
        if add_generation_prompt {
            out.push_str("<|im_start|>assistant\n");
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Only runs when a real HF model directory is pointed at via
    /// `INFER_TEST_MODEL_PATH`. Mirrors the infer-side convention.
    #[test]
    fn encode_decode_roundtrip() {
        let Ok(model_dir) = std::env::var("INFER_TEST_MODEL_PATH") else {
            eprintln!("INFER_TEST_MODEL_PATH unset; skipping tokenizer roundtrip");
            return;
        };
        let path = PathBuf::from(model_dir).join("tokenizer.json");
        if !path.exists() {
            eprintln!("no tokenizer.json at {}; skipping", path.display());
            return;
        }
        let tok = TrainTokenizer::from_file(&path).expect("load tokenizer");
        assert!(tok.vocab_size() > 1000, "vocab suspiciously small");

        let ids = tok.encode("hello world", false).expect("encode");
        assert!(!ids.is_empty());
        let text = tok.decode(&ids, true).expect("decode");
        // Tokenizers sometimes add/remove leading whitespace; trim and compare.
        assert_eq!(text.trim(), "hello world");
    }

    #[test]
    fn chat_template_shape() {
        let msgs = [
            ChatMsg {
                role: "user",
                content: "hi",
            },
            ChatMsg {
                role: "assistant",
                content: "hello",
            },
        ];
        let rendered = TrainTokenizer::render_chat(&msgs, true);
        assert!(rendered.contains("<|im_start|>user\nhi<|im_end|>"));
        assert!(rendered.contains("<|im_start|>assistant\nhello<|im_end|>"));
        assert!(rendered.ends_with("<|im_start|>assistant\n"));
    }
}
