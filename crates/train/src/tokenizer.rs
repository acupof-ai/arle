//! Thin wrapper around HF `tokenizers` for the training side.
//!
//! Training needs both encode + decode + chat-template assembly against
//! existing vocabs (the supported Qwen families share the `<|im_start|>` template).

use std::{fmt::Display, path::Path};

use tokenizers::Tokenizer;

use autograd::{AutogradError, Result};

pub struct ChatTokenizer {
    inner: Tokenizer,
}

impl ChatTokenizer {
    pub fn from_file(path: &Path) -> Result<Self> {
        let inner = Tokenizer::from_file(path).map_err(|e| tokenizer_error("load", e))?;
        Ok(Self { inner })
    }

    pub fn encode(&self, text: &str, add_special: bool) -> Result<Vec<u32>> {
        let enc = self
            .inner
            .encode(text, add_special)
            .map_err(|e| tokenizer_error("encode", e))?;
        Ok(enc.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32], skip_special: bool) -> Result<String> {
        self.inner
            .decode(ids, skip_special)
            .map_err(|e| tokenizer_error("decode", e))
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }
}

fn tokenizer_error(context: &str, err: impl Display) -> AutogradError {
    tokenizer_message(&format!("tokenizer {context} failed: {err}"))
}

fn tokenizer_message(message: &str) -> AutogradError {
    AutogradError::TapeInvariant(Box::leak(message.to_string().into_boxed_str()))
}
