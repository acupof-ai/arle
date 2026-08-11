//! Thin wrapper around HF `tokenizers` for the training side.
//!
//! Mirrors the split already in `infer/` (which only ever decodes today).
//! Training needs both encode + decode + chat-template assembly against
//! existing vocabs (the supported Qwen families share the `<|im_start|>` template).

use std::{collections::HashSet, fmt::Display, path::Path};

use tokenizers::{
    AddedToken, Tokenizer, models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace,
};

use autograd::{AutogradError, Result};

const UNK_TOKEN: &str = "[UNK]";

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

pub fn write_wordlevel_tokenizer(
    path: &Path,
    tokens: impl IntoIterator<Item = impl Into<String>>,
    special_tokens: impl IntoIterator<Item = impl Into<String>>,
) -> Result<()> {
    let special_tokens = special_tokens
        .into_iter()
        .map(Into::into)
        .collect::<Vec<String>>();
    let mut seen = HashSet::from([UNK_TOKEN.to_string()]);
    let mut vocab = vec![(UNK_TOKEN.to_string(), 0u32)];
    for token in tokens
        .into_iter()
        .map(Into::into)
        .chain(special_tokens.iter().cloned())
    {
        if !seen.insert(token.clone()) {
            continue;
        }
        let next_id = u32::try_from(vocab.len())
            .map_err(|_| tokenizer_message("tokenizer vocab length exceeded u32::MAX"))?;
        vocab.push((token, next_id));
    }

    let model = WordLevel::builder()
        .vocab(vocab.into_iter().collect())
        .unk_token(UNK_TOKEN.into())
        .build()
        .map_err(|e| tokenizer_error("build wordlevel", e))?;
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(Whitespace));
    let special_tokens = special_tokens
        .into_iter()
        .map(|token| AddedToken::from(token, true).special(true))
        .collect::<Vec<_>>();
    tokenizer.add_special_tokens(&special_tokens);
    tokenizer
        .save(path, false)
        .map_err(|e| tokenizer_error("save", e))?;
    Ok(())
}

fn tokenizer_error(context: &str, err: impl Display) -> AutogradError {
    tokenizer_message(&format!("tokenizer {context} failed: {err}"))
}

fn tokenizer_message(message: &str) -> AutogradError {
    AutogradError::TapeInvariant(Box::leak(message.to_string().into_boxed_str()))
}
