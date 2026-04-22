//! Thin wrapper around HF `tokenizers` for the training side.
//!
//! Mirrors the split already in `infer/` (which only ever decodes today).
//! Training needs both encode + decode + chat-template assembly against
//! pretrained vocabs (Qwen3 family, GLM4 share the `<|im_start|>` template).

use std::{collections::HashSet, fmt::Display, path::Path};

use tokenizers::{
    AddedToken, Tokenizer, models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace,
};

use autograd::{AutogradError, Result};

const CHATML_SPECIAL_TOKENS: [&str; 4] = [
    "<|im_start|>user\n",
    "<|im_start|>assistant\n",
    "<|im_start|>system\n",
    "<|im_end|>",
];
const UNK_TOKEN: &str = "[UNK]";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedSpecialToken {
    pub id: u32,
    pub token: String,
}

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

    /// Encode and return (ids, byte-offsets) per token. Needed when we must
    /// locate a sub-string's tokens after encoding the whole input as one
    /// string (e.g. assistant-body span inside a full chat turn — piecewise
    /// re-encoding isn't stable under BPE merges at boundaries).
    #[allow(clippy::type_complexity)]
    pub fn encode_with_offsets(
        &self,
        text: &str,
        add_special: bool,
    ) -> Result<(Vec<u32>, Vec<(usize, usize)>)> {
        let enc = self
            .inner
            .encode(text, add_special)
            .map_err(|e| tokenizer_error("encode", e))?;
        Ok((enc.get_ids().to_vec(), enc.get_offsets().to_vec()))
    }

    pub fn decode(&self, ids: &[u32], skip_special: bool) -> Result<String> {
        self.inner
            .decode(ids, skip_special)
            .map_err(|e| tokenizer_error("decode", e))
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }

    pub fn resolve_special_token(
        &self,
        name: &str,
        explicit_id: Option<u32>,
        explicit_token: Option<&str>,
        inferred_tokens: &[&str],
    ) -> Result<Option<ResolvedSpecialToken>> {
        if let Some(token) = explicit_token {
            let id = self.token_to_id(token).ok_or_else(|| {
                tokenizer_message(&format!("tokenizer is missing {name} token {token:?}"))
            })?;
            if let Some(explicit_id) = explicit_id {
                if explicit_id != id {
                    return Err(tokenizer_message(&format!(
                        "{name} token/id mismatch: token {token:?} maps to {id}, not {explicit_id}"
                    )));
                }
            }
            return Ok(Some(ResolvedSpecialToken {
                id,
                token: token.to_string(),
            }));
        }

        if let Some(explicit_id) = explicit_id {
            let token = self.id_to_token(explicit_id).ok_or_else(|| {
                tokenizer_message(&format!(
                    "tokenizer is missing {name} token id {explicit_id}"
                ))
            })?;
            return Ok(Some(ResolvedSpecialToken {
                id: explicit_id,
                token,
            }));
        }

        Ok(inferred_tokens.iter().find_map(|token| {
            self.token_to_id(token).map(|id| ResolvedSpecialToken {
                id,
                token: (*token).to_string(),
            })
        }))
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

pub fn write_chatml_wordlevel_tokenizer(
    path: &Path,
    tokens: impl IntoIterator<Item = impl Into<String>>,
) -> Result<()> {
    write_wordlevel_tokenizer(path, tokens, CHATML_SPECIAL_TOKENS)
}

fn tokenizer_error(context: &str, err: impl Display) -> AutogradError {
    tokenizer_message(&format!("tokenizer {context} failed: {err}"))
}

fn tokenizer_message(message: &str) -> AutogradError {
    AutogradError::TapeInvariant(Box::leak(message.to_string().into_boxed_str()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tempfile::tempdir;

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
        let tok = ChatTokenizer::from_file(&path).expect("load tokenizer");
        assert!(tok.vocab_size() > 1000, "vocab suspiciously small");

        let ids = tok.encode("hello world", false).expect("encode");
        assert!(!ids.is_empty());
        let text = tok.decode(&ids, true).expect("decode");
        // Tokenizers sometimes add/remove leading whitespace; trim and compare.
        assert_eq!(text.trim(), "hello world");
    }

    #[test]
    fn write_chatml_wordlevel_tokenizer_roundtrips_words() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("tokenizer.json");
        write_chatml_wordlevel_tokenizer(&path, ["hello", "world", "agent"]).expect("write");
        let tok = ChatTokenizer::from_file(&path).expect("load tokenizer");
        let ids = tok.encode("hello world", false).expect("encode");
        assert_eq!(
            tok.decode(&ids, true).expect("decode").trim(),
            "hello world"
        );
    }

    #[test]
    fn resolve_special_token_prefers_explicit_token_and_checks_id_match() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("tokenizer.json");
        write_wordlevel_tokenizer(&path, ["hello"], ["<eos>", "<bos>"]).expect("write");
        let tok = ChatTokenizer::from_file(&path).expect("load tokenizer");
        let eos_id = tok.token_to_id("<eos>").expect("eos id");
        let resolved = tok
            .resolve_special_token("eos", Some(eos_id), Some("<eos>"), &["</s>"])
            .expect("resolve")
            .expect("present");
        assert_eq!(resolved.id, eos_id);
        assert_eq!(resolved.token, "<eos>");

        let err = tok
            .resolve_special_token("eos", Some(eos_id + 1), Some("<eos>"), &["</s>"])
            .expect_err("mismatch should fail");
        assert!(
            err.to_string().contains("mismatch"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn resolve_special_token_can_infer_common_fallback() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("tokenizer.json");
        write_wordlevel_tokenizer(&path, ["hello"], ["<|endoftext|>"]).expect("write");
        let tok = ChatTokenizer::from_file(&path).expect("load tokenizer");
        let resolved = tok
            .resolve_special_token("bos", None, None, &["<s>", "<|endoftext|>"])
            .expect("resolve")
            .expect("present");
        assert_eq!(resolved.token, "<|endoftext|>");
    }
}
