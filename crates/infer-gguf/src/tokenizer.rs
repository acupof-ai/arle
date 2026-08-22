//! Build a `tokenizers::Tokenizer` directly from a GGUF's embedded vocab.
//!
//! Some checkpoints (e.g. the custom-vocab Qwen3.6-27B) ship their tokenizer
//! *only* inside the GGUF metadata — there is no sibling `tokenizer.json` to
//! load. The relevant keys, written by llama.cpp's GGUF converter, are:
//!   - `tokenizer.ggml.model`      — vocab family; `"gpt2"` = byte-level BPE.
//!   - `tokenizer.ggml.tokens`     — `[String; vocab]`, id = array index. The
//!     strings are already GPT-2 *byte-level encoded* (space → `Ġ`, etc.).
//!   - `tokenizer.ggml.merges`     — `["a b", …]` ordered BPE merge rules.
//!   - `tokenizer.ggml.token_type` — `[i32; vocab]`; 1=NORMAL, 2=UNKNOWN,
//!     3=CONTROL, 4=USER_DEFINED, 5=UNUSED/BYTE. Types 3/4 are the special
//!     tokens (`<|im_start|>`, `<think>`, …) that must encode atomically.
//!   - `tokenizer.ggml.{bos,eos,padding}_token_id`, `tokenizer.ggml.add_bos_token`.
//!
//! We reconstruct a GPT-2 byte-level BPE [`Tokenizer`]: a `BPE` model from the
//! (token→id) vocab + merges, a `ByteLevel` pre-tokenizer and decoder, with the
//! control/user-defined tokens registered as special added tokens. This mirrors
//! how llama.cpp's `gpt2` tokenizer and HF's Qwen `tokenizer.json` behave.

use anyhow::{Context, Result, anyhow, bail};
use tokenizers::{
    AddedToken, Tokenizer,
    decoders::byte_level::ByteLevel as ByteLevelDecoder,
    models::bpe::{BPE, Vocab},
    pre_tokenizers::byte_level::ByteLevel as ByteLevelPre,
    processors::byte_level::ByteLevel as ByteLevelPost,
};

use crate::gguf::{GgufFile, GgufValue};

/// Tokenizer built from a GGUF's embedded vocab, plus the special-token ids
/// the GGUF declares (prompt assembly / stop conditions).
pub struct GgufTokenizer {
    inner: Tokenizer,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub add_bos_token: bool,
}

impl GgufTokenizer {
    /// Supports `tokenizer.ggml.model == "gpt2"` (GPT-2 byte-level BPE;
    /// covers Qwen / Qwen3.5 / Qwen3.6 vocabs).
    pub fn from_gguf(g: &GgufFile) -> Result<Self> {
        let model = g
            .get_str("tokenizer.ggml.model")
            .context("GGUF missing tokenizer.ggml.model")?;
        if model != "gpt2" {
            bail!("GGUF tokenizer model {model:?} unsupported (only gpt2 byte-level BPE is wired)");
        }

        let tokens = str_array(g, "tokenizer.ggml.tokens")?;
        let merges_raw = str_array(g, "tokenizer.ggml.merges")?;
        let token_types = i32_array(g, "tokenizer.ggml.token_type")?;
        if token_types.len() != tokens.len() {
            bail!(
                "token_type len {} != tokens len {}",
                token_types.len(),
                tokens.len()
            );
        }

        // token -> id, in id order (array index is the id).
        let vocab: Vocab = tokens
            .iter()
            .enumerate()
            .map(|(id, tok)| (tok.clone(), u32::try_from(id).expect("vocab id fits u32")))
            .collect();

        // GGUF stores each merge as "left right"; HF BPE wants the pair split
        // on the FIRST space (token strings never contain a raw space — spaces
        // are byte-level encoded as `Ġ`).
        let merges: Vec<(String, String)> = merges_raw
            .iter()
            .map(|m| {
                m.split_once(' ')
                    .map(|(a, b)| (a.to_string(), b.to_string()))
                    .ok_or_else(|| anyhow!("malformed merge rule {m:?} (no space)"))
            })
            .collect::<Result<_>>()?;

        let bpe = BPE::builder()
            .vocab_and_merges(vocab, merges)
            // The vocab is exhaustive byte-level; there is no <unk>. Leaving
            // unk unset means an unknown piece errors, but byte-level pre-tok
            // guarantees every input byte maps to a known base token.
            .ignore_merges(true)
            .build()
            .map_err(|e| anyhow!("build BPE from GGUF vocab: {e}"))?;

        let mut tokenizer = Tokenizer::new(bpe);
        // GPT-2 byte-level, Qwen-style: no prefix space, trim offsets;
        // `add_bos_token` is false for this vocab, so BOS is never prepended
        // automatically.
        tokenizer.with_pre_tokenizer(Some(ByteLevelPre::new(
            /* add_prefix_space */ false, /* trim_offsets */ true, true,
        )));
        tokenizer.with_decoder(Some(ByteLevelDecoder::default()));
        tokenizer.with_post_processor(Some(ByteLevelPost::new(false, true, true)));

        // Register CONTROL (3) and USER_DEFINED (4) tokens as special so they
        // encode as single ids and are skipped when decoding with skip_special.
        let specials: Vec<AddedToken> = token_types
            .iter()
            .enumerate()
            .filter(|&(_, &t)| t == 3 || t == 4)
            .map(|(id, _)| AddedToken::from(tokens[id].clone(), true).special(true))
            .collect();
        if !specials.is_empty() {
            tokenizer.add_special_tokens(&specials);
        }

        Ok(Self {
            inner: tokenizer,
            bos_token_id: g.get_u64("tokenizer.ggml.bos_token_id").map(|v| v as u32),
            eos_token_id: g.get_u64("tokenizer.ggml.eos_token_id").map(|v| v as u32),
            add_bos_token: g.get_bool("tokenizer.ggml.add_bos_token").unwrap_or(false),
        })
    }

    /// `add_special` matches registered special tokens; it does not auto-insert
    /// BOS (this vocab sets `add_bos_token=false`).
    pub fn encode(&self, text: &str, add_special: bool) -> Result<Vec<u32>> {
        let enc = self
            .inner
            .encode(text, add_special)
            .map_err(|e| anyhow!("encode failed: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32], skip_special: bool) -> Result<String> {
        self.inner
            .decode(ids, skip_special)
            .map_err(|e| anyhow!("decode failed: {e}"))
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }
}

fn str_array(g: &GgufFile, key: &str) -> Result<Vec<String>> {
    match g.get(key) {
        Some(GgufValue::Array(items)) => items
            .iter()
            .map(|v| {
                v.as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| anyhow!("{key}: array element is not a string"))
            })
            .collect(),
        Some(_) => bail!("{key} is not an array"),
        None => bail!("GGUF missing {key}"),
    }
}

fn i32_array(g: &GgufFile, key: &str) -> Result<Vec<i32>> {
    match g.get(key) {
        Some(GgufValue::Array(items)) => items
            .iter()
            .map(|v| match v {
                GgufValue::I32(n) => Ok(*n),
                GgufValue::U32(n) => Ok(*n as i32),
                other => other
                    .as_u64()
                    .map(|n| n as i32)
                    .ok_or_else(|| anyhow!("{key}: element {other:?} is not an int")),
            })
            .collect(),
        Some(_) => bail!("{key} is not an array"),
        None => bail!("GGUF missing {key}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trips ASCII + Chinese through the GGUF-embedded vocab. Skips
    /// cleanly when the on-box 27B GGUF is absent.
    #[test]
    #[ignore = "needs the on-box Qwen3.6-27B GGUF"]
    fn gguf_tokenizer_roundtrips() {
        let Ok(model) = std::env::var("INFER_GGUF_TEST_MODEL") else {
            eprintln!("skip: set INFER_GGUF_TEST_MODEL to the on-box Qwen3.6-27B GGUF path");
            return;
        };
        let path = std::path::Path::new(&model);
        if !path.exists() {
            eprintln!("skip: {} not present", path.display());
            return;
        }
        let g = GgufFile::open(path).expect("open GGUF");
        let tok = GgufTokenizer::from_gguf(&g).expect("build tokenizer from GGUF");
        eprintln!(
            "vocab_size={}, bos={:?}, eos={:?}, add_bos={}",
            tok.vocab_size(),
            tok.bos_token_id,
            tok.eos_token_id,
            tok.add_bos_token
        );
        assert!(tok.vocab_size() > 200_000, "vocab suspiciously small");

        for text in [
            "Hello world",
            "The capital of France is",
            "请用一句话介绍你自己。",
        ] {
            let ids = tok.encode(text, false).expect("encode");
            assert!(!ids.is_empty(), "encode produced no ids for {text:?}");
            let back = tok.decode(&ids, true).expect("decode");
            eprintln!("{text:?} -> {ids:?} -> {back:?}");
            assert_eq!(back, text, "round-trip mismatch for {text:?}");
        }

        // Special tokens must encode atomically.
        let im_start = tok.encode("<|im_start|>", true).expect("encode special");
        eprintln!("<|im_start|> -> {im_start:?}");
        assert_eq!(im_start.len(), 1, "control token should be a single id");
    }
}
