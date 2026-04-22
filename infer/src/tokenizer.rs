use anyhow::{Result, anyhow};
use tokenizers::Tokenizer as HfTokenizer;
use tokenizers::tokenizer::{
    DecodeStream as HfDecodeStream, DecoderWrapper, ModelWrapper, NormalizerWrapper,
    PostProcessorWrapper, PreTokenizerWrapper,
};

#[allow(dead_code)]
type InnerDecodeStream<'a> = HfDecodeStream<
    'a,
    ModelWrapper,
    NormalizerWrapper,
    PreTokenizerWrapper,
    PostProcessorWrapper,
    DecoderWrapper,
>;

#[derive(Clone)]
pub struct Tokenizer {
    inner: HfTokenizer,
}

#[allow(dead_code)]
pub(crate) struct IncrementalDecoder<'a> {
    tokenizer: &'a Tokenizer,
    inner: InnerDecodeStream<'a>,
    token_ids: Vec<u32>,
    emitted_text: String,
}

impl Tokenizer {
    pub fn from_file(path: &str) -> Result<Self> {
        let tokenizer_path = format!("{}/tokenizer.json", path);
        let inner = HfTokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        Ok(Self { inner })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("Encode error: {}", e))?;
        Ok(encoding.get_ids().to_vec())
    }

    fn decode_strict(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids, true)
            .map_err(|e| anyhow::anyhow!("Decode error: {}", e))
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        match self.decode_strict(ids) {
            Ok(text) => Ok(text),
            Err(err) => {
                log::warn!(
                    "Tokenizer decode failed for {} token(s): {:#}. Falling back to lossy incremental decode.",
                    ids.len(),
                    err
                );
                let lossy = self.decode_lossy(ids);
                if lossy.is_empty() && !ids.is_empty() {
                    Err(err)
                } else {
                    Ok(lossy)
                }
            }
        }
    }

    #[allow(dead_code)]
    pub(crate) fn incremental_decoder(&self) -> IncrementalDecoder<'_> {
        IncrementalDecoder {
            tokenizer: self,
            inner: self.inner.decode_stream(true),
            token_ids: Vec::new(),
            emitted_text: String::new(),
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    fn decode_lossy(&self, ids: &[u32]) -> String {
        let mut decoder = self.incremental_decoder();
        let mut out = String::new();

        for &id in ids {
            match decoder.step(id) {
                Ok(Some(chunk)) => out.push_str(&chunk),
                Ok(None) => {}
                Err(err) => {
                    log::warn!(
                        "Skipping undecodable token {} during fallback decode: {:#}",
                        id,
                        err
                    );
                    out.push('\u{fffd}');
                    decoder = self.incremental_decoder();
                }
            }
        }

        if let Some(tail) = decoder.finish_lossy() {
            out.push_str(&tail);
        }

        out
    }
}

#[allow(dead_code)]
impl IncrementalDecoder<'_> {
    pub(crate) fn step(&mut self, token_id: u32) -> Result<Option<String>> {
        self.token_ids.push(token_id);
        let chunk = self
            .inner
            .step(token_id)
            .map_err(|e| anyhow!("Streaming decode error for token {}: {}", token_id, e))?;

        if let Some(ref text) = chunk {
            self.emitted_text.push_str(text);
        }

        Ok(chunk)
    }

    /// Text emitted so far by `step()` (and `finish()`). Used for stop-sequence checks.
    pub(crate) fn emitted_text(&self) -> &str {
        &self.emitted_text
    }

    pub(crate) fn finish(&mut self) -> Result<Option<String>> {
        let decoded = self.tokenizer.decode_strict(&self.token_ids)?;
        let suffix = decoded.strip_prefix(&self.emitted_text).ok_or_else(|| {
            anyhow!(
                "Streaming decoder state mismatch: emitted text is not a prefix of final decode"
            )
        })?;

        if suffix.is_empty() {
            Ok(None)
        } else {
            self.emitted_text.push_str(suffix);
            Ok(Some(suffix.to_string()))
        }
    }

    fn finish_lossy(&mut self) -> Option<String> {
        let decoded = self.tokenizer.decode_strict(&self.token_ids).ok()?;
        let suffix = decoded.strip_prefix(&self.emitted_text)?;
        if suffix.is_empty() {
            None
        } else {
            self.emitted_text.push_str(suffix);
            Some(suffix.to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/Qwen3-4B");

    #[test]
    #[ignore = "requires model weights at models/Qwen3-4B"]
    fn test_load_tokenizer() {
        let tokenizer = Tokenizer::from_file(MODEL_PATH).unwrap();
        assert!(tokenizer.vocab_size() > 0);
    }

    #[test]
    #[ignore = "requires model weights at models/Qwen3-4B"]
    fn test_encode_decode() {
        let tokenizer = Tokenizer::from_file(MODEL_PATH).unwrap();

        let text = "Hello, world!";
        let ids = tokenizer.encode(text).unwrap();
        // ids: [9707, 11, 1879, 0]
        assert_eq!(ids, vec![9707, 11, 1879, 0]);

        let decoded = tokenizer.decode(&ids).unwrap();

        assert_eq!(decoded, text);
    }

    #[test]
    #[ignore = "requires model weights at models/Qwen3-4B"]
    fn test_chinese() {
        let tokenizer = Tokenizer::from_file(MODEL_PATH).unwrap();

        let text = "你好，世界！";
        let ids = tokenizer.encode(text).unwrap();

        let decoded = tokenizer.decode(&ids).unwrap();

        assert_eq!(decoded, text);
    }

    #[test]
    #[ignore = "requires model weights at models/Qwen3-4B"]
    fn test_incremental_decode_matches_full_decode_for_chinese() {
        let tokenizer = Tokenizer::from_file(MODEL_PATH).unwrap();
        let text = "北京，简称“京”，是中国的首都。";
        let ids = tokenizer.encode(text).unwrap();

        let mut decoder = tokenizer.incremental_decoder();
        let mut streamed = String::new();
        for id in ids.iter().copied() {
            if let Some(chunk) = decoder.step(id).unwrap() {
                streamed.push_str(&chunk);
            }
        }
        if let Some(tail) = decoder.finish().unwrap() {
            streamed.push_str(&tail);
        }

        assert_eq!(streamed, tokenizer.decode(&ids).unwrap());
    }

    // Attributes HTTP-server per-token cost vs the once-at-end batch decode
    // used by `metal_bench`. See docs/experience/wins/2026-04-18-*.
    const QWEN35_TOKENIZER_DIR: &str = "/path/to/.cache/huggingface/hub/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3";

    #[test]
    #[ignore = "micro-bench; requires Qwen3.5-4B tokenizer in HF cache"]
    fn bench_incremental_decoder_step() {
        use std::time::Instant;

        let tokenizer = Tokenizer::from_file(QWEN35_TOKENIZER_DIR).unwrap();

        let prompt = "The quick brown fox jumps over the lazy dog. Large language \
                      models excel at next-token prediction, and the per-step tokenizer \
                      decoding cost can dominate when the GPU forward is already optimized. \
                      We measure it here to see how bad the HTTP per-token decode is.";
        let ids = tokenizer.encode(prompt).unwrap();
        assert!(ids.len() >= 32, "got {} ids", ids.len());

        let run = |label: &str, ids: &[u32]| {
            let mut dec = tokenizer.incremental_decoder();
            let t0 = Instant::now();
            for &id in ids {
                let _ = dec.step(id).unwrap();
            }
            let dt_us = t0.elapsed().as_micros();
            let n = ids.len();
            eprintln!(
                "{label:8}: {n:3} tokens in {dt_us:6} us → {:6.1} us/tok  ({:5.2} ms per 128-tok gen)",
                dt_us as f64 / n as f64,
                (dt_us as f64 / n as f64) * 128.0 / 1000.0,
            );
        };

        run("warm 1", &ids);
        run("warm 2", &ids);
        run("TIMED", &ids);
        run("TIMED", &ids);
        run("TIMED", &ids);
        run("TIMED", &ids);
        run("TIMED", &ids);

        let t0 = Instant::now();
        let _s = tokenizer.decode(&ids).unwrap();
        let dt_us = t0.elapsed().as_micros();
        eprintln!(
            "batch  : {:3} tokens in {:6} us total ({:.1} us/tok, {:.2} ms per 128-tok gen)",
            ids.len(),
            dt_us,
            dt_us as f64 / ids.len() as f64,
            (dt_us as f64 / ids.len() as f64) * 128.0 / 1000.0,
        );
    }
}
