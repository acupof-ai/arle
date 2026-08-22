//! Structured output: OpenAI `response_format` → an `infer_core::GrammarHook`.
//!
//! The engine never sees xgrammar. It holds a callback, calls it with each
//! committed token, and puts the returned bitmask on the next step's sampling
//! params; the matcher and its lifetime live here, next to the tokenizer that
//! defines the vocabulary it was compiled against.

use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, bail};
use infer_core::GrammarHook;
use serde::{Deserialize, Serialize};
use xgrammar_sys::{
    CompiledGrammar, CompilerConfig, GrammarCompiler, GrammarMatcher, MatcherConfig,
};

use crate::tokenizer::OpenAiTokenizer;

/// OpenAI `response_format`. `json_object` is a bare "must be valid JSON";
/// `json_schema` carries the schema the output must validate against.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseFormat {
    Text,
    JsonObject,
    JsonSchema { json_schema: JsonSchemaSpec },
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct JsonSchemaSpec {
    #[serde(default)]
    pub name: String,
    pub schema: serde_json::Value,
}

impl ResponseFormat {
    /// Cache key. `None` = no constraint.
    fn key(&self) -> Option<String> {
        match self {
            Self::Text => None,
            Self::JsonObject => Some("{}".to_string()),
            Self::JsonSchema { json_schema } => Some(json_schema.schema.to_string()),
        }
    }
}

/// Compiles and caches grammars for one model's vocabulary.
pub struct GrammarCache {
    compiler: Mutex<GrammarCompiler>,
    vocab_size: usize,
    compiled: Mutex<Vec<(String, Arc<CompiledGrammar>)>>,
}

impl GrammarCache {
    pub fn new(tokenizer: &OpenAiTokenizer) -> Result<Self> {
        let vocab = tokenizer.vocab_by_id();
        let compiler = GrammarCompiler::new(&vocab, CompilerConfig::default())
            .context("build xgrammar compiler from the model vocabulary")?;
        Ok(Self {
            vocab_size: compiler.vocab_size(),
            compiler: Mutex::new(compiler),
            compiled: Mutex::new(Vec::new()),
        })
    }

    /// `Ok(None)` means the format imposes no constraint, so the request
    /// decodes unmodified.
    pub fn hook(&self, format: &ResponseFormat) -> Result<Option<GrammarHook>> {
        let Some(key) = format.key() else {
            return Ok(None);
        };
        let grammar = self.compile(&key)?;
        let matcher = GrammarMatcher::new(grammar, MatcherConfig::default())
            .context("create xgrammar matcher")?;
        Ok(Some(new_hook(matcher, self.vocab_size)))
    }

    fn compile(&self, key: &str) -> Result<Arc<CompiledGrammar>> {
        let mut cache = self.compiled.lock().expect("grammar cache poisoned");
        if let Some((_, g)) = cache.iter().find(|(k, _)| k == key) {
            return Ok(g.clone());
        }
        let mut compiler = self.compiler.lock().expect("grammar compiler poisoned");
        let grammar = Arc::new(
            compiler
                .compile_json_schema(key, false)
                .with_context(|| format!("compile JSON schema: {key}"))?,
        );
        // Bounded so a schema-per-request client cannot grow this without limit;
        // compilation is the expensive step, matcher creation is not.
        if cache.len() >= 64 {
            cache.remove(0);
        }
        cache.push((key.to_string(), grammar.clone()));
        Ok(grammar)
    }
}

fn new_hook(matcher: GrammarMatcher, vocab_size: usize) -> GrammarHook {
    let words = xgrammar_sys::bitmask_size(vocab_size).unwrap_or(0);
    let state = Mutex::new((matcher, vec![0u32; words]));
    GrammarHook(Arc::new(move |token: Option<u32>| {
        let mut guard = state.lock().ok()?;
        let (matcher, buf) = &mut *guard;
        if let Some(t) = token {
            // A token the grammar rejects can only come from a path that did not
            // consult the mask (a rejected speculative draft is never committed).
            // Drop the constraint rather than mask against a desynced matcher.
            if !matcher.accept_token(t).ok()? {
                return None;
            }
        }
        if matcher.is_terminated() {
            return None;
        }
        matcher.fill_next_token_bitmask(buf).ok()?;
        Some(Arc::from(buf.as_slice()))
    }))
}

/// Resolve a request's `response_format` against the model's cache. A format
/// that constrains but has no backend is an error, not silent free-form text.
pub fn resolve(
    cache: Option<&GrammarCache>,
    format: Option<ResponseFormat>,
) -> Result<Option<GrammarHook>> {
    let Some(format) = format else {
        return Ok(None);
    };
    match cache {
        Some(cache) => cache.hook(&format),
        None if format.key().is_none() => Ok(None),
        None => bail!("response_format needs the grammar backend: rebuild with --features grammar"),
    }
}
