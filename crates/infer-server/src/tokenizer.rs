//! Tokenizer and chat-template adapter for the OpenAI v1 facade (COLD).
//!
//! Chat rendering follows the HF-ecosystem contract so that **new model
//! onboarding is zero-code**: the checkpoint's own `chat_template`
//! (`tokenizer_config.json`, Jinja) is the first truth and is evaluated with
//! minijinja — the same mechanism HF/vLLM/TGI use. Builtin renderers exist
//! only for the genuine gaps:
//!   - DeepSeek-V4 ships its format as Python (`encoding/encoding_dsv4.py`),
//!     not as a Jinja template → builtin official renderer.
//!   - A checkpoint with neither gets ChatML + a load-time warning.

use std::path::Path;

use anyhow::{Context, Result, anyhow, ensure};
use tokenizers::Tokenizer;
use tokenizers::decoders::DecoderWrapper;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;

use crate::schema::ChatMessage;

#[derive(Clone, Debug)]
enum ChatTemplate {
    /// The checkpoint's own Jinja `chat_template`, rendered with the standard
    /// HF context (`messages`, `add_generation_prompt`, `bos_token`,
    /// `eos_token`). This is the general path — any new model that ships a
    /// template works with zero ARLE code.
    Jinja {
        source: String,
        bos_token: String,
        eos_token: String,
    },
    /// Official DeepSeek-V4 format (fullwidth-bar specials, non-thinking
    /// "chat" mode). Canonical reference: the checkpoint's
    /// `encoding/encoding_dsv4.py` (issue #66).
    BuiltinDeepseekV4,
    /// Last-resort Qwen ChatML (warned at load).
    BuiltinChatMl,
}

/// Streaming detokenizer. A codepoint's bytes can span a token boundary, so
/// decoding each delta on its own replaces the split character with U+FFFD —
/// one per orphaned byte, so a 4-byte emoji arrives as two of them. Holds the
/// tokens carrying an incomplete tail until the bytes that finish it arrive.
#[derive(Default)]
pub struct IncrementalDetokenizer {
    pending: Vec<u32>,
}

/// A codepoint is at most 4 bytes and a token at least 1, so a partial tail
/// spans at most 4 tokens; a longer replacement run is genuinely invalid output
/// and must be emitted rather than buffered forever.
const MAX_PENDING_TOKENS: usize = 4;

impl IncrementalDetokenizer {
    pub fn push(&mut self, tok: &OpenAiTokenizer, ids: &[u32]) -> String {
        self.pending.extend_from_slice(ids);
        let text = tok.decode(&self.pending).unwrap_or_default();
        if !text.ends_with(char::REPLACEMENT_CHARACTER) || self.pending.len() > MAX_PENDING_TOKENS {
            self.pending.clear();
            return text;
        }
        let n = self.pending.len();
        for k in 1..=n {
            let head = tok.decode(&self.pending[..n - k]).unwrap_or_default();
            if !head.ends_with(char::REPLACEMENT_CHARACTER) {
                self.pending.drain(..n - k);
                return head;
            }
        }
        String::new()
    }

    /// Emit whatever is still held. The stream is over, so an unfinished
    /// codepoint will never complete.
    pub fn flush(&mut self, tok: &OpenAiTokenizer) -> String {
        if self.pending.is_empty() {
            return String::new();
        }
        let text = tok.decode(&self.pending).unwrap_or_default();
        self.pending.clear();
        text
    }
}

#[derive(Clone)]
pub struct OpenAiTokenizer {
    inner: Tokenizer,
    template: ChatTemplate,
}

impl OpenAiTokenizer {
    /// Load `tokenizer.json` from a model dir and resolve the chat template:
    /// checkpoint `chat_template` / `chat_template.jinja` → builtin
    /// per-architecture → ChatML + warn.
    pub fn from_model_dir(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let tokenizer_path = model_dir.join("tokenizer.json");
        let inner = Tokenizer::from_file(&tokenizer_path)
            .map_err(|err| anyhow!("load tokenizer {} failed: {err}", tokenizer_path.display()))?;
        let template = resolve_chat_template(model_dir)?;
        match &template {
            ChatTemplate::Jinja { source, .. } => {
                log::info!(
                    "chat template: checkpoint chat_template ({} chars) from {}",
                    source.len(),
                    model_dir.display()
                );
            }
            ChatTemplate::BuiltinDeepseekV4 => {
                log::info!(
                    "chat template: builtin DeepSeek-V4 (checkpoint ships no Jinja template)"
                );
            }
            ChatTemplate::BuiltinChatMl => {
                log::warn!(
                    "chat template: {} ships no chat_template and matches no builtin — \
                     falling back to Qwen ChatML, which may mis-render non-Qwen models",
                    model_dir.display()
                );
            }
        }
        Ok(Self { inner, template })
    }

    /// Encode text into token ids without adding special tokens.
    ///
    /// DSv4's BOS (`<｜begin▁of▁sentence｜>`, id 0) is in the vocab but missing
    /// from `added_tokens`, so the BPE pre-tokenizer splits it into mojibake
    /// subwords. Strip the BOS prefix and prepend id 0 directly.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let bos = "<｜begin▁of▁sentence｜>";
        let (body, prepend_bos) = match text.strip_prefix(bos) {
            Some(rest) => (rest, true),
            None => (text, false),
        };
        let encoding = self
            .inner
            .encode(body, false)
            .map_err(|err| anyhow!("tokenize prompt failed: {err}"))?;
        let mut ids = encoding.get_ids().to_vec();
        if prepend_bos {
            ids.insert(0, 0);
        }
        Ok(ids)
    }

    /// Vocabulary indexed by token id, for grammar-compiler construction.
    /// Holes (ids the vocab map skips) come back empty.
    pub fn vocab_by_id(&self) -> Vec<String> {
        let map = self.inner.get_vocab(true);
        let mut out =
            vec![String::new(); map.values().copied().max().map_or(0, |m| m as usize + 1)];
        for (tok, id) in map {
            out[id as usize] = tok;
        }
        out
    }

    /// Decode token ids into text, skipping special tokens.
    pub fn decode(&self, token_ids: &[u32]) -> Result<String> {
        self.inner
            .decode(token_ids, true)
            .map_err(|err| anyhow!("decode generated tokens failed: {err}"))
    }

    /// Force a byte-level (GPT-2) decoder, overriding whatever the checkpoint's
    /// `tokenizer.json` declared. DeepSeek-OCR ships a byte-level BPE vocab
    /// (`Ġ`=space, `Ċ`=newline) but a mismatched Metaspace/ByteFallback decoder,
    /// so its raw decode leaks `Ġ`/`Ċ` glyphs and drops spaces — useless OCR
    /// text. The reference HF `tokenizers` library reproduces the same bug; the
    /// fix is to decode with `ByteLevel`, which maps the byte alphabet back to
    /// real UTF-8. Scoped to the OCR load path; other models keep their decoder.
    pub fn force_byte_level_decoder(&mut self) {
        self.inner
            .with_decoder(Some(DecoderWrapper::ByteLevel(ByteLevel::default())));
    }

    /// `true` if this checkpoint's template defaults to thinking-on. Only
    /// DeepSeek-V4-Flash does: it is reasoning-trained and degenerates
    /// (looping, bare `</think>` leaks) when forced non-thinking. Jinja/ChatML
    /// stay thinking-off (unchanged default behavior).
    #[must_use]
    pub fn defaults_thinking_on(&self) -> bool {
        matches!(self.template, ChatTemplate::BuiltinDeepseekV4)
    }

    pub fn render_chat(&self, messages: &[ChatMessage]) -> Result<String> {
        self.render_chat_with_kwargs(messages, None)
    }

    /// Render chat messages, passing optional `chat_template_kwargs` (e.g.
    /// `enable_thinking`) into the Jinja template context. `None` kwargs render
    /// byte-identically to [`Self::render_chat`]. Builtin (non-Jinja) renderers
    /// have no template variables and ignore the kwargs.
    ///
    /// Thin wrapper over [`Self::render_chat_full`] with no tools and
    /// thinking-off — the byte-identical legacy path for tool-less callers.
    pub fn render_chat_with_kwargs(
        &self,
        messages: &[ChatMessage],
        chat_template_kwargs: Option<&serde_json::Map<String, serde_json::Value>>,
    ) -> Result<String> {
        self.render_chat_full(messages, chat_template_kwargs, &[], false, None)
    }

    /// Full chat render: threads tool definitions, per-message tool calls, and
    /// the thinking / reasoning-effort switches into whichever renderer the
    /// checkpoint resolved to. Tool-less, thinking-off calls render
    /// byte-identically to the legacy [`Self::render_chat_with_kwargs`].
    pub fn render_chat_full(
        &self,
        messages: &[ChatMessage],
        chat_template_kwargs: Option<&serde_json::Map<String, serde_json::Value>>,
        tools: &[chat::OpenAiToolDefinition],
        thinking: bool,
        reasoning_effort: Option<&str>,
    ) -> Result<String> {
        ensure!(
            !messages.is_empty(),
            "messages must contain at least one message"
        );
        // Strict checkpoint templates (e.g. Qwen) `raise_exception` unless every
        // system message is at the front; some clients (e.g. the Eli agent
        // framework) place the system prompt after prior context. Hoist system
        // messages to the front (stable) so any reasonable order renders — no clone
        // when already ordered.
        let hoisted;
        let messages: &[ChatMessage] = if needs_system_hoist(messages) {
            hoisted = hoist_system_first(messages);
            &hoisted
        } else {
            messages
        };
        match &self.template {
            ChatTemplate::Jinja {
                source,
                bos_token,
                eos_token,
            } => render_jinja(
                source,
                bos_token,
                eos_token,
                messages,
                chat_template_kwargs,
                tools,
            ),
            ChatTemplate::BuiltinDeepseekV4 => Ok(render_deepseek_v4(
                messages,
                tools,
                thinking,
                reasoning_effort,
            )),
            ChatTemplate::BuiltinChatMl => render_chatml(messages, tools),
        }
    }
}

/// `true` if any system message follows a non-system message — i.e. the system
/// block isn't already at the front (so a hoist is needed).
fn needs_system_hoist(messages: &[ChatMessage]) -> bool {
    messages
        .iter()
        .skip_while(|m| m.role == "system")
        .any(|m| m.role == "system")
}

fn hoist_system_first(messages: &[ChatMessage]) -> Vec<ChatMessage> {
    messages
        .iter()
        .filter(|m| m.role == "system")
        .cloned()
        .chain(messages.iter().filter(|m| m.role != "system").cloned())
        .collect()
}

/// Render the checkpoint's Jinja template with the standard HF context.
///
/// The environment is built per call — chat rendering is the COLD facade path
/// and per-call compile keeps [`OpenAiTokenizer`] `Clone` without sharing a
/// non-`Clone` minijinja environment.
pub(crate) fn render_jinja(
    source: &str,
    bos_token: &str,
    eos_token: &str,
    messages: &[ChatMessage],
    chat_template_kwargs: Option<&serde_json::Map<String, serde_json::Value>>,
    tools: &[chat::OpenAiToolDefinition],
) -> Result<String> {
    use minijinja::{Environment, UndefinedBehavior, context};

    let mut env = Environment::new();
    // HF templates probe optional context (`tools`, `enable_thinking`, …) with
    // `if`; lenient undefined makes those probes falsy instead of erroring.
    env.set_undefined_behavior(UndefinedBehavior::Lenient);
    // HF chat templates are written against Python jinja2 and lean on str
    // methods (`.startswith`, `.split`, …) — pycompat supplies them (the same
    // shim TGI ships).
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_function(
        "raise_exception",
        |msg: String| -> std::result::Result<minijinja::Value, minijinja::Error> {
            Err(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                format!("chat template raise_exception: {msg}"),
            ))
        },
    );
    env.add_template("chat", source)
        .map_err(|err| anyhow!("compile checkpoint chat_template failed: {err}"))?;

    let rows: Vec<minijinja::Value> = messages
        .iter()
        .map(|m| {
            let content = minijinja::Value::from_serialize(m.template_content());
            // Omit `tool_calls` when empty so a tool-less message is byte-identical
            // even under templates that probe `is defined`, not just truthiness.
            if m.tool_calls.is_empty() {
                context! { role => m.role.as_str(), content => content }
            } else {
                let tool_calls =
                    minijinja::Value::from_serialize(template_tool_calls(&m.tool_calls));
                context! { role => m.role.as_str(), content => content, tool_calls => tool_calls }
            }
        })
        .collect();
    let base = context! {
        messages => rows,
        add_generation_prompt => true,
        bos_token => bos_token,
        eos_token => eos_token,
    };
    // Expose `tools` only when present — absent (not empty-list) keeps tool-less
    // renders byte-identical for templates probing `tools is defined`/`is not none`.
    let base = if tools.is_empty() {
        base
    } else {
        context! { tools => minijinja::Value::from_serialize(tools), ..base }
    };
    // Pass-through `chat_template_kwargs` (e.g. `enable_thinking`) as top-level
    // template variables, merged ON TOP of the standard HF context. When absent
    // the render is byte-identical to before — only the `Some` arm changes the
    // context. Spread-merge keeps the base keys; the kwargs cannot shadow them
    // because the standard keys are reserved by HF templates anyway.
    let render_context = match chat_template_kwargs {
        Some(kwargs) if !kwargs.is_empty() => {
            let extra = minijinja::Value::from_serialize(kwargs);
            // base spread LAST so the reserved HF keys (messages,
            // add_generation_prompt, bos/eos) always win over any kwargs.
            context! { ..extra, ..base }
        }
        _ => base,
    };
    env.get_template("chat")
        .expect("template registered above")
        .render(render_context)
        .map_err(|err| anyhow!("render checkpoint chat_template failed: {err}"))
}

/// Tool calls in HF chat-template convention: `function.arguments` is a
/// mapping, not the OpenAI wire's JSON string — templates iterate it
/// (`arguments|items`), so parse for the render (HF/vLLM do the same);
/// unparseable arguments fall back to the raw string.
fn template_tool_calls(calls: &[chat::OpenAiToolCall]) -> Vec<serde_json::Value> {
    calls
        .iter()
        .map(|call| {
            let arguments = serde_json::from_str::<serde_json::Value>(&call.function.arguments)
                .unwrap_or_else(|_| serde_json::Value::String(call.function.arguments.clone()));
            serde_json::json!({
                "id": call.id,
                "type": call.call_type,
                "function": {"name": call.function.name, "arguments": arguments},
            })
        })
        .collect()
}

/// The `chat` crate's OpenAI wire messages — the shared shape its DeepSeek-V4
/// and ChatML+tools renderers consume.
fn to_openai_messages(messages: &[ChatMessage]) -> Vec<chat::OpenAiChatMessage> {
    messages.iter().map(ChatMessage::to_openai).collect()
}

/// DeepSeek-V4 prompt via the `chat` crate — the single source of the DSML tool
/// format and the thinking / non-thinking generation prefix.
fn render_deepseek_v4(
    messages: &[ChatMessage],
    tools: &[chat::OpenAiToolDefinition],
    thinking: bool,
    reasoning_effort: Option<&str>,
) -> String {
    chat::openai_messages_to_deepseek_v4_prompt(
        &to_openai_messages(messages),
        tools,
        &chat::DeepSeekV4ChatTemplateOptions {
            thinking,
            reasoning_effort: reasoning_effort.map(str::to_owned),
        },
    )
}

/// Last-resort Qwen ChatML rendering. A tool-less render stays byte-identical to
/// the legacy path; when tools are present the shared `chat` ChatML+tools
/// renderer takes over (one source of the tool block + native XML call format).
fn render_chatml(messages: &[ChatMessage], tools: &[chat::OpenAiToolDefinition]) -> Result<String> {
    if !tools.is_empty() {
        return Ok(chat::openai_messages_to_prompt(
            &to_openai_messages(messages),
            tools,
        ));
    }
    let mut out = String::new();
    for message in messages {
        let role = message.role.trim();
        ensure!(!role.is_empty(), "message role must not be empty");
        out.push_str("<|im_start|>");
        out.push_str(role);
        out.push('\n');
        out.push_str(&message.content_text());
        out.push_str("<|im_end|>\n");
    }
    out.push_str("<|im_start|>assistant\n");
    Ok(out)
}

/// Resolve the chat template for a model dir:
/// 1. `tokenizer_config.json` `chat_template` (string, or HF list-of-named —
///    `default` preferred) → [`ChatTemplate::Jinja`];
/// 2. `chat_template.jinja` next to the tokenizer → [`ChatTemplate::Jinja`];
/// 3. `config.json` `architectures` starting with `DeepseekV4` →
///    [`ChatTemplate::BuiltinDeepseekV4`];
/// 4. otherwise [`ChatTemplate::BuiltinChatMl`] (caller warns).
fn resolve_chat_template(model_dir: &Path) -> Result<ChatTemplate> {
    let tok_cfg = read_json(&model_dir.join("tokenizer_config.json"))?;
    if let Some(cfg) = &tok_cfg
        && let Some(source) = extract_chat_template(cfg)
    {
        return Ok(ChatTemplate::Jinja {
            source,
            bos_token: extract_token(cfg, "bos_token").unwrap_or_default(),
            eos_token: extract_token(cfg, "eos_token").unwrap_or_default(),
        });
    }
    if let Some(source) = read_chat_template_file(model_dir)? {
        return Ok(ChatTemplate::Jinja {
            source,
            bos_token: tok_cfg
                .as_ref()
                .and_then(|cfg| extract_token(cfg, "bos_token"))
                .unwrap_or_default(),
            eos_token: tok_cfg
                .as_ref()
                .and_then(|cfg| extract_token(cfg, "eos_token"))
                .unwrap_or_default(),
        });
    }
    let model_cfg = read_json(&model_dir.join("config.json"))?;
    let is_dsv4 = model_cfg
        .as_ref()
        .and_then(|v| v.get("architectures"))
        .and_then(|a| a.as_array())
        .is_some_and(|archs| {
            archs
                .iter()
                .filter_map(|v| v.as_str())
                .any(|name| name.starts_with("DeepseekV4"))
        });
    Ok(if is_dsv4 {
        ChatTemplate::BuiltinDeepseekV4
    } else {
        ChatTemplate::BuiltinChatMl
    })
}

fn read_json(path: &Path) -> Result<Option<serde_json::Value>> {
    if !path.exists() {
        return Ok(None);
    }
    let raw = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    Ok(Some(
        serde_json::from_str(&raw).with_context(|| format!("parse {}", path.display()))?,
    ))
}

fn read_chat_template_file(model_dir: &Path) -> Result<Option<String>> {
    let path = model_dir.join("chat_template.jinja");
    if !path.exists() {
        return Ok(None);
    }
    std::fs::read_to_string(&path)
        .with_context(|| format!("read {}", path.display()))
        .map(Some)
}

/// `chat_template` is a string, or (HF multi-template form) a list of
/// `{name, template}` — prefer `default`, else the first entry.
fn extract_chat_template(cfg: &serde_json::Value) -> Option<String> {
    match cfg.get("chat_template")? {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Array(entries) => entries
            .iter()
            .find(|e| e.get("name").and_then(|n| n.as_str()) == Some("default"))
            .or_else(|| entries.first())
            .and_then(|e| e.get("template"))
            .and_then(|t| t.as_str())
            .map(str::to_owned),
        _ => None,
    }
}

fn extract_token(cfg: &serde_json::Value, key: &str) -> Option<String> {
    match cfg.get(key)? {
        serde_json::Value::String(s) => Some(s.clone()),
        other => other
            .get("content")
            .and_then(|c| c.as_str())
            .map(str::to_owned),
    }
}
