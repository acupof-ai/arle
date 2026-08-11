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

#[cfg(test)]
use crate::schema::ChatContent;
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
    /// This backend intentionally exposes `/v1/completions` only because the
    /// checkpoint ships no chat template and no verified builtin renderer.
    UnsupportedChat { reason: String },
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
            ChatTemplate::UnsupportedChat { reason } => {
                log::warn!(
                    "chat template disabled for {}: {reason}",
                    model_dir.display()
                );
            }
        }
        Ok(Self { inner, template })
    }

    /// Load a tokenizer for `/v1/completions` while making
    /// `/v1/chat/completions` fail closed.
    pub fn from_model_dir_without_chat(
        model_dir: impl AsRef<Path>,
        reason: impl Into<String>,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let tokenizer_path = model_dir.join("tokenizer.json");
        let inner = Tokenizer::from_file(&tokenizer_path)
            .map_err(|err| anyhow!("load tokenizer {} failed: {err}", tokenizer_path.display()))?;
        Ok(Self {
            inner,
            template: ChatTemplate::UnsupportedChat {
                reason: reason.into(),
            },
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
            ChatTemplate::UnsupportedChat { reason } => {
                anyhow::bail!("chat completions are not supported for this tokenizer: {reason}")
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.to_string(),
            content: Some(ChatContent::Text(content.to_string())),
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
        }
    }

    #[test]
    fn system_messages_hoisted_to_front_for_strict_templates() {
        // Eli-style order: context/user before the system prompt.
        let out = hoist_system_first(&[
            msg("user", "hi"),
            msg("system", "be brief"),
            msg("assistant", "ok"),
        ]);
        assert_eq!(out[0].role, "system");
        assert_eq!(out[1].role, "user");
        assert_eq!(out[2].role, "assistant");
        assert!(needs_system_hoist(&[msg("user", "hi"), msg("system", "s")]));
        assert!(!needs_system_hoist(&[
            msg("system", "s"),
            msg("user", "hi")
        ]));
        assert!(!needs_system_hoist(&[
            msg("user", "hi"),
            msg("assistant", "yo")
        ]));
    }

    /// A 4-byte emoji streamed one token at a time must arrive intact, not as
    /// two U+FFFD. Needs a real byte-level vocab, so it runs against
    /// `INFER_TEST_MODEL_PATH` (or the repo-local Qwen3.5-0.8B).
    #[test]
    fn streaming_detokenizer_rejoins_split_codepoints() {
        let dir = std::env::var("INFER_TEST_MODEL_PATH")
            .unwrap_or_else(|_| "models/Qwen3.5-0.8B".to_string());
        let path = std::path::Path::new(&dir);
        if !path.join("tokenizer.json").exists() {
            eprintln!("no tokenizer.json under {dir}; skipping split-codepoint test");
            return;
        }
        let tok = OpenAiTokenizer::from_model_dir_without_chat(path, "test").expect("load");
        let text = "1. ✅ ok\n2. 🚀 go";
        let ids = tok.encode(text).expect("encode");
        // Whole-sequence decode is the reference; per-token streaming must match.
        assert_eq!(tok.decode(&ids).expect("decode"), text);

        // The defect being fixed: decoding each token alone splits the emoji.
        let naive: String = ids
            .iter()
            .map(|&id| tok.decode(&[id]).unwrap_or_default())
            .collect();
        assert!(
            naive.contains(char::REPLACEMENT_CHARACTER),
            "vocab does not split this codepoint, so the test proves nothing"
        );

        let mut detok = IncrementalDetokenizer::default();
        let streamed: String = ids.iter().map(|&id| detok.push(&tok, &[id])).collect();
        let streamed = streamed + &detok.flush(&tok);
        assert_eq!(
            streamed, text,
            "streaming decode dropped or replaced a split codepoint"
        );
        assert!(!streamed.contains(char::REPLACEMENT_CHARACTER));
    }

    /// Bytes that never form a valid codepoint must still be emitted, or a
    /// stream would stall forever holding them.
    #[test]
    fn streaming_detokenizer_does_not_buffer_forever() {
        let dir = std::env::var("INFER_TEST_MODEL_PATH")
            .unwrap_or_else(|_| "models/Qwen3.5-0.8B".to_string());
        let path = std::path::Path::new(&dir);
        if !path.join("tokenizer.json").exists() {
            eprintln!("no tokenizer.json under {dir}; skipping invalid-run test");
            return;
        }
        let tok = OpenAiTokenizer::from_model_dir_without_chat(path, "test").expect("load");
        // A run of continuation bytes: the emoji's tail tokens without its lead.
        let ids = tok.encode("🚀🚀🚀").expect("encode");
        let mut detok = IncrementalDetokenizer::default();
        let mut out = String::new();
        for &id in &ids[1..] {
            out.push_str(&detok.push(&tok, &[id]));
        }
        out.push_str(&detok.flush(&tok));
        assert!(
            !out.is_empty(),
            "invalid bytes were buffered instead of emitted"
        );
    }

    // A trimmed Qwen-style ChatML jinja template exercising the standard HF
    // context shape (messages/add_generation_prompt) plus loop + if probes.
    const QWEN_STYLE_TEMPLATE: &str = "{%- for message in messages %}\
{{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}\
{%- endfor %}\
{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{%- endif %}";

    #[test]
    fn jinja_renders_checkpoint_template() {
        let out = render_jinja(
            QWEN_STYLE_TEMPLATE,
            "",
            "<|im_end|>",
            &[msg("system", "be brief"), msg("user", "hi")],
            None,
            &[],
        )
        .unwrap();
        assert_eq!(
            out,
            "<|im_start|>system\nbe brief<|im_end|>\n\
             <|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn jinja_lenient_undefined_allows_tools_probe() {
        let out = render_jinja(
            "{%- if tools %}TOOLS{%- endif %}{{ messages[0].content }}",
            "",
            "",
            &[msg("user", "x")],
            None,
            &[],
        )
        .unwrap();
        assert_eq!(out, "x");
    }

    /// Qwen-family templates render tool schemas via `{{ tool | tojson }}` —
    /// requires minijinja's `json` feature (tools were unrenderable without it).
    #[test]
    fn jinja_tojson_filter_renders_tools() {
        let tools = vec![chat::OpenAiToolDefinition {
            tool_type: "function".into(),
            function: chat::OpenAiFunctionDefinition {
                name: "get_weather".into(),
                description: Some("Get the weather".into()),
                parameters: Some(serde_json::json!({"type": "object"})),
            },
        }];
        let out = render_jinja(
            "{%- for tool in tools %}{{ tool | tojson }}{%- endfor %}",
            "",
            "",
            &[msg("user", "x")],
            None,
            &tools,
        )
        .unwrap();
        assert!(out.contains(r#""name":"get_weather""#), "got: {out}");
    }

    /// Qwen-family templates iterate `tool_call.function.arguments|items` —
    /// history arguments must render as a mapping (HF convention), not the
    /// OpenAI wire's JSON string.
    #[test]
    fn jinja_tool_call_arguments_render_as_mapping() {
        let message: ChatMessage = serde_json::from_value(serde_json::json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"},
            }]
        }))
        .unwrap();
        let out = render_jinja(
            "{%- for tc in messages[0].tool_calls %}\
             {%- for k, v in tc.function.arguments|items %}{{ k }}={{ v }}{%- endfor %}\
             {%- endfor %}",
            "",
            "",
            &[message],
            None,
            &[],
        )
        .unwrap();
        assert_eq!(out, "city=Paris");
    }

    #[test]
    fn jinja_enable_thinking_kwarg_toggles_branch() {
        // A Qwen-style `{% if enable_thinking %}` probe: the kwarg must reach
        // the template context and flip the branch.
        const THINK_TEMPLATE: &str = "{{ messages[0].content }}{%- if enable_thinking %}<think>{%- else %}</think>{%- endif %}";

        let kwargs_on = serde_json::Map::from_iter([(
            "enable_thinking".to_string(),
            serde_json::Value::Bool(true),
        )]);
        let kwargs_off = serde_json::Map::from_iter([(
            "enable_thinking".to_string(),
            serde_json::Value::Bool(false),
        )]);

        let on = render_jinja(
            THINK_TEMPLATE,
            "",
            "",
            &[msg("user", "hi")],
            Some(&kwargs_on),
            &[],
        )
        .unwrap();
        let off = render_jinja(
            THINK_TEMPLATE,
            "",
            "",
            &[msg("user", "hi")],
            Some(&kwargs_off),
            &[],
        )
        .unwrap();
        // absent kwargs must render byte-identically to the explicit `false`
        // case (lenient-undefined makes the probe falsy) — backward compat.
        let absent = render_jinja(THINK_TEMPLATE, "", "", &[msg("user", "hi")], None, &[]).unwrap();

        assert_eq!(on, "hi<think>");
        assert_eq!(off, "hi</think>");
        assert_ne!(on, off, "enable_thinking must toggle the thinking branch");
        assert_eq!(
            absent, off,
            "absent kwargs must match explicit enable_thinking=false"
        );
    }

    #[test]
    fn jinja_receives_content_parts_as_sequences() {
        let message: ChatMessage = serde_json::from_value(serde_json::json!({
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
            ]
        }))
        .unwrap();
        let out = render_jinja(
            "{%- for item in messages[0].content -%}{%- if item.type == 'text' -%}{{ item.text }}{%- elif item.type == 'image' -%}<|image|>{%- endif -%}{%- endfor -%}",
            "",
            "",
            &[message],
            None,
            &[],
        )
        .unwrap();
        assert_eq!(out, "look<|image|>");
    }

    #[test]
    fn jinja_raise_exception_surfaces_as_error() {
        let err = render_jinja(
            "{{ raise_exception('bad roles') }}",
            "",
            "",
            &[msg("user", "x")],
            None,
            &[],
        )
        .unwrap_err();
        assert!(err.to_string().contains("render checkpoint chat_template"));
    }

    fn dsv4_tokenizer() -> OpenAiTokenizer {
        OpenAiTokenizer {
            inner: Tokenizer::new(tokenizers::models::wordlevel::WordLevel::default()),
            template: ChatTemplate::BuiltinDeepseekV4,
        }
    }

    #[test]
    fn deepseek_v4_render_full_single_turn_non_thinking() {
        // Tool-less, thinking-off render is byte-identical to the legacy builtin's
        // single-turn output (the `chat` crate is now the single source).
        let rendered = dsv4_tokenizer()
            .render_chat_full(
                &[
                    msg("system", "You are a helpful assistant."),
                    msg("user", "What's the capital of France?"),
                ],
                None,
                &[],
                false,
                None,
            )
            .unwrap();
        assert_eq!(
            rendered,
            "<｜begin▁of▁sentence｜>You are a helpful assistant.\
             <｜User｜>What's the capital of France?<｜Assistant｜></think>"
        );
    }

    #[test]
    fn deepseek_v4_render_full_thinking_opens_think_and_renders_tools() {
        let tools = vec![chat::OpenAiToolDefinition {
            tool_type: "function".into(),
            function: chat::OpenAiFunctionDefinition {
                name: "shell".into(),
                description: Some("Run a shell command".into()),
                parameters: None,
            },
        }];
        let rendered = dsv4_tokenizer()
            .render_chat_full(&[msg("user", "list files")], None, &tools, true, None)
            .unwrap();
        // DSML tool schema block is present and the generation prefix opens a
        // thinking block (reasoning model default).
        assert!(rendered.contains("｜DSML｜tool_calls"));
        assert!(rendered.contains(r#""name": "shell""#) || rendered.contains(r#""name":"shell""#));
        assert!(rendered.ends_with("<｜Assistant｜><think>"));
    }

    #[test]
    fn defaults_thinking_on_only_for_deepseek_v4() {
        assert!(dsv4_tokenizer().defaults_thinking_on());
        let chatml = OpenAiTokenizer {
            inner: Tokenizer::new(tokenizers::models::wordlevel::WordLevel::default()),
            template: ChatTemplate::BuiltinChatMl,
        };
        assert!(!chatml.defaults_thinking_on());
        let jinja = OpenAiTokenizer {
            inner: Tokenizer::new(tokenizers::models::wordlevel::WordLevel::default()),
            template: ChatTemplate::Jinja {
                source: "{{ messages[0].content }}".into(),
                bos_token: String::new(),
                eos_token: String::new(),
            },
        };
        assert!(!jinja.defaults_thinking_on());
    }

    #[test]
    fn chatml_fallback_unchanged() {
        let rendered = render_chatml(&[msg("user", "hi")], &[]).unwrap();
        assert_eq!(
            rendered,
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn unsupported_chat_template_fails_closed() {
        let tokenizer = OpenAiTokenizer {
            inner: Tokenizer::new(tokenizers::models::wordlevel::WordLevel::default()),
            template: ChatTemplate::UnsupportedChat {
                reason: "no verified template".to_string(),
            },
        };
        let err = tokenizer.render_chat(&[msg("user", "hi")]).unwrap_err();
        assert!(err.to_string().contains("not supported"));
        assert!(err.to_string().contains("no verified template"));
    }

    #[test]
    fn template_extraction_handles_hf_list_form() {
        let cfg: serde_json::Value = serde_json::json!({
            "chat_template": [
                {"name": "tool_use", "template": "T"},
                {"name": "default", "template": "D"},
            ]
        });
        assert_eq!(extract_chat_template(&cfg).as_deref(), Some("D"));
        let cfg2: serde_json::Value =
            serde_json::json!({"chat_template": "S", "eos_token": {"content": "<eos>"}});
        assert_eq!(extract_chat_template(&cfg2).as_deref(), Some("S"));
        assert_eq!(extract_token(&cfg2, "eos_token").as_deref(), Some("<eos>"));
    }

    #[test]
    fn resolves_external_chat_template_file() {
        let suffix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "infer-server-chat-template-file-{}-{suffix}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("tokenizer_config.json"),
            r#"{"chat_template": null, "bos_token": "<bos>", "eos_token": "<eos>"}"#,
        )
        .unwrap();
        std::fs::write(
            dir.join("chat_template.jinja"),
            "{{ bos_token }}{{ eos_token }}",
        )
        .unwrap();
        std::fs::write(dir.join("config.json"), "{}").unwrap();

        let template = resolve_chat_template(&dir).unwrap();
        let ChatTemplate::Jinja {
            source,
            bos_token,
            eos_token,
        } = template
        else {
            panic!("external template should resolve to Jinja");
        };
        assert_eq!(source, "{{ bos_token }}{{ eos_token }}");
        assert_eq!(bos_token, "<bos>");
        assert_eq!(eos_token, "<eos>");
        let _ = std::fs::remove_dir_all(dir);
    }
}

#[cfg(test)]
mod real_checkpoint_tests {
    use super::*;

    /// Render the REAL Qwen checkpoint template if the repo-local model dir is
    /// present (skips otherwise, mirroring the model-gated test convention).
    /// Catches minijinja incompatibilities (tojson/namespace/loop) that the
    /// trimmed fixture cannot.
    #[test]
    fn real_qwen_template_renders() {
        let dir = Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../models/Qwen3-0.6B"
        ));
        if !dir.join("tokenizer_config.json").exists() {
            eprintln!("skip: models/Qwen3-0.6B not present");
            return;
        }
        let template = resolve_chat_template(dir).unwrap();
        let ChatTemplate::Jinja {
            source,
            bos_token,
            eos_token,
        } = template
        else {
            panic!("Qwen checkpoint should resolve to its own chat_template");
        };
        let out = render_jinja(
            &source,
            &bos_token,
            &eos_token,
            &[
                ChatMessage {
                    role: "system".into(),
                    content: Some(ChatContent::Text("be brief".into())),
                    tool_calls: Vec::new(),
                    tool_call_id: None,
                    name: None,
                },
                ChatMessage {
                    role: "user".into(),
                    content: Some(ChatContent::Text("hi".into())),
                    tool_calls: Vec::new(),
                    tool_call_id: None,
                    name: None,
                },
            ],
            None,
            &[],
        )
        .unwrap();
        assert!(out.contains("<|im_start|>user\nhi<|im_end|>"), "got: {out}");
        assert!(out.ends_with("<|im_start|>assistant\n"), "got: {out}");
    }

    #[test]
    fn real_diffusion_gemma_external_template_renders_if_cached() {
        let Some(home) = std::env::var_os("HOME") else {
            eprintln!("skip: HOME not set");
            return;
        };
        let root = Path::new(&home).join(
            ".cache/huggingface/hub/models--mlx-community--diffusiongemma-26B-A4B-it-4bit/snapshots",
        );
        let Ok(entries) = std::fs::read_dir(&root) else {
            eprintln!("skip: DiffusionGemma HF snapshot not present");
            return;
        };
        let Some(dir) = entries
            .filter_map(std::result::Result::ok)
            .map(|entry| entry.path())
            .find(|path| path.join("chat_template.jinja").exists())
        else {
            eprintln!("skip: DiffusionGemma chat_template.jinja not present");
            return;
        };

        let template = resolve_chat_template(&dir).unwrap();
        let ChatTemplate::Jinja {
            source,
            bos_token,
            eos_token,
        } = template
        else {
            panic!("DiffusionGemma should resolve external chat_template.jinja");
        };
        let out = render_jinja(
            &source,
            &bos_token,
            &eos_token,
            &[ChatMessage {
                role: "user".into(),
                content: Some(ChatContent::Text("hi".into())),
                tool_calls: Vec::new(),
                tool_call_id: None,
                name: None,
            }],
            None,
            &[],
        )
        .unwrap();
        assert!(out.contains("<|turn>user"), "got: {out}");
        assert!(out.contains("hi"), "got: {out}");
        assert!(out.contains("<|turn>model"), "got: {out}");
    }
}
