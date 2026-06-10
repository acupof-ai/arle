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

use crate::schema::ChatMessage;

/// How `render_chat` produces the prompt, resolved once at load time.
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

/// Tokenizer and chat-template adapter for the OpenAI v1 facade.
#[derive(Clone)]
pub struct OpenAiTokenizer {
    inner: Tokenizer,
    template: ChatTemplate,
}

// Official DSv4 prompt pieces (encoding_dsv4.py constants; fullwidth bars).
const DSV4_BOS: &str = "<｜begin▁of▁sentence｜>";
const DSV4_EOS: &str = "<｜end▁of▁sentence｜>";
const DSV4_USER: &str = "<｜User｜>";
const DSV4_ASSISTANT: &str = "<｜Assistant｜>";
const DSV4_THINK_END: &str = "</think>";

impl OpenAiTokenizer {
    /// Load `tokenizer.json` from a model dir and resolve the chat template:
    /// checkpoint `chat_template` → builtin per-architecture → ChatML + warn.
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

    /// Render OpenAI chat messages into the model's prompt form.
    pub fn render_chat(&self, messages: &[ChatMessage]) -> Result<String> {
        ensure!(
            !messages.is_empty(),
            "messages must contain at least one message"
        );
        match &self.template {
            ChatTemplate::Jinja {
                source,
                bos_token,
                eos_token,
            } => render_jinja(source, bos_token, eos_token, messages),
            ChatTemplate::BuiltinDeepseekV4 => render_deepseek_v4(messages),
            ChatTemplate::BuiltinChatMl => render_chatml(messages),
        }
    }
}

/// Render the checkpoint's Jinja template with the standard HF context.
///
/// The environment is built per call — chat rendering is the COLD facade path
/// and per-call compile keeps [`OpenAiTokenizer`] `Clone` without sharing a
/// non-`Clone` minijinja environment.
fn render_jinja(
    source: &str,
    bos_token: &str,
    eos_token: &str,
    messages: &[ChatMessage],
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
            context! {
                role => m.role.as_str(),
                content => m.content.as_deref().unwrap_or(""),
            }
        })
        .collect();
    env.get_template("chat")
        .expect("template registered above")
        .render(context! {
            messages => rows,
            add_generation_prompt => true,
            bos_token => bos_token,
            eos_token => eos_token,
        })
        .map_err(|err| anyhow!("render checkpoint chat_template failed: {err}"))
}

/// Last-resort Qwen ChatML rendering.
fn render_chatml(messages: &[ChatMessage]) -> Result<String> {
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

/// Official DeepSeek-V4 rendering, non-thinking "chat" mode, text-only v1
/// (no tools / response_format / reasoning_content — those raise instead of
/// silently mis-rendering). Mirrors `encoding_dsv4.py`:
///   - `system`: bare content (immediately after bos)
///   - `user`: `<｜User｜>` + content
///   - `assistant`: content + `<｜end▁of▁sentence｜>`
///   - final generation suffix: `<｜Assistant｜></think>` (chat mode)
fn render_deepseek_v4(messages: &[ChatMessage]) -> Result<String> {
    let mut out = String::from(DSV4_BOS);
    for (index, message) in messages.iter().enumerate() {
        let role = message.role.trim();
        let content = message.content.as_deref().unwrap_or("");
        match role {
            "system" => {
                ensure!(
                    index == 0,
                    "DeepSeek-V4 template: system message must be first"
                );
                out.push_str(content);
            }
            "user" => {
                out.push_str(DSV4_USER);
                out.push_str(content);
            }
            "assistant" => {
                out.push_str(DSV4_ASSISTANT);
                out.push_str(DSV4_THINK_END);
                out.push_str(content);
                out.push_str(DSV4_EOS);
            }
            other => anyhow::bail!(
                "DeepSeek-V4 template: unsupported role `{other}` (text-only \
                 system/user/assistant in v1; tools ride the DSML format, not yet wired)"
            ),
        }
    }
    ensure!(
        messages.last().map(|m| m.role.trim()) != Some("assistant"),
        "DeepSeek-V4 template: conversation must end with a user/system message"
    );
    out.push_str(DSV4_ASSISTANT);
    out.push_str(DSV4_THINK_END);
    Ok(out)
}

/// Resolve the chat template for a model dir:
/// 1. `tokenizer_config.json` `chat_template` (string, or HF list-of-named —
///    `default` preferred) → [`ChatTemplate::Jinja`];
/// 2. `config.json` `architectures` starting with `DeepseekV4` →
///    [`ChatTemplate::BuiltinDeepseekV4`];
/// 3. otherwise [`ChatTemplate::BuiltinChatMl`] (caller warns).
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

/// `bos_token`/`eos_token` is a bare string or an AddedToken `{content: …}`.
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
            content: Some(content.to_string()),
        }
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
        )
        .unwrap();
        assert_eq!(out, "x");
    }

    #[test]
    fn jinja_raise_exception_surfaces_as_error() {
        let err = render_jinja(
            "{{ raise_exception('bad roles') }}",
            "",
            "",
            &[msg("user", "x")],
        )
        .unwrap_err();
        assert!(err.to_string().contains("render checkpoint chat_template"));
    }

    #[test]
    fn deepseek_v4_single_turn_with_system() {
        let rendered = render_deepseek_v4(&[
            msg("system", "You are a helpful assistant."),
            msg("user", "What's the capital of France?"),
        ])
        .unwrap();
        assert_eq!(
            rendered,
            "<｜begin▁of▁sentence｜>You are a helpful assistant.\
             <｜User｜>What's the capital of France?<｜Assistant｜></think>"
        );
    }

    #[test]
    fn deepseek_v4_multi_turn_appends_eos_per_assistant() {
        let rendered = render_deepseek_v4(&[
            msg("user", "hi"),
            msg("assistant", "hello"),
            msg("user", "again"),
        ])
        .unwrap();
        assert_eq!(
            rendered,
            "<｜begin▁of▁sentence｜><｜User｜>hi<｜Assistant｜></think>hello\
             <｜end▁of▁sentence｜><｜User｜>again<｜Assistant｜></think>"
        );
    }

    #[test]
    fn deepseek_v4_rejects_trailing_assistant_and_odd_roles() {
        assert!(render_deepseek_v4(&[msg("user", "q"), msg("assistant", "a")]).is_err());
        assert!(render_deepseek_v4(&[msg("tool", "x")]).is_err());
        assert!(render_deepseek_v4(&[msg("user", "q"), msg("system", "late")]).is_err());
    }

    #[test]
    fn chatml_fallback_unchanged() {
        let rendered = render_chatml(&[msg("user", "hi")]).unwrap();
        assert_eq!(
            rendered,
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        );
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
                    content: Some("be brief".into()),
                },
                ChatMessage {
                    role: "user".into(),
                    content: Some("hi".into()),
                },
            ],
        )
        .unwrap();
        assert!(out.contains("<|im_start|>user\nhi<|im_end|>"), "got: {out}");
        assert!(out.ends_with("<|im_start|>assistant\n"), "got: {out}");
    }
}
