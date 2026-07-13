//! CC-trajectory converter (`arle train cc-convert`): raw `/v1/messages` dumps
//! (serve `--dump-messages-dir`) → verl-style token records
//! (`prompt_ids`/`response_ids`/`response_mask`) for the agent-OPD masked-CE
//! replay (`agent-opd --replay-records`). Backend-independent, no CUDA.
//!
//! Pipeline per attempt window: pick the dump with the LARGEST `messages`
//! array (the session-final request = full conversation) → map it through the
//! serve's own [`infer_server::messages_body_to_chat_request`] → render as
//! ChatML with per-turn supervised byte spans
//! ([`chat::render_structured_chatml_with_spans`]) → tokenize with byte
//! offsets → mask = tokens overlapping an assistant turn's supervised span.
//!
//! Render-truth note: the serve renders through the checkpoint's own Jinja
//! `chat_template` (`infer_server::OpenAiTokenizer::render_chat_full`), which
//! exposes no supervision spans. For Qwen-family checkpoints that template is
//! ChatML; the span-carrying ChatML renderer here is the same one the training
//! stack supervises against, so replayed records match the TRAIN-side format.

use std::fs;
use std::ops::Range;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, ensure};
use serde::{Deserialize, Serialize};

/// One capture window grouping dumps into a single attempt. Dumps whose
/// filename epoch falls in `[t_start_ms, t_end_ms)` belong to the window.
#[derive(Debug, Clone, Deserialize)]
pub struct CcWindow {
    pub label: String,
    pub t_start_ms: u64,
    pub t_end_ms: u64,
    /// Attempt reward (1.0 pass / 0.0 fail). Old windows.jsonl (passing-only)
    /// lack the field → default 1.0 preserves today's passing = reward-1.0 flow.
    #[serde(default = "default_reward")]
    pub reward: f32,
}

fn default_reward() -> f32 {
    1.0
}

/// One converted attempt: a verl-style token record plus mask accounting.
#[derive(Debug, Serialize)]
pub struct CcRecord {
    pub label: String,
    pub prompt_ids: Vec<u32>,
    pub response_ids: Vec<u32>,
    pub response_mask: Vec<u8>,
    /// Attempt reward carried from the window (SAO advantage input). Default
    /// 1.0 keeps replay records from pre-reward windows on the passing-only path.
    #[serde(default = "default_reward")]
    pub reward: f32,
    pub masked_tokens: usize,
    pub total_tokens: usize,
}

/// Convert the dumps under `dump_dir` into one JSONL record per window at
/// `out_path` (no windows = one record over the whole dir). Returns per-window
/// summaries for the caller to log.
pub fn run_cc_convert(
    dump_dir: &Path,
    tokenizer_path: &Path,
    out_path: &Path,
    windows: &[CcWindow],
) -> Result<Vec<CcRecord>> {
    let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
        .map_err(|err| anyhow!("load tokenizer {}: {err}", tokenizer_path.display()))?;
    let dumps = list_dumps(dump_dir)?;
    ensure!(
        !dumps.is_empty(),
        "no *.json dumps in {}",
        dump_dir.display()
    );

    let whole_dir = [CcWindow {
        label: "all".to_owned(),
        t_start_ms: 0,
        t_end_ms: u64::MAX,
        reward: default_reward(),
    }];
    let windows = if windows.is_empty() {
        &whole_dir[..]
    } else {
        windows
    };

    let mut records = Vec::with_capacity(windows.len());
    for window in windows {
        let Some(body) = fullest_dump_in_window(&dumps, window)? else {
            eprintln!(
                "[cc-convert] window {} [{}, {}) matched no dump; skipped",
                window.label, window.t_start_ms, window.t_end_ms
            );
            continue;
        };
        records.push(
            convert_body(&window.label, window.reward, body, &tokenizer)
                .with_context(|| format!("convert window {}", window.label))?,
        );
    }
    ensure!(
        !records.is_empty(),
        "no window matched any dump in {}",
        dump_dir.display()
    );

    let mut out = String::new();
    for record in &records {
        out.push_str(&serde_json::to_string(record)?);
        out.push('\n');
    }
    fs::write(out_path, out).with_context(|| format!("write {}", out_path.display()))?;
    Ok(records)
}

/// `<epoch_ms>_<seq>.json` files under `dir`, keyed by their epoch prefix.
fn list_dumps(dir: &Path) -> Result<Vec<(u64, PathBuf)>> {
    let mut dumps = Vec::new();
    for entry in fs::read_dir(dir).with_context(|| format!("read dump dir {}", dir.display()))? {
        let path = entry?.path();
        if path.extension().is_none_or(|ext| ext != "json") {
            continue;
        }
        let epoch_ms = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .and_then(|stem| stem.split('_').next())
            .and_then(|prefix| prefix.parse::<u64>().ok());
        match epoch_ms {
            Some(ms) => dumps.push((ms, path)),
            None => eprintln!(
                "[cc-convert] {} has no <epoch_ms>_ prefix; skipped",
                path.display()
            ),
        }
    }
    dumps.sort();
    Ok(dumps)
}

/// The session-final request of a window = the dump with the LARGEST
/// `messages` array (each CC turn resends the whole conversation).
fn fullest_dump_in_window(
    dumps: &[(u64, PathBuf)],
    window: &CcWindow,
) -> Result<Option<serde_json::Value>> {
    let mut best: Option<(usize, serde_json::Value)> = None;
    for (_, path) in dumps
        .iter()
        .filter(|(ms, _)| (window.t_start_ms..window.t_end_ms).contains(ms))
    {
        let raw = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        let body: serde_json::Value =
            serde_json::from_str(&raw).with_context(|| format!("parse {}", path.display()))?;
        let len = body
            .get("messages")
            .and_then(serde_json::Value::as_array)
            .map_or(0, Vec::len);
        if best.as_ref().is_none_or(|(best_len, _)| len > *best_len) {
            best = Some((len, body));
        }
    }
    Ok(best.map(|(_, body)| body))
}

/// One dump body → one token record: serve-identical request mapping, ChatML
/// render with spans, tokenize with byte offsets, assistant-span mask.
fn convert_body(
    label: &str,
    reward: f32,
    body: serde_json::Value,
    tokenizer: &tokenizers::Tokenizer,
) -> Result<CcRecord> {
    let chat_request = infer_server::messages_body_to_chat_request(body)
        .context("map /v1/messages body onto the chat request")?;
    let messages: Vec<chat::ChatMessage> =
        chat_request.messages.iter().map(to_chat_message).collect();
    ensure!(!messages.is_empty(), "dump has no messages");
    // No generation prompt: this is a supervision record, not an inference
    // prompt, and the last turn of the fullest request is user/tool anyway.
    let rendered = chat::render_structured_chatml_with_spans(&messages, false);
    let supervised: Vec<Range<usize>> = rendered
        .spans
        .iter()
        .map(|span| span.supervised.clone())
        .filter(|range| !range.is_empty())
        .collect();
    ensure!(
        !supervised.is_empty(),
        "no assistant turns to supervise in the rendered conversation"
    );

    let encoding = tokenizer
        .encode(rendered.prompt.as_str(), false)
        .map_err(|err| anyhow!("tokenize rendered prompt: {err}"))?;
    let ids = encoding.get_ids();
    let offsets = encoding.get_offsets();
    ensure!(!ids.is_empty(), "rendered prompt tokenized to zero tokens");
    // Sanity: offsets must cover the rendered prompt (byte offsets contract).
    let last_end = offsets.last().map_or(0, |&(_, end)| end);
    ensure!(
        last_end <= rendered.prompt.len(),
        "token offsets exceed the rendered prompt ({last_end} > {})",
        rendered.prompt.len()
    );

    let mask = mask_from_offsets(offsets, &supervised);
    let first_masked = mask
        .iter()
        .position(|&m| m == 1)
        .ok_or_else(|| anyhow!("supervised spans matched no tokens"))?;
    ensure!(
        first_masked > 0,
        "first token is supervised — the record would have an empty prompt"
    );

    let masked_tokens = mask.iter().filter(|&&m| m == 1).count();
    Ok(CcRecord {
        label: label.to_owned(),
        prompt_ids: ids[..first_masked].to_vec(),
        response_ids: ids[first_masked..].to_vec(),
        response_mask: mask[first_masked..].to_vec(),
        reward,
        masked_tokens,
        total_tokens: ids.len(),
    })
}

/// Map the serve's OpenAI-shaped message onto the `chat` crate's structured
/// message (the shape the span renderer supervises).
fn to_chat_message(message: &infer_server::ChatMessage) -> chat::ChatMessage {
    chat::ChatMessage {
        role: chat::ChatRole::from(message.role.as_str()),
        content: message.content_text(),
        tool_calls: message
            .tool_calls
            .iter()
            .map(|call| {
                // HF-convention arguments mapping; unparseable falls back raw.
                let arguments = serde_json::from_str(&call.function.arguments)
                    .unwrap_or_else(|_| serde_json::Value::String(call.function.arguments.clone()));
                chat::ToolCall::new(&call.function.name, arguments)
            })
            .collect(),
    }
}

/// Byte spans → per-token mask: a token is supervised (1) iff its byte range
/// overlaps any supervised span.
fn mask_from_offsets(offsets: &[(usize, usize)], supervised: &[Range<usize>]) -> Vec<u8> {
    offsets
        .iter()
        .map(|&(start, end)| {
            u8::from(
                supervised
                    .iter()
                    .any(|span| start < span.end && end > span.start),
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::mask_from_offsets;
    use chat::{ChatMessage, render_structured_chatml_with_spans};

    /// End-to-end mask over a rendered user→assistant→tool→assistant
    /// conversation with a synthetic byte-per-token tokenization: exactly the
    /// assistant supervised bytes get mask=1.
    #[test]
    fn assistant_spans_map_to_token_mask() {
        let rendered = render_structured_chatml_with_spans(
            &[
                ChatMessage::user("fix the bug"),
                ChatMessage::assistant("looking", vec![]),
                ChatMessage::tool_result("shell", "grep output"),
                ChatMessage::assistant("done", vec![]),
            ],
            false,
        );
        let supervised: Vec<_> = rendered
            .spans
            .iter()
            .map(|span| span.supervised.clone())
            .filter(|range| !range.is_empty())
            .collect();
        assert_eq!(supervised.len(), 2, "two assistant turns supervised");

        // Fake tokenizer: one token per byte of the rendered prompt.
        let offsets: Vec<(usize, usize)> = (0..rendered.prompt.len()).map(|i| (i, i + 1)).collect();
        let mask = mask_from_offsets(&offsets, &supervised);

        let masked: usize = mask.iter().map(|&m| usize::from(m)).sum();
        let supervised_bytes: usize = supervised.iter().map(std::ops::Range::len).sum();
        assert_eq!(masked, supervised_bytes);
        // Every masked byte sits inside an assistant supervised span; the user
        // turn (span 0) and tool turn contribute none.
        for (i, &m) in mask.iter().enumerate() {
            let inside = supervised.iter().any(|span| span.contains(&i));
            assert_eq!(m == 1, inside, "byte {i}");
        }
        // Prompt split point: everything before the first masked token is
        // prompt — the whole user turn precedes it.
        let first = mask.iter().position(|&m| m == 1).expect("has masked");
        assert!(first > 0);
        assert!(rendered.prompt[..first].contains("fix the bug"));
    }

    /// Partial-overlap rule: a token straddling a span boundary counts.
    #[test]
    fn straddling_token_is_supervised() {
        let mask = mask_from_offsets(&[(0, 4), (4, 8), (8, 12)], &[6..10]);
        assert_eq!(mask, vec![0, 1, 1]);
    }
}
