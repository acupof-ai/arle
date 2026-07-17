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
    /// Require the dump body's `model` to equal this (the cc harness tags each
    /// sample's model id) — concurrent samples overlap in time, so wall-clock
    /// alone would cross-attribute conversations. `None` = time-only (serial).
    #[serde(default)]
    pub model: Option<String>,
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
    /// Generation-time behavior logprobs, one per MASKED response token in
    /// mask order (= `capture_rollout_logprobs` target order) — from the
    /// serve sidecars' `gen_logprobs`. Empty when any contributing request
    /// lacked capture (greedy, Metal, pre-P6 serve, re-render fallback).
    pub gen_logprobs: Vec<f32>,
    /// Attempt reward carried from the window (SAO advantage input).
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
    let records = convert_cc_dumps(dump_dir, &tokenizer, windows)?;
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

/// In-memory core: dumps under `dump_dir` → one token record per matched
/// window. Unmatched or un-convertible windows are skipped with a note, so the
/// result may be empty (the cc harness treats an all-failed group as
/// trainable-empty, not fatal — the CLI wrapper above enforces non-empty).
pub fn convert_cc_dumps(
    dump_dir: &Path,
    tokenizer: &tokenizers::Tokenizer,
    windows: &[CcWindow],
) -> Result<Vec<CcRecord>> {
    let dumps = list_dumps(dump_dir)?;
    if dumps.is_empty() {
        eprintln!("[cc-convert] no *.json dumps in {}", dump_dir.display());
        return Ok(Vec::new());
    }

    let whole_dir = [CcWindow {
        label: "all".to_owned(),
        t_start_ms: 0,
        t_end_ms: u64::MAX,
        reward: default_reward(),
        model: None,
    }];
    let windows = if windows.is_empty() {
        &whole_dir[..]
    } else {
        windows
    };

    let mut records = Vec::with_capacity(windows.len());
    for window in windows {
        let mut matched = dumps_in_window(&dumps, window)?;
        // Session-final request = the dump with the LARGEST `messages` array
        // (each CC turn resends the whole conversation).
        let Some(final_idx) = (0..matched.len()).max_by_key(|&i| messages_len(&matched[i].0))
        else {
            eprintln!(
                "[cc-convert] window {} [{}, {}) matched no dump; skipped",
                window.label, window.t_start_ms, window.t_end_ms
            );
            continue;
        };
        // A single un-convertible window (e.g. a failed rollout with no assistant
        // turn — now collected because SAO keeps failing attempts) must not abort
        // the whole round's records; skip it and keep the rest.
        let converted =
            match merged_sidecar_record(&window.label, window.reward, &matched, final_idx) {
                // Serve-written engine tokens: token-exact, no re-render drift.
                Some(record) => Ok(record),
                None => {
                    let (body, _) = matched.swap_remove(final_idx);
                    convert_body(&window.label, window.reward, body, tokenizer)
                }
            };
        match converted {
            Ok(record) => records.push(record),
            Err(err) => eprintln!("[cc-convert] window {}: {err:#}; skipped", window.label),
        }
    }
    Ok(records)
}

/// Newest dump's prompt-portion tokens (everything before the first assistant
/// supervised byte; the whole render when no assistant turn yet). Feeds the
/// post-merge prefix warm-up: one max_tokens=1 prefill re-populates the shared
/// cc prefix the LoRA re-merge's cache flush dropped. `None` = no dump yet.
pub fn newest_dump_prompt_ids(
    dump_dir: &Path,
    tokenizer: &tokenizers::Tokenizer,
) -> Result<Option<Vec<u32>>> {
    let Some((_, path)) = list_dumps(dump_dir)?.into_iter().next_back() else {
        return Ok(None);
    };
    // Serve-written sidecar first: the exact prompt tokens the serve rendered
    // (token-exact for any chat template); ChatML re-render as fallback.
    if let Some(sidecar) = read_tokens_sidecar(&path) {
        return Ok(Some(sidecar.prompt_token_ids));
    }
    let rendered = render_dump(read_json(&path)?)?;
    let cutoff = supervised_spans(&rendered)
        .iter()
        .map(|range| range.start)
        .min()
        .unwrap_or(rendered.prompt.len());
    let encoding = tokenizer
        .encode(rendered.prompt.as_str(), false)
        .map_err(|err| anyhow!("tokenize rendered prompt: {err}"))?;
    let ids = encoding.get_ids();
    let prompt_len = encoding
        .get_offsets()
        .iter()
        .position(|&(_, end)| end > cutoff)
        .unwrap_or(ids.len());
    Ok((prompt_len > 0).then(|| ids[..prompt_len].to_vec()))
}

/// `<epoch_ms>_<seq>.json` files under `dir`, keyed by their epoch prefix.
fn list_dumps(dir: &Path) -> Result<Vec<(u64, PathBuf)>> {
    let mut dumps = Vec::new();
    for entry in fs::read_dir(dir).with_context(|| format!("read dump dir {}", dir.display()))? {
        let path = entry?.path();
        if path.extension().is_none_or(|ext| ext != "json") {
            continue;
        }
        let stem = path.file_stem().and_then(|stem| stem.to_str());
        // `<stem>.tokens.json` sidecars ride beside dumps, not as dumps.
        if stem.is_some_and(|stem| stem.ends_with(".tokens")) {
            continue;
        }
        let epoch_ms = stem
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

/// A window's model-matching dumps in epoch order: `(body, path)`.
fn dumps_in_window(
    dumps: &[(u64, PathBuf)],
    window: &CcWindow,
) -> Result<Vec<(serde_json::Value, PathBuf)>> {
    let mut matched = Vec::new();
    for (_, path) in dumps
        .iter()
        .filter(|(ms, _)| (window.t_start_ms..window.t_end_ms).contains(ms))
    {
        let body = read_json(path)?;
        if window.model.as_deref().is_some_and(|m| body["model"] != m) {
            continue;
        }
        matched.push((body, path.clone()));
    }
    Ok(matched)
}

fn read_json(path: &Path) -> Result<serde_json::Value> {
    let raw = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_str(&raw).with_context(|| format!("parse {}", path.display()))
}

/// Dump body → serve-identical chat request → span-carrying ChatML render.
/// No generation prompt: this is a supervision record, not an inference
/// prompt, and the last turn of the fullest request is user/tool anyway.
fn render_dump(body: serde_json::Value) -> Result<chat::RenderedChatMl> {
    let chat_request = infer_server::messages_body_to_chat_request(body)
        .context("map /v1/messages body onto the chat request")?;
    let messages: Vec<chat::ChatMessage> =
        chat_request.messages.iter().map(to_chat_message).collect();
    ensure!(!messages.is_empty(), "dump has no messages");
    Ok(chat::render_structured_chatml_with_spans(&messages, false))
}

/// Non-empty supervised byte ranges (the assistant turns) of a render.
fn supervised_spans(rendered: &chat::RenderedChatMl) -> Vec<Range<usize>> {
    rendered
        .spans
        .iter()
        .map(|span| span.supervised.clone())
        .filter(|range| !range.is_empty())
        .collect()
}

fn messages_len(body: &serde_json::Value) -> usize {
    body.get("messages")
        .and_then(serde_json::Value::as_array)
        .map_or(0, Vec::len)
}

/// Parse `<dump>.tokens.json` when present and usable (non-empty prompt+gen);
/// anything else falls back to the re-render path.
fn read_tokens_sidecar(dump_path: &Path) -> Option<infer_server::TokensSidecar> {
    let path = infer_server::tokens_sidecar_path(dump_path);
    let raw = fs::read_to_string(&path).ok()?;
    match serde_json::from_str::<infer_server::TokensSidecar>(&raw) {
        Ok(s) if !s.prompt_token_ids.is_empty() && !s.gen_token_ids.is_empty() => Some(s),
        Ok(_) => None,
        Err(err) => {
            eprintln!(
                "[cc-convert] {}: {err}; falling back to re-render",
                path.display()
            );
            None
        }
    }
}

/// Engine-token record with prefix-merged supervision: the final request's
/// tokens ARE the token-exact multi-turn history, and each earlier request's
/// gen segment recurs verbatim inside the final prompt (request i's prompt+gen
/// is a prefix of request i+1's prompt modulo template glue). Mask = union of
/// every located earlier gen segment (monotonic forward scan — turn order) ∪
/// the final gen. Earlier dumps without a sidecar are skipped: an incomplete
/// request's gen never entered the history. `None` (no usable final sidecar,
/// an unlocatable earlier segment — compaction/branching/template drift — or a
/// mask starting at token 0) → the caller re-renders; never a silently
/// narrower mask.
fn merged_sidecar_record(
    label: &str,
    reward: f32,
    matched: &[(serde_json::Value, PathBuf)],
    final_idx: usize,
) -> Option<CcRecord> {
    let final_sidecar = read_tokens_sidecar(&matched[final_idx].1)?;
    let prompt_len = final_sidecar.prompt_token_ids.len();
    let mut mask = vec![0u8; prompt_len];
    let mut cursor = 0usize;
    // Behavior logprobs in mask order (segments located at monotonic cursor
    // positions); all-or-nothing — any uncaptured contributing request drops
    // the whole vector so it can never misalign with the mask.
    let mut logprobs = Some(Vec::new());
    let mut push_logprobs = |lps: &[f32], n_gen: usize| {
        match &mut logprobs {
            Some(all) if lps.len() == n_gen => all.extend_from_slice(lps),
            slot => *slot = None,
        };
    };
    for (i, (_, path)) in matched.iter().enumerate() {
        if i == final_idx {
            continue;
        }
        let Some(earlier) = read_tokens_sidecar(path) else {
            continue;
        };
        let gen_ids = &earlier.gen_token_ids;
        let Some(pos) = find_subsequence(&final_sidecar.prompt_token_ids[cursor..], gen_ids) else {
            eprintln!(
                "[cc-convert] window {label}: earlier gen segment ({} tokens, {}) not in the \
                 final prompt; re-render fallback",
                gen_ids.len(),
                path.display()
            );
            return None;
        };
        let start = cursor + pos;
        mask[start..start + gen_ids.len()].fill(1);
        cursor = start + gen_ids.len();
        push_logprobs(&earlier.gen_logprobs, gen_ids.len());
    }
    push_logprobs(
        &final_sidecar.gen_logprobs,
        final_sidecar.gen_token_ids.len(),
    );
    let first_masked = mask.iter().position(|&m| m == 1).unwrap_or(prompt_len);
    if first_masked == 0 {
        eprintln!("[cc-convert] window {label}: supervised segment at token 0; re-render fallback");
        return None;
    }
    let ids: Vec<u32> = final_sidecar
        .prompt_token_ids
        .iter()
        .chain(&final_sidecar.gen_token_ids)
        .copied()
        .collect();
    mask.resize(ids.len(), 1);
    let mut record = split_record(label, reward, &ids, &mask, first_masked);
    record.gen_logprobs = logprobs.unwrap_or_default();
    Some(record)
}

/// Split at the first supervised token: everything before is prompt, the rest
/// is the (masked) response.
fn split_record(
    label: &str,
    reward: f32,
    ids: &[u32],
    mask: &[u8],
    first_masked: usize,
) -> CcRecord {
    CcRecord {
        label: label.to_owned(),
        prompt_ids: ids[..first_masked].to_vec(),
        response_ids: ids[first_masked..].to_vec(),
        response_mask: mask[first_masked..].to_vec(),
        gen_logprobs: Vec::new(),
        reward,
        masked_tokens: mask.iter().filter(|&&m| m == 1).count(),
        total_tokens: ids.len(),
    }
}

/// First occurrence of `needle` in `haystack` (naive scan: ~20K-token prompt ×
/// few-hundred-token needles × a handful of turns). Callers guarantee a
/// non-empty needle (`read_tokens_sidecar` requires non-empty gen).
fn find_subsequence(haystack: &[u32], needle: &[u32]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// One dump body → one token record: serve-identical request mapping, ChatML
/// render with spans, tokenize with byte offsets, assistant-span mask.
fn convert_body(
    label: &str,
    reward: f32,
    body: serde_json::Value,
    tokenizer: &tokenizers::Tokenizer,
) -> Result<CcRecord> {
    let rendered = render_dump(body)?;
    let supervised = supervised_spans(&rendered);
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

    Ok(split_record(label, reward, ids, &mask, first_masked))
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

    /// A 3-request window (two earlier turns + final) with engine-token
    /// sidecars round-trips to ONE record built from the final sidecar's
    /// tokens whose mask supervises exactly the three gen segments — earlier
    /// gens located by exact subsequence match in the final prompt, template
    /// glue tokens (3, 4) and the leading prompt left unsupervised. The dump
    /// bodies alone have no assistant turn, so only the sidecar path can
    /// produce this record — proving it took precedence over the re-render.
    #[test]
    fn sidecar_round_trips_engine_tokens() {
        let dir = tempfile::tempdir().expect("tempdir");
        let dump = |n: usize| {
            let messages: Vec<String> = (0..n)
                .map(|i| format!(r#"{{"role":"user","content":"m{i}"}}"#))
                .collect();
            format!(
                r#"{{"model":"m","max_tokens":8,"messages":[{}]}}"#,
                messages.join(",")
            )
        };
        let write = |name: &str, contents: &str| {
            std::fs::write(dir.path().join(name), contents).expect("write");
        };
        write("1000_0.json", &dump(1));
        write(
            "1000_0.tokens.json",
            r#"{"prompt_token_ids":[1,2],"gen_token_ids":[10,11],"gen_logprobs":[-0.1,-0.2]}"#,
        );
        write("1001_0.json", &dump(3));
        write(
            "1001_0.tokens.json",
            r#"{"prompt_token_ids":[1,2,10,11,3],"gen_token_ids":[20,21],"gen_logprobs":[-0.3,-0.4]}"#,
        );
        write("1002_0.json", &dump(5));
        write(
            "1002_0.tokens.json",
            r#"{"prompt_token_ids":[1,2,10,11,3,20,21,4],"gen_token_ids":[30],"gen_logprobs":[-0.5]}"#,
        );
        // Never reached: the sidecar path skips tokenization entirely.
        let tokenizer =
            tokenizers::Tokenizer::new(tokenizers::models::wordlevel::WordLevel::default());
        let records = super::convert_cc_dumps(dir.path(), &tokenizer, &[]).expect("convert");
        assert_eq!(records.len(), 1);
        let record = &records[0];
        // Full sequence [1,2,10,11,3,20,21,4,30] splits at the first
        // supervised token; mask = the three gen segments exactly.
        assert_eq!(record.prompt_ids, vec![1, 2]);
        assert_eq!(record.response_ids, vec![10, 11, 3, 20, 21, 4, 30]);
        assert_eq!(record.response_mask, vec![1, 1, 0, 1, 1, 0, 1]);
        assert_eq!((record.masked_tokens, record.total_tokens), (5, 9));
        // Behavior logprobs: one per masked token, in mask order.
        assert_eq!(record.gen_logprobs, vec![-0.1, -0.2, -0.3, -0.4, -0.5]);
    }
}
