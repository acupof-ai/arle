//! CC-trajectory converter (`arle train cc-convert`): raw `/v1/messages` dumps
//! (serve `--dump-messages-dir`) → verl-style token records
//! (`prompt_ids`/`response_ids`/`response_mask`) for the agent-OPD masked-CE
//! replay (`agent-opd --replay-records`). Backend-independent, no CUDA.
//!
//! One record PER REQUEST with a token sidecar: prompt = sidecar prompt tokens
//! (mask 0), response = gen tokens (mask 1). Token-exact by construction — both
//! halves come from the engine, no re-render, no prefix matching. Real cc traffic
//! compacts/rewrites history between requests, so the prefix-merge relation
//! genuinely doesn't hold (pod: 100% of windows fell back to re-render).
//!
//! Fallback (window with NO sidecars): the session-final request (largest
//! `messages` array) → the serve's `messages_body_to_chat_request` → ChatML with
//! per-turn supervised byte spans → tokenize with byte offsets → mask = tokens
//! overlapping an assistant turn's supervised span. The serve's own Jinja
//! renderer exposes no spans; for Qwen checkpoints it is ChatML anyway, and the
//! span-carrying renderer here is the same one the training stack supervises
//! against, so replayed records match the TRAIN-side format.

use std::fs;
use std::ops::Range;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, ensure};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize)]
pub struct CcWindow {
    pub label: String,
    pub t_start_ms: u64,
    pub t_end_ms: u64,
    /// Old windows.jsonl (passing-only) lack this field → default 1.0.
    #[serde(default = "default_reward")]
    pub reward: f32,
    /// Timeout/harness error — records marked truncated so the DAPO filter drops them.
    #[serde(default)]
    pub errored: bool,
    /// Match dump body's `model` — concurrent samples overlap in time.
    #[serde(default)]
    pub model: Option<String>,
}

fn default_reward() -> f32 {
    1.0
}

#[derive(Debug, Serialize)]
pub struct CcRecord {
    /// SAO's task key is the prefix before the first `#`.
    pub label: String,
    pub prompt_ids: Vec<u32>,
    pub response_ids: Vec<u32>,
    pub response_mask: Vec<u8>,
    /// One per masked response token, in mask order (= capture_rollout_logprobs target).
    pub gen_logprobs: Vec<f32>,
    pub reward: f32,
    pub truncated: bool,
    pub masked_tokens: usize,
    pub total_tokens: usize,
}

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
        errored: false,
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
        if matched.is_empty() {
            eprintln!(
                "[cc-convert] window {} [{}, {}) matched no dump; skipped",
                window.label, window.t_start_ms, window.t_end_ms
            );
            continue;
        }
        // One record per sidecar-covered request. Earlier turns' gen tokens
        // recur inside later requests' prompts (masked 0), so each turn is
        // supervised exactly once — its own record.
        let before = records.len();
        for (seq, (_, path)) in matched.iter().enumerate() {
            if let Some(sidecar) = read_tokens_sidecar(path) {
                records.push(request_record(
                    &format!("{}#r{seq}", window.label),
                    window.reward,
                    window.errored,
                    sidecar,
                ));
            }
        }
        let covered = records.len() - before;
        if covered > 0 {
            if covered < matched.len() {
                eprintln!(
                    "[cc-convert] window {}: {}/{} requests lack a token sidecar; skipped",
                    window.label,
                    matched.len() - covered,
                    matched.len()
                );
            }
            continue;
        }
        // No sidecars: re-render the session-final request = the dump with the
        // largest `messages` array (each CC turn resends the whole conversation).
        let final_idx = (0..matched.len())
            .max_by_key(|&i| messages_len(&matched[i].0))
            .expect("matched is non-empty");
        let (body, _) = matched.swap_remove(final_idx);
        match convert_body(
            &window.label,
            window.reward,
            window.errored,
            body,
            tokenizer,
        ) {
            Ok(record) => records.push(record),
            Err(err) => eprintln!("[cc-convert] window {}: {err:#}; skipped", window.label),
        }
    }
    Ok(records)
}

fn request_record(
    label: &str,
    reward: f32,
    truncated: bool,
    sidecar: infer_server::TokensSidecar,
) -> CcRecord {
    let n_gen = sidecar.gen_token_ids.len();
    CcRecord {
        label: label.to_owned(),
        total_tokens: sidecar.prompt_token_ids.len() + n_gen,
        prompt_ids: sidecar.prompt_token_ids,
        response_ids: sidecar.gen_token_ids,
        response_mask: vec![1; n_gen],
        gen_logprobs: if sidecar.gen_logprobs.len() == n_gen {
            sidecar.gen_logprobs
        } else {
            Vec::new()
        },
        reward,
        truncated,
        masked_tokens: n_gen,
    }
}

/// Newest dump's prompt tokens (before the first assistant supervised byte).
/// Feeds the post-merge prefix warm-up: re-populate the shared cc prefix the
/// LoRA re-merge's cache flush dropped. `None` = no dump yet.
pub fn newest_dump_prompt_ids(
    dump_dir: &Path,
    tokenizer: &tokenizers::Tokenizer,
) -> Result<Option<Vec<u32>>> {
    let Some((_, path)) = list_dumps(dump_dir)?.into_iter().next_back() else {
        return Ok(None);
    };
    // Serve-written sidecar first (token-exact); ChatML re-render as fallback.
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

fn list_dumps(dir: &Path) -> Result<Vec<(u64, PathBuf)>> {
    let mut dumps = Vec::new();
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        // Callers treat "no dumps" as a skipped conversion.
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(dumps),
        Err(e) => {
            return Err(e).with_context(|| format!("read dump dir {}", dir.display()));
        }
    };
    for entry in entries {
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

fn render_dump(body: serde_json::Value) -> Result<chat::RenderedChatMl> {
    let chat_request = infer_server::messages_body_to_chat_request(body)
        .context("map /v1/messages body onto the chat request")?;
    let messages: Vec<chat::ChatMessage> =
        chat_request.messages.iter().map(to_chat_message).collect();
    ensure!(!messages.is_empty(), "dump has no messages");
    Ok(chat::render_structured_chatml_with_spans(&messages, false))
}

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

fn split_record(
    label: &str,
    reward: f32,
    truncated: bool,
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
        truncated,
        masked_tokens: mask.iter().filter(|&&m| m == 1).count(),
        total_tokens: ids.len(),
    }
}

fn convert_body(
    label: &str,
    reward: f32,
    truncated: bool,
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

    Ok(split_record(
        label,
        reward,
        truncated,
        ids,
        &mask,
        first_masked,
    ))
}

fn to_chat_message(message: &infer_server::ChatMessage) -> chat::ChatMessage {
    chat::ChatMessage {
        role: chat::ChatRole::from(message.role.as_str()),
        content: message.content_text(),
        tool_calls: message
            .tool_calls
            .iter()
            .map(|call| {
                // HF-convention arguments; unparseable falls back raw.
                let arguments = serde_json::from_str(&call.function.arguments)
                    .unwrap_or_else(|_| serde_json::Value::String(call.function.arguments.clone()));
                chat::ToolCall::new(&call.function.name, arguments)
            })
            .collect(),
    }
}

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
