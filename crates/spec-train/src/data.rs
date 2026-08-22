//! JSONL conversations to tokenized [`Sample`]s.
//!
//! Input is one `{"id": .., "conversations": [{"role": .., "content": ..}]}`
//! per line — what `scripts/spec_train_data.py` writes after regenerating the
//! assistant turns through our own target, with each turn's reasoning already
//! wrapped back into `<think>`/`</think>`.
//!
//! One conversation yields ONE sample, rendered exactly as the serve renders it
//! at the moment it generates the conversation's last assistant turn, and only
//! that turn is supervised. DeepSpec supervises every assistant turn
//! (`deepspec/data/parser.py:114-138`) because its non-thinking render puts each
//! turn after the same prefix the target sampled it under; our template deletes
//! every history turn's reasoning, so those answers no longer follow the
//! reasoning that produced them and supervising them would train a conditional
//! the trunk never emits.

use anyhow::{Context, Result, anyhow, bail, ensure};
use serde::Deserialize;
use std::ops::Range;
use std::path::Path;
use tokenizers::Tokenizer;

use crate::block::anchor_candidates;
use crate::trainer::Sample;

const IM_START: &str = "<|im_start|>";
const IM_END: &str = "<|im_end|>";
const THINK_START: &str = "<think>";
const THINK_END: &str = "</think>";
/// What the template emits between the reasoning and the answer.
const CLOSER: &str = "\n</think>\n\n";
/// DeepSpec `scripts/data/prepare_target_cache.py --min-loss-tokens`, counted
/// the same way: supervised tokens left after truncation.
const MIN_SUPERVISED: usize = 14;

#[derive(Debug, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct Conversation {
    #[serde(default)]
    pub id: u64,
    pub conversations: Vec<Message>,
}

#[derive(Debug, Clone, Copy)]
pub struct Limits {
    /// The trunk's `vocab_size` — the width of the logit row every target id
    /// indexes through `gather_last_dim`.
    pub vocab_size: usize,
    /// The draft's `mask_token_id`, which every non-anchor row carries.
    pub mask_token_id: u32,
    pub max_len: usize,
}

pub fn load_samples(data: &Path, tokenizer: &Path, limits: Limits) -> Result<Vec<Sample>> {
    ensure!(
        limits.max_len > 1,
        "max_len {} anchors nothing",
        limits.max_len
    );
    let tk = Tokenizer::from_file(tokenizer)
        .map_err(|e| anyhow!("load tokenizer {}: {e}", tokenizer.display()))?;
    let top_id = tk.get_vocab(true).values().copied().max().unwrap_or(0) as usize;
    ensure!(
        top_id < limits.vocab_size,
        "tokenizer {} reaches id {top_id}, past the trunk's vocab_size {} — those ids \
         index off the end of the logits",
        tokenizer.display(),
        limits.vocab_size
    );
    ensure!(
        (limits.mask_token_id as usize) < limits.vocab_size,
        "mask_token_id {} is outside the trunk's vocab_size {}",
        limits.mask_token_id,
        limits.vocab_size
    );

    let raw = std::fs::read_to_string(data).with_context(|| format!("read {}", data.display()))?;
    let mut samples = Vec::new();
    let (mut truncated, mut cut, mut short, mut unsup) = (0usize, 0usize, 0usize, 0usize);
    for (i, line) in raw.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let at = format!("{}:{}", data.display(), i + 1);
        let conv: Conversation =
            serde_json::from_str(line).with_context(|| format!("{at}: parse conversation"))?;
        match to_sample(&conv, &tk, limits).with_context(|| at)? {
            Outcome::Kept {
                sample,
                truncated: t,
            } => {
                truncated += usize::from(t);
                samples.push(sample);
            }
            Outcome::Cut => cut += 1,
            Outcome::Short => short += 1,
            Outcome::Unsupervised => unsup += 1,
        }
    }
    println!(
        "{}: {} samples, {truncated} truncated at max_len {}, {cut} dropped (max_len cut the \
         supervised turn), {short} dropped (under {MIN_SUPERVISED} supervised tokens), \
         {unsup} dropped (no trailing assistant turn)",
        data.display(),
        samples.len(),
        limits.max_len,
    );
    ensure!(
        !samples.is_empty(),
        "{} yields no usable sample: {cut} dropped where max_len {} cut the supervised turn, \
         {short} dropped under {MIN_SUPERVISED} supervised tokens, {unsup} dropped with no \
         trailing assistant turn",
        data.display(),
        limits.max_len
    );
    Ok(samples)
}

pub enum Outcome {
    /// `truncated` marks a row shortened to `max_len` with its supervision intact.
    Kept { sample: Sample, truncated: bool },
    /// `max_len` ate supervised tokens.
    Cut,
    /// Under [`MIN_SUPERVISED`] supervised tokens, or no two consecutive ones to
    /// anchor a block.
    Short,
    /// No trailing assistant turn to supervise. Raw prompt corpora carry these;
    /// regeneration cannot, since it ends every row on a generated turn.
    Unsupervised,
}

/// Tokenize one conversation and mark its last assistant turn as trainable.
///
/// The mask is derived from token byte offsets against the generated span, so
/// it stays correct whatever the tokenizer does to the boundaries — and a token
/// that crosses a boundary is an error, because supervising it would supervise
/// prompt bytes and dropping it would lose a generated token.
pub fn to_sample(conv: &Conversation, tokenizer: &Tokenizer, limits: Limits) -> Result<Outcome> {
    if !ends_in_a_supervisable_turn(conv) {
        return Ok(Outcome::Unsupervised);
    }
    let rendered = render(conv)?;
    let encoding = tokenizer
        .encode(rendered.text.as_str(), false)
        .map_err(|e| anyhow!("tokenize conversation {}: {e}", conv.id))?;
    let offsets = encoding.get_offsets();
    ensure!(
        offsets.iter().any(|&(s, e)| s < e),
        "conversation {}: the tokenizer reports no byte offsets, so the loss mask cannot \
         be placed",
        conv.id
    );

    let generated = &rendered.generated;
    let take = encoding.len().min(limits.max_len);
    let truncated = take < encoding.len();
    // The supervised turn is last, so max_len cuts the answer, never the prompt:
    // a cut leaves a fragment the trunk never decoded, so the row goes.
    if truncated && offsets[take - 1].1 < generated.end {
        return Ok(Outcome::Cut);
    }
    let mut loss_mask = Vec::with_capacity(take);
    for &(start, end) in &offsets[..take] {
        let inside = start >= generated.start && end <= generated.end;
        let outside = end <= generated.start || start >= generated.end;
        ensure!(
            inside || outside,
            "conversation {}: token bytes {start}..{end} straddle the generated span \
             {}..{} — the tokenizer and the chat template disagree",
            conv.id,
            generated.start,
            generated.end
        );
        loss_mask.push(inside && start < end);
    }

    if loss_mask.iter().filter(|&&m| m).count() < MIN_SUPERVISED
        || anchor_candidates(&loss_mask).is_empty()
    {
        return Ok(Outcome::Short);
    }
    Ok(Outcome::Kept {
        sample: Sample {
            input_ids: encoding.get_ids()[..take].to_vec(),
            loss_mask,
        },
        truncated,
    })
}

struct Rendered {
    text: String,
    /// Bytes the trunk generated for the last assistant turn — everything after
    /// the generation prompt, `<|im_end|>` included. Exactly what the draft has
    /// to predict at serve.
    generated: Range<usize>,
}

/// The shape [`render`] supervises: a final assistant turn after the last user
/// one. Checked before rendering so a corpus row of any other shape is counted
/// and skipped rather than killing the load.
fn ends_in_a_supervisable_turn(conv: &Conversation) -> bool {
    let msgs = &conv.conversations;
    let Some(last_query) = msgs.iter().rposition(|m| m.role == "user") else {
        return false;
    };
    msgs.len() - 1 > last_query && msgs[msgs.len() - 1].role == "assistant"
}

/// The Qwen3.5 / Qwen3.6 chat template for the shapes this corpus has (an
/// optional leading system turn, then user/assistant, no tools, no vision).
/// Byte-checked against both checkpoints' own `chat_template` by
/// `renders_what_the_checkpoint_template_renders`.
fn render(conv: &Conversation) -> Result<Rendered> {
    let msgs = &conv.conversations;
    ensure!(!msgs.is_empty(), "conversation {} is empty", conv.id);
    let last_query = msgs
        .iter()
        .rposition(|m| m.role == "user")
        .ok_or_else(|| anyhow!("conversation {} has no user turn", conv.id))?;
    let last = msgs.len() - 1;
    ensure!(
        msgs[last].role == "assistant" && last > last_query,
        "conversation {} must end in the assistant turn it supervises",
        conv.id
    );

    let mut text = String::new();
    let mut generated = 0..0;
    for (i, m) in msgs.iter().enumerate() {
        let content = m.content.trim();
        match m.role.as_str() {
            "system" => {
                ensure!(
                    i == 0,
                    "conversation {}: a system turn must come first",
                    conv.id
                );
                push_turn(&mut text, "system", content);
            }
            "user" => push_turn(&mut text, "user", content),
            // Before the last user query the template drops the reasoning.
            "assistant" if i <= last_query => {
                push_turn(&mut text, "assistant", split_think(content).1)
            }
            "assistant" => {
                let (reasoning, answer) = split_think(content);
                text.push_str(IM_START);
                text.push_str("assistant\n");
                text.push_str(THINK_START);
                text.push('\n');
                // Thinking-on stops the generation prompt here; with no reasoning
                // the turn is the thinking-off form, whose prompt runs through the
                // closer, so only the answer was generated.
                generated.start = text.len()
                    + if reasoning.is_empty() {
                        CLOSER.len()
                    } else {
                        0
                    };
                text.push_str(reasoning);
                text.push_str(CLOSER);
                text.push_str(answer);
                text.push_str(IM_END);
                generated.end = text.len();
                text.push('\n');
            }
            other => bail!("conversation {}: unexpected role {other}", conv.id),
        }
    }
    Ok(Rendered { text, generated })
}

fn push_turn(text: &mut String, role: &str, content: &str) {
    text.push_str(IM_START);
    text.push_str(role);
    text.push('\n');
    text.push_str(content);
    text.push_str(IM_END);
    text.push('\n');
}

/// The template's own reasoning split: everything after the last `<think>` in
/// the head before the FIRST closer, and the answer after the LAST closer.
fn split_think(content: &str) -> (&str, &str) {
    let Some(first) = content.find(THINK_END) else {
        return ("", content);
    };
    let head = content[..first].trim_end_matches('\n');
    let reasoning = head.rsplit(THINK_START).next().unwrap_or(head);
    let last = content.rfind(THINK_END).unwrap_or(first);
    (
        reasoning.trim(),
        content[last + THINK_END.len()..].trim_start_matches('\n'),
    )
}
