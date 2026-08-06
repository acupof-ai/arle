//! JSONL conversations to tokenized [`Sample`]s.
//!
//! Input is one `{"id": .., "conversations": [{"role": .., "content": ..}]}`
//! per line — what `scripts/spec_train_data.py` writes after regenerating the
//! assistant turns through our own target, with each turn's reasoning already
//! wrapped back into `<think>`/`</think>`.
//!
//! One conversation yields ONE sample, rendered exactly as the serve renders it
//! at the moment it generates the conversation's last assistant turn, and only
//! that turn is supervised. The checkpoint template strips the reasoning from
//! every assistant turn before the last user query, so an earlier turn's text
//! in this rendering is not what the trunk emitted there and training on it
//! would be off-distribution.

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

/// What a sample must fit for the trainer to consume it.
#[derive(Debug, Clone, Copy)]
pub struct Limits {
    /// The trunk's `vocab_size` — the width of the logit row every target id
    /// indexes through `gather_last_dim`.
    pub vocab_size: usize,
    /// The draft's `mask_token_id`, which every non-anchor row carries.
    pub mask_token_id: u32,
    pub max_len: usize,
}

/// Read a conversations JSONL and tokenize it with the model's own tokenizer.
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
    for (i, line) in raw.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let at = format!("{}:{}", data.display(), i + 1);
        let conv: Conversation =
            serde_json::from_str(line).with_context(|| format!("{at}: parse conversation"))?;
        let sample = to_sample(&conv, &tk, limits).with_context(|| at.clone())?;
        ensure!(
            !anchor_candidates(&sample.loss_mask).is_empty(),
            "{at}: no two consecutive supervised positions survive max_len {} — the sample \
             can anchor no block",
            limits.max_len
        );
        samples.push(sample);
    }
    ensure!(
        !samples.is_empty(),
        "{} holds no conversations",
        data.display()
    );
    Ok(samples)
}

/// Tokenize one conversation and mark its last assistant turn as trainable.
///
/// The mask is derived from token byte offsets against the generated span, so
/// it stays correct whatever the tokenizer does to the boundaries — and a token
/// that crosses a boundary is an error, because supervising it would supervise
/// prompt bytes and dropping it would lose a generated token.
pub fn to_sample(conv: &Conversation, tokenizer: &Tokenizer, limits: Limits) -> Result<Sample> {
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
    // The supervised turn is last, so max_len cuts the answer, never the prompt.
    ensure!(
        take == encoding.len() || offsets[take - 1].1 >= generated.end,
        "conversation {}: max_len {} cuts the supervised turn — raise --max-len to {} \
         or the trainer sees a fragment of the answer",
        conv.id,
        limits.max_len,
        encoding.len()
    );
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

    Ok(Sample {
        input_ids: encoding.get_ids()[..take].to_vec(),
        loss_mask,
    })
}

struct Rendered {
    text: String,
    /// Bytes the trunk generated for the last assistant turn — everything after
    /// the generation prompt, `<|im_end|>` included. Exactly what the draft has
    /// to predict at serve.
    generated: Range<usize>,
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

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::decoders::fuse::Fuse;
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
    use tokenizers::tokenizer::SplitDelimiterBehavior;

    fn conv() -> Conversation {
        let msg = |role: &str, content: &str| Message {
            role: role.into(),
            content: content.into(),
        };
        Conversation {
            id: 1,
            conversations: vec![
                msg("user", "ask one"),
                msg("assistant", "<think>\nfirst thought\n</think>\n\nreply one"),
                msg("user", "ask two"),
                msg(
                    "assistant",
                    "<think>\nsecond thought\n</think>\n\nreply two",
                ),
            ],
        }
    }

    fn limits(max_len: usize) -> Limits {
        Limits {
            vocab_size: 1024,
            mask_token_id: 0,
            max_len,
        }
    }

    /// Every char its own token, decoded by concatenation: offsets tile the
    /// rendered text exactly, so the mask is exercised against real boundaries
    /// and the masked ids decode back to a literal substring.
    fn tokenizer(text: &str) -> Tokenizer {
        let mut vocab = ahash::AHashMap::new();
        for c in text.chars() {
            let next = vocab.len() as u32;
            vocab.entry(c.to_string()).or_insert(next);
        }
        vocab.insert("<unk>".into(), vocab.len() as u32);
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("<unk>".into())
            .build()
            .unwrap();
        let mut t = Tokenizer::new(model);
        t.with_pre_tokenizer(Some(
            Split::new(
                SplitPattern::Regex(r"\s|\S".into()),
                SplitDelimiterBehavior::Isolated,
                false,
            )
            .unwrap(),
        ));
        t.with_decoder(Some(Fuse::new()));
        t
    }

    /// The render must be what the trunk actually prefills, or the draft trains
    /// on a context the serve never produces. The expected string is the output
    /// of `models/ThinkingCap-Qwen3.6-27B-FP8/chat_template.jinja` (and of
    /// `models/Qwen3.5-0.8B` `tokenizer_config.json` `chat_template`, which
    /// renders this conversation identically) on the same messages: history
    /// reasoning stripped, the final turn carrying its `<think>` block.
    #[test]
    fn renders_what_the_checkpoint_template_renders() {
        let r = render(&conv()).unwrap();
        assert_eq!(
            r.text,
            "<|im_start|>user\nask one<|im_end|>\n\
             <|im_start|>assistant\nreply one<|im_end|>\n\
             <|im_start|>user\nask two<|im_end|>\n\
             <|im_start|>assistant\n<think>\nsecond thought\n</think>\n\nreply two<|im_end|>\n"
        );
        // The generation prompt is everything up to and including `<think>\n`,
        // so the supervised bytes are exactly the decode region.
        assert_eq!(
            &r.text[r.generated.clone()],
            "second thought\n</think>\n\nreply two<|im_end|>"
        );

        // No reasoning is the thinking-off form: the template still emits the
        // empty block, but the prompt runs through it, so only the answer was
        // generated.
        let plain = Conversation {
            id: 2,
            conversations: vec![
                Message {
                    role: "user".into(),
                    content: "ask".into(),
                },
                Message {
                    role: "assistant".into(),
                    content: "answer".into(),
                },
            ],
        };
        let r = render(&plain).unwrap();
        assert!(
            r.text
                .ends_with("assistant\n<think>\n\n</think>\n\nanswer<|im_end|>\n")
        );
        assert_eq!(&r.text[r.generated.clone()], "answer<|im_end|>");
    }

    /// The one gate that catches a mask landing off the assistant turn: the
    /// masked ids must decode to the generated text and nothing else.
    #[test]
    fn the_loss_mask_decodes_to_exactly_the_generated_turn() {
        let dir = std::env::temp_dir().join(format!("spec-train-data-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (data, tok_path) = (dir.join("train.jsonl"), dir.join("tokenizer.json"));
        let c = conv();
        let text = render(&c).unwrap().text;
        tokenizer(&text).save(&tok_path, false).unwrap();
        std::fs::write(
            &data,
            serde_json::json!({
                "id": 1,
                "conversations": c.conversations.iter()
                    .map(|m| serde_json::json!({"role": m.role, "content": m.content}))
                    .collect::<Vec<_>>(),
            })
            .to_string()
                + "\n",
        )
        .unwrap();

        let samples = load_samples(&data, &tok_path, limits(4096)).unwrap();
        let tk = Tokenizer::from_file(&tok_path).unwrap();
        let marked: Vec<u32> = samples[0]
            .input_ids
            .iter()
            .zip(&samples[0].loss_mask)
            .filter(|(_, m)| **m)
            .map(|(&id, _)| id)
            .collect();
        assert!(!marked.is_empty());
        assert_eq!(
            tk.decode(&marked, false).unwrap(),
            "second thought\n</think>\n\nreply two<|im_end|>"
        );

        // Truncation that leaves the assistant turn out is an error, not a
        // sample the trainer silently skips.
        let err = load_samples(&data, &tok_path, limits(8)).unwrap_err();
        assert!(err.to_string().contains("train.jsonl:1"), "{err}");
        assert!(
            format!("{err:#}").contains("cuts the supervised turn"),
            "{err:#}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// An id past the trunk's vocab reaches `gather_last_dim` and reads out of
    /// bounds; catching it at load is the only cheap place.
    #[test]
    fn a_tokenizer_wider_than_the_trunk_is_rejected() {
        let dir = std::env::temp_dir().join(format!("spec-train-vocab-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (data, tok_path) = (dir.join("train.jsonl"), dir.join("tokenizer.json"));
        let text = render(&conv()).unwrap().text;
        tokenizer(&text).save(&tok_path, false).unwrap();
        std::fs::write(&data, "").unwrap();

        let err = load_samples(
            &data,
            &tok_path,
            Limits {
                vocab_size: 4,
                ..limits(4096)
            },
        )
        .unwrap_err();
        assert!(err.to_string().contains("vocab_size 4"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
