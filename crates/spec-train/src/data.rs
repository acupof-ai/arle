//! JSONL conversations to tokenized [`Sample`]s.
//!
//! Input is one `{"id": .., "conversations": [{"role": .., "content": ..}]}`
//! per line — what `scripts/spec_train_data.py` writes after regenerating the
//! assistant turns through our own target.

use anyhow::{Result, ensure};
use chat::{ChatMlMessage, render_chatml_with_spans};
use serde::Deserialize;
use std::path::Path;
use tokenizers::Tokenizer;

use crate::trainer::Sample;

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

pub fn read_jsonl(path: &Path) -> Result<Vec<Conversation>> {
    std::fs::read_to_string(path)?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| Ok(serde_json::from_str(l)?))
        .collect()
}

/// Read a conversations JSONL and tokenize it with the model's own tokenizer.
pub fn load_samples(data: &Path, tokenizer: &Path, max_len: usize) -> Result<Vec<Sample>> {
    let tk = Tokenizer::from_file(tokenizer)
        .map_err(|e| anyhow::anyhow!("load tokenizer {}: {e}", tokenizer.display()))?;
    read_jsonl(data)?
        .iter()
        .map(|c| to_sample(c, &tk, max_len))
        .collect()
}

/// Tokenize one conversation and mark the assistant turns as trainable.
///
/// The mask is derived from token byte offsets against the rendered turn
/// spans, so it stays correct whatever the tokenizer does to the boundaries.
pub fn to_sample(conv: &Conversation, tokenizer: &Tokenizer, max_len: usize) -> Result<Sample> {
    ensure!(
        !conv.conversations.is_empty(),
        "conversation {} is empty",
        conv.id
    );
    let messages: Vec<ChatMlMessage<'_>> = conv
        .conversations
        .iter()
        .map(|m| ChatMlMessage {
            role: &m.role,
            content: &m.content,
        })
        .collect();
    let rendered = render_chatml_with_spans(&messages, false);

    let supervised: Vec<_> = rendered
        .spans
        .iter()
        .zip(&conv.conversations)
        .filter(|(_, m)| m.role == "assistant")
        .map(|(s, _)| s.supervised.clone())
        .collect();

    let encoding = tokenizer
        .encode(rendered.prompt.as_str(), false)
        .map_err(|e| anyhow::anyhow!("tokenize conversation {}: {e}", conv.id))?;
    let take = encoding.len().min(max_len);
    let loss_mask = encoding.get_offsets()[..take]
        .iter()
        .map(|&(start, end)| {
            start < end && supervised.iter().any(|s| start >= s.start && end <= s.end)
        })
        .collect();

    Ok(Sample {
        input_ids: encoding.get_ids()[..take].to_vec(),
        loss_mask,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;

    fn conv() -> Conversation {
        let msg = |role: &str, content: &str| Message {
            role: role.into(),
            content: content.into(),
        };
        Conversation {
            id: 1,
            conversations: vec![
                msg("user", "ask one"),
                msg("assistant", "reply one"),
                msg("user", "ask two"),
                msg("assistant", "reply two"),
            ],
        }
    }

    /// Whitespace word-level over whatever the render produced, so offsets are
    /// real and the mask is exercised end to end.
    fn tokenizer(text: &str) -> Tokenizer {
        let mut vocab = ahash::AHashMap::new();
        for (i, w) in text.split_whitespace().enumerate() {
            vocab.entry(w.to_string()).or_insert(i as u32);
        }
        vocab.insert("<unk>".into(), vocab.len() as u32);
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("<unk>".into())
            .build()
            .unwrap();
        let mut t = Tokenizer::new(model);
        t.with_pre_tokenizer(Some(Whitespace {}));
        t
    }

    /// Only assistant content is trainable — a trainable prompt token would
    /// anchor blocks on context the draft never sees at serve time.
    #[test]
    fn the_mask_covers_the_assistant_turns_only() {
        let c = conv();
        let messages: Vec<ChatMlMessage<'_>> = c
            .conversations
            .iter()
            .map(|m| ChatMlMessage {
                role: &m.role,
                content: &m.content,
            })
            .collect();
        let rendered = render_chatml_with_spans(&messages, false);
        let tk = tokenizer(&rendered.prompt);
        let sample = to_sample(&c, &tk, 4096).unwrap();

        let encoding = tk.encode(rendered.prompt.as_str(), false).unwrap();
        let marked: Vec<&str> = encoding
            .get_offsets()
            .iter()
            .zip(&sample.loss_mask)
            .filter(|(_, m)| **m)
            .map(|((s, e), _)| &rendered.prompt[*s..*e])
            .collect();
        assert!(marked.contains(&"reply"), "assistant content not marked");
        assert!(!marked.contains(&"ask"), "user content was marked");
        assert_eq!(sample.input_ids.len(), sample.loss_mask.len());
        assert!(sample.loss_mask.iter().any(|&m| m));
    }

    #[test]
    fn truncation_keeps_ids_and_mask_aligned() {
        let c = conv();
        let messages: Vec<ChatMlMessage<'_>> = c
            .conversations
            .iter()
            .map(|m| ChatMlMessage {
                role: &m.role,
                content: &m.content,
            })
            .collect();
        let tk = tokenizer(&render_chatml_with_spans(&messages, false).prompt);
        let sample = to_sample(&c, &tk, 5).unwrap();
        assert_eq!(sample.input_ids.len(), 5);
        assert_eq!(sample.loss_mask.len(), 5);
    }
}
