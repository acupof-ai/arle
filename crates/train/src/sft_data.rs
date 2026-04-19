use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

use autograd::{AutogradError, Result};
use serde::Deserialize;

use crate::tokenizer::TrainTokenizer;

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct SftExample {
    pub messages: Vec<ChatMessage>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizedSft {
    pub input_ids: Vec<u32>,
    pub labels: Vec<i32>,
}

pub fn load_jsonl(path: &Path) -> Result<Vec<SftExample>> {
    let file = File::open(path).map_err(|err| {
        tape_invariant(format!(
            "failed to open SFT JSONL {}: {err}",
            path.display()
        ))
    })?;
    let reader = BufReader::new(file);
    let mut examples = Vec::new();

    for (line_index, line) in reader.lines().enumerate() {
        let line = line.map_err(|err| {
            tape_invariant(format!(
                "failed to read line {} from {}: {err}",
                line_index + 1,
                path.display()
            ))
        })?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let example = serde_json::from_str::<SftExample>(trimmed).map_err(|err| {
            tape_invariant(format!(
                "failed to parse SFT JSONL line {} from {}: {err}",
                line_index + 1,
                path.display()
            ))
        })?;
        examples.push(example);
    }

    Ok(examples)
}

pub fn tokenize_example(
    example: &SftExample,
    tokenizer: &TrainTokenizer,
    max_seq_len: usize,
) -> Result<TokenizedSft> {
    const IM_END: &str = "<|im_end|>";
    let mut input_ids = Vec::new();
    let mut labels = Vec::new();

    for message in &example.messages {
        // Encode each full turn as ONE string so BPE merges at boundaries
        // (prefix↔content, content↔<|im_end|>) are respected. Using the
        // combined form is what Qwen's chat template actually feeds the model
        // at inference; piecewise-encoded concatenation can diverge.
        let turn = format!(
            "<|im_start|>{}\n{}{IM_END}\n",
            message.role, message.content
        );
        let (turn_ids, offsets) = tokenizer.encode_with_offsets(&turn, false)?;
        let supervise = message.role == "assistant";

        // Locate the assistant body via **byte offsets** in the combined
        // encoding, not piecewise token counts. Encoding the header alone
        // isn't stable — e.g. content that starts with `\n` causes the
        // header-trailing `\n` and the content-leading `\n` to merge into a
        // single token, so the standalone header is one token longer than
        // the header slice of `turn_ids`. Byte offsets remain accurate
        // regardless of merges.
        let header_byte_end = format!("<|im_start|>{}\n", message.role).len();
        let body_byte_end = header_byte_end + message.content.len() + IM_END.len();

        let body_start = if supervise {
            offsets
                .iter()
                .position(|&(s, _)| s >= header_byte_end)
                .unwrap_or(turn_ids.len())
        } else {
            turn_ids.len()
        };
        let body_end = if supervise {
            offsets
                .iter()
                .position(|&(s, _)| s >= body_byte_end)
                .unwrap_or(turn_ids.len())
        } else {
            turn_ids.len()
        };

        for (i, &token_id) in turn_ids.iter().enumerate() {
            input_ids.push(token_id);
            let supervised_here = supervise && i >= body_start && i < body_end;
            labels.push(if supervised_here {
                token_id as i32
            } else {
                -100
            });
        }
    }

    if input_ids.len() > max_seq_len {
        input_ids.truncate(max_seq_len);
        labels.truncate(max_seq_len);
    }

    Ok(TokenizedSft { input_ids, labels })
}

fn tape_invariant(message: String) -> AutogradError {
    AutogradError::TapeInvariant(Box::leak(message.into_boxed_str()))
}
