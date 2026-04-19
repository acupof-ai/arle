use std::{
    fs::File,
    io::{BufRead, BufReader},
    iter,
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
    let end_ids = tokenizer.encode("<|im_end|>", false)?;
    let newline_ids = tokenizer.encode("\n", false)?;
    let mut input_ids = Vec::new();
    let mut labels = Vec::new();

    for message in &example.messages {
        let prefix = format!("<|im_start|>{}\n", message.role);
        let prefix_ids = tokenizer.encode(&prefix, false)?;
        let content_ids = tokenizer.encode(&message.content, false)?;
        let supervise = message.role == "assistant";

        input_ids.extend_from_slice(&prefix_ids);
        labels.extend(iter::repeat_n(-100, prefix_ids.len()));

        input_ids.extend_from_slice(&content_ids);
        labels.extend(
            content_ids
                .iter()
                .map(|&token_id| if supervise { token_id as i32 } else { -100 }),
        );

        input_ids.extend_from_slice(&end_ids);
        labels.extend(
            end_ids
                .iter()
                .map(|&token_id| if supervise { token_id as i32 } else { -100 }),
        );

        input_ids.extend_from_slice(&newline_ids);
        labels.extend(iter::repeat_n(-100, newline_ids.len()));
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
