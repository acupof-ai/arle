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
    let trailing_nl_len = tokenizer.encode("\n", false)?.len();
    let mut input_ids = Vec::new();
    let mut labels = Vec::new();

    for message in &example.messages {
        // Encode each full turn as ONE string so BPE merges at boundaries
        // (prefix↔content, content↔<|im_end|>) are respected. Using the
        // combined form is what Qwen's chat template actually feeds the model
        // at inference; piecewise-encoded concatenation can diverge.
        let turn = format!(
            "<|im_start|>{}\n{}<|im_end|>\n",
            message.role, message.content
        );
        let turn_ids = tokenizer.encode(&turn, false)?;
        let supervise = message.role == "assistant";

        // Locate assistant body span within the combined encoding. The prefix
        // ends with a special `<|im_start|>...\n` token boundary, which BPE
        // will not merge across, so encoding the header alone gives a stable
        // token count for the header portion of `turn_ids`.
        let header = format!("<|im_start|>{}\n", message.role);
        let header_len = tokenizer.encode(&header, false)?.len();
        let body_end = turn_ids.len().saturating_sub(trailing_nl_len);

        for (i, &token_id) in turn_ids.iter().enumerate() {
            input_ids.push(token_id);
            let supervised_here = supervise && i >= header_len && i < body_end;
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
