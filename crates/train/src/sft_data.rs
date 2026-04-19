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
        //
        // A token belongs to the body if its span `[s, e)` overlaps
        // `[header_byte_end, body_byte_end)`. Using `e > header_byte_end`
        // (rather than `s >= header_byte_end`) catches the boundary-crossing
        // token that merges the header's trailing `\n` with the content's
        // leading `\n` — otherwise that token gets dropped from supervision.
        let header_byte_end = format!("<|im_start|>{}\n", message.role).len();
        let body_byte_end = header_byte_end + message.content.len() + IM_END.len();
        let (body_start, body_end) = if supervise {
            body_span(&offsets, header_byte_end, body_byte_end, turn_ids.len())
        } else {
            (turn_ids.len(), turn_ids.len())
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

/// Find the token-index range `[body_start, body_end)` whose byte spans
/// overlap `[header_byte_end, body_byte_end)`. A token counts as body if its
/// span ends past `header_byte_end` (picks up tokens that straddle the
/// header/body boundary — e.g. a merged `\n\n` when content begins with a
/// newline) and its start is strictly inside the body range.
fn body_span(
    offsets: &[(usize, usize)],
    header_byte_end: usize,
    body_byte_end: usize,
    total_tokens: usize,
) -> (usize, usize) {
    let start = offsets
        .iter()
        .position(|&(_, e)| e > header_byte_end)
        .unwrap_or(total_tokens);
    let end = offsets
        .iter()
        .position(|&(s, _)| s >= body_byte_end)
        .unwrap_or(total_tokens);
    (start, end.max(start))
}

fn tape_invariant(message: String) -> AutogradError {
    AutogradError::TapeInvariant(Box::leak(message.into_boxed_str()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn body_span_simple_non_merged() {
        // `<|im_start|>assistant\n` + `hello` + `<|im_end|>` + `\n`
        // header_byte_end = 22, body_byte_end = 37
        let offsets = vec![(0, 22), (22, 27), (27, 37), (37, 38)];
        let (s, e) = body_span(&offsets, 22, 37, offsets.len());
        assert_eq!((s, e), (1, 3));
    }

    #[test]
    fn body_span_includes_boundary_crossing_token() {
        // Simulates Qwen BPE merging the header-trailing `\n` with the
        // content-leading `\n` into a single `\n\n` token whose span straddles
        // the header/body boundary (Codex finding on 5004b4c).
        // turn = `<|im_start|>assistant\n\nHello<|im_end|>\n`
        // header_byte_end = 22, body_byte_end = 22 + 6 + 10 = 38.
        let offsets = vec![
            (0, 12),  // <|im_start|>
            (12, 21), // assistant
            (21, 23), // \n\n merged — straddles the boundary at 22
            (23, 28), // Hello
            (28, 38), // <|im_end|>
            (38, 39), // trailing \n
        ];
        let (s, e) = body_span(&offsets, 22, 38, offsets.len());
        // Must include the merged-\n\n token at index 2 (its end=23 > 22),
        // not advance to `Hello` at 3.
        assert_eq!((s, e), (2, 5));
    }

    #[test]
    fn body_span_empty_content() {
        // turn = `<|im_start|>assistant\n<|im_end|>\n` — content is empty.
        // header_byte_end = body_byte_end - 10 (just <|im_end|>).
        let offsets = vec![(0, 22), (22, 32), (32, 33)];
        let (s, e) = body_span(&offsets, 22, 32, offsets.len());
        // <|im_end|> at (22, 32) straddles body start; should be included.
        assert_eq!((s, e), (1, 2));
    }
}
