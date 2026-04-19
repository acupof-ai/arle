//! Convert common public SFT dataset schemas to the `messages` format
//! consumed by [`sft_data::load_jsonl`].
//!
//! The canonical on-disk schema for `train_sft` is one JSON object per
//! line, shaped like:
//!
//! ```json
//! {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
//! ```
//!
//! Most public instruction-tuning datasets ship in a different shape.
//! This module converts each line-oriented record into the canonical
//! `messages` form. Unsupported / malformed lines are reported with
//! their line number and skipped.
//!
//! Supported input formats:
//!
//! | Format     | Input schema                                                    |
//! |------------|-----------------------------------------------------------------|
//! | `chat`     | `{"messages": [...]}` — passthrough                             |
//! | `dolly`    | `{"instruction", "context"?, "response"}` — Databricks Dolly-15k |
//! | `alpaca`   | `{"instruction", "input"?, "output"}` — Stanford Alpaca style   |
//! | `sharegpt` | `{"conversations": [{"from", "value"}, ...]}` — ShareGPT/Vicuna |
//!
//! ShareGPT role mapping: `human` → `user`, `gpt`/`chatgpt`/`assistant` →
//! `assistant`, `system` → `system`. Unknown roles are dropped.

use std::{
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    path::Path,
};

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputFormat {
    Chat,
    Dolly,
    Alpaca,
    ShareGpt,
}

impl InputFormat {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "chat" => Ok(Self::Chat),
            "dolly" => Ok(Self::Dolly),
            "alpaca" => Ok(Self::Alpaca),
            "sharegpt" => Ok(Self::ShareGpt),
            other => Err(anyhow!(
                "unknown format '{other}' (expected: chat, dolly, alpaca, sharegpt)"
            )),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct ChatRecord {
    messages: Vec<Message>,
}

#[derive(Debug, Deserialize)]
struct DollyRecord {
    instruction: String,
    #[serde(default)]
    context: String,
    response: String,
}

#[derive(Debug, Deserialize)]
struct AlpacaRecord {
    instruction: String,
    #[serde(default)]
    input: String,
    output: String,
}

#[derive(Debug, Deserialize)]
struct ShareGptTurn {
    from: String,
    value: String,
}

#[derive(Debug, Deserialize)]
struct ShareGptRecord {
    conversations: Vec<ShareGptTurn>,
}

#[derive(Debug, Deserialize)]
struct ChatPassthrough {
    messages: Vec<Message>,
}

pub struct ConvertStats {
    pub total_lines: usize,
    pub written: usize,
    pub skipped: usize,
}

pub fn convert_file(input: &Path, output: &Path, format: InputFormat) -> Result<ConvertStats> {
    if paths_alias(input, output) {
        return Err(anyhow!(
            "--input and --output resolve to the same file ({}); in-place conversion would truncate the source before reading",
            input.display()
        ));
    }
    let reader = BufReader::new(
        File::open(input).with_context(|| format!("open input {}", input.display()))?,
    );
    let mut writer = BufWriter::new(
        File::create(output).with_context(|| format!("create output {}", output.display()))?,
    );

    let mut stats = ConvertStats {
        total_lines: 0,
        written: 0,
        skipped: 0,
    };

    for (line_index, line) in reader.lines().enumerate() {
        let line =
            line.with_context(|| format!("read line {} of {}", line_index + 1, input.display()))?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        stats.total_lines += 1;

        let messages = match convert_line(trimmed, format) {
            Ok(m) => m,
            Err(err) => {
                eprintln!(
                    "[data_adapter] skipping line {} ({input}): {err}",
                    line_index + 1,
                    input = input.display()
                );
                stats.skipped += 1;
                continue;
            }
        };

        if messages.is_empty() {
            stats.skipped += 1;
            continue;
        }

        let record = ChatRecord { messages };
        serde_json::to_writer(&mut writer, &record)?;
        writer.write_all(b"\n")?;
        stats.written += 1;
    }

    writer.flush()?;
    Ok(stats)
}

fn paths_alias(a: &Path, b: &Path) -> bool {
    match (a.canonicalize(), b.canonicalize()) {
        (Ok(ca), Ok(cb)) => ca == cb,
        _ => a == b,
    }
}

fn convert_line(line: &str, format: InputFormat) -> Result<Vec<Message>> {
    match format {
        InputFormat::Chat => {
            let rec: ChatPassthrough = serde_json::from_str(line)?;
            Ok(rec.messages)
        }
        InputFormat::Dolly => {
            let rec: DollyRecord = serde_json::from_str(line)?;
            let user = if rec.context.trim().is_empty() {
                rec.instruction
            } else {
                format!("{}\n\n{}", rec.instruction, rec.context)
            };
            Ok(vec![
                Message {
                    role: "user".to_string(),
                    content: user,
                },
                Message {
                    role: "assistant".to_string(),
                    content: rec.response,
                },
            ])
        }
        InputFormat::Alpaca => {
            let rec: AlpacaRecord = serde_json::from_str(line)?;
            let user = if rec.input.trim().is_empty() {
                rec.instruction
            } else {
                format!("{}\n\n{}", rec.instruction, rec.input)
            };
            Ok(vec![
                Message {
                    role: "user".to_string(),
                    content: user,
                },
                Message {
                    role: "assistant".to_string(),
                    content: rec.output,
                },
            ])
        }
        InputFormat::ShareGpt => {
            let rec: ShareGptRecord = serde_json::from_str(line)?;
            let mut messages = Vec::with_capacity(rec.conversations.len());
            for turn in rec.conversations {
                let role = match turn.from.as_str() {
                    "human" | "user" => "user",
                    "gpt" | "chatgpt" | "assistant" => "assistant",
                    "system" => "system",
                    _ => continue,
                };
                messages.push(Message {
                    role: role.to_string(),
                    content: turn.value,
                });
            }
            Ok(messages)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dolly_with_context() {
        let line = r#"{"instruction":"Q","context":"C","response":"A"}"#;
        let msgs = convert_line(line, InputFormat::Dolly).unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, "user");
        assert_eq!(msgs[0].content, "Q\n\nC");
        assert_eq!(msgs[1].role, "assistant");
        assert_eq!(msgs[1].content, "A");
    }

    #[test]
    fn dolly_without_context() {
        let line = r#"{"instruction":"Q","context":"","response":"A"}"#;
        let msgs = convert_line(line, InputFormat::Dolly).unwrap();
        assert_eq!(msgs[0].content, "Q");
    }

    #[test]
    fn alpaca_with_input() {
        let line = r#"{"instruction":"Q","input":"I","output":"A"}"#;
        let msgs = convert_line(line, InputFormat::Alpaca).unwrap();
        assert_eq!(msgs[0].content, "Q\n\nI");
        assert_eq!(msgs[1].content, "A");
    }

    #[test]
    fn sharegpt_role_mapping() {
        let line = r#"{"conversations":[{"from":"human","value":"hi"},{"from":"gpt","value":"yo"},{"from":"unknown","value":"x"}]}"#;
        let msgs = convert_line(line, InputFormat::ShareGpt).unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, "user");
        assert_eq!(msgs[1].role, "assistant");
    }

    #[test]
    fn chat_passthrough() {
        let line = r#"{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"hello"}]}"#;
        let msgs = convert_line(line, InputFormat::Chat).unwrap();
        assert_eq!(msgs.len(), 2);
    }
}
