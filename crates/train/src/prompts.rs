//! Prompt loading utilities for OPD examples.

use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
};

use serde::Deserialize;
use thiserror::Error;
use tokenizers::Tokenizer;

#[derive(Debug, Clone)]
pub struct LoadedPromptSets {
    pub train: Vec<Vec<u32>>,
    pub heldout: Vec<Vec<u32>>,
    pub prompt_file: PathBuf,
    pub tokenizer_path: PathBuf,
    pub jsonl_rows: usize,
    pub default_max_tokens: usize,
    pub truncated_rows: usize,
}

#[derive(Debug, Error)]
pub enum PromptLoadError {
    #[error("prompt max_tokens must be positive, got {0}")]
    InvalidDefaultMaxTokens(usize),
    #[error("heldout prompt count must be positive, got {0}")]
    InvalidHeldoutCount(usize),
    #[error("missing tokenizer.json at {0}")]
    MissingTokenizer(PathBuf),
    #[error("failed to load tokenizer {path}: {message}")]
    TokenizerLoad { path: PathBuf, message: String },
    #[error("failed to open prompts file {path}: {source}")]
    OpenPromptFile {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to read prompts file {path} line {line}: {source}")]
    ReadPromptLine {
        path: PathBuf,
        line: usize,
        source: std::io::Error,
    },
    #[error("invalid JSON in prompts file {path} line {line}: {source}")]
    InvalidPromptJson {
        path: PathBuf,
        line: usize,
        source: serde_json::Error,
    },
    #[error("prompts file {path} line {line} has empty text")]
    EmptyPromptText { path: PathBuf, line: usize },
    #[error("prompts file {path} line {line} has non-positive max_tokens {max_tokens}")]
    InvalidRowMaxTokens {
        path: PathBuf,
        line: usize,
        max_tokens: usize,
    },
    #[error("tokenizer encode failed for prompts file {path} line {line}: {message}")]
    TokenizePrompt {
        path: PathBuf,
        line: usize,
        message: String,
    },
    #[error("tokenizer produced no tokens for prompts file {path} line {line}")]
    EmptyTokenizedPrompt { path: PathBuf, line: usize },
    #[error(
        "prompts file {path} produced {count} prompts, need more than heldout_count={heldout_count} for 1+ train prompt + heldout split"
    )]
    NotEnoughPrompts {
        path: PathBuf,
        count: usize,
        heldout_count: usize,
    },
}

#[derive(Debug, Deserialize)]
struct JsonlPrompt {
    text: String,
    #[serde(default)]
    max_tokens: Option<usize>,
}

pub fn load_jsonl_prompt_sets(
    model_dir: &Path,
    prompt_file: &Path,
    default_max_tokens: usize,
    heldout_count: usize,
) -> Result<LoadedPromptSets, PromptLoadError> {
    if default_max_tokens == 0 {
        return Err(PromptLoadError::InvalidDefaultMaxTokens(default_max_tokens));
    }
    if heldout_count == 0 {
        return Err(PromptLoadError::InvalidHeldoutCount(heldout_count));
    }

    let tokenizer_path = model_dir.join("tokenizer.json");
    if !tokenizer_path.is_file() {
        return Err(PromptLoadError::MissingTokenizer(tokenizer_path));
    }
    let tokenizer =
        Tokenizer::from_file(&tokenizer_path).map_err(|err| PromptLoadError::TokenizerLoad {
            path: tokenizer_path.clone(),
            message: err.to_string(),
        })?;

    let file = File::open(prompt_file).map_err(|source| PromptLoadError::OpenPromptFile {
        path: prompt_file.to_path_buf(),
        source,
    })?;
    let reader = BufReader::new(file);
    let mut prompts = Vec::new();
    let mut jsonl_rows = 0usize;
    let mut truncated_rows = 0usize;

    for (idx, line) in reader.lines().enumerate() {
        let line_no = idx + 1;
        let line = line.map_err(|source| PromptLoadError::ReadPromptLine {
            path: prompt_file.to_path_buf(),
            line: line_no,
            source,
        })?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        jsonl_rows += 1;
        let record = serde_json::from_str::<JsonlPrompt>(trimmed).map_err(|source| {
            PromptLoadError::InvalidPromptJson {
                path: prompt_file.to_path_buf(),
                line: line_no,
                source,
            }
        })?;
        if record.text.trim().is_empty() {
            return Err(PromptLoadError::EmptyPromptText {
                path: prompt_file.to_path_buf(),
                line: line_no,
            });
        }
        let max_tokens = record.max_tokens.unwrap_or(default_max_tokens);
        if max_tokens == 0 {
            return Err(PromptLoadError::InvalidRowMaxTokens {
                path: prompt_file.to_path_buf(),
                line: line_no,
                max_tokens,
            });
        }

        let encoding = tokenizer
            .encode(record.text.as_str(), false)
            .map_err(|err| PromptLoadError::TokenizePrompt {
                path: prompt_file.to_path_buf(),
                line: line_no,
                message: err.to_string(),
            })?;
        let mut ids = encoding.get_ids().to_vec();
        if ids.is_empty() {
            return Err(PromptLoadError::EmptyTokenizedPrompt {
                path: prompt_file.to_path_buf(),
                line: line_no,
            });
        }
        if ids.len() > max_tokens {
            ids.truncate(max_tokens);
            truncated_rows += 1;
        }
        prompts.push(ids);
    }

    if prompts.len() <= heldout_count {
        return Err(PromptLoadError::NotEnoughPrompts {
            path: prompt_file.to_path_buf(),
            count: prompts.len(),
            heldout_count,
        });
    }

    let split_at = prompts.len() - heldout_count;
    let heldout = prompts.split_off(split_at);
    Ok(LoadedPromptSets {
        train: prompts,
        heldout,
        prompt_file: prompt_file.to_path_buf(),
        tokenizer_path,
        jsonl_rows,
        default_max_tokens,
        truncated_rows,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::write_wordlevel_tokenizer;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn load_jsonl_prompt_sets_splits_last_rows_as_heldout() {
        let dir = tempdir().expect("tempdir");
        write_wordlevel_tokenizer(
            &dir.path().join("tokenizer.json"),
            ["alpha", "beta", "gamma", "delta", "epsilon"],
            ["<eos>"],
        )
        .expect("tokenizer");
        let prompts = dir.path().join("prompts.jsonl");
        let mut file = File::create(&prompts).expect("create prompts");
        writeln!(file, r#"{{"text":"alpha beta","max_tokens":8}}"#).expect("write");
        writeln!(file, r#"{{"text":"gamma delta","max_tokens":8}}"#).expect("write");
        writeln!(file, r#"{{"text":"epsilon alpha","max_tokens":8}}"#).expect("write");

        let loaded = load_jsonl_prompt_sets(dir.path(), &prompts, 8, 1).expect("load");
        assert_eq!(loaded.train.len(), 2);
        assert_eq!(loaded.heldout.len(), 1);
        assert_eq!(loaded.jsonl_rows, 3);
        assert_eq!(loaded.truncated_rows, 0);
    }

    #[test]
    fn load_jsonl_prompt_sets_truncates_row_max_tokens() {
        let dir = tempdir().expect("tempdir");
        write_wordlevel_tokenizer(
            &dir.path().join("tokenizer.json"),
            ["alpha", "beta", "gamma"],
            ["<eos>"],
        )
        .expect("tokenizer");
        let prompts = dir.path().join("prompts.jsonl");
        let mut file = File::create(&prompts).expect("create prompts");
        writeln!(file, r#"{{"text":"alpha beta gamma","max_tokens":2}}"#).expect("write");
        writeln!(file, r#"{{"text":"gamma beta alpha","max_tokens":3}}"#).expect("write");

        let loaded = load_jsonl_prompt_sets(dir.path(), &prompts, 8, 1).expect("load");
        assert_eq!(loaded.train[0].len(), 2);
        assert_eq!(loaded.heldout[0].len(), 3);
        assert_eq!(loaded.truncated_rows, 1);
    }
}
