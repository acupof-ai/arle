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
    pub train_completions: Vec<Option<Vec<u32>>>,
    pub heldout_completions: Vec<Option<Vec<u32>>>,
    pub prompt_file: PathBuf,
    pub tokenizer_path: PathBuf,
    pub jsonl_rows: usize,
    pub default_max_tokens: usize,
    pub truncated_rows: usize,
    pub completion_rows: usize,
    pub truncated_completion_rows: usize,
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
    #[error("prompts file {path} line {line} needs either text or prompt_ids")]
    MissingPrompt { path: PathBuf, line: usize },
    #[error("prompts file {path} line {line} has empty prompt_ids")]
    EmptyPromptIds { path: PathBuf, line: usize },
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
    #[error("prompts file {path} line {line} has empty completion text")]
    EmptyCompletionText { path: PathBuf, line: usize },
    #[error("prompts file {path} line {line} has empty completion_ids")]
    EmptyCompletionIds { path: PathBuf, line: usize },
    #[error("prompts file {path} line {line} has non-positive completion_max_tokens {max_tokens}")]
    InvalidCompletionMaxTokens {
        path: PathBuf,
        line: usize,
        max_tokens: usize,
    },
    #[error("tokenizer encode failed for prompts file {path} line {line} completion: {message}")]
    TokenizeCompletion {
        path: PathBuf,
        line: usize,
        message: String,
    },
    #[error("tokenizer produced no completion tokens for prompts file {path} line {line}")]
    EmptyTokenizedCompletion { path: PathBuf, line: usize },
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
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    prompt_ids: Option<Vec<u32>>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    completion: Option<String>,
    #[serde(default)]
    target: Option<String>,
    #[serde(default)]
    completion_ids: Option<Vec<u32>>,
    #[serde(default)]
    completion_max_tokens: Option<usize>,
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
    let mut completions = Vec::new();
    let mut jsonl_rows = 0usize;
    let mut truncated_rows = 0usize;
    let mut completion_rows = 0usize;
    let mut truncated_completion_rows = 0usize;

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
        let max_tokens = record.max_tokens.unwrap_or(default_max_tokens);
        if max_tokens == 0 {
            return Err(PromptLoadError::InvalidRowMaxTokens {
                path: prompt_file.to_path_buf(),
                line: line_no,
                max_tokens,
            });
        }

        let mut ids = if let Some(prompt_ids) = record.prompt_ids {
            if prompt_ids.is_empty() {
                return Err(PromptLoadError::EmptyPromptIds {
                    path: prompt_file.to_path_buf(),
                    line: line_no,
                });
            }
            prompt_ids
        } else if let Some(text) = record.text.as_ref() {
            if text.trim().is_empty() {
                return Err(PromptLoadError::EmptyPromptText {
                    path: prompt_file.to_path_buf(),
                    line: line_no,
                });
            }
            let encoding = tokenizer.encode(text.as_str(), false).map_err(|err| {
                PromptLoadError::TokenizePrompt {
                    path: prompt_file.to_path_buf(),
                    line: line_no,
                    message: err.to_string(),
                }
            })?;
            let ids = encoding.get_ids().to_vec();
            if ids.is_empty() {
                return Err(PromptLoadError::EmptyTokenizedPrompt {
                    path: prompt_file.to_path_buf(),
                    line: line_no,
                });
            }
            ids
        } else {
            return Err(PromptLoadError::MissingPrompt {
                path: prompt_file.to_path_buf(),
                line: line_no,
            });
        };
        if ids.len() > max_tokens {
            ids.truncate(max_tokens);
            truncated_rows += 1;
        }
        prompts.push(ids);

        if let Some(mut completion_ids) = record.completion_ids {
            if completion_ids.is_empty() {
                return Err(PromptLoadError::EmptyCompletionIds {
                    path: prompt_file.to_path_buf(),
                    line: line_no,
                });
            }
            let completion_max_tokens = record.completion_max_tokens.unwrap_or(max_tokens);
            if completion_max_tokens == 0 {
                return Err(PromptLoadError::InvalidCompletionMaxTokens {
                    path: prompt_file.to_path_buf(),
                    line: line_no,
                    max_tokens: completion_max_tokens,
                });
            }
            if completion_ids.len() > completion_max_tokens {
                completion_ids.truncate(completion_max_tokens);
                truncated_completion_rows += 1;
            }
            completion_rows += 1;
            completions.push(Some(completion_ids));
        } else if let Some(completion_text) = record.completion.as_ref().or(record.target.as_ref())
        {
            if completion_text.trim().is_empty() {
                return Err(PromptLoadError::EmptyCompletionText {
                    path: prompt_file.to_path_buf(),
                    line: line_no,
                });
            }
            let completion_max_tokens = record.completion_max_tokens.unwrap_or(max_tokens);
            if completion_max_tokens == 0 {
                return Err(PromptLoadError::InvalidCompletionMaxTokens {
                    path: prompt_file.to_path_buf(),
                    line: line_no,
                    max_tokens: completion_max_tokens,
                });
            }
            let completion_encoding =
                tokenizer
                    .encode(completion_text.as_str(), false)
                    .map_err(|err| PromptLoadError::TokenizeCompletion {
                        path: prompt_file.to_path_buf(),
                        line: line_no,
                        message: err.to_string(),
                    })?;
            let mut completion_ids = completion_encoding.get_ids().to_vec();
            if completion_ids.is_empty() {
                return Err(PromptLoadError::EmptyTokenizedCompletion {
                    path: prompt_file.to_path_buf(),
                    line: line_no,
                });
            }
            if completion_ids.len() > completion_max_tokens {
                completion_ids.truncate(completion_max_tokens);
                truncated_completion_rows += 1;
            }
            completion_rows += 1;
            completions.push(Some(completion_ids));
        } else {
            completions.push(None);
        }
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
    let heldout_completions = completions.split_off(split_at);
    Ok(LoadedPromptSets {
        train: prompts,
        heldout,
        train_completions: completions,
        heldout_completions,
        prompt_file: prompt_file.to_path_buf(),
        tokenizer_path,
        jsonl_rows,
        default_max_tokens,
        truncated_rows,
        completion_rows,
        truncated_completion_rows,
    })
}
