use std::path::Path;

use anyhow::{Context, Result, bail};

use crate::args::TrainOpdArgs;

pub(super) struct OpdPromptSource {
    pub(super) train_prompts: Vec<Vec<u32>>,
    pub(super) train_completions: Vec<Option<Vec<u32>>>,
    pub(super) eval_ids: Vec<u32>,
    pub(super) report_prompt_ids: Vec<u32>,
    pub(super) completion_rows: usize,
}

pub(super) fn load_opd_prompt_source(
    args: &TrainOpdArgs,
    student_dir: &Path,
    vocab_size: usize,
) -> Result<OpdPromptSource> {
    if let Some(prompt_file) = args.prompts_file.as_deref() {
        let loaded = train::prompts::load_jsonl_prompt_sets(
            student_dir,
            prompt_file,
            args.prompt_max_tokens,
            1,
        )
        .with_context(|| format!("load OPD prompt corpus from {}", prompt_file.display()))?;
        validate_prompt_collection("--prompts-file train prompt", &loaded.train, vocab_size)?;
        validate_prompt_collection("--prompts-file heldout prompt", &loaded.heldout, vocab_size)?;
        validate_completion_collection(
            "--prompts-file train completion",
            &loaded.train_completions,
            loaded.train.len(),
            vocab_size,
        )?;
        validate_completion_collection(
            "--prompts-file heldout completion",
            &loaded.heldout_completions,
            loaded.heldout.len(),
            vocab_size,
        )?;
        let eval_ids = match args.eval_ids.as_deref() {
            Some(raw) => {
                let ids = parse_prompt_ids(Some(raw))?;
                validate_token_ids("--eval-ids", &ids, vocab_size)?;
                ids
            }
            None => loaded
                .heldout
                .first()
                .cloned()
                .unwrap_or_else(|| loaded.train[0].clone()),
        };
        eprintln!(
            "[arle train opd] loaded prompt corpus {}: train={} heldout={} rows={} truncated={} completion_rows={} truncated_completion_rows={}",
            prompt_file.display(),
            loaded.train.len(),
            loaded.heldout.len(),
            loaded.jsonl_rows,
            loaded.truncated_rows,
            loaded.completion_rows,
            loaded.truncated_completion_rows,
        );
        let report_prompt_ids = loaded.train[0].clone();
        return Ok(OpdPromptSource {
            train_prompts: loaded.train,
            train_completions: loaded.train_completions,
            eval_ids,
            report_prompt_ids,
            completion_rows: loaded.completion_rows,
        });
    }

    let prompt_ids = parse_prompt_ids(args.prompt_ids.as_deref())?;
    validate_token_ids("--prompt-ids", &prompt_ids, vocab_size)?;
    let eval_ids = match args.eval_ids.as_deref() {
        Some(raw) => {
            let ids = parse_prompt_ids(Some(raw))?;
            validate_token_ids("--eval-ids", &ids, vocab_size)?;
            ids
        }
        None => prompt_ids.clone(),
    };
    Ok(OpdPromptSource {
        train_prompts: vec![prompt_ids.clone()],
        train_completions: vec![None],
        eval_ids,
        report_prompt_ids: prompt_ids,
        completion_rows: 0,
    })
}

pub(super) fn parse_prompt_ids(raw: Option<&str>) -> Result<Vec<u32>> {
    let raw = raw.unwrap_or("1,3,8");
    raw.split(',')
        .map(|piece| {
            piece
                .trim()
                .parse::<u32>()
                .with_context(|| format!("invalid prompt id `{piece}` (expected u32)"))
        })
        .collect()
}

fn validate_token_ids(label: &str, ids: &[u32], vocab_size: usize) -> Result<()> {
    if ids.is_empty() {
        bail!("{label} must contain at least one token id");
    }
    if ids.iter().any(|&id| (id as usize) >= vocab_size) {
        bail!("{label} token ids must be < {vocab_size} (student vocab size); got {ids:?}");
    }
    Ok(())
}

fn validate_prompt_collection(label: &str, prompts: &[Vec<u32>], vocab_size: usize) -> Result<()> {
    if prompts.is_empty() {
        bail!("{label} must contain at least one prompt");
    }
    for (idx, prompt) in prompts.iter().enumerate() {
        validate_token_ids(&format!("{label}[{idx}]"), prompt, vocab_size)?;
    }
    Ok(())
}

fn validate_completion_collection(
    label: &str,
    completions: &[Option<Vec<u32>>],
    prompts_len: usize,
    vocab_size: usize,
) -> Result<()> {
    if completions.len() != prompts_len {
        bail!(
            "{label} length {} must match prompt count {prompts_len}",
            completions.len()
        );
    }
    for (idx, completion) in completions.iter().enumerate() {
        if let Some(tokens) = completion {
            validate_token_ids(&format!("{label}[{idx}]"), tokens, vocab_size)?;
        }
    }
    Ok(())
}
