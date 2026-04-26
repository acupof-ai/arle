//! HuggingFace Hub dataset file downloader.
//!
//! Thin wrapper over `hf-hub`'s sync API that resolves a
//! `(repo_id, filename)` pair from a Dataset repo into a local cached
//! path under `~/.cache/huggingface/hub/datasets--<org>--<repo>/…`.
//!
//! Auth follows the same contract as `infer::hf_hub`: reads `HF_TOKEN`
//! from env if present; otherwise relies on `~/.cache/huggingface/token`
//! written by `huggingface-cli login`.
//!
//! This is deliberately minimal: no tokenization, no decompression, no
//! streaming. Callers chain the returned path into `sft_data::load_jsonl`
//! (or their own loader). That keeps the dataset layer orthogonal to
//! the tokenization + training loop so new formats plug in piecewise.
//!
//! Intended wire-up:
//!
//! ```bash
//! DATA=$(arle data download --repo allenai/tulu-3-sft-mixture \
//!        --file data/train.jsonl)
//! arle train sft --data "$DATA" ...
//! ```

use std::path::PathBuf;

use anyhow::{Context, Result};
use hf_hub::{
    Repo, RepoType,
    api::sync::{Api, ApiBuilder},
};

/// Resolve `filename` inside the HF dataset repo `repo_id` to a local
/// cache path, downloading if necessary. Subsequent calls are served
/// from cache — no re-download.
pub fn download_dataset_file(repo_id: &str, filename: &str) -> Result<PathBuf> {
    let api = build_api().context("failed to initialise HuggingFace API")?;
    let repo = api.repo(Repo::new(repo_id.to_string(), RepoType::Dataset));
    repo.get(filename)
        .with_context(|| format!("failed to download '{filename}' from dataset '{repo_id}'"))
}

fn build_api() -> Result<Api> {
    let mut builder = ApiBuilder::new();
    if let Ok(token) = std::env::var("HF_TOKEN")
        && !token.is_empty()
    {
        builder = builder.with_token(Some(token));
    }
    builder.build().context("ApiBuilder::build failed")
}
