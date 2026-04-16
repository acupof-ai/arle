//! Progress-aware model download from HuggingFace Hub.
//!
//! Wraps `hf-hub`'s sync API with `indicatif` multi-progress bars for a
//! polished download experience.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use console::style;
use hf_hub::api::Progress;
use hf_hub::{Repo, RepoType, api::sync::ApiRepo};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

/// Download a model from HuggingFace Hub with per-file progress bars.
///
/// Returns the local cache directory containing all downloaded files.
pub(crate) fn download_model_with_progress(model_id: &str) -> Result<PathBuf> {
    let api = infer::hf_hub::build_api().context("failed to initialise HuggingFace API")?;
    let repo = api.repo(Repo::new(model_id.to_string(), RepoType::Model));

    let info = repo
        .info()
        .with_context(|| format!("failed to fetch repo info for '{model_id}'"))?;

    let filenames: Vec<String> = info.siblings.iter().map(|s| s.rfilename.clone()).collect();

    eprintln!(
        "  {} {}",
        style("downloading").cyan().bold(),
        style(model_id).bold()
    );
    eprintln!();

    let mp = MultiProgress::new();

    // ── mandatory files ──────────────────────────────────────────────────
    let mandatory = ["config.json", "tokenizer.json", "tokenizer_config.json"];
    for name in &mandatory {
        if filenames.iter().any(|f| f == name) {
            fetch_with_bar(&repo, name, model_id, &mp)?;
        }
    }

    // ── optional config files ────────────────────────────────────────────
    let optional = [
        "special_tokens_map.json",
        "generation_config.json",
        "vocab.json",
        "merges.txt",
        "model.safetensors.index.json",
    ];
    for name in &optional {
        if filenames.iter().any(|f| f == name) {
            fetch_with_bar(&repo, name, model_id, &mp)?;
        }
    }

    // ── weight shards ────────────────────────────────────────────────────
    let weight_files: Vec<&str> = filenames
        .iter()
        .filter(|f| {
            let p = Path::new(f.as_str());
            let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");
            (ext == "safetensors" || ext == "bin")
                && !f.contains("adapter")
                && !(ext == "bin" && has_safetensors_twin(&filenames, f))
        })
        .map(String::as_str)
        .collect();

    if weight_files.is_empty() {
        bail!("no weight files found in HF repo '{model_id}'");
    }

    for name in &weight_files {
        fetch_with_bar(&repo, name, model_id, &mp)?;
    }

    // ── resolve cache dir ────────────────────────────────────────────────
    let first = mandatory
        .iter()
        .find(|name| filenames.iter().any(|f| f == *name))
        .unwrap_or(&"config.json");

    let file_path = repo
        .get(first)
        .with_context(|| format!("failed to resolve cache path for '{first}'"))?;

    let cache_dir = file_path
        .parent()
        .map_or_else(|| file_path.clone(), Path::to_path_buf);

    eprintln!();
    eprintln!(
        "  {} {}",
        style("ready").green().bold(),
        style(cache_dir.display()).dim()
    );
    eprintln!();

    Ok(cache_dir)
}

fn fetch_with_bar(
    repo: &ApiRepo,
    filename: &str,
    model_id: &str,
    mp: &MultiProgress,
) -> Result<PathBuf> {
    let pb = mp.add(ProgressBar::new(0));
    pb.set_style(
        ProgressStyle::with_template(
            "  {prefix:>30}  {bar:25.cyan/dim} {percent:>3}%  {bytes}/{total_bytes}  {bytes_per_sec}",
        )
        .unwrap()
        .progress_chars("━╸─"),
    );

    // Truncate filename for display
    let display_name = if filename.len() > 28 {
        format!("...{}", &filename[filename.len() - 25..])
    } else {
        filename.to_string()
    };
    pb.set_prefix(display_name);

    let progress = IndicatifProgress { bar: pb.clone() };

    let result = repo
        .download_with_progress(filename, progress)
        .with_context(|| format!("failed to download '{filename}' from '{model_id}'"));

    pb.finish();
    result
}

#[derive(Clone)]
struct IndicatifProgress {
    bar: ProgressBar,
}

impl Progress for IndicatifProgress {
    fn init(&mut self, size: usize, _filename: &str) {
        self.bar.set_length(size as u64);
    }

    fn update(&mut self, size: usize) {
        self.bar.inc(size as u64);
    }

    fn finish(&mut self) {
        self.bar.finish();
    }
}

fn has_safetensors_twin(all: &[String], bin_file: &str) -> bool {
    let stem = bin_file.strip_suffix(".bin").unwrap_or(bin_file);
    let twin = format!("{stem}.safetensors");
    all.iter().any(|f| f == &twin)
}
