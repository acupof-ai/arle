use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use hf_hub::{
    Repo, RepoType,
    api::sync::{ApiBuilder, ApiRepo},
};

/// Resolve a model id or filesystem path to a local directory containing the
/// model's `config.json`, safetensors weights, and `tokenizer.json` (HF cache
/// snapshots colocate all three). Downloads from HuggingFace if not found
/// locally. Exposed so the `infer-api` adapter can resolve the same directory
/// for both the executor weights and the `OpenAiTokenizer`.
pub fn resolve_model_path(model_id_or_path: &str) -> Result<PathBuf> {
    if let Some(local) = resolve_local_weighted_model_path(model_id_or_path) {
        log::info!("Using local model path: {}", local.display());
        return Ok(local);
    }

    if looks_like_local_path(model_id_or_path) {
        anyhow::bail!(
            "model path does not exist: '{}' (looks like a filesystem path, skipping HuggingFace Hub lookup)",
            model_id_or_path
        );
    }

    log::info!("Model not found locally -- downloading from HuggingFace: {model_id_or_path}");
    download_from_hub(model_id_or_path)
}

fn resolve_local_weighted_model_path(model_id_or_path: &str) -> Option<PathBuf> {
    let local = Path::new(model_id_or_path);
    if is_weighted_model_dir(local) {
        return Some(local.to_path_buf());
    }

    local_model_search_candidates(model_id_or_path)
        .into_iter()
        .find(|candidate| is_weighted_model_dir(candidate))
}

fn download_from_hub(model_id: &str) -> Result<PathBuf> {
    let mut builder = ApiBuilder::new();
    if let Ok(token) = std::env::var("HF_TOKEN")
        && !token.is_empty()
    {
        builder = builder.with_token(Some(token));
    }
    let api = builder
        .build()
        .context("failed to initialise HuggingFace API")?;
    let repo = api.repo(Repo::new(model_id.to_string(), RepoType::Model));
    let info = repo
        .info()
        .with_context(|| format!("failed to fetch repo info for '{model_id}'"))?;
    let filenames: Vec<String> = info.siblings.iter().map(|s| s.rfilename.clone()).collect();

    for name in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "model.safetensors.index.json",
    ] {
        if filenames.iter().any(|file| file == name) {
            fetch_file(&repo, name, model_id)?;
        }
    }

    let weight_files: Vec<&str> = filenames
        .iter()
        .filter(|file| {
            Path::new(file.as_str())
                .extension()
                .is_some_and(|ext| ext == "safetensors")
                && !file.contains("adapter")
        })
        .map(String::as_str)
        .collect();
    anyhow::ensure!(
        !weight_files.is_empty(),
        "no safetensors files found in HF repo '{model_id}'"
    );
    for name in &weight_files {
        fetch_file(&repo, name, model_id)?;
    }

    let first = filenames
        .iter()
        .find(|file| file.as_str() == "config.json")
        .map_or("config.json", String::as_str);
    let file_path = repo
        .get(first)
        .with_context(|| format!("failed to resolve cache path for '{first}'"))?;
    Ok(file_path
        .parent()
        .map_or_else(|| file_path.clone(), Path::to_path_buf))
}

fn fetch_file(repo: &ApiRepo, filename: &str, model_id: &str) -> Result<PathBuf> {
    log::info!("  download {filename}");
    repo.get(filename)
        .with_context(|| format!("failed to download '{filename}' from '{model_id}'"))
}

fn local_model_search_candidates(model_id_or_path: &str) -> Vec<PathBuf> {
    let model_name = model_id_or_path
        .rsplit('/')
        .next()
        .unwrap_or(model_id_or_path)
        .trim();

    let mut candidates: Vec<PathBuf> = common_local_roots()
        .into_iter()
        .map(|base| base.join(model_name))
        .collect();

    if let Some((org, repo)) = model_id_or_path.split_once('/') {
        let cache_repo_dir = huggingface_hub_root().join(format!("models--{org}--{repo}"));
        candidates.extend(snapshot_dirs(&cache_repo_dir));
        candidates.push(cache_repo_dir.join("snapshots").join("main"));
    }

    candidates
}

fn common_local_roots() -> Vec<PathBuf> {
    let cwd_roots = std::env::current_dir()
        .ok()
        .into_iter()
        .flat_map(|cwd| [cwd.join("models"), cwd.join("infer").join("models")]);
    let home_roots = home_dir().into_iter().flat_map(|home| {
        [
            home.join("models"),
            home.join("llm"),
            home.join(".cache").join("mlx-models"),
        ]
    });
    cwd_roots.chain(home_roots).collect()
}

fn huggingface_hub_root() -> PathBuf {
    if let Ok(hf_home) = std::env::var("HF_HOME")
        && !hf_home.trim().is_empty()
    {
        return PathBuf::from(hf_home).join("hub");
    }
    home_dir()
        .map(|home| home.join(".cache").join("huggingface").join("hub"))
        .unwrap_or_else(|| PathBuf::from(".cache/huggingface/hub"))
}

fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

fn snapshot_dirs(cache_repo_dir: &Path) -> Vec<PathBuf> {
    let snapshot_root = cache_repo_dir.join("snapshots");
    let Ok(entries) = std::fs::read_dir(&snapshot_root) else {
        return Vec::new();
    };
    let mut snapshots: Vec<PathBuf> = entries
        .filter_map(std::result::Result::ok)
        .map(|entry| entry.path())
        .collect();
    snapshots.sort_unstable_by(|a, b| b.cmp(a));
    snapshots
}

fn is_weighted_model_dir(path: &Path) -> bool {
    if !path.is_dir() || !path.join("config.json").exists() {
        return false;
    }
    let Ok(entries) = std::fs::read_dir(path) else {
        return false;
    };
    entries.filter_map(std::result::Result::ok).any(|entry| {
        let p = entry.path();
        p.is_file() && p.extension().is_some_and(|e| e == "safetensors")
    })
}

fn looks_like_local_path(input: &str) -> bool {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return false;
    }
    trimmed.starts_with('/')
        || trimmed.starts_with("./")
        || trimmed.starts_with("../")
        || trimmed.starts_with('~')
        || trimmed.contains('\\')
        || trimmed.matches('/').count() > 1
}
