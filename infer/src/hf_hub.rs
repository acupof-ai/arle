//! HuggingFace Hub automatic model download.
//!
//! Mirrors sglang's `hf_utils.py` approach:
//! 1. If the string looks like an existing local path, use it directly.
//! 2. Otherwise treat it as a HF model ID (`org/repo` or `repo`), download
//!    all relevant files to `~/.cache/huggingface/hub/` and return the cache dir.
//!
//! Relevant files downloaded:
//! - `config.json` — model architecture / hyper-params
//! - `tokenizer.json` + `tokenizer_config.json` + `special_tokens_map.json`
//! - `*.safetensors` (preferred over pickle)
//! - `model.safetensors.index.json` (sharded weight index, if present)
//! - `generation_config.json` (optional)
//!
//! Authentication: reads `HF_TOKEN` env var, then `~/.cache/huggingface/token`.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use hf_hub::{
    Repo, RepoType,
    api::sync::{Api, ApiBuilder, ApiRepo},
};

const DEFAULT_DISCOVERY_CANDIDATES: &[&str] = &[
    "mlx-community/Qwen3-0.6B-4bit",
    "mlx-community/Qwen3-0.6B-bf16",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3.5-4B",
];

/// Resolve a model source string to a local directory containing model files.
///
/// # Arguments
/// * `model_id_or_path` — Either an existing local path or a HuggingFace model
///   ID (e.g. `"Qwen/Qwen2.5-0.5B-Instruct"`, `"mlx-community/Qwen2.5-0.5B-4bit"`).
///
/// Returns the path to the directory that contains `config.json` and weight files.
///
/// Short-circuit: if the input *looks like* a filesystem path
/// (see [`looks_like_local_path`]) but that path does not exist locally, the
/// call fails immediately with a clear error rather than falling through to
/// a HuggingFace Hub download. Repo-id inputs (e.g. `Qwen/Qwen3-0.6B`) still
/// fall through to the hub as before.
pub fn resolve_model_path(model_id_or_path: &str) -> Result<PathBuf> {
    if let Some(local) = resolve_local_model_path(model_id_or_path) {
        log::info!("Using local model path: {}", local.display());
        return Ok(local);
    }

    if looks_like_local_path(model_id_or_path) {
        anyhow::bail!(
            "model path does not exist: '{}' (looks like a filesystem path, skipping HuggingFace Hub lookup)",
            model_id_or_path
        );
    }

    log::info!("Model not found locally — downloading from HuggingFace: {model_id_or_path}");
    download_from_hub(model_id_or_path)
}

/// Resolve a checkpoint source for callers that need config + weights but not a
/// tokenizer.
///
/// This is used for auxiliary checkpoints such as Metal DFlash draft models,
/// which are often stored in the local HuggingFace cache without tokenizer
/// files.
pub fn resolve_weighted_model_path(model_id_or_path: &str) -> Result<PathBuf> {
    if let Some(local) = resolve_local_weighted_model_path(model_id_or_path) {
        log::info!("Using local weighted model path: {}", local.display());
        return Ok(local);
    }

    log::info!(
        "Weighted model not found locally — downloading from HuggingFace: {model_id_or_path}"
    );
    download_from_hub(model_id_or_path)
}

/// Resolve a model source using local paths and local HuggingFace cache only.
///
/// Returns `None` when no local candidate exists.
pub fn resolve_local_model_path(model_id_or_path: &str) -> Option<PathBuf> {
    let local = Path::new(model_id_or_path);
    if local.exists() {
        return Some(local.to_path_buf());
    }

    local_model_search_candidates(model_id_or_path)
        .into_iter()
        .find(|candidate| is_model_dir(candidate))
}

/// Resolve a model source using local paths and local HuggingFace cache only.
///
/// Unlike [`resolve_local_model_path`], this only requires `config.json` plus at
/// least one local weight shard; tokenizer files are optional.
pub fn resolve_local_weighted_model_path(model_id_or_path: &str) -> Option<PathBuf> {
    let local = Path::new(model_id_or_path);
    if is_weighted_model_dir(local) {
        return Some(local.to_path_buf());
    }

    local_model_search_candidates(model_id_or_path)
        .into_iter()
        .find(|candidate| is_weighted_model_dir(candidate))
}

/// Discover the best local model from a curated candidate list.
///
/// Prefers the explicit candidate order. Returns the candidate label that matched
/// plus the resolved local path.
pub fn discover_local_model() -> Option<(String, PathBuf)> {
    discover_local_model_from(DEFAULT_DISCOVERY_CANDIDATES)
}

/// Same as [`discover_local_model`] but with a caller-provided priority list.
pub fn discover_local_model_from(candidates: &[&str]) -> Option<(String, PathBuf)> {
    candidates.iter().find_map(|candidate| {
        resolve_local_model_path(candidate).map(|path| ((*candidate).to_string(), path))
    })
}

/// Resolve a model source for CLI-style callers.
///
/// Prefers an explicit flag value, then `AGENT_INFER_MODEL`, then local model
/// auto-discovery using [`discover_local_model`].
pub fn resolve_model_source(explicit_model_path: Option<&str>) -> Result<String> {
    if let Some(model_path) = explicit_model_path
        && !model_path.trim().is_empty()
    {
        return Ok(model_path.to_string());
    }

    if let Ok(model) = std::env::var("AGENT_INFER_MODEL")
        && !model.trim().is_empty()
    {
        return Ok(model);
    }

    if let Some((candidate, local_path)) = discover_local_model() {
        log::info!(
            "Auto-detected local model '{}' at {}",
            candidate,
            local_path.display()
        );
        return Ok(candidate);
    }

    anyhow::bail!(
        "No model specified and no local model was auto-detected. Pass --model-path or set AGENT_INFER_MODEL."
    )
}

/// Download a model from HuggingFace Hub and return the local cache directory.
///
/// Files are stored under `~/.cache/huggingface/hub/models--<org>--<repo>/snapshots/<sha>/`.
/// Subsequent calls with the same model ID are served from cache (no re-download).
pub fn download_from_hub(model_id: &str) -> Result<PathBuf> {
    download_repo_assets_from_hub(model_id, true)
}

/// Download config + tokenizer assets without model weights.
///
/// Used by lightweight development backends that need prompt/tokenizer metadata
/// but do not execute the full neural model.
pub fn download_runtime_assets_from_hub(model_id: &str) -> Result<PathBuf> {
    download_repo_assets_from_hub(model_id, false)
}

fn download_repo_assets_from_hub(model_id: &str, include_weights: bool) -> Result<PathBuf> {
    let api = build_api().context("failed to initialise HuggingFace API")?;
    let repo = api.repo(Repo::new(model_id.to_string(), RepoType::Model));

    // Fetch repo metadata to discover which files exist.
    let info = repo
        .info()
        .with_context(|| format!("failed to fetch repo info for '{model_id}'"))?;

    let filenames: Vec<String> = info.siblings.iter().map(|s| s.rfilename.clone()).collect();

    log::info!(
        "Hub repo '{}' has {} files — selecting required ones",
        model_id,
        filenames.len()
    );

    // ── mandatory files ────────────────────────────────────────────────────
    let mandatory = ["config.json", "tokenizer.json", "tokenizer_config.json"];
    for name in &mandatory {
        if filenames.iter().any(|f| f == name) {
            fetch_file(&repo, name, model_id)?;
        } else {
            log::warn!("'{model_id}': expected file '{name}' not found in repo");
        }
    }

    // ── optional config files ─────────────────────────────────────────────
    let optional = [
        "special_tokens_map.json",
        "generation_config.json",
        "vocab.json",
        "merges.txt",
        "model.safetensors.index.json",
    ];
    for name in &optional {
        if filenames.iter().any(|f| f == name) {
            fetch_file(&repo, name, model_id)?;
        }
    }

    // ── weight shards (safetensors preferred, no pickle) ──────────────────
    if include_weights {
        let weight_files: Vec<&str> = filenames
            .iter()
            .filter(|f| {
                let p = std::path::Path::new(f.as_str());
                let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");
                (ext == "safetensors" || ext == "bin")
                    // Skip adapter / lora weight files
                    && !f.contains("adapter")
                    // Prefer safetensors; skip .bin when a .safetensors twin exists
                    && !(ext == "bin" && has_safetensors_twin(&filenames, f))
            })
            .map(String::as_str)
            .collect();

        if weight_files.is_empty() {
            anyhow::bail!("no weight files (.safetensors or .bin) found in HF repo '{model_id}'");
        }

        for name in &weight_files {
            fetch_file(&repo, name, model_id)?;
        }
    }

    // ── derive local cache dir from the first downloaded file ─────────────
    let first = mandatory
        .iter()
        .find(|name| filenames.iter().any(|f| f == *name))
        .unwrap_or(&"config.json");

    // `repo.get()` returns the full path to the cached file; its parent is
    // the snapshot directory that contains all downloaded files.
    let file_path = repo
        .get(first)
        .with_context(|| format!("failed to resolve cache path for '{first}'"))?;

    let cache_dir = file_path
        .parent()
        .map_or_else(|| file_path.clone(), Path::to_path_buf);

    log::info!("Model '{}' ready at: {}", model_id, cache_dir.display());
    Ok(cache_dir)
}

// ── helpers ───────────────────────────────────────────────────────────────────

pub fn build_api() -> Result<Api> {
    let mut builder = ApiBuilder::new();

    // Honour HF_TOKEN for private / gated models.
    if let Ok(token) = std::env::var("HF_TOKEN") {
        if !token.is_empty() {
            log::debug!("HF_TOKEN found — using for authentication");
            builder = builder.with_token(Some(token));
        }
    }

    builder.build().context("ApiBuilder::build failed")
}

fn local_model_search_candidates(model_id_or_path: &str) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    let model_name = model_id_or_path
        .rsplit('/')
        .next()
        .unwrap_or(model_id_or_path)
        .trim();

    for base in common_local_roots() {
        candidates.push(base.join(model_name));
    }

    if let Some((org, repo)) = parse_hf_model_id(model_id_or_path) {
        let cache_repo_dir = huggingface_hub_root().join(format!("models--{org}--{repo}"));
        candidates.extend(snapshot_dirs(&cache_repo_dir));
        candidates.push(cache_repo_dir.join("snapshots").join("main"));
    }

    candidates
}

fn common_local_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();

    if let Ok(cwd) = std::env::current_dir() {
        roots.push(cwd.join("models"));
        roots.push(cwd.join("infer").join("models"));
    }

    if let Some(home) = home_dir() {
        roots.push(home.join("models"));
        roots.push(home.join("llm"));
        roots.push(home.join(".cache").join("mlx-models"));
    }

    roots
}

fn huggingface_hub_root() -> PathBuf {
    if let Ok(hf_home) = std::env::var("HF_HOME")
        && !hf_home.trim().is_empty()
    {
        return PathBuf::from(hf_home).join("hub");
    }

    if let Some(home) = home_dir() {
        return home.join(".cache").join("huggingface").join("hub");
    }

    PathBuf::from(".cache/huggingface/hub")
}

fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

fn parse_hf_model_id(model_id_or_path: &str) -> Option<(String, String)> {
    let (org, repo) = model_id_or_path.split_once('/')?;
    if org.is_empty() || repo.is_empty() {
        return None;
    }
    Some((org.to_string(), repo.to_string()))
}

/// Heuristic: does the input *look like* a filesystem path rather than a
/// HuggingFace repo id?
///
/// Treated as path-like:
/// - starts with `/`, `./`, `../`, or `~`
/// - contains a backslash (Windows-style separator)
/// - contains more than one forward slash (`a/b/c`)
///
/// Treated as a repo id (not path-like):
/// - bare name (`foo`)
/// - single-slash `org/repo` form (`Qwen/Qwen3-0.6B`, `databricks/databricks-dolly-15k`)
///
/// Note: HuggingFace repo names can legitimately contain dots
/// (`Qwen/Qwen3.5-4B`), so a dot alone does **not** mark the input as path-like.
pub(crate) fn looks_like_local_path(input: &str) -> bool {
    let s = input.trim();
    if s.is_empty() {
        return false;
    }
    if s.starts_with('/') || s.starts_with("./") || s.starts_with("../") || s.starts_with('~') {
        return true;
    }
    if s.contains('\\') {
        return true;
    }
    s.matches('/').count() > 1
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
    snapshots.sort();
    snapshots.reverse();
    snapshots
}

fn is_model_dir(path: &Path) -> bool {
    path.is_dir() && path.join("config.json").exists() && path.join("tokenizer.json").exists()
}

fn is_weighted_model_dir(path: &Path) -> bool {
    if !path.is_dir() || !path.join("config.json").exists() {
        return false;
    }

    let Ok(entries) = std::fs::read_dir(path) else {
        return false;
    };

    entries.filter_map(std::result::Result::ok).any(|entry| {
        // HuggingFace snapshot directories commonly expose files as symlinks to
        // the blob store, so use `Path::is_file()` instead of `DirEntry::file_type()`.
        if !entry.path().is_file() {
            return false;
        }
        matches!(
            entry.path().extension().and_then(|ext| ext.to_str()),
            Some("safetensors" | "bin")
        )
    })
}

fn fetch_file(repo: &ApiRepo, filename: &str, model_id: &str) -> Result<PathBuf> {
    log::info!("  ↓ {filename}");
    repo.get(filename)
        .with_context(|| format!("failed to download '{filename}' from '{model_id}'"))
}

/// Returns true when there is a `.safetensors` counterpart for a `.bin` file.
/// e.g. `model.bin` → looks for `model.safetensors` in the file list.
fn has_safetensors_twin(all: &[String], bin_file: &str) -> bool {
    let stem = bin_file.strip_suffix(".bin").unwrap_or(bin_file);
    let twin = format!("{stem}.safetensors");
    all.iter().any(|f| f == &twin)
}

#[cfg(test)]
mod tests {
    use super::{
        common_local_roots, has_safetensors_twin, looks_like_local_path, parse_hf_model_id,
        resolve_local_model_path, resolve_local_weighted_model_path, resolve_model_path,
    };

    #[test]
    fn twin_detection() {
        let files = vec![
            "model.bin".to_string(),
            "model.safetensors".to_string(),
            "pytorch_model.bin".to_string(),
        ];
        assert!(has_safetensors_twin(&files, "model.bin"));
        assert!(!has_safetensors_twin(&files, "pytorch_model.bin"));
    }

    #[test]
    fn parse_hf_model_id_requires_org_and_repo() {
        assert_eq!(
            parse_hf_model_id("mlx-community/Qwen3-0.6B-4bit"),
            Some(("mlx-community".to_string(), "Qwen3-0.6B-4bit".to_string()))
        );
        assert_eq!(parse_hf_model_id("Qwen3-0.6B-4bit"), None);
    }

    #[test]
    fn resolve_local_model_path_accepts_existing_directory() {
        let tmp = tempfile::tempdir().expect("tempdir");
        std::fs::write(tmp.path().join("config.json"), "{}").expect("config");
        std::fs::write(tmp.path().join("tokenizer.json"), "{}").expect("tokenizer");

        assert_eq!(
            resolve_local_model_path(tmp.path().to_str().expect("utf8")),
            Some(tmp.path().to_path_buf())
        );
    }

    #[test]
    fn resolve_local_weighted_model_path_accepts_weight_only_directory() {
        let tmp = tempfile::tempdir().expect("tempdir");
        std::fs::write(tmp.path().join("config.json"), "{}").expect("config");
        std::fs::write(tmp.path().join("model.safetensors"), "").expect("weights");

        assert_eq!(
            resolve_local_weighted_model_path(tmp.path().to_str().expect("utf8")),
            Some(tmp.path().to_path_buf())
        );
    }

    #[test]
    fn resolve_local_weighted_model_path_accepts_symlinked_weights() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let blob_dir = tempfile::tempdir().expect("blob tempdir");
        std::fs::write(tmp.path().join("config.json"), "{}").expect("config");
        let blob = blob_dir.path().join("model.safetensors");
        std::fs::write(&blob, "").expect("weights");
        std::os::unix::fs::symlink(&blob, tmp.path().join("model.safetensors")).expect("symlink");

        assert_eq!(
            resolve_local_weighted_model_path(tmp.path().to_str().expect("utf8")),
            Some(tmp.path().to_path_buf())
        );
    }

    #[test]
    fn common_local_roots_include_repo_models_dir() {
        let cwd = std::env::current_dir().expect("current_dir");
        let roots = common_local_roots();
        assert!(roots.contains(&cwd.join("models")));
        assert!(roots.contains(&cwd.join("infer").join("models")));
    }

    #[test]
    fn looks_like_local_path_classifies_paths() {
        // Path-like: absolute, relative with ./ or ../, home, multiple slashes.
        assert!(looks_like_local_path("/does/not/exist"));
        assert!(looks_like_local_path("/tmp/models"));
        assert!(looks_like_local_path("./models"));
        assert!(looks_like_local_path("../models/Qwen3-0.6B"));
        assert!(looks_like_local_path("~/models"));
        assert!(looks_like_local_path("foo/bar/baz"));
        assert!(looks_like_local_path("C:\\models\\Qwen3"));
    }

    #[test]
    fn looks_like_local_path_classifies_repo_ids() {
        // HF repo ids (single slash, no leading special char) are NOT paths.
        assert!(!looks_like_local_path("Qwen/Qwen3-0.6B"));
        assert!(!looks_like_local_path("databricks/databricks-dolly-15k"));
        assert!(!looks_like_local_path("mlx-community/Qwen3-0.6B-4bit"));
        // Dots are legal in HF repo names (e.g. Qwen3.5).
        assert!(!looks_like_local_path("Qwen/Qwen3.5-4B"));
        // Bare name.
        assert!(!looks_like_local_path("Qwen3-0.6B"));
        // Empty.
        assert!(!looks_like_local_path(""));
    }

    #[test]
    fn resolve_model_path_fails_fast_for_missing_local_path() {
        // /definitely/does/not/exist/<pid> looks like an absolute path, so it
        // must short-circuit without touching the network.
        let bogus = format!(
            "/definitely/does/not/exist-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        );
        let err = resolve_model_path(&bogus).expect_err("must bail for missing path");
        let msg = format!("{err:#}");
        assert!(msg.contains("does not exist"), "bad error message: {msg}");
        assert!(msg.contains(&bogus), "error should mention input: {msg}");
    }
}
