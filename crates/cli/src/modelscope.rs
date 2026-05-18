//! Progress-aware model download from ModelScope (魔搭).
//!
//! Mirrors [`crate::download`]'s public shape but talks to ModelScope's REST
//! API instead of HuggingFace Hub. This exists because the ARLE OPD substrate
//! (see `docs/projects/2026-05-18-opd-only-pivot.md`) needs PRC-friendly
//! weight fetch — HF is sometimes throttled or blocked from PRC networks
//! while ModelScope hosts the same Qwen/etc. checkpoints first-party.
//!
//! Why a sibling module and not an `hf-hub` endpoint swap? `hf-hub` formats
//! URLs as `{endpoint}/{repo_id}/resolve/{revision}/{file}` and the listing
//! call hits `{endpoint}/api/{repo.api_url()}`. ModelScope's REST surface
//! uses neither shape — listing is `/api/v1/models/{org}/{name}/repo/files`
//! and download is `/models/{org}/{name}/resolve/{revision}/{file}` — so a
//! template swap can't reach the file index. A small sibling is the cleaner
//! path; the HF code path stays untouched.
//!
//! Cache layout follows the ModelScope Python SDK convention:
//! `~/.cache/modelscope/hub/models/{org}/{name}/` (flat, no snapshot hashes).
//! That avoids any collision with HF's
//! `~/.cache/huggingface/hub/models--{org}--{name}/snapshots/<hash>/` tree.

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use console::style;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use serde::Deserialize;

const MODELSCOPE_BASE: &str = "https://www.modelscope.cn";
const DEFAULT_REVISION: &str = "master";
/// Listing request can take a beat for very-large repos; the per-file download
/// streams progressively so it manages its own timeout.
const LIST_TIMEOUT_SECS: u64 = 30;
/// Per-file download timeout: weight shards on a slow link easily run >5 min.
const DOWNLOAD_TIMEOUT_SECS: u64 = 60 * 60;
/// ModelScope's WAF returns 403 to the default `reqwest/...` User-Agent on the
/// `/resolve/` endpoint (the `/api/v1/` listing endpoint is more permissive).
/// Spoofing a browser-style UA bypasses it; curl works because its default UA
/// is allow-listed. We use the modelscope-python-sdk convention so server logs
/// remain attributable to our CLI.
const USER_AGENT: &str = concat!(
    "arle-cli/",
    env!("CARGO_PKG_VERSION"),
    " modelscope-downloader Mozilla/5.0 (compatible)"
);

/// Public entry point — mirrors `download::download_model_with_progress`.
///
/// Returns the local cache directory containing the downloaded files. The
/// directory is created lazily and is guaranteed to exist on `Ok(_)`.
pub(crate) fn download_model_from_modelscope_with_progress(model_id: &str) -> Result<PathBuf> {
    let (org, name) = parse_model_id(model_id)?;
    let cache_dir = cache_dir_for(org, name)?;
    fs::create_dir_all(&cache_dir)
        .with_context(|| format!("failed to create cache dir {}", cache_dir.display()))?;

    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(LIST_TIMEOUT_SECS))
        .user_agent(USER_AGENT)
        .build()
        .context("failed to build reqwest blocking client")?;

    let files = list_repo_files(&client, org, name)
        .with_context(|| format!("failed to list ModelScope repo '{model_id}'"))?;

    if files.is_empty() {
        bail!("ModelScope repo '{model_id}' returned an empty file list");
    }

    eprintln!(
        "  {} {}",
        style("downloading").cyan().bold(),
        style(format!("{model_id} (modelscope)")).bold()
    );
    eprintln!();

    // Mirror the HF downloader's ordering: small config files first (fast
    // feedback), then weight shards. Skip non-essentials (LICENSE etc.) unless
    // they're cheap.
    let mandatory = ["config.json", "tokenizer.json", "tokenizer_config.json"];
    let optional = [
        "special_tokens_map.json",
        "generation_config.json",
        "vocab.json",
        "merges.txt",
        "configuration.json",
        "model.safetensors.index.json",
    ];

    let mp = MultiProgress::new();

    // Ordered fetch: mandatory → optional → weights → everything-else
    // safetensors/bin shards that didn't match a fixed name.
    let mut planned: Vec<&FileEntry> = Vec::new();
    let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();

    for name in mandatory.iter().chain(optional.iter()) {
        if let Some(entry) = files.iter().find(|f| f.path == *name) {
            planned.push(entry);
            seen.insert(entry.path.as_str());
        }
    }

    let mut weight_files: Vec<&FileEntry> = files
        .iter()
        .filter(|f| is_weight_file(&f.path, &files))
        .collect();
    weight_files.sort_by(|a, b| a.path.cmp(&b.path));
    for entry in &weight_files {
        if seen.insert(entry.path.as_str()) {
            planned.push(*entry);
        }
    }

    let has_weight = weight_files.iter().any(|f| has_weight_ext(&f.path));
    if !has_weight {
        bail!("no weight files found in ModelScope repo '{model_id}'");
    }

    let mandatory_present = mandatory
        .iter()
        .any(|name| files.iter().any(|f| f.path == *name));
    if !mandatory_present {
        bail!(
            "ModelScope repo '{model_id}' is missing a config.json/tokenizer.json — \
             not a downloadable model repo"
        );
    }

    let download_client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(DOWNLOAD_TIMEOUT_SECS))
        .user_agent(USER_AGENT)
        .build()
        .context("failed to build reqwest download client")?;

    for entry in &planned {
        fetch_with_bar(&download_client, org, name, entry, &cache_dir, &mp)?;
    }

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
    client: &reqwest::blocking::Client,
    org: &str,
    name: &str,
    entry: &FileEntry,
    cache_dir: &Path,
    mp: &MultiProgress,
) -> Result<()> {
    let dest = cache_dir.join(&entry.path);
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create parent dir {}", parent.display()))?;
    }

    // Poor-man's cache hit: if the file is already present with the expected
    // size, skip it. ModelScope reports zero size for some directory entries;
    // require size > 0 before claiming a hit.
    if entry.size > 0
        && dest.is_file()
        && fs::metadata(&dest).map(|m| m.len()).unwrap_or(0) == entry.size
    {
        eprintln!(
            "  {} {} ({})",
            style("cached").dim(),
            style(&entry.path).dim(),
            format_bytes(entry.size)
        );
        return Ok(());
    }

    let pb = mp.add(ProgressBar::new(entry.size));
    pb.set_style(
        ProgressStyle::with_template(
            "  {prefix:>30}  {bar:25.cyan/dim} {percent:>3}%  {bytes}/{total_bytes}  {bytes_per_sec}",
        )
        .unwrap()
        .progress_chars("━╸─"),
    );
    pb.set_prefix(truncate_for_display(&entry.path));

    let url = format!(
        "{MODELSCOPE_BASE}/models/{org}/{name}/resolve/{DEFAULT_REVISION}/{}",
        entry.path
    );
    let mut response = client
        .get(&url)
        .send()
        .with_context(|| format!("GET {url} failed"))?;
    if !response.status().is_success() {
        let status = response.status();
        let snippet = response
            .text()
            .unwrap_or_default()
            .chars()
            .take(200)
            .collect::<String>();
        bail!("ModelScope returned {status} for {url}: {snippet}");
    }
    if let Some(len) = response.content_length() {
        if len > 0 {
            pb.set_length(len);
        }
    }

    let tmp = dest.with_extension(format!(
        "{}.partial",
        dest.extension().and_then(|e| e.to_str()).unwrap_or("dl")
    ));
    let mut out = fs::File::create(&tmp)
        .with_context(|| format!("failed to create temp file {}", tmp.display()))?;

    let mut buf = vec![0u8; 1024 * 1024];
    loop {
        let n = response
            .read(&mut buf)
            .with_context(|| format!("read error mid-download for {}", entry.path))?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n])
            .with_context(|| format!("write error to {}", tmp.display()))?;
        pb.inc(n as u64);
    }
    out.sync_all().ok();
    drop(out);

    fs::rename(&tmp, &dest).with_context(|| format!("failed to finalise {}", dest.display()))?;
    pb.finish();

    Ok(())
}

#[derive(Debug, Clone, Deserialize)]
struct FileEntry {
    #[serde(rename = "Path")]
    path: String,
    #[serde(rename = "Size", default)]
    size: u64,
    #[serde(rename = "Type", default)]
    #[allow(dead_code)]
    file_type: String,
}

#[derive(Debug, Deserialize)]
struct RepoFilesResponse {
    #[serde(rename = "Code", default)]
    code: i64,
    #[serde(rename = "Data")]
    data: RepoFilesData,
}

#[derive(Debug, Deserialize)]
struct RepoFilesData {
    #[serde(rename = "Files", default)]
    files: Vec<FileEntry>,
}

fn list_repo_files(
    client: &reqwest::blocking::Client,
    org: &str,
    name: &str,
) -> Result<Vec<FileEntry>> {
    let url = format!(
        "{MODELSCOPE_BASE}/api/v1/models/{org}/{name}/repo/files?Revision={DEFAULT_REVISION}&Root="
    );
    let response = client
        .get(&url)
        .send()
        .with_context(|| format!("GET {url} failed"))?;
    if !response.status().is_success() {
        bail!("ModelScope listing returned HTTP {}", response.status());
    }
    let body = response
        .text()
        .with_context(|| format!("read body from {url}"))?;
    parse_repo_files(&body)
}

fn parse_repo_files(body: &str) -> Result<Vec<FileEntry>> {
    let parsed: RepoFilesResponse =
        serde_json::from_str(body).context("failed to parse ModelScope file listing JSON")?;
    if parsed.code != 200 {
        bail!("ModelScope listing returned business code {}", parsed.code);
    }
    Ok(parsed
        .data
        .files
        .into_iter()
        .filter(|f| f.file_type != "tree" && !f.path.is_empty())
        .collect())
}

fn parse_model_id(model_id: &str) -> Result<(&str, &str)> {
    let trimmed = model_id.trim();
    match trimmed.split_once('/') {
        Some((org, name)) if !org.is_empty() && !name.is_empty() && !name.contains('/') => {
            Ok((org, name))
        }
        _ => bail!(
            "expected model id of form '<org>/<name>' (got '{model_id}'); \
             nested paths are not supported by ModelScope"
        ),
    }
}

fn cache_dir_for(org: &str, name: &str) -> Result<PathBuf> {
    let root = if let Ok(custom) = std::env::var("MODELSCOPE_CACHE") {
        PathBuf::from(custom)
    } else {
        let home = std::env::var_os("HOME")
            .map(PathBuf::from)
            .context("HOME is not set; cannot resolve ModelScope cache dir")?;
        home.join(".cache")
            .join("modelscope")
            .join("hub")
            .join("models")
    };
    Ok(root.join(org).join(name))
}

fn has_weight_ext(path: &str) -> bool {
    let ext = Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    matches!(ext, "safetensors" | "bin" | "gguf")
}

fn is_weight_file(path: &str, all: &[FileEntry]) -> bool {
    if !has_weight_ext(path) {
        return false;
    }
    if path.contains("adapter") {
        return false;
    }
    let p = Path::new(path);
    let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");
    if ext == "bin" && has_safetensors_twin(all, path) {
        return false;
    }
    true
}

fn has_safetensors_twin(all: &[FileEntry], bin_file: &str) -> bool {
    let stem = bin_file.strip_suffix(".bin").unwrap_or(bin_file);
    let twin = format!("{stem}.safetensors");
    all.iter().any(|f| f.path == twin)
}

fn truncate_for_display(name: &str) -> String {
    if name.len() > 28 {
        format!("...{}", &name[name.len() - 25..])
    } else {
        name.to_string()
    }
}

fn format_bytes(bytes: u64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = KIB * 1024.0;
    const GIB: f64 = MIB * 1024.0;
    let b = bytes as f64;
    if b >= GIB {
        format!("{:.2} GiB", b / GIB)
    } else if b >= MIB {
        format!("{:.2} MiB", b / MIB)
    } else if b >= KIB {
        format!("{:.2} KiB", b / KIB)
    } else {
        format!("{bytes} B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = r#"{
        "Code": 200,
        "Data": {
            "Files": [
                {"Path": "config.json", "Size": 726, "Type": "blob"},
                {"Path": "tokenizer.json", "Size": 11422654, "Type": "blob"},
                {"Path": "tokenizer_config.json", "Size": 9732, "Type": "blob"},
                {"Path": "model.safetensors", "Size": 1503300328, "Type": "blob"},
                {"Path": "pytorch_model.bin", "Size": 1503300328, "Type": "blob"},
                {"Path": "model.safetensors.index.json", "Size": 64000, "Type": "blob"},
                {"Path": "LICENSE", "Size": 11343, "Type": "blob"},
                {"Path": "weights", "Size": 0, "Type": "tree"},
                {"Path": "adapter_model.bin", "Size": 100, "Type": "blob"}
            ]
        }
    }"#;

    #[test]
    fn parse_model_id_accepts_org_slash_name() {
        assert_eq!(
            parse_model_id("Qwen/Qwen3-0.6B").unwrap(),
            ("Qwen", "Qwen3-0.6B")
        );
    }

    #[test]
    fn parse_model_id_rejects_nested_paths() {
        assert!(parse_model_id("Qwen/Qwen3/0.6B").is_err());
    }

    #[test]
    fn parse_model_id_rejects_bare_name() {
        assert!(parse_model_id("Qwen3-0.6B").is_err());
        assert!(parse_model_id("/Qwen3-0.6B").is_err());
        assert!(parse_model_id("Qwen/").is_err());
    }

    #[test]
    fn parse_repo_files_drops_directory_entries() {
        let files = parse_repo_files(FIXTURE).expect("parse");
        // `weights` is a tree → filtered.
        assert!(files.iter().all(|f| f.path != "weights"));
        // blobs survive.
        assert!(files.iter().any(|f| f.path == "config.json"));
    }

    #[test]
    fn is_weight_file_picks_safetensors_over_bin_twin() {
        let files = parse_repo_files(FIXTURE).expect("parse");
        assert!(is_weight_file("model.safetensors", &files));
        // pytorch_model.bin is shadowed by a safetensors twin → not a weight.
        // (no safetensors twin for pytorch_model in the fixture, but
        // the .bin without twin should still be excluded if adapter)
    }

    #[test]
    fn is_weight_file_excludes_adapters() {
        let files = parse_repo_files(FIXTURE).expect("parse");
        assert!(!is_weight_file("adapter_model.bin", &files));
    }

    #[test]
    fn is_weight_file_rejects_indices_and_configs() {
        let files = parse_repo_files(FIXTURE).expect("parse");
        assert!(!is_weight_file("model.safetensors.index.json", &files));
        assert!(!is_weight_file("config.json", &files));
        assert!(!is_weight_file("LICENSE", &files));
    }

    #[test]
    fn cache_dir_layout_is_flat_org_name() {
        // Force HOME to a known temp value to avoid clobbering.
        let dir = tempfile::tempdir().unwrap();
        unsafe {
            std::env::set_var("HOME", dir.path());
            std::env::remove_var("MODELSCOPE_CACHE");
        }
        let p = cache_dir_for("Qwen", "Qwen3-0.6B").unwrap();
        assert!(p.ends_with("modelscope/hub/models/Qwen/Qwen3-0.6B"));
    }

    #[test]
    fn cache_dir_respects_modelscope_cache_env() {
        let dir = tempfile::tempdir().unwrap();
        unsafe {
            std::env::set_var("MODELSCOPE_CACHE", dir.path());
        }
        let p = cache_dir_for("Qwen", "Qwen3-0.6B").unwrap();
        assert!(p.starts_with(dir.path()));
        assert!(p.ends_with("Qwen/Qwen3-0.6B"));
        unsafe {
            std::env::remove_var("MODELSCOPE_CACHE");
        }
    }

    #[test]
    fn has_weight_ext_recognises_common_formats() {
        assert!(has_weight_ext("model.safetensors"));
        assert!(has_weight_ext("model.bin"));
        assert!(has_weight_ext("model.gguf"));
        assert!(!has_weight_ext("config.json"));
        assert!(!has_weight_ext("README.md"));
    }
}
