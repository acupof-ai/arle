//! HuggingFace hub cache discovery — model auto-discovery wizard data source.
//!
//! Walks `~/.cache/huggingface/hub/models--*/snapshots/*/` looking for
//! snapshot dirs containing `config.json` or `model.safetensors`, filters to
//! supported model families, and sorts newest-first.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// Supported model family substrings (case-insensitive).
///
/// Keep in sync with `docs/support-matrix.md`.
const SUPPORTED_FAMILIES: &[&str] = &["qwen3", "qwen2.5", "qwen3.5"];

/// A discovered HuggingFace-cache snapshot ready for the picker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HubSnapshot {
    /// Human-friendly model id (`Qwen/Qwen3-4B`), decoded from `models--…`.
    pub(crate) model_id: String,
    /// Full filesystem path to the snapshot directory.
    pub(crate) path: PathBuf,
}

/// Decode a path of the form
/// `~/.cache/huggingface/hub/models--org--repo/snapshots/<hash>/` into the
/// canonical model id `org/repo`.
///
/// Returns `None` if the path shape does not match.
pub(crate) fn decode_hub_snapshot_path(path: &Path) -> Option<String> {
    // Expect .../models--X--Y.../snapshots/<hash>
    let parent = path.parent()?; // snapshots
    if parent.file_name()?.to_str()? != "snapshots" {
        return None;
    }
    let repo_dir = parent.parent()?; // models--X--Y...
    let repo_name = repo_dir.file_name()?.to_str()?;
    let rest = repo_name.strip_prefix("models--")?;

    // `org--repo[--...]` → first `--` splits org from the rest; any additional
    // `--` inside the repo name are rare but preserved as `-`.
    let (org, repo) = rest.split_once("--")?;
    let repo = repo.replace("--", "-");
    Some(format!("{org}/{repo}"))
}

/// Return the user's HuggingFace hub cache dir, honouring `HF_HOME` /
/// `HUGGINGFACE_HUB_CACHE` if present.
pub(crate) fn hub_cache_root() -> Option<PathBuf> {
    if let Some(v) = std::env::var_os("HUGGINGFACE_HUB_CACHE") {
        return Some(PathBuf::from(v));
    }
    if let Some(v) = std::env::var_os("HF_HOME") {
        return Some(PathBuf::from(v).join("hub"));
    }
    let home = std::env::var_os("HOME")?;
    Some(PathBuf::from(home).join(".cache/huggingface/hub"))
}

fn is_family_supported(model_id: &str) -> bool {
    let lc = model_id.to_ascii_lowercase();
    SUPPORTED_FAMILIES.iter().any(|f| lc.contains(f))
}

fn snapshot_has_usable_content(path: &Path) -> bool {
    path.join("config.json").exists() || path.join("model.safetensors").exists()
}

fn snapshot_mtime(path: &Path) -> SystemTime {
    fs::metadata(path)
        .and_then(|m| m.modified())
        .unwrap_or(SystemTime::UNIX_EPOCH)
}

/// Discover HF cache snapshots matching the supported families. Sorted by
/// mtime descending (newest first).
pub(crate) fn discover_hub_snapshots() -> Vec<HubSnapshot> {
    let Some(root) = hub_cache_root() else {
        return Vec::new();
    };
    let Ok(read) = fs::read_dir(&root) else {
        return Vec::new();
    };

    let mut out: Vec<HubSnapshot> = Vec::new();
    for repo_entry in read.flatten() {
        let repo_dir = repo_entry.path();
        let repo_name = repo_dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default();
        if !repo_name.starts_with("models--") {
            continue;
        }
        let snapshots_dir = repo_dir.join("snapshots");
        let Ok(snaps) = fs::read_dir(&snapshots_dir) else {
            continue;
        };
        for snap_entry in snaps.flatten() {
            let snap_path = snap_entry.path();
            if !snap_path.is_dir() {
                continue;
            }
            if !snapshot_has_usable_content(&snap_path) {
                continue;
            }
            let Some(model_id) = decode_hub_snapshot_path(&snap_path) else {
                continue;
            };
            if !is_family_supported(&model_id) {
                continue;
            }
            out.push(HubSnapshot {
                model_id,
                path: snap_path,
            });
        }
    }

    out.sort_by_key(|s| std::cmp::Reverse(snapshot_mtime(&s.path)));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn decode_hub_snapshot_path_simple() {
        let p =
            PathBuf::from("/home/u/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/abc123");
        assert_eq!(
            decode_hub_snapshot_path(&p).as_deref(),
            Some("Qwen/Qwen3-4B")
        );
    }

    #[test]
    fn decode_hub_snapshot_path_multi_segment_repo() {
        // Some model names contain `--` in the repo — exceedingly rare but
        // survives as `-` in the decoded id.
        let p = PathBuf::from(
            "/home/u/.cache/huggingface/hub/models--mlx-community--Qwen3-0.6B-4bit/snapshots/dead",
        );
        assert_eq!(
            decode_hub_snapshot_path(&p).as_deref(),
            Some("mlx-community/Qwen3-0.6B-4bit")
        );
    }

    #[test]
    fn decode_hub_snapshot_path_rejects_non_snapshot_paths() {
        let p = PathBuf::from("/tmp/something/else");
        assert_eq!(decode_hub_snapshot_path(&p), None);
    }

    #[test]
    fn decode_hub_snapshot_path_rejects_missing_prefix() {
        let p = PathBuf::from("/cache/hub/not-models--Qwen--X/snapshots/abc");
        assert_eq!(decode_hub_snapshot_path(&p), None);
    }

    #[test]
    fn is_family_supported_matches_supported_qwen_families() {
        assert!(is_family_supported("Qwen/Qwen3-4B"));
        assert!(is_family_supported("mlx-community/qwen3.5-4b-4bit"));
        assert!(is_family_supported("Qwen/Qwen2.5-7B"));
        assert!(!is_family_supported("mistralai/Mistral-7B"));
        assert!(!is_family_supported("meta-llama/Llama-3-8B"));
    }
}
