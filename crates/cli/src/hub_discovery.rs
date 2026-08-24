//! Walks `~/.cache/huggingface/hub` snapshots, filtered by `config.json`
//! `model_type`/`architectures` (not the id name), sorted newest-first.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// Architectures the picker must never offer as a primary, standalone model.
///
/// `DFlashDraftModel` is the speculative-decoding *draft* half of a DFlash
/// pair — it has no tokenizer and only ~5–8 layers, so loading it as a
/// target produces a confusing "tokenizer.json not found" error. The
/// repo names match `qwen3*` so the substring filter alone lets them
/// through; the only honest signal lives in `config.json#architectures`.
const DRAFT_ONLY_ARCHITECTURES: &[&str] = &["DFlashDraftModel"];

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HubSnapshot {
    pub(crate) model_id: String,
    pub(crate) path: PathBuf,
}

pub(crate) fn decode_hub_snapshot_path(path: &Path) -> Option<String> {
    let parent = path.parent()?;
    if parent.file_name()?.to_str()? != "snapshots" {
        return None;
    }
    let repo_dir = parent.parent()?;
    let repo_name = repo_dir.file_name()?.to_str()?;
    let rest = repo_name.strip_prefix("models--")?;

    // First `--` splits org from repo; any additional `--` inside the repo
    // name are rare but preserved as `-`.
    let (org, repo) = rest.split_once("--")?;
    let repo = repo.replace("--", "-");
    Some(format!("{org}/{repo}"))
}

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

/// Does the **compiled backend's** serve path actually load this checkpoint?
///
/// The authoritative signal is `config.json#model_type` + `architectures`, NOT
/// the model-id name — a substring family match let non-servable look-alikes
/// through (Qwen3-0.6B is `model_type=qwen3`, the MTP draft is `qwen3_5_mtp`)
/// (Metal only). The per-backend sets are the serve-path truth:
/// Metal requires the Qwen3.5 `layer_types` config (`infer-metal/config.rs`) so
/// only `qwen3_5` / `qwen3_5_moe` load; CUDA serves the
/// Qwen3 / Qwen3.5 dense+MoE families and DeepSeek-V4 (mirrors
/// `infer_api::classify_cuda_model`). Measured 2026-06-14: `benchmarks/README.md`.
/// Best-effort: unreadable / unparseable config → `false` (don't offer what we
/// can't classify).
fn snapshot_is_servable(path: &Path) -> bool {
    let cfg = path.join("config.json");
    let Ok(raw) = fs::read_to_string(&cfg) else {
        return false;
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&raw) else {
        return false;
    };
    let model_type = v.get("model_type").and_then(|x| x.as_str()).unwrap_or("");
    let arch_contains = |needle: &str| {
        v.get("architectures")
            .and_then(|a| a.as_array())
            .is_some_and(|a| {
                a.iter()
                    .any(|s| s.as_str().is_some_and(|s| s.contains(needle)))
            })
    };
    let mut servable = false;
    #[cfg(feature = "metal")]
    {
        servable |= matches!(model_type, "qwen3_5" | "qwen3_5_moe")
            || arch_contains("Qwen3_5")
            // DeepSeek-OCR VLM (Metal-only) — the `arle ocr` model.
            || model_type == "deepseekocr"
            || arch_contains("UnlimitedOCR");
    }
    #[cfg(feature = "cuda")]
    {
        servable |= model_type == "deepseek_v4"
            || arch_contains("DeepseekV4")
            || (model_type.starts_with("qwen3") && model_type != "qwen3_5_mtp")
            || arch_contains("Qwen3");
    }
    #[cfg(not(any(feature = "metal", feature = "cuda")))]
    {
        servable |= (model_type.starts_with("qwen3") && model_type != "qwen3_5_mtp")
            || arch_contains("Qwen3");
    }
    servable
}

fn snapshot_has_usable_content(path: &Path) -> bool {
    path.join("config.json").exists() || path.join("model.safetensors").exists()
}

/// Best-effort: `false` on any parse/read failure so a malformed config
/// doesn't hide a usable target model.
fn snapshot_is_draft_only(path: &Path) -> bool {
    let cfg = path.join("config.json");
    let Ok(raw) = fs::read_to_string(&cfg) else {
        return false;
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&raw) else {
        return false;
    };
    let Some(archs) = v.get("architectures").and_then(|a| a.as_array()) else {
        return false;
    };
    archs.iter().any(|item| {
        item.as_str()
            .is_some_and(|s| DRAFT_ONLY_ARCHITECTURES.contains(&s))
    })
}

fn snapshot_mtime(path: &Path) -> SystemTime {
    fs::metadata(path)
        .and_then(|m| m.modified())
        .unwrap_or(SystemTime::UNIX_EPOCH)
}

pub(crate) fn discover_hub_snapshots() -> Vec<HubSnapshot> {
    let Some(root) = hub_cache_root() else {
        return Vec::new();
    };
    let Ok(read) = fs::read_dir(&root) else {
        return Vec::new();
    };

    let mut out: Vec<HubSnapshot> = read
        .flatten()
        .filter(|e| {
            e.path()
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("models--"))
        })
        .flat_map(|repo_entry| {
            let snapshots_dir = repo_entry.path().join("snapshots");
            fs::read_dir(snapshots_dir).into_iter().flatten().flatten()
        })
        .filter_map(|snap_entry| {
            let snap_path = snap_entry.path();
            if !snap_path.is_dir()
                || !snapshot_has_usable_content(&snap_path)
                || !snapshot_is_servable(&snap_path)
                || snapshot_is_draft_only(&snap_path)
            {
                return None;
            }
            let model_id = decode_hub_snapshot_path(&snap_path)?;
            Some(HubSnapshot {
                model_id,
                path: snap_path,
            })
        })
        .collect();

    out.sort_by_key(|s| std::cmp::Reverse(snapshot_mtime(&s.path)));
    out
}
