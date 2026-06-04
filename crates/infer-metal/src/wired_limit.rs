//! MLX wired-memory limit helpers for clean Metal executor construction.

use std::path::{Path, PathBuf};

/// Compute the legacy default wired limit: model weight bytes + 1 GiB.
pub(crate) fn auto_wired_limit_bytes(model_dir: &Path) -> Option<usize> {
    const HEADROOM: u64 = 1 << 30;
    let estimate = model_weight_size_for_path(model_dir)?;
    let limit = (estimate.bytes + HEADROOM) as usize;
    log::info!(
        "auto wired_limit = {} GiB ({} bytes; model dir {})",
        limit / (1 << 30),
        limit,
        estimate.source.display()
    );
    Some(limit)
}

#[derive(Debug, Clone)]
struct ModelWeightSize {
    bytes: u64,
    source: PathBuf,
}

fn model_weight_size_for_path(path: &Path) -> Option<ModelWeightSize> {
    if path.is_file() {
        let bytes = weight_file_len(path)?;
        return Some(ModelWeightSize {
            bytes,
            source: path.to_path_buf(),
        });
    }

    let snapshot_dir = if path.is_dir() && path.ends_with("snapshots") {
        std::fs::read_dir(path)
            .ok()?
            .filter_map(Result::ok)
            .find(|e| e.path().is_dir())
            .map(|e| e.path())
    } else if path.is_dir() {
        Some(path.to_path_buf())
    } else {
        None
    }?;

    let total = sum_weight_files(&snapshot_dir).unwrap_or(0);
    (total > 0).then_some(ModelWeightSize {
        bytes: total,
        source: snapshot_dir,
    })
}

fn weight_file_len(path: &Path) -> Option<u64> {
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
    if matches!(ext, "safetensors" | "bin" | "gguf" | "npz") {
        let meta = std::fs::metadata(path).ok()?;
        if meta.is_file() {
            return Some(meta.len());
        }
    }
    None
}

fn sum_weight_files(dir: &Path) -> std::io::Result<u64> {
    let mut total = 0u64;
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let Ok(meta) = std::fs::metadata(&path) else {
            continue;
        };
        if !meta.is_file() {
            continue;
        }
        let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
        if matches!(ext, "safetensors" | "bin" | "gguf" | "npz") {
            total += meta.len();
        }
    }
    Ok(total)
}
