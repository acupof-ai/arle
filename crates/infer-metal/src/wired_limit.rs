//! MLX wired-memory limit helpers for clean Metal executor construction.

use std::path::{Path, PathBuf};

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

pub(crate) fn model_weight_bytes(model_dir: &Path) -> Option<usize> {
    model_weight_size_for_path(model_dir).and_then(|estimate| usize::try_from(estimate.bytes).ok())
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

/// Accept the weight-file extensions MLX loads (`safetensors`/`bin`/`gguf`/`npz`),
/// so the wired-limit size estimate counts the same files the loader reads.
fn is_weight_file_ext(path: &Path) -> bool {
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
    matches!(ext, "safetensors" | "bin" | "gguf" | "npz")
}

fn weight_file_len(path: &Path) -> Option<u64> {
    if is_weight_file_ext(path) {
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
        if is_weight_file_ext(&path) {
            total += meta.len();
        }
    }
    Ok(total)
}
