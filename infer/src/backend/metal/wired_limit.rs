//! Auto-computed MLX wired-memory limit for warm-loading Qwen3.6-class MoE
//! weights on Apple Silicon.
//!
//! Pinning the model weights via `mlx::set_wired_limit` stops macOS from paging
//! out cold expert weights under memory pressure — on the Qwen3.6 35B-A3B
//! baseline that pageout blows up p99 ITL by 5-20×. Auto-pinning dropped c=1
//! p99 from 86 ms to 15 ms (-83%) on first validation.
//!
//! This helper lives in the lib (not the `metal_serve` binary) so both the HTTP
//! server entry point and the in-process CLI load path (`server_engine::loaded`)
//! pin weights identically. See `docs/experience/wins/2026-05-07-bench-qwen36-mle-perf.md`.

use std::path::PathBuf;

/// Compute an auto wired-memory limit (model weight bytes + 1 GiB headroom) for
/// the given model path or HuggingFace cache id. Follows HF cache snapshot
/// symlinks. Returns `None` when no weight files can be sized (caller should
/// fall back to no pinning).
pub fn auto_wired_limit_bytes(model_path: &str) -> Option<usize> {
    const HEADROOM: u64 = 1 << 30;
    let candidates = [
        PathBuf::from(model_path),
        PathBuf::from(env!("HOME"))
            .join(".cache/huggingface/hub")
            .join(format!("models--{}", model_path.replace('/', "--")))
            .join("snapshots"),
    ];

    for candidate in &candidates {
        let snapshot_dir = if candidate.is_dir() && candidate.ends_with("snapshots") {
            std::fs::read_dir(candidate)
                .ok()?
                .filter_map(Result::ok)
                .find(|e| e.path().is_dir())
                .map(|e| e.path())
        } else if candidate.is_dir() {
            Some(candidate.clone())
        } else {
            None
        };
        let Some(dir) = snapshot_dir else { continue };
        let total = sum_weight_files(&dir).unwrap_or(0);
        if total == 0 {
            continue;
        }
        let limit = (total + HEADROOM) as usize;
        log::info!(
            "auto wired_limit = {} GiB ({} bytes; model dir {})",
            limit / (1 << 30),
            limit,
            dir.display()
        );
        return Some(limit);
    }

    None
}

fn sum_weight_files(dir: &std::path::Path) -> std::io::Result<u64> {
    let mut total = 0u64;
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        // Follow symlinks here — HF cache snapshots are symlinks into a sibling
        // blobs/ directory, so `entry.metadata()` (which doesn't traverse on
        // Unix) reports the symlink's own ~12-byte size and undercounts
        // catastrophically. `std::fs::metadata` follows the link.
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
