//! Opt-in MLX wired-memory limit for warm-loading Qwen3.6-class MoE weights on
//! Apple Silicon.
//!
//! Pinning the model weights via `mlx::set_wired_limit` stops macOS from paging
//! out cold expert weights under memory pressure — on the Qwen3.6 35B-A3B
//! baseline that pageout blows up p99 ITL by 5-20×. Auto-pinning dropped c=1
//! p99 from 86 ms to 15 ms (-83%) on first validation.
//!
//! This helper lives in the lib (not the `metal_serve` binary) so callers that
//! explicitly choose weight pinning can compute the same limit. See
//! `docs/experience/wins/2026-05-07-bench-qwen36-mle-perf.md`.

use std::path::{Path, PathBuf};

/// Compute an auto wired-memory limit (model weight bytes + 1 GiB headroom) for
/// the given model path or HuggingFace cache id. Follows HF cache snapshot
/// symlinks. Returns `None` when no weight files can be sized (caller should
/// fall back to no pinning).
pub fn auto_wired_limit_bytes(model_path: &str) -> Option<usize> {
    const HEADROOM: u64 = 1 << 30;
    let mut candidates = Vec::with_capacity(2);
    candidates.push(PathBuf::from(model_path));
    if let Some(hub_dir) = runtime_huggingface_hub_dir() {
        candidates.push(
            hub_dir
                .join(format!("models--{}", model_path.replace('/', "--")))
                .join("snapshots"),
        );
    }

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

fn runtime_huggingface_hub_dir() -> Option<PathBuf> {
    env_path("HUGGINGFACE_HUB_CACHE")
        .or_else(|| env_path("HF_HOME").map(|home| home.join("hub")))
        .or_else(|| runtime_home_dir().map(|home| home.join(".cache/huggingface/hub")))
}

fn runtime_home_dir() -> Option<PathBuf> {
    env_path("HOME").or_else(|| env_path("USERPROFILE"))
}

fn env_path(key: &str) -> Option<PathBuf> {
    std::env::var_os(key)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn sum_weight_files(dir: &Path) -> std::io::Result<u64> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;
    use std::sync::{Mutex, MutexGuard};

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    struct EnvGuard {
        _lock: MutexGuard<'static, ()>,
        saved: Vec<(&'static str, Option<OsString>)>,
    }

    impl EnvGuard {
        fn new(keys: &[&'static str]) -> Self {
            let lock = ENV_LOCK.lock().expect("env lock");
            let saved = keys
                .iter()
                .map(|&key| (key, std::env::var_os(key)))
                .collect();
            Self { _lock: lock, saved }
        }

        fn set_path(&self, key: &'static str, value: &Path) {
            // SAFETY: process environment mutation is serialized by ENV_LOCK
            // and restored by EnvGuard::drop before the lock is released.
            unsafe { std::env::set_var(key, value) };
        }

        fn remove(&self, key: &'static str) {
            // SAFETY: process environment mutation is serialized by ENV_LOCK
            // and restored by EnvGuard::drop before the lock is released.
            unsafe { std::env::remove_var(key) };
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            for (key, value) in self.saved.iter().rev() {
                // SAFETY: process environment mutation is serialized by
                // ENV_LOCK; the guard still owns the lock while restoring.
                unsafe {
                    match value {
                        Some(value) => std::env::set_var(key, value),
                        None => std::env::remove_var(key),
                    }
                }
            }
        }
    }

    #[test]
    fn auto_wired_limit_uses_runtime_home_for_hf_cache() {
        let guard = EnvGuard::new(&["HOME", "USERPROFILE", "HF_HOME", "HUGGINGFACE_HUB_CACHE"]);
        let home = tempfile::tempdir().expect("temp home");
        guard.set_path("HOME", home.path());
        guard.remove("USERPROFILE");
        guard.remove("HF_HOME");
        guard.remove("HUGGINGFACE_HUB_CACHE");

        let model_id = format!("runtime-home-test/{}", std::process::id());
        let snapshot_dir = home
            .path()
            .join(".cache/huggingface/hub")
            .join(format!("models--{}", model_id.replace('/', "--")))
            .join("snapshots")
            .join("abc123");
        std::fs::create_dir_all(&snapshot_dir).expect("create snapshot");
        std::fs::write(snapshot_dir.join("model.safetensors"), b"weight-bytes")
            .expect("write weights");
        std::fs::write(snapshot_dir.join("tokenizer.json"), b"ignored").expect("write metadata");

        assert_eq!(
            auto_wired_limit_bytes(&model_id),
            Some((1 << 30) + b"weight-bytes".len())
        );
    }
}
