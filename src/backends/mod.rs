//! Runtime backend registry: one builder per compiled-in backend, registered
//! at binary startup before any engine load. Multiproc worker ranks re-exec
//! the same binary, so `register_all` in `main` covers them too.

#[cfg(feature = "cpu")]
mod cpu;
#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "hip")]
mod hip;
#[cfg(feature = "metal")]
mod metal;
#[cfg(feature = "vulkan")]
mod vulkan;

/// Register every backend this binary compiled in. Called once from `main`
/// before `cli::run()`.
pub fn register_all() {
    #[cfg(feature = "cuda")]
    infer_api::register_backend("cuda", cuda::build);
    #[cfg(feature = "metal")]
    infer_api::register_backend("metal", metal::build);
    #[cfg(feature = "hip")]
    infer_api::register_backend("hip", hip::build);
    #[cfg(feature = "vulkan")]
    infer_api::register_backend("vulkan", vulkan::build);
    #[cfg(feature = "cpu")]
    infer_api::register_backend("cpu", cpu::build);
}

/// Resolve `model_path` to a `.gguf` checkpoint: either the file itself or a
/// directory containing exactly one `*.gguf`. No HF-repo resolution (that
/// is the Metal facade's surface); a plain file-path check with a clear
/// error is the MVP contract for GGUF-only backends.
#[cfg(any(feature = "hip", feature = "vulkan"))]
pub(crate) fn resolve_gguf_path(
    model_path: &str,
    backend_label: &str,
) -> anyhow::Result<std::path::PathBuf> {
    use anyhow::{Context, bail, ensure};

    let path = std::path::Path::new(model_path);
    if path.is_file() {
        ensure!(
            path.extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf")),
            "{backend_label} backend serves GGUF checkpoints only; {model_path} is not a .gguf file"
        );
        return Ok(path.to_path_buf());
    }
    if path.is_dir() {
        let mut ggufs: Vec<std::path::PathBuf> = std::fs::read_dir(path)
            .with_context(|| format!("read model dir {model_path}"))?
            .filter_map(|entry| entry.ok().map(|e| e.path()))
            .filter(|p| {
                p.is_file()
                    && p.extension()
                        .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
            })
            .collect();
        return match ggufs.len() {
            1 => Ok(ggufs.remove(0)),
            0 => bail!(
                "no .gguf file in {model_path}; the {backend_label} backend serves GGUF checkpoints only"
            ),
            n => bail!("{n} .gguf files in {model_path}; pass the .gguf file path explicitly"),
        };
    }
    bail!(
        "{backend_label} model path {model_path} not found \
         (expected a .gguf file or a directory containing exactly one)"
    )
}
