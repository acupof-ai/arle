//! Runtime backend registry: one builder per compiled-in backend, registered
//! at binary startup before any engine load. Multiproc worker ranks re-exec
//! the same binary, so `register_all` in `main` covers them too.

#[path = "backends/cpu.rs"]
#[cfg(feature = "cpu")]
mod cpu;
#[path = "backends/cuda.rs"]
#[cfg(feature = "cuda")]
mod cuda;
#[path = "backends/hip.rs"]
#[cfg(feature = "hip")]
mod hip;
#[path = "backends/metal.rs"]
#[cfg(feature = "metal")]
mod metal;
#[path = "backends/vulkan.rs"]
#[cfg(feature = "vulkan")]
mod vulkan;

/// Register every backend this binary compiled in. Called once from `main`
/// before `cli::run()`. Multiproc worker ranks re-exec the same binary, so
/// `register_all` in `main` covers them too. Capabilities declared here gate
/// feature requests in one place (`infer_api::check_load_capabilities`); the
/// builders themselves carry no per-feature name checks.
pub fn register_all() {
    use infer_api::BackendCapabilities as C;
    #[cfg(feature = "cuda")]
    infer_api::register_backend(
        "cuda",
        cuda::build,
        C {
            mtp_spec_decode: true,
            kv_ssd_tier: true,
        },
    );
    #[cfg(feature = "metal")]
    infer_api::register_backend(
        "metal",
        metal::build,
        C {
            mtp_spec_decode: false,
            // The generic Metal executor spills to an L3 page tier; the
            // DeepSeek-OCR VLM executor does not — that model-kind-conditional
            // limit is checked inside the Metal builder after model resolution.
            kv_ssd_tier: true,
        },
    );
    #[cfg(feature = "hip")]
    infer_api::register_backend("hip", hip::build, C::NONE);
    #[cfg(feature = "vulkan")]
    infer_api::register_backend("vulkan", vulkan::build, C::NONE);
    #[cfg(feature = "cpu")]
    infer_api::register_backend("cpu", cpu::build, C::NONE);
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
