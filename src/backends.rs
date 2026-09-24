//! Runtime backend registry: one builder per compiled-in backend, registered
//! at binary startup before any engine load. Multiproc worker ranks re-exec
//! the same binary, so `register_all` in `main` covers them too.

#[path = "backends/cpu.rs"]
#[cfg(feature = "cpu")]
mod cpu;
#[path = "backends/cuda.rs"]
#[cfg(feature = "cuda")]
mod cuda;
#[path = "backends/metal.rs"]
#[cfg(feature = "metal")]
mod metal;

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
    #[cfg(feature = "cpu")]
    infer_api::register_backend("cpu", cpu::build, C::NONE);
}
