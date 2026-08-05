//! Public `InferenceEngine` adapter over the rewrite stack.
//!
//! Exposes the same public contract as the legacy `infer::server_engine` (the
//! [`InferenceEngine`] trait, [`LoadedInferenceEngine`] enum, request/output/
//! stream/telemetry types) over the rewrite stack, so consumers can later swap
//! `infer` -> `infer-api` with zero code changes. Backend selection is by
//! compiled feature (`metal`/`cuda`/`hip`/`vulkan`/`cpu`). Every request flows
//! `tokenize -> ServeHandle::submit -> collect -> detokenize` via
//! [`ServeInferenceEngine`].
//!
//! # Gaps vs. the legacy contract (follow-ups)
//!
//! Public types carry the fields so the swap stays zero-change; the data is
//! unavailable until the rewrite stack grows the hook.
//!
//! - **Streaming** — `ServeHandle` is blocking-`collect` only.
//!   [`InferenceEngine::complete_stream`] emits the full text + a terminal delta,
//!   not incremental tokens.
//! - **Telemetry** — [`InferenceEngine::telemetry`] returns empty
//!   [`EngineTelemetry`] (no scheduler counters across the thread channel; also
//!   omits the legacy `model_arch` field).
//! - **Per-token logprobs** — [`CompletionOutput::token_logprobs`] is always empty.
//! - **`session_id` / `trace_context` / `cancel`** — carried on
//!   [`CompletionRequest`] but not yet honored; every current consumer passes `None`.
//! - **CUDA backend** — wired + typechecks, but [`LoadedInferenceEngine::load`]
//!   errors: the real CUDA forward + builder are lead-owned.
//! - **Train-only CUDA methods + LoRA types** — `forward_token_logits`,
//!   `remerge_student_lora`, weight offload/reload, and the `StudentLora*` types
//!   need direct model access the host-only `ServeHandle` doesn't expose; `train`
//!   stays on legacy `infer` until the CUDA path grows an OPD control surface.

mod loaded;
mod serve;
mod serve_engine;
#[cfg(feature = "cuda")]
mod student_lora;
mod types;

pub use infer_core::engine_forward_busy_micros;
pub use infer_seam::{CommBackend, CudaRuntimeFlags, MetalRuntimeFlags};

#[must_use]
pub const fn kernel_build_id() -> &'static str {
    #[cfg(feature = "cuda")]
    {
        cuda_kernels::KERNEL_BUILD_ID
    }
    #[cfg(not(feature = "cuda"))]
    {
        "unreported"
    }
}
/// Kernel families compiled into this binary, comma-separated. A fast path
/// missing here is a build defect, not a runtime choice — reading it is the
/// cheapest way to tell a stub build from a real one.
#[must_use]
pub const fn kernel_capabilities() -> &'static str {
    #[cfg(feature = "cuda")]
    {
        cuda_kernels::KERNEL_CAPABILITIES
    }
    #[cfg(not(feature = "cuda"))]
    {
        ""
    }
}

#[cfg(feature = "cuda")]
pub use loaded::CudaWorkerEngine;
#[cfg(any(
    feature = "metal",
    feature = "cuda",
    feature = "hip",
    feature = "vulkan",
    feature = "cpu"
))]
pub use loaded::LoadedInferenceEngine;
/// Multiproc-serve spawn gate: which CUDA checkpoints join the env-driven TP
/// world (DSv4 + Qwen3.5/3.6 MoE). Consumed by `cli::serve_multiproc`.
#[cfg(feature = "cuda")]
pub use loaded::cuda_model_takes_multiproc_serve;
#[cfg(feature = "cuda")]
pub use loaded::{DSV4_AUTO_CONTEXT_CEILING, cuda_model_is_dsv4};
pub use loaded::{EngineLoadConfig, KvCacheDtype, KvTierBudget};
#[cfg(feature = "cuda")]
pub use serve::serve_coordinator_http;
pub use serve::{
    DEFAULT_MTP_DRAFT_TOKENS, DEFAULT_MTP_DRAFT_TOPK, ServeHttpOptions, ServeSpecOptions,
    ServeSpecType, default_kv_ssd_root, serve_http, validate_kv_ssd_config,
};
#[cfg(any(
    feature = "metal",
    feature = "cuda",
    feature = "hip",
    feature = "vulkan",
    feature = "cpu"
))]
pub use serve::{ServeThread, bind_and_serve, serve_router_on_thread};
pub use serve_engine::ServeInferenceEngine;
// DSv4 multiproc-serve control-plane relay, re-exported from `infer-server` so
// the `cli` coordinator/worker scaffold (`cli::serve_multiproc`) reaches it at
// the `infer-api` surface without depending on `infer-server` directly (mirrors
// the `infer-cuda` re-export pattern above).
pub use infer_server::{
    BuildIdentity, PendingRelayCoordinator, RelayChannel, RelayCompletionDelta, RelayCoordinator,
    RelayEnvelope, RelayWorker, SamplingDefaults, ServeShutdown, TcpChannel, WireRequest,
    WireStats, build_identity, coordinator_local_router, set_messages_dump_dir,
    set_sampling_defaults, set_tick_broadcaster,
};
// Per-step student LoRA re-merge contract (OPD P2), re-exported from `infer-cuda`
// so consumers see them at the `infer-api` surface (mirrors the legacy
// `infer::server_engine::StudentLora*` path the `train` crate couples to).
#[cfg(feature = "cuda")]
pub use infer_cuda::{
    SharedFp8BaseProjection, StudentLoraLayer, StudentLoraMatrices, StudentLoraProjection,
    StudentLoraProjectionUpdate, StudentLoraUpdate, set_qwen35_moe_experts_bf16_resident,
};
/// Rank-0 NCCL `unique_id` mint for the multiproc-serve coordinator.
#[cfg(feature = "nccl")]
pub use infer_cuda::{mint_nccl_unique_id_hex, nccl_unique_id_from_env};
#[cfg(feature = "metal")]
pub use infer_metal::recommended_max_working_set_size_bytes as metal_recommended_max_working_set_size_bytes;
#[cfg(feature = "cuda")]
pub use student_lora::{LoraHalf, load_student_lora_update, parse_student_adapter_name};
#[cfg(feature = "cuda")]
pub use types::RawLogits;
pub use types::{
    ChatPromptImage, ChatPromptMessage, CompletionOutput, CompletionRequest, CompletionStreamDelta,
    CompletionStreamError, EngineTelemetry, FinishReason, InferenceEngine, MultimodalChatRequest,
    PrefillPathStats, SamplingParams, SessionId, TokenUsage,
};
