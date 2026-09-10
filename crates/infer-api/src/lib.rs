//! Public `InferenceEngine` adapter over the rewrite stack.
//!
//! Backend selection is a runtime decision: the leaf binary registers one
//! builder per compiled-in backend via [`register_backend`], and
//! [`LoadedInferenceEngine::load_with_config`] dispatches by
//! [`EngineLoadConfig::backend`]. Every request flows `tokenize ->
//! ServeHandle::submit -> collect -> detokenize` via [`ServeInferenceEngine`].
//!
//! # Gaps (follow-ups)
//!
//! Public types carry the fields so consumers compile unchanged; the data is
//! unavailable until the rewrite stack grows the hook.
//!
//! - **Streaming** — `ServeHandle` is blocking-`collect` only.
//!   [`InferenceEngine::complete_stream`] emits the full text + a terminal delta,
//!   not incremental tokens.
//! - **Telemetry** — [`InferenceEngine::telemetry`] returns only queue/active
//!   counters (no latency / batch-occupancy / spec metrics).
//! - **Train-only CUDA methods + LoRA types** — `forward_token_logits`,
//!   `remerge_student_lora`, weight offload/reload, and the `StudentLora*` types
//!   need direct model access the host-only `ServeHandle` doesn't expose.

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
pub use loaded::LoadedInferenceEngine;
#[cfg(feature = "cuda")]
pub use loaded::build_cuda_engine;
/// Multiproc-serve spawn gate: which CUDA checkpoints join the env-driven TP
/// world (DSv4 + Qwen3.5/3.6 MoE). Consumed by `cli::serve_multiproc`.
#[cfg(feature = "cuda")]
pub use loaded::cuda_model_takes_multiproc_serve;
pub use loaded::{
    BackendBuilderFn, BackendCapabilities, EngineLoadConfig, KvCacheDtype, KvTierBudget,
    default_backend, is_backend_registered, register_backend,
};
#[cfg(feature = "cuda")]
pub use serve::serve_coordinator_http_dp;
pub use serve::{
    DEFAULT_MTP_DRAFT_TOKENS, DEFAULT_MTP_DRAFT_TOPK, ServeHttpOptions, ServeSpecOptions,
    ServeSpecType, checkpoint_has_mtp_head, default_kv_ssd_root, serve_http,
    validate_kv_ssd_config,
};
pub use serve::{ServeThread, serve_router_on_thread};
pub use serve_engine::{ServeInferenceEngine, model_id_from_path};
// DSv4 multiproc-serve control-plane relay, re-exported from `infer-server` so
// the `cli` coordinator/worker scaffold (`cli::serve_multiproc`) reaches it at
// the `infer-api` surface without depending on `infer-server` directly (mirrors
// the `infer-cuda` re-export pattern above).
pub use infer_server::{
    BuildIdentity, RelayCompletionDelta, RelayCoordinator, RelayEnvelope, RelayWorker,
    SamplingDefaults, ServeShutdown, WireStats, build_identity, coordinator_local_router,
    set_messages_dump_dir, set_sampling_defaults,
};
// Per-step student LoRA re-merge contract (OPD P2), re-exported from `infer-cuda`
// so consumers see them at the `infer-api` surface.
#[cfg(feature = "cuda")]
pub use infer_cuda::{
    SharedFp4BaseProjection, SharedFp8BaseProjection, StudentLoraLayer, StudentLoraMatrices,
    StudentLoraProjection, StudentLoraProjectionUpdate, StudentLoraUpdate, run_kernel_bench,
    set_qwen35_moe_experts_bf16_resident,
};
/// Rank-0 NCCL `unique_id` mint for the multiproc-serve coordinator.
#[cfg(feature = "nccl")]
pub use infer_cuda::{mint_nccl_unique_id_hex, nccl_unique_id_from_env};
#[cfg(feature = "cuda")]
pub use student_lora::{LoraHalf, load_student_lora_update, parse_student_adapter_name};
#[cfg(feature = "cuda")]
pub use types::RawLogits;
pub use types::{
    ChatPromptImage, ChatPromptMessage, CompletionOutput, CompletionRequest, CompletionStreamDelta,
    CompletionStreamError, EngineTelemetry, FinishReason, InferenceEngine, MultimodalChatRequest,
    SamplingParams, TokenUsage,
};
