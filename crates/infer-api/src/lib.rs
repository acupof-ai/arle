//! Public `InferenceEngine` adapter over the rewrite stack.
//!
//! `infer-api` exposes the SAME public contract the legacy `infer::server_engine`
//! provides — the [`InferenceEngine`] trait, [`LoadedInferenceEngine`] enum, and
//! the request/output/stream/telemetry types — implemented over the rewrite
//! stack (`infer-core` `Engine` + `infer-server` `ServeHandle` + `infer-plan`
//! `SamplingParams`). Consumers (`agent` / `cli` / `train`) can later swap
//! `infer` -> `infer-api` with zero code changes. This crate is **additive**: it
//! migrates no consumer.
//!
//! Backend selection is by compiled feature flag, mirroring `infer`:
//! `metal` (Apple Silicon), `cuda` (Linux + NVIDIA), `cpu` (portable smoke / CI).
//!
//! # Wiring
//!
//! Every request flows `tokenize -> ServeHandle::submit -> collect ->
//! detokenize`, projecting the rewrite `CompletedRequest` back into the legacy
//! [`CompletionOutput`] shape (finish reason, [`TokenUsage`], token ids). The
//! shared body lives in [`ServeInferenceEngine`]; each [`LoadedInferenceEngine`]
//! variant dispatches to it.
//!
//! # Gaps vs. the legacy contract (follow-ups)
//!
//! These are the precise points where the rewrite stack cannot yet match the
//! legacy surface. The public types still carry the fields so the swap stays
//! zero-change; the data is simply unavailable until the rewrite stack grows
//! the hook.
//!
//! - **Streaming** — the rewrite `ServeHandle` exposes blocking `collect` only,
//!   with no per-token streaming hook. [`InferenceEngine::complete_stream`] runs
//!   the request to completion, then emits the full text + a terminal finish
//!   delta. Consumers receive a correct result but not incremental tokens. True
//!   token-granular streaming needs a `ServeHandle` streaming API.
//! - **Telemetry** — the rewrite `ServeHandle` does not surface scheduler
//!   counters (queue depth, occupancy, TTFT, ITL) across its thread channel.
//!   [`InferenceEngine::telemetry`] returns the empty [`EngineTelemetry`] shape.
//!   (`EngineTelemetry` also omits the legacy `model_arch` field, which is an
//!   `infer`-internal `ModelArchSummary` type.)
//! - **Per-token logprobs** — not surfaced by the rewrite stack;
//!   [`CompletionOutput::token_logprobs`] is always empty.
//! - **`session_id` / `trace_context` / `cancel`** — carried on
//!   [`CompletionRequest`] for surface compatibility but not yet honored by the
//!   rewrite engine (no sticky routing, no trace propagation, no mid-generation
//!   cancel). Every current consumer passes `None`.
//! - **CUDA backend** — structurally wired (the [`LoadedInferenceEngine::Cuda`]
//!   variant + dispatch typecheck under the Mac CUDA-Rust recipe), but
//!   [`LoadedInferenceEngine::load`] on CUDA returns an error: the real CUDA
//!   forward + `CudaExecutor` model builder + tokenizer resolution are Phase-0 /
//!   lead-owned and this crate may not edit them.
//! - **Train-only CUDA methods + LoRA types** — `forward_token_logits`,
//!   `remerge_student_lora`, `offload_engine_weights`, `reload_engine_weights`,
//!   and the `StudentLora{Layer,Matrices,Update}` types that `train` uses are
//!   NOT yet ported. They depend on the CUDA scheduler's direct model access,
//!   which the rewrite `ServeHandle` (thread-isolated, host-only seam) does not
//!   expose. `train` must stay on legacy `infer` until the rewrite CUDA path
//!   grows an OPD control surface. This is the single largest blocker to
//!   deleting `infer` for the `train` consumer specifically.

mod loaded;
mod serve_engine;
mod types;

pub use loaded::EngineLoadConfig;
#[cfg(any(feature = "metal", feature = "cuda", feature = "cpu"))]
pub use loaded::LoadedInferenceEngine;
pub use serve_engine::ServeInferenceEngine;
pub use types::{
    CompletionOutput, CompletionRequest, CompletionStreamDelta, CompletionStreamError,
    EngineTelemetry, FinishReason, InferenceEngine, PrefillPathStats, SamplingParams, SessionId,
    TokenUsage,
};
