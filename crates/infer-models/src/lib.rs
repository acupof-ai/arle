//! Model architectures (`ModelArch` impls) — R3 track.
//!
//! Stub: to be filled with the `infer_seam::ModelArch` implementations for the
//! Qwen3 / Qwen3.5 / DeepSeek-V4 families, written against `Communicator`
//! (TP/EP collectives) and `KvPool`. The numerical forward math is PORTED from
//! the existing `infer/src/model/*` (not re-derived) and gated by the parity
//! suite (kv_precision_parity / greedy_consistency / e2e) per the
//! verification-targets doc.
//!
//! Depends only on the stable `infer-plan` + `infer-seam` contracts (+ the
//! `*-spec` config crates when the real ports land).
