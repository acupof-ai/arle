# Global dead-code deletion, wave 4 — non-CUDA crates, 2026-08-19

> Status: verified (local Mac: check + clippy + CPU/Metal test lanes)

## Goal

Fourth wave of the global dead-code deletion: remove zero-caller code in the
non-CUDA crates that the CUDA-focused waves 1–3 did not reach. Zero runtime
change is the correctness criterion.

## Scope

18 files, **+20 / −446 (net −426 lines)** across 10 crates. Every deletion
verified as zero-caller before removal (repo-root grep across all file types,
including cfg-gated callers, re-exports, doc links, and macro/FFI paths).
Findings from a 47-agent workflow scan (3 scan agents + adversarial verify),
37 confirmed dead, 5 refuted as live.

| Crate | Deleted symbols | Lines |
|-------|----------------|------:|
| infer-server | `EchoExecutor` struct+impl, `ServeHandle::shutdown`, `ServeHandle::spawn_with_engine_builder`, `ServeHandle::resume_admissions`, `set_tick_broadcaster`, `RelayCompletionDelta::text`, `PendingRelayCoordinator::accept` | ~130 |
| infer-api | `serve_coordinator_http`, `ServeSpecType::label`, `ServeSpecOptions::requested`, `ServeHttpOptions::new`, `CompletionRequest::new`, `MultimodalChatRequest::new`, `ChatPromptMessage::system/assistant`, `CompletionStreamDelta::text`, `LoadedInferenceEngine::resume_admissions` | ~120 |
| infer-api/serve_engine | `ServeInferenceEngine::resume_admissions` (transitively dead after `LoadedInferenceEngine::resume_admissions` removal) | ~7 |
| infer-seam | `BufferedDiffusionExecutor::new`, `BufferedDiffusionExecutor::into_inner` | ~10 |
| agent | `run_turn_interruptibly` (delegates to `_with_callbacks`; zero callers) | ~22 |
| infer-core | `Engine::submit_request` (`submit_request_with_options` is the live path) | ~5 |
| infer-util | `init_default` | ~5 |
| kv-native-sys | `chunk_sub` re-export (function still used internally in `kv_tier.rs`) | ~1 |
| infer-moe | `RoutingDecision::expert_ids/weights`, `ScoringFunc::from_config_str`, `TopkMethod::from_config_str`, `MoeConfig::num_shared_experts` field (write-only) | ~40 |
| infer-topo | `build_pp_groups`, `build_attn_dp_groups`, `build_attn_owner_groups`, `build_moe_dp_groups`, `build_moe_tp_groups`, `RankCoord` write-only fields (`world_rank`/`tp_rank`/`pp_rank`/`moe_tp_rank`/`moe_ep_rank`/`moe_dp_rank`) | ~110 |
| autograd | Add missing `#[cfg(feature = "cuda")]` gate on `CudaFp4E2M1GroupStorage` (pre-existing build break in `cpu,no-cuda` lane from W4AFP8 merge) | +1 |

Re-export cleanup: `infer-server/lib.rs` (`set_tick_broadcaster`),
`infer-api/lib.rs` (`serve_coordinator_http`, `set_tick_broadcaster`),
`infer-topo/lib.rs` (5 dead `build_*_groups`).

Doc fixes: `quiesce_admissions` doc comment retargeted from
`Self::resume_admissions` to `Self::ensure_kv_pool_and_resume_admissions`
(the live pairing); `RelayEnvelope::WorkerHello` doc retargeted from
`PendingRelayCoordinator::accept` to `accept_symmetric`;
`serve_coordinator_http_dp` doc updated after `serve_coordinator_http` deletion.

## The resume_admissions chain

Three methods formed a dead call chain:
`LoadedInferenceEngine::resume_admissions` → `ServeInferenceEngine::resume_admissions`
→ `ServeHandle::resume_admissions`. The live OPD bracket uses
`quiesce_admissions` + `ensure_kv_pool_and_resume_admissions` (which calls
`engine.resume_serving()` directly, bypassing this chain). All three deleted.

## What was NOT deleted (refuted as live)

- `ChatPromptMessage::user` — called via `Self::user` inside `user_with_images`
- `LoadedInferenceEngine::forward_training_taps` — live via `arle train spec-draft`
- `ServeInferenceEngine::resume_admissions` — refuted as dead (had a caller:
  `LoadedInferenceEngine::resume_admissions`); deleted in the same commit once
  the caller was removed
- `ServeInferenceEngine::forward_training_taps` — live via spec-draft
- `TeacherForward` trait import — live (trait methods on `InferTeacher`)

## Pre-existing build break fixed

`CudaFp4E2M1GroupStorage` (W4AFP8, commit `2a3a2164f`) had
`#[cfg_attr(feature = "no-cuda", allow(dead_code))]` but no
`#[cfg(feature = "cuda")]` gate. In the `cpu,no-cuda` lane the struct compiled
without the `Arc` import (gated on `any(metal, cuda)`), breaking the build.
Added the missing gate. The `Drop` impl and enum variant were already gated.

## Verification

```
cargo check -p infer-util -p infer-core -p infer-moe -p infer-topo
  -p infer-seam -p kv-native-sys                              → clean
cargo check -p infer-server                                    → clean
cargo check -p infer-api --no-default-features --features metal,no-cuda → clean
cargo clippy -p infer-util -p infer-core -p infer-moe -p infer-topo
  -p infer-seam -p kv-native-sys -p infer-server -- -D warnings → zero warnings
cargo test -p arle --profile release-fast
  --no-default-features --features cpu,no-cuda,cli            → 5/5 pass
cargo test -p cli --release --no-default-features
  --features metal,no-cuda                                    → 4/4 pass
```

Pre-existing clippy error on HEAD (not from this wave):
`autograd/src/backend/cpu_math.rs:87` manual `is_multiple_of` — left for the
owner of that file.

## Coexistence with in-flight W4AFP8 work

The user's uncommitted W4AFP8 changes (`infer-cuda/src/{loader,moe}.rs`,
`dsv4/weights.rs` `w4afp8_gemv_tables` field) were left untouched. The
`weights.rs` commit contains only the `num_shared_experts` param removal
(staged via `git apply --cached` on a single-hunk patch); the user's
`w4afp8_gemv_tables` addition remains in the working tree.

## Rule

Dead-code scans should verify re-exports and doc links, not just direct
callers — a pub re-export with zero downstream callers is as dead as the
original definition. And a `cfg_attr(allow(dead_code))` is not a cfg gate:
a struct that references CUDA types needs `#[cfg(feature = "cuda")]`, not
just a warning suppression.
