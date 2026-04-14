# 2026-04-15 · Route-A revert + CUDA internal hygiene + extraction blueprint

## Context

Going into this session, the workspace was in a half-finished state from a
previous attempt to atomize `infer` into 8 crates (`infer-core`,
`infer-engine`, `infer-observability`, `infer-policy`, plus the existing
`infer-agent` / `infer-cli` / `infer-chat` / `infer-tools`). The compute
core (`backend/`, `model/`, `ops/`, `weight_loader.rs`) had been physically
relocated under `crates/infer-engine/src/`, but only via 12 `#[path]`
redirects in `infer/src/lib.rs` — every file in those directories still used
`use crate::backend::…` against the `infer` crate, and `crates/infer-engine`'s
own `lib.rs` was just a thin `AgentEngine` facade that did `use infer::…` to
reach back into the parent. The split was cosmetic. Compile times had not
improved, and there was no second consumer of any of the four shell crates.

In parallel, the engine layer was carrying a duplicate naming scheme:
`agent_engine.rs::AgentCompleteRequest` / `AgentCompleteOutput` /
`AgentEngine` / `LoadedAgentEngine` were exact field-by-field duplicates of
`server_engine.rs::CompleteRequest` / `CompleteOutput` / `ServerEngine` /
`LoadedServerEngine`, and `infer-chat` had two `ChatMessage` types (one for
the OpenAI wire format in `lib.rs`, one for the internal protocol in
`protocol.rs`) re-exported with confusing `Protocol*` aliases.

The user wanted the duplication eliminated, the names made unambiguous, and
the workspace shape consolidated. The deeper question — whether to extract
the CUDA layer at all — was deferred to a strategic discussion at the end.

## What Worked

The session landed five focused commits and one reverted-then-replaced
attempt, in this order:

1. **`d902090` — Route-A workspace revert.** Moved 117 source files back from
   `crates/infer-engine/src/` to `infer/src/` via `git mv`. Folded
   `infer-core` (RequestId/SessionId/InferenceMode/RequestEventKind),
   `infer-observability` (EngineEvent/EventSink), and `infer-policy`
   (admission/chunking policies) into `infer/src/{types,events,scheduler/policy}.rs`.
   Deleted the duplicate `agent_engine` facade and unified the engine
   contract under `infer::server_engine::{InferenceEngine, LoadedInferenceEngine}`.
   Renamed all the engine types for unambiguous semantics (`ServerEngine` →
   `InferenceEngine`, `CompleteRequest` → `CompletionRequest`, `Usage` →
   `TokenUsage`, etc.). Renamed all the OpenAI wire types in `infer-chat`
   to `OpenAi*` so the canonical protocol names could be re-exported clean
   from `infer_chat::protocol`. Net diff: 138 files, +991 / -1198 lines.
   274 CPU lib tests + 280 Metal lib tests passing throughout.

2. **`909e8cc` — Metal bridge cleanup.** User-landed in parallel: removed
   two dead C++ bridge sources (`metal_fused_ops.cpp`, `metal_fused_capi.cpp`)
   that had been superseded when the Metal path was rebuilt around `mlx-sys`.

3. **`d3136ba` — Dead Triton kernels + doc cleanup.** Identified and deleted
   four Triton kernels that were compiled by `build.rs` but had zero callers
   from any `.rs` file: `flash_attention_prefill_kernel.py` (HD128 FA, replaced
   in production by `flashinfer_single_prefill`), `attention_decode_kernel.py`
   (no FFI binding ever existed), `attention_reduce_kernel.py` (partner to
   the dead decode kernel), and the single-token `embedding_kernel` spec in
   `basic_kernels.py` (no FFI binding; `embedding_decode` and
   `embedding_batched` cover the live cases). Deleted the orphan `extern "C"`
   declaration `flash_attention_prefill_cuda` in `infer/src/backend/cuda/ffi.rs`.
   Removed the vestigial `replaced_cuda_files` BTreeSet in `build.rs` whose
   guarded files (`activation.cu`, `elementwise.cu`, `embedding.cu`) had
   already been deleted from disk in some earlier round but kept their
   bookkeeping. Also did the final Route-A doc sweep (5 markdown files).

4. **`26c8f39` — `ffi.rs` split + `cuda::prelude`.** Split the 1483-line
   `infer/src/backend/cuda/ffi.rs` into 10 domain submodules
   (`ffi/{attention, gemm, kv, norm, quant, sampling, embedding, elementwise,
   recurrent, misc}.rs`) with `pub(crate) use submodule::*;` re-exports from
   the parent so all 60+ existing `crate::backend::cuda::ffi::xxx` call sites
   resolved unchanged. Introduced `infer/src/backend/cuda/prelude.rs` as a
   single import point for the seven cross-cutting types that 25+ model and
   ops files share (`DeviceContext`, `DeviceVec`, `DeviceMatrix`,
   `RawDevicePtr`, `HiddenStates`, `PagedKVPool`, `FlashInferDecodeMetadata`).
   Auto-derived Triton's `cargo:rerun-if-changed` from a directory walk so
   the build script can no longer drift from the actual kernel set.

5. **`efcc991` — Prelude discipline tightened.** A first attempt put
   `TokenKVPool` in the prelude. The user reverted that, correctly: it has
   only 3 callers and shouldn't bloat the proto-API surface. The commit
   codifies the rule: a symbol enters the prelude only if it has ≥3
   consumers outside `backend/cuda/`, is stable for 6+ months, and would
   not force any currently-private `infer` type to become cross-crate `pub`
   on extraction.

After the code work, the strategic discussion: should the CUDA layer (and
`ops/`) be extracted to a standalone crate? Asked Codex for an independent
opinion in read-only consult mode. Codex came back with **partial agreement**
on the recommendation (option A — don't extract now) but a sharper independent
angle: the just-committed `prelude.rs` is itself a proto-crate boundary, and
keeping it disciplined lets a future kernel-only crate extraction land as a
mechanical 1-day refactor instead of a re-architecture.

The session also produced the forward blueprint
(`docs/plans/cuda-kernel-crate-extraction.md`) that documents the six trip
wires from `ROADMAP.md::Missing` (FA-3 H100, MLA / DeepSeek-V3, NCCL tensor
parallel, FP8 GEMM, speculative decode GPU, second consumer of kernel layer)
and spells out the exact `git mv` sequence and `Cargo.toml` / `build.rs`
edits to execute when any one of them fires. The plan stays narrow: kernel
crate only, `bootstrap.rs` stays in `infer` because it pulls
model/scheduler/tokenizer, no `infer-ops` extraction, no
`infer-scheduler-core` extraction, no `*-sys` split.

## Rule

**Cosmetic crate splits are worse than monolithic crates.** A crate that
exists only to host files reachable only via `#[path]` redirects, or whose
contents `use parent::…` to reach back into the same workspace, is not a
real boundary — it is a relocation pretending to be one. The Route-A revert
reset four such crates back to their single-crate origin in one focused PR.
The takeaway is not "never split" — it is "the split must do work the
monolith couldn't, and that work has to be observable as a real consumer or
a real artifact, not as a `Cargo.toml` row."

**The proto-API discipline rule:** when you anticipate a future crate split,
landing the split's *public API surface* as an internal module today
(`backend/cuda/prelude.rs`) is the cheapest and lowest-risk way to lock in
the architectural commitment. A narrow, explicitly-policed proto-API is
worth more than ten "future plans" in `docs/plans/` — because consumers are
already migrated to it, and the eventual extraction becomes a `pub(crate)`
→ `pub` diff on a known small set of symbols, plus a `git mv`. The trip
wires for executing the migration go in writing, sourced from `ROADMAP.md`
not from speculation, so when the trigger fires the response is "execute
the plan", not "decide again".

**Naming that survives a year, not a quarter.** Rename `Complete` → `Completion`
because "Complete" is a verb-ambiguous prefix that drifts into "the complete
config" or "marked as complete". Rename `Usage` → `TokenUsage` because
`Usage` could mean memory, GPU, quota, or anything else. Rename `ServerEngine`
→ `InferenceEngine` because the type is used outside HTTP servers. Two
ChatMessage types in the same crate is a name collision pretending to be a
type system, even if `pub use` aliases hide it; the wire format gets the
explicit `OpenAi*` prefix and the protocol type keeps the canonical short
name. Code is read more often than written; ambiguity in the read costs more
than verbosity in the write.
