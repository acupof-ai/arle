# oMLX-Inspired Optimizations For ARLE

Date: 2026-05-02
Status: execution plan, docs-only
Scope: Metal serving, MLX bridge, prefix/KV cache, multimodal cache, agent control plane

This report maps each oMLX acceleration source to the ARLE code that already
exists today, then names the smallest implementation path that preserves ARLE's
runtime-first boundaries.

## External Signals

- oMLX README: tiered KV cache uses a hot in-memory tier plus cold SSD tier;
  cold blocks are persisted in safetensors and restored on matching prefixes,
  even after restart. It also advertises continuous batching and an EnginePool
  for LLM/VLM/embedding/reranker models:
  <https://github.com/jundot/omlx/blob/main/README.md>
- oMLX HN launch note: coding-agent long-context TTFT drops from 30-90s to
  1-3s on follow-up requests by restoring cached KV from SSD:
  <https://news.ycombinator.com/item?id=47247294>
- Apple MLX M5 note: MLX uses Metal 4 TensorOps/MPP on M5 Neural Accelerators;
  reported TTFT speedups are 3.33-4.06x while decode gains are 1.19-1.27x, and
  the accelerator path requires macOS 26.2+:
  <https://machinelearning.apple.com/research/exploring-llms-mlx-m5>
- vLLM-Omni prefix cache design: multimodal correctness needs both KV prefix
  cache and encoder-output cache keyed by image/media hash:
  <https://docs.vllm.ai/projects/vllm-omni/en/latest/design/feature/prefix_caching/>

## Decision Summary

| ID | Optimization | ARLE landing path | Priority | Main owner files |
| --- | --- | --- | --- | --- |
| P0-1 | Metal T2 SSD KV persistence | Extend current Metal Qwen3.5 prefix snapshots to disk first, then converge onto block-level Metal KV/Radix path | P0 | `infer/src/backend/metal/runtime.rs`, `request_state.rs`, `infer/src/kv_tier/transport/disk.rs` |
| P0-2 | `mlx-sys` 0.32+ / M5 Neural Accelerators | Update vendored MLX and bridge signatures; gate by SDK/macOS checks | P0 | `crates/mlx-sys/vendor/mlx/`, `crates/mlx-sys/src/*`, `infer/src/backend/metal/mlx.rs` |
| P1-1 | Multimodal content-hash prefix cache | Reuse the existing multimodal plan's `mm_hash` + `EncoderCache` + `RadixCache` shape | P1 | `infer/src/vision/*`, `infer/src/prefix_cache.rs`, `infer/src/http_server/openai_v1.rs` |
| P1-2 | Metal `kv_pool` LRU + watermarks | Promote `MetalKVPool` from refcount pool to eviction-capable hot pool | P1, blocks canonical P0-1B | `infer/src/backend/metal/kv_pool.rs`, `prefix_cache.rs`, `runtime.rs` |
| P1-3 | Metal scalar materialize audit | Audit all hot-path `.item()` / `eval()` / `async_eval()` boundaries | P1 | `infer/src/backend/metal/*`, `crates/mlx-sys/src/*` |
| P2-1 | DFlash with continuous batching | Generalize existing Qwen3.5 DFlash batch partitioning and cache sync | P2 | `infer/src/backend/metal/runtime.rs`, `dflash.rs`, `request_state.rs`, `gdr.rs` |
| P2-2 | Multi-model engine pool | Add control-plane engine registry around `LoadedInferenceEngine`; keep runtime truth in `infer` | P2 | `infer/src/server_engine/loaded.rs`, `crates/cli/src/serve.rs`, `src/main.rs` |

Execution order: P0-2 first, P1-2 second, P0-1 third. P1-3 can run in parallel
after P0-2 compiles. P1-1 waits for the multimodal branch. P2 items wait until
P0 data proves the Metal lane is worth widening.

## Current ARLE Facts

- Metal scheduler hot path is `run_metal_scheduler_runtime` in
  `infer/src/backend/metal/runtime.rs`. It drives decode-first continuous
  batching plus optional prefill chunks.
- Metal Qwen3.5 live prefix reuse is not yet the generic
  `backend/metal/prefix_cache.rs` bridge. The hot code is
  `MetalLivePrefixRuntime` in `runtime.rs`, which stores in-memory
  `Qwen35PrefixSnapshot` values exported/imported by
  `infer/src/backend/metal/request_state.rs`.
- `infer/src/backend/metal/kv_pool.rs` has a token-slot `SlotLedger`,
  refcounts, scatter/gather helpers, and unit tests. It has no LRU,
  watermarks, disk payload extraction, or coordinator path.
- `infer/src/backend/metal/prefix_cache.rs` wraps `RadixCache` and maps
  blocks to Metal token slots, but it is still a bridge/accounting layer.
- CUDA already has the richer tiered-KV shape:
  `infer/src/prefix_cache.rs`, `infer/src/kv_tier/*`, and
  `infer/src/scheduler/cuda/*` wire `RadixCache`, `ReadmissionPlan`,
  `Coordinator`, and `DiskStore`.
- `infer/src/kv_tier/transport/disk.rs` currently stores one content-addressed
  block per `BlockFingerprint` using a postcard header plus opaque raw payload
  bytes. Do not force safetensors into ARLE unless a later benchmark proves the
  format change is worth the churn.
- `crates/mlx-sys/vendor/mlx/mlx/version.h` pins MLX `0.31.1`. The bridge is
  vendored; this is not a one-line FetchContent tag change.
- Metal memory-limit hooks already exist:
  `mlx_get_active_memory`, `mlx_get_peak_memory`, `mlx_set_memory_limit`,
  `mlx_set_cache_limit`, and `mlx_set_wired_limit` are bridged through
  `infer/src/backend/metal/mlx.rs`.
- OpenAI chat content currently flattens non-text parts away in
  `crates/chat/src/lib.rs`; multimodal work must change that before P1-1 can
  become real.

## P0-1 Metal T2 SSD KV Persistence

### oMLX Source Of Speed

oMLX turns repeated agent prefixes from "recompute the full prompt" into
"restore matching KV blocks from SSD". The signature workload is a coding agent
with a large, slowly shifting prompt prefix.

### ARLE Mapping

There are two ARLE landing layers:

1. **P0-1A: Qwen3.5 snapshot persistence.** This is the fastest useful path
   because Qwen3.5's current hot prefix reuse already exports
   `Qwen35PrefixSnapshot { token_ids, kv_flat, gdr_flat, cache_len, kv_capacity }`.
   Persist those snapshots through `DiskStore` and load them during
   `MetalQwen35PrefixRuntime::prepare_request`.
2. **P0-1B: canonical block-level Metal T2.** After P1-2, promote
   `MetalKVPool` + `MetalPrefixCache` into the hot path and persist block
   payloads with `BlockFingerprint`, `BlockLocation::Disk`, and
   `ReadmissionPlan` semantics. This is the long-term parity path with CUDA.

P0-1A should not introduce a third cache. It should extend
`MetalQwen35PrefixRuntime` with a disk-backed index and then make P0-1B delete
or absorb that special path once block-level Metal is live.

### Implementation Shape

- Add a Metal disk-cache options struct:
  `MetalKvDiskOptions { dir, max_bytes, high_watermark, low_watermark,
  fsync_each_block }`. Thread it through `MetalBackendOptions`,
  `metal_serve`, `metal_request`, and `scripts/start_metal_serve.sh`.
- Add `Qwen35PrefixSnapshot` encode/decode helpers. Required fields:
  token ids, model/config fingerprint, tensor count, tensor names, dtype,
  shape, cache length, capacity, and raw bytes. The missing primitive today is
  an MLX array byte export/import for bf16/f16 arrays; add that to
  `crates/mlx-sys/src/mlx_bridge.cpp` and wrap it in `metal/mlx.rs`.
- Use `DiskStore::put_block` / `get_block` instead of inventing a parallel
  filesystem layout. Use `BlockFingerprint::compute_from_tokens` or a new
  model-aware fingerprint helper only at the persistence boundary.
- On publish, spill block-aligned snapshots once memory pressure crosses the
  high watermark. Keep hot snapshots in memory until the low watermark target
  is reached.
- On prepare, check memory first, then disk. A disk hit imports the snapshot,
  records prefix-hit metrics, and should not run prefill for the reused span.
- Add startup reconciliation: scan disk records, discard wrong model/config
  fingerprints, and build an index without trusting paths from persisted data.

### Acceptance

- New Metal test or smoke harness: first run of a 30k-token Qwen3.5 prompt
  populates disk; second run with the same prefix imports from disk and has
  TTFT at least 3x lower; process restart keeps the hit.
- Metrics expose memory hits, disk hits, disk misses, disk bytes, and import
  latency.
- `cargo test --release --no-default-features --features metal,no-cuda`
  passes on Apple Silicon.
- `cargo check -p infer --no-default-features --features cuda,no-cuda` passes
  on Mac to protect always-on skeletons.
- Bench entry under `docs/experience/wins/` is mandatory because this touches
  `infer/src/backend/metal/` and `kv_tier`.

### Risks

- MLX raw-byte export can accidentally scalarize or synchronize the hot path.
  Serialize only after the snapshot has already been materialized for prefix
  publication, and keep writes off the scheduler critical step.
- Snapshot format is Qwen3.5-specific because it includes GDR state. Do not
  pretend it is a generic KV block format.
- `BlockId` is a pool slot id, not a content hash. Only `BlockFingerprint`
  crosses the disk boundary.

## P0-2 Upgrade `mlx-sys` To MLX 0.32+

### oMLX / Apple Source Of Speed

The M5 gain is MLX-level, not an ARLE kernel rewrite. Apple's published data
shows TTFT speedups around 3.3-4.1x for tested LLMs while generation improves
only about 1.2x because decode is bandwidth-bound.

### ARLE Mapping

- Reality check on 2026-05-02: upstream MLX tags do not expose a 0.32+ line to
  vendor. M5 support is already present before ARLE's vendored 0.31.1 snapshot,
  so the shipped action is a load-time MLX/macOS/chip/NAX diagnostic plus a
  pending M5 bench gate, not a forced vendored rewrite to a nonexistent tag.
- When a newer upstream snapshot is selected, update
  `crates/mlx-sys/vendor/mlx/` from 0.31.1 and include local vendored
  dependencies if upstream CMake changed them. Do not bump for the M5 claim
  alone while 0.31.1 already covers the MLX-side accelerator gate.
- Rebuild `crates/mlx-sys/src/mlx_bridge.cpp`,
  `mlx_qwen35_model.cpp`, `mlx_dflash_draft_model.cpp`,
  `mlx_qwen35_moe_block.cpp`, and `mlx_common.h` against the new API.
- Keep all MLX access behind `infer/src/backend/metal/mlx.rs`; do not leak
  `mlx-sys` into scheduler/model cross-backend modules.
- Add a build-time or startup diagnostic for macOS 26.2+ when users request
  the M5 accelerator path. Older Macs must still compile and run, just without
  the accelerator claim.

### Acceptance

- `cargo build --release --no-default-features --features metal,no-cuda`.
- `cargo test --release --no-default-features --features metal,no-cuda`.
- RoPE tripwires still pass: batched decode must keep `[B, H, S, D]` and
  array `rope_offsets`, including same-length batches.
- M5 bench: `scripts/bench_guidellm.sh metal-mlx-032-m5` against a 0.31.1
  baseline. If no M5 is local, create a `pending-remote` wins entry.
- M3/M4 bench: prove no regression outside M5. If regression appears, keep a
  compile-time fallback or defer the bump.

### Risks

- MLX minor bumps can change dtype behavior or lazy-eval boundaries. Treat any
  JSON baseline drift as a correctness investigation first, not an automatic
  regenerate.
- Bridge functions that expose Metal 4 behavior must remain optional on older
  SDKs.

## P1-1 Multimodal Content-Hash Prefix Cache

### Source Of Speed

The reusable work in VLM requests has two parts: visual encoder output and LLM
KV over the expanded image-token span. oMLX advertises VLMs on the same tiered
cache stack; vLLM-Omni documents why encoder cache and KV prefix cache both
matter for correctness.

### ARLE Mapping

Use the existing plan in
`docs/plans/2026-05-01-multimodal-vision-cuda-metal.md` instead of inventing a
new cache:

- `infer/src/vision/hash.rs`: canonical media hash. Prefer the existing plan's
  BLAKE3 `mm_hash` over SHA-256 so it aligns with ARLE `BlockFingerprint`
  practice.
- `infer/src/vision/encoder_cache.rs`: byte-budgeted
  `EncoderCache<mm_hash -> vision_embeds>`.
- `infer/src/prefix_cache.rs`: include `mm_hash` in block fingerprinting when
  the token span contains vision placeholder ids.
- `infer/src/http_server/openai_v1.rs` and `crates/chat/src/lib.rs`: stop
  flattening image parts away; normalize URL/base64/file inputs into canonical
  decoded pixels before hashing.

### Acceptance

- Same image, different surrounding text: encoder cache hit, no false KV hit
  unless the text prefix also matches.
- Same text, different image: no KV prefix hit.
- Same image and same text prefix: both encoder and KV prefix hit.
- Bench target: reproduce a large TTFT drop on repeated image prompts; do not
  accept a headline multiplier without raw before/after tables.

## P1-2 Metal `kv_pool` LRU + Watermarks

### Why It Blocks P0-1B

`MetalKVPool` can allocate, refcount, scatter, gather, and release token slots,
but it does not decide what to evict. T2 persistence needs an explicit policy
for which blocks become cold.

### ARLE Mapping

- Extend `SlotLedger` in `infer/src/backend/metal/kv_pool.rs` with
  `last_access_tick`, `pin_count` or active-request protection, and byte/token
  accounting.
- Add `register_access(slots)` at decode and prefix-attach points.
- Add `select_eviction_candidates(target_tokens)` and return block-aligned
  candidates only.
- Connect `infer/src/backend/metal/prefix_cache.rs` to the same candidate
  model so `RadixCache` remains the source of truth for cached prefixes.
- Expose high/low watermarks in Metal runtime config. Reuse CUDA defaults only
  after a Metal bench confirms they make sense.

### Acceptance

- CPU-only unit tests for LRU order, refcount protection, block alignment, and
  watermark selection.
- Metal smoke: under a tiny pool, old inactive prefixes spill first; active
  rows are never evicted.

## P1-3 Metal Scalar Materialize Audit

### Source Of Speed

Metal/MLX is lazy. `.item()`, `eval()`, and `async_eval()` are scheduling
boundaries. A misplaced scalar read can turn an overlapped graph into a
synchronous per-token stall.

### ARLE Mapping

Run:

```bash
rg -n 'eval\(|async_eval\(|\.item\(' infer/src/backend/metal crates/mlx-sys/src
```

Classify every hit:

- **Green:** setup, tests, load-time validation, or explicit sampling boundary.
- **Yellow:** once per prefill chunk or once per request, requires comment if
  retained.
- **Red:** per-token or per-row inside scheduler hot loops, must be removed or
  justified by a failing correctness test.

Known sensitive files: `runtime.rs`, `request_state.rs`, `qwen35.rs`,
`sampling.rs`, `mlx.rs`, `crates/mlx-sys/src/mlx_bridge.cpp`, and
`mlx_qwen35_model.cpp`.

### Acceptance

- Add a `docs/experience/wins/2026-05-XX-metal-eval-audit.md` entry listing
  every hot-path boundary and its classification.
- If code changes are made, run a matched Metal before/after bench. Small
  effects require matched A/B; do not claim a 1-3% win from unmatched runs.

## P2-1 DFlash With Continuous Batching

### Current ARLE State

`infer/src/backend/metal/runtime.rs` already partitions Qwen3.5 decode rows:

- DFlash rows `>= 2` go through `execute_qwen35_dflash_packed_batch`.
- One DFlash row falls back to `execute_decode_single`.
- Plain rows go through `execute_qwen35_packed_decode_batch`.

That means the P2 question is no longer "can DFlash batch at all"; it is
"can DFlash remain efficient and correct in mixed, long-lived continuous
batches with varlen rows, cache shrink/grow, and fallback rows?"

### Implementation Shape

- Define a mixed-batch contract: DFlash-ready rows, stale DFlash rows, and
  plain rows must all update per-row cursors and packed KV/GDR state exactly
  once.
- Promote fallback metrics into acceptance gates so a "batched" path that
  falls back most ticks is rejected.
- Keep Qwen3.5/Qwen3.6 only. Do not widen to Qwen3 until a target/draft pair
  has the same validation level.

### Acceptance

- Mixed test: 3 DFlash rows + 5 plain rows, with one stale DFlash row, no cache
  cursor drift, no double token accounting.
- Long-context bench: DFlash mixed workload improves decode throughput by at
  least 1.3x over plain continuous batching at matched prompt/output shapes.

## P2-2 Multi-Model Engine Pool

### Source Of Speed

oMLX keeps LLM, VLM, embedding, and reranker engines in one service and evicts
least-recently-used models under a process memory limit. This speeds agent
workflows by avoiding repeated model process startup and by colocating helper
models.

### ARLE Mapping

ARLE has one loaded engine at a time:

- `infer/src/server_engine/loaded.rs` owns `LoadedInferenceEngine`.
- `crates/cli/src/lib.rs` loads a single engine for the local agent loop.
- `crates/cli/src/serve.rs` delegates to a single backend serving binary.

Add an `EnginePool` only around `LoadedInferenceEngine`; do not create a second
runtime truth surface. The pool should own:

- model id / alias / type
- loaded engine handle
- TTL and manual pin
- last-used time
- memory budget estimate
- load/unload lifecycle events

The first version should be control-plane only. Do not share KV pools across
models, and do not mutate backend internals to make the pool work.

### Acceptance

- `arle serve` can expose `/v1/models` with multiple configured models but load
  at most the memory budget allows.
- LRU unload never drops an active request.
- Embedding/reranker model types are explicit stubs or real implementations;
  no silent route to a text-generation engine.

## Global Verification Rules

- Docs-only edits to this file are bench-exempt.
- Any implementation under `infer/src/`, `crates/mlx-sys/src/`, or
  `crates/cuda-kernels/csrc/` must add a dated wins/errors bench entry per
  `docs/bench-and-trace-spec.md`.
- Required checks by implementation lane:
  - Metal: `cargo test --release --no-default-features --features metal,no-cuda`
  - Mac CUDA type surface: `cargo check -p infer --no-default-features --features cuda,no-cuda`
  - Workspace CPU sanity when relevant: `cargo test --release`
- Non-trivial runtime diffs require `codex review --uncommitted` before commit.

## Open Questions Before Code

1. P0-1A snapshot disk format: raw opaque `DiskStore` payload is the default;
   safetensors should be a deliberate format change only if MLX load/save APIs
   make it clearly cheaper.
2. P0-1B convergence: decide whether `MetalLivePrefixRuntime` is deleted after
   block-level Metal KV lands or remains as a Qwen3.5 fast path. The no-half
   states rule favors deletion or full absorption.
3. P0-2 fallback: decide whether MLX 0.31.1 remains as a build feature if
   0.32+ regresses M3/M4 or older SDK support.
