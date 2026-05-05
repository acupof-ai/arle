# `infer::model` — Agent Guide

Model implementations (`qwen3`, `qwen35`) + the `ModelForward` and
`DecodeContextOps` traits. Load before editing any model or adding a new one.

## Refactor posture

- Keep model code simple and uniform. Prefer deletion-style refactors: remove
  obsolete model-specific detours, collapse duplicate shape/config logic, and
  keep one canonical contract per runtime behavior.

## Trait contracts

**`ModelForward`** (`model.rs`) — the deep interface the scheduler calls:

- `type State: GenerationState + Send` — **per-request mutable state**. Weights
  stay in `&self` so one model instance serves N slots.
- `type DecodeContext: DecodeContextOps + Send` — pre-allocated batched-decode
  buffers owned by the scheduler; **one per scheduler**, not per-request.
- `forward_prefill(tokens, state)` — multi-token path, populates KV.
- `forward_decode(token, state)` — single-token path.
- `forward_decode_batch(...)` — the batched decode path; **do not** fall back
  to sequential decode in production models.
- `select_token_with_logprob` — greedy-capable backends should override so the
  scheduler can surface logprobs without a second pass.
- `sample_batch_greedy` — return `Some` for the fast path, `None` to force the
  scheduler to fall back to `select_tokens_batch`.
- `forward_prefill_with_pool` — optional dual-write prefill (not yet the
  production path; the scheduler still uses `migrate_kv_range_to_paged`).

**`GenerationState`** (`model.rs`) — state that must be resettable, truncatable,
and snapshottable:

- `truncate_to(len)`, `reset()` — slot reuse.
- `set_max_seq_len`, `set_kv_dtype` — **must be called before
  the KV cache is first initialized**; after that they are silent no-ops.
- `migrate_kv_to_paged` / `migrate_kv_range_to_paged` — contiguous → paged pool
  migration; called after prefill, before the first decode step.
- `save_prefix_snapshot` / `restore_prefix_snapshot` + `supports_partial_prefix`
  — the scheduler downgrades partial hits to MISS when `supports_partial_prefix`
  is `false` (hybrid models), and uses snapshots only on exact-full hits.

**`DecodeContextOps`** (`model.rs`) — what the scheduler can do with a model's
decode buffers independent of architecture:
`upload_token_ids`, `update_metadata`, `plan_attention`, `set_batch_size`,
`invalidate_graph_cache`, `logprobs_host`. Returns `true` from `update_metadata`
when `kv_indices` was reallocated so the scheduler knows to drop the captured
CUDA graph.

## Module layout

Flat layout with `#[path = "model/<name>.rs"]`. Each model has a **directory**
of peers next to its root file:

```
model/qwen3.rs   + model/qwen3/{config, weights, forward, prefill, decode, batch_decode, decode_buffers, lora}.rs
model/qwen35.rs  + model/qwen35/{config, weights, forward, prefill, decode_buffers, batch_decode, recurrent_state, prefill_buffers, single_token_buffers, gguf_host}.rs
model/common.rs  — cross-model CUDA graph glue
model/cuda_graph.rs — CudaGraphState (capture-on-first-call, replay-thereafter) for decode-path graph reuse
model/generation_state.rs — GenerationStateBase shared scaffolding
model/kv_cache.rs — KVCacheDtype, KVFormat (re-exports from cuda_kernels)
model/layer_communicator.rs — F0.8 inert TP/DP/CP `LayerCollective` surface; defines the method shape F1+ forward paths will call, with exact single-rank pass-through. Not yet wired into Qwen forward call sites.
```

Hybrid models (Qwen3.5) add `recurrent_state`, `prefill_buffers`,
`single_token_buffers` because linear-attention layers need separate
O(1) recurrent state. `qwen35/gguf_host.rs` carries the GGUF tensor
layout helpers used by the host-side load path. `qwen3/lora.rs` is a
forward-only HuggingFace PEFT loader (M2b Phase 1: types + loader; the
hot-path wiring through `ops/linear.rs` is M2b Phase 2 and not yet in).

## Invariants

1. **Weights are `&self`.** If you need `&mut`, you're re-architecting —
   stop and talk to the user.
2. **`num_kv_layers()` on hybrid models counts full-attention layers only.**
   Linear attention layers have O(1) recurrent state, not KV pages.
3. **`create_decode_context` is lazy.** The scheduler calls it once, after
   the first slot has its state, so the model knows the pool geometry.
4. **`forward_prefill` must populate KV in the state's contiguous cache.**
   The scheduler migrates it into the paged pool after prefill completes.
   Direct paged writes go through `forward_prefill_with_pool` only if you've
   also wired the scheduler to use it (currently: no). When the scheduler
   does use paged prefill, that path must handle every chunk size, including
   `len == 1`, and must write KV into the paged pool rather than silently
   falling back to contiguous decode.
5. **`DecodeContext` lives on the scheduler for the lifetime of the run.**
   Don't allocate GPU buffers inside `forward_decode_batch` — use the context.
6. **`layer_communicator.rs` is inert until F2.** TP=1 collectives are
   strictly no-ops; TP>1 currently fails fast. Do not stub out the method
   bodies into something other than the documented pass-through, and do
   not call them from production forward paths until the F2 NCCL
   forward-collective wiring lands. The plan lives at
   [`docs/plans/2026-04-28-single-node-multi-gpu.md`](../../../docs/plans/2026-04-28-single-node-multi-gpu.md).
7. **`cuda_graph.rs` is capture-on-first-call.** A second decode tick
   replays the graph; metadata changes that invalidate the captured graph
   must clear `CudaGraphState` (the `update_metadata` true return on
   `DecodeContext` is the existing signal). Do not introduce a
   per-step capture loop — that defeats the cost amortization.

## Active priorities touching this module

- **P0 long-context.** Varlen FP8 KV path + mixed-batch attention go
  through `qwen3/forward.rs` + `qwen35/forward.rs`. Phase 2 spec-decode
  hooks (sparse-KV draft views) consume `SparseKvDraftView` /
  `SpecVerifyRequest` exposed via `model.rs`.
- **P0' multi-GPU F0–F4.** TP-aware weight sharding lands in
  `qwen3/weights.rs` + `qwen35/weights.rs` via `TpLoadContext`;
  forward wiring stages through `layer_communicator.rs`. Production
  TP>1 forward gates on F2 collective integration.
- **P0'' DeepSeek V4 prep.** DS3 MLA cache + kernels are designed
  (`docs/plans/2026-05-01-mla-kernel-design.md`); no model file in this
  tree implements them yet — they will land alongside DS1 registry +
  DS2 block-FP8 metadata once F2 collectives are real.

## Adding a new model

1. Create `model/<name>.rs` + `model/<name>/{config, weights, forward, prefill, decode, batch_decode, decode_buffers}.rs`.
2. Implement `ModelForward::State` + `DecodeContext` with the `_batched_into`
   ops from `infer::ops` — see [`infer/src/ops/AGENTS.md`](../ops/AGENTS.md).
3. Register in `infer/src/model_registry.rs` (model ID → builder).
4. Add a `BackendInferenceEngine` arm in `server_engine.rs::LoadedInferenceEngine::load`.
5. Add a greedy baseline under `infer/test_data/` and wire an E2E test.

## Pointers

- `crates/qwen3-spec/` and `crates/qwen35-spec/` — canonical config + tensor-name
  contract shared between train and infer.
- `docs/projects/agent-first-architecture.md` — current model/runtime priorities
  including the hybrid Qwen3.5 attention surface.
- `feedback_mlx_rope_layout.md` / `feedback_mlx_rope_axis.md` (auto-memory) —
  MLX `fast::rope` layout gotchas for the Metal side.
