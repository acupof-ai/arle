# Unified Batched Decode Over KvPool

> **This is the authoritative forward plan for the DSv4 perf campaign's throughput
> axis.** Provenance: the campaign root-cause was a wall-clock @4096 trace
> (`2026-06-06-dsv4-pd-systematic-analysis.md`
> §3.5 found the R6 executor is single-row-only → c>1 doesn't scale, 1.63× @c=8); the
> single-request kernel bottlenecks were fixed by adopting official kernels (decode
> [`../experience/wins/2026-06-07-dsv4-official-dsa-default-on.md`](../experience/wins/2026-06-07-dsv4-official-dsa-default-on.md),
> prefill
> [`../experience/wins/2026-06-07-dsv4-prefill-official-kernels-default-on.md`](../experience/wins/2026-06-07-dsv4-prefill-official-kernels-default-on.md))
> per the retro
> `../experience/errors/2026-06-06-handrolled-kernels-vs-adopt-official-retro.md`;
> targets are anchored on the
> H20 reference baseline. The Phase 3
> shared-KV-pool foundation landed (c=1 byte-identical, c=2/c=4 row-parity) — see the
> wins entry `2026-06-07-dsv4-shared-kv-pool-foundation.md` once committed.
> The session code cleanup that should ride alongside this work:
> `2026-06-07-dsv4-code-cleanup-audit.md`.

## Context

DSv4 batched decode, paged KV, and continuous batching must be engine-generic
abstractions. They are not DSv4-only infrastructure. DSv4 is the first adapter
because it has the hardest KV layout: sliding-window BF16 state, compressed
CSA/HCA state, DSA index keys, and FlashMLA FP8 KV. The engine still owns the
generic request, slot, page, and scheduling decisions.

The existing seam is the starting point:

- `crates/infer-seam/src/kv.rs`: `KvPool = KvQuery + KvAllocator +
  KvPrefixStore`; all methods are host slot/page/token-id based.
- `crates/infer-seam/src/lib.rs`: `BackendExecutor::submit(plan, kv)` already
  passes `ForwardPlan + &mut dyn KvPool` to backends.
- `crates/infer-plan/src/lib.rs`: `ForwardPlan` carries backend-independent
  decode/prefill rows.
- `docs/plans/backend-unification.md`: continuous batching, paged KV, prefix
  cache, and tiered KV are the shared runtime plane.

The current DSv4 CUDA path bypasses the seam: the top-level CUDA executor passes
`host_kv`, but DSv4 drops it and keeps per-slot attention KV state inside
`Dsv4SlotState`. That is the layering problem to fix.

## Generic Contract

The seam stays host-only. Device tensors, CUDA streams, graphs, collectives, and
model-specific KV physical layouts remain below `BackendExecutor`.

Add a host-only descriptor produced from `ForwardPlan + dyn KvPool`:

```rust
pub struct KvBatchDescriptor {
    pub mode: ForwardMode,
    pub rows: Vec<KvBatchRow>,
    pub flat_page_ids: Vec<u32>,
}

pub struct KvBatchRow {
    pub slot: usize,
    pub token: u32,
    pub seq_len: usize,
    pub append_pos: usize,
    pub append_len: usize,
    pub slot_epoch: u64,
    pub page_range: std::ops::Range<usize>,
}
```

Semantics:

- `slot` is the engine slot id.
- `seq_len` is the logical KV length before the current row writes.
- `append_pos` is the logical write start for this row.
- `append_len` is 1 for decode and `tokens.len()` for prefill.
- `slot_epoch` is the current occupant epoch; backends can invalidate stale
  cached device views.
- `page_range` indexes `flat_page_ids`, which contains the slot page ids in
  logical order for the row span needed by the backend.

This descriptor does not prescribe CUDA block tables, Metal packed offsets, or
DSv4 compressed-page metadata. It is the batch-addressable host description that
each backend lowers into its own device buffers.

## Per-Model Adapter Boundary

Backend executors lower the generic descriptor into model-specific device views.
The model adapter owns physical KV layout and attention metadata.

```rust
pub trait ModelKvAdapter {
    type DecodeDeviceBatch;
    type PrefillDeviceBatch;

    fn prepare_decode_batch(
        &mut self,
        kv: &dyn KvPool,
        desc: &KvBatchDescriptor,
    ) -> anyhow::Result<Self::DecodeDeviceBatch>;

    fn prepare_prefill_batch(
        &mut self,
        kv: &dyn KvPool,
        desc: &KvBatchDescriptor,
    ) -> anyhow::Result<Self::PrefillDeviceBatch>;

    fn truncate_slot(&mut self, slot: usize, new_len: usize) -> anyhow::Result<()>;
}
```

Generic engine responsibilities:

- request admission and scheduling;
- page allocation, truncation, prefix retention, and seq_len accounting through
  `KvPool`;
- building `ForwardPlan` and `KvBatchDescriptor`;
- applying `StepOutput`.

Per-model adapter responsibilities:

- physical KV pool allocation and device page-table upload;
- translating `slot/page/position` into model layout;
- running the model's batched attention backend;
- model-specific rollback and state mutation.

Examples:

- DSv4 adapter: DSA index-key pool, FlashMLA FP8 KV pool, sliding-window ring,
  compressed c4/c128 pools, official DSA indexer, FlashMLA attention metadata.
- Qwen adapter: GQA paged K/V pool and its attention backend.
- Gemma adapter: same generic contract with SWA/local-window attention layout.

## SGLang Posture

SGLang keeps the same split:

- scheduler batch owns `req_to_token_pool`, token-to-KV allocator, and batch
  rows;
- `ForwardBatch` carries `input_ids`, `req_pool_indices`, `seq_lens`, and
  `out_cache_loc`;
- model-specific token-to-KV pools own physical layout;
- attention backends consume `req_to_token` plus the model-specific KV pool.

ARLE's equivalent should be:

- `infer-core` and `KvPool`: scheduling and request-to-page ownership;
- `KvBatchDescriptor`: host batch view over scheduled rows;
- backend lowering: descriptor to device tensors;
- per-model adapter: physical KV layout and attention metadata.

## DSv4 WIP Folding

The existing uncommitted DSv4 shared-pool work is useful, but it belongs in the
DSv4 adapter:

- `Dsv4AttentionPools` becomes `Dsv4KvLayout` / `Dsv4KvAdapter`.
- `Dsv4LayerAttentionState` becomes a per-slot logical view into shared pools,
  not an owner of the physical FlashMLA/DSA buffers.
- Static per-slot page bands are acceptable for the first DSv4 adapter tranche,
  but they must be expressed through generic `KvPool` page ids and slot epochs,
  not hidden `slot * stride` arithmetic in the model path.
- The DSA and FlashMLA code remains the DSv4 attention adapter, not engine-core.

## Implementation Phases

### Phase 1: Seam Helper

Add `KvBatchDescriptor` and CPU tests with a mock `KvPool`. Verify N-row decode
and prefill descriptor construction: slot ids, seq_lens, append positions,
epochs, and page ranges.

### Phase 2: Executor Lowering

Pass the host descriptor to CUDA backends, including DSv4. DSv4 may still loop
rows internally. The goal is routing DSv4 through the seam, not throughput.

### Phase 3: DSv4 Adapter

Fold the DSv4 shared-pool WIP into `Dsv4KvAdapter`. Move physical KV storage out
of per-slot ownership. Keep c=1 output byte-identical and keep c=2/c=4
byte-parity against single-row decode.

### Phase 4: Batched Official DSA

Use the descriptor's N rows and page ids to drive official DSA indexer metadata
and produce `selected[N, index_topk]`.

### Phase 5: Batched FlashMLA Decode

Widen FlashMLA decode state to `b=N, s_q=1`: indices, topk lengths, split
metadata, LSE/out scratch, and scheduler metadata.

### Phase 6: Batched MoE And Collectives

Route N tokens through grouped MoE and run one attention all-reduce and one MoE
all-reduce over `[N, hidden]`.

### Phase 7: Other Model Adapters

Add Qwen and Gemma adapters against the same generic contract. Engine-core and
the scheduler remain unchanged.

## Gates

Every phase is refactor-first:

- CPU seam tests before CUDA work.
- DSv4 c=1 byte-identical output before/after.
- `INFER_DSV4_BATCH_DECODE_VALIDATE=2,4` byte-parity at every CUDA phase.
- No throughput claim until Phases 4-6 are complete.

---

# COMPLETE CODE PLAN (per-model adapters + file map + Metal) — authoritative spec

Every model implements `ModelKvAdapter`. The adapter owns three things: (1) **KvLayout** =
the physical KV pool shapes/dtypes (page-id addressed via the generic `KvPool`); (2)
**AttentionBackend** = the official/OSS kernel that consumes the descriptor's page rows + the
KvLayout; (3) `prepare_{decode,prefill}_batch` = lower the host `KvBatchDescriptor` (N rows)
into the model's device batch. Official-kernel choices come from the 2026-06-06 adoption specs
(`2026-06-06-{qwen35,qwen36,gemma4}-official-adoption-spec.md`). The engine, planner, `KvPool`,
descriptor, batched-MoE, and collectives are SHARED — never per-model.

## Adapter A — DSv4 (first; hardest KV) — `crates/infer-cuda/src/dsv4.rs` + `attention.rs`
- **Dsv4KvLayout** (per layer, shared pools, page-id addressed — fold the WIP `Dsv4AttentionPools` here):
  - SW ring: BF16, window=128, shared `[max_slots×128×kv]` pool, per-slot ring view.
  - Compressed c4/c128: FP8 CSA/HCA compressed KV, shared pool + page table.
  - DSA index-key cache: FP8 paged `[page][64][128 fp8 | 64 fp32 scales]` (official fused_store layout).
  - FlashMLA FP8 KV: 584 B/token (nope-448-FP8 + rope-64-BF16 + 7 scales + 1), shared, page-id addressed.
- **Dsv4AttentionBackend**: official DSA indexer (`fp8_paged_mqa_logits` + `deepseek_v4_topk_transform_512`) + FlashMLA `sparse_fwd` (prefill) / `sparse_decode` (decode). All already default-on single-row; batching = widen to b=N.
- prepare_decode_batch: descriptor N rows → per-row page table + `start_pos[N]` → batched DSA `selected[N,topk]` → FlashMLA `b=N,s_q=1`. prepare_prefill_batch: chunked-prefill rows → `sparse_fwd`.

## Adapter B — Qwen3.5 / Qwen3.6 (hybrid: GatedDeltaNet + gated GQA + dense/MoE) — `qwen35.rs`
- **QwenKvLayout**: full-attn layers → GQA paged K/V pool (`[page][kv_heads][head_dim]`, FP8/BF16);
  GDN (linear-attn) layers → fixed-size RECURRENT state per slot (conv1d state + GDN recurrent state) —
  NOT paged KV; the adapter manages it as per-slot state batched over N.
- **QwenAttentionBackend**: full-attn → FlashInfer/FA3 paged GQA (b=N); GDN → FLA `chunk`/`fused_recurrent`
  (b=N); conv1d → causal_conv1d. (Vendor FLA + FlashInfer + causal_conv1d — per qwen35 spec.)
- Qwen3.6 = Qwen3.5 hybrid + MoE (256 experts top-8) → the SHARED batched DeepGEMM grouped MoE.
- KEY: the generic descriptor drives the full-attn paged part; GDN recurrent state is a per-slot
  batched view the adapter exposes alongside (the descriptor's page rows are empty for GDN layers).

## Adapter C — Gemma4 (GQA + alternating local-SWA/global) — new `gemma4.rs` + `crates/gemma-spec`
- **GemmaKvLayout**: per-layer-type KV — global layers → full paged GQA KV; local-SWA layers →
  sliding-window paged KV (window per config). Per-layer-type RoPE theta; Q/K/V RMSNorm; scaling=1.
- **GemmaAttentionBackend**: FlashInfer/FA3 with `window_size` + per-layer-type theta + soft-cap (b=N).
  (Vendor FlashInfer — per gemma4 spec.)

## SHARED (not per-model) — `crates/infer-cuda/src/moe.rs`, `tp.rs`, the ops layer
- Batched grouped MoE (DeepGEMM) over N tokens + the TP all-reduce over `[N,hidden]` are ONE shared
  path, parameterized by the model's expert config (DSv4 + Qwen3.6 reuse it). Widen the one-token
  decode-scratch guard (moe.rs:874/1104) to N.

## File-level phase map
| Phase | Files | Status |
|---|---|---|
| 1 seam | `infer-seam/src/kv_batch.rs` (KvBatchDescriptor) + CPU MockKvPool tests | DONE `4e8f1989` |
| 2 lowering | `infer-cuda/src/executor.rs` (submit→descriptor) | DONE `d70181ce` |
| 3 DSv4 adapter | `dsv4.rs` (Dsv4KvAdapter, slot→view), `attention.rs` (Dsv4KvLayout from Dsv4AttentionPools, page-id driven), `ModelKvAdapter` trait | IN PROGRESS |
| 4 batched DSA | `attention.rs` official DSA → `selected[N,topk]` (batched index builder) | pending |
| 5 batched FlashMLA | `attention.rs` Dsv4FlashMlaDecodeState→max_batch (b=N,s_q=1) | pending |
| 6 batched MoE+comm | `moe.rs` (N-token grouped, drop 1-token scratch guard), `dsv4.rs` (1 all-reduce/[N,hidden]) → **c-sweep here** | pending |
| 7 other models | `qwen35.rs` (QwenKvAdapter), new `gemma4.rs`+`gemma-spec` (GemmaKvAdapter); engine/planner/KvPool UNCHANGED | pending |

## Metal convergence (backend-unification end state)
`MetalExecutor` (infer-metal) implements the SAME `BackendExecutor::submit(plan, &mut dyn KvPool)`
+ a Metal `ModelKvAdapter` (MLX KV layout). The generic `KvBatchDescriptor` + engine scheduling are
shared CUDA↔Metal. Metal already has continuous batching (mlx-lm `BatchKVCache` pattern); it plugs
into the same seam — converging the two schedulers onto one plane (`backend-unification.md` goal).

## Gates (full) — refactor-first, every phase
CPU seam tests before CUDA; DSv4 c=1 byte-identical; `INFER_DSV4_BATCH_DECODE_VALIDATE=2,4`
byte-parity; per-model adapters gated on that model's needle + same-config-twice floor; throughput
c-sweep (aggregate tok/s + per-request ITL) ONLY after Phase 6; per-phase commit+push. The engine
core (planner, KvPool, descriptor, scheduler) stays UNCHANGED across all model adapters — that's the
unification invariant.

