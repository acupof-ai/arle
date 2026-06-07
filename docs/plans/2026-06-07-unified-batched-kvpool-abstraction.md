# Unified Batched Decode Over KvPool

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

