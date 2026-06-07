# DSv4 Shared KV Pool Foundation

## Context

Phase 3 of DSv4 continuous batching needs one batch-addressable KV/index base per
layer, matching SGLang's TokenToKVPool posture. Before this change, each
`Dsv4LayerAttentionState` owned its own FlashMLA FP8 KV pool and official DSA
index-key cache, so later batched DSA/FlashMLA calls had no shared base pointer
or stable per-slot page row.

## What Worked

Moved the DSv4 FlashMLA FP8 KV pool and official DSA index-key cache to
shared per-layer pools owned by `Dsv4KvAdapter`, and lowered the generic
`KvBatchDescriptor` into a DSv4 batch view before the existing row loop runs.
The first tranche keeps static per-slot bands inside each shared pool, so
single-row math and selected-index semantics are unchanged while later batched
decode can address `[slot, page]` through a shared base.

This is a storage/adapter foundation only. It does not claim a throughput win;
the true b=N DSA/FlashMLA/MoE kernels are deliberately deferred to the reviewed
Phase 4-6 plan.

Verification on the H20 pod:

- Build: `cargo build --release -p infer-cuda --features cuda,nccl,deepep --example dsv4_parity`.
- Gate: `INFER_DSV4_BATCH_DECODE_VALIDATE=2,4 INFER_DSV4_MAX_NEW=8 scripts/dsv4_multigpu_parity.sh`.
- c=1 reference tokens stayed byte-identical to the Phase 1-2 foundation:
  `[11111, 14, 778, 344, 990, 270, 6102, 294]`.
- c=2 byte parity: PASS.
- c=4 byte parity: PASS.

## Rule

For batch-enabling refactors, first move ownership without changing math. Static
slot bands are acceptable as a foundation only if c=1 output and c=2/c=4
row-parity stay byte-identical.
