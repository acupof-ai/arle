# DSv4 B=1 decode: foundation-lever investigation — measured facts + open question

> Status: investigation (no code landed beyond the c1+c2 cleanup) — 2026-07-07

Scope: after the alloc-removal sweep washed (`errors/2026-07-07-dsv4-alloc-removal-sweep-wall-wash.md`),
investigated whether the "foundation" levers (per-step ctx.sync removal, single-proc
TP, decode CUDA graph) can move the DSv4 B=1/MTP decode wall. This records what is
MEASURED vs INFERRED so the direction isn't re-explored blind.

## Measured facts (this session)

1. **Per-rank steady GPU-busy ≈ 69.6%, idle-gap ≈ 22.5%** (after excluding 353
   `>1ms` warmup/boundary outliers = 1011ms). Source: `/host/kern141_decode2.sqlite`
   (TP=4/EP=4 DSv4-Flash-FP8 MTP-on, 07-03), rank0 globalPid, steady middle-third
   window, busy = union of kernel intervals across its streams (19,21).
   Caveat: capture mixes 4 rank pids; single-pid union may miss a comm/copy stream
   → 69.6% is a ±few-point estimate. Direction (~70%, not 100%) is solid.

2. **Idle-gap size split**: 65.1% of clean gap is `<5µs`, 32.8% is `5-20µs`, 2.1%
   is `>20µs` (611 gaps). Intra-stream (single compute stream, no comm/sync) gap
   median **1344ns**, p10 1120ns, floor 480ns.

3. **CUDA API wall shares** (same trace, `cuda_api_sum`): `cudaLaunchKernel` 39.8%,
   `cuStreamSynchronize` 26.6%, `cuMemAllocAsync`+Free 7.7%, `cuMemsetD8Async` 9.1%,
   zero `cuGraphLaunch`. ~1271 kernel launches/step (10.75M insts / ~8458 steps).

4. **A Qwen3.5/3.6 whole-decode-step CUDA graph exists and was measured**
   (`wins/2026-06-11-qwen35-whole-decode-step-cuda-graph.md`): **+5.5%** tok/s
   (43.11 vs 40.86, σ≈0.07), TP=1, ~1074 launches/step. Predicted +30-75% from the
   launch-count formula; actual +5.5%. Quote: "with host issue removed, ~23ms/token
   is GPU-timeline." This is an already-run graph — the launch-removal ceiling on a
   sibling model is single-digit %.

5. **DSv4 decode graph has never run on the real path**: `attention.rs:4232` bails
   `ARLE_DSV4_DECODE_GRAPH=1` on any CSA layer, and DSv4-Flash is all-CSA. So the
   prior "graph wash / GPU-bound" DSv4 kills (2026-06-08/06-10) were measured on a
   path where the graph never executed.

6. **CSA graph READ is already graph-safe** (`csa_select_official_batched` n=1 +
   persistent slot_id/key_count buffers, gated `ARLE_DSV4_DECODE_GRAPH_CSA=1`,
   `attention.rs:1697`) but runs EAGER. The remaining blocker is the CSA cache
   **WRITE** (`dsv4_dsa_cache_write_batched`, `attention.rs:7906-7955`): per-step
   host-uploaded offset/count/ptr arrays (`Dsv4DsaCacheWriteBatchPtrs`) with a
   host-tracked growing `packed_rows`/`dst_row`/`newly_packed` — not graph-capturable
   until a device-driven index-key packer lands.

7. **Comm is skew-bound, wall-neutral to faster collectives** (prior, re-confirmed):
   one-shot allreduce AR med 51.9→25.8µs but B=1 wall +0.6% (noise)
   (`errors/2026-06-10-dsv4-oneshot-comm-wall-neutral-skew-bound.md`); NCCL AR
   ~17-22µs flat 14KB-459KB (`wins/2026-06-15-dsv4-allreduce-latency-floor-measured.md`).
   Collectives ≈8.6% GPU-busy.

8. **MTP is the only measured B=1 wall multiplier** (+71% historically); linear
   depth-K killed (accept pinned 1/4, `errors/2026-06-11-dsv4-mtp-depth-k-draft-quality-wall.md`);
   deeper needs a 2-head draft (training). Default depth=2 topk=1
   (`dsv4.rs:460/462`), measured ~2.34 tok/step (`2026-07-02-dsv4-6ms-token-plan.md:41`).

## Inferred (NOT measured on DSv4 — flagged)

- The ~1.3µs intra-stream gaps are CPU launch-dispatch (host issue), largely
  overlap-absorbed by the GPU — inferred from fact 4 (Qwen3.6 graph recovered
  5.5%, not the predicted 30-75%), not from a direct DSv4 overlap measurement.
- A DSv4 all-CSA decode graph would recover single-digit % (Qwen3.6 is the
  same-family anchor; DSv4 TP=4 adds non-capturable skew-bound comm). DSv4 graph
  wall was never measured — this is a high-confidence prediction, not a result.

## Open question (unresolved, would need a real experiment)
The only clean way to settle DSv4 graph value: land a device-driven CSA index-key
packer (fact 6), remove the bail (fact 5), run the all-CSA decode graph once, A/B
the wall. Ceiling is bounded by fact 4 (~5.5% sibling) minus DSv4 comm skew.
Cost/benefit: a correctness-critical multi-file device state-machine change for a
single-digit-% ceiling — not started.

## Direction
DSv4 B=1 decode kernel/alloc/graph levers are at or near saturation: alloc-sweep
0% (measured), graph ceiling single-digit % (sibling-measured, DSv4 unmeasured),
comm skew-bound (measured wall-neutral). The remaining measured-effective lever
class is amortization (MTP, capped by the 1-head arch) and batching (DP-attn,
c>1) — architectural, outside the kernel-optimization scope.
