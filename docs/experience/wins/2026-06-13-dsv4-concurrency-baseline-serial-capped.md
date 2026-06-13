# DSv4 concurrency baseline (d2 spec) — aggregate is SERIAL-CAPPED at 53 tok/s; replicated-attn deleted

## Context

Goal: completely build DP-attention for c>1 throughput, test concurrency, and
deletion-refactor the killed replicated-attn lane. Step 1 is the honest
concurrency baseline — does today's lockstep serve scale aggregate throughput
with concurrent clients at all? Without that number, "DP-attn for throughput"
is a hypothesis, not a licensed lever.

This entry is also the mandatory bench entry for the **replicated-attn
deletion refactor** (below): that refactor removes opt-in (default-OFF) code
only, so the DSv4 default decode path is byte-identical — the baseline here
*is* the post-refactor number.

## Params / Env

- 8×H20, TP=8/EP=8, DSv4-Flash, CUDA 12.9, sm_90a.
- Worktree binary (`/data01/build/arle-dsv4`), `--spec-type mtp
  --mtp-draft-tokens 2` (d2 chain-fold, default-on), `MOE_BACKEND=allreduce`,
  `EXPERT_BACKEND=deepgemm`, native DeepGEMM, incremental KV, auto NUMA pin.
- Client: `conc_bench.py` — warm once, then C concurrent threads, each a
  128-token `temperature=0` completion to `/v1/completions`. Serial GPU
  (no other client; a wedged 2h client had contaminated earlier runs —
  killed, serve restarted clean, gated on a real non-empty completion).

## Results — aggregate is FLAT, wall-clock scales linearly with C

| c | agg tok/s | per-req tok/s | wall (s) | wall vs c×(c=1) |
|---|---|---|---|---|
| 1 | 53.3 | 53.3 | 2.40 | — |
| 4 | 53.2 | 13.3 | 9.62 | 4×2.40 = 9.62 ✓ |
| 8 | 53.2 | 6.6 | 19.25 | 8×2.40 = 19.25 ✓ |
| 16 | 53.2 | 3.3 | 38.49 | 16×2.40 = 38.49 ✓ |

**Aggregate throughput is dead flat at ~53 tok/s regardless of C; wall-clock
is exactly C × single-stream.** The lockstep single-scheduler serve processes
each concurrent decode request as its own B=1 forward, back-to-back — **zero
batching benefit.** per-req = agg / C is the signature of pure serialization.

This is the precise, measured motivation for the throughput work: the ceiling
is not the GPU, it is the **single-scheduler lockstep** admission model that
never co-batches concurrent decodes (batched-decode lowering is the open
keystone #60; DP-attn #89 is the architecture that breaks the single
scheduler).

## Deletion refactor — replicated-attn removed (default byte-identical)

replicated-attn (opt-in `ARLE_DSV4_REPLICATED_ATTN=1`) was correct but −48%
at B=1 ([kill entry](../errors/2026-06-13-dsv4-replicated-attn-bandwidth-kill.md)).
It is the *opposite* of DP-attn (every rank redundantly computes ALL requests
vs. each rank computing DIFFERENT requests), and its scalar grouped
o-projection is the wrong impl for any salvage (DP-attn needs a vectorized
o-proj). Keeping it unwired would be a multi-week half-state. Deleted whole:

- `attention.rs`: `dsv4_replicated_attn_enabled`, the `replicated` gate +
  param threaded through `try_flashmla_decode_attention` / `mla_oproj`, the
  grouped-o-projection branch, all `if replicated`/`!replicated` arms collapse
  to the existing sharded path.
- `dsv4.rs`: `wq_b_full`/`wo_a_groups`/`wo_b_full` fields,
  `replicated_attn_active()`, and the 4 AR-skip gates → unconditional
  all-reduce (the default path always took these).
- `loader.rs`: `load_dsv4_wo_a_groups` + the full-width conditional load.

Every kept branch is the pre-existing non-replicated default; replicated was
default-off, so the default decode path is byte-for-byte unchanged. The
grouped-o-projection design is preserved in the kill entry for re-derivation
(vectorized) when DP-attn is built. Mac CUDA-Rust typecheck clean; pod rebuild
+ needle smoke confirm the default path (see verify section in commit).

## Rule

- **Baseline concurrency before claiming a throughput lever.** The flat
  aggregate (53 @ c=1..16) proves the bottleneck is the admission model, not
  the GPU — DP-attn / batched-decode is licensed by *this* curve, not by a
  source-survey of "lockstep looks serial."
- **per-req = agg/C is the serialization fingerprint.** When wall-clock is
  exactly C × single-stream, the server is not batching; no kernel-level work
  changes that ceiling.
- **Delete the whole killed lane, not just its gate.** A killed opt-in path
  whose impl is the wrong shape for the successor (scalar vs vectorized
  o-proj) is dead code in the final ideal state — remove it, preserve the
  design in the kill entry, re-derive correctly when the successor lands.
