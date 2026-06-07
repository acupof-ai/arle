# DSv4 batched decode Phase 6a: grouped MoE/shared → +31–35% decode throughput at c=4–8

## Context

The Step-A layer-major batched DSv4 decode driver (`forward_decode_batch_stream_impl`,
default-off behind `INFER_DSV4_BATCHED_DECODE`) kept the MoE and shared-expert halves
as **per-row loops** for correctness isolation: each row copied to a `[hidden,1]`
scratch, run through the single-token MoE/shared, copied back, with a `ctx.sync()`
**per row** (43 layers × N rows host syncs per decode step). That serialization
caps c-scaling.

Phase 6a replaces both per-row loops with **grouped-over-N** calls — the existing
prefill path (`dsv4_moe_forward`/`dsv4_shared_expert_forward` with
`decode_scratch=None`): one router GEMM + one DeepGEMM grouped expert GEMM over
N×topk routes, and one batched SwiGLU pair for the shared expert. The per-row MoE
scratch and 2N host syncs/layer are deleted. Attention stays per-row for now
(Phase 5/6b).

## What worked

Throughput A/B on the 8×H20 pod, same needle prompt (37 tok, passcode "73914"),
12 decode steps, grouped (`BATCHED=1`) vs per-row executor (`BATCHED=0`), back to
back in one process (`scripts/dsv4_multigpu_parity.sh`, decode-loop wall time only):

| c | grouped tok/s | per-row tok/s | Δ tok/s | grouped ms/step | per-row ms/step | Δ ms/step |
|---|---|---|---|---|---|---|
| 2 | 35.68 | 32.83 | **+8.7%**  | 56.05  | 60.92  | −8.0%  |
| 4 | 43.50 | 33.08 | **+31.5%** | 91.96  | 120.93 | −24.0% |
| 8 | 49.71 | 36.73 | **+35.3%** | 160.93 | 217.80 | −26.1% |

The win grows with c (grouped GEMM amortizes launch + improves GPU occupancy over
N×single-token calls). ms/step at c=8 dropped 217.8 → 160.9.

**Correctness (needle gate, `INFER_DSV4_BATCH_MATCH_PREFIX=3`, gate_exit=0):** every
row at c=2/4/8 retrieves the answer `[223,30793,929]` (" 73914") bit-identically.
c=4 and c=8 are *fully* byte-identical to the c=1 reference (all 12 tokens);
c=2 diverges only at idx4 (after the answer), the legitimate N=2 all-reduce tiling
differing from N=1 (see the numerics-derivation wins entry). `ref_self_parity=true`
throughout (c=1 deterministic). The divergent c=2 tail stays coherent — it re-emits
the passcode.

## Rule

- Grouped-over-N is a drop-in for DSv4 decode MoE/shared: the prefill path already
  handles seq_len=N (`decode_scratch=None`); the pooled/`decode_scratch` path is the
  N==1-only one. Don't hand-roll a batched MoE — reuse the prefill grouped path.
- Per-row loops + per-row `ctx.sync()` are c-scaling killers. Removing 2N host
  syncs/layer is most of this win; the grouped GEMM occupancy gain is the rest.
- Gate batched perf on **decode-loop wall time** (exclude per-slot prefills) and on
  **needle retrieval**, not byte-parity (byte-parity is shape-dependent — c=4/8 happen
  to match c=1, c=2 doesn't; all are correct).
- Next levers (bigger): attention is still per-row with a per-row sync (43×N
  syncs/step) — Phase 6b removes the sync (verify no FlashMLA private-stream race via
  the needle gate), Phase 5 batches FlashMLA b=N. Absolute tok/s (~50 at c=8) is
  attention-bound until then.
