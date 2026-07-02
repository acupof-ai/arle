# Linear-attention chunked backward — the next OPD writeback wall

**Status:** planned (decomposed, not started). **Owner:** next kernel session.
**Commissioned by:** the 2026-07-02 OPD optimization series
([full10 e2e](../experience/wins/2026-07-02-agent-opd-full10-e2e.md)).

## Why

After the day's fixes the toy agent-OPD round is 27.8s; backward is 10.0s and
**LinearAttention backward is 4.2s of it** (45 calls × 93ms, the top op in the
inner-checkpoint profile). The kernel
(`linear_attention_chunked_scan_backward_f32`,
`crates/autograd/src/backend_cuda/kernels/linear_attention.cu:533`) runs ONE
block per (batch × value_head) = 48 blocks on a 78-SM H20 — structurally
under-occupied — and scans tokens sequentially twice per chunk (forward
recompute + reverse grad pass) with 256 threads striding 128-wide loops.

## Why not the cheap splits

- **Split value_dim across blocks:** forward recompute is value-channel
  separable, but the reverse pass's `dq_vec`/`dk_vec` are per-token reductions
  ACROSS value channels (kernel lines ~700+); per-token cross-block reduction
  = grid sync per token × 1010 — worse than the scan.
- **Split key_dim:** `delta[v]` needs the full `k·S[:,v]` dot per token — same
  coupling, other axis.

## The plan — chunked GEMM backward (fla-style)

Same trick the forward already uses (`gdr_prefill_chunk_*` TileLang stages):
express the intra-chunk work as dense chunk-level matmuls, keep only the
64-token chunk boundary state/grad-state carries sequential (16 carries at
seq≈1010 instead of 1010 token steps).

Reference algorithm: flash-linear-attention's chunked gated-delta-rule
backward (`fla/ops/gated_delta_rule` — chunkwise dq/dk/dv/dbeta/dg via
per-chunk GEMMs against saved chunk states + a reverse chunk-state-grad
recurrence). Adopt the ALGORITHM structure; emit as either
(a) native CUDA stages next to the existing kernel, or
(b) TileLang stages in `tools/tilelang/gated_delta_rule.py` beside the seven
forward stages (preferred — same codegen/bundle pipeline, and the kernel
bundle now ships via the release artifact so no consumer cost).

Concrete steps:
1. Derive the chunkwise backward equations for THIS kernel's exact forward
   (k-normalization + exp-gate decay + beta delta rule + RMS-gated output —
   the forward recompute at kernel lines 560-690 is the spec; the fla
   reference must be adapted to the k-norm and the `preact`/conv chain which
   stay in the existing epilogue kernels).
2. New saved-context requirement check: the device forward already saves
   `chunk_state` (per-chunk incoming states) — sufficient for chunk-local
   recompute inside GEMM stages; no new forward cost.
3. Stage kernels: per chunk (parallel over 16 chunks × 48 heads = 768 blocks):
   recompute {q,k,v,g_cumsum,beta} tiles → chunk-local grads via GEMMs;
   sequential pass only for the [num_heads × 128×128] grad-state carry.
4. Wire behind the existing `linear_attention_backward_device` entry with the
   old scan kernel as the fallback (envelope guard), A/B per
   [bench spec](../bench-and-trace-spec.md): per-op backward profile
   (`ARLE_OPD_BACKWARD_PROFILE=1`) LinearAttention seconds + toy-round loss
   band + `chunked_sdpa_backward_matches_unchunked`-style parity test vs the
   scan kernel on small shapes.

## Expected gain / kill threshold

93ms → target ≤25ms per call (occupancy 48→768 blocks + GEMM math):
LA backward 4.2s → ~1.1s, backward 10.0s → ~7s, round 27.8s → ~25s (−10%).
Kill if the A/B shows <30% op-level gain after step 3 — the carry recurrence
may dominate at 128×128 state (measure, don't assume).

## Outcome (2026-07-02, ec23705e)

Shipped: 3-stage transfer-operator design (chunk_transfer/carry/grad, mono
kept behind ARLE_LA_BACKWARD_MONO=1). Parity: qwen35+qwen36-27b shapes, all
grads max_abs <= 1.2e-4. Same-binary env-flip A/B at seq~1010:
mono 4.135s -> chunked 3.186s per round (92 -> 71ms/call) = 29.7% op gain —
AT the pre-declared kill threshold. Verdict per the kill rule: keep the
kernel, stop iterating this design. Decoded floor: the per-token
__syncthreads chain (mono ~= 16 chunks x 2 scans x 64 tokens x ~45us matches
92ms exactly); stage-parallelism cannot beat it while the intra-chunk math
stays a per-token loop. The real next step is the full fla-style chunkwise
GEMM formulation (no token loop) — a separate derivation, out of scope here.
