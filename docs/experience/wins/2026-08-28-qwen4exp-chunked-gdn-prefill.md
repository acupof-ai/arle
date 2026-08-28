# qwen4_exp chunked (WY-form) GDN prefill: the serial linattn scan collapses from n_tokens to n_chunks

## Context / Goal

After the expert-grouped MoE round, the GPU-timestamp drain decomposition
showed the prefill's largest single slice was `pf.linattn`: the gated-delta
recurrence ran as ONE `qwen35_gated_delta_net` dispatch per linear layer per
chunk — 48 near-serial workgroups walking all 256 tokens — 2.8 s of a 7.2 s
GPU drain at chunk 256. Target: port llama.cpp's `build_delta_net_chunking`
(delta-net-base.cpp, ca3d5a3e1, non-KDA arm: per-value-head scalar decay,
CS=64) as an OPT-IN lane, because the default prefill's crown jewel — the
prefill=decode 0.000e0 bit-exact gate — cannot survive any reassociation.

## What Worked

**Two kernels, math ported rather than the ggml graph**
(`qwen4_gdn_chunk_intra.comp` + `qwen4_gdn_chunk_state.comp`):

- INTRA, parallel over (64-token chunk, value head) workgroups: per-token
  scalars (same formulas as the serial kernel: l2norm eps 1e-6, `ssm_a *
  softplus(a + dt_bias)`, sigmoid beta), cumsum of log-decays, decay-masked
  `k_beta K^T` and `q K^T`, `T = (I + tril)^{-1}` by in-place forward
  substitution in shared memory (CS steps — the only serial piece, and it is
  64, not seq_len), then `u = T V_beta`, `k_cumdecay`, `q exp(G)`,
  `k exp(G_last - G)` into a scratch slot.
- STATE, serial only in `n_chunks`, inside the workgroup: `v_new = u - kcd S`,
  `out = qg S + kq v_new`, `S <- S*exp(G_last) + kg^T v_new`, on the SAME
  resident f32 state buffer decode advances. Grid = (head, 32-column state
  stripe): the state splits by value column, so 192 workgroups and no
  cross-workgroup sync.
- All f32 everywhere (scratch, state, accumulation) — the GEMM lane died as a
  default precisely because sub-f32 staging saturates into expert flips; this
  lane's only deviation from serial is reassociation. Decay factors are
  exp(non-positive) by construction, so underflow-to-0 is the correct limit
  and the HF clamp(max=50) is unreachable.
- 2 dispatches per (layer, chunk) instead of 1 — prefill records ~10 per
  linear layer already; dispatch count was never the linattn problem
  (near-serial occupancy was).

**Proof stack, each layer catching what the previous cannot:**

- `device_gdn_chunked.rs`: device vs a host chunked oracle (association-exact:
  worst 4.4e-5 out / 1.5e-4 state incl. FMA contraction), host chunked vs the
  serial per-token rule (pure WY reassociation: worst 8.9e-5), partial tails
  (seq 7/100), a chunk-boundary-exact 64, the real 48-head shape, and a
  two-call state continuation. Mutations kill it loudly: flipped decay-mask
  sign reads NON-FINITE; dropped state decay reads 7.1e0 out / 2.8e2 state
  against 5e-4 bounds.
- `chunked_gdn_drift_stays_in_the_reassociation_envelope`
  (tests/qwen4_prefill.rs): the 4-layer SubsetF32 fixture where every GEMV is
  decode's own — layer-0 S agrees with decode to ~4e-7 ABSOLUTE (the
  reassociation itself); downstream MoE amplification takes the 1e-3-floored
  metric to 3.7e-1 worst (absolute ~5e-4), 5x under the GEMM lane's honest
  2.0e0; argmax equal at widths 96/24/7. Env hygiene is panic-safe: the gate
  is removed BEFORE the asserts, so a red envelope can never leak the lane
  into the bit-exact gate running later in the same process (the first cut
  did exactly that — the gate "failed" at 1.1e-1 until the leak was fixed).
- The bit-exact gate itself: **max rel 0.000e0 at both widths with the lane
  off** — the default path is byte-for-byte untouched (the lane is a
  record-time branch, exactly the `ARLE_QWEN4_PREFILL_GEMM` pattern).

**The full-scale receipt (`ARLE_QWEN4_PREFILL_CHUNKED_AB=1`)** — one ~70 GiB
load, chunk 256 over the same 512-token prompt, lane off then on back to back;
`moe_ids` (new, reads the ids fence the prefill already pays) captures every
expert selection both ways:

- tok/s: **50.3 → 55.8** (+11%, Performance mode, same sitting) — real but
  far below the hoped 25%+, because the same profile re-ranked the wall:
  `pf.moe.ids_fence` holds **7.46 s of the 9.27 s chunk-256 prefill (80%)**.
  The linattn GPU work the lane removes was only one tributary draining into
  that fence; the fence STRUCTURE (per-(layer,chunk) flush + host regroup) is
  the next prefill lever, not any single kernel under it.
- expert selections over 512 tokens × 36 MoE layers: **5,627 set flips /
  12,225 order flips over 24,576 (token,layer) rows (23%)**; final-logits
  max rel **1.74e2**; argmax **FLIPPED** (OFF 198 vs ON 271). The 4-layer
  envelope (argmax equal, drift 5× under the GEMM lane's) did not survive 48
  layers of 512-expert routers — the same amplification that keeps the GEMM
  lane opt-in. **Verdict: the lane stays `ARLE_QWEN4_PREFILL_CHUNKED_GDN=1`
  opt-in, and the A/B prints this receipt instead of gating on it.**
- chunk-64 side receipt: 0.6 tok/s with the ids fence at 654 s — the fence
  cost scales with chunk COUNT, which is the 80% claim seen from the other
  side.

## Rule

- A serial scan whose grid is heads-only leaves the GPU idle in exact
  proportion to seq_len; the WY/chunk form buys parallelism with ~3x the
  FLOPs and wins whenever the serial dimension, not bandwidth, is the wall.
- A reassociation lane can never be default where the gate is bit-exactness —
  ship it env-gated with a calibrated envelope AND the downstream discrete
  receipt (expert flips), and let the receipt, not the drift scalar, decide
  promotion.
- Test-process env vars are state: remove the gate BEFORE asserting, or a
  failing envelope test silently corrupts every test after it.
