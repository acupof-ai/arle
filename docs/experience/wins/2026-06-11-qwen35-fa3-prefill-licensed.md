# Qwen3.5/3.6 FA3 prefill attention LICENSED — 3k −36%, 6k −53%, default ON

**Date:** 2026-06-11. **Commits:** `3c7952c2` (vendor) → `8ebb623e` (shim) →
`41984bf2` (wiring) → `852cb9c7` (prepare_scheduler link fix) → default flip.
**Backend:** CUDA H20, Qwen3.6-35B-A3B. **Status: LICENSED, default ON**
(`ARLE_QWEN35_FA3=0` same-binary fallback; build needs
`ARLE_CUDA_ENABLE_FA3=1`, stub builds auto-fall-back via link marker).

## Context

Post-license re-profile pinned `nonpaged_prefill_attention` at **42.1% of
prefill GPU time** (avg 5.0 ms/launch, no tensor cores, ~10× off the
148-TFLOPS roofline). Adoption per "先抄业界最好的": SGLang runs this exact
shape (q16/kv2/HD256, gate + partial-RoPE outside the kernel) on FA3.
Survey + plan: `docs/plans/2026-06-11-qwen35-fa3-hd256-adoption.md`.
Vendored Dao-AILab flash-attention @ `fc8cbad6` (5 hdim256/bf16/sm90 fwd
units + combine + prepare_scheduler, cutlass pin `71275920`), torch-free
shim fills `Flash_fwd_params` mirroring `mha_fwd`'s non-varlen b=1 flow and
calls `run_mha_fwd_<90, bf16, 256, 256>` directly. Head-major slot caches
(`[h_k, max_seq, d]`) feed FA3's TMA descriptors via per-tensor head strides
— zero relayout. Decode (graph lane) untouched.

## Results (same binary `852cb9c7`, two env flips, sequential same-session)

| shape | in-tree | FA3 | Δ |
|---|---|---|---|
| 512-tok prefill (warm ×2) | 0.289 s | 0.274 s | −5.2% |
| 3k prefill ×3 | 1.38 s (σ=0) | **0.88 s** (σ=0) | **−36.2%** |
| 6k prefill ×3 | 3.38 s (σ=0) | **1.58 s** (σ=0) | **−53.3%** |
| 3k needle ×2 | PASS 11.82/11.86 s | PASS 11.40/11.45 s | −3.5% |
| 6k deep needle ×2 | PASS 5.21/5.19 s | **PASS** 3.38/3.36 s | −35% |
| c=2 3k mixed | 4.67 s | 3.65 s | −21.8% |
| decode control (256-tok gen) | 96.66 tok/s | 95.72 tok/s | −1% (untouched path, noise) |

Mechanism check: predicted −36.6% at 3k (42.1% share × ~87% kernel-time
removal) — measured −36.2%. Gains scale with context exactly as replacing a
quadratic-inefficient kernel should: −5% @512, −36% @3k, −53% @6k.
Prefill throughput at 3k: 2226 → **3491 tok/s**; at 6k: 1775 → **3797 tok/s**.

## Build/integration notes

- `flash_prepare_scheduler.cu` must be in the FA3 compile list even for the
  non-varlen path — the launch template's runtime `VARLEN_SWITCH` references
  `prepare_varlen_num_blocks` from every fwd instantiation (`852cb9c7`).
  The first compile gate (archive-only) could NOT catch this: undefined
  symbols only surface when a Rust caller pulls the objects into the binary
  link.
- Non-varlen + causal selects `DynamicPersistentTileScheduler` — needs only
  the zeroed `tile_count_semaphore` (verified against `tile_scheduler.hpp`;
  the `num_nheads_in_l2_ptr` machinery is Varlen-scheduler-only).
- nvcc flags mirror `hopper/setup.py`: `-DNDEBUG` is upstream-marked
  perf-critical; `CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED` mandatory.
- FA3 build adds ~5 min wall to a cold cuda-kernels compile (6 heavy units,
  parallel); opt-in `ARLE_CUDA_ENABLE_FA3=1` keeps non-FA3 iteration fast.

## Tradeoffs (per the no-free-lunch rule)

- +28 MB vendored source (hopper headers + pinned cutlass include).
- sm_90a-only: the runtime gate + stub marker keep other SMs on the in-tree
  kernel; sm_89 consumer cards never see FA3 code.
- Two attention implementations behind one gate until the in-tree kernel is
  deleted — deletion deferred until FA3 covers decode (step 5: batched
  decode via the vendored paged/packgqa/split units; then the in-tree
  HD256 kernels can go).

## Next lever (re-ranked)

6k prefill is now GDR-recurrent-bound: attention's 42% share collapsed, so
`gated_delta_rule_prefill_recurrent` (28.0% pre-FA3, now ~45%+ of the
remaining prefill window) is board #3's FlashQLA chunked kernel — the
remaining 1.58 s at 6k has a sub-second target.

## Rule

- An archive-level symbol check passes builds that a binary link will fail:
  the stub/marker pattern needs ONE Rust call site landed before the build
  gate run, or the gate proves nothing about link completeness.
- "Quadratic kernel replaced" claims must show the gain GROWING with
  context length across ≥3 sizes — a flat Δ% would mean the win came from
  somewhere else.
