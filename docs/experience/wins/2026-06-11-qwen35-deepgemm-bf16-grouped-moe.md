# Qwen3.5/3.6 MoE expert GEMMs → DeepGEMM SM90 BF16 m-grouped — pending-remote, 2026-06-11

**Status: `pending-remote`** — code landed Mac-side (typecheck + CPU unit tests
only); all GPU verification is pod-side (8×H20, sm_90a). No perf claim is made
by this entry. The default is byte-identical: `ARLE_QWEN35_DEEPGEMM` is OFF.

## Goal

- Replace the hand CUDA-core grouped MoE expert GEMMs (~3.9 TFLOP/s class,
  `csrc/gemm/moe_grouped_gemm.cu`) in the Qwen3.5/3.6 path with DeepGEMM SM90
  BF16 m-grouped GEMMs (vendored `deepseek-ai/DeepGEMM` @ `714dd1a4`,
  `sm90_bf16_gemm.cuh`, `MGroupedMasked` + `MGroupedContiguous`), behind
  `ARLE_QWEN35_DEEPGEMM` (default OFF).
- Target shape: Qwen3.6-35B-A3B on H20 — 256 routed experts top-8, gate/up
  `[512, 2048]`, down `[2048, 512]`, BF16; decode R=8, prefill chunk
  R=16384 routes (~4.1 TFLOP routed compute/chunk → ~1 s on CUDA cores vs
  ~40–120 ms expected at DeepGEMM-class rates incl. alignment padding).

## Hypothesis

- Prefill routed-expert time drops ~8–20× (padded-cap contiguous layout does
  ≤ ~3× nominal FLOPs at R=16384/G=256, still ≫ the CUDA-core rate).
- Decode (masked, R=8): wash-to-small-win — B=1 decode is GPU-bound elsewhere;
  the masked GEMM computes 8 × BLOCK_M-row tiles vs the hand kernel's 8 rows.

## Dispatch + contracts (for the pod A/B)

- `R = T·topk ≤ 128` → **masked** (`[G, 128, K]` bands, `masked_m = counts`,
  shapes routing-independent ⇒ CUDA-graph-safe). R ≤ 128 is the only
  host-provable bound on `max_g count_g` (counts are device-resident).
- `R > 128` → **contiguous** with 128-aligned per-group segments
  (`moe_exclusive_scan_aligned_i32`) and host row cap
  `align(R,128) + 128·min(R,G)`; pad rows carry `m_indices = -1` +
  route-slot `-1` (scatter skips them).
- **m_indices contract** (re-derived from vendor `scheduler/gemm.cuh`): the
  contiguous kernel resolves the B group ONCE per BLOCK_M=128 tile from
  `m_indices[tile_start]`. The existing **DSv4 prefill contiguous path
  violates this** (compact unaligned segments via the plain exclusive scan in
  `deepgemm_grouped_experts`, `infer-cuda/src/moe.rs`) — boundary tiles
  compute the tail rows against the wrong expert. CPU test
  `compact_layout_violates_per_tile_group_contract` documents it; DSv4 fix is
  a separate follow-up (this entry does not touch the DSv4 path).

## Pod verification plan (lead)

1. Build with `ARLE_CUDA_ENABLE_DEEPGEMM_NATIVE=1`; unit-kernel A/B of
   masked + contiguous BF16 GEMMs vs the hand path on random tensors
   (relerr gate, both TP=1 G=256 and TP=2 G=128 shapes).
2. e2e correct-inference gate (NOT byte-identity — MoE non-determinism):
   smoke ×3 same-config consistency + needle retrieval,
   `ARLE_QWEN35_DEEPGEMM=1` vs baseline envelope.
3. `scripts/bench_guidellm.sh` same-binary two-env A/B vs the latest
   Qwen3.6/H20 baseline; Δ% rows for TTFT / ITL / output tok/s; roofline row
   for the prefill grouped GEMM.
4. Default stays OFF until 2+3 license the flip.

## Results

- `pending-remote` — to be filled by the pod run.

## Problems

- First decode/prefill after a cold JIT cache pays an nvcc `-cubin` compile
  per (shape, layout) tuple (DG_JIT_CACHE_DIR-cached, same as the DSv4 FP8
  path).
- Contiguous prefill computes up to `cap − Σ align(count_g,128)` extra pad
  tiles because the aligned total is device-resident and the TMA descriptors
  need a host `m`; refinement (psum layout / device-side total) deferred.

## Learnings

- Filled after the pod run.
