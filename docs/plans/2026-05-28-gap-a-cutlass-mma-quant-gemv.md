# GAP-A · CUTLASS-MMA path for `dsv4_fp8_gemv_batch_tiled_kernel`

Date: 2026-05-28
Driver: GAP-A from
[`docs/research/2026-05-28-arle-kernel-vs-sota-audit.md`](../research/2026-05-28-arle-kernel-vs-sota-audit.md).
Owner: this thread (CUDA-kernels).

## §0 SOLID gate — confounders surfaced up front

1. **Pod target is H20 (SM_90), not L4 (SM_89).** The audit's
   "~4–6 % decode wall-clock" projection is for L4. On H20 the same audit
   states: *"on H20 (SM_90) DeepGEMM already wraps the WGMMA path so the gap
   is narrower"* (line 134). Pod-side bench numbers are the only currently-
   available wall-clock evidence; **L4 wall-clock impact must remain
   `Hypothesis` until an SM_89 box is provisioned.** Do not retro-anchor the
   SM_90 result onto SM_89 framing.
2. **Roofline risk on H20.** Scalar kernel today does 1 FMA / FP8 byte
   loaded (32 FLOP/B at `DSV4_BATCH_TILE=32`). H20 HBM3 sustained ≈ 4 TB/s,
   FP32 SIMT throughput ≈ 67 TFLOP/s → roofline crossover ≈ 17 FLOP/B. At
   `B=1..4` (most common decode) effective FLOP/B is 1–4, **BW-bound**.
   At `B=32` it is 32, **compute-bound** but capped by FP32 SIMT, not Tensor
   Cores. **Tensor-core MMA only helps if compute is the actual bottleneck**
   — must be verified per-shape in Phase 2.
3. **DeepGEMM `block_m >= 64` is wrong at GEMV shape.** Routing M≤16
   batches through `m_grouped_gemm_nt_masked` pads M up to 64, throwing
   away 75–94 % of TC work. That is why the audit calls for a **fresh**
   MMA path, not "just call DeepGEMM with M padding".
4. **One variable per commit.** This task changes one file
   (`quantized_gemv.cu` → new `quantized_gemv_mma.cu` + dispatch shim).
   Sibling WIP files (`decode_attention_quantized.cu`, `kv_quant.cu`,
   `qwen35/*.rs`, `scheduler/cuda/*.rs`, etc.) are off-limits per the
   task brief and per `feedback_no_half_states.md`.

License-or-kill gate (Phase 2) is the SOLID anchor: if the micro
experiment shows MMA < 1.5× scalar on the canonical shape, **kill the
axis cleanly** with an errors entry. Do not "make it work harder".

## Op identity (locked)

Per-shape contract for the DSv4 FP8 batched GEMV decode path:

| Symbol | Value | Source |
|---|---|---|
| Op | `Y[b, n] = Σ_k A[n, k] · X[b, k]` over BF16 input, FP8E4M3 weight, E8M0 block scale | `dsv4_fp8_gemv_batch_tiled_kernel` (`quantized_gemv.cu:392`) |
| A layout | `weight[N, K]` row-major FP8E4M3, byte-packed | line 416 |
| Scale layout | `scales[scale_rows, scale_cols]` E8M0, block size `block_h × block_w` where `block_h = ⌈N/scale_rows⌉`, `block_w = ⌈K/scale_cols⌉` | lines 411–417 |
| X layout | `input[B, K]` row-major BF16 | line 438 |
| Y layout | `output[B, N]` row-major BF16 | line 464 |
| B | 1..32 (`DSV4_BATCH_TILE=32`); decode shape is 1..16 | `linear.rs:2048`, `mlp.rs:373` |
| K canonical (DSv4) | 7168 (hidden) / 16384 (intermediate) | per audit §GAP-A |
| N canonical | 2048 / 4096 / 7168 | per DSv4 config |
| Scale layout (DSv4 std) | `block_w = 128` (granK), `block_h = 128` (granM) | `dsv4_deepgemm_ops.cu`, line 41 `kScaleGranK = 128` |

Caller bindings:

- `infer/src/ops/linear.rs:2047-2060` (single-matrix Dsv4Fp8 batch)
- `infer/src/model/deepseek/mlp.rs:373-384` (per-expert grouped GEMV
  segment)
- `crates/cuda-kernels/csrc/gemm/dsv4_grouped_gemm.cu` (GAP-D; sister
  kernel, deliberately out of scope this task)

## Industry reference choice

Three candidates were considered. Selected with rationale:

| Reference | Shape fit | LoC complexity | Decision |
|---|---|---|---|
| DeepGEMM `m_grouped_gemm_nt_masked` (vendored) | `block_m=64..128`, drops M≤16 → padded 4–8× waste | wraps cubins, already integrated | **Reject for GEMV shape.** Already used for full-tile case. Keep as fallback for B ≥ 32 (existing path). |
| FlashInfer `bgmv` family | `m16n8k16` (Ampere/Ada), `m16n8k32` (Hopper). Direct M=1..16 fit. | ~350 LoC per dtype | **Primary reference for SM_89**; identical scale layout (per-channel, but adaptable to per-block). |
| SGLang `fp8_blockwise_scaled_grouped_mm.cu` | CUTLASS 3.x grouped, M≤16 fast path, K=128 block scale | ~600 LoC, CUTLASS-dep | **Primary reference for SM_90.** Block-scaled FP8 layout matches DSv4 exactly. |

Practical choice: **port the inner mainloop**. Avoid pulling the CUTLASS
3.x `Gemm` template machinery whole — it would double crate compile
time and entangle the build. Hand-write a `mma.m16n8k32` (FP8E4M3 →
FP32 accumulator) mainloop with `cp.async.cg.shared.global` weight
loads, per-K-block-128 scale-aware accumulation. SGLang is the *layout*
reference, FlashInfer is the *PTX-issue-style* reference.

## Design — `dsv4_fp8_gemv_batch_mma_kernel` (SM_89+SM_90 single source)

Tile shape (decode):

- `BLOCK_M = 16` (one warp-tile of `m16n8k32`)
- `BLOCK_N = 64` (4 warps fanned across N)
- `BLOCK_K = 128` (matches DSv4 scale granularity → one scale per K-block)
- Threads per block: 128 (4 warps).
- Grid: `(⌈N / BLOCK_N⌉, ⌈B / BLOCK_M⌉)`.
- cp.async stages: 3 (pipeline weight + activation tiles).
- Smem: A tile `16 × 128` BF16 (4 KB), B tile `64 × 128` FP8 (8 KB),
  3 stages = 36 KB. Below the 100 KB SM_89 cap.

PTX-level mainloop (per K-block, per warp):

1. `cp.async.cg.shared.global [smem_a + …], [global_x + …], 16` — load
   `16 × 128 / 4 / 32 = 2` `int4` per thread for A tile (BF16).
2. `cp.async.cg.shared.global [smem_b + …], [global_w + …], 16` — load
   B tile (FP8). 128 threads × 16 B = 2 KB per cycle.
3. `cp.async.commit_group; cp.async.wait_group 2` — 3-stage pipeline.
4. Per K-block: decode E8M0 scale once into a register (1 load, 1
   multiply).
5. `ldmatrix.sync.aligned.m8n8.x4.shared.b16` for A tile.
6. Per `m16n8k32` instruction:
   - Dequantize FP8 → BF16 in registers (8 elements / thread). FlashInfer
     does this with `cvt.rn.bf16x2.f8x2_e4m3` (Hopper/Ada PTX).
   - Multiply by the K-block E8M0 scale (1 FFMA / 2 elements via `bf16x2`).
   - Issue `mma.m16n8k32.row.col.bf16.bf16.bf16.f32`.
7. Accumulate into per-warp `float regs[16][2]`.

Epilogue: warp-shuffle reduction across N axis is not needed
(`BLOCK_N` is split across warps; each warp owns its N-tile). Direct
`stmatrix.sync.aligned.m8n8.x4.shared.b16` to smem then coalesced store
back to `output[B, N]`.

Tail handling: `B % BLOCK_M != 0` → masked write at epilogue.

## SM_89 vs SM_90 dispatch

- SM_89 (L4, Ada): `mma.m16n8k32` FP8E4M3 → FP32 is native; `ldmatrix`
  available; cp.async OK. **No tcgen05 / WGMMA.**
- SM_90 (H20, Hopper): same path works **AND** Hopper has WGMMA
  (`wgmma.mma_async.sync.aligned.m64n*k32`) that's natively 4× larger
  per-issue. For now ship **single source** with `mma.m16n8k32` —
  WGMMA only helps when `BLOCK_M ≥ 64`, which is the wrong shape for
  decode. Revisit if Phase 4 H20 bench shows compute-bound at `B=16`
  with 16k-row epilogue.
- SM ≤ 86 (no FP8 MMA): keep scalar fallback in the dispatch shim.

## Phase 2 license-or-kill micro-experiment

**File** (temporary, gitignored): `crates/cuda-kernels/csrc/gemm/_gap_a_micro.cu`

**Goal**: prove `mma.m16n8k32` beats the scalar kernel by ≥ 1.5× on the
canonical decode shape on H20 (pod). 1.5× is the floor; if MMA only
ties, the BW-roofline claim (§0 confounder 2) is confirmed and the axis
dies.

**Shape**: `B = 1, 4, 16`; `N = 2048`; `K = 7168`. Three sub-experiments.

**Bench harness**: criterion-driven (mirror `infer/benches/ops/ops_cuda_bench.rs`
shape conventions). 200 warmup iters, 1000 measurement iters,
cudaEvent-based ms.

**Build**: ship the micro file behind `cargo:rustc-cfg=gap_a_micro` so
the FFI symbol is only emitted in micro builds; do not ship in default
build. Avoid scope creep.

**PASS criteria** (all three needed):
- `B=16, N=2048, K=7168`: MMA / scalar speedup ≥ 1.5×.
- `B=4`: MMA / scalar speedup ≥ 1.2× (BW-bound, smaller margin).
- `B=1`: MMA / scalar speedup ≥ 0.9× (no regression; this shape is
  pure BW-bound, MMA shouldn't help but mustn't hurt).

**FAIL criteria**:
- Any of the above missed → KILL: write
  `docs/experience/errors/2026-05-28-cuda-gap-a-mma-kill.md`,
  document numbers + roofline calc, stop. Do NOT proceed to Phase 3.

**Confounders to actively rule out** (run controls):
- Activation tile load BW dominates → run B=16 with `K=256`
  (compute-light shape). If MMA gain holds → compute-bound confirmed.
- Scalar kernel is occupancy-bound (16 active warps per SM today vs MMA
  4 warps) → record `ncu --metrics achieved_occupancy` for both.
- L2 cache hit rate confounder (small-shape micro fits in L2) → run
  `B=16, N=4096, K=16384` (won't fit in L2 of H20's 60 MB) and confirm.

## Phase 3 full-port plan (gated on Phase 2 PASS)

If Phase 2 passes:

1. **New file**: `crates/cuda-kernels/csrc/gemm/quantized_gemv_mma.cu`.
   ~400–600 LoC. Kernel + extern-C entrypoint:
   ```cpp
   extern "C" cudaError_t dsv4_fp8_gemv_batch_mma_cuda(
       const uint8_t* weight, const uint8_t* scales,
       const __nv_bfloat16* input, __nv_bfloat16* output,
       int B, int N, int K, int scale_rows, int scale_cols,
       cudaStream_t stream);
   ```
2. **Dispatch shim** in existing `quantized_gemv.cu` `dsv4_fp8_gemv_batch_cuda`
   entry — route to MMA path when `B ≥ 2 && N % BLOCK_N == 0 && K % BLOCK_K == 0`,
   else fall through to scalar. **Per `feedback_no_half_states.md` the
   shim is a *temporary 1-commit* state**: once parity confirmed (Phase 4),
   delete the scalar tiled kernel in a follow-up commit. The single-token
   `dsv4_fp8_gemv_batch_kernel` (B=1 path, no tile) stays — it's pure
   BW-bound and MMA cannot help.
3. **FFI binding** in `crates/cuda-kernels/src/ffi/gemm.rs` only changes
   if we add a separate FFI entry; per shim design, **no FFI change**
   (existing `dsv4_fp8_gemv_batch_cuda` is the dispatcher).
4. **Parity test** in `infer/src/ops/tests.rs`: compare MMA path output
   vs scalar on `B=4, N=128, K=512` (small enough for CI Mac
   `cuda,no-cuda` typecheck; real bit-match deferred to pod).
   Tolerance: max |rel diff| ≤ 1e-3 (BF16 + FP32-accum vs MMA-accum
   should match within last-bit BF16 rounding).
5. **build.rs**: no change needed — file is auto-discovered by
   `collect_cu_files`.
6. **GAP-D follow-on** (deliberately out of scope this task): once
   `quantized_gemv_mma.cu` is in, the `dsv4_grouped_gemm.cu` scalar
   path adds one grid Z axis and is mechanical.

## Phase 4 validation

Per task brief:

1. Local: `cargo check -p infer --no-default-features --features cuda,no-cuda`
2. Pod: `~/bin/pod-exec 'cd /data01/build/arle && CUDA_HOME=/usr/local/cuda CUDARC_CUDA_VERSION=12060 cargo build --release --features cuda,nccl -p infer --bin infer 2>&1 | tail -40'`
3. Pod TPOT A/B: DSv4 c=1, c=16; this commit vs previous-main. Tool:
   `scripts/bench_dsv4_trace_http.py` per
   [`docs/bench-and-trace-spec.md`](../bench-and-trace-spec.md).
4. Wins entry under
   `docs/experience/wins/2026-05-28-cuda-gap-a-cutlass-mma-quant-gemv-{phase}.md`
   with real Δ% TPOT, framed per spec §3 (per-NVTX *and* per-wall-clock
   per CLAUDE.md §0 framing rule — wall-clock is ground truth).

## Trip wires (auto-kill if hit)

- Phase 2 micro speedup < 1.5× at `B=16` → KILL.
- Phase 3 parity test max |rel diff| > 1e-3 → root-cause before commit;
  do NOT relax tolerance.
- Phase 4 TPOT shows < 1 % wall-clock win at c=16 → KILL the
  shim (revert to scalar default), file as `wins/` with caveat that
  axis is real on L4 but not H20.

## Cross-refs

- Audit doc: [`docs/research/2026-05-28-arle-kernel-vs-sota-audit.md`](../research/2026-05-28-arle-kernel-vs-sota-audit.md)
- DSv4 binding constraints: [`docs/research/2026-05-15-dsv4-decode-memaccess-binding-constraints.md`](../research/2026-05-15-dsv4-decode-memaccess-binding-constraints.md)
- Industry reference repos cited (FlashInfer `bgmv`, SGLang
  `fp8_blockwise_scaled_grouped_mm`, DeepGEMM `m_grouped_gemm_nt_masked`)
  — links in audit doc §Cross-refs.
