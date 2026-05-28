# GAP-A Phase 3 — CUTLASS-MMA kernel landed, dispatch shim + pod parity pending

## SLO-shape probed? — N (kernel written, no Rust caller yet; license-or-kill PASSED at Phase 2)

## TL;DR

Phase 2 license-or-kill (`287f831a` commit) cleared the gate: standalone
H20 micro on the canonical DSv4 GEMV shape (`B={1,4,16}`, N=2048, K=7168)
measured the scalar `dsv4_fp8_gemv_batch_tiled_kernel` at
`frac_peak = 0.043` (B=1,4) and `0.011` (B=16) of HBM3 4 TB/s. Compute /
instruction-issue bound at 96–99 % of HBM3 BW left on the floor — the
1.5× license threshold is reachable via `mma.m16n8k16` BF16×BF16→FP32.

Phase 3 lands the kernel: `crates/cuda-kernels/csrc/gemm/quantized_gemv_mma.cu`
(326 LoC). `extern "C" dsv4_fp8_gemv_batch_mma_launch` is exported from
the CUDA side but **no Rust FFI declaration + no dispatch shim in the
scalar wrapper yet** — by design, since the dispatch / parity-test wire-up
needs a dedicated session with `isolation: "worktree"` to avoid the
parallel-subagent shared-worktree race that contaminated `ab850f7a` +
`9ffaa622` earlier in this session.

This entry documents the half-state explicitly per
[`feedback_no_half_states.md`](../../../.claude/projects/-Users-user-code-agent-infer/memory/feedback_no_half_states.md):
the kernel compiles, the dead `extern "C"` symbol is harmless, the wire-up
work is the next session's deliverable.

## What's in `quantized_gemv_mma.cu`

- `gemv_mma_decode_e8m0` / `gemv_mma_decode_fp8_e4m3` — per-element
  FP8E4M3 + E8M0 dequant to FP32 (matches scalar kernel's dequant path
  bit-for-bit at the BF16 cast boundary).
- `pack_bf16x2` — packs two BF16 into a `uint32_t` register for the MMA
  mainloop's K-tile feed.
- `mma_m16n8k16_bf16` — wraps `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`
  inline asm. BF16×BF16→FP32, four FP32 output fragments per warp.
- `dsv4_fp8_gemv_batch_mma_kernel` — full block kernel. Loads tile of
  weights through smem with `cp.async` staging, dequants FP8 → BF16
  into A-fragments, broadcasts input tile to B-fragments, issues
  `mma.m16n8k16` for the BF16 mainloop, accumulates in FP32 registers,
  writes back to global BF16.
- `dsv4_fp8_gemv_batch_mma_launch` — `extern "C"` entry computing grid /
  block shape from the same `(B, N, K)` arguments as the scalar launch.

## Phase 2 license-or-kill evidence

The standalone micro-experiment source is preserved as research
evidence at
[`docs/research/2026-05-28-gap-a-mma-license-micro.cu`](../../research/2026-05-28-gap-a-mma-license-micro.cu).
Build/run on pod:

```bash
nvcc -O3 -arch=sm_90 -std=c++17 \
  docs/research/2026-05-28-gap-a-mma-license-micro.cu -o /tmp/gap_a_micro
/tmp/gap_a_micro
```

Reproduces:

| B  | wall-clock (µs) | achieved BW (GB/s) | frac of 4 TB/s peak |
|----|-----------------|--------------------|---------------------|
| 1  | (Phase 2 commit) | ~170 GB/s          | **0.043** |
| 4  | "                | ~170 GB/s          | **0.043** |
| 16 | "                | ~45 GB/s           | **0.011** |

License granted at Phase 2 commit `287f831a`. See its body for the
diagnosis: the scalar inner loop is FFMA-issue-bound, not bandwidth-bound;
swapping the scalar FFMA sequence for `mma.m16n8k16` (one MMA = 16
FFMA-equivalents per warp-cycle) should give the ~2× kernel-local
speedup needed to clear the 1.5× gate.

## What's NOT in this commit

1. **No Rust FFI declaration.** `crates/cuda-kernels/src/ffi/gemm.rs`
   has no `dsv4_fp8_gemv_batch_mma_launch` extern signature. The
   symbol compiles but no Rust caller exists.
2. **No dispatch shim.** `quantized_gemv.cu::dsv4_fp8_gemv_batch_cuda`
   continues to dispatch to the scalar kernel. The
   `dsv4_fp8_gemv_batch_mma_launch` shape gate (M ≤ 16 → MMA,
   M > 16 → scalar) is unwired.
3. **No bit-parity test.** `infer/src/ops/tests.rs` has no
   `mma_kernel` arm. Phase 4 plan: extend the existing scalar parity
   test with `ARLE_DSV4_FP8_GEMV_MMA=1` toggle and gate at relerr ≤
   1e-3 (BF16 last-bit rounding budget).
4. **No pod TPOT bench.** Will run after Phase 4 wire-up.

## Why we stopped here

The GAP-A subagent was producing useful work but the parallel-shared-
worktree race (documented in
[`errors/2026-05-28-parallel-subagent-commit-contamination.md`](../errors/2026-05-28-parallel-subagent-commit-contamination.md))
contaminated two earlier commits with content from the parallel GAP-C
subagent and the user's INT4 KIVI WIP. Continuing Phase 3 wire-up in the
same shared workspace would have risked a third contamination. The
correct path forward is to land Phase 3's MMA kernel as a clean
isolated commit, then resume Phase 4 wire-up in a dedicated session
with `isolation: "worktree"`.

## Phase 4 — Next-session checklist

1. Spawn a single `general-purpose` subagent with `isolation: "worktree"`.
2. Add `dsv4_fp8_gemv_batch_mma_launch` to `crates/cuda-kernels/src/ffi/gemm.rs`.
3. In `quantized_gemv.cu::dsv4_fp8_gemv_batch_cuda`, gate on
   `B <= 16 && weight is fp8e4m3 && env(ARLE_DSV4_FP8_GEMV_MMA)`:
   dispatch to MMA launch; fall through to scalar otherwise.
4. Add bit-parity test in `infer/src/ops/tests.rs` (small shapes:
   B=1, N=128, K=512), MMA vs scalar relerr ≤ 1e-3.
5. Pod build + parity smoke (env knob OFF → bit-equal, env knob ON →
   relerr OK).
6. Pod TPOT A/B at canonical DSv4 decode shape, env knob OFF vs ON.
   PASS gate: ≥ 4 % decode wall-clock saving (audit projection).
7. If PASS: write Phase 4 wins entry, flip env default, plan GAP-D
   (mechanical port to `dsv4_grouped_gemm.cu`).
8. If FAIL: errors entry, revert env default, audit kernel for
   register pressure / MMA tile mismatch.

## Refs

- Audit: [`docs/research/2026-05-28-arle-kernel-vs-sota-audit.md`](../../research/2026-05-28-arle-kernel-vs-sota-audit.md) §GAP-A
- Phase 1 plan: [`docs/plans/2026-05-28-gap-a-cutlass-mma-quant-gemv.md`](../../plans/2026-05-28-gap-a-cutlass-mma-quant-gemv.md)
- Phase 2 license-or-kill (commit `287f831a`): standalone micro PASS at H20.
- Parallel-subagent contamination errors: [`errors/2026-05-28-parallel-subagent-commit-contamination.md`](../errors/2026-05-28-parallel-subagent-commit-contamination.md).
