# CUDA kernel systematic audit (2026-05-30) — what needs work, ranked by the measured DSv4 profile

Multi-agent audit of all 66 `crates/cuda-kernels/csrc/` kernels + 11 TileLang AOT
generators, ranked against the measured DSv4-Flash 8×H20 TP=8 **prefill** profile
(per-phase trace: 84% MoE FFN = all-reduce 52% of FFN + native expert GEMM 41%;
FlashMLA attention ~16%). 57 kernels flagged; the synthesis below is impact-ranked.

## Headline — the kernels are NOT the easy win

- The **dominant FFN phase (all-reduce, 52% of FFN) is NOT kernel-optimizable**: A4
  compute/comm overlap was already measured and KILLED (<0.006% wall-clock,
  `errors/2026-05-28-dsv4-a4-multi-stream-overlap-kill`). Its ratio is also inflated by
  the sync-instrumented trace it came from. It is a ceiling problem, not a kernel bug.
- The single real kernel lever is the **expert GEMM (41% of FFN)**.
- Several long-standing flags are **STALE and were corrected**: the "M-blind grouped
  GEMM" is already M-tiled (fixed `ac1f0ccc`, bought only 13%); the route swiglu/scatter
  kernels are <1% of prefill; `deepgemm` now JIT-compiles (`38bf157b`) and is wired
  (`mlp.rs:4921`) — the blocker is no longer the wiring.

## P0 — the real levers (on the measured bottleneck)

1. **Expert FP8 GEMM via deepgemm** — `gemm/deepgemm_native.cu` + `dsv4_deepgemm_ops.cu`.
   JIT now compiles + dispatch is wired, but it **crashes at runtime on H20**
   ("unspecified launch failure" + nondeterministic output,
   `errors/2026-05-27-b335-deepgemm-runtime-crash-h20`). This is the 41%-of-FFN, claimed
   ~2.5× lever. **Fix: compute-sanitizer one prefill+decode with `EXPERT=deepgemm`
   (~30 min on pod)** → pinpoint the illegal-access kernel; suspect the H20 SM9.0
   cluster/TMA feature path or a `packed_x` scale-stride layout mismatch (NOT the wiring).
   *Biggest single lever; the deepgemm path has the highest ceiling.*
2. **Expert FP8 GEMM scalar fallback** — `dsv4_fp8_grouped_gemm_batch_kernel`,
   `gemm/dsv4_grouped_gemm.cu:60-190`. NOT M-blind (already M-tile=32). Real issue: the
   inner loop (153-168) is **pure scalar FP32-FFMA — no tensor cores, no cp.async**.
   **Fix: cp.async double-buffer weight+scale tiles + port the accumulate to
   `mma.m16n8k16`.** Toolchain-independent alternative to deepgemm. Caveat: roofline says
   ~13% alone — necessary, not sufficient.
3. **native-deepep dispatch host-poll** — `deepep-sys/csrc/deepep_buffer.cpp:379-391`. A
   `while(true)` busy-loop spinning on `moe_recv_counter_host` until notify_dispatch
   signals = a per-layer CPU↔GPU sync that **serializes prefill** (this is *why*
   native-deepep is SLOWER than all-reduce at prefill despite +46% at decode). **Fix:
   device-side capacity sizing (fixed-capacity recv buffer)** so the recv count no longer
   gates expert compute on the host. The structural lever that makes the +46% decode win
   transfer to prefill.

(The all-reduce itself is re-ranked OUT of kernel-fix P0 — not optimizable.)

## P1 — real, but off the DSv4 prefill critical path

- **`marlin_w4_fp8_kernel.cu`** (`run_marlin_w4_fp8_prefill`) — 100% `cudaErrorMemoryAllocation`
  under sustained conc>1 (`errors/2026-05-10-pf83`); per-call workspace alloc. Default OFF
  (`INFER_MARLIN_W4_FP8_PREFILL` opt-in), Qwen W4 path. Fix: persistent workspace pool +
  pre-request budget check + a sustained-load smoke gate.
- **`dsv4_fp8_gemv_batch_mma_kernel`** (`quantized_gemv_mma.cu`) — decode tensor-core GEMV
  behind `ARLE_DSV4_FP8_GEMV_MMA` knob; for B≤16 decode MMA should always beat scalar but
  parity unvalidated at scale. Fix: nsys A/B at B=4,8,16, validate numerics, make default.
- **INT4 KV quant** (`kv_quant.cu` `quantize_paged_kv_int4_*` 983-1174) — two-barrier
  (reduce, stage, pack) adds decode latency at low batch. Fix: single-pass reduce+pack in
  registers. (INT4 KV quality separately gated.)
- **TileLang gated-delta FullRow-WGMMA** (`gdr_prefill_batch.cu:137`) — HANGS on sm_90 for
  seq_len≤32 (TileLang short-tile codegen bug). Mitigated (routed to recurrent, OFF). Qwen
  hybrid path, not DSv4. Needs upstream TileLang fix.
- **tilelang CUDA-12.2 pin** — `tilelang≥0.1.10` c++20 fold-expr breaks nvcc at `-arch=sm_90a`
  on the CUDA-12.2 pod. Pin <0.1.10 until pod ≥ CUDA-12.3. Gates ALL TileLang regen.

## P2 — down-ranked / per-model / cold

- `dsv4_route.cu` swiglu/scale/scatter (1D element-wise, warp divergence) — but <1% of
  prefill; fuse only if a fresh trace promotes it >2%.
- `quantized_gemv.cu` W4A16/W8A16/W2A16 scalar GEMV — same lever as the P1 MMA path
  (extend MMA to W4/W8 variants).
- `decode_prep_paged.cu` / `_hd256.cu` — low-batch occupancy underutilization (launch
  config, not algorithmic); profile first.
- `dsv4_csa_select_kernel` (`misc/dsv4_attention.cu:1110`) — per-token sequential bitonic
  sort over 4096 keys, ~141ms/call, but only in CSA-select prefill + attention is ~16%.
  Parallelize the sort only if a trace promotes attention.
- misc Qwen-dense/GDR ops (`conv1d_prefill_batch.cu`, batched `norm.cu`, `fused_mlp.cu`,
  `dsv4_mhc.cu`, `gdr_prefill_solve.cu` single-thread Gauss-Jordan) — genuinely
  unoptimized but each <1% on the DSv4 trace; optimize per-model when that model's trace
  promotes them.
- **dead code** — `attention/prefill_attention.cu`, `attention/fused_attention.cu` have NO
  Rust FFI binding (only `nonpaged_prefill_attention_cuda` is wired). Candidates for
  deletion. `tilelang` HD256 prefill/decode + FP8 variants exist but aren't built (roadmap-gated).

## The takeaway

Fast DSv4 prefill is gated on the **expert FP8 GEMM**, and the fastest path (deepgemm) now
COMPILES + is WIRED — only a **runtime H20 crash** stands between it and the ~2.5× win.
That `compute-sanitizer` root-cause (~30 min) is the single highest-value kernel action.
The toolchain-independent fallback (cp.async + MMA on the scalar grouped GEMM) is the
parallel hedge. Everything else is either already optimized, not on the hot path, or a
per-model/cold concern. The "lots of kernels need work" intuition is mostly STALE flags.
