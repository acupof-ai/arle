# Qwen3.6 FP8 prefill: GEMV fallback → GEMM (dequant + DeepGEMM default) — ~16× — pending-remote

`pending-remote`: measured on the 8×H20 pod (Qwen3.6-27B-FP8, GPU0). Commits land on
main; this entry is the bench gate per §Benchmarks.

## Context
The KV capacity probe (打爆) surfaced a pathologically slow prefill: 18 410-token prefill =
**147.5 s**. Per-stage CUDA profiling (`ARLE_QWEN35_PROFILE=1`, 12K prefill) attributed it —
**not** to attention (TileLang HD256 paged attn is already FlashAttention-tiled: `full_paged/
attention` = 0.4 s) — but to the **FP8 linear layers running a memory-bound GEMV**:

| stage (12K prefill, w/ profile) | before |
|---|---|
| `qwen/fp8/gemv_batch` | 100 788 ms |
| `qwen/dense_ffn` | 71 444 ms |
| `qwen/full_paged/attention` (TileLang Flash) | 419 ms |

Root cause: `gemm_batch` tried DeepGEMM → `try_fp8_dequant_bf16_gemm_batch` → scalar GEMV.
The dequant→GEMM was gated `!qwen_fp8_dense_sm_supports_deepgemm` (sm<9 only), so on Hopper with
DeepGEMM **not compiled** (`native_bridge=not_compiled`, the opt-in `ARLE_CUDA_ENABLE_DEEPGEMM_NATIVE`
unset) prefill skipped the GEMM and fell to the per-token GEMV (one 17408×5120 chunk projection =
**53 ms** via GEMV vs ~2.4 ms for a GEMM).

## What Worked
1. **`c6a23bdb` — prefill never GEMV.** Removed the sm<9 gate on `try_fp8_dequant_bf16_gemm_batch`:
   we only reach it after DeepGEMM declined, so large-M (prefill) uses the dequant→cuBLAS-BF16 GEMM
   on ANY arch. Decode (M < MIN_M) keeps GEMV (optimal there). Portable safety net.
2. **`66549bfa` — DeepGEMM FP8-native default-ON.** build.rs auto-detects sm_90 + vendored source
   (mirrors the FlashMLA auto-detect) and compiles the native bridge by default; opt out with
   `ARLE_CUDA_DISABLE_DEEPGEMM_NATIVE=1`. Was opt-in, so prod shipped without it.
3. **`feb89e4e` — prompt cap.** `max_prompt_tokens` 32 768 → per-request capacity
   (`total_pages × page_size − 1/8`), so moderate prompts stop aborting.

Measured after (same 12K profile / wall clock):

| | before | after |
|---|---|---|
| FP8 dense GEMM stage | gemv_batch 100.8 s | dequant→GEMM 6.4 s; **DeepGEMM warm** |
| dense_ffn | 71.4 s | 4.6 s |
| **18K prefill wall** | **147.5 s** | **~9 s (~16×)** (warm DeepGEMM 13.8K = 7.0 s w/ profile) |

DeepGEMM auto-enabled (`cargo:warning DeepGEMM native enabled (sm_90 + vendored source)`),
`deepgemm_native_cuda.o` compiled (default-on did not break the build), serve no longer logs
"DeepGEMM disabled".

## Rule
- **Prefill must never run GEMV.** GEMV (memory-bound, per-token) is decode-only; multi-token
  prefill is always a GEMM (DeepGEMM FP8-native > dequant→cuBLAS-BF16 > **never** GEMV). A
  `sm<9`-gated GEMM fallback silently drops Hopper-without-DeepGEMM onto GEMV.
- **A "kernel is slow" claim must be attributed per-stage before optimizing.** The 147 s looked
  like an attention/paging problem; per-stage profiling proved it was the FP8 GEMM path — the
  attention was already Flash. Survey ≠ evidence; the CUDA-event per-stage breakdown is.
- **DeepGEMM cold JIT ≠ DeepGEMM slow.** First prefill JIT-compiles each shape (~22 s cold);
  warm = 7 s. Set `DG_JIT_CACHE_DIR` to persist the JIT across restarts in production.
- Default the FP8-native path ON (auto-detect sm_90 + vendored), don't hide it behind opt-in —
  prod silently ran the ~20× slower path for want of a flag.
