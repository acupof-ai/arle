# T5b lm_head GEMV → cuBLASLt: −2.8% decode ITL — CUDA, 2026-08-03

> Status: **Shipped, default path** (#196 T5b). c=1 W8A16 decode ITL p50
> **21.37 → 20.77 ms**; cumulative vs pre-#196 baseline **26.88 → 20.77
> (−22.7%)**. Same 32k c=1 protocol; graphed lane (17 captures / 4100+
> replays); greedy byte-identical (argmax stable under the kernel swap).

## What shipped

Two deletions, one reroute:

- `ops::gemv` (the `DeviceVec` lane — lm_head decode logits) now issues an
  N=1 `gemm_cuda` instead of the hand-written kernel: the T4 nsys showed
  `gemv_handwritten_kernel` reading the 1.5 GB lm_head at **~1.1 TB/s**
  (1.40 ms/step) while SGLang's nvjet does the identical shape at 2.2 TB/s
  (0.70 ms).
- The `gemm_small_n_uses_gemv` guard + per-column loop in `gemv.cu` are
  deleted outright — every bf16 N≤4 GEMM now rides cuBLASLt. The kernel
  itself stays for the direct `gemv_cuda` FFI callers (DSv4).

## Learnings

- **The first two "no-op" benches were a routing lesson**: lm_head reaches
  the GPU via `ops::gemv` → `gemv_cuda` directly, NOT through `gemm_cuda`'s
  guard — deleting the guard alone changed nothing, and a byte-identical
  result + flat ITL is exactly what "your patch isn't on the path" looks
  like. Trace the Rust dispatch to the FFI symbol before editing kernels.
- A concurrent pod-tree resync silently dropped an applied-but-uncommitted
  patch mid-cycle (`git status` clean looked like "applied"); the tell was
  `grep` of the source on the pod, not the APPLIED echo.
