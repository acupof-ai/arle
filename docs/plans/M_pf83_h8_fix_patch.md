# M_pf83 H8 fix — proposed 1-line patch (clear sticky cudaGetLastError at wrapper entry)

> Per `c9abe8e` H8 hypothesis: `gemm_w4_fp8_marlin_cuda` returns code 2
> because `cudaGetLastError()` at the wrapper end (line 250-256) reads
> a sticky error from a PRIOR CUDA call, not from this kernel.
> Quick repro test + 1-line fix proposed below.

## §1 Repro test (5 min codex pickup)

Add at top of `gemm_w4_fp8_marlin_cuda` body (after line 138):

```cpp
extern "C" int gemm_w4_fp8_marlin_cuda(...) {
+ // H8 diagnostic: log + clear any pre-existing sticky CUDA error
+ {
+   cudaError_t prev_err = cudaGetLastError();
+   if (prev_err != cudaSuccess) {
+     fprintf(stderr, "[gemm_w4_fp8_marlin_cuda] cleared pre-existing CUDA error: %d (%s)\n",
+             prev_err, cudaGetErrorString(prev_err));
+   }
+ }
  if (prob_m == 0 || prob_n == 0 || prob_k == 0) {
    return 0;
  }
  ...
```

Build + run:
```bash
cd /home/ckl/projects/arle
CUDA_HOME=/opt/cuda cargo build --release -p infer --features cuda
RUST_MIN_STACK=33554432 INFER_HYBRID_W4A8_PREFILL=1 \
  INFER_MARLIN_W4_FP8_PREFILL=1 \
  target/release/infer --model-path infer/models/Qwen3-4B-W4-hybrid-zpfix \
  --port 8000 \
  > /tmp/pf83-h8-test.log 2>&1 &
sleep 30
curl -s -X POST http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3-4B-W4-hybrid-zpfix","prompt":"Hello","max_tokens":3,"temperature":0,"stream":false}'

# Check stderr for the H8 diagnostic message
grep "cleared pre-existing CUDA error" /tmp/pf83-h8-test.log

pkill -f "target/release/infer.*--port 8000"
```

### Interpretation

| Diagnostic fires? | error code in message | Conclusion |
|-------------------|----------------------|------------|
| YES every call | code 2 | **H8 CONFIRMED**: prior call leaves sticky OOM, surfaces here |
| YES some calls | varies | Sticky errors leak from some paths, not all |
| NO never | n/a | **H8 DISPROVEN**: kernel itself returns code 2; investigate H1'/H2/H6 |

## §2 Fix (if H8 confirmed) — also 1 line

Two options:

### Option A — clear at entry (defensive)

Keep the diagnostic clear at function entry. This prevents leaked errors
from prior calls from being mis-attributed to this kernel.

But this masks the underlying bug — somewhere ELSE in the codebase a
kernel is leaking errors that should be handled.

### Option B — synchronize before checking last error (defensive + diagnostic)

```cpp
- cudaError_t err = cudaGetLastError();
- if (err != cudaSuccess) {
+ cudaError_t err = cudaDeviceSynchronize();
+ cudaError_t kernel_err = cudaGetLastError();  // post-sync, fresh check
+ if (err != cudaSuccess || kernel_err != cudaSuccess) {
+   err = (err != cudaSuccess) ? err : kernel_err;
    return static_cast<int>(err);
  }
```

This:
1. Synchronizes the stream (forces all in-flight kernels to complete)
2. Returns sync error (most recent kernel) if any
3. Falls back to last error code

Better diagnostic + still functional. Slightly more expensive (sync per
call) — only acceptable if we're already in error path.

### Option C — find the LEAKING caller (real fix)

Use `cudaPeekAtLastError()` + `CUDA_LAUNCH_BLOCKING=1` to identify which
EARLIER kernel is leaving the sticky error. Fix THAT kernel's error
handling. This is the "right" fix but requires bisecting the call chain.

## §3 Recommended sequence

1. Apply repro test (Option A diagnostic) → confirm/disprove H8 (5 min)
2. If H8 confirmed: apply Option B short-term + Option C long-term
3. If H8 disproven: pivot to H1' static-scratch refactor per
   `cd7732a` §7

## §4 Cross-references

- `c9abe8e` H8 hypothesis introduction
- `cd7732a` H1' refined hypothesis (per-call alloc pattern)
- `2472e8a` initial 5-hypothesis ranking
- `0cde63d` PF8.3 RUNTIME KILL evidence (101380 failures)
- `marlin_w4_fp8_kernel.cu` lines 138 (entry), 250-256 (current
  cudaGetLastError check)

## §5 Estimated total fix effort (if H8 confirmed)

- Repro test apply + verify: 10 min codex
- Option B fix apply + cargo check: 10 min
- Test bench v11 with --rate "1,2,4" --max-seconds 30: 5 min
- Per skill v1.11.0+ #29: PAIR with greedy_consistency at conc=2,4
  to confirm fix vs single-request lucky-PASS
- Total: ~40 min for codex pickup → PF8.5 license-decision-grade
  numbers

## §6 H8 build timing observation (2026-05-10 08:12 update)

cargo build --release with H8 diagnostic patch (81672c3) takes longer
than initial bench v3-v10 builds because:
- Adding `#include <stdio.h>` to marlin_w4_fp8_kernel.cu invalidates
  the file's incremental compile cache
- nvcc re-compiles the file from scratch including marlin_pf8/marlin_template.h
  (2081 LOC of templates)
- Total expected: 5-10 min for this re-compile

cargo PID 1950646 started 08:08, binary mtime still 07:12 at 08:12 (~4
min in). ETA build complete: 08:14-15. Run pf83_h8_verify.sh after.
