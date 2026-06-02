# DSv4 DeepGEMM JIT Cross-Process Lock

## Context

Target workload remains DSv4-Flash TP8 + EAGLE, 256K/1500, hot GPU cache:
TTFT ~0.44s, TPOT ~4.85ms, E2E ~7.7s, output throughput ~196 tok/s.

During the TP8 debug-fallback EAGLE smoke for the batched attention-half change,
the short decode request completed with real output, but the next 1K prefill
request failed before producing tokens:

```text
ptxas fatal   : Output file '/root/.deep_gemm/tmp/arle-.../kernel.cubin' could not be opened
DeepSeek V4 DeepGEMM w13 GEMM failed: DriverError(CUDA_ERROR_UNKNOWN, "unknown error")
```

This is a startup/runtime correctness blocker for any DSv4 path that relies on
DeepGEMM. A 256K prefill target cannot depend on eight ranks racing to JIT the
same cubin.

## What Worked

The ARLE native DeepGEMM bridge used a process-local mutex, but TP8 multiproc
serving runs eight processes. Each rank could choose the same digest and compile
into the same temporary directory:

```text
$DG_JIT_CACHE_DIR/tmp/arle-<digest>/kernel.cubin
```

The fix keeps the hot cache path unchanged and only serializes first-time JIT
publication for a given kernel digest:

- Add a per-kernel `flock` file under `$DG_JIT_CACHE_DIR/locks/`.
- Re-check the final `cache/kernel.<name>.<digest>/kernel.cubin` under that
  cross-process lock.
- Compile into a pid/time/counter-unique temporary directory.
- Publish by directory rename, and fail explicitly if the cache cannot be
  published.

This preserves concurrent compilation for distinct kernel digests while
removing the same-kernel temp-file race.

## Verification

Local checks:

- `cargo fmt --check`
- `cargo check -p infer --no-default-features --features no-cuda`
- `CUDARC_CUDA_VERSION=12080 cargo check -p infer --no-default-features --features cuda,nccl,no-cuda`
- `git diff --check`

Remote verification is pending:

- release-fast CUDA rebuild on `/data01/build/arle`.
- Repeat TP8 debug-fallback EAGLE smoke with MTP weights loaded.
- Confirm the 1K prefill no longer fails with the `kernel.cubin` ptxas output
  race.
- Confirm no `infer` process or GPU compute app remains after the smoke.

## Rule

Runtime JIT caches used by multiproc serving need cross-process publication
locks, not only in-process mutexes. Otherwise the first real request can fail
even when the CUDA source compiled successfully at build time.
