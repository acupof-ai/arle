# A zero-grid availability probe, mistaken for a long-prompt crash — CUDA, 2026-08-13

> Status: The zero-grid launch is explained and removed. The original engine
> death is NOT reproducible on the current build and its cause is unknown.

## Context

A user report: a 64k-token system prompt plus 38 tool definitions killed the
engine. Serving ThinkingCap-Qwen3.6-27B-FP8, CUDA, paged KV. The failure looked
length-dependent:

```
ERROR infer_server::execution: infer-server engine step failed:
      DriverError(CUDA_ERROR_UNKNOWN, "unknown error")
```

The process then `exit(75)`s, so every later request returns `engine thread
closed; cannot submit` — one prompt takes the whole server down.

## Phenomenon

`compute-sanitizer` reported no invalid memory access; the failure was
`cuLaunchKernel` returning `CUDA_ERROR_INVALID_VALUE`. An `LD_PRELOAD`
interposer on `cuLaunchKernel` captured the configuration:

```
[LAUNCHFAIL] rc=1 grid=(0,1,1) block=(128,1,1) shmem=24832
[LAUNCHFAIL]   arg4(i32)=0     # num_tokens
```

`gridDim.x == 0`. `shmem=24832` matches one AOT wrapper,
`flashqla_gdr_cumsum_h48_sm90`, whose generated host code computes
`grid_x = ceildiv_i32(seq_len, 64)`, which is 0 at `seq_len == 0`.

This launch is real, but it is **deliberate and benign**. `qwen35.rs`
`fq_kernels_available` probed for stub builds by calling the cumsum wrapper with
`seq_len = 0` and reading the return code — a `OnceLock`, fired once per
process, at the first GDN chunked prefill.

## Root Cause

Two separate findings.

**The zero-grid launch is the probe, not a crash trigger.** Three reasons it
cannot explain the reported failure:

1. The probe is one-shot and length-independent. The 27964-token request, which
   succeeds, fires it too.
2. A launch-configuration error returns synchronously from `cuLaunchKernel` and
   does not poison the context, so it cannot resurface as `CUDA_ERROR_UNKNOWN`
   at a later sync.
3. A `Backtrace::force_capture()` guard placed at the single GDN dispatch
   (`advance_linear_conv_gdr`) never fired — not on any passing request, and not
   on the run that did kill the process.

**The engine death is unreproducible on the current build.** Measured on a
private tree, sm_90, BF16 KV, empty `DG_JIT_CACHE_DIR`, no DSpark, free GPU:

| prompt tokens | concurrency | result |
|---|---|---|
| 33554 | 1 | 200 |
| 61992 | 1 | 200 |
| 61992 | 4 | 200 ×4, server alive |

The serve command in the original crash log carried
`--spec-type dspark --mtp-draft-model /root/dspark-fr-native`. DSpark was never
ruled out at 33554 — only at 22374 — so the earlier "independent of DSpark"
claim was unsupported. That draft checkpoint no longer exists, so the DSpark
configuration has not been retested.

One later 62k run did kill the process, but with a different signature: no
`prefix-lookup` log line, no `ERROR`, no `exit(75)`, no `dmesg` segfault. It
coincided with another job taking the GPU and with a concurrent sync rewriting
the build tree. Silent disappearance with no CUDA error is consistent with an
external kill, not with this code path.

## Fix

`crates/infer-cuda/src/qwen35.rs` — the probe now checks the two conditions the
AOT dispatch wrapper itself switches on, without launching anything:

```rust
let ok = cuda_kernels::KERNEL_CAPABILITIES.split(',').any(|c| c == "flashqla")
    && ctx.compute_capability() == (9, 0);
```

`KERNEL_CAPABILITIES` gains `flashqla` in `build.rs` when an sm_90 target is
present — the same condition that emits the wrapper — and the flashqla rows are
emitted for sm_90 only. Verified on the pod: no
`FlashQLA chunked GDR unavailable` warning, so the chunked path stays enabled.
The probe no longer needs a bound driver context, and compute-sanitizer no
longer reports an illegal launch on every run.

The `seq_len == 0` guard added to `advance_linear_conv_gdr` was removed. It
caught nothing and encoded a wrong theory.

Scope note: the claim that AOT wrappers generally lack zero guards was wrong.
Exactly one wrapper derives a grid dimension from a count.

## Rule

**A launch failure found by tracing is not the failure under investigation.**
The `LD_PRELOAD` trace answered "what launch fails" correctly and the answer was
a deliberate probe. The question that needed asking — "does this failure
correlate with the input that crashes?" — was never asked. A one-shot event
cannot explain a length-dependent one.

**A reproducing trigger is not a root cause.** "Zero tokens reach the GDN cumsum"
was measured; "a snapshot cut produces them" was invented to explain it. Both
that and the planner-alignment story were falsified in under a minute by
replaying the real arithmetic (last chunk of 33554: `start=32768 end=33554`,
cuts `[32768, 33552]`, forward 784, tail 2 — no zero-length segment anywhere).

**Re-verify the premise on a clean binary before building a chain on it.** The
33554 boundary came from a serve whose flags included DSpark and whose build
tree was stale. On a clean build the same prompt passes, and so does 62k at c=4.
Hours went into explaining a boundary that does not exist in the current tree.

**Do not probe for capability by performing an illegal operation.** The
information was already available from the build manifest and the device
properties. Probing by launching left a permanent false positive in every
sanitizer run, which is how it ended up misread as a bug.

**A single request must never be able to kill the engine.** Whatever the
original cause, an over-length prompt cleared admission and took every in-flight
request down. That containment gap is independent and still open.

## Environment notes

- The shared container tree `/host/arle-build` is synced concurrently by other
  sessions. `tools/tilelang/` is untracked, so each sync deletes it, and
  `build.rs:918` reads it for the kernel identity hash — every build there fails
  until it is restored from `/host/nvme0/arle_k/tools`. A sync landing mid-build
  also deletes the build's cwd (`sh: 0: getcwd() failed`). Gate work from a
  private copy.
- The node tree carried `csrc/attention/decode_attention_varlen_int8.cu`,
  deleted from the repo in `64be73980`, with corrupted content
  (`__nv_fp8_e4m32float`). Every CUDA build there failed at `build.rs:2490`
  regardless of the change under test.
