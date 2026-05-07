# 2026-05-07 · M_nsys P1 validated — longctx 4k/c=4 nsys captures full kernel data

## Priority & ROI

**Priority**: P0 diagnostic-infra unblock. Without M_nsys P1 the
remaining long-ctx 4k/c=4 prefill TTFT 800 ms gap (vLLM 1.68×
faster, post-Phase-1A-v3-default-Split) cannot be diagnosed
because nsys 2025.6 silently dropped kernel data on this exact
workload (`5.8 MB / 0 kernel data` in prior `profile_nsys_guidellm.sh`
runs, mirroring SGLang #2776).

**ROI evidence (this entry)**:
- nsys-rep: **894 MB** (vs prior **5.8 MB / 0-kernel** at the same
  shape) — kernel-density silent drop bypassed.
- Kernel summary: 27 distinct kernels with full per-call timing.
- Bench numbers under instrumentation match control within noise
  (TTFT 1988 ms vs 1976 ms control = +0.6%, ITL 19.36 ms vs
  19.4 ms control, out tok/s 153.13 vs 153.83 control).
- Unblocks H_LP3 (per-chunk launch overhead) diagnosis on the
  surviving long-ctx 4k/c=4 hypothesis after codex `c219434`
  license-killed H_LP1 + H_LP2.

**Negative case**:
- 894 MB nsys-rep is large — consumes disk + slow stats parse
  (~30s vs <5s for 5 MB). For per-chunk launch-overhead analysis
  we may want to scope capture window to prefill-only via shorter
  `--max-seconds`. Still acceptable per current cost.
- Profile run wedged twice during dev (PID resolution + /tmp tmpfs
  exhaustion) — both fixed in `28b56d0`. If this script is reused
  on a host with different /tmp size, may regress.

**Kill criteria** (not fired): would have killed if .nsys-rep
< 50 MB or kernel summary < 10 kernels at this workload.

## Goal

Validate that `scripts/profile_nsys_signal.sh` (M_nsys P1 wrapper,
commit `f791425` + fixes `28b56d0`) actually captures kernel data
at the workload that previously produced empty `.nsys-rep` files.

## Bench command

```bash
scripts/profile_nsys_signal.sh longctx-4k-c4-h_lp3 \
  --server-args "--model-path infer/models/Qwen3-4B --port 8000 \
                 --max-seq-len 5120 --num-slots 8" \
  --concurrencies 4 --max-seconds 60 --warmup 10
```

Same shape as `bench-output/2026-05-07-longctx-4k-c4` (the original
P0' default-split bench where ARLE TTFT 1976 ms vs vLLM 1177 ms).

## Results

### Capture sizes

| File | Size | Note |
|---|---:|---|
| `trace.nsys-rep` | **894 MB** | vs prior 5.8 MB at same shape |
| `trace.sqlite` | 3.1 GB | nsys export |
| `cuda_gpu_kern_sum.txt` | 6.2 KB | 27 distinct kernels |
| `cuda_api_sum.txt` | 4.3 KB | full API breakdown |
| `bench-anchor.log` | 10 KB | guidellm metrics |

### Bench under instrumentation (vs control)

| Metric | This run (nsys) | Control (`P0' default split`) | Δ |
|---|---:|---:|---:|
| TTFT p50 | 1988.5 ms | 1976.4 ms | +0.6% |
| TTFT p99 | 2060.7 ms | (n/a) | n/a |
| ITL p50 | 19.36 ms | 19.4 ms | -0.2% |
| out tok/s | 153.13 | 153.83 | -0.5% |
| total tok | 8192 | 8192 | 0% |

nsys overhead = within bench noise. The capture-range=cudaProfilerApi
+ SIGUSR1/USR2 path adds essentially zero overhead vs no-nsys.

### Kernel summary highlights

```text
Time (%)  Total Time (ns)  Instances     Name
   27.9%   17.3 s          417,195       cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2
   22.5%   14.0 s          165,960       cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x1
   18.2%   11.3 s           82,872       decode_attention_fp8_partial_kernel<128>
   14.4%    9.0 s            4,212       cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128
    6.2%    3.8 s            1,116       kernel_kernel
    4.8%    3.0 s              756       cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256
    0.7%    455 ms          82,872       decode_attention_merge_kernel<128>
    0.4%    269 ms           3,924       prefill_attention_paged_qk_norm_rope_hd128_kernel
    0.4%    260 ms          82,872       decode_prep_paged_kernel
    0.1%     63 ms           3,924       prefill_attention_paged_kv_write_hd128_kernel
```

Decode dominates (decode_attention_fp8_partial<128> = 18.2%).
Prefill kernels (qk_norm_rope + kv_write) total **<1% wall time**
across 7,848 instances — at this workload the prefill TTFT gap
is NOT dominated by per-chunk attention launch overhead. (More
precise per-chunk H_LP3 needs a TTFT-only scoped capture; see
follow-up below.)

### CUDA API summary highlights

```text
Time (%)  Total Time   Calls          Name
   70.7%   8.32 s     13,096,325      cuEventQuery     ← F4-Small async readback poll loop
   23.1%   2.72 s          2,302      cuGraphLaunch    ← decode CUDA Graph replay
    1.7%   203 ms         20,084      cudaLaunchKernel
    1.1%   131 ms         16,341      cuMemcpyHtoDAsync_v2
    1.0%   123 ms             13      cuStreamSynchronize  ← only 13 syncs in 60s, F4-Small confirmed
    0.5%    61 ms          7,015      cuMemcpyDtoDAsync_v2
    0.4%    46 ms          8,604      cuLaunchKernel
    0.0%    14 µs              1      cuProfilerStart  ← M_nsys P0+P1 fired
    0.0%    13 µs              1      cuProfilerStop
```

`cuProfilerStart`/`cuProfilerStop` each fired exactly once — the
SIGUSR1/USR2 → `install_cuda_profiler_signal_handlers` (commit
`9b1fb8c`) chain works end-to-end.

`cuStreamSynchronize` = 13 calls / 122 ms total = 0.2% of wall
time. F4-Small async readback path confirmed in production —
no per-step decode sync chain remaining.

`cuEventQuery` = 13 M calls / 70.7% API time. This is the F4-Small
poll loop (`is_event_recorded` poll), expected for async path.

## What it tells us about H_LP3

**At 60s of decode-dominated workload**, prefill kernel time is
< 1% wall. Per-chunk launch overhead is real (3,924 prefill
launches over ~8 prefill cycles = ~490 launches per cycle) but
amortized across the 60s capture.

To validate H_LP3 properly we need a **TTFT-only scoped capture**:
- `--max-seconds 5` (just enough for 1-2 prefill cycles)
- `--warmup 0`
- Compare vLLM trace at same shape

That's the next move; deferred to next /loop tick because GPU is
contended with codex M_b.1 BF16 split-KV bench.

## Cross-references

- M_nsys P0 signal handler: commit `9b1fb8c`
- M_nsys P1 wrapper: commit `f791425` (initial) + `28b56d0` (fixes:
  TMPDIR redirect + PID tree walk + venv PATH)
- Codex H_LP1+H_LP2 license-kill: commit `c219434`
- Long-ctx 4k/c=4 baseline (no nsys): `bench-output/2026-05-07-longctx-4k-c4`
- Capture artifacts: `bench-output/2026-05-07-longctx-4k-c4-h_lp3-profile-nsys-signal/`
- M_nsys plan: [`docs/plans/M_nsys-cuda-profiler-api-integration.md`](../../plans/M_nsys-cuda-profiler-api-integration.md)

## Rule

- **`cudaProfilerApi` capture-range solves the kernel-density drop**.
  Validated on the exact workload (longctx 4k/c=4) where SGLang
  #2776 shows the same upstream bug unresolved since Jan 2025.
  ARLE has working long-ctx GPU traces.
- **TMPDIR redirect is mandatory on hosts with small /tmp**. nsys
  CUPTI injection storage hit 4.8 GB on a 60s capture; default
  /tmp tmpfs (16 GB on this box) gets exhausted by leftover
  sessions. The `28b56d0` fix points TMPDIR at repo-root
  `.nsys-tmp/` (gitignored, on /home).
- **Sub-process resolution under nsys requires a tree walk**, not
  direct PPID — nsys 2025.6 spawns target via a forked launcher
  ([nsys-launcher] `<defunct>` in ps output). `28b56d0` added a
  two-level pgrep walk + pstree fallback.
- **nsys 2025.6 instrumentation overhead is negligible** at the
  longctx 4k/c=4 shape — within bench noise (+0.6% TTFT, -0.5%
  out tok/s). Safe to leave on for diagnostic runs.
