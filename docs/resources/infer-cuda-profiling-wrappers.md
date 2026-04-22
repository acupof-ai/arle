# Infer CUDA Profiling Wrappers

Last updated: 2026-04-22

This page is the operator-facing guide for the repo-owned CUDA profiling
entrypoints:

- [`scripts/profile_nsys_guidellm.sh`](../../scripts/profile_nsys_guidellm.sh)
- [`scripts/profile_ncu_guidellm.sh`](../../scripts/profile_ncu_guidellm.sh)

They sit on top of the existing
[`scripts/bench_guidellm.sh`](../../scripts/bench_guidellm.sh) workflow, so
every deep CUDA capture stays bench-anchored instead of becoming an orphan
trace.

## What These Wrappers Standardize

- output directories under `bench-output/`
- an explicit bench anchor
- sidecar metadata files
- profiler flags that are easy to forget
- one short markdown summary per capture

They do not replace the bench/report process in
[`docs/bench-and-trace-spec.md`](../bench-and-trace-spec.md). If the capture
is verification-grade, write the paired
`docs/experience/wins/YYYY-MM-DD-profile-<backend>-<model>-<what>.md` entry.

## Prerequisites

- infer server already running on the target URL
- `guidellm`, `curl`, `jq`, `python3`, `lsof`
- `nsys` for timeline captures
- `ncu` for kernel captures

The wrappers resolve the server PID from the `--target` port via `lsof`.
Pass `--server-pid` when auto-discovery is not appropriate.

## Layer 2: Timeline

Use `Nsight Systems` when the question is:
"Across CPU threads, CUDA API, copies, and kernels, where does the time go?"

Default entrypoint:

```bash
scripts/profile_nsys_guidellm.sh <label> --target http://127.0.0.1:8000
```

Default behavior:

- attach to the existing infer server
- drive a short `bench_guidellm.sh --fast` load as the bench anchor
- capture a short steady-state CUDA timeline
- export:
  - `.nsys-rep`
  - `.sqlite`
  - `cuda_gpu_kern_sum.txt`
  - `cuda_api_sum.txt`
  - `summary.md`

Common patterns:

```bash
# Slightly wider steady-state window.
scripts/profile_nsys_guidellm.sh cuda-qwen3 \
  --target http://127.0.0.1:8000 \
  --delay-seconds 8 \
  --duration-seconds 12

# Reuse an existing bench anchor and replay its exact guidellm command.
scripts/profile_nsys_guidellm.sh cuda-qwen3 \
  --bench bench-output/2026-04-22-cuda-qwen3

# Safe contract check on a machine without Nsight installed.
scripts/profile_nsys_guidellm.sh cuda-qwen3 --dry-run
```

Timeline output contract:

- top kernels
- top CUDA APIs
- copy-related API lines
- approximate launches per completion token
- a direct link back to the bench anchor

## Layer 3: Kernel Deep Dive

Use `Nsight Compute` only after the timeline already points at a hotspot.

Default entrypoint:

```bash
scripts/profile_ncu_guidellm.sh <label> --family attention
```

Supported kernel families:

- `attention`
- `sampling`
- `paged-kv`
- `dequant`
- `fused-op`

Examples:

```bash
# Default short load, focused on attention kernels.
scripts/profile_ncu_guidellm.sh cuda-qwen3 \
  --family attention \
  --target http://127.0.0.1:8000

# Explicit regex for a known kernel family.
scripts/profile_ncu_guidellm.sh cuda-qwen3 \
  --kernel 'regex:decode_attention_int8_.*_kernel' \
  --launch-skip 8 \
  --launch-count 2

# Reuse an existing bench anchor.
scripts/profile_ncu_guidellm.sh cuda-qwen3 \
  --family paged-kv \
  --bench bench-output/2026-04-22-cuda-qwen3
```

Kernel output contract:

- `.ncu-rep`
- raw `ncu.log`
- `summary.md`
- kernel selector, launch skip/count, and bench anchor recorded in `env.txt`

The wrapper defaults to `--set full` so the capture contains occupancy,
memory-throughput, stall, and roofline-style sections without engineers
having to rebuild the `ncu` command line from scratch.

## Bench Anchoring Rules

There are two supported modes:

1. Auto-anchor:
   the wrapper runs `bench_guidellm.sh --fast` and uses that load as the
   profiling anchor.
2. Existing anchor:
   pass `--bench bench-output/...`; the wrapper replays the exact guidellm
   command stored in `command.txt` and links the profile back to that bench.

Use auto-anchor for quick diagnosis. Use an existing anchor when the profile
must explain a specific recorded regression.

## Output Layout

`profile_nsys_guidellm.sh` writes:

```text
bench-output/<date>-<label>-profile-nsys[-runN]/
  command.txt
  env.txt
  sha256.txt
  nsys.log
  trace.nsys-rep
  trace.sqlite
  cuda_gpu_kern_sum.txt
  cuda_api_sum.txt
  summary.md
  bench-anchor.log
```

`profile_ncu_guidellm.sh` writes:

```text
bench-output/<date>-<label>-profile-ncu[-runN]/
  command.txt
  env.txt
  sha256.txt
  ncu.log
  trace.ncu-rep
  summary.md
  bench-anchor.log
```

## Notes

- The wrappers are intentionally bench-first, not always-on. They are for
  diagnosis, not background monitoring.
- `--dry-run` is the fastest way to validate PID resolution, output naming,
  and the exact profiler command on a machine that does not have Nsight
  installed.
- Raw `.nsys-rep` / `.ncu-rep` captures stay in `bench-output/`; do not
  commit them.
