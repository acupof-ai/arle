# `arle kernel` — per-kernel runner with a CPU f32 reference (W3), GPU run pending

Date: 2026-09-09. Lane `d`. PR #251 (commit `7373bead8`).

Status: **pending-remote** — the host half (argument parse, registry, CPU
reference, output contract) is merged and exercised on Mac; the device half
needs one CUDA release build on the pod. No kernel number has been recorded.

## Context

Per-kernel iteration previously required either editing an example or
reading nsys. `arle kernel <name> --shape M,N,K --ref cpu`
(`crates/infer-cuda/src/kernel_bench.rs`) registers real tensors, runs one
named kernel, and prints device time and max relative error against a CPU f32
reference that dequantizes and accumulates in f32. The registry is an
explicit table; an unregistered kernel surfaces by absence (`unknown kernel
…; registered: …`). Two kernels are registered: `fp4-gemv` (the dense NVFP4
GEMV with no production caller since the dispatch convergence left dense M=1
on Marlin) and `marlin-fp4-gemm` (the serving arm). The runner prints
`kernel= shape= time_ms= max_rel= status=PASS/FAIL` with
`PASS_MAX_REL=1e-2`; `fp4-gemv` is M=1-only.

## Remote commands (H20 pod)

Build a release binary with the bench-valid receipt (bench-labelled pod
runs reject non-release builds):

```bash
bash scripts/pod.sh build kab --release --features cuda --bin arle
```

Run both M=1 comparisons (the wrapper invokes
`arle kernel {fp4-gemv,marlin-fp4-gemm} --shape 1,34816,5120 --ref cpu
--iters 100` on the pod):

```bash
bash scripts/kernel_ab_fp4.sh kab
```

Parameters: shape `1,34816,5120` is M=1 decode against hidden 5120 /
intermediate 17408 / 64 layers; `--iters 100` for the mean device time.
M=1 is the default; Marlin really runs at M=1 (no M=1 bail, unlike gemv).

Closure: this W3 task closes when both `kernel=` lines print time_ms and
max_rel with `status=PASS`. The trailing `verdict:` line settles the open
`marlin-m1-gemv` question — whether the killed dedicated GEMV beats the
Marlin tensor-core GEMM at decode M=1; "M=1 goes to marlin" then holds by
construction.

## Net

Host half: CLI wiring, registry, and CPU reference merged in #251; no
serving-path change (dev tooling, exempt from the bench-entry rule per the
commit body). The pending item is the two pod commands above, not a code
change.
