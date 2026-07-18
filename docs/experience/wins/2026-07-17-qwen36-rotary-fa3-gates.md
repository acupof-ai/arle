# Qwen3.6 rotary and FA3 gates fail closed

> Status: pending-remote — clean H20 build passed; GPU 1 correctness and throughput remain blocked by a foreign serve.

## Context

Qwen3.6 configs carry HD256 partial RoPE plus `mrope_section`; FA3 is Hopper-only, while T1 binaries also target Ampere and Ada.

## What Worked

- Derive `rotary_dim` without float truncation and validate text-only mRoPE sections against `rotary_dim / 2`.
- Reject invalid HD256 paged and fused-attention C ABI arguments before launch.
- Build real FA3 only when requested, vendored, and the target set includes SM90.
- Select FA3 only when the linked marker is real and `DeviceContext` reports exact compute capability 9.0.
- Enable FA3 in canonical T1, H20, and Docker builds; keep SM70 and Blackwell lanes explicitly off.

Local gates now cover qwen35-spec, Mac CUDA/no-CUDA type checks, prebuilt export,
strict lever summaries, candidate qualification, release validation, and
receipt-bound pod flow. The five shell contract tests run in no-GPU Linux CI and
in the pre-push snapshot before Rust compilation.

The artifact path now generates one candidate, proves cold consumption without
TileLang, binds per-GPU evidence to the exact candidate/kernel/product identity,
aggregates the required profiles, and adds qualification as a sidecar without
changing payload bytes. Pod sync/build/run are source- and receipt-bound; kill
refuses stale or foreign process identities.

H20 clean build passed on source `925fd69b7512` plus the local diff
(`LOCAL_STATE_SHA256=2965976e9c2e95c0b6e2ba52222e71487c6784e334eafacec1da65e98e0f8051`):
`BUILD_EXIT=0`, 345 crates, 8m41s. TileLang AOT generated all SM90 kernels from
`/host/tilelang-preserve/.venv`; the dispatch objects export unmangled GDR and
HD256 attention C symbols. Embedded kernel ID:
`bundle:0f6510c6c3b9e343d2afcc320608a7eb965152d61f80dddbfeefa2bbfaa59acf`.

Correctness and throughput are not measured. GPU 1 remained at 87,757 MiB for
the full 60-minute wait, owned by foreign PID 1269240 (`Qwen3-0.6B` serve); it
was not killed. The earlier needle log is invalid because the server rejected a
stale `--num-slots` flag and never became ready.

## Rule

A Hopper kernel needs three independent gates: build request, linked implementation marker, and exact runtime device capability. Bench pending-remote: rerun the Qwen3.6 H20 needle gate and the canonical `1,4,8,16` throughput grid against the rolling champion.
