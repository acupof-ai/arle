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

H20 exact receipt build passed on source `4f34f471ece0`:
`BUILD_EXIT=0` in 3m48s. Source digest
`74c0e7530d8b1d3840f1a0a3a5947d1732da702a61d9eae1e4f160e94c6c6fe5`;
product SHA `79f83ab3fd01d4f7c506d812d22d839a8b00a1d2980bb8b90a1d1c93e2c2f087`;
producer and embedded kernel ID both
`bundle:639529e383719a407eb7c2be4090036d263eded4f2601bc33469f9b05e639d38`.
Receipt: `/root/arle-ops/builds/h20-4f34f471/receipt`.

Correctness and throughput remain unmeasured. GPU 1 still held 87,757 MiB under
foreign PID 1269240; it was not signalled. DSv4 was also blocked because the
full eight-GPU topology was not free. No matching Qwen3.6 CUDA champion exists
in `docs/baselines.md`, so the eventual throughput run must re-anchor before a
delta is claimed.

## Rule

A Hopper kernel needs three independent gates: build request, linked implementation marker, and exact runtime device capability. Bench pending-remote: rerun the Qwen3.6 H20 needle gate and the canonical `1,4,8,16` throughput grid against the rolling champion.
