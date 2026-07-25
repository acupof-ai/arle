# Qwen3.6 rotary and FA3 gates fail closed

> Status: pending-remote — H20 correctness and throughput passed; formal SM80/86/89/90 qualification remains.

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

H20 single-GPU verification passed on source `6733d6ba768c` with product SHA
`880a5889c1891b136de7e3a7bc4d3a076e7a954a64e323ac6c3dc4cbf6ebca0c`
and producer/embedded/runtime kernel ID
`bundle:0ec5f02d6fdb6798623cb829dbc749ea372c4e47eb411c2b33ecc4e492b3e985`.
The strict `115,300,446,2000,8000 × 3` needle gate returned **15/15 exact,
DET**. Evidence:
`/host/arle-evidence/q36sg-strict-20260719T041545Z/`.

The canonical 120-second `1,4,8,16` grid completed 278/278 requests with zero
errors, timeouts, incomplete responses, or correctness failures:

| c | complete | out tok/s | total tok/s | TTFT p50/p99 ms | ITL p50/p99 ms |
|---:|---:|---:|---:|---|---|
| 1 | 38 | 51.6 | 1190.0 | 454.1 / 1232.6 | 14.481 / 14.664 |
| 4 | 56 | 84.8 | 1672.1 | 7783.8 / 10203.1 | 0.0138 / 29.786 |
| 8 | 88 | 113.6 | 2646.1 | 10732.2 / 12729.7 | 0.0135 / 24.953 |
| 16 | 96 | 124.2 | 2893.2 | 20195.2 / 26754.2 | 0.0135 / 38.063 |

This is an absolute warm-prefix anchor, not a delta: the checked eight-prompt
workload reused prefixes in one server lifecycle, raising prefix hits from 79.5%
to 97.1%, and `docs/baselines.md` has no matching Qwen3.6 single-H20 champion.
Raw evidence:
`/host/arle-evidence/q36sg-bench-20260719T050500Z/`.

Only receipt-owned process groups were stopped. The selected GPUs returned to
0 MiB; foreign holders on GPUs 1 and 4 were unchanged and were not signalled.
DSv4 TP=4/EP=4 was subsequently verified on free GPUs `3,5,6,7`; the current
champion row lives in [baselines.md](../../baselines.md).
Formal publication still requires the same candidate's physical SM80, SM86,
SM89, and SM90 fragments.

## Rule

A Hopper kernel needs three independent gates: build request, linked implementation marker, and exact runtime device capability. A repeated-prompt grid is a warm-prefix anchor; cache-sensitive qualification needs unique prompts or a declared cold restart.
