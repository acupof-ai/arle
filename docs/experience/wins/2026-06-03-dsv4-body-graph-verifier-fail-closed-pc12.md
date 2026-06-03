# DSv4 Body Graph Verifier Fail-Closed PC12

## Context

Target workload remains DSv4-Flash TP8 + EAGLE, 256K/1500, hot GPU cache:

| Metric | Target |
| --- | ---: |
| TTFT | ~0.44 s |
| TPOT | ~4.85 ms |
| E2E | ~7.7 s |
| Output throughput | ~196 tok/s |

PC11 proved that synthetic decode warmup captured zero DSv4 body graphs because
the dummy slots did not have real compressed/SW/FP8 cache substrate. PC12
followed the real serving path instead of treating warmup as evidence.

## What Worked

The fix separates three cases that were previously conflated:

- Synthetic decode warmup is now tagged with a thread-local scope. Debug logs
  distinguish `synthetic-warmup` from `serving-decode`.
- Explicit `ARLE_DSV4_DECODE_BODY_CUDA_GRAPH=1` can route c=1 decode through
  the batched DSv4 body graph run site. Without that env, the old c=1 per-row
  fallback is preserved.
- Synthetic warmup can no longer write DSv4 body graph warm signatures or graph
  cache entries. A real serving request must perform its own eager warm step
  before capture.
- Batched sparse EAGLE verifier now fail-closes body graph replay by calling
  `force_eager_once()` when `ARLE_DSV4_DECODE_BODY_CUDA_GRAPH=1`. The verifier
  appends speculative target tokens and may roll them back; its body graph
  replay is not graph-safe yet.

This is a correctness/unblocker tranche, not a performance pass.

## Evidence

Local checks:

- `cargo fmt --check`
- `git diff --check`
- `cargo check -p infer --no-default-features --features no-cuda`
- `CUDARC_CUDA_VERSION=12080 cargo check --tests -p infer --no-default-features --features cuda,no-cuda`
- `CUDARC_CUDA_VERSION=12080 cargo check -p infer --no-default-features --features cuda,nccl,no-cuda`

Remote build:

- PASS. `/tmp/dsv4_pc12_verifier_body_graph_gate_build.log` built the
  `release-fast` DSv4 binary with prebuilt CUDA artifacts in 16.1 s.

Root-cause isolation:

- Before synthetic warm isolation,
  `/tmp/dsv4_pc12_n1_body_graph_probe_1780457429` showed serving c=1 finally
  reached the body graph run site, but synthetic warmup had already produced
  `Warmed` signatures for B=1. The real request then went
  `Eager -> Captured -> Replayed` and failed HTTP 500 with
  `CUDA_ERROR_ILLEGAL_ADDRESS` surfacing at `H2D DSv4 start_pos`.
- Control run `/tmp/dsv4_pc12_n1_control_no_batchverifier_1780457624` disabled
  `ARLE_DSV4_SPEC_VERIFY_BATCH`. It returned HTTP 200, generated 32 tokens,
  contained `406`, and had `illegal=0`, but body graph stayed eager. This
  isolated the illegal-address failure to batched verifier body replay, not the
  no-batch verifier route.
- After synthetic warm isolation but before verifier fail-close,
  `/tmp/dsv4_pc12_batchverifier_after_synth_iso_1780457850` proved the fake
  warm signature was gone: real serving ran
  `Eager -> Warmed -> Captured -> Replayed`. It still failed HTTP 500 with
  `illegal=25`, so the remaining root cause was verifier body replay itself,
  not synthetic warm leakage.

Final gate:

- PASS. `/tmp/dsv4_pc12_batchverifier_after_synth_iso_1780458009` ran
  DSv4-Flash TP8, debug-fallback, EAGLE, accepted drafts, batch verifier, FP8
  shared KV, FlashMLA decode, and explicit body graph env for c=1, 32 output
  tokens.
- HTTP 200, `completion_tokens=32`, answer contained `406`, `illegal=0`.
- Body graph debug confirmed fail-close inside serving verifier:
  `run=Eager ready=false force=true` for 8 ranks, then
  `run=Eager ready=true force=true` for 8 ranks; `capture=0`.
- Remote post-probe cleanup showed no active `infer` or timeout processes and
  no compute apps in `nvidia-smi`.

## Rule

Synthetic warmup evidence cannot license serving CUDA graph capture. Warm/cache
state created under dummy slots must not carry into real requests.

Batched speculative verification is also a separate graph-safety contract. It
mutates and rolls back target KV/state, so body graph replay must stay disabled
there until a matched verifier replay test proves HTTP 200, full token budget,
no illegal access, and output correctness. None of these debug-fallback gates
can be compared with the 256K/1500 hot-cache performance target.
