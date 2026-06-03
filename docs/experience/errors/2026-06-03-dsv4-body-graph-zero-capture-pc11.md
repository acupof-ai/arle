# DSv4 Body Graph Zero Capture PC11

## Context

Target workload remains DSv4-Flash TP8 + EAGLE, 256K/1500, hot GPU cache:
TTFT ~0.44 s, TPOT ~4.85 ms, E2E ~7.7 s, output throughput ~196 tok/s.

PC9 made the target EAGLE route fail closed unless internal-MTP accepted
drafts are enabled. PC10 proved a small debug-fallback functional gate with
accepted drafts. The next target blocker was full-decode CUDA graph replay:
without body graph capture/replay, any debug-fallback or eager verifier run is
only a correctness/unblocker probe, not a performance comparison.

## Root Cause

The DSv4 body graph path is enabled by env, but the current startup warmup is
synthetic. `warmup_cuda_graphs()` allocates dummy one-token slots and calls
decode over batch sizes up to `cuda_graph_max_bs`; it does not run a real DSv4
prefill first.

That means the model reaches the DSv4 body-graph readiness check before
per-slot attention substrate exists:

- With FlashMLA decode enabled, readiness rejects on missing FP8 shared-KV SW
  bootstrap.
- With FlashMLA decode disabled, readiness rejects on missing compressed cache.
- Enabling batched EAGLE verification removes one eager verifier blocker, but
  it does not make synthetic warmup materialize DSv4 cache metadata.

This licenses only a narrow conclusion: current DSv4 body graph warmup captures
zero graphs because the synthetic warmup lacks the cache substrate that the
body graph readiness contract requires. It does not license a performance win,
and it does not prove the final 256K/1500 hot-cache root cause is complete.

## Evidence

Remote DSv4 pod, `/data01/build/arle`, commit `536d5168`.

Probe 1: `/tmp/dsv4_pc11_body_graph_b5_eagle_1780455603`.

- Env included `ARLE_DSV4_DECODE_BODY_CUDA_GRAPH=1`,
  `ARLE_DSV4_DECODE_BODY_CUDA_GRAPH_MAX_BS=5`,
  `ARLE_DSV4_NCCL_GRAPH_CAPTURE=1`, FlashMLA decode, and accepted EAGLE
  drafts.
- Request served successfully: HTTP 200, request trace error `null`, 8
  completion tokens, TTFT ~392 ms, total ~1323 ms.
- Spec metrics: draft 15, verified 15, accepted 4, acceptance rate 0.266667.
- Body graph evidence:
  - `body_capture_b5=0`
  - `body_capture_b4=0`
  - `body_rejects=120`
  - `body_replay_errors=0`
  - `illegal_address=0`
- Reject aggregation included rows 0-3 rejecting on
  `FP8 shared KV SW cache not bootstrapped`, plus batch 6-16 exceeding max
  graph batch 5.

Probe 2: `/tmp/dsv4_pc11_body_graph_b16_eagle_c1_1780455709`.

- Same route, with `ARLE_DSV4_DECODE_BODY_CUDA_GRAPH_MAX_BS=16`, c=1,
  `max_tokens=32`.
- Request served successfully: request trace error `null`, 32 completion
  tokens, TTFT ~380 ms, total ~4232 ms, output throughput ~7.56 tok/s.
- Spec metrics: draft 74, verified 74, accepted 16, acceptance rate 0.216216.
- Body graph evidence:
  - `body_capture_total=0`
  - `body_capture_b1=0`
  - `body_capture_b5=0`
  - `body_capture_b16=0`
  - `body_rejects=128`
  - `body_replay_errors=0`
  - `illegal_address=0`
- Reject aggregation showed 8 rejects each for warmup slots 0-15:
  `FP8 shared KV SW cache not bootstrapped`.

Probe 3: `/tmp/dsv4_pc11_body_graph_no_flashmla_eagle_c1_1780455943`.

- Same route as probe 2, but with `ARLE_DSV4_FLASHMLA_DECODE=0`, c=1,
  `max_tokens=16`.
- Request served successfully: HTTP 200, 16 completion tokens, answer contained
  `406`.
- Spec metrics: draft 40, verified 40, accepted 6, acceptance rate 0.15.
- Body graph evidence:
  - `body_capture_total=0`
  - `body_capture_b1=0`
  - `body_capture_b16=0`
  - `body_rejects=120`
  - `body_replay_errors=0`
  - `illegal_address=0`
- Reject aggregation changed to `missing compressed cache`.

Probe 4: `/tmp/dsv4_pc11_spec_verify_batch_body_graph_1780456027`.

- Same as probe 3, plus `ARLE_DSV4_SPEC_VERIFY_BATCH=1`.
- Request served successfully: HTTP 200, 16 completion tokens, answer contained
  `406`.
- Spec metrics: draft 36, verified 36, accepted 8, acceptance rate 0.222222.
- Body graph evidence:
  - `body_capture_total=0`
  - `body_capture_b1=0`
  - `body_capture_b6=0`
  - `body_rejects=120`
  - `body_replay_errors=0`
  - `illegal_address=0`
- Reject aggregation still showed `missing compressed cache`.

Post-probe process checks showed no lingering `infer` or `timeout` compute
processes, and remote HEAD stayed at `536d5168`.

## Fix Direction

Do not flip full-decode graph support on this evidence.

The next fix must make the body graph capture path use real DSv4 cache
substrate, or fail closed with a precise startup/metric reason such as
`body_graph_warmup_synthetic_cache_incomplete`. Viable next steps are:

- Add DSv4-specific body graph diagnostics that distinguish synthetic warmup
  rejects from real-request capture/replay attempts.
- Add a real DSv4 graph warmup path that materializes compressed/indexer/SW/FP8
  cache state before capture.
- Keep `ARLE_DSV4_SPEC_VERIFY_BATCH=1` as a required graph-capable EAGLE
  verifier condition, but do not treat it as sufficient for body graph capture.

## Rule

`body_graph_enabled=true` is not evidence that DSv4 captured a body graph. The
report must include capture/replay counters and the reject reason. For the
256K/1500 hot-cache target, debug-fallback or eager verifier runs remain
unblocker evidence only until TTFT, TPOT, E2E, output throughput, and EAGLE
acceptance are measured together on the target route.
