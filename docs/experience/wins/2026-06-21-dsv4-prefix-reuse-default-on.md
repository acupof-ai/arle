# DSv4 cross-request prefix KV reuse — default-ON (env tag deleted, pod-verified)

## Context
DSv4-Flash has no radix prefix reuse (`reusable_prefix_blocks=0`). A slot-level
position-0 prefix store was landed default-OFF + WIP (`25940112`) because the
on-path reuse correctness hadn't been cleanly verified (earlier warm-JIT probes
were confounded by cold-start). The physical chain-map work showed the b=1
input-sensitivity lives in the PREFILL (TTFT), and prefix reuse is the direct
lever for shared-prefix prefill — so it was worth verifying + defaulting on.

## What Worked
**Clean correctness + speedup gate** (`prefix_reuse_verify.py`): thorough warmup
FIRST (kills the JIT confound that caused the earlier flip-flopping), then
pure-prefill TTFT (`max_tokens=1`) of a cold-cache prefix vs an in-cache repeat,
plus needle retrieval through the reused KV. Run on the pod, TP4 GPUs 0-3,
**with NO `ARLE_DSV4_PREFIX_CACHE` env set** (proving default-on):

| | result |
|---|---|
| cold-cache prefill (P_new, P_newX) | 0.538 / 0.532 s |
| in-cache reuse (P1, P1#2, P3) | **0.046 s** (×3, identical) |
| **prefill speedup** | **11.65×** |
| needle correct (P1→ZQ7K9X, P3→MB4T2P, no cross-mixing) | **True** |

So the reused whole-slot KV is valid (the secret is retrieved through it, and two
distinct prefixes don't cross-match) and the reuse skips the prefill (11.65×).

**Code change** (`executor.rs`): deleted the `ARLE_DSV4_PREFIX_CACHE` enable env
("the tag"), removed the now-dead `enabled` field + its 4 gate checks (match_len,
take, insert, capture) + the `disabled_store_never_matches_or_stores` test; the
store is always active. Kept `ARLE_DSV4_PREFIX_CACHE_BYTES` as a pure sizing knob
(default 4 GiB). Local `cargo check -p infer-api cuda,no-cuda` clean; pod
`dsv4_fast_build` BUILD_EXIT=0.

## Bench impact
Default-on changes behavior ONLY for shared-prefix requests (cache hit → 11.65×
faster prefill). Non-shared workloads see a cache miss = no-op → the decode
baseline is byte-unchanged (the steady-state TPOT ~25.7 ms/tok is untouched; this
lever is prefill/TTFT-only). No throughput regression path.

## Rule
- A KV-reuse path's correctness gate is **needle retrieval through the reused KV
  + no cross-prefix mixing** (per the project's correct-inference gate, NOT
  token-exact — MoE non-determinism). Run it with the feature's env UNSET to prove
  default-on, not just env-on.
- The earlier "reuse verification confounded" was a **cold-start/JIT artifact**:
  warm the serve THOROUGHLY before the cold-vs-reuse TTFT delta, or the first
  long-prompt request's prefill+JIT swamps the signal (same class as the
  `+666%` decode-TPOT cold artifact).
