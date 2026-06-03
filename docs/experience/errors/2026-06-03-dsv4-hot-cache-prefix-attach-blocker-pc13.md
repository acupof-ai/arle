# DSv4 Hot-Cache Prefix Attach Blocker PC13

## Context

Target workload remains DSv4-Flash TP8 + EAGLE, single 8-GPU node,
256K/1500, hot GPU-cache hit:

| Metric | Target |
| --- | ---: |
| TTFT | ~0.44 s |
| TPOT | ~4.85 ms |
| E2E | ~7.7 s |
| Output throughput | ~196 tok/s |

PC12 made the accepted-EAGLE + batched-verifier route safe enough to continue
probing, but it did not prove performance. The next question was whether the
current path can even measure the target hot-cache lane.

## Root Cause

The current DSv4 scheduler can publish radix prefix blocks, but DeepSeek V4
does not yet support the GPU prefix reuse contract needed by the target:

- `Deepseek::supports_cross_slot_prefix_attach()` returns `false`.
- `DeepseekState::supports_partial_prefix()` returns `false`.
- DeepSeek has no prefix snapshot/restore implementation for its per-layer
  attention compressor/indexer/SW/FP8 metadata.

Therefore a repeated prompt may produce a radix hit, but the measured request
still falls back to prefill recompute unless the future DSv4 metadata restore
contract is implemented. This kills the "GPU cache hot hit" premise before the
256K/1500 number is meaningful.

## Evidence

Remote DSv4 pod, `/data01/build/arle`, commit `69f336cd`.

Sanity probe harness:

- New helper: `scripts/dsv4_hot_cache_probe.py`.
- Remote temp copy: `/tmp/dsv4_hot_cache_probe.py`.
- It runs one warm request and one measured streaming request against an
  already-started server, then reports TTFT, TPOT, E2E, output throughput, and
  request trace prefix counters.

Short prompt control:

- Artifact: `/tmp/dsv4_hot_probe_sanity_1780459069`.
- Shape: prompt words 128, actual prompt tokens 153, measured output tokens 8.
- Because `short_prompt_bypass_tokens=256`, prefix lookup was bypassed.
- Measured request: TTFT `0.407 s`, TPOT `130.899 ms`, E2E `1.323 s`,
  output throughput `6.045 tok/s`.

Prefix-eligible sanity:

- Artifact: `/tmp/dsv4_hot_probe_sanity512_1780459179`.
- Shape: prompt words 512, actual prompt tokens 537, measured output tokens 8.
- Warm request published the prefix; measured request logged
  `radix_hit=528 reusable_prefix=528 cached_len=537`.
- But measured trace still showed `direct_gpu_attach=false`,
  `lookup_reusable_tokens=528`, `matched_tokens=0`,
  `resume_prefill_tokens=537`, and full prompt prefill work.
- Measured request: TTFT `1.280 s`, TPOT `85.959 ms`, E2E `1.882 s`,
  output throughput `4.251 tok/s`.

Post-run cleanup check showed no active `infer` process and no compute app.

## Fix Direction

Do not run or publish the 256K/1500 target as a hot-cache result until the
measured request trace proves `direct_gpu_attach=true` or an equivalent DSv4
prefix-restore path with near-zero `resume_prefill_tokens`.

The next implementation tranche needs one of:

- a DSv4 cross-slot prefix attach contract that reconstructs or carries
  compressor/indexer/SW/FP8 metadata from persistent structures, or
- a DSv4 prefix snapshot/restore implementation based on the existing
  speculative verifier snapshot code, plus a matched correctness probe.

Only after that should the campaign re-run the 256K/1500 hot-cache EAGLE
target and compare TTFT, TPOT, E2E, and output throughput together.

## Rule

A radix hit is not a hot GPU-cache hit. For DSv4, the report must include the
request trace prefix fields. `lookup_reusable_tokens>0` with
`direct_gpu_attach=false` is still a prefill-recompute run and cannot be
compared with the 256K/1500 hot-cache target.
