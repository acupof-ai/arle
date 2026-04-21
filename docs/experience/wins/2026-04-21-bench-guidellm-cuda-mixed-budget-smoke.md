# CUDA c16 mixed-budget smoke — `Qwen/Qwen3-4B`, 2026-04-21

## Goal

- Verify that splitting pure-prefill budget from decode-active mixed resident
  capacity raises effective KV capacity and fixes the c16 long-context
  underfill shape.

## Hypothesis

- If the scheduler stops reserving a `max_prefill_tokens=16384` sized mixed
  resident buffer on every decode-active step, `TokenKVPool` capacity should
  rise materially and the same c16 / 4096-in workload should lift `peak active`
  and `peak running_batch`.

## Params

- Wrapper: `scripts/bench_guidellm.sh`
- Mode: exploration smoke (`--fast`)
- Profile: `concurrent`
- Concurrency: `16`
- Data: `prompt_tokens=4096,output_tokens=256`
- Duration: `30s`
- Trace polling: `500ms`

## Env

- **Backend:** CUDA
- **Model:** `Qwen/Qwen3-4B`
- **Hardware:** NVIDIA L4 24GB
- **Commit base:** `772098f` + local mixed-budget scheduler changes
- **Server launch:** `./target/release/infer --model-path Qwen/Qwen3-4B --port 8000 --num-slots 16 --max-seq-len 4608 --enable-mixed-chunk=true --chunked-prefill-size 4096 --max-prefill-tokens 16384`
- **Non-default flags:** `--num-slots 16 --max-seq-len 4608 --enable-mixed-chunk=true --chunked-prefill-size 4096 --max-prefill-tokens 16384`

## Results

- Startup budget changed from `TokenKVPool: 24352 max tokens` to
  `TokenKVPool: 34672 max tokens`.
- Mixed resident workspace now initializes with
  `mixed_prefill_tokens=4096, prefill_budget_tokens=16384`.
- Service trace shape improved from an underfilled active set to a stable
  `active=8` plateau.

### Headline table

| run | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual |
|---|---|---|---|---|---|---|
| before (`2026-04-21-cuda-l4-mixed-width-probe`) | 12806.1 | 24148.1 | 38.67 | 38.73 | 47.1 | 0.133 |
| after (`2026-04-21-cuda-l4-mixed-budget-smoke`) | 4128.8 | 5661.1 | 62.16 | 111.64 | 66.68 | 0.233 |

### Service trace

| run | peak waiting | peak active | peak running_batch | peak prefill_queue | peak kv_util |
|---|---|---|---|---|---|
| before | 14 | 2 | 2 | 1 | 97.4% |
| after | 14 | 8 | 8 | 4 | 100.0% |

## Problems

- ITL regressed once the active set filled out. This is expected for the smoke:
  the system is finally trading more decode bandwidth across more live
  requests instead of serializing them.
- Canonical sweep was started after the smoke but interrupted by the user, so
  this entry intentionally records the completed smoke A/B only.

## Learnings

- The dominant c16 gap was not a remaining queue-ordering bug. It was
  persistent mixed resident memory being sized from the pure-prefill batch
  budget.
- On L4 with Qwen3-4B BF16 KV, recovering roughly `10k+` token capacity was
  enough to move the workload from `active≈2`/`5` behavior to a stable
  `active≈8` plateau.
- A clean split between:
  - pure-prefill batch budget
  - decode-active mixed resident capacity
  is a better approximation of SGLang's "static KV pool + runtime headroom"
  model than treating both as one number.

## Artefacts

- Raw (before): `bench-output/2026-04-21-2026-04-21-cuda-l4-mixed-width-probe/benchmarks.json`
- Trace summary (before): `bench-output/2026-04-21-2026-04-21-cuda-l4-mixed-width-probe/service_stats_trace_summary.md`
- Raw (after): `bench-output/2026-04-21-2026-04-21-cuda-l4-mixed-budget-smoke/benchmarks.json`
- Trace summary (after): `bench-output/2026-04-21-2026-04-21-cuda-l4-mixed-budget-smoke/service_stats_trace_summary.md`

## Notes

- Code change since the previous smoke: split scheduler/runtime accounting so
  pure prefill still uses `max_prefill_tokens`, while decode-active mixed uses
  a smaller persistent `mixed_prefill_token_budget`.
- Next step: rerun a clean canonical sweep and land a canonical wins entry once
  the service is left uninterrupted for the full run.
