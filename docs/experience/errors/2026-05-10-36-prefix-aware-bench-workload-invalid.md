# #36 PrefixAwareAdmission Bench A/B - workload invalid despite gate firing

## Context

#36 PrefixAwareAdmission substrate was already present on `main`. Codex added
the missing observability counter in `079639c feat(scheduler): expose
PrefixAware admission deferrals`, then ran a manual server-restart A/B because
`scripts/bench_guidellm.sh` only probes an already-running server and does not
apply server flags passed after `--`.

The goal was to validate whether `--admission-policy prefix-aware` closes the
multi-tenant TTFT gap by preserving warm prefix-cache benefit under cold-request
pressure.

## Hypothesis

With a low cold soft cap, PrefixAware admission should defer cold requests while
allowing warm/prefix-hit requests through. The treatment arm should show:

- `prefix_aware_admit_deferrals > 0`
- non-zero prefix hit / skip distribution
- TTFT p50 improvement vs `queue-bound`
- no cold starvation (`cold p95 <= 3x warm p95`)

## Commands

Both arms used the same current binary and model:

```bash
CUDA_HOME=/opt/cuda \
TORCH_CUDA_ARCH_LIST=8.9 \
INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
INFER_HYBRID_W4A8_PREFILL=1 \
INFER_PREFILL_GRAPH=1 \
./target/release/infer \
  --model-path infer/models/Qwen3-4B-W4-hybrid-zpfix \
  --port 8765 \
  --num-slots 8 \
  --max-seq-len 5120 \
  --admission-policy <queue-bound|prefix-aware> \
  --cold-headroom 253
```

`--cold-headroom 253` was used to make the default internal
`max_waiting_requests=256` behave like `cold_soft_cap=3`; the local CLI does
not expose `--max-waiting-requests`.

Bench command per arm:

```bash
PATH=/home/ckl/projects/arle/.venv/bin:$PATH \
  scripts/bench_guidellm.sh 36-bench-<A|B> \
  --target http://127.0.0.1:8765 \
  --model Qwen3-4B-W4-hybrid-zpfix \
  --processor infer/models/Qwen3-4B \
  --concurrencies 8 --max-seconds 120 --warmup 10 \
  --data 'prompt_tokens=2048,prompt_tokens_stdev=512,output_tokens=128,output_tokens_stdev=32'
```

## Environment

- Commit: `3e83741` with counter commit `079639c` included
- GPU: RTX 4070 Ti SUPER 16 GiB
- CUDA: `/opt/cuda`, `TORCH_CUDA_ARCH_LIST=8.9`
- Model: `infer/models/Qwen3-4B-W4-hybrid-zpfix`
- Feature path: CUDA, W4 hybrid prefill enabled, prefill graph enabled

## Results

### Client-side GuideLLM output

Both arms completed real traffic, but GuideLLM marked the result invalid
because TTFT/ITL were recorded as `0.0` despite successful non-empty outputs.
This is the same streaming timing limitation seen in recent graph-capture runs,
so client TTFT/ITL cannot be used for license.

| Arm | Policy | Completed input toks | Incomplete input toks | Output toks | Errors | Output tok/s mean | Req/s mean | Request latency mdn/p95 | GuideLLM TTFT/ITL |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| A | queue-bound | 3,037,206 | 12,208 | 189,410 | 0 | 1,722.1 | 13.5 | 0.4s / 1.8s | invalid 0.0 / 0.0 |
| B | prefix-aware | 2,962,118 | 14,378 | 184,664 | 0 | 1,678.7 | 13.1 | 0.6s / 1.0s | invalid 0.0 / 0.0 |

### Server-side counters

| Arm | Policy | Peak waiting | Peak active | Peak running_batch | Prefix hit peak/q75 | Prefix skip peak | Deferrals | engine_ttft_us after | engine_itl_p50_us after |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | queue-bound | 7 | 8 | 8 | 0.0% / 0.0% | 0.0% | 0 | 150,000 | 15,000 |
| B | prefix-aware | 7 | 8 | 7 | 0.0% / 0.0% | 0.0% | 8,962 | 5,000 | 10,000 |

Raw artefacts:

- A: `bench-output/2026-05-10-36-bench-A-queuebound-run2/`
- B: `bench-output/2026-05-10-36-bench-B-prefixaware/`

## Problems

The PrefixAware gate did fire: treatment arm recorded
`prefix_aware_admit_deferrals=8962`.

However, the workload had no sessionized warm-vs-cold structure:

- `prefix_hit_rate` stayed `0.0%`
- `prefix_skip_rate` stayed `0.0%`
- `prefix_request_hit_rate` stayed `0.0%`
- `matched_prefix_tokens=0`
- `session_affinity_hit=0`

So this A/B only proves that the cold-pressure gate can defer requests. It does
not exercise the intended policy objective: preserve prefix-cache benefit for
warm sessions while cold arrivals queue behind the soft cap.

There is also no trustworthy client TTFT/ITL delta because GuideLLM wrote
invalid `0.0` TTFT/ITL values in both arms.

## Root Cause

The documented `prompt_tokens=2048,prompt_tokens_stdev=512,...` GuideLLM
workload generates independent cold requests. It does not send stable
`session_id` values or repeated shared prefixes in a way that populates the
runtime prefix-cache counters.

`PrefixAwareAdmission` is a warm-vs-cold admission policy. A cold-only workload
can trigger deferrals, but it cannot validate that warm requests receive better
service.

## Decision

Do not license or kill PrefixAwareAdmission performance from this run.

Status:

- Counter instrumentation: LANDED (`079639c`)
- Gate-trigger evidence: PASS (`8962` deferrals)
- Policy-benefit evidence: INVALID (0% prefix hits, no warm requests)
- GuideLLM client TTFT/ITL: INVALID (0.0 timing bug)

## Next Reproducer

Use a sessionized warm/cold burst workload instead of random independent
prompts:

1. Warm 4 sessions with the same 2k-6k system prefix and stable `session_id`.
2. Issue turn 2/3 requests for those warm sessions while injecting cold
   one-shot requests.
3. Run `queue-bound` vs `prefix-aware` with the same `cold_headroom`.
4. Require all of:
   - `prefix_aware_admit_deferrals > 0`
   - non-zero prefix hit rate / matched prefix tokens
   - warm TTFT p50 improvement >= 20%
   - cold p95 <= 3x warm p95

If GuideLLM cannot express this request pattern with `session_id`, use a small
custom HTTP benchmark derived from `scripts/bench_multitenant_burst.py` and
still capture `/v1/stats` before/during/after.

## Rule

For admission-policy benches, counter evidence must prove both sides of the
policy:

- pressure occurred (`waiting`, `deferrals`)
- the protected class existed (`prefix_hit_rate`, `matched_prefix_tokens`,
  `session_id`/affinity hits)

Deferrals on a cold-only workload are not a win; they are only a gate-smoke.
