# #36 Layer 2 warm-mix bench - GuideLLM JSONL path invalid, direct HTTP bench required

## Context

Layer 1 #36 A/B established that PrefixAware admission can fire under cold
pressure, but the random synthetic GuideLLM workload had 0% prefix hits:

- Counter instrumentation landed in `079639c`
- Layer 1 invalidation entry: `9a8c6d5`
- Layer 1 treatment deferrals: `prefix_aware_admit_deferrals=8962`
- Layer 1 prefix reuse evidence: `prefix_hit_rate=0.0%`

Claude then landed `5453ee4` with:

- `scripts/gen_36_warm_prefix_mix.py`
- `docs/research/2026-05-10-36-followup-warm-mix-bench-spec.md`

Codex ran Layer 2 using that deterministic warm/cold JSONL workload.

## Goal

Type: diagnosis.

Validate that PrefixAwareAdmission is exercised under a warm/cold prefix-mix
workload and determine whether the GuideLLM JSONL path can produce a valid
license/kill TTFT delta.

## Hypothesis

The warm-mix JSONL should prove both required mechanisms:

- prefix reuse exists (`prefix_hit_rate > 0`)
- PrefixAware defers cold requests (`prefix_aware_admit_deferrals > 0`)

If GuideLLM records valid TTFT/ITL, use the A/B delta for license. If not,
switch to a direct HTTP benchmark loop.

## Commands

Generated workload:

```bash
PATH=/home/ckl/projects/arle/.venv/bin:$PATH \
  ./scripts/gen_36_warm_prefix_mix.py \
    --tokenizer infer/models/Qwen3-4B/tokenizer.json \
    --out bench-output/36-warm-mix.jsonl \
    --num-requests 256 --warm-fraction 0.6 --num-sessions 4 \
    --shared-prefix-tokens 1024 --tail-tokens 256 --output-tokens 128
```

Output:

```text
wrote 256 rows to bench-output/36-warm-mix.jsonl
  warm:    153 (4 sessions x ~38 reqs each, 1024-tok shared prefix)
  cold:    103 (unique random 1280-tok prompts)
  output:  128 tokens per request
  seed:    41262 (deterministic)
```

Server commands used the corrected `--cold-headroom 253` workaround because
the local CLI does not expose `--max-waiting-requests`:

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

Bench command per arm:

```bash
PATH=/home/ckl/projects/arle/.venv/bin:$PATH \
  scripts/bench_guidellm.sh 36-warmmix-<A|B> \
    --target http://127.0.0.1:8765 \
    --model Qwen3-4B-W4-hybrid-zpfix \
    --processor infer/models/Qwen3-4B \
    --concurrencies 8 --max-seconds 120 --warmup 10 \
    --data bench-output/36-warm-mix.jsonl
```

## Environment

- Commit: `3e83741` plus `5453ee4` already on `main`
- Counter commit: `079639c`
- GPU: RTX 4070 Ti SUPER 16 GiB
- CUDA: `/opt/cuda`, `TORCH_CUDA_ARCH_LIST=8.9`
- Model: `infer/models/Qwen3-4B-W4-hybrid-zpfix`
- Runtime flags: CUDA, W4 hybrid prefill, prefill graph, 8 slots,
  `max_seq_len=5120`, `cold_headroom=253`

## Results

### Mechanism counters

Layer 2 did exercise prefix reuse. This is the important improvement over
Layer 1.

| Arm | Policy | Requests seen by server | Peak waiting | Peak active | Prefix hit peak/q75 | Prefix hit after | Prefix skip peak | Deferrals |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | queue-bound | 257 | 0 | 8 | 93.8% / 84.6% | 56.4% | 75.0% | 0 |
| B | prefix-aware | 257 | 7 | 8 | 92.7% / 56.4% | 56.4% | 74.2% | 90 |

Raw artefacts:

- A: `bench-output/2026-05-10-36-warmmix-A-queuebound/`
- B: `bench-output/2026-05-10-36-warmmix-B-prefixaware/`

### GuideLLM validity

GuideLLM output is invalid in both arms:

| Arm | Client duration | Client completed tokens | Client request TTFT | Wrapper verdict |
|---|---:|---:|---:|---|
| A | 1.3s, despite `--max-seconds 120` and warmup 10s | 9,533 input / 112 output | p50 0.0ms | invalid: TTFT p50 0.0 |
| B | -4.2s duration | 0 input / 0 output | p50 0.0ms | invalid: no successful requests |

The B-arm contradiction is decisive: `/v1/stats` showed 257 requests reached
the server, while GuideLLM recorded 0 successful requests and negative
duration.

## Root Cause

The GuideLLM JSONL path is not usable for this finite warm-mix workload:

- It reports `inf unique requests`, but drains the 256-row JSONL almost
  immediately instead of sustaining the requested 120s window.
- It records `TTFT p50=0.0` even when the server returns non-empty outputs.
- In the PrefixAware arm it reports 0 completed requests while `/v1/stats`
  proves the server processed the full workload.

This is a bench-tool/input-path failure, not a PrefixAware runtime failure.

## Secondary Finding

The current generator uses shared prompt prefixes, not explicit `session_id`.
That is enough to exercise the prefix cache (`prefix_hit_rate=56.4%` after),
but not the session-affinity path:

- `session_affinity_hit=0`
- `session_affinity_miss=0`
- `matched_prefix_tokens=0`

For the next license run, the direct HTTP benchmark should include explicit
session IDs if the OpenAI-compatible request path supports them.

## Decision

Do not license or kill PrefixAwareAdmission performance from Layer 2.

Status:

- Layer 1 substrate: PASS, gate fires under cold-only pressure
- Layer 2 mechanism: PASS, warm-mix generates prefix hits
- Layer 2 PrefixAware pressure: PASS, treatment deferrals recorded
- Layer 2 TTFT license: INVALID, GuideLLM JSONL timing/accounting broken

## Next Step

Replace GuideLLM for this workload with a direct HTTP benchmark loop:

1. Read `bench-output/36-warm-mix.jsonl`.
2. Preserve row labels from generator order: first 153 warm, last 103 cold.
3. Send `/v1/completions` or `/v1/chat/completions` at concurrency 8.
4. Measure client-side:
   - first-token latency for streaming responses
   - full request latency
   - output tokens
   - warm vs cold p50/p95/p99
5. Capture `/v1/stats` before/during/after.
6. Require:
   - `prefix_hit_rate > 0`
   - `prefix_aware_admit_deferrals > 0`
   - warm p50 TTFT improves >= 20%
   - aggregate throughput does not regress
   - cold p95 <= 3x warm p95

This benchmark can live as a small purpose-built script, likely derived from
`scripts/bench_multitenant_burst.py`, because GuideLLM cannot currently express
this finite session/prefix-mix workload reliably.

## Rule

For finite JSONL workloads, trust GuideLLM only if all three views agree:

- client duration matches requested run shape
- client completed request/token counts match the input cardinality
- `/v1/stats` request count matches client counts

If these disagree, use `/v1/stats` only for mechanism evidence and switch to a
direct HTTP benchmark for license/kill latency numbers.
