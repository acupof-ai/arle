# P2.4 adaptive spec longctx smoke invalid

## Context

Phase 2 P2.4 added per-request adaptive speculative acceptance tracking:
`AcceptanceTracker::observe_step(accepted, drafted)` now keeps a rolling
64-step token-weighted window, CUDA requests hold an optional tracker when
spec decode is enabled, and decode readback disables speculation per request
when the configured `SchedulerConfig::spec_acceptance_threshold` is not met.

Validation and bench were run on `2026-05-01` before committing the patch.

## Root Cause

The implementation-level tracker tests passed, but the required longctx-32k
c=4 60s smoke did not produce a valid performance result.

Spec-enabled command:

```bash
./target/release/infer \
  --model-path infer/models/Qwen3-4B \
  --port 8000 \
  --kv-cache-dtype fp8 \
  --num-slots 16 \
  --max-seq-len 131072 \
  --mem-fraction-static 0.85 \
  --max-num-batched-tokens 16384 \
  --max-prefill-tokens 16384 \
  --schedule-policy fcfs \
  --spec-enabled \
  --spec-draft-k 5 \
  --spec-acceptance-threshold 0.3 \
  --spec-draft-model self

WORKLOAD=longctx-32k LONGCTX_CONCURRENCIES=4 LONGCTX_MAX_SECONDS=60 \
  scripts/bench_guidellm.sh p24-adaptive-spec-c4-smoke \
  --target http://127.0.0.1:8000 \
  --model Qwen3-4B \
  --processor infer/models/Qwen3-4B
```

Result: invalid GuideLLM result set, `0` successful requests, `131072`
incomplete input tokens, `8` incomplete output tokens. Service trace:
`bench-output/2026-05-01-p24-adaptive-spec-c4-smoke/service_stats_trace_summary.md`.

Key service trace evidence:

| metric | value |
|---|---:|
| peak active | 4 |
| peak kv_util | 95.7% |
| plan labels | `idle=420901`, `decode=8`, `prefill=18`, `mixed=4` |
| spec counters after | `draft=11`, `verified=11`, `accepted=11`, `accept_rate=100.0%` |
| completed requests | 0 |

This means the adaptive threshold did not disable speculation because the
current P2.3 greedy canary accepted every verified token. The invalid result
was dominated by long final mixed rows, not by a low acceptance-rate fallback.

A no-spec control using the same current binary and the same 60s c=4 workload
also produced an invalid result:

```bash
./target/release/infer \
  --model-path infer/models/Qwen3-4B \
  --port 8000 \
  --kv-cache-dtype fp8 \
  --num-slots 16 \
  --max-seq-len 131072 \
  --mem-fraction-static 0.85 \
  --max-num-batched-tokens 16384 \
  --max-prefill-tokens 16384 \
  --schedule-policy fcfs

WORKLOAD=longctx-32k LONGCTX_CONCURRENCIES=4 LONGCTX_MAX_SECONDS=60 \
  scripts/bench_guidellm.sh p24-nospec-c4-smoke-control \
  --target http://127.0.0.1:8000 \
  --model Qwen3-4B \
  --processor infer/models/Qwen3-4B
```

Result: invalid GuideLLM result set, `0` successful requests, `131072`
incomplete input tokens, `8` incomplete output tokens. Artifact:
`bench-output/2026-05-01-p24-nospec-c4-smoke-control/`.

The deterministic GPU correctness test also fails on the existing P2.3
verifier path:

```bash
ZIG=/content/workspace/agent-infer/.toolchains/zig/zig-x86_64-linux-0.16.0/zig \
  cargo test -p infer --features cuda --test spec_decode_correctness
```

Failure:

```text
spec_decode_greedy_is_bit_identical_for_three_prompts ... FAILED
prompt: "Write a tiny Rust function name."
plain: " It should be a function that takes a string and returns a"
spec:  " It's a function that takes a string and returns the number"
```

## Fix

Do not claim a P2.4 throughput win from this patch. The adaptive tracker slice
can land as plumbing, but P2.5 must first repair the P2.3 verifier correctness
gap and the current longctx c=4 60s smoke invalid result before any speculative
decode speedup claim.

## Rule

Spec decode patches must pass the greedy bit-identical correctness test before
throughput results are accepted. A GuideLLM run with zero successful requests is
an error artifact, even when service-side counters show useful diagnostic data.
