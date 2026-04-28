# Metal Qwen3.5 GuideLLM sweep produced no decode batch

## Context

On 2026-04-28, after adding Metal decode-path counters, I attempted the
canonical GuideLLM run against the local Metal Qwen3.5-0.8B GGUF server:

```bash
./scripts/start_metal_serve.sh models/Qwen3.5-0.8B-GGUF 8013 -- --warmup 0

scripts/bench_guidellm.sh metal-decode-observability \
  --target http://127.0.0.1:8013 \
  --model Qwen3.5-0.8B-GGUF \
  --processor models/Qwen3.5-0.8B \
  --trace-interval-ms 1000
```

The run was stopped after about 15 minutes because it still had not produced
`benchmarks.json` and the server had accumulated a long queue. Raw artifacts
were left under:

```text
bench-output/2026-04-28-metal-decode-observability/
```

The service trace had 865 successful `/v1/stats` samples. Headline counters:

```text
Peak waiting: 256
Peak active: 1
Peak running_batch: 1
Peak prefill_queue: 1
Peak kv_util: 0.0%
After: metal_decode=batch:0/0,scalar:44789,fallback:0,qwen35_packed:0/0
```

While shutting down the overloaded server, the runtime also logged repeated
prefill failures:

```text
Metal prefill chunk failed for RequestId(...): MLX error: qwen35_session_begin requires an inactive session
```

## Root Cause

The new counters show the serving gap directly: the scheduler may have waiting
work and occasional `batch_width=2`, but the Qwen3.5 decode path did not enter
the packed GPU batch at all during this long-prompt sweep. The workload stayed
effectively single-active (`peak_active=1`) and every observed decode row ran
through the scalar path.

This points at an admission/prefill lifecycle bottleneck rather than a single
decode kernel bottleneck. Under the canonical 4096-in / 256-out sweep, long
prefills and the current request-state ownership prevent multiple Qwen3.5 rows
from reaching the persistent packed decode cache together.

The `qwen35_session_begin requires an inactive session` errors are a separate
cleanup/lifecycle smell: after cancellation or overload, at least one compiled
Qwen3.5 session can remain active when a later prefill tries to begin a new
session. The run was aborted before reducing this to a smaller reproducer.

## Fix

No performance fix landed in this tranche. What landed is observability:

- `infer_metal_decode_batches_total`
- `infer_metal_decode_batched_rows_total`
- `infer_metal_decode_scalar_rows_total`
- `infer_metal_decode_batch_fallback_rows_total`
- `infer_metal_qwen35_packed_decode_batches_total`
- `infer_metal_qwen35_packed_decode_rows_total`

These counters now make the scheduler-vs-backend gap explicit in both
Prometheus and `/v1/stats`.

Next implementation work should be ordered:

1. Reduce the session-lifecycle failure to a small cancellation/backlog test.
2. Fix Qwen3.5 compiled-session cleanup before further high-rate sweeps.
3. Rework Metal admission/prefill so concurrent long prompts can reach decode
   together and actually enter `Qwen35PackedDecodeBatch`.

## Rule

Do not treat `scheduler_scheduled_decode_rows` or `batch_width` as proof that
Metal is using a batched GPU decode path. For Metal serving work, always pair
scheduled-row counters with backend-path counters:

```text
scheduled_decode_rows vs metal_decode=batch/scalar/fallback/qwen35_packed
```

If `qwen35_packed` remains zero under a concurrent workload, the next fix is
admission/prefill/state ownership, not another single-request decode kernel
tweak.
