# W3 c=16 ARLE deadlock — retry-backoff harness is correct,but problem is deeper

> Companion to [`a672b08`](2026-05-08-w3-bench-capacity-503-admission-backlog.md)
> initial 503 finding。Harness retry-backoff(`e7b4765`)was committed,
> validated structurally(no streaming bugs),and re-tested at W3 c=16:
>
> **Same 0.7% success rate**(1/136 turns)despite 31s exponential backoff
> per turn(1+2+4+8+16 = 31s,5 retries)。/v1/stats reveals **prefill
> deadlock,not transient queue saturation**:`active=16, prefill_queue=15,
> prefill_rows=0, tokens_out=0` — 14 GB pinned with zero prefill activity。

## Setup

```bash
CUDA_HOME=/opt/cuda TORCH_CUDA_ARCH_LIST=8.9 \
  ./target/release/infer --model-path infer/models/Qwen3-4B-W4A16-sym-g128-marlin \
  --port 8000 --num-slots 16 --max-seq-len 5120

python scripts/bench_agent_trace.py \
  --workload agent-w3-short-multiturn \
  --num-concurrent 16
# (with retry-backoff harness from e7b4765)
```

## Result

```
turns OK: 1 / 136 (0.7%)
135 errors: "HTTP 503 after 5 503 retries: Server is at capacity..."
TTFT p50/p99 (1 success): 420 ms (consistent with no-spec W4A16 W3-light)
```

Per-error breakdown:
- All 135 errors are AFTER 5 retries × exponential backoff (31s total wait)
- Total elapsed per failing turn: 31100 ms (entire backoff window exhausted)

## /v1/stats observation — DEADLOCK signal

After bench end:
```
active=16, waiting=0, scheduled=0, prefill_queue=15
decode_rows=0, prefill_rows=0, running_batch=0
batch_width=0, decode_tokens=0, prefill_tokens=0, tokens_out=0
step_last=0.0ms, step_phase_us=adm:72, prefill:403356, decode:0
plan_label=idle:0, decode:0, prefill:1, split:1, mixed:0
peak_mem=14151.6MB
engine_active_requests=16, engine_batch_occupancy=0.0106
session_affinity_miss=16, resume_prefill_tokens=1084
```

Diagnostic interpretations:

1. **`active=16, prefill_queue=15`**: 16 sessions admitted to scheduler,
   15 still waiting for prefill turn. Should be 1 actively prefilling.
2. **`prefill_rows=0, prefill_tokens=0`**: But ZERO active prefill rows.
   The 1 supposedly-active session is stalled mid-step.
3. **`step_phase_us=prefill:403356`**: 403 seconds of accumulated prefill
   time across the whole bench wall-time window — but tokens_out=0.
4. **`peak_mem=14151 MB`**: Most of 16 GB GPU memory pinned, suggests
   16 sessions' KV slots are reserved but not making progress.
5. **`session_affinity_miss=16`**: All sessions failed to find affinity
   (cold sessions; first-time prefix). Expected for W3 cold.
6. **`resume_prefill_tokens=1084`**: Only 1084 tokens of resume seen
   across 16 × 1024 = 16384 expected. Suggests prefill kicked off but
   stalled.

This is **NOT** transient queue saturation — that would resolve in
seconds as prefill chunks complete. This is **scheduling deadlock or
resource starvation**:
- Hypothesis A: `chunked_prefill` chose chunk-size that interacts badly
  with c=16 slot rotation
- Hypothesis B: All 16 slots reserved → KV admission can't free
  intermediate state → progress halts
- Hypothesis C: Prefill plan label `prefill:1, split:1` suggests one
  prefill-split is stuck and nothing else can run (HOL blocking?)

## Why harness retry-backoff was the right fix at the wrong level

`e7b4765` retry-backoff is structurally correct (handles transient 503
gracefully). At c=4 and below, ARLE may have transient 503 during chunk
admission churn — retry-backoff would help. But at c=16 the failure
mode is qualitatively different: **deadlock with active KV reservation**.
No amount of backoff resolves a deadlock.

The harness fix was committed at `e7b4765` and remains valid for c≤8
admission resilience. Should NOT be reverted — but **NOT sufficient
alone for c=16 W3 baseline**.

## Master strategy implication

Master §7.1 P0.0 W3/W4 baseline mandate at c=16 is BLOCKED on:

1. **Codex substrate**: ARLE admission/scheduler at c=16 — investigate
   `infer/src/scheduler/cuda/runtime/admission.rs` + `chunked_prefill.rs`
   for the deadlock. Likely needs:
   - Larger `--max-num-batched-tokens` (currently 16384 = c=16 × 1024)
   - Explicit prefill rotation (round-robin instead of HOL)
   - OR scheduler invariant fix (active vs prefill_queue counter race)

2. **Workaround**: bench at lower concurrency (c=4 or c=8) to validate
   harness + W3 workload semantically, even if off-spec for master §2.1.

3. **Spec-decode axis re-test**: W3/W4 at production spec is now blocked
   on substrate fix; classical KILL evidence (3 KILLs at 4k+32k) +
   Medusa REQUIRED (master §7.4 update `5acbe94`) framework remains.

## Skill methodology validation

Per anti-pattern #13 (NULL elimination):
- Harness fix `e7b4765`: structurally valid, validated by re-test (no
  streaming bugs in retry path) — REAL elimination of "transient 503
  recovery" hypothesis.
- This entry: NULL elimination of "harness-side fix sufficient" — the
  remaining bug is ARLE-side scheduling.

Codex action recommended: investigate deadlock per /v1/stats signature
above. Read `infer/src/scheduler/cuda/core/scheduler.rs` for active vs
prefill_queue invariants. Likely substrate-LOC ~100-300 fix.

## Cross-references

- Initial 503 discovery: [`2026-05-08-w3-bench-capacity-503-admission-backlog.md`](2026-05-08-w3-bench-capacity-503-admission-backlog.md) (`a672b08`)
- Harness retry-backoff: [`scripts/bench_agent_trace.py`](../../../scripts/bench_agent_trace.py) (`e7b4765`)
- Master §7.1 P0.0 W3/W4 baseline mandate: master strategy §7.1
- Spec-decode classical KILL chain: `5f26675` `3ac5f4d` `8f2b227`
- Master §7.4 P1.1 Medusa REQUIRED: `5acbe94`
- ARLE scheduler core: `infer/src/scheduler/cuda/core/scheduler.rs`
- ARLE admission: `infer/src/scheduler/cuda/runtime/admission.rs`

## Rule

When `active=N` but `prefill_rows=0` AND `tokens_out=0` over a long
time window: **deadlock signature** (resource reserved but no progress).
Distinguish from transient saturation by checking forward progress:
no progress → substrate bug, not capacity-tuning issue. Backoff fixes
do not resolve deadlocks.

For ARLE W3 c=16, this is a substrate-level scheduling fix needed,
not a harness or CLI tuning fix.
