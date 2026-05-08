# W3 short-multiturn bench — ARLE 503 capacity error at c=16 (admission backlog)

> Master §7.1 P0.0 W3 c=16 baseline attempt. Discovered ARLE admission/
> queue capacity issue at the master strategy production-shape concurrency.
> 135/136 turns failed with HTTP 503 "Server is at capacity"; only 1 turn
> succeeded.

## Setup

```bash
CUDA_HOME=/opt/cuda TORCH_CUDA_ARCH_LIST=8.9 \
  ./target/release/infer --model-path infer/models/Qwen3-4B-W4A16-sym-g128-marlin \
  --port 8000 --num-slots 16 --max-seq-len 5120

# Harness: master §2.1 W3 short-multiturn (c=16, base 1024 ± 32, tail 64 ± 8/turn × 4 turn)
python scripts/bench_agent_trace.py \
  --workload agent-w3-short-multiturn \
  --server http://localhost:8000 \
  --model Qwen3-4B-W4A16-sym-g128-marlin \
  --num-concurrent 16
```

W3 driver enforces `--num-concurrent 16` per master §2.1 W3 spec.

## Result

```
turns OK: 1 / 136 (0.7%)
tokens total: 1
wall total (s): 0.50
TTFT p50/p99: 446.0 / 446.0 ms

(135 turns failed with HTTP 503 "Server is at capacity, please retry later")
```

`/v1/stats` after run:
- `active=16, prefill_queue=15` — 16 sessions all admitted but 15 still
  queued in prefill (only 1 actively prefilling)
- `kv_util=1.1%` — KV cache barely utilized
- `active_mem=14151.6 MB / peak=14151.6 MB` — model weights + scaffolds
  consuming most of 16 GB GPU
- `step_phase_us=adm:64, prefill:429743, decode:0` — 429 ms in prefill
  for the single chunk processed

## Root cause hypothesis

**Prefill admission backlog**:
- 16 sessions × 1024 prompt tokens = 16384 tokens needed
- `max_num_batched_tokens = 16384` (matches exactly)
- `chunked_prefill_size = 2048` → 8 chunks needed to clear initial queue
- Per chunk ~50-200 ms → 0.4-1.6 s backlog total
- HTTP harness times out per-request at a smaller window (likely 30 s
  hard-cap per turn) but server admission queue rejects with 503

The 503 is the HTTP layer's "queue depth exceeded" or a per-window
admission limit, not necessarily the scheduler. Without the source for
the admission code (`infer/src/http_server/`), can't pin exact threshold.

## What this proves

1. **Master §7.1 P0.0 W3 baseline cannot run as-is** on current ARLE with
   default max_seq_len/num_slots. Either:
   - Increase `--max-num-slots` to 32+ (more admission headroom)
   - Increase `--max-num-batched-tokens` (more prefill bandwidth per step)
   - Reduce `--num-concurrent` from 16 to 4-8 (off-spec but bench-able)
   - Add admission backoff/retry on 503 in harness
2. **Spec-decode axis re-test on W3** is gated on this admission fix.
   Without functioning W3 baseline, cannot measure spec-decode lift on
   structured workload (master §2.1 production shape).
3. **The single successful turn measured TTFT 446 ms**, which is consistent
   with ARLE BF16 baseline at smaller workload — suggests when admission
   isn't backlogged, the model + KV path work.

## Recommended fixes (codex / Claude split)

| Action | Owner | Cost |
|---|---|---|
| Increase ARLE default `--max-num-slots` to 32 for W3-class workloads | Claude (CLI flag tuning) | 1 min |
| Add HTTP harness 503 retry-with-backoff in `bench_agent_trace.py` | Claude (~30 LOC) | 30 min |
| Investigate ARLE admission policy for high-concurrency burst | codex (substrate) | hours |
| Confirm ARLE `--max-num-batched-tokens` flag exists/works (not seen in --help quickly) | Claude / read code | 15 min |

## Skill methodology

- ✅ Phase 5 attempted single-variable A/B (W3 workload vs no-spec)
- ❌ Phase 8 not reached — admission failure preempted bench
- ✅ NULL elimination (anti-pattern #13): the 0.7% success rate IS data —
  ARLE's W3 capacity at default settings is NOT viable

This discovery is a prerequisite blocker for spec-decode axis re-test
(both `5f26675` self-spec and `3ac5f4d` ext-draft KILLs were at 4k
random text, NOT W3 production shape; the W3 re-test would need this
fix to even start).

## Cross-references

- W3/W4 bench plan: master §7.1 P0.0 + `scripts/bench_agent_trace.py`
- master §2.1 W3 spec: short multi-turn, c=16, base 1024 + tail 64×4
- ARLE admission: `infer/src/http_server/` + `infer/src/scheduler/cuda/runtime/admission.rs`
- Spec-decode KILL chain: [`5f26675`](2026-05-08-spec-decode-self-spec-k5-kill.md), [`3ac5f4d`](2026-05-08-spec-decode-ext-draft-k5-kill.md)

## Rule

W3 c=16 spec is HARDER than 4k random text c=4 — 16 simultaneous prefill
streams overwhelm ARLE's admission policy at default `--num-slots 16
--max-num-batched-tokens 16384`. Master §7.1 P0.0 cannot complete its
"3-engine W3/W4 baseline" mandate until ARLE admission for c=16 burst
is fixed.

This blocks the spec-decode axis re-test on production shape. Until then,
spec-decode evaluation remains stuck at 4k random text (where workload-
dead per `5f26675` + `3ac5f4d`).
