# TP/CP NCCL collectives → comm_stream — CUDA, 2026-08-17

> Status: **Shipped; decode regression found post-ship.** The A/B bench below
> was never run (table left empty, "pending"); the "wash by construction"
> hypothesis was wrong. Measured 2026-08-17 at TP=8, 128K context: decode
> 78.7 → 55–59 tok/s (−25–30%). Partial fix (all-reduce back to compute
> stream) recovered to 63 tok/s; remaining −19% gap under investigation.
> Single-GPU (no NCCL) is unaffected.
>
> **Root-cause fix: event pool** (`9a82dbe4d`, 2026-08-17). The regression
> was per-fence `cuEventCreate`/`cuEventDestroy` — 80 all-reduces × 2 fences
> = 160 event allocations per decode step at TP=8. The pool reuses events
> (`Arc<Mutex<Vec<CudaEvent>>>` in `DeviceContext`), eliminating steady-state
> allocation. Bench: pending-remote ([entry](2026-08-17-cuda-event-pool-pipeline-fences.md)).

## Goal

Move NCCL collectives off the compute stream so communication can overlap
compute. The strictly-dependent decode chain (AR output → residual add →
next layer) has no slack, so this is a wash on decode by construction; the
value is the stream plumbing that enables the T3 CP-decode merge path
(attn_tp all-reduce + cp row-gather per layer) to overlap compute.

## Hypothesis

Bracketing each NCCL collective with `comm_waits_for_compute` /
`compute_waits_for_comm` fences and running the NCCL enqueue on
`comm_stream` produces identical results with no decode regression (the
fences add event create/destroy overhead but no host stall).

**Disproven.** The fences allocate a new CUDA event per call
(`new_event(None)` in `record_pipeline_fence`). At TP=8 a decode step issues
80 all-reduces (64 MLP + 16 attention) × 2 fences = 160 event allocations
per step, adding ~3–5 ms/step host overhead. The strictly-dependent chain
has no slack to hide it.

## Parameters

```bash
# A/B: baseline = a59c6c661^, treatment = a59c6c661
# ThinkingCap-27B-FP8, TP>=2 (NCCL arm, not one-shot)
python3 scripts/bench_throughput.py \
  --url <url> \
  --model ThinkingCap-27B-FP8 \
  --prompts-jsonl bench-agent-119k-16x8.jsonl \
  --concurrency-grid 1,4,8,16 \
  --requests-per-concurrency 128 \
  --max-tokens 214 \
  --seed 20260416 \
  --timeout-seconds 900 \
  --output bench-output/comm-stream/bench
```

- Baseline: `a59c6c661^` (NCCL on compute stream, no fences)
- Treatment: `a59c6c661` (NCCL on comm_stream, fenced)
- Trials: 3 (matched A/B, simultaneous)

## Environment

- Host / GPU: 8×H20 pod (sm_90)
- Driver / CUDA: TBD
- Model / dtype: ThinkingCap-27B-FP8
- TP / EP / slots / KV: TP=8 (or ≥2)
- Server flags: default

## Results

Smoke test (2026-08-17, pod 8×H20, TP=8, ThinkingCap-Qwen3.6-27B-FP8):
serve ready, all 8 workers loaded, coherent generation (thinking model),
no NCCL errors. Needle gate ×3 (LENGTHS=8000, RUNS=3): exact=3/3,
correctness PASS.

| concurrency | arm | completed | errors | decode tok/s | req/s | TTFT p50/p99 ms | ITL p50/p99 ms | delta |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | baseline | — | — | — | — | — | — | — |
| 1 | treatment | — | — | — | — | — | — | — |

A/B bench: **never run.** The "wash by construction" claim was accepted
without measurement. The decode regression was found by a separate
128K decode-rate probe (TP=8, CP=1, `decode_rate_probe.py
--target-tokens 128000 --max-tokens 128`):

| build | decode tok/s @ 128K | Δ vs baseline |
|---|---:|---:|
| pre-comm-stream (4bcefcb57) | 78.7 | — |
| comm-stream (a59c6c661) | 55–59 | −25–30% |
| + all-reduce → compute stream fix | 63 | −20% |

The remaining −19% gap (63 vs 78.7) is under investigation. Single-GPU
(no NCCL, no fences) is unaffected by construction.

Raw artifacts: `/host/arle-runs/212-comm/` on pod.

## Problems

**Decode regression shipped undetected.** The A/B bench was deferred
("pending") and the "wash by construction" hypothesis was accepted without
measurement. The per-fence event allocation cost is host-side, not GPU-side,
so it does not appear in a GPU profile — only a wall-clock decode probe
catches it. The regression was found by a separate 128K decode-rate probe,
not by this entry's bench.

## Learnings

PASS (smoke + needle). The fence machinery (`comm_waits_for_compute` /
`compute_waits_for_comm`) already existed in tensor.rs and was used by the
dsv4 shared-expert overlap path; this change generalizes it to all NCCL
collectives. The dsv4 prefill shared-expert overlap is preserved by keeping
the moe all-reduce on the compute stream via `all_reduce_sum_on(Compute)`.
One-shot all-reduce stays on the compute stream (small-message fast path;
moving it is a follow-up — `arle_car_allreduce_bf16_into` already takes a
stream parameter).

**"Wash by construction" is not a bench.** A strictly-dependent chain has no
slack to hide overhead, so any added host cost lands directly on the critical
path. The A/B table must be filled before shipping, not after. The
`decode_rate_probe.py` 128K probe is the gate for decode-path changes at
TP≥2 — smoke + needle only validates correctness, not performance.
