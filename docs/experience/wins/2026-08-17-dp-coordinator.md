# DP coordinator: least-in-flight multi-group routing — CUDA, 2026-08-17

> Status: pending-remote

## Goal

Enable data-parallel serving: multiple independent TP groups on disjoint GPUs,
with the coordinator routing each request to the least-in-flight group. This
scales throughput past the single-group ceiling by adding full model replicas.

## Hypothesis

A `DpCoordinator` wrapping `Vec<Arc<CoordinatorHandle>>` with `Deref`-based
least-in-flight selection adds zero overhead on the DP=1 path (single-group
DpCoordinator, one Deref indirection) and scales linearly with DP count at
high concurrency. Handler bodies stay unchanged — Deref coercion fixes the
target group per call site, so per-group state (sinks, submit_tx, in_flight)
is consistent within each function call.

## Parameters

```bash
# Smoke: DP=2, TP=2, 4 GPUs
INFER_DP_SIZE=2 INFER_TP_SIZE=2 INFER_CUDA_DEVICES=0,1,2,3 \
  arle serve --model <dsv4-model> --port 8080
# Verify: both groups load, requests route to both, output coherent

# A/B: DP=1 vs DP=2 at high concurrency
python3 scripts/bench_throughput.py \
  --url <url> \
  --model <dsv4-model> \
  --prompts-jsonl bench-agent-119k-16x8.jsonl \
  --concurrency-grid 16,32,64 \
  --requests-per-concurrency 128 \
  --max-tokens 214 \
  --seed 20260416 \
  --timeout-seconds 900 \
  --output bench-output/dp-coordinator/bench
```

- Baseline: DP=1 (same binary, `INFER_DP_SIZE=1`)
- Treatment: DP=2 (`INFER_DP_SIZE=2`)
- Trials: 3 (matched A/B, simultaneous)

## Environment

- Host / GPU: 8×H20 pod (sm_90)
- Driver / CUDA: 12.8
- Model / dtype: DSv4 MoE or ThinkingCap-27B-FP8
- TP / DP: TP=2, DP=2 (4 GPUs) or TP=4, DP=2 (8 GPUs)
- Server flags: `INFER_DP_SIZE=2`

## Design

- `DpCoordinator` in `coordinator.rs`: wraps `Vec<Arc<CoordinatorHandle>>`,
  implements `Deref<Target=CoordinatorHandle>` with least-in-flight `select()`.
- `coordinator_router` always wraps in `DpCoordinator` (single-group for DP=1);
  `dp_coordinator_router` creates one `CoordinatorHandle` per relay.
- Handler signatures change from `State<Arc<CoordinatorHandle>>` to
  `State<Arc<DpCoordinator>>`; bodies unchanged (Deref coercion).
- Helper functions (`streaming_submit`, `encode`, `decode`, etc.) take
  `&CoordinatorHandle` instead of `&Arc<CoordinatorHandle>`.
- `bind_relay_and_spawn_workers` returns `Vec<MultiprocCoordinator>`, loops
  DP times with per-group NCCL unique ID + relay port.
- `spawn_workers` assigns GPU `group * world_size + rank`.
- `INFER_DP_SIZE` env var (default 1), consistent with `INFER_TP_SIZE`.

## Results

| concurrency | arm | completed | errors | output tok/s | req/s | TTFT p50/p99 ms | ITL p50/p99 ms | delta |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 16 | DP=1 | | | | | | | — |
| 16 | DP=2 | | | | | | | |

Smoke + A/B bench: pending (pod gate).

## Problems

None yet.

## Learnings

pending-remote. The `Deref` approach is the lowest-entropy routing layer:
one struct, one trait impl, zero handler body changes. The per-group NCCL
unique ID is load-bearing — sharing one ID across groups deadlocks the NCCL
rendezvous (each group expects `world_size` ranks, but `world_size * dp_size`
try to join).
