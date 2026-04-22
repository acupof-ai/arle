# Qwen3 tiered-KV T1/T2/T3 readmission validation

## Goal

- Verify the full CUDA tiered-KV chain on `Qwen3-4B` L4: T1 host recall, T2 disk store+recall, T3 shared-fs store+recall.
- Separate correctness of slower-tier recall from the canonical `c16` throughput lane.
- Confirm what “prefetch KV” means in the current tree.

## Hypothesis

- The host-tier store path is currently under-driven by a coarse percentage gate, so T1→T2/T3 store can remain cold even after repeated T0→T1 demotion.
- The slower-tier recall heuristic is overly optimistic about cold-prefill speed, so disk/shared-fs hits can be incorrectly downgraded to recompute.
- The runtime still uses on-demand staged fetch, not proactive prefetch-plan orchestration.

## Commands

Initial multiprefix store-check with byte-level T1 drain gate:

```bash
python3 scripts/trace_tier_kv.py \
  --model-path /root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
  --port 8054 \
  --num-slots 16 \
  --max-seq-len 4608 \
  --mem-fraction-static 0.94 \
  --chunked-prefill-size 4096 \
  --max-prefill-tokens 16384 \
  --max-tokens 8 \
  --warm-prefixes 16 \
  --warm-repeats 3 \
  --churn-requests 40 \
  --replay-prefixes 4 \
  --replay-repeats 2 \
  --t1-host-pinned-high-water 0.20 \
  --t1-host-pinned-low-water 0.10 \
  --t1-host-pinned-keepalive-ticks 512 \
  --disk-store-root /tmp/infer-kv-disk-trace-qwen3-c16-multiprefix16-storecheck-drainbytes \
  --cluster-shared-root /tmp/infer-kv-shared-trace-qwen3-c16-multiprefix16-storecheck-drainbytes \
  --trace-dir bench-output/2026-04-22-tier-kv-trace-qwen3-c16-multiprefix16-storecheck-drainbytes/traces \
  --log-path bench-output/2026-04-22-tier-kv-trace-qwen3-c16-multiprefix16-storecheck-drainbytes/infer.log \
  --out bench-output/2026-04-22-tier-kv-trace-qwen3-c16-multiprefix16-storecheck-drainbytes/summary.json
```

Focused T2 disk recall validation after slower-tier heuristic fix:

```bash
python3 scripts/trace_tier_kv.py \
  --model-path /root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
  --port 8056 \
  --num-slots 16 \
  --max-seq-len 4608 \
  --mem-fraction-static 0.94 \
  --chunked-prefill-size 4096 \
  --max-prefill-tokens 16384 \
  --max-tokens 8 \
  --warm-prefixes 1 \
  --warm-repeats 3 \
  --churn-requests 80 \
  --replay-prefixes 1 \
  --replay-repeats 2 \
  --t1-host-pinned-high-water 0.20 \
  --t1-host-pinned-low-water 0.10 \
  --t1-host-pinned-keepalive-ticks 512 \
  --disk-store-root /tmp/infer-kv-disk-trace-qwen3-singleprefix-disk-recall-v2 \
  --trace-dir bench-output/2026-04-22-tier-kv-trace-qwen3-singleprefix-disk-recall-v2/traces \
  --log-path bench-output/2026-04-22-tier-kv-trace-qwen3-singleprefix-disk-recall-v2/infer.log \
  --out bench-output/2026-04-22-tier-kv-trace-qwen3-singleprefix-disk-recall-v2/summary.json
```

Focused T3 shared-fs recall validation after slower-tier heuristic fix:

```bash
python3 scripts/trace_tier_kv.py \
  --model-path /root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
  --port 8057 \
  --num-slots 16 \
  --max-seq-len 4608 \
  --mem-fraction-static 0.94 \
  --chunked-prefill-size 4096 \
  --max-prefill-tokens 16384 \
  --max-tokens 8 \
  --warm-prefixes 1 \
  --warm-repeats 3 \
  --churn-requests 80 \
  --replay-prefixes 1 \
  --replay-repeats 2 \
  --t1-host-pinned-high-water 0.20 \
  --t1-host-pinned-low-water 0.10 \
  --t1-host-pinned-keepalive-ticks 512 \
  --disk-store-root /tmp/infer-kv-disk-trace-qwen3-singleprefix-remote-recall-v2 \
  --cluster-shared-root /tmp/infer-kv-shared-trace-qwen3-singleprefix-remote-recall-v2 \
  --trace-dir bench-output/2026-04-22-tier-kv-trace-qwen3-singleprefix-remote-recall-v2/traces \
  --log-path bench-output/2026-04-22-tier-kv-trace-qwen3-singleprefix-remote-recall-v2/infer.log \
  --out bench-output/2026-04-22-tier-kv-trace-qwen3-singleprefix-remote-recall-v2/summary.json
```

## Environment

- **Backend:** CUDA
- **Model:** `Qwen3-4B`
- **Hardware:** NVIDIA L4
- **Commit base:** `b70c03b` plus local runtime fixups in this tranche
- **Feature set:** `cargo build --release -p infer --bin infer`
- **Non-default runtime flags:** `--num-slots 16 --max-seq-len 4608 --mem-fraction-static 0.94 --chunked-prefill-size 4096 --max-prefill-tokens 16384`
- **Tier overrides:** `t1_high_water=0.20`, `t1_low_water=0.10`, `t1_keepalive_ticks=512`

## Results

### 1. Before the final fixes

| Run | Result |
|---|---|
| `bench-output/2026-04-22-tier-kv-trace-qwen3-c16-multiprefix16-storecheck-fixed/summary.json` | `kv_store=sub:0,done:0,fail:0,rej:0`; replay stayed `tier_src=h:36/d:0/r:0` |
| `bench-output/2026-04-22-tier-kv-trace-qwen3-singleprefix-disk-recall/summary.json` | `kv_store=sub:357,done:357,fail:0,rej:0`, but replay still fell back to cold prefill because slower-tier hits were marked `recompute_advised` |

### 2. After the byte-level T1 drain gate

| Run | Store activity | Recall outcome |
|---|---:|---|
| `bench-output/2026-04-22-tier-kv-trace-qwen3-c16-multiprefix16-storecheck-drainbytes/summary.json` | `kv_store=sub:45,done:45,fail:0,rej:0` | replay still `tier_src=h:21/d:0/r:0` |

Materialized objects from that run:

| Tier | Count |
|---|---:|
| T2 disk | `9` files under `/tmp/infer-kv-disk-trace-qwen3-c16-multiprefix16-storecheck-drainbytes` |
| T3 shared-fs | `36` files under `/tmp/infer-kv-shared-trace-qwen3-c16-multiprefix16-storecheck-drainbytes` |

### 3. After the slower-tier recall heuristic fix

| Run | Replay `tier_src` | `tier_recall` | `tier_promoted` | `tier_fallback` | `kv_store` |
|---|---|---:|---:|---:|---|
| `bench-output/2026-04-22-tier-kv-trace-qwen3-singleprefix-disk-recall-v2/summary.json` | `h:0/d:16/r:0` | `100.0%` | `16` | `0` | `sub:359,done:359,fail:0,rej:0` |
| `bench-output/2026-04-22-tier-kv-trace-qwen3-singleprefix-remote-recall-v2/summary.json` | `h:0/d:1/r:15` | `100.0%` | `16` | `0` | `sub:359,done:359,fail:0,rej:0` |

### 4. Current prefetch status

- Live runtime path is still **on-demand staged fetch**, not proactive prefetch.
- Evidence:
  - runtime uses `submit_fetch(...)` in `infer/src/scheduler/cuda/runtime.rs:622`
  - `submit_prefetch_plan(...)` exists only on the coordinator handle in `infer/src/kv_tier/coordinator.rs:571`
  - `TieredKvPolicy::allow_prefetch(...)` exists in `infer/src/kv_tier/policy.rs:40`
  - there is still no runtime call site for `submit_prefetch_plan(...)`

## Problems

- The first percentage-only T1 drain fix was necessary but not sufficient: it activated T1→T2/T3 store, but replay still stayed on host because the default slower-tier lookup heuristic was overestimating cold-prefill speed.
- `LookupHeuristics::default()` previously used `prefill_tokens_per_sec = 30_000`, which matched optimistic kernel-rate intuition rather than observed end-to-end long-prefill wall time on this model/host.
- Because `lookup_or_stage()` OR-ed `recompute_advised` across staged blocks, that optimistic default could downgrade every disk/shared-fs hit back to cold prefill.

## Learnings

- **T1 fix:** host-tier drain must be gated in bytes, not just rounded percentage watermarks; otherwise discrete block granularity can leave T1 forever “just under” high-water and never drain.
- **T2/T3 fix:** the slower-tier lookup model must use realistic end-to-end prefill speed, not an optimistic kernel-only number.
- **Current architecture:** slower-tier store and recall are now end-to-end live for all three tiers, but “prefetch” still means request-triggered staged fetch, not proactive background prefetch planning.
- **Throughput relevance:** canonical random-prompt `c16` stays cold on prefix reuse, so these tier fixes are correctness/coverage work first, not the next throughput lever.

## Artefacts

- Multiprefix store-check before heuristic fix: `bench-output/2026-04-22-tier-kv-trace-qwen3-c16-multiprefix16-storecheck-drainbytes/summary.json`
- Disk recall: `bench-output/2026-04-22-tier-kv-trace-qwen3-singleprefix-disk-recall-v2/summary.json`
- Remote recall: `bench-output/2026-04-22-tier-kv-trace-qwen3-singleprefix-remote-recall-v2/summary.json`
- Disk objects: `/tmp/infer-kv-disk-trace-qwen3-c16-multiprefix16-storecheck-drainbytes`
- Shared objects: `/tmp/infer-kv-shared-trace-qwen3-c16-multiprefix16-storecheck-drainbytes`
