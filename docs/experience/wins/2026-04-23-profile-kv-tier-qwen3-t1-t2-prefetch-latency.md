# Qwen3-4B L4 tiered-KV T1/T2 latency and deferred prefetch validation

## Goal

- Validate three runtime behaviors on `Qwen3-4B` / L4 after the direct-host and deferred-prefetch scheduler changes: T1 host staged readmission latency, T2 disk staged readmission latency, and best-effort deferred prefetch while requests wait for admission.

## Hypothesis

- Host-only staged prefixes should bypass the coordinator fetch queue and resume almost immediately.
- Disk-backed staged prefixes should still promote through the same staged-prefix path, but with a higher ready latency than T1.
- Deferred requests with a slower-tier staged prefix should be able to queue a fetch before a slot becomes free, so the first admitted replay can read from T1 instead of disk.

## Commands

T1-focused host-only validation:

```bash
python3 scripts/trace_tier_kv.py \
  --model-path /root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
  --port 8062 \
  --num-slots 16 \
  --max-seq-len 4608 \
  --mem-fraction-static 0.94 \
  --chunked-prefill-size 4096 \
  --max-prefill-tokens 16384 \
  --max-tokens 8 \
  --warm-prefixes 1 \
  --warm-repeats 3 \
  --churn-requests 16 \
  --replay-prefixes 1 \
  --replay-repeats 2 \
  --t1-host-pinned-high-water 0.99 \
  --t1-host-pinned-low-water 0.97 \
  --t1-host-pinned-keepalive-ticks 4096 \
  --disk-store-root /tmp/infer-kv-disk-trace-qwen3-t1-speed-v2-20260423 \
  --trace-dir bench-output/2026-04-23-tier-kv-trace-qwen3-t1-speed-v2/traces \
  --log-path bench-output/2026-04-23-tier-kv-trace-qwen3-t1-speed-v2/infer.log \
  --out bench-output/2026-04-23-tier-kv-trace-qwen3-t1-speed-v2/summary.json
```

T2-focused disk recall validation:

```bash
python3 scripts/trace_tier_kv.py \
  --model-path /root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
  --port 8063 \
  --num-slots 16 \
  --max-seq-len 4608 \
  --mem-fraction-static 0.94 \
  --chunked-prefill-size 4096 \
  --max-prefill-tokens 16384 \
  --max-tokens 8 \
  --warm-prefixes 1 \
  --warm-repeats 3 \
  --churn-requests 64 \
  --replay-prefixes 1 \
  --replay-repeats 2 \
  --t1-host-pinned-high-water 0.20 \
  --t1-host-pinned-low-water 0.10 \
  --t1-host-pinned-keepalive-ticks 512 \
  --disk-store-root /tmp/infer-kv-disk-trace-qwen3-t2-speed-20260423 \
  --trace-dir bench-output/2026-04-23-tier-kv-trace-qwen3-t2-speed/traces \
  --log-path bench-output/2026-04-23-tier-kv-trace-qwen3-t2-speed/infer.log \
  --out bench-output/2026-04-23-tier-kv-trace-qwen3-t2-speed/summary.json
```

Deferred-prefetch validation under blocked admission:

```bash
python3 scripts/trace_tier_kv.py \
  --model-path /root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
  --port 8064 \
  --num-slots 4 \
  --max-seq-len 4608 \
  --mem-fraction-static 0.94 \
  --chunked-prefill-size 4096 \
  --max-prefill-tokens 16384 \
  --max-tokens 8 \
  --warm-prefixes 1 \
  --warm-repeats 3 \
  --churn-requests 32 \
  --replay-prefixes 1 \
  --replay-repeats 4 \
  --replay-concurrency 4 \
  --replay-blockers 4 \
  --replay-blocker-max-tokens 256 \
  --t1-host-pinned-high-water 0.20 \
  --t1-host-pinned-low-water 0.10 \
  --t1-host-pinned-keepalive-ticks 512 \
  --disk-store-root /tmp/infer-kv-disk-trace-qwen3-prefetch-20260423 \
  --trace-dir bench-output/2026-04-23-tier-kv-trace-qwen3-prefetch/traces \
  --log-path bench-output/2026-04-23-tier-kv-trace-qwen3-prefetch/infer.log \
  --out bench-output/2026-04-23-tier-kv-trace-qwen3-prefetch/summary.json
```

## Environment

- **Backend:** CUDA
- **Model:** `Qwen3-4B`
- **Hardware:** NVIDIA L4
- **Commit base:** `42ce889` plus local direct-T1 / deferred-prefetch scheduler changes in this tranche
- **Feature set:** `cargo build --release -p infer --bin infer`
- **Non-default runtime flags:** `--num-slots 16 --max-seq-len 4608 --mem-fraction-static 0.94 --chunked-prefill-size 4096 --max-prefill-tokens 16384`

## Capture params

- Runtime trace source: scheduler logs + `/v1/stats` snapshots from `scripts/trace_tier_kv.py`
- Prompt shape: `4096` prompt tokens, `8` output tokens for warm/churn/replay; blocker phase uses `256` output tokens
- Focus window: staged-prefix admission and replay after churn, not full steady-state throughput

## Bench anchor

- Canonical throughput anchor for the same code/workload: `docs/experience/wins/2026-04-23-bench-guidellm-qwen3-4b-l4-c16-tier-prefetch-42ce889.md`

## Results

| Run | Evidence | Result |
|---|---|---|
| T1 host | `bench-output/2026-04-23-tier-kv-trace-qwen3-t1-speed-v2/infer.log` | Request `19` resumed through `staged_prefix=256`, then `staged prefix ready in 16.3ms src=h:16/d:0/r:0 waiters=1`; the next replay was already `radix_gpu_attach=4096` |
| T2 disk | `bench-output/2026-04-23-tier-kv-trace-qwen3-t2-speed/infer.log` | Request `67` resumed through `staged_prefix=256`, then `staged prefix ready in 37.6ms src=h:0/d:16/r:0 waiters=1`; the next replay was already `radix_gpu_attach=4096` |
| Deferred prefetch | `bench-output/2026-04-23-tier-kv-trace-qwen3-prefetch/infer.log` | `Prefetch 109 queued: matched=192 src=h:0/d:12/r:0`, then `Prefetch 109 ready: 17.8ms materialized=12 src=h:0/d:12/r:0`; the first replay hit `staged prefix ready in 51.3ms src=h:12/d:0/r:0`, and later replays were `radix_gpu_attach=192` |

Replay latency summary from the trace harness:

| Run | Replay elapsed (ms) | Notes |
|---|---|---|
| T1 host | `[903.6, 298.2]` | first replay includes staged-prefix resume + remaining prompt prefill; second replay is full GPU reuse |
| T2 disk | `[936.3, 299.1]` | disk fetch cost raises staged-ready latency by `21.3ms` vs T1, but end-to-end replay shape stays similar |
| Deferred prefetch | `[16125.0, 16165.1, 16229.7, 16229.3]` | replay started while four blocker requests occupied all slots, forcing the prefetch to happen before admission |

Tier counters from the trace summaries:

| Run | Key counters |
|---|---|
| T1 host | `kv_store=sub:0,done:0,fail:0,rej:0`; no disk spill was required |
| T2 disk | `tier_recall=100.0%`, `tier_src=h:0/d:16/r:0`, `kv_store=sub:281,done:281,fail:0,rej:0` |
| Deferred prefetch | `kv_store=sub:120,done:120,fail:0,rej:0`; final stats snapshot is not sufficient to prove prefetch, so the runtime log lines above are the authoritative evidence |

## Top-N kernels

- Not collected in this run. This is a scheduler/runtime trace, not an `nsys` / `ncu` capture.

## Launches per token

- Not collected in this run. The question here is tier readmission behavior, not kernel launch density.

## Findings

- Direct host staged readmission is now a real fast path. Host-only staged prefixes no longer need to round-trip through the coordinator fetch queue before promotion.
- T2 disk staged readmission is working on the same adopt/promote path as T1; the observable extra cost in this harness is about `21.3ms` (`37.6ms - 16.3ms`) at the staged-prefix-ready point.
- Deferred prefetch is now real. The runtime can queue a slower-tier fetch while a request is still deferred, materialize the returned blocks into T1, and then let the first admitted replay consume host-pinned blocks instead of going back to disk.

## Problems

- These traces validate behavior, not peak bandwidth. The reusable staged prefix in the T1/T2 runs was `256` tokens rather than a full `4096` token prompt.
- The canonical `c16` benchmark lane remains prefix-cold, so this trace tranche does not explain the remaining throughput gap versus `sglang`.

## Learnings

- For host-only staged prefixes, bypassing the fetch queue is the right canonical path; keeping them on the slower-tier machinery only adds avoidable latency.
- For slower tiers, prefetch proof needs explicit queued/ready/materialized runtime events; a final `/v1/stats` snapshot can miss the transient fetch activity entirely.
- Deferred prefetch helps correctness and reuse-heavy recall latency, but it does not move the canonical `c16` throughput lane unless the workload actually reuses prefixes.

## Artefacts

- T1 summary: `bench-output/2026-04-23-tier-kv-trace-qwen3-t1-speed-v2/summary.json`
- T1 log: `bench-output/2026-04-23-tier-kv-trace-qwen3-t1-speed-v2/infer.log`
- T2 summary: `bench-output/2026-04-23-tier-kv-trace-qwen3-t2-speed/summary.json`
- T2 log: `bench-output/2026-04-23-tier-kv-trace-qwen3-t2-speed/infer.log`
- Prefetch summary: `bench-output/2026-04-23-tier-kv-trace-qwen3-prefetch/summary.json`
- Prefetch log: `bench-output/2026-04-23-tier-kv-trace-qwen3-prefetch/infer.log`

