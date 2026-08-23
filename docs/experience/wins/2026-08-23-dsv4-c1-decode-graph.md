# DSv4 c=1 decode graph: correct, below the eager arm — CUDA, 2026-08-23

> Status: Shipped opt-in (`ARLE_DSV4_DECODE_GRAPH=1`); default stays eager.

## Goal

c=1 decode ITL on DeepSeek-V4-Flash-0731, TP=4, 32k agent prompts.

## Hypothesis

Capturing the 43-layer decode body into one CUDA graph per slot removes
~6300 per-step launches on the host side; expected ITL gain in the 10–20%
band seen on the Qwen3.6 TP decode graph.

## Parameters

```bash
python3 scripts/bench_throughput.py \
  --url http://localhost:8300 --model DeepSeek-V4-Flash-0731 \
  --prompts-jsonl bench-agent-32k-16x8.jsonl \
  --concurrency-grid 1 --requests-per-concurrency 8 \
  --max-tokens 256 --temperature 0 --timeout-seconds 900
```

- Baseline: `8484ad783`, build `c1-graph-v15`, `ARLE_DSV4_DECODE_GRAPH=0` (eager)
- Treatment: same binary, graph armed
- Prompt tokens: p50 28566 / min 28556 / max 28577
- Completion tokens: 256 exact (ignore_eos)
- Trials: 1 per arm, sequential, same GPUs (0,1,2,4)

## Environment

- 8×H20 box, pod `sglang-test`; GPUs 3/6/7 held by other serves
- CUDA 12.x driver; `--comm-backend nccl` (oneshot off)
- DeepSeek-V4-Flash-0731, NVFP4 experts, BF16 KV, TP=4, 4 slots/rank
- `--spec-type none`, mempool retain on

## Results

| concurrency | arm | completed | errors | decode tok/s | TTFT p50/p99 ms | ITL p50/p99 ms | delta |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | eager | 8 | 0 | 40.4 | 7838 / 7906 | 24.5 / 42.0 | — |
| 1 | graph | 8 | 0 | 28.4 | 8360 / 8411 | 30.2 / 156.1 | ITL p50 +23%, p99 +272% |

Correctness (MMLU 5-shot, 200 samples, greedy, same seed):

| arm | accuracy | invalid |
|---|---:|---:|
| eager | 171/200 = 85.5% | 0 |
| graph | 166/199 = 83.4% | 1 |

6/200 predictions differ; every one is graph-wrong / eager-right (items
14, 49, 73, 135, 162, 196). Three 1024-token generations per arm read
fluently in both arms; the graph arm shows a higher rate of dropped
letters inside words ("jaged", "clathered", "unaccold", "scrift",
"tremb") than the eager arm ("hoed", "kel", "scscribbled").

Raw artifacts: `/host/arle-ops/runs/c1g/bench-{graph,eager}/tp.json`,
`.../bench-*/long.txt`, `.../mmlu-{graph,eager}-r1/`.

## Problems

Six independent defects stood between "capture attempted" and "replay
correct", each found by the next probe after the previous fix:

1. `record_pipeline_fence` probed the event pool with `cuEventQuery`
   during capture → `STREAM_CAPTURE_UNSUPPORTED` and the capture was
   invalidated. Fix: skip the pool probe when `cuStreamIsCapturing`.
2. The 3 hash-routed MoE layers `clone_htod` the token ids per step →
   3 host memcpy nodes, rejected by the graph audit. Fix: read the
   persistent pre-replay buffer in graph mode.
3. Replay advanced `slot.seq_len` only; `compressed.seq_len` and
   `fp8_kv_comp_packed_rows` drifted. Fix: `advance_after_replay`.
4. `graph_stream_clone` returned buffer 0 (the embedding), so the LM head
   sampled from the input. Fix: last layer's `ffn_stream` index.
5. `CudaSlice::clone()` in cudarc is a D2D copy into a fresh allocation,
   so every "persistent" graph buffer handed to a kernel was a throwaway
   copy; replay wrote freed memory and the LM head read a frozen buffer.
   Fix: `StepBuf::Alias` over `upgrade_device_ptr` in `ManuallyDrop`.
6. A new request's `slot.reset()` re-arms the SW-ring bootstrap, but the
   slot's graph kept replaying the post-bootstrap body. Fix: drop the
   slot's graph on reset (the Qwen3.6 executor already does this).

Defects 1, 2 and 5 were located with an `LD_PRELOAD` shim on
`cuMemcpyHtoDAsync` that prints a backtrace when the stream is
capturing; the release binary is symbol-stripped, so the shim's frames
were resolved against a `strip = "none"` build with `nm`.

## Learnings

KILL as default, keep opt-in. The captured body carries 1296
alloc / 1295 free nodes (every per-step `uninit`/`alloc_zeros` in the
layer loop, MoE scratch, and mHC params); under `AUTO_FREE_ON_LAUNCH`
each replay re-allocates them through the async pool, and the p99 ITL
(156 ms vs 42 ms) says that path stalls. The one-sided MMLU misses and
the typo rate point at a remaining replay-vs-eager divergence that the
alloc churn may also explain (a pooled block reused before its reader
finishes is silent corruption, not a fault).

Next wall, in order: (1) a per-model `DecodeWorkspace` so the layer
loop performs zero per-step allocations and the audit reads
`0 alloc`; (2) delete the host-side row counters that mirror
`start_pos_device`; (3) a dry capture at executor build that fails the
serve on any alloc/host node. (1) is the precondition for re-measuring
ITL; the correctness diff is re-run on the same 200 items after it.
