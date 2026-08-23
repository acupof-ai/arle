# DSv4 c=1 decode graph: default ON — CUDA, 2026-08-23

> Status: Shipped default ON (`ARLE_DSV4_DECODE_GRAPH=0` selects the eager arm).

## Goal

c=1 decode ITL on DeepSeek-V4-Flash, TP=4, 32k agent prompts.

## Hypothesis

Capturing the 43-layer decode body into one CUDA graph per slot removes
~6300 per-step launches on the host side; expected ITL gain in the 10–20%
band seen on the Qwen3.6 TP decode graph.

## Parameters

```bash
python3 scripts/bench_throughput.py \
  --url http://localhost:8300 --model DeepSeek-V4-Flash-0731 \
  --prompts-jsonl bench-agent-32k-16x8.jsonl \
  --concurrency-grid 1,8,16 --requests-per-concurrency 16 \
  --max-tokens 256 --temperature 0 --timeout-seconds 900
```

- Binary: `1a48d179f`, build `c1-graph-v24b`
- Baseline: `ARLE_DSV4_DECODE_GRAPH=0` (eager); Treatment: same binary, default (graph armed)
- Prompt tokens: p50 28568; completion tokens 256 exact (ignore_eos)
- Trials: 1 per arm, sequential, same GPUs (0,1,2,4)

## Environment

- 8×H20 box, pod `sglang-test`; other GPUs held by other serves
- CUDA 12.x driver; `--comm-backend nccl` (oneshot off)
- DeepSeek-V4-Flash-0731 (NVFP4 experts) and `-FP8`, BF16 KV, TP=4, 4 slots/rank
- `--spec-type none`, mempool retain on

## Results

NVFP4 experts, 16 requests per point:

| c | arm | decode tok/s | ITL p50 ms | ITL p99 ms | TTFT p50 ms |
|---:|---|---:|---:|---:|---:|
| 1 | eager | 35.6 | 24.5 | 122.6 | 7926 |
| 1 | graph | **43.2** (+21.3%) | **22.3** (−9.0%) | **44.5** (−63.7%) | 7898 |
| 8 | eager | 22.4 | 40.5 | 99.4 | 11409 |
| 8 | graph | 22.4 | 40.8 | 99.0 | 11507 |
| 16 | eager | 17.8 | 51.4 | 104.0 | 22270 |
| 16 | graph | 17.7 | 51.9 | 103.9 | 22530 |

FP8 experts, same box and workload, 8 requests:

| c | arm | decode tok/s | ITL p50 ms | ITL p99 ms |
|---:|---|---:|---:|---:|
| 1 | eager | 52.4 | 18.5 | 41.9 |
| 1 | graph | **59.5** (+13.5%) | **16.3** (−11.9%) | 42.0 |

All points 16/16 or 8/8 completed, zero errors. c=8 and c=16 are unchanged —
the graph only arms at c=1, so those points are a no-op control that confirms
the gate.

Capture audit on the shipped build: **0 alloc nodes, 0 free nodes, 0 host
memcpy nodes, 0 host callback nodes**; `alloc warnings since launch: 0`.

Correctness (MMLU 5-shot, 200 samples, greedy, same seed):

| arm | accuracy | invalid | per-item diffs vs eager |
|---|---:|---:|---:|
| eager | 171/200 = 85.5% | 0 | — |
| graph | 171/200 = 85.5% | 0 | **0** |

DSpark control (`--spec-type dspark`, draft `-DSpark-draft-fp8`), run in both
orders because the first ordering showed an 11% ITL gap that did not reproduce:

| arm | decode tok/s | ITL p50 ms | ITL p99 ms | graph captures |
|---|---:|---:|---:|---:|
| graph off | 15.4 | 65.8 | 109.8 | 0 |
| graph on | 15.2 | 66.4 | 110.2 | 0 |

`captures = 0` in both arms is the direct proof that the `self.dspark.is_some()`
gate (`crates/infer-cuda/src/executor/dsv4.rs:235`) keeps DSpark on the eager
path; the 0.9% p50 delta is run-order noise. The first ordering's 74.2 vs 65.9
was cold-start, not a regression.

Needle ladder (`scripts/needle_gate.py 512,4096,16384,32768 3 0.0`,
`NEEDLE_MAX_TOKENS=48`, chat endpoint), same binary both arms:

| length | graph off | graph on |
|---:|---|---|
| 512 | exact=0 partial=3 miss=0 | exact=0 partial=3 miss=0 |
| 4096 | exact=0 partial=3 miss=0 | exact=0 partial=3 miss=0 |
| 16384 | exact=0 partial=3 miss=0 | exact=0 partial=3 miss=0 |
| 32768 | exact=0 partial=3 miss=0 | exact=0 partial=3 miss=0 |

Envelope identical, `miss=0` at every length including 32768, which crosses
page boundaries and exercises prefix restore. `partial` rather than `exact` is
the harness scoring the model's prose answer (`...stated earlier: "738291"`)
instead of a bare needle token; it is the same in both arms and is not a
correctness signal. Treatment arm proven engaged: 48 `captured slot` lines in
the graph arm, **0** in the eager arm.

## Problems

Eleven independent defects stood between "capture attempted" and "replay
correct at 0 alloc nodes", each found by the next probe after the previous fix:

1. `record_pipeline_fence` probed the event pool with `cuEventQuery` during
   capture → `STREAM_CAPTURE_UNSUPPORTED`, which also invalidated the capture.
   Fix: skip the pool probe when `cuStreamIsCapturing` (`cac2729d4`).
2. The 3 hash-routed MoE layers `clone_htod` the token ids per step → 3 host
   memcpy nodes, rejected by the audit. Fix: read the persistent pre-replay
   buffer in graph mode (`82391a0fd`).
3. Replay advanced `slot.seq_len` only; `compressed.seq_len` and
   `fp8_kv_comp_packed_rows` drifted. Fix: `advance_after_replay` (`ca602dfc8`).
4. `graph_stream_clone` returned buffer 0 (the embedding), so the LM head
   sampled from the input. Fix: last layer's `ffn_stream` index (`f11e2cf0f`).
5. `CudaSlice::clone()` in cudarc is a D2D copy into a fresh allocation, so
   every "persistent" graph buffer handed to a kernel was a throwaway copy;
   replay wrote freed memory and the LM head read a frozen buffer. Fix:
   `StepBuf::Alias` over `upgrade_device_ptr` in `ManuallyDrop` (`c72c6c1a6`).
6. A new request's `slot.reset()` re-arms the SW-ring bootstrap, but the slot's
   graph kept replaying the post-bootstrap body. Fix: drop the slot's graph on
   reset (`8484ad783`).
7. 1296 alloc / 1295 free nodes in the captured body (every per-step
   `uninit`/`alloc_zeros` in the layer loop, MoE scratch, mHC params). Under
   `AUTO_FREE_ON_LAUNCH` each replay re-allocated them through the async pool
   and the p99 stalled at 156 ms. Fix: keyed persistent scratch
   (`GraphBufKey = (layer, GraphSlot)`) plus `StepBuf`/`StepSlice`, so graph
   mode aliases a model-owned buffer and eager mode still allocates
   (`c63517aaa`, then the tail down to zero).
8. The CSA key-cache pack baked its `newly_packed > 0` branch and row range
   into the graph, so replay either skipped the write or repeated the captured
   row. Fix: a device-gated one-row pack kernel keyed on `start_pos_device`,
   plus advancing `dsa_official.packed_rows` on replay (`c8ac2542a`).
9. Capture rollback called `slot.truncate`, which rewinds the FlashMLA pool
   cursor that `prepare_kv_batch` had already advanced; eager fallback in the
   same tick then broke the `pool seq_len == append_pos` invariant. Fix:
   `rewind_host_counters` touches host counters only (`5cf706dae`).
10. The batched (n>1) decode lane replaced and freed the same per-layer O-LoRA
    staging that a c=1 graph had captured, so returning to c=1 replayed freed
    addresses. Fix: fixed n=1 staging inside the fused scratch (`5cf706dae`).
11. **The one that survived four versions.** Six MMLU items were graph-wrong /
    eager-right, reproducibly. Root cause: a prefix-cache restore (and a KV-tier
    promote) hands a slot to a new occupant with a new page band and fresh
    ring/compressor state, while the previous occupant's captured graph stayed
    armed and replayed over it. `ARLE_DISABLE_PREFIX_CACHE=1` collapsing the
    diff to 1 item was the probe that located it. Fix: drop the slot's graph on
    restore and on promote (`450a7d208`).

Defects 1, 2 and 5 were located with an `LD_PRELOAD` shim on
`cuMemcpyHtoDAsync_v2` (hooked through `dlsym`, since cudarc loads libcuda
dynamically) that prints a backtrace when the stream is capturing; the release
binary is symbol-stripped, so the shim's frames were resolved against a
`strip = "none"` build with `nm`.

Defect 7's fix regressed once: the constructor sized the n=1 O-LoRA staging
from `dsv4_oproj_group_dims(config)` while the runtime used the TP-local
`shape.cols_per_group`, so the mismatch re-introduced 86 alloc nodes. The
staging now resizes on the eager warm step when the loaded table dims differ
(`7b761258b`).

## Learnings

**Ship default ON.** The gain is real at c=1 on both quantization lanes
(+21.3% NVFP4, +13.5% FP8 decode tok/s) and the p99 collapse (122.6 → 44.5 ms)
matters more than the p50 for an agent workload. c≥8 is untouched because the
gate is c=1-only, and DSpark is provably untouched (0 captures).

**Zero alloc nodes is the precondition, not an optimization.** At 1296 alloc
nodes the graph was 23% *slower* than eager. The whole gain lives in removing
per-step allocation from the replay, not in removing launch overhead. Any
future graph on this family should audit `mem_alloc_nodes == 0` before
measuring anything.

**A captured graph is bound to a slot's page band, not just to the slot.**
Every path that re-points a slot at different pages — request reset, prefix
restore, tier promote — must drop that slot's graph. Missing one of the three
produced a 3% MMLU miss rate that four rounds of numerical debugging could not
explain, because the bug was in occupancy bookkeeping rather than in any
kernel.

**`CudaSlice::clone()` is a copy, not an alias.** This is the single fact that
cost the most debugging time. Any "persistent buffer" plumbed into a graph
must be aliased through `upgrade_device_ptr` + `ManuallyDrop`.

Remaining, not blocking: (1) delete the host-side row counters that mirror
`start_pos_device` — they are now written twice, once by the eager path and
once by `advance_after_replay`; (2) a dry capture at executor build that fails
the serve on any alloc or host node, so a future regression is caught at boot
rather than by a bench.

## Artifacts

- `/host/arle-ops/runs/c1g/bench-{graph,eager}-cN/tp.json` (NVFP4 c=1/8/16)
- `/host/arle-ops/runs/c1g/bench-{graph,eager}-fp8/tp.json` (FP8 c=1)
- `/host/arle-ops/runs/c1g/bench-dspark-r2-{0,1}.json`
- `/host/arle-ops/runs/c1g/mmlu-{graph,eager}-r2/`
- `/host/arle-ops/runs/c1g/needle_ab.out`

## Follow-up — 2026-08-23, codex round 3

One P1 survived two prior review rounds and both bench arms:
`advance_decode_len` (`crates/infer-cuda/src/attention/dsa.rs:967`) divided by
the raw `compress_ratio`. `SparseIndexed` layers carry `compress_ratio == 0`
and share the indexer at ratio 1 (`attention/dsa.rs:761`), and only
`SlidingWindow` returns early, so the first graph replay on a GLM-family
checkpoint would divide by zero and panic. It never showed on
DeepSeek-V4-Flash because that checkpoint has no `SparseIndexed` layer, and the
eager path never calls this function at all — its only two callers are
`advance_after_replay` and `truncate_decode_len`, both introduced by this work.

Fix: `total_len / ratio.max(1)`, which reproduces the eager path's absolute
`start_pos + seq_len` (`attention.rs:5629`) for the shared indexer.

The graph remains unvalidated on GLM-5.2 / `SparseIndexed`; that family's
verification is pending-remote independently of this work.

Rule: a counter helper written for one model family must be checked against
every `mode`/`ratio` combination the family enum admits, not only the one the
bench model exercises. A `.max(1)` in the caller is not protection when the
callee divides first.
