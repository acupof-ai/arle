# Longctx 4k/c=4 prefill gap — chunk-size fix killed

## Priority & ROI

**Priority**: P0 diagnosis for the remaining CUDA longctx TTFT gap. The
current production baseline is already decode-parity with vLLM at this
shape, but TTFT is still slower.

**ROI basis**: the candidate fix was cheap to test by launch flags only:
`--chunked-prefill-size 4096` and a full-prompt budget control. If it
had closed >=30% of TTFT without hurting ITL/tok/s, the default runtime
envelope could be changed in one small scheduler commit.

**Negative case**: if larger chunks do not materially reduce TTFT, or if
they trade TTFT for worse decode/throughput, do not change defaults.

**Kill criteria**: H_LP2 needs >=30% TTFT p50 improvement vs
`p0prime-default-split-c4` with no >5% ITL or out tok/s regression.
Otherwise move to kernel/profile evidence instead of config tuning.

## Goal

Diagnosis: determine whether the 4k/c=4 TTFT gap is caused by
`chunked_prefill_size=2048` splitting each request into two chunks.

## Hypothesis

H_LP2: increasing `chunked_prefill_size` to 4096 should make each 4k
request a single chunk, reduce prefill steps, and move TTFT from ARLE
1976 ms closer to vLLM's 1177 ms.

H_LP1 control: GuideLLM's tokenizer produces 4097 prompt tokens, so a
full-row control with `chunked_prefill_size=4097` and
`max_num_batched_tokens=max_prefill_tokens=17408` should remove the
1-token tail and show whether the remaining issue is just the
16384-token step boundary.

## Source Survey

- `SchedulerConfig::resolve_runtime_envelope()` auto-picks
  `chunked_prefill_size` from HBM. On the 4070 Ti SUPER 16 GiB,
  `pick_chunked_prefill_size_for_hbm()` resolves to 2048.
- `max_prefill_tokens` defaults to `max_num_batched_tokens` (16384),
  not to chunk size, so the whole-step prefill budget is already wide.
- `PrefillBudget::from_scheduler_for_decode_slots()` enforces the
  total token/request budget. With `prefill_max_requests=None`, there
  is no operator cap on request count.
- `select_prefill_candidates()` selects multiple queued slots until
  token/page budget is exhausted. Cross-request batching is not missing.
- `prefill_reservation()` still caps each slot to one
  `prefill_chunk_size()` row per scheduler step. Unfinished rows are
  requeued by `finish_prefill_batch()`.
- `Qwen3PrefillContext` supports one in-flight prefill batch at a
  time. This preserves ordering but means the scheduler cannot overlap
  multiple prefill waves.

Observed logs confirm the implementation: ARLE packs multiple requests
when they are already queued, but the first request in a GuideLLM c=4
burst often arrives ~20-500 ms before the other three and launches a
single-row prefill immediately.

## Command

Baseline from [`2026-05-07-m3.9-mixed-policy-budget-fix.md`](2026-05-07-m3.9-mixed-policy-budget-fix.md):

```bash
target/release/infer --model-path infer/models/Qwen3-4B --port 8000 \
  --num-slots 8 --max-seq-len 12288 --max-prefill-tokens 16384

PATH=.venv/bin:$PATH scripts/bench_guidellm.sh p0prime-default-split-c4 \
  --concurrencies 4 --max-seconds 60 --warmup 10
```

H_LP2 scout:

```bash
target/release/infer --model-path infer/models/Qwen3-4B --port 8000 \
  --num-slots 8 --max-seq-len 12288 --max-prefill-tokens 16384 \
  --chunked-prefill-size 4096 --scheduler-mixed-policy split

PATH=.venv/bin:$PATH scripts/bench_guidellm.sh lp-hlp2-chunk4096-c4 \
  --concurrencies 4 --max-seconds 60 --warmup 10
```

H_LP1 full-row control:

```bash
target/release/infer --model-path infer/models/Qwen3-4B --port 8000 \
  --num-slots 8 --max-seq-len 12288 \
  --max-num-batched-tokens 17408 --max-prefill-tokens 17408 \
  --chunked-prefill-size 4097 --scheduler-mixed-policy split

PATH=.venv/bin:$PATH scripts/bench_guidellm.sh lp-hlp1-full4097budget-c4 \
  --concurrencies 4 --max-seconds 60 --warmup 10
```

## Environment

- **Backend:** CUDA
- **Model:** Qwen3-4B BF16, `infer/models/Qwen3-4B`
- **Hardware:** RTX 4070 Ti SUPER 16 GiB, CUDA 13.2, sm_89
- **Commit:** `a5bef64`
- **Feature set:** release CUDA build
- **Env:** `NVCC_CCBIN=/usr/bin/g++-14`,
  `INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python`,
  `TORCH_CUDA_ARCH_LIST=8.9`

## Results

| run | envelope delta | TTFT p50 | ITL p50 | out tok/s | completed in/out | incomplete in/out | plan labels |
|---|---|---:|---:|---:|---:|---:|---|
| baseline `p0prime-default-split-c4` | chunk=2048, max_prefill=16384 | 1976.4 ms | 19.27 ms | 153.83 | 131104 / 8192 | 0 / 0 | prefill=34, split=0, mixed=0 |
| H_LP2 `lp-hlp2-chunk4096-c4` | chunk=4096, max_prefill=16384 | 1983.8 ms | 19.23 ms | 153.77 | 131104 / 8192 | 0 / 0 | prefill=22, split=0, mixed=0 |
| H_LP1 full-row control | chunk=4097, max_prefill=17408, max_batch=17408 | 1551.5 ms | 21.09 ms | 135.16 | 118813 / 7424 | 12288 / 758 | prefill=20, split=11, mixed=0 |
| vLLM s8 reference | max-num-seqs=8 equivalent shape | 1177 ms | 18.8 ms | 159.1 | n/a | n/a | n/a |

## Delta vs Baseline

| metric | baseline | H_LP2 chunk4096 | delta | H_LP1 full-row | delta |
|---|---:|---:|---:|---:|---:|
| TTFT p50 | 1976.4 ms | 1983.8 ms | +0.4% slower | 1551.5 ms | -21.5% |
| ITL p50 | 19.27 ms | 19.23 ms | -0.2% | 21.09 ms | +9.4% slower |
| out tok/s | 153.83 | 153.77 | -0.0% | 135.16 | -12.1% |
| completed output tokens | 8192 | 8192 | 0% | 7424 | -9.4% |

H_LP2 is dead: reducing prefill plan count from 34 to 22 did not move
TTFT. The full-row H_LP1 control recovers only 21.5% TTFT and pays for
it with worse ITL, lower throughput, and incomplete output tokens.

## Problems

- The full-row control introduced `split=11` and worse decode latency.
  It is not a valid production default even though TTFT p50 improved.
- Larger chunks increase activation/workspace pressure and move the run
  closer to the max-batched-token edge; the service completed fewer
  requests in the same 60 s window.
- The vLLM TTFT target remains 31.8% faster even after the aggressive
  full-row control.

## Learnings

- **Chunk-size is not the primary 4k/c=4 gap.** The scheduler already
  has enough whole-step prefill budget; changing chunk size mainly
  changes plan count, not the dominant prefill wall time.
- **2048 vs 4096 is a false fix at this shape.** It removes bookkeeping
  but not the long-row prefill compute cost.
- **Full-row packing is a tradeoff, not a free win.** It helps TTFT but
  hurts ITL/out tok/s and causes incomplete work under the same bench
  duration.
- **Next evidence should be kernel-side.** The useful next step is nsys
  / ncu on the pure-prefill 4x4097 path after M_nsys P1, comparing
  ARLE's HD128 paged prefill kernels against vLLM/Triton attention at
  the same qlen/KV shape.

## Decision

No runtime default change. Keep production default at HBM-picked
`chunked_prefill_size=2048`, `max_prefill_tokens=16384`, and
`scheduler_mixed_policy=split`.

The H_LP2 fix candidate is killed. H_LP1 needs a separate design if we
want an opt-in TTFT/ITL tradeoff knob, but it is not the vLLM parity fix.

## Artefacts

- Baseline raw: `bench-output/2026-05-07-p0prime-default-split-c4/`
- H_LP2 raw: `bench-output/2026-05-07-lp-hlp2-chunk4096-c4/`
- H_LP2 server log:
  `bench-output/server-logs/2026-05-07T19-29-50-lp-hlp2-chunk4096-c4.log`
- H_LP1 full-row raw:
  `bench-output/2026-05-07-lp-hlp1-full4097budget-c4/`
- H_LP1 server log:
  `bench-output/server-logs/2026-05-07T19-32-55-lp-hlp1-full4097budget-c4.log`

## Rule

For low-concurrency longctx, test the exact prompt token count from
GuideLLM before changing chunk defaults. If the full-row control cannot
clear the kill threshold, stop tuning scheduler envelope flags and move
to kernel/profile evidence.
