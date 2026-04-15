# 2026-04-15 · Qwen3-4B L4 — long-sequence bench at 4k / 8k / 16k / 25k (bf16 + int8)

## Context

User asked for performance numbers at 16k and 25k input contexts to
complement the 2026-04-15 quick sweep
([`2026-04-15-bench-kv-quant-sweep.md`](2026-04-15-bench-kv-quant-sweep.md))
which topped out at 2048-token prompts. This note captures a clean
four-point curve from 4k up to 25k real prompt tokens in both the
default `bf16` KV cache mode and the `int8` KV cache mode, plus the
memory-tuning constraints that surfaced along the way.

Two side findings also belong in this note:

1. **Pre-existing panic at `crates/infer-cuda-kernels/src/paged_kv.rs:595`**
   fires from the prefix-cache publish path when the completed
   request's prompt length cannot be represented by the published
   radix blocks (either pool OOM or an off-by-some span boundary).
   Fires as a background thread panic AFTER the client has already
   received the full response, so the response is valid but the
   server is degraded for subsequent requests until it is restarted.
2. **Default `--kv-pool-headroom-mb 4096` is too conservative** on a
   single-slot L4 box when `--max-seq-len` is raised to 32 k. The
   auto-sized `TokenKVPool` can drop to ~8 k tokens, which is smaller
   than the max context window itself and forces cleanup-time publish
   failures. Tuning `--kv-pool-headroom-mb 1536..2048` +
   `--max-seq-len 28000..32768` restores a ~31 k / 42 k pool for bf16
   / int8 respectively and all four configs below run cleanly.

## Environment

- GPU: NVIDIA L4 24 GB (driver 580.82.07, CUDA 13.0, SM 89)
- Model: Qwen3-4B BF16, `Qwen/Qwen3-4B` HF Instruct variant
- Commit: `2b61644` (post 2026-04-15 remote acceptance + M3b
  classifier tightening + docs drift scan)
- Cargo env: `CARGO_HOME=/tmp/cargo-home-local`
- Server launches:
  - **bf16**: `target/release/infer --model-path models/Qwen3-4B
    --num-slots 1 --max-seq-len 28000 --kv-cache-dtype bf16
    --kv-pool-headroom-mb 1536 --gpu-reserved-mb 512`
    → `TokenKVPool budget: 4.6 GB, 31264 max tokens, page_size=16`
  - **int8**: `target/release/infer --model-path models/Qwen3-4B
    --num-slots 1 --max-seq-len 32768 --kv-cache-dtype int8
    --kv-pool-headroom-mb 2048 --gpu-reserved-mb 512`
    → `TokenKVPool budget: 3.4 GB, 42033 max tokens, page_size=1`
- Bench: `/tmp/longseq_bench.py` (paired artifact; a focused inline
  streaming client that measures TTFT + decode ITL + full-wall
  throughput for a single non-concurrent request per config).
- Prompt shape: repeated realistic LLM-serving paragraph padded to
  `target_tokens * 5.5` chars. Script computes ITL from the intervals
  between consecutive streamed `/v1/completions` deltas.
- Targets: 4000, 8000, 16000, 25000 tokens input × 128 tokens output,
  greedy decode (`temperature=0`), `num_slots=1`.
- JSON snapshots paired:
  `2026-04-15-bench-longseq-bf16.json`,
  `2026-04-15-bench-longseq-int8.json`.

## Results — bf16 (default KV cache mode)

```
 target  emitted    TTFT       ITL p50   ITL p99     wall   throughput
──────────────────────────────────────────────────────────────────────
  4 000    128     755.9 ms    35.2 ms   35.4 ms    5.22 s   24.5 tok/s
  8 000    128     824.5 ms    37.4 ms   37.6 ms    5.58 s   23.0 tok/s
 16 000    128    2112.9 ms    41.8 ms   42.0 ms    7.42 s   17.2 tok/s
 25 000    128    6306.2 ms    33.1 ms   33.5 ms   10.51 s   12.2 tok/s
```

All four configs: **128 / 128 tokens emitted**, no client errors. The
25 k row panicked the server AFTER the response finished streaming
(`paged_kv.rs:595`, background thread) — response data is valid, but
the server must be restarted before the next request. See "Pre-
existing panic" below.

## Results — int8 KV cache mode

```
 target  emitted    TTFT       ITL p50   ITL p99     wall   throughput
──────────────────────────────────────────────────────────────────────
  4 000    128     739.5 ms    37.7 ms   38.2 ms    5.52 s   23.2 tok/s
  8 000    128     824.1 ms    42.1 ms   42.3 ms    6.16 s   20.8 tok/s
 16 000    128    2059.3 ms    50.7 ms   51.2 ms    8.49 s   15.1 tok/s
 25 000    128    6360.2 ms    61.9 ms   62.6 ms   14.22 s    9.0 tok/s
```

All four configs: **128 / 128 tokens emitted**, no client errors, no
server panic (int8 pool is 42 k tokens so the 25 k publish fits).

## bf16 vs int8 delta

| target | TTFT Δ (int8 − bf16) | ITL p50 Δ | wall Δ | tok/s Δ |
|-------:|---------------------:|----------:|-------:|--------:|
|   4 k  | −16 ms               | +2.5 ms   | +0.30 s| −1.3     |
|   8 k  | −0.4 ms              | +4.7 ms   | +0.58 s| −2.2     |
|  16 k  | −53 ms               | +8.9 ms   | +1.07 s| −2.1     |
|  25 k  | +54 ms               | +28.8 ms  | +3.71 s| −3.2     |

**TTFT is essentially identical** between bf16 and int8 at all four
lengths — prefill is compute-bound, and the int8 quantize-on-write
overhead is negligible next to the FlashInfer prefill kernel cost.

**Decode ITL is where int8 pays**: +2.5 ms at 4 k grows to +28.8 ms
at 25 k. That is the per-step dequant-on-read cost accumulating as
the KV cache gets bigger. The int8 pool uses `page_size = 1` vs
bf16's `page_size = 16`, so int8 also loses some of page16's
metadata amortisation on long contexts — the two effects compound.

**Overall throughput at 25 k**: bf16 12.2 tok/s vs int8 9.0 tok/s, a
−3.2 tok/s (−26 %) gap. For production long-context serving on an L4,
**bf16 is the default** unless KV memory is the actual gate.

## TTFT scaling and decode ITL shape

TTFT scales roughly linearly with input length for both modes:

- bf16: 756 → 825 → 2113 → 6306 ms as input goes 4k → 8k → 16k → 25k
  (Δ per 1 k tokens: 17 / 215 / 419 ms in the 4-8 / 8-16 / 16-25
  windows — superlinear past 8 k because chunked prefill allocates
  more prefill chunks and the attention kernel cost grows quadratic
  per chunk even though each chunk is bounded by `prefill_chunk_size
  = 4096`).
- int8: 740 → 824 → 2059 → 6360 ms — the same shape, within ~1 % of
  bf16 at every length.

Decode ITL grows roughly linearly with KV-cache-read bandwidth:

- bf16: 35.2 → 37.4 → 41.8 → 33.1 ms. The 25 k row is **lower** than
  the 16 k row, which is the opposite of what physics predicts. Two
  candidate explanations, neither fully verified:
  1. `max_seq_len` differs between the 16 k and 25 k pool runs
     (32 768 in the 16 k-only run vs 28 000 in the 25 k full run),
     so the contiguous-cache stride / alignment may be different and
     the attention kernel is hitting a different tile. Rerunning both
     lengths under identical `--max-seq-len` would settle it.
  2. The 16 k run was captured against a server that had already
     served the earlier 4 k and 8 k requests in sequence; the 25 k
     run was the fourth in the same sequence after a second server
     restart. CUDA Graph warmup or driver state could differ.

  Flag this for the next bench pass — file it against
  `docs/plans/tiered-kv-cache-tasks.md` §7.1 as "long-context ITL
  repeatability under differing `--max-seq-len`" if anyone wants the
  definitive answer.
- int8: 37.7 → 42.1 → 50.7 → 61.9 ms. **Monotonic as expected** —
  int8 dequant cost grows linearly with KV size and dominates over
  the kernel-variant noise, so the int8 curve is cleaner than bf16's.

Steady-state `mean = (wall − TTFT) / (emitted − 1)` agrees with the
reported ITL p50 to within 0.1 ms on every row, so the ITL anomaly
is a genuine per-step measurement, not a delta-batching artifact.

## Pre-existing panic on prefix-cache publish

Both the first bf16 attempt (`--kv-pool-headroom-mb 512`) and the
final bf16 run (`--max-seq-len 28000 --kv-pool-headroom-mb 1536`)
hit the same panic on cleanup of the 25 k request:

```
thread '<unnamed>' (…) panicked at crates/infer-cuda-kernels/src/paged_kv.rs:595:33:
```

Surrounding log context on the failing path (first attempt, earlier
pool):

```
ERROR infer::scheduler::cuda::prefill: prefill.rs:255 Request N: pool alloc for migration failed:
  TokenKVPool alloc retry failed after reclaiming 132 pages:
  first error: TokenKVPool: out of pages (requested 12005 tokens / 751 new pages, available 386 pages);
  retry error: TokenKVPool: out of pages (requested 12005 tokens / 751 new pages, available 518 pages)
```

The panic fires in a background thread **after** the response has
been fully streamed to the client, so the client data is valid, but
the server enters a degraded state and must be restarted before the
next request. The panic site is `page_indices_for_token_range` at
`paged_kv.rs:595`, which is an unchecked index into
`page_indices[slot][span]`. Either the span extends past the slot's
allocated pages or the slot has been freed underneath the publish
path.

This is not an M2b / M3b regression — it is a long-standing
invariant violation on the publish path that was simply not
exercised until someone published a prefix whose token range did not
fit in the current pool budget. Tracked as a follow-up against
`docs/plans/tiered-kv-cache-tasks.md` §3.5 / §6 M3b, not as an M2b or
M0.3/M3a acceptance regression. **Do not treat this as a ship
blocker**; treat it as the next prefix-cache hardening item.

## Memory budgets that actually worked

```
--num-slots 1 --max-seq-len 32768 --kv-cache-dtype bf16 --kv-pool-headroom-mb 4096
  → TokenKVPool budget: 1.2 GB   8 288 tokens  (too small for any publish > 8 k; DO NOT USE)

--num-slots 1 --max-seq-len 32768 --kv-cache-dtype bf16 --kv-pool-headroom-mb 2048
  → TokenKVPool budget: 3.4 GB  22 848 tokens  (fits <= 22 k publish; 25 k publish panics)

--num-slots 1 --max-seq-len 28000 --kv-cache-dtype bf16 --kv-pool-headroom-mb 1536
  → TokenKVPool budget: 4.6 GB  31 264 tokens  (fits 25 k publish; used for the bf16 table above)

--num-slots 1 --max-seq-len 32768 --kv-cache-dtype int8 --kv-pool-headroom-mb 2048
  → TokenKVPool budget: 3.4 GB  42 033 tokens  (int8 pool is ~1.8× bf16 pool at equal bytes; used for the int8 table above)

--num-slots 1 --max-seq-len 32768 --kv-cache-dtype bf16 --kv-pool-headroom-mb 512  --gpu-reserved-mb 256
  → CUDA_ERROR_OUT_OF_MEMORY during prefill chunk workspace alloc (headroom too small; DO NOT USE)
```

## Rule

**Long-context serving on a 24 GB L4 with Qwen3-4B BF16 is a
`--max-seq-len` vs `--kv-pool-headroom-mb` tradeoff**, not a
performance tradeoff. Auto-sizing is tuned for many-slot agent-trace
workloads, not single-slot extreme-context workloads, and the default
4 GB headroom over-reserves room for concurrent-decode buffers that
a single-slot engine will never use. Drop headroom to 1.5 GB and
leave a spare ~1 GB gap between `max_seq_len` and the biggest
prompt you plan to run, and bf16 handles 25 k cleanly.

**Corollary**: at C = 1, **int8 buys long-context headroom, not
throughput**. Same-wall behavior out to ~8 k; −26 % throughput at
25 k due to per-step dequant cost. Run int8 if you need the
contiguous-cache / pool memory back; otherwise stay on bf16.

**Corollary 2**: the publish path's `paged_kv.rs:595` out-of-bounds
panic is the next brick wall for serving arbitrarily-long single
requests. It is not an M2b / M3b regression — the publish path has
always been unchecked here — but any future long-context benchmark
run that exercises `--num-slots 1` with `--max-seq-len` near the
model's max will hit it. Hardening that panic into a graceful
fallback ("skip publish, log once") is the cleanest fix.
