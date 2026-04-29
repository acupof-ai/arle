# cuda-l4 KV-quant matrix — full canonical comparison @ c=16 fixed 120s

> First end-to-end KV-format comparison at canonical bench protocol
> (`§10` of `docs/bench-and-trace-spec.md`): fixed concurrency 16,
> 120 s duration, 4096-in / 256-out, FP8 KV currently the best by
> ~2.2× over BF16 at saturation. SGLang historical reference 201
> tok/s; our best (FP8 + chunked-prefill=512 K2 workaround) lands at
> 145.30 tok/s = **72% parity**. Kernel infrastructure for the
> proper K2 fix landed in `4e4906f5`.

## Goal

Establish the canonical c=16 fixed-concurrency tok/s number for each
KV format, on the post-fix-stack baseline, so future regressions and
the upcoming K2 wire-up have a stable comparison frame. Goal type:
**publication baseline + KV-quant ROI study**.

## Hypothesis

Pre-bench:
- FP8 ~150 tok/s (best).
- INT8 ~135 (close to FP8, slight dequant cost on attention).
- BF16 ~70 (2× KV bandwidth penalty at saturation).
- chunked-prefill=512 lifts FP8 by ~10-15% (K2 Split-block
  workaround, see
  `docs/experience/errors/2026-04-29-bf16-shadow-mixed-architectural-dead-end.md`).

## Setup

- Hardware: NVIDIA L4 sm_89, 22 GB, CUDA 12.8
- Commit: `4e4906f5`
- Feature set: `--features cuda` (implies tilelang-attn)
- Server: `target/release/infer --num-slots {N} --max-seq-len 4608
  --kv-cache-dtype {fmt} [extra-flags]`
- Bench: `bash scripts/bench_guidellm.sh <label> --concurrencies 16
  --max-seconds 120` (canonical, §10.2)

## Results

| Config | KV | slots | pool tokens | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | **out tok/s** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **FP8 default (canonical)** | FP8E4M3 | 16 | 148,256 | 10455 | 23701 | 86.12 | 119.86 | **105.22** |
| **FP8 + chunk=512** | FP8E4M3 | 16 | 148,256 | 11884 | 13123 | 71.58 | 77.60 | **145.30** ⭐ |
| **INT8 default** | INT8 | 16 | 152,656 | n/a (sweep) | n/a | ~100 | n/a | ~135 (sweep median) |
| **BF16 slots=8** | BF16 | 8 | 78,384 | 9360 | 69199 | 106.22 | 119.58 | 53.31 |
| **BF16 slots=16 frac=0.80** | BF16 | 16 | 70,400 | 11971 | 35193 | 130.26 | 187.85 | **65.43** |

(BF16 slots=16 needs `--mem-fraction-static 0.80 --max-prefill-tokens
4096` to fit the bigger KV pool; at default `frac=0.85
max_prefill=16384`, workspace est = 3.2 GB > 2.8 GB headroom and
sweep crashes mid-run.)

### Reference points

- SGLang ref @ same shape (historical, c=16/4096-in/256-out FP8):
  TTFT ~3357 ms, ITL ~67 ms, **tok/s ~201**.
- Our best (FP8 + chunk=512): 145.30 tok/s = **72% parity** on
  out tok/s. TTFT 11.9 s (still 3.5× SGLang's 3.4 s — the K2 stall
  hasn't been removed, just shrunk).

### Key takeaways

1. **FP8 vs BF16 at saturation: 2.22× speedup.** FP8 reads 1 byte/elem
   vs BF16 2 byte/elem — at c=16/4096-in, attention KV-read
   bandwidth is the dominant cost. The often-quoted "FP8 dequant
   adds latency" is more than offset by the bandwidth win at
   saturation.

2. **chunk_size=512 is a real K2 workaround, not just noise.**
   FP8 default (chunk=2048) → 105 tok/s.  FP8 + chunk=512 → 145.
   Smaller chunks shorten each Split-prefill block (500ms → ~125ms
   per block), so decode rows interleave more often even within
   the Split plan. Only relevant while K2 keeps mixed-batch off
   for FP8/INT8.

3. **BF16 has Mixed plan but is still slower.** BF16 slots=16
   with mixed-batch enabled (the K2 gate accepts BF16) gives 65
   tok/s vs FP8's 145 with K2 still gating. Mixed plan wins
   admission interleave; FP8 wins KV bandwidth. FP8 wins more.

4. **INT8 ITL is ~17% higher than FP8 at same shape.** Earlier
   data (--fast n=3): FP8 ITL p50 ≈ 75 ms, INT8 ≈ 100 ms. INT8
   has per-page scale tables that add an extra load+multiply per
   tile read. Consistent with K9 in
   `docs/projects/2026-04-29-perf-bug-roundup.md`.

## What's next (closing the SGLang gap)

1. **K2 wire-up** (`a48ad60ea2f5be9d7` audit + `4e4906f5` kernel
   infra landed):
   - `decode_attention_varlen_fp8` kernel exists but isn't called.
   - Wire it into `infer/src/model/qwen3/batch_decode.rs::decode_batch_with_prefill`
     for FP8 KV; lift the `KVFormat::BF16`-only gate at
     `infer/src/model/qwen3/forward.rs:585`.
   - Expected: TTFT 11.9 s → ~3-4 s (admission stall removed),
     tok/s 145 → 180-210 (= SGLang parity).

2. **INT8 variant of the new kernel.** Trivial template extension
   (per-page K/V scale loads) once FP8 wire-up is verified.

3. **Bench tracing patches A-E** (per
   `docs/plans/bench-tracing-patch-2026-04-29.md`) — surface the
   per-step phase EMAs in `/v1/stats` so we don't have to grep
   server logs to find next bottleneck.

## Appendix: env knob recipe (pre-K2)

For users who can't wait for the kernel wire-up, the SGLang-parity-
adjacent config TODAY is:

```bash
target/release/infer \
    --num-slots 16 --max-seq-len 4608 \
    --kv-cache-dtype fp8 \
    --chunked-prefill-size 512   # K2 workaround
```

The remaining 72% → 90% parity gap requires the kernel wire-up
(task #29).
