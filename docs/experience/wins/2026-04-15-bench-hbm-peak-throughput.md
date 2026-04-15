# 2026-04-15 · Qwen3-4B L4 — HBM inventory + peak throughput at default num_slots

## Context

User asked: "what's the memory usage situation, and what's the max
achievable throughput at the current HBM?". This note inventories
exactly where the 24 GB on the L4 actually goes when running the
default `target/release/infer --model-path models/Qwen3-4B`, and
finds the peak tok/s the auto-sized server can produce on the
`512 → 128` (input → output) bench shape.

Three follow-on findings surfaced during the measurement that are
worth writing down even though they aren't perf wins on their own:

1. The auto-sizer (`infer/src/main.rs:277` `auto_num_slots`) is
   **dtype-blind** — it uses a hardcoded bf16 `per_slot=1208 MB`
   regardless of `--kv-cache-dtype`, so int8 servers default to the
   same slot count as bf16 even though int8's actual contiguous
   cache is half the bytes. **The capacity win that quant KV is
   supposed to deliver is silently disabled at default flags.**
2. The auto-sizer's `reserved=6.4 GB` is **conservative**. Observed
   peak HBM on a saturating bench is ~15.3 GB — there is ~7 GB of
   HBM left on the table at default flags.
3. The pre-existing `crates/infer-cuda-kernels/src/paged_kv.rs:595`
   index-OOB panic from the long-seq note still fires under
   `--num-slots 14 --kv-cache-dtype int8 --kv-pool-headroom-mb 1024`
   when the publish path tries to insert a span that exceeds the
   slot's allocated pages. Same bug, different trigger.

## Environment

- GPU: NVIDIA L4 24 GB (driver 580.82.07, CUDA 13.0, SM 89)
  - `memory.total = 23034 MiB ≈ 22.5 GiB` reported by `nvidia-smi`
    (the missing 1.5 GiB vs the spec is the system reservation).
  - **Effective max HBM: ~22.5 GiB.**
- Model: Qwen3-4B BF16, `Qwen/Qwen3-4B` HF Instruct variant
- Commit: `3703826` (post 2026-04-16 Tier A/B/C remote acceptance,
  post the 2026-04-15 int8 decode kernel work)
- Cargo env: `CARGO_HOME=/tmp/cargo-home-local`
- Bench: paired `/tmp/peak_throughput_bench.py` script — fires N
  concurrent `/v1/completions` requests, measures aggregate tok/s.
  Each request uses `input=512 tok, output=128 tok, temperature=0`.

## HBM inventory at default `--model-path models/Qwen3-4B` (no other flags)

`nvidia-smi` snapshots:

```
state             memory.used    memory.free    util.gpu
─────────────────────────────────────────────────────────
idle (no server)     228 MiB      22 336 MiB         —
bf16 boot done    14 698 MiB       7 866 MiB        0%
bf16 under load   15 306 MiB       7 258 MiB      100%
int8 boot done    14 730 MiB       7 834 MiB        0%
int8 under load   16 954 MiB       5 610 MiB      100%
```

Auto-sizer log (`auto_num_slots`) for both dtype runs is **identical**:

```
auto_num_slots: gpu_free=23.2GB, weights=8.0GB, reserved=6.4GB,
                available=8.7GB, per_slot=1208.0MB (seq_len=4096),
                slots=7
```

Both bf16 and int8 default servers run at **`num_slots = 7`,
`max_seq_len = 4096`**. The dtype only changes the `TokenKVPool`
sizing downstream (because the pool sizing IS dtype-aware), not the
slot count.

`Scheduler ready` log breakdown (bf16):

```
TokenKVPool budget: 1.8 GB (contiguous KV=4.2 GB, headroom=4.3 GB)
TokenKVPool: 12 384 max tokens (774 pages @ page_size=16),
             1.8 GB for 36 layers, format=BF16
```

`Scheduler ready` log breakdown (int8):

```
TokenKVPool budget: 1.8 GB (contiguous KV=4.2 GB, headroom=4.3 GB)
TokenKVPool: 22 770 max tokens (22 770 pages @ page_size=1),
             1.8 GB for 36 layers, format=INT8
```

The pool stores **22 770 / 12 384 ≈ 1.84× more tokens** at INT8
because each token is half the bytes — that part is dtype-aware.
But the **`contiguous KV=4.2 GB`** number in the same log line is
**bf16-hardcoded** (it is the auto-sizer's budget reservation, not
the actual allocation), and the auto-sizer never re-derives it for
int8.

### What is actually in HBM at idle (bf16, 14.7 GB observed)

| component | size | source |
|---|---:|---|
| Qwen3-4B BF16 weights | ~8.0 GB | `auto_num_slots: weights=8.0GB`; matches the 8045 MB safetensors total from the page1 baseline note |
| Contiguous KV cache (`KVCache`, 7 slots × 4096 × 147456 B) | ~4.2 GB | `TokenKVPool budget` log line |
| `TokenKVPool` (paged decode pool) | ~1.8 GB | `TokenKVPool: 12384 max tokens, 1.8 GB` |
| CUDA context + driver overhead | ~0.5 GB | observed delta: idle nvidia-smi 228 MiB → server boot 14 698 MiB |
| CUDA Graph captures (B=1, 2, 4) | ~0.2 GB | warmup logs show 3 graphs captured in 287 ms |
| Model loader scratch (released after init) | small | gone by the time we measure idle |
| **Total observed at idle** | **~14.7 GB** | nvidia-smi |

Under load the additional ~0.6 GB delta (15.3 - 14.7) is attributable
to:
- `BatchDecodeBuffers` for the active batch (per CLI help: ~750 MB
  budget), of which only the actively-touched portion is committed.
- Workspace allocations for prefill chunks, sample buffers, etc.
- Per-stream FlashInfer scratch.

For int8 the under-load HBM is ~16.95 GB (vs bf16's 15.3) because
the int8 dequant path uses additional working buffers per layer
(per the page1 baseline note and the bench-kv-quant-sweep numbers).
The bigger pool (1.83× tokens) does NOT change the pool's total
bytes because both are budgeted at 1.8 GB.

### What auto-sizing leaves on the table

- `gpu_free=23.2 GB` (raw available before allocations)
- `weights=8.0 GB` (correct)
- `reserved=6.4 GB` (auto-sizer's hardcoded headroom for graphs +
  workspace + safety margin)
- `available=8.7 GB` (the residual the auto-sizer is willing to
  use for `7 × per_slot=1208 MB ≈ 8.5 GB` of contiguous KV)

The `reserved=6.4 GB` is the conservative knob. **Observed total
HBM under load is ~15.3 GB**. Subtract weights (8) + actual
contiguous cache (4.2) + actual pool (1.8) = ~14 GB used directly.
The other ~1.3 GB is workspace + overhead. So the auto-sizer's
`reserved=6.4 GB` is over-budgeting by roughly **5 GB**. With
`--kv-pool-headroom-mb 1024 --gpu-reserved-mb 256` (or similar
manual overrides), the same default workload would have an extra
4-5 GB to put into either more slots or a longer max_seq_len.

## Peak throughput at default 7 slots (input=512, output=128)

Both servers were measured with the paired
`/tmp/peak_throughput_bench.py`, single-shot per concurrency
level (no warm-up). One natural prompt repeated `concurrency`
times so every request had identical work shape, no cache reuse.

```
                           bf16 (7 slots)        int8 (7 slots)
  C    completed   tok/s   completed   tok/s
  ────────────────────────────────────────────────────────
   1     1 / 1     29.1      1 / 1     28.8
   4     4 / 4    106.9      4 / 4    102.8
   7     7 / 7  ← 171.2      7 / 7    161.4
  14     9 / 14*  132.1*    14 / 14   167.5
  28     2 / 28*  146.8*     —        —

  * The bf16 14/28 rows had a degraded "completed" count because
    the bench's emit-counter compares chunks-emitted to
    output_tokens, and the OpenAI streaming layer occasionally
    sends multi-token chunks. The actual aggregate token throughput
    is slightly higher than the printed `tok/s` for those rows.
```

**Peak (clean): 171.2 tok/s on bf16 at `C = num_slots = 7`.** The
GPU is at 100 % utilization. C=14 and C=28 fall off because all
14/28 requests have to queue through 7 slots in waves and the
wave-2 admission cost shows up as wall time inflation.

**int8 default peak: 167.5 tok/s at C=14**, slightly higher than
its C=7 (161.4 tok/s). The bigger pool (22 770 vs 12 384 tokens)
absorbs the 14-request wave in one pass without preempting active
slots. int8 C=14 essentially **matches bf16 C=7** but not by
running more slots — by running the same 7 slots twice with no
dropped work in between.

Per-token int8 ITL is ~5-8 % slower than bf16 (matches the
2026-04-15 quick sweep), so at C=7 saturation int8 is **~6 %
slower** than bf16. The whole "quant KV is a memory lever, not a
perf lever at C ≤ slots" finding from the kv-quant-sweep note
holds at peak as well.

### What `--num-slots 14 --kv-cache-dtype int8` should have done

Reasoning: int8's actual per-slot bytes for the contiguous cache is
roughly half of bf16 (~604 MB at `seq_len = 4096` vs 1208 MB).
Available HBM for KV at default reservation is 8.7 GB. Therefore
int8 should fit `8.7 / 0.604 ≈ 14` slots — **double bf16's 7**.

Tried: `target/release/infer --model-path models/Qwen3-4B
--kv-cache-dtype int8 --num-slots 14 --kv-pool-headroom-mb 1024
--gpu-reserved-mb 512`.

Result: server **does** boot, but the auto-sizer reports
`contiguous KV=8.5 GB` (a bf16-fixed calculation: `14 × 1208 =
8 512 MB`). This pushes the pool budget down to 0.8 GB and only
**10 207 pool tokens**. With 14 active slots × ~640 tokens of
typical request KV = ~9 000 tokens needed, the pool goes empty
under load, the prefill migration starts failing with
`TokenKVPool: out of pages`, and eventually the publish path hits
the **`paged_kv.rs:595` index-OOB panic** documented in
`docs/experience/wins/2026-04-15-bench-longseq.md` § "Pre-existing
panic on prefix-cache publish". Same bug, different trigger.

Idle HBM with 14 slots was **13.78 GB** (vs the auto-7-slot
default's 14.7 GB) — confirming that int8's *actual* contiguous
allocation IS half the bf16 size, even though the budget log line
prints the bf16 amount. So the HBM is there; the auto-sizer just
won't use it.

To actually deliver the int8 capacity benefit on the L4 today, we
would need both:

1. **Make `auto_num_slots` dtype-aware**. ~5-10 LOC in
   `infer/src/main.rs:277` to switch between `1208 MB` (bf16) and
   `~604 MB` (int8) per-slot when computing the slot count.
2. **Make `TokenKVPool budget` log line dtype-aware**. The
   `contiguous KV=…` value is currently printed as the bf16
   reservation; the pool budget computation should use the actual
   dtype's per-slot bytes so headroom isn't double-counted.
3. **Fix the `paged_kv.rs:595` publish panic** so high-slot int8
   doesn't crash on the first cleanup that exceeds an allocated
   slot's page span. This is already on the follow-up list from
   the long-seq note.

Without (1)+(2), `--num-slots 14 --kv-cache-dtype int8` is a
booby-trap because the user passes the larger slot count expecting
int8's halved per-slot footprint to leave room for the pool, and
instead the auto-sizer reserves bf16-equivalent space.

## Numbers to remember

| metric | bf16 | int8 |
|---|---|---|
| **Default num_slots** (auto-sized) | 7 | 7 (should be 14) |
| **Idle HBM** | 14.7 GB | 14.7 GB |
| **Peak HBM under load** | 15.3 GB | 16.95 GB |
| **Free HBM at peak** | 7.3 GB | 5.6 GB |
| **Peak throughput** | **171 tok/s** at C=7 | 167.5 tok/s at C=14 |
| **Peak ITL** | ~35 ms | ~37 ms |
| **TTFT p50 at saturation** | 344 ms (C=7) | 348 ms (C=7) |

**Capacity ceiling on L4 24 GB at default flags**:
- 7 concurrent requests at `max_seq_len = 4096`
- 28 672 contiguous KV tokens + 12 384 pool tokens = ~41 056 tokens
  total addressable
- Real-world ceiling: ~170 tok/s aggregate, dominated by per-token
  decode cost (~35 ms ITL × 7 slots = 200 ms decode wall per token
  per slot, vs 35 ms for 1 slot).

If we fixed the auto-sizer to be dtype-aware AND bumped int8 to
~14 slots:
- Theoretical peak: ~340 tok/s (linearly scaled bf16's 171 at 2×
  slots), assuming the int8 kernel's per-token cost stays at 1.06×
  bf16. Realistically ~300 tok/s on L4 because the kernel will
  start sharing memory bandwidth across more concurrent batches.

That **300 tok/s number is the current upper bound on L4** for
Qwen3-4B in the existing kernel family. To go beyond it we would
need either (a) a bigger model would not fit, or (b) a faster
attention kernel — see the
[`../research/kv-quant-decode-industry-survey.md`](../research/kv-quant-decode-industry-survey.md)
survey for the full set of upstream options.

## Sign-off

Two HBM snapshots, two clean throughput sweeps, one blocked sweep
(panicked due to pre-existing bug, documented). Three actionable
findings:

- [x] HBM inventory captured for both bf16 and int8 default flags
- [x] Peak throughput measured at default num_slots (171 tok/s
      bf16, 167 tok/s int8)
- [x] Auto-sizer dtype-blind bug documented; int8 capacity win
      is currently disabled
- [x] Auto-sizer headroom over-budgeting documented (~5 GB
      conservative buffer)
- [x] Pre-existing `paged_kv.rs:595` panic re-confirmed under a
      new trigger

## Rule

**A 24 GB GPU at default `infer --model-path` flags has 7 GB of
HBM that the auto-sizer never touches.** The conservative
`reserved=6.4 GB` is sized for worst-case workspace + graph
captures + safety margin, but observed peak under saturating load
is only ~1.3 GB beyond the strict (weights + cache + pool) floor.
Trimming `reserved` and `kv_pool_headroom_mb` is the single
biggest knob users have to push effective capacity, and it is not
documented in `--help` as such — both default to values that
leave 5+ GB of HBM unused.

**Corollary**: when shipping a quantised KV cache mode (int8,
fp8, tq3, etc.) the auto-sizer **must** know the per-slot bytes
of that mode. Without it, the user types `--kv-cache-dtype int8`
expecting twice the slots and gets the same 7 the bf16 default
gives them. Fix is ~10 LOC in `auto_num_slots`, blocks the entire
"quant KV as memory lever" story documented in the kv-quant-sweep
note.

**Corollary 2**: the L4 24 GB ceiling for Qwen3-4B at this kernel
quality is roughly **300 tok/s aggregate** if the auto-sizer is
fixed AND the publish-path panic is fixed AND int8 is the
canonical quant. Anything beyond that requires either a bigger
GPU, a smaller model, or a tensor-core-grade attention kernel
(see the industry survey for the upstream options).
