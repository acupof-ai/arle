# 2026-04-15 · INT8 decode kernel num_splits bump — final partial win

## Context

Iteration on top of
[`2026-04-15-bench-longseq-int8-optionA.md`](2026-04-15-bench-longseq-int8-optionA.md).
After Option A landed the SMEM-tile + cp.async pipeline and recovered
~4-8 % ITL at long context, the residual gap vs bf16 at 25k was
still 24 ms. User asked me to figure out more.

I spent a cycle on two wrong hypotheses first, then found the real
lever by reading the kernel's split-KV grid sizing.

## What I tried and what failed

**Hypothesis 1 (refuted)** — batching `pool_idx` loads into one
warp-coalesced read would eliminate scatter serialisation inside the
prefetch loop. Code written, built clean, measured **regression** of
0.3-1.5 ms across all configs. Root cause: the original kernel's
`kv_indices[tok_start_global + global_t]` inside the prefetch loop
already reads the SAME address across all 32 lanes of the warp
(because `t` doesn't depend on `lane_id`), so the hardware already
did a **broadcast load**, not a scatter. The "fix" added one
`__syncthreads()` without recovering a problem that wasn't there.
Reverted, documented as a dead-end in the commit message.

**Hypothesis 2 (empirically-tested, clean result)** — the split-KV
grid was undersized. `choose_decode_num_splits` at
`crates/cuda-kernels/csrc/attention/decode_attention_quantized.cu:472`
picks `num_splits = ceil(target_blocks / total_q_heads)` where
`target_blocks = kTargetBlocksPerSm × num_SMs`. The default
`kTargetBlocksPerSm = 4` gives `target_blocks = 4 × 58 = 232` and
`desired_splits = ceil(232 / 32) = 8` for Qwen3-4B (32 Q heads,
batch=1). At 25k tokens, each CTA then processes 3125 tokens, and
the entire grid is **128 blocks = 512 warps**, which is only 14 %
of the L4's maximum concurrent warp capacity. Compute-bound on
softmax/reduce with low SM occupancy.

Bumping to `kTargetBlocksPerSm = 32` gives `target_blocks = 1856`,
`desired_splits = ceil(1856 / 32) = 58`, clamped to `kMaxSplits = 32`
(which is the workspace pre-allocation ceiling in
`crates/cuda-kernels/src/paged_kv.rs:333`). So `num_splits = 32`
at runtime, producing **1024 blocks = 4096 warps** — closer to SM
saturation, and each CTA only handles 781 tokens at 25k so the
hand-written softmax loop stays out of the way of the SM scheduler.

## Environment

- GPU: NVIDIA L4 24 GB (driver 580.82.07, CUDA 13.0, SM 89)
- Model: Qwen3-4B BF16, `Qwen/Qwen3-4B` HF Instruct variant
- Commit at validation: `6ba6340` (post the 2026-04-15 Metal batch-decode
  tranche + the `189bd17` `waiting_count` fix) + the one-line kernel
  edit proposed in this note
- Server: same flags as the Option A bench (`--num-slots 1
  --max-seq-len 32768 --kv-cache-dtype int8 --kv-pool-headroom-mb 2048
  --gpu-reserved-mb 512 --port 8001`)
- Cargo env: `CARGO_HOME=/tmp/cargo-home-local`
- Bench: `/tmp/longseq_bench.py --configs 4000,8000,16000,25000
  --output-tokens 128`, greedy decode, single request

Important fact about the bench timing: the FIRST splits=32 run I did
was pre-pull, on a commit that still had the Tier A waiting_count
bug (`498c0cc` left two `waiting_count.fetch_add(1)` in
`runtime.rs`). That bug was fixed in `189bd17`. Under the bug, each
request got double-counted in the waiting channel, the scheduler
got stuck in a retry loop that queued a second server-side request
for every bench POST, and my bench measured wildly inflated TTFT
(8k jumped to 6 s, 16k to 11 s). Re-ran after pulling `189bd17`;
the numbers below are the clean post-pull measurement.

## Results — int8 long-seq progression

```
                bf16     int8 BASE   int8 OPT-A   int8 splits=32
input           ITL      ITL         ITL          ITL
─────────────────────────────────────────────────────────────────
 4 000 tokens   35.2 ms  37.7 ms     37.0 ms      37.1 ms
 8 000 tokens   37.4 ms  42.1 ms     40.9 ms      40.3 ms
16 000 tokens   41.8 ms  50.7 ms     48.5 ms      47.0 ms
25 000 tokens   33.1 ms  61.9 ms     56.9 ms      55.2 ms
```

Cumulative int8 improvement vs BASE:

| input | BASE ITL | final ITL | delta | pct |
|---|---:|---:|---:|---:|
|  4 k | 37.7 | 37.1 | −0.6 | −1.6 % |
|  8 k | 42.1 | 40.3 | −1.8 | −4.3 % |
| 16 k | 50.7 | 47.0 | −3.7 | −7.3 % |
| 25 k | 61.9 | 55.2 | −6.7 | **−10.8 %** |

All 16 configs (4 × 4 sweeps) emitted 128/128 tokens cleanly — no
early-EOS drift like the initial Option A run had at 25k. TTFT and
wall time unchanged from Option A alone within noise.

Steady-state decode throughput at 25k:
- int8 BASE:       128 tokens in (14.22 − 6.36) = 7.86 s = 16.3 tok/s
- int8 final:      128 tokens in (13.37 − 6.36) = 7.01 s = 18.3 tok/s
- **+12.3 %** steady-state decode speedup from Option A + splits=32.

## What actually helps vs what doesn't

`kTargetBlocksPerSm` is a **one-constant change in one .cu file**
(line 489 of `decode_attention_quantized.cu`) that also applies to
the FP8 and TurboQuant variants since they share the same
`choose_decode_num_splits` helper. No new kernels, no new types, no
workspace resizing (paged_kv.rs already pre-allocates at
`num_splits = 32`). Zero correctness risk — the split-KV phase-2
merge has always been exercised, we're just giving it more
partials per query head to merge over.

What **didn't** help:

- Pool-index SMEM batching (the hypothesis 1 above) — regression.
  The `kv_indices[...]` load inside the prefetch loop was already
  broadcast across the warp; "fixing" it added a sync without
  reducing any real work. Lesson: **verify the assumed memory
  pattern before writing the optimisation**; do not assume every
  scalar-looking global load is a per-lane scatter.

## Residual gap and where the next 22 ms at 25k come from

After Option A + splits=32, int8 at 25k sits at 55.2 ms ITL vs
bf16's 33.1 ms. The remaining **22 ms gap** is almost certainly
structural to the kernel family, not algorithmic:

1. **No tensor cores for the QK matmul.** Our int8 kernel uses
   CUDA-core FP32 FMAs for the dot product. FlashInfer's bf16 paged
   kernel uses bf16 WMMA (tensor cores) on SM89, which is ~4-8×
   faster for the same matmul shape. Since the bulk of the kernel's
   ALU time is QK + online softmax, this alone explains most of
   the gap. Fixing it requires an IMMA (integer MMA) path —
   `mma.sync.aligned.m16n16k16.s32.s8.s8.s32` on SM89 — which is a
   new kernel, not a tuning knob.

2. **`page_size = 1` still.** BF16 paged pool uses `page_size = 16`;
   INT8 stays at 1. That is 16× more `kv_indices` indirections per
   decode step, and although my bench showed hardware broadcast
   coalescing takes most of the sting out, the pool-pointer chain
   is still longer than bf16's. This is the "Option B" fix
   documented in the Option A note — would require rewriting
   `kv_cache_to_paged_int8_kernel` + `decode_attention_int8` +
   `batch_decode.rs:449,494` together.

3. **Online softmax reduction overhead.** Each of the ~6250 tokens
   per warp does one `warp_reduce_sum(qk)` (5 shuffles) + three
   `__expf` calls. At 25k that's ~400-800 µs per warp just in
   reduction / special-function time, on top of the memory traffic.
   FlashInfer amortises this differently via warp specialisation
   (some warps do loads, others do reduces, pipelined). Our kernel
   has no such specialisation.

None of those are "tune this one constant". They are each their
own commit-sized project. For now, **the splits=32 change is the
last straightforward lever**; beyond here the return on effort
drops hard until we land a TC-based int8 decode kernel.

## Sign-off

- [x] kernel edit is one line, scoped to `choose_decode_num_splits`
- [x] build clean
- [x] all 4 bench configs emit 128/128 tokens
- [x] 25k ITL improved from 56.9 ms → 55.2 ms (Option A → splits=32),
      and from 61.9 ms → 55.2 ms (BASE → final)
- [x] steady-state decode throughput at 25k up 12.3 % vs BASE
- [x] raw JSON saved as `2026-04-15-bench-longseq-int8-splits32-postpull.json`

## Rule

**When a kernel looks compute-bound, check the grid size before
re-writing the inner loop.** The `choose_decode_num_splits` helper
on this tree was shipping an effective `num_splits = 8` on L4 for
Qwen3-4B at `batch_size = 1`. That was 14 % of the L4's warp
capacity on a split-KV kernel that was **designed for
oversubscription**. Bumping the target-blocks-per-SM constant from
4 to 32 was a one-line change that captured more improvement than
400 lines of SMEM tiling and cp.async pipelining combined. The
tiling work wasn't wasted — it bought 4-8 % on its own — but the
grid-size lever was the bigger fish and it was sitting in plain
sight next to a workspace ceiling already provisioned for 32
splits.

**Corollary**: the next 20 % of perf on this kernel costs real
work — either an IMMA-based decode path (major rewrite of the
integer matmul loop to use SM89 int8 tensor cores) or the full
page_size=16 pool lift (Option B, 5-8 files). Anything between
those two levels of effort is probably not there. Stop looking for
"one more trick" and either commit to the larger fix or accept
that int8 on L4 runs at ~60 % of bf16 perf for long contexts.

**Corollary 2 (methodology)**: always verify the suspected memory
access pattern against the kernel's actual broadcast/coalescing
behaviour **before** writing the "fix". The hypothesis-1 regression
in this round cost one build + one bench cycle and an extra commit
to revert, because I assumed `kv_indices[...]` inside a warp was a
scatter without checking. `lane_id` has to appear in the address
expression for a read to actually be scattered across the warp.
