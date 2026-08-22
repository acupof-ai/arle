# Long-context decode attention on Qwen3.6-27B — where the time goes, and the order to fix it

The first long-agent baseline for ThinkingCap-Qwen3.6-27B-FP8 (1×H20, multi-turn
32k) put decode, not prefill, at the top of the cost. Prefill runs at ~3.9–4.1k
tok/s and is flat across c=1/4/8 — it is saturated and uninteresting. Decode is
77 ms/token at c=1 on a step whose KV read has a 0.52 ms roofline.

This document separates what is measured from what is read off the source from
what is still a hypothesis, and orders the work so each step is licensed by the
one before it.

## What is measured

Arm: no-spec, `bench-agent-32k-8x8.jsonl` (8 sessions × 8 turns, sha
`78c70bda…`), 64 requests/point, max_tokens 214, greedy, GPU 0. Prefix hit rate
0.9585 against the TraceLab 95.7% reference.

| c | TTFT p50 | prefill tok/s | TPOT p50 | TPOT p99 | decode tok/s/req | aggregate decode tok/s |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 12.4 s | 4086 | 77.0 ms | 80.9 ms | 13.0 | 13.0 |
| 4 | 9.1 s | 3951 | 95.2 ms | 8937 ms | 10.5 | 42.0 |
| 8 | 13.5 s | 3949 | 140.5 ms | 9553 ms | 7.1 | 56.9 |

Decode speed is `1000 / TPOT`; the wall clock is mostly prefill on a 32k
workload.

TPOT is linear in context. Anchoring on a measured short prompt rather than
extrapolating (see the byte ledger below): **26.6 ms at 66 tokens, 72.1 ms at
32k → 1.57 ms per 1k context tokens.** The context-free step is 26.6 ms, not the
5 ms an earlier two-point extrapolation produced — that value sat below the
7.3 ms physical weight-read floor, which is how it was caught.

A single-request ITL probe (no bench harness, no queueing) puts c=1 at 32k at
**ITL p50 72.1 ms, 3 trials within 0.4%** — the number the rest of this document
decomposes.

TPOT p99 blowing to ~9 s at c≥4 while p50 stays at 95/140 ms is a separate
issue — queueing/preemption tail, not decode speed. It gets its own
investigation, not this one.

## What the source says

Geometry: 64 layers, `full_attention_interval 4` → **16 full-attention layers**,
`num_attention_heads 24`, `num_key_value_heads 4` (GQA ratio 6), `head_dim 256`,
bf16 KV. So **64 KB of KV per context token** and **2.1 GB read per decode step
at 32k** — 0.52 ms at 4 TB/s.

### The hot kernel is TileLang, and it took three wrong turns to find it

`ARLE_QWEN35_PROFILE=1`, one 32k request, 190 decode steps, per-step CUDA time:

| label | ms/step | calls/step |
|---|---:|---:|
| `qwen/full_attention` (parent, 16 layers) | 56.76 | 16 |
| → **`qwen/full_paged/attention`** | **50.84** | 16 |
| `qwen/linear_attention` (parent, 48 GDN layers) | 20.35 | 48 |
| `qwen/dense_ffn` | 16.37 | 64 |
| `qwen/full_paged/qkv_gemm` | 2.42 | 16 |
| everything else | ~5.4 | |
| **total** | **≈95** | |

**Attention is 53% of the step: 3.18 ms per layer at 32k.** Per layer the KV is
134 MB, so that is **42 GB/s — 1% of peak**, matching the context-term estimate
from the short-prompt anchor independently.

It resolves through `ffi::resolve_paged_attn_v1` to a **TileLang AOT kernel**,
`tools/tilelang/batch_decode_paged_hd256.py` → `batch_decode_paged_hd256_q24_kv4`.
Four diseases, all in that one 285-line file, all read off the source:

1. **64× M-padding.** `BLOCK_M = 64` with one real query row:
   ```python
   # Rows 1..63 are padding to satisfy TileLang's tensor-core
   # M-divisibility constraint; they are masked out below.
   ```
   98.4% of every QK and PV GEMM is padding.
2. **One CTA per query head.** `T.Kernel(1, num_q_heads, batch_size)` — the KV
   cache is per KV head, so the `gqa_ratio` = 6 blocks of a group each walk it
   independently.
3. **No split-KV.** At c=1 the grid is **24 CTAs × 128 threads = 3072 threads**
   on a 78-SM, 159 744-thread GPU — about 2% occupancy — and each CTA walks
   32k serially in 2048 `BLOCK_N=16` iterations at `num_stages=2`.
4. **`BLOCK_N = 16` is one page, chosen for a GPU we do not serve.** The
   docstring is explicit: `BLOCK_N=64` needs 128 KB of shared and "would not
   load on L4 / sm_89 (99 KB cap)". H20 lifts to 228 KB.

(1) and (2) are what PackGQA fixes — pack the group's `gqa_ratio` real rows into
M instead of padding, which simultaneously drops BLOCK_M waste from 63/64 to
10/16 and cuts KV traffic 6×. (3) is split-KV. (4) is an SM-conditional tile.
Upstream TileLang ships this shape in `example_gqa_decode*`, which this file's
own docstring cites.

### Three wrong turns, recorded because each was cheap to avoid

- **Hand-wrote a GQA + warp-per-key + scaled-splits rewrite of
  `fused_gqa_attention_decode_batched_kernel`** (`c00efdb9c`): measured −20.5%,
  reverted (`fcf709e0f`). That kernel is in the **non-paged** lane, which this
  configuration never enters.
- **Turned on `--qwen35-fa3-decode`** (since deleted): ITL p50 72.4 ms vs 72.1
  off. Its call site was in the same unused non-paged lane. Four configurations —
  batched/per-row × FA3 on/off — all landed at 72.1–72.4 ms, which should have
  been read immediately as "none of these four is executing".
- **Concluded from that identity that attention was not the bottleneck**: wrong,
  caused by an aggregation bug that dropped `full_paged/attention` from the
  per-step table.

The cost of all three was one command: `ARLE_QWEN35_PROFILE=1`. **Profile before
touching a kernel, and confirm the kernel you are editing is the one that runs.**

## The byte ledger (measured from the checkpoint's tensor table)

Batch 1 has no weight reuse, so a decode step streams essentially the whole
model out of HBM. Summed from `model.safetensors.index.json`, FP8:

| what | bytes/step | share |
|---|---:|---:|
| MLP (gate+up+down × 64 layers) | 17.12 GB | 58% |
| GDN (`in_proj_qkv` + `in_proj_z` + `out_proj` × 48 layers) | 5.54 GB | 19% |
| `lm_head` | 2.54 GB | 8.7% |
| full-attention weights (q/k/v/o × 16 layers) | 1.68 GB | 5.7% |
| **KV cache @ 32k** | **2.10 GB** | **7.2%** |
| GDN recurrent state (48 × [48,128,128] fp32, r+w) | 0.30 GB | 1.0% |
| **total** | **≈29.3 GB** | |

Floor at 4.0 TB/s: **7.3 ms**. Real stacks land at 35–65% of peak → 11–21 ms →
48–91 tok/s, which brackets the Qwen3-32B FP8 reference on this same H20
(46.2 tok/s, Qwen's own speed benchmark).

The constant term is measured, not extrapolated: a 66-token prompt, no spec,
c=1, on today's binary gives **37.60 tok/s = 26.6 ms/token** (6 trials, spread
0.6%) — against `baselines.md`'s `6aa4ca6d1` row of 38.62 tok/s at 128 tokens,
i.e. inside the ±3% drift band. **Short-context decode has not regressed.**
Splitting the 32k measurement against that anchor:

| | bytes | time | effective | % peak |
|---|---:|---:|---:|---:|
| weights (constant term) | 26.9 GB | 26.6 ms | 1.01 TB/s | **25%** |
| KV (context term) | 2.10 GB | 51.4 ms | 41 GB/s | **1.0%** |

KV is 7.2% of the bytes and 66% of the time, at 1% of peak bandwidth. The weight
path is not catastrophic but it is not healthy either — 25% against a 35–65%
band leaves 1.4–2.6× on the table there too.

Context slope from this anchor: **1.57 ms per 1k context tokens** (51.4 ms /
32.6k). An earlier estimate of 2.09 ms/1k came from extrapolating two
cold/warm points 2.5k apart back to zero; that extrapolation also put the
constant term at 5 ms, below the 7.3 ms physical weight floor, which is how it
was caught. Use the short-prompt anchor, not the extrapolation.

**Nothing here is a regression — it is an exposure.** Every 27B row before
2026-07-26 used 128-token prompts (`RETIRED short-prompt fingerprint` in
`baselines.md`); the multi-turn 32k workload became mandatory one day earlier
under bench spec rule 5. The context term was always there and was never
measured at the target length.

This also explains why concurrency buys nothing here: weight traffic amortizes
across a batch, KV traffic does not. A KV-bound engine cannot batch its way out
— which is exactly the measured 8.6 → 9.9 aggregate decode tok/s from c=1 to
c=8.

## What this deletes

Full-attention decode had five implementations of one job. The profile said
exactly one of them ran at c=1: `batch_decode_paged_hd256` (TileLang), 53% of
the step.

Closed out 2026-07-27/28:

- `fused_gqa_attention_single_token_kernel` — deleted (`5dc0d28e7`, 185 lines,
  zero Rust callers).
- `--qwen35-fa3-decode` — deleted. FA3 went into the paged lane unconditionally
  instead, for every query length; a second differently-gated FA3 entry that
  the decode graph can never use is the half-state.
- `--qwen35-batched-decode-attention` — deleted. It was a same-binary A/B knob;
  `head_dim != 256` still selects the per-row arm, which is the real selector.
- `fused_gqa_attention_decode_batched_kernel` + `attention_decode_reduce_batched_kernel`
  — **kept, and the premise here was wrong.** They are not dead: `forward_decode_batch`
  is the contiguous-KV lane for OPD weight offload, where no paged pool exists.
  The paged lane never reaches them; the offload build has nothing else.

The rule holds — converge on one flow — but "unreachable from the serving path"
is not "unreachable". Check every caller of the lane, not every caller of the
kernel.

## Order of work

**Step 1 — make coverage honest.** `qwen/linear_attention` (20.35 ms/step) sums
to 13.9 ms across its wrapped children; `qwen/full_attention` (56.76) to 54.9.
Close both gaps before optimizing either, and add a `qwen/step` wrapper so the
profile is checkable against ITL.

**Step 2 — route sm_90 paged decode to the vendored FA3, keep TileLang for the
rest.** FA3 hopper is sm_90-only, so this is a capability split, not two
implementations of one job: **sm_90 → FA3 paged split-KV + PackGQA; sm_70 and
anything else → the existing TileLang kernel.** PackGQA and split-KV are exactly
defects (1)–(3) above, already implemented and validated upstream.

Three facts checked against the vendored source, so the integration has no
unknowns left:

- **No relayout.** `paged_kv.h:117` builds `mK_paged = make_tensor(ptr_K,
  shape_K, stride_K)(_, _, bidh, _)` over dims `(page_size, head_dim, head,
  page)` with an arbitrary CuTe stride tuple. Our HND pool
  `[page, kv_head, page_size, head_dim]` is expressible as
  `stride_K = (head_dim, 1, page_size*head_dim, kv_dim*page_size)` — the last
  term is the `stride_page` the call site already computes
  (`qwen35.rs:6174`).
- **`pagedkv_tma = false`.** qwen35's `SUPPORTED_PAGE_SIZE = 16` is smaller than
  FA3's hdim256 `kBlockN`, and the TMA paged path asserts
  `page_size % kBlockN == 0` (`mainloop_fwd_sm90_tma_gmma_ws.hpp:537`). The
  non-TMA `PagedKVManager` gather path has no such constraint and is what the
  vendored `flash_fwd_hdim256_bf16_paged{,_split}_sm90.cu` units serve.
- **Start at `b = 1`, one call per request.** The shim is already written for
  `params.b = 1`; a request's page slice is `kv_indices + kv_indptr[i]`, which is
  a rectangular 1-row page table for free. FA3's ragged input wants a padded
  `[b, max_pages]` table, so batching is a second step — take it only if c≥4
  measures worse than the TileLang batched launch it replaces.

Touches: `arle_fa3_shim.cu` (paged fields + branch), `ffi/attention.rs` (struct
mirror), `qwen35.rs:6407` (dispatch + per-request loop). The `--qwen35-fa3-decode`
flag does not survive this — FA3 becomes the sm_90 path, not an arm.

**Step 3 — nothing to do about `BLOCK_N` on sm_90.** The 16-per-page tile exists
for L4's 99 KB shared cap; once sm_90 leaves this kernel, the constraint only
binds the devices that actually have it. Revisit only if sm_70 becomes a
serving target again.

**Step 4 — re-measure, then delete.** Same long-agent dataset, same
`ARLE_QWEN35_PROFILE` decomposition, plus the 4K/c=1 needle. Only then does the
deletion list above land.

Steps 2–3 are GPU kernel work: each needs `ncu` before/after and a dated
`wins/` entry, re-measured at 32k — not on short prompts, where the entire
context term is invisible.

## Rule

**Profile first, and prove the kernel you are about to edit is the one that
runs.** Three optimization attempts here — a hand-written kernel, a vendored
FA3 flag, and a four-way configuration sweep — all landed on code the serving
path never executes, because the model has two full-attention lanes and the
flags only reach the unused one. One `ARLE_QWEN35_PROFILE=1` run answered it,
and every step before that was wasted.

Report prefill and decode separately or the headline number is meaningless: the
same run is a saturated 4086 tok/s prefill plus a decode at 1% of memory peak,
and only one of those is worth an engineer's week.

And when a portability constraint sets a performance constant — `BLOCK_N = 16`
so the cubin loads on L4 — write down which device pays for it. This one costs
the H20 serving path a 4× longer inner loop for a GPU that is not in the
support matrix.
