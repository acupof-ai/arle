# DSpark draft attention was launched per slot at 192 blocks — CUDA, 2026-08-08

> Status: **mechanism measured, serve A/B and correctness gate pending-remote.**
> Do not quote an end-to-end number from this entry until the Results section
> below is filled in.

## Problem

`nonpaged_prefill_attention_kernel` is 30.5% of a decode tick on the
decode-shaped c=16 capture — the largest single line
([the re-anchor entry](2026-08-08-decode-shaped-reanchor-draft-attention-is-30pct.md)).
Two 2026-08-01 rewrites of this kernel were reverted after failing to transfer
to the serve.

## Root cause

The cost was never in the kernel's arithmetic.

Every one of the 39,690 launches in the 60 s window carries grid
**(32, 6, 1) = 192 blocks**, on one stream, 7.5 µs apart — 16 slots × 5 draft
layers, one launch per slot. 192 blocks is ~2.5 per SM on an H20's 78.

`dspark_draft_blocks` batched the draft GEMMs and left the ring kernels
per-slot, which its own doc comment stated (`dspark.rs:1266`, "Only the ring
kernels stay per-slot"). Each slot owns a separate `df.k_ctx[li]` ring
allocation, so one launch could not address them all.

**This is why the two 08-01 rewrites missed.** Both were tuned by `ncu` at
**3072** blocks, from a harness sweeping rows 12→96 out of the model config. At
3072 blocks the kernel is genuinely ALU-bound (SM 80.15%, ALU 61.9%, L2 hit
99.58%, DRAM 0.06%). At 192 it is occupancy-starved. Two different kernels as
far as optimization goes.

## Fix

`nonpaged_prefill_attention_kernel` takes an optional array of per-slot ring
base pointers and promotes `blockIdx.z` to a slot axis; the new
`nonpaged_prefill_attention_ring_varlen_batched_cuda` entry launches grid
(heads, block, slots). `dspark.rs` stages each layer's k/v ring bases inside the
existing per-slot loop and runs attention once after it. Grid becomes
(32, 6, 16) = 3072 blocks, one launch per layer instead of 16.

Commit `3a8f99b1f`.

## Measured — pinned shape

`crates/cuda-kernels/tools/nonpaged_attn_bench.cu batch 16 6 <kv_len>`, 16 slots
× block 6, 20 iterations after 3 warmups, H20 GPU 0. **Output bit-identical in
every arm.**

| kv_len | per-slot | batched | Δ |
|---:|---:|---:|---:|
| 512 | 3.545 ms | 1.023 | −71.1% |
| 1024 | 7.203 | 2.057 | −71.4% |
| 1376 | 9.013 | 2.766 | −69.3% |
| 2048 | 13.283 | 4.197 | −68.4% |

Flat across the window, so the win is structural rather than window-dependent.

**The harness sits on the serve's operating point.** At `kv_len` 1376 a
192-block launch costs 563 µs against the serve's 558 µs mode (+0.9%). The
serve duration histogram is bimodal — 47% of launches in the 550–600 µs bin
(full window), the rest spread down as slots fill.

## Results — serve A/B

Pending-remote. Matched A/B, two binaries from the same HEAD (`3a8f99b1f` and
its parent), decode-shaped c=16, arms swapped across GPU 0 and GPU 1.

| | BASE | NEW | Δ |
|---|---|---|---|
| output tok/s | | | |
| ITL mean | | | |
| accept_rate | | | |

## Correctness gate

Pending-remote. `scripts/needle_gate.py` ladder ×3 same-config repeats per arm,
at depth 0.0 and 0.5.

The harness bit-identity covers the kernel math only. The failure mode this
change can actually produce lives in the caller — a wrong slot index, a wrong
window-table offset, or the k/v pointer-array halves swapped — any of which
makes one draft slot attend another slot's ring. That is a correctness bug the
math check cannot see, which is why the gate is required here and not optional.

## Learnings

**Pin a kernel microbench's shape from the trace's grid dims, not from the
model config.** The config gave 3072 blocks and a 2048-token window; the serve
runs 192 blocks and behaves like a 1376-token window. Three rewrites were sized
against the config.

**"Only X stays per-slot" in a doc comment is an unpriced cost.** It was written
when the batched draft landed and stayed true for seven weeks, through a
root-cause note ([`project_decode_attention_throws_away_batch`]) that predicted
exactly this symptom and named an 8-second check for it — count attention
launches per step at 8 concurrent decoders. The check was never run on the draft
lane.
