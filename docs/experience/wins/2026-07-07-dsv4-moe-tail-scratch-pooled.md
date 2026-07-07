# DSv4 compact-FP8 MoE decode-tail scratch pooled

> Status: measured — 2026-07-07. Commit `4f589cfb`.

## Change
`dsv4_moe_forward_decode_fp8` allocated 8 device buffers per MoE layer per step.
Added `Dsv4MoeTailScratch` (moe.rs, sized to `DSV4_DECODE_CONTIG_MAX_ROUTES = 128`)
on the kv_adapter (`moe_tail_scratch`, allocated when the model has a MoE layer).
Threaded via `tail: Option<&mut Dsv4MoeTailScratch>` on `dsv4_moe_forward` →
`dsv4_moe_forward_masked_tail` → `dsv4_moe_forward_decode_fp8`; batched-stream path
passes `Some`, other 4 call sites pass `None` (throwaway scratch).

Buffer init:
- reused as-is: offsets, scan_total, packed_hidden, packed_weight, act, expert_out.
- re-init per step (`reinit`, live `[0,rows)` span): counts=0, cursors=0,
  route_out=0, packed_route_slot=-1.

VRAM budget (`dsv4.rs`) adds `moe_tail_scratch_bytes` to the fixed term.

## Compile
BUILD_EXIT=0 (cuda,nccl,deepep), clippy-clean. First build failed E0425 at
moe.rs:3250 (`tail` not in scope) — `dsv4_moe_forward_masked_tail` is the middle
hop between `dsv4_moe_forward` and `dsv4_moe_forward_decode_fp8` and also needed
the `tail` param. Fixed in `4f589cfb`.

## Correctness (greedy, TP=4/EP=4, DSv4-Flash-FP8, GPU 4-7, MTP-on)
- "capital of France, one word" → content "Paris".
- "three primary colors" → coherent reasoning_content.
- No garbage / NaN / empty generation.

## Wall-clock
Measured in the cumulative A/B:
`errors/2026-07-07-dsv4-alloc-removal-sweep-wall-wash.md` (c1+c2 vs baseline
−0.27% mean wall, within ±0.7% run-to-run spread).
