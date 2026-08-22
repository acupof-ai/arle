# Closing the LoRA merge loop for an NVFP4 base — CUDA, 2026-08-22

> Status: Shipped

## Context

`rubric-opd` and `agent-opd` train a LoRA and sync it back into the co-resident
rollout engine. That sync promotes each quantized target to dense BF16, merges
`W = base + scale·(B·A)`, and requants. NVFP4 could not enter it: the repack
releases the group bytes at load, so the only resident form is the Marlin
layout, and the promotion accepted dense BF16 and FP8 block-scaled only. Every
NVFP4 round died at `layer 0 mlp.gate_proj: got Fp4E2M1Group`.

`train opd` was unaffected — it never syncs — which is why the 2026-08-19 NVFP4
entry could report a working step without meeting this.

## What Worked

`dequantize_fp4_marlin_to_bf16` is the existing `dequantize_fp4_marlin_to_fp8`
kernel minus its per-128x128 divisor. That divisor exists because DeepGEMM
takes the block power of two back as `sfb`; a BF16 consumer wants the value
itself. Same tile walk, same `marlin_fp4_scale_tail` de-permutation.

Two things the promotion has to do beyond the dequant:

- **Build the FP8 slots.** Without `qweight_u8`, `requant_merged_matrix`
  returns early and the weight stays dense BF16 at 4x the NVFP4 bytes. On a 27B
  all-linear merge that is an OOM. From the first merge on, the engine serves
  FP8 rather than NVFP4 — the residency win applies to the rollout and the
  training step, not to a post-sync engine.
- **Park the tiles, do not free them.** A `--share-frozen-base` student aliases
  the packed bytes. They move to `retired_marlin`; left in `marlin_packed` the
  FP8 arm this weight requants into would pick them up and read FP4 tiles as
  FP8.

## Result

`rubric-opd`, `ThinkingCap-Qwen3.6-27B-NVFP4`, greedy + seed 1234, 20 prompts,
`--lora-target-set attention-qv`, single H20:

- `borrowing 256 resident NVFP4 base projections from the rollout engine
  (zero-copy, marlin layout)`
- round 0 `accepted=20 distinct=20 parse_err=0 trained=2 mean_loss=0.3363`
- LoRA sync completed, exit 0

## Limits

`--lora-target-set all-linear` still OOMs during the sync on a 27B. That is not
an NVFP4 property: the FP8 arms fail the same way at the same settings (share →
`BF16 promotion alloc failed` at layer 51; noshare → engine thread OOM). The
merge path's residency on a 27B all-linear is a separate open item.

The NVFP4 loss (0.3363) differs from the same run against the FP8 checkpoint
because the repack flushes lifted values under 2.0 to zero, so the two engines
hold different weights. See
`docs/experience/errors/2026-08-22-marlin-fp4-parity-wrong-oracle.md`.

## Rule

A quantized format is not an OPD base until the merge lane accepts it. Check
what the round does after the backward, not just whether a step runs.
