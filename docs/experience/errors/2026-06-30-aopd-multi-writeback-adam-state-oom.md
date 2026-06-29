# Agent-OPD: Multi-Writeback Adam State OOM — 4 Accepted Trajectories × 50 GB Adam State Persists Between Writebacks

## Context

Run 3 of the Agent-OPD training loop on the 8×H20 box (GPU 4, 97508 MiB).
Config: `--share-frozen-base` (400 FP8 base projections zero-copy from rollout
engine), `--lora-layer-start 32`, `--samples-per-prompt 4`, `--rounds 1`,
`ARLE_OPD_WRITEBACK_OFFLOAD=1`. No `--writeback-cap`.

Prior walls cleared:
- ELKEID setsid → `process_group(0)` fix (`ea2e6133`)
- Two-copy base OOM → `--share-frozen-base` (default)
- N×N causal mask OOM → seq=7463 is safe (222 MB mask vs 1.32 GB at seq=18168)
- SDPA chunk intermediates → nested `checkpoint` fix (`0b7a1d89`)

Run 3 progress:
- Baseline eval: pass_rate=0.3333 (1/3)
- Rollout: 4/4 samples passed (turns=5 each), seq_len=7463 total, 626 target tokens
- **Writeback 1** (trajectory 0): SUCCESSFUL — `loss=0.181633`
  - forward_hidden_states: 964 seconds
  - fused_ce: 0.195 seconds
  - backward: 1269 seconds
  - optimizer_cleanup: 19 seconds
  - VRAM: 34155 → 46833 (forward) → 83825 MiB (backward peak)
- **Writeback 2** (trajectory 1): STARTED at 83825 MiB → killed (run 3 terminated)

## Root Cause

`masked_writeback_ce_step` is called ONCE PER ACCEPTED TRAJECTORY (4 times for
4 passes). The **AdamW optimizer state** (m + v tensors in FP32) is created
during the first writeback's `optimizer_cleanup` and **persists between writebacks**.

VRAM accounting after writeback 1 optimizer_cleanup:

| Component | VRAM |
|-----------|------|
| Base floor (model + shared FP8 base) | 34155 MiB |
| Adam m+v for LoRA params (FP32) | ~49670 MiB |
| **Total persistent** | **83825 MiB** |

The gradient tensors (another ~25 GB) ARE freed after optimizer_cleanup. But
the Adam state is **intentionally persistent** (accumulates across writeback
calls for proper momentum tracking).

For writeback 2 (same trajectory length):
- Starting VRAM: 83825 MiB (13683 MiB free)
- Forward adds checkpoints: +12678 → **96503 MiB** (within 97508 by only 1005 MiB)
- Backward needs gradient tensors: +~25 GB → **~121 GB** → OOM during backward

The optimizer state is ~50 GB because LoRA (rank=64, layers 32-63 of Qwen3.6-27B)
covers attention + MoE expert projections across 32 layers, yielding ~6.25B LoRA
parameters × 2 (Adam m+v) × 4 bytes (FP32) ≈ 50 GB.

## Fix

Add `--writeback-cap 1` to limit writeback to **1 trajectory per round**.

With cap=1:
- Only the first accepted trajectory trains
- `trained_pairs = 1` (not 4)
- One writeback: VRAM peaks at 83825 MiB (safe), then persists as Adam state
- Round ends before a second writeback would OOM
- `eval_round_1.jsonl` and LoRA adapter are written

The cap reduces training signal per round (1 instead of 4 examples) but is
the correct trade-off on a single H20 with 32-layer LoRA.

## Evidence

From `/tmp/aopd3_run.log` (run 3):
```
[masked-writeback] DONE loss=0.181633 total_targets=626
[agent-opd] released inference scratch
[agent-opd] released rollout KV pool
[opd-vram] agent-opd pre-writeback: used=83825MiB free=13683MiB total=97508MiB
[masked-writeback] offload_checkpoints=true
[masked-writeback] seq_len=7463 total_targets=626 chunk_rows=2048
[opd-vram] masked-writeback pre forward_hidden_states: used=83825MiB free=13683MiB total=97508MiB
```

The "pre-writeback" trace for writeback 2 shows 83825 MiB — same as writeback
1's backward PEAK. Confirmed: Adam state did not release.

## Rule

- **Without `--writeback-cap`, N accepted trajectories → N consecutive writebacks.**
  After writeback 1, the Adam optimizer state (~50 GB for 32-layer LoRA on 27B)
  persists permanently in GPU VRAM. Writeback 2 starts at 83 GB floor vs
  34 GB floor for writeback 1. At 97.5 GB limit, only 1 writeback fits.
- **Use `--writeback-cap 1` for single-GPU 27B model + 32-layer LoRA.** Multiple
  writebacks require either (a) a larger GPU cluster, (b) fewer LoRA layers,
  (c) smaller LoRA rank, or (d) 8-bit Adam (reduces optimizer state 4×).
- **`loss=0.181633` from writeback 1 proves the training mechanics work.** The
  wall is purely VRAM management between writebacks, not the writeback itself.

Claude-Session: https://claude.ai/code/session_01FWW6tY9aXDyx9n5NNCKEb6
