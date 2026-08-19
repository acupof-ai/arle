# CP training gradients regressed 6.4× against single-card, and the gate that would catch it cannot run — 2026-08-19

## Context

Refreshing the stale timing rows in `docs/baselines.md` on `9c2c84675`. The
workload is deterministic (`--synthetic-writeback-seq` builds the trajectory
from `(i % 30000) + 1`, the base is frozen, LoRA `B` is zero-init), so the
loss at step 1 is reproducible. The refresh compared against the 2026-08-05
rows and the numbers did not match.

## Phenomenon

27B, seq=32768, LoRA r16 α32 `attention-qv`, `/data00/ThinkingCap-Qwen3.6-27B-FP8`:

| arm | loss | grad_norm |
|---|---:|---:|
| 2026-08-05 baseline row, cp=2 (`15caff0d0`) | 10.871086 | 2.263385 |
| 2026-08-19 **cp=1** | **10.870087** | **2.197122** |
| 2026-08-19 cp=2, FlashQLA on | 11.664682 | 1.401418e1 |
| 2026-08-19 cp=2, FlashQLA off | 11.665197 | 1.403425e1 |

Single-card reproduces the 2026-08-05 numbers — loss to 4 decimals, grad_norm
to 3%. **cp=2 does not: loss +7.3%, grad_norm 6.4×.** The 2026-08-05 row
states "Both ranks print identical loss and grad_norm (post-all-reduce)" at
10.871086 / 2.263385, so CP agreed with single-card then and does not now.

The 0.8B correctness model behaves the same way on the single-card side:
cp=1 grad_norm 3.466840 today against the 2026-08-05 correctness row's
3.464900 (5.6e-4).

Under CP the sequence-sharded gradient is all-reduced, so cp=1 and cp=N are
supposed to agree — the 2026-08-05 correctness rows assert exactly that
(cp=1 3.464900 / cp=2 3.459982 / cp=4 3.464276, spread 1.4e-3).

## Cause

Unknown. Ruled out by measurement:

- **Not LoRA rank/alpha.** r16/α32 and r32/α64 both give loss 11.664682 to
  every digit — `B` is zero-init so the adapter contributes nothing at step 1.
  (The 2026-08-05 row does not record its rank; this closes that gap.)
- **Not FlashQLA GDN chunkwise.** `--gdr-chunkwise-prefill false` moves
  backward 43.4 s → 102.2 s (2.35×), so the arm demonstrably engaged, and the
  loss/grad_norm are unchanged at 11.665197 / 14.034.
- **Not the single-card path.** cp=1 reproduces the old numbers on both models.

166 commits touch `crates/train/src`, `crates/autograd/src`, or
`crates/cuda-kernels/csrc/attention` between the 2026-08-05 rows and
`9c2c84675`. A bisect over that range is the next step.

## Why it went unnoticed

The comparison that would have caught it is dead. The repo's CP correctness
rows are cp=1 vs cp=2 vs cp=4 grad_norm on `qwen35-08b-clean` at seq=2048.
Run today, the cp=2 arm does not produce a number — it errors:

```
flashqla GDN head geometry H=8/Hg=8 not built
(have 32/16, 48/16, 24/8, 12/4, 16/8, 16/16)
```

FlashQLA went default-on at `15caff0d0`/`fa742a038` (2026-08-05) and has no
kernel for the 0.8B's per-rank geometry under CP, so the gate model cannot run
the very configuration it exists to check. The 27B's CP geometry is 24/8, which
IS in the built list — so the 27B runs, and produces wrong gradients quietly.

## Impact

Every CP training number measured after 2026-08-05 was produced by a path whose
gradients do not match single-card. That includes today's cp=4 seq ladder
([wins](../wins/2026-08-19-cp4-seq-ceiling-229376-and-17x-step.md)): its wall
clock and peak-VRAM figures stand as resource measurements — the same tensors
are allocated and the same kernels run — but the ceiling was not measured on a
numerically correct path, and no CP training run since 2026-08-05 should be
treated as converging on the right gradient.

## Rule

A correctness gate that errors is not a correctness gate that fails — it
disappears. FlashQLA's default flip removed the 0.8B CP arm's ability to
produce a number at all, and a missing row reads the same as a row nobody
re-ran. Any gate whose arm can go from "value" to "error" needs the error to
be loud where the gate's results are read, not only in the run log.

## Follow-up

1. Bisect `15caff0d0`..`9c2c84675` on 27B cp=2 seq=32768 grad_norm; the arm is
   ~85 s per point.
2. Build the FlashQLA H=8/Hg=8 geometry, or make the 0.8B CP correctness arm
   run on the recurrent path, so the gate produces a number again.
3. Re-run the cp=4 seq ladder once the gradient regression is fixed — the
   ceiling may move if the fix changes what backward allocates.
