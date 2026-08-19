# CP ring FA3 pair offsets were halved for two days — one `* 2`, 6.4× wrong gradients — 2026-08-19

Fixes [`2026-08-19-cp-training-gradients-regressed-and-the-gate-is-dead.md`](../errors/2026-08-19-cp-training-gradients-regressed-and-the-gate-is-dead.md).
Commit `ad1192864`.

## Context

A timing refresh of the stale 2026-08-05 training rows found CP disagreeing with
single-card on a deterministic workload: 27B seq=32768 gave loss 11.664682 /
grad_norm 14.014 at cp=2 against 10.870087 / 2.197122 at cp=1.

## Root cause

`652f87cb8` (2026-08-17, "2D KV ownership sharding") dropped `* 2` from the five
run-offset expressions in the FA3 pair route — three in
`ring_block_fwd_merge_fa3`, two in `ring_block_bwd_fa3`:

```rust
- q: (q_ptr + (((bi * h * s) + pair.q.row) * d * 2) as u64) as *const ffi::Half,
+ q: (q_ptr + (((bi * h * s) + pair.q.row) * d)     as u64) as *const ffi::Half,
```

`q_ptr` is a raw `CUdeviceptr` **byte** address from `device_ptr()`, and the
element is bf16. Every run offset was halved.

**Why it stayed hidden for two days.** The offset is
`((bi*h*s) + pair.q.row) * d`. For a shard's FIRST run `pair.q.row == 0`, and at
batch 1 the whole expression is 0 — and 0 halved is still 0. Only the second and
later runs of a *non-contiguous* zigzag shard read from a wrong address. A cp
rank that happens to hold one contiguous range is unaffected.

## Verification

`cp_hidden_parity`, 2×H20, cp=2, seq=16 — the same gate that last passed at
`083e2e89a` (2026-08-16):

| | 2026-08-16 (last pass) | before fix | after fix |
|---|---:|---:|---:|
| `cp_vs_cpu_f32` | 3.16e-2 | **12.40047** | **3.159571e-2** |
| `single_vs_cpu_f32` | 3.49e-2 | 3.492630e-2 | 3.492630e-2 |
| `ce_cp_vs_cpu` | 3.90e-4 | 4.042e-2 | **3.901e-4** |
| verdict | PASS | FAIL | PASS |

The fix reproduces the last known-good numbers to their printed precision.

Per-row, before → after, rank 0 shard rows `[0,1,2,3, 12,13,14,15]`:

| row | run | before | after |
|---|---|---:|---:|
| g3 | first | 1.7282e-2 | 1.7282e-2 |
| g12 | second | **1.2400e1** | 1.9941e-2 |
| g13 | second | **7.8943e0** | 1.9047e-2 |
| g14 | second | **9.1841e0** | 1.7584e-2 |

rank 1 holds one contiguous run `[4..11]` and was correct throughout, before and
after.

End-to-end, 27B seq=32768, LoRA r16 α32 attention-qv, `--synthetic-writeback-seq`:

| arm | loss | grad_norm |
|---|---:|---:|
| 2026-08-05 baseline row, cp=2 | 10.871086 | 2.263385 |
| cp=1 today | 10.870087 | 2.197122 |
| cp=2 before fix | 11.664682 | 1.401418e1 |
| **cp=2 after fix** | **10.870859** | **2.152082** |

cp=2 loss now matches cp=1 to 9e-5 relative and the 2026-08-05 row to 2e-5;
grad_norm is within 2%, the bf16/MoE band.

## What worked — the sequence that found it

1. **A deterministic workload made the mismatch visible at all.** The synthetic
   writeback is reproducible by construction, so a 7% loss move was a fact, not
   noise.
2. **The cp curve separated the faults.** loss 10.870/11.665/11.699 and
   grad_norm 2.20/14.01/20.26 at cp=1/2/4: a cp-independent forward error plus a
   gradient error tracking ≈10.0·√cp. Two components, one cause.
3. **Cheap eliminations before any bisect.** LoRA rank (r16 and r32 give the same
   loss to every digit) and FlashQLA (its off-arm moved backward 43.4 s → 102.2 s,
   proving it engaged, and the numbers did not move).
4. **A restored host gate exonerated the ring math** — forward against
   full-softmax, backward against finite differences, on ragged blocks with
   zigzag positions. That pushed the search to the device dispatch.
5. **A passing dated gate bracketed the window.** The 2026-08-16 T1 entry
   recorded `cp_hidden_parity` PASS, which cut the range from 166 commits to the
   four code commits after it that touch the ring surface. Only one of those
   edited `ring_attention.rs`, and it edited exactly five pointer lines.

No bisect build was needed.

## Why three layers of gating missed it

- `652f87cb8`'s own gate was "needle ladder ×3 at world=4 — **pending-remote**".
  That is an *inference* gate; the commit also changed the *training* ring.
- The training CP gate (`cp_hidden_parity`) was never re-run after 2026-08-16.
- The 0.8B CP correctness arm, which compares cp=1/2/4 grad_norm, cannot run at
  all — FlashQLA has no kernel for that model's per-rank CP head geometry, so it
  errors instead of producing a number.

## Rule

A commit that edits a shared kernel must run the gates of **every** consumer, not
the gate of the feature it was written for. `652f87cb8` was an inference feature
whose diff reached into the training ring; its stated gate could not have caught
this even if it had run.

And when a pointer offset is bytes, say so in the type system or the name — the
deleted `* 2` was a bare literal with nothing to make its meaning survive a
refactor. The fix uses `size_of::<ffi::Half>()`.

## Follow-up

The cp=4 seq ladder ([entry](2026-08-19-cp4-seq-ceiling-229376-and-17x-step.md))
was measured on the broken path. Its walls and peaks stand as resource
measurements; re-run it on the fixed binary before treating 229376 as the
ceiling.
