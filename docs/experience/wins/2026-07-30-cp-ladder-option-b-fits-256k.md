# CP seq-ceiling ladder: option B fits 256K on 4×H20 — no ring needed

Date: 2026-07-30
Scope: measurement only (no code change). Binary HEAD `562fc87ab`, `cuda,nccl`.
Driver: `scripts/run_cp_ceiling.sh` (untracked helper).

## Context

The N-D design doc's CP premise was "attention holds full KV + full scores → OOM
at 256K → must build the ring (option A)." That premise was never measured. This
ladder measures where option B (all-gather-prefix + seq-chunked recompute, the
committed default) actually peaks — the one number that decides whether the CP
ring / EP all-to-all / linear-attn boundary-grad sharding is needed at all.

## Measured (cp=4, 27B FP8 ThinkingCap, 4×H20 96 GB, synthetic writeback)

| seq | RUN_EXIT | peak mem/rank | forward | backward | outcome |
|---|---|---|---|---|---|
| 65536 | 0 | ~61 GB | 71s | 309s | FIT — optimizer step, loss 2.77 |
| 131072 | 0 | ~81 GB | 218s | 914s | FIT — optimizer step, loss 2.65 |
| 196608 | 0 | ~79 GB | 444s | ~30min | FIT — optimizer step, loss 1.78 |
| 262144 | 137* | **96.4 GB** | 749s | 3253s | **FIT, no OOM**, then hung post-backward |

*137 = my `kill -9` of the hung process, not the workload. Forward AND backward
both completed at 96.4/97.9 GB (~1.4 GB headroom), no OOM.

## What worked (the verdict)

**No rung OOMs. Option B alone reaches 256K.** Two facts the ladder settled:
- Peak does NOT scale O(full_seq) in the danger phase — 131072→196608 peak
  *dropped* (81→79 GB) because backward is seq-chunked-recompute-bounded, not
  KV-bound. At 262144 the runtime's `[ckpt-gate]` auto-engaged gradient
  checkpointing (modeled 1.99 TB without ckpt → fit under 96 GB with it).
- The doc's "full scores" term does not exist: the fused forward is a flash-2
  kernel (`nonpaged_prefill_attention.cu`), no `[seq,seq]` transient. The
  documented single-card walls are activations (q_proj grad, MLP intermediate),
  which option B already shards to O(seq/N).

Decision: the CP ring (option A), EP all-to-all, and linear-attn sharding are
NOT required to reach 256K on memory grounds. Do not build them for the ceiling.

## The real next wall (separate from the memory verdict)

262144 completed forward+backward with no OOM, then **hung ~40 min at 0% util in
the post-backward CP collective** — 3 of 4 ranks printed `phase=backward`, the
4th didn't; all parked in `hrtimer_nanosleep` (NCCL busy-wait). This is a
seq-scale-specific collective desync, NOT a memory wall. Ruled out (forensics +
controlled repro): the ckpt-gate (engage byte-identical across ranks), the
empty-shard `sum·0` path (no shard is empty at 65536 local), and generic
checkpoint-recompute-under-CP (seq=8192 + forced checkpoint completes in
lockstep). The surviving suspect is the >65535 chunked-SDPA branch
(`attention.rs:171`) — 262144/cp=4 is the only rung whose local seq (65536)
crosses it. Tracked as the next binding constraint, not this entry's win.

## Rule

Measure the wall before building the fix for it. The ring's flash-2 merge math
was authored and CPU-gated on the *assumption* that attention OOMs at 256K; the
ladder showed it doesn't. A device data-plane (CP ring, EP, PP) is gated behind a
measurement that names which term binds — a completed-vs-OOM ladder, not an
extrapolation. Here the answer was "none binds; option B fits," and the actual
next wall (a collective hang) was invisible to the memory argument entirely.
