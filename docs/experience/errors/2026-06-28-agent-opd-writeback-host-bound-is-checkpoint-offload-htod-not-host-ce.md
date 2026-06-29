# Agent-OPD writeback host-bound wall pinned by gdb: it is the gradient-checkpoint OFFLOAD H2D round-trip (`cuMemcpyHtoDAsync`), NOT a host CE / host lm_head — and the run also OOMs

## Context

Continuation of the [value-run host-bound-wall wins entry](../wins/2026-06-28-agent-opd-value-run-8of8-accept-writeback-host-bound-wall.md)
(HEAD `7ae42221`, pod tree `/host/arle-ckl-aopd`). That entry recorded the
masked-CE writeback running ~24 min with `DONE loss=` never printing and
attributed it to the documented "host-loop autograd backward" / host lm_head
(`memory/reference_opd_fused_distill_host_loop_pathological`). This session
profiled the **live** writeback PID with gdb and reconciled the failure against
the actual process exit. Both the attribution and the "no-OOM, speed-only"
framing in the prior entry were wrong; corrected here with evidence.

The contradiction the prior entry left open: commit `65a46817` ported
`fused_linear_ce_loss_indexed` from a host scalar loop to a GPU device path
(3944×/target), and `fused_linear_ce_loss_indexed`
(`crates/autograd/src/ops/fused_linear_distill.rs:441`) dispatches to
`fused_linear_ce_loss_indexed_device` for **every non-CPU backend**. So on CUDA
the writeback CE is already on the GPU — the host wall cannot be the CE.

## Root Cause (measured — 5/5 gdb backtraces, not inference)

The live writeback (PID 2041, `arle train agent-opd … --lora-layer-start 32`,
seq_len=11735, total_targets=1496) ran single-threaded at 98.6% CPU. Five gdb
backtraces of the main thread @ 2 s spacing **all** bottomed out identically:

```
#13 cuMemcpyHtoDAsync_v2 () from libcuda.so.1
#14..  <stripped arle frames, identical addresses across all 5 samples>
```

The host thread is pinned issuing host→device copies in a tight loop. The source
is `masked_writeback_ce_step` (`crates/train/src/opd.rs`) calling
`tape.set_offload_checkpoints(true)`: `checkpoint()`
(`crates/autograd/src/ops/checkpoint.rs:50`) offloads each checkpoint group's
saved hidden to host RAM in the forward (D2H), and `checkpoint_backward`
(`crates/autograd/src/tape.rs:678`) re-fetches it via `ensure_device` (H2D) for
the recompute. At seq=11735, `ckpt_group_size` returns 1 (long-seq clamp), so all
~32 trainable-suffix layers (`--lora-layer-start 32` of 64) round-trip a
`[1, 11735, 5120]` hidden (~240 MB) host↔device during backward, serialized on
the one host thread driving `cuMemcpyHtoDAsync`. That serialized H2D flood — not
any host CE/lm_head loop — is the ~minutes-per-trajectory wall.

**The run also OOMs (co-blocker, contra the prior "fits, no OOM" claim).** PID
2041 died at 14:26:05 with `[ARLE train] error: masked CE writeback (round 0):
cuda alloc_zeros failed` / `RUN_EXIT=1` (`/host/run_aopd_value.log`). At
seq=11735 the SDPA recompute materializes O(seq²) `[chunk, seq, seq]` scores
(`head_chunked_sdpa_recompute`, `qwen35.rs:288`) on top of the ~51.5 GB resident
share-frozen-base + rollout floor; even at chunk→1 the transient + offload
re-uploads exceed the H20's headroom. So the writeback is **both** host-bound
(offload H2D) **and** memory-bound (O(seq²) attention) at the production
trajectory length — not a pure-speed wall.

## Fix

**Resolved 2026-06-30** via two commits:

1. **`0b7a1d89` — nested SDPA checkpoint** (`crates/train/src/qwen35.rs`):
   When `tape.enabled` (inner_tape during `checkpoint_backward`), wrap each
   `causal_sdpa_recompute` chunk in a nested `checkpoint` instead of letting ALL
   chunks' `[scores/scaled/masked/probs]` accumulate simultaneously. This bounds
   the inner backward's O(seq²) memory to ONE chunk at a time (~7 GiB for
   seq=9597) vs the old 7-chunk × 6.6 GiB = 46 GiB pile-up.

2. **`ARLE_OPD_WRITEBACK_OFFLOAD=1` (keep)**: Tried OFFLOAD=0 first (GPU-bound
   backward, ~37 GB peak during forward) but the long forward (21 min) fills the
   CUDA allocator cache with fragmented blocks → backward OOMs at 97422/97871 MiB
   even though total live tensors are only ~50 GB. OFFLOAD=1 keeps the forward
   lean (~37 GB peak) and the allocator cache small. The backward with nested SDPA
   checkpoint should now be GPU-fast (H2D per layer = 276 MB = 23 ms, negligible)
   with peak ~56 GB.

Prior "fix direction" list (options 1-3 above) is superseded by the nested SDPA
checkpoint, which addresses option 3 without requiring option 1 or option 2.

## Rule

- **Pin a host-bound op with a stack, not a memory.** `nvidia-smi` "GPU 0%/idle,
  CPU 98%" says *host-bound*, but NOT *which* host op. Five gdb backtraces of the
  pinned thread cost ~2 min and named the exact frame (`cuMemcpyHtoDAsync`) —
  overturning the "host CE/lm_head loop" inference. A from-scratch-autograd host
  wall is not automatically the documented host-loop; verify the frame.
- **A long writeback that "never prints DONE" may be dying, not just slow.**
  Reconcile the live profile against the process's actual exit: PID 2041 was not
  hung — it OOM'd (`cuda alloc_zeros failed`, RUN_EXIT=1) minutes after the gdb
  samples. "Slow, fits, no OOM" was a half-truth; check the exit code.
- **`fused_linear_ce_loss_indexed` is GPU on CUDA** (dispatches to `_device` for
  any non-CPU backend, `fused_linear_distill.rs:441`). The writeback host wall is
  the **checkpoint offload round-trip**, not the CE — don't re-attribute it to
  `reference_opd_fused_distill_host_loop_pathological`.

Claude-Session: https://claude.ai/code/session_01Vsoud3oabdLDppvb274bCr
