# OPD writeback forward peak — as-you-go free — CUDA, 2026-07-28

> Status: Shipped (commit `e736c485a`). Forward wall cleared. The full step
> still OOMs in backward; that lever (seq-chunked attention forward-recompute in
> the checkpoint replay) is a standalone autograd/kernel project, **deferred —
> Phase 7a closed 2026-07-28**. See the Learnings note and the research doc's
> backward per-op probe for why. 40960 single-card lossless writeback is not a
> current product need; this win stands on its own target (forward peak freed).

## Goal

seq=40960 masked-CE writeback on one H20 (97508 MiB), `all-linear` LoRA — the config that OOMs in forward at checkpoint group 3 (SwiGLU `mul [40960,17408]`=2720 MiB, 15.9 MiB free). No precision/attention/sequence/mask/loss change.

## Hypothesis

The checkpointed forward runs tape-disabled and frees intermediates only once at the closure's exit (`checkpoint.rs` `free_new_except`), so the single-layer peak = the SUM of one layer's intermediates. Freeing dead SwiGLU + LoRA transients as each consumer finishes (`store.free()`, gated `!tape.enabled`) cuts that peak ~19-22 GB. Numerically a no-op.

## Parameters

```bash
ARLE_OPD_VRAM_TRACE=1 arle train agent-opd \
  --student-model ThinkingCap-Qwen3.6-27B-FP8 \
  --synthetic-writeback-seq 40960 \
  --writeback-offload true \
  --lora-target-set all-linear
```

- Baseline: clean HEAD `585e49337` — forward OOM at group 3/64, 15.9 MiB free.
- Treatment: `e736c485a` — `qwen35.rs` Dense-MLP + `lora.rs` forward free dead transients when `!tape.enabled`.

## Environment

- Host / GPU: single H20, 97508 MiB, GPU 1 pinned.
- Model / dtype: ThinkingCap-Qwen3.6-27B-FP8, LoRA masked-CE writeback.
- Parallelism: single GPU. Flags identical except the code change.

## Results

| metric | baseline `585e49337` | treatment `e736c485a` |
|---|---|---|
| forward groups completed | 3 / 64 (OOM) | **64 / 64** |
| per-group used (layers 1-63) | — | flat 77492 MiB, +8 MiB/group (retained ckpt inputs) |
| forward-end peak | died | used=79060 / free=18447 MiB |
| full-step rc | OOM (fwd) | OOM (**backward**) — wall moved |

Forward timing: `forward_hidden_states 322 s`, `fused_ce 1.8 s`. `live_tensors` 1850→1913 across 64 groups (only retained inputs grow; transients freed in-group). Loss not emitted — step died in backward, so the no-op claim is not yet loss-verified.

Raw: `/host/wb-diag/wb-run.log`, marker `/host/wb-diag/RUN_EXIT` (=1), launch `/host/wb-diag/run_wb.sh`.

## Problems

Full step still OOMs — the same 2720 MiB SwiGLU-shaped alloc now fails in **backward** (`add [40960,17408]`), not forward. Move 1 is `!tape.enabled`-gated so it does not fire on backward's tape-enabled recompute. At forward-end the mempool hoarded 37898 MiB (`pool_reserved=74656` vs `pool_used_current=35957`) while driver free=18447 — whether releasing that hoard lets backward complete, or it is live-bytes/fragmentation, is the next A/B (`--cuda-mempool-retain false`, no rebuild). Note the wall-decomposition doc's attention-qv arm freed to 56 GiB free and STILL failed (fragmentation), so hoard-release is not a guaranteed fix.

## Learnings

PASS on its target: the forward single-layer peak is cut and the group-3 death is gone — Move 1 does exactly what it claimed, numerically a no-op by construction. It is necessary at every length, insufficient alone: the wall is now backward. Next lever is the mempool retain A/B (cheap, no rebuild), then per its outcome either backward-side MLP liveness or chunked SwiGLU. GDN dead-store (old Move 2) targets the wrong shape and is deferred.
