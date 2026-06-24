# SPMD multiproc redesign (B) — DSv4 deepep_ll EP boot deadlock FIXED, EP=4 fully works

## Context

The deepep_ll (NVSHMEM EP) path deadlocked at boot
([root cause](2026-06-24-cuda-moe-fp8-serving-ep-pod.md)): rank 0 ran in the parent
process as BOTH the relay coordinator AND TP rank 0's in-process engine — a
control/data-plane role conflation. During the deepep_ll boot collectives the
coordinator role could park rank 0 off the barrier while workers spun → hang.

Fix = SPMD ([design](../../plans/2026-06-24-multiproc-control-data-plane-redesign.md),
[spec](../../plans/2026-06-24-multiproc-B-impl-spec.md)): the parent becomes a thin
coordinator owning NO TP rank; all N ranks are symmetric spawned workers → deadlock
impossible by construction. Commit `f72d94f3` (workflow-implemented, adversarially
reviewed + repaired, comments-tightened).

## What Worked (8×H20 pod, commit `f72d94f3`, EP=4 GPUs 4-7)

- **Boot deadlock GONE**: EP=4 deepep_ll serves **READY in ~18-24 s** (was an
  infinite barrier hang). The EngineReady handshake (coordinator waits for all N
  engines before opening HTTP) + symmetric spawn did it.
- **EP correctness PROVEN**: needle exact ×3 DET at 115→2000 (sequential through the
  coordinator → relay → worker → completion path). 4000/8000 NONDET = the prompt
  exceeds `INFER_DSV4_MAX_SEQ_LEN=4096`, not EP.
- **EP concurrency WORKS**: clean c跑 (short prompts) **c=1/2/4/8/16 all ok=N/N**,
  zero owned-token overflow, ~40-60 tok/s aggregate (c=8 peaks 60). VRAM 93.6/96 GB
  per rank at `max_tok=512`.
- **The earlier "concurrency NON-FUNCTIONAL / hard memory blocker" verdict was
  WRONG** — it was the boot deadlock PLUS a confound: the prior c跑 ran right after
  a needle whose 8000-tok requests (over the 4096 arena) spiked `owned_n` to 904 >
  512 and crashed the workers. Short-prompt concurrency never overflows.

## Verdict
DSv4 deepep_ll EP is FUNCTIONAL end-to-end at EP=4 (boot + correctness + concurrency).
TP=1 path byte-unchanged (B gated on `world_size>1`). Coordinator path is
non-streaming (blocking completion); per-token streaming over the relay is a later
stage.

## Open
- **EP=8 (production, GPUs 0-7) untested** — ckl reclaimed GPU 0,1 mid-window; needs
  all 8 free. Expected to work (same B path; weights halve → more `max_tok` headroom).
- **Long-prompt-under-concurrency overflow**: chunked prefill must cap per-forward
  tokens at `num_max_dispatch_tokens_per_rank` for deepep_ll (else a long prompt's
  prefill chunk can exceed the LL cap). Short/normal serving is unaffected.
- shm relay transport — still KILLed (per the design's measurement); revisit only if
  the per-step relay RTT proves material at high concurrency.

## Rule

- **A "structural / non-functional" verdict on a feature that won't even boot is
  suspect until the boot path itself is fixed.** EP "concurrency" was declared a hard
  blocker while the real wall was a boot deadlock; once B fixed boot, concurrency
  worked. Fix boot first, then judge the next layer — and isolate confounds (the
  long-prompt needle residue) before blaming the layer under test.
- **A role-conflation deadlock is fixed by separating the roles, not patching the
  ordering** — SPMD (thin coordinator + symmetric workers) makes the deadlock
  impossible by construction.
