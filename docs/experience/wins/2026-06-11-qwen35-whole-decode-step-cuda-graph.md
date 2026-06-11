# Qwen3.5/3.6 whole-decode-step CUDA graph (env-gated OFF)

**Date:** 2026-06-11. **Backend:** CUDA, Qwen3.6-35B-A3B, H20, TP=1 only.
**Status: pending-remote** — same-binary `ARLE_QWEN35_DECODE_GRAPH=0/1` flip
on the pod; license ≥ +10% tok/s + needle gate + replay-counter evidence.

## Context

Post device-router decode sits at 40.8 tok/s vs ~750 tok/s HBM floor; the
per-token cost is ~1,074 kernel launches' host issue + inter-launch gaps —
the orchestration-bound regime where DSv4's whole-step graph verdict (WASH,
GPU-bound) inverts. The three blockers fell in order: host MoE routing
(`874f8cfb` device router), per-call allocations (`1e0f05e1` workspace —
plus argmax's last alloc moved to a persistent slot in this tranche),
decode-shape variability (B=1, R=8, seq=1 constant; hybrid `878c5ff2` keeps
decode off DeepGEMM JIT).

## What Worked (implementation; numbers pending)

- Capture scope embedding → 40 layers → final norm → lm_head GEMV, ending at
  the persistent logits slot; sampling outside (dense pattern). 18-kernel
  capture-safety table on `forward_decode_step_captured`.
- Per-token scalars device-resident: token id + position staged pre-replay
  (dense stage1 pattern); NEW `nonpaged_prefill_attention_devpos_cuda` reads
  kv_len from the same device buffer the prep kernel already used (eager
  decode uses the same entry — no parallel half-state path).
- ONE graph key per slot (state addresses bake per-slot; ≤ num_slots keys);
  dedicated decode workspace so prefill can never reshape captured
  addresses; epoch + 3-pointer bake fingerprint guard; request-boundary
  rearm (warm-1 then recapture-free replay); OPD offload/LoRA invalidate.
- Replay/capture AtomicU64 counters + periodic log — the bench license
  requires REUSE evidence, not capture-exists (skill anti-pattern #6/#26).
- Bonus hardening in graph.rs: a mid-closure error during capture now
  terminates the capture before propagating — previously the stream stayed
  in capture mode and poisoned all subsequent eager work (latent bug for
  every graph user).
- Gates: `ARLE_QWEN35_DECODE_GRAPH` default OFF ∧ TP=1 (NCCL not capturable)
  ∧ device-router eligible; any capture failure → warn + eager downgrade.

## Formula

24.5 ms/token − (1,074 × 3–8 µs host issue + reclaimed gaps) → predicted
**14–19 ms/token (+30–75% tok/s)**; HBM floor 1.7 ms remains far below —
post-license re-profile (nsys) re-ranks the next lever.

## Rule

- A graph WASH verdict is regime-specific: re-evaluate capture when the
  binding constraint flips from GPU-bound to host-issue-bound (and vice
  versa) — the same lever flips sign across regimes.
