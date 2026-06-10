# DSv4 c≥2 serving lane EXISTS: lockstep admission lands, batched decode +57% agg @ c=8 end-to-end

**Date:** 2026-06-10. **Backend:** CUDA DSv4-Flash FP8 TP=8/EP=8, 8×H20, CUDA 12.9,
allreduce MoE transport. **Commit:** `23b69249` (+ `cd421794` Layer-1 split).

## Goal

Make the DSv4 multi-rank serving stack run c≥2 at all (it crashed, then
deadlocked — the two stacked blockers of
[the Layer-2 error entry](../errors/2026-06-10-dsv4-c2-layer2-lockstep-admission-deadlock.md)),
then measure the first end-to-end concurrent throughput numbers.

## Hypothesis

Per-tick admission broadcast (SGLang `recv_requests`+`broadcast_pyobj` shape —
one `TickAdmissions{seq, requests}` before every rank-0 engine step, workers
stepping a directly-owned engine exactly once per envelope) closes the
admission-vs-tick race; with the lane running, `INFER_DSV4_BATCHED_DECODE=1`
(Phase 6a grouped MoE, previously harness-only) scales aggregate throughput
with c.

## Params / Env

`arle serve` 8-rank multiproc, `INFER_DSV4_MAX_SEQ_LEN=16384`, deepgemm-native,
needle-free short prompts. Harness `scripts/dsv4_c_sweep.py` (per-c burst +
1 s-stagger lanes, 96 tok/req, distinct capital-city prompts with
expected-substring contamination probes) + `dsv4_ab_bench.py` (B=1 lane).
Same binary throughout; the batched arm is the same serve re-launched with
`INFER_DSV4_BATCHED_DECODE=1`.

## Results

Correctness/liveness (previously: instant engine death pre-`cd421794`, NCCL
deadlock pre-`23b69249`):

- **All 6 lanes × both arms complete; server alive throughout; every request
  returns.** Probe hits 19–24/24 per sweep; every miss is a coherent on-topic
  continuation of its OWN prompt (Japan→"world's largest city by population,
  with over 13 millio…", Egypt→"the Egyptian capital…") — zero cross-slot
  contamination.
- Plan fingerprints (debug run): **1150/1150 ticks byte-uniform across ranks
  1–7**, including **140 Mixed (prefill+decode) ticks** — the exact tick-9
  shape that deadlocked pre-fix now executes uniformly.

Throughput (aggregate tok/s over the lane window):

| lane | per-row (batched OFF) | batched ON | Δ |
|---|---|---|---|
| B=1 p50 | 38.85 | 37.97 | −2.3% (B=1 never takes the batched path; day-noise) |
| c=2 burst / stagger | 37.96 / 38.51 | 54.94 / 48.59 | **+45% / +26%** |
| c=4 burst / stagger | 39.53 / 39.59 | 56.51 / 56.41 | **+43% / +42%** |
| c=8 burst / stagger | 38.81 / 39.13 | 60.78 / 61.68 | **+57% / +58%** |

- B=1 lockstep overhead: 38.85 vs same-day pre-lockstep control 38.98 =
  **−0.3%, noise** — one localhost TCP envelope per 26 ms step is free.
  (A −4.3% reading earlier was the `RUST_LOG=infer_cuda=debug` fingerprint
  logging, not the lockstep — measure clean.)
- Per-row arm is flat in c (~38–39.6 agg) as expected: N sequential single-row
  forwards per step. Batched arm scales; absolute ceiling is still the per-row
  attention loop (Phase 5 batched FlashMLA pending — harness-direct Phase 6a/6b
  measured 51.75 agg @ c=8 decode-only; serving adds prefill interleave, and
  61.7 here includes prefill windows).

## What landed (`23b69249`)

- `RelayEnvelope::TickAdmissions{seq, requests}` supersedes the per-request
  `Request` relay; rank-0 `engine_loop` drains→broadcasts→admits→steps so
  exactly one envelope precedes every step (empty on pure-decode ticks).
- Workers: `CudaWorkerEngine` (directly-owned `infer_core::Engine`, no
  background loop) stepped once per envelope; `seq` gap = fatal.
  `submit_replicated` + the ServeHandle worker path deleted.
- Config parity: coordinator serializes its resolved `EngineLoadConfig` into
  `ARLE_WORKER_ENGINE_CONFIG`; workers refuse to run without it.
- Soundness preconditions verified in-tree: CUDA `submit` synchronous + `poll`
  always `Ready` (one forward per `Engine::step`); sampling
  `(seed,position)`-SplitMix64 deterministic (`seed=None→0`); plan-building
  pure in engine state.

## Problems / residual

- Phase 5 (batched FlashMLA b=N) is now the throughput ceiling: attention is
  still a per-row loop inside the batched driver.
- `INFER_DSV4_BATCHED_DECODE` stays opt-in pending a default-flip c-sweep with
  TTFT/ITL percentiles (this entry's gate is lane-existence + aggregate).
- deepep_ll batched A/B is now RUNNABLE (next: batched-decode MoE over the LL
  transport — `forward_decode_batch_stream_impl` still refuses deepep).
- rank0 emits no `[dsv4-plan]` DEBUG lines (worker-side uniformity carried the
  check); coordinator logger filter still to fix.
- Original design sketch (admit-tick stamping with margin K) was killed in
  review: empty plans don't synchronize (counter drift) and idle stamps are
  unreachable (self-deadlock). Message-per-step has neither hole.

## Learnings

- **Two stacked blockers need two verified fixes**: the executor `ensure!` was
  the loud one; the admission race was the structural one. The lane only
  counts as existing after burst+stagger lanes pass with liveness + per-rank
  fingerprint uniformity.
- **Make admission part of the lockstep, not a racer against it** — any
  free-running ingestion into per-rank schedulers desyncs on timing alone, no
  matter how deterministic the planners are.
- Expected-substring probes need semantic slack (greedy continuations may
  paraphrase); judge MISSes by coherence-with-own-prompt before suspecting
  contamination.

## Refs

- Plan: [`docs/plans/2026-06-10-dsv4-multirank-lockstep-admission.md`](../../plans/2026-06-10-dsv4-multirank-lockstep-admission.md)
- Layer-2 evidence: [`errors/2026-06-10-dsv4-c2-layer2-lockstep-admission-deadlock.md`](../errors/2026-06-10-dsv4-c2-layer2-lockstep-admission-deadlock.md)
- Phase 6a harness numbers: [`wins/2026-06-07-dsv4-batched-decode-grouped-moe-throughput.md`](2026-06-07-dsv4-batched-decode-grouped-moe-throughput.md)
- Pod logs: `c_sweep_lockstep{,_clean,_batched}.log`, `ab_lockstep_*.log` (build dir)
