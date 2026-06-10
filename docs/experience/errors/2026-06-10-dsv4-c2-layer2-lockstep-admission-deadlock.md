# DSv4 c≥2 has a second blocker behind the single-row ensure: cross-rank admission desync → NCCL deadlock

**Date:** 2026-06-10. **Backend:** CUDA DSv4-Flash FP8 TP=8/EP=8, 8×H20, allreduce MoE transport.
**Binaries:** `cd421794` (mixed-plan split) + `78553406` (plan fingerprint log).

## Context

The 2026-06-10 deepep_ll entry blamed c≥2 engine death on the DSv4 executor's
single-row `ensure!`. Fixing that (Layer 1, `cd421794`: per-prefill sub-steps +
decode sub-batch via `KvBatchDescriptor::subset`, planner untouched) revealed a
second, deeper blocker.

## What happened (evidence, same pod session)

- B=1 sanity through the refactored submit: 5.8 s, `" Paris…"`, correct — the
  single-prefill plan now routes through `subset(0..1)` (identity rebase) and
  decode ticks through the unchanged decode-only branch. Layer 1 does not
  regress B=1.
- c=2 burst: **no crash — a deadlock.** Both requests hang to a 90 s HTTP
  timeout; all 8 GPUs sit at 100% util / ~120 W (collective spin, not work).
- Plan fingerprints (`[dsv4-plan] rank tick prefill decode`,
  `RUST_LOG=infer_cuda=debug`): all 7 workers byte-uniform ticks 0–8 (sanity
  prefill + 7 decodes + c2-request-A prefill `(0,0,5)`); **tick 9 on every
  worker is `decode=[(0,5)]` with no prefill row for request B, and tick 10
  never appears.** rank 0 alone had drained B from its HTTP-local queue → its
  tick 9 was Mixed (prefill-B sub-step + decode-A) → different NCCL collective
  sequence/shape than the workers' decode-only tick → all ranks spin forever.
  (A uniform tick 9 would have completed in ~25 ms like ticks 1–7.)

## Root cause

The multi-rank serve runs a full symmetric Engine per rank fed by an **async**
request relay (rank 0 broadcasts at `admit_submission`; worker relay threads
inject into their engines whenever the TCP message lands). "Identical request
order ⇒ identical batches" fails because admission is timing-coupled to each
rank's tick top: a request landing inside the race window (the whole 25–160 ms
step) is planned into forward #k on one rank and #k+1 on another. B=1 cannot
hit the race (single in-flight request); any c≥2 hits it almost immediately.
Documented as multirank follow-up #3 (2026-06-08) — now reproducible at will
with the fingerprint log.

Second source, config: workers build with `EngineLoadConfig::default()` while
rank 0 uses the CLI-resolved config — any non-default planner knob diverges
deterministically (likely mechanism of follow-up #2, the ≥2-chunk prefill
crash at c=1).

## Fix direction

Per-tick admission broadcast (SGLang `recv_requests` + `broadcast_pyobj`
shape): rank 0 drains its queue at tick top, sends `TickAdmissions{seq, L}` to
workers, all ranks admit the same L before planning the same tick. Design +
file map: [`docs/plans/2026-06-10-dsv4-multirank-lockstep-admission.md`](../../plans/2026-06-10-dsv4-multirank-lockstep-admission.md)
— awaiting ck ack (architectural).

## Rule

- **"c≥2 crashes" was two stacked blockers, not one.** Fixing the loud first
  failure (executor ensure) is necessary but not sufficient; verify the lane
  end-to-end before declaring it exists. Layer 1's mixed-tick execution is
  itself only verifiable after Layer 2 lands (mixed ticks currently only occur
  divergently → deadlock strikes first).
- **100% GPU util at ~⅓ TDP with zero progress = collective desync spin**, not
  load. Check power draw before believing utilization.
- **A per-forward plan fingerprint log is the cheapest distributed-lockstep
  oracle**: 7 worker streams uniform → silence at tick 9 localized the
  divergence to rank 0's admission timing in one capture, no GPU-side
  debugging at all. Keep `[dsv4-plan]` (`78553406`) as the permanent
  regression surface for every multirank change.
- deepep_ll batched-lane A/B remains blocked on this; B=1 deepep root-cause is
  NOT blocked (single request) — measured separately.

## Refs

- `docs/experience/errors/2026-06-10-dsv4-deepep-ll-b1-regression-no-batch-lane.md` (Layer 1 discovery)
- `docs/experience/wins/2026-06-08-dsv4-multirank-serve-rewire.md` (follow-ups #2/#3 first noted)
- Pod artifacts: `desync_repro.log`, `c_sweep_allreduce_perrow.log`, `plan_diff.py` (build dir)
