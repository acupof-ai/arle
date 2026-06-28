# Agent-OPD writeback: first GPU-completed masked-CE loss captured (3.82), host-bound op pinned by gdb (checkpoint-offload H2D, not host CE) — full `trained_pairs`/held-out Δ still blocked by node-governance crash-loop

## Context

Continuation of the [value-run host-bound-wall entry](2026-06-28-agent-opd-value-run-8of8-accept-writeback-host-bound-wall.md).
Goal (per brief): pin the writeback host-bound op with EVIDENCE, then capture
`trained_pairs`/`mean_loss` + per-round held-out Δ — or name the exact remaining
wall. Pod tree `/host/arle-ckl-aopd`, existing binary built Jun 28 12:36 (HEAD
`7ae42221`-era), data on `/host`.

## What Worked (case-as-fact, measured)

**Host-bound op PINNED with 5/5 gdb backtraces — it is the gradient-checkpoint
OFFLOAD H2D round-trip, NOT a host CE/lm_head loop.** The live value-run writeback
(PID 2041, seq=11735, 1496 targets, 98.6% CPU single-thread) was sampled with gdb
×5 @ 2 s; every backtrace bottomed out in `cuMemcpyHtoDAsync_v2 (libcuda.so.1)`.
Source: `masked_writeback_ce_step` sets `tape.set_offload_checkpoints(true)`
(`opd.rs`), so `checkpoint()` (`checkpoint.rs:50`) offloads each group's hidden to
host (D2H) and `checkpoint_backward` (`tape.rs:678`) re-fetches via `ensure_device`
(H2D) during the backward recompute, serialized on the one host thread. This
**overturns** the prior entry's "host-loop autograd / host lm_head" attribution:
`fused_linear_ce_loss_indexed` already dispatches to the GPU `_device` path for any
non-CPU backend (`fused_linear_distill.rs:441`), so the CE is on-device; the host
wall is the offload H2D. Full root-cause in
[the errors entry](../errors/2026-06-28-agent-opd-writeback-host-bound-is-checkpoint-offload-htod-not-host-ce.md).

**The live run did not just hang — it OOM'd.** PID 2041 died at 14:26:05 with
`[ARLE train] error: masked CE writeback (round 0): cuda alloc_zeros failed`,
`RUN_EXIT=1` (`/host/run_aopd_value.log`). At seq=11735 the SDPA recompute
materializes O(seq²) `[chunk, seq, seq]` scores
(`head_chunked_sdpa_recompute`, `qwen35.rs:288`) on top of the ~51.5 GB resident
share-frozen-base floor → OOM. The writeback is BOTH host-bound (offload H2D) AND
memory-bound (O(seq²) attention) at the production trajectory length. Corrects the
prior entry's "fits, no OOM, speed only".

**First GPU-completed masked-CE loss captured.** Synthetic writeback
(`--synthetic-writeback-seq 512`, 256 masked targets, offload ON) on a free H20 ran
fully GPU-bound (`nvidia-smi` 100% / ~360 W throughout) and printed:
```
[masked-writeback] DONE loss=3.819344 total_targets=256
[synthetic-writeback] DONE loss=3.819344 elapsed=166.036491299s
```
A finite, sensible CE → the GPU writeback path is correct end-to-end and DOES
complete; the wall is per-trajectory throughput, not correctness. At seq=512 the
GPU stays 100% (offload H2D is tiny; the cost is the 32-layer MoE backward
RECOMPUTE) with brief 0% dips at each checkpoint-group offload boundary — the same
mechanism that floods the host thread at long seq.

**Baseline held-out eval (un-tuned student) reproduced: pass_rate = 0.3333 (1/3).**
`0ea40e0` pass `[exit 0]`, `12734fa` fail `[exit 4]`, `5e36960` fail `[exit 1]`,
all 3 edited — no timeouts/errors mis-bucketed as failures (clean harness). Dumped
to `/host/aopd_evalout_value/eval_round_base.jsonl`. This is the baseline the
round-1 Δ would be read against.

**Case-decoded the accepted training rollout.** Task `ansible__ansible-f327e65`,
8/8 samples `passed=true [exit 0]`: the student located the FQCN-validation gap
(`is_valid_collection_name` in `_collection_finder.py` uses bare regex `^\w+\.\w+$`
while `_is_fqcn` in `dataclasses.py` checks `not iskeyword(...)`), added
`import keyword`, and rewrote the validator to reject Python-keyword parts — the
gold fix. Training targets are genuine and high quality.

## Result — full `trained_pairs`/`mean_loss` + held-out Δ BLOCKED by node-governance crash-loop (NOT code)

The numeric `trained_pairs`/`mean_loss` and round-1 held-out Δ could NOT be
captured. A reduced-cost real run (`--max-turns 6 --max-tokens 512
--samples-per-prompt 2 --writeback-cap 1`, existing binary) was launched to fit the
writeback in a short window, but the `sglang-test` container is in a hard
**CrashLoopBackOff** (observed attempt 7+, container ID changed mid-session,
crash cadence ~2-3 min then exponential backoff → DOWN across 3 consecutive 25 s
checks). Every relaunch died mid-baseline-eval (got through 2 of 3 eval tasks
before the container bounced); the load (~90 s) + baseline eval (~3-5 min) +
rollout + writeback chain cannot complete inside a 2-3 min window. The persistent
`/host` state (model, data, binary, my code edits) survives each restart, but the
process does not. This is the documented node-governance kill
([agent-opd-full-loop-killed-by-node-governance-not-code](2026-06-27-agent-opd-full-loop-killed-by-node-governance-not-code.md),
memory `project_new_h20_sglang_box_devops`), not a code defect — proven by the
seq=512 writeback completing cleanly on the same binary.

## Honest verdict (a/b/c per the brief)

**(c) — still cannot capture the full signal; exact remaining wall named.** Two
walls, both attributed:
1. **Writeback throughput/OOM (code, characterized):** at the production seq=11735
   the writeback is host-bound on checkpoint-offload H2D *and* OOMs on O(seq²)
   attention. The captured seq=512 loss (3.82, 166 s/256-targets) proves the path
   is correct and bounds the cost. Landed an `ARLE_OPD_WRITEBACK_OFFLOAD=0` gate
   (default unchanged) + per-phase timing to make the next attempt GPU-bound at
   moderate seq; the structural fix (async/batched offload, or bound the O(seq²)
   recompute) is the license-or-kill target.
2. **Container CrashLoopBackOff (infra, blocking):** the immediate blocker to the
   end-to-end numbers. Needs a stable container window ≥ ~10 min (or a checkpointed
   resume) — outside code. Bounded retry exhausted this session.

A rigorous capability claim would still need multi-seed (≥5) + Wilson CI on a
larger held-out set (n=3 is noise-dominated) even once the run completes.

## Bench

Exempt: agent-OPD training path, not a serving hot path. The committed code change
(`opd.rs` offload gate + phase timing, commit `5cc3df28`) defaults to offload ON =
prior behavior; default serve/CLI byte-identical. Captured numbers are above.

## Rule

- **Pin a host-bound op with a STACK, not a `nvidia-smi`.** "GPU idle, CPU 98%"
  says host-bound but not which op; 5 gdb backtraces (~2 min) named
  `cuMemcpyHtoDAsync` and overturned the "host CE loop" inference. A from-scratch
  autograd host wall is not automatically the documented host-loop — verify the
  frame.
- **A "never prints DONE" writeback may be dying, not slow.** Reconcile the live
  profile against the process exit: PID 2041 OOM'd (`alloc_zeros failed`,
  RUN_EXIT=1), it did not hang. Check the exit code before concluding "speed only".
- **Capture the cheapest real number first.** A seq=512 synthetic writeback
  (loss=3.82, 166 s) proved correctness + bounded cost with the EXISTING binary in
  one run — no rebuild, no waiting on the blocked production path.
- **Exponential-backoff crash-loop ≠ retry harder.** When the container restart
  interval is shorter than the job's minimum runtime, more relaunches cannot help;
  name the wall (needs a ≥10 min stable window) and stop — the seq=512 completion
  already isolated code from infra.

Claude-Session: https://claude.ai/code/session_01Vsoud3oabdLDppvb274bCr
