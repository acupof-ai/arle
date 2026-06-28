# Agent-OPD value run: 8/8 training rollouts pass, masked-CE writeback runs — but the per-trajectory host-bound backward (~24 min, GPU idle / CPU 98%) blocks capturing `trained_pairs`/`mean_loss`

## Context

Continuation of the [6/6-accept run](2026-06-28-agent-opd-train-task-validated-6of6-accept-pod-crashloop-block.md),
which proved the loop produces accepted rollouts but crashed (pod CrashLoopBackOff)
before the writeback logged its numbers. This run targeted the same capture
(`trained_pairs`/`mean_loss` per round + held-out Δ) on a **stabilized** container
(uptime held 37+ min, far past the prior 3-6 min SIGKILL cadence).

Binary: HEAD `7ae42221` in the own pod tree `/host/arle-ckl-aopd` (spawner symbols
verified present in the built `target/release/arle`). Data re-staged from persistent
`/host` (train `/host/agent_opd_task.jsonl` + held-out `/host/agent_opd_eval.jsonl`
+ `/host/staged` + `/host/eval_staged`; `GIT_CONFIG_GLOBAL=/host/aopd_gitconfig`).
Config (the validated 6/6-accept regime, leaned to front-load training into the
window): `--rounds 2 --samples-per-prompt 8 --rollout-temperature 1.0 --max-turns 16
--max-tokens 768 --lora-layer-start 32 --rollout-num-slots 1 --eval-every 1
--eval-temperature 0.0`, detached via `setsid` + short-poll. GPU 0, all 8 H20 free.

## What Worked (case-as-fact, measured)

**8/8 training rollouts PASSED — strongest accept signal yet.** All eight samples
(0-7) of the validated task `ansible__ansible-f327e65` logged
`passed=true (turns=16) :: [exit 0]` — better than the prior run's 6/6 (which
crashed mid-sample-6). The accept→writeback gate is decisively reachable.

**Container stayed up 37+ min — the crash-loop blocker is gone this session.**
The prior run's blocker (node-governance SIGKILL on a 3-6 min cadence) did NOT
recur; the container loaded the 29 GB 27B-FP8 share-frozen-base engine + student,
ran the base eval, and ran all 8 training rollouts without a single restart.

**Baseline held-out eval reproduced exactly: pass_rate = 0.3333 (1/3).**
`ansible__ansible-0ea40e0` pass (`[exit 0]`), `12734fa` fail (`[exit 4]`),
`5e36960` fail (`[exit 1]`), all 3 edited — byte-identical to both prior runs'
baselines. Dumped to `/host/aopd_evalout_value/eval_round_base.jsonl`.

**The masked-CE writeback IS executing — the LoRA update is in flight.** After
the 8th rollout: `[agent-opd] released inference scratch` + `released rollout KV
pool` (freeing rollout memory for the CE backward), then
`[masked-writeback] seq_len=11735 total_targets=1496 chunk_rows=2048` — the first
accepted trajectory's `train_on_accepted` masked-CE step. This is the exact step
that produces `trained_pairs`/`mean_loss`; the prior run never reached it.

**Case-decoded an accepted training rollout (this run's `agent-opd-debug`
trace).** The student again located the bug at turn 12 — "`is_valid_collection_name`
in `_collection_finder.py` only uses the regex `^\w+\.\w+$` … while `_is_fqcn` in
`dataclasses.py` properly checks `not iskeyword(...)`" — then imported `keyword`
and rewrote the validator to reject Python-keyword parts. The exact gold fix,
reproduced independently of the prior run.

## Result — writeback numbers blocked by the host-bound masked-CE backward (NOT a container crash)

`trained_pairs`/`mean_loss` and the post-training held-out Δ could NOT be
captured, but the blocker is **different** from the prior run and is decisively
attributed. The first `train_on_accepted` step ran **~24 min and did not
complete** (`[masked-writeback] DONE loss=` never printed). Fine-grained profiling
(10 samples @ 0.5 s) shows the step is **host-bound, not GPU-bound**:

```
GPU util mostly 0 % @ 124 W (idle power), rare 13 % blips   ← GPU starved
proc CPU = 98 %, state Rl                                   ← saturated on host
```

The coarse `nvidia-smi` "100 % / 413 W" readings caught only the sparse GPU
bursts; at 0.5 s granularity the H20 is idle ~80 % of the time while one CPU core
is pinned. This is the documented OPD **host-loop autograd wall** (the from-scratch
`crates/autograd` masked-CE backward runs the heavy work on the host, GPU starved
— `memory/reference_opd_fused_distill_host_loop_pathological`). At ~24 min for
trajectory 1 of 8, round-0 writeback alone is ~3 hrs; the held-out Δ further. Not
capturable in any reasonable window even on a stable container.

Note the seq_len: 11735 (< the 15858 that OOM'd in the
[forward-activation-wall errors entry](../errors/2026-06-26-agent-opd-forward-activation-wall-after-logits-fix.md));
it fits in memory (steady 44817 MiB, no OOM) — the blocker here is **speed**, not
memory.

## Honest verdict (a/b/c per the brief)

**Variant of (c) — infra/perf blocker, but NOT the anticipated container
crash-loop.** Training is *provably real and in flight* (8/8 accept + the
masked-CE step executing), so this is NOT a capability ceiling and NOT a null.
But the numeric `trained_pairs`/`mean_loss` and the held-out Δ are blocked by the
**host-bound writeback backward** (~24 min/trajectory × 8 ≈ 3 hrs), a NEW wall the
prior crash-loop had been masking. A real value signal still requires making the
masked-CE writeback GPU-bound (the host-loop is the next license-or-kill target),
and a rigorous capability claim still needs multi-seed (≥5) + Wilson CI per the
small-n-eval rule.

## Bench

Exempt: agent-OPD training path, not a serving hot path. Default serve/CLI is
byte-identical (spawner gated on `ARLE_SPAWNER_SOCKET`). No code change this run
(config + pod-data re-staging only); binary is HEAD `7ae42221`.

## Rule

- **A stable container can still block the writeback — profile GPU-vs-host before
  blaming the pod.** When a masked-CE step runs minutes with no `DONE`, sample
  `nvidia-smi` at 0.5 s: coarse "100 % util" hides a starved GPU. GPU 0 %/idle-W +
  CPU 98 % = host-bound autograd (the documented host-loop wall), not a hang and
  not a container crash. The fix is to move the backward on-device, not to wait or
  relaunch.
- **`[masked-writeback] seq_len=…` prints once per accepted trajectory at step
  START; `DONE loss=…` prints at step END.** A lone `seq_len` line with no `DONE`
  means trajectory 1 of N is still in its backward — count starts vs DONEs to know
  how many of the `trained_pairs` have actually trained.
- **8/8 > 6/6 confirms the accept gate is not the bottleneck.** Two independent
  stable windows both solve the validated task at the default turn/token budget;
  the remaining wall is writeback throughput, not rollout capability.
