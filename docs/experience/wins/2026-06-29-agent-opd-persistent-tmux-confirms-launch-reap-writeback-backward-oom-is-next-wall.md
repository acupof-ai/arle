# Agent-OPD "dies at loading student (~80s)" = the NON-PERSISTENT launch being reaped (hypothesis a), NOT a pod-side kill — persistent tmux ran the FURTHEST EVER (baseline eval + 4 rollouts + sample-0 PASS + writeback forward), then hit a NEW clean wall: the masked-CE writeback BACKWARD OOMs (`cuda alloc_zeros failed`) on a 10780-token trajectory

## Context

Mainline brief: decide why agent-OPD runs die at "loading student" (~80s) on the
8×H20 box. A fresh diagnosis had already FALSIFIED the prior "container
crash-loop / node-governance" attribution (container pid 1 = `sleep infinity`,
12h+ uptime, cgroup `memory.max` unlimited, dmesg ZERO OOM/coredump/kill at the
15:55/16:30 death times). Two live hypotheses:
- **(a) launch method** — a run launched detached via `~/bin/pod 'setsid … &'`
  gets reaped when the `~/bin/pod` exec session ends (the documented
  "exec 143 reaps; setsid alone may be insufficient" hazard).
- **(b) a real untraced pod-side governance kill.**

The runs that died (`run_aopd_ckl2.sh`, the 15:55/16:30 deaths) ended with
`exec -a arleCKL … >> log 2>&1` in the **FOREGROUND** of the `~/bin/pod` (crictl
exec) session — NO tmux, NO setsid, NO `&`. The one earlier full completion
(`a6f74cc2`) and the 15:29 `run_ckl.log` had caught a stable window.

Binary HEAD on pod = `/host/arle-ckl-aopd/target/release/arle` (built Jun 28
15:50, carries the `ccedd788` scoring fix + the pre-CUDA sandbox-spawner from
[precuda-spawner-closes-loop](2026-06-28-agent-opd-precuda-spawner-closes-loop-on-elkeid-pod.md)).
Config: `--samples-per-prompt 4 --writeback-cap 1 --rounds 1 --eval-every 1
--max-turns 16 --max-tokens 768 --lora-layer-start 32 --rollout-num-slots 1`,
1 train task (`ansible__ansible-f327e65`) + 3 held-out eval tasks, data on `/host`.

## What Worked — STEP 1 decides (a) vs (b): persistent tmux SURVIVES the load

Launched FULLY detached from any `~/bin/pod` exec: `tmux new-session -d -s aopd`
running `exec -a arleCKL /host/arle-ckl-aopd/target/release/arle …` (marker
`arleCKL`, free GPU 1 @ 0 MiB), then let the `~/bin/pod` exec RETURN and
re-checked in separate short exec calls.

**Verdict = (a), unambiguous.** With the run in a persistent tmux that survives
the exec teardown, it ran the **furthest any agent-OPD run has ever gone on this
box**, ~34 min, well past every prior death point:

| Milestone | Result | Prior runs |
|---|---|---|
| survive "loading student" (~80s) | **YES** — GPU 1 allocated 36 GB @ 100%, `arleCKL` pid 51905 @ 93% CPU, tmux ALIVE *after* the `~/bin/pod` exec returned | **DIED here** (15:55, 16:30) — GPU 0, log stops at "loading student" |
| round-0 baseline held-out eval (3 tasks) | **DONE** — `pass_rate=0.3333` (1/3) | reached only twice before |
| 4 training rollouts (sample 0–3) | **DONE** — sample 0 **passed=true `[exit 0]`**, 1–3 `no edits (MaxTurns 16)` | sample 0 was `git apply … patch does not apply` (UNSCORABLE) on the prior run |
| masked-CE writeback (accepted sample 0) | **STARTED** — `seq_len=10780 total_targets=1422`, `phase=forward_hidden_states seconds=1428.547`, `phase=fused_ce seconds=0.297` | never reached `trained_pairs>0` |

The prior 15:55/16:30 deaths were the **non-persistent foreground
`~/bin/pod` exec launch being reaped** when the exec session ended — exactly the
documented `exec 143 reaps; setsid alone insufficient` hazard. There is **no
pod-side governance kill** (b) on this path: the container is stable, the
sandbox-spawner already dodges the ELKEID fork-hook, and the persistent run died
~34 min in for a completely different, **clean, self-caught** reason (below) —
GPU released to 0 MiB, no SIGKILL, no leaked process.

**`ccedd788` scoring fix is verified working (case-as-fact):** sample 0
`passed=true (turns=16) :: [exit 0]` — the model added the Python-keyword check to
`AnsibleCollectionRef.is_valid_collection_name` (turn 15 `replace`), and the
hidden `test_patch` now **applies + the tests pass**. This is the FIRST scorable
accepted train rollout on this box; the prior run's `trained_pairs=0` was the
`git apply` offset mismatch, now fixed. So `trained_pairs` would have been 1.

## The NEW wall (case-as-fact): masked-CE writeback BACKWARD OOMs on the 10780-token trajectory

```
[masked-writeback] seq_len=10780 total_targets=1422 chunk_rows=2048
[masked-writeback] phase=forward_hidden_states seconds=1428.547   # ~24 min FORWARD over 10780 tok
[masked-writeback] phase=fused_ce seconds=0.297                   # fused-CE itself is ~free
[ARLE train] error: masked CE writeback (round 0): cuda alloc_zeros failed
```

Watched live: GPU mem 50 GB (forward + checkpoint offload) → **95.2 GB** at the
backward (activation gradients over the full 10780-token forward graph) → the
next `alloc_zeros` exceeds the 97.8 GB H20 → **clean, caught OOM**. NOT a
SIGKILL/reap: ARLE's own error path printed it, the process exited, GPU released
to 0 MiB, marker hygiene intact (no foreign proc touched, all 8 GPUs 0 MiB after).

**Root cause = `--writeback-window` (default 2048) bounds only the
`[window, vocab]` LOGITS tile, NOT the hidden-states forward+backward over the
full sequence.** `phase=forward_hidden_states` re-forwards the **entire 10780-tok
prefix**; the backward through that full graph materializes activation grads to
95 GB. Sample 0 was the longest trajectory precisely *because* it was the one
that successfully multi-turn-edited to a PASS — so the value-producing rollout is
also the one that OOMs the writeback. The window flag does not bound this path.

## Verdict

- **(a) CONFIRMED** — the "~80s death" was launch reaping, fixed by a persistent
  tmux launch (full detach from the `~/bin/pod` exec). **(b) ruled out** on this
  path. This is the mainline question, answered.
- **Value signal: baseline captured, training-Δ BLOCKED by the writeback OOM.**
  Held-out baseline `pass_rate=0.3333` (1/3) is real (the base Qwen3.6-27B-FP8
  student edits all 3 held-out tasks, solves 1). `trained_pairs` would have been
  **1** (sample 0 scorable+passed — the `ccedd788` fix's first payoff), but the
  writeback OOM'd before the AdamW step, so **no LoRA update, no post-train eval,
  no Δ** (no `eval_round_1.jsonl`, no `/host/agentopd_ckl5/` adapter dir). The
  Δ is not "flat" — it is **unmeasured** (training never landed a weight step).
- A rigorous capability claim needs `trained_pairs>0` → AdamW step → post-eval Δ,
  AND multi-seed (≥5) + Wilson CI per the small-n-eval rule (3-task eval is far
  below that bar regardless). Not reached this run.

## Next step (cheap, file:line) to land `trained_pairs>0` → a real Δ

The writeback backward must be bounded to fit 97.8 GB on a ~11k-token trajectory:
1. **True sequence-windowed forward+backward** (not just the logits tile):
   re-forward + backward each `--writeback-window` slice with grad-checkpointing
   across windows, so peak activation-grad VRAM is `O(window)`, not `O(seq_len)`.
   The current `forward_hidden_states` is whole-sequence — that's the OOM.
2. OR cap the accepted-trajectory length fed to writeback (truncate/skip the
   tail beyond a VRAM-safe token budget), OR raise `--max-tokens`/lower
   `--max-turns` so the accepted rollout is shorter (sample 0 was 10780 tok at
   16 turns × 768 tok).
3. OR `--lora-layer-start` higher (fewer trainable layers ⇒ fewer stored grads),
   OR offload optimizer/grad state to host across windows.

Forward already took ~24 min for one 10780-tok trajectory (the host-bound
writeback wall, now quantified as `forward_hidden_states`, not the CE) — the
windowed rewrite also fixes the latency, not just the OOM.

## Run facts

- Launch: `tmux new-session -d -s aopd` → `exec -a arleCKL
  /host/arle-ckl-aopd/target/release/arle train agent-opd …` on GPU 1 (free).
  `ps aux` showed `argv[0]=arleCKL` throughout; tmux survived the `~/bin/pod`
  exec return (the decisive (a)-vs-(b) control). Clean teardown: all 8 GPUs
  0 MiB, no `arleCKL`, no tmux server after the OOM.
- Timeline: 03:20 load → 03:24 baseline eval done (0.3333) → 03:26–03:29 4
  rollouts (sample 0 PASS) → 03:30 writeback start → 03:54
  `forward_hidden_states 1428.547s` + `fused_ce 0.297s` → backward OOM, exit.
- Artifacts: `/host/aopd_evalout_ckl5/eval_round_base.jsonl` (baseline only).
  No `eval_round_1.jsonl`, no adapter dir. Log: `/host/run_ckl5.log`.
- Marker hygiene held: only ever launched/killed `arleCKL`; no foreign process
  or GPU touched; ckl's local infer-cuda WIP left untouched.

## Bench

Exempt: agent-OPD training path, not a serving hot path. No code change this run
(persistent-launch + config + pod-data only); default serve/CLI byte-identical.
Per the mandatory-bench rule this is the training-axis diagnosis + baseline
capture, not a guidellm serving delta.

## Rule

- **"Dies at loading student (~80s)" with the launcher ending in a foreground
  `exec … >> log 2>&1` under `~/bin/pod` (crictl exec) is the EXEC-REAP hazard,
  not a pod-side kill.** The discriminator is a single control: relaunch in a
  persistent `tmux` that survives the exec teardown. If it then runs past the
  load (GPU allocates, rollout lines appear), the prior deaths were reaping —
  `setsid &` alone is insufficient; the run must be owned by a tmux/setsid server
  that does NOT die with the `~/bin/pod` exec.
- **A self-caught `cuda alloc_zeros failed` (GPU released to 0 MiB, ARLE's own
  error path prints it) is a clean OOM, NOT the silent-SIGKILL/reap class.** Do
  not conflate the two: the reap leaves the log stopped mid-line with the process
  gone and GPU non-zero; the OOM prints a named error, exits, and frees the GPU.
- **`--writeback-window` bounds only the `[window, vocab]` logits tile, not the
  hidden-states forward+backward over the full trajectory.** A long accepted
  rollout (~11k tok) OOMs the writeback BACKWARD at ~95 GB on a 97.8 GB H20.
  The value-producing rollout (the one that PASSED) is also the longest, so this
  is on the critical path — the fix is a true sequence-windowed forward+backward,
  not just the logits-tile window.
- **Decode the writeback phase timers before blaming "host-bound CE".** The
  `forward_hidden_states 1428s` vs `fused_ce 0.297s` split shows the ~24-min wall
  is the whole-sequence layer forward, NOT the cross-entropy — the fix targets
  the forward/backward windowing, not the CE.

Claude-Session: https://claude.ai/code/session_01Vsoud3oabdLDppvb274bCr
