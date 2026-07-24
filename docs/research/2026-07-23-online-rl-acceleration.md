# Accelerating ARLE online RL (agent-OPD loop)

> Status: Active — 2026-07-23. Evidence-grounded survey of the acceleration space
> for the agent-OPD online-RL loop (rollout → pytest reward → DAPO/masked-CE
> update → LoRA remerge). Deep-research synthesis; code claims source-traced to
> file:line, throughput numbers from the 2026-07-11 profile, industry claims
> external.

## Verdict

**"Make online RL async" does not transfer to ARLE.** The AReaL 2.77× / verl 1.7×
headline speedups come from **disaggregating** generation GPUs from training GPUs
so rollout ∥ update. ARLE is single-box, single-process, **colocated by design**
(`infer_student.rs:34` one `Arc<Mutex<LoadedInferenceEngine>>`; base weights
zero-copy shared via `--share-frozen-base`, `train_cli.rs:3086`). On one GPU set,
rollout-decode and writeback-backward both saturate the same SMs — they **cannot**
run simultaneously. verl states its 1.7× is "vs the colocate setup", i.e. the win
IS de-colocation, which ARLE deliberately rejects.

The three GPU stages are serial not because of a lock but because of **VRAM mutual
exclusion**: the rollout KV pool must be physically torn down before writeback
backward (`train_cli.rs:3523 release_kv_pool`; a resident pool+scratch atop the
writeback forward OOMed at 97.4 GB > 96 GB). Rollout needs the pool; writeback
needs it gone. This is the load-bearing wall — physics, not software.

So acceleration here is **not** an async-systems problem. It is three different
problems: **(1) rollout concurrency starvation, (2) wasted rollout on
gradient-free trajectories, (3) the writeback VRAM wall.**

## Where the wall goes (measured, 2026-07-11, 27B FP8, 4×2 synthetic round = 173.5 s)

| Stage | % round | ms/call × n | GPU/host | Overlappable? |
|---|---|---|---|---|
| rollout_decode | **40.4%** | 8764 × 8 | mixed | no (VRAM + lock) |
| writeback (fwd+bwd+opt) | **33.1%** | 14366 × 4 | gpu | no (VRAM exclusion) |
| eval | **23.0%** | 39868 × 1 | mixed | no (round-tail barrier) |
| pytest reward | 3.3% | 714 × 8 | **host** | already ∥ sibling decode |
| sync_lora (weight remerge) | 0.0% | 60 × 1 | gpu | already optimal (was 60–83 s) |

96.5% of the round is three non-overlapping GPU stages; provable GPU-idle = **3.5%**
(= host pytest) on synthetic, growing to ~15–18% on real SWE-Pro (seconds-to-minutes
pytest). **The GPU-stage-overlap ceiling on this box is that 3.5–18%, not a 2×.**

## Ranked levers (ARLE-specific)

### New top lever (2026-07-23 pod case-facts): share the ~31K CC-preamble prefix

Decoded sidecars from the busytimer-fixed run inverted the workload model:
every CC request carries a **30.9–31.1K-token prompt** (claude-CLI system
prompt + tool schemas + conversation) and generates only **12–96 coherent
tokens**. Consequences, measured on one H20:

- **Wall-clock**: ~150–300 s per turn is 31K prefill (chunked at 4096 —
  `--chunked-prefill-size 22000` clamps, `loaded.rs:2095`), not generation.
- **KV**: 8 concurrent sessions × 31K ≈ 248K ≈ the 250K-token pool →
  exhaustion (engine death pre-a9d0c5412; parks after).
- ~~**Root cause**: Qwen3.6 hybrid has prefix reuse structurally OFF
  (`reusable_prefix_blocks` → 0)~~ — **RETRACTED 2026-07-24, source-survey
  error.** The `|_| false` closure (`qwen35.rs:318`) only rejects
  host-demoted pages; resident radix pages count fully, and the recurrent
  sidecar save/restore is wired end-to-end since #85 (`52e2fdb47` 06-29,
  periodic stride snapshots `312d22c8c` 07-13) — all in the measured run's
  HEAD `0a42841ad`. The turn-1 vs turns-2–4 latency gap (lever 1 below) is
  consistent with same-session reuse working. What actually limits sharing:
  ① publish happens on **finish**, so a group's concurrent cold-start turn 1
  × 8 can't share (each prefills its own 31K → the pool exhaustion);
  ② every LoRA re-merge drops the whole prefix cache
  (`serve_engine.rs:376` — correctness-mandated, stale-epoch KV).

**The levers, in cost order** (2026-07-24 update — the 31K source is found):

1. **Stage CC workdirs outside the repo (config-only, do first — LANDED).**
   The dumps show `claude -p` walks up from the task workdir (under
   `/host/arle-build`) and ingests the repo `CLAUDE.md` agent contract — that
   IS most of the 31K. Root: `agent_opd_curve.sh` staged sandboxes under
   `$OUT` = repo-relative `runs/`. Fixed: `WORK_ROOT` default
   `/tmp/agent-opd-work` + a `boot_workdir` ancestor-CLAUDE.md warning.
   Turn 1 measured 178–245 s vs 21–80 s for turns 2–4 — dominated by exactly
   this prefill. Pending pod verification: prompt_tokens per request should
   drop ~31K → few K. Durable follow-up: `claude --bare` at the spawn point
   (`cc_harness.rs:270`) disables CLAUDE.md auto-discovery path-independently
   (also closes the `~/.claude/CLAUDE.md` vector the path fix can't) — but it
   drops hooks/auto-memory/workdir-level CLAUDE.md too, a rollout-behavior
   change needing pod-CLI support check + its own A/B before flipping.
2. **Prefix reuse — already built; residual gap is concurrent cold starts.**
   With lever 1 landed the shared preamble shrinks to CC's own system prompt
   + tool schemas (few K), so the residual win is small. If a measured run
   still shows redundant prefill: serialize a group's first sample until its
   preamble publishes, then admit the rest (they attach the published
   prefix). Gate any work here on a measured prefix-hit counter, not source
   survey.

### Tier 0 — RESOLVED 2026-07-24: gpu_busy_frac 0.30–0.34 → GO

Measured on a healthy hard-gated run (busytimer-s4, 1×H20, SPEC=off SAMPLES=4;
[wins](../experience/wins/2026-07-24-agent-opd-gpu-busy-frac-measured-go.md)):
the GPU forwards ~135 s of each ~410 s group — **~2/3 of the rollout wall is
idle** on CC-side latency. Doubling concurrency 4→8 held busy-frac flat (0.29),
so occupancy has headroom for many more overlapped groups.

> **Collapse the 4 serial group-rollouts into one `num_slots=8` concurrent
> mega-rollout** (`train_cli.rs:3396` inner loop + `:3026` num_slots). The
> continuous batcher already exists (`serve_engine.rs:126-133`); it is merely
> starved at C≤2 (K=2 concurrent `claude` sessions). Upper bound: 4× serial →
> ~1× concurrent on the GPU-active portion.

Now unblocked to build. Prerequisites in place: KV exhaustion parks instead of
killing the engine (`a9d0c5412`), and the prompt-side levers above shrink the
per-session KV footprint the mega-rollout multiplies. The A/B must clear
reward/loss parity (concurrent rollout = staleness drift), not just wall.

What already exists (2026-07-24 audit — nothing is broken, the loop is
design-serial): within-group sample concurrency (`cc_harness.rs:192-219`,
measured 8-way), next-group boot-ahead during rollout+train
(`train_cli.rs:3394`), and one-group rollout/train overlap via `--staleness 1`
(IS-corrected, curve-script default ON). The mega-rollout's new part is only
width: >1 group **rolling** simultaneously (staleness is capped at 1, the
published safe envelope — widening it moves the IS-correction goalposts).

### Tier 1 — cheap, safe, attacks measured waste (highest ROI)

> **Wire `scripts/comfort_band.py` as the default corpus-prep step.** ~50% of
> rollouts currently produce zero gradient: zero-variance always-pass groups, plus
> variance-bearing 30K-token trajectories that the 23K writeback cap SKIPS
> (`mean_loss=0.0000`, errors/2026-07-22). comfort_band offline-filters to
> intermediate-difficulty **and** ≤23K tasks, converting wasted rollout GPU into
> gradient-bearing rollout. Total-wall-to-target win, zero hot-path risk.

Not redundant with the runtime P5 `TaskSelection` (`train_cli.rs:599`): P5 skips
zero-variance/too-easy tasks **online**, but a 30K variance-bearing task passes
P5's filter (non-zero variance → `zv_streak` never increments) and wastes its
40%-of-round rollout every round. **Length pre-filtering is the gap P5 structurally
cannot close** — exactly comfort_band's role. Wired into `agent_opd_curve.sh`
2026-07-23 (profile 1 round → filter → train), default on, `COMFORT_BAND=0` opts out.

### Tier 2 — the writeback (33%, biggest single stage; training-kernel axis)

Writeback is VRAM-bound and cannot overlap; only its internals give:
- **seq-adaptive grad-checkpoint offload**: shipped (backward −36% @seq~1276), but
  `WRITEBACK_OFFLOAD_MIN_SEQ=4096` gate — wrongly-on-short regressed; **tune the gate**.
- **23K `max_update_seq` VRAM wall** (`update_strategy.rs:641`): real fix is
  sequence-parallel / activation-offload writeback to admit 30K trajectories
  (recovers their wasted 40%-of-round rollout). A training-VRAM project, not RL structure.
- LA chunkwise-GEMM backward / bf16-native autograd (P1.2/P1.3): autograd-kernel work.

### Tier 3 — real-corpus-gated (no-op on synthetic)

- **`--staleness 1` pipeline** (already built, `train_cli.rs:3390`) hides host-bound
  pytest/boot behind the next group's GPU rollout. On synthetic <3.5% (no-op); pays
  only on real SWE-Pro. **Blocked** by the 97.4 GB OOM (Rolling keeps the resident KV
  pool during writeback — shrink the pool first); needs a DAPO/ratio preset
  (`rejection_ce` errors at `:3322`) + truncated-IS for the k=1 off-policy variance.
- **Verify the prefix-cache flush on real prompt lengths.** On synthetic (~2560 tok)
  `prefix_warmup` hides it; on real SWE-Pro (22K–200K shared prefix, `cc_harness.rs:27,32`)
  the re-prefill spills onto the next rollout's critical path. Measurement, not build.
- **DAPO filter-before-capture reorder** (DAPO path only): hoist `DropZeroAdvGroup`
  (`update_strategy.rs:270`) ahead of `capture_rollout_logprobs` (`train_cli.rs:3595`)
  so dropped trajectories don't pay a capture forward. Zero on the `rejection_ce` default.

## Killed-in-disguise (the async playbook's traps)

| Lever | Why it fails on ARLE |
|---|---|
| Double-buffered weights (θ_{t-1} rollout ∥ θ_t optimize) | Targets a **measured 0.0%** (sync_lora 60 ms; flush already overlapped by prefix_warmup); +27 GB into an OOM budget = **net negative** |
| Disaggregate rollout/train GPUs on one box | Halving each stage's TP degrades throughput 1.7–2× → overlap is wash-or-loss. Only pays when GPUs are **added** |
| LoRA-delta-at-rollout to skip the prefix flush | Stale-epoch KV = on-policy corruption; explicitly guarded (`serve_engine.rs:363-372`) |
| Partial / interruptible rollout | pytest scores the **complete** trajectory (`cc_harness.rs:307`); a truncated agent turn = empty diff = reward 0 |
| Truncate stragglers by wall | Selection bias against hard-but-correct long trajectories |
| DAPO oversample-and-refill | Adds rollout wall to buy gradient density — wrong direction for wall-clock |

## Already banked (do not re-file as wins)

Continuous batching (exists, starved at C≤2) · fast on-device weight sync (60 ms, was
60–83 s) · LoRA-delta-only sync · verl-colocation placement (ARLE is past it — same
process, zero-copy base) · DAPO zero-variance drop · eval downfreq (23%→11.5%) · DSpark
rollout decode (−29%).

## Action queue

Neither top move needs free GPUs to *prepare*:
1. **Done 2026-07-23:** comfort_band wired as default corpus-prep in `agent_opd_curve.sh`.
   Acceptance (trainable fraction rises ~50%→~100%, `mean_loss>0`) is a pod-gated run.
2. **Next — a derivation, not a build.** The GPU-busy fraction of the 40.4% rollout
   wall decides whether the Tier-0 mega-rollout lever is worth building, and the
   telemetry already exists: `rollout_tok_per_sec` is emitted per group
   (`train_cli.rs:3493`, metrics.jsonl kind=group), and the engine exposes monotonic
   `throughput_stats()` {steps, generated_tokens, requests_completed} (`infer-core/src/lib.rs:779`)
   + `spec_decode_stats()` (`infer-api/src/loaded.rs:2450`). GPU-busy % ≈
   `rollout_tok_per_sec ÷ peak_decode_tok_s` (peak = one serve bench at batch≈K, or
   the known 27B-FP8+MTP decode rate). ≪ peak → idle-bound → mega-rollout wins; ≈ peak
   → GPU-bound, do not build. **Auto-tracked as of 2026-07-23:** `infer-core`
   Engine.step() now times each forward's submit→poll-Ready wall into a process-global
   `engine_forward_busy_micros()`; `train_cli` samples it around `cc_rollout` and emits
   **`gpu_busy_secs` + `gpu_busy_frac`** per group in metrics.jsonl (kind=group). So the
   Tier-0 go/no-go is now a direct read of `gpu_busy_frac` from any agent-OPD round — no
   derivation, no serve bench needed.

## Reward-bearing rollout ratio (2026-07-23)

Follow-up deep-dive: raise the fraction of rolled-out sample-groups that yield a
non-zero-advantage gradient (a zero-gradient rollout wastes ~40%-of-round GPU).

**Two premise corrections from the grounding:**
- **Reward is already dense, not binary.** `sandbox.rs:465` returns
  `passed/|fail_to_pass| ∈ [0,1]`; `RewardShape::Dense` is default-ON (`cc_harness.rs:58`).
  Do NOT flip `--reward-shape binary` (reverts it, lowers the ratio).
- **The ratio is computable from existing metrics** — `1 − mean(zero_variance)` over
  kind=group rows. One `--task-selection false` profile run gives the current ratio +
  the p-histogram; no new counter needed.

**First principle:** a group of k on a task with pass-rate p is reward-bearing with
prob `R(p,k) = 1 − p^k − (1−p)^k`, dominated by p (k=8: p=0.5→0.99, p=0.9→0.57,
p=0.99→0.08). So curriculum (move the sampled-p toward 0.5) is the top lever, mostly free.

**Landed 2026-07-23:**
- **Variance-weighted task selection** (`train_cli.rs` `TaskSelection`): `select()` now
  runs each task with prob ∝ its reward-bearing variance `p(1−p)` (0.1 floor), using the
  online `ema_pass` estimate — replacing the reactive zv-streak skip that only trimmed
  the tails. Concentrates the fixed rollout budget on the p≈0.5 band. Free (zero extra
  rollouts); ~1.6× ratio if the corpus averages p≈0.9. Gated by the existing
  `--task-selection`.
- **Min-tests granularity floor** (`comfort_band.py --min-tests 2`): drops tasks with
  `|fail_to_pass| < 2` where dense reward degenerates to binary (the toy corpus averages
  ~1.85 tests/task). Offline, free.
- **Script knobs** (`agent_opd_curve.sh`): `REWARD_SHAPE` (dense|anchored|binary),
  `REPLAY_REUSE` (amortize spent rollout via the built ReplayBuffer), `CB_MIN_TESTS`.

**Deferred (with reason):**
- **Rollout temperature 0.3→~0.8** (free ratio lift) — BLOCKED by the FP8/hd256 temp>0
  corruption (#48); unblock that sampler first.
- **Adaptive-k pilot gating** (the only lever that raises the *within-round* rolled-out
  ratio) — needs the measured p-distribution to tune k0 (≥3–4 to avoid false-killing
  p≈0.5 tasks); build after the profile run.

**Killed-in-disguise:** DAPO oversample-refill — ARLE already reaches ~100% *trained*-batch
ratio via `DropZeroAdvGroup` (trains a smaller batch); refill adds +75%(p=0.9)–+1200%(p=0.99)
rollout for zero net gradient. Adaptive-k for *more* samples on hard tasks (R(p,k) negative
ROI at extremes). PRM (displaces pytest, the ground-truth verifier).

**Next:** the pod profile run measures the actual ratio + p-histogram → sizes the
variance-weighting and unblocks adaptive-k k0 tuning.

## Anchors

- Loop: `crates/cli/src/train_cli.rs:3340` (round) / `:3396` (group) / `:3403` rollout /
  `:3520-3525` VRAM release / `:3614` writeback / `:3701` sync_lora / `:3766` eval
- Engine lock: `crates/train/src/infer_student.rs:34`; weight sync `:324` → `:423`
- Selection: `train_cli.rs:599` P5 TaskSelection; corpus filter `scripts/comfort_band.py`
- Profile: `docs/experience/wins/2026-07-11-agent-opd-round-profile-ms-breakdown.md`
- VRAM wall: `docs/experience/errors/2026-07-22-agent-opd-dapo-null-gradient-trajectory-vram-wall.md`
- Prior RL infra research: `docs/research/2026-07-21-rl-algo-infra-deepresearch.md`
