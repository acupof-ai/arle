# Deep Research: is ARLE's agentic-RL algorithm + infra best-in-class? (2026-07-21)

> Status: Active. Six parallel web-grounded research streams (algorithm core,
> multi-turn agentic, reward, infra systems, rollout acceleration, frontier),
> each benchmarked against 2025–2026 SOTA and against ARLE's actual setup
> (agent-OPD, cc-harness multi-turn SWE rollouts, GRPO, Qwen3.6-27B ThinkingCap
> FP8 + LoRA, 8×H20, in-process serve + autograd, train-infer-unified). Labels:
> **[M]** measured in the cited source, **[H]** hypothesis / mechanistic-only.

## Verdict

**The algorithm toolkit ARLE needs is already built — the gap is which knobs we
select, not code.** `update_strategy.rs` already ships DAPO (clip-higher
0.2/0.28 + dynamic sampling + overlong filter), Dr.GRPO / GSPO (`std_norm=false`,
sequence-level IS), `DropTruncated`/`DropZeroVarAndTruncated` compact filtering,
`--task-selection` (GRESO zero-variance skip), and `--staleness 0..=1`
(one-step-off async). The single biggest miss is a **config choice we just made
wrong**: `--sync every-group` (strict on-policy) — three independent streams
converge that it is overkill and the most expensive default in our regime.

Two things genuinely need building: (1) the **reward** (our dense partial-test
fraction is the exact design a controlled study flags as miscalibrated), and
(2) the **frontier moat** — bitwise-exact on-policy RL, which train-infer-unified
gives ARLE structurally and which is worth a headline experiment.

## The change list — flag-flip vs code

| Change | Evidence | Flag or code | Priority |
|---|---|---|---|
| `--sync every-group` → `--staleness 1` (one-step-off) | I1 async 2.2–2.8× at reward parity [M]; A1 strict sync makes IS-ratio≡1 → GSPO/clip-higher inert; A2 straggler tax worst in agentic | **flag exists** | P0 |
| `--update-strategy grpo` → `dapo` (or `dr-grpo`) | A1/A2 DeepSWE bundle (no-std, no-KL, clip-high, overlong filter) = SWE-bench-V 42.2% [M]; drops std bias fatal at G=4 | **flag exists** (preset) | P0 |
| `--task-selection true` (dynamic sampling) | A2 all-pass/all-fail groups at G=4 waste the batch; DAPO dynamic sampling | **flag exists** | P0 |
| `--samples-per-prompt 4` → `8` | A1/A2/A3 G=4 std estimate noisy + sparse-reward zero-gradient groups common | **flag exists** | P1 |
| `--kv-cache-dtype fp8` in the rollout lane | I2 NVIDIA FP8-KV+attn +48% [M]; correctness already licensed (CLAUDE.md) | **flag exists** (opt-in) | P1 |
| keep `--mtp-draft-tokens 3`; gate on tail occupancy | I2 agentic low-occupancy = memory-bound = spec-decode's win regime (TLT 3.2–3.6× BS=1 vs 1.7× BS=32 [M]) | flag on; elastic gate = code | P1/P2 |
| Reward: dense fraction → **binary-anchored + validity-gate (`valid·correct`)** | A3 Pass-Rate study: dense loses 2pp pass@16 vs binary [M]; VeRPO/TritonRL anchor recipe | **code** (`trajectory_scorer.rs`) | P1 |
| Reward↔held-out divergence monitor; timeout as its own bucket; LLM-judge spot-check (use teacher) | A3 divergence widens as hacking is discovered [M]; timeout-as-fail = our own OPD false-KILL | **code** (metrics + scorer) | P1 |
| Bitwise-exact on-policy: log per-token `rollout_logprob − train_logprob`, assert K3 KL ≈ 0 | F1 rollout-training mismatch is the central 2025-26 RL-infra pain; MoE KL 1e-3–1e-1 trains-collapse [M]; ARLE erases it by construction | **code** (instrumentation) | P1 (headline) |
| Task-level rollout concurrency (extend eval-concurrency to rollout) | I1 SkyRL 1.55×, ~90% util, no new GPUs [M] | **code** (host scheduler) | P2 |
| Prefix/turn KV reuse across the G samples (RadixCache extension) | I2 TreePO/SRT amortize shared prefix, compounds with spec-decode [M] | **code** | P2 |

## Convergent consensus (cross-validated, ≥2 independent streams)

1. **Strict every-group sync is the wrong default** — A1 (IS-ratio≡1 ⇒ half the
   SOTA inert), I1 (all frontier systems dropped it; 2.2–2.8× at parity), A2
   (agentic length-variance straggler tax worst here). → one-step-off + IS
   correction. **The single highest-confidence finding.**
2. **The DeepSWE "GRPO++" bundle** (no-std / no-KL / clip-high / max-len norm /
   compact-filtering / RLOO baseline) is the closest public precedent to our
   setup (Qwen3-32B RL-only → SWE-bench-Verified **42.2% pass@1** [M]). ARLE's
   `dapo` preset already encodes most of it.
3. **G=4 is too small** for a sparse/execution reward — noisy std, frequent
   zero-variance groups. Raise G and/or lean on dynamic sampling.
4. **Our dense partial-test-fraction reward is miscalibrated** — controlled A/B
   shows dense does **not** beat binary (loses 2pp pass@16) [M]; move to a hard
   binary anchor + a small *calibrated* dense term behind a validity gate.
5. **Spec-decode vs task-concurrency is a false choice** — do both in sequence
   (batch to fill occupancy, elastic MTP spec-decode on the drained tail). Our
   MTP head is self-speculative → never stale → we skip the online-draft-retrain
   loop everyone else builds.

## The synthesis that resolves the apparent contradiction

"Frontier says stay on-policy (bitwise-exact), algorithm/infra say go
off-policy (staleness)" is **not** a contradiction — the two are orthogonal:

- **Bitwise-exact on-policy** is a *numerical/backend* property: same operators
  ⇒ rollout-scoring and train-scoring produce identical logprobs (K3 KL ≈ 0).
- **Staleness** is a *temporal* property: reuse a rollout across ≥1 weight update.

ARLE can hold **both**: exact scoring **and** one-step reuse — and because it is
bitwise-exact, the IS correction for staleness is *clean* (the only divergence
is the genuine weight-version delta, not backend FP noise that everyone else's
TIS/MIS patches must also absorb). This is a combination no seam-based stack
(vLLM+FSDP) can offer. It also folds into I2's frontier: the **spec-decode verify
pass already emits the target logprob at every accepted position** — so the
decoupled π_behavior recompute largely *is* the verify pass; fuse them and the
draft, the target, and the logprob-source become one object.

## Per-axis distilled

**Algorithm (A1).** Consensus: kill KL-to-ref, kill std-norm, fix length
aggregation, filter degenerate groups — all live in our presets. GSPO's real
measured win is **MoE routing stability** (removes Routing Replay) — relevant iff
ThinkingCap-27B is MoE; verify. Under staleness=0 GSPO/clip-higher are no-ops;
they only bite once we go one-step-off. Skip VAPO/value-based (wrong for LoRA),
GMPO/CISPO (marginal for us). Cites: DAPO 2503.14476, Dr.GRPO 2503.20783, GSPO
2507.18071, DeepSWE (together.ai), "GRPO is secretly off-policy" 2509.24203.

**Multi-turn (A2).** Per-trajectory GRPO **is** the SWE-agent field standard
(DeepSWE, R2E-Gym, Kimi-Dev, Qwen3-Coder) — do **not** switch to turn-level
(GiGPO/MT-GRPO win on ALFWorld/WebShop where states repeat; SWE trajectories
never revisit a state, and we have no per-turn reward to blend). Highest-value:
**compact filtering** (mask truncated/timeout/max-step), then dynamic sampling /
larger G. Reward-hacking is the sleeper risk — Cursor: **63% of Opus-4.8 "fixes"
came from git history** [M]. Cites: DeepSWE, RLEF 2410.02089, GiGPO 2505.10978,
Cursor reward-hacking study.

**Reward (A3).** Pass-Rate study [M]: dense partial-fraction does not beat binary
and induces intra-group gradient conflict (57% of tasks) — worst at G=4. Move to
VeRPO/TritonRL **binary anchor + calibrated dense behind `valid·correct` gate**.
Missing guardrails: LLM-judge spot-check (best detector, 0-FN in EvilGenie [M];
use our teacher), reward↔held-out divergence monitor (the hacking alarm), timeout
bucketing (not pass/fail). **Do not add a length reward** — ThinkingCap is already
length-minimized; use length only as a drop-filter. Cites: Pass-Rate 2605.02944,
VeRPO 2601.03525, TritonRL 2510.17891, EvilGenie 2511.21654, SpecBench 2605.21384.

**Infra (I1).** ARLE is structurally *ahead* on the two axes everyone pays to fix:
weight-sync (ours is an in-process on-device LoRA merge — the 7–53s cross-process
broadcast numbers are not our regime) and the rollout-train seam. Behind on one:
async. Highest-value = **task-level rollout concurrency** (SkyRL 1.55×, ~90% util,
reuses our batching, zero new GPUs), then **one-step-off async**. Disaggregation
and weight-sync optimization are *not* on our critical path. Cites: AReaL
2505.24298, ROLL-Flash 2510.11345, SkyRL-Agent 2511.16108, APRIL 2509.18521.

**Rollout accel (I2).** Spec-decode via MTP is a **real win** for our regime
(agentic low-occupancy = memory-bound; TLT 3.2–3.6× at BS=1). Our MTP head can't
go stale (moves with the target on LoRA re-merge) — *verify the LoRA merge reaches
the MTP head's input features; if MTP acceptance decays across groups, refresh the
head in rollout bubbles*. Largest un-taken lever: **FP8 KV+attention in the
rollout lane** (correctness already licensed). FP8 caveat: run the π_behavior
recompute at the *same* FP8 recipe as generation → ratio→1, no TIS; never FP8-gen
vs BF16-recompute with a bare ratio. Cites: TLT 2511.16665, ReSpec 2510.26475,
Jet-RL 2601.14243, NVIDIA end-to-end-FP8-RL, SPEC-RL 2509.23232.

## Frontier bets (ranked)

1. **★ Bitwise-exact on-policy RL as a runtime guarantee (F1).** Train-infer-unified
   erases the rollout-training mismatch seam that the field patches with IS; the
   mismatch is worst on MoE (our flagship target). Cheapest headline: a K3-KL≈0
   plot vs the published 1e-3–1e-1 band, then an injected-mismatch A/B to show it
   *matters* on our workload. Precondition that makes bets 2–4 trustworthy.
2. **Runtime-fused OPD** — teacher-logprob capture inside the same generation step
   (OPD: 7–10× fewer steps / 50–100× less compute than RL to match a teacher [M,
   math]); no separate teacher deployment, no weight-sync round-trip.
3. **Spec-decode-native RL (I2)** — fold the π_behavior recompute into the
   spec-decode verify pass; the MTP head becomes draft + trained-object +
   logprob-source in one. Gate on the measured accepted-fraction (a countable
   number) before refactoring.
4. **Cost-aware reward from runtime telemetry (F1)** — shape reward on *measured*
   decode-ms / KV-pressure / acceptance, not token count; "cheap AND correct."
   High confounder risk (batch-dependent) → matched-A/B only.
5. **MTP head as densified RL signal** — joint MTP aux loss during GRPO (2605.28184)
   + acceptance-as-difficulty signal.

## Recommended config + phased plan

**Phase 2 (now, all flag flips — no code):**
`--update-strategy dapo --staleness 1 --rollout-temperature 1.0
--samples-per-prompt 8 --task-selection true --sync every-group→(removed; staleness
drives it) --mtp-draft-tokens 3 --kv-cache-dtype fp8` on the ThinkingCap student.
One-variable discipline: flip **staleness** and **strategy** first (biggest,
cross-validated), hold G and dtype for the second A/B. Gate on the correct-inference
probe + the 二八 three signals (held-out delta / completion_tokens /
zero_variance_frac).

**Phase 2.5 (code, ranked):** ① reward → binary-anchored + `valid·correct` gate
+ divergence monitor + timeout bucket; ② K3-KL≈0 instrumentation (headline);
③ task-level rollout concurrency.

**Phase 3:** frontier bet #1 injected-mismatch A/B; then #2/#3 as the runtime moat.

## SOLID caveats

- Async 2.2–2.8× are the authors' shapes (mostly 32B multi-node math/agentic) —
  the **achievable envelope**, not a guaranteed number on 8×H20; SkyRL's 1.55×
  (agentic, single-cluster, tool-exec overlap) is the most transferable.
- "Weight-sync is free for ARLE" is inference from the in-process/LoRA design —
  verify with one on-device LoRA-merge timing probe before deprioritizing.
- GSPO's MoE win applies iff ThinkingCap-27B is MoE — **verify the arch** (its
  config was `Qwen3_5ForConditionalGeneration`, hybrid linear-attn, head_dim=256;
  MoE-ness unconfirmed here).
- Every "flag exists" row above was grep-confirmed in `update_strategy.rs` /
  `args.rs` but **not yet run end-to-end** with these values — reachability, not
  acceptance. The c-sweep + held-out gate still decides.

## Sources (per-axis, deduped)

DAPO 2503.14476 · Dr.GRPO 2503.20783 · GSPO 2507.18071 + Qwen blog · CISPO
2506.13585 · VAPO 2504.05118 · GRPO-off-policy 2509.24203 · DeepSWE (together.ai)
· SWE-RL 2502.18449 · RLEF 2410.02089 · R2E-Gym · Kimi-Dev 2509.23045 · GiGPO
2505.10978 · MT-GRPO 2505.11821 · Pass-Rate-vs-Binary 2605.02944 · VeRPO
2601.03525 · TritonRL 2510.17891 · Kimi-k1.5 2501.12599 · EvilGenie 2511.21654 ·
SpecBench 2605.21384 · Cursor reward-hacking study · AReaL 2505.24298 · ROLL-Flash
2510.11345 · SkyRL-Agent 2511.16108 · APRIL 2509.18521 · verl HybridFlow ·
checkpoint-engine · LMSYS P2P RDMA · SPEC-RL 2509.23232 · ReSpec 2510.26475 · TLT
2511.16665 · Draft-OPD 2605.29343 · MagicDec · Jet-RL 2601.14243 · NVIDIA
end-to-end-FP8-RL · FP16-mismatch 2510.26788 · Tree-GRPO 2509.21240 · Thinking
Machines On-Policy-Distillation · Rollout-Training-Mismatch NeurIPS'25
(OpenReview 8MHqvb4lK9) · TIS/AIS 2605.14220 / 2605.13907 · Joint-MTP-in-RL
2605.28184.
